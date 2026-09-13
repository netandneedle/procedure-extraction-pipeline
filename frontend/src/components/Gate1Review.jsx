/**
 * Gate1Review — Procedure draft review for Gate 1.
 *
 * Two views (toggled):
 *   Cards — per-draft review cards with narrative, tags, actions
 *   Flow  — vertical DAG of attack chain + sequence override table
 *
 * The Flow DAG re-renders live when the analyst edits sequence overrides.
 *
 * Reject reasons: bad_chunk_boundary, wrong_technique, hallucinated,
 *   duplicate, too_vague, not_a_procedure, wrong_technique_granularity
 */
import { useState, useMemo, useCallback, useEffect } from "react";
import { DragDropContext, Droppable, Draggable } from "@hello-pangea/dnd";
import TechniqueEditor from "./TechniqueEditor";
import ProvenanceBadge from "./ProvenanceBadge";
import { fetchTechniques } from "../api/techniques";
import ReviewerBrief from "./ReviewerBrief";
import SuggestionChip from "./SuggestionChip";
import InfoDot from "./InfoDot";
import { hint } from "../lib/glossary";
import {
  indexByDraft,
  indexByPromotion,
  isBulkAcceptable,
  verdictForRecommendation,
} from "../lib/reviewerSuggestions";

// "Why" options for Re-map techniques. All of these route to
// extract_techniques (none is bad_chunk_boundary) and feed both the
// feedback flywheel and the AI's retry hint. "Not a procedure" / "duplicate"
// are reasons to Discard the procedure, not to re-map it — they live in the
// Discard note instead.
const REMAP_REASONS = [
  { value: "wrong_technique", label: "Wrong technique" },
  { value: "wrong_technique_granularity", label: "Wrong granularity (parent vs sub-technique)" },
  { value: "hallucinated", label: "Not supported by the source" },
  { value: "too_vague", label: "Too vague to map" },
];

const CHUNK_PROBLEMS = [
  { value: "overlap", label: "Overlap with another chunk" },
  { value: "split_needed", label: "Needs to be split" },
  { value: "merge_needed", label: "Needs to merge with another chunk" },
  { value: "wrong_boundary", label: "Boundary in the wrong place" },
];

// Procedure-level verdicts — scope-explicit so a new analyst can tell what
// each one acts on (the whole procedure) vs. the per-technique ✎/✕ controls
// on the technique chips. Each verdict maps onto the unchanged backend wire
// contract (action + reject_reason) via verdictOf / setVerdict:
//   approve  -> action "approve" (or "edit" if techniques were changed)
//   discard  -> action "remove"
//   remap    -> action "reject" + a technique reason  -> extract_techniques
//   rechunk  -> action "reject" + "bad_chunk_boundary" -> chunk_behaviors
const VERDICTS = [
  { key: "approve", label: "Approve", rerun: false,
    desc: "Keep it; your technique edits ship with the procedure." },
  { key: "discard", label: "Discard procedure", rerun: false,
    desc: "Drop this procedure from the bundle." },
  { key: "remap", label: "Re-map techniques", rerun: true,
    desc: "Have the AI redo the technique mapping. Re-runs extraction for the whole source — every procedure regenerates for re-review." },
  { key: "rechunk", label: "Re-chunk", rerun: true,
    desc: "The chunk boundaries are wrong (split / merge). Rebuilds chunking for the whole source — every procedure regenerates." },
];

// Single source for verdict theming — the card border, radio styles, flow
// view, and tally all derive from this one map. (They were previously five
// hand-maintained structures that had already drifted: aqua vs bright-aqua
// borders, fg2 vs gray discard text.) `border` is the muted card shade,
// `borderBright` the emphasized flow-node shade.
const VERDICT_THEME = {
  approve: {
    border: "border-gb-green", borderBright: "border-gb-bright-green",
    bg: "bg-gb-tag-reliability-bg", text: "text-gb-bright-green",
    accent: "accent-gb-bright-green", short: "approve",
  },
  discard: {
    border: "border-gb-bg2", borderBright: "border-gb-gray",
    bg: "bg-gb-bg1", text: "text-gb-gray",
    accent: "accent-gb-gray", short: "discard",
  },
  remap: {
    border: "border-gb-aqua", borderBright: "border-gb-bright-aqua",
    bg: "bg-gb-bg0-s", text: "text-gb-bright-aqua",
    accent: "accent-gb-bright-aqua", short: "re-map",
  },
  rechunk: {
    border: "border-gb-orange", borderBright: "border-gb-bright-orange",
    bg: "bg-gb-tag-gate-bg", text: "text-gb-bright-orange",
    accent: "accent-gb-bright-orange", short: "re-chunk",
  },
};

// Derived views of VERDICT_THEME, kept so render sites stay terse.
const VERDICT_STYLES = Object.fromEntries(
  Object.entries(VERDICT_THEME).map(([k, t]) => [
    k, { on: `${t.border} ${t.bg}`, text: t.text, accent: t.accent },
  ])
);
const VERDICT_LABEL = Object.fromEntries(
  Object.entries(VERDICT_THEME).map(([k, t]) => [k, t.short])
);

/** Colors for verdicts in the flow view (bright border + theme bg). */
const ACTION_BORDER_COLORS = Object.fromEntries(
  Object.entries(VERDICT_THEME).map(([k, t]) => [k, t.borderBright])
);
const ACTION_BG_COLORS = Object.fromEntries(
  Object.entries(VERDICT_THEME).map(([k, t]) => [k, t.bg])
);

/** Validate sequence overrides against the current decisions map.
 *
 * Returns a map of draft_id -> list of human-readable issue strings. Only
 * populated for rows that have at least one problem. Two failure modes
 * surfaced (A + validator scope):
 *   - Dangling predecessor: a predecessor index that doesn't match any
 *     existing row's sequence_index.
 *   - Forward predecessor: predecessor_index >= current row's
 *     sequence_index — violates DAG ordering (step cannot depend on
 *     itself or on a later step).
 *
 * We don't block submit on errors — the analyst may be mid-edit — we
 * just paint the offending inputs and show a count above the table.
 */
function validateSequenceOverrides(decisions, nodeIds) {
  const seqToId = {};
  nodeIds.forEach((id) => {
    const seq = decisions[id]?.sequence_index;
    if (seq != null) seqToId[seq] = id;
  });
  const errors = {};
  nodeIds.forEach((id) => {
    const d = decisions[id];
    if (!d) return;
    const seq = d.sequence_index;
    const issues = [];
    (d.predecessor_indices ?? []).forEach((p) => {
      if (!(p in seqToId)) {
        issues.push(`predecessor ${p} does not exist`);
      } else if (p >= seq) {
        issues.push(`predecessor ${p} is not earlier than ${seq}`);
      }
    });
    if (issues.length > 0) errors[id] = issues;
  });
  return errors;
}

/** Initialize decision for a draft. */
function initDecision(draft) {
  return {
    draft_id: draft.id ?? draft.draft_id,
    action: "approve",
    reject_reason: null,
    rationale: "",
    // Chunk feedback fields (for BAD_CHUNK_BOUNDARY rejections)
    chunk_problem: null,
    related_draft_id: null,
    chunk_guidance: "",
    // Sequence fields (for flow view)
    sequence_index: draft.sequence_index ?? 0,
    predecessor_indices: draft.predecessor_indices ?? [],
    branch_point: draft.branch_point ?? false,
    convergence_point: draft.convergence_point ?? false,
  };
}

/** Comma-separated predecessor list, committed on blur or Enter.
 *
 *  Controlled-by-parsed-value could not be typed into: "1," parsed to [1]
 *  and re-rendered as "1", so the comma vanished and the next digit made
 *  "12". Local text while focused; the parsed list only leaves on commit. */
function PredecessorsInput({ value, onCommit, className, placeholder, title }) {
  const [text, setText] = useState(() => (value ?? []).join(", "));
  const [focused, setFocused] = useState(false);
  // Follow the parent's value while not editing (an AI apply, an undo).
  useEffect(() => {
    if (!focused) setText((value ?? []).join(", "));
  }, [value, focused]);

  const commit = () => {
    const indices = text
      .split(",")
      .map((s) => parseInt(s.trim(), 10))
      .filter((n) => !Number.isNaN(n));
    onCommit(indices);
    setText(indices.join(", "));
  };

  return (
    <input
      type="text"
      value={text}
      onChange={(e) => setText(e.target.value)}
      onFocus={() => setFocused(true)}
      onBlur={() => { setFocused(false); commit(); }}
      onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); e.currentTarget.blur(); } }}
      placeholder={placeholder}
      title={title}
      className={className}
    />
  );
}

export default function Gate1Review({
  payload,
  onSubmit,
  submitting,
  // AI reviewer additions. All optional — absent means this gate ran in
  // plain review mode and the panel behaves exactly as it always has.
  recommendations = null,
  brief = null,
  onSaveBrief = null,
  savingBrief = false,
}) {
  const drafts = payload?.items ?? [];
  const context = payload?.context ?? {};
  const techniqueMappings = context.technique_mappings ?? {};
  // C+A+D review lane: possible-bucket picks the technique-extraction step
  // parked for analyst inspection. Keyed by chunk_id; each pick has the same
  // shape as a regular technique mapping plus `confidence_bucket: "possible"`.
  // The analyst can promote a pick into the bundle by clicking its toggle —
  // promoted picks ride along on the Gate 1 submit body's `promotions` array.
  const techniqueMappingsForReview = context.technique_mappings_for_review ?? {};
  const chunks = context.chunks ?? [];

  // Chunk lookup: chunk_id -> chunk object (for showing source text + overlap data)
  const chunkLookup = useMemo(() => {
    const map = {};
    chunks.forEach((c) => { map[c.chunk_id] = c; });
    return map;
  }, [chunks]);

  // Reverse lookup: chunk_id -> draft_id (for overlap badge "overlaps with <draft name>")
  const chunkToDraft = useMemo(() => {
    const map = {};
    drafts.forEach((d) => {
      if (d.chunk_id) map[d.chunk_id] = d;
    });
    return map;
  }, [drafts]);

  const [activeView, setActiveView] = useState("cards");
  // Track which draft cards have the "Source Text" section expanded
  const [expandedChunks, setExpandedChunks] = useState(() => new Set());
  const [decisions, setDecisions] = useState(() => {
    const map = {};
    drafts.forEach((d) => {
      const dec = initDecision(d);
      map[dec.draft_id] = dec;
    });
    return map;
  });

  // ATT&CK technique catalogue for search/autocomplete
  const [catalogue, setCatalogue] = useState([]);
  useEffect(() => {
    fetchTechniques().then(setCatalogue).catch(console.error);
  }, []);

  // Per-draft technique arrays (mutable copy, keyed by draft_id)
  const [draftTechniques, setDraftTechniques] = useState(() => {
    const map = {};
    drafts.forEach((d) => {
      const did = d.id ?? d.draft_id;
      map[did] = [...(d.techniques ?? [])];
    });
    return map;
  });

  // Track which drafts have had techniques actually modified by the analyst
  const [techniquesModified, setTechniquesModified] = useState(() => new Set());
  // Tracks which drafts the analyst actually touched on the sequence/flow
  // editor (drag-reorder, predecessor edit, branch/converge toggle). Without
  // this, every reject submission would emit the form's default sequence
  // values as "edits" and pollute the audit trail.
  const [sequenceModified, setSequenceModified] = useState(() => new Set());
  const SEQUENCE_FIELDS = new Set([
    "sequence_index", "predecessor_indices", "branch_point", "convergence_point",
  ]);

  // Promoted possible-bucket picks. Key shape: `${chunkId}|${techniqueId}`.
  // On submit we convert the set to a list of {chunk_id, technique_id}
  // entries the backend's gate_1 node consumes via gate1_promotions.
  const [promotedKeys, setPromotedKeys] = useState(() => new Set());

  // Which AI recommendations the analyst has applied. Separate from
  // `decisions` so "applied" survives a later manual change and the chip can
  // offer an undo.
  const [appliedRecs, setAppliedRecs] = useState(() => new Set());

  // Technique list as it stood just BEFORE a recommendation was applied, so
  // undo restores what the analyst had rather than the extractor's original.
  // Resetting to the original would silently discard their own edits — an
  // undo that destroys unrelated work is worse than no undo.
  const [preApplyTechniques, setPreApplyTechniques] = useState({});

  /** draft_id -> recommendation, and `${chunk}|${tid}` -> promotion rec. */
  const recsByDraft = useMemo(() => indexByDraft(recommendations), [recommendations]);
  const recsByPromotion = useMemo(
    () => indexByPromotion(recommendations), [recommendations],
  );

  /** How many recommendations a bulk accept would apply. */
  const bulkAcceptableCount = useMemo(
    () =>
      (recommendations?.drafts ?? []).filter(isBulkAcceptable).length +
      (recommendations?.promotions ?? []).filter(isBulkAcceptable).length,
    [recommendations],
  );

  /** Toggle a possible-bucket pick's promoted state. */
  const togglePromote = useCallback((chunkId, techniqueId) => {
    const key = `${chunkId}|${techniqueId}`;
    setPromotedKeys((prev) => {
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  /** Update techniques for a draft and mark it as edited. */
  const updateTechniques = useCallback((draftId, techniques) => {
    setDraftTechniques((prev) => ({ ...prev, [draftId]: techniques }));
    setTechniquesModified((prev) => new Set(prev).add(draftId));
    // Auto-set action to "edit" when techniques are modified
    setDecisions((prev) => {
      const d = prev[draftId];
      if (!d) return prev;
      return {
        ...prev,
        [draftId]: {
          ...d,
          action: d.action === "approve" ? "edit" : d.action,
        },
      };
    });
  }, []);

  /** Update a single decision field (reject reason, chunk problem,
   *  rationale, or a sequence field in the flow editor). */
  const updateDecision = useCallback((draftId, field, value) => {
    if (SEQUENCE_FIELDS.has(field)) {
      // Analyst actually touched a sequence field — flag this draft so
      // handleSubmit emits sequence overrides for it.
      setSequenceModified((prev) => new Set(prev).add(draftId));
    }
    setDecisions((prev) => ({
      ...prev,
      [draftId]: { ...prev[draftId], [field]: value },
    }));
  }, []);

  /** Set the procedure-level verdict, translating it to the backend wire
   *  fields (action + reject_reason). Re-chunk pre-fills the chunk-problem /
   *  related-draft from overlap detection so the analyst doesn't re-enter it. */
  const setVerdict = useCallback((draftId, verdict) => {
    setDecisions((prev) => {
      const d = prev[draftId];
      if (!d) return prev;
      const updated = { ...d };
      if (verdict === "approve") {
        // Keep "edit" if techniques were already changed, else plain approve.
        updated.action = techniquesModified.has(draftId) ? "edit" : "approve";
        updated.reject_reason = null;
        updated.chunk_problem = null;
        updated.related_draft_id = null;
      } else if (verdict === "discard") {
        updated.action = "remove";
        updated.reject_reason = null;
      } else if (verdict === "remap") {
        updated.action = "reject";
        // Preserve an existing technique reason; never bad_chunk_boundary here.
        updated.reject_reason =
          updated.reject_reason && updated.reject_reason !== "bad_chunk_boundary"
            ? updated.reject_reason
            : "wrong_technique";
        updated.chunk_problem = null;
      } else if (verdict === "rechunk") {
        updated.action = "reject";
        updated.reject_reason = "bad_chunk_boundary";
        if (!updated.chunk_problem) {
          const draftObj = drafts.find((x) => (x.id ?? x.draft_id) === draftId);
          const chunk = draftObj ? chunkLookup[draftObj.chunk_id] : null;
          const overlaps = chunk?.potential_overlaps ?? [];
          if (overlaps.length > 0) {
            updated.chunk_problem = "overlap";
            const best = overlaps.reduce((a, b) => (a.score >= b.score ? a : b));
            const rel = chunkToDraft[best.chunk_id];
            if (rel) updated.related_draft_id = rel.id ?? rel.draft_id;
          } else {
            updated.chunk_problem = "wrong_boundary";
          }
        }
      }
      return { ...prev, [draftId]: updated };
    });
  }, [techniquesModified, drafts, chunkLookup, chunkToDraft]);

  /** Apply one AI recommendation to a draft.
   *
   * Two separable parts: the verdict, and any technique removals. They are
   * applied together here but SCORED separately in the outcome diff — "keep
   * this draft but drop T1105" is two claims, and an analyst who keeps the
   * draft while restoring the technique has agreed with one and overridden
   * the other.
   */
  const handleApplyRec = useCallback((draftId) => {
    const rec = recsByDraft[draftId];
    if (!rec) return;

    const drops = rec.remove_technique_ids ?? [];
    if (drops.length) {
      const current = draftTechniques[draftId] ?? [];
      setPreApplyTechniques((prev) =>
        draftId in prev ? prev : { ...prev, [draftId]: current },
      );
      const kept = current.filter((t) => !drops.includes(t.technique_id));
      // Only touch the list if something actually goes — updateTechniques
      // flips the draft to "edit", and doing that for a no-op would show the
      // analyst a change that isn't there.
      if (kept.length !== current.length) updateTechniques(draftId, kept);
    }

    setVerdict(draftId, verdictForRecommendation(rec));
    if (rec.action === "reject" && rec.reject_reason) {
      updateDecision(draftId, "reject_reason", rec.reject_reason);
    }
    // The reason is what the pattern learner reads off a removal; without
    // it the correction log records a blank.
    if (rec.action === "remove" && rec.remove_reason) {
      updateDecision(draftId, "remove_reason", rec.remove_reason);
    }
    // Same rule as applyRecommendation: an analyst's own words win.
    if (rec.rationale && !decisions[draftId]?.rationale) {
      updateDecision(draftId, "rationale", rec.rationale);
    }
    setAppliedRecs((prev) => new Set(prev).add(draftId));
  }, [recsByDraft, draftTechniques, decisions, updateTechniques, setVerdict, updateDecision]);

  /** Undo an applied recommendation: back to approve with the original
   *  technique list. */
  const handleDismissRec = useCallback((draftId) => {
    if (draftId in preApplyTechniques) {
      updateTechniques(draftId, preApplyTechniques[draftId]);
      setPreApplyTechniques((prev) => {
        const next = { ...prev };
        delete next[draftId];
        return next;
      });
    }
    setVerdict(draftId, "approve");
    // Clear only the rationale the recommendation supplied. Blanking the
    // field unconditionally would delete the analyst's own note as the price
    // of undoing an AI suggestion.
    const rec = recsByDraft[draftId];
    if (rec?.rationale && decisions[draftId]?.rationale === rec.rationale) {
      updateDecision(draftId, "rationale", "");
    }
    if (rec?.remove_reason && decisions[draftId]?.remove_reason === rec.remove_reason) {
      updateDecision(draftId, "remove_reason", null);
    }
    setAppliedRecs((prev) => {
      const next = new Set(prev);
      next.delete(draftId);
      return next;
    });
  }, [preApplyTechniques, recsByDraft, decisions, updateTechniques, setVerdict, updateDecision]);

  /** Apply every HIGH-confidence recommendation in one click.
   *
   * Medium and low are deliberately excluded — see the same rule at Gate 0.
   * If one click could accept everything, assist mode becomes autopilot
   * wearing a human's badge and the override signal never materialises. */
  const handleAcceptAI = useCallback(() => {
    (recommendations?.drafts ?? [])
      .filter(isBulkAcceptable)
      .forEach((rec) => handleApplyRec(rec.draft_id));
    (recommendations?.promotions ?? [])
      .filter(isBulkAcceptable)
      .forEach((rec) => {
        if (!promotedKeys.has(`${rec.chunk_id}|${rec.technique_id}`)) {
          togglePromote(rec.chunk_id, rec.technique_id);
        }
      });
  }, [recommendations, handleApplyRec, promotedKeys, togglePromote]);


  /** Decision tally, by procedure-level verdict. */
  const tally = useMemo(() => {
    const counts = { approve: 0, discard: 0, remap: 0, rechunk: 0 };
    Object.values(decisions).forEach((d) => {
      counts[verdictForRecommendation(d)] += 1;
    });
    return counts;
  }, [decisions]);

  /** Build DAG nodes from decisions (sorted by sequence_index). */
  const dagNodes = useMemo(() => {
    return drafts
      .map((draft) => {
        const d = decisions[draft.id ?? draft.draft_id];
        return {
          id: draft.id ?? draft.draft_id,
          name: draft.name ?? draft.title ?? "Untitled",
          verdict: verdictForRecommendation(d),
          sequence_index: d?.sequence_index ?? 0,
          predecessor_indices: d?.predecessor_indices ?? [],
          branch_point: d?.branch_point ?? false,
          convergence_point: d?.convergence_point ?? false,
        };
      })
      .sort((a, b) => a.sequence_index - b.sequence_index);
  }, [drafts, decisions]);

  /** Per-row validation errors for the sequence override table.
   *  Keyed by draft id. Empty object = clean. */
  const sequenceErrors = useMemo(
    () => validateSequenceOverrides(decisions, dagNodes.map((n) => n.id)),
    [decisions, dagNodes],
  );

  /** Drag-end handler for the sequence override table.
   *  Reorders rows, renumbers SEQs to 1..N contiguous, and remaps every
   *  predecessor_index through the old->new sequence map so DAG edges
   *  survive the shuffle. */
  const handleSequenceDragEnd = useCallback((result) => {
    if (!result.destination) return;
    const src = result.source.index;
    const dst = result.destination.index;
    if (src === dst) return;

    // Reorder a shallow copy of dagNodes to get the new row order.
    const reordered = [...dagNodes];
    const [moved] = reordered.splice(src, 1);
    if (!moved) return;
    reordered.splice(dst, 0, moved);

    // Build old_seq -> new_seq map using the CURRENT decisions (not the
    // stale dagNodes snapshot) so typed edits since last render are honored.
    const oldToNew = {};
    reordered.forEach((node, idx) => {
      const oldSeq = decisions[node.id]?.sequence_index;
      const newSeq = idx + 1;
      if (oldSeq != null) oldToNew[oldSeq] = newSeq;
    });

    setDecisions((prev) => {
      const next = { ...prev };
      reordered.forEach((node, idx) => {
        const d = prev[node.id];
        if (!d) return;
        // Any predecessor whose old seq isn't in the map stays put (defensive
        // — shouldn't happen on a clean drag but covers hand-typed refs that
        // pointed at nonexistent rows before the drag).
        const remappedPreds = (d.predecessor_indices ?? []).map(
          (p) => oldToNew[p] ?? p,
        );
        next[node.id] = {
          ...d,
          sequence_index: idx + 1,
          predecessor_indices: remappedPreds,
        };
      });
      return next;
    });
    // Drag-reorder touches every draft's sequence_index (renumbered 1..N)
    // and predecessor_indices (remapped through old→new). Flag them all so
    // handleSubmit emits the overrides; without this the visual reorder
    // wouldn't survive the submit.
    setSequenceModified((prev) => {
      const next = new Set(prev);
      for (const node of reordered) next.add(node.id);
      return next;
    });
  }, [dagNodes, decisions]);

  /** Whether this submission loops back to an earlier stage (and to which),
   *  vs. proceeds to the next gate.
   *
   *  - bad_chunk_boundary reject -> re-chunk (chunk_behaviors)
   *  - any other reject          -> re-map techniques (extract_techniques);
   *    every reject loops — the verdict model makes "approve + edited
   *    techniques" the only apply-and-proceed fix path
   *  - edit / approve / remove   -> proceed (edits apply, no loop)
   *
   *  This mirrors gate_1's routing so the button label tells the truth.
   *  rerunTarget is the optimistic pipeline status for the Kanban — when a
   *  re-chunk and a re-map are co-submitted, the chunk route wins (same
   *  priority as the backend: wrong chunking implies wrong techniques). */
  const rerunInfo = useMemo(() => {
    let chunkRerun = false;
    let techRerun = false;
    Object.values(decisions).forEach((d) => {
      const v = verdictForRecommendation(d);
      if (v === "rechunk") chunkRerun = true;
      else if (v === "remap") techRerun = true;
    });
    return {
      chunkRerun,
      techRerun,
      willRerun: chunkRerun || techRerun,
      rerunTarget: chunkRerun ? "chunking" : techRerun ? "extracting_techniques" : null,
    };
  }, [decisions]);
  const willRerun = rerunInfo.willRerun;

  /** Submit handler. */
  const handleSubmit = useCallback(() => {
    const reviews = Object.values(decisions).map((d) => {
      const review = { draft_id: d.draft_id, action: d.action };
      if (d.action === "reject" && d.reject_reason) {
        review.reject_reason = d.reject_reason;
      }
      if (d.action === "remove" && d.remove_reason) {
        review.remove_reason = d.remove_reason;
      }
      if (d.rationale) review.rationale = d.rationale;

      // Include technique edits only if the analyst modified them AND the
      // verdict keeps the procedure (approve/edit). Clean split: editing a
      // technique is a fix that ships under Approve. Re-map / Re-chunk
      // discard fixes and let the AI redo the mapping, and Discard drops the
      // procedure — forwarding edits with "remove" would log a phantom
      // "analyst corrected techniques to X" on a draft they deleted.
      if ((d.action === "approve" || d.action === "edit") && techniquesModified.has(d.draft_id)) {
        review.analyst_edits = {
          ...(review.analyst_edits ?? {}),
          techniques: draftTechniques[d.draft_id],
        };
        // Ensure action is at least "edit" if techniques were changed
        if (review.action === "approve") {
          review.action = "edit";
        }
      }

      // Include structured chunk feedback for BAD_CHUNK_BOUNDARY rejections
      if (d.action === "reject" && d.reject_reason === "bad_chunk_boundary") {
        review.analyst_edits = {
          ...(review.analyst_edits ?? {}),
          chunk_problem: d.chunk_problem,
          related_draft_id: d.related_draft_id,
          chunk_guidance: d.chunk_guidance || d.rationale || "",
        };
        // Use chunk_guidance as rationale if no separate rationale set
        if (!review.rationale && d.chunk_guidance) {
          review.rationale = d.chunk_guidance;
        }
      }

      // Include sequence overrides only when the analyst actually touched
      // the sequence/flow editor for this draft. The prior unconditional
      // emission polluted every reject with form-default sequence values.
      // Same keep-the-procedure split as techniques above. The backend
      // applies analyst_edits only under action "edit", so a reorder on an
      // approved draft must ship as an edit or it is silently dropped.
      if ((d.action === "approve" || d.action === "edit") && sequenceModified.has(d.draft_id)) {
        review.analyst_edits = {
          ...(review.analyst_edits ?? {}),
          sequence_index: d.sequence_index,
          predecessor_indices: d.predecessor_indices,
          branch_point: d.branch_point,
          convergence_point: d.convergence_point,
        };
        if (review.action === "approve") review.action = "edit";
      }
      return review;
    });
    // C+A+D review-lane promotions: convert "${chunkId}|${techniqueId}" keys
    // back into the wire-format {chunk_id, technique_id} dicts the backend
    // expects on gate1_promotions.
    const promotions = [...promotedKeys].map((key) => {
      const [chunk_id, technique_id] = key.split("|");
      return { chunk_id, technique_id };
    });

    // Pass the rerun flag AND its target so GateReviewPanel can set the
    // RIGHT optimistic status + flash copy — a remap-only submission loops
    // to extract_techniques, not chunking, and the Kanban card should say
    // so instead of jumping to the Procedure Review column.
    // Body shape matches Gate1Submit ({reviews, promotions}); submitGate1
    // accepts either the legacy array or the new object.
    onSubmit({ reviews, promotions }, {
      willRerun,
      rerunTarget: rerunInfo.rerunTarget,
    });
  }, [
    decisions, draftTechniques, techniquesModified, sequenceModified,
    rerunInfo, willRerun, onSubmit, promotedKeys,
  ]);

  return (
    <div className="flex flex-col gap-4">
      {/* The reviewer's read of the report, if this gate ran in assist mode. */}
      {brief && onSaveBrief && (
        <ReviewerBrief brief={brief} onSave={onSaveBrief} saving={savingBrief} />
      )}

      {/* Only HIGH-confidence recommendations — see handleAcceptAI. */}
      {bulkAcceptableCount > 0 && (
        <div className="flex items-center gap-2">
          <button
            onClick={handleAcceptAI}
            className="px-2.5 py-1 rounded text-[11px] font-semibold font-data bg-gb-purple/15 text-gb-bright-purple border border-gb-purple hover:bg-gb-bg2 transition-colors"
            title="Applies only the AI's high-confidence recommendations. Medium and low stay for you to judge."
          >
            🤖 Accept {bulkAcceptableCount} high-confidence
          </button>
          {recommendations?.overall_notes && (
            <span className="text-[11px] text-gb-fg4 leading-snug">
              {recommendations.overall_notes}
            </span>
          )}
        </div>
      )}

      {/* Context bar */}
      <div className="flex gap-4 text-[12px] text-gb-fg4 font-data">
        <span>Drafts: <strong className="text-gb-fg1">{drafts.length}</strong></span>
        {context.validated_entities?.length > 0 && (
          <span>Entities: <strong className="text-gb-fg1">{context.validated_entities.length}</strong></span>
        )}
        {context.chunks?.length > 0 && (
          <span>Chunks: <strong className="text-gb-fg1">{context.chunks.length}</strong></span>
        )}
      </div>

      {/* View toggle */}
      <div className="flex gap-1 bg-gb-bg1 rounded-md p-0.5 w-fit">
        {["cards", "flow"].map((v) => (
          <button
            key={v}
            onClick={() => setActiveView(v)}
            className={`px-3 py-1 rounded text-[11px] font-semibold capitalize transition-colors ${
              activeView === v
                ? "bg-gb-bg0-s text-gb-bright-yellow"
                : "text-gb-fg4 hover:text-gb-fg1"
            }`}
          >
            {v}
          </button>
        ))}
      </div>

      {/* ── Cards View ─────────────────────────────────── */}
      {activeView === "cards" && (
        <div className="flex flex-col gap-3">
          {drafts.map((draft) => {
            const draftId = draft.id ?? draft.draft_id;
            const d = decisions[draftId];
            if (!d) return null;

            // Backend shape (from app/nodes/llm/drafting.py `_process_drafts`):
            //   techniques: [{ technique_id, technique_name, tactic, confidence }]
            //   kill_chain_phases: [{ kill_chain_name, phase_name }]
            //   platforms: [string]
            //   raw_command_lines: [string]
            const techniques = draft.techniques ?? [];
            const platforms = draft.platforms ?? [];
            const rawCommandLines = draft.raw_command_lines ?? [];

            // Derive tactics from live edited techniques (draftTechniques)
            // so tactic tags update reactively when techniques are
            // added, edited, or removed via the TechniqueEditor.
            const currentTechniques = draftTechniques[draftId] ?? techniques;
            const tactics = Array.from(
              new Set(currentTechniques.map((t) => t.tactic).filter(Boolean))
            );
            const verdict = verdictForRecommendation(d);
            const isEdited = techniquesModified.has(draftId);

            return (
              <div
                key={draftId}
                className={`p-4 rounded-lg border bg-gb-bg0-s ${
                  VERDICT_THEME[verdict].border
                }${verdict === "discard" ? " opacity-60" : ""}`}
              >
                {/* Draft name + edited badge */}
                <div className="flex items-start justify-between gap-2 mb-2">
                  <h3 className="text-[13px] font-semibold text-gb-fg0">
                    {draft.name ?? draft.title ?? "Untitled Draft"}
                  </h3>
                  {isEdited && (
                    <span className="shrink-0 font-data text-[9px] px-1.5 py-0.5 rounded bg-gb-bright-orange/15 text-gb-bright-orange border border-gb-orange">
                      edited
                    </span>
                  )}
                </div>

                {/* Overlap warning badge (Layer 2 detection: lexical + entity-aware) */}
                {(() => {
                  const chunk = chunkLookup[draft.chunk_id];
                  const overlaps = chunk?.potential_overlaps ?? [];
                  if (overlaps.length === 0) return null;

                  // Resolve overlapping chunk IDs to draft names
                  const overlapItems = overlaps.map((o) => {
                    const otherDraft = chunkToDraft[o.chunk_id];
                    const name = otherDraft?.name ?? otherDraft?.title ?? o.chunk_id;
                    return {
                      name,
                      score: o.score,
                      shared_tokens: o.shared_tokens ?? [],
                      shared_entities: o.shared_entities ?? [],
                      detection: o.detection ?? "lexical",
                    };
                  });

                  return (
                    <div className="mb-3 px-3 py-2 rounded-lg bg-gb-tag-gate-bg border border-gb-orange">
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-gb-bright-orange text-[13px]">⚠</span>
                        <span className="text-[11px] font-semibold text-gb-bright-orange font-data">
                          Potential overlap detected
                        </span>
                      </div>
                      {overlapItems.map((o, idx) => (
                        <div key={idx} className="ml-5 mb-1">
                          <p className="text-[11px] text-gb-fg3 font-data">
                            {Math.round(o.score * 100)}% similar to <strong className="text-gb-fg1">{o.name}</strong>
                            <span className={`ml-1.5 text-[9px] px-1 py-0.5 rounded ${
                              o.detection === "entity"
                                ? "bg-gb-purple/20 text-gb-bright-purple border border-gb-purple"
                                : "bg-gb-yellow/20 text-gb-bright-yellow border border-gb-yellow"
                            }`}>
                              {o.detection === "entity" ? "entity match" : "lexical"}
                            </span>
                          </p>
                          {o.shared_entities.length > 0 && (
                            <p className="text-[10px] text-gb-fg4 font-data mt-0.5">
                              Shared: {o.shared_entities.join(", ")}
                            </p>
                          )}
                          {o.shared_entities.length === 0 && o.shared_tokens.length > 0 && (
                            <p className="text-[10px] text-gb-fg4 font-data mt-0.5">
                              Tokens: {o.shared_tokens.slice(0, 6).join(", ")}{o.shared_tokens.length > 6 ? "..." : ""}
                            </p>
                          )}
                        </div>
                      ))}
                    </div>
                  );
                })()}

                {/* Narrative / description */}
                {draft.description && (
                  <p className="text-[12px] text-gb-fg3 leading-relaxed mb-3">
                    {draft.description}
                  </p>
                )}

                {/* Source chunk text (collapsible) — split into the
                    chunker's narrative summary AND the verbatim
                    source_excerpt. Provenance badge tells the analyst
                    whether the excerpt is prose, code, vision-extracted
                    figure, or paraphrased away. */}
                {chunkLookup[draft.chunk_id] && (
                  <div className="mb-3">
                    <button
                      onClick={() => setExpandedChunks((prev) => {
                        const next = new Set(prev);
                        next.has(draftId) ? next.delete(draftId) : next.add(draftId);
                        return next;
                      })}
                      className="font-data text-[10px] font-semibold text-gb-fg4 uppercase tracking-wider hover:text-gb-fg1 transition-colors inline-flex items-center gap-2"
                    >
                      <span>
                        {expandedChunks.has(draftId) ? "▾" : "▸"} Source chunk
                      </span>
                      <ProvenanceBadge
                        provenance={chunkLookup[draft.chunk_id].source_provenance}
                        compact
                      />
                    </button>
                    {expandedChunks.has(draftId) && (
                      <div className="mt-1 p-2.5 rounded bg-gb-bg0 border border-gb-bg1 max-h-64 overflow-y-auto space-y-2">
                        <div>
                          <p className="font-data text-[9px] uppercase tracking-wider text-gb-fg4 mb-0.5">
                            Chunk summary (LLM)
                          </p>
                          <p className="text-[11px] text-gb-fg3 leading-relaxed whitespace-pre-wrap">
                            {chunkLookup[draft.chunk_id].text}
                          </p>
                        </div>
                        {chunkLookup[draft.chunk_id].source_excerpt && (
                          <div className="pt-2 border-t border-gb-bg1">
                            <p className="font-data text-[9px] uppercase tracking-wider text-gb-fg4 mb-0.5">
                              Verbatim quote (source)
                            </p>
                            <p className="text-[11px] text-gb-fg2 leading-relaxed whitespace-pre-wrap font-data">
                              {chunkLookup[draft.chunk_id].source_excerpt}
                            </p>
                          </div>
                        )}
                        {chunkLookup[draft.chunk_id].context?.actor && (
                          <p className="text-[10px] text-gb-fg4 font-data pt-1.5 border-t border-gb-bg1">
                            Actor: {chunkLookup[draft.chunk_id].context.actor}
                            {chunkLookup[draft.chunk_id].context.malware?.length > 0 && (
                              <> | Malware: {chunkLookup[draft.chunk_id].context.malware.join(", ")}</>
                            )}
                            {chunkLookup[draft.chunk_id].context.tools?.length > 0 && (
                              <> | Tools: {chunkLookup[draft.chunk_id].context.tools.join(", ")}</>
                            )}
                          </p>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* Technique editor (inline tags with add/edit/remove).
                    Heading spells out the scope: these controls act on
                    individual techniques and ship — distinct from the
                    procedure-level verdict below. */}
                <div className="mb-3">
                  <p className="text-[10px] font-data font-semibold text-gb-fg4 uppercase tracking-wider mb-1">
                    Techniques{" "}
                    <span className="normal-case font-normal text-gb-gray">
                      — fix (✎) or drop (✕) a single mapping here; ships, no re-run
                    </span>
                  </p>
                  <TechniqueEditor
                    techniques={draftTechniques[draftId] ?? techniques}
                    onChange={(updated) => updateTechniques(draftId, updated)}
                    catalogue={catalogue}
                  />
                </div>

                {/* Review for Inclusion lane: C+A+D possible-bucket picks
                    that the technique-extraction step parked for analyst
                    review. Each entry has a Promote toggle that lifts it
                    into the bundle on submit. */}
                {(techniqueMappingsForReview[draft.chunk_id] ?? []).length > 0 && (
                  <div className="mb-3 rounded border border-gb-yellow/40 bg-gb-yellow/5 p-2.5">
                    <p className="text-[10px] font-data font-semibold text-gb-bright-yellow uppercase tracking-wider mb-1.5">
                      Review for Inclusion ({techniqueMappingsForReview[draft.chunk_id].length})
                      <InfoDot term="review-for-inclusion" />
                    </p>
                    {/* "Possible-bucket picks" named an internal confidence
                        tier the analyst never sees anywhere else in the UI. */}
                    <p className="text-[10px] text-gb-fg3 mb-2 leading-snug">
                      The AI judged these plausible but could not tie them
                      confidently to what this procedure sets out to do.
                      Promote any that belong in the bundle.
                    </p>
                    <div className="flex flex-col gap-1.5">
                      {techniqueMappingsForReview[draft.chunk_id].map((p) => {
                        const key = `${draft.chunk_id}|${p.technique_id}`;
                        const isPromoted = promotedKeys.has(key);
                        return (
                          <div
                            key={key}
                            className={`flex items-center justify-between gap-2 rounded px-2 py-1.5 ${
                              isPromoted
                                ? "bg-gb-bright-aqua/10 border border-gb-aqua"
                                : p.denylisted
                                  ? "bg-gb-bright-red/10 border border-gb-red"
                                  : "bg-gb-bg0 border border-gb-bg1"
                            }`}
                          >
                            <div className="min-w-0 flex-1">
                              <div className="flex items-center gap-2 text-[11px] font-data">
                                <span className="text-gb-bright-yellow font-semibold">{p.technique_id}</span>
                                <span className="text-gb-fg2 truncate">{p.technique_name}</span>
                                <span className="text-gb-fg4 text-[10px]">{p.tactic}</span>
                                {p.denylisted && (
                                  <span
                                    className="text-[10px] px-1 py-0.5 rounded bg-gb-bright-red/15 text-gb-bright-red border border-gb-red"
                                    title={p.denylist_reason || hint("technique-denylist")}
                                  >
                                    ⊘ denylist
                                  </span>
                                )}
                                <span className="text-gb-fg4 text-[10px] ml-auto">conf {Number(p.confidence ?? 0).toFixed(2)}</span>
                              </div>
                              {p.source_quote && (
                                <p className="text-[10px] text-gb-fg3 mt-0.5 italic">&ldquo;{p.source_quote}&rdquo;</p>
                              )}
                              {p.rationale && (
                                <p className="text-[10px] text-gb-fg4 mt-0.5">{p.rationale}</p>
                              )}
                            </div>
                            <button
                              type="button"
                              onClick={() => togglePromote(draft.chunk_id, p.technique_id)}
                              className={`shrink-0 px-2 py-0.5 rounded text-[10px] font-data font-semibold transition-colors ${
                                isPromoted
                                  ? "bg-gb-bright-aqua text-gb-bg0 hover:bg-gb-aqua"
                                  : "bg-gb-bg1 text-gb-fg2 hover:bg-gb-bg2 border border-gb-bg2"
                              }`}
                              title={
                                isPromoted
                                  ? "Click to undo promotion"
                                  : p.denylisted
                                    ? `${hint("promote-technique")} Overrides the denylist, for this source only.`
                                    : hint("promote-technique")
                              }
                            >
                              {isPromoted ? "↑ promoted" : "promote"}
                            </button>
                          </div>
                        );
                      })}
                      {/* The AI's promotion recommendations for this chunk,
                          rendered under the lane they act on. Evidence only —
                          the promote buttons above are the action. */}
                      {techniqueMappingsForReview[draft.chunk_id]
                        .filter((p) => recsByPromotion[`${draft.chunk_id}|${p.technique_id}`])
                        .map((p) => (
                          <SuggestionChip
                            key={`rec-${draft.chunk_id}-${p.technique_id}`}
                            rec={recsByPromotion[`${draft.chunk_id}|${p.technique_id}`]}
                          />
                        ))}
                    </div>
                  </div>
                )}

                {/* Tags: tactics, platforms */}
                <div className="flex flex-wrap gap-1.5 mb-3">
                  {tactics.map((t) => (
                    <span key={t} className="font-data text-[10px] px-1.5 py-0.5 rounded bg-gb-tag-channel-bg text-gb-bright-purple border border-gb-purple">
                      {t}
                    </span>
                  ))}
                  {platforms.map((p) => (
                    <span key={p} className="font-data text-[10px] px-1.5 py-0.5 rounded bg-gb-tag-reliability-bg text-gb-bright-green border border-gb-green">
                      {p}
                    </span>
                  ))}
                </div>

                {/* Command lines (verbatim from source, pre-serialization).
                    These are the operative strings the analyst needs to see
                    to judge technique mapping correctness. */}
                {rawCommandLines.length > 0 && (
                  <div className="mb-3">
                    <p className="text-[10px] font-data font-semibold text-gb-fg4 uppercase tracking-wider mb-1">
                      Command lines ({rawCommandLines.length})
                    </p>
                    <div className="rounded bg-gb-bg0 border border-gb-bg1 p-2 max-h-40 overflow-y-auto">
                      {rawCommandLines.map((cmd, i) => (
                        <pre
                          key={i}
                          className="font-data text-[11px] text-gb-fg1 whitespace-pre-wrap break-all leading-relaxed"
                        >
                          {cmd}
                        </pre>
                      ))}
                    </div>
                  </div>
                )}

                {/* Confidence + detail gap indicator */}
                <div className="flex items-center gap-3 mb-3">
                  {draft.confidence != null && (
                    <p className="text-[11px] font-data text-gb-fg4">
                      Confidence: <strong className="text-gb-fg1">{draft.confidence}%</strong>
                    </p>
                  )}
                  {draft.detail_gap && (
                    <span
                      title="Source lacks specificity for this behavior. Consider rejecting or requesting clarification."
                      className="font-data text-[10px] px-1.5 py-0.5 rounded bg-gb-tag-gate-bg text-gb-bright-orange border border-gb-orange"
                    >
                      detail gap
                    </span>
                  )}
                </div>

                {/* Procedure-level decision — scope-explicit verdicts. The
                    per-technique ✎/✕ controls above act on a single mapping;
                    these act on the whole procedure. */}
                <div className="pt-3 border-t border-gb-bg1">
                  <p className="text-[10px] font-data font-semibold text-gb-fg4 uppercase tracking-wider mb-1.5">
                    This procedure
                  </p>
                  {/* The AI's verdict for this draft, with its evidence. */}
                  {recsByDraft[draftId] && (
                    <SuggestionChip
                      rec={recsByDraft[draftId]}
                      applied={appliedRecs.has(draftId)}
                      onApply={() => handleApplyRec(draftId)}
                      onDismiss={() => handleDismissRec(draftId)}
                    />
                  )}
                  <div className="flex flex-col gap-1">
                    {VERDICTS.map((v) => {
                      const selected = verdict === v.key;
                      const styles = VERDICT_STYLES[v.key];
                      return (
                        <label
                          key={v.key}
                          className={`flex items-start gap-2 px-2 py-1.5 rounded border cursor-pointer transition-colors ${
                            selected ? styles.on : "border-gb-bg2 bg-gb-bg0 hover:border-gb-fg4"
                          }`}
                        >
                          <input
                            type="radio"
                            name={`verdict-${draftId}`}
                            checked={selected}
                            onChange={() => setVerdict(draftId, v.key)}
                            className={`mt-0.5 ${styles.accent}`}
                          />
                          <span className="min-w-0">
                            <span className="flex items-center gap-1.5 flex-wrap">
                              <span className={`text-[11px] font-semibold ${selected ? styles.text : "text-gb-fg2"}`}>
                                {v.label}
                              </span>
                              {v.rerun && (
                                <span className="font-data text-[9px] px-1 py-0.5 rounded bg-gb-bright-orange/15 text-gb-bright-orange border border-gb-orange">
                                  re-runs · slow
                                </span>
                              )}
                            </span>
                            <span className="block text-[10px] text-gb-fg4 leading-snug mt-0.5">
                              {v.desc}
                            </span>
                          </span>
                        </label>
                      );
                    })}
                  </div>

                  {/* Clean split: only Approve ships inline technique fixes.
                      A send-back has the AI redo the mapping; a discard drops
                      the procedure entirely. Warn whenever edits won't ship. */}
                  {verdict !== "approve" && isEdited && (
                    <p className="mt-1.5 text-[10px] text-gb-bright-yellow leading-snug">
                      ⚠ Your inline technique edits won&apos;t be sent —{" "}
                      {verdict === "discard"
                        ? "discarding drops the whole procedure."
                        : "a send-back has the AI redo the mapping."}{" "}
                      Choose <strong>Approve</strong> to keep your edits.
                    </p>
                  )}

                  {/* "Why" for Re-map techniques (feeds the AI retry + flywheel) */}
                  {verdict === "remap" && (
                    <div className="mt-2 space-y-2">
                      <select
                        value={d.reject_reason ?? "wrong_technique"}
                        onChange={(e) => updateDecision(draftId, "reject_reason", e.target.value)}
                        className="w-full font-data text-[11px] bg-gb-bg1 text-gb-fg1 border border-gb-bg2 rounded px-2 py-1"
                      >
                        {REMAP_REASONS.map((r) => (
                          <option key={r.value} value={r.value}>{r.label}</option>
                        ))}
                      </select>
                      <input
                        type="text"
                        value={d.rationale}
                        onChange={(e) => updateDecision(draftId, "rationale", e.target.value)}
                        placeholder="What's the right answer / what to fix? (guides the AI's retry + teaches the system)"
                        className="w-full bg-gb-bg1 text-gb-fg1 border border-gb-bg2 rounded px-2 py-1 text-[11px] font-data placeholder-gb-gray"
                      />
                    </div>
                  )}

                  {/* "Why" for Re-chunk */}
                  {verdict === "rechunk" && (
                    <div className="mt-2 p-2.5 bg-gb-bg0 border border-gb-bg2 rounded-lg space-y-2">
                      <p className="text-[10px] font-data font-semibold text-gb-fg4 uppercase tracking-wider">
                        Chunk boundary feedback
                      </p>
                      <select
                        value={d.chunk_problem ?? ""}
                        onChange={(e) => updateDecision(draftId, "chunk_problem", e.target.value || null)}
                        className="w-full font-data text-[11px] bg-gb-bg1 text-gb-fg1 border border-gb-bg2 rounded px-2 py-1"
                      >
                        <option value="">Select problem type...</option>
                        {CHUNK_PROBLEMS.map((p) => (
                          <option key={p.value} value={p.value}>{p.label}</option>
                        ))}
                      </select>
                      {(d.chunk_problem === "overlap" || d.chunk_problem === "merge_needed") && (
                        <select
                          value={d.related_draft_id ?? ""}
                          onChange={(e) => updateDecision(draftId, "related_draft_id", e.target.value || null)}
                          className="w-full font-data text-[11px] bg-gb-bg1 text-gb-fg1 border border-gb-bg2 rounded px-2 py-1"
                        >
                          <option value="">Select related procedure...</option>
                          {drafts
                            .filter((other) => (other.id ?? other.draft_id) !== draftId)
                            .map((other) => {
                              const otherId = other.id ?? other.draft_id;
                              return (
                                <option key={otherId} value={otherId}>
                                  {other.name ?? otherId}
                                </option>
                              );
                            })}
                        </select>
                      )}
                      <input
                        type="text"
                        value={d.chunk_guidance ?? ""}
                        onChange={(e) => updateDecision(draftId, "chunk_guidance", e.target.value)}
                        placeholder="Where should the boundary be? (e.g., 'The DLL sideloading belongs with the next chunk')"
                        className="w-full bg-gb-bg1 text-gb-fg1 border border-gb-bg2 rounded px-2 py-1 text-[11px] font-data placeholder-gb-gray"
                      />
                    </div>
                  )}

                  {/* Optional note for Discard (teaches the flywheel why) */}
                  {verdict === "discard" && (
                    <input
                      type="text"
                      value={d.rationale}
                      onChange={(e) => updateDecision(draftId, "rationale", e.target.value)}
                      placeholder="Why drop it? e.g. not a real procedure, duplicate of another (optional)"
                      className="mt-2 w-full bg-gb-bg1 text-gb-fg1 border border-gb-bg2 rounded px-2 py-1 text-[11px] font-data placeholder-gb-gray"
                    />
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ── Flow View ──────────────────────────────────── */}
      {activeView === "flow" && (
        <div className="flex flex-col gap-4">
          {/* DAG visualization */}
          <div className="p-4 bg-gb-bg0-s border border-gb-bg2 rounded-lg">
            <div className="flex flex-col items-center gap-0">
              {dagNodes.map((node, idx) => (
                <div key={node.id} className="flex flex-col items-center">
                  {/* Arrow from previous node */}
                  {idx > 0 && (
                    <div className="w-px h-6 bg-gb-fg4" />
                  )}

                  {/* Node */}
                  <div
                    className={`relative px-4 py-2.5 rounded-lg border-2 min-w-[200px] text-center ${
                      ACTION_BORDER_COLORS[node.verdict] ?? "border-gb-bg2"
                    } ${ACTION_BG_COLORS[node.verdict] ?? "bg-gb-bg0-s"}`}
                  >
                    <p className="text-[12px] font-semibold text-gb-fg0">{node.name}</p>
                    <span className={`text-[10px] font-data mt-0.5 inline-block ${
                      VERDICT_THEME[node.verdict]?.text ?? "text-gb-gray"
                    }`}>
                      {VERDICT_LABEL[node.verdict] ?? node.verdict}
                    </span>

                    {/* Branch / convergence indicators */}
                    {node.branch_point && (
                      <span className="absolute -right-2 top-1/2 -translate-y-1/2 text-[10px] font-data text-gb-bright-yellow bg-gb-bg0 px-1 rounded border border-gb-bg2">
                        fork
                      </span>
                    )}
                    {node.convergence_point && (
                      <span className="absolute -left-2 top-1/2 -translate-y-1/2 text-[10px] font-data text-gb-bright-aqua bg-gb-bg0 px-1 rounded border border-gb-bg2">
                        join
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Sequence override table */}
          <div className="p-3 bg-gb-bg0-s border border-gb-bg2 rounded-lg">
            <div className="flex items-center justify-between mb-2">
              <h4 className="text-[12px] font-semibold text-gb-fg0">Sequence overrides</h4>
              {Object.keys(sequenceErrors).length > 0 && (
                <span
                  className="text-[10px] font-data text-gb-bright-red"
                  title={Object.entries(sequenceErrors)
                    .map(([id, issues]) => {
                      const n = dagNodes.find((nd) => nd.id === id);
                      return `${n?.name ?? id}: ${issues.join("; ")}`;
                    })
                    .join("\n")}
                >
                  ⚠ {Object.keys(sequenceErrors).length} sequence issue
                  {Object.keys(sequenceErrors).length === 1 ? "" : "s"}
                </span>
              )}
            </div>
            <p className="text-[10px] font-data text-gb-fg4 mb-2">
              Drag rows to reorder. Sequence numbers and predecessor refs
              renumber automatically.
            </p>
            <DragDropContext onDragEnd={handleSequenceDragEnd}>
              <table className="w-full text-[11px]">
                <thead>
                  <tr className="bg-gb-bg0">
                    <th
                      aria-label="Drag handle"
                      className="px-2 py-1.5 border-b border-gb-bg2 w-[24px]"
                    />
                    <th className="text-left px-3 py-1.5 text-[10px] font-semibold text-gb-fg4 uppercase tracking-wider font-data border-b border-gb-bg2">
                      Draft
                    </th>
                    <th className="text-left px-3 py-1.5 text-[10px] font-semibold text-gb-fg4 uppercase tracking-wider font-data border-b border-gb-bg2 w-[60px]">
                      Seq
                    </th>
                    <th className="text-left px-3 py-1.5 text-[10px] font-semibold text-gb-fg4 uppercase tracking-wider font-data border-b border-gb-bg2 w-[120px]">
                      Predecessors
                    </th>
                    <th className="text-center px-3 py-1.5 text-[10px] font-semibold text-gb-fg4 uppercase tracking-wider font-data border-b border-gb-bg2 w-[90px]">
                      Branch?{" "}
                      <span
                        className="inline-flex items-center justify-center w-3.5 h-3.5 rounded-full border border-gb-fg4 text-[9px] text-gb-fg4 cursor-help ml-0.5 align-middle"
                        title={hint("branch-point")}
                      >
                        i
                      </span>
                    </th>
                    <th className="text-center px-3 py-1.5 text-[10px] font-semibold text-gb-fg4 uppercase tracking-wider font-data border-b border-gb-bg2 w-[100px]">
                      Converge?{" "}
                      <span
                        className="inline-flex items-center justify-center w-3.5 h-3.5 rounded-full border border-gb-fg4 text-[9px] text-gb-fg4 cursor-help ml-0.5 align-middle"
                        title={hint("convergence-point")}
                      >
                        i
                      </span>
                    </th>
                  </tr>
                </thead>
                <Droppable droppableId="seq-overrides">
                  {(dropProvided) => (
                    <tbody
                      ref={dropProvided.innerRef}
                      {...dropProvided.droppableProps}
                    >
                      {dagNodes.map((node, idx) => {
                        const d = decisions[node.id];
                        if (!d) return null;
                        const rowErrors = sequenceErrors[node.id] ?? [];
                        const hasError = rowErrors.length > 0;
                        return (
                          <Draggable
                            key={node.id}
                            draggableId={String(node.id)}
                            index={idx}
                          >
                            {(dragProvided, snap) => (
                              <tr
                                ref={dragProvided.innerRef}
                                {...dragProvided.draggableProps}
                                className={
                                  snap.isDragging
                                    ? "bg-gb-bg2 shadow-lg"
                                    : ""
                                }
                              >
                                <td
                                  {...dragProvided.dragHandleProps}
                                  className="px-2 py-2 border-b border-gb-bg1 text-center text-gb-fg4 hover:text-gb-fg1 cursor-grab active:cursor-grabbing select-none font-data"
                                  title="Drag to reorder"
                                >
                                  ⠿
                                </td>
                                <td className="px-3 py-2 border-b border-gb-bg1 text-gb-fg1">
                                  {node.name}
                                </td>
                                <td className="px-3 py-2 border-b border-gb-bg1">
                                  <input
                                    type="number"
                                    value={d.sequence_index}
                                    onChange={(e) =>
                                      updateDecision(
                                        node.id,
                                        "sequence_index",
                                        parseInt(e.target.value, 10) || 0,
                                      )
                                    }
                                    className="w-[40px] text-center bg-gb-bg1 text-gb-fg1 border border-gb-bg2 rounded px-1 py-0.5 font-data text-[11px]"
                                  />
                                </td>
                                <td className="px-3 py-2 border-b border-gb-bg1">
                                  <PredecessorsInput
                                    value={d.predecessor_indices}
                                    onCommit={(indices) =>
                                      updateDecision(
                                        node.id,
                                        "predecessor_indices",
                                        indices,
                                      )
                                    }
                                    placeholder="none"
                                    title={
                                      hasError ? rowErrors.join("; ") : ""
                                    }
                                    className={`w-full bg-gb-bg1 text-gb-fg1 border rounded px-2 py-0.5 font-data text-[11px] placeholder-gb-gray ${
                                      hasError
                                        ? "border-gb-bright-red"
                                        : "border-gb-bg2"
                                    }`}
                                  />
                                </td>
                                <td className="px-3 py-2 border-b border-gb-bg1 text-center">
                                  <input
                                    type="checkbox"
                                    checked={d.branch_point}
                                    onChange={(e) =>
                                      updateDecision(
                                        node.id,
                                        "branch_point",
                                        e.target.checked,
                                      )
                                    }
                                    className="accent-gb-bright-yellow"
                                  />
                                </td>
                                <td className="px-3 py-2 border-b border-gb-bg1 text-center">
                                  <input
                                    type="checkbox"
                                    checked={d.convergence_point}
                                    onChange={(e) =>
                                      updateDecision(
                                        node.id,
                                        "convergence_point",
                                        e.target.checked,
                                      )
                                    }
                                    className="accent-gb-bright-aqua"
                                  />
                                </td>
                              </tr>
                            )}
                          </Draggable>
                        );
                      })}
                      {dropProvided.placeholder}
                    </tbody>
                  )}
                </Droppable>
              </table>
            </DragDropContext>
          </div>
        </div>
      )}

      {/* A re-run loop regenerates EVERY draft downstream, so co-submitted
          approves/discards don't stick — the analyst re-reviews everything
          on the next pass. Say so before they hit submit. */}
      {willRerun && (tally.approve + tally.discard) > 0 && (
        <p className="text-[11px] text-gb-bright-yellow leading-snug pt-2">
          ⚠ A re-run regenerates <strong>every</strong> procedure — your{" "}
          {tally.approve + tally.discard} approve/discard decision
          {tally.approve + tally.discard === 1 ? "" : "s"} will be re-presented
          for review on the next pass.
        </p>
      )}

      {/* Footer: tally + submit */}
      <div className="flex items-center justify-between pt-3 border-t border-gb-bg2">
        <div className="flex gap-3 text-[11px] font-data">
          {VERDICTS.map((v) =>
            (v.key === "approve" || tally[v.key] > 0) && (
              <span key={v.key} className={VERDICT_THEME[v.key].text}>
                <strong>{tally[v.key]}</strong> {VERDICT_THEME[v.key].short}
              </span>
            )
          )}
        </div>
        <button
          onClick={handleSubmit}
          disabled={submitting}
          className={`px-4 py-1.5 rounded-md text-[12px] font-semibold transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
            willRerun
              ? "bg-gb-orange text-gb-bg0-h hover:bg-gb-bright-orange"
              : "bg-gb-green text-gb-bg0-h hover:bg-gb-bright-green"
          }`}
        >
          {submitting
            ? "Submitting..."
            : rerunInfo.chunkRerun
              ? "Reject & Rerun Chunking"
              : rerunInfo.techRerun
                ? "Reject & Re-map Techniques"
                : "Submit Review"}
        </button>
      </div>
    </div>
  );
}
