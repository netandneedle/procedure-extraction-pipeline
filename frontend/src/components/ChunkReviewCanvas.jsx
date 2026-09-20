/**
 * ChunkReviewCanvas — analyst review surface for the chunk-validation gate.
 *
 * Three panes:
 *   left:   source-text with chunk-span highlighting
 *   center: React Flow DAG (chunks + precedes edges)
 *   right:  selected-chunk detail / edit form
 *
 * Full submit flow. Analyst can edit fields, drop chunks,
 * mutate precedes edges, add new chunks the LLM missed, and submit the
 * accumulated review as either Approve (proceed to extract_techniques)
 * or Reject (loop back to chunk_behaviors with feedback).
 *
 * Edit-field whitelist mirrors backend `_CHUNK_EDITABLE_FIELDS` in
 * gates.py: text, source_excerpt, behavioral_confidence, branch_point,
 * convergence_point, chain_root, chain_label. (context has no form field
 * here — it's a free-form dict without a natural form-field surface.)
 *
 * Added-chunk limitation: the backend allocates real chunk_ids in
 * gate_chunks._build_added_chunk based on (next_seq, text), so the
 * frontend can't predict them. We block edge connections that involve
 * a synthetic "add-N" id at handleConnect time. Analyst-added chunks
 * land at the end of the sequence with no precedes edges; if mid-flow
 * insertion is needed, reject the gate with a comment instead.
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  ReactFlow, Background, Controls, MiniMap,
  Handle, Position,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import ProvenanceBadge from "./ProvenanceBadge";
import ReviewerBrief from "./ReviewerBrief";
import SuggestionChip from "./SuggestionChip";
import InfoDot from "./InfoDot";
import { hint } from "../lib/glossary";
import { newId } from "../lib/ids";
import { ellipsize } from "../lib/strings";
import { applyPositions, useLayoutPositions } from "../lib/graphLayout";
import {
  edgeRecKey,
  indexByChunk,
  isBulkAcceptable,
  pendingEdgeRecs,
  restoreChunkSnapshot,
} from "../lib/reviewerSuggestions";

const NODE_WIDTH = 240;
const NODE_HEIGHT = 110;
// Layout tuning for lib/graphLayout.js. 60/110 leaves room for the
// 240px chunk cards without overlap.
const CHUNK_LAYOUT = { rankdir: "TB", nodesep: 60, ranksep: 110, marginx: 20, marginy: 20 };

/** Tactic shortname -> friendly label. */
const TACTIC_LABEL = {
  reconnaissance: "Recon",
  "resource-development": "Resource Dev",
  "initial-access": "Initial Access",
  execution: "Execution",
  persistence: "Persistence",
  "privilege-escalation": "Priv Esc",
  "defense-evasion": "Defense Evasion",
  "credential-access": "Cred Access",
  discovery: "Discovery",
  "lateral-movement": "Lateral Move",
  collection: "Collection",
  "command-and-control": "C2",
  exfiltration: "Exfil",
  impact: "Impact",
};

/**
 * Custom React Flow node for a chunk. Renders the same content as the
 * default node, plus visual states for selected/edited and source/target
 * handles for edge editing.
 *
 * `data` shape: { chunk, isSelected, isEdited }.
 */
function ChunkNode({ data }) {
  const { chunk, isSelected, isEdited, isDropped, isAdded, aiRec } = data;
  const tactic = (chunk.context?.tactics?.[0]) || "";
  const tacticChip = tactic ? TACTIC_LABEL[tactic] || tactic : "";
  const conf = typeof chunk.behavioral_confidence === "number"
    ? Math.round(chunk.behavioral_confidence * 100)
    : 0;

  // Border priority: dropped > selected > added > edited > default.
  const border = isDropped
    ? "border-gb-bright-red ring-1 ring-gb-bright-red/40"
    : isSelected
      ? "border-gb-bright-blue ring-2 ring-gb-bright-blue/40"
      : isAdded
        ? "border-gb-bright-green ring-1 ring-gb-bright-green/30"
        : isEdited
          ? "border-gb-bright-yellow ring-1 ring-gb-bright-yellow/30"
          : "border-gb-bg2";

  // Dropped nodes dim and strike through; the analyst can still click them
  // to undo the drop from the side panel.
  const dim = isDropped ? "opacity-50 line-through decoration-gb-bright-red/70" : "";

  return (
    <div
      className={`rounded-md border bg-gb-bg0-h px-2.5 py-2 transition-colors ${border} ${dim}`}
      style={{ width: NODE_WIDTH, minHeight: NODE_HEIGHT }}
    >
      <Handle type="target" position={Position.Top} style={{ background: "#928374", width: 7, height: 7 }} />
      <div className="flex items-center justify-between mb-0.5 no-underline">
        {tacticChip ? (
          <span className="text-[9px] font-data uppercase tracking-wide text-gb-bright-orange/80 no-underline">
            {tacticChip}
          </span>
        ) : <span />}
        {/* Deterministic flow warnings from the chunker: today, an incoming
            edge that runs from a later tactic to an earlier one — the report's
            exposition order taken as attack order. Warn-only; the analyst
            decides whether it is an inversion or a genuine loop-back. */}
        {Array.isArray(chunk.flow_warnings) && chunk.flow_warnings.length > 0 && (
          <span
            title={chunk.flow_warnings.join("\n")}
            className="text-[9px] font-data uppercase tracking-wide text-gb-bright-yellow no-underline mr-1 cursor-help"
          >
            ⚠ order
          </span>
        )}
        {/* The AI flag rides alongside the analyst's own state rather than
            replacing it: "this is edited AND the reviewer wants it dropped"
            is the case worth seeing at a glance. Approvals are not flagged —
            a badge on every node is a badge on none. */}
        {aiRec && aiRec.action !== "approve" && (
          <span
            title={`AI (${aiRec.confidence}): ${aiRec.action} — ${aiRec.rationale ?? ""}`}
            className="text-[9px] font-data uppercase tracking-wide text-gb-bright-purple no-underline mr-1 cursor-help"
          >
            🤖 {aiRec.action}
          </span>
        )}
        {isDropped ? (
          <span className="text-[9px] font-data uppercase tracking-wide text-gb-bright-red no-underline">
            dropped
          </span>
        ) : isAdded ? (
          <span className="text-[9px] font-data uppercase tracking-wide text-gb-bright-green no-underline">
            added
          </span>
        ) : isEdited ? (
          <span className="text-[9px] font-data uppercase tracking-wide text-gb-bright-yellow no-underline">
            edited
          </span>
        ) : null}
      </div>
      <div className="text-[12px] text-gb-fg1 leading-snug font-medium text-left">
        {ellipsize(chunk.text, 100)}
      </div>
      {chunk.source_excerpt && (
        <div className="text-[10px] text-gb-fg4 italic mt-1 leading-tight text-left">
          {ellipsize(chunk.source_excerpt, 80)}
        </div>
      )}
      <div className="text-[9px] font-data text-gb-fg4 mt-1.5 flex gap-2 items-center no-underline flex-wrap">
        <span>rel: {conf}</span>
        {chunk.branch_point && (
          <span className="text-gb-bright-purple cursor-help" title={hint("branch-point")}>↗ branch</span>
        )}
        {chunk.convergence_point && (
          <span className="text-gb-bright-aqua cursor-help" title={hint("convergence-point")}>↘ converge</span>
        )}
        {chunk.chain_root && (
          <span
            title={chunk.chain_label
              ? `Chain root — begins '${chunk.chain_label}' as a separate attack chain`
              : "Chain root — begins a separate attack chain"}
            className="font-data text-[9px] font-medium px-1.5 py-0.5 rounded border bg-[rgba(180,142,173,0.15)] text-gb-bright-purple border-gb-purple cursor-help"
          >
            🔗 {chunk.chain_label || "Chain root"}
          </span>
        )}
        {!chunk.chain_root && chunk.chain_label && (
          <span
            title={`Part of chain: ${chunk.chain_label}`}
            className="font-data text-[9px] text-gb-bright-purple/70 cursor-help"
          >
            ⛓ {chunk.chain_label}
          </span>
        )}
        {chunk.source_provenance && (
          <ProvenanceBadge provenance={chunk.source_provenance} compact />
        )}
      </div>
      <Handle type="source" position={Position.Bottom} style={{ background: "#928374", width: 7, height: 7 }} />
    </div>
  );
}

const NODE_TYPES = { chunk: ChunkNode };

// Hoisted to module scope so React Flow doesn't see fresh object references
// on every render — otherwise it logs the "nodeTypes/defaultEdgeOptions has
// changed" warning and may invalidate edge memoization.
const FIT_VIEW_OPTIONS = { padding: 0.2 };
const DEFAULT_EDGE_OPTIONS = {
  type: "smoothstep",
  style: { stroke: "#928374", strokeWidth: 1.5 },
};

/**
 * Compute the live edge set from chunk precedes_ids + the analyst's edge
 * mutation log. Returns both the canonical "from->to" key set and an
 * `addedKeys` subset (edges introduced by the analyst, rendered in green
 * dashed style so the diff is visible at a glance).
 *
 * Mutation semantics: edgeOps is replayed in order, so {add A->B} then
 * {remove A->B} cleanly undoes itself, and the same pair can be toggled
 * any number of times. The submission diff collapses opposing
 * pairs against the original edge set before sending.
 */
export function computeLiveEdges(chunks, edgeOps) {
  const original = new Set();
  const seenIds = new Set(chunks.map((c) => c.chunk_id));
  for (const c of chunks) {
    for (const t of c.precedes_ids || []) {
      if (!seenIds.has(t)) continue;
      original.add(`${c.chunk_id}->${t}`);
    }
  }
  // Replay ops against `live` only. `addedKeys` is derived AFTERWARDS as
  // `live − original`. Tracking
  // `added` incrementally during the loop can diverge from set-difference
  // semantics under toggling and mislead the analyst (green-dashed paint
  // on an edge that already exists in the original chunk graph).
  const live = new Set(original);
  for (const op of edgeOps) {
    if (!seenIds.has(op.from) || !seenIds.has(op.to)) continue;
    const key = `${op.from}->${op.to}`;
    if (op.action === "add") live.add(key);
    else if (op.action === "remove") live.delete(key);
  }
  const added = new Set();
  for (const k of live) {
    if (!original.has(k)) added.add(k);
  }
  return { liveKeys: live, addedKeys: added };
}

/**
 * Build React Flow nodes/edges from the chunks list, decorating each node
 * with selection/edited/dropped state and each edge with its add-vs-original
 * provenance for the custom renderers.
 */
function buildGraph(
  chunks,
  { selectedId, selectedEdgeKey, editedIds, droppedIds, edgeOps, recsByChunk = {} },
) {
  const { liveKeys, addedKeys } = computeLiveEdges(chunks, edgeOps);

  // Unpositioned: the component overlays layout positions afterwards, so
  // this (cheap) decoration can run on every click and keystroke while the
  // dagre pass runs only when the ids or edges change.
  const nodes = chunks.map((c) => ({
    id: c.chunk_id,
    type: "chunk",
    width: NODE_WIDTH,
    height: NODE_HEIGHT,
    data: {
      chunk: c,
      isSelected: c.chunk_id === selectedId,
      isEdited: editedIds.has(c.chunk_id),
      isDropped: droppedIds.has(c.chunk_id),
      isAdded: !!c._isAdded,
      aiRec: recsByChunk[c.chunk_id] || null,
    },
  }));

  const edges = Array.from(liveKeys).map((key) => {
    const [source, target] = key.split("->");
    const isAdded = addedKeys.has(key);
    const isSelected = key === selectedEdgeKey;
    let style;
    if (isSelected) {
      // Selected: bright orange + thicker so the click target is unambiguous.
      // Keeps the dashed pattern when also-added so analyst still knows.
      style = isAdded
        ? { stroke: "#fe8019", strokeWidth: 3, strokeDasharray: "5 4" }
        : { stroke: "#fe8019", strokeWidth: 3 };
    } else if (isAdded) {
      style = { stroke: "#b8bb26", strokeWidth: 1.8, strokeDasharray: "5 4" };
    } else {
      style = { stroke: "#928374", strokeWidth: 1.5 };
    }
    return {
      id: isAdded ? `e+${key}` : `e-${key}`,
      source,
      target,
      type: "smoothstep",
      animated: false,
      selected: isSelected,
      style,
    };
  });

  return { nodes, edges };
}

/**
 * Build a one-pass normalization index for parsedText. Used by flexibleFind
 * to map whitespace-collapsed offsets back to original-text offsets.
 *
 * Returns `{normalized, map}` where:
 *  - `normalized` is parsedText with every run of whitespace collapsed to
 *    a single space.
 *  - `map[i]` is the original-text index of normalized character i. The
 *    map has length `normalized.length + 1`; the trailing entry is a
 *    sentinel equal to parsedText.length for end-position lookup.
 *
 * Cost: O(n) once per parsedText. SourcePane wraps this in useMemo so the
 * index lives with the component, not as a module-scoped Map (a
 * module-scoped cache leaked across panels and churned memory).
 */
function buildNormIndex(parsedText) {
  if (!parsedText) return { normalized: "", map: [0] };
  const norm = [];
  const map = [];
  let inSpace = false;
  for (let i = 0; i < parsedText.length; i++) {
    const ch = parsedText[i];
    if (/\s/.test(ch)) {
      if (!inSpace) {
        norm.push(" ");
        map.push(i);
        inSpace = true;
      }
    } else {
      norm.push(ch);
      map.push(i);
      inSpace = false;
    }
  }
  map.push(parsedText.length);
  return { normalized: norm.join(""), map };
}

/**
 * Whitespace-flexible substring search.
 *
 * Backend `_finalize_chunks` uses `parsed_text.find(excerpt)` which is
 * strict-match. PDF parsing introduces hard line breaks (Docling reflow
 * artifacts) that don't appear in the LLM-emitted excerpt, so the strict
 * find returns -1 even when the excerpt is verbatim "by reading."
 *
 * Three strategies, in order:
 *  1. Strict find on whitespace-normalized strings.
 *  2. Prefix-anchor + TAIL-ANCHOR verification. The LLM frequently
 *     paraphrases the tail of an excerpt; we accept a prefix match only
 *     if a short tail snippet also appears nearby. Without the tail
 *     check the returned span can blanket unrelated content past the
 *     real chunk boundary.
 *  3. Short literal-anchor fallback. Caps the returned span to a small
 *     fixed window so we never report a span that runs past the actual
 *     passage.
 *
 * Returns [start, end] in original-text coordinates, or null on miss.
 *
 * `normIndex` is the precomputed `{normalized, map}` from buildNormIndex.
 * Passing it explicitly (rather than caching inside the function) keeps
 * lifetime tied to the calling component — SourcePane wraps it in
 * useMemo so the index goes away when the gate-review panel closes.
 */
function flexibleFind(parsedText, normIndex, excerpt) {
  if (!parsedText || !excerpt) return null;
  const cached =
    normIndex && normIndex.normalized !== undefined
      ? normIndex
      : buildNormIndex(parsedText);
  const normNeedle = excerpt.replace(/\s+/g, " ").trim();
  if (!normNeedle) return null;

  // Strategy 1: strict find on whitespace-normalized strings.
  let idx = cached.normalized.indexOf(normNeedle);
  let matchedLen = normNeedle.length;

  // Strategy 2: prefix-anchor with TAIL VERIFICATION. Try shrinking
  // prefixes from 95% down to 50% in 5% steps; for each candidate,
  // require a short tail anchor (last 30 chars / 20% of needle) to
  // also appear within a 1.5× needle-length window past the prefix
  // start. The span is trimmed to where the tail actually ends, so
  // we never blanket unverified bytes between prefix and tail.
  if (idx === -1) {
    const minLen = Math.max(40, Math.floor(normNeedle.length * 0.5));
    const tailLen = Math.min(30, Math.floor(normNeedle.length * 0.2));
    const tailAnchor =
      tailLen >= 15 ? normNeedle.slice(normNeedle.length - tailLen) : null;
    for (
      let len = Math.floor(normNeedle.length * 0.95);
      len >= minLen;
      len -= Math.max(5, Math.floor(normNeedle.length * 0.05))
    ) {
      const prefix = normNeedle.slice(0, len);
      const candidateIdx = cached.normalized.indexOf(prefix);
      if (candidateIdx === -1) continue;
      if (tailAnchor) {
        const searchStart = candidateIdx + len;
        const searchEnd = Math.min(
          candidateIdx + Math.floor(normNeedle.length * 1.5),
          cached.normalized.length,
        );
        const tailIdx = cached.normalized.indexOf(tailAnchor, searchStart);
        if (tailIdx === -1 || tailIdx >= searchEnd) continue;
        idx = candidateIdx;
        // Trim span to where the verified tail actually ends.
        matchedLen = tailIdx + tailAnchor.length - candidateIdx;
        break;
      }
      // No tail anchor (needle too short): cap matchedLen to the
      // verified prefix only so we don't blanket trailing unverified bytes.
      idx = candidateIdx;
      matchedLen = len;
      break;
    }
  }

  // Strategy 3: literal short-anchor fallback. Anchor on the first 30
  // chars of the original excerpt; return a tight, fixed-size span
  // (80 chars) so we never blanket past the real passage — the rest
  // of the excerpt may not actually be at this location.
  if (idx === -1) {
    const literalAnchor = excerpt.slice(0, 30).trim();
    if (literalAnchor.length >= 15) {
      const literalIdx = parsedText.indexOf(literalAnchor);
      if (literalIdx !== -1) {
        const SHORT_SPAN = 80;
        const approxEnd = Math.min(
          literalIdx + SHORT_SPAN,
          parsedText.length,
        );
        return [literalIdx, approxEnd];
      }
    }
  }

  if (idx === -1) return null;
  const start = cached.map[idx];
  const endNormIdx = Math.min(idx + matchedLen, cached.normalized.length);
  const end = cached.map[endNormIdx] ?? parsedText.length;
  if (end <= start) return null;
  return [start, end];
}

/**
 * Build the run-list of plain/highlighted segments to render in SourcePane.
 *
 * Each chunk with a valid `source_span = [start, end]` becomes a <mark>
 * over `parsed_text[start:end]`. Overlapping spans are dropped (later
 * spans that start before the previous span ends are skipped) to keep
 * the DOM clean — overlap is rare in practice (the LLM emits disjoint
 * excerpts) and showing nested marks confuses the click target.
 *
 * For chunks WITHOUT a backend-derived span, `flexibleFind` retries the
 * lookup with whitespace normalization. PDF reflow line breaks are the
 * usual culprit for backend find() failures; the fallback recovers most
 * of those cases at render time so the source pane stays in sync with
 * the canvas selection.
 */
export function buildSourceSegments(parsedText, normIndex, chunks) {
  if (!parsedText) return [];

  const spans = chunks
    .map((c) => {
      // Prefer the backend-derived span; otherwise retry with the
      // whitespace-flexible search against the chunk's source_excerpt.
      let span = Array.isArray(c.source_span) && c.source_span.length === 2
        ? c.source_span
        : null;
      if (!span && typeof c.source_excerpt === "string" && c.source_excerpt) {
        span = flexibleFind(parsedText, normIndex, c.source_excerpt);
      }
      if (!span) return null;
      return { chunk_id: c.chunk_id, start: span[0], end: span[1] };
    })
    .filter(Boolean)
    .filter((s) => s.start >= 0 && s.end > s.start && s.end <= parsedText.length)
    .sort((a, b) => a.start - b.start);

  // Drop any span that starts before the previous kept span ended.
  const kept = [];
  let lastEnd = 0;
  for (const s of spans) {
    if (s.start < lastEnd) continue;
    kept.push(s);
    lastEnd = s.end;
  }

  const out = [];
  let cursor = 0;
  for (const s of kept) {
    if (s.start > cursor) {
      out.push({ type: "text", content: parsedText.slice(cursor, s.start) });
    }
    out.push({
      type: "mark",
      chunk_id: s.chunk_id,
      content: parsedText.slice(s.start, s.end),
    });
    cursor = s.end;
  }
  if (cursor < parsedText.length) {
    out.push({ type: "text", content: parsedText.slice(cursor) });
  }
  return out;
}

/**
 * Distinguish frontend-synthetic chunk IDs (used for analyst-added chunks
 * that haven't been allocated a real backend ID yet) from real ones.
 *
 * Backend allocates real IDs in `gate_chunks._build_added_chunk` based on
 * (next_seq, text). The frontend can't predict these, so we use synthetic
 * "add-N" IDs for layout/rendering and refuse to include them in edge
 * mutations at submit time.
 */
const SYNTHETIC_ID_PREFIX = "add-";
function isSyntheticId(id) {
  return typeof id === "string" && id.startsWith(SYNTHETIC_ID_PREFIX);
}

/** Reasons match backend `ChunkGateRejectReason` enum values one-for-one. */
const REJECT_REASONS = [
  { value: "missed_procedures", label: "Missed procedures" },
  { value: "over_chunked", label: "Over-chunked (too granular)" },
  { value: "under_chunked", label: "Under-chunked (too coarse)" },
  { value: "bad_boundaries", label: "Bad boundaries" },
  { value: "bad_descriptions", label: "Bad descriptions" },
  { value: "bad_flow", label: "Bad sequencing / flow" },
  { value: "other", label: "Other (free-form)" },
];

/**
 * Build the on-wire ChunkGateSubmit payload from local state.
 *
 * Diff semantics:
 *   - decisions[]: only chunks the analyst actively touched (drop, edit).
 *     The backend's safe-default approves any unmentioned chunk, so we
 *     skip approve entries to keep the payload minimal and focused on
 *     analyst intent.
 *   - added_chunks[]: raw fields per AddedChunkItem schema (no chunk_id
 *     or sequence_index — the backend allocates those).
 *   - edges[]: net mutations against the original precedes_ids set —
 *     opposing add/remove pairs collapse, edges involving synthetic IDs
 *     are filtered out (backend can't resolve them; v1 limitation).
 *   - is_sequential: only when the analyst clicked the chip; null (the
 *     default) keeps the auto-detected flag and sends nothing.
 */
export function buildSubmitPayload({
  originalChunks, edits, droppedIds, addedChunks, edgeOps, operatorOverrides,
  conditionEdits, mergeGroups, sequentialOverride = null,
}) {
  const decisions = [];
  // Chunks absorbed by a merge are handled by their survivor's decision —
  // the backend drops them and rewires their edges, so emitting a separate
  // drop here would be redundant.
  const absorbed = new Set();
  for (const ids of Object.values(mergeGroups || {})) {
    for (const id of ids) absorbed.add(id);
  }
  for (const c of originalChunks) {
    if (absorbed.has(c.chunk_id)) continue;
    const partners = (mergeGroups || {})[c.chunk_id];
    if (partners && partners.length > 0) {
      const d = { chunk_id: c.chunk_id, action: "merge", merge_with: partners };
      const cEdits = edits[c.chunk_id];
      if (cEdits && Object.keys(cEdits).length > 0) d.edits = cEdits;
      decisions.push(d);
      continue;
    }
    if (droppedIds.has(c.chunk_id)) {
      decisions.push({ chunk_id: c.chunk_id, action: "drop" });
      continue;
    }
    const cEdits = edits[c.chunk_id];
    if (cEdits && Object.keys(cEdits).length > 0) {
      decisions.push({ chunk_id: c.chunk_id, action: "edit", edits: cEdits });
    }
  }

  const realIds = new Set(originalChunks.map((c) => c.chunk_id));
  const original = new Set();
  for (const c of originalChunks) {
    for (const t of c.precedes_ids || []) {
      if (realIds.has(t)) original.add(`${c.chunk_id}->${t}`);
    }
  }
  const final = new Set(original);
  for (const op of edgeOps) {
    if (isSyntheticId(op.from) || isSyntheticId(op.to)) continue;
    if (!realIds.has(op.from) || !realIds.has(op.to)) continue;
    const key = `${op.from}->${op.to}`;
    if (op.action === "add") final.add(key);
    else final.delete(key);
  }
  const edges = [];
  for (const k of original) {
    if (!final.has(k)) {
      const [from, to] = k.split("->");
      edges.push({ action: "remove", from, to });
    }
  }
  for (const k of final) {
    if (!original.has(k)) {
      const [from, to] = k.split("->");
      edges.push({ action: "add", from, to });
    }
  }

  // Strip frontend-only fields from added chunks before sending. Drop any
  // chunks the analyst added and then dropped — visually they show as
  // strikethrough on the canvas, but there's no point shipping them.
  const cleanedAdded = addedChunks
    .filter((a) => !droppedIds.has(a.tmp_id))
    .map(({ tmp_id, _isAdded, ...rest }) => rest);

  // Operator overrides: {operator_id: kind}. The backend merges these
  // into chunk_operators by operator_id; overrides whose operator_id no
  // longer exists in the post-edit geometry silently drop at normalize.
  const overrideList = Object.entries(operatorOverrides || {}).map(
    ([operator_id, kind]) => ({ operator_id, kind })
  );

  // Condition edits: keyed by chunk_id on the analyst's mutation map.
  // Entries shaped {action: "set"|"clear", description, pattern,
  // pattern_type, on_true_ids, on_false_ids}. Sent verbatim to the
  // backend; the gate processor re-validates against precedes_ids.
  const conditionList = Object.entries(conditionEdits || {}).map(
    ([chunk_id, edit]) => ({ chunk_id, ...edit })
  );

  const submission = {};
  if (decisions.length) submission.decisions = decisions;
  if (cleanedAdded.length) submission.added_chunks = cleanedAdded;
  if (edges.length) submission.edges = edges;
  if (overrideList.length) submission.operator_overrides = overrideList;
  if (conditionList.length) submission.condition_edits = conditionList;
  // Sequentiality override: the auto-detected flag is otherwise frozen
  // after entity extraction. Only sent when the analyst clicked the chip;
  // null means "keep what was detected".
  if (typeof sequentialOverride === "boolean") submission.is_sequential = sequentialOverride;
  return submission;
}

/**
 * Inline form for analyst-supplied chunks the LLM missed. Rendered in
 * the side-panel column when toggled; submission appends to addedChunks
 * with a synthetic `tmp_id` for canvas identity. The backend allocates
 * the real chunk_id at gate processing time.
 */
function AddChunkForm({ onAdd, onCancel }) {
  const [text, setText] = useState("");
  const [excerpt, setExcerpt] = useState("");
  const [confidence, setConfidence] = useState(0.7);
  const [branch, setBranch] = useState(false);
  const [converge, setConverge] = useState(false);

  const handleSave = () => {
    const trimmed = text.trim();
    if (!trimmed) return;
    onAdd({
      text: trimmed,
      source_excerpt: excerpt.trim(),
      behavioral_confidence: confidence,
      branch_point: branch,
      convergence_point: converge,
    });
  };

  return (
    <div className="rounded-lg border border-gb-bright-green/60 bg-gb-bg0 p-3 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase font-data text-gb-bright-green tracking-wide">New chunk</span>
        <button
          type="button"
          onClick={onCancel}
          className="text-[10px] text-gb-fg4 hover:text-gb-fg2 transition-colors"
        >
          Cancel
        </button>
      </div>
      <div>
        <label className="block text-[10px] uppercase font-data text-gb-fg4 mb-1">Text *</label>
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          rows={3}
          autoFocus
          placeholder="Behavioral description, e.g. 'The adversary cleared event logs via wevtutil.'"
          className="w-full px-2 py-1.5 rounded border border-gb-bg2 bg-gb-bg0-h text-gb-fg1 text-[12px] font-data resize-y focus:outline-none focus:border-gb-bright-green"
        />
      </div>
      <div>
        <label className="block text-[10px] uppercase font-data text-gb-fg4 mb-1">
          Source excerpt<InfoDot term="source-excerpt" />{" "}
          <span className="normal-case text-gb-gray">(optional, verbatim)</span>
        </label>
        <textarea
          value={excerpt}
          onChange={(e) => setExcerpt(e.target.value)}
          rows={2}
          placeholder="Verbatim sentence(s) from the source supporting this chunk."
          className="w-full px-2 py-1.5 rounded border border-gb-bg2 bg-gb-bg0-h text-gb-fg1 text-[12px] italic resize-y focus:outline-none focus:border-gb-bright-green"
        />
      </div>
      <div>
        <div className="flex items-center justify-between mb-1">
          <label className="text-[10px] uppercase font-data text-gb-fg4">Confidence</label>
          <span className="text-[11px] font-data text-gb-fg2">{Math.round(confidence * 100)}%</span>
        </div>
        <input
          type="range"
          min={0}
          max={1}
          step={0.05}
          value={confidence}
          onChange={(e) => setConfidence(parseFloat(e.target.value))}
          className="w-full accent-gb-bright-green"
        />
      </div>
      <div className="flex items-center gap-4">
        <label className="flex items-center gap-2 text-[12px] text-gb-fg2 cursor-pointer">
          <input type="checkbox" checked={branch} onChange={(e) => setBranch(e.target.checked)} className="accent-gb-bright-purple" />
          Branch point
        </label>
        <label className="flex items-center gap-2 text-[12px] text-gb-fg2 cursor-pointer">
          <input type="checkbox" checked={converge} onChange={(e) => setConverge(e.target.checked)} className="accent-gb-bright-aqua" />
          Converge point
        </label>
      </div>
      <p className="text-[10px] text-gb-fg4 leading-relaxed">
        Added chunks land at the end of the sequence. To insert mid-flow, reject the gate and re-run chunking with feedback.
      </p>
      <div className="flex justify-end gap-2 mt-1">
        <button
          type="button"
          onClick={handleSave}
          disabled={!text.trim()}
          className="px-3 py-1.5 rounded text-[12px] font-medium bg-gb-bright-green/20 text-gb-bright-green border border-gb-bright-green/60 hover:bg-gb-bright-green/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Add chunk
        </button>
      </div>
    </div>
  );
}

/**
 * Reject modal — analyst picks one of the ChunkGateRejectReason enum
 * values and adds free-form comments. Submitted as `{reject: {reason,
 * comments}}` which routes the gate back to chunk_behaviors with the
 * comments injected into the rerun prompt (see _build_rerun_feedback_context).
 */
function RejectModal({
  onCancel, onConfirm, submitting,
  // Pre-fill from an AI re-chunk recommendation. The analyst still confirms
  // and can rewrite both fields — the reviewer opens the form, it does not
  // submit it. Uninitialised when opened from the toolbar.
  initialReason = "missed_procedures", initialComments = "",
}) {
  // A reason the dropdown does not offer would render the select blank and
  // submit whatever the browser picked. Only reachable through the AI
  // pre-fill, which writes the reviewer's reason straight in — so it is worth
  // the two lines even though a contract test now pins the two lists.
  const [reason, setReason] = useState(
    REJECT_REASONS.some((r) => r.value === initialReason)
      ? initialReason
      : "other",
  );
  const [comments, setComments] = useState(initialComments);

  return (
    <div
      role="dialog"
      aria-modal="true"
      className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 animate-fade-in"
    >
      <div className="w-[520px] max-w-[90vw] rounded-lg border border-gb-bright-orange/60 bg-gb-bg0-h shadow-2xl">
        <div className="px-4 py-3 border-b border-gb-bg2 flex items-center justify-between">
          <h3 className="text-[14px] font-semibold text-gb-bright-orange">Reject and re-run chunking</h3>
          <button
            type="button"
            onClick={onCancel}
            className="w-6 h-6 flex items-center justify-center rounded text-gb-fg4 hover:text-gb-fg1 hover:bg-gb-bg1 transition-colors text-[14px]"
            aria-label="Close"
          >
            ✕
          </button>
        </div>
        <div className="p-4 flex flex-col gap-3">
          <p className="text-[12px] text-gb-fg2 leading-relaxed">
            The pipeline will discard the current chunks and re-run <code className="font-data text-gb-bright-yellow">chunk_behaviors</code> with your feedback as a high-priority hint in the prompt.
          </p>
          <div>
            <label className="block text-[10px] uppercase font-data text-gb-fg4 mb-1">Reason</label>
            <select
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              className="w-full px-2 py-1.5 rounded border border-gb-bg2 bg-gb-bg0 text-gb-fg1 text-[12px] focus:outline-none focus:border-gb-bright-orange"
            >
              {REJECT_REASONS.map((r) => (
                <option key={r.value} value={r.value}>{r.label}</option>
              ))}
            </select>
          </div>
          <div>
            <label className="block text-[10px] uppercase font-data text-gb-fg4 mb-1">
              Comments <span className="normal-case text-gb-gray">(injected into rerun prompt)</span>
            </label>
            <textarea
              value={comments}
              onChange={(e) => setComments(e.target.value)}
              rows={4}
              placeholder="Specific guidance the LLM should follow on the next pass."
              className="w-full px-2 py-1.5 rounded border border-gb-bg2 bg-gb-bg0 text-gb-fg1 text-[12px] font-data resize-y focus:outline-none focus:border-gb-bright-orange"
            />
          </div>
        </div>
        <div className="px-4 py-3 border-t border-gb-bg2 flex justify-end gap-2">
          <button
            type="button"
            onClick={onCancel}
            disabled={submitting}
            className="px-3 py-1.5 rounded text-[12px] font-medium text-gb-fg2 hover:bg-gb-bg1 transition-colors"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => onConfirm({ reason, comments: comments.trim() })}
            disabled={submitting}
            className="px-3 py-1.5 rounded text-[12px] font-medium bg-gb-bright-orange/20 text-gb-bright-orange border border-gb-bright-orange/60 hover:bg-gb-bright-orange/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {submitting ? "Rerunning…" : "↻ Reject & rerun"}
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * Source pane — renders parsed_text with <mark> highlights on each
 * chunk's source_span. Clicking a highlight selects the chunk;
 * selecting a chunk on the canvas auto-scrolls the corresponding mark
 * into view (driven by parent via the `selectedId` prop).
 */
function SourcePane({ parsedText, chunks, selectedId, editedIds, droppedIds, onSelectChunk }) {
  const markRefs = useRef({});

  // Memoize the normalization index — replaces the prior module-scoped
  // Map cache. Tied to the component lifecycle so it goes away when
  // the gate-review panel closes.
  const normIndex = useMemo(
    () => buildNormIndex(parsedText || ""),
    [parsedText],
  );

  const segments = useMemo(
    () => buildSourceSegments(parsedText || "", normIndex, chunks),
    [parsedText, normIndex, chunks],
  );

  const linkedCount = useMemo(
    () => segments.filter((s) => s.type === "mark").length,
    [segments],
  );

  // Scroll the selected mark into view when selection changes.
  useEffect(() => {
    if (!selectedId) return;
    const el = markRefs.current[selectedId];
    if (el && typeof el.scrollIntoView === "function") {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }, [selectedId]);

  if (!parsedText) {
    return (
      <div className="rounded-lg border border-gb-bg2 bg-gb-bg0 p-3 text-[12px] text-gb-fg4">
        No parsed source text available.
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-gb-bg2 bg-gb-bg0 flex flex-col overflow-hidden">
      <div className="px-3 py-2 border-b border-gb-bg2 bg-gb-bg0-h flex items-center justify-between">
        <span className="text-[10px] uppercase font-data tracking-wide text-gb-fg4">Source</span>
        <span className="text-[10px] font-data text-gb-fg4">
          {linkedCount}/{chunks.length} linked
        </span>
      </div>
      <div className="overflow-y-auto p-3 text-[12px] text-gb-fg2 leading-relaxed whitespace-pre-wrap font-data">
        {segments.map((seg, i) => {
          if (seg.type === "text") {
            return <span key={i}>{seg.content}</span>;
          }
          const isSel = seg.chunk_id === selectedId;
          const isEdited = editedIds.has(seg.chunk_id);
          const isDropped = droppedIds.has(seg.chunk_id);
          // Drop wins so the analyst can't lose track of removed chunks; selected
          // still gets the orange ring on top of the strikethrough.
          const base = isDropped
            ? "bg-gb-bright-red/20 text-gb-fg4 line-through decoration-gb-bright-red"
            : isSel
              ? "bg-gb-bright-orange text-gb-bg0 font-semibold"
              : isEdited
                ? "bg-gb-bright-yellow/30 text-gb-fg0 hover:bg-gb-bright-yellow/40"
                : "bg-gb-bright-yellow/10 text-gb-fg1 hover:bg-gb-bright-yellow/25";
          const ring = isSel ? "ring-2 ring-gb-bright-orange" : "";
          const cls = `${base} ${ring}`;
          return (
            <mark
              key={i}
              ref={(el) => {
                if (el) markRefs.current[seg.chunk_id] = el;
              }}
              className={`rounded px-0.5 cursor-pointer transition-colors ${cls}`}
              data-chunk-id={seg.chunk_id}
              onClick={(e) => {
                e.stopPropagation();
                onSelectChunk(seg.chunk_id);
              }}
            >
              {seg.content}
            </mark>
          );
        })}
      </div>
    </div>
  );
}

/**
 * Attack Flow attack-condition editor surfaced inside the SidePanel.
 *
 * Two modes:
 *   * No condition (and no successors) → render a hint explaining when
 *     a condition is appropriate. No "Mark as condition" button — it'd
 *     have nothing to partition.
 *   * No condition + has successors → "Mark as condition" button to
 *     bootstrap one with the current successors all on the on_true side.
 *   * Condition present → description textarea + pattern + partition
 *     picker (per-successor radio: true / false). "Clear condition"
 *     removes it.
 *
 * Edits dispatch to onConditionSet({fields}) or onConditionClear(); the
 * parent merges over the current condition before submit and writes
 * ConditionEditItem entries.
 */
function ConditionEditor({ chunk, condition, onConditionSet, onConditionClear }) {
  const chunkId = chunk?.chunk_id;
  const successors = chunk?.precedes_ids || [];
  const hasCondition = condition != null;

  if (!hasCondition && successors.length === 0) {
    return (
      <div className="pt-2 border-t border-gb-bg1">
        <p className="text-[10px] uppercase font-data text-gb-fg4 mb-1">
          Condition<InfoDot term="flow-condition" />
        </p>
        <p className="text-[11px] text-gb-gray leading-relaxed">
          No downstream successors. Add a precedes edge first if this chunk's flow forks on a runtime check.
        </p>
      </div>
    );
  }

  if (!hasCondition) {
    return (
      <div className="pt-2 border-t border-gb-bg1">
        <p className="text-[10px] uppercase font-data text-gb-fg4 mb-1">
          Condition<InfoDot term="flow-condition" />
        </p>
        <p className="text-[11px] text-gb-gray mb-2 leading-relaxed">
          When this chunk's flow forks on a runtime check (e.g., "if EDR present, abort; else continue"), mark it as a condition and partition the downstream chunks.
        </p>
        <button
          type="button"
          onClick={() => onConditionSet(chunkId, {
            description: "",
            on_true_ids: [...successors],
            on_false_ids: [],
          })}
          className="text-[11px] px-2 py-1 rounded border border-gb-bright-aqua/40 text-gb-bright-aqua hover:bg-gb-bright-aqua/10 transition-colors"
        >
          + Mark as condition
        </button>
      </div>
    );
  }

  const trueSet = new Set(condition.on_true_ids || []);
  const falseSet = new Set(condition.on_false_ids || []);

  const togglePartition = (succId, side) => {
    // Move succId to the named side; remove from the other to keep
    // every successor on exactly one branch. "unset" pulls it from
    // both (rare; supports analyst-only-one-side cases).
    const newTrue = new Set(trueSet);
    const newFalse = new Set(falseSet);
    newTrue.delete(succId);
    newFalse.delete(succId);
    if (side === "true") newTrue.add(succId);
    else if (side === "false") newFalse.add(succId);
    onConditionSet(chunkId, {
      on_true_ids: [...newTrue],
      on_false_ids: [...newFalse],
    });
  };

  return (
    <div className="pt-2 border-t border-gb-bg1 space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-[10px] uppercase font-data text-gb-bright-aqua">
          ✓ Condition
        </p>
        <button
          type="button"
          onClick={() => onConditionClear(chunkId)}
          className="text-[10px] text-gb-bright-red hover:text-gb-red transition-colors"
        >
          ✕ Clear
        </button>
      </div>

      <div>
        <label className="block text-[10px] uppercase font-data text-gb-fg4 mb-1">
          Description
        </label>
        <textarea
          value={condition.description || ""}
          onChange={(e) => onConditionSet(chunkId, { description: e.target.value })}
          rows={2}
          placeholder="actor checks whether the host is domain-joined"
          className="w-full px-2 py-1.5 rounded border border-gb-bg2 bg-gb-bg0-h text-gb-fg1 text-[12px] resize-y focus:outline-none focus:border-gb-bright-aqua"
        />
      </div>

      <div className="grid grid-cols-3 gap-1.5">
        <div className="col-span-2">
          <label className="block text-[10px] uppercase font-data text-gb-fg4 mb-1">
            Pattern (optional)
          </label>
          <input
            type="text"
            value={condition.pattern || ""}
            onChange={(e) => onConditionSet(chunkId, { pattern: e.target.value })}
            placeholder="HKLM\\...\\Sense"
            className="w-full px-2 py-1 rounded border border-gb-bg2 bg-gb-bg0-h text-gb-fg1 text-[11px] font-data focus:outline-none focus:border-gb-bright-aqua"
          />
        </div>
        <div>
          <label className="block text-[10px] uppercase font-data text-gb-fg4 mb-1">
            Type
          </label>
          <select
            value={condition.pattern_type || "plain"}
            onChange={(e) => onConditionSet(chunkId, { pattern_type: e.target.value })}
            disabled={!condition.pattern}
            className="w-full px-1.5 py-1 rounded border border-gb-bg2 bg-gb-bg0 text-gb-fg1 text-[11px] font-data focus:outline-none focus:border-gb-bright-aqua disabled:opacity-50"
          >
            <option value="plain">plain</option>
            <option value="stix">stix</option>
            <option value="regex">regex</option>
          </select>
        </div>
      </div>

      <div>
        <p className="text-[10px] uppercase font-data text-gb-fg4 mb-1">
          Successor partition
        </p>
        {successors.length === 0 ? (
          <p className="text-[11px] text-gb-gray italic">No successors. Add precedes edges before partitioning.</p>
        ) : (
          <div className="space-y-1">
            {successors.map((succId) => {
              const side = trueSet.has(succId) ? "true"
                : falseSet.has(succId) ? "false"
                : "unset";
              return (
                <div key={succId} className="flex items-center gap-1.5 text-[11px]">
                  <span className="font-data text-gb-fg2 truncate flex-1" title={succId}>
                    {succId}
                  </span>
                  {(["true", "false"]).map((s) => (
                    <button
                      key={s}
                      type="button"
                      onClick={() => togglePartition(succId, side === s ? "unset" : s)}
                      className={`px-1.5 py-0.5 rounded font-data text-[10px] border transition-colors ${
                        side === s
                          ? s === "true"
                            ? "bg-gb-bright-aqua/30 border-gb-bright-aqua text-gb-bright-aqua"
                            : "bg-gb-bright-yellow/30 border-gb-bright-yellow text-gb-bright-yellow"
                          : "border-gb-bg2 text-gb-gray hover:text-gb-fg2"
                      }`}
                    >
                      {s === "true" ? "ON TRUE" : "ON FALSE"}
                    </button>
                  ))}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}


/**
 * Side panel for editing the selected chunk. Edits go into the parent's
 * `edits` map; this component is a controlled form over the live (merged)
 * chunk and reports field changes upward.
 */
function MergeControl({ chunk, absorbed, candidates, onMergeInto, onUnmerge }) {
  const [target, setTarget] = useState("");
  if (absorbed.length > 0) {
    return (
      <div className="rounded-md border border-gb-bright-purple/40 bg-gb-bright-purple/10 p-3 text-[12px] text-gb-fg2">
        <p className="font-semibold text-gb-bright-purple mb-1">
          Merging {absorbed.length} chunk{absorbed.length > 1 ? "s" : ""} in
        </p>
        <p className="text-[11px] text-gb-fg4 mb-2">
          {absorbed.join(", ")} will be absorbed into this procedure and their
          flow edges rewired onto it.
        </p>
        <button
          type="button"
          onClick={() => onUnmerge?.(chunk.chunk_id)}
          className="text-[10px] text-gb-bright-aqua hover:text-gb-bright-blue"
        >
          ↺ Undo merge
        </button>
      </div>
    );
  }
  if (candidates.length === 0) return null;
  return (
    <div className="rounded-md border border-gb-bg2 bg-gb-bg0 p-3">
      <label className="block text-[10px] uppercase tracking-wide text-gb-fg4 mb-1">
        Merge another chunk into this one
      </label>
      <div className="flex gap-2">
        <select
          value={target}
          onChange={(e) => setTarget(e.target.value)}
          className="flex-1 rounded border border-gb-bg2 bg-gb-bg0 px-2 py-1 text-[11px] text-gb-fg1"
        >
          <option value="">Select a chunk...</option>
          {candidates.map((c) => (
            <option key={c.chunk_id} value={c.chunk_id}>
              {c.sequence_index}. {(c.text || "").slice(0, 46)}
            </option>
          ))}
        </select>
        <button
          type="button"
          disabled={!target}
          onClick={() => { onMergeInto?.(chunk.chunk_id, target); setTarget(""); }}
          className="rounded px-2 py-1 text-[11px] text-gb-bright-purple hover:text-gb-purple disabled:text-gb-fg4 disabled:cursor-not-allowed"
        >
          Merge
        </button>
      </div>
      <p className="mt-1 text-[10px] text-gb-fg4 leading-relaxed">
        Use when one chunk describes the RESULT of another rather than a
        separate objective — the texts combine and the chain re-links.
      </p>
    </div>
  );
}


/**
 * The AI recommendations that have no node to sit on.
 *
 * Per-chunk advice rides its own node and its own SuggestionChip. Procedures
 * the chunker MISSED have no node by definition, edges are about a pair
 * rather than a chunk, and a re-chunk is about the whole pass — so without a
 * home of their own these three would be produced and never seen. This panel
 * is that home: the deselected state, which is where the analyst starts.
 */
function UnanchoredRecs({
  recommendations, appliedAdds, appliedEdges, liveEdgeKeys,
  onApplyAdd, onDismissAdd, onApplyEdge, onOpenReject,
}) {
  const adds = recommendations?.added_chunks ?? [];
  const edgeRecs = pendingEdgeRecs(recommendations, liveEdgeKeys);
  const reject = recommendations?.reject ?? null;
  const notes = recommendations?.overall_notes;
  if (!adds.length && !edgeRecs.length && !reject && !notes) return null;

  return (
    <div className="mt-3 border-t border-gb-bg2 pt-3">
      <p className="text-[10px] uppercase font-data text-gb-bright-purple mb-2">
        🤖 AI reviewer
      </p>

      {notes && (
        <p className="text-[11px] text-gb-fg2 leading-snug mb-2">{notes}</p>
      )}

      {/* A re-chunk throws away every chunk on the canvas AND every edit made
          on this pass, so it is never part of bulk accept and never one
          click — it opens the reject form pre-filled and the analyst
          confirms. */}
      {reject && (
        <div className="mb-3 rounded border border-gb-bright-orange/50 bg-gb-bright-orange/10 px-2 py-1.5">
          <p className="text-[11px] font-semibold text-gb-bright-orange">
            Recommends re-chunking — {String(reject.reason).replace(/_/g, " ")}
            <span className="font-normal text-gb-fg4"> ({reject.confidence})</span>
          </p>
          {reject.rationale && (
            <p className="text-[11px] text-gb-fg2 mt-1 leading-snug">{reject.rationale}</p>
          )}
          <p className="text-[10px] text-gb-fg4 mt-1 leading-snug">
            This discards every chunk on the canvas and every edit on this pass.
          </p>
          <button
            type="button"
            onClick={() => onOpenReject(reject)}
            className="mt-1.5 text-[10px] font-data text-gb-bright-orange hover:text-gb-fg1 transition-colors"
          >
            open the reject form →
          </button>
        </div>
      )}

      {adds.length > 0 && (
        <>
          <p className="text-[10px] uppercase font-data text-gb-fg4 mb-1">
            Procedures it says were missed ({adds.length})
          </p>
          {adds.map((rec, i) => (
            <div key={`add-${i}`} className="mb-2">
              <p className="text-[11px] text-gb-fg1 leading-snug">{rec.text}</p>
              <SuggestionChip
                rec={rec}
                applied={appliedAdds.has(i)}
                onApply={() => onApplyAdd(rec, i)}
                onDismiss={() => onDismissAdd(i)}
              />
            </div>
          ))}
        </>
      )}

      {edgeRecs.length > 0 && (
        <>
          <p className="text-[10px] uppercase font-data text-gb-fg4 mb-1 mt-3">
            Sequencing ({edgeRecs.length})
          </p>
          {edgeRecs.map((rec) => (
            <div key={edgeRecKey(rec)} className="mb-2">
              <p className="text-[11px] font-data text-gb-fg1 leading-snug">
                {rec.action === "add" ? "link" : "unlink"} {rec.from_chunk_id}
                {" \u2192 "}
                {rec.to_chunk_id}
              </p>
              <SuggestionChip
                rec={rec}
                applied={appliedEdges.has(edgeRecKey(rec))}
                onApply={() => onApplyEdge(rec)}
                onDismiss={() => onApplyEdge(rec, true)}
              />
            </div>
          ))}
        </>
      )}
    </div>
  );
}


function SidePanel({
  chunk, hasEdits, isDropped, onFieldChange, onResetEdits, onToggleDrop,
  operatorsForChunk = [], operatorOverrides = {}, onOperatorKindChange,
  condition = null, onConditionSet, onConditionClear,
  mergeGroups = {}, mergeCandidates = [], onMergeInto, onUnmerge,
  // AI reviewer additions. All optional — absent means this gate ran in
  // plain review mode and the panel behaves exactly as it always has.
  rec = null, recApplied = false, onApplyRec, onDismissRec,
  unanchored = null,
}) {
  if (!chunk) {
    return (
      <div className="rounded-lg border border-gb-bg2 bg-gb-bg0 p-4 text-[12px] text-gb-fg4 overflow-y-auto">
        <p className="font-medium text-gb-fg2 mb-2">No chunk selected</p>
        <p className="mb-2">Click a node on the canvas to inspect or edit it. Edits accumulate locally and are submitted together when you approve the gate.</p>
        <p className="text-[11px] text-gb-gray mt-3 leading-relaxed">
          Tip: drag from the bottom of one node to the top of another to add a precedes edge. Click an edge then press Delete to remove it.
        </p>
        {unanchored}
      </div>
    );
  }

  const conf = typeof chunk.behavioral_confidence === "number"
    ? chunk.behavioral_confidence
    : 0.5;

  return (
    <div className="rounded-lg border border-gb-bg2 bg-gb-bg0 p-3 flex flex-col gap-3 overflow-y-auto">
      <div className="flex items-center justify-between">
        <span className="text-[10px] font-data text-gb-fg4">{chunk.chunk_id}</span>
        <div className="flex items-center gap-3">
          {hasEdits && !isDropped && (
            <button
              type="button"
              onClick={onResetEdits}
              className="text-[10px] text-gb-bright-yellow hover:text-gb-bright-orange transition-colors"
            >
              Reset edits
            </button>
          )}
          <button
            type="button"
            onClick={onToggleDrop}
            className={`text-[10px] transition-colors ${
              isDropped
                ? "text-gb-bright-aqua hover:text-gb-bright-blue"
                : "text-gb-bright-red hover:text-gb-red"
            }`}
          >
            {isDropped ? "↺ Undo drop" : "✕ Drop chunk"}
          </button>
        </div>
      </div>

      {rec && (
        <SuggestionChip
          rec={rec}
          applied={recApplied}
          onApply={onApplyRec}
          onDismiss={onDismissRec}
        />
      )}

      {!isDropped && (
        <MergeControl
          chunk={chunk}
          absorbed={mergeGroups[chunk.chunk_id] || []}
          candidates={mergeCandidates}
          onMergeInto={onMergeInto}
          onUnmerge={onUnmerge}
        />
      )}

      {isDropped ? (
        <div className="rounded-md border border-gb-bright-red/40 bg-gb-bright-red/10 p-3 text-[12px] text-gb-fg2 leading-relaxed">
          <p className="font-semibold text-gb-bright-red mb-1">Marked for drop</p>
          <p>
            This chunk will be removed when you approve the gate. Click "Undo drop" to keep it, or click another node to continue review.
          </p>
        </div>
      ) : (
        <>
          <div>
            <label className="block text-[10px] uppercase font-data text-gb-fg4 mb-1">Text</label>
            <textarea
              value={chunk.text || ""}
              onChange={(e) => onFieldChange("text", e.target.value)}
              rows={3}
              className="w-full px-2 py-1.5 rounded border border-gb-bg2 bg-gb-bg0-h text-gb-fg1 text-[12px] font-data resize-y focus:outline-none focus:border-gb-bright-yellow"
            />
          </div>

          <div>
            <label className="block text-[10px] uppercase font-data text-gb-fg4 mb-1">
              Source excerpt<InfoDot term="source-excerpt" />{" "}
              <span className="normal-case text-gb-gray">(verbatim from report)</span>
            </label>
            <textarea
              value={chunk.source_excerpt || ""}
              onChange={(e) => onFieldChange("source_excerpt", e.target.value)}
              rows={3}
              className="w-full px-2 py-1.5 rounded border border-gb-bg2 bg-gb-bg0-h text-gb-fg1 text-[12px] italic resize-y focus:outline-none focus:border-gb-bright-yellow"
            />
          </div>

          <div>
            <div className="flex items-center justify-between mb-1">
              <label className="text-[10px] uppercase font-data text-gb-fg4">Confidence</label>
              <span className="text-[11px] font-data text-gb-fg2">{Math.round(conf * 100)}%</span>
            </div>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={conf}
              onChange={(e) => onFieldChange("behavioral_confidence", parseFloat(e.target.value))}
              className="w-full accent-gb-bright-yellow"
            />
          </div>

          <div className="flex items-center gap-4">
            <label className="flex items-center gap-2 text-[12px] text-gb-fg2 cursor-pointer">
              <input
                type="checkbox"
                checked={!!chunk.branch_point}
                onChange={(e) => onFieldChange("branch_point", e.target.checked)}
                className="accent-gb-bright-purple"
              />
              Branch point
            </label>
            <label className="flex items-center gap-2 text-[12px] text-gb-fg2 cursor-pointer">
              <input
                type="checkbox"
                checked={!!chunk.convergence_point}
                onChange={(e) => onFieldChange("convergence_point", e.target.checked)}
                className="accent-gb-bright-aqua"
              />
              Converge point
            </label>
          </div>

          {/* Attack Flow operator-kind overrides — one dropdown per operator
              anchored on this chunk. Defaults: AND for converge, OR for
              branch. Analyst flip persists into chunk_operators on submit. */}
          {operatorsForChunk.map((op) => {
            const overrideKind = operatorOverrides[op.operator_id];
            const effectiveKind = overrideKind || op.kind;
            const roleLabel = op.role === "converge" ? "Converge" : "Branch";
            return (
              <div key={op.operator_id} className="rounded-md border border-gb-bg2 bg-gb-bg0-h p-2.5">
                <div className="flex items-center justify-between mb-1.5">
                  <label className="text-[10px] uppercase font-data text-gb-fg4">
                    {roleLabel} operator
                  </label>
                  <span className="text-[10px] font-data text-gb-gray">
                    {op.operator_id.slice(0, 8)}
                  </span>
                </div>
                <select
                  value={effectiveKind}
                  onChange={(e) => onOperatorKindChange(op.operator_id, e.target.value)}
                  className="w-full px-2 py-1 rounded border border-gb-bg2 bg-gb-bg0 text-gb-fg1 text-[12px] font-data focus:outline-none focus:border-gb-bright-yellow"
                >
                  <option value="AND">AND — all branches required</option>
                  <option value="OR">OR — any branch suffices</option>
                  <option value="XOR">XOR — exactly one branch</option>
                </select>
                {overrideKind && overrideKind !== op.kind && (
                  <p className="text-[10px] text-gb-bright-yellow mt-1.5">
                    Overridden from {op.kind}
                  </p>
                )}
              </div>
            );
          })}

          {/* Attack Flow attack-condition: present when chunk has a
              precondition (from chunker or analyst-added). Description +
              partition of downstream successors between on_true / on_false. */}
          <ConditionEditor
            chunk={chunk}
            condition={condition}
            onConditionSet={onConditionSet}
            onConditionClear={onConditionClear}
          />

          {/* Chain-separation controls. chain_root flips the chunk to a
              fresh-start root in a multi-intrusion source; chain_label
              names the chain so the canvas chip-tags it. */}
          <div className="pt-2 border-t border-gb-bg1 space-y-2">
            <label className="flex items-center gap-2 text-[12px] text-gb-fg2 cursor-pointer">
              <input
                type="checkbox"
                checked={!!chunk.chain_root}
                onChange={(e) => onFieldChange("chain_root", e.target.checked)}
                className="accent-gb-bright-purple"
              />
              <span title="Marks this chunk as the start of a new attack chain. The chunker's orphan-link backstop will skip it, preserving the disconnection from the prior chain.">
                Chain root (separate attack chain)
              </span>
            </label>
            <div>
              <label className="block text-[10px] uppercase font-data tracking-wider text-gb-fg4 mb-1">
                Chain label
              </label>
              <input
                type="text"
                value={chunk.chain_label || ""}
                onChange={(e) => onFieldChange("chain_label", e.target.value)}
                placeholder={chunk.chain_root ? "e.g. Veeam intrusion" : "(inherited from chain root)"}
                className="w-full px-2 py-1 text-[12px] bg-gb-bg0 border border-gb-bg2 rounded text-gb-fg1 placeholder-gb-bg4 outline-none focus:border-gb-bright-purple font-data"
              />
            </div>
          </div>
        </>
      )}

      <div className="text-[10px] font-data text-gb-fg4 mt-2 leading-relaxed">
        <div>seq: {chunk.sequence_index ?? "—"}</div>
        <div>precedes: {(chunk.precedes_ids || []).length === 0 ? "(end)" : chunk.precedes_ids.join(", ")}</div>
      </div>

      {unanchored}
    </div>
  );
}


/** Side panel rendered when the analyst clicks an edge. Provides the
 *  discoverable Delete affordance — the React Flow native interaction
 *  (click + Backspace/Delete) still works, but isn't visible without
 *  this UI cue. */
function EdgePanel({ edgeKey, chunkLookup, onDelete }) {
  if (!edgeKey) return null;
  const [src, tgt] = edgeKey.split("->");
  const srcChunk = chunkLookup[src];
  const tgtChunk = chunkLookup[tgt];

  const renderRef = (id, chunk) => (
    <div className="rounded border border-gb-bg2 bg-gb-bg1 p-2">
      <div className="text-[10px] font-data text-gb-fg4">{id}</div>
      {chunk ? (
        <>
          <div className="text-[11px] text-gb-fg2 mt-0.5">
            seq {chunk.sequence_index ?? "—"}
          </div>
          <div className="text-[11px] text-gb-fg3 mt-1 line-clamp-2">
            {chunk.text ?? ""}
          </div>
        </>
      ) : (
        <div className="text-[11px] text-gb-bright-orange mt-0.5 italic">
          (chunk dropped or removed)
        </div>
      )}
    </div>
  );

  return (
    <div className="rounded-lg border border-gb-bg2 bg-gb-bg0 p-3 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-[11px] text-gb-fg2 font-medium">Edge selected</span>
        <button
          type="button"
          onClick={onDelete}
          className="text-[10px] text-gb-bright-red hover:text-gb-red transition-colors font-medium"
          title="Delete this precedes edge (or press Delete / Backspace)"
        >
          ✕ Delete edge
        </button>
      </div>

      <div className="flex flex-col gap-2 text-[11px]">
        {renderRef(src, srcChunk)}
        <div className="text-center text-gb-fg4 text-[14px] leading-none">↓</div>
        {renderRef(tgt, tgtChunk)}
      </div>

      <p className="text-[10px] text-gb-gray leading-relaxed">
        Tip: pressing <kbd className="px-1 py-0.5 bg-gb-bg1 rounded text-[9px]">Delete</kbd> or <kbd className="px-1 py-0.5 bg-gb-bg1 rounded text-[9px]">Backspace</kbd> with an edge selected does the same thing.
      </p>
    </div>
  );
}

export default function ChunkReviewCanvas({
  payload,
  onSubmit,
  submitting,
  // AI reviewer additions. All optional — absent means this gate ran in
  // plain review mode and the canvas behaves exactly as it always has.
  recommendations = null,
  brief = null,
  onSaveBrief = null,
  savingBrief = false,
}) {
  const [edits, setEdits] = useState({});
  const [selectedId, setSelectedId] = useState(null);
  // selectedEdgeKey: "{source_chunk_id}->{target_chunk_id}" of the currently
  // selected edge. Mutually exclusive with selectedId — clicking a node
  // clears the edge selection and vice versa. Drives the EdgePanel render
  // (edge-delete discoverability).
  const [selectedEdgeKey, setSelectedEdgeKey] = useState(null);
  // chunk_ids the analyst marked for drop. Stored as a Set for O(1) lookup
  // in the renderers; reset by re-toggling the same chunk's drop button.
  const [droppedIds, setDroppedIds] = useState(() => new Set());
  // survivor chunk_id -> [absorbed chunk_ids]. A merge is one analyst gesture
  // that the backend expands into edit-survivor + drop-absorbed + rewire.
  const [mergeGroups, setMergeGroups] = useState(() => ({}));
  // Append-only log of edge mutations. Replayed in `computeLiveEdges` to
  // derive the rendered edge set; the submission diff collapses pairs
  // against the original edges before sending.
  const [edgeOps, setEdgeOps] = useState([]);
  // Analyst-supplied chunks the LLM missed. Each entry carries a synthetic
  // `tmp_id` for canvas identity; the backend allocates the real chunk_id
  // at gate processing time. Stripped before submit.
  const [addedChunks, setAddedChunks] = useState([]);
  const [showAddForm, setShowAddForm] = useState(false);
  const [showRejectModal, setShowRejectModal] = useState(false);
  // Operator-kind overrides keyed by operator_id: {op_id: "AND"|"OR"|"XOR"}.
  // Operators not present in this map use their geometry-derived default
  // (AND for converge, OR for branch) surfaced via payload.chunk_operators.
  const [operatorOverrides, setOperatorOverrides] = useState({});
  // Analyst override of the auto-detected sequentiality. null = untouched;
  // a boolean is sent as `is_sequential` on submit (approve OR reject) and
  // decides downstream whether the bundle gets PRECEDES edges, operators
  // and conditions at all.
  const [sequentialOverride, setSequentialOverride] = useState(null);
  // Condition edits keyed by chunk_id: {chunk_id: {action: "set"|"clear",
  // description, pattern, pattern_type, on_true_ids, on_false_ids}}.
  // Only chunks the analyst actively touched appear here; everything else
  // uses the live extracted condition from payload.chunk_conditions.
  const [conditionEdits, setConditionEdits] = useState({});

  // Which AI recommendations the analyst has applied, per channel.
  const [appliedRecs, setAppliedRecs] = useState(() => new Set());
  const [appliedEdgeRecs, setAppliedEdgeRecs] = useState(() => new Set());
  // rec index -> the tmp_id of the chunk it created, so undo can find it.
  const [appliedAddRecs, setAppliedAddRecs] = useState(() => ({}));
  // Pre-filled reject form, opened from the reviewer's re-chunk advice.
  const [rejectPrefill, setRejectPrefill] = useState(null);
  // chunk_id -> the analyst's own state for that chunk, captured the moment
  // BEFORE a recommendation was applied over it.
  //
  // Undo restores this snapshot rather than clearing the fields. Clearing is
  // the bug this codebase has already shipped twice: an undo that discards
  // the wording, the drop, or the merge the analyst had set by hand, as the
  // price of rejecting an unrelated AI suggestion.
  const [preApply, setPreApply] = useState({});

  const editedIds = useMemo(() => new Set(Object.keys(edits)), [edits]);

  /** chunk_id -> recommendation. Empty when there is no AI review. */
  const recsByChunk = useMemo(() => indexByChunk(recommendations), [recommendations]);

  // liveChunks = (originals + edits) ⊕ added chunks. Synthetic added chunks
  // are tagged `_isAdded: true` so the canvas paints the green badge.
  // sequence_index is computed locally so dagre lays them at the bottom; the
  // backend overwrites it with the authoritative value at processing time.
  const liveChunks = useMemo(() => {
    const originals = (payload?.chunks ?? []).map((c) => ({
      ...c,
      ...(edits[c.chunk_id] || {}),
    }));
    const baseSeq = originals.length;
    const synthetic = addedChunks.map((a, i) => ({
      chunk_id: a.tmp_id,
      text: a.text,
      source_excerpt: a.source_excerpt,
      context: {},
      behavioral_confidence: a.behavioral_confidence,
      branch_point: a.branch_point,
      convergence_point: a.convergence_point,
      sequence_index: baseSeq + i + 1,
      precedes_ids: [],
      source_span: null,
      _isAdded: true,
    }));
    return [...originals, ...synthetic];
  }, [payload?.chunks, edits, addedChunks]);

  const { nodes: unplacedNodes, edges } = useMemo(
    () => buildGraph(liveChunks, {
      selectedId, selectedEdgeKey, editedIds, droppedIds, edgeOps, recsByChunk,
    }),
    [liveChunks, selectedId, selectedEdgeKey, editedIds, droppedIds, edgeOps, recsByChunk],
  );
  const positions = useLayoutPositions(unplacedNodes, edges, CHUNK_LAYOUT);
  const nodes = useMemo(() => applyPositions(unplacedNodes, positions), [unplacedNodes, positions]);

  /** `${from}->${to}` for every edge currently on the canvas. Lets the
   *  recommendation list hide advice the analyst's graph already satisfies. */
  const liveEdgeKeys = useMemo(
    () => computeLiveEdges(liveChunks, edgeOps).liveKeys,
    [liveChunks, edgeOps],
  );

  const addedEdgeCount = useMemo(
    () => edges.filter((e) => e.id.startsWith("e+")).length,
    [edges],
  );

  const handleNodeClick = useCallback((_evt, node) => {
    setSelectedId(node.id);
    setSelectedEdgeKey(null);
  }, []);

  const handleEdgeClick = useCallback((_evt, edge) => {
    if (!edge?.source || !edge?.target) return;
    setSelectedEdgeKey(`${edge.source}->${edge.target}`);
    setSelectedId(null);
  }, []);

  const handlePaneClick = useCallback(() => {
    setSelectedId(null);
    setSelectedEdgeKey(null);
  }, []);

  /** Imperative delete from the side-panel button. Mirrors what pressing
   *  Delete on a selected edge does, but is the analyst's discoverable
   *  affordance — the keyboard shortcut alone has no visible cue. */
  const handleDeleteSelectedEdge = useCallback(() => {
    if (!selectedEdgeKey) return;
    const [src, tgt] = selectedEdgeKey.split("->");
    if (!src || !tgt) return;
    setEdgeOps((prev) => [...prev, { action: "remove", from: src, to: tgt }]);
    setSelectedEdgeKey(null);
  }, [selectedEdgeKey]);

  const handleFieldChange = useCallback(
    (field, value) => {
      if (!selectedId) return;
      if (isSyntheticId(selectedId)) {
        // Synthetic chunks don't go through the edits map — they live in
        // addedChunks and the AddedChunkItem schema is the source of truth.
        // Route field changes there so the canvas + submission stay in sync.
        setAddedChunks((prev) =>
          prev.map((a) => (a.tmp_id === selectedId ? { ...a, [field]: value } : a)),
        );
        return;
      }
      setEdits((prev) => ({
        ...prev,
        [selectedId]: { ...(prev[selectedId] || {}), [field]: value },
      }));
    },
    [selectedId],
  );

  const handleResetEdits = useCallback(() => {
    if (!selectedId) return;
    if (isSyntheticId(selectedId)) return;  // No "reset" surface for synthetic chunks; use Drop instead.
    setEdits((prev) => {
      const next = { ...prev };
      delete next[selectedId];
      return next;
    });
  }, [selectedId]);

  const handleToggleDrop = useCallback(() => {
    if (!selectedId) return;
    setDroppedIds((prev) => {
      const next = new Set(prev);
      if (next.has(selectedId)) next.delete(selectedId);
      else next.add(selectedId);
      return next;
    });
  }, [selectedId]);

  /** React Flow fires onConnect when the analyst drags from a source handle
   *  onto a target handle. We record the mutation; the next render's
   *  computeLiveEdges replay paints the new edge in green dashed style.
   *
   *  Edges involving synthetic added-chunk IDs are blocked: the backend
   *  allocates real IDs at processing time so we can't preserve them. The
   *  analyst gets no UI feedback besides the connection silently failing,
   *  which matches React Flow's convention; the AddChunkForm note explains
   *  the limitation up-front. */
  const handleConnect = useCallback((connection) => {
    const src = connection?.source;
    const tgt = connection?.target;
    if (!src || !tgt || src === tgt) return;
    if (isSyntheticId(src) || isSyntheticId(tgt)) return;
    setEdgeOps((prev) => [...prev, { action: "add", from: src, to: tgt }]);
  }, []);

  /** React Flow fires onEdgesDelete when the analyst presses Delete or
   *  Backspace on a selected edge. Record one remove op per deleted edge. */
  const handleEdgesDelete = useCallback((deleted) => {
    if (!deleted?.length) return;
    setEdgeOps((prev) => [
      ...prev,
      ...deleted.map((e) => ({ action: "remove", from: e.source, to: e.target })),
    ]);
  }, []);

  const handleAddChunk = useCallback((raw) => {
    setAddedChunks((prev) => {
      // random id slice — never reuses an id even across add/drop/add
      // cycles. The prior `prev.length + 1` scheme produced duplicate ids
      // when the analyst dropped an entry mid-sequence and added a new one,
      // causing React key collisions.
      const tmpId = `${SYNTHETIC_ID_PREFIX}${newId().slice(0, 8)}`;
      return [...prev, { ...raw, tmp_id: tmpId }];
    });
    setShowAddForm(false);
  }, []);

  const handleApprove = useCallback(() => {
    if (!onSubmit) return;
    const submission = buildSubmitPayload({
      originalChunks: payload?.chunks ?? [],
      edits,
      droppedIds,
      addedChunks,
      edgeOps,
      operatorOverrides,
      conditionEdits,
      mergeGroups,
      sequentialOverride,
    });
    onSubmit(submission, { willRerun: false });
  }, [onSubmit, payload?.chunks, edits, droppedIds, addedChunks, edgeOps, operatorOverrides, conditionEdits, mergeGroups, sequentialOverride]);

  // Merge the selected chunk INTO the given survivor. The survivor keeps its
  // identity and flow position; the selected chunk is absorbed.
  const handleMergeInto = useCallback((survivorId, absorbedId) => {
    if (!survivorId || !absorbedId || survivorId === absorbedId) return;
    if (isSyntheticId(survivorId) || isSyntheticId(absorbedId)) return;
    setMergeGroups((prev) => {
      const next = { ...prev };
      // Absorbing a chunk that is itself a survivor folds its group upward,
      // so a chain of merges collapses into one group rather than nesting.
      const inherited = next[absorbedId] || [];
      delete next[absorbedId];
      const existing = next[survivorId] || [];
      next[survivorId] = Array.from(
        new Set([...existing, absorbedId, ...inherited]),
      );
      return next;
    });
    setSelectedId(survivorId);
  }, []);

  const handleUnmerge = useCallback((survivorId) => {
    setMergeGroups((prev) => {
      const next = { ...prev };
      delete next[survivorId];
      return next;
    });
  }, []);

  const handleOperatorKindChange = useCallback((operatorId, kind) => {
    setOperatorOverrides((prev) => ({ ...prev, [operatorId]: kind }));
  }, []);

  /** Set/update a condition on a chunk. `fields` carries any subset of
   *  {description, pattern, pattern_type, on_true_ids, on_false_ids}.
   *  The gate processor does NOT merge: a "set" REPLACES the condition and
   *  one without a description is dropped outright. So the first edit to a
   *  chunk seeds from the chunker's condition in the payload, and every
   *  submitted "set" is whole. */
  const handleConditionSet = useCallback((chunkId, fields) => {
    const baseline = payload?.chunk_conditions?.[chunkId] || {};
    setConditionEdits((prev) => ({
      ...prev,
      [chunkId]: { ...baseline, ...(prev[chunkId] || {}), ...fields, action: "set" },
    }));
  }, [payload?.chunk_conditions]);

  /** Remove the condition on a chunk. */
  const handleConditionClear = useCallback((chunkId) => {
    setConditionEdits((prev) => ({ ...prev, [chunkId]: { action: "clear" } }));
  }, []);

  // ── AI reviewer application ────────────────────────────────────────
  //
  // Every apply snapshots the analyst's own state for that chunk first, so
  // undo restores what they had rather than resetting to the chunker's
  // output. Applying is otherwise expressed entirely in the same state the
  // analyst's own clicks write — there is no second, parallel "AI decisions"
  // channel that could disagree with the canvas at submit time.

  const handleApplyChunkRec = useCallback((chunkId) => {
    const rec = recsByChunk[chunkId];
    if (!rec) return;

    setPreApply((prev) => (
      chunkId in prev ? prev : {
        ...prev,
        [chunkId]: {
          edits: edits[chunkId],
          dropped: droppedIds.has(chunkId),
          merge: mergeGroups[chunkId],
        },
      }
    ));

    if (rec.action === "drop") {
      setDroppedIds((prev) => new Set(prev).add(chunkId));
    } else if (rec.action === "edit") {
      const patch = {};
      if (rec.edited_text) patch.text = rec.edited_text;
      if (rec.edited_source_excerpt) patch.source_excerpt = rec.edited_source_excerpt;
      // Chain fields: the reviewer's fix for a shared capability mis-rooted
      // as a chain (a kit made the sole entry point). Booleans are applied
      // as sent — `false` is a real edit here, not an absence.
      if (typeof rec.edited_chain_root === "boolean") patch.chain_root = rec.edited_chain_root;
      if (rec.edited_chain_label) patch.chain_label = rec.edited_chain_label;
      if (Object.keys(patch).length) {
        setEdits((prev) => ({ ...prev, [chunkId]: { ...(prev[chunkId] || {}), ...patch } }));
      }
    } else if (rec.action === "merge") {
      // Only ids that are real, present, and not this chunk. A merge naming
      // a chunk that isn't on the canvas would be dropped by the gate
      // anyway; filtering here means the canvas shows what will happen.
      const present = new Set(liveChunks.map((c) => c.chunk_id));
      const partners = (rec.merge_with ?? []).filter(
        (id) => id !== chunkId && present.has(id) && !isSyntheticId(id),
      );
      if (partners.length) {
        partners.forEach((id) => handleMergeInto(chunkId, id));
      }
    }
    // approve needs no mutation: gate_chunks keeps any chunk it isn't told
    // about, so recording the click is the whole effect.
    setAppliedRecs((prev) => new Set(prev).add(chunkId));
  }, [recsByChunk, edits, droppedIds, mergeGroups, liveChunks, handleMergeInto]);

  const handleDismissChunkRec = useCallback((chunkId) => {
    const restored = restoreChunkSnapshot(chunkId, preApply[chunkId], {
      edits, droppedIds, mergeGroups,
    });
    setEdits(restored.edits);
    setDroppedIds(restored.droppedIds);
    setMergeGroups(restored.mergeGroups);
    setPreApply((prev) => {
      const next = { ...prev };
      delete next[chunkId];
      return next;
    });
    setAppliedRecs((prev) => {
      const next = new Set(prev);
      next.delete(chunkId);
      return next;
    });
  }, [preApply, edits, droppedIds, mergeGroups]);

  /** Apply — or, with `undo`, reverse — one sequencing recommendation.
   *
   *  `edgeOps` is an append-only log replayed to derive the rendered edges,
   *  so the reversal of an op is simply the opposite op. No snapshot needed:
   *  the log IS the history. */
  const handleApplyEdgeRec = useCallback((rec, undo = false) => {
    if (!rec?.from_chunk_id || !rec?.to_chunk_id) return;
    const key = edgeRecKey(rec);
    const action = undo
      ? (rec.action === "add" ? "remove" : "add")
      : rec.action;
    setEdgeOps((prev) => [
      ...prev,
      { action, from: rec.from_chunk_id, to: rec.to_chunk_id },
    ]);
    setAppliedEdgeRecs((prev) => {
      const next = new Set(prev);
      if (undo) next.delete(key);
      else next.add(key);
      return next;
    });
  }, []);

  const handleApplyAddRec = useCallback((rec, index) => {
    const tmpId = `${SYNTHETIC_ID_PREFIX}${newId().slice(0, 8)}`;
    setAddedChunks((prev) => [...prev, {
      text: rec.text,
      source_excerpt: rec.source_excerpt ?? "",
      // The reviewer's confidence is a three-way label, not the 0-1 score
      // this field carries, and mapping between them would invent precision.
      // 0.7 is the same default an analyst-added chunk gets.
      behavioral_confidence: 0.7,
      branch_point: false,
      convergence_point: false,
      tmp_id: tmpId,
    }]);
    setAppliedAddRecs((prev) => ({ ...prev, [index]: tmpId }));
  }, []);

  const handleDismissAddRec = useCallback((index) => {
    const tmpId = appliedAddRecs[index];
    if (tmpId) setAddedChunks((prev) => prev.filter((a) => a.tmp_id !== tmpId));
    setAppliedAddRecs((prev) => {
      const next = { ...prev };
      delete next[index];
      return next;
    });
  }, [appliedAddRecs]);

  /** Every HIGH-confidence recommendation, across the three applicable
   *  channels. The re-chunk ask is deliberately NOT one of them: it discards
   *  the whole pass including the recommendations being accepted alongside
   *  it, which is not something a bulk button should be able to do. */
  const bulkAcceptable = useMemo(() => {
    const chunks = (recommendations?.chunks ?? []).filter(isBulkAcceptable);
    const adds = (recommendations?.added_chunks ?? [])
      .map((rec, index) => ({ rec, index }))
      .filter(({ rec }) => isBulkAcceptable(rec));
    const edgeRecs = pendingEdgeRecs(recommendations, liveEdgeKeys)
      .filter(isBulkAcceptable);
    return { chunks, adds, edgeRecs, total: chunks.length + adds.length + edgeRecs.length };
  }, [recommendations, liveEdgeKeys]);

  const handleAcceptAI = useCallback(() => {
    bulkAcceptable.chunks.forEach((rec) => handleApplyChunkRec(rec.chunk_id));
    bulkAcceptable.edgeRecs.forEach((rec) => handleApplyEdgeRec(rec));
    bulkAcceptable.adds
      .filter(({ index }) => !(index in appliedAddRecs))
      .forEach(({ rec, index }) => handleApplyAddRec(rec, index));
  }, [
    bulkAcceptable, appliedAddRecs,
    handleApplyChunkRec, handleApplyEdgeRec, handleApplyAddRec,
  ]);

  const handleReject = useCallback(({ reason, comments }) => {
    if (!onSubmit) return;
    const body = { reject: { reason, comments } };
    // The rerun chunker's prompt depends on the flag, so a flip must ride
    // along with the reject rather than wait for the next pass.
    if (typeof sequentialOverride === "boolean") body.is_sequential = sequentialOverride;
    onSubmit(body, { willRerun: true });
    setShowRejectModal(false);
  }, [onSubmit, sequentialOverride]);

  // Whether any local state would actually change the chunk set on submit.
  // Drives the "no-op approve" hint without disabling the button — the
  // analyst should still be able to confirm "yes, the LLM got it right."
  const hasChanges = useMemo(
    () =>
      Object.keys(edits).length > 0
      || droppedIds.size > 0
      || addedChunks.length > 0
      || edgeOps.length > 0
      || Object.keys(operatorOverrides).length > 0
      || Object.keys(conditionEdits).length > 0
      // Merges were missing here, so a merge-only pass showed "Approve
      // as-is" on a button that was about to submit a merge.
      || Object.keys(mergeGroups).length > 0
      || sequentialOverride !== null,
    [edits, droppedIds, addedChunks, edgeOps, operatorOverrides, conditionEdits, mergeGroups, sequentialOverride],
  );
  const effectiveSequential = sequentialOverride ?? payload?.is_sequential;

  const selectedChunk = useMemo(
    () => liveChunks.find((c) => c.chunk_id === selectedId) || null,
    [liveChunks, selectedId],
  );

  // Operators anchored on the currently-selected chunk. Surfaces the
  // operator-kind dropdowns in the side panel. A chunk can be both a
  // branch and a convergence (two operators, two dropdowns); both
  // appear simultaneously.
  const operatorsForSelected = useMemo(() => {
    if (!selectedId) return [];
    const ops = payload?.chunk_operators || {};
    return Object.entries(ops)
      .filter(([, meta]) => meta?.anchor_chunk_id === selectedId)
      .map(([operator_id, meta]) => ({ operator_id, ...meta }));
  }, [selectedId, payload?.chunk_operators]);

  // Resolved condition for the selected chunk. Local analyst edits
  // (conditionEdits[selectedId]) win over the payload condition; a
  // "clear" edit returns null so the side panel hides the section.
  const conditionForSelected = useMemo(() => {
    if (!selectedId) return null;
    const edit = conditionEdits[selectedId];
    if (edit) {
      if (edit.action === "clear") return null;
      // Merge any edit fields over the payload baseline so the form
      // shows the analyst's latest values even when only one field
      // was touched.
      const base = payload?.chunk_conditions?.[selectedId] || {};
      return { ...base, ...edit };
    }
    return payload?.chunk_conditions?.[selectedId] || null;
  }, [selectedId, conditionEdits, payload?.chunk_conditions]);

  if (!payload?.chunks?.length) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-gb-fg4">
        <p className="text-[13px]">No chunks were extracted from this source.</p>
        <p className="text-[11px] mt-2">
          You can reject and re-run chunking, or approve the empty result to skip technique extraction.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      {/* Turn one of the reviewer's transcript, editable. A correction here
          propagates to every later gate instead of being re-argued at each.
          Guarded like the other three gates: the brief arrives after the
          canvas mounts, and it is only meaningful with a save handler. */}
      {brief && onSaveBrief && (
        <ReviewerBrief brief={brief} onSave={onSaveBrief} saving={savingBrief} />
      )}

      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="text-[11px] text-gb-fg4 font-data flex items-center gap-3 flex-wrap">
          {/* Sequentiality chip: explains why the canvas may have many
              disconnected components (catalog mode) vs a single connected
              DAG (sequential mode). Hover for the auto-detect rationale;
              click to override it. The override ships with the submit and
              is what decides whether any edge drawn here reaches the
              bundle — a source misread as a catalogue otherwise loses
              every PRECEDES edge at serialization. */}
          {typeof payload.is_sequential === "boolean" && (
            <>
              <button
                type="button"
                onClick={() => setSequentialOverride((v) => (v === null ? !payload.is_sequential : null))}
                className={`px-1.5 py-0.5 rounded font-medium cursor-pointer ${
                  effectiveSequential
                    ? "bg-gb-bright-blue/15 text-gb-bright-blue border border-gb-bright-blue/40"
                    : "bg-gb-bright-orange/15 text-gb-bright-orange border border-gb-bright-orange/40"
                }`}
                title={payload.sequentiality_rationale || "No rationale available"}
              >
                Sequential: {effectiveSequential ? "yes" : "no"}
                {sequentialOverride !== null ? " (overridden)" : ""}
              </button>
              <InfoDot term="sequential" />
              <span>·</span>
            </>
          )}
          <span>{liveChunks.length} chunks</span>
          <span>·</span>
          <span>
            {edges.length} precedes edges<InfoDot term="precedes" />
          </span>
          <span>·</span>
          <span>{(payload.parsed_text ?? "").length.toLocaleString()} chars of source</span>
          {editedIds.size > 0 && (
            <>
              <span>·</span>
              <span className="text-gb-bright-yellow">{editedIds.size} edited</span>
            </>
          )}
          {droppedIds.size > 0 && (
            <>
              <span>·</span>
              <span className="text-gb-bright-red">{droppedIds.size} dropped</span>
            </>
          )}
          {addedChunks.length > 0 && (
            <>
              <span>·</span>
              <span className="text-gb-bright-green">+{addedChunks.length} chunks</span>
            </>
          )}
          {addedEdgeCount > 0 && (
            <>
              <span>·</span>
              <span className="text-gb-bright-green">+{addedEdgeCount} edges</span>
            </>
          )}
        </div>
        <div className="flex items-center gap-2">
          {/* Only HIGH-confidence recommendations — see handleAcceptAI. */}
          {bulkAcceptable.total > 0 && (
            <button
              type="button"
              onClick={handleAcceptAI}
              disabled={submitting}
              title="Applies only the AI's high-confidence recommendations. Medium and low stay for you to judge, and a re-chunk is never included."
              className="px-3 py-1.5 rounded text-[12px] font-medium bg-gb-purple/10 text-gb-bright-purple border border-gb-purple/50 hover:bg-gb-purple/15 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
            >
              🤖 Accept {bulkAcceptable.total} high-confidence
            </button>
          )}
          <button
            type="button"
            onClick={() => { setShowAddForm(true); setSelectedId(null); }}
            disabled={submitting || showAddForm}
            className="px-3 py-1.5 rounded text-[12px] font-medium bg-gb-bright-green/10 text-gb-bright-green border border-gb-bright-green/40 hover:bg-gb-bright-green/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            + Add chunk
          </button>
          <button
            type="button"
            onClick={() => setShowRejectModal(true)}
            disabled={submitting}
            className="px-3 py-1.5 rounded text-[12px] font-medium bg-gb-bright-orange/10 text-gb-bright-orange border border-gb-bright-orange/40 hover:bg-gb-bright-orange/20 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            ↻ Reject &amp; rerun
          </button>
          <button
            type="button"
            onClick={handleApprove}
            disabled={submitting}
            title={hasChanges ? `Submit ${editedIds.size} edits, ${droppedIds.size} drops, ${addedChunks.length} adds, ${addedEdgeCount} edge changes` : "Approve all chunks as-is"}
            className="px-3 py-1.5 rounded text-[12px] font-semibold bg-gb-bright-aqua/20 text-gb-bright-aqua border border-gb-bright-aqua/60 hover:bg-gb-bright-aqua/30 transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
          >
            {submitting ? "Submitting…" : hasChanges ? "✓ Submit changes" : "✓ Approve as-is"}
          </button>
        </div>
      </div>

      <div
        className="grid gap-3"
        style={{ height: "70vh", minHeight: 420, gridTemplateColumns: "360px 1fr 320px" }}
      >
        <SourcePane
          parsedText={payload.parsed_text}
          chunks={liveChunks}
          selectedId={selectedId}
          editedIds={editedIds}
          droppedIds={droppedIds}
          onSelectChunk={setSelectedId}
        />

        <div className="rounded-lg border border-gb-bg2 bg-gb-bg0 overflow-hidden">
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={NODE_TYPES}
            fitView
            fitViewOptions={FIT_VIEW_OPTIONS}
            proOptions={{ hideAttribution: true }}
            nodesDraggable
            nodesConnectable
            elementsSelectable
            edgesFocusable
            deleteKeyCode={["Backspace", "Delete"]}
            onNodeClick={handleNodeClick}
            onEdgeClick={handleEdgeClick}
            onPaneClick={handlePaneClick}
            onConnect={handleConnect}
            onEdgesDelete={handleEdgesDelete}
            defaultEdgeOptions={DEFAULT_EDGE_OPTIONS}
          >
            <Background gap={16} size={1} color="rgba(255,255,255,0.04)" />
            <MiniMap
              pannable
              zoomable
              style={{ background: "rgba(0,0,0,0.4)" }}
              nodeStrokeWidth={2}
              maskColor="rgba(0,0,0,0.5)"
            />
            <Controls showInteractive={false} />
          </ReactFlow>
        </div>

        {showAddForm ? (
          <AddChunkForm onAdd={handleAddChunk} onCancel={() => setShowAddForm(false)} />
        ) : selectedEdgeKey ? (
          <EdgePanel
            edgeKey={selectedEdgeKey}
            chunkLookup={Object.fromEntries(
              liveChunks.map((c) => [c.chunk_id, c])
            )}
            onDelete={handleDeleteSelectedEdge}
          />
        ) : (
          <SidePanel
            chunk={selectedChunk}
            hasEdits={selectedId ? editedIds.has(selectedId) : false}
            isDropped={selectedId ? droppedIds.has(selectedId) : false}
            onFieldChange={handleFieldChange}
            onResetEdits={handleResetEdits}
            onToggleDrop={handleToggleDrop}
            operatorsForChunk={operatorsForSelected}
            operatorOverrides={operatorOverrides}
            onOperatorKindChange={handleOperatorKindChange}
            condition={conditionForSelected}
            onConditionSet={handleConditionSet}
            onConditionClear={handleConditionClear}
            mergeGroups={mergeGroups}
            mergeCandidates={liveChunks.filter(
              (c) => c.chunk_id !== selectedId && !c._isAdded,
            )}
            onMergeInto={handleMergeInto}
            onUnmerge={handleUnmerge}
            rec={selectedId ? recsByChunk[selectedId] || null : null}
            recApplied={selectedId ? appliedRecs.has(selectedId) : false}
            onApplyRec={() => selectedId && handleApplyChunkRec(selectedId)}
            onDismissRec={() => selectedId && handleDismissChunkRec(selectedId)}
            unanchored={
              <UnanchoredRecs
                recommendations={recommendations}
                appliedAdds={new Set(Object.keys(appliedAddRecs).map(Number))}
                appliedEdges={appliedEdgeRecs}
                liveEdgeKeys={liveEdgeKeys}
                onApplyAdd={handleApplyAddRec}
                onDismissAdd={handleDismissAddRec}
                onApplyEdge={handleApplyEdgeRec}
                onOpenReject={(rej) => {
                  setRejectPrefill(rej);
                  setShowRejectModal(true);
                }}
              />
            }
          />
        )}
      </div>

      <p className="text-[11px] text-gb-fg4">
        Edits, drops, merges, edge mutations and added chunks are submitted
        together when you approve. Rejecting re-runs chunking with your
        comments injected as a high-priority hint in the prompt — which
        discards every chunk and every edit on this pass.
      </p>

      {showRejectModal && (
        <RejectModal
          onCancel={() => { setShowRejectModal(false); setRejectPrefill(null); }}
          onConfirm={handleReject}
          submitting={submitting}
          initialReason={rejectPrefill?.reason ?? "missed_procedures"}
          initialComments={
            rejectPrefill
              ? [rejectPrefill.rationale, rejectPrefill.comments]
                  .filter(Boolean).join("\n\n")
              : ""
          }
        />
      )}
    </div>
  );
}
