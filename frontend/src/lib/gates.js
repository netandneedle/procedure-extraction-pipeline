/**
 * lib/gates.js — single source of truth for gate metadata.
 *
 * Python identifiers and user-facing numbering differ: backend node names
 * are positional (gate_0, gate_1, gate_2) and the chunk-as-procedure gate
 * (gate_chunks) sits between gate_0 and gate_1. To analysts this reads as
 * Gate 0/1/2/3.
 *
 * Adding a new gate is one place: append a Gate entry to GATES below. All
 * per-component derived maps (titles, labels, status mappings, etc.) are
 * computed from this list at module load.
 *
 * Mirror of the backend `_gate_registry.py`; keep field semantics aligned.
 */

/** Display order matches the pipeline's gate sequence.
 *  `column` is the Kanban column (lib/pipelineStatus.js COLUMN_IDS) the
 *  gate pause and its resuming_from_* successor sit in. */
export const GATES = [
  {
    status: "gate_0",
    column: "entity_review",
    enableKey: "entities",
    routeId: 0,
    label: "Gate 0 — Entities",
    title: "Gate 0 — Entity Review",
    enableLabel: "Entity Review",
    hintTerm: "gate-entities",
    statusLabel: "Entity review",
    statusColor: "text-gb-bright-yellow",
    shortLabel: "G0",
    nextStatus: "resuming_from_gate_0",
    // An AI reviewer exists for this gate. Gates without one fall through to
    // human review whatever their mode says (backend enforces this too, in
    // pipeline._REVIEWING_STATUS) — so this flag controls what the UI is
    // allowed to OFFER, not what it can force.
    aiReviewer: true,
  },
  {
    status: "gate_chunks",
    column: "procedure_review",
    enableKey: "chunks",
    routeId: "chunks",
    label: "Gate 1 — Procedures",
    title: "Gate 1 — Procedure Review",
    enableLabel: "Procedure Review",
    hintTerm: "gate-chunks",
    statusLabel: "Procedure review",
    statusColor: "text-gb-bright-orange",
    shortLabel: "G1",
    nextStatus: "resuming_from_gate_chunks",
    aiReviewer: true,
  },
  {
    status: "gate_1",
    column: "technique_review",
    enableKey: "procedures",
    routeId: 1,
    label: "Gate 2 — Techniques",
    title: "Gate 2 — Technique Review",
    enableLabel: "Technique Review",
    hintTerm: "gate-procedures",
    statusLabel: "Technique review",
    statusColor: "text-gb-bright-aqua",
    shortLabel: "G2",
    nextStatus: "resuming_from_gate_1",
    aiReviewer: true,
  },
  {
    status: "gate_2",
    column: "bundle_review",
    enableKey: "bundle",
    routeId: 2,
    label: "Gate 3 — Bundle",
    title: "Gate 3 — Bundle Review",
    enableLabel: "Bundle Review",
    hintTerm: "gate-bundle",
    statusLabel: "Bundle review",
    statusColor: "text-gb-bright-purple",
    shortLabel: "G3",
    nextStatus: "resuming_from_gate_2",
    aiReviewer: true,
  },
];

/** Status name -> route id ("chunks" for string-keyed gate, int otherwise). */
export const STATUS_TO_GATE = Object.fromEntries(
  GATES.map((g) => [g.status, g.routeId])
);

/** Route id -> user-facing title for the slide-over panel header. */
export const GATE_TITLES = Object.fromEntries(
  GATES.map((g) => [g.routeId, g.title])
);

/** Route id -> next pipeline status after a non-rerun submit. */
export const GATE_NEXT_STATUS = Object.fromEntries(
  GATES.map((g) => [g.routeId, g.nextStatus])
);

/** Status name -> compact label used on SourceCard chips + Review CTA. */
export const GATE_LABELS = Object.fromEntries(
  GATES.map((g) => [g.status, g.label])
);

/** Status name -> Tailwind color class for the SourceCard footer label. */
export const GATE_STATUS_COLORS = Object.fromEntries(
  GATES.map((g) => [g.status, g.statusColor])
);

/** Status name -> SourceCard footer status label. */
export const GATE_STATUS_LABELS = Object.fromEntries(
  GATES.map((g) => [g.status, g.statusLabel])
);

/** Backend gates_enabled keys in pipeline order — drives AddSourceModal. */
export const GATE_ENABLE_KEYS = GATES.map((g) => g.enableKey);

/** Backend gates_enabled key -> checkbox label. */
export const GATE_ENABLE_LABELS = Object.fromEntries(
  GATES.map((g) => [g.enableKey, g.enableLabel])
);

/** Backend gates_enabled key -> lib/glossary.js term describing what you
 *  actually review at that gate. */
export const GATE_HINT_TERMS = Object.fromEntries(
  GATES.map((g) => [g.enableKey, g.hintTerm])
);

/** Ordered review-chip descriptors for StatsBar. `title` is what the compact
 *  G0/G1/G2/G3 label stands for — without it the chips are unreadable to
 *  anyone who has not memorized the numbering. */
export const REVIEW_GATES = GATES.map((g) => ({
  status: g.status,
  label: g.shortLabel,
  title: g.title,
}));

/** Set of gate-pause statuses — for `isGate` checks. */
export const GATE_STATUSES = new Set(GATES.map((g) => g.status));

/**
 * Backend correction "gate area" -> gate status. Areas are the keys the
 * feedback flywheel uses for captured corrections / delta scoring
 * (feedback_synthesis._AREA_FOR_CATEGORY); they don't all match enableKey
 * ("techniques" reviews happen at gate_1 whose enableKey is "procedures"),
 * hence the explicit map. Lets correction UIs derive per-gate accents from
 * GATES instead of hardcoding a parallel palette.
 */
export const AREA_TO_GATE_STATUS = {
  entities: "gate_0",
  chunks: "gate_chunks",
  techniques: "gate_1",
  relationships: "gate_2",
};

/**
 * Per-gate review MODE — how an ENABLED gate gets reviewed.
 *
 * Separate from the enable/disable flag on purpose. `gates_enabled` decides
 * whether a gate is reviewed at all; mode decides by whom. See the backend's
 * app.graph.state.normalize_gate_modes for why they are not one field.
 *
 * "auto" is the third: the AI decides and the pipeline keeps going with nobody
 * watching. It is listed last and styled as the exception because it is one —
 * a gate on auto produces no agreement data, so the Reviewer tab goes quiet
 * for it. That is the trade, not an oversight.
 */
export const GATE_MODE_OPTIONS = [
  {
    value: "review",
    label: "Human review",
    help: "You review this gate yourself. Default.",
  },
  {
    value: "assist",
    label: "AI Assisted Review",
    help: "An AI analyst reads the report and recommends; you still decide.",
  },
  {
    value: "auto",
    label: "AI decides (unattended)",
    help:
      "The AI applies its own decisions and the pipeline continues without " +
      "pausing. Nothing is reviewed, and this gate stops contributing to the " +
      "Reviewer tab.",
  },
];

/** Modes where the pipeline runs a gate without a human. */
export const UNATTENDED_GATE_MODES = new Set(["auto"]);

/** Gate enable keys that have an AI reviewer implemented. */
export const GATE_AI_REVIEWER_KEYS = new Set(
  GATES.filter((g) => g.aiReviewer).map((g) => g.enableKey)
);
