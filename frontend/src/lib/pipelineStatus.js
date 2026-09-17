/**
 * lib/pipelineStatus.js — the one table of pipeline statuses.
 *
 * Mirror of the backend's PipelineStatus enum (app/graph/state.py). Every
 * component that needs to know something about a status — which Kanban
 * column it sits in, its label and color on the card, how far along the
 * progress bar is, whether the WebSocket should stay open for it, whether
 * the stats bar counts it as processing — derives that from here.
 *
 * Before this table there were five hand-listed copies across four files.
 * Each status-adding commit updated the ones a contract test scanned and
 * missed the ones it did not: the stats bar undercounted "Processing" for
 * every AI-reviewer phase and the whole figure-extraction pass, and a card
 * in a status no column listed simply vanished from the board.
 *
 * Gate pauses and their resuming_from_* successors are derived from
 * lib/gates.js rather than repeated here. tests/test_contracts.py scans this
 * file and gates.js against the backend enum; keep one literal per status.
 */
import { GATES } from "./gates";

const BLUE = "text-gb-bright-blue";
const PURPLE = "text-gb-bright-purple";

/** Kanban columns, in board order. KanbanBoard owns labels and accents. */
export const COLUMN_IDS = [
  "queued", "entity_review", "procedure_review", "technique_review",
  "bundle_review", "complete", "failed",
];

// Non-gate statuses, in pipeline order. `terminal` marks the states a
// source is *left* in — no live WebSocket, not "processing".
const NON_GATE_STATUSES = [
  { status: "queued", label: "Queued", color: "text-gb-fg4", progress: 0, column: "queued", terminal: true },
  { status: "parsing", label: "Parsing...", color: BLUE, progress: 10, column: "queued" },
  { status: "extracting_figures", label: "Reading figures...", color: BLUE, progress: 15, column: "queued" },
  { status: "classifying_sections", label: "Mapping sections...", color: BLUE, progress: 18, column: "queued" },
  { status: "extracting_entities", label: "Extracting entities...", color: BLUE, progress: 20, column: "queued" },
  { status: "reviewing_entities", label: "AI reviewing entities...", color: PURPLE, progress: 24, column: "entity_review" },
  { status: "chunking", label: "Chunking...", color: BLUE, progress: 35, column: "procedure_review" },
  { status: "reviewing_chunks", label: "AI reviewing procedures...", color: PURPLE, progress: 38, column: "procedure_review" },
  { status: "extracting_techniques", label: "Mapping techniques...", color: BLUE, progress: 55, column: "technique_review" },
  // The `procedures` gate key is user-facing "Gate 2 — Techniques"; the
  // chunk gate is the one analysts call procedure review. Labeling both
  // "procedures" would put the same words in two different columns.
  { status: "reviewing_procedures", label: "AI reviewing techniques...", color: PURPLE, progress: 62, column: "technique_review" },
  { status: "drafting", label: "Drafting procedures...", color: BLUE, progress: 65, column: "technique_review" },
  { status: "normalizing", label: "Normalizing...", color: BLUE, progress: 80, column: "bundle_review" },
  { status: "reviewing_bundle", label: "AI reviewing bundle...", color: PURPLE, progress: 82, column: "bundle_review" },
  { status: "serializing", label: "Serializing STIX...", color: BLUE, progress: 90, column: "complete" },
  { status: "distributing", label: "Writing to Neo4j...", color: BLUE, progress: 95, column: "complete" },
  { status: "synthesizing_feedback", label: "Learning from review...", color: BLUE, progress: 98, column: "complete" },
  { status: "completed", label: "Complete", color: "text-gb-bright-green", progress: 100, column: "complete", terminal: true },
  { status: "failed", label: "Failed", color: "text-gb-bright-red", progress: 0, column: "failed", terminal: true },
];

// A gate pause, and the transient status the gate node writes after
// applying decisions so the card stays in its column while the pipeline
// restarts (see PipelineStatus.RESUMING_FROM_GATE_N in the backend).
const GATE_STATUSES = GATES.flatMap((g) => [
  { status: g.status, label: g.statusLabel, color: g.statusColor, progress: 0, column: g.column, gate: true },
  { status: g.nextStatus, label: "Resuming...", color: BLUE, progress: 0, column: g.column },
]);

export const PIPELINE_STATUSES = [...NON_GATE_STATUSES, ...GATE_STATUSES];

const BY_ID = Object.fromEntries(PIPELINE_STATUSES.map((s) => [s.status, s]));

/** Statuses a source is in-flight in: keep the WebSocket subscribed. */
export const ACTIVE_STATUSES = new Set(
  PIPELINE_STATUSES.filter((s) => !s.terminal).map((s) => s.status),
);

/** In-flight and not waiting on a human: the card shows a progress bar
 *  and the stats bar counts it as processing. */
export const PROCESSING_STATUSES = new Set(
  PIPELINE_STATUSES.filter((s) => !s.terminal && !s.gate).map((s) => s.status),
);

/** Statuses that live in one Kanban column, in table order. */
export function statusesInColumn(columnId) {
  return PIPELINE_STATUSES.filter((s) => s.column === columnId).map((s) => s.status);
}

/** Human label, or the raw status for one this table does not know
 *  (better a visible enum value than a blank). */
export function statusLabel(status) {
  return BY_ID[status]?.label ?? status;
}

export function statusColor(status) {
  return BY_ID[status]?.color ?? "text-gb-fg4";
}

export function statusProgress(status) {
  return BY_ID[status]?.progress ?? 0;
}
