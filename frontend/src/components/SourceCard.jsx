/**
 * SourceCard — a single Kanban card representing a source in the queue.
 *
 * Shows: title, type tag, reliability, analyst avatar,
 * timestamp, progress bar (when processing), and gate CTA (when at a gate).
 */
import { useState } from "react";
import { Draggable } from "@hello-pangea/dnd";
import CorrectionsModal, {
  CORRECTIONS_SEVERITY_STYLES,
  highestSeverity,
} from "./CorrectionsModal";
import {
  GATE_ENABLE_KEYS,
  GATE_ENABLE_LABELS,
  GATE_LABELS,
  GATE_STATUSES,
  UNATTENDED_GATE_MODES,
} from "../lib/gates";
import { hint } from "../lib/glossary";
import {
  PROCESSING_STATUSES,
  statusColor,
  statusLabel,
  statusProgress,
} from "../lib/pipelineStatus";


function timeAgo(dateStr) {
  if (!dateStr) return "";
  const diff = Date.now() - new Date(dateStr).getTime();
  const mins = Math.floor(diff / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

/** Build a tooltip string for the corrections chip showing the per-severity
 *  breakdown. Skips severities with zero entries. Trailing line tells the
 *  analyst the chip is clickable. Uses proper noun forms ("repair" not
 *  "repaired") so plurals work cleanly. */
function correctionsTooltip(corrections) {
  const counts = { hard_fail: 0, repaired: 0, warn: 0, auto_fix: 0 };
  for (const c of corrections) {
    const sev = c?.severity;
    if (sev in counts) counts[sev]++;
  }
  const nounForms = {
    hard_fail: ["failure", "failures"],
    repaired: ["repair", "repairs"],
    warn: ["warning", "warnings"],
    auto_fix: ["fix", "fixes"],
  };
  const parts = ["hard_fail", "repaired", "warn", "auto_fix"]
    .filter((s) => counts[s] > 0)
    .map((s) => {
      const [singular, plural] = nounForms[s];
      return `${counts[s]} ${counts[s] === 1 ? singular : plural}`;
    });
  return `${parts.join(", ")} — click for detail`;
}

export default function SourceCard({ source, index, onGateReview, onStartPipeline, onDelete }) {
  // Gates this source runs with no human: enabled, but set to let the AI
  // decide. Computed here rather than on the server so a card reflects the
  // setting even before the run reaches that gate.
  const modes = source.gate_modes || {};
  const enabled = source.gates_enabled || {};
  const unattendedGates = GATE_ENABLE_KEYS.filter(
    (k) => enabled[k] !== false && UNATTENDED_GATE_MODES.has(modes[k]),
  ).map((k) => GATE_ENABLE_LABELS[k]);

  const isGate = GATE_STATUSES.has(source.status);
  const isProcessing = PROCESSING_STATUSES.has(source.status);
  const isComplete = source.status === "completed";
  const isQueued = source.status === "queued";
  const isFailed = source.status === "failed";
  const canRun = (isQueued || isFailed) && typeof onStartPipeline === "function";
  const progress = statusProgress(source.status);

  // Bundle validator corrections — surfaced as a chip on the card with
  // severity-aware styling. Color follows the highest severity present
  // (hard_fail > repaired > warn > auto_fix) so a glance tells the
  // analyst whether to investigate.
  const corrections = Array.isArray(source.bundle_corrections)
    ? source.bundle_corrections
    : [];
  const correctionsSeverity = highestSeverity(corrections);
  const [correctionsOpen, setCorrectionsOpen] = useState(false);

  return (
    <Draggable draggableId={source.id} index={index}>
      {(provided, snapshot) => (
        <div
          ref={provided.innerRef}
          {...provided.draggableProps}
          {...provided.dragHandleProps}
          className={`
            rounded-lg p-3 cursor-grab transition-all
            ${isGate
              ? "bg-gb-bg0-s border border-gb-orange border-l-[3px] border-l-gb-bright-orange"
              : isComplete
                ? "bg-gb-bg0-s border border-gb-green"
                : "bg-gb-bg0-s border border-gb-bg2"
            }
            ${snapshot.isDragging ? "shadow-lg ring-1 ring-gb-bright-blue" : ""}
            hover:border-gb-bright-blue hover:shadow-[0_0_0_1px_var(--color-gb-bright-blue-dim)]
          `}
        >
          {/* Title + delete */}
          <div className="flex items-start justify-between gap-2 mb-1.5">
            <h3 className="text-[13px] font-semibold text-gb-fg0 leading-tight flex-1 min-w-0">
              {source.title}
            </h3>
            {typeof onDelete === "function" && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  onDelete(source);
                }}
                title="Delete source"
                aria-label="Delete source"
                className="shrink-0 text-gb-gray hover:text-gb-bright-red transition-colors leading-none text-[14px] px-1 -mt-0.5"
              >
                ×
              </button>
            )}
          </div>

          {/* Tags */}
          <div className="flex flex-wrap gap-1.5 mb-2">
            <span className="font-data text-[10px] font-medium px-1.5 py-0.5 rounded bg-gb-tag-type-bg text-gb-bright-blue border border-gb-blue">
              {source.source_type}
            </span>
            {/* The channel tag is gone: with the field removed from Add Source
                every card read "manual", so it distinguished nothing. */}
            {/* Unattended: at least one gate is set to let the AI decide, so
                part of this bundle was never seen by a person. Worth saying on
                the card — otherwise a source nobody reviewed is visually
                identical to one that was reviewed carefully, and the Reviewer
                tab has no data for it to explain the difference. */}
            {unattendedGates.length > 0 && (
              <span
                className="font-data text-[10px] font-medium px-1.5 py-0.5 rounded bg-gb-bright-purple/15 text-gb-bright-purple border border-gb-purple cursor-help"
                title={
                  `AI decided unattended at: ${unattendedGates.join(", ")}. ` +
                  "No human reviewed these stages."
                }
              >
                ⚡ unattended
              </span>
            )}
            {source.source_reliability != null && (
              <span
                className="font-data text-[10px] font-medium px-1.5 py-0.5 rounded bg-gb-tag-reliability-bg text-gb-bright-green border border-gb-green"
                title={hint("source-reliability")}
              >
                rel: {source.source_reliability}
              </span>
            )}
            {isGate && (
              <span className="font-data text-[10px] font-medium px-1.5 py-0.5 rounded bg-gb-tag-gate-bg text-gb-bright-orange border border-gb-orange">
                {GATE_LABELS[source.status] || source.status}
              </span>
            )}
            {isComplete && source.objects_written != null && (
              <span className="font-data text-[10px] font-medium px-1.5 py-0.5 rounded bg-gb-tag-reliability-bg text-gb-bright-green border border-gb-green">
                {source.objects_written} objects
              </span>
            )}
            {isComplete &&
              Array.isArray(source.persistence_errors) &&
              source.persistence_errors.length > 0 && (
                <span
                  title={source.persistence_errors.join("\n")}
                  className="font-data text-[10px] font-medium px-1.5 py-0.5 rounded bg-gb-tag-gate-bg text-gb-bright-yellow border border-gb-yellow cursor-help"
                >
                  ⚠ {source.persistence_errors.length} persistence warning
                  {source.persistence_errors.length === 1 ? "" : "s"}
                </span>
              )}
            {correctionsSeverity && (
              <button
                type="button"
                onClick={(e) => {
                  e.stopPropagation();
                  setCorrectionsOpen(true);
                }}
                title={correctionsTooltip(corrections)}
                className={`font-data text-[10px] font-medium px-1.5 py-0.5 rounded border transition-colors hover:brightness-125 ${CORRECTIONS_SEVERITY_STYLES[correctionsSeverity].badge}`}
              >
                {CORRECTIONS_SEVERITY_STYLES[correctionsSeverity].icon}{" "}
                {corrections.length} correction
                {corrections.length === 1 ? "" : "s"}
              </button>
            )}
            {isFailed && source.error && (
              <span
                title={source.error}
                className="font-data text-[10px] font-medium px-1.5 py-0.5 rounded bg-[rgba(251,73,52,0.12)] text-gb-bright-red border border-gb-red cursor-help max-w-[240px] truncate inline-block"
              >
                ✕ {source.error.length > 40 ? source.error.slice(0, 40) + "…" : source.error}
              </span>
            )}
          </div>

          {/* Gate summary + CTA */}
          {isGate && (
            <>
              <p className="font-data text-[11px] text-gb-fg4 mt-1">
                {source.entity_count != null && `${source.entity_count} entities`}
                {source.draft_count != null && source.draft_count > 0 && ` · ${source.draft_count} drafts`}
              </p>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onGateReview?.(source);
                }}
                className="font-data block w-full mt-2 py-1 rounded text-[11px] font-semibold border border-gb-orange bg-gb-tag-gate-bg text-gb-bright-orange text-center transition-colors hover:bg-gb-bg2 hover:border-gb-bright-orange"
              >
                Review {GATE_LABELS[source.status]?.split(" — ")[1] || "Items"} →
              </button>
            </>
          )}

          {/* Run button for queued/failed sources.
              Click = run with the gates chosen in Add Source.
              Shift+Click = run unattended (every gate auto-approves). */}
          {canRun && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onStartPipeline(source, e.shiftKey);
              }}
              title="Click to run with the gates you chose. Shift+Click to auto-approve every gate."
              className="font-data block w-full mt-2 py-1 rounded text-[11px] font-semibold border border-gb-green bg-gb-tag-reliability-bg text-gb-bright-green text-center transition-colors hover:bg-gb-bg2 hover:border-gb-bright-green"
            >
              {isFailed ? "Retry ▶" : "Run ▶"}
            </button>
          )}

          {/* Processing progress bar */}
          {isProcessing && (
            <div className="h-[3px] bg-gb-bg1 rounded-sm mt-2 overflow-hidden">
              <div
                className="h-full bg-gb-bright-blue rounded-sm transition-[width] duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
          )}

          {/* Footer: status + time */}
          <div className="flex items-center justify-between mt-2 pt-2 border-t border-gb-bg1">
            <span className={`font-data text-[11px] font-semibold ${statusColor(source.status)}`}>
              {statusLabel(source.status)}
            </span>
            <span className="font-data text-[11px] text-gb-gray">
              {timeAgo(source.updated_at)}
            </span>
          </div>
          <CorrectionsModal
            isOpen={correctionsOpen}
            corrections={corrections}
            title={`Validator corrections — ${source.title}`}
            onClose={() => setCorrectionsOpen(false)}
          />
        </div>
      )}
    </Draggable>
  );
}
