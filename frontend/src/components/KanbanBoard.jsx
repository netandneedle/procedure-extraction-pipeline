/**
 * KanbanBoard — main orchestrator for the source queue.
 *
 * Collapses every PipelineStatus value into 7 human-readable columns:
 *   Queued | Entity Review | Procedure Review | Technique Review |
 *   Bundle Review | Complete | Failed
 *
 * Column names follow the user-facing gate numbering, not the backend node
 * names: "Procedure Review" is gate_chunks (chunks are the procedures),
 * "Technique Review" is gate_1 (the post-draft technique mapping), and
 * "Bundle Review" is gate_2.
 *
 * Processing and serialization statuses appear as status labels on the card
 * rather than as dedicated columns.
 *
 * Uses @hello-pangea/dnd for drag-and-drop between columns.
 */
import { useCallback, useMemo } from "react";
import { DragDropContext, Droppable } from "@hello-pangea/dnd";
import SourceCard from "./SourceCard";
import { statusesInColumn } from "../lib/pipelineStatus";

/** Column definitions: id, label, accent dot class. Which statuses map here
 *  comes from the one status table in lib/pipelineStatus.js.
 *  accentClass uses a static Tailwind class (not interpolated) so the class
 *  is always present in the source for Tailwind's scanner to detect. */
const COLUMNS = [
  {
    id: "queued",
    statuses: statusesInColumn("queued"),
    label: "Queued",
    accentClass: "bg-gb-fg4",
  },
  {
    id: "entity_review",
    statuses: statusesInColumn("entity_review"),
    label: "Entity Review",
    accentClass: "bg-gb-bright-yellow",
  },
  {
    id: "procedure_review",
    statuses: statusesInColumn("procedure_review"),
    label: "Procedure Review",
    accentClass: "bg-gb-bright-orange",
  },
  {
    id: "technique_review",
    statuses: statusesInColumn("technique_review"),
    label: "Technique Review",
    accentClass: "bg-gb-bright-aqua",
  },
  {
    id: "bundle_review",
    statuses: statusesInColumn("bundle_review"),
    label: "Bundle Review",
    accentClass: "bg-gb-bright-purple",
  },
  {
    id: "complete",
    statuses: statusesInColumn("complete"),
    label: "Complete",
    accentClass: "bg-gb-bright-green",
  },
  {
    id: "failed",
    statuses: statusesInColumn("failed"),
    label: "Failed",
    accentClass: "bg-gb-bright-red",
  },
];

/** Quick lookup: status string -> column id. */
const STATUS_TO_COLUMN = {};
COLUMNS.forEach((col) => {
  col.statuses.forEach((s) => {
    STATUS_TO_COLUMN[s] = col.id;
  });
});

/**
 * When a card is dropped into a new column, determine the target status.
 * Only "queued" is a meaningful manual move target — processing columns
 * are driven by the pipeline, not drag-and-drop.
 */
const COLUMN_DROP_STATUS = {
  queued: "queued",
  failed: "failed",
};

export default function KanbanBoard({ sources, onStatusChange, onGateReview, onStartPipeline, onDelete }) {
  /** Bucket sources into columns. */
  const columnData = useMemo(() => {
    const buckets = {};
    COLUMNS.forEach((col) => {
      buckets[col.id] = [];
    });
    sources.forEach((src) => {
      const colId = STATUS_TO_COLUMN[src.status] || "queued";
      buckets[colId].push(src);
    });
    return buckets;
  }, [sources]);

  /** Handle drag end: move card between columns when allowed. */
  const handleDragEnd = useCallback(
    (result) => {
      const { destination, source: dragSource, draggableId } = result;
      if (!destination) return;
      if (destination.droppableId === dragSource.droppableId) return;

      const targetStatus = COLUMN_DROP_STATUS[destination.droppableId];
      if (!targetStatus) {
        // Can't manually drag into processing/review/serializing/completed
        return;
      }

      onStatusChange?.(draggableId, targetStatus);
    },
    [onStatusChange]
  );

  return (
    <DragDropContext onDragEnd={handleDragEnd}>
      <div className="flex gap-3 px-4 py-3 overflow-x-auto min-h-0 flex-1">
        {COLUMNS.map((col) => {
          const items = columnData[col.id] || [];
          return (
            <Droppable key={col.id} droppableId={col.id}>
              {(provided, snapshot) => (
                <div
                  ref={provided.innerRef}
                  {...provided.droppableProps}
                  className={`
                    flex flex-col w-[280px] min-w-[280px] rounded-xl
                    bg-gb-bg0 border border-gb-bg2
                    ${snapshot.isDraggingOver ? "border-gb-bright-blue bg-gb-bg0-s" : ""}
                  `}
                >
                  {/* Column header */}
                  <div className="flex items-center justify-between px-3 py-2.5 border-b border-gb-bg2">
                    <div className="flex items-center gap-2">
                      <div className={`w-2 h-2 rounded-full ${col.accentClass}`} />
                      <h2 className="text-[13px] font-semibold text-gb-fg1">
                        {col.label}
                      </h2>
                    </div>
                    <span className="font-data text-[11px] font-medium text-gb-fg4 bg-gb-bg1 px-1.5 py-0.5 rounded">
                      {items.length}
                    </span>
                  </div>

                  {/* Card list */}
                  <div className="flex flex-col gap-2 p-2 overflow-y-auto flex-1 min-h-[120px]">
                    {items.map((src, idx) => (
                      <SourceCard
                        key={src.id}
                        source={src}
                        index={idx}
                        onGateReview={onGateReview}
                        onStartPipeline={onStartPipeline}
                        onDelete={onDelete}
                      />
                    ))}
                    {provided.placeholder}
                    {items.length === 0 && (
                      <p className="text-[11px] text-gb-gray text-center py-6">
                        No sources
                      </p>
                    )}
                  </div>
                </div>
              )}
            </Droppable>
          );
        })}
      </div>
    </DragDropContext>
  );
}
