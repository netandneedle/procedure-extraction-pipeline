/**
 * StatsBar — summary stats displayed below the top nav.
 * Total sources, per-gate review counts, processing, completed today.
 */
import { REVIEW_GATES } from "../lib/gates";
import { PROCESSING_STATUSES } from "../lib/pipelineStatus";

function isToday(dateStr) {
  if (!dateStr) return false;
  const d = new Date(dateStr);
  const now = new Date();
  return d.toDateString() === now.toDateString();
}

export default function StatsBar({ sources }) {
  const total = sources.length;
  const counts = REVIEW_GATES.map((g) => ({
    label: g.label,
    title: g.title,
    count: sources.filter((s) => s.status === g.status).length,
  }));
  const reviewTotal = counts.reduce((acc, c) => acc + c.count, 0);
  const processing = sources.filter((s) => PROCESSING_STATUSES.has(s.status)).length;
  const completedToday = sources.filter(
    (s) => s.status === "completed" && isToday(s.updated_at)
  ).length;
  const chips = counts.filter((c) => c.count > 0);

  return (
    <div className="flex gap-5 px-6 py-2 bg-gb-bg0-h border-b border-gb-bg1 text-xs text-gb-gray">
      <div className="flex items-center gap-1.5">
        Total: <span className="font-data font-semibold text-gb-fg1">{total}</span>
      </div>
      <div className="flex items-center gap-1.5">
        Review: <span className="font-data font-semibold text-gb-bright-orange">{reviewTotal}</span>
        {/* One span per chip rather than a single joined string, so each
            G0/G1/G2/G3 can carry the gate name it stands for. */}
        {chips.length > 0 && (
          <span className="font-data text-gb-fg4">
            (
            {chips.map((c, i) => (
              <span key={c.label} title={c.title} className="cursor-help">
                {i > 0 ? " " : ""}{c.label}:{c.count}
              </span>
            ))}
            )
          </span>
        )}
      </div>
      <div className="flex items-center gap-1.5">
        Processing: <span className="font-data font-semibold text-gb-bright-blue">{processing}</span>
      </div>
      <div className="flex items-center gap-1.5">
        Completed today: <span className="font-data font-semibold text-gb-bright-green">{completedToday}</span>
      </div>
    </div>
  );
}
