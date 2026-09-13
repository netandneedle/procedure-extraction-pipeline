/**
 * CorrectionsModal — displays bundle_corrections from the validate_bundle node.
 *
 * Each correction is a structured record:
 *   {rule, severity, message, ref_id?, ref_field?, holder_id?,
 *    holder_type?, before?, after?, recovered_from?}
 *
 * severity precedence (worst first): hard_fail > repaired > warn > auto_fix
 *
 *   - hard_fail (red)    — bundle didn't ship; analyst needs to act
 *   - repaired  (orange) — validator covered for an upstream bug; investigate
 *   - warn      (gray)   — informational
 *   - auto_fix  (yellow) — routine housekeeping
 */
import { useMemo } from "react";

const SEVERITY_ORDER = ["hard_fail", "repaired", "warn", "auto_fix"];

const SEVERITY_STYLES = {
  hard_fail: {
    label: "Hard fail",
    badge: "bg-[rgba(251,73,52,0.12)] text-gb-bright-red border-gb-red",
    headerText: "text-gb-bright-red",
    icon: "✕",
  },
  repaired: {
    label: "Repaired",
    badge: "bg-gb-tag-gate-bg text-gb-bright-orange border-gb-orange",
    headerText: "text-gb-bright-orange",
    icon: "↻",
  },
  warn: {
    label: "Warning",
    badge: "bg-gb-bg1 text-gb-fg4 border-gb-bg2",
    headerText: "text-gb-fg4",
    icon: "⚠",
  },
  auto_fix: {
    label: "Auto-fix",
    badge: "bg-gb-tag-gate-bg text-gb-bright-yellow border-gb-yellow",
    headerText: "text-gb-bright-yellow",
    icon: "✓",
  },
};

function CorrectionEntry({ entry }) {
  const refLabel = entry.holder_id || entry.ref_id;
  return (
    <div className="border border-gb-bg2 rounded-md px-3 py-2 mb-2 bg-gb-bg0-s">
      <div className="flex items-baseline gap-2 mb-1">
        <span className="font-data text-[11px] font-semibold text-gb-fg1">
          {entry.rule}
        </span>
        {entry.ref_field && (
          <span className="font-data text-[10px] text-gb-fg4">
            .{entry.ref_field}
          </span>
        )}
      </div>
      {entry.message && (
        <p className="text-[12px] text-gb-fg2 mb-1.5 leading-relaxed">
          {entry.message}
        </p>
      )}
      {refLabel && (
        <p className="font-data text-[10px] text-gb-fg4 mb-1 truncate">
          {entry.holder_type ? `${entry.holder_type} ` : ""}
          {refLabel}
        </p>
      )}
      {entry.recovered_from && (
        <p className="font-data text-[10px] text-gb-bright-aqua mb-1">
          recovered from: {entry.recovered_from}
        </p>
      )}
      {(entry.before !== undefined || entry.after !== undefined) && (
        <div className="font-data text-[10px] mt-1.5 grid grid-cols-[60px_1fr] gap-x-2 gap-y-0.5">
          {entry.before !== undefined && (
            <>
              <span className="text-gb-fg4">before:</span>
              <span className="text-gb-bright-red break-all">
                {JSON.stringify(entry.before)}
              </span>
            </>
          )}
          {entry.after !== undefined && (
            <>
              <span className="text-gb-fg4">after:</span>
              <span className="text-gb-bright-green break-all">
                {JSON.stringify(entry.after)}
              </span>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default function CorrectionsModal({ isOpen, corrections = [], title, onClose }) {
  // Group + count by severity. Memoized so the buckets don't re-shuffle
  // on each render when the parent re-renders for unrelated reasons.
  const groups = useMemo(() => {
    const buckets = { hard_fail: [], repaired: [], warn: [], auto_fix: [] };
    for (const entry of corrections) {
      const sev = entry?.severity || "auto_fix";
      if (buckets[sev]) {
        buckets[sev].push(entry);
      }
    }
    return buckets;
  }, [corrections]);

  if (!isOpen) return null;

  const total = corrections.length;
  const heading = title || "Bundle validator corrections";

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-16 bg-gb-bg0-h/75 overflow-y-auto"
      onClick={onClose}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="bg-gb-bg0-s border border-gb-bg2 rounded-xl p-6 w-[680px] max-w-[90vw] mb-16"
      >
        <div className="flex items-baseline justify-between mb-4">
          <h2 className="text-base font-semibold text-gb-fg0">
            {heading}
          </h2>
          <span className="font-data text-[11px] text-gb-fg4">
            {total} {total === 1 ? "entry" : "entries"}
          </span>
        </div>

        {total === 0 ? (
          <p className="text-[13px] text-gb-fg4 leading-relaxed">
            No corrections — bundle validated cleanly.
          </p>
        ) : (
          <div className="space-y-5">
            {SEVERITY_ORDER.map((sev) => {
              const entries = groups[sev];
              if (!entries.length) return null;
              const style = SEVERITY_STYLES[sev];
              return (
                <div key={sev}>
                  <div className="flex items-baseline gap-2 mb-2">
                    <span
                      className={`font-data text-[11px] font-medium px-2 py-0.5 rounded border ${style.badge}`}
                    >
                      {style.icon} {style.label}
                    </span>
                    <span className={`font-data text-[11px] ${style.headerText}`}>
                      {entries.length}
                    </span>
                  </div>
                  {entries.map((entry, i) => (
                    <CorrectionEntry key={`${sev}-${i}`} entry={entry} />
                  ))}
                </div>
              );
            })}
          </div>
        )}

        <div className="flex justify-end mt-5 pt-4 border-t border-gb-bg1">
          <button
            type="button"
            onClick={onClose}
            className="px-3.5 py-1.5 rounded-md text-[13px] border border-gb-bg2 text-gb-fg4 hover:bg-gb-bg1 hover:text-gb-fg1 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

/**
 * Helper: pick the highest severity bucket present in a corrections array.
 * Returns null when the array is empty/missing. Useful for one-glance
 * styling on a chip or row badge.
 */
export function highestSeverity(corrections) {
  if (!Array.isArray(corrections) || corrections.length === 0) return null;
  const present = new Set(
    corrections.map((c) => c?.severity).filter(Boolean)
  );
  return SEVERITY_ORDER.find((s) => present.has(s)) || null;
}

export const CORRECTIONS_SEVERITY_STYLES = SEVERITY_STYLES;
