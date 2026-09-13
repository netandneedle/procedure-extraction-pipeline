import StixNodeIcon from "./StixNodeIcon";

/**
 * Tooltip shown when the cursor hovers a node in any of the three
 * viewers. Floats at (x + 12, y + 12) relative to its positioned
 * parent, so the caller is responsible for rendering this inside a
 * `position: relative` container (the viewport wrapper).
 *
 * Props:
 *   node    { type, name, id, mitreId? } — null/undefined hides the tooltip.
 *   x, y    Cursor coordinates relative to the positioned parent.
 *
 * Why custom (vs native `title=…`):
 *   - Native title has a ~500 ms delay and the OS controls styling.
 *   - We want instant + theme-consistent + a colored icon badge.
 *   - We want type + name + id together (native title only shows one
 *     string).
 *
 * The tooltip is render-only — it never receives clicks, never blocks
 * the cursor-down event path (pointerEvents: none).
 */
export default function NodeHoverTooltip({ node, x, y }) {
  if (!node) return null;
  // Cap to a sane width — long procedure names get an ellipsis below.
  return (
    <div
      className="pointer-events-none absolute z-40 max-w-[320px] rounded border border-gb-bg2 bg-gb-bg0/95 px-2 py-1.5 shadow-lg"
      style={{ left: x + 12, top: y + 12 }}
    >
      <div className="flex items-start gap-2">
        <StixNodeIcon type={node.type} size={18} className="shrink-0 mt-0.5" />
        <div className="min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="text-[9px] font-data text-gb-fg4 uppercase tracking-wide">
              {node.type}
            </span>
            {node.mitreId && (
              <span className="text-[9px] font-data text-gb-bright-blue">
                {node.mitreId}
              </span>
            )}
          </div>
          <div
            className="text-[11px] font-data text-gb-fg1 leading-tight break-words"
            title={node.name}
          >
            {node.name}
          </div>
          {node.id && (
            <div className="text-[9px] font-data text-gb-fg4 truncate mt-0.5">
              {node.id}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
