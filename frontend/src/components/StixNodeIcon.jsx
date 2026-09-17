import React from "react";
import { getTypeConfig } from "../lib/bundleGraphConstants";

/**
 * Rounded-square badge with a colored fill and a white MDI glyph, a
 * shape STIX analysts already read from other tooling. Used by the
 * React Flow viewers (BundleReviewCanvas, BundleFlowView) and by node
 * labels in the side panels. Canvas-based BundleGraph rasterizes the
 * same MDI paths directly via Path2D instead of mounting this
 * component.
 *
 * Props:
 *   type   STIX type string (e.g. "x-procedure", "attack-pattern").
 *   size   Outer badge size in CSS pixels (default 40). The glyph
 *          renders at ~60% of size, centered.
 *   className  Extra class for the wrapper (e.g. "shadow-md").
 *   title  Optional <title> for accessibility / hover hint.
 */
export default function StixNodeIcon({ type, size = 40, className = "", title }) {
  const cfg = getTypeConfig(type);
  const radius = Math.max(2, Math.round(size * 0.18));
  const inner = Math.round(size * 0.6);
  const inset = Math.round((size - inner) / 2);

  return (
    <svg
      width={size}
      height={size}
      viewBox={`0 0 ${size} ${size}`}
      className={className}
      role="img"
      aria-label={title || type}
    >
      {title ? <title>{title}</title> : null}
      <rect
        x={0.5}
        y={0.5}
        width={size - 1}
        height={size - 1}
        rx={radius}
        ry={radius}
        fill={cfg.color}
        stroke="rgba(0,0,0,0.25)"
        strokeWidth={1}
      />
      <g transform={`translate(${inset}, ${inset}) scale(${inner / 24})`}>
        <path d={cfg.icon} fill="#fff" />
      </g>
    </svg>
  );
}
