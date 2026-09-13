/**
 * lib/graphLayout.js — dagre layout for the three React Flow canvases.
 *
 * Positions depend on the node set (ids and sizes) and the edge set, and on
 * nothing else. Selection, hover, edited/dropped flags, focus dimming and
 * edge labels change how a node LOOKS, not where it sits — yet every canvas
 * used to re-run dagre over the whole graph on each of those, so a keystroke
 * in a side-panel textarea re-laid-out a 33-chunk DAG and re-rendered every
 * node and the MiniMap.
 *
 * `useLayoutPositions` keys the layout on a signature of exactly those
 * inputs and hands back the same Map until one of them changes. Callers
 * build their (cheap) decorated nodes every render and overlay positions
 * with `applyPositions`.
 *
 * This replaced three diverged copies of the same function (chunk canvas,
 * bundle canvas, flow view); only one honoured per-node width/height, so
 * the same attack-operator was a narrow chip in one view and a full-width
 * card in another.
 */
import dagre from "dagre";
import { useRef } from "react";

const DEFAULTS = {
  rankdir: "TB",
  nodesep: 60,
  ranksep: 110,
  marginx: 20,
  marginy: 20,
  width: 220,
  height: 70,
};

function sizeOf(node, opts) {
  return {
    w: node.width ?? opts.width,
    h: node.height ?? opts.height,
  };
}

/** Top-left position per node id, from a dagre run over these nodes and edges. */
export function layoutPositions(nodes, edges, options = {}) {
  const opts = { ...DEFAULTS, ...options };
  const g = new dagre.graphlib.Graph();
  g.setGraph({
    rankdir: opts.rankdir,
    nodesep: opts.nodesep,
    ranksep: opts.ranksep,
    marginx: opts.marginx,
    marginy: opts.marginy,
  });
  g.setDefaultEdgeLabel(() => ({}));
  for (const n of nodes) {
    const { w, h } = sizeOf(n, opts);
    g.setNode(n.id, { width: w, height: h });
  }
  for (const e of edges) g.setEdge(e.source, e.target);
  dagre.layout(g);
  const positions = new Map();
  for (const n of nodes) {
    const pos = g.node(n.id);
    const { w, h } = sizeOf(n, opts);
    positions.set(n.id, { x: pos.x - w / 2, y: pos.y - h / 2 });
  }
  return positions;
}

/** Everything the layout depends on, as one string. Order is kept: dagre
 *  is order-sensitive, so a reordered node list is a different layout. */
export function layoutSignature(nodes, edges, options = {}) {
  const opts = { ...DEFAULTS, ...options };
  const head = `${opts.rankdir}/${opts.nodesep}/${opts.ranksep}/${opts.marginx}/${opts.marginy}`;
  const ns = nodes.map((n) => { const { w, h } = sizeOf(n, opts); return `${n.id}:${w}x${h}`; }).join("|");
  const es = edges.map((e) => `${e.source}>${e.target}`).join("|");
  return `${head}#${ns}#${es}`;
}

/** Cached `layoutPositions`: the same Map comes back until the signature
 *  changes, so decorating nodes never pays for a layout. */
export function useLayoutPositions(nodes, edges, options = {}) {
  const sig = layoutSignature(nodes, edges, options);
  const cache = useRef({ sig: null, positions: new Map() });
  if (cache.current.sig !== sig) {
    cache.current = { sig, positions: layoutPositions(nodes, edges, options) };
  }
  return cache.current.positions;
}

/** Nodes with `position` filled from a positions Map. */
export function applyPositions(nodes, positions) {
  return nodes.map((n) => ({ ...n, position: positions.get(n.id) ?? { x: 0, y: 0 } }));
}
