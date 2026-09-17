import { describe, expect, it } from "vitest";

import {
  applyPositions,
  layoutPositions,
  layoutSignature,
} from "../graphLayout";

const nodes = [
  { id: "a", width: 200, height: 80 },
  { id: "b", width: 200, height: 80 },
  { id: "c", width: 90, height: 40 },
];
const edges = [
  { source: "a", target: "b" },
  { source: "b", target: "c" },
];

describe("layoutSignature", () => {
  it("ignores decoration and depends only on ids, sizes, edges and options", () => {
    const decorated = nodes.map((n) => ({ ...n, data: { isSelected: n.id === "b" } }));
    expect(layoutSignature(decorated, edges)).toBe(layoutSignature(nodes, edges));
  });

  it("changes when a node, an edge, or the rank direction changes", () => {
    const base = layoutSignature(nodes, edges);
    expect(layoutSignature(nodes.slice(0, 2), edges)).not.toBe(base);
    expect(layoutSignature(nodes, edges.slice(0, 1))).not.toBe(base);
    expect(layoutSignature(nodes, edges, { rankdir: "LR" })).not.toBe(base);
  });
});

describe("layoutPositions", () => {
  it("positions every node and honors per-node width when centering", () => {
    const pos = layoutPositions(nodes, edges, { rankdir: "TB" });
    expect([...pos.keys()].sort()).toEqual(["a", "b", "c"]);
    // A linear chain stacks top-to-bottom: each rank strictly below the last.
    expect(pos.get("b").y).toBeGreaterThan(pos.get("a").y);
    expect(pos.get("c").y).toBeGreaterThan(pos.get("b").y);
    // The narrow node is centered on the same axis as the wide ones, so its
    // top-left x sits (200 - 90) / 2 to the right of theirs.
    expect(pos.get("c").x - pos.get("a").x).toBeCloseTo(55, 5);
  });
});

describe("applyPositions", () => {
  it("overlays positions and falls back to the origin for an unknown id", () => {
    const pos = new Map([["a", { x: 1, y: 2 }]]);
    const out = applyPositions([{ id: "a" }, { id: "zz" }], pos);
    expect(out[0].position).toEqual({ x: 1, y: 2 });
    expect(out[1].position).toEqual({ x: 0, y: 0 });
  });
});
