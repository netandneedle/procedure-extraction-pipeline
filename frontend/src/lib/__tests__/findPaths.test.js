/**
 * Tests for the Explorer's path-finding.
 *
 * Graph algorithms are easy to get subtly wrong in ways a build can't see
 * and a screenshot won't reveal: an off-by-one in the BFS layering silently
 * returns paths that are not shortest, and a missing visited-check turns a
 * cyclic bundle into a hang. Both are plausible here — STIX bundles are
 * cyclic by nature (a procedure and its observables reference each other).
 */

import { describe, expect, it } from "vitest";

import {
  buildAdjacency,
  collectPathElements,
  findAllShortestPaths,
} from "../findPaths.js";

const edges = (...pairs) => pairs.map(([source, target]) => ({ source, target }));

describe("buildAdjacency", () => {
  it("is undirected — an edge is walkable both ways", () => {
    const adj = buildAdjacency(edges(["a", "b"]));
    expect([...adj.get("a")]).toEqual(["b"]);
    expect([...adj.get("b")]).toEqual(["a"]);
  });

  it("accepts node objects as well as id strings", () => {
    const adj = buildAdjacency([{ source: { id: "a" }, target: { id: "b" } }]);
    expect(adj.get("a").has("b")).toBe(true);
  });

  it("drops self-loops", () => {
    // A STIX object referencing itself would otherwise make a node its own
    // neighbor and pad every path through it.
    const adj = buildAdjacency(edges(["a", "a"]));
    expect(adj.has("a")).toBe(false);
  });

  it("ignores edges with a missing endpoint", () => {
    const adj = buildAdjacency([{ source: "a" }, { target: "b" }, {}]);
    expect(adj.size).toBe(0);
  });

  it("dedupes parallel edges", () => {
    const adj = buildAdjacency(edges(["a", "b"], ["a", "b"]));
    expect(adj.get("a").size).toBe(1);
  });

  it("tolerates null/undefined input", () => {
    expect(buildAdjacency(null).size).toBe(0);
    expect(buildAdjacency(undefined).size).toBe(0);
  });
});

describe("findAllShortestPaths", () => {
  it("finds a simple chain", () => {
    const adj = buildAdjacency(edges(["a", "b"], ["b", "c"]));
    expect(findAllShortestPaths(adj, "a", "c")).toEqual([["a", "b", "c"]]);
  });

  it("returns the node itself when start === end", () => {
    const adj = buildAdjacency(edges(["a", "b"]));
    expect(findAllShortestPaths(adj, "a", "a")).toEqual([["a"]]);
  });

  it("returns nothing when the nodes are in disconnected components", () => {
    const adj = buildAdjacency(edges(["a", "b"], ["c", "d"]));
    expect(findAllShortestPaths(adj, "a", "d")).toEqual([]);
  });

  it("returns nothing for unknown nodes", () => {
    const adj = buildAdjacency(edges(["a", "b"]));
    expect(findAllShortestPaths(adj, "a", "ghost")).toEqual([]);
    expect(findAllShortestPaths(adj, "ghost", "b")).toEqual([]);
  });

  it("returns EVERY shortest path when routes tie", () => {
    // a-b-d and a-c-d are both length 3; surfacing only one would hide a
    // real relationship from the analyst.
    const adj = buildAdjacency(edges(["a", "b"], ["a", "c"], ["b", "d"], ["c", "d"]));
    const paths = findAllShortestPaths(adj, "a", "d");
    expect(paths).toHaveLength(2);
    expect(paths.map((p) => p.join(">")).sort()).toEqual(["a>b>d", "a>c>d"]);
  });

  it("excludes longer routes when a shorter one exists", () => {
    const adj = buildAdjacency(
      edges(["a", "b"], ["b", "z"], ["a", "c"], ["c", "d"], ["d", "z"]),
    );
    const paths = findAllShortestPaths(adj, "a", "z");
    expect(paths).toEqual([["a", "b", "z"]]);
  });

  it("terminates on a cyclic graph", () => {
    // STIX bundles are cyclic by construction. A missing visited-check
    // would hang the renderer here rather than fail visibly.
    const adj = buildAdjacency(edges(["a", "b"], ["b", "c"], ["c", "a"]));
    expect(findAllShortestPaths(adj, "a", "c")).toEqual([["a", "c"]]);
  });

  it("honors maxDepth by refusing paths that are too deep", () => {
    const adj = buildAdjacency(edges(["a", "b"], ["b", "c"], ["c", "d"]));
    expect(findAllShortestPaths(adj, "a", "d", { maxDepth: 1 })).toEqual([]);
    expect(findAllShortestPaths(adj, "a", "d", { maxDepth: 10 })).toHaveLength(1);
  });

  it("caps the number of returned paths", () => {
    // Many parallel routes must not freeze the renderer.
    const pairs = [];
    for (let i = 0; i < 30; i += 1) {
      pairs.push(["start", `mid${i}`], [`mid${i}`, "end"]);
    }
    const adj = buildAdjacency(edges(...pairs));
    const paths = findAllShortestPaths(adj, "start", "end", { maxPaths: 5 });
    expect(paths.length).toBeLessThanOrEqual(5);
  });

  it("every returned path is a real walk of the graph", () => {
    // Guards the reconstruction: a parent-map bug can emit a sequence whose
    // consecutive nodes are not actually adjacent.
    const adj = buildAdjacency(
      edges(["a", "b"], ["a", "c"], ["b", "d"], ["c", "d"], ["d", "e"]),
    );
    for (const path of findAllShortestPaths(adj, "a", "e")) {
      for (let i = 0; i < path.length - 1; i += 1) {
        expect(adj.get(path[i]).has(path[i + 1])).toBe(true);
      }
      expect(path[0]).toBe("a");
      expect(path[path.length - 1]).toBe("e");
    }
  });
});

describe("collectPathElements", () => {
  it("collects the nodes and edge keys along the paths", () => {
    const out = collectPathElements([["a", "b", "c"]]);
    expect([...out.nodeSet].sort()).toEqual(["a", "b", "c"]);
    // Edge keys are order-normalized (a|b, not b|a) so an undirected
    // edge has one identity regardless of traversal direction.
    expect([...out.edgeSet].sort()).toEqual(["a|b", "b|c"]);
  });

  it("handles an empty path list", () => {
    const out = collectPathElements([]);
    expect(out.nodeSet.size).toBe(0);
    expect(out.edgeSet.size).toBe(0);
  });

  it("dedupes nodes shared by several paths", () => {
    const out = collectPathElements([["a", "b", "d"], ["a", "c", "d"]]);
    expect([...out.nodeSet].sort()).toEqual(["a", "b", "c", "d"]);
  });
});
