/**
 * Tests for the bundle-viewer resolution layer.
 *
 * This is what turns raw STIX ids into something an analyst can read, and
 * what derives the Flow tab's graph. Its defensive behaviour matters as much
 * as its happy path: bundles legitimately contain refs to objects that are
 * NOT embedded (ATT&CK attack-patterns are referenced, never copied in), so
 * "unresolvable" is normal and must degrade to a readable label rather than
 * a blank slot or a crash.
 */

import { describe, expect, it } from "vitest";

import {
  buildResolutionIndex,
  extractFlowGraph,
  resolveRef,
  resolveRefs,
} from "../bundleResolution.js";

const bundle = (...objects) => ({ type: "bundle", objects });

const PROC = "x-procedure--11111111-1111-4111-8111-111111111111";
const PROC2 = "x-procedure--22222222-2222-4222-8222-222222222222";

const procedure = (id, name) => ({ type: "x-procedure", id, name });
const precedes = (src, tgt) => ({
  type: "relationship", relationship_type: "precedes",
  id: `relationship--${src.slice(-4)}${tgt.slice(-4)}`,
  source_ref: src, target_ref: tgt,
});

describe("buildResolutionIndex", () => {
  it("indexes objects by id", () => {
    const index = buildResolutionIndex(bundle(procedure(PROC, "Delete Shadow Copies")));
    expect(index.get(PROC).displayName).toBe("Delete Shadow Copies");
    expect(index.get(PROC).type).toBe("x-procedure");
  });

  it("still indexes an unknown object type", () => {
    // Better a typed row than a missing one — a new STIX type should not
    // make the detail panel go blank.
    const index = buildResolutionIndex(bundle({ type: "brand-new-type", id: "brand-new-type--x", name: "N" }));
    expect(index.get("brand-new-type--x").type).toBe("brand-new-type");
  });

  it("skips objects without an id", () => {
    const index = buildResolutionIndex(bundle({ type: "x-procedure", name: "no id" }));
    expect(index.size).toBe(0);
  });

  it.each([
    ["null", null],
    ["undefined", undefined],
    ["no objects array", { type: "bundle" }],
    ["objects not an array", { type: "bundle", objects: "nope" }],
  ])("returns an empty index for %s", (_label, input) => {
    expect(buildResolutionIndex(input).size).toBe(0);
  });
});

describe("resolveRef", () => {
  const index = buildResolutionIndex(bundle(procedure(PROC, "Named Procedure")));

  it("resolves a ref that is in the bundle", () => {
    expect(resolveRef(index, PROC).displayName).toBe("Named Procedure");
  });

  it("returns null for a falsy ref", () => {
    expect(resolveRef(index, null)).toBeNull();
    expect(resolveRef(index, "")).toBeNull();
  });

  it("degrades to a readable label for an unembedded ref", () => {
    // ATT&CK attack-patterns are referenced but never copied into the
    // bundle, so this is the normal case, not an error.
    const out = resolveRef(index, "attack-pattern--99999999-9999-4999-8999-999999999999");
    expect(out).toBeTruthy();
    expect(out.displayName).toBeTruthy();
    expect(out.rawId).toBe("attack-pattern--99999999-9999-4999-8999-999999999999");
  });
});

describe("resolveRefs", () => {
  const index = buildResolutionIndex(bundle(procedure(PROC, "A"), procedure(PROC2, "B")));

  it("resolves a list in order", () => {
    expect(resolveRefs(index, [PROC, PROC2]).map((r) => r.displayName)).toEqual(["A", "B"]);
  });

  it.each([
    ["null", null],
    ["undefined", undefined],
    ["a non-array", "not-a-list"],
  ])("returns an empty list for %s", (_label, input) => {
    expect(resolveRefs(index, input)).toEqual([]);
  });
});

describe("extractFlowGraph", () => {
  it("builds nodes and precedes edges", () => {
    const g = extractFlowGraph(bundle(
      procedure(PROC, "first"), procedure(PROC2, "second"), precedes(PROC, PROC2),
    ));
    expect(g.nodes.map((n) => n.name)).toEqual(["first", "second"]);
    expect(g.edges).toEqual([{ source: PROC, target: PROC2 }]);
  });

  it("ignores non-precedes relationships", () => {
    // The Flow tab shows sequencing only; a `uses` edge belongs in the
    // Bundle tab, not the kill-chain view.
    const uses = { ...precedes(PROC, PROC2), relationship_type: "uses" };
    const g = extractFlowGraph(bundle(procedure(PROC, "a"), procedure(PROC2, "b"), uses));
    expect(g.edges).toEqual([]);
  });

  it("drops a precedes edge whose endpoint is not a flow node", () => {
    // A dangling edge would render as a line to nowhere.
    const g = extractFlowGraph(bundle(
      procedure(PROC, "a"),
      precedes(PROC, "x-procedure--33333333-3333-4333-8333-333333333333"),
    ));
    expect(g.edges).toEqual([]);
  });

  it("dedupes duplicate precedes edges", () => {
    const g = extractFlowGraph(bundle(
      procedure(PROC, "a"), procedure(PROC2, "b"),
      precedes(PROC, PROC2), { ...precedes(PROC, PROC2), id: "relationship--dup" },
    ));
    expect(g.edges).toHaveLength(1);
  });

  it.each([
    ["null", null],
    ["no objects", { type: "bundle" }],
  ])("returns an empty graph for %s", (_label, input) => {
    const g = extractFlowGraph(input);
    expect(g.nodes).toEqual([]);
    expect(g.edges).toEqual([]);
  });
});
