/**
 * Consistency tests for the bundle-graph visual constants.
 *
 * These are four parallel lookup tables — type config, edge colours, filter
 * layers, layer colours — that have to agree with each other. Nothing
 * enforced that. A STIX type present in KNOWN_TYPES but absent from every
 * FILTER_LAYER renders as a node no filter can show or hide, which looks
 * like a rendering bug and is really a missing table entry.
 *
 * The same class as the backend contract tests: not "is this value right?"
 * but "do these tables still describe the same world?".
 */

import { describe, expect, it } from "vitest";

import {
  EDGE_COLORS,
  FILTER_LAYERS,
  FILTER_LAYER_NAMES,
  KNOWN_TYPES,
  LAYER_COLORS,
  SKIP_TYPES,
  edgeColor,
  getTypeConfig,
  layerForType,
  collectMetaIds,
} from "../bundleGraphConstants.js";

// Snapshot the BUILT-IN types at import time, before any test runs.
// getTypeConfig deliberately memoises unknown types INTO KNOWN_TYPES so a
// colour stays stable across views in a session — which means the table is
// not constant, and asserting over it after other tests have called
// getTypeConfig would be asserting over their leftovers.
const BUILTIN_TYPES = Object.keys(KNOWN_TYPES);

describe("getTypeConfig", () => {
  it("returns the configured entry for a known type", () => {
    expect(getTypeConfig("x-procedure")).toBe(KNOWN_TYPES["x-procedure"]);
  });

  it("falls back rather than returning undefined for an unknown type", () => {
    // A new STIX type must not crash the renderer.
    const cfg = getTypeConfig("brand-new-type");
    expect(cfg).toBeTruthy();
    expect(cfg.color).toBeTruthy();
  });

  it.each([["null", null], ["undefined", undefined], ["empty", ""]])(
    "falls back for %s", (_l, input) => {
      expect(getTypeConfig(input)).toBeTruthy();
    },
  );
});

describe("edgeColor", () => {
  it("returns the configured colour for a known relationship type", () => {
    const [known] = Object.keys(EDGE_COLORS);
    expect(edgeColor(known)).toBe(EDGE_COLORS[known]);
  });

  it("falls back for an unknown relationship type", () => {
    expect(edgeColor("invented-relationship")).toBeTruthy();
  });

  it("falls back for null", () => {
    expect(edgeColor(null)).toBeTruthy();
  });
});

describe("layerForType", () => {
  it("puts x-procedure in a real layer", () => {
    expect(FILTER_LAYER_NAMES).toContain(layerForType("x-procedure"));
  });

  it("returns a usable layer for an unknown type", () => {
    // Otherwise the node exists but no filter toggle governs it.
    const layer = layerForType("brand-new-type");
    expect(layer === null || FILTER_LAYER_NAMES.includes(layer)).toBe(true);
  });
});

describe("the tables agree with each other", () => {
  it("scans something (anti-vacuity)", () => {
    expect(BUILTIN_TYPES.length).toBeGreaterThan(10);
    expect(FILTER_LAYER_NAMES.length).toBeGreaterThan(1);
  });

  it("every known type is reachable through some filter layer", () => {
    const orphans = BUILTIN_TYPES.filter((t) => {
      if (SKIP_TYPES.has(t)) return false;
      const layer = layerForType(t);
      return !layer || !FILTER_LAYER_NAMES.includes(layer);
    });
    expect(orphans, `types with no filter layer: ${orphans.join(", ")}`).toEqual([]);
  });

  it("every filter layer has a colour", () => {
    const missing = FILTER_LAYER_NAMES.filter((name) => !LAYER_COLORS[name]);
    expect(missing, `layers with no colour: ${missing.join(", ")}`).toEqual([]);
  });

  it("every type named by a filter layer is a known type", () => {
    const unknown = [];
    for (const [layer, types] of Object.entries(FILTER_LAYERS)) {
      for (const t of types) {
        if (!BUILTIN_TYPES.includes(t)) unknown.push(`${layer}:${t}`);
      }
    }
    expect(unknown, `filter layers naming unknown types: ${unknown.join(", ")}`).toEqual([]);
  });

  it("no type is claimed by two filter layers", () => {
    // Two layers owning one type makes the toggles fight: hiding one layer
    // leaves the node visible via the other.
    const seen = new Map();
    const dupes = [];
    for (const [layer, types] of Object.entries(FILTER_LAYERS)) {
      for (const t of types) {
        if (seen.has(t)) dupes.push(`${t} in ${seen.get(t)} and ${layer}`);
        else seen.set(t, layer);
      }
    }
    expect(dupes, dupes.join("; ")).toEqual([]);
  });
});

describe("getTypeConfig memoisation", () => {
  it("registers an unknown type so its colour is stable across views", () => {
    const a = getTypeConfig("stable-type-check");
    const b = getTypeConfig("stable-type-check");
    expect(b).toBe(a);
    expect(KNOWN_TYPES["stable-type-check"]).toBe(a);
  });

  it("does NOT register falsy input", () => {
    // An object with no `type` would otherwise add literal "null" /
    // "undefined" / "" keys to the shared table, which surface as phantom
    // entries anywhere the table is iterated. Found while writing these
    // tests: the earlier consistency check failed because previous tests
    // had polluted KNOWN_TYPES this way.
    getTypeConfig(null);
    getTypeConfig(undefined);
    getTypeConfig("");
    for (const junk of ["null", "undefined", ""]) {
      expect(Object.keys(KNOWN_TYPES)).not.toContain(junk);
    }
  });
});

describe("collectMetaIds", () => {
  it("returns every extension-definition id and the identity it names as author", () => {
    const objects = [
      { type: "extension-definition", id: "extension-definition--a", created_by_ref: "identity--author" },
      { type: "identity", id: "identity--author" },
      { type: "identity", id: "identity--source" },
      { type: "x-procedure", id: "x-procedure--p", created_by_ref: "identity--source" },
    ];
    expect([...collectMetaIds(objects)].sort()).toEqual([
      "extension-definition--a",
      "identity--author",
    ]);
  });

  it("leaves the source identity alone and tolerates empty input", () => {
    expect(collectMetaIds([{ type: "identity", id: "identity--source" }]).size).toBe(0);
    expect(collectMetaIds(undefined).size).toBe(0);
  });
});
