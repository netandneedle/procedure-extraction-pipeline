/**
 * Shared visual constants for bundle/relationship graph views.
 *
 * Both the read-only Explorer (BundleGraph.jsx, canvas-based) and the
 * editable Gate 2 review (BundleReviewCanvas.jsx, React Flow-based)
 * import from this module so they render consistently — same colors,
 * same icons, same layer groupings — without duplicating the registry.
 *
 * Heavy logic (parseBundle, force-simulation helpers) stays in
 * BundleGraph.jsx; this module is constants + tiny lookup helpers only.
 *
 * Icons: SVG path strings from @mdi/js (Material Design Icons,
 * Apache 2.0). MDI was picked over @mui/icons-material because it's
 * tree-shakeable raw strings — no MUI peer dep, no Tailwind/MUI theme
 * collision. The icon picks largely mirror OpenCTI's ItemIcon
 * conventions (LockPattern for attack-pattern, Biohazard for malware,
 * ChessKnight for campaign, etc.) so analysts familiar with that tool
 * read our viewers without retraining. The one custom pick is
 * `mdiPlaylistPlay` for x-procedure — procedures are ordered,
 * executable steps; the playlist-play glyph reads that intent.
 */
import {
  mdiPlaylistPlay,
  mdiLockPattern,
  mdiChessKnight,
  mdiDiamondStone,
  mdiAccountMultipleOutline,
  mdiBiohazard,
  mdiTools,
  mdiBugOutline,
  mdiRouter,
  mdiFingerprint,
  mdiProgressWrench,
  mdiRadar,
  mdiChartBoxOutline,
  mdiChip,
  mdiDatabase,
  mdiConsoleLine,
  mdiTimelineOutline,
  mdiPlayBoxOutline,
  mdiSitemap,
  mdiHelpCircleOutline,
  mdiCog,
  mdiFileOutline,
  mdiWeb,
  mdiIpNetwork,
  mdiLinkVariant,
  mdiEmailOutline,
  mdiFolderCog,
  mdiSwapHorizontal,
  mdiFolderOutline,
  mdiLockOutline,
  mdiApplicationOutline,
  mdiAccountCircleOutline,
  mdiMapMarker,
  mdiAccountTie,
  mdiFileDocumentOutline,
  mdiShieldOutline,
  mdiHelpCircle,
} from "@mdi/js";

// ── Type registry ────────────────────────────────────────────────────
// One entry per STIX object type the pipeline emits. `r` is the canvas
// render radius (canvas-based BundleGraph reads it; React Flow viewers
// use a fixed badge size). `icon` is an MDI SVG path string. Unknown
// types fall back to a rotating palette + question-mark icon via
// getTypeConfig().
export const KNOWN_TYPES = {
  "x-procedure":                { color: "#d08770", r: 18, icon: mdiPlaylistPlay,             group: "Threat Layer" },
  "attack-pattern":             { color: "#b48ead", r: 12, icon: mdiLockPattern,              group: "Threat Layer" },
  "campaign":                   { color: "#bf616a", r: 14, icon: mdiChessKnight,              group: "Threat Layer" },
  "intrusion-set":              { color: "#bf616a", r: 14, icon: mdiDiamondStone,             group: "Threat Layer" },
  "malware":                    { color: "#d08770", r: 12, icon: mdiBiohazard,                group: "Threat Layer" },
  "tool":                       { color: "#d08770", r: 11, icon: mdiTools,                    group: "Threat Layer" },
  "threat-actor":               { color: "#bf616a", r: 13, icon: mdiAccountMultipleOutline,   group: "Threat Layer" },
  "vulnerability":              { color: "#bf616a", r: 10, icon: mdiBugOutline,               group: "Threat Layer" },
  "infrastructure":             { color: "#81a1c1", r: 10, icon: mdiRouter,                   group: "Threat Layer" },
  "x-mitre-detection-strategy": { color: "#a3be8c", r: 16, icon: mdiRadar,                    group: "Detection Layer" },
  "x-mitre-analytic":           { color: "#a3be8c", r: 14, icon: mdiChartBoxOutline,          group: "Detection Layer" },
  "x-mitre-data-component":     { color: "#8fbcbb", r: 13, icon: mdiChip,                     group: "Detection Layer" },
  "x-mitre-data-source":        { color: "#8fbcbb", r: 11, icon: mdiDatabase,                 group: "Detection Layer" },
  "indicator":                  { color: "#bf616a", r: 15, icon: mdiFingerprint,              group: "Detection Layer" },
  "course-of-action":           { color: "#a3be8c", r: 11, icon: mdiProgressWrench,           group: "Detection Layer" },
  "attack-flow":                { color: "#ebcb8b", r: 14, icon: mdiTimelineOutline,          group: "Attack Flow" },
  "attack-action":              { color: "#ebcb8b", r: 10, icon: mdiPlayBoxOutline,           group: "Attack Flow" },
  "attack-operator":            { color: "#ebcb8b", r: 10, icon: mdiSitemap,                  group: "Attack Flow" },
  "attack-condition":           { color: "#b48ead", r: 10, icon: mdiHelpCircleOutline,        group: "Attack Flow" },
  "process":                    { color: "#5e81ac", r: 7,  icon: mdiCog,                      group: "Observables" },
  "file":                       { color: "#5e81ac", r: 6,  icon: mdiFileOutline,              group: "Observables" },
  "domain-name":                { color: "#81a1c1", r: 7,  icon: mdiWeb,                      group: "Observables" },
  "ipv4-addr":                  { color: "#81a1c1", r: 7,  icon: mdiIpNetwork,                group: "Observables" },
  "ipv6-addr":                  { color: "#81a1c1", r: 7,  icon: mdiIpNetwork,                group: "Observables" },
  "url":                        { color: "#81a1c1", r: 6,  icon: mdiLinkVariant,              group: "Observables" },
  "email-addr":                 { color: "#81a1c1", r: 6,  icon: mdiEmailOutline,             group: "Observables" },
  "windows-registry-key":       { color: "#5e81ac", r: 6,  icon: mdiFolderCog,                group: "Observables" },
  "network-traffic":            { color: "#5e81ac", r: 7,  icon: mdiSwapHorizontal,           group: "Observables" },
  "directory":                  { color: "#5e81ac", r: 6,  icon: mdiFolderOutline,            group: "Observables" },
  "mutex":                      { color: "#5e81ac", r: 5,  icon: mdiLockOutline,              group: "Observables" },
  "software":                   { color: "#5e81ac", r: 7,  icon: mdiApplicationOutline,       group: "Observables" },
  "user-account":               { color: "#5e81ac", r: 6,  icon: mdiAccountCircleOutline,     group: "Observables" },
  "location":                   { color: "#81a1c1", r: 8,  icon: mdiMapMarker,                group: "Observables" },
  "identity":                   { color: "#4c566a", r: 8,  icon: mdiAccountTie,               group: "Infrastructure" },
  "report":                     { color: "#4c566a", r: 10, icon: mdiFileDocumentOutline,      group: "Infrastructure" },
  "marking-definition":         { color: "#4c566a", r: 5,  icon: mdiShieldOutline,            group: "Infrastructure" },
  "x-log-source":               { color: "#8fbcbb", r: 10, icon: mdiConsoleLine,              group: "Detection Layer" },
};

// Per-kind palette for attack-operator nodes. AND = teal/aqua (the
// "all required" sense maps to a cool, deterministic color); OR = amber
// (the type's base color, signaling alternative paths); XOR = red
// (mutually exclusive — easy to spot at a glance).
export const OPERATOR_KIND_COLORS = {
  AND: "#8fbcbb",
  OR:  "#ebcb8b",
  XOR: "#bf616a",
};

/** Resolve operator kind to its visual color, with a safe fallback. */
export function operatorKindColor(kind) {
  return OPERATOR_KIND_COLORS[kind] || OPERATOR_KIND_COLORS.OR;
}

// Fallback icon when getTypeConfig auto-registers an unknown STIX type.
export const FALLBACK_ICON = mdiHelpCircle;

const FALLBACK_COLORS = ["#d08770", "#ebcb8b", "#a3be8c", "#b48ead", "#88c0d0", "#81a1c1", "#bf616a"];
const _BUILTIN_TYPES = new Set(Object.keys(KNOWN_TYPES));

/** Resolve a STIX type to its visual config. Unknown types are auto-
 * registered with a rotating fallback color so successive views stay
 * consistent within a session. */
export function getTypeConfig(type) {
  if (KNOWN_TYPES[type]) return KNOWN_TYPES[type];
  // Do not register falsy input. An object with no `type` would otherwise
  // add literal "null"/"undefined"/"" keys to KNOWN_TYPES, which then show
  // up in anything that iterates the table (a legend, a filter list) as
  // phantom entries. Fall back without memoising.
  if (!type) {
    return { color: FALLBACK_COLORS[0], r: 10, icon: FALLBACK_ICON, group: "Other" };
  }
  const unknownCount = Object.keys(KNOWN_TYPES).filter(
    (k) => !_BUILTIN_TYPES.has(k),
  ).length;
  const config = {
    color: FALLBACK_COLORS[unknownCount % FALLBACK_COLORS.length],
    r: 10, icon: FALLBACK_ICON, group: "Other",
  };
  KNOWN_TYPES[type] = config;
  return config;
}

// ── Edge styling ─────────────────────────────────────────────────────
// Covers explicit SROs (uses, targets, etc.), STIX embedded refs the
// BundleGraph passes shows as dashed (x-components, etc.), and Attack
// Flow specifics (flow-start, flow-effect).
export const EDGE_COLORS = {
  "uses-technique":      "rgba(180,142,173,0.4)",
  "uses":                "rgba(208,135,112,0.4)",
  "has-observable":      "rgba(94,129,172,0.25)",
  "implements-technique":"rgba(180,142,173,0.5)",
  "belongs-to-tactic":   "rgba(180,142,173,0.35)",
  "subtechnique-of":     "rgba(180,142,173,0.3)",
  "mitigates":           "rgba(163,190,140,0.4)",
  "detects":             "rgba(163,190,140,0.4)",
  "indicates":           "rgba(191,97,106,0.35)",
  "uses-data-component": "rgba(143,188,187,0.35)",
  "has-analytic":        "rgba(163,190,140,0.4)",
  "derived-from":        "rgba(143,188,187,0.3)",
  "related-to":          "rgba(76,86,106,0.25)",
  "exploits":            "rgba(208,135,112,0.4)",
  "attributed-to":       "rgba(191,97,106,0.4)",
  "targets":             "rgba(191,97,106,0.4)",
  "marked-by":           "rgba(76,86,106,0.2)",
  "report-references":   "rgba(76,86,106,0.25)",
  "revoked-by":          "rgba(76,86,106,0.2)",
  "from-source":         "rgba(76,86,106,0.2)",
  "sighting-of":         "rgba(208,135,112,0.35)",
  "precedes":            "rgba(235,203,139,0.5)",
  "describes":           "rgba(76,86,106,0.3)",
  "component-of":        "rgba(143,188,187,0.4)",
  "embedded-ref":        "rgba(76,86,106,0.15)",
  // BundleGraph (Explorer) embedded-ref and observable-tree refs
  "flow-start":          "rgba(235,203,139,0.7)",
  "flow-effect":         "rgba(235,203,139,0.55)",
  "references":          "rgba(136,192,208,0.35)",
  "based-on":            "rgba(136,192,208,0.45)",
  "consists-of":         "rgba(208,135,112,0.45)",
  "has-log-source":      "rgba(143,188,187,0.45)",
  "dst":                 "rgba(94,129,172,0.4)",
  "src":                 "rgba(94,129,172,0.4)",
  "image":               "rgba(163,190,140,0.4)",
  "parent":              "rgba(163,190,140,0.4)",
  "parent_directory":    "rgba(163,190,140,0.4)",
  "content":             "rgba(136,192,208,0.35)",
  "resolves_to":         "rgba(94,129,172,0.35)",
  "belongs_to":          "rgba(94,129,172,0.35)",
  "observed-at":         "rgba(180,142,173,0.35)",
  "observed-by":         "rgba(180,142,173,0.3)",
  "observed-data":       "rgba(180,142,173,0.35)",
  // BundleGraph embedded-ref (dashed) variants
  "x-components":        "rgba(208,135,112,0.35)",
  "x-command":           "rgba(208,135,112,0.3)",
  "x-source":            "rgba(76,86,106,0.35)",
  "x-fingerprint":       "rgba(136,192,208,0.3)",
};

export function edgeColor(type) {
  return EDGE_COLORS[type] || "rgba(76,86,106,0.25)";
}

// ── Layer filter definitions ────────────────────────────────────────
// Each layer maps a label to the set of STIX types it controls. When
// a layer is off, nodes of those types (and edges connecting them) are
// hidden from both rendering and simulation.
export const FILTER_LAYERS = {
  "Procedures & Flow": new Set([
    "x-procedure", "attack-flow", "attack-action", "attack-operator", "attack-condition",
  ]),
  "Threat Context": new Set([
    "campaign", "intrusion-set", "malware", "tool", "threat-actor",
    "vulnerability", "infrastructure", "identity", "location", "report",
  ]),
  "Techniques & Observables": new Set([
    "attack-pattern",
    "process", "file", "domain-name", "ipv4-addr", "ipv6-addr", "url",
    "email-addr", "windows-registry-key", "network-traffic", "directory",
    "mutex", "software", "user-account",
  ]),
  "Detection": new Set([
    "x-mitre-detection-strategy", "x-mitre-analytic", "x-mitre-data-component",
    "x-mitre-data-source", "x-log-source", "indicator", "course-of-action",
  ]),
};

export const FILTER_LAYER_NAMES = Object.keys(FILTER_LAYERS);

// Color accents for each layer toggle button.
export const LAYER_COLORS = {
  "Procedures & Flow":       "#d08770",
  "Threat Context":          "#bf616a",
  "Techniques & Observables":"#b48ead",
  "Detection":               "#a3be8c",
};

// Types we never render — bundle metadata that adds noise without value.
export const SKIP_TYPES = new Set([
  "marking-definition", "extension-definition", "language-content",
]);

/** Ids of the bundle's metadata objects: every extension-definition plus the
 *  identity each names as its author. The serializer embeds both so a STIX
 *  consumer can resolve the extensions it declares; in a graph they would
 *  only float as orphan "author" nodes on every bundle. */
export function collectMetaIds(objects) {
  const ids = new Set();
  for (const obj of objects || []) {
    if (!obj || obj.type !== "extension-definition" || !obj.id) continue;
    ids.add(obj.id);
    if (typeof obj.created_by_ref === "string" && obj.created_by_ref) ids.add(obj.created_by_ref);
  }
  return ids;
}

/** Find the layer name that controls a given STIX type. Returns null
 *  when the type isn't covered by any layer (e.g. relationship,
 *  marking-definition). */
export function layerForType(stixType) {
  for (const [name, types] of Object.entries(FILTER_LAYERS)) {
    if (types.has(stixType)) return name;
  }
  return null;
}
