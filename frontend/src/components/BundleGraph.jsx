/**
 * BundleGraph — canvas-based STIX 2.1 bundle visualizer.
 *
 * Ported from viewer.html (Sherman Chu, CC BY-NC 4.0).
 * Renders STIX objects as a force-directed graph on an HTML5 canvas
 * with pan, zoom, node dragging, layer filters, and click-to-inspect
 * detail panel.
 *
 * Props:
 *   bundleJson  — full STIX 2.1 bundle object ({ type: "bundle", objects: [...] })
 *   className   — optional className for the wrapper div
 */
import { useRef, useEffect, useState, useCallback, useMemo } from "react";

import {
  FILTER_LAYERS,
  FILTER_LAYER_NAMES,
  LAYER_COLORS,
  SKIP_TYPES,
  collectMetaIds,
  getTypeConfig,
  edgeColor,
} from "../lib/bundleGraphConstants";
import { buildResolutionIndex } from "../lib/bundleResolution";
import StixNodeIcon from "./StixNodeIcon";
import NodeSearchBox from "./NodeSearchBox";
import NodeHoverTooltip from "./NodeHoverTooltip";
import {
  buildAdjacency,
  findAllShortestPaths,
  collectPathElements,
  pathEdgeKey,
} from "../lib/findPaths";

// Cache MDI SVG path strings as Path2D instances so we don't reparse
// per-frame. Keyed by path string (cheap identity test against the
// `icon` field returned by getTypeConfig).
const _pathCache = new Map();
function getIconPath2D(pathString) {
  if (!pathString) return null;
  let p = _pathCache.get(pathString);
  if (!p) {
    p = new Path2D(pathString);
    _pathCache.set(pathString, p);
  }
  return p;
}

// Layer Y positions for initial layout
const LAYERS = {
  "campaign": -350, "intrusion-set": -380, "threat-actor": -380,
  "malware": -200, "tool": -200,
  "x-procedure": -100,
  "attack-pattern": 60, "vulnerability": 40,
  "course-of-action": 120,
  "process": -200, "file": -180, "domain-name": -160,
  "ipv4-addr": -160, "url": -160,
  "identity": -480, "report": -450,
  "indicator": 550,
  "x-mitre-detection-strategy": 220,
  "x-mitre-analytic": 340,
  "x-mitre-data-component": 450,
};

// Fields to hide in the detail panel (meta / already shown in header)
const DETAIL_SKIP_FIELDS = new Set([
  "type", "id", "spec_version", "name", "description", "confidence",
]);

// Largest per-frame node move (world units) below which the force
// simulation counts as settled and stops.
const SETTLED_EPSILON = 0.02;

/** Command lines of a procedure: the Process SCOs in x_components_refs.
 *  x_command_lines left the schema in v1.0.0; commands live on the
 *  components now, which is where the Flow view already reads them. */
function commandLinesFor(obj, byId) {
  if (obj.type !== "x-procedure") return [];
  return (obj.x_components_refs || [])
    .map((ref) => byId[ref])
    .filter((o) => o?.type === "process" && o.command_line)
    .map((o) => o.command_line);
}


// ── Bundle parser ────────────────────────────────────────────────────
function parseBundle(bundle) {
  const objects = bundle?.objects || [];
  // Extension definitions and the identities they name as author: metadata
  // about the object types, not the intrusion. Rendering them would hang
  // the same two "author" nodes off every bundle.
  const metaIds = collectMetaIds(objects);
  const nodeMap = {};
  const edges = [];
  const rawObjects = {}; // stix id -> raw STIX object
  const byId = Object.fromEntries(objects.map((o) => [o.id, o]));
  // Same per-type labels as the Flow view, so a Process SCO reads as its
  // command line here too instead of "process--3f1a2b4c…".
  const resolution = buildResolutionIndex(bundle);

  // Pass 1: nodes
  for (const obj of objects) {
    if (!obj.id || !obj.type || SKIP_TYPES.has(obj.type) || metaIds.has(obj.id) || obj.type === "relationship") continue;

    const tids = (obj.external_references || [])
      .filter((r) => r.source_name === "mitre-attack")
      .map((r) => r.external_id);

    const label = obj.name || obj.value
      || resolution.get(obj.id)?.displayName || obj.id.slice(0, 20);

    rawObjects[obj.id] = obj;
    nodeMap[obj.id] = {
      id: obj.id,
      type: obj.type,
      label: tids.length ? `${tids[0]}: ${label}` : label,
      name: obj.name || label,
      desc: obj.description || "",
      cmdlines: commandLinesFor(obj, byId),
      confidence: obj.confidence || 0,
      rawObj: obj,
      x: 0, y: 0, vx: 0, vy: 0,
    };
  }

  // Pass 2: SRO edges
  for (const obj of objects) {
    if (obj.type !== "relationship" || !obj.source_ref || !obj.target_ref) continue;
    let edgeType = obj.relationship_type || "related-to";
    const src = nodeMap[obj.source_ref];
    const tgt = nodeMap[obj.target_ref];
    if (edgeType === "uses" && src && tgt && tgt.type === "attack-pattern") {
      edgeType = "uses-technique";
    }
    edges.push({ source: obj.source_ref, target: obj.target_ref, type: edgeType });
  }

  // Pass 2b: Sighting edges
  for (const obj of objects) {
    if (obj.type !== "sighting") continue;
    if (obj.sighting_of_ref && nodeMap[obj.sighting_of_ref]) {
      edges.push({ source: obj.id, target: obj.sighting_of_ref, type: "sighting-of" });
    }
    if (Array.isArray(obj.where_sighted_refs)) {
      for (const ref of obj.where_sighted_refs) {
        if (nodeMap[ref]) edges.push({ source: obj.id, target: ref, type: "observed-at" });
      }
    }
    if (Array.isArray(obj.observed_data_refs)) {
      for (const ref of obj.observed_data_refs) {
        if (nodeMap[ref]) edges.push({ source: obj.id, target: ref, type: "observed-data" });
      }
    }
    if (obj.created_by_ref && nodeMap[obj.created_by_ref]) {
      edges.push({ source: obj.id, target: obj.created_by_ref, type: "observed-by" });
    }
  }

  // Pass 3: implicit edges (embedded refs)
  const SCO_REF_FIELDS = ["dst_ref", "src_ref", "image_ref", "parent_ref", "parent_directory_ref", "content_ref", "resolves_to_refs", "belongs_to_refs"];
  const ATTACK_FLOW_EXT = "extension-definition--fb9c968a-745b-4ade-9b25-c324172197f4";

  for (const obj of objects) {
    if (!obj.id || SKIP_TYPES.has(obj.type) || metaIds.has(obj.id) || obj.type === "relationship" || obj.type === "sighting") continue;

    if (obj.x_technique_refs) {
      for (const ref of obj.x_technique_refs) {
        if (nodeMap[ref]) edges.push({ source: obj.id, target: ref, type: "uses-technique" });
      }
    }
    if (obj.object_refs) {
      for (const ref of obj.object_refs) {
        if (nodeMap[ref]) edges.push({ source: obj.id, target: ref, type: "references" });
      }
    }
    if (Array.isArray(obj.x_vulnerability_refs)) {
      for (const ref of obj.x_vulnerability_refs) {
        if (nodeMap[ref]) edges.push({ source: obj.id, target: ref, type: "exploits" });
      }
    }
    if (obj.type === "attack-flow" && Array.isArray(obj.start_refs)) {
      for (const ref of obj.start_refs) {
        if (nodeMap[ref]) edges.push({ source: obj.id, target: ref, type: "flow-start" });
      }
    }
    if (obj.extensions && obj.extensions[ATTACK_FLOW_EXT]) {
      const af = obj.extensions[ATTACK_FLOW_EXT];
      if (Array.isArray(af.start_refs)) {
        for (const ref of af.start_refs) {
          if (nodeMap[ref]) edges.push({ source: obj.id, target: ref, type: "flow-start" });
        }
      }
      if (Array.isArray(af.effect_refs)) {
        for (const ref of af.effect_refs) {
          if (nodeMap[ref]) edges.push({ source: obj.id, target: ref, type: "flow-effect" });
        }
      }
    }
    for (const field of SCO_REF_FIELDS) {
      const val = obj[field];
      if (!val) continue;
      const label = field.replace(/_refs?$/, "");
      if (typeof val === "string") {
        if (nodeMap[val]) edges.push({ source: obj.id, target: val, type: label });
      } else if (Array.isArray(val)) {
        for (const ref of val) {
          if (nodeMap[ref]) edges.push({ source: obj.id, target: ref, type: label });
        }
      }
    }
    if (obj.x_mitre_data_source_ref && nodeMap[obj.x_mitre_data_source_ref]) {
      edges.push({ source: obj.id, target: obj.x_mitre_data_source_ref, type: "derived-from" });
    }
    if (Array.isArray(obj.x_log_source_refs)) {
      for (const ref of obj.x_log_source_refs) {
        if (nodeMap[ref]) edges.push({ source: obj.id, target: ref, type: "has-log-source" });
      }
    }
    if (obj.x_mitre_data_component_ref && nodeMap[obj.x_mitre_data_component_ref]) {
      edges.push({ source: obj.id, target: obj.x_mitre_data_component_ref, type: "derived-from" });
    }
  }

  // Pass 4: generic embedded-ref edge pass
  // Scans every non-relationship object for *_ref / *_refs properties
  // that point to nodes in the bundle but weren't covered by Passes 2-3.
  // These get dashed edges so analysts can see structural links
  // (e.g. x_components_refs) without cluttering the graph.
  const HANDLED_REF_FIELDS = new Set([
    // Pass 2 / 2b (relationship & sighting objects are skipped, but
    // list their fields here in case they appear on other types)
    "source_ref", "target_ref", "sighting_of_ref",
    "where_sighted_refs", "observed_data_refs",
    // Pass 3 explicit
    "x_technique_refs",
    "object_refs", "x_vulnerability_refs", "start_refs",
    "x_mitre_data_source_ref", "x_log_source_refs",
    "x_mitre_data_component_ref",
    // Not on x-procedure in v0.5.0-draft: kept here so any legacy bundle
    // still carrying them doesn't sprout generic Pass-4 edges. Sequencing
    // lives in PRECEDES SROs + the attack-flow object; observables live on
    // has-observable SROs, which Pass 1 already draws.
    "x_effect_refs", "x_flow_ref", "x_command_ref", "x_observable_refs",
    // Attack Flow extension refs (handled via extensions dict)
    "effect_refs",
    // Pass 3 SCO ref fields
    "dst_ref", "src_ref", "image_ref", "parent_ref",
    "parent_directory_ref", "content_ref", "resolves_to_refs",
    "belongs_to_refs",
    // Noise: hub-and-spoke meta refs
    "created_by_ref", "object_marking_refs",
    "sample_refs", "sample_ref",
  ]);
  // Build a quick lookup of existing edges so we don't double-up
  const existingEdgeKeys = new Set(edges.map((e) => `${e.source}|${e.target}`));

  for (const obj of objects) {
    if (!obj.id || SKIP_TYPES.has(obj.type) || metaIds.has(obj.id) || obj.type === "relationship" || obj.type === "sighting") continue;
    for (const [field, val] of Object.entries(obj)) {
      if (HANDLED_REF_FIELDS.has(field)) continue;
      if (!field.endsWith("_ref") && !field.endsWith("_refs")) continue;
      // extensions object contains nested dicts, not refs
      if (field === "extensions") continue;

      const edgeLabel = field.replace(/_refs?$/, "").replace(/_/g, "-");
      const refs = Array.isArray(val) ? val : (typeof val === "string" ? [val] : []);
      for (const ref of refs) {
        if (!nodeMap[ref]) continue;
        const key = `${obj.id}|${ref}`;
        if (existingEdgeKeys.has(key)) continue;
        existingEdgeKeys.add(key);
        edges.push({ source: obj.id, target: ref, type: edgeLabel, dashed: true });
      }
    }
  }

  // De-dupe
  const seen = new Set();
  const deduped = [];
  for (const e of edges) {
    const key = `${e.source}|${e.target}|${e.type}`;
    if (seen.has(key)) continue;
    seen.add(key);
    deduped.push(e);
  }
  edges.length = 0;
  edges.push(...deduped);

  // Initial layout
  const typeGroups = {};
  const nodes = Object.values(nodeMap);
  for (const n of nodes) {
    (typeGroups[n.type] || (typeGroups[n.type] = [])).push(n);
  }
  for (const [t, group] of Object.entries(typeGroups)) {
    const layerY = LAYERS[t] || 0;
    for (let i = 0; i < group.length; i++) {
      group[i].x = (i - group.length / 2) * 80 + (Math.random() - 0.5) * 30;
      group[i].y = layerY + (Math.random() - 0.5) * 40;
    }
  }

  // Resolve edge refs to node objects
  const links = edges
    .map((e) => ({ source: nodeMap[e.source], target: nodeMap[e.target], type: e.type, dashed: !!e.dashed }))
    .filter((l) => l.source && l.target);

  return { nodes, edges, links, nodeMap, rawObjects };
}


// ── Visibility filter helper ────────────────────────────────────────
function applyLayerFilter(allNodes, allLinks, activeLayers) {
  // Build set of visible STIX types from active layers.
  // Types not in ANY layer definition are always visible (uncategorized).
  const hiddenTypes = new Set();
  for (const [name, typeSet] of Object.entries(FILTER_LAYERS)) {
    if (!activeLayers[name]) {
      for (const t of typeSet) hiddenTypes.add(t);
    }
  }
  const nodes = allNodes.filter((n) => !hiddenTypes.has(n.type));
  const visibleIds = new Set(nodes.map((n) => n.id));
  const links = allLinks.filter((l) => visibleIds.has(l.source.id) && visibleIds.has(l.target.id));
  return { nodes, links };
}


// ── React component ──────────────────────────────────────────────────
export default function BundleGraph({ bundleJson, className = "" }) {
  const canvasRef = useRef(null);
  const stateRef = useRef({
    allNodes: [], allLinks: [], // full set (unfiltered)
    nodes: [], links: [],       // visible (filtered) set
    nodeMap: {},
    transform: { x: 0, y: 0, k: 0.55 },
    showLabels: true,
    showEdges: true,
    showEdgeLabels: false,
    gravity: true,
    // Path-trace state. pathStartNode + pathEndNode hold the picked
    // {id, name, type} for banner display; pathNodes/pathEdges hold
    // the union of every shortest-path's elements so the draw loop
    // can dim everything else without re-running BFS per frame.
    pathStartNode: null,
    pathEndNode: null,
    pathNodes: new Set(),
    pathEdges: new Set(),
    pathCount: 0,
    simAlpha: 0,
    simRunning: false,
    dragging: false,
    dragNode: null,
    panning: false,
    panStart: null,
    transformStart: null,
    selectedNode: null,
    W: 0, H: 0,
  });
  const rafRef = useRef(null);
  const [, forceRender] = useState(0);
  const [showLabels, setShowLabels] = useState(true);
  const [showEdges, setShowEdges] = useState(true);
  const [showEdgeLabels, setShowEdgeLabels] = useState(false);
  const [gravity, setGravity] = useState(true);
  // Legend open/closed, remembered per browser. Whoever collapses it wants it
  // out of the way for more than one bundle; making them re-close it on every
  // selection is the same annoyance in slow motion. Wrapped because storage
  // throws outright in some privacy modes, and a legend is not worth a crash.
  const [legendOpen, setLegendOpen] = useState(() => {
    try {
      return localStorage.getItem("bundleGraph.legendOpen") !== "0";
    } catch {
      return true;
    }
  });
  useEffect(() => {
    try {
      localStorage.setItem("bundleGraph.legendOpen", legendOpen ? "1" : "0");
    } catch {
      /* storage unavailable — the toggle still works for this session */
    }
  }, [legendOpen]);
  const [selectedNode, setSelectedNode] = useState(null);
  const [hoverNode, setHoverNode] = useState(null);
  const [hoverPos, setHoverPos] = useState({ x: 0, y: 0 });
  // Mirror of stateRef.path* — kept in React state for banner rerender.
  // The renderer reads stateRef.pathNodes / pathEdges directly so we
  // don't pay for set comparisons each frame.
  const [pathInfo, setPathInfo] = useState({
    start: null, end: null, count: 0,
  });

  // Layer filter state: all on by default
  const [activeLayers, setActiveLayers] = useState(() => {
    const init = {};
    for (const name of FILTER_LAYER_NAMES) init[name] = true;
    return init;
  });

  // Parse bundle when it changes
  useEffect(() => {
    if (!bundleJson) return;
    const parsed = parseBundle(bundleJson);
    const s = stateRef.current;
    s.allNodes = parsed.nodes;
    s.allLinks = parsed.links;
    s.nodeMap = parsed.nodeMap;
    // Apply current filters
    const filtered = applyLayerFilter(parsed.nodes, parsed.links, activeLayers);
    s.nodes = filtered.nodes;
    s.links = filtered.links;
    s.transform = { x: 0, y: 0, k: 0.55 };
    s.simAlpha = 1;
    s.simRunning = true;
    s.selectedNode = null;
    setSelectedNode(null);
    forceRender((n) => n + 1);
    startSim();
  }, [bundleJson]);

  // Re-filter when layers change (but not on initial mount / bundle change)
  const isFirstRender = useRef(true);
  useEffect(() => {
    if (isFirstRender.current) {
      isFirstRender.current = false;
      return;
    }
    const s = stateRef.current;
    if (!s.allNodes.length) return;
    const filtered = applyLayerFilter(s.allNodes, s.allLinks, activeLayers);
    s.nodes = filtered.nodes;
    s.links = filtered.links;
    // Restart simulation so gravity rebalances around visible nodes
    s.simAlpha = 0.6;
    s.simRunning = true;
    // Dismiss detail panel if selected node is now hidden
    if (s.selectedNode && !filtered.nodes.includes(s.selectedNode)) {
      s.selectedNode = null;
      setSelectedNode(null);
    }
    forceRender((n) => n + 1);
    startSim();
  }, [activeLayers]);

  // Resize observer
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      const rect = canvas.parentElement.getBoundingClientRect();
      canvas.width = rect.width;
      canvas.height = rect.height;
      stateRef.current.W = rect.width;
      stateRef.current.H = rect.height;
      draw();
    });
    ro.observe(canvas.parentElement);
    return () => ro.disconnect();
  }, []);

  // ── Drawing ──────────────────────────────────────────────────────
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const s = stateRef.current;
    const { W, H, transform, nodes, links } = s;
    if (!W || !H) return;

    ctx.save();
    ctx.clearRect(0, 0, W, H);
    ctx.translate(W / 2 + transform.x, H / 2 + transform.y);
    ctx.scale(transform.k, transform.k);

    // Build focus set: when a node is selected, only it and its
    // direct neighbors (1-hop) render at full opacity. Everything
    // else dims to ~15% so the local neighborhood pops visually.
    // Path-trace mode supersedes single-node focus: when the analyst
    // has shift-clicked a second node, we union the node + edge sets
    // of every shortest path between A and B instead.
    let focusIds = null; // null = no focus mode (everything full)
    let focusEdges = null;
    if (s.pathNodes && s.pathNodes.size > 0) {
      focusIds = s.pathNodes;
      focusEdges = new Set();
      for (let i = 0; i < links.length; i++) {
        const l = links[i];
        const key = pathEdgeKey(l.source.id, l.target.id);
        if (s.pathEdges.has(key)) {
          focusEdges.add(i);
        }
      }
    } else if (s.selectedNode) {
      focusIds = new Set([s.selectedNode.id]);
      focusEdges = new Set();
      for (let i = 0; i < links.length; i++) {
        const l = links[i];
        if (l.source.id === s.selectedNode.id || l.target.id === s.selectedNode.id) {
          focusIds.add(l.source.id);
          focusIds.add(l.target.id);
          focusEdges.add(i);
        }
      }
    }

    // Edges
    if (s.showEdges) {
      // Edge labels render at midpoint when the toggle is on AND the
      // current zoom is high enough that text won't pile up. Threshold
      // picked empirically against one ransomware bundle (1185 edges,
      // 1474 objects) — below k≈0.85 the labels overlap each other
      // even with the in-focus filter applied. Skip labels for dimmed
      // and dashed (embedded-ref) edges; those are visual noise.
      const labelsOn = s.showEdgeLabels && s.transform.k >= 0.85;
      // Collect midpoints to draw labels in a second pass so they
      // layer above all edges and arrowheads.
      const labelDraws = [];
      for (let i = 0; i < links.length; i++) {
        const l = links[i];
        const dimmed = focusEdges && !focusEdges.has(i);
        ctx.strokeStyle = edgeColor(l.type);
        ctx.lineWidth = l.type === "precedes" ? 1.8 : (l.dashed ? 0.7 : 0.8);
        ctx.globalAlpha = dimmed ? 0.06 : (l.dashed ? 0.55 : 0.9);
        if (l.dashed) ctx.setLineDash([6, 4]);
        else if (l.type === "detects") ctx.setLineDash([4, 3]);
        else ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(l.source.x, l.source.y);
        ctx.lineTo(l.target.x, l.target.y);
        ctx.stroke();
        ctx.setLineDash([]);

        // Arrowhead (skip for dimmed edges to reduce visual noise)
        if (!dimmed) {
          const dx = l.target.x - l.source.x;
          const dy = l.target.y - l.source.y;
          const len = Math.sqrt(dx * dx + dy * dy) || 1;
          const tr = (getTypeConfig(l.target.type)?.r || 6) + 3;
          const ax = l.target.x - (dx / len) * tr;
          const ay = l.target.y - (dy / len) * tr;
          const angle = Math.atan2(dy, dx);
          ctx.fillStyle = ctx.strokeStyle;
          ctx.beginPath();
          ctx.moveTo(ax, ay);
          ctx.lineTo(ax - 6 * Math.cos(angle - 0.4), ay - 6 * Math.sin(angle - 0.4));
          ctx.lineTo(ax - 6 * Math.cos(angle + 0.4), ay - 6 * Math.sin(angle + 0.4));
          ctx.closePath();
          ctx.fill();

          // Stash label info (skip dashed embedded-ref edges — they
          // pollute the canvas with x-components / x-source noise).
          if (labelsOn && !l.dashed) {
            labelDraws.push({
              x: (l.source.x + l.target.x) / 2,
              y: (l.source.y + l.target.y) / 2,
              text: l.type,
            });
          }
        }
      }

      // Second pass: edge labels stacked above the edge layer so
      // arrowheads / other edges don't bleed through.
      if (labelDraws.length) {
        ctx.globalAlpha = 1;
        ctx.font = "9px 'IBM Plex Mono', monospace";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        for (const ld of labelDraws) {
          const textW = ctx.measureText(ld.text).width;
          const pad = 3;
          // Dark pill background so the label reads against the
          // graph + edge colors.
          ctx.fillStyle = "rgba(29,32,33,0.85)";
          ctx.fillRect(
            ld.x - textW / 2 - pad,
            ld.y - 6,
            textW + pad * 2,
            12,
          );
          ctx.fillStyle = "#a89984";
          ctx.fillText(ld.text, ld.x, ld.y);
        }
      }
    }

    // Nodes — each node is a rounded-square badge with a white MDI
    // glyph inside, the same badge the React Flow viewers draw with
    // their StixNodeIcon component.
    for (const n of nodes) {
      const cat = getTypeConfig(n.type);
      const isSelected = s.selectedNode && s.selectedNode.id === n.id;
      const inFocus = !focusIds || focusIds.has(n.id);
      ctx.globalAlpha = inFocus ? 1 : 0.12;
      const r = cat.r;
      const badge = 2 * r; // edge length of the badge square
      const x0 = n.x - r;
      const y0 = n.y - r;
      const rxy = Math.max(2, r * 0.32);

      // Badge fill
      ctx.fillStyle = cat.color;
      ctx.strokeStyle = isSelected ? "#ebcb8b" : "rgba(216,222,233,0.3)";
      ctx.lineWidth = isSelected ? 2.5 : 1;
      ctx.beginPath();
      ctx.roundRect(x0, y0, badge, badge, rxy);
      // Subtle glow for x-procedure (preserves prior hex-shape emphasis).
      if (n.type === "x-procedure") {
        ctx.shadowColor = cat.color;
        ctx.shadowBlur = 10;
        ctx.fill();
        ctx.shadowBlur = 0;
      } else {
        ctx.fill();
      }
      ctx.stroke();

      // Glyph: white MDI icon centered at ~60% of badge size. MDI paths
      // use a 24x24 viewBox so the scale factor is iconSize / 24.
      const iconPath = getIconPath2D(cat.icon);
      if (iconPath) {
        const iconSize = badge * 0.6;
        const offset = (badge - iconSize) / 2;
        ctx.save();
        ctx.translate(x0 + offset, y0 + offset);
        ctx.scale(iconSize / 24, iconSize / 24);
        ctx.fillStyle = "#fff";
        ctx.fill(iconPath);
        ctx.restore();
      }

      // Labels (skip for dimmed nodes to reduce clutter)
      if (s.showLabels && inFocus) {
        ctx.globalAlpha = isSelected ? 1 : (focusIds ? 0.8 : 1);
        ctx.font = n.type === "x-procedure"
          ? "600 10px 'IBM Plex Mono', monospace"
          : "9px 'IBM Plex Sans', sans-serif";
        ctx.fillStyle = "#d8dee9";
        ctx.textAlign = "center";
        const lbl = n.label.length > 35 ? n.label.slice(0, 33) + ".." : n.label;
        ctx.fillText(lbl, n.x, n.y + r + 12);
      }
    }

    ctx.restore();
  }, []);

  // ── Force simulation ─────────────────────────────────────────────
  const simTick = useCallback(() => {
    const s = stateRef.current;
    if (!s.simRunning || s.simAlpha < 0.001) {
      s.simRunning = false;
      return;
    }
    s.simAlpha *= 0.995;

    // Spring forces (edges) — only visible links
    for (const l of s.links) {
      const dx = l.target.x - l.source.x;
      const dy = l.target.y - l.source.y;
      const dist = Math.sqrt(dx * dx + dy * dy) || 1;
      const ideal = l.type.includes("observable") ? 80 : l.type.includes("technique") ? 120 : 100;
      const f = (dist - ideal) * 0.0005 * s.simAlpha;
      const fx = (dx / dist) * f;
      const fy = (dy / dist) * f;
      l.source.vx += fx;
      l.source.vy += fy;
      l.target.vx -= fx;
      l.target.vy -= fy;
    }

    // Repulsion forces — only visible nodes
    const nodes = s.nodes;
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const a = nodes[i], b = nodes[j];
        const dx = b.x - a.x, dy = b.y - a.y;
        const d2 = dx * dx + dy * dy;
        if (d2 > 40000) continue;
        const d = Math.sqrt(d2) || 0.1;
        const f = (-200 * s.simAlpha) / (d * d);
        const fx = (dx / d) * f, fy = (dy / d) * f;
        a.vx += fx; a.vy += fy;
        b.vx -= fx; b.vy -= fy;
      }
    }

    // Integrate, tracking the largest move so the loop can stop once the
    // picture is static. Alpha alone decays to 0.001 over ~1,400 frames
    // (~23 s) while 0.85 damping settles positions in a few hundred; the
    // remaining frames ran the O(n^2) repulsion and a full redraw of an
    // unchanged canvas.
    let maxMove = 0;
    for (const n of nodes) {
      n.vx *= 0.85;
      n.vy *= 0.85;
      n.x += n.vx;
      n.y += n.vy;
      if (isNaN(n.x)) { n.x = Math.random() * 100; n.vx = 0; }
      if (isNaN(n.y)) { n.y = Math.random() * 100; n.vy = 0; }
      const moved = Math.abs(n.vx) + Math.abs(n.vy);
      if (moved > maxMove) maxMove = moved;
    }

    draw();
    if (maxMove < SETTLED_EPSILON) {
      s.simRunning = false;
      return;
    }
    if (s.simAlpha >= 0.001) {
      rafRef.current = requestAnimationFrame(simTick);
    }
  }, [draw]);

  const startSim = useCallback(() => {
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    rafRef.current = requestAnimationFrame(simTick);
  }, [simTick]);

  // Cleanup RAF on unmount
  useEffect(() => {
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, []);

  // ── Interaction helpers ──────────────────────────────────────────
  const screenToWorld = useCallback((sx, sy) => {
    const s = stateRef.current;
    return {
      x: (sx - s.W / 2 - s.transform.x) / s.transform.k,
      y: (sy - s.H / 2 - s.transform.y) / s.transform.k,
    };
  }, []);

  const findNode = useCallback((wx, wy) => {
    const nodes = stateRef.current.nodes;
    for (let i = nodes.length - 1; i >= 0; i--) {
      const n = nodes[i];
      const r = (getTypeConfig(n.type)?.r || 6) + 3;
      if (Math.abs(n.x - wx) < r && Math.abs(n.y - wy) < r) return n;
    }
    return null;
  }, []);

  // ── Mouse handlers ───────────────────────────────────────────────
  const handleWheel = useCallback((e) => {
    e.preventDefault();
    const s = stateRef.current;
    const factor = e.deltaY > 0 ? 0.92 : 1.08;
    s.transform.k = Math.max(0.1, Math.min(5, s.transform.k * factor));
    draw();
  }, [draw]);

  const handleMouseDown = useCallback((e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const sx = e.clientX - rect.left;
    const sy = e.clientY - rect.top;
    const w = screenToWorld(sx, sy);
    const n = findNode(w.x, w.y);
    const s = stateRef.current;
    if (n) {
      s.dragNode = n;
      s.simRunning = false;
    } else {
      s.panning = true;
      s.panMoved = false;
      s.panStart = { x: e.clientX, y: e.clientY };
      s.transformStart = { x: s.transform.x, y: s.transform.y };
    }
  }, [screenToWorld, findNode]);

  const handleMouseMove = useCallback((e) => {
    const s = stateRef.current;
    if (s.dragNode) {
      const rect = canvasRef.current.getBoundingClientRect();
      const w = screenToWorld(e.clientX - rect.left, e.clientY - rect.top);
      s.dragNode.x = w.x;
      s.dragNode.y = w.y;
      draw();
    } else if (s.panning && s.panStart) {
      s.panMoved = true;
      s.transform.x = s.transformStart.x + (e.clientX - s.panStart.x);
      s.transform.y = s.transformStart.y + (e.clientY - s.panStart.y);
      draw();
    } else {
      // Hover hit-test for the tooltip. Suppressed during drag/pan so
      // the tooltip doesn't flicker. Stored in stateRef + react state
      // so the DOM tooltip rerenders without round-tripping through
      // the canvas draw loop.
      const rect = canvasRef.current.getBoundingClientRect();
      const sx = e.clientX - rect.left;
      const sy = e.clientY - rect.top;
      const w = screenToWorld(sx, sy);
      const n = findNode(w.x, w.y);
      if (n !== s.hoverNode) {
        s.hoverNode = n || null;
        setHoverNode(n || null);
      }
      // Only while a node is under the cursor: the tooltip is the sole
      // reader, and a fresh {x, y} object never passes Object.is, so
      // setting it on every mousemove re-rendered the whole Explorer at
      // pointer rate — legend, detail panel and JSON.stringify per field.
      if (n) setHoverPos({ x: sx, y: sy });
    }
  }, [screenToWorld, findNode, draw]);

  const handleMouseLeave = useCallback(() => {
    if (stateRef.current.hoverNode) {
      stateRef.current.hoverNode = null;
      setHoverNode(null);
    }
  }, []);

  const handleMouseUp = useCallback(() => {
    const s = stateRef.current;
    if (s.dragNode) {
      s.dragNode.vx = 0;
      s.dragNode.vy = 0;
      s.dragNode = null;
      if (s.gravity) {
        s.simRunning = true;
        s.simAlpha = 0.1;
        startSim();
      }
    }
    s.panning = false;
    s.panStart = null;
  }, [startSim]);

  const handleClick = useCallback((e) => {
    const s = stateRef.current;
    // mouseup has already cleared `panning` by the time the browser
    // dispatches click, so that flag never guarded anything: every pan
    // ended by deselecting the node. "Did this gesture move" survives.
    if (s.panMoved) {
      s.panMoved = false;
      return;
    }
    const rect = canvasRef.current.getBoundingClientRect();
    const w = screenToWorld(e.clientX - rect.left, e.clientY - rect.top);
    const n = findNode(w.x, w.y);

    // Shift-click activates path-trace mode. The first selected node
    // (s.selectedNode) acts as the start; the shift-clicked node is
    // the target. A plain click clears any active trace and falls
    // back to single-node focus.
    if (e.shiftKey && n && s.selectedNode && s.selectedNode.id !== n.id) {
      const adj = buildAdjacency(s.allLinks);
      const paths = findAllShortestPaths(adj, s.selectedNode.id, n.id);
      const { nodeSet, edgeSet } = collectPathElements(paths);
      s.pathStartNode = s.selectedNode;
      s.pathEndNode = n;
      s.pathNodes = nodeSet;
      s.pathEdges = edgeSet;
      s.pathCount = paths.length;
      setPathInfo({
        start: { id: s.selectedNode.id, name: s.selectedNode.name, type: s.selectedNode.type },
        end: { id: n.id, name: n.name, type: n.type },
        count: paths.length,
      });
      draw();
      return;
    }

    // Plain click: clear any active trace, single-node focus.
    if (s.pathNodes.size) {
      s.pathStartNode = null;
      s.pathEndNode = null;
      s.pathNodes = new Set();
      s.pathEdges = new Set();
      s.pathCount = 0;
      setPathInfo({ start: null, end: null, count: 0 });
    }
    s.selectedNode = n || null;
    setSelectedNode(n || null);
    draw();
  }, [screenToWorld, findNode, draw]);

  const clearPathTrace = useCallback(() => {
    const s = stateRef.current;
    s.pathStartNode = null;
    s.pathEndNode = null;
    s.pathNodes = new Set();
    s.pathEdges = new Set();
    s.pathCount = 0;
    setPathInfo({ start: null, end: null, count: 0 });
    draw();
  }, [draw]);

  // Attach wheel listener with passive: false
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.addEventListener("wheel", handleWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", handleWheel);
  }, [handleWheel]);

  // Toggle handlers
  const toggleLabels = useCallback(() => {
    stateRef.current.showLabels = !stateRef.current.showLabels;
    setShowLabels(stateRef.current.showLabels);
    draw();
  }, [draw]);

  const toggleEdges = useCallback(() => {
    stateRef.current.showEdges = !stateRef.current.showEdges;
    setShowEdges(stateRef.current.showEdges);
    draw();
  }, [draw]);

  const toggleEdgeLabels = useCallback(() => {
    stateRef.current.showEdgeLabels = !stateRef.current.showEdgeLabels;
    setShowEdgeLabels(stateRef.current.showEdgeLabels);
    draw();
  }, [draw]);

  // Jump-to-node: center the viewport on a node and select it. The
  // transform's (x, y) is in screen-space relative to canvas center, so
  // (transform.x, transform.y) = (-node.x * k, -node.y * k) centers
  // that world-space point at (W/2, H/2) on screen. The zoom is bumped
  // to 1.1 so we land above the edge-label threshold without making the
  // graph feel cramped.
  const jumpToNode = useCallback(
    (matched) => {
      const s = stateRef.current;
      const fullNode = s.allNodes.find((n) => n.id === matched.id);
      if (!fullNode) return;
      s.transform.k = 1.1;
      s.transform.x = -fullNode.x * s.transform.k;
      s.transform.y = -fullNode.y * s.transform.k;
      s.selectedNode = fullNode;
      setSelectedNode(fullNode);
      draw();
    },
    [draw],
  );

  // Build the {id, displayName, type, mitreId} list once per bundle.
  // Reads directly from bundleJson so it's available on first render —
  // parseBundle runs in an effect against stateRef, which wouldn't
  // trigger a re-render of this memo. Cheap: one pass over objects[].
  const searchResolution = useMemo(() => buildResolutionIndex(bundleJson), [bundleJson]);
  const searchableNodes = useMemo(() => {
    const objects = bundleJson?.objects || [];
    const metaIds = collectMetaIds(objects);
    const out = [];
    for (const obj of objects) {
      if (!obj.id || !obj.type) continue;
      if (SKIP_TYPES.has(obj.type) || metaIds.has(obj.id)) continue;
      if (obj.type === "relationship" || obj.type === "sighting") continue;
      const mitre = (obj.external_references || []).find(
        (r) => r.source_name === "mitre-attack",
      );
      out.push({
        id: obj.id,
        displayName: obj.name || obj.value
          || searchResolution.get(obj.id)?.displayName || obj.id.slice(0, 20),
        type: obj.type,
        mitreId: mitre?.external_id || null,
      });
    }
    return out;
  }, [bundleJson]);

  const toggleGravity = useCallback(() => {
    const s = stateRef.current;
    s.gravity = !s.gravity;
    setGravity(s.gravity);
    if (!s.gravity) {
      s.simRunning = false;
      s.simAlpha = 0;
      for (const n of s.nodes) { n.vx = 0; n.vy = 0; }
    } else {
      s.simRunning = true;
      s.simAlpha = 0.3;
      startSim();
    }
  }, [startSim]);

  const toggleLayer = useCallback((name) => {
    setActiveLayers((prev) => ({ ...prev, [name]: !prev[name] }));
  }, []);

  const resetView = useCallback(() => {
    const s = stateRef.current;
    s.transform = { x: 0, y: 0, k: 0.55 };
    if (s.gravity) {
      s.simAlpha = 1;
      s.simRunning = true;
      startSim();
    }
    draw();
  }, [draw, startSim]);

  // Count stats (visible only)
  const nodeCount = stateRef.current.nodes.length;
  const edgeCount = stateRef.current.links.length;
  const totalCount = stateRef.current.allNodes.length;

  if (!bundleJson) {
    return (
      <div className={`flex items-center justify-center h-full text-gb-bg4 text-sm ${className}`}>
        Select a bundle to view its graph
      </div>
    );
  }

  return (
    <div className={`flex flex-col h-full ${className}`}>
      {/* Pane header */}
      <div className="flex items-center justify-between px-3.5 py-2 bg-gb-bg0 border-b border-gb-bg1 shrink-0 flex-wrap gap-y-1">
        <span className="text-[11px] font-semibold text-gb-fg3 uppercase tracking-wider flex items-center gap-1.5">
          <span className="text-[13px]">&#128300;</span>
          STIX Bundle
          <span className="font-normal text-gb-gray font-data text-[10px] ml-1">
            {nodeCount === totalCount
              ? `${nodeCount} objects`
              : `${nodeCount}/${totalCount} objects`}{" "}
            &middot; {edgeCount} relationships
          </span>
        </span>
        <div className="flex gap-1 flex-wrap">
          {/* Layer filters */}
          {FILTER_LAYER_NAMES.map((name) => {
            const on = activeLayers[name];
            const col = LAYER_COLORS[name] || "#88c0d0";
            return (
              <button
                key={name}
                onClick={() => toggleLayer(name)}
                className="px-2 py-0.5 rounded text-[10px] font-data border transition-colors"
                style={
                  on
                    ? { color: col, borderColor: col, backgroundColor: col + "18" }
                    : { color: "#665c54", borderColor: "#3c3836" }
                }
                title={`Toggle ${name}`}
              >
                {name}
              </button>
            );
          })}
          <span className="w-px bg-gb-bg2 mx-0.5" />
          <NodeSearchBox
            nodes={searchableNodes}
            onJump={jumpToNode}
            placeholder="Find node…"
          />
          <button
            onClick={resetView}
            className="px-2 py-0.5 rounded text-[10px] font-data text-gb-fg4 border border-gb-bg2 hover:text-gb-fg1 transition-colors"
          >
            Reset
          </button>
          <button
            onClick={toggleLabels}
            className={`px-2 py-0.5 rounded text-[10px] font-data border transition-colors ${
              showLabels
                ? "text-gb-bright-blue border-gb-bright-blue bg-gb-bright-blue-dim"
                : "text-gb-fg4 border-gb-bg2"
            }`}
          >
            Labels
          </button>
          <button
            onClick={toggleEdges}
            className={`px-2 py-0.5 rounded text-[10px] font-data border transition-colors ${
              showEdges
                ? "text-gb-bright-blue border-gb-bright-blue bg-gb-bright-blue-dim"
                : "text-gb-fg4 border-gb-bg2"
            }`}
          >
            Edges
          </button>
          <button
            onClick={toggleEdgeLabels}
            className={`px-2 py-0.5 rounded text-[10px] font-data border transition-colors ${
              showEdgeLabels
                ? "text-gb-bright-blue border-gb-bright-blue bg-gb-bright-blue-dim"
                : "text-gb-fg4 border-gb-bg2"
            }`}
            title="Show relationship-type labels on edges (only at higher zoom)"
          >
            Edge labels
          </button>
          <button
            onClick={toggleGravity}
            className={`px-2 py-0.5 rounded text-[10px] font-data border transition-colors ${
              gravity
                ? "text-gb-bright-blue border-gb-bright-blue bg-gb-bright-blue-dim"
                : "text-gb-fg4 border-gb-bg2"
            }`}
          >
            Gravity
          </button>
        </div>
      </div>

      {/* Canvas + overlays */}
      <div className="flex-1 relative overflow-hidden bg-gb-bg0-h">
        <canvas
          ref={canvasRef}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseLeave}
          onClick={handleClick}
          className="w-full h-full cursor-grab active:cursor-grabbing"
        />

        {/* Path-trace banner — pinned top-center while a trace is
            active. Renders nothing in the common case. Discoverability:
            when a node is selected and trace is inactive, surface a
            short hint about shift-clicking. */}
        {pathInfo.start && pathInfo.end && (
          <div className="absolute top-2 left-1/2 -translate-x-1/2 z-30 flex items-center gap-2 px-3 py-1.5 rounded border border-gb-bright-orange bg-gb-bg0/95 shadow-lg">
            <StixNodeIcon type={pathInfo.start.type} size={16} />
            <span className="font-data text-[11px] text-gb-fg1 max-w-[180px] truncate" title={pathInfo.start.name}>
              {pathInfo.start.name}
            </span>
            <span className="font-data text-[11px] text-gb-bright-orange">→</span>
            <StixNodeIcon type={pathInfo.end.type} size={16} />
            <span className="font-data text-[11px] text-gb-fg1 max-w-[180px] truncate" title={pathInfo.end.name}>
              {pathInfo.end.name}
            </span>
            <span className="font-data text-[10px] text-gb-fg4 ml-1">
              {pathInfo.count === 0
                ? "no path"
                : `${pathInfo.count} path${pathInfo.count === 1 ? "" : "s"}`}
            </span>
            <button
              type="button"
              onClick={clearPathTrace}
              className="font-data text-[10px] text-gb-fg4 hover:text-gb-fg1 ml-1"
              title="Clear path trace"
            >
              ✕
            </button>
          </div>
        )}
        {!pathInfo.start && selectedNode && (
          <div className="absolute top-2 left-1/2 -translate-x-1/2 z-30 px-3 py-1 rounded border border-gb-bg2 bg-gb-bg0/90 font-data text-[10px] text-gb-fg4">
            Shift-click another node to trace paths from <span className="text-gb-fg2">{selectedNode.name}</span>.
          </div>
        )}

        {/* Hover tooltip — shown while not dragging/panning. */}
        <NodeHoverTooltip
          node={hoverNode && {
            type: hoverNode.type,
            name: hoverNode.name || hoverNode.label || "",
            id: hoverNode.id,
            mitreId: (hoverNode.rawObj?.external_references || [])
              .find((r) => r.source_name === "mitre-attack")?.external_id,
          }}
          x={hoverPos.x}
          y={hoverPos.y}
        />

        {/* Object detail panel (right side drawer) */}
        {selectedNode && (
          <div className="absolute top-0 right-0 w-[380px] h-full bg-gb-bg0/95 backdrop-blur-md border-l border-gb-bg2 shadow-2xl z-50 flex flex-col overflow-hidden">
            {/* Detail header */}
            <div className="px-4 pt-3 pb-2 border-b border-gb-bg2 shrink-0">
              <div className="flex items-start justify-between">
                <div className="flex-1 min-w-0 pr-2">
                  <div className="font-semibold text-[14px] truncate" style={{ color: getTypeConfig(selectedNode.type).color }}>
                    {selectedNode.name}
                  </div>
                  <div className="font-data text-[9px] text-gb-bg4 mt-0.5 break-all">
                    {selectedNode.type} | {selectedNode.id}
                  </div>
                </div>
                <button
                  onClick={() => { stateRef.current.selectedNode = null; setSelectedNode(null); draw(); }}
                  className="text-gb-bg4 hover:text-gb-fg1 text-[16px] leading-none shrink-0 mt-0.5"
                >
                  &#x2715;
                </button>
              </div>
            </div>

            {/* Detail body */}
            <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
              {/* Description */}
              {selectedNode.desc && (
                <div>
                  <div className="text-[9px] font-semibold text-gb-fg4 uppercase tracking-wider mb-1">Description</div>
                  <div className="text-gb-fg2 text-[11px] leading-relaxed">{selectedNode.desc}</div>
                </div>
              )}

              {/* Command lines */}
              {selectedNode.cmdlines.length > 0 && (
                <div>
                  <div className="text-[9px] font-semibold text-gb-fg4 uppercase tracking-wider mb-1">Command Lines</div>
                  {selectedNode.cmdlines.map((cmd, i) => (
                    <pre key={i} className="bg-black/30 border border-gb-bg2 rounded px-2 py-1.5 font-data text-[9.5px] text-gb-bright-aqua whitespace-pre-wrap break-all mt-1">
                      {cmd}
                    </pre>
                  ))}
                </div>
              )}

              {/* All STIX fields */}
              {selectedNode.rawObj && (
                <div>
                  <div className="text-[9px] font-semibold text-gb-fg4 uppercase tracking-wider mb-1">All Fields</div>
                  <div className="space-y-1.5">
                    {Object.entries(selectedNode.rawObj)
                      .filter(([k]) => !DETAIL_SKIP_FIELDS.has(k))
                      .map(([key, val]) => (
                        <div key={key} className="text-[11px]">
                          <span className="font-data text-gb-bright-blue text-[10px]">{key}</span>
                          <div className="text-gb-fg2 mt-0.5 ml-2">
                            {renderFieldValue(val)}
                          </div>
                        </div>
                      ))
                    }
                  </div>
                </div>
              )}

              {/* Connected edges */}
              {(() => {
                const s = stateRef.current;
                const incoming = s.links.filter((l) => l.target.id === selectedNode.id);
                const outgoing = s.links.filter((l) => l.source.id === selectedNode.id);
                if (!incoming.length && !outgoing.length) return null;
                return (
                  <div>
                    <div className="text-[9px] font-semibold text-gb-fg4 uppercase tracking-wider mb-1">
                      Relationships ({incoming.length + outgoing.length})
                    </div>
                    <div className="space-y-0.5 text-[10px] font-data">
                      {outgoing.map((l, i) => (
                        <div key={`o${i}`} className="text-gb-fg3 truncate">
                          <span className="text-gb-bright-orange">&#8594;</span>{" "}
                          <span className="text-gb-fg4">{l.type}</span>
                          {l.dashed && <span className="text-gb-bg4 ml-0.5" title="embedded ref">(ref)</span>}
                          {" "}<span className="text-gb-fg2">{l.target.name || l.target.label}</span>
                        </div>
                      ))}
                      {incoming.map((l, i) => (
                        <div key={`i${i}`} className="text-gb-fg3 truncate">
                          <span className="text-gb-bright-aqua">&#8592;</span>{" "}
                          <span className="text-gb-fg4">{l.type}</span>
                          {l.dashed && <span className="text-gb-bg4 ml-0.5" title="embedded ref">(ref)</span>}
                          {" "}<span className="text-gb-fg2">{l.source.name || l.source.label}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })()}
            </div>
          </div>
        )}

        {/* Legend. Collapsible: on a bundle with a full detection layer it
            runs to ~18 type rows and covers the lower-left of the graph,
            which is exactly where a force layout tends to push the long
            tail of leaf nodes. Collapsed it is a single title bar. */}
        <div className="absolute bottom-3 left-3 bg-gb-bg0/90 backdrop-blur-md border border-gb-bg2 rounded-lg font-data text-[10px] min-w-[140px] max-w-[220px]">
          <button
            type="button"
            onClick={() => setLegendOpen((v) => !v)}
            className="w-full flex items-center gap-1.5 px-2.5 py-1.5 text-left hover:bg-gb-bg1/50 rounded-lg"
            aria-expanded={legendOpen}
            title={legendOpen ? "Hide legend" : "Show legend"}
          >
            <span className="text-gb-fg4 w-2 shrink-0">{legendOpen ? "\u25be" : "\u25b8"}</span>
            <span className="font-bold text-gb-fg2 text-[10px]">STIX Types</span>
          </button>
          {legendOpen && (
            <div className="px-2.5 pb-2.5 max-h-[46vh] overflow-y-auto">
              {buildLegendItems(stateRef.current.nodes)}
            </div>
          )}
        </div>

        {/* Stats */}
        <div className="absolute bottom-3 right-3 bg-gb-bg0/90 backdrop-blur-md border border-gb-bg2 rounded-lg px-3 py-2 font-data text-[10px] text-gb-fg4">
          <span className="text-gb-gray">Visible:</span>{" "}
          <span className="text-gb-fg2 font-semibold">{nodeCount}</span> /{" "}
          <span className="text-gb-gray">{totalCount}</span> &middot;{" "}
          <span className="text-gb-gray">Edges:</span>{" "}
          <span className="text-gb-fg2 font-semibold">{edgeCount}</span>
        </div>
      </div>
    </div>
  );
}


/** Render a STIX field value for the detail panel. */
function renderFieldValue(val) {
  if (val === null || val === undefined) {
    return <span className="text-gb-bg4 italic">null</span>;
  }
  if (typeof val === "boolean") {
    return <span className="text-gb-bright-purple">{val ? "true" : "false"}</span>;
  }
  if (typeof val === "number") {
    return <span className="text-gb-bright-yellow">{val}</span>;
  }
  if (typeof val === "string") {
    // STIX refs are clickable-looking
    if (val.includes("--")) {
      return <span className="text-gb-bright-aqua font-data text-[10px] break-all">{val}</span>;
    }
    if (val.length > 200) {
      return <span className="text-gb-fg3 break-all">{val.slice(0, 200)}...</span>;
    }
    return <span className="text-gb-fg3 break-all">{val}</span>;
  }
  if (Array.isArray(val)) {
    if (val.length === 0) return <span className="text-gb-bg4 italic">[]</span>;
    // Short string arrays inline
    if (val.length <= 5 && val.every((v) => typeof v === "string" && v.length < 60)) {
      return (
        <div className="space-y-0.5">
          {val.map((item, i) => (
            <div key={i} className="text-gb-fg3 font-data text-[10px] break-all pl-1 border-l border-gb-bg2">
              {typeof item === "string" && item.includes("--")
                ? <span className="text-gb-bright-aqua">{item}</span>
                : String(item)}
            </div>
          ))}
        </div>
      );
    }
    // Complex arrays as JSON
    return (
      <pre className="bg-black/20 border border-gb-bg2 rounded px-2 py-1 text-[9px] text-gb-fg3 whitespace-pre-wrap break-all max-h-[200px] overflow-y-auto font-data">
        {JSON.stringify(val, null, 2)}
      </pre>
    );
  }
  if (typeof val === "object") {
    return (
      <pre className="bg-black/20 border border-gb-bg2 rounded px-2 py-1 text-[9px] text-gb-fg3 whitespace-pre-wrap break-all max-h-[200px] overflow-y-auto font-data">
        {JSON.stringify(val, null, 2)}
      </pre>
    );
  }
  return <span className="text-gb-fg3">{String(val)}</span>;
}


/** Build legend JSX from current (visible) nodes. */
function buildLegendItems(nodes) {
  const counts = {};
  for (const n of nodes) counts[n.type] = (counts[n.type] || 0) + 1;

  const groups = {};
  for (const t of Object.keys(counts)) {
    const g = getTypeConfig(t).group || "Other";
    (groups[g] || (groups[g] = [])).push(t);
  }

  const ORDER = ["Threat Layer", "Detection Layer", "Attack Flow", "Observables", "Infrastructure", "Other"];
  const items = [];

  for (const gName of ORDER) {
    if (!groups[gName]) continue;
    items.push(
      <div key={`g-${gName}`} className="text-gb-fg4 text-[9px] font-semibold uppercase tracking-wider mt-1.5 first:mt-0">
        {gName}
      </div>
    );
    for (const t of groups[gName].sort()) {
      const label = t.replace("x-mitre-", "").replace("x-", "").replace("attack-", "");
      items.push(
        <div key={t} className="flex items-center gap-1.5 py-0.5 text-gb-fg3">
          <StixNodeIcon type={t} size={14} title={t} className="shrink-0" />
          {label} ({counts[t]})
        </div>
      );
    }
  }
  return items;
}
