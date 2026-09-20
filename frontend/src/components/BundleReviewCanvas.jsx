/**
 * BundleReviewCanvas — Gate 2 (Bundle Review) editable graph.
 *
 * Renders the relationship_preview + drafts + validated_entities as a
 * React Flow graph. Visual styling matches the read-only Explorer
 * (BundleGraph.jsx) by importing the shared type/edge/layer constants
 * from `lib/bundleGraphConstants.js`. Edit affordances mirror the
 * ChunkReviewCanvas pattern (drag-to-add, click-edge + Backspace to
 * delete, side-panel inline edit).
 *
 * Mutations are tracked locally and converted into Gate2ReviewItem[]
 * on submit. The backend's existing per-relationship handler in
 * gate_2 already supports approve / edit / remove / added (via
 * rel_id startsWith "added_"), so no backend schema change is needed.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ReactFlow, Background, Controls, MiniMap,
  Handle, Position,
  ReactFlowProvider, useReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";

import {
  EDGE_COLORS,
  FILTER_LAYER_NAMES,
  LAYER_COLORS,
  SKIP_TYPES,
  getTypeConfig,
  operatorKindColor,
  layerForType,
} from "../lib/bundleGraphConstants";
import StixNodeIcon from "./StixNodeIcon";
import NodeSearchBox from "./NodeSearchBox";
import NodeHoverTooltip from "./NodeHoverTooltip";
import ReviewerBrief from "./ReviewerBrief";
import InfoDot from "./InfoDot";
import { hint } from "../lib/glossary";
import { newId } from "../lib/ids";
import { ellipsize } from "../lib/strings";
import { applyPositions, useLayoutPositions } from "../lib/graphLayout";
import SuggestionChip from "./SuggestionChip";
import { isBulkAcceptable } from "../lib/reviewerSuggestions";
import {
  buildAdjacency,
  findAllShortestPaths,
  collectPathElements,
  pathEdgeKey,
} from "../lib/findPaths";

const NODE_WIDTH = 220;
const NODE_HEIGHT = 70;
// 80/120 leaves enough room for ~220px-wide labels without overlap at
// typical node densities; margins keep edge labels off the canvas edge.
const BUNDLE_LAYOUT = { rankdir: "TB", nodesep: 80, ranksep: 120, marginx: 30, marginy: 30 };

// Common STIX relationship types the canvas's drag-to-add picker
// shows by default. Backend validator (Gate2ReviewItem.validate_rel_type)
// is the authoritative whitelist; this is just the convenient shortlist.
const COMMON_REL_TYPES = [
  "uses", "targets", "precedes", "indicates",
  "has-observable", "attributed-to", "mitigates", "detects",
  "exploits", "component-of",
];


// ── Helpers ──────────────────────────────────────────────────────────

/**
 * Stable id for a graph node derived from name+type. Same name+type
 * across multiple relationship_preview entries → one node. We avoid
 * relying on the upstream STIX UUIDs because the gate_2 preview only
 * carries names; UUIDs come later at serialization time.
 */
function nodeKeyFor(name, type) {
  return `${(type || "").toLowerCase()}::${(name || "").toLowerCase()}`;
}


/**
 * React Flow custom node that renders a colored chip per STIX type.
 * Color/shape derived from the shared lib so the visual treatment
 * matches Explorer (BundleGraph.jsx). isOutOfFocus dims the node when
 * focus mode or a path trace is active; layer-hidden nodes are not
 * rendered at all.
 */
function BundleNode({ data }) {
  const { name, type, isSelected, isOutOfFocus, isAdded, operator, conditionDescription } = data;
  const config = getTypeConfig(type) || { color: "#88c0d0" };
  // attack-operator nodes override the type color with a kind-specific
  // palette (AND=teal, OR=amber, XOR=red) so the analyst can spot
  // disjunctive / mutually-exclusive flows at a glance.
  const baseColor = type === "attack-operator" && operator
    ? operatorKindColor(operator)
    : config.color;
  // Visual state stack: hidden > out-of-focus > selected > added > default.
  let outerClasses = "border";
  let style = {
    background: `${baseColor}1a`, // 10% opacity bg
    borderColor: `${baseColor}80`,
  };
  if (isOutOfFocus) {
    outerClasses += " opacity-30";
  }
  if (isSelected) {
    outerClasses += " ring-2 ring-gb-bright-blue";
    style.borderColor = "#83a598";
  } else if (isAdded) {
    outerClasses += " ring-1 ring-gb-bright-green";
    style.borderColor = "#b8bb26";
  }
  return (
    <div
      className={`rounded px-2.5 py-1.5 ${outerClasses}`}
      style={{ ...style, width: NODE_WIDTH, minHeight: NODE_HEIGHT }}
    >
      <Handle type="target" position={Position.Top} style={{ background: baseColor }} />
      <div className="flex items-start gap-2">
        <StixNodeIcon type={type} size={22} title={type} className="shrink-0 mt-0.5" />
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <span className="text-[10px] font-data text-gb-fg4 uppercase tracking-wide">
              {type}
            </span>
            {type === "attack-operator" && operator && (
              <span
                className="text-[10px] font-data uppercase tracking-wide px-1 rounded"
                style={{ background: `${baseColor}33`, color: baseColor }}
              >
                {operator}
              </span>
            )}
            {isAdded && (
              <span className="ml-auto text-[9px] font-data text-gb-bright-green uppercase">
                added
              </span>
            )}
          </div>
          <div
            className="font-data text-[12px] text-gb-fg1 leading-tight break-words"
            title={name}
          >
            {ellipsize(name, 80)}
          </div>
          {type === "attack-condition" && conditionDescription && (
            <div
              className="text-[10px] italic text-gb-fg3 leading-tight mt-0.5"
              title={conditionDescription}
            >
              {ellipsize(conditionDescription, 60)}
            </div>
          )}
        </div>
      </div>
      <Handle type="source" position={Position.Bottom} style={{ background: baseColor }} />
    </div>
  );
}

const NODE_TYPES = { bundle: BundleNode };

// Hoisted to module scope. React Flow
// 12 warns and may invalidate edge memoization when these are recreated per
// render.
const FIT_VIEW_OPTIONS = { padding: 0.15, minZoom: 0.45, maxZoom: 1 };


/**
 * Build the React Flow graph (nodes + edges) from a Gate 2 payload.
 *
 * Inputs:
 *   payload.items                        — relationship_preview list
 *   payload.context.drafts               — procedure drafts
 *   payload.context.validated_entities   — entities surviving Gate 0
 *
 * Output: { nodes, edges } where each node carries display metadata
 * the layer filter and edit handlers consume; each edge maps 1:1 to
 * a relationship_preview entry plus any analyst-added rel.
 */
function parseGate2Payload({ items, context, addedRels, removedRelIds, editedRels }) {
  const nodeMap = new Map(); // nodeKey → { name, type, key }

  const drafts = context.drafts ?? [];
  const entities = context.validated_entities ?? [];

  // Pass 1: nodes from drafts (procedures) and entities. We only emit
  // nodes that participate in a visible edge later, but we register
  // every candidate so the relationship pass can resolve names → keys.
  for (const d of drafts) {
    const name = d.name || d.draft_id || "unnamed-procedure";
    const key = nodeKeyFor(name, "x-procedure");
    if (!nodeMap.has(key)) {
      nodeMap.set(key, { key, name, type: "x-procedure" });
    }
  }
  for (const e of entities) {
    if (e.gate_action === "remove") continue;
    const name = e.edited_value || e.value || "";
    if (!name) continue;
    const stixType = entityTypeToStixType(e.edited_type || e.entity_type);
    if (!stixType || SKIP_TYPES.has(stixType)) continue;
    const key = nodeKeyFor(name, stixType);
    if (!nodeMap.has(key)) {
      nodeMap.set(key, { key, name, type: stixType });
    }
  }

  // Pass 2: edges from relationship_preview, filtered by removed.
  const edges = [];
  for (const rel of items ?? []) {
    if (removedRelIds.has(rel.id)) continue;
    const editedFor = editedRels[rel.id]; // may override fields
    const relType = editedFor?.edited_rel_type ?? rel.relationship_type ?? "uses";
    const { srcName, tgtName, srcType, tgtType, srcKey, tgtKey } =
      relEndpointKeys(rel, editedRels, drafts);
    if (!nodeMap.has(srcKey)) nodeMap.set(srcKey, { key: srcKey, name: srcName, type: srcType });
    if (!nodeMap.has(tgtKey)) nodeMap.set(tgtKey, { key: tgtKey, name: tgtName, type: tgtType });
    edges.push({
      id: rel.id,
      source: srcKey,
      target: tgtKey,
      relType,
      reviewable: rel.reviewable !== false,
      isAdded: false,
      isEdited: !!editedFor,
    });
  }

  // Pass 3: analyst-added rels (front-end only until submit).
  for (const a of addedRels) {
    const srcKey = nodeKeyFor(a.source_name, a.source_type);
    const tgtKey = nodeKeyFor(a.target_name, a.target_type);
    if (!nodeMap.has(srcKey)) nodeMap.set(srcKey, { key: srcKey, name: a.source_name, type: a.source_type });
    if (!nodeMap.has(tgtKey)) nodeMap.set(tgtKey, { key: tgtKey, name: a.target_name, type: a.target_type });
    edges.push({
      id: a.id,
      source: srcKey,
      target: tgtKey,
      relType: a.relationship_type,
      reviewable: true,
      isAdded: true,
      isEdited: false,
    });
  }

  return { nodeMap, edges };
}


function entityTypeToStixType(et) {
  // Map our internal EntityType values to canonical STIX types so the
  // visual config table matches.
  const m = {
    intrusion_set: "intrusion-set",
    threat_actor: "threat-actor",
    malware: "malware",
    tool: "tool",
    campaign: "campaign",
    vulnerability: "vulnerability",
    organization: "identity",
    location: "location",
    victim_sector: "identity",
    infrastructure: "infrastructure",
    ioc_hash: "file",
    ioc_ip: "ipv4-addr",
    ioc_domain: "domain-name",
    ioc_url: "url",
    ioc_email: "email-addr",
    ioc_file_path: "file",
    ioc_registry_key: "windows-registry-key",
    ioc_mutex: "mutex",
    ioc_command_line: "process",
    ioc_process_name: "process",
    software: "software",
    user_account: "user-account",
  };
  return m[et] || et;
}


/** A relationship's endpoint names, types and node keys, with any analyst
 *  edit applied. The ONE place this is computed: the canvas, the AI-review
 *  filter and the focus buttons all have to agree on what a node's key is,
 *  and a raw name compared against a `type::name` key never matches. */
function relEndpointKeys(rel, editedRels, drafts) {
  const editedFor = editedRels?.[rel.id];
  const srcName = editedFor?.edited_source ?? rel.source_name ?? "";
  const tgtName = editedFor?.edited_target ?? rel.target_name ?? "";
  const srcType = rel.source_type || guessTypeFromName(srcName, drafts) || "identity";
  const tgtType = rel.target_type || guessTypeFromName(tgtName, drafts) || "identity";
  return {
    srcName, tgtName, srcType, tgtType,
    srcKey: nodeKeyFor(srcName, srcType),
    tgtKey: nodeKeyFor(tgtName, tgtType),
  };
}

function guessTypeFromName(name, drafts) {
  if (!name) return null;
  // If it matches a draft name, it's an x-procedure.
  if (drafts.some((d) => d.name === name)) return "x-procedure";
  // CVE-* → vulnerability
  if (/^CVE-\d{4}-\d+/i.test(name)) return "vulnerability";
  return null;
}


/** Dagre layout. The rankdir, nodesep, and ranksep defaults are tuned
 * for real-source bundles (~50 nodes, hub-and-spoke topology) — the
 * earlier tighter spacing crushed labels into an unreadable horizontal
 * band on one campaign-report dry run. */


function buildReactFlowGraph({
  nodeMap,
  edges,
  selectedNodeKey,
  selectedEdgeId,
  layerActive,
  focusKey,
  focusNeighbors,
  tab, // "flow" | "bundle"
  showEdgeLabels, // explicit toggle; off by default in dense Bundle tab
  pathNodes,      // Set<string> | null — when set, only these dim-in
  pathEdges,      // Set<string> | null — when set, only these edges dim-in
}) {
  // Path-trace mode supersedes the 1-hop focus dimming. When active,
  // we test membership in the path sets instead of focusNeighbors.
  const traceActive = !!(pathNodes && pathNodes.size > 0);
  // Flow tab: only x-procedure nodes + precedes edges. Strips the
  // bundle to its kill-chain spine — reads as a clean linear or
  // branching sequence. Click-to-focus still works for selecting
  // the procedure to drill into.
  const isFlowTab = tab === "flow";

  // Compute layer-hidden state. Layer-hidden nodes are EXCLUDED from
  // the dagre layout entirely (not just dimmed) — including them
  // crushed real-scale bundles into an unreadable horizontal band.
  // Untyped / unrecognized types stay visible so coverage gaps don't
  // become invisible. In Flow tab, layer toggles are bypassed: we
  // unconditionally hide everything that isn't an x-procedure.
  const layerHidden = new Set();
  for (const n of nodeMap.values()) {
    if (isFlowTab) {
      if (n.type !== "x-procedure") layerHidden.add(n.key);
    } else {
      const layer = layerForType(n.type);
      if (layer && !layerActive[layer]) layerHidden.add(n.key);
    }
  }

  const rfNodes = [];
  for (const n of nodeMap.values()) {
    if (layerHidden.has(n.key)) continue; // skip from layout entirely
    const isOutOfFocus = traceActive
      ? !pathNodes.has(n.key)
      : focusKey && !focusNeighbors.has(n.key);
    rfNodes.push({
      id: n.key,
      type: "bundle",
      data: {
        name: n.name,
        type: n.type,
        isSelected: n.key === selectedNodeKey,
        isOutOfFocus,
        isAdded: !!n.isAdded,
      },
      position: { x: 0, y: 0 },
      width: NODE_WIDTH,
      height: NODE_HEIGHT,
    });
  }

  const rfEdges = [];
  // Count edges hidden by the layer filter and (in Flow tab) the
  // precedes-only filter. Surfaced in the stats row as a usability hint
  // so analysts don't think "0 edges" means the relationship_preview is
  // broken when in fact a layer toggle is hiding everything.
  let hiddenByLayer = 0;
  let hiddenByFlowFilter = 0;
  for (const e of edges) {
    if (layerHidden.has(e.source) || layerHidden.has(e.target)) {
      hiddenByLayer += 1;
      continue;
    }
    if (isFlowTab && e.relType !== "precedes") {
      hiddenByFlowFilter += 1;
      continue;
    }
    const color = EDGE_COLORS[e.relType] || "rgba(76,86,106,0.5)";
    const isSelected = e.id === selectedEdgeId;
    // Path-trace edges dim by membership in pathEdges, keyed the same way
    // collectPathElements keys them — a hand-rolled copy of that format
    // would dim every edge the day the lib's changed. Outside trace mode,
    // 1-hop focus dims edges where either endpoint sits outside the
    // focused neighborhood.
    const edgeKey = pathEdgeKey(e.source, e.target);
    const isOutOfFocus = traceActive
      ? !pathEdges.has(edgeKey)
      : focusKey && !(focusNeighbors.has(e.source) && focusNeighbors.has(e.target));
    rfEdges.push({
      id: e.id,
      source: e.source,
      target: e.target,
      // Label visibility is gated by the showEdgeLabels toggle. Selected
      // edges always render the label so the analyst can confirm what
      // they clicked, even when global labels are off.
      label: (showEdgeLabels || isSelected) ? e.relType : undefined,
      labelStyle: { fill: "#928374", fontSize: 9 },
      labelBgStyle: { fill: "#1d2021", fillOpacity: 0.85 },
      style: {
        stroke: e.isAdded ? "#b8bb26" : color.replace(/,\s*0\.\d+\)/, ",0.85)"),
        strokeWidth: isSelected ? 2.5 : (e.isAdded ? 1.8 : 1.3),
        strokeDasharray: e.isAdded ? "4 3" : undefined,
        opacity: isOutOfFocus ? 0.15 : 1,
      },
      data: { relType: e.relType, isAdded: e.isAdded, isEdited: e.isEdited },
    });
  }

  // Unpositioned; the component overlays cached layout positions.
  return {
    nodes: rfNodes,
    edges: rfEdges,
    hiddenByLayer,
    hiddenByFlowFilter,
    totalDataEdges: edges.length,
  };
}


function neighborKeysFor(focusKey, edges) {
  const set = new Set([focusKey]);
  for (const e of edges) {
    if (e.source === focusKey) set.add(e.target);
    if (e.target === focusKey) set.add(e.source);
  }
  return set;
}


// ── Add-edge form (in side panel) ────────────────────────────────────

function AddEdgeForm({ pendingAdd, onConfirm, onCancel }) {
  const [relType, setRelType] = useState("uses");
  return (
    <div className="rounded border border-gb-aqua/40 bg-gb-aqua/5 p-2.5">
      <p className="text-[10px] font-data font-semibold text-gb-bright-aqua uppercase tracking-wider mb-1.5">
        Add relationship
      </p>
      <div className="text-[11px] font-data text-gb-fg2 mb-2">
        <div><span className="text-gb-fg4">source: </span>{pendingAdd.source_name} <span className="text-gb-fg4 ml-1">({pendingAdd.source_type})</span></div>
        <div><span className="text-gb-fg4">target: </span>{pendingAdd.target_name} <span className="text-gb-fg4 ml-1">({pendingAdd.target_type})</span></div>
      </div>
      <label className="text-[10px] font-data text-gb-fg4 uppercase tracking-wider">
          type<InfoDot term="relationship-type" />
        </label>
      <select
        value={relType}
        onChange={(e) => setRelType(e.target.value)}
        className="w-full mt-1 mb-2 bg-gb-bg1 border border-gb-bg2 rounded px-2 py-1 text-[11px] font-data text-gb-fg1"
      >
        {COMMON_REL_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
      </select>
      <div className="flex gap-1.5">
        <button
          type="button"
          onClick={() => onConfirm(relType)}
          className="px-2 py-1 rounded bg-gb-bright-aqua text-gb-bg0 text-[11px] font-data font-semibold hover:bg-gb-aqua"
        >
          Add
        </button>
        <button
          type="button"
          onClick={onCancel}
          className="px-2 py-1 rounded bg-gb-bg1 text-gb-fg2 text-[11px] font-data hover:bg-gb-bg2 border border-gb-bg2"
        >
          Cancel
        </button>
      </div>
    </div>
  );
}


// ── Side panel ──────────────────────────────────────────────────────

function SidePanel({
  selectedNode,
  selectedEdge,
  pendingAdd,
  onEditEdgeRelType,
  onRemoveEdge,
  onUndoAdded,
  onConfirmAdd,
  onCancelAdd,
}) {
  if (pendingAdd) {
    return <AddEdgeForm pendingAdd={pendingAdd} onConfirm={onConfirmAdd} onCancel={onCancelAdd} />;
  }
  if (selectedEdge) {
    return (
      <div className="rounded border border-gb-bg2 bg-gb-bg0 p-2.5">
        <p className="text-[10px] font-data font-semibold text-gb-fg4 uppercase tracking-wider mb-2">
          Relationship
        </p>
        <div className="text-[11px] font-data text-gb-fg2 mb-3 space-y-0.5">
          <div><span className="text-gb-fg4">id: </span><span className="text-gb-fg2">{selectedEdge.id}</span></div>
          <div><span className="text-gb-fg4">source: </span>{selectedEdge.sourceName}</div>
          <div><span className="text-gb-fg4">target: </span>{selectedEdge.targetName}</div>
        </div>
        <label className="text-[10px] font-data text-gb-fg4 uppercase tracking-wider">
          type<InfoDot term="relationship-type" />
        </label>
        <select
          value={selectedEdge.relType}
          onChange={(e) => onEditEdgeRelType(selectedEdge.id, e.target.value)}
          className="w-full mt-1 mb-3 bg-gb-bg1 border border-gb-bg2 rounded px-2 py-1 text-[11px] font-data text-gb-fg1"
        >
          {COMMON_REL_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
        </select>
        <div className="flex gap-1.5">
          {selectedEdge.isAdded ? (
            <button
              type="button"
              onClick={() => onUndoAdded(selectedEdge.id)}
              className="px-2 py-1 rounded bg-gb-bg1 text-gb-bright-yellow text-[11px] font-data hover:bg-gb-bg2 border border-gb-yellow"
            >
              ↺ Undo add
            </button>
          ) : (
            <button
              type="button"
              onClick={() => onRemoveEdge(selectedEdge.id)}
              className="px-2 py-1 rounded bg-gb-bg1 text-gb-bright-red text-[11px] font-data hover:bg-gb-bg2 border border-gb-red"
            >
              ✕ Remove
            </button>
          )}
        </div>
        {selectedEdge.isEdited && (
          <p className="mt-2 text-[10px] font-data text-gb-bright-yellow">
            (edited — submit will record the new type)
          </p>
        )}
      </div>
    );
  }
  if (selectedNode) {
    return (
      <div className="rounded border border-gb-bg2 bg-gb-bg0 p-2.5">
        <p className="text-[10px] font-data font-semibold text-gb-fg4 uppercase tracking-wider mb-2">
          {selectedNode.type}
        </p>
        <p className="text-[12px] font-data text-gb-fg1 break-words">
          {selectedNode.name}
        </p>
        <p className="mt-2 text-[10px] font-data text-gb-fg4">
          Drag from this node&apos;s bottom handle to another node to create a new relationship.
        </p>
      </div>
    );
  }
  return (
    <div className="rounded border border-gb-bg1 bg-gb-bg0/40 p-2.5">
      <p className="text-[10px] text-gb-fg4 leading-snug">
        Click an edge to inspect or edit. Drag from a node&apos;s bottom handle to another node to create a relationship. Press Backspace on a selected edge to remove it.
      </p>
    </div>
  );
}


// ── Submit-payload builder ──────────────────────────────────────────

/** Convert local mutation state into the Gate2Submit body. The backend's
 * existing per-relationship handler (gate_2 node) consumes this shape:
 *   reviews: [{rel_id, action, edited_rel_type?, edited_source?, edited_target?, rationale?}]
 * Analyst-added rels emit rel_id starting with "added_" — that prefix
 * is the existing handler's signal to treat them as additions.
 */
function buildSubmitBody({ items, removedRelIds, editedRels, addedRels }) {
  const reviews = [];
  // Existing rels: emit only when changed (remove or edit). Approves are
  // implicit — gate_2 treats unmentioned rels as approved per the
  // existing batch-mode default.
  for (const rel of items ?? []) {
    if (removedRelIds.has(rel.id)) {
      reviews.push({ rel_id: rel.id, action: "remove" });
    } else if (editedRels[rel.id]) {
      const e = editedRels[rel.id];
      const review = { rel_id: rel.id, action: "edit" };
      if (e.edited_rel_type) review.edited_rel_type = e.edited_rel_type;
      if (e.edited_source) review.edited_source = e.edited_source;
      if (e.edited_target) review.edited_target = e.edited_target;
      reviews.push(review);
    }
  }
  // Analyst-added rels.
  for (const a of addedRels) {
    reviews.push({
      rel_id: a.id, // already starts with "added_"
      action: "approve",
      edited_rel_type: a.relationship_type,
      edited_source: a.source_name,
      edited_target: a.target_name,
      // The serializer resolves endpoints by (name, type); without the
      // type, a name shared by a malware and a tool node is ambiguous and
      // the row is skipped rather than guessed.
      edited_source_type: a.source_type,
      edited_target_type: a.target_type,
    });
  }
  return { reviews };
}


// ── Main component ──────────────────────────────────────────────────

function BundleReviewCanvasInner({
  payload,
  onSubmit,
  submitting,
  // AI reviewer additions. All optional — absent means this gate ran in
  // plain review mode and the canvas behaves exactly as it always has.
  recommendations = null,
  brief = null,
  onSaveBrief = null,
  savingBrief = false,
}) {
  const items = payload?.items ?? [];
  const context = payload?.context ?? {};

  // React Flow viewport control. Used by the search box to center on
  // a jumped-to node.
  const { setCenter, getNode } = useReactFlow();

  // Mutation state.
  const [removedRelIds, setRemovedRelIds] = useState(() => new Set());
  const [editedRels, setEditedRels] = useState({}); // rel_id → {edited_rel_type?, edited_source?, edited_target?}

  // Which AI recommendations the analyst has applied.
  const [appliedRecs, setAppliedRecs] = useState(() => new Set());
  // What the analyst had for a relationship BEFORE an AI recommendation was
  // applied, so undo restores it rather than wiping their own edit. The
  // other three gates keep the same snapshot (preApplyDecisions etc.).
  const [preApplyRels, setPreApplyRels] = useState({});

  /** rel_id -> recommendation. Empty when there is no AI review. */
  const recsByRel = useMemo(() => {
    const out = {};
    for (const rec of recommendations?.relationships ?? []) {
      if (rec?.rel_id) out[rec.rel_id] = rec;
    }
    return out;
  }, [recommendations]);

  const bulkAcceptableCount = useMemo(
    () => (recommendations?.relationships ?? []).filter(isBulkAcceptable).length,
    [recommendations],
  );
  const [addedRels, setAddedRels] = useState([]);   // [{id, source_name, source_type, target_name, target_type, relationship_type}]

  // UI state.
  const [selectedNodeKey, setSelectedNodeKey] = useState(null);
  const [selectedEdgeId, setSelectedEdgeId] = useState(null);
  const [pendingAdd, setPendingAdd] = useState(null); // {source_*, target_*}
  // Default the noisier layers OFF: Techniques & Observables alone can
  // contribute 50%+ of the nodes on a real source (every distinct
  // technique pick + observable), and Detection is rarely populated for
  // this pipeline. Starting with Procedures & Flow + Threat Context
  // gives the analyst a readable kill-chain view they can drill into
  // by toggling layers on. Without this default the dagre layout
  // crushes 50+ node bundles into an unreadable horizontal band.
  const [layerActive, setLayerActive] = useState(() => ({
    "Procedures & Flow": true,
    "Threat Context": true,
    "Techniques & Observables": false,
    "Detection": false,
  }));
  const [focusKey, setFocusKey] = useState(null);

  // Tab state. "flow" = procedure-only kill-chain view (LR layout, only
  // x-procedure nodes + precedes edges). "bundle" = full layered graph
  // with layer filters + click-to-focus. Default to flow because it's
  // the most-readable entry point on real-scale bundles; analyst clicks
  // a procedure in flow → drills into bundle focused on that procedure.
  const [tab, setTab] = useState("flow");

  // Edge label visibility. Off by default in Bundle tab (87+ edges in a
  // typical real-scale bundle = wall of labels). Selected edges still
  // show their label regardless of this toggle.
  const [showEdgeLabels, setShowEdgeLabels] = useState(false);

  // Hover tooltip state: { type, name, id } | null + cursor position
  // relative to the React Flow container.
  const [hoverNode, setHoverNode] = useState(null);
  const [hoverPos, setHoverPos] = useState({ x: 0, y: 0 });

  // Path-trace state. start/end are { id, name, type } so the banner
  // renders without re-deriving from nodeMap. nodes/edges are the
  // union of all shortest-path elements; render dimming uses these
  // instead of the 1-hop focusNeighbors when set.
  const [pathTrace, setPathTrace] = useState({
    start: null, end: null, nodes: null, edges: null, count: 0,
  });

  // Build node + edge model.
  const { nodeMap, edges: modelEdges } = useMemo(
    () => parseGate2Payload({ items, context, addedRels, removedRelIds, editedRels }),
    [items, context, addedRels, removedRelIds, editedRels],
  );

  // Ordered procedure list for Next/Prev cycling in Bundle tab.
  const procedureKeys = useMemo(() => {
    const drafts = context.drafts ?? [];
    return drafts
      .map((d) => nodeKeyFor(d.name || d.draft_id, "x-procedure"))
      .filter((k) => nodeMap.has(k));
  }, [context, nodeMap]);

  // When the user enters the Bundle tab, default focus to the first
  // procedure so the canvas isn't an overwhelming wall on landing.
  // Re-runs only when tab changes (not on every node update).
  useEffect(() => {
    if (tab === "bundle" && !focusKey && procedureKeys.length > 0) {
      setFocusKey(procedureKeys[0]);
    }
    if (tab === "flow") {
      // Flow tab: clear focus so the whole sequence is visible.
      setFocusKey(null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tab]);

  const focusNeighbors = useMemo(
    () => focusKey ? neighborKeysFor(focusKey, modelEdges) : new Set(),
    [focusKey, modelEdges],
  );

  // Build React Flow graph.
  const { nodes: unplacedNodes, edges, hiddenByLayer, hiddenByFlowFilter, totalDataEdges } = useMemo(
    () => buildReactFlowGraph({
      nodeMap, edges: modelEdges,
      selectedNodeKey, selectedEdgeId,
      layerActive, focusKey, focusNeighbors, tab,
      showEdgeLabels,
      pathNodes: pathTrace.nodes,
      pathEdges: pathTrace.edges,
    }),
    [nodeMap, modelEdges, selectedNodeKey, selectedEdgeId, layerActive, focusKey, focusNeighbors, tab, showEdgeLabels, pathTrace.nodes, pathTrace.edges],
  );
  const positions = useLayoutPositions(unplacedNodes, edges, BUNDLE_LAYOUT);
  const nodes = useMemo(() => applyPositions(unplacedNodes, positions), [unplacedNodes, positions]);

  // Jump-to-node target list. Drawn from the raw model nodeMap so the
  // search hits every node — including ones currently layer-hidden.
  // Jumping into a hidden layer auto-enables it (handled in jumpToNode).
  const searchableNodes = useMemo(() => {
    return Array.from(nodeMap.values()).map((n) => ({
      id: n.key,
      displayName: n.name || n.key,
      type: n.type,
      mitreId: null,
    }));
  }, [nodeMap]);

  const jumpToNode = useCallback(
    (matched) => {
      // If the target is on a hidden layer, flip that layer on so the
      // jump lands on a visible node rather than a phantom focus state.
      const layer = layerForType(matched.type);
      if (layer && !layerActive[layer]) {
        setLayerActive((m) => ({ ...m, [layer]: true }));
      }
      // Force the Bundle tab — Flow tab only renders x-procedures, so
      // jumping to any other type would land on an invisible node.
      if (tab === "flow" && matched.type !== "x-procedure") {
        setTab("bundle");
      }
      setSelectedNodeKey(matched.id);
      setSelectedEdgeId(null);
      // setCenter needs node coordinates. After the layer flip + tab
      // switch, the next React Flow render will reposition the node; we
      // defer the center call by a microtask so getNode returns the new
      // position rather than the pre-flip one.
      setTimeout(() => {
        const rfNode = getNode(matched.id);
        if (rfNode) {
          const x = rfNode.position.x + (rfNode.width || NODE_WIDTH) / 2;
          const y = rfNode.position.y + (rfNode.height || NODE_HEIGHT) / 2;
          setCenter(x, y, { zoom: 1.0, duration: 400 });
        }
      }, 50);
    },
    [layerActive, tab, setCenter, getNode],
  );

  // Procedure cycling — used in Bundle tab.
  const cycleProcedure = useCallback((dir) => {
    if (procedureKeys.length === 0) return;
    const idx = focusKey ? procedureKeys.indexOf(focusKey) : -1;
    const nextIdx = (idx + dir + procedureKeys.length) % procedureKeys.length;
    setFocusKey(procedureKeys[nextIdx]);
    setSelectedNodeKey(null);
    setSelectedEdgeId(null);
    setPendingAdd(null);
  }, [focusKey, procedureKeys]);

  // ── Handlers ──────────────────────────────────────────────────

  const handleNodeClick = useCallback((evt, node) => {
    // Shift-click activates path-trace mode: BFS from the previously-
    // selected node to the shift-clicked one, union the resulting
    // shortest-path nodes/edges, and dim everything else.
    if (evt.shiftKey && selectedNodeKey && selectedNodeKey !== node.id) {
      const adj = buildAdjacency(modelEdges);
      const paths = findAllShortestPaths(adj, selectedNodeKey, node.id);
      const { nodeSet, edgeSet } = collectPathElements(paths);
      const startNode = nodeMap.get(selectedNodeKey);
      const endNode = nodeMap.get(node.id);
      setPathTrace({
        start: startNode ? { id: startNode.key, name: startNode.name, type: startNode.type } : null,
        end: endNode ? { id: endNode.key, name: endNode.name, type: endNode.type } : null,
        nodes: nodeSet,
        edges: edgeSet,
        count: paths.length,
      });
      setSelectedEdgeId(null);
      setPendingAdd(null);
      // Don't change focusKey — path-trace state takes over dimming.
      return;
    }

    // Plain click: clear any active trace.
    if (pathTrace.start) {
      setPathTrace({ start: null, end: null, nodes: null, edges: null, count: 0 });
    }
    setSelectedNodeKey(node.id);
    setSelectedEdgeId(null);
    setPendingAdd(null);
    if (tab === "flow") {
      // In Flow tab: clicking a procedure jumps to Bundle tab focused
      // on it. The Flow view exists to navigate the kill chain;
      // editing happens in Bundle.
      setFocusKey(node.id);
      setTab("bundle");
    } else {
      // In Bundle tab: focus follows clicks. Re-clicking the focused
      // node cycles back to "no focus" so the analyst can briefly
      // see the full graph before re-focusing.
      setFocusKey((prev) => (prev === node.id ? null : node.id));
    }
  }, [tab, selectedNodeKey, modelEdges, nodeMap, pathTrace.start]);

  const clearPathTrace = useCallback(() => {
    setPathTrace({ start: null, end: null, nodes: null, edges: null, count: 0 });
  }, []);

  const handleEdgeClick = useCallback((_evt, edge) => {
    setSelectedEdgeId(edge.id);
    setSelectedNodeKey(null);
    setPendingAdd(null);
  }, []);

  const handlePaneClick = useCallback(() => {
    setSelectedNodeKey(null);
    setSelectedEdgeId(null);
    setPendingAdd(null);
    setFocusKey(null);
  }, []);

  const handleConnect = useCallback((conn) => {
    const src = nodeMap.get(conn.source);
    const tgt = nodeMap.get(conn.target);
    if (!src || !tgt) return;
    setPendingAdd({
      source_name: src.name,
      source_type: src.type,
      target_name: tgt.name,
      target_type: tgt.type,
    });
    setSelectedNodeKey(null);
    setSelectedEdgeId(null);
  }, [nodeMap]);

  const confirmAdd = useCallback((relType) => {
    if (!pendingAdd) return;
    const id = `added_${newId().slice(0, 12)}`;
    setAddedRels((prev) => [...prev, { id, ...pendingAdd, relationship_type: relType }]);
    setPendingAdd(null);
  }, [pendingAdd]);

  const cancelAdd = useCallback(() => setPendingAdd(null), []);

  const handleEdgesDelete = useCallback((deleted) => {
    // React Flow fires this when the user selects an edge and presses
    // Delete/Backspace. For added rels, undo the add; for original
    // rels, mark as removed.
    setAddedRels((prev) => prev.filter((a) => !deleted.some((d) => d.id === a.id)));
    setRemovedRelIds((prev) => {
      const next = new Set(prev);
      for (const d of deleted) {
        if (!d.id.startsWith("added_")) next.add(d.id);
      }
      return next;
    });
    setSelectedEdgeId(null);
  }, []);

  const editEdgeRelType = useCallback((relId, newType) => {
    if (relId.startsWith("added_")) {
      // Adjust the relationship_type on a newly-added rel.
      setAddedRels((prev) => prev.map((a) =>
        a.id === relId ? { ...a, relationship_type: newType } : a,
      ));
    } else {
      // Edit an existing rel.
      setEditedRels((prev) => ({
        ...prev,
        [relId]: { ...(prev[relId] ?? {}), edited_rel_type: newType },
      }));
    }
  }, []);

  const removeEdge = useCallback((relId) => {
    setRemovedRelIds((prev) => new Set(prev).add(relId));
    setSelectedEdgeId(null);
  }, []);

  /** Apply one AI relationship recommendation. */
  const handleApplyRec = useCallback((relId) => {
    const rec = recsByRel[relId];
    if (!rec) return;
    setPreApplyRels((prev) => (
      relId in prev ? prev : {
        ...prev,
        [relId]: { removed: removedRelIds.has(relId), edited: editedRels[relId] },
      }
    ));
    if (rec.action === "remove") {
      setRemovedRelIds((prev) => new Set(prev).add(relId));
    } else if (rec.action === "edit") {
      setEditedRels((prev) => ({
        ...prev,
        [relId]: {
          ...(prev[relId] ?? {}),
          ...(rec.edited_rel_type ? { edited_rel_type: rec.edited_rel_type } : {}),
          ...(rec.edited_source ? { edited_source: rec.edited_source } : {}),
          ...(rec.edited_target ? { edited_target: rec.edited_target } : {}),
        },
      }));
    }
    // approve needs no mutation — gate_2 treats an unmentioned relationship
    // as approved, so recording the click is the whole effect.
    setAppliedRecs((prev) => new Set(prev).add(relId));
  }, [recsByRel, removedRelIds, editedRels]);

  /** Undo an applied recommendation: back to what the analyst had. */
  const handleDismissRec = useCallback((relId) => {
    const before = preApplyRels[relId] ?? { removed: false, edited: undefined };
    setRemovedRelIds((prev) => {
      const next = new Set(prev);
      if (before.removed) next.add(relId); else next.delete(relId);
      return next;
    });
    setEditedRels((prev) => {
      const next = { ...prev };
      if (before.edited) next[relId] = before.edited; else delete next[relId];
      return next;
    });
    setPreApplyRels((prev) => {
      const next = { ...prev };
      delete next[relId];
      return next;
    });
    setAppliedRecs((prev) => {
      const next = new Set(prev);
      next.delete(relId);
      return next;
    });
  }, [preApplyRels]);

  /** Apply every HIGH-confidence recommendation. Medium and low are
   *  deliberately excluded — same rule as the other two gates. */
  const handleAcceptAI = useCallback(() => {
    (recommendations?.relationships ?? [])
      .filter(isBulkAcceptable)
      .forEach((rec) => handleApplyRec(rec.rel_id));
  }, [recommendations, handleApplyRec]);

  const undoAdded = useCallback((relId) => {
    setAddedRels((prev) => prev.filter((a) => a.id !== relId));
    setSelectedEdgeId(null);
  }, []);

  const toggleLayer = useCallback((name) => {
    setLayerActive((prev) => ({ ...prev, [name]: !prev[name] }));
  }, []);

  const handleSubmit = useCallback(() => {
    const body = buildSubmitBody({ items, removedRelIds, editedRels, addedRels });
    onSubmit(body);
  }, [items, removedRelIds, editedRels, addedRels, onSubmit]);

  // ── Selection lookup helpers for the side panel ─────────────

  const selectedNode = selectedNodeKey ? nodeMap.get(selectedNodeKey) : null;
  const selectedEdge = useMemo(() => {
    if (!selectedEdgeId) return null;
    const e = modelEdges.find((x) => x.id === selectedEdgeId);
    if (!e) return null;
    const src = nodeMap.get(e.source);
    const tgt = nodeMap.get(e.target);
    return {
      id: e.id,
      relType: e.relType,
      sourceName: src?.name ?? "?",
      targetName: tgt?.name ?? "?",
      isAdded: e.isAdded,
      isEdited: e.isEdited,
    };
  }, [selectedEdgeId, modelEdges, nodeMap]);

  // ── Stats ──────────────────────────────────────────────────

  const stats = {
    total: edges.length,
    added: addedRels.length,
    removed: removedRelIds.size,
    edited: Object.keys(editedRels).length,
    nodes: nodes.length,
  };
  const hasChanges = stats.added + stats.removed + stats.edited > 0;

  /** Recommendations for the focused procedure, or all when unfocused.
   *  Follows the canvas's own focus model rather than adding a second one —
   *  the panel is meant to answer "what does the AI think about what I am
   *  looking at". */
  const visibleRecs = useMemo(() => {
    const all = recommendations?.relationships ?? [];
    if (!focusKey) return all;
    const drafts = context?.drafts ?? [];
    const inFocus = new Set(
      (payload?.items ?? [])
        .filter((r) => {
          const { srcKey, tgtKey } = relEndpointKeys(r, editedRels, drafts);
          return srcKey === focusKey || tgtKey === focusKey;
        })
        .map((r) => r.id),
    );
    return all.filter((r) => inFocus.has(r.rel_id));
  }, [recommendations, focusKey, payload, context, editedRels]);

  return (
    <div className="flex flex-col h-full">
      {/* The reviewer's read of the report, if this gate ran in assist mode. */}
      {brief && onSaveBrief && (
        <div className="mb-2">
          <ReviewerBrief brief={brief} onSave={onSaveBrief} saving={savingBrief} />
        </div>
      )}
      {/* Toolbar row 1: tab switcher + procedure cycling + submit */}
      <div className="flex items-center gap-3 mb-1.5 px-1">
        {/* Tab switcher */}
        <div className="flex rounded border border-gb-bg2 overflow-hidden">
          <button
            type="button"
            onClick={() => setTab("flow")}
            title={hint("bundle-flow-tab")}
            className={`text-[11px] font-data px-3 py-1 ${tab === "flow" ? "bg-gb-bright-orange text-gb-bg0 font-semibold" : "bg-gb-bg1 text-gb-fg3 hover:bg-gb-bg2"}`}
          >
            Flow
          </button>
          <button
            type="button"
            onClick={() => setTab("bundle")}
            title={hint("bundle-bundle-tab")}
            className={`text-[11px] font-data px-3 py-1 ${tab === "bundle" ? "bg-gb-bright-orange text-gb-bg0 font-semibold" : "bg-gb-bg1 text-gb-fg3 hover:bg-gb-bg2"}`}
          >
            Bundle
          </button>
        </div>
        {/* Edge-label toggle. Off by default — dense bundles drown in
            labels otherwise. Selected edge still shows its label. */}
        <button
          type="button"
          onClick={() => setShowEdgeLabels((v) => !v)}
          className={`text-[11px] font-data px-2.5 py-1 rounded border transition-colors ${
            showEdgeLabels
              ? "text-gb-bright-blue border-gb-bright-blue bg-gb-bright-blue-dim"
              : "text-gb-fg4 border-gb-bg2 hover:text-gb-fg2"
          }`}
          title="Show relationship-type labels on all edges"
        >
          Edge labels
        </button>
        {/* Search-and-jump: live filter on node names, jumps to + selects
            the picked node. Hidden-layer targets auto-enable their layer. */}
        <NodeSearchBox nodes={searchableNodes} onJump={jumpToNode} />
        {/* Procedure cycling — only in Bundle tab where focus is meaningful */}
        {tab === "bundle" && procedureKeys.length > 0 && (
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={() => cycleProcedure(-1)}
              className="text-[11px] font-data px-2 py-1 rounded bg-gb-bg1 text-gb-fg2 hover:bg-gb-bg2 border border-gb-bg2"
              title="Previous procedure"
            >
              ←
            </button>
            <span className="text-[11px] font-data text-gb-fg3 whitespace-nowrap">
              procedure {focusKey ? procedureKeys.indexOf(focusKey) + 1 : "?"} / {procedureKeys.length}
            </span>
            <button
              type="button"
              onClick={() => cycleProcedure(1)}
              className="text-[11px] font-data px-2 py-1 rounded bg-gb-bg1 text-gb-fg2 hover:bg-gb-bg2 border border-gb-bg2"
              title="Next procedure"
            >
              →
            </button>
            <button
              type="button"
              onClick={() => setFocusKey(null)}
              className="text-[10px] font-data px-2 py-1 rounded bg-gb-bg1 text-gb-fg4 hover:bg-gb-bg2 border border-gb-bg2 ml-1"
              title="Show all (clear focus)"
            >
              show all
            </button>
          </div>
        )}
        <div className="ml-auto flex items-center gap-2">
          {/* Only HIGH-confidence recommendations — see handleAcceptAI. */}
          {bulkAcceptableCount > 0 && (
            <button
              type="button"
              onClick={handleAcceptAI}
              className="px-2.5 py-1 rounded text-[11px] font-data font-semibold bg-gb-purple/15 text-gb-bright-purple border border-gb-purple hover:bg-gb-bg2"
              title="Applies only the AI's high-confidence recommendations. Medium and low stay for you to judge."
            >
              🤖 Accept {bulkAcceptableCount} high-confidence
            </button>
          )}
          <button
            type="button"
            onClick={handleSubmit}
            disabled={submitting}
            className="px-3 py-1 rounded text-[11px] font-data font-semibold bg-gb-bright-aqua text-gb-bg0 hover:bg-gb-aqua disabled:opacity-50"
          >
            {submitting ? "Submitting…" : hasChanges ? "✓ Submit changes" : "✓ Approve as-is"}
          </button>
        </div>
      </div>
      {/* Toolbar row 2: stats + layer toggles. Layers only meaningful in
          Bundle tab (Flow forces procedure-only). */}
      <div className="flex items-center gap-3 mb-2 px-1">
        <div className="text-[11px] font-data text-gb-fg2 flex gap-3 items-center flex-wrap">
          <span>{stats.nodes} nodes</span>
          {/* Edge stat — surface a hidden-by-filter hint when applicable.
              When the layer toggle hides procedures, every procedure-anchored
              edge disappears too (covers ~99% of typical bundles); without
              this hint the analyst sees "0 edges" and assumes the data is
              broken. */}
          {(() => {
            const hidden = (hiddenByLayer || 0) + (hiddenByFlowFilter || 0);
            if (hidden === 0) {
              return <span>{stats.total} edges</span>;
            }
            const reasonParts = [];
            if (hiddenByLayer > 0) reasonParts.push(`${hiddenByLayer} hidden by layer filter`);
            if (hiddenByFlowFilter > 0) reasonParts.push(`${hiddenByFlowFilter} non-precedes hidden in Flow`);
            const tooltip = reasonParts.join("; ");
            const visible = stats.total;
            const total = totalDataEdges ?? (visible + hidden);
            const className = visible === 0
              ? "text-gb-bright-orange font-medium"
              : "";
            return (
              <span className={className} title={tooltip}>
                {visible} of {total} edges visible
                <span className="text-gb-fg4 ml-1">
                  · {hidden} hidden
                </span>
              </span>
            );
          })()}
          {stats.edited > 0 && <span className="text-gb-bright-yellow">{stats.edited} edited</span>}
          {stats.removed > 0 && <span className="text-gb-bright-red">{stats.removed} removed</span>}
          {stats.added > 0 && <span className="text-gb-bright-green">+{stats.added} added</span>}
        </div>
        {tab === "bundle" && (
          <div className="flex items-center gap-1.5 ml-auto">
            <span className="text-[10px] font-data uppercase text-gb-fg4">
              layers<InfoDot term="bundle-layers" />
            </span>
            {FILTER_LAYER_NAMES.map((name) => {
              const active = layerActive[name];
              // Stronger on/off distinction: inactive
              // toggles get a strikethrough + dimmer opacity + dashed
              // border so a glance at the toolbar makes the off-state
              // unambiguous. Prior `opacity-40` alone was too subtle to
              // catch when "0 edges" was the symptom.
              return (
                <button
                  key={name}
                  type="button"
                  onClick={() => toggleLayer(name)}
                  title={active ? `Hide ${name}` : `Show ${name}`}
                  className={`text-[10px] font-data px-1.5 py-0.5 rounded transition-all ${
                    active
                      ? "border"
                      : "border border-dashed opacity-30 line-through"
                  }`}
                  style={{
                    background: active ? `${LAYER_COLORS[name]}1a` : "transparent",
                    borderColor: active ? `${LAYER_COLORS[name]}80` : `${LAYER_COLORS[name]}40`,
                    color: LAYER_COLORS[name],
                  }}
                >
                  {name}
                </button>
              );
            })}
          </div>
        )}
      </div>

      {/* Path-trace banner — pinned just below the toolbar when a
          trace is active. Quiet (no extra row) otherwise. */}
      {pathTrace.start && pathTrace.end && (
        <div className="flex items-center gap-2 px-3 py-1 mb-1 rounded border border-gb-bright-orange bg-gb-bg0/95">
          <span className="font-data text-[10px] text-gb-bright-orange uppercase tracking-wide">
            Path trace
          </span>
          <StixNodeIcon type={pathTrace.start.type} size={16} />
          <span className="font-data text-[11px] text-gb-fg1 max-w-[200px] truncate" title={pathTrace.start.name}>
            {pathTrace.start.name}
          </span>
          <span className="font-data text-[11px] text-gb-bright-orange">→</span>
          <StixNodeIcon type={pathTrace.end.type} size={16} />
          <span className="font-data text-[11px] text-gb-fg1 max-w-[200px] truncate" title={pathTrace.end.name}>
            {pathTrace.end.name}
          </span>
          <span className="font-data text-[10px] text-gb-fg4 ml-1">
            {pathTrace.count === 0
              ? "no path"
              : `${pathTrace.count} path${pathTrace.count === 1 ? "" : "s"}`}
          </span>
          <button
            type="button"
            onClick={clearPathTrace}
            className="ml-auto font-data text-[10px] text-gb-fg4 hover:text-gb-fg1"
            title="Clear path trace"
          >
            Clear ✕
          </button>
        </div>
      )}

      <div className="grid h-[calc(100%-2.5rem)]" style={{ gridTemplateColumns: "1fr 320px", gap: "0.75rem" }}>
        <div
          className="rounded border border-gb-bg1 bg-gb-bg0 relative"
          onMouseMove={(e) => {
            // Only update cursor position when a tooltip is actually
            // visible. The previous "React will skip rerender if the
            // tooltip itself is null" claim is wrong — the parent still
            // rerenders on every pixel of movement. Gating on hoverNode
            // drops hundreds of
            // wasted renders/sec when the cursor is over empty canvas.
            if (!hoverNode) return;
            const r = e.currentTarget.getBoundingClientRect();
            setHoverPos({ x: e.clientX - r.left, y: e.clientY - r.top });
          }}
        >
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={NODE_TYPES}
            onNodeClick={handleNodeClick}
            onNodeMouseEnter={(_, node) => setHoverNode({
              type: node.data?.type,
              name: node.data?.name,
              id: node.id,
            })}
            onNodeMouseLeave={() => setHoverNode(null)}
            onEdgeClick={handleEdgeClick}
            onPaneClick={handlePaneClick}
            onConnect={handleConnect}
            onEdgesDelete={handleEdgesDelete}
            nodesConnectable
            edgesFocusable
            deleteKeyCode={["Backspace", "Delete"]}
            fitView
            fitViewOptions={FIT_VIEW_OPTIONS}
            minZoom={0.15}
            maxZoom={2}
          >
            <Background gap={16} size={1} color="rgba(255,255,255,0.04)" />
            <Controls showInteractive={false} />
            <MiniMap
              pannable
              zoomable
              style={{ background: "rgba(0,0,0,0.4)" }}
              nodeStrokeWidth={2}
              nodeColor={(node) => getTypeConfig(node.data?.type)?.color || "#fbf1c7"}
              nodeStrokeColor={(node) => getTypeConfig(node.data?.type)?.color || "#fbf1c7"}
              maskColor="rgba(0,0,0,0.5)"
            />
          </ReactFlow>
          <NodeHoverTooltip node={hoverNode} x={hoverPos.x} y={hoverPos.y} />
        </div>
        <div className="overflow-y-auto px-1">
          <SidePanel
            selectedNode={selectedNode}
            selectedEdge={selectedEdge}
            pendingAdd={pendingAdd}
            onEditEdgeRelType={editEdgeRelType}
            onRemoveEdge={removeEdge}
            onUndoAdded={undoAdded}
            onConfirmAdd={confirmAdd}
            onCancelAdd={cancelAdd}
          />
          {/* AI reviewer recommendations for what is currently in focus.
              Clicking one focuses that relationship's procedure, so the
              panel and the canvas stay in step. */}
          {visibleRecs.length > 0 && (
            <div className="mt-3 border-t border-gb-bg2 pt-2">
              <p className="text-[12px] font-semibold text-gb-bright-purple mb-1.5">
                🤖 AI review ({visibleRecs.length}
                {focusKey ? " here" : ""})
              </p>
              {recommendations?.overall_notes && !focusKey && (
                <p className="text-[11px] text-gb-fg4 mb-2 leading-snug">
                  {recommendations.overall_notes}
                </p>
              )}
              <div className="flex flex-col gap-1.5">
                {visibleRecs.map((rec) => {
                  const rel = (payload?.items ?? []).find((r) => r.id === rec.rel_id);
                  return (
                    <div key={rec.rel_id}>
                      {rel && (
                        <button
                          type="button"
                          onClick={() => setFocusKey(
                            relEndpointKeys(rel, editedRels, context?.drafts ?? []).srcKey,
                          )}
                          className="w-full text-left text-[10px] font-data text-gb-fg2 hover:text-gb-bright-purple truncate"
                          title="Focus this procedure on the canvas"
                        >
                          {rel.source_name} --{rel.relationship_type}--&gt; {rel.target_name}
                        </button>
                      )}
                      <SuggestionChip
                        rec={rec}
                        applied={appliedRecs.has(rec.rel_id)}
                        onApply={() => handleApplyRec(rec.rel_id)}
                        onDismiss={() => handleDismissRec(rec.rel_id)}
                      />
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}


// React Flow's useReactFlow() hook (for setCenter / fitView) must be
// called inside a ReactFlowProvider. The viewer wraps its inner body
// in the provider so toolbar widgets like NodeSearchBox can drive the
// viewport. Mirrors BundleFlowView's pattern.
export default function BundleReviewCanvas(props) {
  return (
    <ReactFlowProvider>
      <BundleReviewCanvasInner {...props} />
    </ReactFlowProvider>
  );
}
