/**
 * BundleFlowView — read-only flow visualization for a completed bundle.
 *
 * Renders the ATT&CK Flow sequencing graph — x-procedure nodes plus the
 * attack-operator / attack-condition nodes the PRECEDES chain routes
 * through — via React Flow + dagre. Click a node to open a side panel
 * that resolves its STIX refs to human-readable values (technique names,
 * source identity, observables, vulnerabilities, etc.) using the
 * bundleResolution lib.
 *
 * This is the Explorer's flow-mode counterpart to BundleGraph (the
 * canvas-based generalized graph view). Unlike BundleReviewCanvas
 * (the gate_2 editing surface), this view has no edit affordances —
 * the bundle is already shipped.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  ReactFlow,
  Background,
  Controls,
  Handle,
  MiniMap,
  Position,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { applyPositions, useLayoutPositions } from "../lib/graphLayout";

import {
  buildResolutionIndex,
  extractFlowGraph,
  resolveRefs,
} from "../lib/bundleResolution";
import ProvenanceBadge from "./ProvenanceBadge";
import StixNodeIcon from "./StixNodeIcon";
import NodeSearchBox from "./NodeSearchBox";
import NodeHoverTooltip from "./NodeHoverTooltip";
import {
  buildAdjacency,
  findAllShortestPaths,
  collectPathElements,
  pathEdgeKey,
} from "../lib/findPaths";
import { getTypeConfig, operatorKindColor } from "../lib/bundleGraphConstants";

const NODE_WIDTH = 220;
const NODE_HEIGHT = 88;
// 80/60 fits typical 5-12 procedure bundles without crowding. Operator and
// condition chips carry their own narrower `width`, which the layout honours.
const FLOW_LAYOUT = { rankdir: "TB", nodesep: 80, ranksep: 60, marginx: 30, marginy: 30, width: NODE_WIDTH, height: NODE_HEIGHT };
// Flow-control nodes are deliberately narrower than procedures so the
// eye reads them as junctions rather than content.
const OPERATOR_WIDTH = 90;
const CONDITION_WIDTH = 140;

// ─── Custom React Flow node ──────────────────────────────────────────

function ProcedureNode({ data }) {
  const ringClass = data.isSelected
    ? "ring-2 ring-gb-bright-orange shadow-lg"
    : "ring-1 ring-gb-bg2 hover:ring-gb-fg4";
  return (
    <div
      className={`bg-gb-bg0-s rounded-md px-3 py-2 transition-shadow w-[220px] flex items-start gap-2 ${ringClass}`}
      style={{ height: NODE_HEIGHT - 4 }}
    >
      <Handle type="target" position={Position.Top} className="!bg-gb-bg2" />
      <Handle type="source" position={Position.Bottom} className="!bg-gb-bg2" />
      <StixNodeIcon type="x-procedure" size={28} title="x-procedure" className="shrink-0 mt-0.5" />
      <div className="min-w-0 flex-1">
        <p className="font-data text-[10px] text-gb-bright-orange mb-0.5">
          {data.tactic || "x-procedure"}
        </p>
        <p
          className="text-[12px] text-gb-fg1 font-medium leading-tight overflow-hidden"
          style={{
            display: "-webkit-box",
            WebkitLineClamp: 3,
            WebkitBoxOrient: "vertical",
          }}
          title={data.name}
        >
          {data.name}
        </p>
      </div>
    </div>
  );
}

/**
 * Render an attack-operator (AND/OR/XOR) as a small kind-colored chip
 * with the operator label centered. Sized smaller than ProcedureNode
 * so the flow visually distinguishes flow-control from procedure
 * content. Reachable only when the Flow-tab plumbing surfaces operators
 * (post-serialize); the BundleNode renderer in BundleReviewCanvas
 * handles operators in the Bundle tab.
 */
function OperatorNode({ data }) {
  const kind = data.operator || "OR";
  const color = data.color || "#ebcb8b";
  const ringClass = data.isSelected
    ? "ring-2 ring-gb-bright-orange shadow-lg"
    : "ring-1 ring-gb-bg2 hover:ring-gb-fg4";
  return (
    <div
      className={`rounded-md px-3 py-2 flex flex-col items-center justify-center transition-shadow ${ringClass}`}
      style={{
        background: `${color}1f`,
        borderColor: `${color}80`,
        borderWidth: 1,
        width: 90,
        height: NODE_HEIGHT - 4,
      }}
    >
      <Handle type="target" position={Position.Top} className="!bg-gb-bg2" />
      <Handle type="source" position={Position.Bottom} className="!bg-gb-bg2" />
      <p className="text-[9px] font-data text-gb-fg4 uppercase tracking-wide">
        operator
      </p>
      <p
        className="text-[14px] font-data font-semibold tracking-wide"
        style={{ color }}
      >
        {kind}
      </p>
    </div>
  );
}

/**
 * Render an attack-condition as a question-mark chip with the
 * description below. Color is gruvbox bright-purple (matches the
 * bundleGraphConstants palette) so conditions visually distinguish
 * from operators (kind-colored) and procedures. Reachable when the
 * Flow tab's plumbing surfaces conditions (post-serialize); the
 * BundleNode renderer in BundleReviewCanvas handles conditions in
 * the Bundle tab.
 */
function ConditionNode({ data }) {
  const color = data.color || "#b48ead";
  const description = data.description || "";
  const ringClass = data.isSelected
    ? "ring-2 ring-gb-bright-orange shadow-lg"
    : "ring-1 ring-gb-bg2 hover:ring-gb-fg4";
  return (
    <div
      className={`rounded-md px-3 py-2 flex flex-col items-center justify-center transition-shadow ${ringClass}`}
      style={{
        background: `${color}1f`,
        borderColor: `${color}80`,
        borderWidth: 1,
        width: 140,
        height: NODE_HEIGHT - 4,
      }}
    >
      <Handle type="target" position={Position.Top} className="!bg-gb-bg2" />
      <Handle type="source" position={Position.Bottom} className="!bg-gb-bg2" />
      <p className="text-[9px] font-data text-gb-fg4 uppercase tracking-wide">
        condition
      </p>
      <p
        className="text-[10px] italic text-gb-fg2 leading-tight text-center mt-0.5 overflow-hidden"
        style={{
          display: "-webkit-box",
          WebkitLineClamp: 3,
          WebkitBoxOrient: "vertical",
        }}
        title={description}
      >
        {description}
      </p>
    </div>
  );
}

const NODE_TYPES = {
  procedure: ProcedureNode,
  operator: OperatorNode,
  condition: ConditionNode,
};

// Hoisted to module scope so React Flow doesn't see fresh object references
// on every render.
const FIT_VIEW_OPTIONS = { padding: 0.15, minZoom: 0.4, maxZoom: 1.2 };
const PRO_OPTIONS = { hideAttribution: true };

// ─── Layout ──────────────────────────────────────────────────────────


// ─── Side panel ──────────────────────────────────────────────────────

function ResolvedField({ label, value }) {
  if (!value) return null;
  return (
    <div className="mb-2">
      <p className="font-data text-[10px] text-gb-fg4 mb-0.5">{label}</p>
      <p className="text-[12px] text-gb-fg1 leading-snug">{value}</p>
    </div>
  );
}

function ResolvedRefList({ label, refs, emptyText = "—" }) {
  if (!refs || refs.length === 0) {
    return (
      <div className="mb-3">
        <p className="font-data text-[10px] text-gb-fg4 mb-0.5">{label}</p>
        <p className="text-[11px] text-gb-bg4">{emptyText}</p>
      </div>
    );
  }
  return (
    <div className="mb-3">
      <p className="font-data text-[10px] text-gb-fg4 mb-0.5">
        {label} ({refs.length})
      </p>
      <ul className="space-y-1">
        {refs.map((r, i) => (
          <li
            key={`${r.rawId}-${i}`}
            className={`text-[12px] leading-snug border-l-2 pl-2 py-0.5 ${
              r.external
                ? "border-gb-bg2 text-gb-fg4"
                : "border-gb-orange text-gb-fg1"
            }`}
          >
            <span className="font-data text-[10px] text-gb-fg4 mr-1">{r.type}</span>
            <span className="break-words">{r.displayName}</span>
            {r.secondary && (
              <span className="font-data text-[10px] text-gb-fg4 block ml-0">
                {r.secondary}
              </span>
            )}
          </li>
        ))}
      </ul>
    </div>
  );
}

/**
 * Detail panel for the flow-control nodes (attack-operator /
 * attack-condition). They have no procedure fields to resolve — what
 * matters is which branches feed in and which fan out, so the panel is
 * just the in/out edge lists plus the operator kind or condition text.
 */
function FlowControlDetailPanel({ node, index, predecessorRefs, successorRefs }) {
  const isOperator = node.type === "attack-operator";
  const kind = node.operator || "OR";
  const color = isOperator
    ? operatorKindColor(kind)
    : getTypeConfig("attack-condition").color;

  return (
    <div className="px-3 py-3 overflow-y-auto h-full">
      <div className="flex items-center gap-2 mb-1">
        <StixNodeIcon type={node.type} size={20} className="shrink-0" />
        <h3
          className="text-[13px] font-semibold leading-tight"
          style={{ color }}
        >
          {isOperator ? `${kind} operator` : "Condition"}
        </h3>
      </div>
      <p className="font-data text-[10px] text-gb-fg4 mb-3">{node.type}</p>

      {isOperator ? (
        <p className="text-[12px] text-gb-fg2 mb-3 leading-relaxed">
          {kind === "AND"
            ? "All inbound branches converge here — every predecessor happens before the flow continues."
            : kind === "XOR"
              ? "Mutually exclusive branches — exactly one outbound path is taken."
              : "Alternative branches — one or more outbound paths may be taken."}
        </p>
      ) : (
        <p className="text-[12px] text-gb-fg2 mb-3 leading-relaxed italic">
          {node.description || "(no description)"}
        </p>
      )}

      <ResolvedRefList
        label="Previous in flow"
        refs={resolveRefs(index, predecessorRefs)}
        emptyText="(start of flow)"
      />
      <ResolvedRefList
        label="Next in flow"
        refs={resolveRefs(index, successorRefs)}
        emptyText="(end of flow)"
      />
    </div>
  );
}

function ProcedureDetailPanel({ procedure, index, successorRefs, observableRefs }) {
  if (!procedure) {
    return (
      <div className="px-3 py-4 text-[12px] text-gb-fg4 leading-relaxed">
        Click a node to inspect its fields.
      </div>
    );
  }
  const tactics = (procedure.kill_chain_phases || [])
    .filter((k) => k.kill_chain_name === "mitre-attack")
    .map((k) => k.phase_name)
    .filter(Boolean);

  return (
    <div className="px-3 py-3 overflow-y-auto h-full">
      <h3 className="text-[13px] font-semibold text-gb-fg0 mb-1 leading-tight">
        {procedure.name}
      </h3>
      <div className="flex items-center gap-2 flex-wrap mb-2">
        <span className="font-data text-[10px] text-gb-fg4">
          {procedure.x_procedure_type || "x-procedure"}
          {typeof procedure.confidence === "number"
            ? ` · confidence ${procedure.confidence}`
            : ""}
        </span>
        {procedure.x_source_provenance && (
          <ProvenanceBadge
            provenance={procedure.x_source_provenance}
            compact
          />
        )}
      </div>
      {procedure.description && (
        <p className="text-[12px] text-gb-fg2 mb-3 leading-relaxed">
          {procedure.description}
        </p>
      )}

      <ResolvedField label="Tactics" value={tactics.join(", ")} />
      <ResolvedField
        label="Platforms"
        value={(procedure.x_platforms || []).join(", ")}
      />

      <ResolvedRefList
        label="Techniques"
        refs={resolveRefs(index, procedure.x_technique_refs)}
      />
      <ResolvedRefList
        label="Source"
        refs={resolveRefs(index, procedure.x_source_refs)}
      />
      <ResolvedRefList
        label="Vulnerabilities"
        refs={resolveRefs(index, procedure.x_vulnerability_refs)}
      />
      <ResolvedRefList
        label="Observables"
        refs={resolveRefs(index, observableRefs)}
      />
      <ResolvedRefList
        label="Components"
        refs={resolveRefs(index, procedure.x_components_refs)}
      />
      <ResolvedRefList
        label="Log sources"
        refs={resolveRefs(index, procedure.x_log_source_refs)}
      />
      <ResolvedRefList
        label="Next in flow"
        refs={resolveRefs(index, successorRefs)}
        emptyText="(end of flow)"
      />

      {procedure.x_fingerprint && (
        <ResolvedField
          label="Fingerprint"
          value={procedure.x_fingerprint}
        />
      )}
    </div>
  );
}

// ─── Inner component (uses React Flow context) ──────────────────────

function FlowInner({ bundle }) {
  const { fitView, setCenter, getNode } = useReactFlow();
  const [selectedId, setSelectedId] = useState(null);
  const [hoverNode, setHoverNode] = useState(null);
  const [hoverPos, setHoverPos] = useState({ x: 0, y: 0 });
  const [pathTrace, setPathTrace] = useState({
    start: null, end: null, nodes: null, edges: null, count: 0,
  });

  // One-pass resolution index. Memoized — bundle reference is stable
  // for the lifetime of this component.
  const resolutionIndex = useMemo(
    () => buildResolutionIndex(bundle),
    [bundle],
  );

  // Pull the flow DAG out of the bundle. Nodes are procedures PLUS the
  // attack-operator / attack-condition nodes the PRECEDES chain routes
  // through — dropping those would sever the chain at every branch.
  const { nodes: flowNodes, edges: rawEdges } = useMemo(
    () => extractFlowGraph(bundle),
    [bundle],
  );

  const procedures = useMemo(
    () => flowNodes.filter((o) => o.type === "x-procedure"),
    [flowNodes],
  );

  // Build React Flow nodes + edges with dagre layout.
  const { nodes: unplacedNodes, edges } = useMemo(() => {
    if (flowNodes.length === 0) return { nodes: [], edges: [] };
    const traceActive = !!(pathTrace.nodes && pathTrace.nodes.size > 0);
    const rfNodes = flowNodes.map((o) => {
      const dimmed = traceActive && !pathTrace.nodes.has(o.id);
      const isSelected = o.id === selectedId;
      const common = {
        id: o.id,
        position: { x: 0, y: 0 },
        style: { opacity: dimmed ? 0.2 : 1 },
        height: NODE_HEIGHT,
      };

      if (o.type === "attack-operator") {
        const kind = o.operator || "OR";
        return {
          ...common,
          type: "operator",
          data: {
            stixType: o.type,
            label: `${kind} operator`,
            operator: kind,
            color: operatorKindColor(kind),
            isSelected,
          },
          width: OPERATOR_WIDTH,
        };
      }

      if (o.type === "attack-condition") {
        return {
          ...common,
          type: "condition",
          data: {
            stixType: o.type,
            label: o.description || "(condition)",
            description: o.description || "",
            color: getTypeConfig("attack-condition").color,
            isSelected,
          },
          width: CONDITION_WIDTH,
        };
      }

      const tactic = (o.kill_chain_phases || [])
        .find((k) => k.kill_chain_name === "mitre-attack")?.phase_name;
      return {
        ...common,
        type: "procedure",
        data: {
          stixType: o.type,
          label: o.name || "(unnamed)",
          name: o.name || "(unnamed)",
          tactic: tactic || null,
          isSelected,
        },
        width: NODE_WIDTH,
      };
    });
    const rfEdges = rawEdges.map((e, i) => {
      const dimmed = traceActive && !pathTrace.edges.has(pathEdgeKey(e.source, e.target));
      return {
        id: `${e.source}→${e.target}-${i}`,
        source: e.source,
        target: e.target,
        animated: false,
        style: {
          stroke: "rgba(235,203,139,0.6)",
          strokeWidth: 1.5,
          opacity: dimmed ? 0.15 : 1,
        },
      };
    });
    return { nodes: rfNodes, edges: rfEdges };
  }, [flowNodes, rawEdges, selectedId, pathTrace.nodes, pathTrace.edges]);
  const positions = useLayoutPositions(unplacedNodes, edges, FLOW_LAYOUT);
  const nodes = useMemo(() => applyPositions(unplacedNodes, positions), [unplacedNodes, positions]);

  // Auto-fit on mount and whenever the flow node set changes.
  useEffect(() => {
    if (nodes.length > 0) {
      const t = setTimeout(() => fitView({ padding: 0.15, duration: 200 }), 50);
      return () => clearTimeout(t);
    }
  }, [nodes.length, fitView]);

  const selectedNode = useMemo(
    () => flowNodes.find((o) => o.id === selectedId) || null,
    [flowNodes, selectedId],
  );

  // Observables each procedure TOUCHES (registry keys, C2 domains, file
  // hashes), read off the has-observable SROs. Not embedded on the
  // procedure: x_observable_refs isn't one of the properties in x-procedure
  // v0.5.0-draft, whose schema is additionalProperties:false. Distinct from
  // x_components_refs, which is what the procedure is MADE of — the process
  // tree of its commands.
  const observableRefsByProcedure = useMemo(() => {
    const map = {};
    for (const obj of bundle?.objects ?? []) {
      if (obj?.type !== "relationship") continue;
      if (obj.relationship_type !== "has-observable") continue;
      if (!obj.source_ref || !obj.target_ref) continue;
      if (!map[obj.source_ref]) map[obj.source_ref] = [];
      map[obj.source_ref].push(obj.target_ref);
    }
    return map;
  }, [bundle]);

  // Neighbours of the selected node, derived from the PRECEDES-edge flow
  // graph (x_effect_refs was removed in v0.5.0-draft — sequencing lives
  // only in the SROs now).
  const selectedSuccessorRefs = useMemo(
    () =>
      selectedId
        ? rawEdges.filter((e) => e.source === selectedId).map((e) => e.target)
        : [],
    [rawEdges, selectedId],
  );

  const selectedPredecessorRefs = useMemo(
    () =>
      selectedId
        ? rawEdges.filter((e) => e.target === selectedId).map((e) => e.source)
        : [],
    [rawEdges, selectedId],
  );

  // Search index for jump-to-procedure. Procedures-only by design —
  // Flow tab doesn't render anything else, so a "process--xxx" hit
  // wouldn't lead anywhere visible.
  const searchableNodes = useMemo(
    () => procedures.map((p) => ({
      id: p.id,
      displayName: p.name || "(unnamed)",
      type: "x-procedure",
      mitreId: null,
    })),
    [procedures],
  );

  const flowControlCount = flowNodes.length - procedures.length;

  const jumpToNode = useCallback(
    (matched) => {
      setSelectedId(matched.id);
      const rfNode = getNode(matched.id);
      if (rfNode) {
        const x = rfNode.position.x + (rfNode.width || NODE_WIDTH) / 2;
        const y = rfNode.position.y + (rfNode.height || NODE_HEIGHT) / 2;
        setCenter(x, y, { zoom: 1.0, duration: 400 });
      }
    },
    [setCenter, getNode],
  );

  if (procedures.length === 0) {
    return (
      <div className="flex-1 flex items-center justify-center text-gb-fg4 text-[13px]">
        Bundle contains no procedures — nothing to render in flow view.
      </div>
    );
  }

  return (
    <div className="flex-1 flex min-h-0">
      <div
        className="flex-1 min-w-0 relative"
        onMouseMove={(e) => {
          // Only update cursor position when a tooltip is actually visible;
          // otherwise every pixel of movement is a setState and a rerender.
          if (!hoverNode) return;
          const r = e.currentTarget.getBoundingClientRect();
          setHoverPos({ x: e.clientX - r.left, y: e.clientY - r.top });
        }}
      >
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={NODE_TYPES}
          onNodeClick={(evt, node) => {
            // Shift-click: trace path from prior selection to this node.
            if (evt.shiftKey && selectedId && selectedId !== node.id) {
              const adj = buildAdjacency(rawEdges);
              const paths = findAllShortestPaths(adj, selectedId, node.id);
              const { nodeSet, edgeSet } = collectPathElements(paths);
              const startRf = nodes.find((n) => n.id === selectedId);
              const endRf = nodes.find((n) => n.id === node.id);
              setPathTrace({
                start: startRf
                  ? { id: startRf.id, name: startRf.data.label, type: startRf.data.stixType }
                  : null,
                end: endRf
                  ? { id: endRf.id, name: endRf.data.label, type: endRf.data.stixType }
                  : null,
                nodes: nodeSet,
                edges: edgeSet,
                count: paths.length,
              });
              return;
            }
            if (pathTrace.start) {
              setPathTrace({ start: null, end: null, nodes: null, edges: null, count: 0 });
            }
            setSelectedId(node.id);
          }}
          onNodeMouseEnter={(_, node) => setHoverNode({
            type: node.data?.stixType || "x-procedure",
            name: node.data?.label || "",
            id: node.id,
          })}
          onNodeMouseLeave={() => setHoverNode(null)}
          onPaneClick={() => setSelectedId(null)}
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable
          fitView
          fitViewOptions={FIT_VIEW_OPTIONS}
          proOptions={PRO_OPTIONS}
        >
          <Background gap={20} size={1} className="!bg-gb-bg0-h" />
          <Controls className="!bg-gb-bg0-s !border-gb-bg2" showInteractive={false} />
          <MiniMap
            pannable
            zoomable
            className="!bg-gb-bg0-s !border-gb-bg2"
            nodeColor={(n) => n.data?.color || "#d08770"}
            maskColor="rgba(40,40,40,0.6)"
          />
        </ReactFlow>
        <div className="absolute top-2 left-2 font-data text-[11px] text-gb-fg4 bg-gb-bg0-h/85 rounded px-2 py-1 border border-gb-bg2">
          {procedures.length} procedure{procedures.length === 1 ? "" : "s"}
          {flowControlCount > 0 && (
            <>
              {" · "}
              {flowControlCount} flow control
            </>
          )}
          {" · "}
          {rawEdges.length} precedes edge{rawEdges.length === 1 ? "" : "s"}
        </div>
        {pathTrace.start && pathTrace.end && (
          <div className="absolute top-2 left-1/2 -translate-x-1/2 z-30 flex items-center gap-2 px-3 py-1 rounded border border-gb-bright-orange bg-gb-bg0/95 shadow-lg">
            <span className="font-data text-[10px] text-gb-bright-orange uppercase tracking-wide">
              Path trace
            </span>
            <span className="font-data text-[11px] text-gb-fg1 max-w-[140px] truncate" title={pathTrace.start.name}>
              {pathTrace.start.name}
            </span>
            <span className="font-data text-[11px] text-gb-bright-orange">→</span>
            <span className="font-data text-[11px] text-gb-fg1 max-w-[140px] truncate" title={pathTrace.end.name}>
              {pathTrace.end.name}
            </span>
            <span className="font-data text-[10px] text-gb-fg4 ml-1">
              {pathTrace.count === 0
                ? "no path"
                : `${pathTrace.count} path${pathTrace.count === 1 ? "" : "s"}`}
            </span>
            <button
              type="button"
              onClick={() => setPathTrace({ start: null, end: null, nodes: null, edges: null, count: 0 })}
              className="font-data text-[10px] text-gb-fg4 hover:text-gb-fg1 ml-1"
              title="Clear path trace"
            >
              ✕
            </button>
          </div>
        )}
        <div className="absolute top-2 right-2">
          <NodeSearchBox
            nodes={searchableNodes}
            onJump={jumpToNode}
            placeholder="Find procedure…"
          />
        </div>
        <NodeHoverTooltip node={hoverNode} x={hoverPos.x} y={hoverPos.y} />
      </div>
      <aside className="w-[340px] flex-shrink-0 border-l border-gb-bg1 bg-gb-bg0-s overflow-hidden flex flex-col">
        <div className="px-3 py-2 border-b border-gb-bg1 flex items-center justify-between">
          <span className="font-data text-[11px] text-gb-fg4">Detail</span>
          {selectedNode && (
            <button
              type="button"
              onClick={() => setSelectedId(null)}
              className="font-data text-[10px] text-gb-fg4 hover:text-gb-fg1"
            >
              Close ✕
            </button>
          )}
        </div>
        {selectedNode && selectedNode.type !== "x-procedure" ? (
          <FlowControlDetailPanel
            node={selectedNode}
            index={resolutionIndex}
            predecessorRefs={selectedPredecessorRefs}
            successorRefs={selectedSuccessorRefs}
          />
        ) : (
          <ProcedureDetailPanel
            procedure={selectedNode}
            index={resolutionIndex}
            successorRefs={selectedSuccessorRefs}
            observableRefs={
              selectedId ? observableRefsByProcedure[selectedId] ?? [] : []
            }
          />
        )}
      </aside>
    </div>
  );
}

// ─── Outer ────────────────────────────────────────────────────────────

export default function BundleFlowView({ bundle }) {
  if (!bundle || !Array.isArray(bundle.objects)) {
    return (
      <div className="flex-1 flex items-center justify-center text-gb-fg4 text-[13px]">
        No bundle loaded.
      </div>
    );
  }
  return (
    <ReactFlowProvider>
      <FlowInner bundle={bundle} />
    </ReactFlowProvider>
  );
}
