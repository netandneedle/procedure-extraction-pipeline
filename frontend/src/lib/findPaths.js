/**
 * Path-tracing utilities used by all three viewers.
 *
 * Strategy: layer-wise BFS that records ALL shortest-path
 * predecessors per node, then DFS-reconstructs every distinct
 * shortest path. We don't return arbitrary simple paths (potentially
 * exponential) — shortest-only keeps output bounded and matches the
 * "how do A and B connect?" question analysts actually ask.
 *
 * Direction: paths treat edges as UNDIRECTED. STIX rel direction
 * matters for some semantics (procedure → technique implements_technique)
 * but for "connect A and B" we want the analyst to see the path
 * regardless of which way each edge points. Each viewer's edge
 * renderer still draws its own directional arrowhead so flow stays
 * visible.
 */

/**
 * Build an undirected adjacency Map<string, Set<string>> from a list
 * of edge-like objects. Each edge must expose a `source` and `target`
 * (either a string id or an object with `.id`).
 */
export function buildAdjacency(edges) {
  const adj = new Map();
  const addEdge = (a, b) => {
    if (!a || !b || a === b) return;
    if (!adj.has(a)) adj.set(a, new Set());
    if (!adj.has(b)) adj.set(b, new Set());
    adj.get(a).add(b);
    adj.get(b).add(a);
  };
  for (const e of edges || []) {
    const s = typeof e.source === "string" ? e.source : e.source?.id;
    const t = typeof e.target === "string" ? e.target : e.target?.id;
    addEdge(s, t);
  }
  return adj;
}

/**
 * Find every shortest path (each as an ordered array of node ids)
 * from startId to endId in an undirected graph.
 *
 * Bounded by:
 *   maxDepth   — BFS stops descending past this layer (default 10).
 *                Larger graphs with deeper actual paths would need this
 *                bumped, but 10 covers every reasonable STIX bundle.
 *   maxPaths   — reconstruction caps at this many paths so a
 *                pathological "many parallel routes" graph can't
 *                freeze the renderer.
 *
 * Returns [] when no path exists. Returns [[startId]] when
 * startId === endId.
 */
export function findAllShortestPaths(
  adjacency,
  startId,
  endId,
  { maxDepth = 10, maxPaths = 100 } = {},
) {
  if (!startId || !endId) return [];
  if (startId === endId) return [[startId]];
  if (!adjacency.has(startId) || !adjacency.has(endId)) return [];

  // Layer-wise BFS recording every predecessor on shortest paths.
  const dist = new Map([[startId, 0]]);
  const parents = new Map();
  let queue = [startId];
  while (queue.length) {
    const nextQueue = [];
    for (const u of queue) {
      const du = dist.get(u);
      if (du >= maxDepth) continue;
      const neighbors = adjacency.get(u);
      if (!neighbors) continue;
      for (const v of neighbors) {
        const dv = dist.get(v);
        if (dv === undefined) {
          dist.set(v, du + 1);
          parents.set(v, new Set([u]));
          nextQueue.push(v);
        } else if (dv === du + 1) {
          // Multiple shortest-path predecessors are tracked so we can
          // enumerate every distinct route.
          parents.get(v).add(u);
        }
      }
    }
    queue = nextQueue;
    if (dist.has(endId)) break;
  }
  if (!dist.has(endId)) return [];

  // DFS backward from endId, accumulating the path forward to endId.
  // `suffix` is the path from `node` to endId, excluding node itself.
  const result = [];
  const dfs = (node, suffix) => {
    if (result.length >= maxPaths) return;
    if (node === startId) {
      result.push([startId, ...suffix]);
      return;
    }
    const ps = parents.get(node);
    if (!ps) return;
    for (const p of ps) {
      if (result.length >= maxPaths) return;
      dfs(p, [node, ...suffix]);
    }
  };
  dfs(endId, []);
  return result;
}

/**
 * Collapse a list of paths into:
 *   - nodeSet:  Set of every node id appearing on any path.
 *   - edgeSet:  Set of "a|b" stringified pairs (sorted) for every
 *               edge traversed on any path.
 *
 * Used by viewers to drive focus/dimming. We stringify unordered
 * pairs so the renderer can hit-check edges without knowing path
 * direction.
 */
export function collectPathElements(paths) {
  const nodeSet = new Set();
  const edgeSet = new Set();
  for (const path of paths) {
    for (let i = 0; i < path.length; i++) {
      nodeSet.add(path[i]);
      if (i > 0) {
        const a = path[i - 1];
        const b = path[i];
        const key = a < b ? `${a}|${b}` : `${b}|${a}`;
        edgeSet.add(key);
      }
    }
  }
  return { nodeSet, edgeSet };
}

/** Stringify-key for an unordered pair, matching collectPathElements. */
export function pathEdgeKey(a, b) {
  return a < b ? `${a}|${b}` : `${b}|${a}`;
}
