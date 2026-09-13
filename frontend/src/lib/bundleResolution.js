/**
 * Bundle ID resolution layer.
 *
 * Walks bundle.objects once, produces a {stix_id → ResolvedRef} index
 * that the Flow view (and any future detail panels) use to render
 * STIX refs as human-readable values.
 *
 * Each ResolvedRef carries enough information for the side panel:
 *   - type: STIX object type (e.g. "attack-pattern", "process")
 *   - displayName: short label ("T1059.001 PowerShell" / "powershell.exe -enc ...")
 *   - secondary: optional second line ("execution" tactic, file hash, etc.)
 *   - rawId: the original STIX UUID (kept so callers can deep-link / audit)
 *
 * Design choices:
 *   - One pass over objects on construction. O(n) build, O(1) lookup.
 *   - Resolution is best-effort. Missing IDs return a ResolvedRef with
 *     displayName = the truncated UUID so the analyst still sees "something".
 *   - ATT&CK technique IDs that aren't embedded (per the validator's
 *     external-OK rule) get their T-number from the UUID's first 8 hex
 *     chars only when explicitly known; otherwise they fall through to
 *     the truncated-UUID branch. Future enhancement: ship a UUID→T-number
 *     map so external attack-pattern refs can render the T-number.
 */

const TYPE_RESOLVERS = {
  "x-procedure": (obj) => ({
    displayName: obj.name || "(unnamed procedure)",
    secondary: obj.x_procedure_type || null,
  }),
  "attack-pattern": (obj) => {
    const externalId = (obj.external_references || []).find(
      (r) => r.source_name === "mitre-attack",
    )?.external_id;
    return {
      displayName: externalId
        ? `${externalId} ${obj.name || ""}`.trim()
        : obj.name || "(unnamed technique)",
      secondary: extractTactic(obj),
    };
  },
  "intrusion-set": (obj) => ({
    displayName: obj.name || "(unnamed intrusion set)",
    secondary: null,
  }),
  "campaign": (obj) => ({
    displayName: obj.name || "(unnamed campaign)",
    secondary: null,
  }),
  "malware": (obj) => ({
    displayName: obj.name || "(unnamed malware)",
    secondary: (obj.malware_types || []).join(", ") || null,
  }),
  "tool": (obj) => ({
    displayName: obj.name || "(unnamed tool)",
    secondary: null,
  }),
  "vulnerability": (obj) => {
    const cve = (obj.external_references || []).find((r) => r.source_name === "cve")?.external_id;
    return {
      displayName: cve || obj.name || "(unnamed vulnerability)",
      secondary: cve ? obj.name : null,
    };
  },
  "identity": (obj) => ({
    displayName: obj.name || "(unnamed identity)",
    secondary: obj.identity_class || null,
  }),
  "location": (obj) => ({
    displayName: obj.name || obj.country || "(unnamed location)",
    secondary: obj.region || null,
  }),
  "infrastructure": (obj) => ({
    displayName: obj.name || "(unnamed infrastructure)",
    secondary: (obj.infrastructure_types || []).join(", ") || null,
  }),
  "report": (obj) => ({
    displayName: obj.name || "(unnamed report)",
    secondary: obj.published || null,
  }),
  // SCOs
  "process": (obj) => ({
    displayName: obj.command_line || "(no command line)",
    secondary: "process",
  }),
  "file": (obj) => {
    const hashEntry = obj.hashes && Object.entries(obj.hashes)[0];
    return {
      displayName: obj.name || (hashEntry ? `${hashEntry[0]}: ${hashEntry[1]}` : "(unnamed file)"),
      secondary: obj.name && hashEntry ? `${hashEntry[0]}: ${hashEntry[1]}` : null,
    };
  },
  "domain-name": (obj) => ({ displayName: obj.value || "(no value)", secondary: "domain" }),
  "ipv4-addr":   (obj) => ({ displayName: obj.value || "(no value)", secondary: "ipv4" }),
  "ipv6-addr":   (obj) => ({ displayName: obj.value || "(no value)", secondary: "ipv6" }),
  "url":         (obj) => ({ displayName: obj.value || "(no value)", secondary: "url" }),
  "email-addr":  (obj) => ({ displayName: obj.value || "(no value)", secondary: "email" }),
  "windows-registry-key": (obj) => ({ displayName: obj.key || "(no key)", secondary: "registry" }),
  "mutex":       (obj) => ({ displayName: obj.name || "(no name)", secondary: "mutex" }),
  "software":    (obj) => ({
    displayName: obj.name || "(unnamed software)",
    secondary: obj.version ? `v${obj.version}` : obj.cpe || null,
  }),
  "user-account": (obj) => ({
    displayName: obj.account_login || obj.user_id || "(unnamed account)",
    secondary: obj.account_type || null,
  }),
  // Detection layer
  "x-mitre-detection-strategy": (obj) => ({
    displayName: obj.name || "(unnamed detection strategy)",
    secondary: null,
  }),
  "x-mitre-analytic": (obj) => ({
    displayName: obj.name || "(unnamed analytic)",
    secondary: null,
  }),
  "x-log-source": (obj) => ({
    displayName: obj.name || "(unnamed log source)",
    secondary: obj.x_data_component || null,
  }),
  "indicator": (obj) => ({
    displayName: obj.name || "(unnamed indicator)",
    secondary: obj.pattern_type || null,
  }),
  // Attack Flow control nodes. These carry no `name`, so without an
  // explicit resolver they fall through to the truncated-UUID branch.
  "attack-operator": (obj) => ({
    displayName: `${obj.operator || "OR"} operator`,
    secondary: "flow control",
  }),
  "attack-condition": (obj) => ({
    displayName: obj.description || "(unnamed condition)",
    secondary: "flow control",
  }),
  "attack-flow": (obj) => ({
    displayName: obj.name || "(unnamed attack flow)",
    secondary: null,
  }),
};

/**
 * STIX types that participate in the ATT&CK Flow sequencing graph.
 *
 * Operators (AND/OR/XOR) and conditions are flow-control nodes that
 * PRECEDES chains route *through* — the serializer's
 * `route_precedes_through_operators` emits `A_proc → operator → B_proc`
 * rather than a direct `A_proc → B_proc`. Filtering the graph to
 * x-procedure only therefore severs the chain at every branch/converge
 * point and orphans everything downstream of it.
 */
export const FLOW_NODE_TYPES = new Set([
  "x-procedure",
  "attack-operator",
  "attack-condition",
]);

function extractTactic(attackPattern) {
  for (const phase of attackPattern.kill_chain_phases || []) {
    if (phase.kill_chain_name === "mitre-attack" && phase.phase_name) {
      return phase.phase_name;
    }
  }
  return null;
}

/** Build a {id → ResolvedRef} index for every embeddable object in the bundle. */
export function buildResolutionIndex(bundle) {
  const index = new Map();
  if (!bundle || !Array.isArray(bundle.objects)) return index;
  for (const obj of bundle.objects) {
    if (!obj || !obj.id) continue;
    const resolver = TYPE_RESOLVERS[obj.type];
    if (!resolver) {
      // Unknown type: still record so the panel can at least show the type.
      index.set(obj.id, {
        rawId: obj.id,
        type: obj.type || "unknown",
        displayName: obj.name || obj.id.slice(0, 24) + "...",
        secondary: null,
      });
      continue;
    }
    const { displayName, secondary } = resolver(obj);
    index.set(obj.id, {
      rawId: obj.id,
      type: obj.type,
      displayName,
      secondary,
    });
  }
  return index;
}

/** Resolve a single ref. Falls back to a truncated-UUID label so the
 *  panel never renders a blank slot. */
export function resolveRef(index, refId) {
  if (!refId) return null;
  const hit = index.get(refId);
  if (hit) return hit;
  // External or missing — render as a typed-prefix + short suffix so
  // the analyst still sees "something" without a wall of UUID.
  const dashIdx = refId.indexOf("--");
  const type = dashIdx > 0 ? refId.slice(0, dashIdx) : "unknown";
  const tail = dashIdx > 0 ? refId.slice(dashIdx + 2, dashIdx + 10) : refId.slice(0, 8);
  return {
    rawId: refId,
    type,
    displayName: `${type} ${tail}…`,
    secondary: "(not embedded in bundle)",
    external: true,
  };
}

/** Resolve a list of refs. Filters out null/empty entries silently. */
export function resolveRefs(index, refList) {
  if (!Array.isArray(refList)) return [];
  return refList
    .map((r) => resolveRef(index, r))
    .filter(Boolean);
}

/** Convenience: derive the precedes-edge graph from the bundle.
 *  Sequencing is expressed entirely by PRECEDES SROs (x_effect_refs was
 *  removed from x-procedure in v0.5.0-draft).
 *
 *  Nodes are every flow participant — procedures plus the
 *  attack-operator / attack-condition nodes the chain routes through
 *  (see FLOW_NODE_TYPES). Keeping only x-procedure would drop the
 *  operator hops and split one chain into disconnected fragments.
 *
 *  Returns {nodes: flow objects, edges: [{source, target}]}. */
export function extractFlowGraph(bundle) {
  if (!bundle || !Array.isArray(bundle.objects)) {
    return { nodes: [], edges: [] };
  }
  const nodes = bundle.objects.filter((o) => FLOW_NODE_TYPES.has(o.type));
  const nodeIds = new Set(nodes.map((n) => n.id));
  const edgeSet = new Set();
  const edges = [];

  // PRECEDES SROs
  for (const o of bundle.objects) {
    if (o.type !== "relationship") continue;
    if (o.relationship_type !== "precedes") continue;
    const src = o.source_ref;
    const tgt = o.target_ref;
    if (!src || !tgt) continue;
    if (!nodeIds.has(src) || !nodeIds.has(tgt)) continue;
    const key = `${src}→${tgt}`;
    if (edgeSet.has(key)) continue;
    edgeSet.add(key);
    edges.push({ source: src, target: tgt });
  }

  return { nodes, edges };
}
