#!/usr/bin/env python3
"""
Load an ATT&CK Enterprise STIX bundle directly into Neo4j.

Single-script loader: reads the bundle JSON, maps STIX types to Neo4j labels,
and executes batched MERGE/CREATE queries via the neo4j driver. The bundle
version is determined by whichever file you pass via --bundle.

Usage:
    python3 scripts/load_attack.py \
        --bundle data/attack/enterprise-attack-19.2.json \
        --uri bolt://localhost:7687 \
        --user neo4j \
        --password <password>

    # Dry run (parse only, no Neo4j connection):
    python3 scripts/load_attack.py \
        --bundle data/attack/enterprise-attack-19.2.json \
        --dry-run
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from neo4j import GraphDatabase
except ImportError:
    print("ERROR: neo4j driver not installed. Run: pip3 install neo4j")
    sys.exit(1)


# ── Type mapping ────────────────────────────────────────────────────────────

TYPE_MAP = {
    "attack-pattern": {
        "label": "AttackPattern",
        "props": ["name", "description", "created", "modified"],
        "serialize": ["external_references", "x_mitre_platforms", "kill_chain_phases"],
        "extra": {"stix_type": "attack-pattern"},
    },
    "x-mitre-tactic": {
        "label": "Tactic",
        "props": ["name", "description", "created", "modified", "x_mitre_shortname"],
        "serialize": ["external_references"],
        "extra": {"stix_type": "x-mitre-tactic"},
    },
    "intrusion-set": {
        "label": "IntrusionSet",
        "props": ["name", "description", "created", "modified"],
        "serialize": ["external_references", "aliases"],
        "extra": {"stix_type": "intrusion-set"},
    },
    "malware": {
        "label": "Malware",
        "props": ["name", "description", "created", "modified", "is_family"],
        "serialize": ["external_references", "malware_types"],
        "extra": {"stix_type": "malware"},
    },
    "tool": {
        "label": "Tool",
        "props": ["name", "description", "created", "modified"],
        "serialize": ["external_references", "x_mitre_platforms", "tool_types"],
        "extra": {"stix_type": "tool"},
    },
    "course-of-action": {
        "label": "CourseOfAction",
        "props": ["name", "description", "created", "modified"],
        "serialize": ["external_references"],
        "extra": {"stix_type": "course-of-action"},
    },
    "x-mitre-data-source": {
        "label": "DataSource",
        "props": ["name", "description", "created", "modified"],
        "serialize": ["external_references", "x_mitre_platforms", "x_mitre_collection_layers"],
        "extra": {"stix_type": "x-mitre-data-source"},
    },
    "x-mitre-data-component": {
        "label": "DataComponent",
        "props": ["name", "description", "created", "modified"],
        "serialize": ["external_references"],
        "extra": {"stix_type": "x-mitre-data-component"},
    },
    "campaign": {
        "label": "Campaign",
        "props": ["name", "description", "created", "modified", "first_seen", "last_seen"],
        "serialize": ["external_references"],
        "extra": {"stix_type": "campaign"},
    },
    "identity": {
        "label": "Identity",
        "props": ["name", "description", "created", "modified", "identity_class"],
        "serialize": ["external_references"],
        "extra": {"stix_type": "identity"},
    },
    "x-mitre-asset": {
        "label": "Asset",
        "props": ["name", "description", "created", "modified"],
        "serialize": ["external_references", "x_mitre_platforms", "x_mitre_sectors"],
        "extra": {"stix_type": "x-mitre-asset"},
    },
    "x-mitre-analytic": {
        "label": "Analytic",
        "props": ["name", "description", "created", "modified"],
        "serialize": ["external_references", "x_mitre_platforms"],
        "extra": {"stix_type": "x-mitre-analytic"},
    },
    "x-mitre-detection-strategy": {
        "label": "DetectionStrategy",
        "props": ["name", "description", "created", "modified"],
        "serialize": ["external_references"],
        "extra": {"stix_type": "x-mitre-detection-strategy"},
    },
}

SKIP_TYPES = {"extension-definition", "marking-definition", "x-mitre-matrix", "x-mitre-collection"}

REL_MAP = {
    "uses": "USES",
    "mitigates": "MITIGATES",
    "subtechnique-of": "SUBTECHNIQUE_OF",
    "detects": "DETECTS",
    "attributed-to": "ATTRIBUTED_TO",
    "targets": "TARGETS",
    "revoked-by": "REVOKED_BY",
    "related-to": "RELATED_TO",
    "has-analytic": "HAS_ANALYTIC",
    "uses-data-component": "USES_DATA_COMPONENT",
}


# ── Object extraction ───────────────────────────────────────────────────────

def extract_props(obj: dict, type_cfg: dict) -> dict:
    """Pull mapped properties from a STIX object into a flat dict for Neo4j."""
    result = {"stix_id": obj["id"]}

    for key in type_cfg["props"]:
        val = obj.get(key)
        if val is not None:
            result[key] = val

    for key in type_cfg.get("serialize", []):
        val = obj.get(key)
        if val is not None:
            result[key] = json.dumps(val)

    for key, val in type_cfg.get("extra", {}).items():
        result[key] = val

    # Promote a few flat top-level fields that downstream Cypher queries need
    # but that STIX expresses as nested / scattered structures:
    #   mitre_id ← external_references[source_name='mitre-attack'].external_id
    #   x_mitre_is_subtechnique / revoked / x_mitre_deprecated ← raw bool flags
    # Without these, lookups like {mitre_id: 'T1059.001'} silently miss.
    for ref in obj.get("external_references", []) or []:
        if ref.get("source_name") == "mitre-attack" and ref.get("external_id"):
            result["mitre_id"] = ref["external_id"]
            break
    for flag in ("x_mitre_is_subtechnique", "revoked", "x_mitre_deprecated"):
        if flag in obj:
            result[flag] = bool(obj[flag])

    return result


# ── Cypher builders ─────────────────────────────────────────────────────────

def build_node_cypher(label: str, sample: dict) -> str:
    """Build MERGE cypher for a node type using UNWIND.

    ON MATCH refreshes every property, not just `modified`. The earlier version
    only touched `modified`, which meant re-running the loader against a
    populated graph created the NEW objects in a release but left the CHANGED
    ones carrying their old name, description and aliases — a partial upgrade
    that looked like a complete one. Refreshing makes a reload the whole
    operation, which is what lets an ATT&CK version bump stay a one-command job
    once the graph holds pipeline output that a wipe would take with it.
    """
    props = sorted(k for k in sample.keys() if k != "stix_id")
    set_clause = ", ".join(f"n.{k} = obj.{k}" for k in props)
    return (
        f"UNWIND $objects AS obj\n"
        f"MERGE (n:{label}:STIXObject {{stix_id: obj.stix_id}})\n"
        f"ON CREATE SET {set_clause}\n"
        f"ON MATCH SET {set_clause}"
    )


def build_rel_cypher(neo4j_type: str) -> str:
    """Build relationship MERGE cypher using UNWIND.

    MERGE, not CREATE. With CREATE, running this loader against an already
    populated graph duplicates every edge it touches — which happened once
    and cost a full wipe and reload to undo. MERGE makes a
    reload idempotent, and idempotent is the property that matters once the
    graph holds pipeline output, because then the wipe is no longer free.
    """
    return (
        f"UNWIND $rels AS r\n"
        f"MATCH (src:STIXObject {{stix_id: r.source_ref}})\n"
        f"MATCH (tgt:STIXObject {{stix_id: r.target_ref}})\n"
        f"MERGE (src)-[:{neo4j_type}]->(tgt)"
    )


BELONGS_TO_TACTIC_CYPHER = """
UNWIND $mappings AS m
MATCH (ap:AttackPattern:STIXObject {stix_id: m.ap_id})
MATCH (t:Tactic {x_mitre_shortname: m.tactic_shortname})
MERGE (ap)-[:BELONGS_TO_TACTIC]->(t)
"""

HAS_ANALYTIC_CYPHER = """
UNWIND $pairs AS p
MATCH (ds:DetectionStrategy:STIXObject {stix_id: p.ds_id})
MATCH (a:Analytic:STIXObject {stix_id: p.analytic_id})
MERGE (ds)-[:HAS_ANALYTIC]->(a)
"""

USES_DATA_COMPONENT_CYPHER = """
UNWIND $pairs AS p
MATCH (a:Analytic:STIXObject {stix_id: p.analytic_id})
MATCH (dc:DataComponent:STIXObject {stix_id: p.dc_id})
MERGE (a)-[:USES_DATA_COMPONENT]->(dc)
"""

INDEX_QUERIES = [
    # A uniqueness constraint backs its own index, so it supersedes the plain
    # stix_id index AND makes a double-MERGE impossible. Neo4j refuses to
    # create it while the plain index still exists, so the drop is ordered
    # first and is not optional. Safe on this data: zero duplicate stix_ids.
    "DROP INDEX stix_id_idx IF EXISTS",
    "CREATE CONSTRAINT stix_id_unique IF NOT EXISTS "
    "FOR (n:STIXObject) REQUIRE n.stix_id IS UNIQUE",
    "CREATE INDEX tactic_shortname_idx IF NOT EXISTS FOR (n:Tactic) ON (n.x_mitre_shortname)",
    # The technique pivot looks techniques up by ATT&CK ID, not by stix_id, and
    # had no index at all.
    "CREATE INDEX mitre_id_idx IF NOT EXISTS FOR (n:AttackPattern) ON (n.mitre_id)",
    # Exact behavioural grouping of procedures (hash of techniques + platforms
    # + tactics). Complements the overlap join, which is a different question.
    "CREATE INDEX procedure_fingerprint_idx IF NOT EXISTS "
    "FOR (n:Procedure) ON (n.x_fingerprint)",
]


# The collection object states which ATT&CK version the bundle is. Nothing
# recorded it before, so the graph could not say what it held — the drift from
# 19.0 to 19.2 was only detectable by counting techniques against the website.
COLLECTION_CYPHER = """
MERGE (c:AttackCollection:STIXObject {stix_id: $stix_id})
SET c.name = $name,
    c.stix_type = 'x-mitre-collection',
    c.x_mitre_version = $version,
    c.modified = $modified,
    c.loaded_at = datetime(),
    c.bundle_file = $bundle_file
"""


# ── Main loader ─────────────────────────────────────────────────────────────

def load(bundle_path: Path, uri: str, user: str, password: str, dry_run: bool):
    # Parse bundle
    print(f"[1/6] Loading bundle: {bundle_path.name}")
    with open(bundle_path) as f:
        bundle = json.load(f)
    all_objects = bundle.get("objects", [])
    print(f"       {len(all_objects)} total STIX objects")

    # Group by type
    by_type: dict[str, list[dict]] = {}
    relationships: list[dict] = []
    skipped_types: dict[str, int] = {}

    collection = None
    for obj in all_objects:
        t = obj.get("type", "")
        if t in SKIP_TYPES:
            # Skipped as a node type, but the collection object is the only
            # place the bundle states its own version, so keep a reference.
            if t == "x-mitre-collection":
                collection = obj
            skipped_types[t] = skipped_types.get(t, 0) + 1
            continue
        if t == "relationship":
            relationships.append(obj)
        elif t in TYPE_MAP:
            by_type.setdefault(t, []).append(obj)
        else:
            skipped_types[t] = skipped_types.get(t, 0) + 1

    print("[2/6] Parsed types:")
    for t in sorted(by_type):
        cfg = TYPE_MAP[t]
        print(f"       {cfg['label']:25s} {len(by_type[t]):5d}")
    print(f"       {'relationships':25s} {len(relationships):5d}")
    if skipped_types:
        print(f"       Skipped: {skipped_types}")

    # Extract properties
    print("[3/6] Extracting properties...")
    node_batches: list[tuple[str, str, list[dict]]] = []
    for stix_type in sorted(by_type):
        cfg = TYPE_MAP[stix_type]
        items = [extract_props(obj, cfg) for obj in by_type[stix_type]]
        cypher = build_node_cypher(cfg["label"], items[0])
        node_batches.append((cfg["label"], cypher, items))

    # Group relationships by type
    rel_batches: list[tuple[str, str, list[dict]]] = []
    rels_by_type: dict[str, list[dict]] = {}
    unmapped_rels: dict[str, int] = {}
    for rel in relationships:
        rt = rel.get("relationship_type", "unknown")
        if rt in REL_MAP:
            rels_by_type.setdefault(rt, []).append(rel)
        else:
            unmapped_rels[rt] = unmapped_rels.get(rt, 0) + 1

    for rt in sorted(rels_by_type):
        neo4j_type = REL_MAP[rt]
        pairs = [{"source_ref": r["source_ref"], "target_ref": r["target_ref"]} for r in rels_by_type[rt]]
        cypher = build_rel_cypher(neo4j_type)
        rel_batches.append((neo4j_type, cypher, pairs))

    if unmapped_rels:
        print(f"       WARN unmapped relationship types: {unmapped_rels}")

    # Derive BELONGS_TO_TACTIC from attack-pattern kill_chain_phases
    tactic_mappings = []
    for ap in by_type.get("attack-pattern", []):
        for phase in ap.get("kill_chain_phases", []):
            if phase.get("kill_chain_name") == "mitre-attack":
                tactic_mappings.append({
                    "ap_id": ap["id"],
                    "tactic_shortname": phase["phase_name"],
                })
    print(f"       Derived BELONGS_TO_TACTIC: {len(tactic_mappings)} mappings")

    # Derive HAS_ANALYTIC from detection-strategy.x_mitre_analytic_refs
    has_analytic_pairs = []
    for ds in by_type.get("x-mitre-detection-strategy", []):
        for analytic_id in ds.get("x_mitre_analytic_refs", []):
            has_analytic_pairs.append({
                "ds_id": ds["id"],
                "analytic_id": analytic_id,
            })
    print(f"       Derived HAS_ANALYTIC: {len(has_analytic_pairs)} mappings")

    # Derive USES_DATA_COMPONENT from analytic.x_mitre_log_source_references
    uses_dc_pairs_set = set()
    for an in by_type.get("x-mitre-analytic", []):
        for ls in an.get("x_mitre_log_source_references", []):
            dc_ref = ls.get("x_mitre_data_component_ref")
            if dc_ref:
                uses_dc_pairs_set.add((an["id"], dc_ref))
    uses_dc_pairs = [{"analytic_id": a, "dc_id": d} for (a, d) in uses_dc_pairs_set]
    print(f"       Derived USES_DATA_COMPONENT (analytic→dc): {len(uses_dc_pairs)} mappings")

    derived_count = (
        (1 if tactic_mappings else 0)
        + (1 if has_analytic_pairs else 0)
        + (1 if uses_dc_pairs else 0)
    )
    total_ops = len(node_batches) + len(rel_batches) + derived_count + 1  # +1 for indexes
    print(f"[4/6] Total batches to execute: {total_ops}")

    if dry_run:
        print("\n  DRY RUN complete. No Neo4j connection made.")
        return

    # Connect and execute
    print(f"[5/6] Connecting to {uri}...")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    try:
        driver.verify_connectivity()
        print("       Connected.")
    except Exception as e:
        print(f"ERROR: Cannot connect: {e}")
        sys.exit(1)

    step = 0
    t0 = time.time()

    try:
        with driver.session() as session:
            # Indexes first
            step += 1
            print(f"  [{step}/{total_ops}] Creating indexes + constraint...")
            for q in INDEX_QUERIES:
                session.run(q)

            if collection is not None:
                session.run(
                    COLLECTION_CYPHER,
                    stix_id=collection["id"],
                    name=collection.get("name", "Enterprise ATT&CK"),
                    version=collection.get("x_mitre_version", ""),
                    modified=collection.get("modified", ""),
                    bundle_file=bundle_path.name,
                )
                print(f"       version marker: "
                      f"{collection.get('name')} v{collection.get('x_mitre_version')}")

            # Node batches
            for label, cypher, items in node_batches:
                step += 1
                t1 = time.time()
                session.execute_write(lambda tx, c=cypher, i=items: tx.run(c, objects=i))
                elapsed = time.time() - t1
                print(f"  [{step}/{total_ops}] {label:25s} {len(items):5d} nodes  ({elapsed:.1f}s)")

            # Relationship batches
            for neo4j_type, cypher, pairs in rel_batches:
                step += 1
                t1 = time.time()
                session.execute_write(lambda tx, c=cypher, p=pairs: tx.run(c, rels=p))
                elapsed = time.time() - t1
                print(f"  [{step}/{total_ops}] {neo4j_type:25s} {len(pairs):5d} rels   ({elapsed:.1f}s)")

            # BELONGS_TO_TACTIC
            if tactic_mappings:
                step += 1
                t1 = time.time()
                session.execute_write(
                    lambda tx: tx.run(BELONGS_TO_TACTIC_CYPHER, mappings=tactic_mappings)
                )
                elapsed = time.time() - t1
                print(f"  [{step}/{total_ops}] {'BELONGS_TO_TACTIC':25s} {len(tactic_mappings):5d} rels   ({elapsed:.1f}s)")

            # HAS_ANALYTIC (derived from detection-strategy.x_mitre_analytic_refs)
            if has_analytic_pairs:
                step += 1
                t1 = time.time()
                session.execute_write(
                    lambda tx: tx.run(HAS_ANALYTIC_CYPHER, pairs=has_analytic_pairs)
                )
                elapsed = time.time() - t1
                print(f"  [{step}/{total_ops}] {'HAS_ANALYTIC':25s} {len(has_analytic_pairs):5d} rels   ({elapsed:.1f}s)")

            # USES_DATA_COMPONENT (derived from analytic.x_mitre_log_source_references)
            if uses_dc_pairs:
                step += 1
                t1 = time.time()
                session.execute_write(
                    lambda tx: tx.run(USES_DATA_COMPONENT_CYPHER, pairs=uses_dc_pairs)
                )
                elapsed = time.time() - t1
                print(f"  [{step}/{total_ops}] {'USES_DATA_COMPONENT':25s} {len(uses_dc_pairs):5d} rels   ({elapsed:.1f}s)")

    finally:
        driver.close()

    total_elapsed = time.time() - t0
    print(f"\n[6/6] Done in {total_elapsed:.1f}s")

    # Final counts
    print(f"\n{'=' * 50}")
    total_nodes = sum(len(items) for _, _, items in node_batches)
    total_rels = sum(len(pairs) for _, _, pairs in rel_batches) + len(tactic_mappings)
    print(f"Nodes created/merged: {total_nodes}")
    print(f"Relationships merged: {total_rels}")
    if collection is not None:
        print(f"ATT&CK version:       v{collection.get('x_mitre_version')} "
              f"({collection.get('modified', '')[:10]})")


def main():
    parser = argparse.ArgumentParser(description="Load ATT&CK STIX bundle into Neo4j")
    parser.add_argument("--bundle", required=True, help="Path to enterprise-attack JSON")
    parser.add_argument("--uri", default="bolt://localhost:7687", help="Neo4j bolt URI")
    parser.add_argument("--user", default="neo4j", help="Neo4j user")
    parser.add_argument("--password", default="neo4j", help="Neo4j password")
    parser.add_argument("--dry-run", action="store_true", help="Parse only, no execution")
    args = parser.parse_args()

    load(
        bundle_path=Path(args.bundle),
        uri=args.uri,
        user=args.user,
        password=args.password,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
