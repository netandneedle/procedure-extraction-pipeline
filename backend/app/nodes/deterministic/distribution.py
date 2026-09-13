"""distribute node: persistence stage of the extraction pipeline.

Writes a validated STIX 2.1 bundle to Neo4j as graph nodes
and relationships.

STIX -> NEO4J MAPPING:
    - Each STIX object becomes a Neo4j node with its type as the label
    - SDO properties map to node properties
    - SCO properties map to node properties
    - SROs become Neo4j relationships between nodes
    - The bundle itself is NOT stored as a node

NODE LABELS and RELATIONSHIP TYPES (Neo4j):
    The full mappings are `_STIX_TYPE_TO_LABEL` (36 labels, e.g. Procedure,
    IntrusionSet, Malware, Tool, File, Process) and `_REL_TYPE_TO_NEO4J`
    (21 STIX relationship types onto 20 Neo4j types, e.g. USES, PRECEDES,
    IMPLEMENTS_TECHNIQUE, HAS_OBSERVABLE)
    below. Catalogue-owned objects (`CATALOGUE_OWNED_TYPES`) are matched,
    never written.

Writes go through the async Neo4j driver, one transaction per batch.

WHAT THIS NODE READS:
    - stix_bundle: The validated STIX 2.1 bundle
    - validation_results: Must all be True to proceed

WHAT THIS NODE WRITES:
    - neo4j_write_status: "success" or error message
    - objects_written: Count of objects written
    - status: DISTRIBUTING or COMPLETED or FAILED
    - current_node: "distribute"
"""

from __future__ import annotations

import logging
import re

from app.config import settings
from app.graph.state import PipelineState, PipelineStatus, resolve_display_title
from app.models.base import async_session
from app.nodes.deterministic.extension_definitions import bundle_meta_ids
from app.services import bundle_store
from app.services.neo4j import write_transaction, write_transaction_counted

logger = logging.getLogger(__name__)

# Cypher identifier whitelist: STIX property names interpolated into
# query strings (e.g. `n.{k} = ${k}`) must match this pattern so a
# crafted property key can't break out of the SET clause. STIX spec
# requires property names to be lowercase letters/digits/underscore,
# optionally with an `x_` custom prefix, so this is strict by intent.
_SAFE_CYPHER_KEY = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Max length for an interpolated Cypher relationship-type identifier. Real
# ATT&CK rel types are short (USES, ATTRIBUTED_TO, ...); this bounds a
# DoS-via-giant-query from a hostile relationship_type.
_MAX_REL_TYPE_LEN = 64


def _sanitize_props(props: dict, *, context: str) -> dict:
    """Drop any property whose key would be unsafe to interpolate into
    Cypher. Logs a warning so we catch extraction bugs or injection
    attempts rather than silently dropping data."""
    safe: dict = {}
    for k, v in props.items():
        if _SAFE_CYPHER_KEY.match(k):
            safe[k] = v
        else:
            logger.warning(
                "distribute: dropping unsafe property key %r in %s",
                k, context,
            )
    return safe

# Mapping from STIX type to Neo4j node label
_STIX_TYPE_TO_LABEL: dict[str, str] = {
    # SDOs
    "attack-pattern": "AttackPattern",
    "identity": "Identity",
    "intrusion-set": "IntrusionSet",
    "threat-actor": "ThreatActor",
    "malware": "Malware",
    "tool": "Tool",
    "campaign": "Campaign",
    "vulnerability": "Vulnerability",
    "location": "Location",
    "indicator": "Indicator",
    "infrastructure": "Infrastructure",
    "x-procedure": "Procedure",
    "course-of-action": "CourseOfAction",
    "report": "Report",
    "marking-definition": "MarkingDefinition",
    # Custom extension SDOs (v0.5.0-draft)
    "x-log-source": "LogSource",
    "attack-flow": "AttackFlow",
    "attack-operator": "AttackOperator",
    "attack-condition": "AttackCondition",
    # ATT&CK detection chain SDOs (emitted by _enrich_detection_chain in
    # serialization.py). Without these labels the detect/has-analytic/
    # uses-data-component SROs would dangle in Neo4j.
    "x-mitre-detection-strategy": "DetectionStrategy",
    "x-mitre-analytic": "Analytic",
    "x-mitre-data-component": "DataComponent",
    "x-mitre-data-source": "DataSource",
    # SCOs
    "ipv4-addr": "IPv4Addr",
    "ipv6-addr": "IPv6Addr",
    "domain-name": "DomainName",
    "url": "URL",
    "email-addr": "EmailAddr",
    "file": "File",
    "windows-registry-key": "WindowsRegistryKey",
    "network-traffic": "NetworkTraffic",
    "process": "Process",
    "directory": "Directory",
    "mutex": "Mutex",
    "software": "Software",
    "user-account": "UserAccount",
}

# Mapping from STIX relationship_type to Neo4j relationship type
_REL_TYPE_TO_NEO4J: dict[str, str] = {
    "uses": "USES",
    "precedes": "PRECEDES",
    "attributed-to": "ATTRIBUTED_TO",
    "indicates": "INDICATES",
    "related-to": "RELATED_TO",
    "targets": "TARGETS",
    "mitigates": "MITIGATES",
    "exploits": "EXPLOITS",
    "component-of": "COMPONENT_OF",
    "implements-technique": "IMPLEMENTS_TECHNIQUE",
    "belongs-to-tactic": "BELONGS_TO_TACTIC",
    "detects": "DETECTS",
    "has-analytic": "HAS_ANALYTIC",
    "uses-data-component": "USES_DATA_COMPONENT",
    "has-observable": "HAS_OBSERVABLE",
    "subtechnique-of": "SUBTECHNIQUE_OF",
    "authored-by": "AUTHORED_BY",
    "derived-from": "DERIVED_FROM",
    "duplicate-of": "DUPLICATE_OF",
    "located-at": "LOCATED_AT",
    "uses-technique": "IMPLEMENTS_TECHNIQUE",
}

# Properties to skip when writing to Neo4j (handled separately)
_SKIP_PROPERTIES = {"type", "spec_version", "id", "source_ref", "target_ref", "relationship_type"}

# STIX types the ATT&CK catalogue owns. The serializer re-emits these so the
# bundle stands alone — one real bundle carried 241 of them — but their
# stix_ids are the catalogue's, and `MERGE ... SET` would overwrite ATT&CK's
# own `created`/`modified`/`name`/`description` with the values from a pipeline
# run. For T1074.001 that meant replacing `created 2020-03-13` with the run
# timestamp, in a graph with no undo.
#
# Their nodes are therefore never written. Relationships pointing at them still
# MATCH the catalogue's own nodes, so nothing is lost from the graph.
CATALOGUE_OWNED_TYPES: frozenset[str] = frozenset({
    "attack-pattern",
    "x-mitre-detection-strategy",
    "x-mitre-analytic",
    "x-mitre-data-component",
})

# Stamped on every node this pipeline creates. Two jobs, both load-bearing:
# it is the exact discriminator for "did we put this here" (the catalogue's own
# marker properties are not reliable — one node has no `mitre_id`), and it is
# what stops an undo from deleting an ATT&CK technique that happens to be
# described by only one report.
PIPELINE_MARKER = "pipeline"


async def distribute(state: PipelineState) -> dict:
    """Stage 6b: Write validated STIX bundle to Neo4j.

    Checks validation results before writing. If any validation
    failed, skips the write and returns an error.

    When neo4j_writes_enabled is False (default), runs in dry-run
    mode: queries are built and counted but not executed. Set the
    env var NEO4J_WRITES_ENABLED=true to enable actual writes.

    Returns partial state update with write status and object count.
    """
    logger.info("distribute: starting")

    bundle = state.get("stix_bundle", {})
    validation_results = state.get("validation_results", {})

    # Pre-check: don't write invalid bundles
    if not all(validation_results.values()):
        failed = [k for k, v in validation_results.items() if not v]
        error_msg = f"Cannot distribute: validation failed for {', '.join(failed)}"
        logger.error("distribute: %s", error_msg)
        return {
            "neo4j_write_status": error_msg,
            "objects_written": 0,
            "status": PipelineStatus.FAILED.value,
            "current_node": "distribute",
            "error": error_msg,
        }

    objects = bundle.get("objects", [])
    if not objects:
        logger.warning("distribute: bundle has no objects")
        return {
            "neo4j_write_status": "success (empty bundle)",
            "objects_written": 0,
            "status": PipelineStatus.COMPLETED.value,
            "current_node": "distribute",
        }

    # Build Cypher statements for batch execution.
    node_queries, rel_queries, describes_queries = _build_cypher_statements(objects)

    logger.info(
        "distribute: prepared %d node queries + %d relationship queries "
        "+ %d describes queries",
        len(node_queries), len(rel_queries), len(describes_queries),
    )

    # Execute via the async Neo4j driver (or count without executing when
    # writes are disabled — see _execute_writes).
    try:
        write_result = await _execute_writes(
            node_queries, rel_queries, describes_queries,
        )
        written = write_result["written"]

        logger.info("distribute: wrote %d objects to Neo4j", written)

        # ── Persist bundle to database ──────────────────────────────
        persistence_errors: list[str] = []
        # A dropped edge is not fatal — the bundle has shipped and the rest of
        # the graph is correct — but it must not be silent either, so it rides
        # the channel that already renders as a warning badge on the card.
        if write_result["dropped"]:
            persistence_errors.append(
                f"{write_result['dropped']} graph relationship(s) dropped: "
                f"endpoint not in graph "
                f"({'; '.join(write_result['dropped_detail'][:3])}"
                f"{', ...' if write_result['dropped'] > 3 else ''})"
            )
        try:
            # Same ladder the serializer uses for the Report SDO, so the
            # bundle row and the Report node cannot disagree on the name.
            bundle_title = resolve_display_title(state, "Untitled Bundle")
            async with async_session() as db:
                await bundle_store.save_bundle(
                    db,
                    source_id=state.get("source_id"),
                    title=bundle_title,
                    bundle=bundle,
                    validation_results=validation_results,
                    source_file_path=state.get("raw_content_path"),
                    source_type=state.get("source_type"),
                    metadata=state.get("metadata", {}),
                    persistence_errors=persistence_errors,
                    bundle_corrections=state.get("bundle_corrections", []),
                )
        except (IOError, OSError) as e:
            # File I/O failure is expected (file deleted, permissions, etc.)
            msg = f"source file read failed: {type(e).__name__}: {e}"
            logger.warning("distribute: %s (non-fatal)", msg)
            persistence_errors.append(msg)
        except Exception as e:
            # DB or unexpected errors: log with traceback but don't fail pipeline
            msg = f"bundle persistence failed: {type(e).__name__}: {e}"
            logger.error("distribute: %s (non-fatal)", msg, exc_info=True)
            persistence_errors.append(msg)

        return {
            "neo4j_write_status": "success",
            "objects_written": written,
            "persistence_errors": persistence_errors,
            "status": PipelineStatus.COMPLETED.value,
            "current_node": "distribute",
        }

    except Exception as e:
        error_msg = f"Neo4j write failed: {type(e).__name__}: {e}"
        logger.exception("distribute: %s", error_msg)
        return {
            "neo4j_write_status": error_msg,
            "objects_written": 0,
            "status": PipelineStatus.FAILED.value,
            "current_node": "distribute",
            "error": error_msg,
        }


def _build_cypher_statements(
    objects: list[dict],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Convert STIX objects into Cypher MERGE statements.

    Returns (node_queries, relationship_queries, describes_queries). Each query
    is a dict with 'query' (Cypher string) and 'params' (parameter dict).

    Objects of a CATALOGUE_OWNED_TYPE get no node query — their nodes belong to
    ATT&CK and writing them would overwrite it. Relationships and DESCRIBES
    edges still reference them and MATCH the catalogue's own nodes.

    Bundle metadata gets no node query either: the extension-definition
    objects and the identities they name as author describe the object
    types the bundle is written in, not the intrusion it reports. They are
    not unmapped types (no warning) — they are deliberately not graph data,
    and the Report never lists them, so the undo boundary is unchanged.
    """
    node_queries: list[dict] = []
    rel_queries: list[dict] = []
    skipped_catalogue = 0
    meta_ids = bundle_meta_ids(objects)

    for obj in objects:
        obj_type = obj.get("type", "")

        if obj_type == "relationship":
            query = _build_rel_query(obj)
            if query:
                rel_queries.append(query)
            continue
        if obj_type in CATALOGUE_OWNED_TYPES:
            skipped_catalogue += 1
            continue
        if obj_type == "extension-definition" or obj.get("id") in meta_ids:
            continue
        query = _build_node_query(obj)
        if query:
            node_queries.append(query)

    if skipped_catalogue:
        logger.info(
            "distribute: %d catalogue-owned object(s) not written (ATT&CK owns "
            "them; their relationships still MATCH the existing nodes)",
            skipped_catalogue,
        )

    return node_queries, rel_queries, _build_describes_queries(objects)


def _build_describes_queries(objects: list[dict]) -> list[dict]:
    """Link the Report SDO to everything it contributed.

    The bundle's Report already lists every non-relationship object it carries
    in `object_refs`, so this is provenance that costs no new modelling: one
    hop answers "which report gave us this procedure" and, backwards, "which
    reports does this cluster span" — the question that makes a cross-source
    cluster mean anything. The same edges are the delete set for an undo.

    Edges are emitted to catalogue-owned objects too (a report does describe
    the techniques it covers), which is why the undo must filter on
    `x_ingested_by` and not on DESCRIBES alone.
    """
    reports = [o for o in objects if o.get("type") == "report" and o.get("id")]
    if not reports:
        return []

    present = {o.get("id") for o in objects}
    queries: list[dict] = []
    for report in reports:
        for ref in report.get("object_refs") or []:
            if not ref or ref not in present or ref == report["id"]:
                continue
            if ref.split("--")[0] == "relationship":
                continue
            queries.append({
                "query": (
                    "MATCH (r:Report:STIXObject {stix_id: $report_id}) "
                    "WITH r "
                    "MATCH (n:STIXObject {stix_id: $object_id}) "
                    "MERGE (r)-[:DESCRIBES]->(n) "
                    "RETURN 1"
                ),
                "params": {"report_id": report["id"], "object_id": ref},
            })
    return queries


def _build_node_query(obj: dict) -> dict | None:
    """Build a MERGE Cypher statement for a STIX object node.

    Uses MERGE (not CREATE) so pre-existing nodes (e.g., AttackPattern
    from ATT&CK load) are matched rather than duplicated.
    """
    stix_type = obj.get("type", "")
    label = _STIX_TYPE_TO_LABEL.get(stix_type)

    if not label:
        logger.warning("No Neo4j label for STIX type: %s", stix_type)
        return None

    stix_id = obj.get("id", "")
    if not stix_id:
        return None

    # Extract properties (skip meta-fields) and drop unsafe keys
    props = {k: v for k, v in obj.items() if k not in _SKIP_PROPERTIES and v is not None}
    props = _sanitize_props(props, context=f"node {stix_type} {stix_id}")

    # Flatten list properties to JSON strings for Neo4j compatibility
    for key, val in list(props.items()):
        if isinstance(val, list):
            if all(isinstance(item, str) for item in val):
                pass  # Neo4j handles string arrays natively
            else:
                import json
                props[key] = json.dumps(val)
        elif isinstance(val, dict):
            import json
            props[key] = json.dumps(val)

    # Three deliberate choices in this one statement:
    #
    #   :STIXObject     the catalogue's own label, applied by
    #                   scripts/load_attack.py to every node it writes. Without
    #                   it, pipeline nodes are invisible to every :STIXObject
    #                   query AND cannot use stix_id's index, which is scoped
    #                   to that label.
    #   n += $props     a map merge instead of interpolating each property name
    #                   into the query text. Keys become data, so the injection
    #                   surface `_sanitize_props` was written to guard is gone
    #                   (the filter stays as hygiene, and to catch extraction
    #                   bugs).
    #   ON MATCH guard  only ever overwrite a node this pipeline created.
    #                   CATALOGUE_OWNED_TYPES covers the collisions we know
    #                   about; this covers a future serializer change that
    #                   reuses an ATT&CK stix_id for something else — the case
    #                   that has no undo.
    query = (
        f"MERGE (n:{label}:STIXObject {{stix_id: $stix_id}}) "
        f"ON CREATE SET n += $props, n.stix_type = $stix_type, "
        f"n.x_ingested_by = $marker "
        f"ON MATCH SET n += CASE WHEN n.x_ingested_by = $marker "
        f"THEN $props ELSE {{}} END"
    )

    params = {
        "stix_id": stix_id,
        "props": props,
        "stix_type": stix_type,
        "marker": PIPELINE_MARKER,
    }

    return {"query": query, "params": params}


def _build_rel_query(obj: dict) -> dict | None:
    """Build a MERGE Cypher statement for a STIX relationship.

    Handles both bundle-internal refs and external refs (e.g.,
    attack-pattern IDs that already exist in Neo4j from ATT&CK load).
    Uses label hints when possible for index-backed lookups.
    """
    rel_type_stix = obj.get("relationship_type", "")
    rel_type = _REL_TYPE_TO_NEO4J.get(rel_type_stix, rel_type_stix.upper().replace("-", "_"))

    # A procedure's technique linkage arrives as a plain `uses` SRO, which is
    # correct STIX and useless as a graph edge: the catalogue already has
    # 18,000+ USES edges, and the pipeline adds procedure->tool and
    # procedure->malware under the same name. The clustering query joins two
    # procedures through a shared endpoint, so with USES it silently treats
    # "both used PsExec" as technique overlap.
    #
    # Renamed here rather than in the bundle. The linkage is already carried
    # twice (x_technique_refs AND the SRO), so a custom `uses-technique` in the
    # bundle would gain nothing and cost partner legibility — while a
    # graph-side name costs nothing, the same trade already made for
    # `stix_type` and `mitre_id`.
    if (
        rel_type_stix == "uses"
        and obj.get("source_ref", "").startswith("x-procedure--")
        and obj.get("target_ref", "").startswith("attack-pattern--")
    ):
        rel_type = "IMPLEMENTS_TECHNIQUE"

    # rel_type is interpolated into the query string, so enforce a safe
    # identifier pattern AND a length cap. _SAFE_CYPHER_KEY blocks unsafe
    # characters but matches arbitrarily long identifiers — a STIX object
    # with relationship_type = "x" * 1000 would yield a 1000-char Cypher
    # identifier (a DoS-via-giant-query, not injection). Real ATT&CK rel
    # types are short; 64 chars is generous headroom.
    if len(rel_type) > _MAX_REL_TYPE_LEN or not _SAFE_CYPHER_KEY.match(rel_type):
        logger.warning(
            "distribute: unsafe or oversized relationship_type %r, dropping rel %s",
            rel_type_stix, obj.get("id", ""),
        )
        return None

    source_ref = obj.get("source_ref", "")
    target_ref = obj.get("target_ref", "")

    if not source_ref or not target_ref:
        return None

    # An edge between two catalogue-owned objects is ATT&CK's own internal
    # structure — the detection chain, mostly — and the catalogue already has
    # it. The bundle re-emits it so it stands alone, and MERGE keys on
    # `stix_id`, which the catalogue's edges do not carry, so it cannot match
    # them and creates a parallel one instead. One bundle left 446 duplicate
    # DETECTS / HAS_ANALYTIC / USES_DATA_COMPONENT edges that way.
    #
    # Worse, they are unreachable by undo: DETACH DELETE only reaches edges on
    # a node we are deleting, and both endpoints here belong to ATT&CK. So
    # they are permanent, and they accumulate once per ingest.
    if (_catalogue_owned_ref(source_ref) and _catalogue_owned_ref(target_ref)):
        return None

    stix_id = obj.get("id", "")

    # Extract relationship properties and drop unsafe keys
    props = {k: v for k, v in obj.items() if k not in _SKIP_PROPERTIES and v is not None}
    props = _sanitize_props(props, context=f"rel {rel_type_stix} {stix_id}")

    prop_setters = ""
    if props:
        prop_setters = " SET " + ", ".join(f"r.{k} = ${k}" for k in props)

    # Use label hints for known STIX ID prefixes (index-backed lookup)
    source_match = _label_hint_match("a", "source_ref", source_ref)
    target_match = _label_hint_match("b", "target_ref", target_ref)

    # RETURN 1 so the caller can tell a written edge from one whose endpoints
    # did not exist — without it the MATCH yields no rows, the MERGE does
    # nothing, and the transaction commits clean.
    # Two MATCH clauses with a WITH between them, not one comma-joined clause.
    # Both endpoints are independent index lookups, so a comma reads to the
    # planner as a disconnected pattern and it warns about a cartesian product
    # on every edge — 1124 notifications for one bundle. Same plan, no noise.
    query = (
        f"MATCH {source_match} "
        f"WITH a "
        f"MATCH {target_match} "
        f"MERGE (a)-[r:{rel_type} {{stix_id: $stix_id}}]->(b)"
        f"{prop_setters} "
        f"RETURN 1"
    )

    params = {
        "stix_id": stix_id,
        "source_ref": source_ref,
        "target_ref": target_ref,
        **props,
    }

    return {"query": query, "params": params}


def _catalogue_owned_ref(stix_ref: str) -> bool:
    """True when a STIX ID names an object the ATT&CK catalogue owns."""
    return stix_ref.split("--")[0] in CATALOGUE_OWNED_TYPES


def _label_hint_match(alias: str, param_name: str, stix_ref: str) -> str:
    """Generate an index-backed MATCH clause for a relationship endpoint.

    Matches on `:STIXObject`, which is where the stix_id index actually lives —
    the earlier version guessed a type label from the STIX ID prefix
    (`(a:AttackPattern {stix_id: ...})`), which no index serves, so every
    endpoint lookup was a label scan. scripts/load_attack.py has always matched
    this way; this brings the write path in line with it.

    Args:
        alias: Cypher variable name (e.g., "a", "b")
        param_name: Parameter name in the query (e.g., "source_ref")
        stix_ref: kept for signature stability; the label is no longer derived
    """
    del stix_ref
    return f"({alias}:STIXObject {{stix_id: ${param_name}}})"


async def _execute_writes(
    node_queries: list[dict],
    rel_queries: list[dict],
    describes_queries: list[dict] | None = None,
) -> dict:
    """Execute Cypher queries against Neo4j.

    Three passes in order, because each depends on the last: nodes, then the
    relationships between them, then the Report's DESCRIBES edges.

    Returns {"written", "dropped", "dropped_detail"}. `dropped` counts edge
    queries whose MATCH found nothing — an endpoint that is not in the graph.
    Cypher treats that as success, so without counting rows it is invisible;
    with real bundles it should stay at zero, and a non-zero value means the
    catalogue and the pipeline have drifted apart on ATT&CK versions.

    When writes are disabled, returns the prepared counts without executing.
    """
    describes_queries = describes_queries or []
    total = len(node_queries) + len(rel_queries) + len(describes_queries)

    if not settings.neo4j_writes_enabled:
        logger.info(
            "distribute: DRY RUN - %d nodes + %d relationships + %d describes "
            "prepared (set NEO4J_WRITES_ENABLED=true to execute)",
            len(node_queries), len(rel_queries), len(describes_queries),
        )
        return {"written": total, "dropped": 0, "dropped_detail": []}

    written = 0
    dropped_detail: list[str] = []

    if node_queries:
        written += await write_transaction(node_queries)
        logger.info("distribute: wrote %d node queries", len(node_queries))

    for label, queries in (("relationship", rel_queries),
                           ("describes", describes_queries)):
        if not queries:
            continue
        counts = await write_transaction_counted(queries)
        landed = sum(1 for c in counts if c)
        written += landed
        for q, c in zip(queries, counts):
            if c:
                continue
            p = q["params"]
            detail = (
                f"{p.get('stix_id') or label} "
                f"{p.get('source_ref') or p.get('report_id')} -> "
                f"{p.get('target_ref') or p.get('object_id')}"
            )
            dropped_detail.append(detail)
            logger.warning(
                "distribute: %s dropped, endpoint not in graph — %s",
                label, detail,
            )
        logger.info(
            "distribute: %d/%d %s queries landed",
            landed, len(queries), label,
        )

    return {
        "written": written,
        "dropped": len(dropped_detail),
        "dropped_detail": dropped_detail[:20],
    }
