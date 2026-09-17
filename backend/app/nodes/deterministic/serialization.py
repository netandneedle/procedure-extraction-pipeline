"""serialize_stix node: Stage 6a of the extraction pipeline.

Converts approved, normalized procedure drafts + entities + detection
rules into a valid STIX 2.1 bundle.

STIX OBJECT MAPPING:
    - ProcedureDraft -> x-procedure SDO (custom extension)
    - Entity (THREAT_ACTOR) -> intrusion-set SDO
    - Entity (MALWARE) -> malware SDO
    - Entity (TOOL) -> tool SDO
    - Entity (CAMPAIGN) -> campaign SDO
    - Entity (IOC_*) -> Observable SCOs (ipv4-addr, domain-name, file, etc.)
    - Entity (VICTIM_SECTOR, VICTIM_GEO) -> identity SDO
    - DetectionRule -> indicator SDO (source-provided only)
    - Relationships -> SROs (uses, indicates, attributed-to, etc.)
    - Sequencing -> PRECEDES SROs + one attack-flow SDO (never embedded
      on the x-procedure)

VALIDATION:
    Four validators run on the assembled bundle:
    1. Schema validation: STIX 2.1 JSON schema + x-procedure v0.5.0-draft
    2. Reference integrity: all id refs resolve within the bundle
    3. ATT&CK Flow: PRECEDES SRO endpoints resolve, sequencing DAG acyclic,
       at most one attack-flow object
    4. Tuple semantics: P = { AP ≠ ∅, LS ≠ ∅, ⟨C⟩ ≠ ∅ } per v0.5.0-draft

WHAT THIS NODE READS:
    - normalized_drafts, validated_entities, detection_rules,
      gate2_decision, metadata, source_id

WHAT THIS NODE WRITES:
    - stix_bundle, validation_results, validation_errors,
      status, current_node
"""

from __future__ import annotations

import ipaddress
import logging
import re
import uuid
from datetime import datetime, timezone

# Matches a full STIX attack-pattern ID whose suffix is a real UUIDv4.
# Used to distinguish real ATT&CK STIX IDs from fabricated T-number
# fallbacks (e.g. "attack-pattern--T1059.001") in enrichment paths.
_ATTACK_PATTERN_UUID_RE = re.compile(
    r"^attack-pattern--[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)

from app.nodes.deterministic.relationship_keys import (
    preview_relationship_key,
    relationship_key,
)
from app.graph.state import (
    EntityType,
    GateAction,
    PipelineState,
    PipelineStatus,
    resolve_display_title,
)
from app.nodes.deterministic.attack_conditions import build_attack_condition_sdos
from app.nodes.deterministic.attack_operators import (
    build_attack_operator_sdos,
    route_precedes_through_operators,
)
from app.nodes.deterministic.extension_definitions import (
    ATTACK_FLOW_EXTENSION_ID,
    ATTACK_FLOW_TYPES,
    X_PROCEDURE_EXTENSION_ID,
    attack_flow_author_identity,
    attack_flow_extension_definition,
    bundle_meta_ids,
    extension_declaration,
    x_procedure_author_identity,
    x_procedure_extension_definition,
)
from app.services import stix_schema
from app.services.neo4j import run_query

logger = logging.getLogger(__name__)

# STIX 2.1 spec version
STIX_SPEC_VERSION = "2.1"

# Custom STIX type for procedures
X_PROCEDURE_TYPE = "x-procedure"

# Mapping from EntityType to STIX object type.
# IOC_IP is handled dynamically (ipv4-addr vs ipv6-addr) in _entity_to_stix.
_ENTITY_TO_STIX_TYPE: dict[str, str] = {
    # SDOs
    EntityType.INTRUSION_SET.value: "intrusion-set",
    EntityType.THREAT_ACTOR.value: "threat-actor",
    EntityType.MALWARE.value: "malware",
    EntityType.TOOL.value: "tool",
    EntityType.CAMPAIGN.value: "campaign",
    EntityType.VULNERABILITY.value: "vulnerability",
    EntityType.ORGANIZATION.value: "identity",
    EntityType.LOCATION.value: "location",
    EntityType.VICTIM_SECTOR.value: "identity",
    EntityType.INFRASTRUCTURE.value: "infrastructure",
    # SCOs
    EntityType.IOC_HASH.value: "file",
    EntityType.IOC_IP.value: "ipv4-addr",  # Default; overridden for IPv6
    EntityType.IOC_DOMAIN.value: "domain-name",
    EntityType.IOC_URL.value: "url",
    EntityType.IOC_EMAIL.value: "email-addr",
    EntityType.IOC_FILE_PATH.value: "file",
    EntityType.IOC_REGISTRY_KEY.value: "windows-registry-key",
    EntityType.IOC_MUTEX.value: "mutex",
    # A bare executable filename is data at rest, so it is a File SCO, not a
    # Process. STIX Process describes a live execution instance (command_line,
    # pid) and points at its binary via image_ref -> File. Qt6Core.dll makes
    # the distinction concrete: a DLL is loaded INTO a process and can never
    # be one. Unmapped, these entities were extracted, reviewed at gate_0,
    # and then silently dropped by _entity_to_stix.
    EntityType.IOC_PROCESS_NAME.value: "file",
    EntityType.SOFTWARE.value: "software",
    EntityType.USER_ACCOUNT.value: "user-account",
}

# STIX types that are SCOs (not SDOs). These don't get created/modified timestamps.
_SCO_TYPES = {
    "ipv4-addr", "ipv6-addr", "domain-name", "url", "email-addr",
    "file", "windows-registry-key", "network-traffic", "process",
    "directory", "mutex", "software", "user-account",
}


async def serialize_stix(state: PipelineState) -> dict:
    """Stage 6a: Build and validate the STIX 2.1 bundle.

    Assembles all pipeline outputs into STIX objects, creates
    relationships, validates the bundle, and returns it.

    Async because it queries Neo4j for log source enrichment
    (DataComponents that detect each procedure's ATT&CK techniques).

    Returns partial state update with stix_bundle, validation_results,
    and validation_errors.
    """
    logger.info("serialize_stix: starting")

    source_id = state.get("source_id", "unknown")
    metadata = state.get("metadata", {})
    # Source.title, resolved once; the Report SDO and the attack-flow name
    # both take it. Empty when neither state nor metadata carries one.
    display_title = resolve_display_title(state, "")
    normalized_drafts = state.get("normalized_drafts", [])
    validated_entities = state.get("validated_entities", [])
    detection_rules = state.get("detection_rules", [])
    all_drafts = state.get("drafts", [])
    # Default True preserves the prior PRECEDES-emission behavior for
    # in-flight checkpoints predating the sequentiality field.
    is_sequential = bool(state.get("is_sequential", True))

    # Build a lookup for original drafts (normalized_drafts reference them)
    draft_lookup = {d["draft_id"]: d for d in all_drafts}

    # Drop procedures that cannot carry a technique. x_technique_refs is
    # REQUIRED by the x-procedure schema, so a draft whose techniques are all
    # missing or unresolved cannot be a valid object — emitting it hard-fails
    # bundle validation and takes the other 30 procedures down with it, which
    # is what happened twice on runs of one ransomware report.
    #
    # This is the *proportionate* reading of the rule the code already states
    # below ("a procedure left with no AP at all gets flagged by the
    # tuple-semantics validator instead of shipping a broken ref"): a missing
    # ATT&CK mapping is a soft finding, not a reason to lose the bundle. The
    # schema check simply hard-fails first and never lets that soft path run.
    #
    # Dropped, not silent: each one is named here, and extract_techniques now
    # warns separately when the pick step is what lost them. A behavior with
    # no technique is still in the chunk record for the analyst; it just
    # cannot be expressed as an x-procedure.
    _mappable, _unmappable = [], []
    for ndraft in normalized_drafts:
        original = draft_lookup.get(ndraft.get("draft_id", ""), {})
        techs = original.get("techniques") or []
        (_mappable if any(t.get("stix_id") for t in techs) else _unmappable).append(ndraft)
    # Every omission is also recorded as a bundle correction. A log line is
    # invisible to the analyst who approved these procedures at Gate 2 — and
    # because serialization runs AFTER that gate, there is no later prompt in
    # which the loss could be noticed. On one ransomware run this would have
    # silently discarded the one procedure the analyst had rejected
    # a whole chunking pass to capture; it survived only because a technique
    # was promoted by hand after the omission was pointed out.
    omission_corrections: list[dict] = []
    if _unmappable:
        for ndraft in _unmappable:
            original = draft_lookup.get(ndraft.get("draft_id", ""), {})
            name = original.get("name") or ndraft.get("name") or "unnamed"
            logger.warning(
                "serialize_stix: omitting procedure %r (draft %s, chunk %s) — "
                "no resolved ATT&CK technique, and x_technique_refs is required "
                "by the x-procedure schema.",
                name, ndraft.get("draft_id", "?"), original.get("chunk_id", "?"),
            )
            omission_corrections.append({
                "rule": "procedure_omitted_no_technique",
                "severity": "warn",
                "message": (
                    f"Procedure {name!r} was approved but omitted from the "
                    f"bundle: no ATT&CK technique resolved, and "
                    f"x_technique_refs is required by the x-procedure schema. "
                    f"Promote a technique for its chunk at Gate 2 to keep it."
                ),
                "ref_id": ndraft.get("draft_id", ""),
                "ref_field": "x_technique_refs",
                "holder_id": original.get("chunk_id", ""),
                "holder_type": "chunk",
            })
        logger.warning(
            "serialize_stix: %d of %d approved procedure(s) omitted for having "
            "no ATT&CK mapping.", len(_unmappable), len(normalized_drafts),
        )
        normalized_drafts = _mappable
    # Chunk lookup for the IoC linking pass: each draft's source chunk
    # carries the artifacts dict
    # that drives procedure→observable HAS-OBSERVABLE SROs.
    chunk_lookup = {c["chunk_id"]: c for c in state.get("chunks", [])}

    objects: list[dict] = []
    id_registry: dict[str, str] = {}  # local_id -> stix_id mapping

    # 1. Create the source identity (report author)
    source_identity = _create_source_identity(metadata, source_id)
    objects.append(source_identity)
    id_registry["source_identity"] = source_identity["id"]

    # 2. Create entity SDOs/SCOs
    for entity in validated_entities:
        action = entity.get("gate_action", "approve")
        if action == GateAction.REMOVE.value:
            continue

        stix_obj = _entity_to_stix(entity)
        if stix_obj:
            objects.append(stix_obj)
            id_registry[entity.get("entity_id", "")] = stix_obj["id"]

    # 3. Create x-procedure SDOs from normalized drafts
    for ndraft in normalized_drafts:
        draft_id = ndraft.get("draft_id", "")
        original_draft = draft_lookup.get(draft_id, {})

        procedure = _draft_to_procedure(
            original_draft, ndraft, source_identity["id"]
        )
        objects.append(procedure)
        id_registry[draft_id] = procedure["id"]

    # 3a. The extension every procedure just declared, and its author. A
    # custom type is only a STIX 2.1 extension if the bundle carries the
    # extension-definition it names; without it a strict consumer sees an
    # unknown custom object. Bundle metadata: excluded from the Report's
    # object_refs, never written to the graph (see extension_definitions).
    if any(o.get("type") == X_PROCEDURE_TYPE for o in objects):
        objects.append(x_procedure_author_identity())
        objects.append(x_procedure_extension_definition())

    # 3b. Create Process SCOs from raw_command_lines and wire them
    # to procedures via components_refs.
    # This bridges the gap between LLM-extracted raw text and
    # the v0.5.0-draft's SCO-based component model.
    # File SCOs already emitted (from ioc_file_path / ioc_process_name
    # entities), keyed by bare filename so a command's leading token can be
    # resolved to the binary it executes.
    file_sco_index: dict[str, str] = {}
    for obj in objects:
        if obj.get("type") != "file":
            continue
        name = (obj.get("name") or "").strip()
        if not name:
            continue
        bare = name.replace("\\", "/").rsplit("/", 1)[-1].lower()
        file_sco_index.setdefault(bare, obj["id"])
        # Index the stem too: commands invoke `tar`, `curl`, `certutil`
        # while the File SCO is named `tar.exe`. Without this the leading
        # token never matches and image_ref stays unwired.
        stem, _, ext = bare.rpartition(".")
        if stem and ext in ("exe", "com", "bat", "cmd", "ps1", "dll"):
            file_sco_index.setdefault(stem, obj["id"])

    for ndraft in normalized_drafts:
        draft_id = ndraft.get("draft_id", "")
        original_draft = draft_lookup.get(draft_id, {})
        procedure_stix_id = id_registry.get(draft_id)

        raw_cmds = original_draft.get("raw_command_lines", [])
        if not raw_cmds or not procedure_stix_id:
            continue

        # Find the procedure object we just created
        procedure_obj = next(
            (o for o in objects if o.get("id") == procedure_stix_id), None
        )
        if not procedure_obj:
            continue

        component_ids = []
        for cmd in raw_cmds:
            process_sco = _build_process_sco(cmd, file_sco_index)
            objects.append(process_sco)
            component_ids.append(process_sco["id"])

        # Add all process SCOs to components_refs
        existing_components = procedure_obj.get("x_components_refs", [])
        procedure_obj["x_components_refs"] = existing_components + component_ids

    # 3b'. IoC linking pass.
    # For each draft, look up its source chunk and match chunk.artifacts
    # values against validated_entities by value. Matching SCO IDs are
    # recorded on the draft so `_build_relationships` emits per-procedure
    # has-observable SROs.
    # This is the IoC-linking pass: a procedure that touches a registry
    # key, C2 domain, file hash, etc. gets a direct edge to the SCO.
    _entity_value_to_id = _build_entity_value_index(validated_entities, id_registry)
    for ndraft in normalized_drafts:
        draft_id = ndraft.get("draft_id", "")
        original_draft = draft_lookup.get(draft_id, {})
        procedure_stix_id = id_registry.get(draft_id)
        if not procedure_stix_id:
            continue

        chunk = chunk_lookup.get(original_draft.get("chunk_id", ""), {})
        artifacts = chunk.get("artifacts", {}) or {}
        if not artifacts:
            continue

        observable_ids: list[str] = []
        seen_obs: set[str] = set()
        for category, values in artifacts.items():
            if not isinstance(values, list):
                continue
            for raw_value in values:
                if not isinstance(raw_value, str):
                    continue
                key = raw_value.strip().lower()
                if not key:
                    continue
                stix_id = _entity_value_to_id.get(key)
                if stix_id and stix_id not in seen_obs:
                    observable_ids.append(stix_id)
                    seen_obs.add(stix_id)

        if not observable_ids:
            continue

        # NOT embedded on the procedure. `x_observable_refs` is not one of
        # the 28 properties in x-procedure v0.5.0-draft, and that schema is
        # `additionalProperties: false` — embedding it made every bundle
        # with IoC-linked procedures schema-invalid, which went unnoticed
        # only while `_validate_schema` was a hand-rolled required-fields
        # check; the full JSON Schema pass it now runs would reject it.
        # The linkage is not lost: it rides the
        # `has-observable` SROs built below, which is also what the
        # distribution layer maps to the HAS_OBSERVABLE Neo4j edge and what
        # the bundle viewers read.
        #
        # Deliberately NOT merged into `x_components_refs` either. Those
        # two mean different things (see `_build_relationships`):
        # components are what the procedure IS — the process tree of its
        # commands — while observables are what it TOUCHES. Merging would
        # also silently satisfy the tuple validator's ⟨C⟩ ≠ ∅ check for
        # procedures that have no commands at all.

        # Stash on the draft so `_build_relationships` emits has-observable
        # SROs from this draft. Mutating the original_draft dict is safe —
        # it's a transient pipeline structure, not a STIX object.
        existing_draft_obs = original_draft.get("observable_refs", []) or []
        merged = list(existing_draft_obs)
        merged_set = set(merged)
        for oid in observable_ids:
            if oid not in merged_set:
                merged.append(oid)
                merged_set.add(oid)
        original_draft["observable_refs"] = merged

    # 3c'. Resolve x_vulnerability_refs: entity_ids -> STIX UUIDs.
    # Drafting populated this with internal entity_ids (CVE-pattern
    # matches against validated_entities). At this point every
    # vulnerability entity has been STIX-IDed and registered, so the
    # entity_id -> stix_id lookup is straightforward.
    for obj in objects:
        if obj.get("type") != X_PROCEDURE_TYPE:
            continue
        raw_vrefs = obj.get("x_vulnerability_refs", [])
        if not raw_vrefs:
            continue
        resolved_vuln: list[str] = []
        for ref in raw_vrefs:
            if ref.startswith("vulnerability--"):
                resolved_vuln.append(ref)
            else:
                stix_id = id_registry.get(ref)
                if stix_id and stix_id.startswith("vulnerability--"):
                    resolved_vuln.append(stix_id)
                else:
                    logger.warning(
                        "serialize_stix: vulnerability_ref '%s' could not be resolved",
                        ref,
                    )
        if resolved_vuln:
            obj["x_vulnerability_refs"] = resolved_vuln
        else:
            obj.pop("x_vulnerability_refs", None)

    # 3d. Log source enrichment: query Neo4j for DataComponents that
    # detect each procedure's ATT&CK techniques, create x-log-source
    # objects, and wire x_log_source_refs on each procedure.
    log_source_objects, procedure_log_refs, no_coverage_proc_ids = await _enrich_log_sources(objects)
    objects.extend(log_source_objects)
    for obj in objects:
        if obj.get("type") != X_PROCEDURE_TYPE:
            continue
        proc_id = obj.get("id", "")
        refs = procedure_log_refs.get(proc_id, [])
        if refs:
            obj["x_log_source_refs"] = refs

    # 3e. Detection chain enrichment: emit full ATT&CK detection chain as
    # SDO stubs so the bundle is self-contained (AttackPattern,
    # DetectionStrategy, Analytic, DataComponent) with SROs tying them
    # together. Deduplicates by STIX ID so repeated techniques across
    # procedures don't produce duplicate nodes.
    chain_objects, chain_rels = await _enrich_detection_chain(objects, source_identity["id"])
    objects.extend(chain_objects)
    objects.extend(chain_rels)

    # 4. Create detection rule indicators
    # Detection rules do NOT become Indicator SDOs. `.cursorrules` is explicit:
    # "IOCs are SCOs, NOT indicators. Detection rules are a separate pipeline."
    # Emitting them here also produced factually wrong STIX — a vendor rule
    # NAME carried as `pattern` with `pattern_type: "sigma"`, which no consumer
    # can parse. The reference is preserved on
    # the Report SDO instead, which is where provenance belongs.
    detection_rule_refs = [
        {
            "source_name": f"detection-rule:{rule.get('rule_type', 'unknown')}",
            "description": (
                rule.get("description")
                or rule.get("rule_content", "")
            )[:512],
        }
        for rule in detection_rules
        if rule.get("rule_content")
    ]

    # 4b. Create the attack-flow object when the bundle has procedure
    # sequencing. start_refs lists every chain root (chunks the chunker
    # explicitly marked, plus any procedure with no incoming forward edge
    # as a fallback). ATT&CK Flow v2.0.0 supports multiple start_refs on a
    # single attack-flow, so multi-intrusion sources collapse to one flow
    # with multiple roots — keeps the validator's "max one attack-flow per
    # bundle" rule intact. Sequencing edges themselves are PRECEDES SROs,
    # built later in _build_relationships.
    flow_obj = _build_attack_flow(
        objects, normalized_drafts, draft_lookup, id_registry,
        source_identity["id"], metadata, display_title=display_title,
    )
    if flow_obj:
        objects.append(flow_obj)

    # 4c. Attack-operator SDOs. Materialize attack-operator objects from
    # the operator entries that normalize inferred (or that the analyst
    # override-marked at gate_chunks). Built before _build_relationships so the precedes
    # emission can route through them. Gated on is_sequential via the
    # normalize step — chunk_operators is empty when False, so this is
    # a no-op for catalog sources.
    chunk_operators = state.get("chunk_operators", {}) or {}
    chunk_to_proc_stix_id: dict[str, str] = {}
    for ndraft in normalized_drafts:
        draft_id = ndraft.get("draft_id", "")
        original = draft_lookup.get(draft_id, {})
        chunk_id = original.get("chunk_id", "")
        proc_stix_id = id_registry.get(draft_id)
        if chunk_id and proc_stix_id:
            chunk_to_proc_stix_id[chunk_id] = proc_stix_id

    operator_sdos, op_id_to_stix_id = build_attack_operator_sdos(
        chunk_operators,
        chunk_to_proc_stix_id,
        source_identity["id"],
        _now_iso(),
        STIX_SPEC_VERSION,
    )
    objects.extend(operator_sdos)

    # 4d. Attack-condition SDOs. Mirror
    # of the operator SDO build: pull conditions extracted by normalize
    # from chunk preconditions, resolve refs through procedure STIX ids,
    # and append. route_precedes_through_operators then routes precedes
    # SROs through both layers (condition wins at source side when
    # both apply; infer_operators already suppresses OR-branch at
    # condition anchors).
    chunk_conditions = state.get("chunk_conditions", {}) or {}
    condition_sdos, cond_anchor_to_stix_id = build_attack_condition_sdos(
        chunk_conditions,
        chunk_to_proc_stix_id,
        source_identity["id"],
        _now_iso(),
        STIX_SPEC_VERSION,
    )
    objects.extend(condition_sdos)

    # 4e. Attack Flow's own definition and author, embedded exactly as CTID's
    # example bundle does, whenever any attack-* object was emitted. One
    # definition covers every Attack Flow type.
    if any(o.get("type") in ATTACK_FLOW_TYPES for o in objects):
        objects.append(attack_flow_author_identity())
        objects.append(attack_flow_extension_definition())

    # 5. Create relationships (SROs)
    relationships = _build_relationships(
        normalized_drafts, draft_lookup, validated_entities,
        id_registry, source_identity["id"],
        is_sequential=is_sequential,
        chunks=state.get("chunks", []) or [],
        chunk_operators=chunk_operators,
        op_id_to_stix_id=op_id_to_stix_id,
        chunk_to_proc_stix_id=chunk_to_proc_stix_id,
        chunk_conditions=chunk_conditions,
        cond_anchor_to_stix_id=cond_anchor_to_stix_id,
        removed_rel_keys=_analyst_removed_rel_keys(state),
    )
    objects.extend(relationships)

    # 6. Create Report SDO last so its object_refs cover every other
    # object assembled above (SDOs, SCOs, and SROs). This ties orphan
    # SDOs (identities, vulnerabilities, etc.) into the graph via
    # `references` edges.
    report = _build_report(
        metadata, objects, source_identity["id"],
        extra_references=detection_rule_refs, display_title=display_title,
    )
    objects.append(report)

    # 7. Assemble the bundle
    bundle = {
        "type": "bundle",
        "id": f"bundle--{uuid.uuid4()}",
        "objects": objects,
    }

    # 7. Validate
    validation_results, validation_errors = _validate_bundle(bundle, no_coverage_proc_ids)

    # The tuple-semantics check appends its C-missing *warnings* to the same
    # list as hard errors, so count them apart: "valid=True, errors=11" was
    # read as a failure more than once when it meant eleven procedures with
    # no command line.
    n_warnings = sum(1 for e in validation_errors if "(warning)" in e)
    logger.info(
        "serialize_stix: bundle has %d objects, valid=%s, errors=%d, warnings=%d",
        len(objects),
        all(validation_results.values()),
        len(validation_errors) - n_warnings,
        n_warnings,
    )

    return {
        "stix_bundle": bundle,
        "validation_results": validation_results,
        "validation_errors": validation_errors,
        # Seeds bundle_corrections; validate_bundle runs next and extends this
        # list rather than starting a fresh one, so the omissions survive into
        # the CorrectionsModal alongside the validator's own findings.
        "bundle_corrections": omission_corrections,
        "status": PipelineStatus.SERIALIZING.value,
        "current_node": "serialize_stix",
    }


# =============================================================================
# Object builders
# =============================================================================

def _create_source_identity(metadata: dict, source_id: str) -> dict:
    """Create an Identity SDO for the report source/author."""
    author = metadata.get("author", "Unknown Source")
    now = _now_iso()

    return {
        "type": "identity",
        "spec_version": STIX_SPEC_VERSION,
        "id": f"identity--{uuid.uuid4()}",
        "created": now,
        "modified": now,
        "name": author,
        "identity_class": "organization",
        "x_source_id": source_id,
    }


def _build_attack_flow(
    objects: list[dict],
    normalized_drafts: list[dict],
    draft_lookup: dict[str, dict],
    id_registry: dict[str, str],
    source_identity_id: str,
    metadata: dict,
    display_title: str = "",
) -> dict | None:
    """Construct an attack-flow SDO when the bundle carries procedure
    sequencing.

    Behavior:
      - Returns None when fewer than 2 procedures exist (no flow needed).
      - start_refs = procedures whose draft.chain_root=True. When no
        chain_root is explicitly marked, falls back to the topological
        roots of the chunk DAG — procedures whose draft is referenced by
        NO other draft's forward edges (draft-level effect_refs, which
        normalize derives by inverting predecessor_indices). Sequencing
        itself is expressed by PRECEDES SROs; the attack-flow object only
        needs the entry points.
      - One attack-flow per bundle — multi-chain sources surface as
        multiple start_refs, not multiple flows. The validator's
        max-one-attack-flow check stays intact.
    """
    procedures = [o for o in objects if o.get("type") == X_PROCEDURE_TYPE]
    if len(procedures) < 2:
        return None

    # Resolve chain_root drafts to procedure STIX IDs.
    explicit_roots: list[str] = []
    for ndraft in normalized_drafts:
        draft_id = ndraft.get("draft_id", "")
        original = draft_lookup.get(draft_id, {})
        if not original.get("chain_root"):
            continue
        proc_stix_id = id_registry.get(draft_id)
        if proc_stix_id:
            explicit_roots.append(proc_stix_id)

    # Fallback: when no chain_root was marked, use the topological roots
    # of the chunk DAG — drafts that no other draft's effect_refs point
    # at. effect_refs are draft_id-space forward edges normalize computed
    # from predecessor_indices, so a draft absent from every effect_refs
    # list is a chain root. Keeps single-chain sources working without
    # needing chain_root=True on chunk 1.
    if not explicit_roots:
        targeted_draft_ids: set[str] = set()
        for ndraft in normalized_drafts:
            original = draft_lookup.get(ndraft.get("draft_id", ""), {})
            for tgt in original.get("effect_refs", []) or []:
                targeted_draft_ids.add(tgt)
        for ndraft in normalized_drafts:
            draft_id = ndraft.get("draft_id", "")
            if draft_id in targeted_draft_ids:
                continue
            proc_stix_id = id_registry.get(draft_id)
            if proc_stix_id:
                explicit_roots.append(proc_stix_id)

    if not explicit_roots:
        # Pathological: every procedure is referenced by at least one
        # other (i.e. there's a cycle). Don't ship a flow object —
        # validator's _check_precedes_cycle will hard-fail anyway.
        return None

    now = _now_iso()
    flow_id = f"attack-flow--{uuid.uuid4()}"

    # The source title when the run has one; a generic label otherwise.
    title = display_title or metadata.get("title") or "Extracted attack flow"

    flow_obj = {
        "type": "attack-flow",
        "spec_version": STIX_SPEC_VERSION,
        "id": flow_id,
        "created": now,
        "modified": now,
        "created_by_ref": source_identity_id,
        "name": title,
        "start_refs": explicit_roots,
        # The Attack Flow 2.0.0 declaration; the definition object itself is
        # appended to the bundle once any attack-* object exists.
        "extensions": extension_declaration(ATTACK_FLOW_EXTENSION_ID),
    }

    # Flow membership is queryable from the flow side (start_refs) and via
    # PRECEDES SROs; the procedure does not carry a back-pointer. (No
    # x_flow_ref stamping — removed in v0.5.0-draft.)
    return flow_obj


def _build_report(
    metadata: dict,
    objects: list[dict],
    source_identity_id: str,
    extra_references: list[dict] | None = None,
    display_title: str = "",
) -> dict:
    """Create a Report SDO enumerating every non-relationship, non-marking
    object in `object_refs`.

    This wires otherwise-dangling SDOs (identities, context locations,
    unexploited CVEs, etc.) to the report via `references` edges in the
    viewer. Relationship, marking-definition, extension-definition, and
    language-content objects are excluded from object_refs per STIX 2.1
    convention, and so are the identities the extension definitions name as
    their authors: the report describes the source's content, not the
    people who defined the object types it is written in.
    """
    now = _now_iso()
    title = display_title or metadata.get("title") or "Untitled CTI Report"
    description = metadata.get("description") or f"Extracted procedures and entities from {metadata.get('author', 'unknown source')}."
    # Normalize publication_date to STIX timestamp format. Accepts
    # "YYYY-MM-DD" and upgrades to "YYYY-MM-DDT00:00:00.000Z"; passes
    # through anything already containing "T". Falls back to `now` if
    # malformed.
    pub_raw = metadata.get("publication_date") or ""
    if pub_raw:
        if "T" in pub_raw:
            published = pub_raw
        elif len(pub_raw) == 10 and pub_raw.count("-") == 2:
            published = f"{pub_raw}T00:00:00.000Z"
        else:
            published = now
    else:
        published = now

    excluded = {"relationship", "marking-definition", "extension-definition", "language-content"}
    meta_ids = bundle_meta_ids(objects)
    object_refs = [
        o["id"] for o in objects
        if o.get("type") not in excluded and o.get("id") and o["id"] not in meta_ids
    ]

    report: dict = {
        "type": "report",
        "spec_version": STIX_SPEC_VERSION,
        "id": f"report--{uuid.uuid4()}",
        "created": now,
        "modified": now,
        "created_by_ref": source_identity_id,
        "name": title,
        "description": description,
        "published": published,
        "report_types": ["threat-report"],
        "object_refs": object_refs,
    }

    references: list[dict] = []
    source_url = metadata.get("source_url")
    if source_url:
        references.append({"source_name": "source", "url": source_url})
    # Vendor detection-rule listings ride here rather than becoming Indicator
    # SDOs — see the note at the detection-rule handling in serialize_stix.
    references.extend(extra_references or [])
    if references:
        report["external_references"] = references

    return report


def _resolve_ip(value: str) -> tuple[str, str] | None:
    """Pick `ipv4-addr` vs `ipv6-addr` and return the address to emit.

    Returns ``(stix_type, address)``, or ``None`` when the value is not a
    usable IP — the caller then skips the entity.

    The previous test was ``":" in value``, which is true of every
    ``host:port`` indicator. ``198.51.100.167:2967`` was emitted as an
    ``ipv6-addr``, the schema rejected it, and because bundle validation is
    all-or-nothing that ONE observable destroyed the whole bundle (one malware-analysis
    source). Any source carrying a host:port IOC hit this.

    So: parse, do not sniff. A trailing ``:port`` is stripped when what
    remains is a valid IPv4 — STIX ``ipv4-addr`` has no port field, and a port
    belongs on a ``network-traffic`` SCO, which is a modeling decision rather
    than part of this fix. A bracketed ``[::1]:443`` is unwrapped the same way.
    Anything still unparseable is dropped: losing one indicator is strictly
    better than losing the bundle it traveled in.
    """
    addr = (value or "").strip()
    if not addr:
        return None

    # `[2001:db8::1]:443` — the bracket form exists precisely because a bare
    # IPv6 address is full of colons, so unwrap before counting any.
    m = re.fullmatch(r"\[(?P<host>[^\]]+)\](?::\d{1,5})?", addr)
    if m:
        addr = m.group("host")

    for candidate in (addr, addr.rsplit(":", 1)[0] if addr.count(":") == 1 else None):
        if not candidate:
            continue
        try:
            ip = ipaddress.ip_address(candidate)
        except ValueError:
            continue
        return ("ipv6-addr" if ip.version == 6 else "ipv4-addr", str(ip))
    return None


def _entity_to_stix(entity: dict) -> dict | None:
    """Convert a validated entity to a STIX SDO or SCO."""
    entity_type = entity.get("entity_type", "")
    stix_type = _ENTITY_TO_STIX_TYPE.get(entity_type)

    if not stix_type:
        logger.warning("No STIX mapping for entity type: %s", entity_type)
        return None

    # Use edited value if analyst corrected it at Gate 0
    value = entity.get("edited_value") or entity.get("value", "")
    effective_type = entity.get("edited_type") or entity_type

    # Re-check mapping with edited type
    if effective_type != entity_type:
        stix_type = _ENTITY_TO_STIX_TYPE.get(effective_type, stix_type)

    if effective_type == EntityType.IOC_IP.value:
        resolved = _resolve_ip(value)
        if resolved is None:
            logger.warning(
                "Skipping IOC_IP entity whose value is not a usable IP "
                "address: %r", value[:60],
            )
            return None
        stix_type, value = resolved

    now = _now_iso()
    stix_id = f"{stix_type}--{uuid.uuid4()}"

    if stix_type in _SCO_TYPES:
        return _build_sco(stix_type, stix_id, value, effective_type, entity)
    else:
        return _build_sdo(stix_type, stix_id, value, now, entity)


def _build_sco(
    stix_type: str, stix_id: str, value: str,
    entity_type: str, entity: dict | None = None,
) -> dict | None:
    """Build a STIX Cyber Observable object.

    Returns None when the SCO can't be constructed soundly (e.g. hash
    with unknown algorithm). Caller treats None as "skip this entity".
    """
    obj = {
        "type": stix_type,
        "spec_version": STIX_SPEC_VERSION,
        "id": stix_id,
    }

    # Type-specific value fields
    if stix_type in ("ipv4-addr", "ipv6-addr"):
        obj["value"] = value
    elif stix_type == "domain-name":
        obj["value"] = value
    elif stix_type == "url":
        obj["value"] = value
    elif stix_type == "email-addr":
        obj["value"] = value
    elif stix_type == "file":
        if entity_type == EntityType.IOC_HASH.value:
            hash_type = _detect_hash_type(value)
            if hash_type is None:
                logger.warning(
                    "serialize_stix: unknown hash length for %r, skipping SCO",
                    value,
                )
                return None
            obj["hashes"] = {hash_type: value}
        elif entity_type in (
            EntityType.IOC_FILE_PATH.value,
            EntityType.IOC_PROCESS_NAME.value,
        ):
            obj["name"] = value
    elif stix_type == "windows-registry-key":
        obj["key"] = _expand_registry_hive(value)
    elif stix_type == "mutex":
        obj["name"] = value
    elif stix_type == "software":
        obj["name"] = value
        # Try to split "Product Version" into name + version
        # e.g. "Apache ActiveMQ 5.15.0" -> name + version
    elif stix_type == "user-account":
        obj["account_login"] = value

    return obj


# STIX 2.1's windows-registry-key schema REJECTS the abbreviated hive names.
# Its `key` carries a NEGATIVE pattern (`not: {pattern: "^HKLM|HKCC|..."}`),
# so "HKLM\\SYSTEM\\..." is invalid and "HKEY_LOCAL_MACHINE\\SYSTEM\\..." is
# what the spec wants. CTI reports write the abbreviation almost without
# exception, so without this expansion any source quoting a registry path
# hard-fails bundle validation — which is how one run died, on three
# perfectly ordinary Terminal Server keys.
_REGISTRY_HIVES = {
    "HKLM": "HKEY_LOCAL_MACHINE",
    "HKCU": "HKEY_CURRENT_USER",
    "HKCR": "HKEY_CLASSES_ROOT",
    "HKU": "HKEY_USERS",
    "HKCC": "HKEY_CURRENT_CONFIG",
}


def _expand_registry_hive(value: str) -> str:
    """Expand an abbreviated registry hive to the full name STIX requires.

    Only the leading hive is touched, and only when it is the whole first
    path segment: "HKLM\\Software" becomes "HKEY_LOCAL_MACHINE\\Software",
    but a key that merely starts with those letters ("HKLMX\\...") is left
    alone. A value already using the full name passes through unchanged, so
    this is safe to apply unconditionally.
    """
    if not value:
        return value
    head, sep, rest = value.partition("\\")
    full = _REGISTRY_HIVES.get(head.strip().upper())
    if full is None:
        return value
    return full + sep + rest


# Canonical business regions, agreed with the analyst. These are NOT
# decomposed; a
# non-standard coinage like AMEA is normalized or decomposed upstream.
CANONICAL_REGIONS = {
    "EMEA": "Europe, Middle East and Africa",
    "APAC": "Asia-Pacific",
    "LATAM": "Latin America",
    "NA": "North America",
    "CIS": "Commonwealth of Independent States",
    "ANZ": "Australia and New Zealand",
}

# STIX region-ov values (UN M49-flavored) are legitimately regions too, even
# though they are not in the canonical six — a source saying "Eastern Europe"
# means a region, not a country. Kept verbatim rather than mapped onto an
# acronym, for the widening reason in _build_sdo.
_M49_REGIONS = {
    "africa", "eastern africa", "middle africa", "northern africa",
    "southern africa", "western africa", "americas", "latin america",
    "latin america and the caribbean", "south america", "caribbean",
    "central america", "northern america", "asia", "central asia",
    "eastern asia", "southern asia", "south asia", "south-eastern asia",
    "southeast asia", "western asia", "europe", "eastern europe",
    "northern europe", "southern europe", "western europe", "oceania",
    "australia and new zealand", "melanesia", "micronesia", "polynesia",
    "antarctica", "middle east", "asia-pacific",
}


def _normalize_place(value: str) -> str:
    """Casefold and flatten separators so 'South-Eastern Asia' and
    'south eastern asia' compare equal."""
    return " ".join(
        (value or "").lower().replace("-", " ").replace("_", " ").split()
    )


def _is_region(value: str) -> bool:
    """True when a location value names a region rather than a country."""
    v = (value or "").strip()
    if not v:
        return False
    if v.upper() in CANONICAL_REGIONS:
        return True
    normalized = _normalize_place(v)
    if normalized in {_normalize_place(r) for r in _M49_REGIONS}:
        return True
    # Expansions of the canonical six, written out ("Europe, Middle East and
    # Africa"). Compared on letters only so punctuation and "&" do not matter.
    letters = "".join(ch for ch in normalized if ch.isalnum())
    return any(
        letters == "".join(ch for ch in expansion.lower() if ch.isalnum())
        for expansion in CANONICAL_REGIONS.values()
    )


def _build_sdo(
    stix_type: str, stix_id: str, value: str, now: str,
    entity: dict | None = None,
) -> dict:
    """Build a STIX Domain Object (non-observable)."""
    entity = entity or {}

    obj = {
        "type": stix_type,
        "spec_version": STIX_SPEC_VERSION,
        "id": stix_id,
        "created": now,
        "modified": now,
        "name": value,
    }

    # Type-specific fields
    if stix_type == "malware":
        obj["is_family"] = True  # Default; analyst can refine
    elif stix_type == "identity":
        # Determine identity_class from entity context
        org_role = entity.get("organization_role", "")
        entity_type = entity.get("entity_type", "")
        if entity_type == EntityType.ORGANIZATION.value:
            obj["identity_class"] = "organization"
        elif entity_type == EntityType.VICTIM_SECTOR.value:
            obj["identity_class"] = "class"
            obj["sectors"] = [value]  # STIX industry-sector-ov value
        else:
            obj["identity_class"] = "unknown"
    elif stix_type == "threat-actor":
        obj["threat_actor_types"] = ["unknown"]  # Analyst can refine
    elif stix_type == "infrastructure":
        # infrastructure_types is optional in the SDO schema, so omitting it
        # validated — and shipped an Infrastructure object carrying a name and
        # nothing else, losing the C2-vs-staging-vs-phishing distinction these
        # entities are extracted for. "unknown" is the spec's own placeholder in
        # infrastructure-type-ov (STIX 2.1 §10.12) and is what a partner can
        # consume; the vocabulary is OPEN (our vendored schema declares
        # `items: {type: string}` with no $ref), so a narrower or custom value
        # is the analyst's to set. Note minItems: 1 — an empty list would NOT
        # validate, so this is the only safe placeholder.
        obj["infrastructure_types"] = ["unknown"]
    elif stix_type == "vulnerability":
        # If value looks like a CVE ID, set external_references
        if value.upper().startswith("CVE-"):
            obj["external_references"] = [{
                "source_name": "cve",
                "external_id": value.upper(),
            }]
    elif stix_type == "location":
        # STIX Location distinguishes `region` from `country`, and this used
        # only the latter — so "EMEA" and "South Asia" both shipped as
        # countries. Route by what the value actually is.
        #
        # Sub-regions are NOT collapsed into a canonical acronym. "South Asia"
        # is not reliably "APAC" (definitions differ on whether India and
        # Pakistan are inside), and silently widening a targeting claim is
        # worse than keeping the source's own word.
        if _is_region(value):
            obj["region"] = value
        else:
            obj["country"] = value  # could later resolve to ISO codes

    return obj


def _build_process_sco(
    command_line: str,
    file_index: dict[str, str] | None = None,
) -> dict:
    """Build a STIX Process SCO from a raw command line string.

    The Process SCO is the v0.5.0-draft's atomic unit for command
    execution. It carries the command_line and, when the binary is known to
    the bundle as a File SCO, a real STIX `image_ref` pointing at it.

    `image_ref` is the standard link from a running process to the file on
    disk it was launched from, and it is what makes "which procedures ran
    this binary" answerable in the graph. Before process names were mapped
    to File SCOs there was nothing to point at, so this recorded a
    non-standard `x_exe_name` string instead; that is kept as a fallback
    label when the binary is not among the extracted entities.
    """
    process_id = f"process--{uuid.uuid4()}"
    sco = {
        "type": "process",
        "spec_version": STIX_SPEC_VERSION,
        "id": process_id,
        "command_line": command_line,
    }

    # Try to extract the executable name from the command line
    # to set as a human-readable label. Simple heuristic: first token.
    tokens = command_line.strip().split()
    if tokens:
        exe = tokens[0]
        # Strip common path prefixes for readability
        if "\\" in exe:
            exe = exe.rsplit("\\", 1)[-1]
        if "/" in exe:
            exe = exe.rsplit("/", 1)[-1]
        lookup = exe.strip().lower()
        index = file_index or {}
        image_id = index.get(lookup)
        if not image_id and "." not in lookup:
            # The command said `tar`; try the executable form.
            image_id = index.get(f"{lookup}.exe")
        if image_id:
            sco["image_ref"] = image_id
        else:
            # No File SCO for this binary — keep the readable label so the
            # information is not lost, even though it is not standard STIX.
            sco["x_exe_name"] = exe

    return sco


# =============================================================================
# Log source enrichment (Neo4j -> x-log-source objects)
# =============================================================================

# Cypher: find DataComponents that detect a set of AttackPatterns.
#
# Matching strategy: match on the real STIX UUID OR the ATT&CK T-number
# carried in the node's external_references JSON. `_draft_to_procedure`
# omits any technique that did not resolve to a real UUID, so in practice
# every ref is a UUID; the T-number path is kept so a ref of the form
# "attack-pattern--T1059.001" (an older bundle, a hand-authored fixture)
# still finds its node.
#
# Detection chain: DetectionStrategy -[DETECTS]-> AttackPattern,
# then DetectionStrategy -[HAS_ANALYTIC]-> Analytic -[USES_DATA_COMPONENT]-> DataComponent.
#
# Uses APOC to parse the external_references JSON string.
_LOG_SOURCE_QUERY = """
MATCH (ap:AttackPattern)
WHERE ap.stix_id IN $stix_ids
   OR any(ref IN apoc.convert.fromJsonList(ap.external_references)
          WHERE ref.source_name = 'mitre-attack' AND ref.external_id IN $mitre_ids)
WITH ap
MATCH (ds:DetectionStrategy)-[:DETECTS]->(ap)
MATCH (ds)-[:HAS_ANALYTIC]->(a:Analytic)-[:USES_DATA_COMPONENT]->(dc:DataComponent)
RETURN DISTINCT
    dc.name AS name,
    dc.stix_id AS stix_id,
    dc.description AS description,
    collect(DISTINCT ap.stix_id) AS technique_stix_ids,
    ds.name AS detection_strategy_name
"""

# Fallback: simpler path without requiring full HAS_ANALYTIC chain.
# Catches cases where DataComponent -[DETECTS]-> AttackPattern directly.
_LOG_SOURCE_FALLBACK_QUERY = """
MATCH (ap:AttackPattern)
WHERE ap.stix_id IN $stix_ids
   OR any(ref IN apoc.convert.fromJsonList(ap.external_references)
          WHERE ref.source_name = 'mitre-attack' AND ref.external_id IN $mitre_ids)
WITH ap
MATCH (ap)<-[:DETECTS]-(ds)
OPTIONAL MATCH (ds)-[:HAS_ANALYTIC]->(a)-[:USES_DATA_COMPONENT]->(dc:DataComponent)
WITH ap, COALESCE(dc, ds) AS source_node
WHERE source_node IS NOT NULL AND source_node.name IS NOT NULL
RETURN DISTINCT
    source_node.name AS name,
    source_node.stix_id AS stix_id,
    source_node.description AS description,
    collect(DISTINCT ap.stix_id) AS technique_stix_ids,
    null AS detection_strategy_name
"""


def _build_log_source(name: str, description: str | None, dc_stix_id: str | None) -> dict:
    """Create an x-log-source STIX object from a DataComponent."""
    now = _now_iso()
    return {
        "type": "x-log-source",
        "spec_version": STIX_SPEC_VERSION,
        "id": f"x-log-source--{uuid.uuid4()}",
        "created": now,
        "modified": now,
        "name": name,
        "description": description or f"Log source: {name}",
        "x_mitre_data_component_ref": dc_stix_id or "",
    }


async def _enrich_log_sources(
    objects: list[dict],
) -> tuple[list[dict], dict[str, list[str]], set[str]]:
    """Query Neo4j for DataComponents and create x-log-source objects.

    For each x-procedure in objects, collects its technique refs,
    queries Neo4j for DataComponents in the detection chain, creates
    x-log-source objects (deduplicated by DataComponent name), and
    returns a mapping of procedure STIX ID -> list of x-log-source IDs.

    Also tracks which procedures have zero detection coverage in ATT&CK
    (all their techniques lack DataComponent mappings). This is distinct
    from a pipeline bug: ATT&CK simply doesn't map detection sources for
    every technique (common in Reconnaissance, Resource Development, Impact).

    Returns:
        (log_source_objects, procedure_log_refs, no_coverage_proc_ids) where:
        - log_source_objects: list of x-log-source STIX dicts to add to bundle
        - procedure_log_refs: dict mapping procedure_stix_id -> [x-log-source IDs]
        - no_coverage_proc_ids: set of procedure STIX IDs whose techniques ALL
          lack detection coverage in ATT&CK (legitimate gap, not a bug)
    """
    # Collect all technique refs across all procedures. extract_techniques
    # resolves T-numbers to real STIX UUIDs and _draft_to_procedure omits
    # anything unresolved, so refs are real UUIDs (attack-pattern--<uuid>);
    # a T-number form (attack-pattern--T1059.001) is tolerated, not expected.
    procedure_techniques: dict[str, list[str]] = {}  # proc_id -> [technique refs]
    all_technique_ids: set[str] = set()

    for obj in objects:
        if obj.get("type") != X_PROCEDURE_TYPE:
            continue
        tech_refs = obj.get("x_technique_refs", [])
        if tech_refs:
            procedure_techniques[obj["id"]] = tech_refs
            all_technique_ids.update(tech_refs)

    if not all_technique_ids:
        logger.info("serialize_stix: no technique refs found, skipping log source enrichment")
        return [], {}, set(procedure_techniques.keys())

    # Split technique refs into real STIX IDs (UUIDs) and mitre_ids (T-numbers).
    # Real UUID: "attack-pattern--0042a9f5-..." -> direct Neo4j match
    # T-number form: "attack-pattern--T1059.001" -> extract T-number
    stix_ids: list[str] = []
    mitre_ids: list[str] = []
    for tid in all_technique_ids:
        if _ATTACK_PATTERN_UUID_RE.match(tid):
            stix_ids.append(tid)
        else:
            mitre_id = tid.replace("attack-pattern--", "")
            mitre_ids.append(mitre_id)

    logger.info(
        "serialize_stix: technique refs: %d STIX IDs, %d bare T-numbers (legacy form)",
        len(stix_ids), len(mitre_ids),
    )

    # Query Neo4j for DataComponents
    technique_to_components: dict[str, list[dict]] = {}
    query_params = {"stix_ids": stix_ids, "mitre_ids": mitre_ids}
    try:
        results = await run_query(_LOG_SOURCE_QUERY, query_params)

        if not results:
            logger.info("serialize_stix: primary log source query returned 0 results, trying fallback")
            results = await run_query(_LOG_SOURCE_FALLBACK_QUERY, query_params)

        logger.info("serialize_stix: found %d DataComponent mappings from Neo4j", len(results))

        # Index components by technique STIX ID. With real UUIDs flowing
        # through from extract_techniques, direct matching works.
        for row in results:
            for tech_stix_id in (row.get("technique_stix_ids") or []):
                technique_to_components.setdefault(tech_stix_id, []).append(row)

        # Tolerance: if any T-number refs are present, build the reverse map
        # so they still get log source coverage.
        if results and mitre_ids:
            try:
                mitre_map_results = await run_query(
                    """MATCH (ap:AttackPattern)
                    WHERE any(ref IN apoc.convert.fromJsonList(ap.external_references)
                              WHERE ref.source_name = 'mitre-attack'
                              AND ref.external_id IN $mitre_ids)
                    WITH ap,
                         [ref IN apoc.convert.fromJsonList(ap.external_references)
                          WHERE ref.source_name = 'mitre-attack' | ref.external_id][0]
                         AS mitre_id
                    RETURN ap.stix_id AS real_id, mitre_id""",
                    {"mitre_ids": mitre_ids},
                )
                for row in mitre_map_results:
                    real_id = row.get("real_id", "")
                    mid = row.get("mitre_id", "")
                    if real_id and mid:
                        fabricated = f"attack-pattern--{mid}"
                        if real_id in technique_to_components:
                            technique_to_components[fabricated] = (
                                technique_to_components[real_id]
                            )
            except Exception as e:
                logger.warning(
                    "serialize_stix: mitre_id reverse mapping failed: %s", e
                )

    except Exception as e:
        logger.warning(
            "serialize_stix: Neo4j log source query failed (%s: %s), "
            "procedures will have empty x_log_source_refs",
            type(e).__name__, e,
        )
        return [], {}, set(procedure_techniques.keys())

    # Create x-log-source objects, deduplicated by component name
    name_to_log_source: dict[str, dict] = {}  # component_name -> x-log-source object
    for row in results:
        name = row.get("name", "")
        if not name or name in name_to_log_source:
            continue
        ls_obj = _build_log_source(
            name=name,
            description=row.get("description"),
            dc_stix_id=row.get("stix_id"),
        )
        name_to_log_source[name] = ls_obj

    # Build procedure -> log source refs mapping.
    # Track which techniques have ANY detection coverage.
    techniques_with_coverage: set[str] = set(technique_to_components.keys())

    procedure_log_refs: dict[str, list[str]] = {}
    no_coverage_proc_ids: set[str] = set()

    for proc_id, tech_refs in procedure_techniques.items():
        ls_ids: set[str] = set()
        has_any_coverage = False
        for tech_id in tech_refs:
            if tech_id in techniques_with_coverage:
                has_any_coverage = True
            for component in technique_to_components.get(tech_id, []):
                comp_name = component.get("name", "")
                if comp_name in name_to_log_source:
                    ls_ids.add(name_to_log_source[comp_name]["id"])
        if ls_ids:
            procedure_log_refs[proc_id] = sorted(ls_ids)
        if not has_any_coverage:
            no_coverage_proc_ids.add(proc_id)

    log_source_objects = list(name_to_log_source.values())
    logger.info(
        "serialize_stix: created %d x-log-source objects, "
        "enriched %d/%d procedures with log source refs, "
        "%d procedures have no ATT&CK detection coverage",
        len(log_source_objects),
        len(procedure_log_refs),
        len(procedure_techniques),
        len(no_coverage_proc_ids),
    )

    return log_source_objects, procedure_log_refs, no_coverage_proc_ids


# =============================================================================
# Detection chain enrichment (full ATT&CK chain as bundle SDOs)
# =============================================================================

# Cypher returns the full detection chain per technique referenced by any
# x-procedure in the bundle. Rows are deduplicated per (AP, DS, A, DC) tuple.
# Minimal fields returned: just enough to build SDO stubs (id, name,
# external_id for ATT&CK refs, description).
_DETECTION_CHAIN_QUERY = """
MATCH (ap:AttackPattern)
WHERE ap.stix_id IN $stix_ids
   OR any(ref IN apoc.convert.fromJsonList(ap.external_references)
          WHERE ref.source_name = 'mitre-attack' AND ref.external_id IN $mitre_ids)
OPTIONAL MATCH (ds:DetectionStrategy)-[:DETECTS]->(ap)
OPTIONAL MATCH (ds)-[:HAS_ANALYTIC]->(a:Analytic)
OPTIONAL MATCH (a)-[:USES_DATA_COMPONENT]->(dc:DataComponent)
OPTIONAL MATCH (dc)-[:DETECTS]->(ap2:AttackPattern) WHERE ap2 = ap
RETURN DISTINCT
    ap.stix_id AS ap_id,
    ap.name AS ap_name,
    ap.description AS ap_description,
    apoc.convert.fromJsonList(ap.external_references) AS ap_refs,
    ap.x_mitre_is_subtechnique AS ap_is_sub,
    ds.stix_id AS ds_id,
    ds.name AS ds_name,
    ds.description AS ds_description,
    apoc.convert.fromJsonList(ds.external_references) AS ds_refs,
    a.stix_id AS a_id,
    a.name AS a_name,
    a.description AS a_description,
    apoc.convert.fromJsonList(a.external_references) AS a_refs,
    dc.stix_id AS dc_id,
    dc.name AS dc_name,
    dc.description AS dc_description,
    apoc.convert.fromJsonList(dc.external_references) AS dc_refs
"""


def _mitre_ext_ref(refs_list) -> dict | None:
    """Extract the mitre-attack external_reference from a list."""
    if not refs_list:
        return None
    for r in refs_list:
        if isinstance(r, dict) and r.get("source_name") == "mitre-attack":
            return {"source_name": "mitre-attack", "external_id": r.get("external_id", "")}
    return None


async def _enrich_detection_chain(
    objects: list[dict],
    source_identity_id: str,
) -> tuple[list[dict], list[dict]]:
    """Emit full ATT&CK detection chain SDOs + SROs for the bundle.

    Walks every x-procedure's x_technique_refs, queries Neo4j for the
    detection chain (AttackPattern -> DetectionStrategy -> Analytic ->
    DataComponent), and emits:

      - `attack-pattern` SDO stubs for each referenced technique
      - `x-mitre-detection-strategy` SDO stubs
      - `x-mitre-analytic` SDO stubs
      - `x-mitre-data-component` SDO stubs
      - SROs: DetectionStrategy->AttackPattern (detects),
              DetectionStrategy->Analytic (has-analytic),
              Analytic->DataComponent (uses-data-component),
              DataComponent->DataSource is skipped (no DataSource in query)

    Dedupes all nodes and edges by (source, target, type) so repeated
    technique references across procedures don't fan out. Tolerates
    missing Neo4j (logs warning, returns empty).
    """
    # Collect technique refs from procedures
    all_technique_ids: set[str] = set()
    for obj in objects:
        if obj.get("type") != X_PROCEDURE_TYPE:
            continue
        all_technique_ids.update(obj.get("x_technique_refs", []))

    if not all_technique_ids:
        return [], []

    # Same split logic as log source enrichment: real UUIDs vs T-number refs
    stix_ids: list[str] = []
    mitre_ids: list[str] = []
    for tid in all_technique_ids:
        if _ATTACK_PATTERN_UUID_RE.match(tid):
            stix_ids.append(tid)
        else:
            mitre_ids.append(tid.replace("attack-pattern--", ""))

    try:
        rows = await run_query(
            _DETECTION_CHAIN_QUERY,
            {"stix_ids": stix_ids, "mitre_ids": mitre_ids},
        )
    except Exception as e:
        logger.warning(
            "serialize_stix: detection chain enrichment failed (%s); "
            "skipping chain SDOs", e,
        )
        return [], []

    now = _now_iso()
    emitted_objects: dict[str, dict] = {}  # stix_id -> object
    emitted_rels: set[tuple[str, str, str]] = set()
    rels: list[dict] = []

    def _stub_sdo(stix_id: str, stix_type: str, name: str, description: str | None, refs_list) -> None:
        if not stix_id or stix_id in emitted_objects:
            return
        ext = _mitre_ext_ref(refs_list)
        sdo = {
            "type": stix_type,
            "spec_version": STIX_SPEC_VERSION,
            "id": stix_id,
            "created": now,
            "modified": now,
            "created_by_ref": source_identity_id,
            "name": name or "",
            "description": description or "",
        }
        if ext:
            sdo["external_references"] = [ext]
        emitted_objects[stix_id] = sdo

    def _add_rel(src: str, rel_type: str, tgt: str) -> None:
        if not src or not tgt:
            return
        key = (src, tgt, rel_type)
        if key in emitted_rels:
            return
        emitted_rels.add(key)
        rels.append(_make_sro(src, rel_type, tgt, source_identity_id))

    for row in rows:
        ap_id = row.get("ap_id")
        ds_id = row.get("ds_id")
        a_id = row.get("a_id")
        dc_id = row.get("dc_id")

        if ap_id:
            _stub_sdo(
                ap_id, "attack-pattern",
                row.get("ap_name"), row.get("ap_description"),
                row.get("ap_refs"),
            )
        if ds_id:
            _stub_sdo(
                ds_id, "x-mitre-detection-strategy",
                row.get("ds_name"), row.get("ds_description"),
                row.get("ds_refs"),
            )
            _add_rel(ds_id, "detects", ap_id)
        if a_id:
            _stub_sdo(
                a_id, "x-mitre-analytic",
                row.get("a_name"), row.get("a_description"),
                row.get("a_refs"),
            )
            _add_rel(ds_id, "has-analytic", a_id)
        if dc_id:
            _stub_sdo(
                dc_id, "x-mitre-data-component",
                row.get("dc_name"), row.get("dc_description"),
                row.get("dc_refs"),
            )
            _add_rel(a_id, "uses-data-component", dc_id)

    logger.info(
        "serialize_stix: detection chain emitted %d SDO stubs, %d SROs",
        len(emitted_objects), len(rels),
    )

    return list(emitted_objects.values()), rels


def _draft_to_procedure(
    draft: dict, ndraft: dict, source_identity_id: str
) -> dict:
    """Convert a procedure draft + normalization data to an x-procedure SDO."""
    now = _now_iso()

    procedure = {
        "type": X_PROCEDURE_TYPE,
        "spec_version": STIX_SPEC_VERSION,
        "id": f"{X_PROCEDURE_TYPE}--{uuid.uuid4()}",
        "created": now,
        "modified": now,
        "name": draft.get("name", ""),
        "description": draft.get("description", ""),
        "created_by_ref": source_identity_id,
        # Declares the x-procedure extension (STIX 2.1 §7.3). The bundle
        # embeds the matching extension-definition; the validator checks both.
        "extensions": extension_declaration(X_PROCEDURE_EXTENSION_ID),
    }

    # Technique references
    # stix_id is populated by extract_techniques via ATT&CK resolution.
    # An unresolved technique is OMITTED, never fabricated. The old fallback
    # emitted `attack-pattern--T1190`, which is not a valid STIX identifier
    # (STIX requires a UUID suffix) and resolves to nothing in or out of the
    # bundle — reference-integrity validation only missed it because it waves
    # through anything prefixed `attack-pattern--`. Dropping is louder in the
    # right way: a procedure left with no AP at all gets flagged by the
    # tuple-semantics validator instead of shipping a broken ref.
    techniques = draft.get("techniques", [])
    if techniques:
        tech_refs = []
        for t in techniques:
            stix_id = t.get("stix_id")
            if stix_id:
                tech_refs.append(stix_id)
            else:
                logger.warning(
                    "serialize: dropping technique %s from %r — no resolved "
                    "ATT&CK STIX UUID. Ref omitted rather than fabricated; "
                    "the procedure loses this mapping.",
                    t.get("technique_id", "?"), draft.get("name", "unnamed"),
                )
        if tech_refs:
            procedure["x_technique_refs"] = tech_refs

    # Kill chain phases — schema-required tactic mapping. Prefer the draft's
    # explicit list; derive from per-technique tactic fields when the draft
    # doesn't set it (drafting today often leaves kill_chain_phases unset
    # despite each TechniqueMapping having a tactic). Without this, the
    # procedure ships without tactic info — the validator's fingerprint
    # formula reads tactics from kill_chain_phases, so a missing list would
    # force a fingerprint_recomputed correction on every bundle.
    kcp = draft.get("kill_chain_phases", [])
    if not kcp:
        seen_tactics: set[str] = set()
        derived: list[dict] = []
        for t in draft.get("techniques", []) or []:
            tactic = t.get("tactic")
            if tactic and tactic not in seen_tactics:
                seen_tactics.add(tactic)
                derived.append({"kill_chain_name": "mitre-attack", "phase_name": tactic})
        kcp = derived
    if kcp:
        procedure["kill_chain_phases"] = kcp

    # Platforms (OpenTide vocab)
    platforms = draft.get("platforms", [])
    if platforms:
        procedure["x_platforms"] = platforms

    # Component refs (ordered SCO sequence: Process, File, Registry, etc.)
    component_refs = draft.get("components_refs", [])
    if component_refs:
        procedure["x_components_refs"] = component_refs

    # Log source refs (x-log-source objects for detection mapping)
    log_source_refs = draft.get("log_source_refs", [])
    if log_source_refs:
        procedure["x_log_source_refs"] = log_source_refs

    # Confidence (composite from normalization)
    procedure["confidence"] = ndraft.get("composite_confidence", 0)

    # Temporal fields
    for field in ("first_observed", "last_observed"):
        val = draft.get(field)
        if val:
            procedure[field] = val

    # Source references. Drafting writes source_refs=[] expecting
    # serialization to populate from the report author identity. Default
    # to [source_identity_id] when the draft didn't supply explicit refs
    # so every procedure carries provenance back to the source. Without
    # this, the validate_bundle node has to backfill at severity=repaired
    # on every bundle (was: silently shipped with empty x_source_refs).
    source_refs = draft.get("source_refs", []) or [source_identity_id]
    procedure["x_source_refs"] = source_refs

    # Vulnerability references — drafts populate this list with internal
    # entity_ids (the drafting node's CVE-pattern detector matches
    # mentions in chunk text + draft name/description against
    # validated_entities of type=vulnerability). The serializer resolves
    # those entity_ids to vulnerability--<UUID> STIX IDs in a post-pass
    # below (after every entity has been STIX-IDed and registered).
    # Stash unresolved refs here; resolve in serialize_stix's main flow.
    vuln_refs = draft.get("vulnerability_refs", [])
    if vuln_refs:
        procedure["x_vulnerability_refs"] = list(vuln_refs)

    # Procedure type (reporting, observed, hypothetical)
    proc_type = draft.get("procedure_type", "reporting")
    procedure["x_procedure_type"] = proc_type

    # Source-fidelity category — propagated from the chunk(s) backing
    # this procedure. Surfaces in the UI as a badge so analysts know
    # whether the evidence is verbatim prose, a code block, a vision-
    # extracted figure, or LLM-paraphrased. Future Story Mode (multi-chunk
    # procedures) will collapse mixed values to "hybrid" at this layer.
    provenance = draft.get("source_provenance")
    if provenance:
        procedure["x_source_provenance"] = provenance

    # Chain-separation propagation: chain_label tags every procedure with
    # the attack-chain it belongs to. Multi-intrusion sources produce
    # multiple labels (e.g. "SharePoint primary intrusion", "Veeam
    # intrusion"). The attack-flow object's start_refs are derived from
    # procedures whose chain_root=True at bundle-assembly time.
    chain_label = draft.get("chain_label") or ""
    if chain_label:
        procedure["x_chain_label"] = chain_label
    if draft.get("chain_root"):
        procedure["x_chain_root"] = True

    # ATT&CK Flow sequencing is NOT embedded on the procedure. The draft's
    # effect_refs (forward edges normalize derived from predecessor_indices)
    # materialize as PRECEDES SROs in _build_relationships and as the
    # attack-flow object's start_refs. x_effect_refs / x_flow_ref were
    # removed in v0.5.0-draft — x-procedure is the intelligence object, not
    # a node in the flow DAG, so it carries no embedded next-step pointers.

    # Behavioral fingerprint (computed at ingestion, never manual)
    fingerprint = ndraft.get("fingerprint")
    if fingerprint:
        procedure["x_fingerprint"] = fingerprint

    return procedure


def _rule_to_indicator(rule: dict, source_identity_id: str) -> dict:
    """Convert a source-provided detection rule to a STIX Indicator SDO."""
    now = _now_iso()

    return {
        "type": "indicator",
        "spec_version": STIX_SPEC_VERSION,
        "id": f"indicator--{uuid.uuid4()}",
        "created": now,
        "modified": now,
        "name": rule.get("description", "") or f"Detection rule ({rule.get('rule_type', 'unknown')})",
        "pattern_type": rule.get("rule_type", "stix"),
        "pattern": rule.get("rule_content", ""),
        "valid_from": now,
        "created_by_ref": source_identity_id,
    }


# =============================================================================
# Relationship builder
# =============================================================================

def _build_entity_value_index(
    entities: list[dict],
    id_registry: dict[str, str],
) -> dict[str, str]:
    """Index entities by lowercase value -> STIX ID for the IoC-linking pass.

    Restricted to SCO-typed entities only (file, ipv4-addr, domain-name, url,
    email-addr, windows-registry-key, mutex, process, software, user-account).
    SDO-typed entities (TOOL, MALWARE, INTRUSION_SET, etc.) are excluded —
    those get attached to procedures via per-procedure `tools_used` /
    `malware_used` USES SROs instead of via has-observable.

    Without this filter, `chunk.artifacts.process_names` like "WScript" or
    "PowerShell" matched TOOL entities by value and produced spurious
    `procedure --has-observable--> tool` SROs.

    Both `value` and `edited_value` are indexed so analyst edits at gate_0
    are reachable. Removed entities are skipped.
    """
    _removed = GateAction.REMOVE.value
    out: dict[str, str] = {}
    for ent in entities:
        if ent.get("gate_action") == _removed:
            continue
        # Resolve the entity's effective STIX type; skip SDOs entirely.
        effective_type = ent.get("edited_type") or ent.get("entity_type", "")
        stix_type = _ENTITY_TO_STIX_TYPE.get(effective_type)
        if stix_type not in _SCO_TYPES:
            continue
        stix_id = id_registry.get(ent.get("entity_id", ""))
        if not stix_id:
            continue
        for key in (ent.get("edited_value"), ent.get("value")):
            if isinstance(key, str) and key.strip():
                out.setdefault(key.strip().lower(), stix_id)
    return out


def _resolve_entity_names(
    names: list[str],
    entities: list[dict],
    id_registry: dict[str, str],
) -> list[str]:
    """Resolve a list of entity NAMES to their STIX IDs via the entity registry.

    Used by `_build_relationships` to translate the per-procedure
    `tools_used` / `malware_used` lists (entity names emitted by the
    drafting LLM) into STIX IDs. Matching is case-insensitive on the
    entity's `edited_value` first, falling back to `value`.

    Returns a deduped list of resolved STIX IDs. Names that don't match
    any entity (e.g. drafting LLM emitted a name the entity extractor
    didn't capture) are silently dropped — the analyst sees the missing
    edge at the bundle review canvas.
    """
    if not names:
        return []
    name_set = {n.strip().lower() for n in names if isinstance(n, str) and n.strip()}
    resolved: list[str] = []
    seen: set[str] = set()
    for ent in entities:
        ent_name = (ent.get("edited_value") or ent.get("value") or "").strip().lower()
        if not ent_name or ent_name not in name_set:
            continue
        stix_id = id_registry.get(ent.get("entity_id", ""))
        if stix_id and stix_id not in seen:
            resolved.append(stix_id)
            seen.add(stix_id)
    return resolved


def _actors_for_procedure(
    original_draft: dict,
    intrusion_sets: list[dict],
    id_registry: dict[str, str],
) -> list[str]:
    """STIX ids of the intrusion sets THIS procedure is attributed to.

    Attribution used to be a cross product: every intrusion-set entity was
    linked to every procedure, every technique and every tool in the source.
    On a report whose entire point was that two actors are UNRELATED
    ("the vendor assesses that the operations are independent"), the bundle asserted
    that OtherGroup and UNC0002 each used all ten of UNC0001's procedures
    and all twenty-one of its techniques — the opposite of what the report
    said. False attribution is the most damaging error class in threat
    intelligence.

    Resolution order:
      1. the LLM's per-procedure `attributed_actors`, validated against the
         intrusion sets the analyst approved at gate_0 — an actor the
         extractor never produced cannot be invented here;
      2. failing that, if the source has EXACTLY ONE intrusion set, use it —
         single-actor reports are the common case and refer to the actor as
         "the group" far more often than by name;
      3. otherwise none. With several actors present and no per-procedure
         signal, guessing is what created the bug; the contrast actors stay
         in the bundle as context SDOs with no fabricated edges.
    """
    named = _resolve_entity_names(
        original_draft.get("attributed_actors", []) or [],
        intrusion_sets,
        id_registry,
    )
    if named:
        return named
    if len(intrusion_sets) == 1:
        single = id_registry.get(intrusion_sets[0].get("entity_id", ""))
        return [single] if single else []
    return []


def _analyst_removed_rel_keys(
    state: PipelineState,
) -> set[tuple[str, str, str, str, str]]:
    """Translate Gate 2 removals from preview ids into content keys.

    `gate2_removed_rel_ids` holds `relationship_preview` ids; the preview
    entries carry the names and types the key is built from. Doing the
    translation here keeps `_build_relationships` free of gate vocabulary.
    """
    removed_ids = set(state.get("gate2_removed_rel_ids") or [])
    if not removed_ids:
        return set()
    preview = state.get("relationship_preview") or []
    keys = {
        preview_relationship_key(rel)
        for rel in preview
        if rel.get("id") in removed_ids
    }
    logger.info(
        "serialize_stix: %d analyst removals resolved to %d relationship keys",
        len(removed_ids), len(keys),
    )
    return keys


def _build_relationships(
    normalized_drafts: list[dict],
    draft_lookup: dict[str, dict],
    entities: list[dict],
    id_registry: dict[str, str],
    source_identity_id: str,
    is_sequential: bool = True,
    chunks: list[dict] | None = None,
    chunk_operators: dict[str, dict] | None = None,
    op_id_to_stix_id: dict[str, str] | None = None,
    chunk_to_proc_stix_id: dict[str, str] | None = None,
    chunk_conditions: dict[str, dict] | None = None,
    cond_anchor_to_stix_id: dict[str, str] | None = None,
    removed_rel_keys: set[tuple[str, str, str, str, str]] | None = None,
) -> list[dict]:
    """Build all STIX Relationship Objects (SROs).

    Relationships created:

    Per-procedure (filtered by per-procedure attribution where noted):
        - procedure USES attack-pattern (dual wiring with x_technique_refs)
        - procedure USES malware (per draft.malware_used, not a source-wide fan-out)
        - procedure USES tool (per draft.tools_used, not a source-wide fan-out)
        - procedure HAS-OBSERVABLE sco (per chunk.artifacts → SCO match)
        - procedure EXPLOITS vulnerability
        - sco COMPONENT-OF procedure (from raw_command_lines)
        - procedure PRECEDES procedure (sequencing, gated on is_sequential)

    Actor-level (intrusion-set + campaign aggregations):
        - intrusion-set USES procedure (one per IS × procedure)
        - intrusion-set ATTRIBUTED-TO threat-actor
        - intrusion-set USES tool / malware / attack-pattern (aggregated
          across all procedures — bounded cardinality, per-IS-actor pivot)
        - intrusion-set TARGETS identity / location / software
        - intrusion-set EXPLOITS vulnerability (gated on procedure evidence)
        - campaign ATTRIBUTED-TO intrusion-set (skipped for MaaS-only sources)
        - campaign USES tool / malware / attack-pattern (aggregated)
        - campaign TARGETS location / software
        - campaign EXPLOITS vulnerability (gated on procedure evidence)

    Known gaps (not yet emitted):
        - malware USES attack-pattern (capability techniques) — would
          require either an additional LLM signal capturing "this
          malware family is capable of these techniques separately
          from the observed-in-this-incident techniques", or an
          analyst-driven step at gate_2. The drafting prompt restricts
          procedures to OBSERVED techniques; pure-capability techniques
          are deliberately excluded. Defer until a real analyst use
          case surfaces the need.
        - tool USES attack-pattern — same shape as malware capability,
          same deferral reason.
    """
    rels: list[dict] = []

    # Index entities by type for relationship building
    _removed = GateAction.REMOVE.value
    intrusion_sets = [e for e in entities if e.get("entity_type") == EntityType.INTRUSION_SET.value and e.get("gate_action") != _removed]
    threat_actors = [e for e in entities if e.get("entity_type") == EntityType.THREAT_ACTOR.value and e.get("gate_action") != _removed]
    malware = [e for e in entities if e.get("entity_type") == EntityType.MALWARE.value and e.get("gate_action") != _removed]
    tools = [e for e in entities if e.get("entity_type") == EntityType.TOOL.value and e.get("gate_action") != _removed]
    campaigns = [e for e in entities if e.get("entity_type") == EntityType.CAMPAIGN.value and e.get("gate_action") != _removed]
    victim_orgs = [e for e in entities if e.get("entity_type") == EntityType.ORGANIZATION.value and e.get("organization_role") == "victim" and e.get("gate_action") != _removed]
    vulnerabilities = [e for e in entities if e.get("entity_type") == EntityType.VULNERABILITY.value and e.get("gate_action") != _removed]
    victim_locations = [e for e in entities if e.get("entity_type") == EntityType.LOCATION.value and e.get("location_role") == "victim" and e.get("gate_action") != _removed]
    software_assets = [e for e in entities if e.get("entity_type") == EntityType.SOFTWARE.value and e.get("gate_action") != _removed]

    for ndraft in normalized_drafts:
        draft_id = ndraft.get("draft_id", "")
        original = draft_lookup.get(draft_id, {})
        procedure_stix_id = id_registry.get(draft_id)
        if not procedure_stix_id:
            continue

        # Dual wiring: procedure USES attack-pattern
        # Complements x_technique_refs embedded on the procedure object.
        # The SRO enables graph traversal; the embedded ref enables
        # bundle-portable filtering without relationship resolution.
        techniques = original.get("techniques", [])
        for tech in techniques:
            tech_stix_id = tech.get("stix_id")
            if tech_stix_id:
                rels.append(_make_sro(
                    procedure_stix_id, "uses", tech_stix_id,
                    source_identity_id,
                ))

        # Component-of: SCO COMPONENT-OF procedure
        # Links each component SCO to the parent procedure for
        # structural decomposition traversal.
        component_refs = original.get("components_refs", [])
        for comp_ref in component_refs:
            if comp_ref:
                rels.append(_make_sro(
                    comp_ref, "component-of", procedure_stix_id,
                    source_identity_id,
                ))

        # Procedure HAS-OBSERVABLE SCO. Per-procedure observables come from
        # the IoC-linking pass in `serialize_stix` (chunk.artifacts matched
        # to validated_entities by value). Distinct from
        # component-of: these are observables the procedure TOUCHES
        # (registry keys, C2 domains, file hashes, IPs) — not the
        # process tree of its commands. Distribution maps has-observable
        # to the HAS_OBSERVABLE Neo4j edge.
        observable_refs = original.get("observable_refs", [])
        for obs_ref in observable_refs:
            if obs_ref:
                rels.append(_make_sro(
                    procedure_stix_id, "has-observable", obs_ref,
                    source_identity_id,
                ))

        # Procedure EXPLOITS vulnerability
        vuln_refs = original.get("vulnerability_refs", [])
        for vref in vuln_refs:
            vuln_stix_id = id_registry.get(vref) or vref
            if vuln_stix_id:
                rels.append(_make_sro(
                    procedure_stix_id, "exploits", vuln_stix_id,
                    source_identity_id,
                ))

        # Intrusion set USES procedure — only actors this procedure is
        # actually attributed to (see _actors_for_procedure).
        for iset_stix_id in _actors_for_procedure(
            original, intrusion_sets, id_registry,
        ):
            rels.append(_make_sro(
                iset_stix_id, "uses", procedure_stix_id,
                source_identity_id,
            ))

        # Procedure USES malware (per-procedure attribution).
        # STIX convention: the active object (the procedure / behavior)
        # uses the artifact (malware family). Per-procedure attribution
        # comes from draft.malware_used (entity NAMES emitted by the
        # drafting LLM). A source-wide fan-out here linked every procedure
        # to every malware in the source (the Rclone-attached-to-everything
        # bug). If draft.malware_used is empty, no malware
        # USES SROs are emitted for this procedure.
        proc_malware = _resolve_entity_names(
            original.get("malware_used", []), malware, id_registry,
        )
        for mw_stix_id in proc_malware:
            rels.append(_make_sro(
                procedure_stix_id, "uses", mw_stix_id,
                source_identity_id,
            ))

        # Procedure USES tool (per-procedure attribution).
        # Same STIX convention as Malware above; same per-procedure
        # filter from draft.tools_used.
        proc_tools = _resolve_entity_names(
            original.get("tools_used", []), tools, id_registry,
        )
        for tool_stix_id in proc_tools:
            rels.append(_make_sro(
                procedure_stix_id, "uses", tool_stix_id,
                source_identity_id,
            ))

    # PRECEDES relationships (sequencing) — batched outside the per-draft
    # loop because operator-routed edges need the full chunk DAG to
    # deduplicate feed-in / feed-out edges (one branch operator with N
    # successors generates N+1 edges total, not N edges per traversal).
    #
    # Skipped entirely for non-sequential sources — the bundle ships
    # as a flat collection of x-procedures without flow scaffolding,
    # mirroring the source's catalog shape. chunk_operators is also
    # empty in that case (normalize skips inference), so the batch call
    # would no-op anyway, but the explicit guard documents the gate.
    if is_sequential:
        for src_stix_id, tgt_stix_id in route_precedes_through_operators(
            chunks or [],
            chunk_operators or {},
            op_id_to_stix_id or {},
            chunk_to_proc_stix_id or {},
            chunk_conditions=chunk_conditions or {},
            cond_anchor_to_stix_id=cond_anchor_to_stix_id or {},
        ):
            rels.append(_make_sro(
                src_stix_id, "precedes", tgt_stix_id, source_identity_id,
            ))

    # MaaS attribution guard: determine if only MaaS malware is present
    # with no explicitly attributed intrusion set. MaaS malware is
    # operated by many unrelated actors, so creating attributed-to SROs
    # from campaigns to intrusion sets would produce false attribution.
    has_maas_only = (
        all(mw.get("is_maas", False) for mw in malware)
        and len(malware) > 0
    )

    # Campaign ATTRIBUTED_TO intrusion set
    # Guarded: skip if only MaaS malware present and the intrusion set
    # was inferred (not explicitly named in the source).
    for campaign in campaigns:
        campaign_stix_id = id_registry.get(campaign.get("entity_id", ""))
        if not campaign_stix_id:
            continue
        for iset in intrusion_sets:
            iset_stix_id = id_registry.get(iset.get("entity_id", ""))
            if not iset_stix_id:
                continue
            # Guard: if MaaS-only and intrusion set was inferred
            # (confidence < 0.8), skip the attribution link.
            iset_confidence = iset.get("confidence", 1.0)
            if has_maas_only and iset_confidence < 0.8:
                logger.info(
                    "MaaS guard: skipping campaign->intrusion-set "
                    "attributed-to for %s (MaaS-only, low confidence)",
                    iset.get("value", ""),
                )
                continue
            rels.append(_make_sro(
                campaign_stix_id, "attributed-to", iset_stix_id,
                source_identity_id,
            ))

    # Intrusion set ATTRIBUTED_TO threat actor (cluster -> real-world actor)
    for iset in intrusion_sets:
        iset_stix_id = id_registry.get(iset.get("entity_id", ""))
        if not iset_stix_id:
            continue
        for ta in threat_actors:
            ta_stix_id = id_registry.get(ta.get("entity_id", ""))
            if ta_stix_id:
                rels.append(_make_sro(
                    iset_stix_id, "attributed-to", ta_stix_id,
                    source_identity_id,
                ))

    # Intrusion set TARGETS victim organizations
    for iset in intrusion_sets:
        iset_stix_id = id_registry.get(iset.get("entity_id", ""))
        if not iset_stix_id:
            continue
        for victim in victim_orgs:
            victim_stix_id = id_registry.get(victim.get("entity_id", ""))
            if victim_stix_id:
                rels.append(_make_sro(
                    iset_stix_id, "targets", victim_stix_id,
                    source_identity_id,
                ))

    # Intrusion-set / Campaign first-class aggregation edges. Without them
    # both are research-orphaned in the bundle — only `attributed-to ←
    # campaign` and `uses → procedure` edges — and an
    # analyst landing on the IS or Campaign node in the BundleGraph couldn't
    # answer "what does this group use, what techniques do they implement"
    # without traversing N procedure nodes.
    #
    # We aggregate per-procedure tool/malware/technique data into per-IS /
    # per-Campaign sets and emit one edge per (actor, unique target).
    # Cardinality is bounded — if 14 procedures all use Rclone, there's
    # still one IS→Rclone edge — so this doesn't recreate the prior
    # N×M tool-fanout bug.
    #
    # vulnerability/exploits edges for IS+Campaign are emitted later in the
    # `Campaign/IntrusionSet EXPLOITS vulnerability (gated on procedure
    # evidence)` block; not duplicated here.
    if intrusion_sets or campaigns:
        # Roll each actor's profile up from the procedures IT is attributed
        # to, rather than from every procedure in the source. A contrast actor
        # ("unlike OtherGroup...") is attributed no procedures and therefore
        # inherits no tooling or techniques.
        per_actor_tools: dict[str, set[str]] = {}
        per_actor_malware: dict[str, set[str]] = {}
        per_actor_techniques: dict[str, set[str]] = {}
        # Campaign-level rollup stays source-wide: a campaign is the container
        # for the activity the report describes, so every extracted procedure
        # belongs to it. Only ACTOR attribution was ever the false claim.
        agg_tool_ids: set[str] = set()
        agg_malware_ids: set[str] = set()
        agg_technique_ids: set[str] = set()

        for ndraft in normalized_drafts:
            draft_id_local = ndraft.get("draft_id", "")
            orig = draft_lookup.get(draft_id_local, {})
            draft_tools = set(
                _resolve_entity_names(orig.get("tools_used", []), tools, id_registry)
            )
            draft_malware = set(
                _resolve_entity_names(orig.get("malware_used", []), malware, id_registry)
            )
            draft_techniques = {
                tech.get("stix_id") for tech in orig.get("techniques", [])
                if isinstance(tech, dict) and tech.get("stix_id")
            }
            agg_tool_ids.update(draft_tools)
            agg_malware_ids.update(draft_malware)
            agg_technique_ids.update(draft_techniques)

            for actor_id in _actors_for_procedure(orig, intrusion_sets, id_registry):
                per_actor_tools.setdefault(actor_id, set()).update(draft_tools)
                per_actor_malware.setdefault(actor_id, set()).update(draft_malware)
                per_actor_techniques.setdefault(actor_id, set()).update(draft_techniques)

        for iset in intrusion_sets:
            iset_stix_id = id_registry.get(iset.get("entity_id", ""))
            if not iset_stix_id:
                continue
            # IS uses tool — analyst pivot for "what's this group's tooling?"
            for tool_id in per_actor_tools.get(iset_stix_id, set()):
                rels.append(_make_sro(
                    iset_stix_id, "uses", tool_id, source_identity_id,
                ))
            # IS uses malware
            for mw_id in per_actor_malware.get(iset_stix_id, set()):
                rels.append(_make_sro(
                    iset_stix_id, "uses", mw_id, source_identity_id,
                ))
            # IS uses attack-pattern — the TTP profile
            for tech_id in per_actor_techniques.get(iset_stix_id, set()):
                rels.append(_make_sro(
                    iset_stix_id, "uses", tech_id, source_identity_id,
                ))

        # Campaign aggregation — same shape as IS. STIX 2.1 Campaign supports
        # `uses` to attack-pattern / malware / tool. Emit only when the
        # campaign exists in the bundle (no campaign → no edges).
        for campaign in campaigns:
            campaign_stix_id = id_registry.get(campaign.get("entity_id", ""))
            if not campaign_stix_id:
                continue
            for tool_id in agg_tool_ids:
                rels.append(_make_sro(
                    campaign_stix_id, "uses", tool_id, source_identity_id,
                ))
            for mw_id in agg_malware_ids:
                rels.append(_make_sro(
                    campaign_stix_id, "uses", mw_id, source_identity_id,
                ))
            for tech_id in agg_technique_ids:
                rels.append(_make_sro(
                    campaign_stix_id, "uses", tech_id, source_identity_id,
                ))

    # Campaign/IntrusionSet TARGETS victim locations
    # Gated on location_role == "victim" (filtered above). Context/origin
    # locations remain in the bundle as SDOs but do not receive SROs.
    for loc in victim_locations:
        loc_stix_id = id_registry.get(loc.get("entity_id", ""))
        if not loc_stix_id:
            continue
        for iset in intrusion_sets:
            iset_stix_id = id_registry.get(iset.get("entity_id", ""))
            if iset_stix_id:
                rels.append(_make_sro(
                    iset_stix_id, "targets", loc_stix_id,
                    source_identity_id,
                ))
        for campaign in campaigns:
            campaign_stix_id = id_registry.get(campaign.get("entity_id", ""))
            if campaign_stix_id:
                rels.append(_make_sro(
                    campaign_stix_id, "targets", loc_stix_id,
                    source_identity_id,
                ))

    # Campaign/IntrusionSet TARGETS victim software (assets)
    # Ungated fan-out: extraction prompt restricts SOFTWARE entities to
    # "software being attacked or exploited", so every Software SCO is
    # treated as a valid target. Attacker tooling lands in TOOL type.
    for sw in software_assets:
        sw_stix_id = id_registry.get(sw.get("entity_id", ""))
        if not sw_stix_id:
            continue
        for iset in intrusion_sets:
            iset_stix_id = id_registry.get(iset.get("entity_id", ""))
            if iset_stix_id:
                rels.append(_make_sro(
                    iset_stix_id, "targets", sw_stix_id,
                    source_identity_id,
                ))
        for campaign in campaigns:
            campaign_stix_id = id_registry.get(campaign.get("entity_id", ""))
            if campaign_stix_id:
                rels.append(_make_sro(
                    campaign_stix_id, "targets", sw_stix_id,
                    source_identity_id,
                ))

    # Campaign/IntrusionSet EXPLOITS vulnerability (gated on procedure evidence)
    # Only emit when at least one procedure in the bundle carries the CVE in
    # its vulnerability_refs. Context-only CVEs do not get SROs.
    exploited_cve_stix_ids: set[str] = set()
    for ndraft in normalized_drafts:
        draft_id = ndraft.get("draft_id", "")
        original = draft_lookup.get(draft_id, {})
        for vref in original.get("vulnerability_refs", []):
            resolved = id_registry.get(vref) or vref
            if resolved:
                exploited_cve_stix_ids.add(resolved)

    for cve_stix_id in exploited_cve_stix_ids:
        for iset in intrusion_sets:
            iset_stix_id = id_registry.get(iset.get("entity_id", ""))
            if iset_stix_id:
                rels.append(_make_sro(
                    iset_stix_id, "exploits", cve_stix_id,
                    source_identity_id,
                ))
        for campaign in campaigns:
            campaign_stix_id = id_registry.get(campaign.get("entity_id", ""))
            if campaign_stix_id:
                rels.append(_make_sro(
                    campaign_stix_id, "exploits", cve_stix_id,
                    source_identity_id,
                ))

    if removed_rel_keys:
        rels = _drop_analyst_removed(
            rels, removed_rel_keys, normalized_drafts, draft_lookup,
            entities, id_registry,
        )

    return rels


def _drop_analyst_removed(
    rels: list[dict],
    removed_rel_keys: set[tuple[str, str, str, str, str]],
    normalized_drafts: list[dict],
    draft_lookup: dict[str, dict],
    entities: list[dict],
    id_registry: dict[str, str],
) -> list[dict]:
    """Drop SROs the analyst removed at Gate 2.

    The removal list arrives keyed by `relationship_preview` id, which the
    caller has already translated into content keys (source name / verb /
    target name / both STIX types). SROs carry only STIX UUIDs, so we build
    the reverse index -- stix_id -> (name, stix_type) -- from the same
    entities and drafts the SROs were built from, then key each SRO the same
    way the preview was keyed.

    Anything we cannot name is KEPT. A removal that silently took out the
    wrong edge would be worse than one that failed to apply, and the analyst
    can see what shipped.
    """
    name_index: dict[str, tuple[str, str]] = {}

    for entity in entities:
        stix_id = id_registry.get(entity.get("entity_id", ""))
        if not stix_id:
            continue
        effective_type = entity.get("edited_type") or entity.get("entity_type", "")
        stix_type = _ENTITY_TO_STIX_TYPE.get(effective_type, "")
        name = entity.get("edited_value") or entity.get("value", "")
        name_index[stix_id] = (name, stix_type)

    for ndraft in normalized_drafts:
        draft_id = ndraft.get("draft_id", "")
        stix_id = id_registry.get(draft_id)
        if not stix_id:
            continue
        original = draft_lookup.get(draft_id, {})
        name_index[stix_id] = (
            original.get("name", draft_id), "x-procedure",
        )
        # attack-pattern ids come off the draft's techniques, which is the
        # same place the preview read their display names from.
        for tech in original.get("techniques", []) or []:
            tech_stix = tech.get("stix_id")
            if tech_stix:
                name_index[tech_stix] = (
                    tech.get("technique_name", tech.get("technique_id", "")),
                    "attack-pattern",
                )

    kept: list[dict] = []
    dropped = 0
    for rel in rels:
        src = name_index.get(rel.get("source_ref", ""))
        tgt = name_index.get(rel.get("target_ref", ""))
        if not src or not tgt:
            kept.append(rel)
            continue
        key = relationship_key(
            src[0], rel.get("relationship_type"), tgt[0], src[1], tgt[1],
        )
        if key in removed_rel_keys:
            dropped += 1
            continue
        kept.append(rel)

    logger.info(
        "serialize_stix: dropped %d/%d SROs removed by the analyst at gate_2",
        dropped, len(rels),
    )
    return kept


def _make_sro(
    source_ref: str,
    relationship_type: str,
    target_ref: str,
    created_by_ref: str,
) -> dict:
    """Create a STIX Relationship Object (SRO)."""
    now = _now_iso()
    return {
        "type": "relationship",
        "spec_version": STIX_SPEC_VERSION,
        "id": f"relationship--{uuid.uuid4()}",
        "created": now,
        "modified": now,
        "relationship_type": relationship_type,
        "source_ref": source_ref,
        "target_ref": target_ref,
        "created_by_ref": created_by_ref,
    }


# =============================================================================
# Validation
# =============================================================================

def _validate_bundle(
    bundle: dict,
    no_coverage_proc_ids: set[str] | None = None,
) -> tuple[dict, list[str]]:
    """Run four validators on the assembled bundle.

    Args:
        bundle: The STIX 2.1 bundle to validate.
        no_coverage_proc_ids: Set of procedure STIX IDs whose techniques
            ALL lack detection coverage in ATT&CK. These procedures get
            a downgraded LS validation (warning instead of error).

    Returns:
        (validation_results dict, list of error strings)
    """
    errors: list[str] = []

    # 1. Schema validation (basic structural checks)
    schema_valid = _validate_schema(bundle, errors)

    # 2. Reference integrity
    ref_valid = _validate_references(bundle, errors)

    # 3. ATT&CK Flow structure
    flow_valid = _validate_attack_flow(bundle, errors)

    # 4. Tuple semantics: P = { AP ≠ ∅, LS ≠ ∅, ⟨C⟩ ≠ ∅ }
    tuple_valid = _validate_tuple_semantics(bundle, errors, no_coverage_proc_ids)

    results = {
        "schema": schema_valid,
        "reference_integrity": ref_valid,
        "attack_flow": flow_valid,
        "tuple_semantics": tuple_valid,
    }

    return results, errors


def _validate_schema(bundle: dict, errors: list[str]) -> bool:
    """STIX 2.1 schema validation.

    Pass 1: structural checks (required fields present). These stay because
    they produce friendlier messages than raw JSON Schema output, and because
    they also cover the ATT&CK Flow extension objects, for which OASIS
    publishes no schema.

    Pass 2: full JSON Schema validation — every object against the vendored
    OASIS STIX 2.1 schema for its type, and x-procedure against
    x_procedure_v3.json. See app.services.stix_schema. When the schema corpus
    can't load, pass 2 contributes nothing and pass 1 remains the floor, so
    a packaging fault can't fail every bundle.
    """
    valid = True
    objects = bundle.get("objects", [])

    if not objects:
        errors.append("Bundle contains no objects")
        return False

    for i, obj in enumerate(objects):
        obj_type = obj.get("type", "unknown")
        obj_id = obj.get("id", f"object-{i}")

        # All objects must have type and id
        if not obj.get("type"):
            errors.append(f"Object {i}: missing 'type' field")
            valid = False
        if not obj.get("id"):
            errors.append(f"Object {i}: missing 'id' field")
            valid = False

        # SDOs must have created/modified (SCOs don't)
        if obj_type not in _SCO_TYPES and obj_type != "relationship":
            if not obj.get("created"):
                errors.append(f"{obj_id}: missing 'created' field")
                valid = False
            if not obj.get("modified"):
                errors.append(f"{obj_id}: missing 'modified' field")
                valid = False

        # Relationships must have source_ref, target_ref, relationship_type
        if obj_type == "relationship":
            for field in ("source_ref", "target_ref", "relationship_type"):
                if not obj.get(field):
                    errors.append(f"{obj_id}: missing '{field}' field")
                    valid = False

        # x-procedure v0.5.0-draft: 6 required fields + name must be non-empty
        if obj_type == X_PROCEDURE_TYPE:
            if not obj.get("name"):
                errors.append(f"{obj_id}: x-procedure missing required 'name' field")
                valid = False
            if not obj.get("spec_version"):
                errors.append(f"{obj_id}: x-procedure missing required 'spec_version' field")
                valid = False

    # Pass 2: full JSON Schema validation against the real schemas.
    schema_errors = stix_schema.validate_bundle_objects(objects)
    if schema_errors:
        valid = False
        errors.extend(f"schema: {e}" for e in schema_errors)
        logger.error(
            "serialize_stix: %d STIX schema violation(s); first: %s",
            len(schema_errors), schema_errors[0],
        )

    return valid


def _validate_references(bundle: dict, errors: list[str]) -> bool:
    """Check that all id references resolve within the bundle."""
    valid = True
    objects = bundle.get("objects", [])

    # Build set of all IDs in the bundle
    all_ids = {obj.get("id") for obj in objects if obj.get("id")}

    # Check reference fields (single-value refs)
    ref_fields = [
        "created_by_ref", "source_ref", "target_ref",
    ]
    # Check reference fields (array refs)
    list_ref_fields = [
        "x_technique_refs", "x_source_refs", "x_vulnerability_refs",
        "x_components_refs", "x_log_source_refs",
    ]

    for obj in objects:
        obj_id = obj.get("id", "unknown")

        for field in ref_fields:
            ref = obj.get(field)
            if ref and ref not in all_ids:
                # Allow technique refs that point to external ATT&CK patterns
                if field == "source_ref" or field == "target_ref":
                    if ref.startswith("attack-pattern--"):
                        continue
                errors.append(f"{obj_id}: {field} '{ref}' not found in bundle")
                valid = False

        for field in list_ref_fields:
            refs = obj.get(field, [])
            for ref in refs:
                if ref and ref not in all_ids:
                    # Allow external ATT&CK pattern references
                    if ref.startswith("attack-pattern--"):
                        continue
                    errors.append(f"{obj_id}: {field} contains '{ref}' not found in bundle")
                    valid = False

    return valid


def _validate_attack_flow(bundle: dict, errors: list[str]) -> bool:
    """Validate ATT&CK Flow sequencing structure.

    v0.5.0-draft expresses sequencing with PRECEDES SROs (procedure /
    attack-operator / attack-condition nodes linked by relationship objects
    of type "precedes") rather than embedding next-step pointers on the
    x-procedure. x-procedure is the intelligence object, not a flow node.

    Checks that:
    1. Every PRECEDES SRO's source_ref and target_ref resolve in-bundle
    2. At most one attack-flow object exists
    3. The PRECEDES sequencing DAG is acyclic
    """
    valid = True
    objects = bundle.get("objects", [])

    procedures = [o for o in objects if o.get("type") == X_PROCEDURE_TYPE]
    if len(procedures) <= 1:
        return True  # No sequencing to validate

    all_ids = {o.get("id") for o in objects if o.get("id")}

    precedes = [
        o for o in objects
        if o.get("type") == "relationship"
        and o.get("relationship_type") == "precedes"
    ]

    # 1. PRECEDES endpoints resolve in-bundle.
    for sro in precedes:
        for end in ("source_ref", "target_ref"):
            ref = sro.get(end)
            if ref and ref not in all_ids:
                errors.append(
                    f"ATT&CK Flow: precedes SRO {sro.get('id')} "
                    f"{end} '{ref}' not found in bundle"
                )
                valid = False

    # 2. At most one attack-flow object (multi-chain sources collapse to
    # one flow with multiple start_refs).
    flow_objs = [o for o in objects if o.get("type") == "attack-flow"]
    if len(flow_objs) > 1:
        errors.append(
            f"ATT&CK Flow: bundle has {len(flow_objs)} attack-flow "
            f"objects — expected at most 1"
        )
        valid = False

    # 3. Cycle detection over the PRECEDES graph (DFS). Nodes are any STIX
    # ID appearing as a precedes endpoint — procedures plus the operator /
    # condition nodes that sequencing routes through.
    adj: dict[str, list[str]] = {}
    nodes: set[str] = set()
    for sro in precedes:
        src = sro.get("source_ref", "")
        tgt = sro.get("target_ref", "")
        if not src or not tgt:
            continue
        adj.setdefault(src, []).append(tgt)
        nodes.add(src)
        nodes.add(tgt)

    visited: set[str] = set()
    in_stack: set[str] = set()

    def has_cycle(node: str) -> bool:
        visited.add(node)
        in_stack.add(node)
        for neighbor in adj.get(node, []):
            if neighbor in in_stack:
                return True
            if neighbor not in visited:
                if has_cycle(neighbor):
                    return True
        in_stack.discard(node)
        return False

    for node in nodes:
        if node not in visited:
            if has_cycle(node):
                errors.append("ATT&CK Flow: cycle detected in sequencing DAG")
                valid = False
                break

    return valid


def _validate_tuple_semantics(
    bundle: dict,
    errors: list[str],
    no_coverage_proc_ids: set[str] | None = None,
) -> bool:
    """Validate x-procedure tuple constraint: P = { AP ≠ ∅, LS ≠ ∅, ⟨C⟩ ≠ ∅ }.

    Every x-procedure SHOULD have:
    - At least one attack-pattern ref (AP ≠ ∅)
    - At least one log source ref (LS ≠ ∅)
    - At least one component ref (⟨C⟩ ≠ ∅)

    Missing elements are warnings, not hard failures -- low-confidence
    procedures may have incomplete decomposition. But the validator
    flags them so Gate 2 analysts can decide.

    Severity rules per tuple element:
    - AP (x_technique_refs): error if confidence >= 70, warning otherwise.
      Every procedure MUST map to at least one technique to be useful.
    - LS (x_log_source_refs): warning only when ATT&CK has no detection
      coverage for the procedure's techniques (framework gap). Error at
      confidence >= 70 otherwise.
    - C (x_components_refs): always warning, never error. Many source
      reports describe behavior narratively without explicit command lines
      or observable artifacts. Missing components reflect source fidelity,
      not a pipeline defect. Confidence already captures this gap.
    """
    valid = True
    no_coverage = no_coverage_proc_ids or set()
    objects = bundle.get("objects", [])
    procedures = [o for o in objects if o.get("type") == X_PROCEDURE_TYPE]

    for proc in procedures:
        proc_id = proc.get("id", "unknown")
        proc_name = proc.get("name", "unnamed")
        confidence = proc.get("confidence", 0)

        has_ap = bool(proc.get("x_technique_refs"))
        has_ls = bool(proc.get("x_log_source_refs"))
        has_c = bool(proc.get("x_components_refs"))

        missing = []
        if not has_ap:
            missing.append("AP (x_technique_refs)")
        if not has_ls:
            missing.append("LS (x_log_source_refs)")
        if not has_c:
            missing.append("C (x_components_refs)")

        if missing:
            # Classify each missing element as error or warning.
            # Only AP can be a hard error. LS and C are always warnings
            # under specific conditions (see docstring).
            ls_excused = not has_ls and proc_id in no_coverage

            error_missing = []
            warn_missing = []
            for m in missing:
                if m.startswith("C "):
                    # C is always warning: source reports often lack
                    # explicit commands or observable artifacts.
                    warn_missing.append(m)
                elif m.startswith("LS ") and ls_excused:
                    warn_missing.append(m + " [no ATT&CK coverage]")
                elif m.startswith("LS ") and confidence < 70:
                    warn_missing.append(m)
                elif m.startswith("AP ") and confidence < 70:
                    warn_missing.append(m)
                else:
                    # AP at confidence >= 70, or LS at >= 70 with coverage
                    error_missing.append(m)

            if error_missing:
                msg = (
                    f"Tuple semantics (error): {proc_id} "
                    f"'{proc_name}' (confidence={confidence}) "
                    f"missing: {', '.join(error_missing)}"
                )
                errors.append(msg)
                valid = False

            if warn_missing:
                msg = (
                    f"Tuple semantics (warning): {proc_id} "
                    f"'{proc_name}' (confidence={confidence}) "
                    f"missing: {', '.join(warn_missing)}"
                )
                errors.append(msg)

    return valid


# =============================================================================
# Helpers
# =============================================================================

def _now_iso() -> str:
    """Current UTC timestamp in STIX format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _detect_hash_type(hash_value: str) -> str | None:
    """Guess hash algorithm from string length. Returns None for unknown
    lengths so the caller can skip rather than mis-typing the SCO."""
    clean = hash_value.strip().lower()
    length = len(clean)

    if length == 32:
        return "MD5"
    elif length == 40:
        return "SHA-1"
    elif length == 64:
        return "SHA-256"
    elif length == 128:
        return "SHA-512"
    else:
        return None
