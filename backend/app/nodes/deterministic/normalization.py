"""normalize node: Stage 5 of the extraction pipeline.

Takes approved procedure drafts from Gate 1 and applies five treatments:
1. Name standardization - map names to canonical vocabulary from Neo4j
2. Composite confidence scoring - weighted blend replacing raw confidence
3. Fingerprint computation - behavioral fingerprint for dedup/similarity
4. Sequencing resolution - convert sequence_index/predecessors to effect_refs
5. Correlation - find similar procedures in the archive (never dedup)

IMPORTANT: This node is deterministic + Neo4j. No LLM calls.
The Neo4j queries are optional: if Neo4j is unreachable, the node
still produces normalized drafts with degraded quality (warnings added).

WHAT THIS NODE READS:
    - drafts: All procedure drafts
    - gate1_approved_draft_ids: Which drafts were approved at Gate 1
    - validated_entities: For context in correlation
    - metadata: Source reliability feeds into composite confidence
    - source_reliability: 0-100, set at ingestion

WHAT THIS NODE WRITES:
    - normalized_drafts: List of NormalizedDraft dicts
    - drafts: Updated with effect_refs (sequencing resolution)
    - status: NORMALIZING
    - current_node: "normalize"
"""

from __future__ import annotations

import logging
from dataclasses import asdict

from app.nodes.deterministic.attack_conditions import extract_conditions
from app.nodes.deterministic.attack_operators import infer_operators
from app.nodes.deterministic.relationship_keys import (
    preview_relationship_id,
)
from app.graph.state import (
    CorrelationResult,
    EntityType,
    GateAction,
    NormalizedDraft,
    PipelineState,
    PipelineStatus,
)

logger = logging.getLogger(__name__)

# Confidence weights for composite scoring.
# Must sum to 1.0.
WEIGHT_SOURCE_RELIABILITY = 0.30
WEIGHT_CONTEXT_COMPLETENESS = 0.25
WEIGHT_BEHAVIORAL_CONFIDENCE = 0.45


def normalize(state: PipelineState) -> dict:
    """Stage 5: Normalize approved procedure drafts.

    Filters to approved drafts only, then applies the five treatments
    listed in the module docstring (name standardization, composite
    confidence, fingerprint, sequencing resolution, correlation).

    Returns partial state update with normalized_drafts.
    """
    logger.info("normalize: starting")

    approved_ids = set(state.get("gate1_approved_draft_ids", []))
    all_drafts = state.get("drafts", [])
    source_reliability = state.get("source_reliability", 50)

    # Filter to approved drafts only
    approved_drafts = [d for d in all_drafts if d.get("draft_id") in approved_ids]

    if not approved_drafts:
        logger.warning("normalize: no approved drafts to normalize")
        return {
            "normalized_drafts": [],
            "status": PipelineStatus.NORMALIZING.value,
            "current_node": "normalize",
        }

    logger.info("normalize: processing %d approved drafts", len(approved_drafts))

    # Resolve sequencing: invert sequence_index / predecessor_indices into
    # draft-level effect_refs (forward edges). The serializer reads these to
    # build the attack-flow object + PRECEDES SROs.
    _resolve_sequencing(approved_drafts)

    normalized: list[dict] = []
    for draft in approved_drafts:
        ndraft = _normalize_draft(draft, source_reliability, state)
        normalized.append(ndraft)

    logger.info("normalize: completed %d normalized drafts", len(normalized))

    # Attack Flow operator inference. Walks the chunk
    # DAG geometry and emits attack-operator entries for branch/converge
    # points so the serializer can route precedes through them instead of
    # flattening to pairwise procedure→procedure edges. existing_operators
    # carries forward any analyst kind overrides set at gate_chunks
    # — same chunk geometry → same operator_id → override
    # survives. is_sequential=False skips inference entirely (catalog
    # sources don't get flow scaffolding).
    chunks = state.get("chunks", []) or []
    is_sequential = bool(state.get("is_sequential", True))
    existing_operators = state.get("chunk_operators", {}) or {}
    # Extract conditions FIRST so we can suppress OR-branch operator
    # inference at any chunk that anchors a condition (the condition
    # replaces the branch operator semantically).
    chunk_conditions = extract_conditions(chunks, is_sequential)
    chunk_operators = infer_operators(
        chunks, is_sequential,
        existing_operators=existing_operators,
        condition_anchors=set(chunk_conditions.keys()),
    )

    # Build relationship preview for Gate 2 review.
    # Uses human-readable names instead of STIX UUIDs (those are assigned later
    # during serialization). Mirrors the relationship types from _build_relationships
    # in serialization.py so the analyst sees the same relationships that will
    # end up in the final STIX bundle.
    validated_entities = state.get("validated_entities", [])
    relationship_preview = _derive_relationship_preview(
        approved_drafts, validated_entities,
    )
    logger.info("normalize: derived %d relationship previews", len(relationship_preview))

    return {
        "normalized_drafts": normalized,
        "relationship_preview": relationship_preview,
        "chunk_operators": chunk_operators,
        "chunk_conditions": chunk_conditions,
        "drafts": all_drafts,  # Write back with resolved effect_refs
        "status": PipelineStatus.NORMALIZING.value,
        "current_node": "normalize",
    }


def _normalize_draft(
    draft: dict,
    source_reliability: int,
    state: PipelineState,
) -> dict:
    """Normalize a single approved draft.

    Applies name standardization, composite confidence, and correlation.
    Each step is isolated so failures don't block the others.
    """
    draft_id = draft.get("draft_id", "")

    # Step 1: Name standardization
    standardized_names = _standardize_names(draft)

    # Step 2: Composite confidence scoring
    behavioral_confidence = draft.get("confidence", 50)
    context_completeness = _assess_context_completeness(draft)

    composite = int(
        (WEIGHT_SOURCE_RELIABILITY * source_reliability)
        + (WEIGHT_CONTEXT_COMPLETENESS * context_completeness)
        + (WEIGHT_BEHAVIORAL_CONFIDENCE * behavioral_confidence)
    )
    # Clamp to 0-100
    composite = max(0, min(100, composite))

    confidence_breakdown = {
        "source_reliability": source_reliability,
        "context_completeness": context_completeness,
        "behavioral_confidence": behavioral_confidence,
        "weights": {
            "source_reliability": WEIGHT_SOURCE_RELIABILITY,
            "context_completeness": WEIGHT_CONTEXT_COMPLETENESS,
            "behavioral_confidence": WEIGHT_BEHAVIORAL_CONFIDENCE,
        },
    }

    # Step 3: Fingerprint computation
    fingerprint = _compute_fingerprint(draft)

    # Step 4: Correlation (placeholder - Neo4j queries added later)
    correlations = _correlate_with_archive(draft, state)

    # Step 5: Enrichment (placeholder - Neo4j queries added later)
    enrichment = _enrich_from_neo4j(draft)

    ndraft = NormalizedDraft(
        draft_id=draft_id,
        composite_confidence=composite,
        confidence_breakdown=confidence_breakdown,
        standardized_names=standardized_names,
        correlations=[CorrelationResult(**c) for c in correlations],
        enrichment=enrichment,
        fingerprint=fingerprint,
    )

    return asdict(ndraft)


def _standardize_names(draft: dict) -> dict:
    """Standardize entity and technique names against canonical vocabulary.

    Currently: basic normalization rules.
    Future: Neo4j lookup for canonical mappings.

    Returns dict of {original: canonical} for any names that changed.
    """
    mappings: dict[str, str] = {}

    # Normalize technique names from the draft's techniques list
    for technique in draft.get("techniques", []):
        name = technique.get("technique_name", "")
        if not name:
            continue

        # Basic normalization: consistent casing, strip whitespace
        normalized = name.strip()

        # Common CTI name standardizations
        canonical = _CANONICAL_NAMES.get(normalized.lower())
        if canonical and canonical != normalized:
            mappings[normalized] = canonical

    # Normalize platform names
    for platform in draft.get("platforms", []):
        canonical = _CANONICAL_PLATFORMS.get(platform.lower())
        if canonical and canonical != platform:
            mappings[platform] = canonical

    return mappings


def _assess_context_completeness(draft: dict) -> int:
    """Score how complete the draft's context is (0-100).

    Checks for presence of key fields that indicate a well-supported
    procedure. Missing fields reduce the score.
    """
    score = 0
    max_score = 100

    # Has a description (required, but check quality)
    description = draft.get("description", "")
    if len(description) > 100:
        score += 25
    elif len(description) > 30:
        score += 15
    elif description:
        score += 5

    # Has command lines (strong evidence)
    if draft.get("raw_command_lines"):
        score += 25

    # Has technique mappings
    techniques = draft.get("techniques", [])
    if len(techniques) >= 1:
        score += 15
    if len(techniques) >= 2:
        score += 5

    # Has temporal data
    if draft.get("first_observed"):
        score += 10

    # Has source references
    if draft.get("source_refs"):
        score += 10

    # Has platform data
    if draft.get("platforms"):
        score += 5

    # Not flagged as detail gap
    if not draft.get("detail_gap", False):
        score += 5

    return min(score, max_score)


def _correlate_with_archive(draft: dict, state: PipelineState) -> list[dict]:
    """Find similar procedures in the Neo4j archive.

    Currently: returns an empty list (not implemented).
    Future: Cypher queries to find procedures with
    overlapping techniques, tools, actors, or temporal clusters.

    Every procedure is a distinct record. Correlation is for
    analytical value only, never dedup or merge.
    """
    # Not implemented; the graph queries this would run are documented above.
    # Example query structure:
    # MATCH (p:Procedure)-[:IMPLEMENTS_TECHNIQUE]->(t:AttackPattern)
    # WHERE t.external_id IN $technique_ids
    # RETURN p.id, p.name, count(t) as overlap
    return []


def _enrich_from_neo4j(draft: dict) -> dict:
    """Pull related mitigations, detections, and techniques from Neo4j.

    Currently: returns an empty dict (not implemented).
    Future: for each technique in the draft, pull:
    - CourseOfAction nodes (mitigations)
    - DetectionStrategy / Analytic nodes
    - Related sub/parent techniques
    """
    # Not implemented; see the docstring.
    return {}


def _compute_fingerprint(draft: dict) -> str:
    """Compute a behavioral fingerprint for the procedure draft.

    Thin wrapper around the canonical formula in `app.utils.fingerprint`.
    Both this call site and `bundle_validator._recompute_fingerprints`
    route through that module so the formula stays in lockstep — drift
    between them is what causes `fingerprint_recomputed` corrections to
    fire on every bundle.
    """
    from app.utils.fingerprint import compute_fingerprint_for_draft
    return compute_fingerprint_for_draft(draft)


def _resolve_sequencing(approved_drafts: list[dict]) -> None:
    """Convert numeric sequencing to draft-ID forward edges (effect_refs).

    Drafts carry sequence_index and predecessor_indices from the chunking
    stage. This function inverts them into a draft-internal `effect_refs`
    list (the draft_ids each procedure leads to). The serializer reads these
    to derive attack-flow start_refs (the chain roots) and to emit PRECEDES
    SROs; they are NOT serialized onto the x-procedure itself.

    Mutates drafts in-place.

    The predecessor model (chunk carries who comes BEFORE it) is inverted
    to the effect model (procedure carries who comes AFTER it) because flow
    sequencing points forward.
    """
    # Build index: sequence_index -> draft
    seq_to_draft: dict[int, dict] = {}
    for draft in approved_drafts:
        idx = draft.get("sequence_index", 0)
        if idx > 0:
            seq_to_draft[idx] = draft

    if not seq_to_draft:
        return  # No sequencing data

    # Invert: for each draft's predecessor_indices, add this draft
    # as an effect of each predecessor.
    for draft in approved_drafts:
        predecessors = draft.get("predecessor_indices", [])
        draft_id = draft["draft_id"]

        for pred_idx in predecessors:
            pred_draft = seq_to_draft.get(pred_idx)
            if pred_draft:
                # Add this draft_id to predecessor's effect_refs.
                # The serializer will resolve draft_ids to STIX IDs.
                pred_draft.setdefault("effect_refs", [])
                if draft_id not in pred_draft["effect_refs"]:
                    pred_draft["effect_refs"].append(draft_id)

    # The serializer builds the attack-flow object and its start_refs from
    # these draft-level effect_refs; no placeholder flow id is needed here.
    sequenced = [d for d in approved_drafts if d.get("sequence_index", 0) > 0]
    if len(sequenced) > 1:
        logger.info(
            "normalize: resolved sequencing for %d procedures", len(sequenced),
        )


# ---------------------------------------------------------------------------
# Canonical name mappings
#
# Currently hardcoded common variations; could be populated from a Neo4j
# canonical vocabulary later.
# ---------------------------------------------------------------------------

_CANONICAL_NAMES: dict[str, str] = {
    "powershell": "PowerShell",
    "cmd": "Windows Command Shell",
    "cmd.exe": "Windows Command Shell",
    "command prompt": "Windows Command Shell",
    "cobalt strike": "Cobalt Strike",
    "cobaltstrike": "Cobalt Strike",
    "mimikatz": "Mimikatz",
    "psexec": "PsExec",
    "certutil": "certutil",
    "certutil.exe": "certutil",
    "bitsadmin": "BITSAdmin",
    "bitsadmin.exe": "BITSAdmin",
    "wmic": "WMIC",
    "wmic.exe": "WMIC",
    "rundll32": "Rundll32",
    "rundll32.exe": "Rundll32",
    "regsvr32": "Regsvr32",
    "regsvr32.exe": "Regsvr32",
    "mshta": "Mshta",
    "mshta.exe": "Mshta",
}

_CANONICAL_PLATFORMS: dict[str, str] = {
    "windows": "windows",
    "linux": "linux",
    "macos": "macos",
    "windows::server": "windows::server",
    "windows::workstation": "windows::workstation",
    "linux::server": "linux::server",
}


# ---------------------------------------------------------------------------
# Relationship preview for Gate 2
#
# Mirrors the SRO logic in serialization._build_relationships but uses
# human-readable names instead of STIX UUIDs (which don't exist yet).
# The analyst reviews these at Gate 2 before serialization runs.
# ---------------------------------------------------------------------------

def _derive_relationship_preview(
    approved_drafts: list[dict],
    validated_entities: list[dict],
) -> list[dict]:
    """Build a list of human-readable relationship previews.

    Each preview dict has:
        id: str               - stable ID for frontend keying and decision tracking
        relationship_type: str
        source_name: str      - human-readable name of the source
        target_name: str      - human-readable name of the target
        source_type: str      - STIX type (e.g. "x-procedure", "intrusion-set")
        target_type: str      - STIX type
        reviewable: bool      - True if analyst should review; False for auto-derived

    Directionality matches serialization output:
        uses:           x-procedure → attack-pattern | tool | malware
        uses:           intrusion-set → x-procedure
        exploits:       x-procedure → vulnerability
        targets:        x-procedure → identity
        attributed-to:  campaign → intrusion-set | intrusion-set → threat-actor
        precedes:       x-procedure → x-procedure

    Categorization (PDM whitepaper §8.4):
        Inherent (reviewable=False):
            - uses: x-procedure → attack-pattern (technique mapping from Gate 1)
        Reviewable (reviewable=True):
            - uses (tool/malware/IS linkage)
            - exploits, attributed-to, targets, precedes

    Coverage vs. serialization._build_relationships:
        Included: uses, exploits, attributed-to, targets, precedes
        Excluded: component-of (SCO refs don't exist until serialization
                  builds them from raw command lines and IOC entities;
                  these are auto-derived and don't need analyst review)
    """
    rels: list[dict] = []
    _removed = GateAction.REMOVE.value
    _counter = 0

    def _next_id() -> str:
        nonlocal _counter
        _counter += 1
        return f"relp_{_counter}"

    def _entity_name(entity: dict) -> str:
        return entity.get("edited_value") or entity.get("value", "")

    # Index entities by type, excluding removed
    def _by_type(etype: str) -> list[dict]:
        return [
            e for e in validated_entities
            if e.get("entity_type") == etype and e.get("gate_action") != _removed
        ]

    intrusion_sets = _by_type(EntityType.INTRUSION_SET.value)
    threat_actors = _by_type(EntityType.THREAT_ACTOR.value)
    malware_list = _by_type(EntityType.MALWARE.value)
    tools_list = _by_type(EntityType.TOOL.value)
    campaigns = _by_type(EntityType.CAMPAIGN.value)
    victim_orgs = [
        e for e in validated_entities
        if e.get("entity_type") == EntityType.ORGANIZATION.value
        and e.get("organization_role") == "victim"
        and e.get("gate_action") != _removed
    ]
    vulnerabilities = _by_type(EntityType.VULNERABILITY.value)

    # Build sequence_index -> draft name for PRECEDES
    seq_to_draft: dict[int, dict] = {}
    for draft in approved_drafts:
        idx = draft.get("sequence_index", 0)
        if idx > 0:
            seq_to_draft[idx] = draft

    for draft in approved_drafts:
        proc_name = draft.get("name", draft.get("draft_id", ""))

        # x-procedure USES attack-pattern — INHERENT from Gate 1
        for tech in draft.get("techniques", []):
            tech_name = tech.get("technique_name", tech.get("technique_id", ""))
            if tech_name:
                rels.append({
                    "id": _next_id(),
                    "relationship_type": "uses",
                    "source_name": proc_name,
                    "target_name": tech_name,
                    "source_type": "x-procedure",
                    "target_type": "attack-pattern",
                    "reviewable": False,
                })

        # x-procedure EXPLOITS vulnerability — REVIEWABLE (analytical judgment)
        # Bundle ref: exploits, x-procedure → vulnerability
        for vref in draft.get("vulnerability_refs", []):
            # Match vulnerability entity by entity_id or value
            vuln_name = vref
            for v in vulnerabilities:
                if v.get("entity_id") == vref or v.get("value") == vref:
                    vuln_name = _entity_name(v)
                    break
            rels.append({
                "id": _next_id(),
                "relationship_type": "exploits",
                "source_name": proc_name,
                "target_name": vuln_name,
                "source_type": "x-procedure",
                "target_type": "vulnerability",
                "reviewable": True,
            })

        # intrusion-set USES x-procedure — REVIEWABLE (attribution claim)
        # Bundle ref: uses, intrusion-set → x-procedure
        for iset in intrusion_sets:
            iset_name = _entity_name(iset)
            rels.append({
                "id": _next_id(),
                "relationship_type": "uses",
                "source_name": iset_name,
                "target_name": proc_name,
                "source_type": "intrusion-set",
                "target_type": "x-procedure",
                "reviewable": True,
            })

        # x-procedure USES malware — REVIEWABLE (tooling judgment).
        # Per-procedure attribution from draft.malware_used. Names match
        # case-insensitively against entity edited_value or value.
        proc_malware_names = {
            n.strip().lower()
            for n in (draft.get("malware_used") or [])
            if isinstance(n, str) and n.strip()
        }
        # Empty means empty — the serializer emits no malware SROs when
        # `malware_used` is empty, so previewing a fan-out here showed the
        # analyst edges that could never ship. A fallback to the source-wide
        # malware list for drafts predating the field fired on an EMPTY list
        # too, which is exactly what the drafting LLM correctly emits for a
        # procedure that uses no malware. On one campaign source that put
        # "Harvest Credentials via Voice Phishing --uses--> GNU shred" in
        # front of the analyst.
        iter_malware = [
            mw for mw in malware_list
            if (_entity_name(mw) or "").strip().lower() in proc_malware_names
        ]
        for mw in iter_malware:
            mw_name = _entity_name(mw)
            rels.append({
                "id": _next_id(),
                "relationship_type": "uses",
                "source_name": proc_name,
                "target_name": mw_name,
                "source_type": "x-procedure",
                "target_type": "malware",
                "reviewable": True,
            })

        # x-procedure USES tool — REVIEWABLE (tooling judgment). Same
        # per-procedure attribution from draft.tools_used.
        proc_tool_names = {
            n.strip().lower()
            for n in (draft.get("tools_used") or [])
            if isinstance(n, str) and n.strip()
        }
        # Same as malware above: empty means empty, matching the serializer.
        iter_tools = [
            t for t in tools_list
            if (_entity_name(t) or "").strip().lower() in proc_tool_names
        ]
        for t in iter_tools:
            tool_name = _entity_name(t)
            rels.append({
                "id": _next_id(),
                "relationship_type": "uses",
                "source_name": proc_name,
                "target_name": tool_name,
                "source_type": "x-procedure",
                "target_type": "tool",
                "reviewable": True,
            })

        # x-procedure PRECEDES x-procedure — REVIEWABLE (flow may change)
        # Bundle ref: precedes, x-procedure → x-procedure
        for pred_idx in draft.get("predecessor_indices", []):
            pred_draft = seq_to_draft.get(pred_idx)
            if pred_draft:
                pred_name = pred_draft.get("name", pred_draft.get("draft_id", ""))
                rels.append({
                    "id": _next_id(),
                    "relationship_type": "precedes",
                    "source_name": pred_name,
                    "target_name": proc_name,
                    "source_type": "x-procedure",
                    "target_type": "x-procedure",
                    "reviewable": True,
                })

    # MaaS attribution guard: same logic as serialization.py
    has_maas_only = (
        len(malware_list) > 0
        and all(mw.get("is_maas", False) for mw in malware_list)
    )

    # campaign ATTRIBUTED-TO intrusion-set — REVIEWABLE (attribution judgment)
    # Bundle ref: attributed-to, campaign → intrusion-set
    for campaign in campaigns:
        for iset in intrusion_sets:
            if has_maas_only and iset.get("confidence", 1.0) < 0.8:
                continue
            camp_name = _entity_name(campaign)
            iset_name = _entity_name(iset)
            rels.append({
                "id": _next_id(),
                "relationship_type": "attributed-to",
                "source_name": camp_name,
                "target_name": iset_name,
                "source_type": "campaign",
                "target_type": "intrusion-set",
                "reviewable": True,
            })

    # intrusion-set ATTRIBUTED-TO threat-actor — REVIEWABLE (attribution judgment)
    for iset in intrusion_sets:
        for ta in threat_actors:
            iset_name = _entity_name(iset)
            ta_name = _entity_name(ta)
            rels.append({
                "id": _next_id(),
                "relationship_type": "attributed-to",
                "source_name": iset_name,
                "target_name": ta_name,
                "source_type": "intrusion-set",
                "target_type": "threat-actor",
                "reviewable": True,
            })

    # x-procedure TARGETS victim organization — REVIEWABLE (victimology judgment)
    # Bundle ref: targets, x-procedure → identity
    for draft in approved_drafts:
        proc_name = draft.get("name", draft.get("draft_id", ""))
        for victim in victim_orgs:
            victim_name = _entity_name(victim)
            rels.append({
                "id": _next_id(),
                "relationship_type": "targets",
                "source_name": proc_name,
                "target_name": victim_name,
                "source_type": "x-procedure",
                "target_type": "identity",
                "reviewable": True,
            })

    # Re-key every entry by content. The `relp_N` values assigned above are
    # positional, so a `normalize` re-run renumbers them and any Gate 2
    # removal the analyst already recorded would silently re-map onto a
    # different relationship. Content-derived ids survive re-derivation,
    # which is what lets serialize_stix honor a removal.
    #
    # Collisions are possible in principle (two drafts sharing a name and
    # linking to the same target) — suffix them so ids stay unique and the
    # gate's id->entry map cannot lose an entry.
    seen_ids: dict[str, int] = {}
    for rel in rels:
        base = preview_relationship_id(rel)
        count = seen_ids.get(base, 0)
        seen_ids[base] = count + 1
        rel["id"] = base if count == 0 else f"{base}_{count}"

    return rels
