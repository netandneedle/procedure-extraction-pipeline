"""validate_bundle node: Stage 6b — check-and-correct on the assembled bundle.

Sits between serialize_stix and distribute. The serializer's job is to build
a STIX bundle; this node's job is to make sure that bundle is structurally
sound before it lands in Neo4j or in the persistent bundle store.

Two-phase design:

1. Recovery + auto-fix passes — non-failing corrections recorded as
   `bundle_corrections` entries on state. Severity buckets:
     - "auto_fix"  — expected housekeeping (SRO direction flip, dedup,
                     fingerprint recompute, role enum coercion)
     - "repaired"  — covered for an upstream bug (e.g. dangling ref
                     recovered from normalize output). Louder severity:
                     these mean the serializer or normalizer dropped
                     something and we should investigate.
     - "warn"      — informational (orphan SDOs, missing objective lead).
                     Surfaced but not load-bearing.

2. Hard-fail checks — when the bundle cannot be made structurally sound,
   set status=FAILED and write hard_fail entries so the source bounces
   to the Failed column with structured detail. Hard-fail short-circuits
   distribute (no Neo4j writes, no bundle store row).

WHAT THIS NODE READS:
    - stix_bundle              : the serializer's output bundle
    - normalized_objects (via normalized_drafts + draft_lookup): for resolver
    - is_sequential            : gates PRECEDES drop pass
    - validated_entities       : for the brand-as-malware check

WHAT THIS NODE WRITES:
    - stix_bundle              : possibly mutated by auto-fix passes
    - bundle_corrections       : list of structured correction records
    - bundle_validation_failed : True when hard-fail occurred
    - status                   : FAILED on hard-fail, SERIALIZING otherwise
    - error                    : summary string when hard-fail occurred
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from app.services.stix_schema import industry_sector_vocab
from app.graph.state import EntityType, PipelineState, PipelineStatus
from app.nodes.deterministic.extension_definitions import (
    ATTACK_FLOW_EXTENSION_ID,
    ATTACK_FLOW_TYPES,
    X_PROCEDURE_EXTENSION_ID,
)
from app.services import stix_schema
from app.services.attack_data import get_attack_data
from app.services.technique_pattern_brands import find_brand_techniques

logger = logging.getLogger(__name__)


# =============================================================================
# Constants
# =============================================================================

X_PROCEDURE_TYPE = "x-procedure"

# SRO directionality contracts. For each relationship_type, source_ref must
# resolve to an object whose type is in `valid_source` and target_ref to one
# whose type is in `valid_target`. When the SRO has a flipped pair (source
# matches valid_target AND target matches valid_source), we flip it and
# record an auto_fix entry. SROs that match neither orientation are left
# alone for the dangling-ref or relationship-shape checks to flag.
_SRO_DIRECTION_RULES: dict[str, dict[str, set[str]]] = {
    "uses": {
        # IntrusionSet/Campaign/x-procedure → Tool/Malware/AttackPattern/Infrastructure/x-procedure.
        # The two-way overlap on x-procedure (it can be both source and
        # target of a `uses` SRO) means same-orientation pairs get a pass
        # without flipping. Only true mismatches (e.g. tool→procedure) flip.
        "valid_source": {"intrusion-set", "campaign", "x-procedure", "threat-actor"},
        "valid_target": {"tool", "malware", "attack-pattern", "infrastructure", "x-procedure"},
    },
    "implements-technique": {
        "valid_source": {"x-procedure"},
        "valid_target": {"attack-pattern"},
    },
    "attributed-to": {
        "valid_source": {"campaign", "x-procedure", "intrusion-set"},
        "valid_target": {"intrusion-set", "threat-actor"},
    },
    "targets": {
        "valid_source": {"campaign", "tool", "malware", "x-procedure", "intrusion-set"},
        "valid_target": {"identity", "location", "software", "infrastructure"},
    },
    "exploits": {
        "valid_source": {"x-procedure", "intrusion-set", "campaign"},
        "valid_target": {"vulnerability"},
    },
    "describes": {
        "valid_source": {"report"},
        "valid_target": {"x-procedure"},
    },
    "mitigates": {
        "valid_source": {"course-of-action"},
        "valid_target": {"attack-pattern"},
    },
}

# Six well-known TLP marking-definition IDs from STIX 2.1 §10.1.4. Refs to
# these are legitimate even when the bundle doesn't embed the markings.
_STANDARD_TLP_MARKING_IDS = {
    "marking-definition--613f2e26-407d-48c7-9eca-b8e91df99dc9",  # TLP:WHITE
    "marking-definition--34098fce-860f-48ae-8e50-ebd3cc5e41da",  # TLP:GREEN
    "marking-definition--f88d31f6-486f-44da-b317-01333bde0b82",  # TLP:AMBER
    "marking-definition--5e57c739-391a-4eb3-b6be-7d15ca92d5ed",  # TLP:RED
    "marking-definition--bab4a63c-aed9-4cf5-a766-dfca5abac2bb",  # TLP 2.0:CLEAR
    "marking-definition--55d920b0-5e8b-4f79-9ee9-91f868d9b421",  # TLP 2.0:AMBER+STRICT
}

# Fields that hold a single STIX ID reference.
_SINGLE_REF_FIELDS = (
    "created_by_ref", "source_ref", "target_ref",
)
# Fields that hold a list of STIX ID references.
_LIST_REF_FIELDS = (
    "x_technique_refs", "x_source_refs", "x_vulnerability_refs",
    "x_components_refs", "x_log_source_refs",
    # x_observable_refs and x_asset_refs are deliberately absent: neither has
    # a producer. x_observable_refs stopped being embedded on procedures when
    # observables moved to has-observable SROs, and x_asset_refs never had
    # one. Both are also undeclared in x_procedure_v3.json, which is
    # additionalProperties:false — so a bundle carrying either now fails
    # schema validation before ref integrity ever looks at it.
    "object_refs",
    "object_marking_refs",
    # ATT&CK Flow: attack-flow.start_refs lists the chain-root procedures
    # the flow begins from. Must resolve in-bundle.
    "start_refs",
)

# x-procedure fields this validator requires to be non-empty. Stricter than
# x_procedure_v3.json, whose `required` list is the STIX envelope plus `name`:
# a procedure with no technique or no source is a pipeline bug.
_X_PROCEDURE_REQUIRED = ("name", "x_technique_refs", "x_source_refs")

# Maximum recovery iterations. Each iteration recovers one or more objects;
# stops when no new dangling refs surface. Five is a generous bound — typical
# SCO chains are 2-3 deep (Process → File + parent Process → File again).
_RECOVERY_FIXPOINT_LIMIT = 5

# Mapping ENtity -> STIX type. Repeated here rather than imported because
# some validation (brand-as-malware) operates on bundle objects whose
# entity_type isn't preserved at this stage.
_VALID_ORG_ROLES = {"victim", "sponsor", "publisher", "author", "other"}
_VALID_LOCATION_ROLES = {"victim", "origin", "context"}


# =============================================================================
# Node entry point
# =============================================================================

def validate_bundle(state: PipelineState) -> dict:
    """Node entry point. Never raises.

    The validator is the last gate before a bundle ships, and it walks
    arbitrary JSON with a lot of type assumptions (refs are strings, list
    fields are lists, ids are hashable). Fuzzing showed those
    assumptions break on malformed input in more places than can be
    sensibly guarded one by one — 173 crashes across 2000 mutants, all
    TypeErrors from type confusion.

    A crash here is the worst possible outcome: the exception escapes the
    node, the runner catches it, and the analyst gets
    "TypeError: unhashable type: 'dict'" on the source row with nothing
    actionable. Converting it into a hard_fail correction costs one wrapper
    and makes "the validator always returns a verdict" true by construction
    rather than by exhaustive hardening.

    The specific realistic malformations (null refs, wrong ref types) are
    still fixed properly upstream — this is the backstop, not the fix.
    """
    try:
        return _validate_bundle_impl(state)
    except Exception as e:  # noqa: BLE001 — see docstring
        logger.exception("validate_bundle crashed on a malformed bundle")
        return _finalize_failed(
            [_hard_fail(
                rule="validator_crashed",
                message=(
                    f"The bundle validator failed on this bundle: "
                    f"{type(e).__name__}: {e}. This is a malformed bundle or "
                    f"a validator bug — either way the bundle cannot ship."
                ),
            )],
            f"validator crashed: {type(e).__name__}: {e}",
        )


def _validate_bundle_impl(state: PipelineState) -> dict:
    """Stage 6b: check and correct the assembled STIX bundle.

    Mutates `state.stix_bundle` in-place via auto-fix passes, records every
    correction in `bundle_corrections`, and either succeeds (status stays
    SERIALIZING) or hard-fails (status flips to FAILED with a summary
    error string). Hard-fails skip distribute and synthesize_feedback
    via route_after_validate.

    The fail-vs-pass decision turns purely on whether any hard_fail entries
    were emitted. Auto-fix and repaired entries are not blocking.
    """
    logger.info("validate_bundle: starting")

    bundle = state.get("stix_bundle") or {}
    objects: list[dict] = bundle.get("objects", []) if isinstance(bundle, dict) else []

    # Collect every correction in order so the audit trail reads
    # chronologically. Seeded from state rather than starting empty:
    # serialize_stix runs first and records the procedures it had to omit, and
    # starting a fresh list here silently discarded them.
    corrections: list[dict] = list(state.get("bundle_corrections") or [])

    # -- Phase 0: bundle envelope sanity --
    if not isinstance(bundle, dict) or bundle.get("type") != "bundle":
        # No bundle to validate — distinct hard-fail case so callers can
        # see "validator ran but the input was unusable" vs "validator ran
        # and found problems."
        corrections.append(_hard_fail(
            rule="bundle_envelope_invalid",
            message="state.stix_bundle is not a STIX bundle (missing or wrong type)",
        ))
        return _finalize_failed(corrections, "bundle envelope invalid")

    if not objects:
        corrections.append(_hard_fail(
            rule="bundle_empty",
            message="bundle.objects is empty — nothing to validate",
        ))
        return _finalize_failed(corrections, "empty bundle")

    # -- Phase 0.5: ref types --
    # Every pass downstream assumes a ref is a string: it does set-membership
    # tests, dict lookups and .startswith on them. A non-string ref is not
    # merely invalid, it is *unhashable* if it happens to be a dict, and it
    # took the entire validator down with a TypeError — four separate sites,
    # all found by fuzzing. Individually guarding each site did
    # not converge, so the class is handled once, here.
    #
    # Short-circuit rather than continue: a bundle with malformed refs cannot
    # ship regardless, and running auto-fix passes over it only risks more
    # crashes for no benefit.
    ref_type_corrections = _check_ref_field_types(objects)
    if ref_type_corrections:
        corrections.extend(ref_type_corrections)
        return _finalize_failed(corrections, "malformed reference field")

    # -- Phase 1: recovery resolver (recover-or-fail) --
    objects, recovery_corrections = _recovery_resolve(objects, state)
    corrections.extend(recovery_corrections)

    # -- Phase 2: auto-fix passes --
    is_sequential = bool(state.get("is_sequential", True))

    objects, fix_corrections = _auto_fix_passes(objects, is_sequential)
    corrections.extend(fix_corrections)

    # -- Phase 3: hard-fail checks --
    hard_fail_corrections = _hard_fail_checks(objects, state)
    corrections.extend(hard_fail_corrections)

    # Update bundle's objects list. The auto-fix passes may have added or
    # dropped objects; reflect that in the bundle envelope.
    bundle["objects"] = objects

    has_hard_fail = any(c.get("severity") == "hard_fail" for c in corrections)

    # Summarize for logging — the audit trail itself lives on state.
    by_severity: dict[str, int] = defaultdict(int)
    for c in corrections:
        by_severity[c.get("severity", "unknown")] += 1
    logger.info(
        "validate_bundle: corrections by severity %s; hard_fail=%s",
        dict(by_severity), has_hard_fail,
    )
    # Surface every hard_fail at WARN so the underlying upstream bug is
    # visible in container logs, not just in the audit trail on state.
    if has_hard_fail:
        for c in corrections:
            if c.get("severity") == "hard_fail":
                logger.warning(
                    "validate_bundle hard_fail: %s — %s",
                    c.get("rule"), c.get("message"),
                )

    if has_hard_fail:
        summary = _summarize_hard_fails(corrections)
        return {
            "stix_bundle": bundle,
            "bundle_corrections": corrections,
            "bundle_validation_failed": True,
            "status": PipelineStatus.FAILED.value,
            "error": summary,
            "current_node": "validate_bundle",
        }

    return {
        "stix_bundle": bundle,
        "bundle_corrections": corrections,
        "bundle_validation_failed": False,
        # Status stays SERIALIZING — distribute will overwrite to DISTRIBUTING
        # almost immediately. We don't introduce a VALIDATING status because
        # success is silent: the analyst doesn't see this node's column on
        # the Kanban; failures land in Failed via the hard-fail path.
        "status": PipelineStatus.SERIALIZING.value,
        "current_node": "validate_bundle",
    }


# =============================================================================
# Phase 1: recovery resolver
# =============================================================================

def _recovery_resolve(
    objects: list[dict], state: PipelineState,
) -> tuple[list[dict], list[dict]]:
    """Try to recover any dangling refs by looking them up in upstream state.

    Iterates a fixpoint: recovering one object can introduce new dangling
    refs (e.g. a Process SCO references a File SCO that's also missing).
    Each iteration scans, recovers, and re-scans. Stops when no new
    refs are dangling, or after _RECOVERY_FIXPOINT_LIMIT iterations.

    Returns (possibly-augmented objects list, correction records).
    """
    corrections: list[dict] = []
    bundle_id_set: set[str] = {o.get("id") for o in objects if o.get("id")}
    upstream_index = _build_upstream_index(state)

    # Recovery sources we've already exhausted for a given ID — avoids
    # re-attempting a known-failed ID every iteration.
    attempted: set[str] = set()

    for iteration in range(_RECOVERY_FIXPOINT_LIMIT):
        dangling = _collect_dangling_refs(objects, bundle_id_set)
        if not dangling:
            break

        recovered_this_iter = 0
        for ctx in dangling:
            if ctx["ref_id"] in attempted:
                continue
            attempted.add(ctx["ref_id"])

            recovered = _resolve_one(ctx, upstream_index)
            if recovered is None:
                continue
            obj, source = recovered
            objects.append(obj)
            bundle_id_set.add(obj["id"])
            recovered_this_iter += 1
            corrections.append({
                "rule": "dangling_ref_recovered",
                "severity": "repaired",
                "ref_id": ctx["ref_id"],
                "ref_field": ctx["ref_field"],
                "holder_id": ctx["holder_id"],
                "holder_type": ctx["holder_type"],
                "recovered_from": source,
                "message": (
                    f"{ctx['holder_type']} {ctx['holder_id']} "
                    f"referenced {ctx['ref_id']} via {ctx['ref_field']}; "
                    f"object was missing from bundle but found in {source}. "
                    "Investigate whether the serializer or normalizer dropped it."
                ),
            })
            logger.warning(
                "validate_bundle: recovered %s from %s "
                "(referenced by %s.%s) — possible upstream bug",
                ctx["ref_id"], source, ctx["holder_id"], ctx["ref_field"],
            )

        if recovered_this_iter == 0:
            # Nothing new this round — anything still dangling is unrecoverable.
            break

    return objects, corrections


def _collect_dangling_refs(
    objects: list[dict], bundle_id_set: set[str],
) -> list[dict]:
    """Walk every object and emit a ResolveContext for each ref that
    points to an ID not present in the bundle, not legitimately external.
    """
    contexts: list[dict] = []

    for obj in objects:
        holder_id = obj.get("id", "<no-id>")
        holder_type = obj.get("type", "<no-type>")

        for field in _SINGLE_REF_FIELDS:
            ref = obj.get(field)
            # Refs must be strings. A non-string is malformed, and it is not
            # merely wrong — an unhashable one (a dict) raised TypeError on
            # the set membership test below and took the whole validator
            # down. Skip it here; the schema check reports it properly.
            if not isinstance(ref, str) or not ref:
                continue
            if ref in bundle_id_set:
                continue
            if _is_legitimately_external(ref, field):
                continue
            contexts.append({
                "ref_id": ref,
                "ref_field": field,
                "holder_id": holder_id,
                "holder_type": holder_type,
            })

        for field in _LIST_REF_FIELDS:
            refs = obj.get(field) or []
            # A bare string here would iterate character by character and
            # emit nonsense dangling refs; anything not a list/tuple is
            # malformed and left to the schema check.
            if not isinstance(refs, (list, tuple)):
                continue
            for ref in refs:
                if not isinstance(ref, str) or not ref:
                    continue
                if ref in bundle_id_set:
                    continue
                if _is_legitimately_external(ref, field):
                    continue
                contexts.append({
                    "ref_id": ref,
                    "ref_field": field,
                    "holder_id": holder_id,
                    "holder_type": holder_type,
                })

    return contexts


def _is_legitimately_external(ref: str, field: str) -> bool:
    """True when a ref is allowed to point outside the bundle.

    Currently:
    - TLP marking definitions in object_marking_refs (well-known IDs).
    - Any attack-pattern UUID, in any ref field. Matches the existing
      serializer's external-OK behavior (it does not embed catalogue
      copies). Hallucinated technique IDs are detected at
      extract_techniques via attack_data.validate_technique_ids; this
      validator is a structural check on the assembled bundle, not a
      content audit on technique pick legitimacy.
    """
    if ref in _STANDARD_TLP_MARKING_IDS:
        return True
    if ref.startswith("attack-pattern--"):
        return True
    return False


def _resolve_one(
    ctx: dict, upstream_index: dict[str, tuple[dict, str]],
) -> tuple[dict, str] | None:
    """Look up a dangling ref in the upstream index. Returns
    (recovered_object, source_label) or None if not found.

    Currently the only upstream source is the normalize step's per-draft
    derivative state. attack-pattern recovery from the catalogue is wired
    via `_attempt_attack_pattern_recovery`.
    """
    ref_id = ctx["ref_id"]
    if ref_id in upstream_index:
        obj, source = upstream_index[ref_id]
        return dict(obj), source

    # ATT&CK pattern recovery — pull the catalogue object and embed.
    if ref_id.startswith("attack-pattern--"):
        attack_obj = _attempt_attack_pattern_recovery(ref_id)
        if attack_obj:
            return attack_obj, "attack_catalogue"

    return None


def _build_upstream_index(state: PipelineState) -> dict[str, tuple[dict, str]]:
    """Build an index of {stix_id: (object_dict, source_label)} from
    upstream pipeline state.

    Sources walked:
    - The bundle's pre-existing objects (so we don't double-recover).
    - normalized_drafts → reconstructed x-procedure stubs would not help,
      so we don't synthesize objects from drafts. Instead we pull from
      `state.stix_bundle.objects` snapshots if present in earlier checkpoints.
    - validated_entities aren't STIX-IDed yet at gate-time, so they don't
      help here. The normalize step's effect on the bundle is what we
      actually want.

    For v1 the practical recovery surface is: an object that was once in
    state.stix_bundle.objects (e.g. the serializer built it but
    accidentally dropped it during a later mutation pass). We walk the
    LangGraph state's prior bundle snapshot if present. Today there isn't
    one — this scaffolding lets us add it without restructuring the API.
    """
    index: dict[str, tuple[dict, str]] = {}

    # Future hook: scan additional state slots here if the pipeline ever
    # accrues a `normalized_objects` snapshot or similar mid-pipeline
    # frozen STIX list. For now the bundle's own objects are the only
    # known source — recovery from the same bundle handles transient
    # ordering glitches and doesn't help with true serializer drops.
    bundle = state.get("stix_bundle") or {}
    for obj in bundle.get("objects", []) or []:
        oid = obj.get("id")
        if oid:
            index[oid] = (obj, "stix_bundle")

    return index


def _attempt_attack_pattern_recovery(ref_id: str) -> dict | None:
    """If `ref_id` matches a STIX UUID for an active ATT&CK technique,
    return the catalogue's attack-pattern object so we can embed it in
    the bundle. Returns None when the catalogue can't be loaded or the
    UUID isn't an active technique.

    The catalogue only resolves attack-pattern UUIDs to technique records
    via `get_technique_record`, which is keyed on T-numbers, not STIX
    UUIDs. So we walk the catalogue once to build a UUID->record map.
    """
    try:
        attack = get_attack_data()
    except Exception as e:
        logger.warning(
            "validate_bundle: attack_data unavailable for recovery: %s", e,
        )
        return None

    # Lazy build a STIX-UUID → record index. The catalogue's all_techniques()
    # already exposes stix_id; convert to a dict here.
    if not hasattr(attack, "_stix_id_to_record"):
        idx: dict[str, dict] = {}
        try:
            for record in attack.all_techniques():
                sid = record.get("stix_id")
                if sid:
                    idx[sid] = record
        except Exception as e:
            logger.warning("validate_bundle: failed to index ATT&CK by stix_id: %s", e)
            return None
        # Cache on the attack data instance for subsequent calls.
        # Using a private attribute name to avoid colliding with future
        # attack_data API additions.
        attack._stix_id_to_record = idx  # type: ignore[attr-defined]

    record = attack._stix_id_to_record.get(ref_id)  # type: ignore[attr-defined]
    if not record:
        return None

    # Build a STIX-shaped attack-pattern dict from the catalogue record.
    # Covers the fields downstream validation actually checks.
    return {
        "type": "attack-pattern",
        "spec_version": "2.1",
        "id": record["stix_id"],
        "name": record.get("name", ""),
        "description": record.get("description", ""),
        "external_references": [
            {"source_name": "mitre-attack", "external_id": record.get("external_id", "")}
        ],
        "x_mitre_platforms": list(record.get("platforms", [])),
        "kill_chain_phases": [
            {"kill_chain_name": "mitre-attack", "phase_name": t}
            for t in record.get("tactics", [])
        ],
    }


# =============================================================================
# Phase 2: auto-fix passes
# =============================================================================

def _auto_fix_passes(
    objects: list[dict], is_sequential: bool,
) -> tuple[list[dict], list[dict]]:
    """Run every auto-fix pass in sequence. Each pass mutates `objects` in
    place and returns its corrections; this function aggregates them.
    """
    corrections: list[dict] = []
    obj_by_id: dict[str, dict] = {o["id"]: o for o in objects if o.get("id")}

    # 1. SRO direction flip.
    corrections.extend(_fix_sro_direction(objects, obj_by_id))

    # 2. SRO dedup (after direction flip so flipped duplicates collapse too).
    objects, dedup_corr = _dedup_sros(objects)
    corrections.extend(dedup_corr)

    # 3. Embedded ref-list dedup.
    corrections.extend(_dedup_embedded_refs(objects))

    # 4. PRECEDES self-loop drop.
    objects, self_loop_corr = _drop_precedes_self_loops(objects)
    corrections.extend(self_loop_corr)

    # 5. PRECEDES drop when the source was non-sequential. The serializer
    # already gates this for cleanly-flowing pipelines (is_sequential=False
    # short-circuits PRECEDES emission), but a backstop here covers cases
    # where state is set late or a future bug reintroduces the SROs.
    if not is_sequential:
        objects, ns_corr = _drop_precedes_when_not_sequential(objects)
        corrections.extend(ns_corr)

    # 6. Embedded x_technique_refs ↔ uses-against-attack-pattern SRO sync.
    obj_by_id = {o["id"]: o for o in objects if o.get("id")}
    corrections.extend(_sync_embedded_technique_refs(objects, obj_by_id))

    # 6b. Backfill x_source_refs from the bundle's source identity when a
    # procedure has none. The drafting node creates drafts with
    # source_refs=[] and the serializer defaults them to the source
    # identity, so this should not fire on a healthy run; it is the backstop
    # that keeps a procedure with no provenance from hard-failing on the
    # x_procedure_missing_required_field check.
    corrections.extend(_backfill_procedure_source_refs(objects))

    # 7. Tactic re-derive from technique catalogue (best-effort).

    # 8. Recompute x_fingerprint on x-procedure objects.
    corrections.extend(_recompute_fingerprints(objects))

    # 9. Coerce invalid org/location roles to "other".
    corrections.extend(_coerce_invalid_roles(objects))

    # 10. Coerce out-of-vocabulary identity sectors to STIX values.
    corrections.extend(_coerce_invalid_sectors(objects))

    return objects, corrections


def _fix_sro_direction(
    objects: list[dict], obj_by_id: dict[str, dict],
) -> list[dict]:
    """Flip SROs whose source/target are inverted relative to the canonical
    direction. Records an auto_fix correction per flip.

    A flip only happens when:
      - relationship_type has a defined direction rule.
      - Both source_ref and target_ref resolve in the bundle.
      - Source is in valid_target AND target is in valid_source. The
        same-orientation case (source in valid_source) gets a pass even
        when the target type isn't valid — that's a different problem
        the dangling-ref or shape checks handle.
    """
    corrections: list[dict] = []

    for obj in objects:
        if obj.get("type") != "relationship":
            continue
        rel_type = obj.get("relationship_type")
        rule = _SRO_DIRECTION_RULES.get(rel_type or "")
        if not rule:
            continue

        src_ref = obj.get("source_ref")
        tgt_ref = obj.get("target_ref")
        # isinstance, not truthiness: an unhashable ref (a dict) blows up on
        # the dict lookup two lines down. Fourth site of the same assumption
        # — refs are strings — all found by fuzzing.
        if not isinstance(src_ref, str) or not isinstance(tgt_ref, str):
            continue
        if not src_ref or not tgt_ref:
            continue

        src_obj = obj_by_id.get(src_ref)
        tgt_obj = obj_by_id.get(tgt_ref)
        if not src_obj or not tgt_obj:
            # Dangling ref; let the recovery / dangling-ref check own it.
            continue

        src_type = src_obj.get("type", "")
        tgt_type = tgt_obj.get("type", "")

        src_valid = src_type in rule["valid_source"]
        if src_valid:
            continue  # already canonical

        # Inverted? source matches valid_target AND target matches valid_source.
        inverted = (src_type in rule["valid_target"]) and (tgt_type in rule["valid_source"])
        if not inverted:
            continue

        before = {"source_ref": src_ref, "target_ref": tgt_ref}
        obj["source_ref"], obj["target_ref"] = tgt_ref, src_ref
        corrections.append({
            "rule": "sro_direction_flipped",
            "severity": "auto_fix",
            "ref_id": obj.get("id"),
            "before": before,
            "after": {"source_ref": tgt_ref, "target_ref": src_ref},
            "message": (
                f"{rel_type} SRO had {src_type}→{tgt_type}; "
                f"flipped to {tgt_type}→{src_type} to match canonical direction."
            ),
        })

    return corrections


def _dedup_sros(
    objects: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Drop SROs with identical (source_ref, target_ref, relationship_type).
    Keep the first occurrence; record one correction per drop.
    """
    corrections: list[dict] = []
    seen: dict[tuple[str, str, str], str] = {}
    kept: list[dict] = []

    for obj in objects:
        if obj.get("type") != "relationship":
            kept.append(obj)
            continue

        key = (
            obj.get("source_ref"),
            obj.get("target_ref"),
            obj.get("relationship_type"),
        )
        # Require all three to be non-empty STRINGS. `all()` alone was not
        # enough: a non-string value (a dict, say) is truthy but unhashable,
        # so `key in seen` raised TypeError and took the whole validator down
        # — found by fuzzing. Anything malformed is kept here
        # and left for the schema and ref-integrity checks to report.
        if not all(isinstance(part, str) and part for part in key):
            kept.append(obj)
            continue

        if key in seen:
            corrections.append({
                "rule": "sro_duplicate_dropped",
                "severity": "auto_fix",
                "ref_id": obj.get("id"),
                "message": (
                    f"Dropped duplicate {key[2]} SRO between {key[0]} and {key[1]} "
                    f"(kept first occurrence {seen[key]})."
                ),
            })
            continue
        seen[key] = obj.get("id", "")
        kept.append(obj)

    return kept, corrections


def _dedup_embedded_refs(objects: list[dict]) -> list[dict]:
    """Dedup duplicates inside embedded ref-list fields. One correction
    per (object, field) where dedup actually changed the list.
    """
    corrections: list[dict] = []

    for obj in objects:
        for field in _LIST_REF_FIELDS:
            value = obj.get(field)
            if not value or not isinstance(value, list):
                continue
            seen: set[str] = set()
            deduped: list[str] = []
            for ref in value:
                if not ref or ref in seen:
                    continue
                seen.add(ref)
                deduped.append(ref)
            if len(deduped) != len(value):
                obj[field] = deduped
                corrections.append({
                    "rule": "embedded_refs_deduplicated",
                    "severity": "auto_fix",
                    "holder_id": obj.get("id"),
                    "ref_field": field,
                    "message": (
                        f"Removed {len(value) - len(deduped)} duplicate ref(s) "
                        f"from {obj.get('id')}.{field}."
                    ),
                })

    return corrections


def _drop_precedes_self_loops(
    objects: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Drop PRECEDES SROs where source==target."""
    corrections: list[dict] = []
    kept: list[dict] = []

    for obj in objects:
        if obj.get("type") == "relationship" and obj.get("relationship_type") == "precedes":
            if obj.get("source_ref") == obj.get("target_ref"):
                corrections.append({
                    "rule": "precedes_self_loop_dropped",
                    "severity": "auto_fix",
                    "ref_id": obj.get("id"),
                    "message": (
                        f"Dropped precedes self-loop on {obj.get('source_ref')}."
                    ),
                })
                continue
        kept.append(obj)

    return kept, corrections


def _drop_precedes_when_not_sequential(
    objects: list[dict],
) -> tuple[list[dict], list[dict]]:
    """When state.is_sequential is False, drop ALL PRECEDES SROs. Catalogs
    of TTPs don't have an inherent order and PRECEDES SROs would manufacture
    one. (Sequencing is no longer embedded on the procedure, so there is
    nothing to clear on the x-procedure objects themselves.)
    """
    corrections: list[dict] = []
    kept: list[dict] = []
    dropped_sros = 0

    for obj in objects:
        if obj.get("type") == "relationship" and obj.get("relationship_type") == "precedes":
            dropped_sros += 1
            continue
        kept.append(obj)

    if dropped_sros:
        corrections.append({
            "rule": "precedes_dropped_non_sequential",
            "severity": "auto_fix",
            "message": (
                f"Source flagged is_sequential=False; "
                f"dropped {dropped_sros} precedes SRO(s)."
            ),
        })

    return kept, corrections


def _sync_embedded_technique_refs(
    objects: list[dict], obj_by_id: dict[str, dict],
) -> list[dict]:
    """For each x-procedure: ensure the embedded x_technique_refs and any
    `uses` SROs from this procedure to attack-pattern targets agree.

    Strategy: union both sets. If an SRO names a technique not in the
    embedded list, append it (auto_fix). If the embedded list names a
    technique with no corresponding SRO, that's left alone — emitting
    new SROs at validate-time would change the bundle's relationship
    count post-gate-2 in a way the analyst didn't review.
    """
    corrections: list[dict] = []

    # Index `uses` SROs by (source_ref, target_ref) where target is an
    # attack-pattern.
    procedure_to_techniques: dict[str, set[str]] = defaultdict(set)
    for obj in objects:
        if obj.get("type") != "relationship":
            continue
        if obj.get("relationship_type") != "uses":
            continue
        # isinstance, not `.get(key, "")`: the default only applies when the
        # key is ABSENT, so a present-but-null ref returns None and
        # .startswith blows up. A relationship with a null or non-string ref
        # crashed the whole validator here — found by fuzzing.
        # It is the last line of defense before a bundle ships, so it has to
        # report malformed input, not die on it. The ref-integrity and schema
        # checks flag such an SRO on their own.
        src = obj.get("source_ref")
        tgt = obj.get("target_ref")
        if not isinstance(src, str) or not isinstance(tgt, str):
            continue
        if not (src.startswith("x-procedure--") and tgt.startswith("attack-pattern--")):
            continue
        procedure_to_techniques[src].add(tgt)

    for obj in objects:
        if obj.get("type") != X_PROCEDURE_TYPE:
            continue
        proc_id = obj.get("id", "")
        embedded = list(obj.get("x_technique_refs", []) or [])
        embedded_set = set(embedded)
        sro_targets = procedure_to_techniques.get(proc_id, set())

        missing_in_embedded = sro_targets - embedded_set
        if missing_in_embedded:
            for tref in sorted(missing_in_embedded):
                embedded.append(tref)
            obj["x_technique_refs"] = embedded
            corrections.append({
                "rule": "embedded_technique_refs_synced",
                "severity": "auto_fix",
                "holder_id": proc_id,
                "ref_field": "x_technique_refs",
                "message": (
                    f"Added {len(missing_in_embedded)} technique ref(s) to "
                    f"{proc_id}.x_technique_refs to match `uses` SROs."
                ),
            })

    return corrections


def _backfill_procedure_source_refs(objects: list[dict]) -> list[dict]:
    """When an x-procedure has no x_source_refs but the bundle has a
    source-author Identity SDO (created_by_ref of the procedure), backfill
    x_source_refs to [created_by_ref]. Records a "repaired" correction so
    the upstream gap is loud — the serializer defaults x_source_refs itself,
    so this auto-fix should not fire in production runs.
    """
    corrections: list[dict] = []

    for obj in objects:
        if obj.get("type") != X_PROCEDURE_TYPE:
            continue
        existing = obj.get("x_source_refs")
        if existing:
            continue
        author_ref = obj.get("created_by_ref")
        if not author_ref:
            continue  # nothing to backfill from; hard-fail check will own it
        obj["x_source_refs"] = [author_ref]
        corrections.append({
            "rule": "x_source_refs_backfilled",
            "severity": "repaired",
            "holder_id": obj.get("id"),
            "ref_field": "x_source_refs",
            "after": [author_ref],
            "message": (
                f"{obj.get('id')} had empty x_source_refs; backfilled with "
                f"created_by_ref={author_ref}. The serializer normally fills "
                f"this; an empty list here means a draft reached serialization "
                f"without one."
            ),
        })

    return corrections


def _extract_attack_external_id(stix_id: str, attack) -> str | None:
    """Helper: STIX UUID → T-number lookup via the attack-pattern bundle's
    UUID index.
    """
    if not hasattr(attack, "_stix_id_to_record"):
        try:
            idx: dict[str, dict] = {}
            for record in attack.all_techniques():
                sid = record.get("stix_id")
                if sid:
                    idx[sid] = record
            attack._stix_id_to_record = idx  # type: ignore[attr-defined]
        except Exception:
            return None
    record = attack._stix_id_to_record.get(stix_id)  # type: ignore[attr-defined]
    if not record:
        return None
    return record.get("external_id")


def _recompute_fingerprints(objects: list[dict]) -> list[dict]:
    """Recompute x_fingerprint on every x-procedure where the fingerprint
    is missing or doesn't match the canonical hash of
    sorted(x_technique_refs) | sorted(x_platforms) | sorted(tactics).

    Tactics are derived from kill_chain_phases entries with
    kill_chain_name='mitre-attack'. The shared helper in
    `app.utils.fingerprint` keeps this formula in lockstep with the
    normalization-time stamp so this corrector stops firing on every
    fresh bundle.
    """
    from app.utils.fingerprint import compute_fingerprint_for_stix_obj

    corrections: list[dict] = []

    for obj in objects:
        if obj.get("type") != X_PROCEDURE_TYPE:
            continue
        expected = compute_fingerprint_for_stix_obj(obj)
        existing = obj.get("x_fingerprint")
        if existing == expected:
            continue
        obj["x_fingerprint"] = expected
        corrections.append({
            "rule": "fingerprint_recomputed",
            "severity": "auto_fix",
            "holder_id": obj.get("id"),
            "ref_field": "x_fingerprint",
            "before": existing,
            "after": expected,
            "message": (
                f"Recomputed x_fingerprint on {obj.get('id')} to match "
                f"current technique/platform/tactic signature."
            ),
        })

    return corrections


# Out-of-vocabulary sector -> nearest STIX `industry-sector-ov` value.
# Two sources of drift:
#   * the extraction prompt listed British spellings and non-STIX values
#     (`defence`, `maritime`, `media`, `pharmaceutical`, `real-estate`)
#   * some real-world sectors have no STIX value at all — one report named
#     "legal & professional services" victims and the closest legitimate
#     value is `commercial`, so that targeting was silently lost.
_SECTOR_ALIASES: dict[str, str] = {
    "defence": "defense",
    "pharmaceutical": "pharmaceuticals",
    "real-estate": "commercial",
    "media": "entertainment",
    "maritime": "transportation",
    "legal": "commercial",
    "legal-services": "commercial",
    "professional-services": "commercial",
    "legal-and-professional-services": "commercial",
}


def _coerce_invalid_sectors(objects: list[dict]) -> list[dict]:
    """Map identity `sectors` values into STIX `industry-sector-ov`.

    `sectors` is declared `items: {type: string}` in identity.json — the
    vocabulary sits unreferenced in `definitions` — so an invalid value
    validates cleanly here and only fails at a downstream consumer that does
    check it, i.e. at the moment of sharing. Coerce where we can, and record
    the original so the source's own phrasing is not lost.
    """
    vocab = set(industry_sector_vocab())
    if not vocab:
        return []

    corrections: list[dict] = []
    for obj in objects:
        if obj.get("type") != "identity":
            continue
        sectors = obj.get("sectors")
        if not isinstance(sectors, list) or not sectors:
            continue

        fixed: list[str] = []
        for raw in sectors:
            value = str(raw).strip().lower()
            if value in vocab:
                fixed.append(value)
                continue
            mapped = _SECTOR_ALIASES.get(value)
            if mapped:
                fixed.append(mapped)
                # Keep the source's wording as a label so the nuance the
                # vocabulary cannot express survives in the bundle.
                labels = obj.setdefault("labels", [])
                original = f"sector:{value}"
                if original not in labels:
                    labels.append(original)
                corrections.append({
                    "rule": "invalid_sector_coerced",
                    "severity": "auto_fix",
                    "holder_id": obj.get("id"),
                    "ref_field": "sectors",
                    "before": value,
                    "after": mapped,
                    "message": (
                        f"Identity {obj.get('id')} had sector {value!r}, which is "
                        f"not in STIX industry-sector-ov; coerced to {mapped!r} "
                        f"and preserved the original as a label."
                    ),
                })
            else:
                corrections.append({
                    "rule": "invalid_sector_dropped",
                    "severity": "warn",
                    "holder_id": obj.get("id"),
                    "ref_field": "sectors",
                    "before": value,
                    "after": None,
                    "message": (
                        f"Identity {obj.get('id')} had sector {value!r}, which is "
                        f"not in STIX industry-sector-ov and has no mapping; "
                        f"dropped rather than shipping a non-standard value."
                    ),
                })

        if fixed != [str(x).strip().lower() for x in sectors]:
            if fixed:
                obj["sectors"] = list(dict.fromkeys(fixed))
            else:
                obj.pop("sectors", None)

    return corrections


def _coerce_invalid_roles(objects: list[dict]) -> list[dict]:
    """Coerce out-of-enum role values on identity / location to 'other'.
    Some upstream paths could emit a free-form role; this is a structural
    safety net rather than something we expect to fire often.
    """
    corrections: list[dict] = []

    for obj in objects:
        otype = obj.get("type", "")
        if otype == "identity":
            role = obj.get("organization_role")
            if role and role not in _VALID_ORG_ROLES:
                obj["organization_role"] = "other"
                corrections.append({
                    "rule": "invalid_role_coerced",
                    "severity": "auto_fix",
                    "holder_id": obj.get("id"),
                    "ref_field": "organization_role",
                    "before": role,
                    "after": "other",
                    "message": (
                        f"Identity {obj.get('id')} had organization_role={role!r} "
                        f"(not in enum); coerced to 'other'."
                    ),
                })
        elif otype == "location":
            role = obj.get("location_role")
            if role and role not in _VALID_LOCATION_ROLES:
                obj["location_role"] = "context"
                corrections.append({
                    "rule": "invalid_role_coerced",
                    "severity": "auto_fix",
                    "holder_id": obj.get("id"),
                    "ref_field": "location_role",
                    "before": role,
                    "after": "context",
                    "message": (
                        f"Location {obj.get('id')} had location_role={role!r} "
                        f"(not in enum); coerced to 'context'."
                    ),
                })

    return corrections


# =============================================================================
# Phase 3: hard-fail checks
# =============================================================================

def _hard_fail_checks(
    objects: list[dict], state: PipelineState,
) -> list[dict]:
    """Run every hard-fail check. Returns one correction per failure;
    severity='hard_fail'. Empty list means the bundle is good to ship.
    """
    corrections: list[dict] = []

    # 0. Full JSON Schema, re-run AFTER the auto-fix passes.
    #    serialize_stix already schema-checked this bundle, but Phase 2 exists
    #    precisely to MUTATE it, and nothing re-checked the result — so any
    #    property an auto-fix added went out unvalidated. That is not
    #    hypothetical: `_rederive_procedure_tactics` wrote an undeclared
    #    `tactics` onto every procedure while the pipeline reported
    #    schema=True, and it shipped that way until a golden-replay test
    #    caught it. An auto-fix emitting an invalid object is a
    #    code bug, so this is a hard fail rather than another auto-fix.
    corrections.extend(_check_post_autofix_schema(objects))

    # 1. Object-level schema (id, type, created/modified for SDOs, etc.).
    corrections.extend(_check_object_schema(objects))

    # 2. Object id uniqueness.
    corrections.extend(_check_object_id_uniqueness(objects))

    # 3. x-procedure required fields.
    corrections.extend(_check_x_procedure_required_fields(objects))

    # 3b. Extension declarations: every custom object names its extension,
    #     and every named extension-definition is in the bundle.
    corrections.extend(_check_extension_declarations(objects))

    # 4. Dangling refs (anything resolver couldn't recover).
    corrections.extend(_check_dangling_refs_unrecoverable(objects))

    # 5. PRECEDES cycle detection (walks attack-operator effect_refs too).
    corrections.extend(_check_precedes_cycle(objects))

    # 5b. attack-operator structural integrity — every operator must
    # have a valid kind, at least one effect_ref, and no dangling refs.
    corrections.extend(_check_attack_operator_integrity(objects))

    # 5c. attack-condition structural integrity — every condition must
    # have a description, at least one ref on either branch, no dangling
    # refs, and (when pattern is set) a valid pattern_type.
    corrections.extend(_check_attack_condition_integrity(objects))

    # 6. Brand-as-malware backstop.
    corrections.extend(_check_brand_as_malware(objects))

    # 7. Multiple attack-flow objects in one bundle (we don't model this yet).
    corrections.extend(_check_multiple_attack_flows(objects))

    return corrections


def _check_post_autofix_schema(objects: list[dict]) -> list[dict]:
    """Validate every object against its real schema after the auto-fixes.

    Complements `_check_object_schema` (hand-rolled required-field checks)
    with the full JSON Schema pass: OASIS STIX 2.1 per type, and
    x_procedure_v3.json for x-procedure. See app.services.stix_schema.

    Degrades to silence when the schema corpus can't load — that is the
    documented behavior of stix_schema.validate_object, and a packaging
    fault must not fail every bundle.
    """
    corrections: list[dict] = []
    for obj in objects:
        for message in stix_schema.validate_object(obj):
            corrections.append(_hard_fail(
                rule="post_autofix_schema_violation",
                message=(
                    f"An auto-fix pass left this object schema-invalid: {message}"
                ),
                holder_id=obj.get("id"),
                holder_type=obj.get("type"),
            ))
    return corrections


def _check_ref_field_types(objects: list[dict]) -> list[dict]:
    """Every reference field must hold a string (or a list of strings).

    This is a precondition for the rest of the validator, not a style rule:
    downstream passes put refs into sets, use them as dict keys and call
    .startswith on them. A dict slipped into a ref field raises
    `TypeError: unhashable type` and aborts validation entirely.
    """
    corrections: list[dict] = []

    for obj in objects:
        holder_id = obj.get("id", "<no-id>")
        holder_type = obj.get("type", "<no-type>")

        for field in _SINGLE_REF_FIELDS:
            value = obj.get(field)
            if value is None or isinstance(value, str):
                continue
            corrections.append(_hard_fail(
                rule="malformed_ref_type",
                message=(
                    f"{holder_id}: {field} must be a STIX identifier string, "
                    f"got {type(value).__name__}."
                ),
                ref_field=field, holder_id=holder_id, holder_type=holder_type,
            ))

        for field in _LIST_REF_FIELDS:
            value = obj.get(field)
            if value is None:
                continue
            if not isinstance(value, (list, tuple)):
                corrections.append(_hard_fail(
                    rule="malformed_ref_type",
                    message=(
                        f"{holder_id}: {field} must be a list of STIX "
                        f"identifiers, got {type(value).__name__}."
                    ),
                    ref_field=field, holder_id=holder_id, holder_type=holder_type,
                ))
                continue
            for entry in value:
                if isinstance(entry, str):
                    continue
                corrections.append(_hard_fail(
                    rule="malformed_ref_type",
                    message=(
                        f"{holder_id}: {field} contains a "
                        f"{type(entry).__name__} where a STIX identifier "
                        f"string was expected."
                    ),
                    ref_field=field, holder_id=holder_id, holder_type=holder_type,
                ))

    return corrections


def _check_object_schema(objects: list[dict]) -> list[dict]:
    """Each object must have type and id. SDOs (non-SCO, non-relationship)
    must also have created and modified. Relationships must have
    source_ref, target_ref, and relationship_type.
    """
    corrections: list[dict] = []
    sco_types = {
        "ipv4-addr", "ipv6-addr", "domain-name", "url", "email-addr",
        "file", "windows-registry-key", "network-traffic", "process",
        "directory", "mutex", "software", "user-account",
    }

    for i, obj in enumerate(objects):
        otype = obj.get("type")
        oid = obj.get("id")

        if not otype:
            corrections.append(_hard_fail(
                rule="object_missing_type",
                ref_id=oid,
                message=f"Object at index {i} has no 'type' field.",
            ))
            continue
        if not oid:
            corrections.append(_hard_fail(
                rule="object_missing_id",
                holder_type=otype,
                message=f"Object of type {otype!r} at index {i} has no 'id' field.",
            ))
            continue

        # SDOs need created + modified.
        if otype not in sco_types and otype != "relationship":
            if not obj.get("created"):
                corrections.append(_hard_fail(
                    rule="sdo_missing_created",
                    ref_id=oid,
                    message=f"{oid}: SDO missing 'created' field.",
                ))
            if not obj.get("modified"):
                corrections.append(_hard_fail(
                    rule="sdo_missing_modified",
                    ref_id=oid,
                    message=f"{oid}: SDO missing 'modified' field.",
                ))

        # Relationships need source/target/type.
        if otype == "relationship":
            for field in ("source_ref", "target_ref", "relationship_type"):
                if not obj.get(field):
                    corrections.append(_hard_fail(
                        rule="relationship_missing_required_field",
                        ref_id=oid,
                        message=f"{oid}: relationship missing {field!r}.",
                    ))

    return corrections


def _check_object_id_uniqueness(objects: list[dict]) -> list[dict]:
    """All ids in the bundle must be unique. Duplicate IDs break ref
    integrity at distribute-time and are a serializer bug.
    """
    corrections: list[dict] = []
    seen: set[str] = set()
    duplicates: set[str] = set()

    for obj in objects:
        oid = obj.get("id")
        if not oid:
            continue
        if oid in seen:
            duplicates.add(oid)
        seen.add(oid)

    for oid in duplicates:
        corrections.append(_hard_fail(
            rule="duplicate_object_id",
            ref_id=oid,
            message=f"Duplicate object id {oid} in bundle.",
        ))

    return corrections


def _check_x_procedure_required_fields(objects: list[dict]) -> list[dict]:
    """Every x-procedure must have name, x_technique_refs (non-empty), and
    x_source_refs (non-empty). The schema itself only requires `name`; the
    two lists are this pipeline's own contract. Missing or empty values
    are hard-fails — a procedure with no technique mapping or no source
    is a pipeline bug, not analyst-correctable from gate_2.
    """
    corrections: list[dict] = []

    for obj in objects:
        if obj.get("type") != X_PROCEDURE_TYPE:
            continue
        oid = obj.get("id", "<no-id>")

        for field in _X_PROCEDURE_REQUIRED:
            value = obj.get(field)
            empty = (
                value is None
                or value == ""
                or (isinstance(value, list) and len(value) == 0)
            )
            if empty:
                corrections.append(_hard_fail(
                    rule="x_procedure_missing_required_field",
                    ref_id=oid,
                    holder_type=X_PROCEDURE_TYPE,
                    ref_field=field,
                    message=(
                        f"{oid}: x-procedure missing required field {field!r}."
                    ),
                ))

    return corrections


def _check_extension_declarations(objects: list[dict]) -> list[dict]:
    """Custom objects must declare their extension, and the bundle must carry
    the definition they declare.

    Three rules, all hard-fails because each is a serializer bug:
      * x_procedure_extension_undeclared — an x-procedure without the
        x-procedure extension-definition id in `extensions`.
      * attack_flow_extension_undeclared — an attack-flow / attack-operator /
        attack-condition without the Attack Flow 2.0.0 id (one definition
        covers every attack-* type).
      * extension_definition_missing — any `extensions` key naming an
        extension-definition the bundle does not contain. Reference integrity
        never looks inside `extensions`, so this is the only check that does.
    """
    corrections: list[dict] = []
    present: set[str] = {o.get("id") for o in objects if o.get("id")}
    required_by_type = {X_PROCEDURE_TYPE: X_PROCEDURE_EXTENSION_ID}
    for t in ATTACK_FLOW_TYPES:
        required_by_type[t] = ATTACK_FLOW_EXTENSION_ID

    for obj in objects:
        obj_type = obj.get("type", "")
        oid = obj.get("id", "<no-id>")
        extensions = obj.get("extensions")
        if not isinstance(extensions, dict):
            extensions = {}

        wanted = required_by_type.get(obj_type)
        if wanted is not None:
            declared = extensions.get(wanted)
            if not isinstance(declared, dict) or declared.get("extension_type") != "new-sdo":
                rule = (
                    "x_procedure_extension_undeclared"
                    if obj_type == X_PROCEDURE_TYPE
                    else "attack_flow_extension_undeclared"
                )
                corrections.append(_hard_fail(
                    rule=rule,
                    ref_id=oid,
                    holder_type=obj_type,
                    ref_field="extensions",
                    message=(
                        f"{oid}: {obj_type} does not declare {wanted} with "
                        f"extension_type 'new-sdo'."
                    ),
                ))

        for key in extensions:
            if not isinstance(key, str) or not key.startswith("extension-definition--"):
                continue
            if key not in present:
                corrections.append(_hard_fail(
                    rule="extension_definition_missing",
                    ref_id=key,
                    holder_type=obj_type,
                    ref_field="extensions",
                    message=(
                        f"{oid}: declares {key} but the bundle carries no such "
                        f"extension-definition object."
                    ),
                ))

    return corrections


def _check_dangling_refs_unrecoverable(objects: list[dict]) -> list[dict]:
    """Any ref the resolver couldn't recover is a hard_fail."""
    corrections: list[dict] = []
    bundle_id_set: set[str] = {o.get("id") for o in objects if o.get("id")}

    dangling = _collect_dangling_refs(objects, bundle_id_set)
    for ctx in dangling:
        corrections.append(_hard_fail(
            rule="dangling_ref_unrecoverable",
            ref_id=ctx["ref_id"],
            ref_field=ctx["ref_field"],
            holder_id=ctx["holder_id"],
            holder_type=ctx["holder_type"],
            message=(
                f"{ctx['holder_type']} {ctx['holder_id']} references "
                f"{ctx['ref_id']} via {ctx['ref_field']}, but the target "
                f"is not in the bundle and could not be recovered from "
                f"upstream state, the ATT&CK catalogue, or the standard "
                f"STIX vocabulary."
            ),
        ))

    return corrections


def _check_precedes_cycle(objects: list[dict]) -> list[dict]:
    """ATT&CK Flow PRECEDES forms a DAG. A cycle is a hard-fail."""
    corrections: list[dict] = []

    # Build adjacency: source_ref -> [target_ref] for every PRECEDES SRO.
    adj: dict[str, list[str]] = defaultdict(list)
    for obj in objects:
        if obj.get("type") != "relationship":
            continue
        if obj.get("relationship_type") != "precedes":
            continue
        src = obj.get("source_ref", "")
        tgt = obj.get("target_ref", "")
        if src and tgt:
            adj[src].append(tgt)

    # Attack Flow v2.0.0 operators carry their own effect_refs (the
    # post-operator targets). A cycle through an operator would otherwise
    # be invisible to this check because op→target isn't a precedes SRO.
    for obj in objects:
        if obj.get("type") != "attack-operator":
            continue
        src = obj.get("id")
        for tgt in obj.get("effect_refs", []) or []:
            if src and tgt:
                adj[src].append(tgt)

    # attack-condition's on_true_refs + on_false_refs play the same
    # role as operator effect_refs — they're flow-graph edges, not
    # precedes SROs. A cycle through a condition (e.g., cond → A → cond)
    # would otherwise slip past this check.
    for obj in objects:
        if obj.get("type") != "attack-condition":
            continue
        src = obj.get("id")
        if not src:
            continue
        for tgt in (obj.get("on_true_refs", []) or []) + (obj.get("on_false_refs", []) or []):
            if tgt:
                adj[src].append(tgt)

    # Standard DFS cycle detection.
    color: dict[str, int] = {}  # 0=unvisited, 1=in_stack, 2=done

    def dfs(node: str) -> str | None:
        color[node] = 1
        for neighbor in adj.get(node, []):
            state = color.get(neighbor, 0)
            if state == 1:
                return f"{node} → {neighbor}"
            if state == 0:
                cycle = dfs(neighbor)
                if cycle:
                    return cycle
        color[node] = 2
        return None

    for node in list(adj.keys()):
        if color.get(node, 0) == 0:
            cycle_edge = dfs(node)
            if cycle_edge:
                corrections.append(_hard_fail(
                    rule="precedes_cycle",
                    message=(
                        f"PRECEDES graph has a cycle (edge {cycle_edge}). "
                        f"Attack-flow sequencing must be a DAG."
                    ),
                ))
                break  # one cycle report is enough

    return corrections


def _check_attack_operator_integrity(objects: list[dict]) -> list[dict]:
    """attack-operator SDO structural checks.

    Validates the SDOs emitted by ``build_attack_operator_sdos``. The
    builder already skips operators that would have <1 resolved
    effect_ref or <2 for branches, so these checks are a safety net for
    operators that arrived through some other path (analyst-edited
    bundle, future LLM-emitted operator, regression in the builder).

    Hard-fails:
      * Operator with invalid ``operator`` kind (must be AND, OR, XOR).
      * Operator with no ``effect_refs``.
      * Operator with a dangling ``effect_ref`` (target not in bundle).
    """
    corrections: list[dict] = []
    allowed_kinds = {"AND", "OR", "XOR"}
    bundle_ids = {o.get("id") for o in objects if o.get("id")}

    for obj in objects:
        if obj.get("type") != "attack-operator":
            continue
        op_id = obj.get("id", "<unknown>")
        kind = obj.get("operator", "")
        effect_refs = obj.get("effect_refs", []) or []

        if kind not in allowed_kinds:
            corrections.append(_hard_fail(
                rule="attack_operator_invalid_kind",
                message=(
                    f"attack-operator {op_id} has invalid `operator` "
                    f"value {kind!r}; must be one of {sorted(allowed_kinds)}."
                ),
                ref_id=op_id,
            ))

        if not effect_refs:
            corrections.append(_hard_fail(
                rule="attack_operator_no_effect_refs",
                message=(
                    f"attack-operator {op_id} has no effect_refs; would "
                    f"leave downstream procedures unreachable through the operator."
                ),
                ref_id=op_id,
            ))

        for ref in effect_refs:
            if ref not in bundle_ids:
                corrections.append(_hard_fail(
                    rule="attack_operator_dangling_effect_ref",
                    message=(
                        f"attack-operator {op_id} references {ref} in "
                        f"effect_refs, but that object is not in the bundle."
                    ),
                    ref_id=op_id,
                ))

    return corrections


def _check_attack_condition_integrity(objects: list[dict]) -> list[dict]:
    """attack-condition SDO structural checks.

    Validates the SDOs emitted by ``build_attack_condition_sdos``. The
    builder already skips conditions with no resolved refs on either
    branch, so these checks are a safety net for conditions that
    arrived through some other path (analyst-edited bundle, future
    LLM-emitted condition outside the chunker, regression in the
    builder).

    Hard-fails:
      * Condition with empty / whitespace-only ``description``.
      * Condition with no refs on either ``on_true_refs`` or
        ``on_false_refs`` (would leave the flow unreachable past it).
      * Condition with a dangling ref on either branch (target not in
        bundle).
      * ``pattern`` set without a valid ``pattern_type``, or with a
        ``pattern_type`` outside {stix, regex, plain}.
    """
    corrections: list[dict] = []
    allowed_pattern_types = {"stix", "regex", "plain"}
    bundle_ids = {o.get("id") for o in objects if o.get("id")}

    for obj in objects:
        if obj.get("type") != "attack-condition":
            continue
        cond_id = obj.get("id", "<unknown>")
        description = (obj.get("description") or "").strip()
        on_true_refs = obj.get("on_true_refs", []) or []
        on_false_refs = obj.get("on_false_refs", []) or []

        if not description:
            corrections.append(_hard_fail(
                rule="attack_condition_no_description",
                message=(
                    f"attack-condition {cond_id} has empty description; "
                    f"analyst review surface needs human-readable prose."
                ),
                ref_id=cond_id,
            ))

        if not on_true_refs and not on_false_refs:
            corrections.append(_hard_fail(
                rule="attack_condition_no_refs",
                message=(
                    f"attack-condition {cond_id} has no refs on either "
                    f"on_true_refs or on_false_refs; would leave the flow "
                    f"unreachable past this point."
                ),
                ref_id=cond_id,
            ))

        for ref in (*on_true_refs, *on_false_refs):
            if ref not in bundle_ids:
                corrections.append(_hard_fail(
                    rule="attack_condition_dangling_ref",
                    message=(
                        f"attack-condition {cond_id} references {ref} "
                        f"in on_true_refs or on_false_refs, but that "
                        f"object is not in the bundle."
                    ),
                    ref_id=cond_id,
                ))

        pattern = obj.get("pattern")
        pattern_type = obj.get("pattern_type")
        # Pair invariant: either both present (and type valid) or both
        # absent. A bare pattern without type, or a type without
        # pattern, is structurally invalid.
        if pattern and pattern_type not in allowed_pattern_types:
            corrections.append(_hard_fail(
                rule="attack_condition_invalid_pattern_type",
                message=(
                    f"attack-condition {cond_id} has pattern set but "
                    f"pattern_type is {pattern_type!r}; must be one of "
                    f"{sorted(allowed_pattern_types)}."
                ),
                ref_id=cond_id,
            ))
        elif pattern_type and not pattern:
            corrections.append(_hard_fail(
                rule="attack_condition_pattern_type_without_pattern",
                message=(
                    f"attack-condition {cond_id} has pattern_type "
                    f"{pattern_type!r} but no pattern text — type "
                    f"requires a paired pattern value."
                ),
                ref_id=cond_id,
            ))

    return corrections


def _check_brand_as_malware(objects: list[dict]) -> list[dict]:
    """Backstop: if a Malware SDO's name matches a known
    technique-pattern brand (ClickFix, EvilProxy, etc.), that's a
    misclassification at entity_extraction. The bundle would carry a
    fictional Malware node; fail loudly so the analyst goes back to gate_0.
    """
    corrections: list[dict] = []

    for obj in objects:
        if obj.get("type") != "malware":
            continue
        name = obj.get("name", "")
        if not name:
            continue
        hits = find_brand_techniques(name)
        if hits:
            brands = ", ".join(sorted(hits.keys()))
            corrections.append(_hard_fail(
                rule="brand_misclassified_as_malware",
                ref_id=obj.get("id"),
                holder_type="malware",
                message=(
                    f"Malware SDO {obj.get('id')} has name {name!r} which matches "
                    f"the technique-pattern brand list ({brands}). "
                    f"Brand names like ClickFix or EvilProxy describe technique "
                    f"patterns, not malware families — re-classify at gate_0."
                ),
            ))

    return corrections


def _check_multiple_attack_flows(objects: list[dict]) -> list[dict]:
    """The pipeline currently emits at most one attack-flow per bundle. A
    second one suggests a serializer bug or a corruption in state."""
    flows = [o for o in objects if o.get("type") == "attack-flow"]
    if len(flows) <= 1:
        return []
    return [_hard_fail(
        rule="multiple_attack_flows",
        message=(
            f"Bundle contains {len(flows)} attack-flow objects; "
            f"v1 supports at most one. Likely a serializer bug."
        ),
    )]


# =============================================================================
# Helpers
# =============================================================================

def _hard_fail(
    rule: str,
    message: str,
    ref_id: str | None = None,
    ref_field: str | None = None,
    holder_id: str | None = None,
    holder_type: str | None = None,
) -> dict:
    """Build a hard_fail correction record. Optional fields are populated
    when the rule has them; otherwise omitted to keep the audit trail
    compact.
    """
    record: dict[str, Any] = {
        "rule": rule,
        "severity": "hard_fail",
        "message": message,
    }
    if ref_id is not None:
        record["ref_id"] = ref_id
    if ref_field is not None:
        record["ref_field"] = ref_field
    if holder_id is not None:
        record["holder_id"] = holder_id
    if holder_type is not None:
        record["holder_type"] = holder_type
    return record


def _summarize_hard_fails(corrections: list[dict]) -> str:
    """One-line summary string suitable for state['error']."""
    fails = [c for c in corrections if c.get("severity") == "hard_fail"]
    by_rule: dict[str, int] = defaultdict(int)
    for c in fails:
        by_rule[c.get("rule", "unknown")] += 1
    parts = [f"{rule}={count}" for rule, count in sorted(by_rule.items())]
    return f"validate_bundle failed: {len(fails)} issue(s) — {', '.join(parts)}"


def _finalize_failed(corrections: list[dict], summary_tail: str) -> dict:
    """Shortcut for envelope-level hard fails where there's no usable
    bundle to return."""
    return {
        "bundle_corrections": corrections,
        "bundle_validation_failed": True,
        "status": PipelineStatus.FAILED.value,
        "error": f"validate_bundle failed: {summary_tail}",
        "current_node": "validate_bundle",
    }
