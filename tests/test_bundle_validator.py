"""Tests for the validate_bundle node.

Coverage targets one test per rule (auto_fix and hard_fail), plus the
recovery resolver's fixpoint + transitive expansion + legitimate-external
classification, plus end-to-end orchestration (the node returns the right
shape on success vs hard-fail).
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest

from app.nodes.deterministic.extension_definitions import (
    ATTACK_FLOW_EXTENSION_ID,
    X_PROCEDURE_EXTENSION_ID,
    attack_flow_author_identity,
    attack_flow_extension_definition,
    extension_declaration,
    x_procedure_author_identity,
    x_procedure_extension_definition,
)
from app.nodes.deterministic.bundle_validator import (
    _attempt_attack_pattern_recovery,
    _backfill_procedure_source_refs,
    _check_attack_condition_integrity,
    _check_attack_operator_integrity,
    _check_brand_as_malware,
    _check_dangling_refs_unrecoverable,
    _check_multiple_attack_flows,
    _check_object_id_uniqueness,
    _check_object_schema,
    _check_precedes_cycle,
    _check_x_procedure_required_fields,
    _coerce_invalid_roles,
    _collect_dangling_refs,
    _dedup_embedded_refs,
    _dedup_sros,
    _drop_precedes_self_loops,
    _drop_precedes_when_not_sequential,
    _fix_sro_direction,
    _is_legitimately_external,
    _recompute_fingerprints,
    _recovery_resolve,
    _sync_embedded_technique_refs,
    validate_bundle,
)


# =============================================================================
# Fixture helpers
# =============================================================================

def _proc(
    name: str = "Test Procedure",
    technique_refs: list[str] | None = None,
    source_refs: list[str] | None = None,
    extra: dict | None = None,
    declared: bool = True,
) -> dict:
    """Build a minimally-valid x-procedure SDO.

    Defaults: x_technique_refs points to an external attack-pattern UUID
    (legit per the serializer's external-OK rule) and x_source_refs is
    empty — most validator tests don't care about source provenance and
    leaving it empty avoids confusion with the dangling-ref check.
    Tests that exercise x_source_refs should pass source_refs explicitly.
    `declared` adds the x-procedure extension declaration the serializer
    always writes; `_make_bundle` embeds the matching definition.
    """
    pid = f"x-procedure--{uuid.uuid4()}"
    obj: dict[str, Any] = {
        "type": "x-procedure",
        "spec_version": "2.1",
        "id": pid,
        "created": "2026-01-01T00:00:00.000Z",
        "modified": "2026-01-01T00:00:00.000Z",
        "name": name,
        "x_technique_refs": technique_refs if technique_refs is not None
            else ["attack-pattern--12345678-1234-4234-8234-123456789012"],
        "x_source_refs": source_refs if source_refs is not None else [],
    }
    if declared:
        obj["extensions"] = extension_declaration(X_PROCEDURE_EXTENSION_ID)
    if extra:
        obj.update(extra)
    return obj


def _identity(name: str = "Test Author") -> dict:
    return {
        "type": "identity",
        "spec_version": "2.1",
        "id": f"identity--{uuid.uuid4()}",
        "created": "2026-01-01T00:00:00.000Z",
        "modified": "2026-01-01T00:00:00.000Z",
        "name": name,
        "identity_class": "organization",
    }


def _sro(rel_type: str, source_ref: str, target_ref: str) -> dict:
    return {
        "type": "relationship",
        "spec_version": "2.1",
        "id": f"relationship--{uuid.uuid4()}",
        "created": "2026-01-01T00:00:00.000Z",
        "modified": "2026-01-01T00:00:00.000Z",
        "relationship_type": rel_type,
        "source_ref": source_ref,
        "target_ref": target_ref,
    }


def _make_bundle(objects: list[dict], with_definitions: bool = True) -> dict:
    """Wrap objects in a bundle. Like the serializer, embed the extension
    definition (plus author identity) for every extension the objects
    declare, so the declaration check passes unless a test opts out."""
    objects = list(objects)
    if with_definitions:
        declared = {
            key
            for o in objects
            for key in (o.get("extensions") or {})
            if isinstance(key, str) and key.startswith("extension-definition--")
        }
        present = {o.get("id") for o in objects}
        if X_PROCEDURE_EXTENSION_ID in declared and X_PROCEDURE_EXTENSION_ID not in present:
            objects += [x_procedure_author_identity(), x_procedure_extension_definition()]
        if ATTACK_FLOW_EXTENSION_ID in declared and ATTACK_FLOW_EXTENSION_ID not in present:
            objects += [attack_flow_author_identity(), attack_flow_extension_definition()]
    return {
        "type": "bundle",
        "id": f"bundle--{uuid.uuid4()}",
        "objects": objects,
    }


def _state_with_bundle(bundle: dict, **overrides) -> dict:
    state = {"stix_bundle": bundle, "is_sequential": True}
    state.update(overrides)
    return state


# =============================================================================
# Phase 1: recovery resolver
# =============================================================================

class TestRecoveryResolver:
    def test_legitimately_external_tlp_marking(self):
        """TLP marking-definition IDs are allowed to be unembedded."""
        assert _is_legitimately_external(
            "marking-definition--613f2e26-407d-48c7-9eca-b8e91df99dc9",
            "object_marking_refs",
        ) is True

    def test_legitimately_external_attack_pattern_in_source_ref(self):
        """attack-pattern UUIDs in SRO source/target refs are external-OK."""
        assert _is_legitimately_external(
            "attack-pattern--12345678-1234-4234-8234-123456789012",
            "source_ref",
        ) is True

    def test_collect_dangling_refs_finds_missing(self):
        """Dangling refs are surfaced with full context."""
        proc = _proc()
        proc["x_components_refs"] = ["process--deadbeef-0000-4000-8000-000000000000"]
        bundle_id_set = {proc["id"]}
        contexts = _collect_dangling_refs([proc], bundle_id_set)
        assert len(contexts) == 1
        assert contexts[0]["ref_id"].startswith("process--")
        assert contexts[0]["ref_field"] == "x_components_refs"
        assert contexts[0]["holder_id"] == proc["id"]
        assert contexts[0]["holder_type"] == "x-procedure"

    def test_collect_dangling_skips_legitimate_external(self):
        """attack-pattern refs in any field are external-OK (matches the
        serializer's behavior — the catalogue is not embedded in v1)."""
        proc = _proc()
        proc["x_technique_refs"] = ["attack-pattern--12345678-1234-4234-8234-123456789012"]
        contexts = _collect_dangling_refs([proc], {proc["id"]})
        assert contexts == []

    def test_recovery_attack_pattern_from_catalogue(self):
        """If a dangling attack-pattern UUID matches a real ATT&CK record,
        the resolver embeds the catalogue object."""
        # T1059 ("Command and Scripting Interpreter") is a stable, active ID.
        from app.services.attack_data import get_attack_data
        attack = get_attack_data()
        record = attack.get_technique_record("T1059")
        if record is None:
            pytest.skip("ATT&CK catalogue unavailable in test env")

        stix_id = record["stix_id"]
        recovered = _attempt_attack_pattern_recovery(stix_id)
        assert recovered is not None
        assert recovered["id"] == stix_id
        assert recovered["type"] == "attack-pattern"
        assert recovered["name"]  # has a name

    def test_recovery_attack_pattern_unknown_returns_none(self):
        """A fabricated attack-pattern UUID can't be recovered."""
        recovered = _attempt_attack_pattern_recovery(
            "attack-pattern--ffffffff-0000-4000-8000-000000000000"
        )
        assert recovered is None

    def test_resolver_fixpoint_recovers_then_quits(self):
        """Resolver runs at most one pass when nothing's dangling."""
        proc = _proc()
        identity = _identity()
        proc["created_by_ref"] = identity["id"]
        proc["x_source_refs"] = [identity["id"]]
        proc["x_technique_refs"] = ["attack-pattern--12345678-1234-4234-8234-123456789012"]
        # No dangling refs → no recovery, no corrections.
        objects, corrections = _recovery_resolve([proc, identity], _state_with_bundle(_make_bundle([proc, identity])))
        assert corrections == []
        assert {o["id"] for o in objects} == {proc["id"], identity["id"]}


# =============================================================================
# Phase 2: auto-fix passes
# =============================================================================

class TestAutoFixDirection:
    def test_uses_direction_flipped_when_inverted(self):
        """A `uses` SRO from tool→procedure is flipped to procedure→tool."""
        proc = _proc()
        tool = {
            "type": "tool",
            "spec_version": "2.1",
            "id": f"tool--{uuid.uuid4()}",
            "created": "2026-01-01T00:00:00.000Z",
            "modified": "2026-01-01T00:00:00.000Z",
            "name": "Mimikatz",
        }
        # Inverted: tool → procedure.
        sro = _sro("uses", tool["id"], proc["id"])
        objects = [proc, tool, sro]
        obj_by_id = {o["id"]: o for o in objects}
        corrections = _fix_sro_direction(objects, obj_by_id)
        assert len(corrections) == 1
        assert corrections[0]["rule"] == "sro_direction_flipped"
        assert corrections[0]["severity"] == "auto_fix"
        assert sro["source_ref"] == proc["id"]
        assert sro["target_ref"] == tool["id"]

    def test_uses_direction_kept_when_canonical(self):
        """A `uses` SRO from intrusion-set→procedure is left alone."""
        proc = _proc()
        iset = {
            "type": "intrusion-set",
            "spec_version": "2.1",
            "id": f"intrusion-set--{uuid.uuid4()}",
            "created": "2026-01-01T00:00:00.000Z",
            "modified": "2026-01-01T00:00:00.000Z",
            "name": "Test Group",
        }
        sro = _sro("uses", iset["id"], proc["id"])
        original_source = sro["source_ref"]
        obj_by_id = {o["id"]: o for o in [proc, iset, sro]}
        corrections = _fix_sro_direction([proc, iset, sro], obj_by_id)
        assert corrections == []
        assert sro["source_ref"] == original_source


class TestAutoFixDedup:
    def test_dedup_sros_drops_exact_duplicate(self):
        proc = _proc()
        ap_id = "attack-pattern--12345678-1234-4234-8234-123456789012"
        sro1 = _sro("uses", proc["id"], ap_id)
        sro2 = _sro("uses", proc["id"], ap_id)  # duplicate
        kept, corrections = _dedup_sros([proc, sro1, sro2])
        assert len(kept) == 2  # proc + first sro
        assert len(corrections) == 1
        assert corrections[0]["rule"] == "sro_duplicate_dropped"

    def test_dedup_sros_keeps_distinct(self):
        proc = _proc()
        ap_id = "attack-pattern--12345678-1234-4234-8234-123456789012"
        sro1 = _sro("uses", proc["id"], ap_id)
        sro2 = _sro("attributed-to", proc["id"], ap_id)
        kept, corrections = _dedup_sros([proc, sro1, sro2])
        assert len(kept) == 3
        assert corrections == []

    def test_dedup_embedded_refs_in_x_components_refs(self):
        proc = _proc()
        proc["x_components_refs"] = ["a", "b", "a", "c", "b"]
        corrections = _dedup_embedded_refs([proc])
        assert proc["x_components_refs"] == ["a", "b", "c"]
        assert len(corrections) == 1
        assert corrections[0]["rule"] == "embedded_refs_deduplicated"


class TestAutoFixPrecedes:
    def test_precedes_self_loop_dropped(self):
        proc = _proc()
        sro = _sro("precedes", proc["id"], proc["id"])
        kept, corrections = _drop_precedes_self_loops([proc, sro])
        assert len(kept) == 1
        assert kept[0] == proc
        assert any(c["rule"] == "precedes_self_loop_dropped" for c in corrections)

    def test_precedes_dropped_when_not_sequential(self):
        proc1 = _proc(name="P1")
        proc2 = _proc(name="P2")
        sro = _sro("precedes", proc1["id"], proc2["id"])
        kept, corrections = _drop_precedes_when_not_sequential([proc1, proc2, sro])
        assert all(o.get("type") != "relationship" for o in kept)
        # Procedures are kept untouched (sequencing lives only in the SROs).
        assert proc1 in kept and proc2 in kept
        assert any(c["rule"] == "precedes_dropped_non_sequential" for c in corrections)


class TestAutoFixSync:
    def test_sync_embedded_technique_refs_appends_missing(self):
        ap_id = "attack-pattern--aaaaaaaa-1111-4111-8111-111111111111"
        proc = _proc(technique_refs=[])
        proc["x_technique_refs"] = []  # empty
        sro = _sro("uses", proc["id"], ap_id)
        objects = [proc, sro]
        obj_by_id = {o["id"]: o for o in objects}
        corrections = _sync_embedded_technique_refs(objects, obj_by_id)
        assert ap_id in proc["x_technique_refs"]
        assert any(c["rule"] == "embedded_technique_refs_synced" for c in corrections)

    def test_backfill_x_source_refs_from_created_by(self):
        identity = _identity()
        proc = _proc(source_refs=[])
        proc["x_source_refs"] = []
        proc["created_by_ref"] = identity["id"]
        corrections = _backfill_procedure_source_refs([proc, identity])
        assert proc["x_source_refs"] == [identity["id"]]
        assert any(c["rule"] == "x_source_refs_backfilled" for c in corrections)
        # severity is "repaired" since this means the upstream serializer dropped
        # the field — louder than auto_fix.
        assert all(c["severity"] == "repaired" for c in corrections if c["rule"] == "x_source_refs_backfilled")


class TestAutoFixFingerprint:
    def test_fingerprint_recomputed_when_missing(self):
        proc = _proc()
        proc["platforms"] = ["windows"]
        proc["tactics"] = ["execution"]
        # No fingerprint set
        corrections = _recompute_fingerprints([proc])
        assert "x_fingerprint" in proc
        assert len(proc["x_fingerprint"]) == 32  # SHA-256 truncated
        assert any(c["rule"] == "fingerprint_recomputed" for c in corrections)

    def test_fingerprint_recomputed_when_stale(self):
        proc = _proc()
        proc["platforms"] = ["windows"]
        proc["tactics"] = ["execution"]
        proc["x_fingerprint"] = "deadbeef" * 4  # wrong
        corrections = _recompute_fingerprints([proc])
        assert proc["x_fingerprint"] != "deadbeef" * 4
        assert len(corrections) == 1


class TestAutoFixRoles:
    def test_invalid_org_role_coerced_to_other(self):
        ident = _identity()
        ident["organization_role"] = "bogus_role"
        corrections = _coerce_invalid_roles([ident])
        assert ident["organization_role"] == "other"
        assert len(corrections) == 1
        assert corrections[0]["before"] == "bogus_role"

    def test_valid_org_role_left_alone(self):
        ident = _identity()
        ident["organization_role"] = "victim"
        corrections = _coerce_invalid_roles([ident])
        assert corrections == []
        assert ident["organization_role"] == "victim"


# =============================================================================
# Phase 3: hard-fail checks
# =============================================================================

class TestHardFailSchema:
    def test_object_missing_id_hard_fails(self):
        bad = {"type": "identity", "name": "no id"}
        corrections = _check_object_schema([bad])
        assert any(c["rule"] == "object_missing_id" for c in corrections)

    def test_relationship_missing_target_ref(self):
        sro = {
            "type": "relationship",
            "id": f"relationship--{uuid.uuid4()}",
            "created": "2026-01-01T00:00:00.000Z",
            "modified": "2026-01-01T00:00:00.000Z",
            "relationship_type": "uses",
            "source_ref": "x-procedure--abc",
            # missing target_ref
        }
        corrections = _check_object_schema([sro])
        assert any(
            c["rule"] == "relationship_missing_required_field"
            for c in corrections
        )

    def test_duplicate_object_id(self):
        proc1 = _proc()
        proc2 = _proc()
        proc2["id"] = proc1["id"]  # collision
        corrections = _check_object_id_uniqueness([proc1, proc2])
        assert len(corrections) == 1
        assert corrections[0]["rule"] == "duplicate_object_id"


class TestHardFailProcedure:
    def test_x_procedure_missing_x_technique_refs(self):
        proc = _proc()
        proc["x_technique_refs"] = []
        corrections = _check_x_procedure_required_fields([proc])
        assert any(
            c["rule"] == "x_procedure_missing_required_field"
            and c["ref_field"] == "x_technique_refs"
            for c in corrections
        )

    def test_x_procedure_missing_name(self):
        proc = _proc()
        proc["name"] = ""
        corrections = _check_x_procedure_required_fields([proc])
        assert any(
            c["rule"] == "x_procedure_missing_required_field"
            and c["ref_field"] == "name"
            for c in corrections
        )


class TestHardFailDangling:
    def test_dangling_unrecoverable_ref_fails(self):
        proc = _proc()
        proc["x_components_refs"] = ["process--deadbeef-0000-4000-8000-000000000000"]
        # No recovery source has this ID; the resolver couldn't fix it.
        corrections = _check_dangling_refs_unrecoverable([proc])
        assert any(c["rule"] == "dangling_ref_unrecoverable" for c in corrections)


class TestHardFailCycle:
    def test_precedes_cycle_detected(self):
        proc1 = _proc()
        proc2 = _proc()
        sro1 = _sro("precedes", proc1["id"], proc2["id"])
        sro2 = _sro("precedes", proc2["id"], proc1["id"])  # cycle
        corrections = _check_precedes_cycle([proc1, proc2, sro1, sro2])
        assert any(c["rule"] == "precedes_cycle" for c in corrections)

    def test_no_cycle_when_dag(self):
        proc1 = _proc()
        proc2 = _proc()
        proc3 = _proc()
        sros = [
            _sro("precedes", proc1["id"], proc2["id"]),
            _sro("precedes", proc2["id"], proc3["id"]),
        ]
        corrections = _check_precedes_cycle([proc1, proc2, proc3] + sros)
        assert corrections == []


class TestHardFailBrand:
    def test_brand_misclassified_as_malware(self):
        """A Malware SDO named ClickFix is a technique-pattern-as-malware violation."""
        malware = {
            "type": "malware",
            "spec_version": "2.1",
            "id": f"malware--{uuid.uuid4()}",
            "created": "2026-01-01T00:00:00.000Z",
            "modified": "2026-01-01T00:00:00.000Z",
            "name": "ClickFix",
        }
        corrections = _check_brand_as_malware([malware])
        assert len(corrections) == 1
        assert corrections[0]["rule"] == "brand_misclassified_as_malware"

    def test_legitimate_malware_passes(self):
        malware = {
            "type": "malware",
            "spec_version": "2.1",
            "id": f"malware--{uuid.uuid4()}",
            "created": "2026-01-01T00:00:00.000Z",
            "modified": "2026-01-01T00:00:00.000Z",
            "name": "Mimikatz",
        }
        corrections = _check_brand_as_malware([malware])
        assert corrections == []


class TestHardFailFlows:
    def test_multiple_attack_flows_fails(self):
        flow1 = {"type": "attack-flow", "id": f"attack-flow--{uuid.uuid4()}",
                 "created": "2026-01-01T00:00:00.000Z",
                 "modified": "2026-01-01T00:00:00.000Z", "name": "F1"}
        flow2 = {"type": "attack-flow", "id": f"attack-flow--{uuid.uuid4()}",
                 "created": "2026-01-01T00:00:00.000Z",
                 "modified": "2026-01-01T00:00:00.000Z", "name": "F2"}
        corrections = _check_multiple_attack_flows([flow1, flow2])
        assert len(corrections) == 1
        assert corrections[0]["rule"] == "multiple_attack_flows"

    def test_single_attack_flow_ok(self):
        flow = {"type": "attack-flow", "id": f"attack-flow--{uuid.uuid4()}",
                "created": "2026-01-01T00:00:00.000Z",
                "modified": "2026-01-01T00:00:00.000Z", "name": "F1"}
        assert _check_multiple_attack_flows([flow]) == []


class TestAttackOperatorIntegrity:
    """Phase 2 — structural checks on attack-operator SDOs."""

    def _op(
        self,
        kind: str = "AND",
        effect_refs: list[str] | None = None,
        op_id: str | None = None,
    ) -> dict:
        return {
            "type": "attack-operator",
            "id": op_id or f"attack-operator--{uuid.uuid4()}",
            "operator": kind,
            "effect_refs": list(effect_refs) if effect_refs is not None else [],
        }

    def test_valid_and_operator_passes(self):
        proc1 = _proc()
        proc2 = _proc()
        op = self._op(kind="AND", effect_refs=[proc1["id"], proc2["id"]])
        assert _check_attack_operator_integrity([op, proc1, proc2]) == []

    def test_invalid_kind_hard_fails(self):
        proc = _proc()
        op = self._op(kind="MAYBE", effect_refs=[proc["id"]])
        corrections = _check_attack_operator_integrity([op, proc])
        assert any(c["rule"] == "attack_operator_invalid_kind" for c in corrections)

    def test_empty_effect_refs_hard_fails(self):
        op = self._op(kind="AND", effect_refs=[])
        corrections = _check_attack_operator_integrity([op])
        assert any(c["rule"] == "attack_operator_no_effect_refs" for c in corrections)

    def test_dangling_effect_ref_hard_fails(self):
        proc = _proc()
        op = self._op(kind="AND", effect_refs=[proc["id"], "x-procedure--missing"])
        corrections = _check_attack_operator_integrity([op, proc])
        dangling = [c for c in corrections if c["rule"] == "attack_operator_dangling_effect_ref"]
        assert len(dangling) == 1
        assert "x-procedure--missing" in dangling[0]["message"]

    def test_non_operators_ignored(self):
        proc = _proc()
        assert _check_attack_operator_integrity([proc]) == []

    def test_xor_and_or_kinds_accepted(self):
        proc1 = _proc()
        proc2 = _proc()
        for kind in ("OR", "XOR"):
            op = self._op(kind=kind, effect_refs=[proc1["id"], proc2["id"]])
            corrections = _check_attack_operator_integrity([op, proc1, proc2])
            assert all(c["rule"] != "attack_operator_invalid_kind" for c in corrections), kind


class TestPrecedesCycleThroughOperators:
    """Cycle detection must walk attack-operator effect_refs, not just
    procedure→procedure precedes SROs."""

    def test_cycle_through_operator_caught(self):
        # proc_a -> op_and -> proc_b -> precedes -> proc_a
        proc_a = _proc()
        proc_b = _proc()
        op = {
            "type": "attack-operator",
            "id": f"attack-operator--{uuid.uuid4()}",
            "operator": "AND",
            "effect_refs": [proc_b["id"]],
        }
        sro_a_to_op = _sro("precedes", proc_a["id"], op["id"])
        sro_b_to_a = _sro("precedes", proc_b["id"], proc_a["id"])
        corrections = _check_precedes_cycle(
            [proc_a, proc_b, op, sro_a_to_op, sro_b_to_a]
        )
        assert any(c["rule"] == "precedes_cycle" for c in corrections)

    def test_no_cycle_through_operator(self):
        proc_a = _proc()
        proc_b = _proc()
        op = {
            "type": "attack-operator",
            "id": f"attack-operator--{uuid.uuid4()}",
            "operator": "AND",
            "effect_refs": [proc_b["id"]],
        }
        sro_a_to_op = _sro("precedes", proc_a["id"], op["id"])
        # No edge back to proc_a — flow is acyclic.
        corrections = _check_precedes_cycle([proc_a, proc_b, op, sro_a_to_op])
        assert all(c["rule"] != "precedes_cycle" for c in corrections)


class TestAttackConditionIntegrity:
    """Phase 3 — structural checks on attack-condition SDOs."""

    def _cond(
        self,
        description: str = "checks something",
        on_true_refs: list[str] | None = None,
        on_false_refs: list[str] | None = None,
        pattern: str | None = None,
        pattern_type: str | None = None,
        cond_id: str | None = None,
    ) -> dict:
        obj = {
            "type": "attack-condition",
            "id": cond_id or f"attack-condition--{uuid.uuid4()}",
            "description": description,
            "on_true_refs": list(on_true_refs) if on_true_refs is not None else [],
            "on_false_refs": list(on_false_refs) if on_false_refs is not None else [],
        }
        if pattern is not None:
            obj["pattern"] = pattern
        if pattern_type is not None:
            obj["pattern_type"] = pattern_type
        return obj

    def test_valid_condition_passes(self):
        proc_b = _proc()
        proc_c = _proc()
        cond = self._cond(on_true_refs=[proc_b["id"]], on_false_refs=[proc_c["id"]])
        assert _check_attack_condition_integrity([cond, proc_b, proc_c]) == []

    def test_empty_description_hard_fails(self):
        proc = _proc()
        cond = self._cond(description="   ", on_true_refs=[proc["id"]])
        corrections = _check_attack_condition_integrity([cond, proc])
        assert any(c["rule"] == "attack_condition_no_description" for c in corrections)

    def test_no_refs_on_either_branch_hard_fails(self):
        cond = self._cond(on_true_refs=[], on_false_refs=[])
        corrections = _check_attack_condition_integrity([cond])
        assert any(c["rule"] == "attack_condition_no_refs" for c in corrections)

    def test_one_sided_partition_passes(self):
        proc = _proc()
        cond = self._cond(on_true_refs=[proc["id"]], on_false_refs=[])
        corrections = _check_attack_condition_integrity([cond, proc])
        assert all(c["rule"] != "attack_condition_no_refs" for c in corrections)

    def test_dangling_ref_hard_fails(self):
        proc = _proc()
        cond = self._cond(
            on_true_refs=[proc["id"], "x-procedure--missing"],
            on_false_refs=[],
        )
        corrections = _check_attack_condition_integrity([cond, proc])
        dangling = [c for c in corrections if c["rule"] == "attack_condition_dangling_ref"]
        assert len(dangling) == 1
        assert "x-procedure--missing" in dangling[0]["message"]

    def test_invalid_pattern_type_hard_fails(self):
        proc = _proc()
        cond = self._cond(
            on_true_refs=[proc["id"]],
            pattern="foo",
            pattern_type="yaml",
        )
        corrections = _check_attack_condition_integrity([cond, proc])
        assert any(c["rule"] == "attack_condition_invalid_pattern_type" for c in corrections)

    def test_pattern_type_without_pattern_hard_fails(self):
        proc = _proc()
        cond = self._cond(
            on_true_refs=[proc["id"]],
            pattern_type="regex",
            # pattern omitted
        )
        corrections = _check_attack_condition_integrity([cond, proc])
        assert any(c["rule"] == "attack_condition_pattern_type_without_pattern" for c in corrections)

    def test_valid_pattern_pair_passes(self):
        proc = _proc()
        for pt in ("stix", "regex", "plain"):
            cond = self._cond(
                on_true_refs=[proc["id"]],
                pattern="something",
                pattern_type=pt,
            )
            corrections = _check_attack_condition_integrity([cond, proc])
            assert all(c["rule"] != "attack_condition_invalid_pattern_type" for c in corrections), pt

    def test_non_conditions_ignored(self):
        proc = _proc()
        assert _check_attack_condition_integrity([proc]) == []


class TestPrecedesCycleThroughConditions:
    """Cycle detection must walk attack-condition on_true_refs +
    on_false_refs (not just precedes SROs)."""

    def test_cycle_through_condition_caught(self):
        # proc_a → cond → proc_b → precedes → proc_a
        proc_a = _proc()
        proc_b = _proc()
        cond = {
            "type": "attack-condition",
            "id": f"attack-condition--{uuid.uuid4()}",
            "description": "test",
            "on_true_refs": [proc_b["id"]],
            "on_false_refs": [],
        }
        sro_a_to_cond = _sro("precedes", proc_a["id"], cond["id"])
        sro_b_to_a = _sro("precedes", proc_b["id"], proc_a["id"])
        corrections = _check_precedes_cycle(
            [proc_a, proc_b, cond, sro_a_to_cond, sro_b_to_a]
        )
        assert any(c["rule"] == "precedes_cycle" for c in corrections)

    def test_no_cycle_through_condition(self):
        proc_a = _proc()
        proc_b = _proc()
        cond = {
            "type": "attack-condition",
            "id": f"attack-condition--{uuid.uuid4()}",
            "description": "test",
            "on_true_refs": [proc_b["id"]],
            "on_false_refs": [],
        }
        sro_a_to_cond = _sro("precedes", proc_a["id"], cond["id"])
        # No edge back to proc_a — flow is acyclic.
        corrections = _check_precedes_cycle([proc_a, proc_b, cond, sro_a_to_cond])
        assert all(c["rule"] != "precedes_cycle" for c in corrections)


# =============================================================================
# Node entry-point orchestration
# =============================================================================

class TestValidateBundleNode:
    def test_clean_bundle_passes(self):
        """Well-formed bundle: status stays SERIALIZING, no hard-fails."""
        identity = _identity()
        ap_id = "attack-pattern--12345678-1234-4234-8234-123456789012"
        proc = _proc()
        proc["created_by_ref"] = identity["id"]
        proc["x_source_refs"] = [identity["id"]]
        proc["x_technique_refs"] = [ap_id]
        # `uses` SRO from procedure to attack-pattern (canonical direction).
        # Target ref to attack-pattern is legitimately external.
        sro = _sro("uses", proc["id"], ap_id)

        bundle = _make_bundle([identity, proc, sro])
        state = _state_with_bundle(bundle)

        result = validate_bundle(state)
        assert result["bundle_validation_failed"] is False
        assert result["status"] == "serializing"
        # No hard_fail entries
        hard_fails = [
            c for c in result["bundle_corrections"]
            if c["severity"] == "hard_fail"
        ]
        assert hard_fails == []

    def test_hard_fail_short_circuits(self):
        """Hard-fail bundle: status flips to FAILED, validation_failed=True,
        error string populated."""
        # Empty bundle → bundle_empty hard-fail.
        bundle = _make_bundle([])
        state = _state_with_bundle(bundle)

        result = validate_bundle(state)
        assert result["bundle_validation_failed"] is True
        assert result["status"] == "failed"
        assert "validate_bundle failed" in result["error"]

    def test_brand_as_malware_routes_to_failed(self):
        """End-to-end: a ClickFix Malware SDO causes hard-fail with detail."""
        identity = _identity()
        bad = {
            "type": "malware",
            "spec_version": "2.1",
            "id": f"malware--{uuid.uuid4()}",
            "created": "2026-01-01T00:00:00.000Z",
            "modified": "2026-01-01T00:00:00.000Z",
            "name": "ClickFix",
        }
        bundle = _make_bundle([identity, bad])
        state = _state_with_bundle(bundle)

        result = validate_bundle(state)
        assert result["bundle_validation_failed"] is True
        assert any(
            c["rule"] == "brand_misclassified_as_malware"
            for c in result["bundle_corrections"]
        )

    def test_auto_fix_does_not_fail(self):
        """Inverted USES SRO: validator flips it but the bundle ships."""
        identity = _identity()
        proc = _proc()
        proc["created_by_ref"] = identity["id"]
        proc["x_source_refs"] = [identity["id"]]
        ap_id = "attack-pattern--12345678-1234-4234-8234-123456789012"
        proc["x_technique_refs"] = [ap_id]
        tool = {
            "type": "tool",
            "spec_version": "2.1",
            "id": f"tool--{uuid.uuid4()}",
            "created": "2026-01-01T00:00:00.000Z",
            "modified": "2026-01-01T00:00:00.000Z",
            "name": "Mimikatz",
        }
        # Inverted USES: tool → procedure
        sro = _sro("uses", tool["id"], proc["id"])
        bundle = _make_bundle([identity, proc, tool, sro])
        state = _state_with_bundle(bundle)

        result = validate_bundle(state)
        assert result["bundle_validation_failed"] is False
        # Auto-fix correction emitted
        assert any(
            c["rule"] == "sro_direction_flipped"
            for c in result["bundle_corrections"]
        )
        # SRO actually flipped in-place
        flipped_sro = next(
            o for o in result["stix_bundle"]["objects"]
            if o.get("type") == "relationship"
        )
        assert flipped_sro["source_ref"] == proc["id"]
        assert flipped_sro["target_ref"] == tool["id"]

    def test_envelope_invalid_short_circuits(self):
        """Bundle envelope missing/wrong type: fail without scanning objects."""
        result = validate_bundle({"stix_bundle": {"type": "not-a-bundle"}})
        assert result["bundle_validation_failed"] is True
        assert any(
            c["rule"] == "bundle_envelope_invalid"
            for c in result["bundle_corrections"]
        )

    def test_non_sequential_drops_precedes(self):
        """is_sequential=False: PRECEDES SROs are dropped via auto-fix."""
        identity = _identity()
        proc1 = _proc(name="P1")
        proc2 = _proc(name="P2")
        proc1["created_by_ref"] = identity["id"]
        proc2["created_by_ref"] = identity["id"]
        ap_id = "attack-pattern--12345678-1234-4234-8234-123456789012"
        proc1["x_technique_refs"] = [ap_id]
        proc2["x_technique_refs"] = [ap_id]
        proc1["x_source_refs"] = [identity["id"]]
        proc2["x_source_refs"] = [identity["id"]]
        sro = _sro("precedes", proc1["id"], proc2["id"])
        bundle = _make_bundle([identity, proc1, proc2, sro])
        state = _state_with_bundle(bundle, is_sequential=False)

        result = validate_bundle(state)
        assert result["bundle_validation_failed"] is False
        # PRECEDES SRO is gone
        kept_sros = [
            o for o in result["stix_bundle"]["objects"]
            if o.get("type") == "relationship"
        ]
        assert kept_sros == []
        assert any(
            c["rule"] == "precedes_dropped_non_sequential"
            for c in result["bundle_corrections"]
        )


# =============================================================================
# Extension declarations
# =============================================================================

from app.nodes.deterministic.bundle_validator import _check_extension_declarations  # noqa: E402


def _x_procedure_meta() -> list[dict]:
    return [x_procedure_author_identity(), x_procedure_extension_definition()]


def _operator(declared: bool = True) -> dict:
    obj: dict[str, Any] = {
        "type": "attack-operator",
        "spec_version": "2.1",
        "id": f"attack-operator--{uuid.uuid4()}",
        "created": "2026-01-01T00:00:00.000Z",
        "modified": "2026-01-01T00:00:00.000Z",
        "operator": "AND",
        "effect_refs": [],
    }
    if declared:
        obj["extensions"] = extension_declaration(ATTACK_FLOW_EXTENSION_ID)
    return obj


class TestExtensionDeclarations:
    """Every custom object names its extension and the bundle carries it."""

    def test_conformant_objects_pass(self):
        objs = [
            _proc(), *_x_procedure_meta(),
            _operator(), attack_flow_author_identity(), attack_flow_extension_definition(),
        ]
        assert _check_extension_declarations(objs) == []

    def test_undeclared_procedure_is_a_hard_fail(self):
        objs = [_proc(declared=False), *_x_procedure_meta()]
        rules = [c["rule"] for c in _check_extension_declarations(objs)]
        assert rules == ["x_procedure_extension_undeclared"]

    def test_wrong_extension_type_counts_as_undeclared(self):
        proc = _proc(extra={"extensions": {X_PROCEDURE_EXTENSION_ID: {"extension_type": "property-extension"}}})
        rules = [c["rule"] for c in _check_extension_declarations([proc, *_x_procedure_meta()])]
        assert rules == ["x_procedure_extension_undeclared"]

    def test_undeclared_attack_operator_is_a_hard_fail(self):
        rules = [c["rule"] for c in _check_extension_declarations([_operator(declared=False)])]
        assert rules == ["attack_flow_extension_undeclared"]

    def test_declared_definition_absent_from_bundle_is_a_hard_fail(self):
        corrections = _check_extension_declarations([_proc()])
        assert [c["rule"] for c in corrections] == ["extension_definition_missing"]
        assert corrections[0]["ref_id"] == X_PROCEDURE_EXTENSION_ID

    def test_full_validator_rejects_a_bundle_missing_its_definition(self):
        bundle = _make_bundle([_identity(), _proc()], with_definitions=False)
        result = validate_bundle(_state_with_bundle(bundle))
        assert result["bundle_validation_failed"] is True
        rules = {c["rule"] for c in result["bundle_corrections"]}
        assert "extension_definition_missing" in rules

    def test_full_validator_accepts_the_embedded_definition(self):
        identity = _identity()
        proc = _proc(source_refs=[identity["id"]])
        proc["created_by_ref"] = identity["id"]
        bundle = _make_bundle([identity, proc])
        result = validate_bundle(_state_with_bundle(bundle))
        assert result["bundle_validation_failed"] is False, result.get("bundle_corrections")
