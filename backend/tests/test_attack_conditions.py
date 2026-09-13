"""Unit tests for attack-condition inference + SDO + routing-interaction.

Covers:
* extract_conditions — pull preconditions off chunks into the
  anchor-keyed dict shape the serializer reads.
* build_attack_condition_sdos — materialize attack-condition SDOs with
  on_true_refs / on_false_refs resolved to procedure STIX ids.
* infer_operators with condition_anchors — suppress OR-branch operator
  emission at any chunk that anchors a condition.
* route_precedes_through_operators with chunk_conditions — verify
  precedes routing inserts attack-condition between source procedure
  and its successors when a condition is present.
"""
from __future__ import annotations

from app.nodes.deterministic.attack_conditions import (
    build_attack_condition_sdos,
    extract_conditions,
)
from app.nodes.deterministic.attack_operators import (
    infer_operators,
    route_precedes_through_operators,
)


def _chunk(chunk_id: str, precedes: list[str] | None = None, precondition: dict | None = None) -> dict:
    c: dict = {"chunk_id": chunk_id, "precedes_ids": list(precedes or [])}
    if precondition is not None:
        c["precondition"] = precondition
    return c


def _precond(desc: str, on_true: list[str], on_false: list[str], pattern: str | None = None, pattern_type: str | None = None) -> dict:
    pre = {
        "description": desc,
        "on_true_ids": list(on_true),
        "on_false_ids": list(on_false),
        # Phase 1 _finalize_chunks also writes on_true_indices /
        # on_false_indices, but extract_conditions reads only the
        # *_ids fields — the indices are LLM-emission shape and don't
        # round-trip here.
    }
    if pattern is not None:
        pre["pattern"] = pattern
    if pattern_type is not None:
        pre["pattern_type"] = pattern_type
    return pre


class TestExtractConditions:
    def test_chunk_without_precondition_skipped(self):
        chunks = [_chunk("A"), _chunk("B")]
        assert extract_conditions(chunks, is_sequential=True) == {}

    def test_precondition_extracted_with_anchor_key(self):
        chunks = [
            _chunk("A", ["B", "C"], precondition=_precond(
                "actor checks domain-join state", on_true=["B"], on_false=["C"],
            )),
            _chunk("B"),
            _chunk("C"),
        ]
        out = extract_conditions(chunks, is_sequential=True)
        assert set(out.keys()) == {"A"}
        meta = out["A"]
        assert meta["description"] == "actor checks domain-join state"
        assert meta["on_true_ids"] == ["B"]
        assert meta["on_false_ids"] == ["C"]

    def test_pattern_round_trip(self):
        chunks = [
            _chunk("A", ["B", "C"], precondition=_precond(
                "checks Sense service",
                on_true=["B"], on_false=["C"],
                pattern="HKLM\\\\SYSTEM\\\\...\\\\Sense",
                pattern_type="regex",
            )),
            _chunk("B"),
            _chunk("C"),
        ]
        out = extract_conditions(chunks, is_sequential=True)
        assert out["A"]["pattern_type"] == "regex"
        assert "Sense" in out["A"]["pattern"]

    def test_both_sides_empty_skipped(self):
        # Defense-in-depth — _finalize_chunks should have dropped these,
        # but extract_conditions guards too.
        chunks = [
            _chunk("A", ["B"], precondition=_precond("vacuous", on_true=[], on_false=[])),
            _chunk("B"),
        ]
        assert extract_conditions(chunks, is_sequential=True) == {}

    def test_empty_description_skipped(self):
        chunks = [
            _chunk("A", ["B"], precondition=_precond("   ", on_true=["B"], on_false=[])),
            _chunk("B"),
        ]
        assert extract_conditions(chunks, is_sequential=True) == {}

    def test_non_sequential_returns_empty(self):
        chunks = [
            _chunk("A", ["B", "C"], precondition=_precond("real check", on_true=["B"], on_false=["C"])),
            _chunk("B"),
            _chunk("C"),
        ]
        assert extract_conditions(chunks, is_sequential=False) == {}

    def test_chunk_without_id_skipped(self):
        chunks = [
            {"precedes_ids": ["B"], "precondition": _precond("test", ["B"], [])},
        ]
        assert extract_conditions(chunks, is_sequential=True) == {}


class TestBuildAttackConditionSdos:
    def test_emits_one_sdo_per_condition(self):
        chunk_conditions = {
            "A": {
                "description": "actor checks domain-join state",
                "pattern": None,
                "pattern_type": None,
                "on_true_ids": ["B"],
                "on_false_ids": ["C"],
            }
        }
        chunk_to_proc = {"A": "x-procedure--a", "B": "x-procedure--b", "C": "x-procedure--c"}
        sdos, anchor_to_stix = build_attack_condition_sdos(
            chunk_conditions, chunk_to_proc,
            source_identity_id="identity--src",
            now_iso="2026-05-23T00:00:00Z",
            spec_version="2.1",
        )
        assert len(sdos) == 1
        (sdo,) = sdos
        assert sdo["type"] == "attack-condition"
        assert sdo["description"] == "actor checks domain-join state"
        assert sdo["on_true_refs"] == ["x-procedure--b"]
        assert sdo["on_false_refs"] == ["x-procedure--c"]
        assert sdo["created_by_ref"] == "identity--src"
        assert sdo["spec_version"] == "2.1"
        assert sdo["id"].startswith("attack-condition--")
        assert anchor_to_stix == {"A": sdo["id"]}

    def test_pattern_emitted_when_both_set(self):
        chunk_conditions = {
            "A": {
                "description": "checks Sense",
                "pattern": "HKLM\\\\SYSTEM\\\\Sense",
                "pattern_type": "regex",
                "on_true_ids": ["B"],
                "on_false_ids": [],
            }
        }
        chunk_to_proc = {"A": "x-procedure--a", "B": "x-procedure--b"}
        sdos, _ = build_attack_condition_sdos(
            chunk_conditions, chunk_to_proc,
            source_identity_id="identity--src",
            now_iso="2026-05-23T00:00:00Z",
            spec_version="2.1",
        )
        (sdo,) = sdos
        assert sdo["pattern"] == "HKLM\\\\SYSTEM\\\\Sense"
        assert sdo["pattern_type"] == "regex"

    def test_pattern_omitted_when_type_invalid(self):
        chunk_conditions = {
            "A": {
                "description": "checks something",
                "pattern": "foo",
                "pattern_type": "yaml",  # not in allowed enum
                "on_true_ids": ["B"],
                "on_false_ids": [],
            }
        }
        chunk_to_proc = {"A": "x-procedure--a", "B": "x-procedure--b"}
        sdos, _ = build_attack_condition_sdos(
            chunk_conditions, chunk_to_proc,
            source_identity_id="identity--src",
            now_iso="2026-05-23T00:00:00Z",
            spec_version="2.1",
        )
        (sdo,) = sdos
        assert "pattern" not in sdo
        assert "pattern_type" not in sdo

    def test_one_sided_partition_keeps_other_empty(self):
        chunk_conditions = {
            "A": {
                "description": "if EDR present, abort",
                "pattern": None,
                "pattern_type": None,
                "on_true_ids": ["B"],
                "on_false_ids": [],
            }
        }
        chunk_to_proc = {"A": "x-procedure--a", "B": "x-procedure--b"}
        sdos, _ = build_attack_condition_sdos(
            chunk_conditions, chunk_to_proc,
            source_identity_id="identity--src",
            now_iso="2026-05-23T00:00:00Z",
            spec_version="2.1",
        )
        (sdo,) = sdos
        assert sdo["on_true_refs"] == ["x-procedure--b"]
        assert sdo["on_false_refs"] == []

    def test_condition_with_no_resolved_refs_skipped(self):
        # B and C exist in chunk_conditions but neither was promoted to
        # a procedure (analyst rejected both drafts). The condition
        # collapses to no refs on either side → skipped.
        chunk_conditions = {
            "A": {
                "description": "test",
                "pattern": None,
                "pattern_type": None,
                "on_true_ids": ["B"],
                "on_false_ids": ["C"],
            }
        }
        chunk_to_proc = {"A": "x-procedure--a"}  # B, C missing
        sdos, anchor_to_stix = build_attack_condition_sdos(
            chunk_conditions, chunk_to_proc,
            source_identity_id="identity--src",
            now_iso="2026-05-23T00:00:00Z",
            spec_version="2.1",
        )
        assert sdos == []
        assert anchor_to_stix == {}


class TestInferOperatorsWithConditionAnchors:
    """Suppress OR-branch operator inference at any chunk that anchors
    an attack-condition. AND-converge operators downstream are
    unaffected."""

    def test_suppress_or_branch_at_condition_anchor(self):
        # A → B, A → C (would normally infer OR at A); A anchors a condition.
        chunks = [
            _chunk("A", ["B", "C"]),
            _chunk("B"),
            _chunk("C"),
        ]
        ops = infer_operators(chunks, is_sequential=True, condition_anchors={"A"})
        # No OR branch at A.
        assert all(op["anchor_chunk_id"] != "A" for op in ops.values())

    def test_converge_downstream_of_condition_still_inferred(self):
        # A branches to B and C (A anchors condition); both converge into D.
        # AND converge at D should still emit.
        chunks = [
            _chunk("A", ["B", "C"]),
            _chunk("B", ["D"]),
            _chunk("C", ["D"]),
            _chunk("D"),
        ]
        ops = infer_operators(chunks, is_sequential=True, condition_anchors={"A"})
        roles = {op["role"] for op in ops.values()}
        assert "converge" in roles
        assert "branch" not in roles

    def test_no_condition_anchors_unchanged_behavior(self):
        chunks = [
            _chunk("A", ["B", "C"]),
            _chunk("B"),
            _chunk("C"),
        ]
        ops_with = infer_operators(chunks, is_sequential=True, condition_anchors=set())
        ops_without = infer_operators(chunks, is_sequential=True)
        assert set(ops_with.keys()) == set(ops_without.keys())


class TestRoutePrecedesThroughConditions:
    def test_condition_routes_replace_direct_branch(self):
        # A has a condition; downstream B (on_true) and C (on_false).
        chunks = [
            _chunk("A", ["B", "C"]),
            _chunk("B"),
            _chunk("C"),
        ]
        chunk_to_proc = {"A": "p-a", "B": "p-b", "C": "p-c"}
        chunk_conditions = {
            "A": {
                "description": "domain-joined check",
                "on_true_ids": ["B"],
                "on_false_ids": ["C"],
            }
        }
        cond_anchor_to_stix = {"A": "attack-condition--cond1"}
        edges = route_precedes_through_operators(
            chunks,
            chunk_operators={},
            op_id_to_stix_id={},
            chunk_to_procedure_stix_id=chunk_to_proc,
            chunk_conditions=chunk_conditions,
            cond_anchor_to_stix_id=cond_anchor_to_stix,
        )
        # Expect: A → cond, cond → B, cond → C. NO direct A→B or A→C.
        assert sorted(edges) == sorted([
            ("p-a", "attack-condition--cond1"),
            ("attack-condition--cond1", "p-b"),
            ("attack-condition--cond1", "p-c"),
        ])
        assert ("p-a", "p-b") not in edges
        assert ("p-a", "p-c") not in edges

    def test_condition_then_converge_chains(self):
        # A has condition → B, C; B and C converge into D (AND).
        chunks = [
            _chunk("A", ["B", "C"]),
            _chunk("B", ["D"]),
            _chunk("C", ["D"]),
            _chunk("D"),
        ]
        chunk_to_proc = {"A": "p-a", "B": "p-b", "C": "p-c", "D": "p-d"}
        # Compute operators with condition anchor suppression.
        ops = infer_operators(chunks, is_sequential=True, condition_anchors={"A"})
        converge_id = next(op_id for op_id, m in ops.items() if m["role"] == "converge")
        op_id_to_stix = {converge_id: "attack-operator--and1"}
        chunk_conditions = {
            "A": {
                "description": "test",
                "on_true_ids": ["B"],
                "on_false_ids": ["C"],
            }
        }
        cond_anchor_to_stix = {"A": "attack-condition--cond1"}
        edges = route_precedes_through_operators(
            chunks,
            chunk_operators=ops,
            op_id_to_stix_id=op_id_to_stix,
            chunk_to_procedure_stix_id=chunk_to_proc,
            chunk_conditions=chunk_conditions,
            cond_anchor_to_stix_id=cond_anchor_to_stix,
        )
        expected = {
            ("p-a", "attack-condition--cond1"),
            ("attack-condition--cond1", "p-b"),
            ("attack-condition--cond1", "p-c"),
            ("p-b", "attack-operator--and1"),
            ("p-c", "attack-operator--and1"),
            ("attack-operator--and1", "p-d"),
        }
        assert set(edges) == expected

    def test_no_conditions_routes_through_operators_normally(self):
        # Regression: empty conditions dict shouldn't change operator
        # routing behavior (Phase 1 tests still cover the non-condition
        # cases but this is a defensive doubled check).
        chunks = [
            _chunk("A", ["C"]),
            _chunk("B", ["C"]),
            _chunk("C"),
        ]
        chunk_to_proc = {"A": "p-a", "B": "p-b", "C": "p-c"}
        ops = infer_operators(chunks, is_sequential=True)
        (op_id,) = ops.keys()
        op_id_to_stix = {op_id: "attack-operator--and1"}
        edges = route_precedes_through_operators(
            chunks, ops, op_id_to_stix, chunk_to_proc,
            chunk_conditions={},
            cond_anchor_to_stix_id={},
        )
        # Direct A→cond pattern from operator routing (unchanged from
        # the operator-only suite).
        assert ("p-a", "attack-operator--and1") in edges
        assert ("p-b", "attack-operator--and1") in edges
        assert ("attack-operator--and1", "p-c") in edges
