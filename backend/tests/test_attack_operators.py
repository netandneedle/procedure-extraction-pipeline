"""Unit tests for attack-operator inference + serialization helpers.

Covers the three pure helpers in
``app/nodes/deterministic/attack_operators.py``:

* :func:`infer_operators` — geometry walk over the chunk DAG
* :func:`build_attack_operator_sdos` — SDO materialization
* :func:`route_precedes_through_operators` — precedes-edge rewiring

The validator-side structural checks on the emitted attack-operator SDOs
live in ``tests/test_bundle_validator.py``.
"""
from __future__ import annotations

from app.nodes.deterministic.attack_operators import (
    build_attack_operator_sdos,
    infer_operators,
    route_precedes_through_operators,
)


def _chunk(chunk_id: str, precedes: list[str] | None = None) -> dict:
    return {"chunk_id": chunk_id, "precedes_ids": list(precedes or [])}


class TestInferOperators:
    def test_linear_chain_has_no_operators(self):
        chunks = [
            _chunk("A", ["B"]),
            _chunk("B", ["C"]),
            _chunk("C", []),
        ]
        assert infer_operators(chunks, is_sequential=True) == {}

    def test_converge_two_predecessors_emits_and(self):
        # A and B both lead to C.
        chunks = [
            _chunk("A", ["C"]),
            _chunk("B", ["C"]),
            _chunk("C", []),
        ]
        ops = infer_operators(chunks, is_sequential=True)
        assert len(ops) == 1
        (meta,) = ops.values()
        assert meta["kind"] == "AND"
        assert meta["role"] == "converge"
        assert meta["anchor_chunk_id"] == "C"
        assert meta["input_chunk_ids"] == ["A", "B"]
        assert meta["output_chunk_ids"] == ["C"]

    def test_branch_two_successors_emits_or(self):
        # A leads to both B and C.
        chunks = [
            _chunk("A", ["B", "C"]),
            _chunk("B", []),
            _chunk("C", []),
        ]
        ops = infer_operators(chunks, is_sequential=True)
        assert len(ops) == 1
        (meta,) = ops.values()
        assert meta["kind"] == "OR"
        assert meta["role"] == "branch"
        assert meta["anchor_chunk_id"] == "A"
        assert meta["input_chunk_ids"] == ["A"]
        assert meta["output_chunk_ids"] == ["B", "C"]

    def test_chunk_that_is_both_branch_and_converge(self):
        # A and B both lead to C; C leads to both D and E.
        chunks = [
            _chunk("A", ["C"]),
            _chunk("B", ["C"]),
            _chunk("C", ["D", "E"]),
            _chunk("D", []),
            _chunk("E", []),
        ]
        ops = infer_operators(chunks, is_sequential=True)
        # Two operators: one converge-AND at C (anchor=C), one branch-OR
        # at C (anchor=C). Distinct operator_ids because role differs.
        roles = sorted(meta["role"] for meta in ops.values())
        assert roles == ["branch", "converge"]
        kinds = sorted(meta["kind"] for meta in ops.values())
        assert kinds == ["AND", "OR"]

    def test_is_sequential_false_returns_empty(self):
        # Even with an obvious converge, catalog sources skip operator
        # inference entirely (they get no flow scaffolding).
        chunks = [
            _chunk("A", ["C"]),
            _chunk("B", ["C"]),
            _chunk("C", []),
        ]
        assert infer_operators(chunks, is_sequential=False) == {}

    def test_stable_id_across_runs(self):
        # Same geometry → same operator_ids across re-runs.
        chunks = [
            _chunk("A", ["C"]),
            _chunk("B", ["C"]),
            _chunk("C", []),
        ]
        ops1 = infer_operators(chunks, is_sequential=True)
        ops2 = infer_operators(chunks, is_sequential=True)
        assert sorted(ops1.keys()) == sorted(ops2.keys())

    def test_existing_overrides_preserve_kind(self):
        chunks = [
            _chunk("A", ["C"]),
            _chunk("B", ["C"]),
            _chunk("C", []),
        ]
        first = infer_operators(chunks, is_sequential=True)
        (op_id,) = first.keys()

        # Analyst flipped the kind to XOR via gate_chunks.
        overridden = {op_id: {**first[op_id], "kind": "XOR"}}

        second = infer_operators(chunks, is_sequential=True, existing_operators=overridden)
        assert second[op_id]["kind"] == "XOR"
        # Geometry fields are still recomputed (don't trust the override
        # for anything other than `kind`).
        assert second[op_id]["input_chunk_ids"] == ["A", "B"]

    def test_existing_override_with_invalid_kind_ignored(self):
        chunks = [
            _chunk("A", ["C"]),
            _chunk("B", ["C"]),
            _chunk("C", []),
        ]
        first = infer_operators(chunks, is_sequential=True)
        (op_id,) = first.keys()
        # Garbage kind value — ignored, defaults to geometry default (AND).
        overridden = {op_id: {"kind": "MAYBE", "role": "converge"}}
        second = infer_operators(chunks, is_sequential=True, existing_operators=overridden)
        assert second[op_id]["kind"] == "AND"

    def test_self_loop_skipped(self):
        # Defensive: a chunk listing itself in precedes_ids would
        # produce a 1-predecessor loop. infer_operators should not emit
        # an AND because the predecessor count is 1, not ≥2.
        chunks = [_chunk("A", ["A"])]
        assert infer_operators(chunks, is_sequential=True) == {}

    def test_dangling_precedes_ids_ignored(self):
        # Chunk A points at C, but C isn't in the list. Predecessor
        # counts should not see ghost edges.
        chunks = [
            _chunk("A", ["B", "C"]),  # C doesn't exist
            _chunk("B", []),
        ]
        # A → B is the only valid edge. B has 1 predecessor → no AND.
        # A has 1 successor (B only, C dropped) → no OR.
        assert infer_operators(chunks, is_sequential=True) == {}


class TestBuildAttackOperatorSdos:
    def test_emits_one_sdo_per_operator(self):
        chunk_operators = {
            "op-aaa111": {
                "kind": "AND",
                "role": "converge",
                "anchor_chunk_id": "C",
                "input_chunk_ids": ["A", "B"],
                "output_chunk_ids": ["C"],
            },
        }
        chunk_to_proc = {"A": "x-procedure--a", "B": "x-procedure--b", "C": "x-procedure--c"}
        sdos, op_id_to_stix = build_attack_operator_sdos(
            chunk_operators, chunk_to_proc,
            source_identity_id="identity--src",
            now_iso="2026-05-22T00:00:00Z",
            spec_version="2.1",
        )
        assert len(sdos) == 1
        (sdo,) = sdos
        assert sdo["type"] == "attack-operator"
        assert sdo["operator"] == "AND"
        assert sdo["effect_refs"] == ["x-procedure--c"]
        assert sdo["created_by_ref"] == "identity--src"
        assert sdo["spec_version"] == "2.1"
        assert sdo["id"].startswith("attack-operator--")
        assert op_id_to_stix == {"op-aaa111": sdo["id"]}

    def test_branch_emits_multi_effect_refs(self):
        chunk_operators = {
            "op-bbb222": {
                "kind": "OR",
                "role": "branch",
                "anchor_chunk_id": "A",
                "input_chunk_ids": ["A"],
                "output_chunk_ids": ["B", "C"],
            },
        }
        chunk_to_proc = {
            "A": "x-procedure--a", "B": "x-procedure--b", "C": "x-procedure--c",
        }
        sdos, _ = build_attack_operator_sdos(
            chunk_operators, chunk_to_proc,
            source_identity_id="identity--src",
            now_iso="2026-05-22T00:00:00Z",
            spec_version="2.1",
        )
        (sdo,) = sdos
        assert sdo["operator"] == "OR"
        assert sorted(sdo["effect_refs"]) == ["x-procedure--b", "x-procedure--c"]

    def test_branch_with_fewer_than_two_resolved_outputs_dropped(self):
        # If one of B/C didn't make it to the bundle (e.g. gate_1 reject),
        # the branch operator collapses to a single effect_ref → invalid.
        chunk_operators = {
            "op-ccc333": {
                "kind": "OR",
                "role": "branch",
                "anchor_chunk_id": "A",
                "input_chunk_ids": ["A"],
                "output_chunk_ids": ["B", "C"],
            },
        }
        chunk_to_proc = {"A": "x-procedure--a", "B": "x-procedure--b"}  # C missing
        sdos, op_id_to_stix = build_attack_operator_sdos(
            chunk_operators, chunk_to_proc,
            source_identity_id="identity--src",
            now_iso="2026-05-22T00:00:00Z",
            spec_version="2.1",
        )
        assert sdos == []
        assert op_id_to_stix == {}

    def test_converge_with_single_effect_ref_still_emits(self):
        # Convergence operators always have exactly ONE output (the
        # convergence point). The 2+ inputs flow IN through precedes,
        # the operator flows OUT to that one procedure.
        chunk_operators = {
            "op-ddd444": {
                "kind": "AND",
                "role": "converge",
                "anchor_chunk_id": "C",
                "input_chunk_ids": ["A", "B"],
                "output_chunk_ids": ["C"],
            },
        }
        chunk_to_proc = {"A": "x-procedure--a", "B": "x-procedure--b", "C": "x-procedure--c"}
        sdos, _ = build_attack_operator_sdos(
            chunk_operators, chunk_to_proc,
            source_identity_id="identity--src",
            now_iso="2026-05-22T00:00:00Z",
            spec_version="2.1",
        )
        assert len(sdos) == 1
        assert sdos[0]["effect_refs"] == ["x-procedure--c"]

    def test_empty_input_returns_empty(self):
        sdos, op_map = build_attack_operator_sdos(
            {}, {}, "identity--src", "now", "2.1",
        )
        assert sdos == []
        assert op_map == {}


class TestRoutePrecedesThroughOperators:
    def test_linear_chain_no_operators_yields_direct_edges(self):
        chunks = [
            _chunk("A", ["B"]),
            _chunk("B", ["C"]),
            _chunk("C", []),
        ]
        chunk_to_proc = {"A": "p-a", "B": "p-b", "C": "p-c"}
        edges = route_precedes_through_operators(
            chunks, chunk_operators={}, op_id_to_stix_id={},
            chunk_to_procedure_stix_id=chunk_to_proc,
        )
        assert sorted(edges) == [("p-a", "p-b"), ("p-b", "p-c")]

    def test_converge_routes_through_and_operator(self):
        chunks = [
            _chunk("A", ["C"]),
            _chunk("B", ["C"]),
            _chunk("C", []),
        ]
        chunk_to_proc = {"A": "p-a", "B": "p-b", "C": "p-c"}
        ops = infer_operators(chunks, is_sequential=True)
        (op_id,) = ops.keys()
        op_id_to_stix = {op_id: "attack-operator--and1"}

        edges = route_precedes_through_operators(
            chunks, chunk_operators=ops, op_id_to_stix_id=op_id_to_stix,
            chunk_to_procedure_stix_id=chunk_to_proc,
        )
        # Expect: A → op, B → op, op → C. NO direct A→C or B→C.
        assert sorted(edges) == sorted([
            ("p-a", "attack-operator--and1"),
            ("p-b", "attack-operator--and1"),
            ("attack-operator--and1", "p-c"),
        ])
        # Defensive: confirm the direct edges are not present.
        assert ("p-a", "p-c") not in edges
        assert ("p-b", "p-c") not in edges

    def test_branch_routes_through_or_operator(self):
        chunks = [
            _chunk("A", ["B", "C"]),
            _chunk("B", []),
            _chunk("C", []),
        ]
        chunk_to_proc = {"A": "p-a", "B": "p-b", "C": "p-c"}
        ops = infer_operators(chunks, is_sequential=True)
        (op_id,) = ops.keys()
        op_id_to_stix = {op_id: "attack-operator--or1"}

        edges = route_precedes_through_operators(
            chunks, chunk_operators=ops, op_id_to_stix_id=op_id_to_stix,
            chunk_to_procedure_stix_id=chunk_to_proc,
        )
        # Expect: A → op, op → B, op → C.
        assert sorted(edges) == sorted([
            ("p-a", "attack-operator--or1"),
            ("attack-operator--or1", "p-b"),
            ("attack-operator--or1", "p-c"),
        ])

    def test_branch_then_converge_chains_operators(self):
        # A branches to B and C; B and C converge into D.
        chunks = [
            _chunk("A", ["B", "C"]),
            _chunk("B", ["D"]),
            _chunk("C", ["D"]),
            _chunk("D", []),
        ]
        chunk_to_proc = {"A": "p-a", "B": "p-b", "C": "p-c", "D": "p-d"}
        ops = infer_operators(chunks, is_sequential=True)
        # Identify operators by role.
        branch_id = next(op for op, m in ops.items() if m["role"] == "branch")
        converge_id = next(op for op, m in ops.items() if m["role"] == "converge")
        op_id_to_stix = {branch_id: "attack-operator--or1", converge_id: "attack-operator--and1"}

        edges = route_precedes_through_operators(
            chunks, chunk_operators=ops, op_id_to_stix_id=op_id_to_stix,
            chunk_to_procedure_stix_id=chunk_to_proc,
        )
        # Expect:
        #   A → op_or (branch feed-in)
        #   op_or → p-b, op_or → p-c (branch feed-out)
        #   p-b → op_and, p-c → op_and (converge feed-in)
        #   op_and → p-d (converge feed-out)
        expected = {
            ("p-a", "attack-operator--or1"),
            ("attack-operator--or1", "p-b"),
            ("attack-operator--or1", "p-c"),
            ("p-b", "attack-operator--and1"),
            ("p-c", "attack-operator--and1"),
            ("attack-operator--and1", "p-d"),
        }
        assert set(edges) == expected

    def test_unresolved_chunk_drops_edges_silently(self):
        # If A's chunk_id has no procedure (e.g. analyst rejected the draft),
        # any precedes edges touching A are dropped.
        chunks = [
            _chunk("A", ["B"]),
            _chunk("B", []),
        ]
        chunk_to_proc = {"B": "p-b"}  # A absent
        edges = route_precedes_through_operators(
            chunks, chunk_operators={}, op_id_to_stix_id={},
            chunk_to_procedure_stix_id=chunk_to_proc,
        )
        assert edges == []

    def test_operator_skipped_by_sdo_builder_routes_around(self):
        # An operator that build_attack_operator_sdos dropped (and so
        # has no op_id_to_stix mapping) is routed-around: direct
        # procedure→procedure precedes resume for that subgraph.
        chunks = [
            _chunk("A", ["C"]),
            _chunk("B", ["C"]),
            _chunk("C", []),
        ]
        chunk_to_proc = {"A": "p-a", "B": "p-b", "C": "p-c"}
        ops = infer_operators(chunks, is_sequential=True)
        # Operator exists in chunk_operators but NOT in op_id_to_stix
        # (simulates a build_attack_operator_sdos skip).
        edges = route_precedes_through_operators(
            chunks, chunk_operators=ops, op_id_to_stix_id={},
            chunk_to_procedure_stix_id=chunk_to_proc,
        )
        # Direct edges resume.
        assert sorted(edges) == [("p-a", "p-c"), ("p-b", "p-c")]


class TestSpliceAbsent:
    """A procedure missing from the bundle must not sever the chain.

    A chunk loses its procedure for several reasons — removed at Gate 1,
    rejected there, or omitted by serialize_stix for having no ATT&CK
    technique. Previously every edge through it was skipped, so A -> B -> C
    with B gone left A and C disconnected and _build_attack_flow read C as a
    new chain root. On one ransomware run 3 of 21 edges were lost this way, and
    nothing said so.
    """

    def test_edge_is_bridged_across_a_removed_chunk(self):
        from app.nodes.deterministic.attack_operators import splice_absent

        successors = {"a": ["b"], "b": ["c"], "c": []}
        spliced, bridged, dropped = splice_absent(successors, {"a", "c"})

        assert spliced["a"] == ["c"], "A must still precede C"
        assert bridged == 1
        assert dropped == 0

    def test_consecutive_removals_still_bridge(self):
        """The chunk-gate splice loses this case; the shared helper must not."""
        from app.nodes.deterministic.attack_operators import splice_absent

        successors = {"a": ["b"], "b": ["c"], "c": ["d"], "d": []}
        spliced, bridged, _ = splice_absent(successors, {"a", "d"})

        assert spliced["a"] == ["d"]
        assert bridged == 1

    def test_edge_into_a_dead_end_is_dropped_and_counted(self):
        from app.nodes.deterministic.attack_operators import splice_absent

        successors = {"a": ["b"], "b": []}
        spliced, bridged, dropped = splice_absent(successors, {"a"})

        assert spliced["a"] == []
        assert (bridged, dropped) == (0, 1)

    def test_branch_through_a_removed_chunk_keeps_both_arms(self):
        """The case that cost one ransomware bundle an operator: a branch whose
        arm ran through a removed chunk looked like a single-output node."""
        from app.nodes.deterministic.attack_operators import splice_absent

        successors = {"a": ["b", "x"], "b": ["c"], "c": [], "x": []}
        spliced, _, _ = splice_absent(successors, {"a", "c", "x"})

        assert sorted(spliced["a"]) == ["c", "x"], "still a branch"

    def test_cycle_among_removed_chunks_terminates(self):
        from app.nodes.deterministic.attack_operators import splice_absent

        successors = {"a": ["b"], "b": ["c"], "c": ["b", "d"], "d": []}
        spliced, _, _ = splice_absent(successors, {"a", "d"})

        assert spliced["a"] == ["d"]

    def test_nothing_absent_is_a_no_op(self):
        from app.nodes.deterministic.attack_operators import splice_absent

        successors = {"a": ["b"], "b": ["c"], "c": []}
        spliced, bridged, dropped = splice_absent(successors, {"a", "b", "c"})

        assert spliced == successors
        assert (bridged, dropped) == (0, 0)

    def test_self_reference_is_never_produced(self):
        """Bridging must not make a chunk precede itself."""
        from app.nodes.deterministic.attack_operators import splice_absent

        successors = {"a": ["b"], "b": ["a", "c"], "c": []}
        spliced, _, _ = splice_absent(successors, {"a", "c"})

        assert "a" not in spliced["a"]
        assert spliced["a"] == ["c"]
