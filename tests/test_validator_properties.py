"""Property tests for the bundle validator.

WHY THIS FILE EXISTS:
The validator is the last line of defense before a bundle ships, and bundle
validation is all-or-nothing — if it misses something, the source completes
and the bad bundle is what you keep. It was tested only with hand-written
cases, and it turned out to have a hole big enough to drive two
schema violations through for months (nothing ever loaded the JSON Schema).
Hand-written cases only cover what someone thought to write down.

These tests take a KNOWN-GOOD bundle built from real captured state, break it
in ways that are unambiguously invalid, and assert the validator objects.

NO NEW DEPENDENCY. `hypothesis` would be the obvious tool, but valid STIX is
highly structured — generic strategies would spend their budget generating
garbage the validator rejects for uninteresting reasons, and getting them to
produce *nearly*-valid bundles means writing most of this anyway. A seeded,
domain-aware mutator is deterministic, reproducible from the seed, and aimed
at the mutations that actually matter.

BASELINE-RELATIVE, and that detail is load-bearing. Neo4j is stubbed here, so
the bundle has no log sources and `tuple_semantics` legitimately fails on the
LS element. An absolute "is it flagged?" check would therefore be true for
EVERY bundle including the unmutated one, and every assertion below would
pass while proving nothing. Each mutation must add a NEW objection.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from app.nodes.deterministic.bundle_validator import validate_bundle
from app.nodes.deterministic.normalization import normalize
from app.nodes.deterministic.serialization import X_PROCEDURE_TYPE, serialize_stix, _validate_bundle
from app.services import stix_schema

GOLDEN = Path(__file__).resolve().parent / "golden" / "cascading-shadows" / "after_gate_1.json"

pytestmark = pytest.mark.skipif(
    not GOLDEN.is_file(),
    reason=(
        "needs a golden fixture captured from a completed run with "
        "backend/scripts/capture_golden.py; the fixture is not shipped "
        "because it carries text derived from the source report"
    ),
)

SEED = 20260820  # deterministic: a failure here reproduces exactly


# ── the known-good bundle, built once ────────────────────────────────


@pytest.fixture(scope="module")
def good():
    """(state, bundle) from real captured state. Module-scoped: normalize +
    serialize + the ATT&CK catalogue load is too slow to repeat per test."""
    import asyncio

    async def build():
        state = json.loads(GOLDEN.read_text())
        w = copy.deepcopy(state)
        w.update(normalize(w))
        with patch(
            "app.nodes.deterministic.serialization.run_query",
            new=AsyncMock(return_value=[]),
        ):
            w.update(await serialize_stix(w))
        return w

    return asyncio.get_event_loop_policy().new_event_loop().run_until_complete(build())


def objections(bundle: dict, state: dict) -> set[str]:
    """Everything the validator objects to: failing check names, plus the
    rule name of every hard-fail correction."""
    results, _ = _validate_bundle(bundle)
    out = validate_bundle({**state, "stix_bundle": copy.deepcopy(bundle)})
    rules = {
        c["rule"] for c in out.get("bundle_corrections", [])
        if c.get("severity") == "hard_fail"
    }
    return {k for k, v in results.items() if not v} | {f"rule:{r}" for r in rules}


@pytest.fixture(scope="module")
def baseline(good):
    return objections(good["stix_bundle"], good)


# ── mutations that must always be caught ─────────────────────────────


def _first(objects, otype):
    return next(i for i, o in enumerate(objects) if o.get("type") == otype)


def _mutations():
    """(label, fn) pairs. Each fn breaks the bundle in place, unambiguously."""

    def drop_name(b):
        b["objects"][_first(b["objects"], X_PROCEDURE_TYPE)].pop("name", None)

    def drop_created(b):
        b["objects"][_first(b["objects"], X_PROCEDURE_TYPE)].pop("created", None)

    def undeclared_property(b):
        b["objects"][_first(b["objects"], X_PROCEDURE_TYPE)]["bogus_field"] = 1

    def malformed_id(b):
        b["objects"][_first(b["objects"], X_PROCEDURE_TYPE)]["id"] = "x-procedure--nope"

    def duplicate_id(b):
        i = _first(b["objects"], X_PROCEDURE_TYPE)
        b["objects"].append(copy.deepcopy(b["objects"][i]))

    def dangling_ref(b):
        b["objects"][_first(b["objects"], "relationship")]["target_ref"] = (
            "file--00000000-0000-4000-8000-000000000000"
        )

    def rel_missing_type(b):
        b["objects"][_first(b["objects"], "relationship")].pop("relationship_type", None)

    def empty_bundle(b):
        b["objects"] = []

    def two_flows(b):
        flow = next(o for o in b["objects"] if o.get("type") == "attack-flow")
        b["objects"].append({
            **copy.deepcopy(flow),
            "id": "attack-flow--99999999-9999-4999-8999-999999999999",
        })

    def confidence_out_of_range(b):
        b["objects"][_first(b["objects"], X_PROCEDURE_TYPE)]["confidence"] = 500

    def wrong_type_for_list_field(b):
        b["objects"][_first(b["objects"], X_PROCEDURE_TYPE)]["x_platforms"] = "windows"

    def precedes_self_loop(b):
        for o in b["objects"]:
            if o.get("relationship_type") == "precedes":
                o["target_ref"] = o["source_ref"]
                return
        # No precedes edge in this bundle — synthesize one.
        pid = b["objects"][_first(b["objects"], X_PROCEDURE_TYPE)]["id"]
        b["objects"].append({
            "type": "relationship", "spec_version": "2.1",
            "id": "relationship--88888888-8888-4888-8888-888888888888",
            "created": "2024-01-01T00:00:00.000Z",
            "modified": "2024-01-01T00:00:00.000Z",
            "relationship_type": "precedes", "source_ref": pid, "target_ref": pid,
        })

    return [
        ("drop required 'name' from a procedure", drop_name),
        ("drop 'created' from a procedure", drop_created),
        ("undeclared property on a procedure", undeclared_property),
        ("malformed STIX identifier", malformed_id),
        ("duplicate object id", duplicate_id),
        ("dangling target_ref", dangling_ref),
        ("relationship missing relationship_type", rel_missing_type),
        ("empty objects list", empty_bundle),
        ("a second attack-flow object", two_flows),
        ("confidence outside 0-100", confidence_out_of_range),
        ("string where a list is declared", wrong_type_for_list_field),
        ("PRECEDES self-loop", precedes_self_loop),
    ]


MUTATIONS = _mutations()


class TestBaselineIsUsable:
    """Guards every other test in this file."""

    def test_baseline_is_narrow(self, baseline):
        """If the unmutated bundle already objects to everything, a
        baseline-relative test can never detect anything."""
        assert baseline == {"tuple_semantics"}, (
            f"expected only the known Neo4j-stubbed LS failure, got {baseline}. "
            f"A wider baseline masks mutations."
        )

    def test_good_bundle_is_schema_clean(self, good):
        """The control must be genuinely valid where it claims to be."""
        assert stix_schema.validate_bundle_objects(good["stix_bundle"]["objects"]) == []


class TestMutationsAreCaught:

    @pytest.mark.parametrize("label,mutate", MUTATIONS, ids=[m[0] for m in MUTATIONS])
    def test_mutation_produces_a_new_objection(self, good, baseline, label, mutate):
        bundle = copy.deepcopy(good["stix_bundle"])
        mutate(bundle)
        new = objections(bundle, good) - baseline
        assert new, (
            f"the validator did not object to: {label}. An invalid bundle that "
            f"validates clean ships — bundle validation is all-or-nothing, so "
            f"nothing downstream will catch it."
        )


class TestValidatorRobustness:
    """Properties that must hold for ANY input, valid or not."""

    def test_validator_never_raises_on_random_damage(self, good):
        """A validator that crashes fails the source with a confusing
        traceback instead of an actionable correction. Seeded, so any
        failure reproduces exactly."""
        rng = random.Random(SEED)
        objs = good["stix_bundle"]["objects"]
        checked = 0

        garbage = [None, "", 123, True, [], [{"x": 1}], {"unexpected": ["nesting"]}]

        for _ in range(80):
            bundle = copy.deepcopy(good["stix_bundle"])
            damage = []
            for _ in range(rng.randint(1, 3)):  # several simultaneous edits
                target = rng.choice(bundle["objects"])
                keys = [k for k in target if k != "type"]
                if not keys:
                    continue
                key = rng.choice(keys)
                if rng.random() < 0.2:
                    target.pop(key, None)
                    damage.append(f"{target.get('type')}.{key}=<dropped>")
                else:
                    value = rng.choice(garbage)
                    target[key] = value
                    damage.append(f"{target.get('type')}.{key}={value!r}")

            try:
                out = validate_bundle({**good, "stix_bundle": bundle})
            except Exception as e:  # noqa: BLE001 — that is the property
                pytest.fail(
                    f"validate_bundle raised {type(e).__name__} (seed={SEED}) "
                    f"after: {damage} -> {e}"
                )
            assert "bundle_corrections" in out, (
                f"validator returned no verdict (seed={SEED}) after: {damage}"
            )
            checked += 1

        assert checked >= 70, f"only {checked} mutants ran — the fuzzer is not working"

    def test_a_crash_becomes_a_verdict_not_an_exception(self, good):
        """The backstop itself. Guarding every type assumption across a
        1500-line module does not converge — fuzzing found 173 crashes in
        2000 mutants across sites that had nothing in common but "assumed a
        string". The node wrapper turns any escape into an actionable
        hard_fail instead of a TypeError on the analyst's source row."""
        with patch(
            "app.nodes.deterministic.bundle_validator._validate_bundle_impl",
            side_effect=RuntimeError("boom"),
        ):
            out = validate_bundle({**good, "stix_bundle": copy.deepcopy(good["stix_bundle"])})

        rules = {c["rule"] for c in out["bundle_corrections"]}
        assert "validator_crashed" in rules
        assert out.get("bundle_validation_failed") is True
        assert "boom" in (out.get("error") or "")

    def test_a_clean_verdict_means_schema_clean(self, good):
        """If the validator says schema=True, the bundle it hands on really
        must be schema-valid. That was FALSE once: auto-fix
        passes mutated the bundle after the check and nothing re-validated."""
        rng = random.Random(SEED + 1)
        objs = good["stix_bundle"]["objects"]
        verdicts = 0

        for _ in range(40):
            bundle = copy.deepcopy(good["stix_bundle"])
            target = rng.choice(bundle["objects"])
            keys = [k for k in target if k not in ("type", "id")]
            if keys:
                target.pop(rng.choice(keys), None)

            out = validate_bundle({**good, "stix_bundle": bundle})
            hard = [c for c in out.get("bundle_corrections", [])
                    if c.get("severity") == "hard_fail"]
            if hard:
                continue  # rejected — nothing to promise about it
            verdicts += 1
            leftover = stix_schema.validate_bundle_objects(out["stix_bundle"]["objects"])
            assert not leftover, (
                f"validator passed a bundle that is not schema-valid "
                f"(seed={SEED + 1}): {leftover[:3]}"
            )

        assert verdicts >= 1, (
            "every mutant was rejected, so the 'clean verdict' property was "
            "never actually exercised"
        )

    def test_auto_fixes_are_idempotent(self, good):
        """Running the validator on its own output must not keep changing it.
        A non-idempotent auto-fix means the bundle you ship depends on how
        many times it happened to be validated."""
        first = validate_bundle({**good, "stix_bundle": copy.deepcopy(good["stix_bundle"])})
        second = validate_bundle({**good, "stix_bundle": copy.deepcopy(first["stix_bundle"])})

        assert second["stix_bundle"]["objects"] == first["stix_bundle"]["objects"], (
            "a second validation pass changed the bundle again — an auto-fix "
            "is not idempotent"
        )
        new_rules = (
            {c["rule"] for c in second.get("bundle_corrections", [])}
            - {c["rule"] for c in first.get("bundle_corrections", [])}
        )
        assert not new_rules, f"second pass raised new corrections: {sorted(new_rules)}"
