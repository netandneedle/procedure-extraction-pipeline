"""Tests for the AI gate reviewer.

Three things carry weight here, and the rest is scaffolding:

  1. BACK-COMPAT. A source with no gate_modes must behave exactly as it did
     before this feature existed. That is the property that makes shipping
     this safe, so it is tested from every shape a real row or checkpoint can
     hold.
  2. THE GROUNDING GUARDRAIL. An AI reviewer is a new place for fabricated
     evidence to enter the pipeline. The mutation test below fails if the
     rule is disabled — without that, a test asserting "unsupported quote is
     downgraded" can pass on a code path that downgrades nothing.
  3. FAIL-SOFT. The reviewer erroring must leave the gate reviewable.
"""

import asyncio
import contextlib
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.models.reviewer import INITIAL_READ

from app.graph.state import (
    DEFAULT_GATE_MODE,
    GATE_KEYS,
    GATE_MODES,
    gate_mode,
    normalize_gate_modes,
)
from app.services.grounding import QUOTE_SUPPORT_THRESHOLD
from app.services.reviewer.outcomes import diff_entity_gate
from app.services.reviewer.runner import apply_grounding


# ── gate modes ───────────────────────────────────────────────────────

class TestNormalizeGateModes:
    def test_none_is_all_review(self):
        assert normalize_gate_modes(None) == {k: "review" for k in GATE_KEYS}

    def test_bare_string_applies_to_every_gate(self):
        assert normalize_gate_modes("assist") == {k: "assist" for k in GATE_KEYS}

    def test_partial_dict_fills_missing_with_review(self):
        out = normalize_gate_modes({"entities": "assist"})
        assert out["entities"] == "assist"
        assert all(out[k] == "review" for k in GATE_KEYS if k != "entities")

    @pytest.mark.parametrize("bad", [
        "nope",                      # unknown bare mode
        {"entities": "nope"},        # unknown mode in a dict
        {"entities": True},          # bool where a mode string belongs
        {"not_a_gate": "assist"},    # unknown gate key
        7,                           # wrong type entirely
    ])
    def test_rejects_garbage(self, bad):
        with pytest.raises(ValueError):
            normalize_gate_modes(bad)

    def test_every_declared_mode_is_accepted(self):
        # Guards against GATE_MODES and the validator drifting apart.
        for mode in GATE_MODES:
            assert normalize_gate_modes(mode)["entities"] == mode


class TestGateModeBackCompat:
    """A state with no gate_modes must read as today's behavior."""

    @pytest.mark.parametrize("state", [
        {},                                          # brand new
        {"gates_enabled": True},                     # legacy bool checkpoint
        {"gates_enabled": {"entities": True}},       # dict, no modes
        {"gate_modes": None},                        # column present, unset
        {"gate_modes": "not a dict"},                # corrupt
        {"gate_modes": {"entities": "bogus"}},       # unknown mode value
        {"gate_modes": {"entities": True}},          # wrong type
    ])
    def test_defaults_to_review(self, state):
        assert gate_mode(state, "entities") == DEFAULT_GATE_MODE == "review"

    def test_reads_a_valid_mode(self):
        assert gate_mode({"gate_modes": {"entities": "assist"}}, "entities") == "assist"

    def test_unlisted_gate_is_review(self):
        state = {"gate_modes": {"entities": "assist"}}
        assert gate_mode(state, "bundle") == "review"


# ── the grounding guardrail ──────────────────────────────────────────

class TestApplyGrounding:
    REPORT_TOKENS = {
        "actor", "deployed", "netsupport", "remote", "access", "trojan",
        "registry", "persistence", "windows", "currentversion",
    }

    def _rec(self, quote, confidence="high"):
        return {
            "entities": [{
                "entity_id": "e1", "action": "remove",
                "confidence": confidence, "rationale": "because",
                "evidence_quote": quote,
            }],
        }

    def test_supported_quote_keeps_its_confidence(self):
        payload = self._rec("the actor deployed NetSupport remote access trojan")
        assert apply_grounding(payload, self.REPORT_TOKENS) == 0
        rec = payload["entities"][0]
        assert rec["confidence"] == "high"
        assert not rec.get("quote_unsupported")
        assert rec["quote_source_support"] >= QUOTE_SUPPORT_THRESHOLD

    def test_fabricated_quote_is_forced_to_low(self):
        payload = self._rec(
            "copying pasting attacker supplied command Windows dialog invented"
        )
        assert apply_grounding(payload, self.REPORT_TOKENS) == 1
        rec = payload["entities"][0]
        assert rec["confidence"] == "low"
        assert rec["quote_unsupported"] is True
        assert rec["quote_source_support"] < QUOTE_SUPPORT_THRESHOLD

    def test_downgrade_is_what_excludes_it_from_bulk_accept(self):
        """The downgrade has to bite, not just annotate.

        'low' is the value the UI's bulk-accept filter excludes. If the rule
        only set a flag and left confidence alone, an unsupported
        recommendation would still be one click from being applied to the
        bundle — which is the whole failure this guardrail exists to stop.
        """
        payload = self._rec("entirely invented phrasing nobody wrote anywhere")
        apply_grounding(payload, self.REPORT_TOKENS)
        assert payload["entities"][0]["confidence"] == "low"

    def test_empty_quote_is_left_alone(self):
        # An honest "I have no quote" is not a fabrication. It stays as the
        # reviewer rated it; the analyst sees no quote and judges accordingly.
        payload = self._rec("")
        assert apply_grounding(payload, self.REPORT_TOKENS) == 0
        assert payload["entities"][0]["confidence"] == "high"

    def test_short_quote_is_not_judged(self):
        # Under the token floor the ratio is too coarse to mean anything.
        payload = self._rec("the actor")
        assert apply_grounding(payload, self.REPORT_TOKENS) == 0
        assert payload["entities"][0]["confidence"] == "high"

    def test_no_source_corpus_disables_the_check(self):
        # Fails OPEN on purpose: a source we cannot read should not have
        # every recommendation rejected.
        payload = self._rec("entirely invented phrasing nobody wrote anywhere")
        assert apply_grounding(payload, set()) == 0
        assert payload["entities"][0]["confidence"] == "high"

    def test_added_entities_are_checked_too(self):
        payload = {"added_entities": [{
            "value": "GHOSTWRITER", "entity_type": "malware",
            "confidence": "high", "rationale": "r",
            "evidence_quote": "fabricated sentence with no overlap whatsoever",
        }]}
        assert apply_grounding(payload, self.REPORT_TOKENS) == 1
        assert payload["added_entities"][0]["confidence"] == "low"

    def test_already_low_is_not_double_counted(self):
        payload = self._rec("fabricated sentence with no overlap", confidence="low")
        # Still flagged, but not counted as a downgrade — it was already there.
        assert apply_grounding(payload, self.REPORT_TOKENS) == 0
        assert payload["entities"][0]["quote_unsupported"] is True


# ── the measurement instrument ───────────────────────────────────────

class TestDiffEntityGate:
    def test_agreement_when_analyst_follows_advice(self):
        payload = {"entities": [
            {"entity_id": "e1", "action": "remove", "confidence": "high"},
        ]}
        out = diff_entity_gate(payload, [{"entity_id": "e1", "action": "remove"}], [])
        assert out["agreement"]["agreed"] == 1
        assert out["agreement"]["overridden"] == 0
        assert out["agreement"]["rate"] == 1.0

    def test_override_is_recorded_with_both_sides(self):
        payload = {"entities": [
            {"entity_id": "e1", "action": "remove", "confidence": "high"},
        ]}
        out = diff_entity_gate(payload, [{"entity_id": "e1", "action": "approve"}], [])
        item = out["agreement"]["items"][0]
        assert item["recommended"] == "remove"
        assert item["actual"] == "approve"
        assert item["agreed"] is False
        assert out["agreement"]["overridden"] == 1

    def test_silence_counts_as_approve(self):
        """The gate keeps an entity nobody mentions, so silence is assent."""
        payload = {"entities": [
            {"entity_id": "e1", "action": "approve", "confidence": "high"},
        ]}
        out = diff_entity_gate(payload, [], [])
        assert out["agreement"]["agreed"] == 1

    def test_a_recommended_removal_ignored_is_an_override(self):
        payload = {"entities": [
            {"entity_id": "e1", "action": "remove", "confidence": "high"},
        ]}
        out = diff_entity_gate(payload, [], [])
        assert out["agreement"]["overridden"] == 1

    def test_addition_taken_and_not_taken(self):
        payload = {"added_entities": [
            {"value": "TAKEN", "entity_type": "tool", "confidence": "high"},
            {"value": "IGNORED", "entity_type": "tool", "confidence": "low"},
        ]}
        out = diff_entity_gate(
            payload, [], [{"value": "taken", "entity_type": "TOOL"}],
        )
        by_ref = {i["ref"]: i for i in out["agreement"]["items"]}
        # Matching is case-insensitive — the analyst may retype the casing.
        assert by_ref["TAKEN"]["agreed"] is True
        assert by_ref["IGNORED"]["agreed"] is False

    def test_no_recommendations_gives_no_rate(self):
        """Not 1.0. Agreeing with zero recommendations is not evidence."""
        out = diff_entity_gate({}, [{"entity_id": "e1", "action": "approve"}], [])
        assert out["agreement"]["total"] == 0
        assert out["agreement"]["rate"] is None

    def test_carries_the_quote_verdict_through(self):
        # So an analysis can ask whether ungrounded recommendations are
        # overridden more often — the question that says whether the
        # guardrail is calibrated.
        payload = {"entities": [{
            "entity_id": "e1", "action": "remove",
            "confidence": "low", "quote_unsupported": True,
        }]}
        out = diff_entity_gate(payload, [{"entity_id": "e1", "action": "approve"}], [])
        assert out["agreement"]["items"][0]["quote_unsupported"] is True


# ── fail-soft ────────────────────────────────────────────────────────

class TestReviewerFailsSoft:
    """The reviewer erroring must leave the gate reviewable by a human.

    This is the contract that makes the feature safe to enable. An assistant
    that can take a pipeline run down with it is a liability, so every failure
    path has to end in "the gate behaves exactly as it does without me".
    """

    @pytest.mark.asyncio
    async def test_llm_failure_returns_none_and_does_not_raise(self, monkeypatch):
        from app.services.reviewer import runner

        async def boom(*a, **k):
            raise RuntimeError("anthropic is down")

        recorded: list[tuple] = []

        async def fake_record_failure(db, source_id, gate_key, error):
            recorded.append((gate_key, error))

        monkeypatch.setattr(runner, "call_llm", boom)
        monkeypatch.setattr(runner.store, "record_failure", fake_record_failure)
        monkeypatch.setattr(runner.store, "load_turns", _async_return([]))

        result = await runner.run_reviewer(
            uuid.uuid4(), "entities", {"parsed_text": "a report", "entities": []},
        )
        assert result is None
        assert recorded and "anthropic is down" in recorded[0][1]

    @pytest.mark.asyncio
    async def test_bookkeeping_failure_is_also_swallowed(self, monkeypatch):
        """Even failing to RECORD the failure must not escalate."""
        from app.services.reviewer import runner

        async def boom(*a, **k):
            raise RuntimeError("anthropic is down")

        async def also_boom(*a, **k):
            raise RuntimeError("postgres is down too")

        monkeypatch.setattr(runner, "call_llm", boom)
        monkeypatch.setattr(runner.store, "record_failure", also_boom)
        monkeypatch.setattr(runner.store, "load_turns", _async_return([]))

        assert await runner.run_reviewer(uuid.uuid4(), "entities", {}) is None

    @pytest.mark.asyncio
    async def test_unimplemented_gate_returns_none_without_calling_the_model(
        self, monkeypatch,
    ):
        from app.services.reviewer import runner

        called = []

        async def tracker(*a, **k):
            called.append(1)

        monkeypatch.setattr(runner, "call_llm", tracker)
        assert await runner.run_reviewer(uuid.uuid4(), "bundle", {}) is None
        assert not called, "no reviewer for this gate — must not spend a call"


class _fake_session:
    """Stand-in for `async with async_session() as db`. The reviewer's DB
    calls are patched out individually; this just satisfies the context
    manager so no real connection is opened."""

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return None

    async def __aexit__(self, *exc):
        return False


def _async_return(value):
    async def _inner(*a, **k):
        return value
    return _inner


# ── the entity vocabulary ────────────────────────────────────────────

class TestDropInvalidEntityTypes:
    """Found on the first live run, not by inspection.

    Asked to review one campaign report, the reviewer proposed adding CLICKFIX with
    `entity_type: "attack_pattern"` — a real STIX type, a plausible answer,
    and not one of ours. The addition channel does not validate type
    (`_build_added_entity` copies whatever it is handed), so it would have
    landed in validated_entities and been carried downstream.

    Two fixes, and both are tested: the tool schema now offers EntityType as
    an enum so the model cannot invent one, and this boundary check catches
    it if the model ignores the enum anyway.
    """

    def test_drops_the_type_that_actually_happened(self):
        from app.services.reviewer.runner import drop_invalid_entity_types

        payload = {"added_entities": [
            {"value": "CLICKFIX", "entity_type": "attack_pattern"},
            {"value": "legal & professional services", "entity_type": "victim_sector"},
        ]}
        assert drop_invalid_entity_types(payload) == 1
        assert [r["value"] for r in payload["added_entities"]] == [
            "legal & professional services",
        ]

    def test_drops_an_edit_naming_an_unknown_type(self):
        from app.services.reviewer.runner import drop_invalid_entity_types

        payload = {"entities": [
            {"entity_id": "e1", "action": "edit", "edited_type": "not_a_type"},
            {"entity_id": "e2", "action": "edit", "edited_type": "malware"},
        ]}
        assert drop_invalid_entity_types(payload) == 1
        assert [r["entity_id"] for r in payload["entities"]] == ["e2"]

    def test_an_entity_recommendation_without_a_type_is_untouched(self):
        # approve/remove don't name a type; there is nothing to validate.
        from app.services.reviewer.runner import drop_invalid_entity_types

        payload = {"entities": [{"entity_id": "e1", "action": "remove"}]}
        assert drop_invalid_entity_types(payload) == 0
        assert len(payload["entities"]) == 1

    def test_an_addition_with_no_type_at_all_is_dropped(self):
        # entity_type is mandatory for an addition — there is no default that
        # would be safe to guess.
        from app.services.reviewer.runner import drop_invalid_entity_types

        payload = {"added_entities": [{"value": "orphan"}]}
        assert drop_invalid_entity_types(payload) == 1
        assert payload["added_entities"] == []

    def test_tool_schema_offers_exactly_the_real_vocabulary(self):
        """Root-cause fix: the model is handed the enum, not a prose hint."""
        from app.graph.state import EntityType
        from app.services.reviewer.gate_reviewers import REVIEW_ENTITIES_TOOL

        props = REVIEW_ENTITIES_TOOL["input_schema"]["properties"]
        offered = set(props["added_entities"]["items"]["properties"]["entity_type"]["enum"])
        assert offered == {e.value for e in EntityType}
        assert "attack_pattern" not in offered


class TestEnforceSectorVocabulary:
    """The second vocabulary gap, and the one that taught the general lesson.

    On one campaign report the reviewer recommended adding victim_sector "legal &
    professional services". Verbatim from the report, quote support 1.0,
    perfectly well-evidenced — and wrong. The extractor had deliberately not
    emitted it: `entity_extraction.py` names that exact string as its worked
    example of a sector with no industry-sector-ov value, and instructs the
    model to pick the closest listed value instead.

    So the reviewer "corrected" a correct decision, with real evidence
    attached, because it held less context than the stage it was reviewing.
    The analyst took the advice and the bundle validator silently dropped the
    sector (`invalid_sector_dropped`, severity warn) — time spent, nothing
    gained, no error raised.

    The general rule this produced: any constrained vocabulary the extractor
    is given must reach the reviewer too.
    """

    def test_drops_the_value_that_actually_happened(self):
        from app.services.reviewer.runner import enforce_sector_vocabulary

        payload = {"added_entities": [
            {"value": "legal & professional services", "entity_type": "victim_sector"},
        ]}
        assert enforce_sector_vocabulary(payload) == 1
        assert payload["added_entities"] == []

    def test_folds_a_valid_sector_field_into_value(self):
        from app.services.reviewer.runner import enforce_sector_vocabulary

        payload = {"added_entities": [{
            "value": "legal & professional services",
            "entity_type": "victim_sector",
            "sector": "commercial",
        }]}
        assert enforce_sector_vocabulary(payload) == 0
        rec = payload["added_entities"][0]
        assert rec["value"] == "commercial"
        assert "sector" not in rec, "the helper field must not reach the gate"

    def test_a_value_already_in_the_vocabulary_passes(self):
        from app.services.reviewer.runner import enforce_sector_vocabulary

        payload = {"added_entities": [
            {"value": "government", "entity_type": "victim_sector"},
        ]}
        assert enforce_sector_vocabulary(payload) == 0
        assert payload["added_entities"][0]["value"] == "government"

    def test_non_sector_entities_are_untouched_but_stripped(self):
        from app.services.reviewer.runner import enforce_sector_vocabulary

        payload = {"added_entities": [
            {"value": "ACME", "entity_type": "organization", "sector": "stray"},
        ]}
        assert enforce_sector_vocabulary(payload) == 0
        rec = payload["added_entities"][0]
        assert rec["value"] == "ACME"
        assert "sector" not in rec

    def test_tool_schema_offers_the_real_vocabulary(self):
        from app.services.reviewer.gate_reviewers import REVIEW_ENTITIES_TOOL
        from app.services.stix_schema import industry_sector_vocab

        props = REVIEW_ENTITIES_TOOL["input_schema"]["properties"]
        offered = set(props["added_entities"]["items"]["properties"]["sector"]["enum"])
        assert offered == set(industry_sector_vocab())
        assert "legal & professional services" not in offered


class TestFlattenedInitialRead:
    """The opening read arrives flattened, and it costs real money.

    On BOTH live runs the model emitted `initial_read` and copies of its
    sub-fields at the top level. Under extra='forbid' that fails validation,
    and the adapter's retry meant every reviewer call spent two reviewer-model requests
    instead of one — a silent 2x on the most expensive call in the pipeline,
    visible only as a WARNING nobody was reading.
    """

    def _base(self, **over):
        payload = {"entities": [], "added_entities": []}
        payload.update(over)
        return payload

    def test_the_shape_both_live_runs_produced(self):
        from app.services.reviewer.models import Gate0Recommendations

        out = Gate0Recommendations.model_validate(self._base(
            initial_read={
                "summary": "nested", "actors": ["A"],
                "attack_chain": ["x"], "thin_areas": [],
            },
            actors=["A"], attack_chain=["x"], thin_areas=[],
            notes_for_later_gates="flat note",
        ))
        assert out.initial_read.summary == "nested"
        # The flat-only field is LIFTED, not discarded.
        assert out.initial_read.notes_for_later_gates == "flat note"

    def test_flat_only_is_not_discarded(self):
        """Dropping strays instead of lifting would lose the whole read here."""
        from app.services.reviewer.models import Gate0Recommendations

        out = Gate0Recommendations.model_validate(self._base(
            summary="flat only", actors=[], attack_chain=[], thin_areas=[],
        ))
        assert out.initial_read is not None
        assert out.initial_read.summary == "flat only"

    def test_nested_wins_a_conflict(self):
        # Both present and disagreeing: prefer the shape the schema asked for.
        from app.services.reviewer.models import Gate0Recommendations

        out = Gate0Recommendations.model_validate(self._base(
            initial_read={
                "summary": "nested", "actors": [],
                "attack_chain": [], "thin_areas": [],
            },
            summary="flat",
        ))
        assert out.initial_read.summary == "nested"

    def test_absent_initial_read_stays_absent(self):
        """Later gates replay the read from the transcript and omit it."""
        from app.services.reviewer.models import Gate0Recommendations

        assert Gate0Recommendations.model_validate(self._base()).initial_read is None

    def test_the_opening_read_is_required_of_the_gate_that_opens_it(self):
        """A field the model MAY omit is a field it sometimes will.

        Observed live: a run where the entities reviewer emitted no
        `initial_read` at all. Nothing failed — the entity review was fine and
        the source completed — but `GET /brief` 404'd and the chunk gate had
        no attack chain to compare its decomposition against, which is the
        whole reason that gate is worth a stateful reviewer. It is also the
        analyst's correction surface, so its absence quietly removes the
        cheapest place in the system to fix a misread.
        """
        from app.services.reviewer.gate_reviewers import (
            REVIEW_CHUNKS_TOOL,
            REVIEW_ENTITIES_TOOL,
            REVIEW_PROCEDURES_TOOL,
            REVIEW_BUNDLE_TOOL,
        )

        schema = REVIEW_ENTITIES_TOOL["input_schema"]
        assert "initial_read" in schema["properties"]
        assert "initial_read" in schema["required"], (
            "the opening read is optional, so the model may omit it — and on "
            "a live run it did"
        )

        # And exactly one gate asks for it. A later gate's read would be
        # anchored on the extraction it is supposed to be judging.
        for tool in (REVIEW_CHUNKS_TOOL, REVIEW_PROCEDURES_TOOL, REVIEW_BUNDLE_TOOL):
            assert "initial_read" not in tool["input_schema"]["properties"], (
                f"{tool['name']} asks for an opening read after the report "
                f"has already been decomposed"
            )

    def test_initial_read_arriving_as_the_summary_string(self):
        """`initial_read` holding the summary PROSE, rest of the read flat.

        One of the two shapes consistent with the live Gate 0 retry
        (`initial_read.summary:missing` with `initial_read` present at the top
        level and `summary` at neither). Before this branch the string was
        discarded and the whole call retried — a second reviewer-model generation on the
        most expensive call in the pipeline. Handled rather than diagnosed
        because the repair is the same either way: a string in that slot IS
        the summary, and reading it that way costs nothing if the model never
        does it again.
        """
        from app.services.reviewer.models import Gate0Recommendations

        out = Gate0Recommendations.model_validate({
            "initial_read": "A vendor report on a ClickFix campaign.",
            "actors": ["UNC0003"],
            "attack_chain": ["fake captcha", "RunMRU write"],
            "thin_areas": ["no PowerShell command shown"],
            "entities": [],
        })
        assert out.initial_read.summary == "A vendor report on a ClickFix campaign."
        assert out.initial_read.actors == ["UNC0003"]
        assert out.initial_read.attack_chain == ["fake captcha", "RunMRU write"]

    def test_the_leaked_parameter_wrapper_is_stripped(self):
        """Shaped like the live run that confirmed the string shape.

        The model's own tool-call serialization leaked into the value. The
        prose after it is fine, so the wrapper is stripped rather than the
        read discarded — this text is shown to the analyst AND replayed to
        every later gate, so markup in it is noise in both places.
        """
        from app.services.reviewer.models import Gate0Recommendations

        out = Gate0Recommendations.model_validate({
            "initial_read": (
                '<parameter name="summary">Example Vendor '
                "reports Campaign 00.001, attributed to UNC0003.</parameter>"
            ),
            "entities": [],
        })
        assert out.initial_read.summary.startswith("Example Vendor")
        assert "<parameter" not in out.initial_read.summary
        assert "</parameter>" not in out.initial_read.summary

    def test_a_clean_string_read_is_left_alone(self):
        from app.services.reviewer.models import Gate0Recommendations

        out = Gate0Recommendations.model_validate(
            {"initial_read": "A plain summary with no markup.", "entities": []},
        )
        assert out.initial_read.summary == "A plain summary with no markup."

    def test_a_string_initial_read_with_nothing_else_flat(self):
        from app.services.reviewer.models import Gate0Recommendations

        out = Gate0Recommendations.model_validate(
            {"initial_read": "Just the summary.", "entities": []},
        )
        assert out.initial_read.summary == "Just the summary."
        assert out.initial_read.actors == []

    def test_unrelated_extra_fields_are_still_rejected(self):
        """Tolerance is scoped to the known flattening, not a blanket allow."""
        import pytest as _pytest
        from app.services.reviewer.models import Gate0Recommendations

        with _pytest.raises(Exception):
            Gate0Recommendations.model_validate(self._base(made_up_field="x"))


# ── Gate 1: procedures + techniques ──────────────────────────────────

class TestDiffProcedureGate:
    """Gate 1's diff scores technique judgment separately from the verdict.

    "Approve this draft but drop T1105" is two claims. An analyst who keeps
    the draft while restoring the technique agreed with one and overrode the
    other — and the technique half is the more interesting signal, because
    technique mapping is what Gate 1 exists to get right.
    """

    def test_verdict_and_removals_are_scored_separately(self):
        from app.services.reviewer.outcomes import diff_procedure_gate

        payload = {"drafts": [{
            "draft_id": "d1", "action": "edit", "confidence": "high",
            "remove_technique_ids": ["T1105", "T1218"],
        }]}
        submitted = [{
            "draft_id": "d1", "action": "edit",
            # Analyst dropped T1105 but kept T1218.
            "analyst_edits": {"techniques": [{"technique_id": "T1218"}]},
        }]
        out = diff_procedure_gate(payload, submitted, [])
        by_ref = {i["ref"]: i for i in out["agreement"]["items"]}
        assert by_ref["d1"]["agreed"] is True
        assert by_ref["d1:T1105"]["agreed"] is True
        assert by_ref["d1:T1218"]["agreed"] is False
        assert out["agreement"]["total"] == 3

    def test_untouched_technique_list_counts_removals_as_overridden(self):
        """Approving without editing means nothing was removed."""
        from app.services.reviewer.outcomes import diff_procedure_gate

        payload = {"drafts": [{
            "draft_id": "d1", "action": "edit", "confidence": "high",
            "remove_technique_ids": ["T1105"],
        }]}
        out = diff_procedure_gate(
            payload, [{"draft_id": "d1", "action": "approve"}], [],
        )
        by_ref = {i["ref"]: i for i in out["agreement"]["items"]}
        assert by_ref["d1"]["agreed"] is False       # edit vs approve
        assert by_ref["d1:T1105"]["actual"] == "kept"
        assert by_ref["d1:T1105"]["agreed"] is False

    def test_silence_on_a_draft_counts_as_approve(self):
        from app.services.reviewer.outcomes import diff_procedure_gate

        payload = {"drafts": [
            {"draft_id": "d1", "action": "approve", "confidence": "high"},
        ]}
        out = diff_procedure_gate(payload, [], [])
        assert out["agreement"]["agreed"] == 1

    def test_promotion_taken_and_declined(self):
        from app.services.reviewer.outcomes import diff_procedure_gate

        payload = {"promotions": [
            {"chunk_id": "c1", "technique_id": "T1219", "confidence": "high"},
            {"chunk_id": "c1", "technique_id": "T1105", "confidence": "low"},
        ]}
        out = diff_procedure_gate(
            payload, [], [{"chunk_id": "c1", "technique_id": "T1219"}],
        )
        by_ref = {i["ref"]: i for i in out["agreement"]["items"]}
        assert by_ref["c1:T1219"]["agreed"] is True
        assert by_ref["c1:T1105"]["actual"] == "left_in_review"

    def test_no_recommendations_gives_no_rate(self):
        from app.services.reviewer.outcomes import diff_procedure_gate

        out = diff_procedure_gate({}, [{"draft_id": "d1", "action": "approve"}], [])
        assert out["agreement"]["total"] == 0
        assert out["agreement"]["rate"] is None


class TestGate1PayloadCarriesTheEvidence:
    """The reviewer's advantage at Gate 1 is evidence the picker never saw.

    If these signals stop reaching the prompt the reviewer degrades into a
    second, pricier technique picker — which is the failure mode this whole
    design is trying to avoid, and it would be invisible from the output.
    """

    def _state(self):
        return {
            "drafts": [{
                "draft_id": "d1", "chunk_id": "c1",
                "name": "Download payload via curl",
                "description": "The actor downloads a payload.",
                "confidence": 70,
            }],
            "chunks": [{
                "chunk_id": "c1", "text": "curl fetched cont.hta",
                "source_excerpt": "curl -s -L -o cont.hta",
            }],
            "technique_mappings": {"c1": [{
                "technique_id": "T1105", "technique_name": "Ingress Tool Transfer",
                "tactic": "command-and-control", "confidence": 0.9,
                "confidence_bucket": "definite", "provenance": "llm_pick",
                "source_quote": "curl -s -L -o cont.hta",
                "quote_unsupported_by_source": True, "quote_source_support": 0.27,
                "bucket_capped": "quote unsupported by report",
            }]},
            "technique_mappings_for_review": {"c1": [{
                "technique_id": "T1219", "technique_name": "Remote Access Tools",
                "tactic": "command-and-control", "confidence": 0.4,
                "confidence_bucket": "possible", "denylisted": True,
            }]},
            "proposals_by_chunk": {"c1": {"objective": "Stage the second payload"}},
            "classified_sections": [{
                "classification": "technique_reference",
                "text": "Techniques observed: T1105, T1204.004",
            }],
        }

    def test_carries_every_guardrail_signal(self):
        from app.services.reviewer.gate_reviewers import build_procedures_turn

        turn = build_procedures_turn(self._state())
        for signal in (
            "QUOTE UNSUPPORTED BY THE REPORT",   # grounding verdict
            "0.27",                              # the actual support score
            "auto-demoted",                      # the guardrail that fired
            "DENYLISTED",                        # analyst rule
            "HELD FOR REVIEW",                   # the promote lane
            "declared objective",                # what scopes the procedure
        ):
            assert signal in turn, f"{signal!r} missing from the Gate 1 prompt"

    def test_surfaces_the_reports_own_attack_table(self):
        """Corroboration the picker never had — from a section deliberately
        held out of the quote-grounding corpus."""
        from app.services.reviewer.gate_reviewers import build_procedures_turn

        turn = build_procedures_turn(self._state())
        assert "THE REPORT'S OWN ATT&CK TABLE" in turn
        assert "REPORT'S OWN TABLE lists this" in turn   # T1105 is in the table

    def test_missing_attack_table_is_not_fatal(self):
        from app.services.reviewer.gate_reviewers import build_procedures_turn

        state = self._state()
        state["classified_sections"] = []
        turn = build_procedures_turn(state)
        assert "DRAFT d1" in turn
        assert "THE REPORT'S OWN ATT&CK TABLE" not in turn

    def test_empty_state_does_not_raise(self):
        from app.services.reviewer.gate_reviewers import build_procedures_turn

        assert "GATE 1" in build_procedures_turn({})


# ── the analyst's pinned rules ───────────────────────────────────────

class TestPinnedRulesReachTheReviewer:
    """Confirmed analyst policy must reach the stage judging the extraction.

    Every extraction node injects pinned `promoted_to_prompt` rules; the
    reviewer injected none. A reviewer that has never heard of a rule the
    analyst promoted can recommend exactly what it forbids — making them
    re-make a correction the flywheel exists to capture once.

    Pinned ONLY. The advisory patterns are the extractor's soft priors, and
    withholding them is what keeps the reviewer an independent check rather
    than a second voice agreeing by construction.
    """

    def test_only_pinned_statuses_are_requested(self, monkeypatch):
        import app.services.feedback_patterns as fp

        seen: dict = {}

        async def fake_fetch(db, **kwargs):
            seen.update(kwargs)
            return []

        monkeypatch.setattr(fp, "fetch_feedback_patterns", fake_fetch)
        import asyncio
        asyncio.run(fp.pinned_rules_addendum(None, categories=("wrong_technique",)))

        assert seen["statuses"] == ("promoted_to_prompt",)
        # Permanent means permanent — a working rule that stops being
        # corrected must not age out of the reviewer's context.
        assert seen["max_age_days"] is None

    def test_no_categories_short_circuits(self, monkeypatch):
        import asyncio

        import app.services.feedback_patterns as fp

        called = []

        async def fake_fetch(db, **kwargs):
            called.append(1)
            return []

        monkeypatch.setattr(fp, "fetch_feedback_patterns", fake_fetch)
        assert asyncio.run(fp.pinned_rules_addendum(None, categories=())) == ""
        assert not called

    def test_failure_returns_empty_rather_than_blocking(self, monkeypatch):
        import asyncio

        import app.services.feedback_patterns as fp

        async def boom(db, **kwargs):
            raise RuntimeError("postgres is down")

        monkeypatch.setattr(fp, "fetch_feedback_patterns", boom)
        assert asyncio.run(
            fp.pinned_rules_addendum(None, categories=("wrong_technique",))
        ) == ""

    @pytest.mark.asyncio
    async def test_run_reviewer_puts_them_in_the_system_prompt(self, monkeypatch):
        """The isolated-helper tests above pass even if the runner never uses
        it. This is the one that fails when the injection is removed."""
        from app.services.reviewer import runner

        captured: dict = {}

        class FakeResponse:
            tool_output = {"entities": [], "added_entities": []}
            # No output_model was used, so the runner falls back to the raw
            # dict — see the payload-source test below for the path that
            # matters.
            validated = None
            input_tokens = out = output_tokens = 0
            cached = False
            model = "claude-opus-5"

        async def fake_call_llm(**kwargs):
            captured["system"] = kwargs["system"]
            return FakeResponse()

        async def fake_rules(db, *, categories):
            captured["categories"] = categories
            return "\n\nPERMANENT RULES (analyst-confirmed — always apply):\n- MARKER"

        class FakeRow:
            id = uuid.uuid4()

        monkeypatch.setattr(runner, "call_llm", fake_call_llm)
        monkeypatch.setattr(runner, "pinned_rules_addendum", fake_rules)
        monkeypatch.setattr(runner, "async_session", _fake_session)
        monkeypatch.setattr(runner.store, "load_turns", _async_return([]))
        monkeypatch.setattr(runner.store, "get_initial_read", _async_return(None))
        monkeypatch.setattr(runner.store, "record_turn", _async_return(FakeRow()))

        await runner.run_reviewer(uuid.uuid4(), "entities", {"parsed_text": "x"})

        assert "MARKER" in captured["system"], (
            "pinned rules were fetched but never reached the model"
        )
        assert "senior CTI analyst" in captured["system"], "base prompt lost"
        # The gate's own categories, not a global dump.
        assert "brand_as_malware" in captured["categories"]

    def test_every_reviewer_declares_categories(self):
        from app.services.reviewer.gate_reviewers import REVIEWERS

        for key, reviewer in REVIEWERS.items():
            assert reviewer.feedback_categories, (
                f"gate '{key}' declares no feedback categories, so the "
                f"analyst's pinned rules would never reach it"
            )


# ── override feedback ────────────────────────────────────────────────

class TestOutcomeFeedback:
    """The system prompt promises the reviewer it will be told about
    overrides. This is the code that makes that true.

    Before it, `turn.outcome` was written for measurement and never read
    back: the model was told to expect a signal it never received, and on a
    gate that loops it could repeat advice the analyst had just rejected.
    """

    def _outcome(self, items, total=None, agreed=None):
        agreed_n = agreed if agreed is not None else sum(1 for i in items if i["agreed"])
        return {"agreement": {
            "total": total if total is not None else len(items),
            "agreed": agreed_n, "items": items,
        }}

    def test_names_the_overrides(self):
        from app.services.reviewer.runner import format_outcome_feedback

        text = format_outcome_feedback(self._outcome([
            {"ref": "dft-1", "recommended": "edit", "actual": "approve", "agreed": False},
            {"ref": "dft-2", "recommended": "approve", "actual": "approve", "agreed": True},
        ]))
        assert "dft-1" in text
        assert "you said edit, they chose approve" in text

    def test_omits_the_agreements(self):
        """Agreement is the default; listing it buries the two lines that
        actually carry information."""
        from app.services.reviewer.runner import format_outcome_feedback

        text = format_outcome_feedback(self._outcome([
            {"ref": "dft-1", "recommended": "edit", "actual": "approve", "agreed": False},
            {"ref": "dft-AGREED", "recommended": "approve", "actual": "approve", "agreed": True},
        ]))
        assert "dft-AGREED" not in text

    def test_silent_when_nothing_was_overridden(self):
        from app.services.reviewer.runner import format_outcome_feedback

        assert format_outcome_feedback(self._outcome([
            {"ref": "d1", "recommended": "approve", "actual": "approve", "agreed": True},
        ])) == ""

    def test_moot_recommendations_are_not_reported_as_overrides(self):
        """Advice the analyst never reached is not advice they rejected.

        A chunk-gate re-chunk discards the decisions, adds and edges channels
        entirely, so every recommendation in them is marked `moot` and not
        agreed. Both are true; only the first is informative. Without this
        filter the reviewer is told "you said drop chk-a, they chose
        moot_rerun" — an override it never actually lost.
        """
        from app.services.reviewer.runner import format_outcome_feedback

        text = format_outcome_feedback(self._outcome([
            {"ref": "chk-a", "recommended": "drop", "actual": "moot_rerun",
             "agreed": False, "moot": True},
        ], total=0, agreed=0))
        assert text == ""

    def test_a_real_override_alongside_a_moot_one_still_reports(self):
        from app.services.reviewer.runner import format_outcome_feedback

        text = format_outcome_feedback(self._outcome([
            {"ref": "chk-a", "recommended": "drop", "actual": "moot_rerun",
             "agreed": False, "moot": True},
            {"ref": "chk-b", "recommended": "drop", "actual": "approve",
             "agreed": False},
        ], total=1, agreed=0))
        assert "chk-b" in text
        assert "chk-a" not in text

    def test_silent_when_not_yet_submitted(self):
        from app.services.reviewer.runner import format_outcome_feedback

        assert format_outcome_feedback(None) == ""
        assert format_outcome_feedback({}) == ""

    @pytest.mark.asyncio
    async def test_replayed_as_a_user_turn_after_the_agent_turn(self, monkeypatch):
        """A USER turn, not an assistant one — the analyst's verdict is not
        something the assistant said."""
        from app.services.reviewer import runner

        class FakeTurn:
            gate_key = "entities"
            pass_number = 1
            agent_notes = "- remove e1 (high): looked wrong"
            outcome = {"agreement": {"total": 1, "agreed": 0, "items": [
                {"ref": "e1", "recommended": "remove", "actual": "approve", "agreed": False},
            ]}}

        monkeypatch.setattr(runner.store, "load_turns", _async_return([FakeTurn()]))
        messages = await runner._build_messages(
            None, uuid.uuid4(), {"parsed_text": "a report"}, "GATE 1 ...",
        )
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant", "user", "user"]
        assert "they chose approve" in messages[2]["content"]

    @pytest.mark.asyncio
    async def test_no_outcome_means_no_extra_turn(self, monkeypatch):
        from app.services.reviewer import runner

        class FakeTurn:
            gate_key = "entities"
            pass_number = 1
            agent_notes = "notes"
            outcome = None

        monkeypatch.setattr(runner.store, "load_turns", _async_return([FakeTurn()]))
        messages = await runner._build_messages(
            None, uuid.uuid4(), {"parsed_text": "a report"}, "GATE 1 ...",
        )
        assert [m["role"] for m in messages] == ["user", "assistant", "user"]


class TestReportTurnIsNotCached:
    """The cache breakpoint was removed after a real run measured it negative.

    A two-call experiment said 32% saving; a three-gate run said cache_rd was
    zero at every later gate, including one 54s after the previous. The prefix
    is matched over tools -> system -> messages and both earlier parts differ
    per gate, so a breakpoint in messages is unreachable across gates --
    leaving a 1.25x write nothing ever reads.

    Pinned so nobody re-adds it from the same reasonable-sounding intuition
    without re-measuring.
    """

    @pytest.mark.asyncio
    async def test_report_turn_is_plain_text(self, monkeypatch):
        from app.services.reviewer import runner

        monkeypatch.setattr(runner.store, "load_turns", _async_return([]))
        messages = await runner._build_messages(
            None, uuid.uuid4(), {"parsed_text": "a report"}, "GATE 0 ...",
        )
        content = messages[0]["content"]
        assert isinstance(content, str), (
            "report turn is a content-block list again -- if that is to carry "
            "cache_control, re-measure cache_rd across gates first"
        )
        assert "a report" in content


class TestCacheTokenAccounting:
    """Prompt caching must not make a call LOOK cheaper than it is.

    The Anthropic API reports cached prompt tokens in separate fields;
    `usage.input_tokens` counts only the uncached remainder. When the report
    turn became a cache breakpoint, a Gate 0 review's reported input dropped
    from 21,492 to 1,407 — not a 15x saving, just ~20k of prompt moving into
    fields nothing recorded.

    A token log that under-reports is worse than none: it invites exactly the
    wrong conclusion about what a feature costs.
    """

    class _Usage:
        def __init__(self, **kw):
            self.input_tokens = kw.get("input_tokens", 0)
            self.output_tokens = kw.get("output_tokens", 0)
            for k in ("cache_creation_input_tokens", "cache_read_input_tokens"):
                if k in kw:
                    setattr(self, k, kw[k])

    def test_response_carries_both_cache_fields(self):
        from app.nodes.llm.llm_adapter import LLMResponse

        r = LLMResponse(input_tokens=1407, cache_creation_tokens=19000)
        assert r.cache_creation_tokens == 19000
        assert r.cache_read_tokens == 0

    def test_missing_fields_do_not_break_older_models(self):
        """Models without prompt caching have no such attributes at all."""
        usage = self._Usage(input_tokens=100, output_tokens=50)
        assert getattr(usage, "cache_creation_input_tokens", 0) or 0 == 0
        assert getattr(usage, "cache_read_input_tokens", 0) or 0 == 0

    def test_log_suffix_appears_only_when_caching_happened(self):
        from app.nodes.llm.llm_adapter import _fmt_cache

        assert _fmt_cache(0, 0) == ""
        assert "cache_write=19000" in _fmt_cache(19000, 0)
        assert "cache_read=19000" in _fmt_cache(0, 19000)

    @pytest.mark.asyncio
    async def test_reviewer_records_all_three_counts(self, monkeypatch):
        """A stored _usage missing the cache fields is the blind instrument."""
        from app.services.reviewer import runner

        class FakeResponse:
            tool_output = {"entities": [], "added_entities": []}
            # No output_model was used, so the runner falls back to the raw
            # dict — see the payload-source test below for the path that
            # matters.
            validated = None
            input_tokens = 1407
            cache_creation_tokens = 19000
            cache_read_tokens = 0
            output_tokens = 5415
            cached = False
            model = "claude-opus-5"

        captured: dict = {}

        async def fake_call_llm(**kwargs):
            return FakeResponse()

        async def fake_record_turn(db, sid, gate, *, payload, **kw):
            captured["payload"] = payload

            class Row:
                id = uuid.uuid4()
            return Row()

        monkeypatch.setattr(runner, "call_llm", fake_call_llm)
        monkeypatch.setattr(runner, "pinned_rules_addendum", _async_return(""))
        monkeypatch.setattr(runner, "async_session", _fake_session)
        monkeypatch.setattr(runner.store, "load_turns", _async_return([]))
        monkeypatch.setattr(runner.store, "get_initial_read", _async_return(None))
        monkeypatch.setattr(runner.store, "record_turn", fake_record_turn)

        await runner.run_reviewer(uuid.uuid4(), "entities", {"parsed_text": "x"})
        usage = captured["payload"]["_usage"]
        assert usage["input_tokens"] == 1407
        assert usage["cache_creation_tokens"] == 19000, (
            "cache tokens missing — the cost of this call would be invisible"
        )
        assert usage["cache_read_tokens"] == 0


class TestPipelineTaskCancellation:
    """Deleting a source must stop the run that was writing to it.

    Without this the detached task keeps going against a row that no longer
    exists and dies on `StaleDataError: UPDATE ... expected to update 1
    row(s); 0 were matched`. Harmless, but it reads like a crash until you
    trace the thread id — which happened repeatedly while building this.
    """

    @pytest.mark.asyncio
    async def test_cancels_the_matching_task_only(self):
        from app.api.routes.pipeline import (
            cancel_pipeline_task,
            launch_pipeline_task,
        )

        mine, theirs = uuid.uuid4(), uuid.uuid4()

        async def forever():
            await asyncio.sleep(3600)

        t_mine = launch_pipeline_task(forever(), mine)
        t_theirs = launch_pipeline_task(forever(), theirs)
        await asyncio.sleep(0)  # let both start

        assert cancel_pipeline_task(mine) is True
        await asyncio.sleep(0)
        assert t_mine.cancelled() or t_mine.cancelling()
        assert not t_theirs.done(), "canceled an unrelated source's run"

        t_theirs.cancel()
        for t in (t_mine, t_theirs):
            with contextlib.suppress(asyncio.CancelledError):
                await t

    @pytest.mark.asyncio
    async def test_unknown_source_is_a_no_op(self):
        from app.api.routes.pipeline import cancel_pipeline_task

        assert cancel_pipeline_task(uuid.uuid4()) is False

    def test_cancellation_escapes_the_runners_error_handler(self):
        """The property that makes canceling safe.

        `stream_with_sync` catches `Exception` and marks the source failed.
        `CancelledError` derives from `BaseException`, so it slips past —
        a canceled run does not leave a 'failed' row behind on its way out.
        If that ever changed, deleting a source mid-run would resurrect it
        in the Failed column.
        """
        assert issubclass(asyncio.CancelledError, BaseException)
        assert not issubclass(asyncio.CancelledError, Exception)

    @pytest.mark.asyncio
    async def test_delete_source_actually_calls_the_cancel(self, monkeypatch):
        """The wiring, not just the mechanism.

        `cancel_pipeline_task` working proves nothing if `delete_source`
        never calls it. Covered here rather than live because the event loop
        is blocked for minutes while SecureBERT encodes the technique
        catalogue — a DELETE issued during that window never reaches the
        handler at all, which is exactly what happened when this was tried
        against a real run.
        """
        import app.services.queue as queue_module
        from app.api.routes import pipeline as pipeline_module

        canceled: list = []
        monkeypatch.setattr(
            pipeline_module, "cancel_pipeline_task",
            lambda sid: canceled.append(sid) or True,
        )

        sid = uuid.uuid4()
        src = SimpleNamespace(
            id=sid, thread_id=None, raw_content_path="/nonexistent/x.pdf",
        )

        async def fake_get_source(db, _sid):
            return src

        db = AsyncMock()
        db.delete = AsyncMock()
        db.commit = AsyncMock()
        monkeypatch.setattr(queue_module, "get_source", fake_get_source)
        monkeypatch.setattr(queue_module.figure_stash, "clear", lambda _s: None)
        monkeypatch.setattr(
            queue_module.reviewer_store, "delete_for_source", _async_return(0),
        )

        assert await queue_module.delete_source(db, sid) is True
        assert canceled == [sid], "delete_source did not cancel the in-flight run"


# ── Gate 2: bundle relationships ─────────────────────────────────────

class TestBundleTurnScope:
    """What the bundle gate shows, and what it deliberately withholds."""

    def _state(self):
        return {"relationship_preview": [
            # inherent — already judged at Gate 1
            {"id": "relp_1", "relationship_type": "uses",
             "source_name": "Download payload", "target_name": "T1105 Ingress Tool Transfer",
             "source_type": "x-procedure", "target_type": "attack-pattern",
             "reviewable": False},
            {"id": "relp_2", "relationship_type": "uses",
             "source_name": "Download payload", "target_name": "curl",
             "source_type": "x-procedure", "target_type": "tool", "reviewable": True},
            {"id": "relp_3", "relationship_type": "targets",
             "source_name": "Deploy NETSUPPORT", "target_name": "Acme Corp",
             "source_type": "x-procedure", "target_type": "identity", "reviewable": True},
        ]}

    def test_withholds_inherent_technique_mappings(self):
        """Showing them again invites the reviewer to relitigate a decision
        the analyst already ruled on at Gate 1 — and they are the largest
        group in a real bundle, so it is not a small cost."""
        from app.services.reviewer.gate_reviewers import build_bundle_turn

        turn = build_bundle_turn(self._state())
        assert "T1105" not in turn, "inherent technique mapping leaked into the prompt"
        assert "relp_1" not in turn

    def test_states_what_was_withheld(self):
        """Silently filtering would leave the reviewer unable to tell a
        withheld edge from a missing one."""
        from app.services.reviewer.gate_reviewers import build_bundle_turn

        turn = build_bundle_turn(self._state())
        assert "technique mapping" in turn
        assert "not shown" in turn

    def test_shows_every_reviewable_relationship(self):
        from app.services.reviewer.gate_reviewers import build_bundle_turn

        turn = build_bundle_turn(self._state())
        assert "relp_2" in turn and "relp_3" in turn
        assert "curl" in turn and "Acme Corp" in turn

    def test_groups_by_source_procedure(self):
        """~87 flat rows is the shape that makes an analyst skim; a reviewer
        reading a flat list has the same problem."""
        from app.services.reviewer.gate_reviewers import build_bundle_turn

        turn = build_bundle_turn(self._state())
        assert "Download payload  [x-procedure]" in turn
        assert "Deploy NETSUPPORT  [x-procedure]" in turn

    def test_singular_grammar_for_one_withheld(self):
        from app.services.reviewer.gate_reviewers import build_bundle_turn

        turn = build_bundle_turn(self._state())
        assert "is a technique mapping" in turn

    def test_no_reviewable_relationships_is_not_an_error(self):
        from app.services.reviewer.gate_reviewers import build_bundle_turn

        state = {"relationship_preview": [
            {"id": "relp_1", "reviewable": False, "relationship_type": "uses",
             "source_name": "p", "target_name": "t"},
        ]}
        assert "no reviewable relationships" in build_bundle_turn(state)

    def test_empty_state_does_not_raise(self):
        from app.services.reviewer.gate_reviewers import build_bundle_turn

        assert "GATE 2" in build_bundle_turn({})


class TestDiffBundleGate:
    def test_agreement_and_override(self):
        from app.services.reviewer.outcomes import diff_bundle_gate

        payload = {"relationships": [
            {"rel_id": "relp_2", "action": "remove", "confidence": "high"},
            {"rel_id": "relp_3", "action": "edit", "confidence": "medium"},
        ]}
        out = diff_bundle_gate(payload, [{"rel_id": "relp_2", "action": "remove"}])
        by_ref = {i["ref"]: i for i in out["agreement"]["items"]}
        assert by_ref["relp_2"]["agreed"] is True
        # Unmentioned ships as derived, which is the same as approve — so an
        # `edit` the analyst ignored is an override.
        assert by_ref["relp_3"]["actual"] == "approve"
        assert by_ref["relp_3"]["agreed"] is False

    def test_no_recommendations_gives_no_rate(self):
        from app.services.reviewer.outcomes import diff_bundle_gate

        out = diff_bundle_gate({}, [{"rel_id": "relp_1", "action": "remove"}])
        assert out["agreement"]["total"] == 0
        assert out["agreement"]["rate"] is None

    def test_signature_matches_the_other_differs(self):
        """All three take (payload, reviews, extras) so the dispatch in
        record_gate_outcome needs no special case."""
        import inspect

        from app.services.reviewer.outcomes import OUTCOME_DIFFERS

        for key, fn in OUTCOME_DIFFERS.items():
            params = list(inspect.signature(fn).parameters)
            assert len(params) == 3, f"{key} differ takes {len(params)} args"


# ── chunk gate: the decomposition itself ─────────────────────────────

class TestChunksTurnScope:
    """What the chunk-gate reviewer is shown, and what it is told is missing.

    The two omissions are deliberate and stated in the prompt: there is no
    split action, and the operator/condition geometry is settled after the
    reviewer rather than by it. Both are cases where saying nothing would let
    silence read as approval.
    """

    def _state(self, **over):
        base = {
            "is_sequential": True,
            "sequentiality_rationale": "the report narrates 'the actor then'",
            "chunks": [
                {"chunk_id": "chk-a", "sequence_index": 1, "text": "Phishing email sent.",
                 "source_excerpt": "victims received an email", "source_provenance": "prose",
                 "behavioral_confidence": 0.9, "precedes_ids": ["chk-b"]},
                {"chunk_id": "chk-b", "sequence_index": 2, "text": "PowerShell ran.",
                 "source_excerpt": "powershell -enc", "source_provenance": "paraphrased",
                 "behavioral_confidence": 0.7, "precedes_ids": []},
            ],
        }
        base.update(over)
        return base

    def test_shows_every_chunk_with_its_flow(self):
        from app.services.reviewer.gate_reviewers import build_chunks_turn

        turn = build_chunks_turn(self._state())
        assert "chk-a" in turn and "chk-b" in turn
        assert "Phishing email sent." in turn
        assert "precedes: chk-b" in turn

    def test_flags_a_chunk_with_no_verbatim_anchor(self):
        """The analogue of an unsupported technique quote.

        `source_provenance: paraphrased` means nothing in the report matched
        the excerpt literally, so the chunk's evidence has been through the
        model's own wording. It is where invented detail enters, and the
        reviewer cannot compute it — it is derived from where source_span
        landed, which the reviewer never sees.
        """
        from app.services.reviewer.gate_reviewers import build_chunks_turn

        turn = build_chunks_turn(self._state())
        assert "NO VERBATIM ANCHOR" in turn
        # ...and only on the chunk that has the problem.
        assert turn.count("NO VERBATIM ANCHOR") == 1

    def test_says_there_is_no_split(self):
        from app.services.reviewer.gate_reviewers import build_chunks_turn

        turn = build_chunks_turn(self._state())
        assert "CANNOT split" in turn
        assert "under_chunked" in turn, (
            "the reviewer is told it cannot split but not what to do "
            "instead, which leaves the finding unexpressible"
        )

    def test_says_operators_and_conditions_are_out_of_scope(self):
        from app.services.reviewer.gate_reviewers import build_chunks_turn

        turn = build_chunks_turn(self._state())
        assert "Not yours at this gate" in turn
        assert "AND/OR/XOR" in turn

    def test_non_sequential_sources_are_told_not_to_tidy_the_graph(self):
        """A catalogue source SHOULD have disconnected chunks.

        Without this the reviewer sees a graph in pieces, reads it as a
        defect, and proposes edges the pipeline deliberately suppressed —
        manufacturing exactly the fictional sequencing the sequentiality
        flag exists to prevent.
        """
        from app.services.reviewer.gate_reviewers import build_chunks_turn

        turn = build_chunks_turn(self._state(is_sequential=False))
        assert "NON-SEQUENTIAL" in turn
        assert "Do not add edges" in turn
        assert "correct output" in turn.lower()

    def test_empty_state_does_not_raise(self):
        from app.services.reviewer.gate_reviewers import build_chunks_turn

        turn = build_chunks_turn({})
        assert "the chunker produced nothing" in turn


class TestChunkFlowSummary:
    """The DAG shape, computed rather than left to the model to derive.

    Each chunk carries only its own forward edges, so "unreachable" and "in
    three pieces" are properties of the whole set. A model asked to hold N
    adjacency lists in its head and derive them will sometimes be wrong, and
    being right here costs nothing.
    """

    def _summary(self, chunks):
        from app.services.reviewer.gate_reviewers import _chunk_flow_summary

        return "\n".join(_chunk_flow_summary(chunks))

    def test_counts_pieces_and_edges(self):
        out = self._summary([
            {"chunk_id": "a", "sequence_index": 1, "precedes_ids": ["b"]},
            {"chunk_id": "b", "sequence_index": 2, "precedes_ids": []},
            {"chunk_id": "z", "sequence_index": 3, "precedes_ids": []},
        ])
        assert "3 chunks, 1 precedes edges, 2 connected pieces" in out

    def test_the_first_chunk_is_not_reported_as_an_undeclared_root(self):
        """The primary chain's start is never marked chain_root.

        `_finalize_chunks` leaves chunk 1 implicit and only marks ADDITIONAL
        chains, so counting it as undeclared would flag the one entry point
        every single source legitimately has — noise on every run, which is
        how a real warning gets ignored.
        """
        out = self._summary([
            {"chunk_id": "a", "sequence_index": 1, "precedes_ids": ["b"]},
            {"chunk_id": "b", "sequence_index": 2, "precedes_ids": []},
            {"chunk_id": "orphan", "sequence_index": 3, "precedes_ids": []},
        ])
        assert "being marked as a new chain: orphan" in out
        assert "a," not in out.split("new chain:")[-1]

    def test_a_declared_second_chain_is_not_flagged(self):
        out = self._summary([
            {"chunk_id": "a", "sequence_index": 1, "precedes_ids": ["b"]},
            {"chunk_id": "b", "sequence_index": 2, "precedes_ids": []},
            {"chunk_id": "veeam", "sequence_index": 3, "precedes_ids": [],
             "chain_root": True, "chain_label": "Veeam intrusion"},
        ])
        assert "new chain" not in out

    def test_a_single_connected_chain_raises_nothing(self):
        out = self._summary([
            {"chunk_id": "a", "sequence_index": 1, "precedes_ids": ["b"]},
            {"chunk_id": "b", "sequence_index": 2, "precedes_ids": []},
        ])
        assert "1 connected piece." in out
        assert "new chain" not in out

    def test_a_shared_segment_is_named_with_its_entry_points(self):
        """Two lures converge on a kit: the summary names the kit as a
        shared entry so the reviewer sees the hourglass without deriving it
        from fifteen adjacency lists."""
        out = self._summary([
            {"chunk_id": "lure-a", "sequence_index": 1, "precedes_ids": ["kit"], "chain_root": True},
            {"chunk_id": "lure-b", "sequence_index": 2, "precedes_ids": ["kit"], "chain_root": True},
            {"chunk_id": "kit", "sequence_index": 3, "precedes_ids": ["tail-a", "tail-b"]},
            {"chunk_id": "tail-a", "sequence_index": 4, "precedes_ids": []},
            {"chunk_id": "tail-b", "sequence_index": 5, "precedes_ids": []},
        ])
        assert "Shared segments" in out
        assert "kit (entered from lure-a, lure-b)" in out

    def test_a_plain_convergence_inside_one_chain_is_not_called_shared(self):
        out = self._summary([
            {"chunk_id": "a", "sequence_index": 1, "precedes_ids": ["b", "c"]},
            {"chunk_id": "b", "sequence_index": 2, "precedes_ids": ["d"]},
            {"chunk_id": "c", "sequence_index": 3, "precedes_ids": ["d"]},
            {"chunk_id": "d", "sequence_index": 4, "precedes_ids": []},
        ])
        assert "Shared segments" not in out

    def test_tactic_order_warnings_are_listed(self):
        out = self._summary([
            {"chunk_id": "kit", "sequence_index": 1, "precedes_ids": ["lure"]},
            {"chunk_id": "lure", "sequence_index": 2, "precedes_ids": [],
             "flow_warnings": ["kit (execution) precedes lure (initial-access): later tactic before earlier one"]},
        ])
        assert "Tactic-order warnings" in out
        assert "lure" in out

    def test_edges_to_chunks_that_do_not_exist_are_ignored(self):
        """A dangling precedes_id is tolerated downstream, so counting it
        here would report an edge the canvas does not draw."""
        out = self._summary([
            {"chunk_id": "a", "sequence_index": 1, "precedes_ids": ["gone"]},
        ])
        assert "1 chunks, 0 precedes edges, 1 connected piece" in out


class TestDiffChunkGate:
    def _payload(self, **over):
        base = {
            "chunks": [
                {"chunk_id": "chk-a", "action": "drop", "confidence": "high"},
                {"chunk_id": "chk-b", "action": "approve", "confidence": "high"},
            ],
        }
        base.update(over)
        return base

    def test_agreement_and_override(self):
        from app.services.reviewer.outcomes import diff_chunk_gate

        out = diff_chunk_gate(
            self._payload(),
            [{"chunk_id": "chk-a", "action": "drop"}],
            {},
        )
        agreement = out["agreement"]
        assert out["gate_key"] == "chunks"
        # chk-a dropped as advised; chk-b unmentioned, which the gate treats
        # as approve — the action the reviewer recommended.
        assert agreement["agreed"] == 2
        assert agreement["overridden"] == 0

    def test_silence_on_a_chunk_counts_as_approve(self):
        from app.services.reviewer.outcomes import diff_chunk_gate

        out = diff_chunk_gate(self._payload(), [], {})
        by_ref = {i["ref"]: i for i in out["agreement"]["items"]}
        assert by_ref["chk-a"]["actual"] == "approve"
        assert by_ref["chk-a"]["agreed"] is False   # it wanted a drop
        assert by_ref["chk-b"]["agreed"] is True

    def test_edge_recommendations_are_scored(self):
        from app.services.reviewer.outcomes import diff_chunk_gate

        out = diff_chunk_gate(
            {"edges": [
                {"action": "add", "from_chunk_id": "a", "to_chunk_id": "b",
                 "confidence": "high"},
                {"action": "remove", "from_chunk_id": "c", "to_chunk_id": "d",
                 "confidence": "medium"},
            ]},
            [],
            {"edges": [{"action": "add", "from": "a", "to": "b"}]},
        )
        items = {i["ref"]: i for i in out["agreement"]["items"]}
        assert items["add a->b"]["agreed"] is True
        assert items["remove c->d"]["agreed"] is False

    def test_an_addition_is_matched_on_its_text(self):
        from app.services.reviewer.outcomes import diff_chunk_gate

        out = diff_chunk_gate(
            {"added_chunks": [{"text": "Shadow copies deleted.", "confidence": "high"}]},
            [],
            {"added_chunks": [{"text": "  shadow copies deleted.  "}]},
        )
        assert out["agreement"]["agreed"] == 1

    def test_a_rerun_moots_the_rest_rather_than_overriding_it(self):
        """A reject makes the other channels unreachable.

        gate_chunks ignores decisions, adds and edges entirely when `reject`
        is set. Scoring them as overrides would report a collapse in
        agreement caused by advice the analyst never got to act on; scoring
        them as agreement would invent consent. Neither is true, so they are
        excluded from the rate and marked.
        """
        from app.services.reviewer.outcomes import diff_chunk_gate

        out = diff_chunk_gate(
            self._payload(),
            [],
            {"reject": {"reason": "under_chunked", "comments": ""}},
        )
        agreement = out["agreement"]
        assert agreement["moot"] == 2
        # Only the unsolicited-rerun record is scored, and it is an override:
        # the reviewer did not ask for one.
        assert agreement["total"] == 1
        assert agreement["agreed"] == 0
        assert [i["kind"] for i in agreement["items"] if not i.get("moot")] == ["rerun"]

    def test_a_rerun_the_reviewer_asked_for_is_agreement(self):
        from app.services.reviewer.outcomes import diff_chunk_gate

        out = diff_chunk_gate(
            {"reject": {"reason": "over_chunked", "confidence": "high"}},
            [],
            {"reject": {"reason": "bad_flow", "comments": "x"}},
        )
        item = out["agreement"]["items"][0]
        assert item["kind"] == "rerun"
        # Scored on the ACTION, not the stated reason: the reason is a hint
        # to the chunker, not a claim the reviewer made about the source.
        assert item["agreed"] is True

    def test_a_rerun_the_analyst_declined_is_an_override(self):
        """The expensive recommendation the analyst refused.

        This is the one most worth capturing: a reviewer that asks to throw
        away a good pass, and an analyst who says no, is precisely the
        evidence that decides whether this gate can ever run unattended.
        """
        from app.services.reviewer.outcomes import diff_chunk_gate

        out = diff_chunk_gate(
            {"reject": {"reason": "over_chunked", "confidence": "high"}}, [], {},
        )
        item = out["agreement"]["items"][0]
        assert item["kind"] == "rerun"
        assert item["actual"] == "kept_this_pass"
        assert item["agreed"] is False

    def test_an_unsolicited_rerun_is_recorded_even_with_no_recommendations(self):
        """The analyst re-chunking unprompted is the single most informative
        thing that can happen at this gate, and recording only what the
        reviewer said would make it invisible."""
        from app.services.reviewer.outcomes import diff_chunk_gate

        out = diff_chunk_gate({}, [], {"reject": {"reason": "bad_flow"}})
        assert [i["kind"] for i in out["agreement"]["items"]] == ["rerun"]
        assert out["agreement"]["items"][0]["recommended"] == "none"

    def test_no_recommendations_gives_no_rate(self):
        from app.services.reviewer.outcomes import diff_chunk_gate

        out = diff_chunk_gate({}, [], {})
        assert out["agreement"]["rate"] is None

    def test_tolerates_a_list_for_extras(self):
        """The uniform differ signature passes `extras` positionally, and
        only this gate sends a dict. A list must degrade, not raise."""
        from app.services.reviewer.outcomes import diff_chunk_gate

        out = diff_chunk_gate(self._payload(), [], [])
        assert out["agreement"]["total"] == 2


class TestRejectRecommendationIsGrounded:
    """The re-chunk ask carries an evidence quote like everything else, but
    arrives as a single object rather than a list — so the grounding walk
    has to reach it explicitly. An ungrounded 'throw the pass away' is the
    most expensive unverified claim the reviewer can make."""

    def test_an_unsupported_quote_downgrades_the_rerun(self):
        from app.services.reviewer.runner import apply_grounding

        payload = {"reject": {
            "reason": "bad_flow", "confidence": "high",
            "evidence_quote": "wholly invented sentence about kerberoasting",
        }}
        downgraded = apply_grounding(
            payload, {"the", "report", "describes", "phishing", "only"},
        )
        assert downgraded == 1
        assert payload["reject"]["confidence"] == "low"
        assert payload["reject"]["quote_unsupported"] is True

    def test_a_supported_quote_survives(self):
        from app.services.reviewer.runner import apply_grounding

        payload = {"reject": {
            "reason": "bad_flow", "confidence": "high",
            "evidence_quote": "the actor deployed ransomware across the estate",
        }}
        apply_grounding(payload, {
            "actor", "deployed", "ransomware", "across", "estate", "the",
        })
        assert payload["reject"]["confidence"] == "high"
        # Only written when the check fails; the model's own default is False.
        assert not payload["reject"].get("quote_unsupported")
        assert payload["reject"]["quote_source_support"] >= 0.5

    def test_chunk_recommendations_are_grounded_too(self):
        from app.services.reviewer.runner import apply_grounding

        payload = {"chunks": [{
            "chunk_id": "chk-a", "action": "drop", "confidence": "high",
            "evidence_quote": "entirely fabricated supporting sentence here",
        }]}
        assert apply_grounding(payload, {"unrelated", "report", "words"}) == 1
        assert payload["chunks"][0]["confidence"] == "low"


class TestChunkTurnSummary:
    """What the chunk gate leaves behind in the transcript for later gates."""

    def test_names_the_chunk_and_its_verb(self):
        from app.services.reviewer.runner import _summarise_turn

        text = _summarise_turn({"chunks": [
            {"chunk_id": "chk-a", "action": "drop", "confidence": "high",
             "rationale": "vendor detection advice, not a procedure"},
        ]})
        assert "drop chk-a" in text
        assert "vendor detection advice" in text

    def test_renders_an_edge_as_a_pair(self):
        from app.services.reviewer.runner import _summarise_turn

        text = _summarise_turn({"edges": [
            {"action": "remove", "from_chunk_id": "chk-a", "to_chunk_id": "chk-b",
             "confidence": "medium", "rationale": "the report states no order"},
        ]})
        assert "remove chk-a -> chk-b" in text

    def test_a_merge_names_what_it_absorbs(self):
        from app.services.reviewer.runner import _summarise_turn

        text = _summarise_turn({"chunks": [
            {"chunk_id": "chk-a", "action": "merge", "merge_with": ["chk-b"],
             "confidence": "high", "rationale": "one objective"},
        ]})
        assert "absorbing chk-b" in text

    def test_the_rerun_ask_survives_into_the_transcript(self):
        from app.services.reviewer.runner import _summarise_turn

        text = _summarise_turn({"reject": {
            "reason": "under_chunked", "confidence": "high",
            "rationale": "every chunk covers two objectives",
            "comments": "split on the objective, not the paragraph",
        }})
        assert "re-chunk the whole source [under_chunked]" in text
        assert "split on the objective" in text

    def test_a_promotion_still_reports_its_technique_not_its_chunk(self):
        """Regression: adding chunk_id to the subject lookup must not steal
        promotions, which carry both ids."""
        from app.services.reviewer.runner import _summarise_turn

        text = _summarise_turn({"promotions": [
            {"chunk_id": "chk-a", "technique_id": "T1059.001",
             "confidence": "high", "rationale": "the report names powershell"},
        ]})
        assert "promote T1059.001" in text


class TestEveryReviewerListIsGrounded:
    """Every list of recommendations must face the grounding check.

    Found live on the Phase 4 run: the bundle gate logged "no recommendations"
    while its differ scored 18. Both read the same payload, so one was looking
    in the wrong place — `_GROUNDED_LISTS` had never gained "relationships"
    when Gate 2 shipped. The stored row confirmed it: 18 recommendations, 18
    carrying an evidence quote, 0 with a support score.

    That is not cosmetic. An unscored quote leaves `quote_unsupported` unset,
    so a fabricated one keeps whatever confidence the model claimed — and
    "high" is precisely what the analyst's bulk-accept takes without opening.
    The check that exists to stop the reviewer inventing evidence was skipping
    a whole gate, silently, for one release.

    A per-gate test would not have caught it, because the omission is between
    a gate's model and a constant in another module. Hence this: derive the
    fields from the models, so a new gate cannot ship ungrounded.
    """

    def test_no_recommendation_list_escapes_the_grounding_walk(self):
        import typing

        from app.services.reviewer.gate_reviewers import REVIEWERS
        from app.services.reviewer.runner import (
            _GROUNDED_LISTS,
            _GROUNDED_SINGLETONS,
        )

        walked = set(_GROUNDED_LISTS) | set(_GROUNDED_SINGLETONS)
        missed: list[str] = []
        for key, reviewer in REVIEWERS.items():
            for name, fld in reviewer.output_model.model_fields.items():
                if name in walked or name == "overall_notes":
                    continue
                # Any field whose type mentions a _Recommendation subclass
                # carries confidence + evidence_quote and so must be walked.
                ann = fld.annotation
                members = [ann, *typing.get_args(ann)]
                if any(
                    isinstance(m, type) and _carries_evidence(m) for m in members
                ):
                    missed.append(f"{key}.{name}")
        assert not missed, (
            f"{missed} carry evidence quotes but are never grounded — a "
            f"fabricated quote there keeps its confidence and stays eligible "
            f"for the analyst's bulk-accept"
        )

    def test_the_walk_names_only_real_fields(self):
        """The mirror of the above: a stale entry grounds nothing and hides
        the fact that nothing is being grounded."""
        from app.services.reviewer.gate_reviewers import REVIEWERS
        from app.services.reviewer.runner import (
            _GROUNDED_LISTS,
            _GROUNDED_SINGLETONS,
        )

        declared: set[str] = set()
        for reviewer in REVIEWERS.values():
            declared |= set(reviewer.output_model.model_fields)
        declared.add("initial_read")  # popped before grounding, never a list
        stale = (set(_GROUNDED_LISTS) | set(_GROUNDED_SINGLETONS)) - declared
        assert not stale, f"grounding walks fields no reviewer emits: {sorted(stale)}"


def _carries_evidence(model: type) -> bool:
    """Is this a recommendation type, i.e. does it have an evidence quote?"""
    fields = getattr(model, "model_fields", None)
    return bool(fields) and "evidence_quote" in fields


class TestPayloadComesFromTheValidatedModel:
    """The runner must store what the VALIDATOR produced, not the raw output.

    `output_model` is not only a gate: its `model_validator` repairs the
    shapes the model actually emits — above all, lifting a flattened or
    stringified `initial_read` back into place. Reading `tool_output` reached
    past every one of those repairs.

    Found live, and only live. Making `initial_read` required did not fix the
    missing brief, because the field WAS arriving — flattened. The validator
    put it right, `validated` held the corrected object, and the runner used
    the raw dict anyway, where `initial_read` is a string and the
    `isinstance(..., dict)` guard drops it on the floor. Every symptom (no
    brief, a 404 on the brief endpoint, a chunk gate with no attack chain)
    followed from this one line.
    """

    @pytest.mark.asyncio
    async def test_a_flattened_opening_read_still_becomes_a_brief(self, monkeypatch):
        from app.services.reviewer import runner
        from app.services.reviewer.models import Gate0Recommendations

        # Exactly the shape the live model produced: the read stringified into
        # `initial_read`, its other fields hoisted to the top level.
        raw = {
            "initial_read": "A vendor report on a ClickFix campaign.",
            "actors": ["UNC0003"],
            "attack_chain": ["fake captcha", "RunMRU write"],
            "thin_areas": [],
            "entities": [],
            "added_entities": [],
        }

        class FakeResponse:
            tool_output = raw
            validated = Gate0Recommendations.model_validate(raw)
            input_tokens = out = output_tokens = 0
            cache_creation_tokens = cache_read_tokens = 0
            cached = False
            model = "claude-opus-5"

        recorded: list[tuple] = []

        async def fake_record_turn(db, sid, gate, *, payload, **kw):
            recorded.append((gate, payload))

            class Row:
                id = uuid.uuid4()
            return Row()

        monkeypatch.setattr(runner, "call_llm", _async_return(FakeResponse()))
        monkeypatch.setattr(runner, "pinned_rules_addendum", _async_return(""))
        monkeypatch.setattr(runner, "async_session", _fake_session)
        monkeypatch.setattr(runner.store, "load_turns", _async_return([]))
        monkeypatch.setattr(runner.store, "get_initial_read", _async_return(None))
        monkeypatch.setattr(runner.store, "record_turn", fake_record_turn)

        out = await runner.run_reviewer(uuid.uuid4(), "entities", {"parsed_text": "x"})

        assert out is not None, "the reviewer failed rather than returning a payload"
        gates = [g for g, _ in recorded]
        assert INITIAL_READ in gates, (
            "no brief was recorded from a read the validator had already "
            "repaired — the runner used the raw output instead"
        )
        brief = next(p for g, p in recorded if g == INITIAL_READ)
        assert brief["initial_read"]["summary"] == "A vendor report on a ClickFix campaign."
        assert brief["initial_read"]["actors"] == ["UNC0003"]

        # And the stray top-level keys must not survive into the gate payload.
        gate_payload = next(p for g, p in recorded if g == "entities")
        assert "actors" not in gate_payload
        assert "attack_chain" not in gate_payload


class TestDisabledGatesCostNothing:
    """A gate nobody reviews must not pay for a review.

    `gates_enabled` and `gate_modes` are separate fields, so
    `{chunks: False}` with `{chunks: "assist"}` is a reachable configuration —
    not through the Add Source modal, which hides the mode selector for an
    unchecked gate, but certainly through the API.

    Today the runner is safe by ORDERING alone: the disabled-gate auto-skip
    `continue`s several statements before the reviewer hook. Nothing states
    that dependency, and the two blocks are ~40 lines apart in a 150-line
    function, so a later edit that moves either one would start billing an
    reviewer-model call at every disabled gate — silently, because a disabled gate
    produces no UI anyone is watching.
    """

    def test_the_auto_skip_precedes_the_reviewer_hook(self):
        import inspect

        from app.api.routes import pipeline

        src = inspect.getsource(pipeline.stream_with_sync)
        skip = src.index("if gate_key and not is_gate_enabled(")
        hook = src.index("await run_reviewer(")
        assert skip < hook, (
            "the reviewer hook now runs before the disabled-gate auto-skip, "
            "so a disabled gate set to assist would pay for an Opus review "
            "nobody will ever see"
        )
        # ...and the skip must actually skip, not fall through.
        between = src[skip:hook]
        assert "continue" in between, (
            "the disabled-gate branch no longer short-circuits, so execution "
            "reaches the reviewer hook anyway"
        )

    def test_the_gate_node_itself_still_self_approves_when_disabled(self):
        """The other half of the contract: the auto-skip resumes the graph so
        the gate node runs and approves itself. If that stopped being true the
        loop would spin rather than advance."""
        from app.graph.state import GateAction
        from app.nodes.gates import gate_chunks

        out = gate_chunks({
            "chunks": [{"chunk_id": "chk-a", "text": "x"}],
            "gates_enabled": {"chunks": False},
            "gate_modes": {"chunks": "assist"},
        })
        assert out["chunks_approved_ids"] == ["chk-a"]
        assert out["chunk_decisions"][0]["action"] == GateAction.APPROVE.value


# ── the agreement readout ────────────────────────────────────────────

def _row(gate_key, *, source="s1", items=None, status="ok", scored=True,
         auto=False):
    """One ReviewerRecommendation, duck-typed. The aggregator takes attributes,
    not ORM rows, precisely so this needs no database."""
    import types

    outcome = None
    if scored:
        scoreable = [i for i in (items or []) if not i.get("moot")]
        agreed = sum(1 for i in scoreable if i["agreed"])
        outcome = {"agreement": {
            "total": len(scoreable),
            "agreed": agreed,
            "rate": round(agreed / len(scoreable), 3) if scoreable else None,
            "items": items or [],
        }}
        if auto:
            outcome["auto"] = True
    return types.SimpleNamespace(
        gate_key=gate_key, source_id=source, status=status, outcome=outcome,
    )


def _item(agreed, *, kind="entity", conf="high", ref="e1", **kw):
    return {
        "kind": kind, "ref": ref, "recommended": "remove",
        "actual": "remove" if agreed else "approve",
        "confidence": conf, "quote_unsupported": False,
        "rationale": "because", "agreed": agreed, **kw,
    }


class TestAgreementAggregation:
    """The four ways this readout could lie, one test each.

    It exists to answer one question — is a gate safe to run unattended? — and
    every one of these failures answers it wrongly while looking like data.
    """

    def _gate(self, result, key):
        return next(g for g in result["gates"] if g["gate_key"] == key)

    def test_rates_are_pooled_not_averaged(self):
        """A source with 2 recommendations must not outweigh one with 40.

        Averaging per-source rates here would read 50%: one source at 0/2 and
        one at 40/40 average to 0.5. Pooled, it is 40/42 — which is the truth
        about how often the analyst took this gate's advice.
        """
        from app.services.reviewer.agreement import aggregate

        rows = [
            _row("entities", source="a", items=[_item(False), _item(False)]),
            _row("entities", source="b", items=[_item(True) for _ in range(40)]),
        ]
        gate = self._gate(aggregate(rows), "entities")
        assert (gate["agreed"], gate["total"]) == (40, 42)
        assert gate["rate"] == 0.952
        assert gate["sources"] == 2

    def test_a_gate_with_no_recommendations_has_no_rate(self):
        """None, not 0.0. "Agreed with all zero of its recommendations" is an
        absence of evidence, and rendering it as 0% agreement would read as the
        reviewer being wrong about everything."""
        from app.services.reviewer.agreement import aggregate

        gate = self._gate(aggregate([_row("bundle", items=[])]), "bundle")
        assert gate["rate"] is None
        assert gate["total"] == 0
        assert gate["reviews"] == 1   # it was reviewed; it just said nothing

    def test_moot_items_are_excluded(self):
        """A chunk-gate re-chunk discards the decisions, adds and edges
        channels wholesale. Those recommendations were never reached, so they
        are neither agreement nor disagreement."""
        from app.services.reviewer.agreement import aggregate

        rows = [_row("chunks", items=[
            _item(True, kind="chunk", ref="chk-a"),
            _item(False, kind="chunk", ref="chk-b", moot=True),
            _item(False, kind="chunk", ref="chk-c", moot=True),
        ])]
        gate = self._gate(aggregate(rows), "chunks")
        assert (gate["agreed"], gate["total"]) == (1, 1)
        assert gate["rate"] == 1.0
        # ...and a moot item must not surface as an override either.
        assert gate["overrides"] == []

    def test_every_rate_arrives_with_its_denominator(self):
        """100% of 3 is not evidence. Any surface rendering `rate` has to be
        able to show n beside it, so n must be in the payload at every level a
        rate appears."""
        from app.services.reviewer.agreement import aggregate

        rows = [_row("entities", items=[
            _item(True, conf="high"), _item(False, conf="low", ref="e2"),
        ])]
        gate = self._gate(aggregate(rows), "entities")

        def has_n(bucket):
            return "total" in bucket and "agreed" in bucket

        assert has_n(gate)
        assert all(has_n(b) for b in gate["by_confidence"].values())
        assert all(has_n(b) for b in gate["by_kind"].values())


class TestAgreementBreakdowns:
    def _gate(self, result, key):
        return next(g for g in result["gates"] if g["gate_key"] == key)

    def test_confidence_split_is_the_number_that_decides_autopilot(self):
        """`isBulkAcceptable` already lets HIGH through without the analyst
        opening it, so that tier is the one whose reliability is being claimed.
        Overall agreement blends in the low-confidence items the design never
        said were safe — a gate can look mediocre overall and be perfect where
        it counts."""
        from app.services.reviewer.agreement import aggregate

        rows = [_row("entities", items=[
            _item(True, conf="high"), _item(True, conf="high"),
            _item(False, conf="low", ref="e3"), _item(False, conf="low", ref="e4"),
        ])]
        gate = self._gate(aggregate(rows), "entities")
        assert gate["rate"] == 0.5                        # unremarkable
        assert gate["by_confidence"]["high"]["rate"] == 1.0   # the real answer
        assert gate["by_confidence"]["low"]["rate"] == 0.0
        assert gate["by_confidence"]["medium"]["rate"] is None

    def test_all_three_tiers_are_always_present(self):
        """A stable shape, so the UI does not gain and lose columns as data
        arrives."""
        from app.services.reviewer.agreement import (
            CONFIDENCE_TIERS,
            aggregate,
        )

        gate = self._gate(aggregate([_row("bundle", items=[_item(True)])]), "bundle")
        assert set(gate["by_confidence"]) == set(CONFIDENCE_TIERS)

    def test_failures_are_counted_not_dropped(self):
        """A reviewer that errors one run in five is not a candidate for
        unattended operation whatever its agreement rate. Failed turns carry no
        outcome, so a readout that only counts scored rows would show a perfect
        gate that half-works."""
        from app.services.reviewer.agreement import aggregate

        rows = [
            _row("procedures", items=[_item(True, kind="draft")]),
            _row("procedures", source="b", status="failed", scored=False),
        ]
        gate = self._gate(aggregate(rows), "procedures")
        assert gate["failures"] == 1
        assert gate["rate"] == 1.0        # the scored half really was agreed
        assert gate["reviews"] == 1       # ...out of two attempts

    def test_recommended_but_not_yet_submitted_is_not_evidence(self):
        """A source paused at the gate has a recommendation and no outcome.
        Counting it either way would invent a verdict from a review that has
        not happened."""
        from app.services.reviewer.agreement import aggregate

        rows = [_row("entities", scored=False)]
        gate = self._gate(aggregate(rows), "entities")
        assert gate["reviews"] == 0
        assert gate["rate"] is None

    def test_overrides_carry_the_reasoning_and_the_analysts_answer(self):
        """The actionable half. A rate says a gate is not trusted; an override
        says why — on a real source, with both sides attached."""
        from app.services.reviewer.agreement import aggregate

        rows = [_row("entities", items=[
            _item(False, ref="ent-7", conf="high"),
        ])]
        gate = self._gate(aggregate(rows), "entities")
        assert len(gate["overrides"]) == 1
        o = gate["overrides"][0]
        assert o["ref"] == "ent-7"
        assert (o["recommended"], o["actual"]) == ("remove", "approve")
        assert o["rationale"] == "because"
        assert o["confidence"] == "high"

    def test_rows_the_agent_applied_itself_are_not_evidence(self):
        """Autopilot outcomes record the reviewer agreeing with itself.

        Counting them drives every rate to 100% exactly when this readout
        matters most — which would be a dial that reads "ready" because
        nobody was there to disagree.
        """
        from app.services.reviewer.agreement import aggregate

        rows = [
            _row("entities", source="s1", items=[_item(True), _item(False)]),
            _row("entities", source="s2", items=[_item(True)] * 40, auto=True),
        ]
        gate = self._gate(aggregate(rows), "entities")
        assert gate["total"] == 2, "the 40 auto items must not be counted"
        assert gate["agreed"] == 1
        assert gate["rate"] == 0.5
        assert gate["reviews"] == 1, "an auto row is not a review either"
        assert gate["sources"] == 1

    def test_an_all_auto_gate_reads_as_no_data(self):
        from app.services.reviewer.agreement import aggregate

        rows = [_row("bundle", items=[_item(True)], auto=True)]
        gate = self._gate(aggregate(rows), "bundle")
        assert gate["rate"] is None
        assert gate["total"] == 0

    def test_the_opening_read_is_not_a_gate(self):
        from app.models.reviewer import INITIAL_READ
        from app.services.reviewer.agreement import aggregate

        out = aggregate([_row(INITIAL_READ, scored=False)])
        assert out["gates"] == []

    def test_gates_come_back_in_pipeline_order(self):
        from app.services.reviewer.agreement import aggregate

        rows = [_row(k, items=[_item(True)]) for k in
                ("bundle", "entities", "procedures", "chunks")]
        keys = [g["gate_key"] for g in aggregate(rows)["gates"]]
        assert keys == ["entities", "chunks", "procedures", "bundle"]

    def test_an_unknown_gate_still_appears(self):
        """A fifth gate must show up here the day it ships, not vanish because
        this module's order list predates it."""
        from app.services.reviewer.agreement import aggregate

        rows = [_row("entities", items=[_item(True)]),
                _row("detections", items=[_item(True, kind="detection")])]
        keys = [g["gate_key"] for g in aggregate(rows)["gates"]]
        assert keys == ["entities", "detections"]

    def test_empty_is_the_ordinary_starting_state(self):
        from app.services.reviewer.agreement import aggregate

        assert aggregate([]) == {"sources_reviewed": 0, "gates": []}
