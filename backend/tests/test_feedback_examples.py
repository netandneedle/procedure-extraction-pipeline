"""Corrected-example channel: build from deltas, persist, retrieve, render.

The DB is faked and the embedding model stubbed, so these run offline.

The load-bearing test in here is the provenance one. Examples are only
trustworthy because a human made them, and every one of the original 63
patterns came from an AI reviewer running unattended and agreeing with itself.
If the gate filter ever regresses, this table becomes the same corpus in a
different shape.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from app.models.feedback_example import FeedbackExample
from app.services import feedback_examples as fx
from app.services import pattern_embedding as pe


# --- fakes -------------------------------------------------------------------

class _FakeResult:
    def __init__(self, items=None):
        self._items = items or []

    def scalars(self):
        return self

    def all(self):
        return list(self._items)

    def first(self):
        return self._items[0] if self._items else None


class _FakeSession:
    def __init__(self, results=None):
        self._results = list(results or [])
        self.added: list = []
        self.commits = 0

    async def execute(self, stmt):
        return self._results.pop(0) if self._results else _FakeResult([])

    def add(self, row):
        self.added.append(row)

    async def flush(self):
        pass

    async def commit(self):
        self.commits += 1


class _CM:
    def __init__(self, session):
        self._s = session

    async def __aenter__(self):
        return self._s

    async def __aexit__(self, *exc):
        return False


def _example(**kw) -> FeedbackExample:
    kw.setdefault("id", uuid.uuid4())
    kw.setdefault("area", "techniques")
    kw.setdefault("action", "remove")
    kw.setdefault("source_title", "Example ransomware writeup")
    kw.setdefault("before", {"technique": "T1082"})
    kw.setdefault("after", {})
    kw.setdefault("rationale", "")
    kw.setdefault("context_snippet", "")
    kw.setdefault("applies_to", {})
    kw.setdefault("occurrence_count", 1)
    kw.setdefault("last_seen_at", datetime.now(timezone.utc))
    return FeedbackExample(**kw)


ENTITY_DELTA = {
    "value": "AMEA (Asia, Middle East, and Africa)",
    "entity_type": "location",
    "action": "remove",
    "llm_confidence": 0.8,
    "edit_rationale": "Regions should not be combined.",
    "context_snippet": "targets across AMEA",
}
TECHNIQUE_DELTA = {
    "draft_id": "dft-1",
    "original_name": "Enumerate Host via curl",
    "chunk_id": "chk-1",
    "action": "edit",
    "reject_reason": None,
    "feedback": "only generically implied",
    "rejected_techniques": [
        {"technique_id": "T1082", "technique_name": "System Information Discovery"},
        {"technique_id": "T1016", "technique_name": "System Network Configuration Discovery"},
    ],
    "corrected_techniques": [{"technique_id": "T1057", "technique_name": "Process Discovery"}],
    "added_techniques": [],
}


def _deltas(entities=(), chunks=(), procedures=()):
    return {"gate_0_entities": list(entities), "gate_chunks": list(chunks),
            "gate_1_procedures": list(procedures)}


# =============================================================================
# Provenance — only a human's corrections become examples
# =============================================================================

class TestOnlyHumanCorrections:
    def test_disabled_gate_contributes_nothing(self):
        state = {"gates_enabled": {"entities": False, "chunks": True,
                                   "procedures": True, "bundle": True}}
        rows = fx.example_rows_from_deltas(state, _deltas(entities=[ENTITY_DELTA]))
        assert rows == []

    def test_auto_mode_contributes_nothing(self):
        """The AI reviewer deciding unattended is the system agreeing with
        itself — the provenance of all 63 original patterns."""
        state = {"gate_modes": {"entities": "auto"}}
        rows = fx.example_rows_from_deltas(state, _deltas(entities=[ENTITY_DELTA]))
        assert rows == []

    @pytest.mark.parametrize("mode", ["review", "assist"])
    def test_human_submitted_modes_contribute(self, mode):
        """assist counts: the AI recommends, a human still submits."""
        state = {"gate_modes": {"entities": mode}}
        rows = fx.example_rows_from_deltas(state, _deltas(entities=[ENTITY_DELTA]))
        assert len(rows) == 1
        assert rows[0]["area"] == "entities"

    def test_one_disabled_gate_does_not_suppress_the_others(self):
        state = {"gates_enabled": {"entities": False, "chunks": True,
                                   "procedures": True, "bundle": True}}
        rows = fx.example_rows_from_deltas(
            state, _deltas(entities=[ENTITY_DELTA], procedures=[TECHNIQUE_DELTA]))
        assert {r["area"] for r in rows} == {"techniques"}


# =============================================================================
# Building
# =============================================================================

class TestBuilding:
    def test_entity_removal_keeps_the_analysts_own_words(self):
        row = fx.example_rows_from_deltas({}, _deltas(entities=[ENTITY_DELTA]))[0]
        assert row["before"]["value"] == ENTITY_DELTA["value"]
        assert row["after"] == {}
        assert row["rationale"] == "Regions should not be combined."
        assert row["applies_to"]["entity_types"] == ["location"]

    def test_entity_edit_records_the_after(self):
        d = {**ENTITY_DELTA, "action": "edit", "edited_type": "region"}
        row = fx.example_rows_from_deltas({}, _deltas(entities=[d]))[0]
        assert row["after"]["entity_type"] == "region"

    def test_one_example_per_rejected_technique(self):
        """A draft where two techniques were removed is two demonstrations.

        They are separately transferable, and keying them together would make
        both undiscoverable to a source that matches only one.
        """
        rows = fx.example_rows_from_deltas({}, _deltas(procedures=[TECHNIQUE_DELTA]))
        assert len(rows) == 2
        assert {r["before"]["technique"] for r in rows} == {"T1082", "T1016"}
        assert all(r["applies_to"]["technique_ids"] for r in rows)

    def test_a_promotion_becomes_an_omission_demonstration(self):
        """The thing a rule-as-check could never express: what was missing."""
        rows = fx.example_rows_from_deltas(
            {}, _deltas(procedures=[{"kind": "promotion", "chunk_id": "c",
                                     "technique_id": "T1210"}]))
        assert len(rows) == 1
        assert rows[0]["action"] == "promote"
        assert rows[0]["applies_to"]["technique_ids"] == ["T1210"]

    def test_wholesale_chunk_reject_carries_reason_and_comments(self):
        """The reject `reason` was dropped before reaching synthesis once
        already, and the rule written without it diagnosed the opposite of
        what the analyst meant."""
        rows = fx.example_rows_from_deltas({}, _deltas(chunks=[{
            "kind": "wholesale_reject", "reason": "missed_procedures",
            "comments": "an alternative initial access vector was skipped"}]))
        assert len(rows) == 1
        assert "missed_procedures" in rows[0]["rationale"]
        assert "alternative initial access" in rows[0]["rationale"]

    def test_repeat_within_one_run_is_collapsed(self):
        """A rewind makes the analyst redo the same decision on the same
        source. Counting it twice would claim cross-source evidence."""
        promo = {"kind": "promotion", "chunk_id": "c", "technique_id": "T1003.001"}
        rows = fx.example_rows_from_deltas({}, _deltas(procedures=[promo, promo]))
        assert len(rows) == 1

    def test_a_delta_that_blows_up_does_not_lose_the_others(self):
        # Patch the dispatch table, not the module attribute: _BUILDER captured
        # the function object at import.
        with patch.dict(fx._BUILDER,
                        {"entities": MagicMock(side_effect=RuntimeError("boom"))}):
            rows = fx.example_rows_from_deltas(
                {}, _deltas(entities=[ENTITY_DELTA], procedures=[TECHNIQUE_DELTA]))
        assert len(rows) == 2 and all(r["area"] == "techniques" for r in rows)


# =============================================================================
# Persisting
# =============================================================================

class TestPersisting:
    async def test_new_example_is_embedded_and_added(self):
        sess = _FakeSession([_FakeResult([])])
        rows = fx.example_rows_from_deltas({}, _deltas(entities=[ENTITY_DELTA]))
        with patch.object(pe, "embed_text", return_value=[1.0, 0.0]):
            out = await fx.persist_examples(sess, rows)
        assert len(sess.added) == 1
        assert out[0].embedding == [1.0, 0.0]
        assert out[0].embedding_model

    async def test_identical_correction_bumps_the_count(self):
        existing = _example(dedup_key="entities|remove|abc", occurrence_count=1)
        sess = _FakeSession([_FakeResult([existing])])
        rows = [{"area": "entities", "action": "remove", "before": {}, "after": {},
                 "dedup_key": "entities|remove|abc", "rationale": "later words"}]
        with patch.object(pe, "embed_text", return_value=None):
            await fx.persist_examples(sess, rows)
        assert existing.occurrence_count == 2
        assert sess.added == []
        assert existing.rationale == "later words"  # backfilled, was empty

    async def test_record_examples_never_raises(self):
        def boom():
            raise RuntimeError("db down")
        with patch.object(fx, "async_session", boom):
            out = await fx.record_examples({}, _deltas(entities=[ENTITY_DELTA]))
        assert "error" in out

    async def test_no_examples_means_no_session(self):
        factory = MagicMock()
        with patch.object(fx, "async_session", factory):
            out = await fx.record_examples(
                {"gate_modes": {"entities": "auto", "chunks": "auto",
                                "procedures": "auto"}},
                _deltas(entities=[ENTITY_DELTA]))
        factory.assert_not_called()
        assert out["built"] == 0


# =============================================================================
# Retrieval + rendering
# =============================================================================

class TestRetrieval:
    def test_shared_technique_id_is_the_strongest_boost(self):
        row = _example(applies_to={"technique_ids": ["T1059.001"]})
        hit = fx._score_example(row, None, {"technique_ids": {"T1059.001"}})
        miss = fx._score_example(row, None, {"technique_ids": {"T1003"}})
        assert hit - miss == pytest.approx(0.5)

    def test_cosine_contributes_when_both_sides_have_a_vector(self):
        row = _example(embedding=[1.0, 0.0, 0.0])
        assert fx._score_example(row, [1.0, 0.0, 0.0], {}) == pytest.approx(1.0)
        assert fx._score_example(row, None, {}) == 0.0

    async def test_failure_yields_no_addendum_not_an_exception(self):
        def boom():
            raise RuntimeError("db down")
        with patch.object(fx, "async_session", boom), \
             patch.object(pe, "embed_text", return_value=None), \
             patch.object(pe, "build_source_representation", return_value=""), \
             patch.object(pe, "extract_source_anchors", return_value={}):
            text, ids = await fx.relevant_examples({}, areas=("entities",), limit=3)
        assert (text, ids) == ("", [])

    async def test_disabled_by_config_short_circuits(self):
        fx.clear_example_cache()
        with patch.object(fx.settings, "feedback_examples_enabled", False):
            assert await fx.relevant_examples_cached(
                {}, areas=("entities",), node="n") == ""

    def test_empty_renders_to_empty_string(self):
        assert fx.format_examples_for_prompt([]) == ""

    def test_render_shows_before_after_and_the_analysts_words(self):
        row = _example(before={"technique": "T1082"},
                       after={"kept_techniques": "T1057"},
                       rationale="only generically implied")
        out = fx.format_examples_for_prompt([row])
        assert "T1082" in out and "T1057" in out
        assert "only generically implied" in out
        assert "Example" in out          # the report it came from
        assert "not rules" in out        # framed as precedent, not instruction

    def test_render_truncates_a_long_value(self):
        row = _example(before={"value": "x" * 500})
        out = fx.format_examples_for_prompt([row])
        assert "x" * 161 not in out

    def test_recurrence_is_shown_only_when_it_happened(self):
        assert "same correction" not in fx.format_examples_for_prompt([_example()])
        assert "3x" in fx.format_examples_for_prompt([_example(occurrence_count=3)])
