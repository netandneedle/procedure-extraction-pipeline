"""Closed-loop tests: surfacing ledger + miss attribution.

The DB is faked (injected via patched async_session) and the embedding model
is stubbed, so these run offline. The attribution LLM call is stubbed out by
an autouse fixture — without it these tests reach the network, and the
deterministic fallback they are pinning would never run.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.models.feedback_pattern import FeedbackPattern
from app.models.feedback_surfacing import FeedbackPatternSurfacing
from app.nodes.llm import feedback_synthesis as fsmod
from app.services import feedback_patterns as fp
from app.services import pattern_embedding as pe


# Captured before the autouse fixture below can replace it, so the tests that
# exercise the adjudicator itself can still reach the real function.
_REAL_ADJUDICATE = fsmod._adjudicate_attribution


@pytest.fixture(autouse=True)
def _no_attribution_call():
    """Default every test to the deterministic fallback path.

    Returning None is what _adjudicate_attribution does when the call could
    not be made; the tests that pin the adjudicator override this.
    """
    with patch.object(fsmod, "_adjudicate_attribution",
                      AsyncMock(return_value=None)):
        yield


# --- fakes -------------------------------------------------------------------

class _FakeResult:
    def __init__(self, items=None):
        self._items = items or []

    def scalars(self):
        return self

    def all(self):
        return list(self._items)


class _FakeSession:
    def __init__(self, results):
        self._results = list(results)
        self.added: list = []
        self.commits = 0

    async def execute(self, stmt):
        return self._results.pop(0)

    def add(self, row):
        self.added.append(row)

    async def commit(self):
        self.commits += 1

    async def refresh(self, row):
        pass


class _CM:
    def __init__(self, session):
        self._s = session

    async def __aenter__(self):
        return self._s

    async def __aexit__(self, *exc):
        return False


def _pattern(**kw) -> FeedbackPattern:
    kw.setdefault("id", uuid.uuid4())
    kw.setdefault("category", "wrong_technique")
    kw.setdefault("pattern", "PowerShell over-picked in malware-capability sections")
    kw.setdefault("occurrence_count", 5)
    kw.setdefault("hit_count", 0)
    kw.setdefault("miss_count", 0)
    kw.setdefault("last_seen_at", datetime.now(timezone.utc))
    return FeedbackPattern(**kw)


def _surfacing(pattern_id, source_id, node="extract_techniques") -> FeedbackPatternSurfacing:
    return FeedbackPatternSurfacing(
        id=uuid.uuid4(), pattern_id=pattern_id, source_id=source_id, node=node,
    )


_SRC = uuid.uuid4()


# =============================================================================
# _record_surfacings (feedback_patterns)
# =============================================================================

class TestRecordSurfacings:
    async def test_inserts_one_row_per_pattern(self):
        sess = _FakeSession([])
        pid1, pid2 = str(uuid.uuid4()), str(uuid.uuid4())
        with patch.object(fp, "async_session", lambda: _CM(sess)):
            await fp._record_surfacings(
                {"source_id": str(_SRC)}, node="draft_procedures",
                surfaced_ids=[pid1, pid2],
            )
        assert len(sess.added) == 2
        assert {str(r.pattern_id) for r in sess.added} == {pid1, pid2}
        assert all(r.node == "draft_procedures" for r in sess.added)
        assert all(r.source_id == _SRC for r in sess.added)
        assert sess.commits == 1

    async def test_skips_without_source_id(self):
        factory = MagicMock()
        with patch.object(fp, "async_session", factory):
            await fp._record_surfacings({}, node="x", surfaced_ids=[str(uuid.uuid4())])
        factory.assert_not_called()

    async def test_skips_without_ids(self):
        factory = MagicMock()
        with patch.object(fp, "async_session", factory):
            await fp._record_surfacings({"source_id": str(_SRC)}, node="x", surfaced_ids=[])
        factory.assert_not_called()


# =============================================================================
# _pattern_was_recorrected (feedback_synthesis)
# =============================================================================

def _rejection(tid: str, name: str = "") -> dict:
    """A Gate 1 edit that REMOVED a technique — the shape that is evidence an
    over-inclusion rule failed to hold."""
    return {
        "draft_id": "dft-1",
        "action": "edit",
        "rejected_techniques": [{"technique_id": tid, "technique_name": name}],
    }


class TestPatternWasRecorrected:
    def test_technique_anchor_match_is_miss(self):
        pattern = _pattern(category="wrong_technique", applies_to={"technique_ids": ["T1059.001"]})
        deltas = {"gate_1_procedures": [_rejection("T1059.001", "PowerShell")]}
        with patch.object(pe, "embed_text", return_value=None):
            area_deltas = fsmod._deltas_by_area(deltas)
        assert fsmod._pattern_was_recorrected(pattern, area_deltas) is True

    def test_no_match_is_hit(self):
        pattern = _pattern(category="wrong_technique", applies_to={"technique_ids": ["T1059.001"]})
        deltas = {"gate_1_procedures": [_rejection("T1003", "OS Credential Dumping")]}
        with patch.object(pe, "embed_text", return_value=None):
            area_deltas = fsmod._deltas_by_area(deltas)
        assert fsmod._pattern_was_recorrected(pattern, area_deltas) is False

    def test_promoting_an_anchored_technique_is_not_a_miss(self):
        """The analyst putting a technique BACK is not evidence that a rule
        naming it failed.

        On one espionage-RAT run the analyst promoted T1070.004 and two rules that
        merely listed it among the ids they were learned from were charged
        with a miss — one of them a rule about deletion on a ransomware crew's
        own leak server, in an espionage report that has no leak server.
        """
        pattern = _pattern(category="wrong_technique",
                           applies_to={"technique_ids": ["T1070.004"]})
        deltas = {"gate_1_procedures": [
            {"kind": "promotion", "technique_id": "T1070.004"}]}
        with patch.object(pe, "embed_text", return_value=None):
            area_deltas = fsmod._deltas_by_area(deltas)
        assert fsmod._pattern_was_recorrected(pattern, area_deltas) is False

    def test_shared_entity_type_alone_is_not_a_miss(self):
        """`vulnerability` is a bucket of hundreds, not a subject.

        A rule about inline ATT&CK ids being mistyped as vulnerabilities took a
        miss on one espionage-RAT run because the analyst removed a redacted CVE placeholder.
        Same bucket, unrelated error.
        """
        pattern = _pattern(category="false_positive_entity",
                           applies_to={"entity_types": ["vulnerability"]})
        deltas = {"gate_0_entities": [
            {"value": "CVE-2024-XXXX", "entity_type": "vulnerability",
             "action": "remove"}]}
        with patch.object(pe, "embed_text", return_value=None):
            area_deltas = fsmod._deltas_by_area(deltas)
        assert fsmod._pattern_was_recorrected(pattern, area_deltas) is False

    def test_semantic_match_is_miss(self):
        pattern = _pattern(category="defender_ioc", embedding=[1.0, 0.0, 0.0], applies_to={})
        deltas = {"gate_0_entities": [{"value": "info@cert.example", "entity_type": "email", "action": "remove"}]}
        # embed_text returns a vector parallel to the pattern's → cosine 1.0 >= threshold
        with patch.object(pe, "embed_text", return_value=[1.0, 0.0, 0.0]):
            area_deltas = fsmod._deltas_by_area(deltas)
            assert fsmod._pattern_was_recorrected(pattern, area_deltas) is True

    def test_wrong_area_is_hit(self):
        # Pattern is an entity-area category; the only correction is in techniques.
        pattern = _pattern(category="defender_ioc", applies_to={"entity_types": ["email"]})
        deltas = {"gate_1_procedures": [{"kind": "promotion", "technique_id": "T1059.001"}]}
        with patch.object(pe, "embed_text", return_value=None):
            area_deltas = fsmod._deltas_by_area(deltas)
        assert fsmod._pattern_was_recorrected(pattern, area_deltas) is False


# =============================================================================
# _score_surfacings (feedback_synthesis)
# =============================================================================

class TestScoreSurfacings:
    async def test_miss_increments_miss_count_and_marks_rows(self):
        pattern = _pattern(category="wrong_technique", applies_to={"technique_ids": ["T1059.001"]})
        surfacing = _surfacing(pattern.id, _SRC)
        sess = _FakeSession([
            _FakeResult(items=[surfacing]),   # unscored surfacings
            _FakeResult(items=[pattern]),     # referenced patterns
        ])
        deltas = {"gate_1_procedures": [_rejection("T1059.001", "PowerShell")]}
        with patch.object(fsmod, "async_session", lambda: _CM(sess)), \
             patch.object(pe, "embed_text", return_value=None):
            result = await fsmod._score_surfacings({"source_id": str(_SRC)}, deltas)
        assert result == {"scored": 1, "hits": 0, "misses": 1, "unscored": 0}
        assert pattern.miss_count == 1
        assert pattern.hit_count == 0
        assert pattern.salience is not None
        assert surfacing.outcome == "miss"
        assert surfacing.scored_at is not None
        assert sess.commits == 1

    async def test_hit_when_no_matching_correction(self):
        pattern = _pattern(category="wrong_technique", applies_to={"technique_ids": ["T1059.001"]})
        surfacing = _surfacing(pattern.id, _SRC)
        sess = _FakeSession([
            _FakeResult(items=[surfacing]),
            _FakeResult(items=[pattern]),
        ])
        deltas = {"gate_1_procedures": [_rejection("T1003", "OS Credential Dumping")]}
        with patch.object(fsmod, "async_session", lambda: _CM(sess)), \
             patch.object(pe, "embed_text", return_value=None):
            result = await fsmod._score_surfacings({"source_id": str(_SRC)}, deltas)
        assert result == {"scored": 1, "hits": 1, "misses": 0, "unscored": 0}
        assert pattern.hit_count == 1
        assert surfacing.outcome == "hit"

    async def test_no_deltas_means_all_hits(self):
        pattern = _pattern()
        surfacing = _surfacing(pattern.id, _SRC)
        sess = _FakeSession([
            _FakeResult(items=[surfacing]),
            _FakeResult(items=[pattern]),
        ])
        with patch.object(fsmod, "async_session", lambda: _CM(sess)), \
             patch.object(pe, "embed_text", return_value=None):
            result = await fsmod._score_surfacings({"source_id": str(_SRC)}, {})
        assert result["hits"] == 1
        assert result["misses"] == 0
        assert pattern.hit_count == 1

    async def test_empty_ledger_returns_zero(self):
        sess = _FakeSession([_FakeResult(items=[])])
        with patch.object(fsmod, "async_session", lambda: _CM(sess)):
            result = await fsmod._score_surfacings({"source_id": str(_SRC)}, {})
        assert result == {"scored": 0, "hits": 0, "misses": 0, "unscored": 0}

    async def test_missing_source_id_is_noop(self):
        factory = MagicMock()
        with patch.object(fsmod, "async_session", factory):
            result = await fsmod._score_surfacings({}, {})
        factory.assert_not_called()
        assert result == {"scored": 0, "hits": 0, "misses": 0, "unscored": 0}

    async def test_duplicate_surfacings_scored_once(self):
        pattern = _pattern(category="wrong_technique", applies_to={"technique_ids": ["T1059.001"]})
        # Two surfacing rows for the same pattern (e.g. two nodes logged it).
        s1 = _surfacing(pattern.id, _SRC, node="extract_techniques")
        s2 = _surfacing(pattern.id, _SRC, node="draft_procedures")
        sess = _FakeSession([
            _FakeResult(items=[s1, s2]),
            _FakeResult(items=[pattern]),
        ])
        deltas = {"gate_1_procedures": [_rejection("T1059.001", "PowerShell")]}
        with patch.object(fsmod, "async_session", lambda: _CM(sess)), \
             patch.object(pe, "embed_text", return_value=None):
            result = await fsmod._score_surfacings({"source_id": str(_SRC)}, deltas)
        assert result["scored"] == 1          # pattern scored once
        assert pattern.miss_count == 1        # counted once, not twice
        assert s1.outcome == "miss" and s2.outcome == "miss"  # both rows closed out

    async def test_best_effort_on_db_error(self):
        def boom():
            raise RuntimeError("db down")
        with patch.object(fsmod, "async_session", boom):
            result = await fsmod._score_surfacings({"source_id": str(_SRC)}, {})
        assert result.get("status") == "error"


# =============================================================================
# compute_salience
# =============================================================================


class TestOutcomeRequiresHumanJudgement:
    """A hit must mean "a human looked at that gate and left it alone".

    Before this, "no correction landed in the area" scored a HIT, so a gate
    that was switched off, or decided by the AI reviewer unattended, credited
    every pattern it surfaced. On a corpus where every run was unattended that
    produced an 84% hit rate measuring the absence of review.
    """

    @staticmethod
    def _run(state_extra: dict, deltas: dict | None = None, category="wrong_technique"):
        pattern = _pattern(category=category,
                           applies_to={"technique_ids": ["T1059.001"]})
        surfacing = _surfacing(pattern.id, _SRC)
        sess = _FakeSession([
            _FakeResult(items=[surfacing]),
            _FakeResult(items=[pattern]),
        ])
        state = {"source_id": str(_SRC), **state_extra}
        return pattern, surfacing, sess, state, (deltas or {})

    async def test_disabled_gate_is_unscored_not_hit(self):
        pattern, surfacing, sess, state, deltas = self._run(
            {"gates_enabled": {"entities": True, "chunks": True,
                               "procedures": False, "bundle": True}})
        with patch.object(fsmod, "async_session", lambda: _CM(sess)), \
             patch.object(pe, "embed_text", return_value=None):
            result = await fsmod._score_surfacings(state, deltas)
        assert surfacing.outcome == "unscored"
        assert result["unscored"] == 1 and result["hits"] == 0
        assert pattern.hit_count == 0
        assert pattern.salience is None

    async def test_auto_mode_is_unscored_the_reviewer_would_grade_itself(self):
        pattern, surfacing, sess, state, deltas = self._run(
            {"gate_modes": {"procedures": "auto"}})
        with patch.object(fsmod, "async_session", lambda: _CM(sess)), \
             patch.object(pe, "embed_text", return_value=None):
            result = await fsmod._score_surfacings(state, deltas)
        assert surfacing.outcome == "unscored"
        assert result["unscored"] == 1 and result["hits"] == 0
        assert pattern.hit_count == 0

    async def test_assist_mode_still_scores_a_human_submits_the_decision(self):
        pattern, surfacing, sess, state, deltas = self._run(
            {"gate_modes": {"procedures": "assist"}})
        with patch.object(fsmod, "async_session", lambda: _CM(sess)), \
             patch.object(pe, "embed_text", return_value=None):
            result = await fsmod._score_surfacings(state, deltas)
        assert surfacing.outcome == "hit"
        assert result["hits"] == 1
        assert pattern.hit_count == 1

    async def test_human_review_with_a_correction_still_misses(self):
        pattern, surfacing, sess, state, deltas = self._run(
            {"gate_modes": {"procedures": "review"}},
            deltas={"gate_1_procedures": [_rejection("T1059.001", "PowerShell")]})
        with patch.object(fsmod, "async_session", lambda: _CM(sess)), \
             patch.object(pe, "embed_text", return_value=None):
            result = await fsmod._score_surfacings(state, deltas)
        assert surfacing.outcome == "miss"
        assert result["misses"] == 1
        assert pattern.miss_count == 1

    async def test_unknown_category_is_unscored_not_scanned_everywhere(self):
        pattern, surfacing, sess, state, deltas = self._run(
            {}, category="not_a_real_category")
        with patch.object(fsmod, "async_session", lambda: _CM(sess)), \
             patch.object(pe, "embed_text", return_value=None):
            result = await fsmod._score_surfacings(state, deltas)
        assert surfacing.outcome == "unscored"
        assert result["unscored"] == 1

    async def test_deleted_pattern_is_unscored_not_credited_as_a_hit(self):
        missing_id = uuid.uuid4()
        surfacing = _surfacing(missing_id, _SRC)
        sess = _FakeSession([
            _FakeResult(items=[surfacing]),
            _FakeResult(items=[]),          # pattern is gone
        ])
        with patch.object(fsmod, "async_session", lambda: _CM(sess)), \
             patch.object(pe, "embed_text", return_value=None):
            result = await fsmod._score_surfacings({"source_id": str(_SRC)}, {})
        assert surfacing.outcome == "unscored"
        assert result["hits"] == 0

class TestComputeSalience:
    def test_hit_heavy_beats_miss_heavy(self):
        now = datetime(2026, 5, 29, tzinfo=timezone.utc)
        hh = fp.compute_salience(hit_count=8, miss_count=0, occurrence_count=8, last_seen_at=now, now=now)
        mh = fp.compute_salience(hit_count=0, miss_count=8, occurrence_count=8, last_seen_at=now, now=now)
        assert hh > mh

    def test_recency_decay(self):
        now = datetime(2026, 5, 29, tzinfo=timezone.utc)
        fresh = datetime(2026, 5, 29, tzinfo=timezone.utc)
        old = datetime(2026, 1, 1, tzinfo=timezone.utc)
        s_fresh = fp.compute_salience(hit_count=3, miss_count=1, occurrence_count=4, last_seen_at=fresh, now=now)
        s_old = fp.compute_salience(hit_count=3, miss_count=1, occurrence_count=4, last_seen_at=old, now=now)
        assert s_fresh > s_old

    def test_occurrence_weight(self):
        now = datetime(2026, 5, 29, tzinfo=timezone.utc)
        hi = fp.compute_salience(hit_count=2, miss_count=1, occurrence_count=20, last_seen_at=now, now=now)
        lo = fp.compute_salience(hit_count=2, miss_count=1, occurrence_count=2, last_seen_at=now, now=now)
        assert hi > lo

    def test_naive_datetime_tolerated(self):
        now = datetime(2026, 5, 29, tzinfo=timezone.utc)
        naive = datetime(2026, 5, 29)  # no tzinfo
        val = fp.compute_salience(hit_count=1, miss_count=0, occurrence_count=1, last_seen_at=naive, now=now)
        assert val >= 0.0


# =============================================================================
# Miss attribution
# =============================================================================


def _llm_reply(attributions):
    return MagicMock(tool_output={"attributions": attributions})


class TestMissAttribution:
    """Which rule a correction is charged to is a judgment, not a lookup.

    The deterministic matcher can only ask "do these name the same technique",
    which is neither necessary nor sufficient. On one espionage-RAT run the analyst
    removed T1036.008 from a curl reconnaissance draft — a textbook instance of
    the rule "techniques appear over-attributed when a step only generically
    implies a capability". That rule scored a HIT, because its `applies_to`
    lists the ids it was LEARNED from (T1059.003 / T1021.001 / T1068), which is
    provenance and not scope. Three unrelated rules took the misses instead.
    """

    @staticmethod
    def _one_pattern_run(category="wrong_technique", applies_to=None):
        pattern = _pattern(category=category,
                           applies_to=applies_to or {"technique_ids": ["T1059.001"]})
        surfacing = _surfacing(pattern.id, _SRC)
        sess = _FakeSession([
            _FakeResult(items=[surfacing]),
            _FakeResult(items=[pattern]),
        ])
        return pattern, surfacing, sess

    async def test_adjudicator_can_charge_a_rule_the_matcher_would_acquit(self):
        # The correction names a technique the rule has never heard of, so the
        # deterministic path scores a hit. The adjudicator sees the claim.
        deltas = {"gate_1_procedures": [_rejection("T1036.008", "Masquerade File Type")]}
        pattern, surfacing, sess = self._one_pattern_run()
        with patch.object(pe, "embed_text", return_value=None):
            assert fsmod._pattern_was_recorrected(
                pattern, fsmod._deltas_by_area(deltas)) is False

        with patch.object(fsmod, "async_session", lambda: _CM(sess)), \
             patch.object(pe, "embed_text", return_value=None), \
             patch.object(fsmod, "_adjudicate_attribution",
                          AsyncMock(return_value={pattern.id})):
            result = await fsmod._score_surfacings({"source_id": str(_SRC)}, deltas)
        assert surfacing.outcome == "miss"
        assert result["misses"] == 1
        assert pattern.miss_count == 1

    async def test_adjudicator_can_acquit_a_rule_the_matcher_would_charge(self):
        deltas = {"gate_1_procedures": [_rejection("T1059.001", "PowerShell")]}
        pattern, surfacing, sess = self._one_pattern_run()
        with patch.object(pe, "embed_text", return_value=None):
            assert fsmod._pattern_was_recorrected(
                pattern, fsmod._deltas_by_area(deltas)) is True

        with patch.object(fsmod, "async_session", lambda: _CM(sess)), \
             patch.object(pe, "embed_text", return_value=None), \
             patch.object(fsmod, "_adjudicate_attribution",
                          AsyncMock(return_value=set())):
            result = await fsmod._score_surfacings({"source_id": str(_SRC)}, deltas)
        assert surfacing.outcome == "hit"
        assert result["hits"] == 1

    async def test_failed_call_falls_back_to_the_matcher(self):
        deltas = {"gate_1_procedures": [_rejection("T1059.001", "PowerShell")]}
        pattern, surfacing, sess = self._one_pattern_run()
        with patch.object(fsmod, "async_session", lambda: _CM(sess)), \
             patch.object(pe, "embed_text", return_value=None), \
             patch.object(fsmod, "_adjudicate_attribution",
                          AsyncMock(return_value=None)):
            await fsmod._score_surfacings({"source_id": str(_SRC)}, deltas)
        assert surfacing.outcome == "miss"

    async def test_no_llm_call_when_nothing_could_be_attributed(self):
        """No corrections in the pattern's own area — nothing to adjudicate."""
        pattern = _pattern(category="wrong_technique", applies_to={})
        call = AsyncMock()
        with patch.object(fsmod, "call_llm", call), \
             patch.object(pe, "embed_text", return_value=None):
            out = await _REAL_ADJUDICATE(
                [pattern], fsmod._deltas_by_area({}))
        assert out == set()
        call.assert_not_called()

    async def test_llm_failure_returns_none_not_an_empty_verdict(self):
        """An empty set means "adjudicated, nothing stuck"; None means "could
        not adjudicate". Collapsing them would silently acquit every rule
        whenever the API was down."""
        pattern = _pattern(category="wrong_technique", applies_to={})
        deltas = {"gate_1_procedures": [_rejection("T1059.001", "PowerShell")]}
        with patch.object(fsmod, "call_llm", AsyncMock(side_effect=RuntimeError("boom"))), \
             patch.object(pe, "embed_text", return_value=None):
            out = await _REAL_ADJUDICATE(
                [pattern], fsmod._deltas_by_area(deltas))
        assert out is None

    async def test_out_of_range_rule_index_is_ignored(self):
        pattern = _pattern(category="wrong_technique", applies_to={})
        deltas = {"gate_1_procedures": [_rejection("T1059.001", "PowerShell")]}
        reply = _llm_reply([{"correction": 0, "rules": [0, 7, -1], "reasoning": "x"}])
        with patch.object(fsmod, "call_llm", AsyncMock(return_value=reply)), \
             patch.object(pe, "embed_text", return_value=None):
            out = await _REAL_ADJUDICATE(
                [pattern], fsmod._deltas_by_area(deltas))
        assert out == {pattern.id}

    def test_rules_prompt_hides_the_ids_a_rule_was_learned_from(self):
        """Showing `applies_to.technique_ids` invites the id-matching this
        call exists to replace."""
        pattern = _pattern(category="wrong_technique",
                           applies_to={"technique_ids": ["T1059.003"]})
        pattern.pattern = "Techniques are over-attributed when a step only implies a capability."
        rendered = fsmod._format_rules_for_attribution([pattern])
        assert "T1059.003" not in rendered
        assert "over-attributed" in rendered

    def test_corrections_prompt_names_the_techniques(self):
        """`T1036.008` alone is not legible as an instance of a rule written
        in prose; `Masquerade File Type` is."""
        rendered = fsmod._format_corrections_for_attribution(
            [("techniques", _rejection("T1036.008", "Masquerade File Type"))])
        assert "T1036.008 (Masquerade File Type)" in rendered
