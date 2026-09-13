"""Unit tests for the relevance-first feedback flywheel.

Covers the pure helpers (no DB / no model), persist-time semantic dedup
(via an injected fake session + patched embeddings), hybrid relevance
ranking, and the source-fingerprint cache. The embedding model and the DB
are stubbed so these run offline.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.config import settings
from app.models.feedback_pattern import FeedbackPattern
from app.services import feedback_patterns as fp
from app.services import pattern_embedding as pe


# =============================================================================
# Fake async session (persist_pattern takes `db` as a param — no real DB)
# =============================================================================

class _FakeResult:
    def __init__(self, scalar=None, items=None):
        self._scalar = scalar
        self._items = items or []

    def scalar_one_or_none(self):
        return self._scalar

    def scalars(self):
        return self

    def all(self):
        return list(self._items)


class _FakeSession:
    """Returns pre-seeded results in execute() call order."""

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


class _FakeSessionCM:
    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, *exc):
        return False


def _row(**kw) -> FeedbackPattern:
    """In-memory FeedbackPattern (column defaults aren't applied off-session,
    so pass what the test needs explicitly)."""
    kw.setdefault("id", uuid.uuid4())
    kw.setdefault("category", "wrong_technique")
    kw.setdefault("pattern", "some generalizable rule about technique mapping")
    kw.setdefault("occurrence_count", 1)
    kw.setdefault("last_seen_at", datetime.now(timezone.utc))
    return FeedbackPattern(**kw)


@pytest.fixture(autouse=True)
def _clear_caches():
    fp.clear_feedback_cache()
    yield
    fp.clear_feedback_cache()


# =============================================================================
# Pure helpers
# =============================================================================

class TestFormatPatterns:
    def test_pinned_and_advisory_render_in_separate_sections(self):
        pin = _row(category="defender_ioc", pattern="emails from letterheads are defender contacts")
        adv = _row(category="wrong_technique", pattern="T1059.001 over-picked", occurrence_count=3)
        out = fp.format_patterns_for_prompt([adv], pinned=[pin])
        assert "PERMANENT RULES" in out
        assert "RECENT ANALYST FEEDBACK" in out
        # Pinned rules render WITHOUT the occurrence tally (read as firm rules);
        # advisory items keep the "(seen Nx)" prefix.
        assert "    - emails from letterheads are defender contacts" in out
        assert "(seen 3x) T1059.001 over-picked" in out

    def test_empty_returns_blank(self):
        assert fp.format_patterns_for_prompt([], pinned=[]) == ""
        assert fp.format_patterns_for_prompt([]) == ""  # back-compat (no pinned arg)


class TestPureHelpers:
    def test_embedding_text_includes_applies_to(self):
        text = fp._embedding_text(
            "PowerShell over-picked",
            {"technique_ids": ["T1059.001"], "tactics": ["execution"], "source_genre": "incident_report"},
        )
        assert "PowerShell over-picked" in text
        assert "T1059.001" in text
        assert "execution" in text
        assert "incident_report" in text

    def test_embedding_text_no_applies_to(self):
        assert fp._embedding_text("just the prose", None) == "just the prose"

    def test_union_applies_to_merges_lists_keeps_genre(self):
        merged = fp._union_applies_to(
            {"technique_ids": ["T1059"], "source_genre": "incident_report"},
            {"technique_ids": ["T1105", "T1059"], "tactics": ["execution"]},
        )
        assert merged["technique_ids"] == ["T1059", "T1105"]
        assert merged["tactics"] == ["execution"]
        assert merged["source_genre"] == "incident_report"

    def test_union_concepts_dedups_preserves_order(self):
        assert fp._union_concepts(["a", "b"], ["b", "c"]) == ["a", "b", "c"]

    def test_pattern_anchor_keywords_from_text_and_keys(self):
        row = _row(pattern="certutil downloads a payload", applies_to={"technique_ids": ["T1105"]})
        kws = fp._pattern_anchor_keywords(row)
        assert "certutil" in kws

    def test_cosine_unit_vectors(self):
        assert pe.cosine([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == pytest.approx(1.0)
        assert pe.cosine([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == pytest.approx(0.0)
        assert pe.cosine(None, [1.0]) == 0.0
        assert pe.cosine([1.0], [1.0, 0.0]) == 0.0  # length mismatch


class TestSourceRepresentation:
    def test_build_representation_prefers_chunks(self):
        state = {
            "title": "Example Ransomware",
            "metadata": {"threat_actor": "Storm-2603"},
            "chunks": [{"text": "Exploited SharePoint then deployed encryptor"}],
            "validated_entities": [{"value": "w.exe", "entity_type": "file"}],
            "technique_mappings": {"c1": [{"technique_id": "T1486", "tactic": "impact"}]},
        }
        rep = pe.build_source_representation(state)
        assert "Example Ransomware" in rep
        assert "Storm-2603" in rep
        assert "SharePoint" in rep
        assert "file:w.exe" in rep
        assert "T1486" in rep

    def test_build_representation_falls_back_to_parsed_text(self):
        state = {"parsed_text": "raw source body about phishing", "chunks": []}
        rep = pe.build_source_representation(state)
        assert "phishing" in rep

    def test_extract_anchors_collects_structured_keys(self):
        state = {
            "validated_entities": [{"value": "1.2.3.4", "entity_type": "ipv4"}],
            "technique_mappings": {"c1": [{"technique_id": "T1059.001", "tactic": "execution"}]},
            "proposals_by_chunk": {"c1": {"proposed_techniques": ["T1105"], "tactics": ["command-and-control"]}},
        }
        anchors = pe.extract_source_anchors(state)
        assert anchors["technique_ids"] == {"T1059.001", "T1105"}
        assert "ipv4" in anchors["entity_types"]
        assert {"execution", "command-and-control"} <= anchors["tactics"]


# =============================================================================
# Hybrid scoring
# =============================================================================

class TestScorePattern:
    def test_semantic_relevance_outranks_high_occurrence_irrelevant(self):
        anchors = {"keywords": set(), "technique_ids": set(), "entity_types": set(), "tactics": set()}
        src = [1.0, 0.0, 0.0]
        relevant = _row(embedding=[1.0, 0.0, 0.0], occurrence_count=1)
        irrelevant_popular = _row(embedding=[0.0, 1.0, 0.0], occurrence_count=50)
        assert fp._score_pattern(relevant, src, anchors) > fp._score_pattern(irrelevant_popular, src, anchors)

    def test_structured_technique_overlap_boosts(self):
        anchors = {"keywords": set(), "technique_ids": {"T1059.001"}, "entity_types": set(), "tactics": set()}
        with_overlap = _row(embedding=None, applies_to={"technique_ids": ["T1059.001"]})
        without = _row(embedding=None, applies_to={"technique_ids": ["T1003"]})
        assert fp._score_pattern(with_overlap, None, anchors) >= 0.5
        assert fp._score_pattern(without, None, anchors) == pytest.approx(0.0)

    def test_null_embedding_lexical_fallback_beats_unrelated_embedded(self):
        # A NULL-embedding pattern that matches the source's technique anchor
        # should still outrank an embedded-but-orthogonal pattern.
        anchors = {"keywords": set(), "technique_ids": {"T1486"}, "entity_types": set(), "tactics": set()}
        src = [1.0, 0.0, 0.0]
        null_but_relevant = _row(embedding=None, applies_to={"technique_ids": ["T1486"]})
        embedded_unrelated = _row(embedding=[0.0, 1.0, 0.0], applies_to={})
        assert fp._score_pattern(null_but_relevant, src, anchors) > fp._score_pattern(embedded_unrelated, src, anchors)

    def test_salience_contributes(self):
        anchors = {"keywords": set(), "technique_ids": set(), "entity_types": set(), "tactics": set()}
        base = _row(embedding=None, salience=None)
        salient = _row(embedding=None, salience=2.0)
        delta = fp._score_pattern(salient, None, anchors) - fp._score_pattern(base, None, anchors)
        assert delta == pytest.approx(settings.feedback_salience_weight * 2.0)


# =============================================================================
# Technique anchors before extract_techniques runs
# =============================================================================
#
# The +0.5 shared-technique term is the largest in `_score_pattern`, and an
# audit measured it changing the injected set on ZERO of nine sources at every
# node. Cause: the three sources it read (`technique_mappings`, the review
# lane, `proposals_by_chunk`) are all written by `extract_techniques`, so the
# term was dead everywhere upstream of that node and dead inside it too,
# because it fetched feedback before its own propose step. These pin the two
# halves of the repair.

def _section(text: str, classification: str = "technique_reference") -> dict:
    return {"section_id": "s1", "classification": classification, "text": text}


class TestTechniqueAnchorsAtEarlyStages:
    def test_vendor_mapping_table_puts_technique_ids_in_play(self):
        state = {"classified_sections": [
            _section("T1204.004 Malicious Copy and Paste | T1059.001 PowerShell"),
        ]}
        assert pe._technique_ids_in_play(state) == {"T1204.004", "T1059.001"}

    def test_only_technique_reference_sections_count(self):
        # A narrative passage that happens to mention a T-ID is not the
        # vendor's considered mapping, and must not become an anchor.
        state = {"classified_sections": [
            _section("we saw T1486 here", classification="behavioral"),
        ]}
        assert pe._technique_ids_in_play(state) == set()

    def test_no_mapping_table_yields_nothing(self):
        # Common and correct: plenty of reports ship no ATT&CK table.
        assert pe._technique_ids_in_play({"classified_sections": []}) == set()
        assert pe._technique_ids_in_play({}) == set()

    def test_vendor_ids_union_with_proposals(self):
        state = {
            "classified_sections": [_section("T1204.004")],
            "proposals_by_chunk": {"chk-1": {"proposed_techniques": ["T1059.001"]}},
        }
        assert pe._technique_ids_in_play(state) == {"T1204.004", "T1059.001"}

    def test_boost_now_separates_patterns_at_entity_stage(self):
        """The headline: the term must actually change the ranking.

        State as `extract_entities` sees it — no entities, no mappings, no
        proposals, only what `classify_sections` left behind. Before the fix
        both patterns scored identically here, which is what "dead term" means.
        """
        state = {
            "parsed_text": "some report body",
            "classified_sections": [_section("T1486 Data Encrypted for Impact")],
        }
        anchors = pe.extract_source_anchors(state)
        assert anchors["technique_ids"] == {"T1486"}

        matching = _row(embedding=None, applies_to={"technique_ids": ["T1486"]})
        other = _row(embedding=None, applies_to={"technique_ids": ["T1003"]})
        assert fp._score_pattern(matching, None, anchors) > fp._score_pattern(other, None, anchors)

    def test_entity_types_stays_empty_at_entity_stage(self):
        """Not a gap to close later — `extract_entities` produces the entities,
        so nothing can name their types beforehand. Pinned so a future reader
        doesn't 'fix' it by synthesising a guess."""
        anchors = pe.extract_source_anchors({"parsed_text": "body", "classified_sections": []})
        assert anchors["entity_types"] == set()


# =============================================================================
# persist_pattern dedup tiers
# =============================================================================

class TestPersistPattern:
    async def test_exact_match_bumps_and_enriches(self):
        existing = _row(occurrence_count=1, applies_to={"technique_ids": ["T1059"]}, embedding=None)
        sess = _FakeSession([_FakeResult(scalar=existing)])
        with patch.object(pe, "embed_text", return_value=[1.0, 0.0]):
            row, created = await fp.persist_pattern(
                sess, source_id=None, category="wrong_technique",
                pattern=existing.pattern, applies_to={"tactics": ["execution"]},
                concepts=["powershell"],
            )
        assert created is False
        assert row is existing
        assert row.occurrence_count == 2
        assert row.applies_to["technique_ids"] == ["T1059"]
        assert row.applies_to["tactics"] == ["execution"]
        assert row.embedding == [1.0, 0.0]  # backfilled
        assert row.embedding_model == settings.embedding_model

    async def test_semantic_near_duplicate_merges(self):
        near = _row(pattern="a near-identical rule", embedding=[1.0, 0.0, 0.0], occurrence_count=3)
        sess = _FakeSession([
            _FakeResult(scalar=None),       # Tier-1 exact miss
            _FakeResult(items=[near]),      # Tier-2 candidates
        ])
        # Clears BOTH bars: cosine 1.0 >= 0.55, and the two texts share
        # "identical"/"rule" for a token overlap of 0.33 >= 0.20.
        with patch.object(pe, "embed_text", return_value=[1.0, 0.0, 0.0]):
            row, created = await fp.persist_pattern(
                sess, source_id=None, category="wrong_technique",
                pattern="a differently-worded but identical rule",
            )
        assert created is False
        assert row is near
        assert row.occurrence_count == 4
        assert sess.added == []  # nothing inserted

    async def test_high_cosine_but_low_overlap_does_not_merge(self):
        """The case a cosine-only threshold cannot get right.

        Measured on the real corpus: a true duplicate scored 0.583
        and a genuinely distinct pair 0.582. No single cosine bar separates
        those, so token overlap is required as well. A false merge silently
        deletes one rule's guidance, which is the expensive error here — hence
        the conjunction, biased toward leaving both rows in place.
        """
        other = _row(pattern="hash strings truncated by an ellipsis are unusable",
                     embedding=[1.0, 0.0, 0.0], occurrence_count=3)
        sess = _FakeSession([
            _FakeResult(scalar=None),
            _FakeResult(items=[other]),
        ])
        with patch.object(pe, "embed_text", return_value=[1.0, 0.0, 0.0]):
            row, created = await fp.persist_pattern(
                sess, source_id=None, category="orphan_ioc",
                pattern="obfuscated placeholder domains from redacted screenshots",
            )
        assert created is True, "cosine 1.0 must not merge on disjoint wording"
        assert other.occurrence_count == 3, "the existing row must be untouched"
        assert len(sess.added) == 1

    async def test_semantic_below_threshold_inserts_new(self):
        far = _row(pattern="unrelated rule", embedding=[0.0, 1.0, 0.0], occurrence_count=3)
        sess = _FakeSession([
            _FakeResult(scalar=None),
            _FakeResult(items=[far]),
        ])
        with patch.object(pe, "embed_text", return_value=[1.0, 0.0, 0.0]):  # cosine 0.0 < 0.90
            row, created = await fp.persist_pattern(
                sess, source_id=None, category="wrong_technique",
                pattern="a genuinely new rule",
                applies_to={"technique_ids": ["T1486"]},
            )
        assert created is True
        assert len(sess.added) == 1
        assert sess.added[0].embedding == [1.0, 0.0, 0.0]
        assert sess.added[0].embedding_model == settings.embedding_model

    async def test_insert_when_embedding_unavailable(self):
        sess = _FakeSession([_FakeResult(scalar=None)])  # no Tier-2 query when embedding is None
        with patch.object(pe, "embed_text", return_value=None):
            row, created = await fp.persist_pattern(
                sess, source_id=None, category="wrong_technique", pattern="rule with no embedding",
            )
        assert created is True
        assert sess.added[0].embedding is None
        assert sess.added[0].embedding_model is None


# =============================================================================
# Hybrid relevance retrieval
# =============================================================================

class TestRelevantFeedbackAddendum:
    async def test_ranks_relevant_over_popular_irrelevant(self):
        relevant = _row(id=uuid.uuid4(), pattern="relevant", embedding=[1.0, 0.0, 0.0], occurrence_count=1)
        popular = _row(id=uuid.uuid4(), pattern="popular but off-topic", embedding=[0.0, 1.0, 0.0], occurrence_count=99)
        # Two queries now: relevance pool, then pinned (empty here).
        sess = _FakeSession([_FakeResult(items=[popular, relevant]), _FakeResult(items=[])])
        state = {"parsed_text": "topic body", "chunks": []}
        with patch.object(fp, "async_session", lambda: _FakeSessionCM(sess)), \
             patch.object(pe, "embed_text", return_value=[1.0, 0.0, 0.0]):
            text, ids = await fp.relevant_feedback_addendum(
                state, categories=("wrong_technique",), limit=15,
            )
        assert ids[0] == str(relevant.id)
        assert "relevant" in text

    async def test_empty_candidates_returns_blank(self):
        sess = _FakeSession([_FakeResult(items=[]), _FakeResult(items=[])])
        with patch.object(fp, "async_session", lambda: _FakeSessionCM(sess)), \
             patch.object(pe, "embed_text", return_value=[1.0]):
            text, ids = await fp.relevant_feedback_addendum(
                {"parsed_text": "x"}, categories=("wrong_technique",),
            )
        assert text == ""
        assert ids == []

    async def test_promoted_to_prompt_is_pinned(self):
        # A pinned rule is ALWAYS surfaced (under PERMANENT RULES) alongside the
        # relevance pool — exempt from the top-N cut.
        candidate = _row(id=uuid.uuid4(), pattern="maybe relevant", embedding=[1.0, 0.0, 0.0])
        pinned = _row(id=uuid.uuid4(), pattern="always treat info@ emails as defender contacts",
                      status="promoted_to_prompt", category="defender_ioc")
        sess = _FakeSession([_FakeResult(items=[candidate]), _FakeResult(items=[pinned])])
        with patch.object(fp, "async_session", lambda: _FakeSessionCM(sess)), \
             patch.object(pe, "embed_text", return_value=[1.0, 0.0, 0.0]):
            text, ids = await fp.relevant_feedback_addendum(
                {"parsed_text": "body", "chunks": []}, categories=("wrong_technique", "defender_ioc"),
            )
        assert str(pinned.id) in ids               # surfaced (for closed-loop scoring)
        assert ids[0] == str(pinned.id)            # pinned listed first
        assert "PERMANENT RULES" in text
        assert "always treat info@ emails as defender contacts" in text
        assert "RECENT ANALYST FEEDBACK" in text   # relevance pool still rendered

    async def test_pinned_only_no_relevance_pool(self):
        pinned = _row(id=uuid.uuid4(), pattern="permanent rule", status="promoted_to_prompt")
        sess = _FakeSession([_FakeResult(items=[]), _FakeResult(items=[pinned])])
        with patch.object(fp, "async_session", lambda: _FakeSessionCM(sess)), \
             patch.object(pe, "embed_text", return_value=[1.0]):
            text, ids = await fp.relevant_feedback_addendum(
                {"parsed_text": "x"}, categories=("wrong_technique",),
            )
        assert ids == [str(pinned.id)]
        assert "PERMANENT RULES" in text
        assert "RECENT ANALYST FEEDBACK" not in text  # nothing in the advisory lane

    async def test_failure_is_best_effort(self):
        def boom():
            raise RuntimeError("db down")
        with patch.object(fp, "async_session", boom):
            text, ids = await fp.relevant_feedback_addendum(
                {"parsed_text": "x"}, categories=("wrong_technique",),
            )
        assert text == ""
        assert ids == []


# =============================================================================
# Source-fingerprint cache
# =============================================================================

class TestRelevantAddendumCache:
    async def test_same_source_hits_cache(self):
        state = {"title": "Report A", "parsed_text": "alpha body", "chunks": []}
        with patch.object(
            fp, "relevant_feedback_addendum",
            new=AsyncMock(return_value=("ADDENDUM-A", ["id1"])),
        ) as m:
            t1 = await fp.relevant_addendum_cached(state, categories=("wrong_technique",), node="extract_techniques")
            t2 = await fp.relevant_addendum_cached(state, categories=("wrong_technique",), node="extract_techniques")
        assert t1 == t2 == "ADDENDUM-A"
        assert m.await_count == 1  # second call served from cache

    async def test_distinct_sources_get_distinct_entries(self):
        state_a = {"title": "Report A", "parsed_text": "alpha body", "chunks": []}
        state_b = {"title": "Report B", "parsed_text": "beta body", "chunks": []}
        with patch.object(
            fp, "relevant_feedback_addendum",
            new=AsyncMock(side_effect=[("A", ["a"]), ("B", ["b"])]),
        ) as m:
            ta = await fp.relevant_addendum_cached(state_a, categories=("wrong_technique",), node="extract_techniques")
            tb = await fp.relevant_addendum_cached(state_b, categories=("wrong_technique",), node="extract_techniques")
        assert ta == "A"
        assert tb == "B"
        assert m.await_count == 2

    async def test_clear_feedback_cache_forces_refetch(self):
        state = {"title": "Report A", "parsed_text": "alpha body", "chunks": []}
        with patch.object(
            fp, "relevant_feedback_addendum",
            new=AsyncMock(return_value=("X", ["id1"])),
        ) as m:
            await fp.relevant_addendum_cached(state, categories=("wrong_technique",), node="extract_techniques")
            fp.clear_feedback_cache()
            await fp.relevant_addendum_cached(state, categories=("wrong_technique",), node="extract_techniques")
        assert m.await_count == 2

    async def test_distinct_node_labels_do_not_share_a_cache_entry(self):
        """`extract_techniques` fetches twice — before and after its propose
        step — so the second sees T-IDs the first could not. Both calls carry
        the same categories and limit, so only the node label separates them."""
        state = {"title": "Report A", "parsed_text": "alpha body", "chunks": []}
        with patch.object(
            fp, "relevant_feedback_addendum",
            new=AsyncMock(side_effect=[("PROPOSE", ["a"]), ("PICK", ["b"])]),
        ) as m:
            first = await fp.relevant_addendum_cached(
                state, categories=("wrong_technique",), node="extract_techniques")
            second = await fp.relevant_addendum_cached(
                state, categories=("wrong_technique",), node="extract_techniques:pick")
        assert first == "PROPOSE"
        assert second == "PICK"
        assert m.await_count == 2

    async def test_pick_refetch_survives_an_unchanged_fingerprint(self):
        """The subtle one. If the propose step surfaces no technique the
        vendor's own mapping table had not already named, the fingerprint is
        IDENTICAL across both fetches. Keyed on the fingerprint alone the
        second call hit the cache and returned the pre-propose addendum — a
        no-op that still looked like a working fix."""
        pre = {"title": "A", "parsed_text": "body",
               "classified_sections": [
                   {"section_id": "s1", "classification": "technique_reference",
                    "text": "T1059.001"}]}
        post = {**pre, "proposals_by_chunk": {"chk-1": {"proposed_techniques": ["T1059.001"]}}}
        assert fp._source_fingerprint(pre) == fp._source_fingerprint(post)  # the trap

        with patch.object(
            fp, "relevant_feedback_addendum",
            new=AsyncMock(side_effect=[("PROPOSE", ["a"]), ("PICK", ["b"])]),
        ) as m:
            await fp.relevant_addendum_cached(
                pre, categories=("wrong_technique",), node="extract_techniques")
            second = await fp.relevant_addendum_cached(
                post, categories=("wrong_technique",), node="extract_techniques:pick")
        assert second == "PICK"
        assert m.await_count == 2

    def test_fingerprint_stable_and_distinct(self):
        a1 = fp._source_fingerprint({"title": "A", "parsed_text": "body-a"})
        a2 = fp._source_fingerprint({"title": "A", "parsed_text": "body-a"})
        b = fp._source_fingerprint({"title": "B", "parsed_text": "body-b"})
        assert a1 == a2
        assert a1 != b


# =============================================================================
# Denylist (deterministic guardrail behind promoted_to_denylist)
# =============================================================================

class TestNormalizeDenylistTerms:
    def test_dedups_trims_and_validates(self):
        out = fp.normalize_denylist_terms({
            "values": ["  Foo ", "foo", "x" * 300, "", "  ", "Bar"],
            "technique_ids": ["t1059.001", "T1059.001", "bogus", "T1486", "t1486"],
        })
        # 'foo' deduped against 'Foo' (case-insensitive, original casing kept);
        # over-long + empty dropped.
        assert out["values"] == ["Foo", "Bar"]
        # uppercased + shape-validated + deduped; 'bogus' dropped.
        assert out["technique_ids"] == ["T1059.001", "T1486"]

    def test_drops_control_chars(self):
        out = fp.normalize_denylist_terms({"values": ["clean", "ctrl\x01x"]})
        assert out["values"] == ["clean"]

    def test_none_and_missing_keys(self):
        empty = {"values": [], "technique_ids": [], "entity_types": []}
        assert fp.normalize_denylist_terms(None) == empty
        assert fp.normalize_denylist_terms({}) == empty

    def test_entity_types_lowercased_and_deduped(self):
        out = fp.normalize_denylist_terms({
            "values": ["google"], "entity_types": ["Organization", "organization", "TOOL"],
        })
        assert out["entity_types"] == ["organization", "tool"]


class TestDenylistMatch:
    def test_value_match_case_insensitive_trimmed(self):
        dl = {"values": {"info@cert.example": {"pattern_id": "p1"}}, "technique_ids": {}}
        assert fp.denylist_match_value("  INFO@CERT.EXAMPLE ", dl)["pattern_id"] == "p1"
        assert fp.denylist_match_value("other", dl) is None
        assert fp.denylist_match_value("", dl) is None

    def test_technique_match_upper(self):
        dl = {"values": {}, "technique_ids": {"T1204.004": {"pattern_id": "p2"}}}
        assert fp.denylist_match_technique("t1204.004", dl)["pattern_id"] == "p2"
        assert fp.denylist_match_technique("T9999", dl) is None
        assert fp.denylist_match_technique("", dl) is None

    def test_value_match_respects_entity_type_scope(self):
        # "google" is denylisted only when it's an organization.
        dl = {"values": {"google": {"pattern_id": "p1", "entity_types": {"organization"}}},
              "technique_ids": {}}
        assert fp.denylist_match_value("google", dl, "organization") is not None
        assert fp.denylist_match_value("google", dl, "ioc_domain") is None
        # No type supplied against a scoped entry -> conservative no-match.
        assert fp.denylist_match_value("google", dl) is None

    def test_value_match_unscoped_matches_any_type(self):
        # Entry with no entity_types scope (or empty) matches any type.
        dl = {"values": {"x@y.com": {"pattern_id": "p1", "entity_types": None}},
              "technique_ids": {}}
        assert fp.denylist_match_value("x@y.com", dl, "ioc_email") is not None
        assert fp.denylist_match_value("x@y.com", dl, "anything") is not None


class TestLoadDenylist:
    async def test_aggregates_promoted_patterns(self):
        r1 = _row(id=uuid.uuid4(), pattern="emails are defender contacts",
                  denylist_terms={"values": ["a@b.com"], "technique_ids": []})
        r2 = _row(id=uuid.uuid4(), pattern="drop this technique",
                  denylist_terms={"values": [], "technique_ids": ["T1204.004"]})
        sess = _FakeSession([_FakeResult(items=[r1, r2])])
        with patch.object(fp, "async_session", lambda: _FakeSessionCM(sess)):
            dl = await fp.load_denylist()
        assert "a@b.com" in dl["values"]
        assert dl["values"]["a@b.com"]["pattern_id"] == str(r1.id)
        assert dl["values"]["a@b.com"]["entity_types"] is None  # unscoped
        assert "T1204.004" in dl["technique_ids"]

    async def test_value_scope_loaded_and_merged(self):
        # Same value denylisted by two patterns with different type scopes ->
        # the loaded scope is the union. A third unscoped pattern would widen
        # it to None (any type); here both are scoped so we get the union.
        r1 = _row(id=uuid.uuid4(), pattern="org noise",
                  denylist_terms={"values": ["google"], "entity_types": ["organization"]})
        r2 = _row(id=uuid.uuid4(), pattern="tool noise",
                  denylist_terms={"values": ["google"], "entity_types": ["tool"]})
        sess = _FakeSession([_FakeResult(items=[r1, r2])])
        with patch.object(fp, "async_session", lambda: _FakeSessionCM(sess)):
            dl = await fp.load_denylist()
        assert dl["values"]["google"]["entity_types"] == {"organization", "tool"}

    async def test_cached_within_ttl(self):
        r1 = _row(denylist_terms={"values": ["x@y.com"], "technique_ids": []})
        calls = {"n": 0}

        def make_cm():
            calls["n"] += 1
            return _FakeSessionCM(_FakeSession([_FakeResult(items=[r1])]))

        with patch.object(fp, "async_session", make_cm):
            await fp.load_denylist()
            await fp.load_denylist()  # served from cache
        assert calls["n"] == 1

    async def test_best_effort_on_error(self):
        def boom():
            raise RuntimeError("db down")
        with patch.object(fp, "async_session", boom):
            dl = await fp.load_denylist()
        assert dl == {"values": {}, "technique_ids": {}}

    async def test_defanged_value_matches_refanged_entity(self):
        """A denylist value entered defanged matches the refanged entity value
        that extract_entities produces (load_denylist refangs both sides)."""
        r = _row(denylist_terms={"values": ["evil[.]com"], "technique_ids": []})
        sess = _FakeSession([_FakeResult(items=[r])])
        with patch.object(fp, "async_session", lambda: _FakeSessionCM(sess)):
            dl = await fp.load_denylist()
        assert "evil.com" in dl["values"]               # key stored refanged
        assert fp.denylist_match_value("evil.com", dl) is not None    # refanged entity
        assert fp.denylist_match_value("evil[.]com", dl) is not None  # defanged query too

    async def test_clear_feedback_cache_forces_reload(self):
        r1 = _row(denylist_terms={"values": ["x@y.com"], "technique_ids": []})
        calls = {"n": 0}

        def make_cm():
            calls["n"] += 1
            return _FakeSessionCM(_FakeSession([_FakeResult(items=[r1])]))

        with patch.object(fp, "async_session", make_cm):
            await fp.load_denylist()
            fp.clear_feedback_cache()
            await fp.load_denylist()
        assert calls["n"] == 2


class TestPromoteDenylist:
    async def test_denylist_stores_normalized_terms(self):
        row = _row(category="defender_ioc")
        sess = _FakeSession([_FakeResult(scalar=row)])
        result = await fp.promote_pattern(
            sess, row.id, action="denylist", by="analyst",
            denylist_terms={"values": ["A@b.com", "a@b.com"], "technique_ids": ["t1204.004"]},
        )
        assert result.status == "promoted_to_denylist"
        assert result.denylist_terms == {
            "values": ["A@b.com"], "technique_ids": ["T1204.004"], "entity_types": [],
        }
        assert result.promoted_action.startswith("denylist:")
        assert "A@b.com" in result.promoted_action
        assert result.promoted_by == "analyst"
        assert sess.commits == 1

    async def test_denylist_empty_terms_is_advisory(self):
        row = _row()
        sess = _FakeSession([_FakeResult(scalar=row)])
        result = await fp.promote_pattern(sess, row.id, action="denylist", by="x", denylist_terms=None)
        assert result.status == "promoted_to_denylist"
        assert result.denylist_terms == {"values": [], "technique_ids": [], "entity_types": []}
        assert "advisory" in result.promoted_action

    async def test_prompt_action_clears_stale_terms(self):
        row = _row(denylist_terms={"values": ["stale"], "technique_ids": []})
        sess = _FakeSession([_FakeResult(scalar=row)])
        result = await fp.promote_pattern(sess, row.id, action="prompt", by="x")
        assert result.status == "promoted_to_prompt"
        assert result.denylist_terms == {}
        assert result.promoted_action.startswith("prompt:")

    async def test_unknown_pattern_returns_none(self):
        sess = _FakeSession([_FakeResult(scalar=None)])
        result = await fp.promote_pattern(sess, uuid.uuid4(), action="denylist", by="x")
        assert result is None

    def test_pattern_to_dict_includes_denylist_terms(self):
        row = _row(denylist_terms={"values": ["x"], "technique_ids": ["T1486"]})
        d = fp.pattern_to_dict(row)
        assert d["denylist_terms"] == {"values": ["x"], "technique_ids": ["T1486"]}
