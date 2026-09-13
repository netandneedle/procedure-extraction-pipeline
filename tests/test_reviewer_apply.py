"""Tests for the auto-apply converters.

These sit on the seam between what the reviewer said and what the pipeline is
told, with no human in between. Everything a human would have caught by reading
the panel before clicking has to be caught here instead.

The load-bearing test is `TestConvertedItemsAreAcceptable`: it feeds each
converter's real output through the real submit schema. A field the converter
invents, misnames, or types wrongly fails there rather than at 2am inside an
unattended run.
"""

from app.schemas.api import (
    AddedChunkItem,
    AddedEntityItem,
    ChunkDecisionItem,
    ChunkEdgeMutation,
    Gate0ReviewItem,
    Gate1PromotionItem,
    Gate1ReviewItem,
    Gate2ReviewItem,
)
from app.services.reviewer.apply import (
    AUTO_APPLIERS,
    apply_bundle,
    apply_chunks,
    apply_entities,
    apply_procedures,
    build_auto_submission,
)


def _rec(**kw):
    """A recommendation with the fields every one of them carries."""
    base = {
        "confidence": "high",
        "rationale": "The report states this plainly.",
        "evidence_quote": "certutil.exe -urlcache -f",
        "quote_source_support": 0.9,
        "quote_unsupported": False,
    }
    base.update(kw)
    return base


# ── Rule 1: sparse ──────────────────────────────────────────────────────

class TestSparse:
    """Emit decisions ONLY for what the reviewer spoke about.

    Not a size optimisation. Gate 0 removes a denylisted entity only when no
    explicit decision arrives for it, so an approve-everything submission
    silently defeats the denylist — which is exactly the defect the analyst
    UI shipped with. Unattended, nobody would ever notice.
    """

    def test_entities_says_nothing_about_unmentioned_entities(self):
        out = apply_entities({"entities": [_rec(entity_id="e1", action="remove")]}, {})
        assert [r["entity_id"] for r in out.channels["gate0_reviews"]] == ["e1"]

    def test_empty_payload_produces_empty_channels_not_blanket_approval(self):
        for fn in (apply_entities, apply_chunks, apply_procedures, apply_bundle):
            out = fn({}, {})
            assert out.decisions == [], fn.__name__
            for value in out.channels.values():
                # chunk gate nests its channel in a dict of lists
                if isinstance(value, dict):
                    assert all(not v for v in value.values()), fn.__name__
                else:
                    assert value == [], fn.__name__

    def test_items_without_an_id_or_action_are_dropped(self):
        out = apply_entities({
            "entities": [
                _rec(entity_id="", action="remove"),
                _rec(entity_id="e2"),
                _rec(entity_id="e3", action="approve"),
            ],
        }, {})
        assert [r["entity_id"] for r in out.channels["gate0_reviews"]] == ["e3"]


# ── Rule 2: confidence is a word here, a number there ───────────────────

class TestConfidenceMapping:
    def test_added_entity_confidence_becomes_a_float(self):
        out = apply_entities({
            "added_entities": [
                _rec(value="evil.test", entity_type="ioc_domain", confidence="medium"),
            ],
        }, {})
        assert out.channels["gate0_added_entities"][0]["confidence"] == 0.7

    def test_added_chunk_confidence_lands_on_behavioral_confidence(self):
        out = apply_chunks({
            "added_chunks": [_rec(text="Delete shadow copies.", confidence="low")],
        }, {})
        added = out.channels["chunk_reviews"]["added_chunks"][0]
        assert added["behavioral_confidence"] == 0.5
        assert "confidence" not in added

    def test_an_unknown_confidence_falls_back_rather_than_crashing(self):
        out = apply_chunks({"added_chunks": [_rec(text="x", confidence="wat")]}, {})
        assert out.channels["chunk_reviews"]["added_chunks"][0][
            "behavioral_confidence"
        ] == 0.7


# ── Rule 3: reviewer-only fields stay out of pipeline state ─────────────

class TestReviewerOnlyFieldsAreStripped:
    def test_grounding_fields_never_reach_the_gate(self):
        out = apply_entities({
            "entities": [_rec(entity_id="e1", action="approve")],
            "added_entities": [_rec(value="v", entity_type="malware")],
        }, {})
        leaked = {"evidence_quote", "quote_source_support", "quote_unsupported"}
        for item in (out.channels["gate0_reviews"]
                     + out.channels["gate0_added_entities"]):
            assert not (leaked & set(item)), item

    def test_rationale_is_truncated_to_what_the_schema_accepts(self):
        out = apply_entities({
            "added_entities": [
                _rec(value="v", entity_type="malware", rationale="x" * 900),
            ],
        }, {})
        assert len(out.channels["gate0_added_entities"][0]["rationale"]) == 500


# ── The chunk gate's shape transforms ───────────────────────────────────

class TestChunkShapes:
    def test_flat_edits_are_nested_under_edits(self):
        out = apply_chunks({
            "chunks": [_rec(
                chunk_id="chk-1", action="edit",
                edited_text="Tighter text.", edited_source_excerpt="verbatim",
            )],
        }, {})
        d = out.channels["chunk_reviews"]["decisions"][0]
        assert d["edits"] == {"text": "Tighter text.", "source_excerpt": "verbatim"}
        assert "edited_text" not in d

    def test_approve_carries_no_empty_edits_key(self):
        out = apply_chunks({"chunks": [_rec(chunk_id="chk-1", action="approve")]}, {})
        assert "edits" not in out.channels["chunk_reviews"]["decisions"][0]

    def test_merge_targets_survive(self):
        out = apply_chunks({
            "chunks": [_rec(chunk_id="chk-1", action="merge", merge_with=["chk-2"])],
        }, {})
        assert out.channels["chunk_reviews"]["decisions"][0]["merge_with"] == ["chk-2"]

    def test_edges_use_the_python_safe_name_the_gate_reads(self):
        # The submit route dumps by_alias=False, so state carries `from_`.
        # Emitting the wire alias `from` would land a key nothing looks for.
        out = apply_chunks({
            "edges": [_rec(action="add", from_chunk_id="a", to_chunk_id="b")],
        }, {})
        assert out.channels["chunk_reviews"]["edges"] == [
            {"action": "add", "from_": "a", "to": "b"},
        ]


# ── Rewinds defer ───────────────────────────────────────────────────────

class TestRewindsAreFlaggedForTheCaller:
    """A rewind throws a whole pass away and re-enters an upstream node.

    It used to defer unconditionally here, because nothing bounded the loop.
    Now there IS a bound — _MAX_AUTO_REWIND_PASSES, enforced in _auto_apply,
    which can count the gate's prior visits and this module cannot. So the
    rewind converts, and carries `rewind=True` so the caller can refuse it.

    The invariant has NOT gone away, it moved: a rewind must never be applied
    without something checking the limit. TestAutoApplyEnforcesThePassLimit
    is where that now lives. What this class guards is that the rewind is
    neither silently dropped nor silently applied — it arrives, and it is
    labelled.
    """

    def test_chunk_rerun_converts_and_is_flagged(self):
        out = apply_chunks({
            "chunks": [_rec(chunk_id="chk-1", action="drop")],
            "reject": _rec(reason="over_chunked", comments="merge 2 and 3"),
        }, {})
        assert out.rewind is True
        assert out.channels["chunk_reviews"]["reject"]["reason"] == "over_chunked"

    def test_a_rejected_draft_converts_and_is_flagged(self):
        out = apply_procedures({
            "drafts": [
                _rec(draft_id="d1", action="approve"),
                _rec(draft_id="d2", action="reject", reject_reason="wrong_technique"),
            ],
        }, {})
        assert out.rewind is True
        actions = {i["draft_id"]: i["action"] for i in out.channels["gate1_reviews"]}
        assert actions == {"d1": "approve", "d2": "reject"}

    def test_ordinary_decisions_do_not_defer(self):
        out = apply_procedures({
            "drafts": [
                _rec(draft_id="d1", action="approve"),
                _rec(draft_id="d2", action="remove"),
            ],
        }, {})
        assert out.rewind is False
        assert len(out.channels["gate1_reviews"]) == 2


# ── The contract that matters ───────────────────────────────────────────

class TestConvertedItemsAreAcceptable:
    """Every item a converter emits must validate against the submit schema.

    Field-name comparison is not enough: `AddedEntityItem.confidence` is a
    float while the recommendation's is "high"/"medium"/"low", so a converter
    that passed the field straight through would match by name and fail by
    type — inside an unattended run, at the point where nobody is looking.
    """

    def test_gate0_items_validate(self):
        out = apply_entities({
            "entities": [_rec(
                entity_id="e1", action="edit",
                edited_value="APT29", edited_type="intrusion_set",
            )],
            "added_entities": [_rec(
                value="evil.test", entity_type="ioc_domain", confidence="medium",
            )],
        }, {})
        for item in out.channels["gate0_reviews"]:
            Gate0ReviewItem(**item)
        for item in out.channels["gate0_added_entities"]:
            AddedEntityItem(**item)

    def test_chunk_items_validate(self):
        out = apply_chunks({
            "chunks": [_rec(chunk_id="chk-1", action="edit", edited_text="t")],
            "added_chunks": [_rec(text="A missed procedure.", source_excerpt="q")],
            "edges": [_rec(action="add", from_chunk_id="a", to_chunk_id="b")],
        }, {})
        sub = out.channels["chunk_reviews"]
        for item in sub["decisions"]:
            ChunkDecisionItem(**item)
        for item in sub["added_chunks"]:
            AddedChunkItem(**item)
        for item in sub["edges"]:
            ChunkEdgeMutation(**item)

    def test_gate1_items_validate(self):
        out = apply_procedures({
            "drafts": [_rec(draft_id="d1", action="approve")],
            "promotions": [_rec(chunk_id="chk-1", technique_id="T1059.001")],
        }, {})
        for item in out.channels["gate1_reviews"]:
            Gate1ReviewItem(**item)
        for item in out.channels["gate1_promotions"]:
            Gate1PromotionItem(**item)

    def test_gate2_items_validate(self):
        out = apply_bundle({
            "relationships": [_rec(
                rel_id="r1", action="edit", edited_rel_type="uses",
            )],
        }, {})
        for item in out.channels["gate2_reviews"]:
            Gate2ReviewItem(**item)


class TestApplierRegistry:
    def test_every_gate_with_a_reviewer_can_run_unattended(self):
        """A gate offered in `auto` with no applier would silently fall through
        to a human forever — the setting would look accepted and do nothing."""
        from app.services.reviewer.gate_reviewers import REVIEWERS

        assert set(AUTO_APPLIERS) == set(REVIEWERS)

    def test_an_unknown_gate_returns_none_rather_than_raising(self):
        assert build_auto_submission("no-such-gate", {}, {}) is None


class TestTechniqueRemovals:
    """`remove_technique_ids` -> the whole-list edit the gate actually applies.

    Regression: a live unattended run had the reviewer ask to drop four
    techniques, and the converter's whitelist silently dropped the field
    instead. The only trace was an auto outcome reading 8/12 — the agent
    recorded as disagreeing with itself, which is the one shape a
    self-submitted outcome should never take.
    """

    STATE = {"drafts": [{
        "draft_id": "d1",
        "techniques": [
            {"technique_id": "T1059.001", "tactic": "execution"},
            {"technique_id": "T1027", "tactic": "defense-evasion"},
            {"technique_id": "T1105", "tactic": "command-and-control"},
        ],
    }]}

    def test_removals_become_a_surviving_technique_list(self):
        out = apply_procedures(
            {"drafts": [_rec(
                draft_id="d1", action="edit",
                remove_technique_ids=["T1059.001", "T1027"],
            )]},
            self.STATE,
        )
        item = out.channels["gate1_reviews"][0]
        assert [t["technique_id"] for t in item["analyst_edits"]["techniques"]] \
            == ["T1105"]
        Gate1ReviewItem(**item)

    def test_a_removal_forces_the_action_to_edit(self):
        # The gate reads analyst_edits only on an edit, so an approve would
        # discard the removal exactly as the original bug did.
        out = apply_procedures(
            {"drafts": [_rec(
                draft_id="d1", action="approve", remove_technique_ids=["T1027"],
            )]},
            self.STATE,
        )
        assert out.channels["gate1_reviews"][0]["action"] == "edit"

    def test_matching_is_case_insensitive_and_whitespace_tolerant(self):
        out = apply_procedures(
            {"drafts": [_rec(
                draft_id="d1", action="edit", remove_technique_ids=[" t1027 "],
            )]},
            self.STATE,
        )
        kept = out.channels["gate1_reviews"][0]["analyst_edits"]["techniques"]
        assert "T1027" not in [t["technique_id"] for t in kept]

    def test_no_removals_means_no_edit_key(self):
        out = apply_procedures(
            {"drafts": [_rec(draft_id="d1", action="approve")]}, self.STATE,
        )
        assert "analyst_edits" not in out.channels["gate1_reviews"][0]

    def test_an_unknown_draft_is_skipped_not_guessed(self):
        out = apply_procedures(
            {"drafts": [_rec(
                draft_id="ghost", action="edit", remove_technique_ids=["T1027"],
            )]},
            self.STATE,
        )
        # No draft to filter against, so no technique list is invented.
        assert "analyst_edits" not in out.channels["gate1_reviews"][0]


class TestEveryConverterProducesSomethingApplicable:
    """No converter hands back an empty submission any more.

    This class used to pin "a defer carries a reason AND empty channels",
    guarding an ordering bug where _auto_apply read the channels first and
    threw the reason away. Converters no longer defer at all — a rewind now
    converts and is flagged — so the guard has moved to the caller, where the
    remaining hand-offs live (TestAutoApplyEnforcesThePassLimit).

    What is still worth pinning here is the property that made the old bug
    possible: a converter that returns nothing is indistinguishable from one
    that crashed. Every converter must produce channels.
    """

    def test_no_converter_returns_empty_channels(self):
        outs = [
            apply_chunks({"reject": _rec(reason="over_chunked", comments="")}, {}),
            apply_procedures(
                {"drafts": [_rec(
                    draft_id="d1", action="reject",
                    reject_reason="wrong_technique",
                )]},
                {},
            ),
            apply_procedures({"drafts": [_rec(draft_id="d1", action="approve")]}, {}),
        ]
        for out in outs:
            assert out.channels, "an empty submission reads as a crash"


class TestAutoApplyEnforcesThePassLimit:
    """The caller's half: a rewind is allowed once, then a person decides.

    This is where the termination guarantee now lives. The converter flags a
    rewind; only _auto_apply can count how many times this gate has already
    been visited, so only it can refuse one.

    It also pins the older contract this class replaced: every hand-off says
    WHY in the log. An earlier ordering bug computed a deferral reason and
    then discarded it, leaving a deliberate hand-off looking exactly like a
    reviewer crash — a gate that paused with nothing logged.

    The refusing path touches neither the graph nor the database, so it can
    be called with a None graph.
    """

    import pytest

    @pytest.mark.asyncio
    async def test_a_rewind_over_the_limit_is_refused_and_says_why(
        self, caplog, monkeypatch,
    ):
        import logging

        from app.api.routes import pipeline as mod

        # The gate has already been reviewed twice: this rewind would be the
        # third pass, past the one retry allowed unattended.
        async def _already_twice(source_id, gate_key):
            return 2

        monkeypatch.setattr(mod, "gate_pass_count", _already_twice)

        payload = {"drafts": [{
            "draft_id": "d1", "action": "reject",
            "reject_reason": "wrong_technique",
            "confidence": "high", "rationale": "mapped the wrong technique",
        }]}
        with caplog.at_level(logging.INFO, logger="app.api.routes.pipeline"):
            applied = await mod._auto_apply(
                None, {}, "src", "thread", "gate_1", "procedures", payload,
                {"drafts": [{"draft_id": "d1", "techniques": []}]},
            )

        assert applied is False, "past the limit, a rewind must not run"
        joined = " ".join(r.getMessage() for r in caplog.records)
        assert "deferring to a human" in joined, (
            "the hand-off was silent — indistinguishable from a reviewer crash"
        )
        assert "limit" in joined, "the log must say what stopped it"

    @pytest.mark.asyncio
    async def test_the_first_rewind_is_allowed(self, monkeypatch):
        """The whole point: autopilot may retry once by itself."""
        from app.api.routes import pipeline as mod

        async def _never_seen(source_id, gate_key):
            return 0

        applied_channels = {}

        class _Graph:
            async def aupdate_state(self, config, channels, as_node=None):
                applied_channels.update(channels)

        async def _noop_outcome(*a, **k):
            return None

        monkeypatch.setattr(mod, "gate_pass_count", _never_seen)
        monkeypatch.setattr(mod, "record_gate_outcome", _noop_outcome)

        payload = {"drafts": [{
            "draft_id": "d1", "action": "reject",
            "reject_reason": "wrong_technique",
            "confidence": "high", "rationale": "mapped the wrong technique",
        }]}
        applied = await mod._auto_apply(
            _Graph(), {}, "src", "thread", "gate_1", "procedures", payload,
            {"drafts": [{"draft_id": "d1", "techniques": []}]},
        )

        assert applied is True
        assert applied_channels["gate1_reviews"][0]["action"] == "reject"


    @pytest.mark.asyncio
    async def test_the_boundary_visit_is_the_last_one_allowed(self, monkeypatch):
        """Visit == limit still rewinds; visit > limit does not.

        gate_pass_count includes the turn being decided, so visit 1 is the
        first rewind. Off by one in either direction is silent — one too many
        is an extra reviewer-model pass on every source, one too few is a hand-off that
        should never have happened — so both sides are pinned here.
        """
        from app.api.routes import pipeline as mod

        payload = {"drafts": [{
            "draft_id": "d1", "action": "reject",
            "reject_reason": "wrong_technique",
            "confidence": "high", "rationale": "mapped the wrong technique",
        }]}
        state = {"drafts": [{"draft_id": "d1", "techniques": []}]}

        class _Graph:
            async def aupdate_state(self, config, channels, as_node=None):
                return None

        async def _noop_outcome(*a, **k):
            return None

        monkeypatch.setattr(mod, "record_gate_outcome", _noop_outcome)

        results = {}
        for visit in (mod._MAX_AUTO_REWIND_PASSES, mod._MAX_AUTO_REWIND_PASSES + 1):
            async def _count(source_id, gate_key, _v=visit):
                return _v

            monkeypatch.setattr(mod, "gate_pass_count", _count)
            results[visit] = await mod._auto_apply(
                _Graph(), {}, "src", "thread", "gate_1", "procedures",
                payload, state,
            )

        assert results[mod._MAX_AUTO_REWIND_PASSES] is True, (
            "the limit-th visit must still be allowed"
        )
        assert results[mod._MAX_AUTO_REWIND_PASSES + 1] is False, (
            "one past the limit must hand over"
        )


class TestRemovalNeverEmptiesAProcedure:
    """x_technique_refs is required, so a removal must not strip a draft bare.

    One ransomware run: extract_techniques emitted T1482 twice, the reviewer
    asked to "drop the duplicate entry so the procedure carries one T1482",
    and remove-by-id took both. The bundle hard-failed on
    x_procedure_missing_required_field. Dedup upstream stops that exact pair,
    but the guard is the floor under every other route to an empty list.
    """

    def test_removal_that_would_empty_the_draft_is_refused(self):
        state = {"drafts": [{
            "draft_id": "d1",
            "techniques": [
                {"technique_id": "T1482"},
                {"technique_id": "T1482"},
            ],
        }]}
        sub = build_auto_submission("procedures", {"drafts": [{
            "draft_id": "d1", "action": "approve",
            "remove_technique_ids": ["T1482"],
        }]}, state)
        item = sub.channels["gate1_reviews"][0]
        assert "analyst_edits" not in item, (
            "an empty technique list must never reach the gate"
        )
        assert item["action"] == "approve"

    def test_removal_still_applies_when_something_survives(self):
        state = {"drafts": [{
            "draft_id": "d1",
            "techniques": [
                {"technique_id": "T1482"},
                {"technique_id": "T1059.001"},
            ],
        }]}
        sub = build_auto_submission("procedures", {"drafts": [{
            "draft_id": "d1", "action": "approve",
            "remove_technique_ids": ["T1482"],
        }]}, state)
        item = sub.channels["gate1_reviews"][0]
        assert [t["technique_id"] for t in item["analyst_edits"]["techniques"]] == [
            "T1059.001"
        ]
        assert item["action"] == "edit"


class TestRewindIsConvertedNotSwallowed:
    """A rewind now converts; the caller decides whether it may run.

    It used to return empty channels and a defer_reason, so autopilot could
    never send work back at all. Three unattended ransomware runs stopped here.
    """

    def test_gate1_reject_reaches_the_gate_and_is_flagged(self):
        sub = build_auto_submission("procedures", {"drafts": [
            {"draft_id": "d1", "action": "reject",
             "reject_reason": "wrong_technique", "rationale": "mapped wrong"},
        ]}, {"drafts": [{"draft_id": "d1", "techniques": []}]})

        assert sub.rewind is True, "the caller must be able to see this is a rewind"
        item = sub.channels["gate1_reviews"][0]
        assert item["action"] == "reject"
        assert item["reject_reason"] == "wrong_technique"

    def test_chunk_reject_reaches_the_gate_and_is_flagged(self):
        sub = build_auto_submission("chunks", {
            "reject": {"reason": "under_chunked", "comments": "split further"},
        }, {})

        assert sub.rewind is True
        assert sub.channels["chunk_reviews"]["reject"] == {
            "reason": "under_chunked", "comments": "split further",
        }

    def test_an_ordinary_submission_is_not_a_rewind(self):
        sub = build_auto_submission("procedures", {"drafts": [
            {"draft_id": "d1", "action": "approve"},
        ]}, {"drafts": [{"draft_id": "d1", "techniques": []}]})

        assert sub.rewind is False


class TestRemoveReasonReachesTheGate:
    """A removal's reason must survive to the correction log.

    remove used to carry no reason at all while reject did, which is what
    pushed the reviewer to say "reject" when it meant "delete this one".
    """

    def test_remove_reason_is_forwarded(self):
        sub = build_auto_submission("procedures", {"drafts": [
            {"draft_id": "d1", "action": "remove",
             "remove_reason": "duplicate", "rationale": "same as d2"},
        ]}, {"drafts": [{"draft_id": "d1", "techniques": []}]})

        item = sub.channels["gate1_reviews"][0]
        assert item["remove_reason"] == "duplicate"
        assert item["action"] == "remove"
