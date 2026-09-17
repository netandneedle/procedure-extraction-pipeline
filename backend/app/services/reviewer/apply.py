"""Turn a reviewer recommendation into the submission a gate accepts.

In assist mode a human reads the recommendation and clicks; the UI builds the
submit body. In auto mode there is no human and no UI, so this module does the
same job in Python. It is the seam between "what the reviewer said" and "what
the pipeline is told" — every auto decision passes through here.

THREE RULES, each of which was a bug waiting to happen:

1. **Sparse, never exhaustive.** Emit a decision ONLY for items the reviewer
   actually spoke about. Every gate already auto-approves what its review list
   does not mention (`gate_1`: "No explicit review -> auto-approve";
   `gate_chunks` and `gate_2` the same; `gate_0` likewise, *except* that an
   unmentioned entity still gets the denylist check).

   That exception is why this rule is load-bearing rather than tidy. An
   explicit `approve` on a denylisted entity **overrides the denylist** — the
   exact defect found in the denylist review, where the frontend submitted a
   decision for every entity and silently defeated the guardrail. Emitting
   approvals we were not asked for would reintroduce it, unattended, with
   nobody watching.

2. **Confidence is a word here and a number there.** `_Recommendation.confidence`
   is `"high" | "medium" | "low"`. `AddedEntityItem.confidence` and
   `AddedChunkItem.behavioral_confidence` are floats in [0, 1]. Passing the
   literal through would fail validation; passing nothing would silently take
   a default that misrepresents how sure the reviewer was. So it is mapped,
   and only where the target is numeric.

3. **Reviewer-only fields never reach the gate.** `evidence_quote`,
   `quote_source_support`, `quote_unsupported` exist for grounding and audit.
   They are recorded on the outcome row, not forwarded into pipeline state.
   Enforced by `_clean`'s whitelist — the only thing standing between a
   recommendation and the pipeline is the list of fields each submit item
   declares.

One thing this module converts but does not authorize: a rewind. A Gate 1
`reject`, or a chunk-gate `reject`, throws a whole pass away and re-enters an
upstream node, so an unattended agent could loop on it forever. The decision
is converted here and flagged `AutoSubmission.rewind`; whether it may RUN is
_auto_apply's call, because only the caller can count how many times this
gate has already been through. Neither applied blindly nor — worse — silently
dropped.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, NamedTuple

logger = logging.getLogger(__name__)

# Rule 2. Chosen to sit inside the bands the pipeline already uses elsewhere
# (a "definite" technique pick lands ~0.9, "possible" ~0.3-0.6), so an added
# item's confidence reads on the same scale as everything beside it.
_CONFIDENCE_TO_FLOAT = {"high": 0.9, "medium": 0.7, "low": 0.5}

# AddedEntityItem.rationale is capped at 500; the reviewer's is uncapped.
_RATIONALE_MAX = 500


class AutoSubmission(NamedTuple):
    """One gate's worth of auto-applied decisions.

    `channels` goes to `aupdate_state`. `decisions` and `extras` go to the
    outcome differ, in the same shapes the HTTP submit routes pass — so the
    recorded outcome is identical in structure whether a human or the agent
    produced it, and one differ serves both.

    `rewind` means applying this would discard the pass and re-enter an
    upstream node. The caller decides whether that is allowed — it can count
    the gate's prior visits and this module cannot.

    There is deliberately no `defer_reason` field any more. Every converter
    now returns something applicable, so the reasons a gate goes to a human
    all live in _auto_apply, which is the only place that knows them: no
    converter for the gate, nothing proposed, or a rewind over the pass
    limit. Each logs its own reason; a field no converter set would just be
    a channel for that logging to go missing down.
    """

    channels: dict[str, Any]
    decisions: list[dict[str, Any]]
    extras: list[dict[str, Any]] | dict[str, Any]
    # True when applying this would throw the pass away and re-enter an
    # upstream node. The conversion happens here; whether it is ALLOWED is the
    # caller's call, because only the caller can count how many passes this
    # gate has already had. See _auto_apply in api/routes/pipeline.py.
    rewind: bool = False


def _confidence_float(rec: dict[str, Any], default: float) -> float:
    return _CONFIDENCE_TO_FLOAT.get(rec.get("confidence"), default)


def _rationale(rec: dict[str, Any]) -> str:
    return (rec.get("rationale") or "")[:_RATIONALE_MAX]


def _clean(rec: dict[str, Any], keep: tuple[str, ...]) -> dict[str, Any]:
    """Project a recommendation onto the fields its submit item declares.

    The whitelist IS rule 3: `evidence_quote`, `quote_source_support` and
    `quote_unsupported` never appear in a `keep` tuple, so they cannot reach
    pipeline state. There was a second blacklist guard here as well; mutation
    testing showed nothing could make it fire, because `keep` is the only
    thing this iterates. Two mechanisms where one is real reads as defense in
    depth and is really just the illusion of it.

    Whitelist rather than blacklist for the same reason: a new field on a
    recommendation model should not start flowing into pipeline state because
    nobody remembered to exclude it.
    """
    return {k: rec[k] for k in keep if k in rec and rec[k] is not None}


def _recs(payload: dict[str, Any], key: str) -> list[dict[str, Any]]:
    return [r for r in (payload.get(key) or []) if isinstance(r, dict)]


# ── Gate 0 — entities ───────────────────────────────────────────────────

def apply_entities(payload: dict[str, Any], state: dict[str, Any]) -> AutoSubmission:
    reviews: list[dict[str, Any]] = []
    for rec in _recs(payload, "entities"):
        if not rec.get("entity_id") or not rec.get("action"):
            continue
        item = _clean(
            rec, ("entity_id", "action", "edited_value", "edited_type", "edited_role"),
        )
        # Gate0ReviewItem.rationale is `str | None`, so the reviewer's
        # reasoning rides along and lands in the correction log — which is
        # what the feedback flywheel reads. Its absence would make every
        # auto correction anonymous.
        item["rationale"] = _rationale(rec)
        reviews.append(item)

    added: list[dict[str, Any]] = []
    for rec in _recs(payload, "added_entities"):
        if not rec.get("value") or not rec.get("entity_type"):
            continue
        item = _clean(
            rec, ("value", "entity_type", "organization_role", "location_role"),
        )
        item["confidence"] = _confidence_float(rec, 1.0)
        item["rationale"] = _rationale(rec)
        added.append(item)

    return AutoSubmission(
        channels={"gate0_reviews": reviews, "gate0_added_entities": added},
        decisions=reviews,
        extras=added,
    )


# ── Gate chunks — the decomposition ─────────────────────────────────────

def apply_chunks(payload: dict[str, Any], state: dict[str, Any]) -> AutoSubmission:
    # A chunk-gate reject re-chunks the whole source, discarding every chunk
    # and every edit on this pass. Same shape as the Gate 1 rewind: converted
    # here, allowed or refused by the caller, which can count passes.
    #
    # Unlike Gate 1 this one is genuinely a re-run — re-chunking a source with
    # different guidance really can produce a better decomposition — so there
    # is no verb-level fix for it, only the limit.
    reject = payload.get("reject")
    if isinstance(reject, dict) and reject.get("reason"):
        return AutoSubmission(
            channels={"chunk_reviews": {"reject": {
                "reason": reject.get("reason"),
                "comments": str(reject.get("comments") or "")[:2000],
            }}},
            decisions=[],
            extras={},
            rewind=True,
        )

    decisions: list[dict[str, Any]] = []
    for rec in _recs(payload, "chunks"):
        if not rec.get("chunk_id") or not rec.get("action"):
            continue
        item = _clean(rec, ("chunk_id", "action", "merge_with"))
        # The one real transform: the reviewer names edited fields flatly,
        # ChunkDecisionItem nests them under `edits` and the gate processor
        # whitelists the keys (_CHUNK_EDITABLE_FIELDS).
        edits = {}
        if rec.get("edited_text") is not None:
            edits["text"] = rec["edited_text"]
        if rec.get("edited_source_excerpt") is not None:
            edits["source_excerpt"] = rec["edited_source_excerpt"]
        if edits:
            item["edits"] = edits
        decisions.append(item)

    added: list[dict[str, Any]] = []
    for rec in _recs(payload, "added_chunks"):
        if not rec.get("text"):
            continue
        item = _clean(rec, ("text", "source_excerpt"))
        item["behavioral_confidence"] = _confidence_float(rec, 0.7)
        added.append(item)

    edges: list[dict[str, Any]] = []
    for rec in _recs(payload, "edges"):
        if not rec.get("from_chunk_id") or not rec.get("to_chunk_id"):
            continue
        # `from_`, not `from`: the submit route dumps `by_alias=False`, so
        # pipeline state carries the Python-safe name and the gate processor
        # reads that. Emitting the wire alias here would land a key nothing
        # looks for.
        edges.append({
            "action": rec.get("action"),
            "from_": rec["from_chunk_id"],
            "to": rec["to_chunk_id"],
        })

    submission = {
        "decisions": decisions,
        "added_chunks": added,
        "edges": edges,
    }
    return AutoSubmission(
        channels={"chunk_reviews": submission},
        decisions=decisions,
        extras={"added_chunks": added, "edges": edges, "reject": None},
    )


def _apply_technique_removals(
    item: dict[str, Any], rec: dict[str, Any], drafts_by_id: dict[str, Any],
) -> None:
    """Turn `remove_technique_ids` into the edit the gate actually applies.

    The reviewer names the techniques to drop; the gate takes a whole
    replacement list (`analyst_edits["techniques"]`) and rebuilds
    kill_chain_phases from it. So the survivors are computed here, from the
    draft as it stands.

    Found by a live unattended run, not by a test: the reviewer asked to drop
    four techniques, the converter's whitelist silently dropped the field
    instead, and the only trace was an auto outcome reading 8/12 — the agent
    recorded as disagreeing with itself. A silent drop costs a full Opus turn
    and changes nothing.

    A removal also forces `action` to "edit": the gate only reads
    analyst_edits on an edit, so leaving an approve alone would discard this
    exactly as before.
    """
    remove = {
        str(t).strip().upper()
        for t in (rec.get("remove_technique_ids") or []) if str(t).strip()
    }
    if not remove:
        return

    draft = drafts_by_id.get(rec.get("draft_id"))
    if draft is None:
        logger.warning(
            "auto-apply: draft %s not in state; cannot drop %s",
            rec.get("draft_id"), sorted(remove),
        )
        return

    kept = [
        t for t in (draft.get("techniques") or [])
        if str(t.get("technique_id", "")).strip().upper() not in remove
    ]
    if not kept:
        # x_technique_refs is REQUIRED by the x-procedure schema, so a
        # procedure stripped to zero techniques hard-fails bundle validation
        # and takes the whole run down with it.
        #
        # Found on one run: extract_techniques emitted T1482 twice on
        # one chunk, and the reviewer asked to "drop the duplicate entry so
        # the procedure carries one T1482". But removal here is by ID, and an
        # ID cannot say "one of two" — so both went, and the bundle failed.
        # _dedupe_picks in technique_extraction now stops that pair from ever
        # reaching the gate; this guard is the floor under every other way a
        # removal could empty a draft.
        #
        # Refusing keeps the procedure valid and loud. Dropping the procedure
        # instead would be a bigger change than the reviewer asked for: it
        # said which techniques were wrong, not that the behavior was not a
        # procedure — that is what action="remove" is for.
        logger.warning(
            "auto-apply: refusing to remove %s from draft %s — it would leave "
            "the procedure with no techniques, and x_technique_refs is "
            "required. Leaving its techniques unchanged.",
            sorted(remove), rec.get("draft_id"),
        )
        return
    item["analyst_edits"] = {"techniques": kept}
    if item.get("action") == "approve":
        item["action"] = "edit"


# ── Gate 1 — procedures and techniques ──────────────────────────────────

def apply_procedures(
    payload: dict[str, Any], state: dict[str, Any],
) -> AutoSubmission:
    # Needed to honor `remove_technique_ids`: the gate replaces a draft's
    # technique list wholesale via analyst_edits, so the survivors have to be
    # computed from the draft as it currently stands.
    drafts_by_id = {
        d.get("draft_id"): d
        for d in (state.get("drafts") or [])
        if isinstance(d, dict) and d.get("draft_id")
    }
    drafts = _recs(payload, "drafts")
    # A reject re-enters an upstream node, discarding this pass. It is now
    # converted like any other decision — the gate's own
    # _compute_rejection_routing handles the route, exactly as it would for an
    # analyst's reject — but it is FLAGGED, because nothing here can see how
    # many passes this gate has already had. The caller enforces the limit.
    rewind = any(d.get("action") == "reject" for d in drafts)

    reviews: list[dict[str, Any]] = []
    for rec in drafts:
        if not rec.get("draft_id") or not rec.get("action"):
            continue
        item = _clean(rec, ("draft_id", "action", "reject_reason", "remove_reason"))
        item["rationale"] = _rationale(rec)
        _apply_technique_removals(item, rec, drafts_by_id)
        reviews.append(item)

    promotions = [
        _clean(rec, ("chunk_id", "technique_id"))
        for rec in _recs(payload, "promotions")
        if rec.get("chunk_id") and rec.get("technique_id")
    ]

    return AutoSubmission(
        channels={"gate1_reviews": reviews, "gate1_promotions": promotions},
        decisions=reviews,
        extras=promotions,
        rewind=rewind,
    )


# ── Gate 2 — bundle relationships ───────────────────────────────────────

def apply_bundle(payload: dict[str, Any], state: dict[str, Any]) -> AutoSubmission:
    reviews: list[dict[str, Any]] = []
    for rec in _recs(payload, "relationships"):
        if not rec.get("rel_id") or not rec.get("action"):
            continue
        reviews.append(_clean(
            rec,
            ("rel_id", "action", "edited_rel_type", "edited_source", "edited_target"),
        ))

    # Per-relationship mode, always — `gate2_reviews` rather than the batch
    # `gate2_review`. An empty list here means "approve everything as-is",
    # which is what the gate does with unmentioned relationships; the batch
    # channel would instead be a wholesale verdict.
    return AutoSubmission(
        channels={"gate2_reviews": reviews},
        decisions=reviews,
        extras=[],
    )


# gates_enabled key -> converter. Mirrors REVIEWERS in gate_reviewers.py and
# OUTCOME_DIFFERS in outcomes.py; a gate with a reviewer but no applier
# cannot run unattended, which a contract test pins.
AUTO_APPLIERS: dict[
    str, Callable[[dict[str, Any], dict[str, Any]], AutoSubmission]
] = {
    "entities": apply_entities,
    "chunks": apply_chunks,
    "procedures": apply_procedures,
    "bundle": apply_bundle,
}


def build_auto_submission(
    gate_key: str, payload: dict[str, Any], state: dict[str, Any] | None = None,
) -> AutoSubmission | None:
    """Convert a recommendation payload for `gate_key`. None if unsupported.

    `state` is the live PipelineState. Only Gate 1 reads it today, to resolve
    technique removals against the draft they apply to, but every converter
    takes it so the registry stays uniform.
    """
    applier = AUTO_APPLIERS.get(gate_key)
    if applier is None:
        logger.warning("no auto applier for gate '%s'", gate_key)
        return None
    return applier(payload or {}, state or {})
