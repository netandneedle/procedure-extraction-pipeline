"""Run the AI gate reviewer for one gate and persist what it said.

FAIL-SOFT IS THE CONTRACT. Every failure path here ends with the gate still
reviewable by a human. The reviewer is an assistant; an assistant that can
take the pipeline down with it is a liability, not a feature. Same convention
as per-figure vision failures and the lexical-only feedback fallback.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from app.graph.state import EntityType
from app.models.base import async_session
from app.models.reviewer import INITIAL_READ
from app.nodes.llm.llm_adapter import call_llm, resolve_target
from app.services.feedback_patterns import pinned_rules_addendum
from app.services.grounding import (
    QUOTE_SUPPORT_THRESHOLD,
    build_source_grounding_tokens,
    quote_support,
)
from app.services.reviewer import store
from app.services.reviewer.gate_reviewers import (
    _SECTOR_VOCAB,
    REVIEWERS,
    build_report_turn,
)
from app.services.reviewer.prompts import REVIEWER_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Recommendation lists in a reviewer payload that carry evidence quotes and
# therefore need grounding. Keeping this explicit (rather than walking every
# list in the payload) means a new field can't silently opt out of the check.
_GROUNDED_LISTS = (
    "entities", "added_entities",
    "chunks", "added_chunks", "edges",
    "drafts", "promotions",
    "relationships",
)

# Recommendations that arrive as a single object rather than a list. Only the
# chunk gate has one: a wholesale re-chunk is not a list because there is
# exactly one decomposition to reject. It still carries an evidence quote and
# so still has to face the grounding check — an ungrounded "re-run everything"
# is the most expensive unverified claim the reviewer can make.
_GROUNDED_SINGLETONS = ("reject",)

# Gates whose tool asks for the opening read. Only the first one does — the
# read is turn one of the transcript, and asking a later gate for it would
# get a read anchored on the extraction it is supposed to be judging.
_GATES_THAT_OPEN_THE_TRANSCRIPT = ("entities",)


def apply_grounding(
    payload: dict[str, Any], source_tokens: set[str],
) -> int:
    """Downgrade recommendations whose evidence quote the report doesn't support.

    An AI reviewer is a NEW place for fabricated evidence to enter the
    pipeline — the exact failure that put `T1204.004` in a bundle at
    confidence 0.95 quoting a phrase occurring zero times in its source. The
    reviewer gets the same instrument pointed at it that the technique picker
    does.

    Forced to "low" rather than dropped, deliberately. A bad quote means the
    justification is unverified, not that the underlying observation is
    wrong — and "low" already carries the consequence that matters: it is
    excluded from the analyst's bulk-accept, so a human must look at it.

    Mutates `payload` in place. Returns how many were downgraded.
    """
    if not source_tokens:
        return 0

    def _check(rec: object) -> bool:
        """Score one recommendation's quote. True when it was downgraded."""
        if not isinstance(rec, dict):
            return False
        quote = (rec.get("evidence_quote") or "").strip()
        if not quote:
            return False
        support = quote_support(quote, source_tokens)
        if support is None:
            return False  # too short to judge — no opinion, no penalty
        rec["quote_source_support"] = round(support, 2)
        if support >= QUOTE_SUPPORT_THRESHOLD:
            return False
        rec["quote_unsupported"] = True
        if rec.get("confidence") == "low":
            return False
        rec["confidence"] = "low"
        return True

    downgraded = 0
    for key in _GROUNDED_LISTS:
        for rec in payload.get(key) or []:
            downgraded += _check(rec)
    for key in _GROUNDED_SINGLETONS:
        downgraded += _check(payload.get(key))
    return downgraded


_VALID_ENTITY_TYPES = frozenset(e.value for e in EntityType)


def drop_invalid_entity_types(payload: dict[str, Any]) -> int:
    """Remove recommendations naming an entity_type the pipeline doesn't have.

    The tool schema already offers the vocabulary as an enum, so this should
    never fire. It exists because the addition channel does NOT validate type
    — `_build_added_entity` copies whatever it is handed straight into
    validated_entities — and on the first live run the reviewer proposed
    `attack_pattern` for CLICKFIX. That is a real STIX type and a plausible
    answer; it is simply not one of ours, and nothing downstream would have
    caught it.

    Dropped rather than downgraded, unlike an unsupported quote. A quote we
    cannot verify still describes a real entity the analyst can judge; a type
    that does not exist has nothing for them to accept. Also note what the
    right answer usually is: a technique pattern like ClickFix should not be
    an entity at ALL, so silently coercing it to some nearest-fit type would
    put a wrong object in the bundle rather than none.

    Mutates `payload` in place. Returns how many were dropped.
    """
    dropped = 0
    for key, field in (("entities", "edited_type"), ("added_entities", "entity_type")):
        kept = []
        for rec in payload.get(key) or []:
            if not isinstance(rec, dict):
                continue
            etype = rec.get(field)
            # Only `added_entities.entity_type` is mandatory; an entity
            # recommendation that doesn't touch the type has nothing to check.
            required = key == "added_entities"
            if etype is None and not required:
                kept.append(rec)
                continue
            if etype in _VALID_ENTITY_TYPES:
                kept.append(rec)
                continue
            logger.warning(
                "reviewer proposed unknown entity_type %r for %r — dropped",
                etype, rec.get("entity_id") or rec.get("value"),
            )
            dropped += 1
        payload[key] = kept
    return dropped


_SECTOR_SET = frozenset(_SECTOR_VOCAB)


def enforce_sector_vocabulary(payload: dict[str, Any]) -> int:
    """Make victim_sector additions carry a real industry-sector-ov value.

    The gate's addition channel carries a sector in `value`, but the tool
    schema asks for it in a separate enum-constrained `sector` field — an
    enum cannot be made conditional on another property in a tool schema, so
    the constraint lives on its own field and is folded in here.

    A recommendation with no usable vocabulary value is DROPPED rather than
    passed through. The bundle validator would otherwise drop the sector from
    the Identity silently (`invalid_sector_dropped`, severity warn), which
    means the analyst accepts a recommendation and gets nothing — the worst
    of both: their time spent, no sector, and no error to notice.

    Mutates `payload` in place. Returns how many were dropped.
    """
    dropped = 0
    kept = []
    for rec in payload.get("added_entities") or []:
        if not isinstance(rec, dict):
            continue
        if rec.get("entity_type") != "victim_sector":
            rec.pop("sector", None)
            kept.append(rec)
            continue
        sector = rec.pop("sector", None)
        candidate = sector if sector in _SECTOR_SET else rec.get("value")
        if candidate in _SECTOR_SET:
            rec["value"] = candidate
            kept.append(rec)
            continue
        logger.warning(
            "reviewer proposed victim_sector %r, which is not an "
            "industry-sector-ov value — dropped (the bundle validator would "
            "have discarded it silently)",
            sector or rec.get("value"),
        )
        dropped += 1
    if "added_entities" in payload:
        payload["added_entities"] = kept
    return dropped


def format_outcome_feedback(outcome: dict[str, Any] | None) -> str:
    """What the analyst actually did with a past review, for the reviewer.

    The system prompt tells the reviewer it will be told when the analyst
    overrides it. This is what makes that true. Without it the model is
    promised a signal it never receives, and on a gate that loops it can
    repeat advice the analyst just rejected.

    Overrides only. Agreement is the default outcome and listing it buries
    the two lines that carry information — and those two lines are the
    labeled disagreement the whole assist-first design exists to produce.

    Returns "" when there is nothing to say (not submitted yet, or the
    analyst took every recommendation).
    """
    agreement = (outcome or {}).get("agreement") or {}
    items = agreement.get("items") or []
    # `moot` items are recommendations the analyst never reached — at the
    # chunk gate a re-chunk discards the decisions, adds and edges channels
    # wholesale. They are marked not-agreed, which is true but not a
    # disagreement, and reporting them as one would tell the reviewer the
    # analyst rejected advice they never saw the consequence of.
    overrides = [
        i for i in items
        if isinstance(i, dict) and not i.get("agreed") and not i.get("moot")
    ]
    if not overrides:
        return ""
    total = agreement.get("total", len(items))
    agreed = agreement.get("agreed", total - len(overrides))
    lines = [
        f"What the analyst did with that review: took {agreed} of {total} "
        f"recommendations. They went a different way on these:",
    ]
    for i in overrides:
        # An addition's ref is the entity VALUE, which for a command line can
        # run to hundreds of characters. Truncated here rather than at the
        # source: the full value stays in the stored outcome for auditing,
        # and only this prompt rendering needs to stay readable.
        ref = str(i.get("ref", "?"))
        if len(ref) > 80:
            ref = ref[:77] + "..."
        lines.append(
            f"  - {ref}: you said {i.get('recommended', '?')}, "
            f"they chose {i.get('actual', '?')}"
        )
    lines.append(
        "Take that as information, not as something to argue with. Do not "
        "re-make a recommendation they already declined unless something in "
        "this gate genuinely changes the picture — and say what changed."
    )
    return "\n".join(lines)


async def _build_messages(
    db, source_id: uuid.UUID, state: dict, current_turn: str,
) -> list[dict]:
    """Report, then every prior reviewer turn, then this gate.

    Only the reviewer's OWN turns are replayed from storage. The gate payloads
    are rebuilt from live state each time — which is the point: between gates
    the analyst may have overridden the reviewer, and it must see what actually
    happened rather than what it proposed.

    Prior turns are replayed as assistant prose rather than as tool_use blocks.
    That is a summary of the conversation, not a byte-exact replay, and it is
    what the reviewer needs: its own conclusions, not its own JSON.
    """
    # NO CACHE BREAKPOINT ON THE REPORT TURN -- measured, and it does not work
    # here.
    #
    # The report is byte-identical across a run's gates, so marking it looked
    # obviously right, and a two-call experiment said 32% saving. A real
    # three-gate run said otherwise:
    #
    #     gate         uncached  cache_wr  cache_rd
    #     entities         3729      8927      8927   <- same-call retry
    #     procedures       9344      7864         0
    #     bundle           6879      7328         0
    #
    # cache_rd is zero at every later gate, including one only 54s after the
    # previous -- far inside the TTL, so this is not expiry. A cache prefix is
    # matched over tools -> system -> messages, and BOTH earlier parts differ
    # per gate: each gate has its own tool schema, and its own pinned-rule
    # categories in the system prompt. A difference anywhere upstream voids
    # everything after it, so a breakpoint in `messages` can never be reached
    # across gates.
    #
    # What is left is paying 1.25x to write a cache nothing reads: ~12% worse
    # per gate. The one place it did pay was the entities retry re-reading its
    # own prefix -- and optimising for a retry means optimising for a bug.
    #
    # Making it work would need identical tools AND system on every gate (pass
    # all three tool schemas everywhere and select with tool_choice). Possible,
    # not obviously worth it; revisit only with a measurement in hand.
    messages: list[dict] = [{"role": "user", "content": build_report_turn(state)}]

    for turn in await store.load_turns(db, source_id):
        notes = (turn.agent_notes or "").strip()
        if not notes:
            continue
        label = (
            "My read of this report:"
            if turn.gate_key == INITIAL_READ
            else f"My review at the {turn.gate_key} gate (pass {turn.pass_number}):"
        )
        messages.append({"role": "assistant", "content": f"{label}\n{notes}"})
        # Environment feedback, so it is a USER turn — the analyst's verdict
        # is not something the assistant said.
        feedback = format_outcome_feedback(turn.outcome)
        if feedback:
            messages.append({"role": "user", "content": feedback})

    messages.append({"role": "user", "content": current_turn})
    return messages


def _subject_of(rec: dict[str, Any]) -> str:
    """What one recommendation is ABOUT, for the replayed summary.

    Order matters: a technique promotion carries both a chunk_id and a
    technique_id, and the technique is the subject — the chunk is only where
    it lives.
    """
    if rec.get("from_chunk_id"):
        return f"{rec['from_chunk_id']} -> {rec.get('to_chunk_id', '?')}"
    for key in ("entity_id", "draft_id", "technique_id", "chunk_id", "value"):
        if rec.get(key):
            return str(rec[key])
    text = (rec.get("text") or "").strip()
    return f'"{text[:80]}"' if text else "?"


def _summarise_turn(payload: dict[str, Any]) -> str:
    """Condense a payload into the prose replayed at the next gate.

    Deliberately lossy. The next gate needs the reviewer's conclusions and
    reasons, not a JSON dump it would have to re-parse — and a compact turn
    keeps the transcript from crowding out the gate being reviewed.
    """
    lines: list[str] = []
    read = payload.get("initial_read")
    if isinstance(read, dict) and read.get("summary"):
        lines.append(read["summary"])
        if read.get("attack_chain"):
            lines.append("Chain: " + " -> ".join(str(s) for s in read["attack_chain"]))
        if read.get("thin_areas"):
            lines.append(
                "Thin: " + "; ".join(str(s) for s in read["thin_areas"])
            )
        if read.get("notes_for_later_gates"):
            lines.append(read["notes_for_later_gates"])

    for key in _GROUNDED_LISTS:
        for rec in payload.get(key) or []:
            if not isinstance(rec, dict):
                continue
            verb = rec.get("action") or ("promote" if rec.get("technique_id") else "add")
            extra = ""
            if rec.get("remove_technique_ids"):
                extra = f" [drop {', '.join(rec['remove_technique_ids'])}]"
            if rec.get("merge_with"):
                extra += f" [absorbing {', '.join(rec['merge_with'])}]"
            if rec.get("reject_reason"):
                extra += f" [{rec['reject_reason']}]"
            lines.append(
                f"- {verb} {_subject_of(rec)}{extra} "
                f"({rec.get('confidence', '?')}): {rec.get('rationale', '')}"
            )

    # The re-chunk ask, if it made one. Loud in the replay because it is the
    # loudest thing this reviewer can say: it throws the pass away.
    reject = payload.get("reject")
    if isinstance(reject, dict) and reject.get("reason"):
        lines.append(
            f"- asked to re-chunk the whole source [{reject['reason']}] "
            f"({reject.get('confidence', '?')}): {reject.get('rationale', '')}"
        )
        if reject.get("comments"):
            lines.append(f"  guidance given: {reject['comments']}")

    if payload.get("overall_notes"):
        lines.append(str(payload["overall_notes"]))
    return "\n".join(lines).strip()


async def run_reviewer(
    source_id: uuid.UUID, gate_key: str, state: dict,
) -> dict[str, Any] | None:
    """Review one gate. Returns the recommendation payload, or None on failure.

    None means the caller should fall through to normal human review. It is
    never an exception: the reviewer failing is a degraded review, not a
    failed run.
    """
    reviewer = REVIEWERS.get(gate_key)
    if reviewer is None:
        logger.warning("no AI reviewer implemented for gate '%s'", gate_key)
        return None

    try:
        async with async_session() as db:
            messages = await _build_messages(
                db, source_id, state, reviewer.build_turn(state),
            )
            # The analyst's confirmed policy for the decisions this gate
            # makes. Pinned rules only — see pinned_rules_addendum for why
            # the advisory patterns are deliberately withheld.
            #
            # Deliberately NOT recorded as a surfacing: the extraction nodes
            # already log these same patterns for this same source, and
            # _score_surfacings dedups by pattern_id, so extra rows would
            # change no score while doubling the ledger.
            rules = await pinned_rules_addendum(
                db, categories=reviewer.feedback_categories,
            )
            reviewer_target = resolve_target("reviewer")
            response = await call_llm(
                system=REVIEWER_SYSTEM_PROMPT + rules,
                messages=messages,
                tools=[reviewer.tool],
                tool_choice={"type": "tool", "name": reviewer.tool["name"]},
                # One resolver for the role — the same one the startup check
                # validates. Blank REVIEWER_PROVIDER inherits llm_provider;
                # set it to run the reviewer on a different VENDOR than
                # extraction, the strongest form of the independence this
                # reviewer exists for.
                model=reviewer_target.model,
                provider=reviewer_target.provider,
                output_model=reviewer.output_model,
            )
            # The VALIDATED model, not the raw tool_output.
            #
            # `output_model` does more than reject bad payloads: its
            # `model_validator` repairs the shapes the model actually emits —
            # most importantly lifting a flattened or stringified
            # `initial_read` back into place. Reading `tool_output` here threw
            # every one of those repairs away and used the unrepaired shape.
            #
            # That was the real reason the opening read kept going missing. It
            # survived making the field required, because the field WAS
            # arriving; it just arrived flattened, was repaired into
            # `validated`, and then this line reached past the repair for the
            # raw dict — where `initial_read` is a string or absent, so
            # `isinstance(..., dict)` is False and no brief is ever stored.
            #
            # Falls back to the raw dict only when no output_model was given,
            # which no gate reviewer does.
            payload: dict[str, Any] = (
                response.validated.model_dump(mode="json")
                if response.validated is not None
                else dict(response.tool_output or {})
            )

            downgraded = apply_grounding(
                payload, build_source_grounding_tokens(state),
            )
            # Entity-vocabulary checks only apply where entities are the
            # subject. Running them elsewhere would walk lists that do not
            # exist and report a misleading zero.
            dropped = 0
            if gate_key == "entities":
                dropped = drop_invalid_entity_types(payload)
                dropped += enforce_sector_vocabulary(payload)

            # Cost is a question for measurement, so record it rather than
            # estimate it. All three token fields, not just input_tokens: the cache
            # fields are counted SEPARATELY from it, so recording one of the
            # three is how a caching change gets scored against a number that
            # never included it. There is no cache breakpoint here today (see
            # _build_messages), which is itself a fact these numbers
            # established — they should keep being recorded so the next
            # attempt can be judged the same way.
            payload["_usage"] = {
                "input_tokens": response.input_tokens,
                "cache_creation_tokens": response.cache_creation_tokens,
                "cache_read_tokens": response.cache_read_tokens,
                "output_tokens": response.output_tokens,
                "cached": response.cached,
                "model": response.model,
            }

            # The opening read is its own transcript turn: the analyst edits it
            # independently, and a correction there must propagate to every
            # later gate without dragging this gate's recommendations along.
            initial_read = payload.pop("initial_read", None)
            if not (isinstance(initial_read, dict) and initial_read.get("summary")):
                # Loud, because the failure is otherwise invisible: the gate
                # review still works, and the brief's absence only shows up
                # later as a 404 nobody is watching and a chunk gate with no
                # attack chain to compare against.
                if gate_key in _GATES_THAT_OPEN_THE_TRANSCRIPT:
                    logger.warning(
                        "reviewer: %s gate returned no usable opening read for "
                        "source %s — later gates lose the attack chain, and the "
                        "analyst has no brief to correct",
                        gate_key, source_id,
                    )
            else:
                existing = await store.get_initial_read(db, source_id)
                if existing is None:
                    await store.record_turn(
                        db, source_id, INITIAL_READ,
                        payload={"initial_read": initial_read},
                        agent_notes=_summarise_turn(
                            {"initial_read": initial_read},
                        ),
                        model=response.model,
                    )

            row = await store.record_turn(
                db, source_id, gate_key,
                payload=payload,
                agent_notes=_summarise_turn(payload),
                model=response.model,
            )

        parts = [
            f"{len(payload.get(k) or [])} {k}"
            for k in _GROUNDED_LISTS if payload.get(k)
        ]
        # A re-chunk ask is a single object, so it does not appear in the list
        # counts — and it is the loudest thing the reviewer can say. Without
        # this a run whose only recommendation was "throw the pass away" logs
        # "no recommendations".
        if isinstance(payload.get("reject"), dict):
            parts.append(f"RE-CHUNK ({payload['reject'].get('reason', '?')})")
        counts = ", ".join(parts) or "no recommendations"
        logger.info(
            "reviewer: %s gate for source %s — %s; %d downgraded for "
            "unsupported quotes, %d dropped for bad vocabulary "
            "(in=%d cache_write=%d cache_read=%d out=%d, cached=%s)",
            gate_key, source_id, counts, downgraded, dropped,
            response.input_tokens, response.cache_creation_tokens,
            response.cache_read_tokens, response.output_tokens, response.cached,
        )
        return {"id": str(row.id), **payload}

    except Exception as e:  # noqa: BLE001 — degraded review, never a failed run
        logger.exception(
            "AI reviewer failed at gate '%s' for source %s; "
            "falling through to human review",
            gate_key, source_id,
        )
        try:
            async with async_session() as db:
                await store.record_failure(db, source_id, gate_key, str(e))
        except Exception:  # noqa: BLE001
            logger.exception("could not record reviewer failure")
        return None
