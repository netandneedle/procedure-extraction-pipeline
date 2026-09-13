"""Persistence for the AI gate reviewer's transcript and its outcomes.

Two responsibilities:

  TRANSCRIPT — load the reviewer's prior turns for a source so the next gate
  call can replay them. Only the AGENT's turns are stored. Environment turns
  (gate payloads) are rebuilt from live state at call time, deliberately: the
  analyst may have overridden things since the last gate, and the reviewer
  must see what actually happened rather than what it proposed.

  OUTCOMES — record what the analyst actually submitted at a gate, diffed
  against what the reviewer recommended. This is the measurement instrument
  that decides whether any gate is ever safe to run unattended.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.reviewer import (
    INITIAL_READ,
    STATUS_FAILED,
    STATUS_OK,
    ReviewerRecommendation,
)

logger = logging.getLogger(__name__)


async def load_turns(
    db: AsyncSession, source_id: uuid.UUID,
) -> list[ReviewerRecommendation]:
    """Every successful reviewer turn for a source, oldest first.

    Failed turns are excluded: a turn that errored has no content to replay,
    and including it would put a hole in the conversation.
    """
    stmt = (
        select(ReviewerRecommendation)
        .where(
            ReviewerRecommendation.source_id == source_id,
            ReviewerRecommendation.status == STATUS_OK,
        )
        # The opening read is turn one by definition, so it is sorted first
        # explicitly rather than trusted to land earliest. Both rows are
        # written in the same call a few milliseconds apart, and `now()` ties
        # would otherwise leave the replay order undefined — the reviewer
        # would see its own gate review before the report read that produced
        # it.
        .order_by(
            (ReviewerRecommendation.gate_key != INITIAL_READ).asc(),
            ReviewerRecommendation.created_at.asc(),
        )
    )
    return list((await db.execute(stmt)).scalars().all())


async def get_initial_read(
    db: AsyncSession, source_id: uuid.UUID,
) -> ReviewerRecommendation | None:
    """The reviewer's opening read of the report, if it has one yet."""
    stmt = (
        select(ReviewerRecommendation)
        .where(
            ReviewerRecommendation.source_id == source_id,
            ReviewerRecommendation.gate_key == INITIAL_READ,
        )
        .order_by(ReviewerRecommendation.created_at.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def next_pass_number(
    db: AsyncSession, source_id: uuid.UUID, gate_key: str,
) -> int:
    """1-based visit count for this gate on this source.

    Gates loop — a Gate 1 reject routes back through extraction and returns
    here with new drafts. Each visit is its own row so the reviewer can be
    shown what changed between attempts, and so an override on pass 1 is not
    scored as agreement on pass 2.
    """
    stmt = select(func.count()).where(
        ReviewerRecommendation.source_id == source_id,
        ReviewerRecommendation.gate_key == gate_key,
    )
    return int((await db.execute(stmt)).scalar() or 0) + 1


async def gate_pass_count(source_id: uuid.UUID, gate_key: str) -> int:
    """How many times this gate has already been reviewed on this source.

    The pass limit for unattended rewinds. Opens its own session for the same
    reason record_gate_outcome does: its caller is the pipeline runner, which
    should not have to lend one.

    No new state field and no checkpoint migration — a row already exists per
    gate visit (see next_pass_number), so the count is free.

    NOTE the boundary: the reviewer writes its recommendation row BEFORE the
    runner applies it, so this count includes the turn being decided. The
    first visit to a gate returns 1, not 0. Callers comparing against a limit
    should read the result as "which visit is this", not "how many have
    already happened".

    Returns 0 on failure rather than raising. A lost count must not strand a
    run mid-pipeline, and 0 is the safe direction: it reads as a visit before
    the first, which at worst grants one rewind a stricter reading would have
    refused. The rewind itself is bounded by the gate's own routing.
    """
    from app.models.base import async_session

    try:
        async with async_session() as db:
            stmt = select(func.count()).where(
                ReviewerRecommendation.source_id == source_id,
                ReviewerRecommendation.gate_key == gate_key,
            )
            return int((await db.execute(stmt)).scalar() or 0)
    except Exception:
        logger.exception(
            "reviewer store: could not count passes for gate %s", gate_key,
        )
        return 0


async def latest_for_gate(
    db: AsyncSession, source_id: uuid.UUID, gate_key: str,
) -> ReviewerRecommendation | None:
    """Most recent turn for one gate — what the assist UI renders."""
    stmt = (
        select(ReviewerRecommendation)
        .where(
            ReviewerRecommendation.source_id == source_id,
            ReviewerRecommendation.gate_key == gate_key,
        )
        .order_by(ReviewerRecommendation.created_at.desc())
        .limit(1)
    )
    return (await db.execute(stmt)).scalar_one_or_none()


async def record_turn(
    db: AsyncSession,
    source_id: uuid.UUID,
    gate_key: str,
    *,
    payload: dict[str, Any],
    agent_notes: str = "",
    model: str = "",
    status: str = STATUS_OK,
    error: str | None = None,
) -> ReviewerRecommendation:
    """Append one reviewer turn."""
    row = ReviewerRecommendation(
        source_id=source_id,
        gate_key=gate_key,
        pass_number=await next_pass_number(db, source_id, gate_key),
        model=model,
        status=status,
        payload=payload,
        agent_notes=agent_notes,
        error=error,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row


async def record_failure(
    db: AsyncSession, source_id: uuid.UUID, gate_key: str, error: str,
) -> None:
    """Record that the reviewer errored and the gate fell through to a human.

    Best-effort by design: this is the failure path, and a failure to record
    a failure must not escalate into a failed pipeline run. The gate is still
    reviewable by a human either way, which is the property that matters.
    """
    try:
        await record_turn(
            db, source_id, gate_key,
            payload={}, status=STATUS_FAILED, error=error[:4000],
        )
    except Exception:  # noqa: BLE001 — never let bookkeeping break the run
        logger.exception(
            "could not record reviewer failure for source %s gate %s",
            source_id, gate_key,
        )


async def record_outcome(
    db: AsyncSession,
    source_id: uuid.UUID,
    gate_key: str,
    outcome: dict[str, Any],
) -> bool:
    """Attach what the analyst actually submitted to the latest turn.

    Returns False when there is nothing to attach it to (the gate ran in
    plain review mode, so no recommendation exists) — that is the normal
    case, not an error.
    """
    row = await latest_for_gate(db, source_id, gate_key)
    if row is None or row.status != STATUS_OK:
        return False
    row.outcome = outcome
    row.outcome_at = datetime.now(timezone.utc)
    await db.commit()
    return True


async def load_outcomes(
    db: AsyncSession, *, limit: int = 2000,
) -> list[ReviewerRecommendation]:
    """Every gate turn worth aggregating, newest first.

    Excludes the opening read: it is not a gate decision and carries no outcome
    by design. Failed turns ARE returned — a reviewer that errors is a fact the
    agreement readout has to report, not one it should quietly drop.

    Rows whose outcome is still NULL come back too: the caller distinguishes
    "recommended but not yet submitted" (the source is paused at that gate)
    from "submitted and scored", and both are true things about a gate.

    `limit` is a backstop against an unbounded scan, not a page size — at four
    rows per source it takes ~500 reviewed sources to reach the default.
    """
    stmt = (
        select(ReviewerRecommendation)
        .where(ReviewerRecommendation.gate_key != INITIAL_READ)
        .order_by(ReviewerRecommendation.created_at.desc())
        .limit(limit)
    )
    return list((await db.execute(stmt)).scalars().all())


async def delete_for_source(db: AsyncSession, source_id: uuid.UUID) -> int:
    """Drop every reviewer row for a source. Called from queue.delete_source."""
    rows = list(
        (
            await db.execute(
                select(ReviewerRecommendation).where(
                    ReviewerRecommendation.source_id == source_id
                )
            )
        ).scalars().all()
    )
    for row in rows:
        await db.delete(row)
    if rows:
        await db.commit()
    return len(rows)


async def record_gate_outcome(
    source_id: uuid.UUID,
    gate_key: str,
    reviews: list[dict],
    extras: list[dict] | dict[str, Any],
    *,
    auto: bool = False,
) -> None:
    """Diff a submission against the reviewer's recommendation and store it.

    Opens its own session: the two callers are an HTTP route and the pipeline
    runner, and neither should have to lend this one theirs.

    `auto=True` marks a decision the agent applied with no human involved.
    That flag is load-bearing, not bookkeeping — `agreement.aggregate` must
    skip these rows. An auto outcome is the reviewer agreeing with itself, so
    counting it would drive the agreement rate to 100% and make the one
    instrument that could justify autonomy report that it already is.

    No-ops when nothing was recommended at this gate (plain review mode) —
    the common case, not an error.

    Wrapped because this is measurement, and measurement must never cost a
    submission. A failure here loses one data point; raising would lose the
    analyst's work, or strand an autonomous run mid-pipeline.
    """
    from app.models.base import async_session
    from app.services.reviewer.outcomes import OUTCOME_DIFFERS

    differ = OUTCOME_DIFFERS.get(gate_key)
    if differ is None:
        return
    try:
        async with async_session() as db:
            row = await latest_for_gate(db, source_id, gate_key)
            if row is None:
                return
            outcome = differ(row.payload or {}, reviews, extras)
            if auto:
                outcome["auto"] = True
            await record_outcome(db, source_id, gate_key, outcome)
            agreement = outcome["agreement"]
            logger.info(
                "reviewer outcome for %s gate %s%s: %d/%d recommendations "
                "followed (%d overridden)",
                source_id, gate_key, " [auto]" if auto else "",
                agreement["agreed"], agreement["total"], agreement["overridden"],
            )
    except Exception:  # noqa: BLE001 — never let bookkeeping break a submit
        logger.warning(
            "could not record reviewer outcome for %s gate %s",
            source_id, gate_key, exc_info=True,
        )
