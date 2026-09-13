"""Feedback-pattern management API.

Analyst-facing surface over the feedback flywheel's knowledge base: list /
filter patterns, promote a pattern to a permanent guardrail (wiring the
promoted_* lifecycle), dismiss a noisy one, or edit its text / structured keys.
Promote / dismiss / edit clear the prompt-addendum cache so the change shows up
in the next run's prompts without waiting for the TTL.

Endpoints:
    GET   /api/feedback-patterns/            — paginated, filterable list
    PATCH /api/feedback-patterns/{id}/promote — promote to prompt | denylist
    PATCH /api/feedback-patterns/{id}/dismiss — mark dismissed
    PATCH /api/feedback-patterns/{id}         — edit text / category / keys
"""

from __future__ import annotations

import asyncio
import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db, get_graph
from app.nodes.llm.feedback_synthesis import captured_corrections
from app.schemas.api import FeedbackPatternEditRequest, FeedbackPatternPromoteRequest
from app.services import feedback_patterns as fp
from app.services import queue as queue_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Sources in these states have nothing in-flight to show: completed sources
# already synthesized their corrections into patterns; queued ones haven't run.
_NO_CAPTURE_STATUSES = frozenset({"completed", "queued"})

# Failed sources stay visible in the captured panel (their corrections never
# synthesized — that visibility is the feature), but a failed run's checkpoint
# is FROZEN: nothing writes to it again unless the source is re-queued, which
# changes its status. Re-deserializing those multi-hundred-KB checkpoints on
# every 8s poll was pure repeated I/O that grew with the project's failure
# history. Memoize per (thread_id, status); bounded FIFO eviction.
_TERMINAL_CAPTURE_STATUSES = frozenset({"failed"})
_terminal_capture_cache: dict[tuple[str, str], dict | None] = {}
_TERMINAL_CACHE_MAX = 512


async def _captured_for_source(graph, s) -> dict | None:
    """Read one source's checkpoint and flatten its captured corrections.

    Best-effort: returns None for unreadable / empty / mis-shaped checkpoints
    (one bad checkpoint mustn't break the list) and for sources with no
    corrections. Terminal (failed) sources are served from the process-local
    memo above — their checkpoints are frozen, so the answer can't change
    until a re-queue flips their status (which changes the cache key).
    """
    cache_key = (str(s.thread_id), s.status)
    is_terminal = s.status in _TERMINAL_CAPTURE_STATUSES
    if is_terminal and cache_key in _terminal_capture_cache:
        return _terminal_capture_cache[cache_key]

    config = {"configurable": {"thread_id": str(s.thread_id)}}
    entry: dict | None = None
    try:
        snap = await graph.aget_state(config)
        # Inside the try on purpose: a checkpoint that READS fine but PARSES
        # badly (mis-shaped legacy/seed state) must also skip, not 500.
        if snap and snap.values:
            items = captured_corrections(snap.values)
            if items:
                entry = {
                    "source_id": str(s.id),
                    "title": s.title,
                    "status": s.status,
                    "corrections": items,
                }
    except Exception as e:  # noqa: BLE001 — one bad checkpoint mustn't break the list
        logger.warning("captured corrections: failed to read %s: %s", s.id, e)
        # Deliberately NOT cached: a read failure may be transient (DB blip),
        # so the next poll should retry rather than pin an empty answer.
        return None

    if is_terminal:
        if len(_terminal_capture_cache) >= _TERMINAL_CACHE_MAX:
            # FIFO eviction via insertion-ordered dict — cheap and fine for
            # a bounded memo of frozen results.
            _terminal_capture_cache.pop(next(iter(_terminal_capture_cache)))
        _terminal_capture_cache[cache_key] = entry
    return entry


@router.get("/captured", summary="In-flight analyst corrections not yet synthesized")
async def list_captured_corrections(
    graph=Depends(get_graph),
    db: AsyncSession = Depends(get_db),
):
    """Surface analyst corrections captured on still-running sources, before
    synthesize_feedback turns them into patterns at run completion.

    Patterns (the rest of this tab) are the *learned rules*, written only when
    a run finishes. This endpoint shows the *raw corrections* the moment they
    are recorded, so an analyst who rejects/edits/discards at a gate sees
    immediate proof their feedback landed — and so corrections on a source that
    loops (or fails) without completing are still visible rather than invisible.

    Reads each active source's LangGraph checkpoint and reuses the same
    _compute_deltas the synthesizer consumes, so the captured view matches what
    will eventually be synthesized. Best-effort per source: a checkpoint that
    can't be read is skipped, never 500s the whole list. Reads run
    concurrently (they're independent; serial reads made latency the SUM of N
    checkpoint round-trips instead of the max), and failed sources are served
    from the frozen-checkpoint memo.
    """
    sources, _ = await queue_service.list_sources(db, limit=200)
    candidates = [
        s for s in sources
        if s.thread_id is not None and s.status not in _NO_CAPTURE_STATUSES
    ]
    entries = await asyncio.gather(
        *(_captured_for_source(graph, s) for s in candidates)
    )
    out = [e for e in entries if e]
    return {"sources": out, "total": sum(len(e["corrections"]) for e in out)}


@router.get("/", summary="List feedback patterns")
async def list_feedback_patterns(
    category: str | None = Query(None),
    status: str | None = Query(None),
    min_salience: float | None = Query(None),
    search: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List patterns ordered most-useful-first (salience, then occurrence,
    then recency). All filters optional."""
    rows, total = await fp.list_patterns(
        db, category=category, status=status, min_salience=min_salience,
        search=search, limit=limit, offset=offset,
    )
    return {"patterns": [fp.pattern_to_dict(r) for r in rows], "total": total}


@router.patch("/{pattern_id}/promote", summary="Promote a pattern to a permanent guardrail")
async def promote_feedback_pattern(
    pattern_id: UUID,
    body: FeedbackPatternPromoteRequest,
    db: AsyncSession = Depends(get_db),
):
    """Turn a learned pattern into an enforced rule.

    `prompt` pins it: always injected for its category, exempt from the
    relevance cut and the age cutoff. `denylist` makes it a deterministic
    guardrail on the confirmed `denylist_terms` (entity values and/or
    technique IDs); a match is removed by default at the gate and can be
    overridden there per source.
    """
    row = await fp.promote_pattern(
        db, pattern_id, action=body.action, by=body.by,
        denylist_terms=body.denylist_terms.model_dump() if body.denylist_terms else None,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Feedback pattern not found")
    return fp.pattern_to_dict(row)


@router.patch("/{pattern_id}/dismiss", summary="Dismiss a pattern (stop consuming it)")
async def dismiss_feedback_pattern(
    pattern_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Stop injecting a pattern. The row is kept for the record."""
    row = await fp.dismiss_pattern(db, pattern_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Feedback pattern not found")
    return fp.pattern_to_dict(row)


@router.patch("/{pattern_id}", summary="Edit a pattern's text / category / keys")
async def edit_feedback_pattern(
    pattern_id: UUID,
    body: FeedbackPatternEditRequest,
    db: AsyncSession = Depends(get_db),
):
    """Edit a pattern's text, category or structured keys. A text change is
    re-embedded so retrieval sees the new wording."""
    row = await fp.update_pattern(
        db, pattern_id,
        pattern=body.pattern, category=body.category,
        applies_to=body.applies_to, concepts=body.concepts,
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Feedback pattern not found")
    return fp.pattern_to_dict(row)
