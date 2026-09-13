"""AI gate reviewer endpoints.

The reviewer's recommendations are a resource of their own rather than a field
on the gate payload: the assist UI fetches them independently, they outlive the
LangGraph checkpoint, and per-gate agreement rates have to be queryable across
sources.

ROUTE ORDER MATTERS. `/{source_id}/brief` is declared before
`/{source_id}/{gate_key}` — otherwise "brief" binds as a gate_key and the
brief endpoints become unreachable. Same trap the chunk-gate routes hit.
`/agreement` is a single segment and cannot collide with either, but it is
declared first anyway so the rule reads the same all the way down.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db, get_graph
from app.api.routes._gate_registry import GATES_BY_NODE
from app.models.reviewer import INITIAL_READ
from app.schemas.api import (
    ReviewerAgreementResponse,
    ReviewerBriefUpdate,
    ReviewerRecommendationResponse,
)
from app.services.reviewer.agreement import aggregate
from app.services.reviewer import run_reviewer, store
from app.services.reviewer.gate_reviewers import REVIEWERS

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/agreement",
    response_model=ReviewerAgreementResponse,
    summary="Per-gate agreement between the reviewer and the analyst",
)
async def get_agreement(
    db: AsyncSession = Depends(get_db),
):
    """How often the analyst took each gate's advice, across every source.

    This is what assist mode is FOR. The recommendations are the product; the
    agreement is the instrument, and it is the only evidence that can say
    whether a gate is ever safe to run unattended.

    An empty result is the ordinary starting state, not an error — no source
    has been reviewed yet.
    """
    rows = await store.load_outcomes(db)
    return ReviewerAgreementResponse(**aggregate(rows))


@router.get(
    "/{source_id}/brief",
    response_model=ReviewerRecommendationResponse,
    summary="Get the reviewer's opening read of the report",
)
async def get_brief(
    source_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """The reviewer's initial read, if it has produced one."""
    row = await store.get_initial_read(db, source_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail="No reviewer brief for this source yet.",
        )
    return ReviewerRecommendationResponse.model_validate(row)


@router.post(
    "/{source_id}/brief",
    response_model=ReviewerRecommendationResponse,
    summary="Correct the reviewer's opening read and re-review the current gate",
)
async def update_brief(
    source_id: uuid.UUID,
    body: ReviewerBriefUpdate,
    db: AsyncSession = Depends(get_db),
    graph=Depends(get_graph),
):
    """Replace the brief, then re-run the reviewer for the gate in progress.

    Because the brief is turn one of the transcript, the corrected version is
    what every later gate replays — the analyst fixes a misread once instead
    of countering it gate by gate.

    Re-running the current gate is what makes the correction visible
    immediately. If the pipeline is not paused at a gate with a reviewer, the
    brief is still updated and takes effect at the next one.
    """
    existing = await store.get_initial_read(db, source_id)
    if existing is None:
        raise HTTPException(
            status_code=404,
            detail="No reviewer brief for this source yet.",
        )

    corrected = body.model_dump()
    # analyst_edited is read by the prompt-builder so the reviewer knows the
    # brief is the analyst's, not its own — it should not quietly re-argue it.
    corrected["analyst_edited"] = True
    existing.payload = {"initial_read": corrected}
    existing.agent_notes = _brief_to_notes(corrected)
    await db.commit()
    await db.refresh(existing)

    gate_key = await _current_reviewable_gate(graph, source_id)
    if gate_key is not None:
        # Drop the stale recommendation so the UI can't show pre-correction
        # advice next to a corrected brief.
        stale = await store.latest_for_gate(db, source_id, gate_key)
        if stale is not None:
            await db.delete(stale)
            await db.commit()
        state = await _thread_state(graph, source_id)
        if state is not None:
            await run_reviewer(source_id, gate_key, state)

    return ReviewerRecommendationResponse.model_validate(existing)


@router.get(
    "/{source_id}/{gate_key}",
    response_model=ReviewerRecommendationResponse,
    summary="Get the reviewer's recommendations for one gate",
)
async def get_recommendations(
    source_id: uuid.UUID,
    gate_key: str,
    db: AsyncSession = Depends(get_db),
):
    """Latest recommendations for a gate.

    404 is the ordinary case, not an error: it means this gate ran in plain
    review mode, or the reviewer has not reached it. The UI renders the gate
    exactly as it always has.
    """
    if gate_key != INITIAL_READ and gate_key not in REVIEWERS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unknown or unimplemented gate '{gate_key}'. "
                f"Implemented: {sorted(REVIEWERS)}"
            ),
        )
    row = await store.latest_for_gate(db, source_id, gate_key)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=f"No reviewer recommendations for gate '{gate_key}'.",
        )
    return ReviewerRecommendationResponse.model_validate(row)


def _brief_to_notes(brief: dict) -> str:
    """Render a brief into the prose replayed at later gates."""
    lines = [brief.get("summary", "")]
    if brief.get("attack_chain"):
        lines.append("Chain: " + " -> ".join(str(s) for s in brief["attack_chain"]))
    if brief.get("thin_areas"):
        lines.append("Thin: " + "; ".join(str(s) for s in brief["thin_areas"]))
    if brief.get("notes_for_later_gates"):
        lines.append(str(brief["notes_for_later_gates"]))
    if brief.get("analyst_edited"):
        lines.append(
            "(The analyst corrected this read. Treat it as authoritative.)"
        )
    return "\n".join(x for x in lines if x).strip()


async def _thread_state(graph, source_id: uuid.UUID) -> dict | None:
    """Current checkpoint state. thread_id == source_id in this pipeline."""
    try:
        snapshot = await graph.aget_state(
            {"configurable": {"thread_id": str(source_id)}}
        )
    except Exception:  # noqa: BLE001 — a brief edit must not 500 on a checkpoint read
        logger.warning("could not read state for %s", source_id, exc_info=True)
        return None
    if snapshot is None or snapshot.values is None:
        return None
    return dict(snapshot.values)


async def _current_reviewable_gate(graph, source_id: uuid.UUID) -> str | None:
    """The gates_enabled key of the gate the graph is paused at, if any."""
    try:
        snapshot = await graph.aget_state(
            {"configurable": {"thread_id": str(source_id)}}
        )
        next_nodes = tuple(snapshot.next or ()) if snapshot else ()
    except Exception:  # noqa: BLE001 — a brief edit must not 500 on this
        logger.warning(
            "could not read next nodes for %s", source_id, exc_info=True,
        )
        return None
    for node in next_nodes:
        gate = GATES_BY_NODE.get(node)
        if gate is not None and gate.state_key in REVIEWERS:
            return gate.state_key
    return None
