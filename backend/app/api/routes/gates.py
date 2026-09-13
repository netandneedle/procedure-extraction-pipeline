"""Gate interaction endpoints for analyst review.

The pipeline pauses at gate nodes (interrupt_before). While paused:
1. Frontend calls GET /pending to fetch review items from checkpointer state.
2. Analyst reviews items in the UI.
3. Frontend calls POST /submit with the full review payload.
4. Backend writes decisions to LangGraph state via update_state() and
   schedules the resume as a BackgroundTask, returning 202 immediately.

Resume semantics:
  Each gate submit writes the analyst decisions to state, then kicks off
  the resume via stream_with_sync (same helper used for the initial
  pipeline run). Because the next leg can take 30+ seconds (Gate 0
  submit triggers a full re-chunk), we do NOT wait for astream to exit
  before responding. The UI tracks progress through the queue status
  column + WebSocket events emitted by stream_with_sync as each node
  runs.

One /submit endpoint per gate. The gate node processes the raw decisions
into downstream state fields (validated_entities, gate1_decisions, etc.).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import get_graph
from app.api.routes._gate_registry import GATES_BY_INT_ID, GATES_BY_NODE
from app.api.routes.pipeline import launch_pipeline_task, stream_with_sync
from app.api.routes.ws import manager as ws_manager
from app.models.base import async_session
from app.nodes.deterministic.attack_conditions import extract_conditions
from app.nodes.deterministic.attack_operators import infer_operators
from app.schemas.api import (
    ChunkGateReviewPayload,
    ChunkGateSubmit,
    Gate0Submit,
    Gate1Submit,
    Gate2Submit,
    GateReviewPayload,
    GateSubmitResponse,
)
from app.services import queue as queue_service
from app.services.reviewer import store as reviewer_store

logger = logging.getLogger(__name__)

router = APIRouter()


# Each gate's metadata (predecessor node, expected/resuming status,
# state field names) comes from app.api.routes._gate_registry.GATES.
# See that module's docstring for the rationale on the `predecessor_node`
# / `as_node` invariant.


async def _get_thread_state(
    graph, thread_id: uuid.UUID,
) -> tuple[dict, tuple, str | None]:
    """Read current state from the LangGraph checkpointer.

    Returns (state_values, next_nodes, checkpoint_id). next_nodes is the
    tuple of node names the graph will run next (populated when paused at
    an interrupt). checkpoint_id is the LangGraph checkpoint identifier
    at read time — used as an optimistic-concurrency token on submits.
    """
    config = {"configurable": {"thread_id": str(thread_id)}}
    state = await graph.aget_state(config)
    if state is None or state.values is None:
        raise HTTPException(
            status_code=404,
            detail=f"No pipeline state found for thread {thread_id}",
        )
    # Defensively pull checkpoint_id — the StateSnapshot's `config` dict
    # may be missing in some LangGraph internal paths and is replaced by
    # MagicMock in tests; only accept a real string.
    checkpoint_id: str | None = None
    try:
        cp = state.config["configurable"]["checkpoint_id"]
        if isinstance(cp, str):
            checkpoint_id = cp
    except (TypeError, KeyError, AttributeError):
        pass
    return state.values, tuple(state.next or ()), checkpoint_id


def _check_checkpoint(submitted: str | None, current: str | None) -> None:
    """Reject stale submits.

    `submitted` is the checkpoint_id the analyst's session captured at
    GET /pending; `current` is what the checkpointer holds right now.
    Mismatch means another submit (or the runner) advanced state between
    fetch and submit — we 409 so the client can refetch and retry.

    `submitted is None` is treated as "client opted out of the race
    guard" — kept for back-compat with any non-UI consumer. The standard
    frontend always echoes the token.
    """
    if submitted is None:
        return
    if current is None:
        # Snapshot didn't expose a checkpoint id — fall through rather
        # than blocking submits on a langgraph implementation detail.
        return
    if submitted != current:
        raise HTTPException(
            status_code=409,
            detail=(
                "Review is stale: the pipeline state has advanced since "
                "you fetched it (someone else may have submitted, or the "
                "runner moved on). Refresh and try again."
            ),
        )


def _validate_gate_id(gate_id: int) -> None:
    """Ensure gate_id is in the int-keyed range (currently 0/1/2)."""
    if gate_id not in GATES_BY_INT_ID:
        valid = ", ".join(str(i) for i in sorted(GATES_BY_INT_ID))
        raise HTTPException(
            status_code=400,
            detail=f"Invalid gate_id: {gate_id}. Must be one of {valid}.",
        )


# =============================================================================
# Chunk-review gate
# =============================================================================
#
# String-keyed endpoints registered BEFORE the int-keyed catch-all so
# /{thread_id}/chunks/... resolves to these handlers rather than 422'ing
# on int validation of "chunks".

async def _flip_to_resuming_chunks(source_id_uuid: uuid.UUID, thread_id: str) -> str:
    """Mirror of `_flip_to_resuming` for the string-keyed chunk gate."""
    resuming_status = GATES_BY_NODE["gate_chunks"].resuming_status
    async with async_session() as db:
        await queue_service.update_status(db, source_id_uuid, resuming_status)
    if ws_manager.has_subscribers(thread_id):
        await ws_manager.broadcast(thread_id, {
            "type": "status_change",
            "thread_id": thread_id,
            "data": {"status": resuming_status, "gate_id": "chunks"},
        })
    return resuming_status


@router.get(
    "/{thread_id}/chunks/pending",
    response_model=ChunkGateReviewPayload,
    summary="Get pending review data for the chunk-review gate",
)
async def get_pending_chunk_review(
    thread_id: uuid.UUID,
    graph=Depends(get_graph),
):
    """Load chunk-review payload: chunks + parsed_text + entities."""
    state, next_nodes, checkpoint_id = await _get_thread_state(graph, thread_id)

    expected = "gate_chunks"
    current_status = state.get("status", "")
    at_gate = expected in next_nodes or current_status == expected
    if not at_gate:
        raise HTTPException(
            status_code=409,
            detail=f"Pipeline is at '{current_status}', not '{expected}'. "
                   f"Cannot fetch chunk review.",
        )

    # Surface chunk_operators live from current geometry so the canvas
    # can render operator-kind dropdowns on branch/converge nodes.
    # Existing overrides on state (from a prior submission to the same
    # gate) are preserved via the existing_operators merge — same
    # contract normalize uses.
    is_sequential = bool(state.get("is_sequential", True))
    chunks_state = state.get("chunks", []) or []
    chunk_operators = infer_operators(
        chunks_state,
        is_sequential,
        existing_operators=state.get("chunk_operators", {}) or {},
        condition_anchors=set(state.get("chunk_conditions", {}) or {}),
    )

    # Surface chunk_conditions live from current chunks (post-finalize
    # preconditions). Merge with any prior analyst edits already in
    # state.chunk_conditions — analyst-edited descriptions / partitions
    # take precedence over what extract_conditions would re-derive from
    # the chunker's emission.
    conditions_state = state.get("chunk_conditions", {}) or {}
    chunk_conditions = extract_conditions(chunks_state, is_sequential)
    for anchor, override in conditions_state.items():
        if anchor in chunk_conditions:
            # Analyst override wins on description / pattern / partition.
            chunk_conditions[anchor] = {**chunk_conditions[anchor], **override}
        else:
            # Analyst added a condition on a chunk the chunker didn't
            # mark — keep it.
            chunk_conditions[anchor] = override

    return ChunkGateReviewPayload(
        thread_id=str(thread_id),
        source_id=state.get("source_id", ""),
        status=current_status,
        chunks=chunks_state,
        parsed_text=state.get("parsed_text", ""),
        validated_entities=state.get("validated_entities", []),
        classified_sections=state.get("classified_sections", []),
        is_sequential=is_sequential,
        sequentiality_rationale=state.get("sequentiality_rationale", ""),
        chunk_operators=chunk_operators,
        chunk_conditions=chunk_conditions,
        checkpoint_id=checkpoint_id,
    )


@router.post(
    "/{thread_id}/chunks/submit",
    response_model=GateSubmitResponse,
    status_code=202,
    summary="Submit chunk-review decisions",
)
async def submit_chunk_review(
    thread_id: uuid.UUID,
    body: ChunkGateSubmit,
    graph=Depends(get_graph),
):
    """Submit chunk-review decisions and kick off the resume.

    The body's shape is documented in `ChunkGateSubmit`. The gate_chunks
    node will:
    - On reject: route back to chunk_behaviors with rerun_feedback.
    - Otherwise: apply per-chunk approve/edit/drop, append added_chunks,
      mutate precedes_ids per edges, then route to extract_techniques.
    """
    state, next_nodes, checkpoint_id = await _get_thread_state(graph, thread_id)
    expected = "gate_chunks"
    at_gate = expected in next_nodes or state.get("status") == expected
    if not at_gate:
        raise HTTPException(
            status_code=409,
            detail=f"Pipeline is at '{state.get('status')}', not '{expected}'.",
        )
    # Optimistic-concurrency guard — see _check_checkpoint docstring.
    _check_checkpoint(body.checkpoint_id, checkpoint_id)

    # by_alias=False so the dict uses Python-safe field names (`from_`
    # instead of the reserved keyword `from`). Wire format is still
    # `{from, to}` — Pydantic's alias accepts that on the way in; we just
    # don't propagate the keyword back into state.chunk_reviews where a
    # future logging or refactor could trip on it. The gate processor reads
    # `from_`.
    # as_node uses the immediate-predecessor from the gate registry so
    # the gate runs next with the analyst payload — see _gate_registry.py
    # for the rationale. Drop our own checkpoint_id before forwarding to
    # the gate processor — it's an HTTP-layer concept.
    config = {"configurable": {"thread_id": str(thread_id)}}
    payload = body.model_dump(by_alias=False, exclude_none=True)
    payload.pop("checkpoint_id", None)
    await graph.aupdate_state(
        config, {"chunk_reviews": payload},
        as_node=GATES_BY_NODE["gate_chunks"].predecessor_node,
    )

    source_id = state.get("source_id") or str(thread_id)
    source_id_uuid = uuid.UUID(source_id)

    # Score the AI reviewer against what the analyst actually did. Four
    # channels rather than the two the other gates have, so the extras go as
    # a dict — see diff_chunk_gate. Best-effort: measurement must never cost
    # a submission.
    await _record_reviewer_outcome(
        source_id_uuid, "chunks", payload.get("decisions") or [],
        {
            "added_chunks": payload.get("added_chunks") or [],
            "edges": payload.get("edges") or [],
            "reject": payload.get("reject"),
        },
    )

    resuming_status = await _flip_to_resuming_chunks(source_id_uuid, str(thread_id))

    # Detached pipeline run — see launch_pipeline_task for the rationale.
    launch_pipeline_task(
        stream_with_sync(graph, str(thread_id), source_id_uuid, None),
        source_id_uuid,
    )

    # gate_id=-1 is a sentinel: the chunk gate isn't in the int-id namespace.
    return GateSubmitResponse(
        thread_id=str(thread_id),
        gate_id=-1,
        status=resuming_status,
        next_node="resuming",
    )


@router.get(
    "/{thread_id}/{gate_id}/pending",
    response_model=GateReviewPayload,
    summary="Get pending review data for a gate",
)
async def get_pending_review(
    thread_id: uuid.UUID,
    gate_id: int,
    graph=Depends(get_graph),
):
    """Load the review payload for a gate.

    Returns the items the analyst needs to review, by internal gate id
    (the UI numbers them 0/2/3; the chunk gate has its own endpoints):
    - gate_id 0: extracted entities
    - gate_id 1: procedure drafts + technique mappings
    - gate_id 2: normalized drafts + entity relationships
    """
    _validate_gate_id(gate_id)
    state, next_nodes, checkpoint_id = await _get_thread_state(graph, thread_id)
    gate = GATES_BY_INT_ID[gate_id]

    # Verify pipeline is actually paused at this gate.
    # Primary check: state.next tells us the graph is interrupted before this gate.
    # Fallback: status field (set by the gate node itself after it runs).
    expected = gate.expected_status
    current_status = state.get("status", "")
    at_gate = expected in next_nodes or current_status == expected
    if not at_gate:
        raise HTTPException(
            status_code=409,
            detail=f"Pipeline is at '{current_status}', not '{expected}'. "
                   f"Cannot fetch review for gate {gate_id}.",
        )

    items = state.get(gate.item_field, [])

    # Build context depending on gate
    context: dict = {}
    if gate_id == 0:
        context["metadata"] = state.get("metadata", {})
        context["parse_warnings"] = state.get("parse_warnings", [])
        # Sequentiality classification — surfaced as a chip in the Gate 0
        # header so the analyst can sanity-check the auto-detect decision
        # before approving entities. The resolved boolean + rationale come
        # from extract_entities (see SOURCE STRUCTURE block in its prompt).
        context["sequentiality"] = state.get("sequentiality", "auto")
        context["is_sequential"] = bool(state.get("is_sequential", True))
        context["sequentiality_rationale"] = state.get(
            "sequentiality_rationale", ""
        )
    elif gate_id == 1:
        context["technique_mappings"] = state.get("technique_mappings", {})
        # Possible-bucket picks the C+A+D pick step parked for analyst review.
        # The analyst can promote any of these into the bundle via the
        # `promotions` array on the Gate 1 submit body. Stays separate from
        # technique_mappings so the UI can render a distinct review lane.
        context["technique_mappings_for_review"] = state.get(
            "technique_mappings_for_review", {}
        )
        context["chunks"] = state.get("chunks", [])
        context["validated_entities"] = state.get("validated_entities", [])
    elif gate_id == 2:
        context["drafts"] = state.get("drafts", [])
        context["normalized_drafts"] = state.get("normalized_drafts", [])
        context["validated_entities"] = state.get("validated_entities", [])
        context["stix_preview"] = {}  # Future: pre-render STIX for preview

    return GateReviewPayload(
        gate_id=gate_id,
        thread_id=str(thread_id),
        source_id=state.get("source_id", ""),
        status=current_status,
        items=items,
        context=context,
        checkpoint_id=checkpoint_id,
    )


async def _flip_to_resuming(source_id_uuid: uuid.UUID, thread_id: str, gate_id: int) -> str:
    """Write the resuming status to the queue + broadcast before resume.

    Called inline from the submit endpoint (before returning 202) so the
    5s polling UI immediately sees the card move out of the "pending
    review" state, rather than waiting for stream_with_sync's first
    status write.
    """
    resuming_status = GATES_BY_INT_ID[gate_id].resuming_status
    async with async_session() as db:
        await queue_service.update_status(db, source_id_uuid, resuming_status)
    if ws_manager.has_subscribers(thread_id):
        await ws_manager.broadcast(thread_id, {
            "type": "status_change",
            "thread_id": thread_id,
            "data": {"status": resuming_status, "gate_id": gate_id},
        })
    return resuming_status


# Both moved into the reviewer package: the differ registry to outcomes.py and
# the recorder to store.py. They are reviewer-domain, and the autonomous runner
# in pipeline.py needs them too — which this module cannot provide, since
# gates.py already imports from pipeline.py.
async def _record_reviewer_outcome(
    source_id: uuid.UUID,
    gate_key: str,
    reviews: list[dict],
    extras: list[dict] | dict[str, Any],
) -> None:
    """Attach the analyst's actual submission to the reviewer's recommendation.

    Thin wrapper so the four submit routes below read unchanged. `auto` is
    left at its default here by definition: reaching this module means a human
    posted the submission.
    """
    await reviewer_store.record_gate_outcome(source_id, gate_key, reviews, extras)


@router.post(
    "/{thread_id}/0/submit",
    response_model=GateSubmitResponse,
    status_code=202,
    summary="Submit Gate 0 review (entities)",
)
async def submit_gate_0(
    thread_id: uuid.UUID,
    body: Gate0Submit,
    graph=Depends(get_graph),
):
    """Submit entity review decisions and kick off the resume.

    Returns 202 immediately after writing reviews to state. The gate_0
    node and downstream chunking run as a background task; status
    updates flow through the queue row + WebSocket events.

    The gate_0 node processes these decisions: approve entities as-is,
    apply edits, or remove entities from downstream processing.
    """
    state, next_nodes, checkpoint_id = await _get_thread_state(graph, thread_id)
    at_gate = "gate_0" in next_nodes or state.get("status") == "gate_0"
    if not at_gate:
        raise HTTPException(
            status_code=409,
            detail=f"Pipeline is at '{state.get('status')}', not 'gate_0'.",
        )
    _check_checkpoint(body.checkpoint_id, checkpoint_id)

    # Write raw reviews to state. This mutates the checkpoint synchronously
    # so a follow-up GET /status reflects the submission immediately.
    # as_node is the immediate predecessor of gate_0 — see the docstring on
    # _gate_registry.py for the rationale.
    config = {"configurable": {"thread_id": str(thread_id)}}
    reviews = [r.model_dump() for r in body.reviews]
    added = [a.model_dump() for a in body.added_entities]
    await graph.aupdate_state(
        config, {"gate0_reviews": reviews, "gate0_added_entities": added},
        as_node=GATES_BY_NODE["gate_0"].predecessor_node,
    )

    # Flip queue status to resuming_from_gate_0 before scheduling the
    # background task — this keeps the Kanban card in the Entity Review
    # column but removes the "Review" affordance on the next poll tick.
    source_id = state.get("source_id") or str(thread_id)
    source_id_uuid = uuid.UUID(source_id)

    # If the AI reviewer recommended anything at this gate, record what the
    # analyst ACTUALLY did against it. This is the agreement data that decides
    # whether the gate is ever safe to run unattended — and it only exists
    # while a human is still in the loop. Best-effort: measurement must never
    # cost a submission.
    await _record_reviewer_outcome(source_id_uuid, "entities", reviews, added)

    resuming_status = await _flip_to_resuming(source_id_uuid, str(thread_id), 0)

    # Fire-and-forget: stream_with_sync drives the graph, writes mid-run
    # status updates, and handles the next gate pause or completion.
    # Detached pipeline run — see launch_pipeline_task for the rationale.
    launch_pipeline_task(
        stream_with_sync(graph, str(thread_id), source_id_uuid, None),
        source_id_uuid,
    )

    return GateSubmitResponse(
        thread_id=str(thread_id),
        gate_id=0,
        status=resuming_status,
        next_node="resuming",
    )


@router.post(
    "/{thread_id}/1/submit",
    response_model=GateSubmitResponse,
    status_code=202,
    summary="Submit Gate 1 review (procedures + techniques)",
)
async def submit_gate_1(
    thread_id: uuid.UUID,
    body: Gate1Submit,
    graph=Depends(get_graph),
):
    """Submit procedure draft review decisions and kick off the resume.

    Returns 202 immediately. The gate_1 node (and whatever comes after
    per routing) runs as a background task.

    The gate_1 node will:
    - Apply edits to approved/edited drafts
    - Compute rejection routing
      (BAD_CHUNK_BOUNDARY -> re-chunk, others -> re-extract techniques)
    - Set gate1_approved_draft_ids for downstream normalize
    """
    state, next_nodes, checkpoint_id = await _get_thread_state(graph, thread_id)
    at_gate = "gate_1" in next_nodes or state.get("status") == "gate_1"
    if not at_gate:
        raise HTTPException(
            status_code=409,
            detail=f"Pipeline is at '{state.get('status')}', not 'gate_1'.",
        )
    _check_checkpoint(body.checkpoint_id, checkpoint_id)

    config = {"configurable": {"thread_id": str(thread_id)}}
    reviews = [r.model_dump() for r in body.reviews]
    promotions = [p.model_dump() for p in body.promotions]
    # as_node is the immediate predecessor of gate_1 — see _gate_registry.py
    # for the rationale (tag the producer of the gate's inputs so the gate
    # runs next with the analyst payload).
    await graph.aupdate_state(
        config,
        {"gate1_reviews": reviews, "gate1_promotions": promotions},
        as_node=GATES_BY_NODE["gate_1"].predecessor_node,
    )

    source_id = state.get("source_id") or str(thread_id)
    source_id_uuid = uuid.UUID(source_id)

    # Agreement data for this gate — see _record_reviewer_outcome.
    await _record_reviewer_outcome(
        source_id_uuid, "procedures", reviews, promotions,
    )

    resuming_status = await _flip_to_resuming(source_id_uuid, str(thread_id), 1)

    # Detached pipeline run — see launch_pipeline_task for the rationale.
    launch_pipeline_task(
        stream_with_sync(graph, str(thread_id), source_id_uuid, None),
        source_id_uuid,
    )

    return GateSubmitResponse(
        thread_id=str(thread_id),
        gate_id=1,
        status=resuming_status,
        next_node="resuming",
    )


@router.post(
    "/{thread_id}/2/submit",
    response_model=GateSubmitResponse,
    status_code=202,
    summary="Submit Gate 2 review (relationships)",
)
async def submit_gate_2(
    thread_id: uuid.UUID,
    body: Gate2Submit,
    graph=Depends(get_graph),
):
    """Submit relationship review decision and kick off the resume.

    Returns 202 immediately. Serialization (approve) or re-normalize
    (reject) runs as a background task.
    """
    state, next_nodes, checkpoint_id = await _get_thread_state(graph, thread_id)
    at_gate = "gate_2" in next_nodes or state.get("status") == "gate_2"
    if not at_gate:
        raise HTTPException(
            status_code=409,
            detail=f"Pipeline is at '{state.get('status')}', not 'gate_2'.",
        )
    _check_checkpoint(body.checkpoint_id, checkpoint_id)

    config = {"configurable": {"thread_id": str(thread_id)}}
    # Per-relationship mode: write reviews list to gate2_reviews.
    # Batch mode: write approved/feedback to gate2_review.
    # as_node is the immediate predecessor of gate_2 — see _gate_registry.py
    # for the rationale.
    g2_predecessor = GATES_BY_NODE["gate_2"].predecessor_node
    if body.reviews is not None:
        reviews = [r.model_dump() for r in body.reviews]
        await graph.aupdate_state(
            config, {"gate2_reviews": reviews}, as_node=g2_predecessor,
        )
    else:
        review = {"approved": body.approved, "feedback": body.feedback}
        await graph.aupdate_state(
            config, {"gate2_review": review}, as_node=g2_predecessor,
        )

    source_id = state.get("source_id") or str(thread_id)
    source_id_uuid = uuid.UUID(source_id)

    # Agreement data for this gate. Batch mode carries no per-relationship
    # decisions, so there is nothing to diff against and this no-ops.
    await _record_reviewer_outcome(
        source_id_uuid, "bundle",
        [r.model_dump() for r in (body.reviews or [])], [],
    )

    resuming_status = await _flip_to_resuming(source_id_uuid, str(thread_id), 2)

    # Detached pipeline run — see launch_pipeline_task for the rationale.
    launch_pipeline_task(
        stream_with_sync(graph, str(thread_id), source_id_uuid, None),
        source_id_uuid,
    )

    return GateSubmitResponse(
        thread_id=str(thread_id),
        gate_id=2,
        status=resuming_status,
        next_node="resuming",
    )
