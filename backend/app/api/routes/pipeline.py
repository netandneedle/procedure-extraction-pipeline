"""Pipeline control endpoints.

Start pipeline runs (background task) and check status.
The pipeline runs asynchronously. POST /run returns immediately with
a thread_id. The client polls GET /status/{thread_id} for progress.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db, get_graph
from app.api.routes._gate_registry import GATES_BY_NODE
from app.api.routes.ws import build_status_event, manager as ws_manager
from app.graph.state import (
    PipelineStatus,
    gate_mode,
    is_gate_enabled,
    normalize_gate_modes,
    normalize_gates,
)
from app.models.base import async_session
from app.schemas.api import (
    PipelineRunRequest,
    PipelineRunResponse,
    PipelineStatusResponse,
)
from app.services import queue as queue_service
from app.services.reviewer import run_reviewer
from app.services.reviewer.apply import build_auto_submission
from app.services.reviewer.store import gate_pass_count, record_gate_outcome

logger = logging.getLogger(__name__)

# LangGraph node name -> gates_enabled dict key. Derived from the gate
# registry so adding a new gate is one place. Used by stream_with_sync's
# auto-skip loop: when the graph pauses at a gate whose key is disabled
# in state.gates_enabled, we drive the graph forward one more step so
# the gate node runs and self-skips.
_GATE_NODE_TO_KEY = {g.node_name: g.state_key for g in GATES_BY_NODE.values()}

# gates_enabled key -> the status shown while the AI reviewer works on it.
#
# Doubles as the registry of which gates HAVE an AI reviewer: a gate absent
# here falls through to human review no matter what its mode says. Membership
# is required rather than derived (f"reviewing_{key}") because Kanban columns
# enumerate statuses with no fallback — a status no column lists makes the
# card disappear from the board rather than merely look wrong.
#
# Add a gate here in the same change that adds its reviewer, its PipelineStatus
# value, and its Kanban mapping. All four or none.
_REVIEWING_STATUS = {
    "entities": PipelineStatus.REVIEWING_ENTITIES.value,
    "chunks": PipelineStatus.REVIEWING_CHUNKS.value,
    "procedures": PipelineStatus.REVIEWING_PROCEDURES.value,
    "bundle": PipelineStatus.REVIEWING_BUNDLE.value,
}

# Upper bound on auto-skip resumes in a single run. The loop below resumes the
# graph past each disabled gate; normally that is at most once per gate. But a
# gate that keeps routing BACKWARDS (a rejection still sitting in state, say)
# re-enters its own interrupt, and the loop would resume it forever — a hot
# spin writing a DB status update per iteration, with no error and no end.
# Found while mutation-testing the loop: a gate that stopped honoring its
# disabled flag hung the test runner outright.
#
# Generous headroom over the real gate count so a legitimate rejection cycle
# is never mistaken for a spin.
_MAX_GATE_AUTO_SKIPS = max(12, len(_GATE_NODE_TO_KEY) * 3)

# Source statuses from which POST /pipeline/run may (re)start a run. A source
# mid-flight or parked at a gate must not be restarted from the top — that
# would race the running graph or discard analyst decisions already applied to
# its checkpoint. 'failed' is included so the Retry button works; see the
# guard in start_pipeline.
_RESTARTABLE_STATUSES = frozenset({"queued", "failed"})

# LangGraph node about-to-run -> the in-progress status to display while it
# runs. LangGraph only checkpoints at node boundaries, so the status a node
# self-reports isn't persisted until it FINISHES — a long node (figure
# extraction, entity extraction) would otherwise show its predecessor's status
# the whole time it runs. Deriving the displayed status from `state.next` (the
# upcoming node) instead makes the Kanban reflect what is RUNNING. Gates are
# intentionally absent: `interrupt_before` halts the stream before a gate, so a
# gate never appears as `next` inside the streaming loop.
_NODE_TO_STATUS = {
    "extract_figures": PipelineStatus.EXTRACTING_FIGURES.value,
    "classify_sections": PipelineStatus.CLASSIFYING_SECTIONS.value,
    "extract_entities": PipelineStatus.EXTRACTING_ENTITIES.value,
    "chunk_behaviors": PipelineStatus.CHUNKING.value,
    "extract_techniques": PipelineStatus.EXTRACTING_TECHNIQUES.value,
    "draft_procedures": PipelineStatus.DRAFTING.value,
    "normalize": PipelineStatus.NORMALIZING.value,
    "serialize_stix": PipelineStatus.SERIALIZING.value,
    "distribute": PipelineStatus.DISTRIBUTING.value,
    # distribute writes COMPLETED, but synthesize_feedback still runs after
    # it — an LLM call. Without this entry the card reads "Complete" while
    # that call is in flight, which is the exact lie this map exists to stop.
    "synthesize_feedback": PipelineStatus.SYNTHESIZING_FEEDBACK.value,
}


def _displayed_status(self_status: str, next_node: str | None) -> str:
    """Pick the status to persist after a node completes.

    Prefer the upcoming node's in-progress status so the card reflects what is
    RUNNING, not what just finished. Three guards:
      * A transient ``resuming_from_gate_*`` status (set by a gate node after
        an analyst submission) is preserved as-is — it intentionally keeps the
        card in the gate column while downstream restarts (CLAUDE.md status
        gotcha). Without this guard the next node (e.g. ``normalize``) would
        immediately overwrite it.
      * FAILED is terminal and always survives. The edge into
        ``synthesize_feedback`` is unconditional, so a run that ``distribute``
        just marked failed still has a next node — and without this guard its
        in-progress status would paper over the failure.
      * If the upcoming node isn't a known long-running stage — a gate, END,
        or absent — fall back to the node's own self-reported status.
    """
    if self_status.startswith("resuming_from_"):
        return self_status
    if self_status == PipelineStatus.FAILED.value:
        return self_status
    if next_node and next_node in _NODE_TO_STATUS:
        return _NODE_TO_STATUS[next_node]
    return self_status

# Strong references to in-flight pipeline tasks. asyncio.create_task
# only weakly references its tasks; without holding a reference here a
# concurrent garbage-collection sweep can abandon a running pipeline
# mid-stream. The done callback removes entries so the set doesn't grow.
_PIPELINE_TASKS: set[asyncio.Task] = set()


# Prefix for the asyncio task name that carries a run's source id. Naming the
# task is how `cancel_pipeline_task` finds it — cheaper than a parallel dict,
# and it also makes the source visible in `asyncio.all_tasks()` when
# debugging a stuck run.
_TASK_NAME_PREFIX = "pipeline:"


def launch_pipeline_task(coro, source_id: uuid.UUID | str | None = None) -> asyncio.Task:
    """Launch a detached pipeline run via asyncio.create_task.

    Previously these went through FastAPI's BackgroundTasks, which runs
    each task *after* the response on the same event loop but serially
    across the worker — a single pipeline's awaits blocked the worker
    from starting the next pipeline. asyncio.create_task fires the run
    immediately and concurrently, so submitting a second source mid-run
    no longer waits for the first to finish.

    Tasks are tracked in `_PIPELINE_TASKS` so they survive GC; any
    unhandled exception inside the task is logged here (asyncio
    otherwise swallows exceptions in unawaited tasks until shutdown).

    `source_id` names the task so `cancel_pipeline_task` can find it when the
    source is deleted mid-run.
    """
    name = f"{_TASK_NAME_PREFIX}{source_id}" if source_id is not None else None
    task = asyncio.create_task(coro, name=name)
    _PIPELINE_TASKS.add(task)

    def _on_done(t: asyncio.Task) -> None:
        _PIPELINE_TASKS.discard(t)
        if t.cancelled():
            return
        exc = t.exception()
        if exc is not None:
            logger.error("pipeline task crashed", exc_info=exc)

    task.add_done_callback(_on_done)
    return task


def cancel_pipeline_task(source_id: uuid.UUID | str) -> bool:
    """Cancel the in-flight run for a source, if there is one.

    Called when a source is deleted. Without this the detached task keeps
    running against a row that no longer exists and dies on
    `StaleDataError: UPDATE ... expected to update 1 row(s); 0 were matched`
    — a traceback that reads like a crash until you trace the thread id.

    Canceling is safe here precisely because `CancelledError` derives from
    `BaseException`: `stream_with_sync`'s `except Exception` does not catch
    it, so the run does not get marked failed on its way out, and `_on_done`
    already returns early for a canceled task.

    Returns True if a task was found and canceled.
    """
    name = f"{_TASK_NAME_PREFIX}{source_id}"
    for task in list(_PIPELINE_TASKS):
        if task.get_name() == name and not task.done():
            task.cancel()
            logger.info("canceled in-flight pipeline task for source %s", source_id)
            return True
    return False


router = APIRouter()


async def _fail_stalled(
    db, source_id: uuid.UUID, thread_id: str, laps: int,
    what: str, gate_name: str,
) -> None:
    """Mark a run failed because the runner kept advancing without progressing.

    Shared by both no-human paths — resuming past disabled gates, and applying
    an autonomous decision. Either can cycle if a gate routes backward into
    one that cannot self-approve, and a stuck source an analyst can see beats
    a silent hot loop nobody notices.
    """
    msg = (
        f"pipeline stalled: {what} {laps} times without advancing (last at "
        f"{gate_name}). Likely a rejection routing back into a gate that "
        f"cannot self-approve."
    )
    logger.error("%s (thread %s)", msg, thread_id)
    await queue_service.update_status(
        db, source_id, PipelineStatus.FAILED.value, error=msg,
    )


def _run_counts(values: dict | None) -> dict[str, int | None]:
    """The card's run counts, from PipelineState. NULL for a stage that has
    not run yet, so the card shows nothing rather than a misleading 0."""
    values = values or {}
    return {
        "entity_count": len(values["entities"]) if values.get("entities") is not None else None,
        "draft_count": len(values["drafts"]) if values.get("drafts") is not None else None,
        "objects_written": values.get("objects_written"),
    }


# How many times an unattended gate may send work back before a human is
# asked. One retry: a re-chunk or re-extraction with fresh guidance can
# genuinely produce a better answer, but a second identical request is a sign
# the rewind is not the fix, and that is a judgment call worth a person.
_MAX_AUTO_REWIND_PASSES = 1


async def _auto_apply(
    graph, config: dict, source_id: uuid.UUID, thread_id: str,
    gate_name: str, gate_key: str, payload: dict, state: dict,
) -> bool:
    """Submit the reviewer's own recommendation, with no human in between.

    Returns True when the graph should resume, False when this gate should
    fall through to a person — which is the honest answer in three cases: no
    converter for the gate, nothing to say, and a rewind that has already had
    its allowance of unattended retries (_MAX_AUTO_REWIND_PASSES).

    The state write is the same one the HTTP submit route makes, tagged with
    the same `as_node` — the gate's immediate predecessor, so LangGraph runs
    the gate next with these inputs rather than skipping it. What it must NOT
    do is call launch_pipeline_task: we are already inside stream_with_sync,
    and re-entering would put two drivers on one thread.
    """
    submission = build_auto_submission(gate_key, payload, state)
    if submission is None:
        return False

    # EVERY hand-off below logs its own reason before returning False. That is
    # not tidiness: an earlier version checked for empty channels before the
    # deferral reason, and since a deferral produced no channels the reason was
    # computed and thrown away — a deliberate hand-off then looked exactly like
    # a reviewer crash, both being a gate that paused with nothing logged.
    # Whatever is added here later must keep saying why.
    if not submission.channels:
        logger.info(
            "gate '%s' auto: reviewer proposed nothing to apply; "
            "handing to a human (thread %s)", gate_key, thread_id,
        )
        return False

    # The pass limit. A rewind throws this pass away and re-enters an upstream
    # node, so without a bound an unattended agent could loop: re-extract,
    # reject, re-extract, reject. One retry is allowed; a second time through
    # the same gate is a human's call.
    #
    # gate_pass_count INCLUDES the turn being applied right now — the reviewer
    # writes its recommendation row before this runs — so it is this visit's
    # number, not the count of earlier ones. First visit reads 1, not 0.
    #
    # Hence `visit > limit`: with a limit of 1, visit 1 rewinds and visit 2
    # hands over, which is exactly one automatic rewind. Getting this boundary
    # wrong in either direction is silent — one too many is an extra Opus pass
    # per source, one too few is a hand-off that should not have happened —
    # so TestAutoApplyEnforcesThePassLimit pins both sides of it.
    #
    # The counter is the ReviewerRecommendation rows the store already writes
    # per visit: no new state, no migration.
    if submission.rewind:
        visit = await gate_pass_count(source_id, gate_key)
        if visit > _MAX_AUTO_REWIND_PASSES:
            logger.info(
                "gate '%s' set to auto but deferring to a human: this is "
                "visit %d and the unattended rewind limit is %d (thread %s)",
                gate_key, visit, _MAX_AUTO_REWIND_PASSES, thread_id,
            )
            return False
        logger.info(
            "gate '%s' auto: applying a rewind on visit %d (limit %d) "
            "(thread %s)",
            gate_key, visit, _MAX_AUTO_REWIND_PASSES, thread_id,
        )

    await graph.aupdate_state(
        config, submission.channels,
        as_node=GATES_BY_NODE[gate_name].predecessor_node,
    )
    # auto=True keeps this out of the agreement readout. An agent scoring its
    # own decisions would report perfect agreement and mean nothing by it.
    await record_gate_outcome(
        source_id, gate_key, submission.decisions, submission.extras, auto=True,
    )
    logger.info(
        "gate '%s' auto-applied %d decision(s) unattended (thread %s)",
        gate_key, len(submission.decisions), thread_id,
    )
    return True


async def stream_with_sync(
    graph,
    thread_id: str,
    source_id: uuid.UUID,
    graph_input,
) -> None:
    """Drive a LangGraph run (initial or resume) and sync status to DB + WS.

    Used by both the initial pipeline start (graph_input = initial_state
    dict) and gate resumes (graph_input = None, LangGraph picks up from
    the checkpointed state).

    At every node transition, this:
      1. Reads state from the checkpointer.
      2. Writes the current status (and any error / persistence_errors)
         to the source queue row.
      3. Broadcasts a status event to any WebSocket subscribers.

    When the run pauses at a gate interrupt, the gate's name
    (e.g. "gate_0") is written as the DB status so the Kanban card
    lands in the correct review column.

    On unhandled exception, the source is marked failed and the error
    is broadcast to subscribers.
    """
    config = {"configurable": {"thread_id": thread_id}}

    # Hold ONE session for the whole run. Previously we opened a fresh
    # async_session() inside every astream iteration — for a typical
    # source that's 10–15 connection cycles plus a couple more at the
    # gate-pause + completion boundaries. Each update_status still
    # commits (so locks don't span the run), but the session/connection
    # is reused. On unhandled exception we open a *fresh* session in
    # the except block because the original may be poisoned mid-tx.
    try:
        async with async_session() as db:
            # Drive the graph in a loop so we can auto-skip past disabled
            # gates. The static `interrupt_before` list set at compile time
            # pauses the graph before EVERY gate, but per-source
            # `gates_enabled` may mark some gates as off. The gate node
            # itself self-skips when it runs (see is_gate_enabled checks
            # in app.nodes.gates), but the interrupt fires before the node
            # executes — so without this loop the graph would sit forever
            # at a "disabled" gate waiting for an analyst action that will
            # never come. Each iteration: stream until the next pause,
            # check whether the pause is at a disabled gate, and if so
            # resume to let the gate run + self-skip; otherwise persist
            # the paused state for the analyst and break.
            current_input = graph_input
            auto_skips = 0
            while True:
                async for _event in graph.astream(current_input, config):
                    # Read current status from state and sync to queue table
                    state = await graph.aget_state(config)
                    if state and state.values:
                        self_status = state.values.get("status", "")
                        # Show the node that's ABOUT to run, not the one that
                        # just finished, so a long node doesn't display its
                        # predecessor's status for minutes. `state.next` is the
                        # upcoming node(s); empty at END, never a gate here
                        # (the stream halts before gate interrupts).
                        next_node = state.next[0] if state.next else None
                        current_status = _displayed_status(self_status, next_node)
                        if current_status:
                            # Forward persistence_errors only when distribute
                            # has populated them (i.e. terminal states). Mid-
                            # run the field is absent, so we pass None and
                            # leave the column untouched.
                            pe = state.values.get("persistence_errors")
                            # Same pattern for bundle_corrections from
                            # validate_bundle: pass through when present,
                            # leave the column untouched otherwise.
                            bc = state.values.get("bundle_corrections")
                            # Forward state["error"] when a node has set it
                            # without raising (parse validation, LLM parse
                            # failures, distribute errors, etc.). Outer
                            # try/except below only catches unhandled
                            # exceptions; without this, Source.error stays
                            # NULL for soft-failed runs and the UI can't
                            # tell the analyst why status flipped to failed.
                            err = state.values.get("error")
                            await queue_service.update_status(
                                db, source_id, current_status,
                                error=err,
                                persistence_errors=pe if pe is not None else None,
                                bundle_corrections=bc if bc is not None else None,
                                run_counts=_run_counts(state.values),
                            )
                            # Broadcast to WebSocket subscribers. Overlay the
                            # derived status onto a COPY (state.values is shared
                            # with LangGraph's checkpoint cache — never mutate
                            # it in place; see H18). Keeps WS + DB consistent.
                            if ws_manager.has_subscribers(thread_id):
                                values = dict(state.values)
                                values["status"] = current_status
                                event = build_status_event(thread_id, values)
                                await ws_manager.broadcast(thread_id, event)

                # astream exited: either the run finished or we paused at
                # an interrupt. state.next is a tuple of node names about
                # to run.
                state = await graph.aget_state(config)
                if not state or not state.next:
                    break  # pipeline finished cleanly

                gate_name = state.next[0]
                gate_key = _GATE_NODE_TO_KEY.get(gate_name)
                # Auto-skip when the paused gate maps to a disabled key.
                # `is_gate_enabled` tolerates both the new dict shape and
                # the legacy bool shape from in-flight checkpoints.
                # graph_input=None tells LangGraph to pick up from the
                # checkpointed state and actually execute the gate node,
                # which will see the disabled flag and emit
                # RESUMING_FROM_GATE_X to advance.
                if gate_key and not is_gate_enabled(state.values or {}, gate_key):
                    auto_skips += 1
                    if auto_skips > _MAX_GATE_AUTO_SKIPS:
                        # Not a slow run — a cycle.
                        await _fail_stalled(
                            db, source_id, thread_id, auto_skips,
                            "resumed past disabled gates", gate_name,
                        )
                        break
                    current_input = None
                    continue

                # AI reviewer, for gates configured as assist or auto.
                #
                # Runs BEFORE the gate status is written so recommendations
                # exist by the time the card offers a Review affordance — an
                # analyst who opens the panel and finds it empty has learned
                # nothing except to distrust it.
                #
                # Never fatal. run_reviewer swallows its own failures and
                # returns None; the gate then behaves exactly as it does in
                # plain review mode. An assistant that can take the pipeline
                # down with it is a liability, not a feature.
                mode = gate_mode(state.values or {}, gate_key) if gate_key else "review"
                if mode in ("assist", "auto") and gate_key not in _REVIEWING_STATUS:
                    # Configured for AI review at a gate whose reviewer hasn't
                    # shipped. Fall through to a human rather than improvise.
                    #
                    # The membership test is doing real work: it also keeps us
                    # from writing a `reviewing_<gate>` status that no Kanban
                    # column lists. Columns enumerate statuses with no
                    # fallback, so an unmapped status doesn't degrade the card
                    # — it removes it from the board entirely.
                    logger.warning(
                        "gate '%s' is set to '%s' but no AI reviewer is "
                        "implemented for it yet; falling through to human "
                        "review. Thread %s.",
                        gate_key, mode, thread_id,
                    )
                    mode = "review"
                if mode in ("assist", "auto"):
                    reviewing_status = _REVIEWING_STATUS[gate_key]
                    await queue_service.update_status(
                        db, source_id, reviewing_status,
                    )
                    if ws_manager.has_subscribers(thread_id):
                        values = dict(state.values or {})
                        values["status"] = reviewing_status
                        await ws_manager.broadcast(
                            thread_id, build_status_event(thread_id, values),
                        )
                    payload = await run_reviewer(
                        source_id, gate_key, state.values or {},
                    )

                    # Autopilot. `payload is None` means the reviewer errored
                    # and has already fallen back to human review — the same
                    # contract assist mode depends on, so there is nothing
                    # extra to handle here.
                    if mode == "auto" and payload is not None:
                        if await _auto_apply(
                            graph, config, source_id, thread_id,
                            gate_name, gate_key, payload, state.values or {},
                        ):
                            # Counted against the same cap as disabled-gate
                            # resumes, because it is the same thing: the
                            # runner advancing a gate with nobody watching.
                            # Forward-only autopilot touches each gate once,
                            # so this only bites when a gate routes backward
                            # — which is precisely the runaway to stop.
                            auto_skips += 1
                            if auto_skips > _MAX_GATE_AUTO_SKIPS:
                                await _fail_stalled(
                                    db, source_id, thread_id, auto_skips,
                                    "auto-applied gate decisions", gate_name,
                                )
                                break
                            current_input = None
                            continue

                # Manual review needed: persist gate name + broadcast,
                # exit loop. Counts ride along so the card's "N entities"
                # line is right the moment it lands in the gate column.
                await queue_service.update_status(
                    db, source_id, gate_name, run_counts=_run_counts(state.values),
                )
                if ws_manager.has_subscribers(thread_id):
                    # COPY before mutating — `state.values` is shared with
                    # LangGraph's in-memory checkpoint cache; writing back
                    # to it can corrupt the next aget_state. We only need
                    # to overlay the gate name for the WS payload, not
                    # mutate state itself.
                    values = dict(state.values or {})
                    values["status"] = gate_name
                    event = build_status_event(thread_id, values)
                    await ws_manager.broadcast(thread_id, event)
                break
    except Exception as e:
        # Log the full traceback before doing anything else. Without this,
        # a soft failure surfaces only as a bare str(e) on the Source row
        # (e.g. "Set changed size during iteration") with no stack to point
        # at the culprit — which is exactly what made this class of bug hard
        # to trace.
        logger.exception(
            "pipeline run failed for thread %s (source %s)", thread_id, source_id
        )
        # Pipeline failed — mark source as failed. Open a *fresh* session:
        # the in-flight `db` above may be in a poisoned state from a mid-
        # transaction error, and we still need to persist the failure.
        #
        # Best-effort: also flush any validator audit (bundle_corrections)
        # captured in state. The mid-run sync loop normally persists these,
        # but if `aget_state` itself raised (e.g. PostgresSaver dropped the
        # connection) the most recent corrections never reached the row.
        # Re-read state defensively here; a failure to read must not mask
        # the original exception.
        bc = None
        try:
            failed_state = await graph.aget_state(config)
            if failed_state and failed_state.values:
                bc = failed_state.values.get("bundle_corrections")
        except Exception:  # noqa: BLE001 — never let the audit read hide the failure
            bc = None
        async with async_session() as db:
            await queue_service.update_status(
                db, source_id, "failed", error=str(e),
                bundle_corrections=bc if bc is not None else None,
            )
        # Broadcast failure to WebSocket subscribers
        if ws_manager.has_subscribers(thread_id):
            await ws_manager.broadcast(thread_id, {
                "type": "failed",
                "thread_id": thread_id,
                "data": {"status": "failed", "error": str(e)},
            })


async def _run_pipeline(
    graph,
    thread_id: str,
    initial_state: dict,
    source_id: uuid.UUID,
) -> None:
    """Background task: invoke the LangGraph pipeline from scratch.

    Thin wrapper around stream_with_sync that passes the initial state.
    The separate entry point is retained so call sites read naturally.
    """
    await stream_with_sync(graph, thread_id, source_id, initial_state)


@router.post(
    "/run",
    response_model=PipelineRunResponse,
    status_code=202,
    summary="Start a pipeline run",
)
async def start_pipeline(
    body: PipelineRunRequest,
    graph=Depends(get_graph),
    db: AsyncSession = Depends(get_db),
):
    """Trigger extraction pipeline for a source.

    The pipeline runs as a background task. Returns immediately with
    the thread_id for status polling.

    The source must exist in the queue and be `queued` or `failed`
    (a failed run can be retried in place).
    """
    source = await queue_service.get_source(db, body.source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    # 'failed' is retryable: the SourceCard renders its run button as
    # "Retry ▶" for failed sources and posts here, and the initial_state
    # below already resets error/correction-log fields specifically so a
    # previously-failed run merges clean values over its checkpoint. Without
    # 'failed' in this guard that button 409s on every card in the Failed
    # column and the retry path it was written for is unreachable.
    if source.status not in _RESTARTABLE_STATUSES:
        raise HTTPException(
            status_code=409,
            detail=(
                f"Source is '{source.status}', must be 'queued' or 'failed' "
                f"to start a run."
            ),
        )

    # Use source ID as thread_id for 1:1 mapping
    thread_id = str(source.id)

    # Record thread_id and update status. Clear any residual error from a
    # prior failed run so the analyst doesn't see a stale red badge
    # alongside an in-progress status.
    await queue_service.set_thread_id(db, source.id, source.id)
    await queue_service.clear_error(db, source.id)
    await queue_service.update_status(db, source.id, "parsing")

    # Determine gates_enabled (request override takes precedence).
    # normalize_gates handles legacy bool rows from databases that
    # haven't run the 20260501 migration yet.
    gates_enabled = normalize_gates(
        body.gates_enabled
        if body.gates_enabled is not None
        else source.gates_enabled
    )
    # Per-gate review mode. Absent on rows predating the column -> all
    # "review", i.e. today's behavior.
    gate_modes = normalize_gate_modes(getattr(source, "gate_modes", None))

    # Build initial state from source record. error, bundle_corrections,
    # gate1_correction_log, and technique_rerun_feedback are explicitly
    # nulled so a retry of a previously-failed (or re-queued) run merges
    # clean values over the checkpoint — thread_id is reused per source,
    # so any field omitted here keeps its prior-run checkpoint value.
    # Without the error reset, state.error from the prior validate_bundle
    # hard-fail bleeds back to the source row via the runner's aget_state
    # sync. Without the correction-log reset, run 1's analyst corrections
    # resurface in run 2: double-counted by synthesize_feedback, shown as
    # "captured this run" in the Feedback tab, and (for rerun feedback
    # stranded by a mid-run failure) injected as phantom-chunk guidance
    # into run 2's first extraction pass.
    initial_state = {
        "source_id": str(source.id),
        "channel": source.channel,
        "source_type": source.source_type,
        "raw_content_path": source.raw_content_path,
        "title": source.title,
        "metadata": source.metadata_,
        "source_reliability": source.source_reliability,
        "gates_enabled": gates_enabled,
        "gate_modes": gate_modes,
        "sequentiality": getattr(source, "sequentiality", "auto") or "auto",
        "extract_figures": bool(getattr(source, "extract_figures", True)),
        "status": "parsing",
        "current_node": "parse_and_validate",
        "error": None,
        "bundle_corrections": [],
        "gate1_correction_log": [],
        "chunk_correction_log": [],
        "technique_rerun_feedback": None,
    }

    # Launch pipeline as a detached asyncio task — see launch_pipeline_task
    # for the rationale (concurrent runs vs serial BackgroundTasks queue).
    launch_pipeline_task(
        _run_pipeline(graph, thread_id, initial_state, source.id), source.id,
    )

    return PipelineRunResponse(
        thread_id=thread_id,
        source_id=str(source.id),
        status="parsing",
    )


@router.get(
    "/status/{thread_id}",
    response_model=PipelineStatusResponse,
    summary="Get pipeline status",
)
async def get_pipeline_status(
    thread_id: uuid.UUID,
    graph=Depends(get_graph),
):
    """Get current pipeline state for a source.

    Reads from LangGraph checkpointer to get real-time node position,
    gate status, entity/chunk/draft counts, and error state.
    """
    config = {"configurable": {"thread_id": str(thread_id)}}
    state_snapshot = await graph.aget_state(config)

    if state_snapshot is None or state_snapshot.values is None:
        raise HTTPException(
            status_code=404,
            detail=f"No pipeline state found for thread {thread_id}",
        )

    state = state_snapshot.values

    return PipelineStatusResponse(
        thread_id=str(thread_id),
        source_id=state.get("source_id", ""),
        status=state.get("status", "unknown"),
        current_node=state.get("current_node"),
        error=state.get("error"),
        gates_enabled=state.get("gates_enabled", True),
        entity_count=len(state.get("entities", [])),
        chunk_count=len(state.get("chunks", [])),
        draft_count=len(state.get("drafts", [])),
        objects_written=state.get("objects_written", 0),
        persistence_errors=state.get("persistence_errors") or [],
        bundle_corrections=state.get("bundle_corrections") or [],
    )
