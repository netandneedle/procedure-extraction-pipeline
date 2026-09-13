"""Source queue service — business logic for Kanban board operations.

Thin layer between API routes and SQLAlchemy. Routes validate HTTP input,
this module handles queries and state transitions.
"""

from __future__ import annotations

import logging
import os
import re
import uuid
from datetime import datetime, timezone

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.source import Source
from app.services import feedback_patterns, figure_stash
from app.services.reviewer import store as reviewer_store

logger = logging.getLogger(__name__)

# Map file extensions to the SourceType values accepted by the pipeline parser.
EXTENSION_TO_SOURCE_TYPE: dict[str, str] = {
    ".pdf": "pdf",
    ".html": "html",
    ".htm": "html",
    ".md": "markdown",
    ".markdown": "markdown",
    ".txt": "free_text",
    ".docx": "docx",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
}


def detect_source_type(filename: str) -> str | None:
    """Return the SourceType value for a filename, or None if the extension is unsupported."""
    _, ext = os.path.splitext(filename.lower())
    return EXTENSION_TO_SOURCE_TYPE.get(ext)


# Strip anything that isn't safe in a filesystem path; keep ASCII letters,
# digits, dot, dash, underscore.
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_filename(original: str) -> str:
    base = os.path.basename(original) or "upload"
    cleaned = _SAFE_FILENAME_RE.sub("_", base).strip("._") or "upload"
    # Cap length to keep paths sane.
    return cleaned[-120:]


async def list_sources(
    db: AsyncSession,
    status: str | None = None,
    claimed_by: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[Source], int]:
    """List sources with optional filters.

    Returns (sources, total_count) for pagination.
    """
    query = select(Source).order_by(Source.created_at.desc())
    count_query = select(func.count(Source.id))

    if status:
        query = query.where(Source.status == status)
        count_query = count_query.where(Source.status == status)
    if claimed_by:
        query = query.where(Source.claimed_by == claimed_by)
        count_query = count_query.where(Source.claimed_by == claimed_by)

    query = query.limit(limit).offset(offset)

    result = await db.execute(query)
    sources = list(result.scalars().all())

    count_result = await db.execute(count_query)
    total = count_result.scalar_one()

    return sources, total


async def get_source(db: AsyncSession, source_id: uuid.UUID) -> Source | None:
    """Get a single source by ID."""
    result = await db.execute(select(Source).where(Source.id == source_id))
    return result.scalar_one_or_none()


async def create_source(
    db: AsyncSession,
    source_type: str,
    title: str,
    raw_content_path: str,
    channel: str = "manual",
    gates_enabled: dict[str, bool] | bool | None = None,
    gate_modes: dict[str, str] | str | None = None,
    source_reliability: int = 50,
    metadata: dict | None = None,
    sequentiality: str = "auto",
    extract_figures: bool = True,
) -> Source:
    """Create a new source in the queue with status=queued."""
    from app.graph.state import normalize_gate_modes, normalize_gates

    source = Source(
        source_type=source_type,
        title=title,
        raw_content_path=raw_content_path,
        channel=channel,
        gates_enabled=normalize_gates(gates_enabled),
        gate_modes=normalize_gate_modes(gate_modes),
        source_reliability=source_reliability,
        metadata_=metadata or {},
        status="queued",
        sequentiality=sequentiality,
        extract_figures=extract_figures,
    )
    db.add(source)
    await db.commit()
    await db.refresh(source)
    return source


async def claim_source(
    db: AsyncSession,
    source_id: uuid.UUID,
    analyst: str,
) -> Source | None:
    """Analyst claims a source. Returns None if not found.

    Raises ValueError if already claimed by someone else.
    """
    source = await get_source(db, source_id)
    if source is None:
        return None
    if source.claimed_by and source.claimed_by != analyst:
        raise ValueError(
            f"Source already claimed by {source.claimed_by}"
        )
    source.claimed_by = analyst
    source.claimed_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(source)
    return source


# The Source columns the runner keeps current from PipelineState.
RUN_COUNT_FIELDS = ("entity_count", "draft_count", "objects_written")


async def update_status(
    db: AsyncSession,
    source_id: uuid.UUID,
    status: str,
    error: str | None = None,
    persistence_errors: list[str] | None = None,
    bundle_corrections: list[dict] | None = None,
    run_counts: dict[str, int | None] | None = None,
) -> Source | None:
    """Update source status (called by pipeline nodes and Kanban drag-and-drop).

    persistence_errors and bundle_corrections are only written when
    explicitly passed (None means "don't touch"). distribute writes
    persistence_errors; validate_bundle writes bundle_corrections.
    Everything else updates status/error without clobbering either list.

    run_counts is {entity_count, draft_count, objects_written} from the
    runner; a key present with a value writes that column (None included,
    so a re-run can reset one), a key absent leaves it alone.

    Returns None if not found.

    No `db.refresh()` after commit. With `expire_on_commit=False` (see
    models/base.py), the in-memory `source` already reflects every field
    we set above. Skipping the refresh halves the DB roundtrips on the
    hot path — update_status fires on every pipeline node transition.
    """
    source = await get_source(db, source_id)
    if source is None:
        return None
    source.status = status
    if error is not None:
        source.error = error
    if persistence_errors is not None:
        source.persistence_errors = list(persistence_errors)
    if bundle_corrections is not None:
        source.bundle_corrections = list(bundle_corrections)
    for field in RUN_COUNT_FIELDS:
        if run_counts is not None and field in run_counts:
            setattr(source, field, run_counts[field])
    source.updated_at = datetime.now(timezone.utc)
    await db.commit()
    return source


async def clear_error(
    db: AsyncSession,
    source_id: uuid.UUID,
) -> Source | None:
    """Clear residual error + bundle_corrections on a source.

    Called by the pipeline controller right before starting a new run so that
    re-running a previously failed source doesn't carry forward stale audit
    state from the prior attempt. The validator's bundle_corrections gets
    cleared too — without this, a source that hard-failed last run shows
    a stale "N corrections" chip on the Kanban while the new run is still
    parsing/chunking. Leaves persistence_errors untouched (those are
    managed by the distribute node and only populate on successful runs).

    Returns None if not found.
    """
    source = await get_source(db, source_id)
    if source is None:
        return None
    dirty = False
    if source.error is not None:
        source.error = None
        dirty = True
    if source.bundle_corrections:
        source.bundle_corrections = []
        dirty = True
    # Last run's counts would sit on the card beside the new run's
    # "Parsing..." until extraction overwrites them.
    for field in RUN_COUNT_FIELDS:
        if getattr(source, field) is not None:
            setattr(source, field, None)
            dirty = True
    if dirty:
        source.updated_at = datetime.now(timezone.utc)
        await db.commit()
        await db.refresh(source)
    return source


async def save_upload(filename: str, data: bytes) -> tuple[str, str]:
    """Persist an uploaded file under settings.upload_dir.

    Returns (absolute_path, detected_source_type).
    Raises ValueError if the extension is unsupported.
    """
    source_type = detect_source_type(filename)
    if source_type is None:
        raise ValueError(
            f"Unsupported file type: {os.path.splitext(filename)[1] or '(none)'}. "
            f"Allowed: {', '.join(sorted(set(EXTENSION_TO_SOURCE_TYPE.keys())))}"
        )

    os.makedirs(settings.upload_dir, exist_ok=True)
    safe_name = _safe_filename(filename)
    path = os.path.join(settings.upload_dir, f"{uuid.uuid4().hex}_{safe_name}")
    # Ensure trailing slash on prefix so the path matches the API validator.
    with open(path, "wb") as fh:
        fh.write(data)
    return path, source_type


async def delete_source(db: AsyncSession, source_id: uuid.UUID) -> bool:
    """Delete a source row, its upload file, and its LangGraph checkpoint.

    Returns True if a row was deleted, False if no row matched.

    Cleans up four places in order:
      1. The uploaded file (only if under upload_dir, so we can't wipe
         user-supplied paths like /data/...).
      2. The source's stashed figure bitmaps, if any.
      3. The DB row.
      4. The LangGraph checkpoint for the source's thread_id, if set.
         Best-effort: failure logs a warning but doesn't fail the call,
         because by this point the row is already gone and we'd otherwise
         leave the caller observing a stale "still there" state.
    """
    source = await get_source(db, source_id)
    if source is None:
        return False

    thread_id = source.thread_id  # captured before delete; may be None

    # Cancel the run before removing anything it writes to. A detached task
    # that outlives its row dies on StaleDataError against a source that no
    # longer exists — noisy, and it reads like a crash. Lazy import for the
    # same reason as the checkpointer below: a service must not import a
    # route module at module scope.
    try:
        from app.api.routes.pipeline import cancel_pipeline_task
        if cancel_pipeline_task(source_id):
            logger.info("delete_source: cancelled in-flight run for %s", source_id)
    except Exception:  # noqa: BLE001 — never block a delete on bookkeeping
        logger.warning(
            "delete_source: could not cancel in-flight run for %s",
            source_id, exc_info=True,
        )

    raw_path = source.raw_content_path or ""
    upload_root = os.path.realpath(settings.upload_dir)
    try:
        real_path = os.path.realpath(raw_path)
        if real_path.startswith(upload_root + os.sep) and os.path.isfile(real_path):
            os.remove(real_path)
    except OSError:
        # File already gone or unreadable — proceed with DB delete.
        pass

    # Rendered figures stashed during parse. Keyed by source_id, so this is
    # the only place they get reclaimed.
    figure_stash.clear(str(source_id))

    # AI reviewer transcript + recommendations. Plain UUID association, not a
    # foreign key, so nothing cascades — this is the only place they are
    # reclaimed. Best-effort: a bookkeeping failure must not block the delete
    # the caller asked for.
    try:
        removed = await reviewer_store.delete_for_source(db, source_id)
        if removed:
            logger.info(
                "delete_source: removed %d reviewer rows for %s",
                removed, source_id,
            )
    except Exception:  # noqa: BLE001
        logger.warning(
            "delete_source: could not remove reviewer rows for %s",
            source_id, exc_info=True,
        )

    # Feedback surfacing ledger. Same shape as the reviewer rows above —
    # plain UUID association, no cascade — and the same consequence if it is
    # skipped: rows that outlive their source can never be joined back to it,
    # so they only distort counts taken over the table.
    try:
        removed = await feedback_patterns.delete_surfacings_for_source(
            db, source_id,
        )
        if removed:
            logger.info(
                "delete_source: removed %d feedback surfacing rows for %s",
                removed, source_id,
            )
    except Exception:  # noqa: BLE001
        logger.warning(
            "delete_source: could not remove surfacing rows for %s",
            source_id, exc_info=True,
        )

    await db.delete(source)
    await db.commit()

    # Flush the LangGraph checkpoint. Sources that never started (no
    # thread_id) have nothing to clean up. Done after the row commit so
    # a checkpointer outage doesn't prevent the user-visible delete.
    if thread_id is not None:
        try:
            from app.graph.checkpointer import get_checkpointer
            checkpointer = await get_checkpointer()
            await checkpointer.adelete_thread(str(thread_id))
        except Exception as exc:  # noqa: BLE001 — never block delete on this
            logger.warning(
                "delete_source: failed to flush checkpoint for thread %s: %s",
                thread_id, exc,
            )

    return True


async def set_thread_id(
    db: AsyncSession,
    source_id: uuid.UUID,
    thread_id: uuid.UUID,
) -> Source | None:
    """Set the LangGraph thread_id for a source (called when pipeline starts)."""
    source = await get_source(db, source_id)
    if source is None:
        return None
    source.thread_id = thread_id
    await db.commit()
    await db.refresh(source)
    return source
