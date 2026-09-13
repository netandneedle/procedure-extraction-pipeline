"""Source queue endpoints for Kanban board operations.

These endpoints power the React Kanban board. A source's `status` walks the
PipelineStatus enum (app.graph.state) from `queued` through the processing
stages and gate pauses to `completed` or `failed`.

The pipeline updates status as it progresses. The frontend reads this table
to render Kanban columns and source cards.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.config import settings
from app.schemas.api import (
    SourceClaim,
    SourceCreate,
    SourceListResponse,
    SourceResponse,
    SourceStatusUpdate,
)
from app.services import queue as queue_service

router = APIRouter()


@router.get(
    "/",
    response_model=SourceListResponse,
    summary="List sources in the queue",
)
async def list_sources(
    status: str | None = Query(None, description="Filter by PipelineStatus value"),
    claimed_by: str | None = Query(None, description="Filter by analyst"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List all sources, filterable by status and analyst. Paginated."""
    sources, total = await queue_service.list_sources(
        db, status=status, claimed_by=claimed_by, limit=limit, offset=offset
    )
    return SourceListResponse(
        sources=[SourceResponse.model_validate(s) for s in sources],
        total=total,
    )


@router.post(
    "/",
    response_model=SourceResponse,
    status_code=201,
    summary="Ingest a new source",
)
async def create_source(
    body: SourceCreate,
    db: AsyncSession = Depends(get_db),
):
    """Add a new source to the queue. Status starts as 'queued'."""
    source = await queue_service.create_source(
        db,
        source_type=body.source_type,
        title=body.title,
        raw_content_path=body.raw_content_path,
        channel=body.channel,
        gates_enabled=body.gates_enabled,
        gate_modes=body.gate_modes,
        source_reliability=body.source_reliability,
        metadata=body.metadata,
        sequentiality=body.sequentiality,
        extract_figures=body.extract_figures,
    )
    return SourceResponse.model_validate(source)


@router.post(
    "/upload",
    summary="Upload a file to be used as a source's raw_content_path",
)
async def upload_source_file(file: UploadFile = File(...)):
    """Accept a multipart file, save it under the uploads dir, and return its path + detected type.

    Enforces settings.upload_max_bytes (50 MB by default) and rejects unsupported extensions.
    Response shape: {"path": "/tmp/pipeline/uploads/<uuid>_<name>", "source_type": "pdf", "filename": "<name>"}
    """
    # Read in a bounded way so we don't OOM on large uploads.
    max_bytes = settings.upload_max_bytes
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"File exceeds {max_bytes // (1024 * 1024)} MB limit.",
            )
        chunks.append(chunk)

    data = b"".join(chunks)
    try:
        path, source_type = await queue_service.save_upload(file.filename or "upload", data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"path": path, "source_type": source_type, "filename": file.filename}


@router.delete(
    "/{source_id}",
    status_code=204,
    summary="Delete a source",
)
async def delete_source(
    source_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Remove a source from the queue. Also deletes its uploaded file if we own it."""
    deleted = await queue_service.delete_source(db, source_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Source not found")
    return None


@router.get(
    "/{source_id}",
    response_model=SourceResponse,
    summary="Get source detail",
)
async def get_source_detail(
    source_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get full source detail including pipeline state."""
    source = await queue_service.get_source(db, source_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return SourceResponse.model_validate(source)


@router.patch(
    "/{source_id}/claim",
    response_model=SourceResponse,
    summary="Claim a source",
)
async def claim_source(
    source_id: uuid.UUID,
    body: SourceClaim,
    db: AsyncSession = Depends(get_db),
):
    """Analyst claims a source for extraction."""
    try:
        source = await queue_service.claim_source(db, source_id, body.analyst)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return SourceResponse.model_validate(source)


@router.patch(
    "/{source_id}/status",
    response_model=SourceResponse,
    summary="Update source status",
)
async def update_source_status(
    source_id: uuid.UUID,
    body: SourceStatusUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update source status (used by pipeline nodes and Kanban drag-and-drop)."""
    source = await queue_service.update_status(db, source_id, body.status)
    if source is None:
        raise HTTPException(status_code=404, detail="Source not found")
    return SourceResponse.model_validate(source)
