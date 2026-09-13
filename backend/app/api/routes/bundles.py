"""Bundle Explorer API endpoints.

Serves completed STIX bundles and their source files for the
Explorer frontend view.

Endpoints:
    GET /api/bundles/           — list completed bundles (metadata only)
    GET /api/bundles/{id}       — get full bundle JSON
    GET /api/bundles/{id}/source — download original source file
    PATCH /api/bundles/{id}     — rename a bundle
    DELETE /api/bundles/{id}    — delete a bundle (cascades to source queue row)
"""

from __future__ import annotations

import logging
import re
from urllib.parse import quote
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.dependencies import get_db
from app.models.base import async_session
from app.schemas.api import BundleRenameRequest
from app.services import bundle_store
from app.services import queue as queue_service

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get(
    "/",
    summary="List completed bundles",
)
async def list_bundles(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """List all completed bundles with metadata (no file bytes or full bundle JSON)."""
    bundles, total = await bundle_store.list_bundles(db, limit=limit, offset=offset)
    return {
        "bundles": [
            {
                "id": str(b.id),
                "source_id": str(b.source_id) if b.source_id else None,
                "title": b.title,
                "object_count": b.object_count,
                "relationship_count": b.relationship_count,
                "procedure_count": b.procedure_count,
                "source_file_name": b.source_file_name,
                "source_file_type": b.source_file_type,
                "persistence_errors": b.persistence_errors or [],
                "bundle_corrections": b.bundle_corrections or [],
                "metadata": b.metadata_,
                "completed_at": b.completed_at.isoformat() if b.completed_at else None,
            }
            for b in bundles
        ],
        "total": total,
    }


@router.get(
    "/{bundle_id}",
    summary="Get full bundle",
)
async def get_bundle(
    bundle_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Get a completed bundle including the full STIX 2.1 JSON."""
    bundle = await bundle_store.get_bundle(db, bundle_id)
    if not bundle:
        raise HTTPException(status_code=404, detail="Bundle not found")
    return {
        "id": str(bundle.id),
        "source_id": str(bundle.source_id) if bundle.source_id else None,
        "title": bundle.title,
        "bundle_json": bundle.bundle_json,
        "object_count": bundle.object_count,
        "relationship_count": bundle.relationship_count,
        "procedure_count": bundle.procedure_count,
        "validation_results": bundle.validation_results,
        "persistence_errors": bundle.persistence_errors or [],
        "bundle_corrections": bundle.bundle_corrections or [],
        "source_file_name": bundle.source_file_name,
        "source_file_type": bundle.source_file_type,
        "metadata": bundle.metadata_,
        "completed_at": bundle.completed_at.isoformat() if bundle.completed_at else None,
    }


@router.get(
    "/{bundle_id}/source",
    summary="Download original source file",
)
async def get_source_file(
    bundle_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Return the original source file as a binary download."""
    file_data, file_name, file_type = await bundle_store.get_source_file(db, bundle_id)
    if file_data is None:
        raise HTTPException(status_code=404, detail="Source file not found for this bundle")
    # Build a safe Content-Disposition per RFC 6266 + RFC 5987:
    # - ASCII fallback with only alphanumerics, dash, underscore, dot
    # - filename* with percent-encoded UTF-8 for clients that honor it
    # Rejecting everything else avoids header-injection (CRLF, quotes,
    # semicolons, URL-encoded nasties) regardless of encoding.
    raw = file_name or "source"
    ascii_fallback = re.sub(r"[^A-Za-z0-9._-]", "_", raw) or "source"
    encoded = quote(raw, safe="")
    return Response(
        content=file_data,
        media_type=file_type or "application/octet-stream",
        headers={
            "Content-Disposition": (
                f'inline; filename="{ascii_fallback}"; '
                f"filename*=UTF-8''{encoded}"
            ),
        },
    )


@router.patch(
    "/{bundle_id}",
    summary="Rename a bundle",
)
async def rename_bundle(
    bundle_id: UUID,
    body: BundleRenameRequest,
    db: AsyncSession = Depends(get_db),
):
    """Update the display title on a completed bundle.

    Title is validated by BundleRenameRequest: stripped, 1-512 chars,
    no ASCII control characters. Returns the updated metadata (same
    shape as the list row, without bundle_json to keep responses cheap).
    """
    bundle = await bundle_store.rename_bundle(db, bundle_id, body.title)
    if bundle is None:
        raise HTTPException(status_code=404, detail="Bundle not found")
    return {
        "id": str(bundle.id),
        "source_id": str(bundle.source_id) if bundle.source_id else None,
        "title": bundle.title,
        "object_count": bundle.object_count,
        "relationship_count": bundle.relationship_count,
        "procedure_count": bundle.procedure_count,
        "source_file_name": bundle.source_file_name,
        "source_file_type": bundle.source_file_type,
        "persistence_errors": bundle.persistence_errors or [],
        "bundle_corrections": bundle.bundle_corrections or [],
        "metadata": bundle.metadata_,
        "completed_at": bundle.completed_at.isoformat() if bundle.completed_at else None,
    }


@router.delete(
    "/{bundle_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a bundle (cascades to source queue row)",
)
async def delete_bundle(
    bundle_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete a completed bundle and its linked source queue row.

    Order:
        1. Delete the bundle row (CompletedBundle) via bundle_store.
           This captures the linked source_id before the row is gone.
        2. If a source_id was returned, delete the source queue row
           (and any uploaded file under upload_dir) via queue_service.

    Steps 1 and 2 use separate sessions so a failure cascading to the
    source row can't roll back the bundle delete. Neo4j is intentionally
    left untouched — graph cleanup is a later decision.

    Returns 204 on success, 404 if no bundle matched the id.
    """
    deleted, source_id = await bundle_store.delete_bundle(db, bundle_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Bundle not found")

    # Cascade to the source queue row in a fresh session. Swallow
    # failures here — the bundle is already gone and surfacing a 500
    # would mislead the analyst into thinking the delete didn't work.
    if source_id is not None:
        try:
            async with async_session() as cascade_db:
                await queue_service.delete_source(cascade_db, source_id)
        except Exception as exc:  # noqa: BLE001
            # Best-effort cascade; bundle delete already committed. Log
            # so an orphan source row is traceable if it turns up later.
            logger.warning(
                "bundles.delete: cascade delete_source(%s) failed after "
                "bundle %s was removed: %s",
                source_id, bundle_id, exc,
            )

    return Response(status_code=status.HTTP_204_NO_CONTENT)
