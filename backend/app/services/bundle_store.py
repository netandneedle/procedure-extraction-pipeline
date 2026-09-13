"""Bundle persistence service — saves and retrieves completed STIX bundles.

Storage abstraction: currently stores bundles and source files directly in
PostgreSQL (JSONB + bytea). To migrate to filesystem + DB reference later,
swap the internal implementation of save_bundle() and get_source_file()
without changing the API contract.

Usage from the distribute node:
    from app.services.bundle_store import save_bundle
    await save_bundle(state)

Usage from API routes:
    from app.services.bundle_store import list_bundles, get_bundle, get_source_file
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
from pathlib import Path
from uuid import UUID

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import defer

from app.config import settings
from app.models.bundle import CompletedBundle

logger = logging.getLogger(__name__)

# MIME type mapping for source types
_SOURCE_TYPE_MIME: dict[str, str] = {
    "pdf": "application/pdf",
    "html": "text/html",
    "markdown": "text/markdown",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "image": "image/png",
    "txt": "text/plain",
}


async def save_bundle(
    db: AsyncSession,
    *,
    source_id: UUID | str | None,
    title: str,
    bundle: dict,
    validation_results: dict | None = None,
    source_file_path: str | None = None,
    source_type: str | None = None,
    metadata: dict | None = None,
    persistence_errors: list[str] | None = None,
    bundle_corrections: list[dict] | None = None,
) -> CompletedBundle:
    """Persist a completed STIX bundle and its source file to the database.

    Args:
        db: Async SQLAlchemy session.
        source_id: FK to sources table (may be None if source was deleted).
        title: Display title for the bundle.
        bundle: The full STIX 2.1 bundle dict.
        validation_results: Results from the 4-stage validation.
        source_file_path: Path to the original source file on disk.
        source_type: Source type string (pdf, html, markdown, etc.).
        metadata: Flexible metadata dict (campaign, TLP, author, etc.).

    Returns:
        The created CompletedBundle instance.
    """
    # Count objects by type
    objects = bundle.get("objects", [])
    object_count = len(objects)
    relationship_count = sum(1 for o in objects if o.get("type") == "relationship")
    procedure_count = sum(1 for o in objects if o.get("type") == "x-procedure")

    # Read source file if path is provided
    source_file_data = None
    source_file_name = None
    source_file_type = None

    if source_file_path:
        try:
            file_path = Path(source_file_path).resolve()
            upload_dir = Path(settings.upload_dir).resolve()
            # Guard against path traversal and prefix-collision (e.g.
            # /var/uploads vs /var/uploads_evil). is_relative_to handles
            # the boundary correctly after .resolve() strips symlinks.
            try:
                is_under_upload = file_path.is_relative_to(upload_dir)
            except AttributeError:
                # Python <3.9 fallback
                is_under_upload = upload_dir in file_path.parents or file_path == upload_dir
            if not is_under_upload:
                logger.warning(
                    "bundle_store: source_file_path %s outside upload directory, skipping",
                    file_path,
                )
            elif file_path.exists() and file_path.is_file():
                # to_thread so a large upload (10s of MB) doesn't block
                # the event loop during the async distribute node.
                source_file_data = await asyncio.to_thread(file_path.read_bytes)
                source_file_name = file_path.name
                # Strip UUID prefix if present (e.g., "a1b2c3d4_report.pdf" -> "report.pdf")
                if "_" in source_file_name:
                    parts = source_file_name.split("_", 1)
                    if len(parts[0]) == 32:  # UUID hex length
                        source_file_name = parts[1]
                # Determine MIME type
                source_file_type = _SOURCE_TYPE_MIME.get(source_type or "")
                if not source_file_type:
                    source_file_type = mimetypes.guess_type(source_file_name)[0] or "application/octet-stream"
                logger.info(
                    "bundle_store: read source file %s (%d bytes, %s)",
                    source_file_name, len(source_file_data), source_file_type,
                )
            else:
                logger.warning("bundle_store: source file not found at %s", file_path)
        except FileNotFoundError:
            logger.warning("bundle_store: source file missing: %s", source_file_path)
        except PermissionError as e:
            # Surface this so the caller can flag it instead of silently
            # dropping the source attachment.
            logger.error("bundle_store: permission denied reading source file: %s", e)
            raise
        except OSError as e:
            logger.error("bundle_store: I/O error reading source file: %s", e)
            raise

    # Convert source_id to UUID with validation
    if isinstance(source_id, str):
        try:
            source_id = UUID(source_id)
        except ValueError:
            logger.warning("bundle_store: invalid source_id UUID string, setting to None")
            source_id = None

    completed = CompletedBundle(
        source_id=source_id,
        title=title,
        bundle_json=bundle,
        source_file_data=source_file_data,
        source_file_name=source_file_name,
        source_file_type=source_file_type,
        object_count=object_count,
        relationship_count=relationship_count,
        procedure_count=procedure_count,
        validation_results=validation_results,
        persistence_errors=list(persistence_errors or []),
        bundle_corrections=list(bundle_corrections or []),
        metadata_=metadata or {},
    )
    db.add(completed)
    await db.commit()
    await db.refresh(completed)

    logger.info(
        "bundle_store: saved bundle %s (%d objects, %d procedures)",
        completed.id, object_count, procedure_count,
    )
    return completed


async def list_bundles(
    db: AsyncSession,
    *,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[CompletedBundle], int]:
    """List completed bundles ordered by completion date (newest first).

    Returns source file metadata but NOT the file bytes (deferred load).

    Returns:
        (bundles, total_count)
    """
    # Count
    count_q = select(func.count()).select_from(CompletedBundle)
    total = (await db.execute(count_q)).scalar() or 0

    # List — defer heavy columns (bundle JSON + file bytes) to avoid loading MBs
    q = (
        select(CompletedBundle)
        .options(defer(CompletedBundle.bundle_json))
        .options(defer(CompletedBundle.source_file_data))
        .order_by(CompletedBundle.completed_at.desc())
        .offset(offset)
        .limit(limit)
    )
    result = await db.execute(q)
    bundles = list(result.scalars().all())

    return bundles, total


async def get_bundle(db: AsyncSession, bundle_id: UUID) -> CompletedBundle | None:
    """Get a single bundle by ID, including the full JSONB bundle."""
    q = select(CompletedBundle).where(CompletedBundle.id == bundle_id)
    result = await db.execute(q)
    return result.scalar_one_or_none()


async def get_source_file(
    db: AsyncSession, bundle_id: UUID
) -> tuple[bytes | None, str | None, str | None]:
    """Get the source file for a bundle.

    Returns:
        (file_bytes, filename, mime_type) — all None if no file stored.
    """
    q = select(
        CompletedBundle.source_file_data,
        CompletedBundle.source_file_name,
        CompletedBundle.source_file_type,
    ).where(CompletedBundle.id == bundle_id)
    result = await db.execute(q)
    row = result.one_or_none()
    if not row:
        return None, None, None
    return row[0], row[1], row[2]


async def rename_bundle(
    db: AsyncSession,
    bundle_id: UUID,
    new_title: str,
) -> CompletedBundle | None:
    """Update the display title on a completed bundle.

    Title normalization and length validation happen in the Pydantic
    schema (BundleRenameRequest). This function trusts the caller to
    have already stripped and capped the value.

    Returns the refreshed row, or None if no bundle matched the id.
    """
    bundle = await get_bundle(db, bundle_id)
    if bundle is None:
        return None
    bundle.title = new_title
    await db.commit()
    await db.refresh(bundle)
    logger.info("bundle_store: renamed bundle %s -> %r", bundle_id, new_title)
    return bundle


async def delete_bundle(
    db: AsyncSession,
    bundle_id: UUID,
) -> tuple[bool, UUID | None]:
    """Delete a completed bundle row.

    Neo4j is intentionally left untouched (writes are disabled in dev
    and a future cleanup strategy hasn't been designed yet). The caller
    (API route) is responsible for cascading to the source queue row
    if that behavior is desired.

    Returns:
        (deleted, source_id) — deleted is True when a row was removed;
        source_id is the FK so the caller can decide whether to cascade
        to the sources table.
    """
    bundle = await get_bundle(db, bundle_id)
    if bundle is None:
        return False, None
    source_id = bundle.source_id
    await db.delete(bundle)
    await db.commit()
    logger.info("bundle_store: deleted bundle %s (source_id=%s)", bundle_id, source_id)
    return True, source_id
