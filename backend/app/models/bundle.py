"""Completed bundle model — stores STIX 2.1 bundles after successful pipeline runs.

Each row represents a completed extraction: the final STIX bundle (as JSONB)
plus a copy of the original source file. The source file is stored as bytea
for now (portable across environments); swap to filesystem + DB reference
later via the bundle_store service abstraction.

This table is linked to the Source queue table via source_id FK, but is
designed to be self-contained: even if the source queue row is deleted,
the bundle and its source file remain accessible.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, LargeBinary, String, func
from sqlalchemy.dialects.postgresql import JSON, JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class CompletedBundle(Base):
    """A completed STIX 2.1 bundle with its original source file."""

    __tablename__ = "completed_bundles"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Link back to source queue (nullable: source queue row may be deleted)
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("sources.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )

    # Display metadata
    title: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        default="Untitled Bundle",
    )

    # The full STIX 2.1 bundle as JSONB (queryable server-side)
    bundle_json: Mapped[dict] = mapped_column(
        JSONB,
        nullable=False,
    )

    # Original source file stored as binary
    source_file_data: Mapped[bytes | None] = mapped_column(
        LargeBinary,
        nullable=True,
        comment="Original source file bytes (PDF, HTML, etc.)",
    )
    source_file_name: Mapped[str | None] = mapped_column(
        String(512),
        nullable=True,
        comment="Original filename for Content-Disposition header",
    )
    source_file_type: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        comment="MIME type of the source file",
    )

    # Bundle statistics (denormalized for fast listing)
    object_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    relationship_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )
    procedure_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0,
    )

    # Validation results from the serialize_stix node
    validation_results: Mapped[dict | None] = mapped_column(
        JSON,
        nullable=True,
    )

    # Non-fatal persistence failures from the distribute node.
    # Mirrors Source.persistence_errors so the Explorer view can show
    # the warning independently of the Kanban source card.
    persistence_errors: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        server_default="[]",
        comment="List of non-fatal persistence failure messages",
    )

    # Structured correction/warning records from the validate_bundle node
    # (Stage 6b). Mirrors Source.bundle_corrections so the Explorer view
    # can show what the validator changed even after the source row is
    # deleted. Same shape as Source.bundle_corrections.
    bundle_corrections: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        server_default="[]",
        comment="List of validator correction records ({rule, severity, message, ...})",
    )

    # Pipeline metadata (source type, campaign, TLP, etc.)
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
    )

    # Timestamps
    completed_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )

    def __repr__(self) -> str:
        return f"<CompletedBundle {self.id} [{self.title[:40]}] {self.object_count} objects>"
