"""Source queue model — the Kanban board's backing table.

Each row represents a threat intelligence source that has been ingested
into the queue. The pipeline reads from this table to find work, and
writes back status updates as it progresses through nodes.

This table is SEPARATE from LangGraph's checkpointer tables. The
checkpointer stores pipeline state (entities, chunks, drafts, etc.).
This table stores queue metadata (who claimed it, when, kanban column).
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.graph.state import DEFAULT_GATE_MODES, DEFAULT_GATES
from app.models.base import Base


class Source(Base):
    """A threat intelligence source in the extraction queue."""

    __tablename__ = "sources"

    # Primary key — also used as LangGraph thread_id for 1:1 mapping.
    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Source properties
    source_type: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        comment="SourceType enum value (pdf, html, markdown, etc.)",
    )
    title: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
        default="Untitled Source",
        comment="Display name for Kanban card",
    )
    raw_content_path: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="File path or blob reference to source content",
    )

    # Pipeline tracking
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="queued",
        index=True,
        comment="PipelineStatus enum value — drives Kanban column",
    )
    channel: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default="manual",
        comment="Channel enum value (manual, automated)",
    )
    gates_enabled: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=lambda: dict(DEFAULT_GATES),
        server_default='{"entities": true, "chunks": true, "procedures": true, "bundle": true}',
        comment="Per-gate enable/disable: {entities, chunks, procedures, bundle} -> bool",
    )

    # Per-gate review MODE, meaningful only for gates that are enabled above.
    # "review" (a human reviews it) | "assist" (the AI reviewer recommends,
    # a human decides) | "auto" (the AI reviewer decides unattended).
    #
    # Deliberately separate from gates_enabled rather than a widening of it:
    # every mode string is truthy, so folding modes into that dict would let
    # any direct `gates_enabled[key]` read treat a disabled gate as enabled.
    # See app.graph.state.normalize_gate_modes for the full rationale.
    gate_modes: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=lambda: dict(DEFAULT_GATE_MODES),
        server_default=(
            '{"entities": "review", "chunks": "review", '
            '"procedures": "review", "bundle": "review"}'
        ),
        comment='Per-gate review mode: {gate_key} -> "review" | "assist" | "auto"',
    )

    # Analyst's pre-flight answer to "does this source include sequential
    # information?" Values: "yes" | "no" | "auto". "auto" defers to
    # entity_extraction's LLM classification. The resolved boolean is written
    # to PipelineState["is_sequential"] and gates the chunker's orphan-link
    # backstop and the serializer's PRECEDES SRO emission.
    sequentiality: Mapped[str] = mapped_column(
        String(8),
        nullable=False,
        default="auto",
        server_default="auto",
        comment='"yes" | "no" | "auto" — does the source describe sequential procedures',
    )

    # When True, the figure_extraction node runs a vision-LLM pass over each
    # figure in the source and inlines the extracted text into parsed_text
    # so the chunker / extractors see figure content as part of the body.
    # Default True: a figure-coverage measurement found 221/221 figures
    # across 14 vendor reports were dropped by Docling's text export. The
    # analyst can disable it for sources known to be all-prose.
    extract_figures: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
        comment="If True, run vision-LLM figure extraction during ingestion",
    )

    # Analyst assignment
    claimed_by: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        default=None,
        comment="Analyst email/username who claimed this source",
    )
    claimed_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )

    # Rich metadata — flexible JSON for author, campaign, TLP, etc.
    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSON,
        nullable=False,
        default=dict,
        comment="Flexible metadata: author, publication_date, campaign, tlp, source_url, etc.",
    )

    # Source reliability score (0-100), set at ingestion.
    source_reliability: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=50,
        comment="Source reliability 0-100, set at ingestion",
    )

    # LangGraph thread_id — set when pipeline starts. Same as id by default,
    # but kept as explicit column for clarity and potential future divergence.
    thread_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        default=None,
        index=True,
        comment="LangGraph thread_id for checkpointer lookup",
    )

    # Error tracking
    error: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None,
        comment="Error message if pipeline failed",
    )

    # Non-fatal persistence failures surfaced from the distribute node
    # (e.g. bundle_store row failed, source file unreadable). The Neo4j
    # write still succeeded — this is a durable signal to the analyst
    # that the bundle's sidecar data didn't land cleanly.
    persistence_errors: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        server_default="[]",
        comment="List of non-fatal persistence failure messages",
    )

    # Structured corrections + warnings emitted by the validate_bundle
    # node (Stage 6b). Each entry: {rule, severity, ref_id?, recovered_from?,
    # holder_id?, before?, after?, message}. severity ∈ {auto_fix, repaired,
    # warn, hard_fail}. Surfaces on the SourceCard so analysts know what the
    # validator changed (or what failed when the source bounced back to gate_2).
    bundle_corrections: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        server_default="[]",
        comment="List of validator correction records ({rule, severity, message, ...})",
    )
    # Run counts the card shows: entities after extraction, drafts after
    # drafting, STIX objects written after distribute. Persisted here because
    # the card reads them off SourceResponse — before these columns existed
    # they lived only on the WebSocket event, so the "N objects" chip
    # appeared for one poll cycle and was gone on any page load. NULL until
    # the run reaches the stage that produces the number.
    entity_count: Mapped[int | None] = mapped_column(
        Integer, nullable=True, default=None,
        comment="Entities extracted in the latest run (NULL before extraction)",
    )
    draft_count: Mapped[int | None] = mapped_column(
        Integer, nullable=True, default=None,
        comment="Procedure drafts in the latest run (NULL before drafting)",
    )
    objects_written: Mapped[int | None] = mapped_column(
        Integer, nullable=True, default=None,
        comment="STIX objects written to Neo4j in the latest run (NULL before distribute)",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    def __repr__(self) -> str:
        return f"<Source {self.id} [{self.status}] {self.title[:40]}>"
