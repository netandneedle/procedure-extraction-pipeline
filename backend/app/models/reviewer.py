"""ReviewerRecommendation model: what the AI reviewer proposed, and what happened.

The AI gate reviewer reads a source once and walks every gate carrying its own
reasoning forward. This table is both its memory and the instrument that
measures it.

DESIGN — why Postgres and not PipelineState:

  1. Measurement. Deciding whether a gate is safe to run unattended means
     asking "across every source, how often did the analyst override the
     reviewer at this gate?" That is a query. Parsing it out of LangGraph
     checkpoints is not.
  2. Checkpoint size. LangGraph rewrites the WHOLE state at every node
     transition — roughly fifteen times per run. A transcript riding in state
     would be rewritten every time for no benefit.
  3. The assist UI needs recommendations as a resource of their own, fetchable
     independently of the gate payload and independent of whether the graph is
     currently paused.

`source_id` is a plain UUID, not a foreign key — the same "these can be
deleted independently" philosophy as FeedbackPatternSurfacing. Cleanup is
explicit in queue.delete_source.

`outcome` is the point of the whole table: what the analyst ACTUALLY submitted,
written back at gate-submit time. Agreement rate falls out of a query rather
than a bespoke harness.
"""

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base

# gate_key values. The four GATE_KEYS plus a synthetic one for the reviewer's
# opening read of the report, which is turn one of the transcript and has no
# gate of its own.
INITIAL_READ = "initial_read"

# status values.
STATUS_OK = "ok"            # recommendations produced
STATUS_FAILED = "failed"    # reviewer errored; gate fell through to a human


class ReviewerRecommendation(Base):
    """One reviewer turn: its recommendations for one gate on one pass."""

    __tablename__ = "reviewer_recommendations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True,
        comment="Source.id this review belongs to (not an FK). Scopes the transcript.",
    )
    gate_key: Mapped[str] = mapped_column(
        String(32), nullable=False,
        comment="entities | chunks | procedures | bundle | initial_read",
    )
    # Gates loop: a Gate 1 reject routes back to extract_techniques and returns
    # to the same gate with new drafts. Each visit is its own row so the
    # transcript shows the reviewer what changed between attempts, and so an
    # override on pass 1 isn't confused with agreement on pass 2.
    pass_number: Mapped[int] = mapped_column(
        Integer, nullable=False, default=1,
        comment="1-based visit count for this gate on this source",
    )
    model: Mapped[str] = mapped_column(
        String(64), nullable=False, default="",
        comment="Model ID that produced this turn",
    )
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default=STATUS_OK,
        comment="ok | failed. A failed row records that the gate fell through to a human.",
    )
    # The recommendation set: submit-schema-shaped items, each carrying
    # confidence / rationale / evidence_quote plus any grounding verdict.
    payload: Mapped[dict[str, Any]] = mapped_column(
        JSONB, nullable=False, default=dict,
        comment="Recommendations, submit-schema shaped + confidence/rationale/quote",
    )
    # The reviewer's own prose for this turn, replayed into the transcript at
    # the next gate. Only the AGENT's turns are stored; environment turns
    # (gate payloads) are regenerated from live state, so the reviewer always
    # sees what actually happened rather than what it proposed.
    agent_notes: Mapped[str] = mapped_column(
        Text, nullable=False, default="",
    )
    error: Mapped[str | None] = mapped_column(
        Text, nullable=True,
        comment="Populated when status='failed'",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    # What the analyst actually submitted at this gate, plus the per-item
    # agree/override diff. Written by the gate submit route. NULL means the
    # gate has not been submitted yet (or ran in auto mode with no human).
    outcome: Mapped[dict[str, Any] | None] = mapped_column(
        JSONB, nullable=True,
        comment="Analyst's actual submission + per-item agreement diff",
    )
    outcome_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    def __repr__(self) -> str:
        return (
            f"<ReviewerRecommendation {self.gate_key} pass={self.pass_number} "
            f"source={self.source_id} status={self.status}>"
        )
