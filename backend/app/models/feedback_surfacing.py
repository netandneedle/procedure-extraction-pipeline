"""FeedbackPatternSurfacing model: the closed-loop ledger.

Records which feedback patterns were SURFACED into which node's prompt for
which source, then — at synthesize_feedback time — whether that surfacing was
a HIT (the guardrail held; no matching correction recurred) or a MISS (the
analyst still made the correction the pattern warns about). Those outcomes
drive each pattern's hit_count / miss_count / salience.

DESIGN: a normalized table rather than a JSON blob on the source row because
hit/miss is cross-run analytics — it joins "what we surfaced for source X" to
"what the analyst corrected in source X", must outlive the per-thread LangGraph
checkpoint, and must be aggregatable for the management UI ("surfaced 40x, hit
32"). pattern_id / source_id are plain UUIDs (NOT foreign keys) — same
"patterns and sources can be deleted independently" philosophy as
FeedbackPattern.source_id.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, String, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class FeedbackPatternSurfacing(Base):
    """One record of a pattern being injected into a node's prompt."""

    __tablename__ = "feedback_pattern_surfacings"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )
    pattern_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True,
        comment="FeedbackPattern.id that was surfaced (not an FK)",
    )
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True,
        comment="Source.id this surfacing happened for (not an FK)",
    )
    node: Mapped[str] = mapped_column(
        String(32), nullable=False,
        comment="Pipeline node that surfaced it: extract_entities | chunk_behaviors | extract_techniques | draft_procedures",
    )
    surfaced_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
    )
    # Scored at synthesize_feedback time: 'hit' | 'miss' | None (pending).
    outcome: Mapped[str | None] = mapped_column(
        String(16), nullable=True,
        comment="hit (guardrail held) | miss (correction recurred) | None (unscored)",
    )
    scored_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
    )

    def __repr__(self) -> str:
        return (
            f"<FeedbackPatternSurfacing {self.id} pattern={self.pattern_id} "
            f"node={self.node} outcome={self.outcome}>"
        )
