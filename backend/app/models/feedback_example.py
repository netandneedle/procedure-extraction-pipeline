"""FeedbackExample model: the analyst's corrections, kept as records.

WHY THIS EXISTS ALONGSIDE FeedbackPattern:
A pattern is what an LLM *wrote about* a correction — a generalisation, one
step removed from anything that happened. That step is where the damage has
been: a hand review of the first 63 patterns dropped roughly one in seven as
simply wrong, and a calibration found that 19 of the 25 rules which
fired against real output never once agreed with the analyst. The failures were
rules whose *idea* was wrong, not whose wording was.

An example cannot be wrong in that way. It is a record of something that
happened: the pipeline produced X, the analyst made it Y, and here is what they
said about it. It can only be IRRELEVANT to the source at hand — and choosing
relevant things is what the retrieval layer already does.

The trade is reach. A rule can cover a case that looks nothing like the one it
came from; an example mostly helps with cases that resemble it, so more of them
are needed before it pays off. That is the bet this table makes.

TWO THINGS IT CAN DO THAT A RULE-AS-CHECK CANNOT:
- Teach an OMISSION. A promoted technique or an analyst-added chunk is a
  demonstration of something that should have been produced and was not.
- Carry the analyst's own words. `rationale` is theirs, not a model's summary
  of theirs.

ONLY HUMAN CORRECTIONS LAND HERE. A gate that was disabled, or decided by the
AI reviewer unattended, produces no examples — see
`services/feedback_examples.py`. Every one of the original 63 patterns was
synthesized from a reviewer agreeing with itself, and that is the mistake this
table is built not to repeat.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class FeedbackExample(Base):
    """One analyst correction, kept verbatim enough to show to a model."""

    __tablename__ = "feedback_examples"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4,
    )

    # Not a foreign key: sources are deleted, examples outlive them. Same
    # decision as FeedbackPattern.source_id, for the same reason.
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True, index=True,
        comment="Source.id this correction came from (nullable; source may be deleted)",
    )
    # Denormalised on purpose. The rendered demonstration names the report it
    # came from ("a vendor ransomware writeup"), which is most of what
    # makes an example legible as a precedent rather than a floating assertion
    # — and it has to survive the source row being deleted.
    source_title: Mapped[str] = mapped_column(
        String(256), nullable=False, default="",
        comment="Title of the report this correction was made on",
    )

    # entities | chunks | techniques. Which gate's review produced it, and so
    # which nodes should be shown it. Deliberately the gate AREA rather than a
    # pattern category: an example is a record of a decision, and the decision
    # belongs to a gate.
    area: Mapped[str] = mapped_column(
        String(16), nullable=False, index=True,
        comment="Gate area: entities | chunks | techniques",
    )
    # remove | edit | drop | reject | promote | add — the analyst's verb.
    action: Mapped[str] = mapped_column(
        String(32), nullable=False, default="",
        comment="What the analyst did",
    )

    before: Mapped[dict] = mapped_column(
        JSON, nullable=False, default=dict,
        comment="What the pipeline produced",
    )
    after: Mapped[dict] = mapped_column(
        JSON, nullable=False, default=dict,
        comment="What the analyst made of it. Empty for a removal.",
    )
    # The analyst's own words, when they gave any. Not a model's paraphrase.
    rationale: Mapped[str] = mapped_column(
        Text, nullable=False, default="",
        comment="The analyst's stated reason, verbatim",
    )
    context_snippet: Mapped[str] = mapped_column(
        Text, nullable=False, default="",
        comment="Surrounding source text, where the delta carried one",
    )

    # Same retrieval machinery as FeedbackPattern: SecureBERT cosine plus
    # structured-key boosts. NULL embedding is legal and scores lexical-only,
    # so a row written while the model is unavailable is still usable.
    embedding: Mapped[list | None] = mapped_column(JSON, nullable=True)
    embedding_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    applies_to: Mapped[dict] = mapped_column(
        JSON, nullable=False, default=dict,
        comment="Structured retrieval keys: technique_ids, entity_types, tactics",
    )

    # Exact-match collapse for the same correction recurring across sources
    # (removing `vssadmin` as a tool, say). Semantic dedup is deliberately NOT
    # applied: two near-identical corrections from different reports are two
    # pieces of evidence that the error is general, and the count is the signal.
    dedup_key: Mapped[str] = mapped_column(
        String(160), nullable=False, index=True, default="",
    )
    occurrence_count: Mapped[int] = mapped_column(Integer, nullable=False, default=1)

    status: Mapped[str] = mapped_column(
        String(24), nullable=False, default="active", index=True,
        comment="active | dismissed",
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
        default=lambda: datetime.now(timezone.utc),
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(),
        default=lambda: datetime.now(timezone.utc),
    )
