"""FeedbackPattern model: cross-source analyst-feedback knowledge base.

Written by the synthesize_feedback node at the end of every successful
pipeline run. Each row captures a structured pattern the synthesizer
extracted from the LLM-output ↔ analyst-decision deltas at the review gates.

Read by LLM nodes at prompt-build time: each node fetches the patterns
whose category matches its concern (e.g., extract_entities fetches
`defender_ioc` and `brand_as_malware`; chunk_behaviors fetches
`over_chunked`, `under_chunked`, `missing_tactic`), ranked by relevance to
the current source. The patterns are prepended to the system prompt as
"RECENT ANALYST FEEDBACK" so future runs benefit from past corrections
without manual prompt edits.

DESIGN: this table is intentionally separate from PipelineState because:
- Patterns are CROSS-SOURCE: querying "all defender_ioc patterns from
  the last 30 days" is the whole point. Per-source state can't do that.
- Patterns OUTLIVE their source: a source might be deleted, but its
  feedback patterns should persist (and contribute to future runs).
- Pattern lifecycle (pending → active → dismissed → promoted) is
  independent of source lifecycle.
"""

import uuid
from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base


class FeedbackPattern(Base):
    """A structured pattern extracted from analyst gate decisions."""

    __tablename__ = "feedback_patterns"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    # Which source generated this pattern. NOT a foreign key — sources can
    # be deleted while patterns persist. Stored as plain UUID for queryability.
    source_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        nullable=True,
        index=True,
        comment="Source.id that generated this pattern (nullable; source may be deleted)",
    )

    # The pattern category. See `FeedbackCategory` in
    # nodes/llm/tool_models.py for the canonical taxonomy.
    # Stored as String (not enum) so the taxonomy can grow without
    # schema migrations.
    category: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="Pattern category: defender_ioc | over_chunked | missing_tactic | ...",
    )

    # Human-readable, searchable description.
    pattern: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Pattern text — short, searchable, matched on for dedup",
    )

    # Evidence: structured data the synthesizer captured to justify the
    # pattern. Free-form JSON; consumers know what to look for per-category.
    # Examples:
    #   defender_ioc: {entity_id, value, gate, rationale, llm_confidence,
    #                  surrounding_figure_caption}
    #   over_chunked: {chunk_ids, merged_into, reason}
    #   missing_tactic: {tactic, evidence_in_source, expected_chunk_count}
    evidence: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        comment="Per-category structured evidence supporting the pattern",
    )

    # --- Semantic retrieval (relevance-first flywheel) -------------------
    # SecureBERT 2.0 embedding of the pattern text (+ a compact applies_to
    # render), stored as a JSON float array. Computed at persist time via
    # app.services.pattern_embedding using the SAME model as
    # technique_retriever. Nullable (no default) so the backfill script can
    # detect un-embedded legacy rows via `WHERE embedding IS NULL`. JSON over
    # bytea keeps it inspectable and matches the `evidence` precedent; bytea
    # is the optimization path if the corpus grows past ~10k rows.
    embedding: Mapped[list | None] = mapped_column(
        JSON,
        nullable=True,
        comment="SecureBERT embedding of the pattern (JSON float array); NULL until backfilled",
    )
    # Which model produced `embedding`, so a future model swap can invalidate
    # and recompute (mirrors EmbeddingRetriever._model_name discipline).
    embedding_model: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        comment="Model id that produced `embedding` (e.g. cisco-ai/SecureBERT2.0-biencoder)",
    )
    # Structured retrieval keys the synthesizer emits, used for exact-match
    # boosts at retrieval time and category-area mapping at scoring time.
    # Shape: {technique_ids: [...], entity_types: [...], tactics: [...],
    #         source_genre: str}
    applies_to: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        server_default="{}",
        comment="Structured retrieval keys: technique_ids / entity_types / tactics / source_genre",
    )
    # Free-form concept tags for search + display.
    concepts: Mapped[list] = mapped_column(
        JSON,
        nullable=False,
        default=list,
        server_default="[]",
        comment="Short concept tags for search/display",
    )
    # Concrete things to block when this pattern is promoted to
    # 'promoted_to_denylist'. Shape: {"values": [...], "technique_ids": [...]}.
    # `values` are literal entity values (case-insensitive exact match —
    # extract_entities tags matches, gate_0 auto-removes them). `technique_ids`
    # are T-IDs dropped from technique picks. Empty {} for non-denylist
    # patterns; a denylist promotion with empty terms degrades to advisory
    # (nothing is enforced deterministically). The analyst confirms/edits these
    # at promote time (pre-filled from evidence/applies_to).
    denylist_terms: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
        server_default="{}",
        comment="Deterministic guardrail targets: {values, technique_ids}",
    )

    # Lifecycle state.
    # - 'active': fresh; consumed by LLM nodes at prompt-build time
    # - 'pending': reserved for analyst-curated patterns; nothing writes it yet
    # - 'dismissed': analyst marked as not useful; not consumed
    # - 'promoted_to_denylist': analyst promoted to a deterministic guardrail
    # - 'promoted_to_prompt': analyst pinned it as a permanent prompt rule
    # - 'archived': low salience and stale (scripts/recompute_salience.py,
    #   90 days by default); not consumed
    status: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="active",
        server_default="active",
        index=True,
        comment="Lifecycle: active | pending | dismissed | promoted_* | archived",
    )

    # Bumped each time the same pattern recurs across runs. Used by LLM
    # nodes to prioritize: high-occurrence patterns are more reliable.
    occurrence_count: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1,
        server_default="1",
        comment="Times this pattern has been seen across all runs",
    )

    # --- Closed loop (does the pattern actually help?) ------------------
    # Set by synthesize_feedback's re-correction detection. A MISS
    # means the pattern was surfaced for a run yet the analyst STILL made the
    # correction it warns about (relevant but ineffective as phrased); a HIT
    # means it was surfaced with no matching correction (the guardrail held).
    hit_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0",
        comment="Runs where this pattern was surfaced and no matching correction recurred",
    )
    miss_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0, server_default="0",
        comment="Runs where this pattern was surfaced but the correction recurred anyway",
    )
    # Bayesian-smoothed hit-rate x recency decay x occurrence weight. Feeds
    # the hybrid retrieval ranking and the decay/archive path. NULL until
    # first scored (treated as 0 contribution by the retriever).
    salience: Mapped[float | None] = mapped_column(
        Float, nullable=True, index=True,
        comment="Composite usefulness score; drives ranking + decay",
    )
    last_scored_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True,
        comment="When hit/miss/salience were last recomputed",
    )

    # Timestamps
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    last_seen_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        comment="Updated each time occurrence_count is bumped",
    )

    # Set when status transitions to a 'promoted_*' value.
    promoted_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )
    promoted_by: Mapped[str | None] = mapped_column(
        String(128),
        nullable=True,
        default=None,
        comment="Analyst who promoted (email or username)",
    )
    promoted_action: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        default=None,
        comment="What the promotion did: e.g., 'denylist:cert.example' or 'prompt:entity_extraction'",
    )

    def __repr__(self) -> str:
        return (
            f"<FeedbackPattern {self.id} [{self.category}] "
            f"x{self.occurrence_count} {self.status}: {self.pattern[:60]}>"
        )
