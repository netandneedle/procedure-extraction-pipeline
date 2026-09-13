"""LLM cache table.

Stores tool_use outputs keyed by sha256 of the input. Eliminates re-billing
and ensures determinism on re-runs of the same pipeline source. Cache key
composition (computed in app.nodes.llm.llm_adapter._build_cache_key) includes
the system prompt — which contains the ATT&CK catalogue text — so any
catalogue version change naturally invalidates the cache via key mismatch.

Concurrent writes use INSERT ... ON CONFLICT DO NOTHING so two pipelines
hitting the same key both succeed without poisoning the cache.
"""

from __future__ import annotations

from sqlalchemy import Column, DateTime, Integer, JSON, String, Text
from sqlalchemy.sql import func

from app.models.base import Base


class LLMCache(Base):
    __tablename__ = "llm_cache"

    cache_key = Column(String(64), primary_key=True)  # sha256 hex digest
    tool_output = Column(JSON, nullable=False)
    raw_text = Column(Text, nullable=False, default="")
    model = Column(String(255), nullable=False)
    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    stop_reason = Column(String(64), nullable=False, default="")
    attempts = Column(Integer, nullable=False, default=1)
    created_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
