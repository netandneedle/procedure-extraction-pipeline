"""Async persistence layer for the LLM cache.

Used by app.nodes.llm.llm_adapter.call_llm to look up and store tool_use
outputs keyed by sha256 of the input. Runs on the async session so cache
I/O doesn't block the FastAPI event loop while an LLM call is in flight.

Concurrent writes use INSERT ... ON CONFLICT DO NOTHING: two pipelines
extracting the same chunks at the same time both succeed without one
clobbering the other's row. Whichever wins the persistence race becomes
the canonical cached value; future reads see a consistent answer even if
the race was non-deterministic.

Failures are non-fatal: if the cache write errors out, the upstream LLM
response is still returned to the caller. We log and move on rather than
poisoning a successful extraction with a cache plumbing failure.
"""

from __future__ import annotations

import logging

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.models.base import async_session
from app.models.llm_cache import LLMCache

logger = logging.getLogger(__name__)


async def get_cached(cache_key: str) -> dict | None:
    """Return cached LLM response fields as a dict, or None on miss.

    Returned dict shape mirrors the columns of LLMCache:
    ``{tool_output, raw_text, model, input_tokens, output_tokens,
    stop_reason, attempts}``.
    """
    try:
        async with async_session() as session:
            result = await session.execute(
                select(LLMCache).where(LLMCache.cache_key == cache_key)
            )
            row = result.scalar_one_or_none()
            if row is None:
                return None
            return {
                "tool_output": row.tool_output,
                "raw_text": row.raw_text,
                "model": row.model,
                "input_tokens": row.input_tokens,
                "output_tokens": row.output_tokens,
                "stop_reason": row.stop_reason,
                "attempts": row.attempts,
            }
    except Exception as e:
        logger.warning(
            "llm_cache: read failed for key=%s err=%s",
            cache_key[:12], e,
        )
        return None


async def write_cache(cache_key: str, payload: dict) -> None:
    """Insert a cache row. ON CONFLICT DO NOTHING ignores duplicate keys.

    Payload should contain the LLMCache columns minus cache_key:
    tool_output, raw_text, model, input_tokens, output_tokens,
    stop_reason, attempts. Extra keys are ignored by SQLAlchemy.
    """
    stmt = (
        pg_insert(LLMCache)
        .values(cache_key=cache_key, **payload)
        .on_conflict_do_nothing(index_elements=["cache_key"])
    )
    try:
        async with async_session() as session:
            await session.execute(stmt)
            await session.commit()
    except Exception as e:
        # Cache write failure must not fail the upstream LLM call.
        logger.warning(
            "llm_cache: write failed for key=%s err=%s",
            cache_key[:12], e,
        )
