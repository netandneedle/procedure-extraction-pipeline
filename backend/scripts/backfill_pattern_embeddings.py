"""One-shot: backfill SecureBERT embeddings onto existing feedback_patterns.

The relevance-first flywheel adds an `embedding` column to
feedback_patterns. The startup auto-migrate adds the column (NULL on existing
rows); this script fills it so legacy patterns participate in semantic ranking
instead of falling back to lexical-only scoring.

Idempotent: only touches rows where `embedding IS NULL`. Re-run safe.

Usage (from backend/ with the venv active):
    python -m scripts.backfill_pattern_embeddings
    python -m scripts.backfill_pattern_embeddings --batch 64
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import select

from app.config import settings
from app.models.base import async_session
from app.models.feedback_pattern import FeedbackPattern
from app.services import pattern_embedding
from app.services.feedback_patterns import _embedding_text


async def main(batch: int) -> None:
    async with async_session() as db:
        rows = list(
            (
                await db.execute(
                    select(FeedbackPattern).where(FeedbackPattern.embedding.is_(None))
                )
            ).scalars().all()
        )
        if not rows:
            print("backfill: no rows with NULL embedding — nothing to do.")
            return

        print(f"backfill: {len(rows)} pattern(s) need embeddings; encoding...")
        texts = [_embedding_text(r.pattern, r.applies_to) for r in rows]

        filled = 0
        for start in range(0, len(rows), batch):
            chunk_rows = rows[start:start + batch]
            chunk_texts = texts[start:start + batch]
            vecs = pattern_embedding.embed_texts(chunk_texts)
            if vecs is None:
                print(
                    "backfill: embedding model unavailable "
                    "(sentence-transformers not installed / load failed). Aborting."
                )
                return
            for row, vec in zip(chunk_rows, vecs):
                row.embedding = vec.astype("float32").tolist()
                row.embedding_model = settings.embedding_model
                filled += 1
            await db.commit()
            print(f"backfill: committed {min(start + batch, len(rows))}/{len(rows)}")

        print(f"backfill: done — {filled} pattern(s) embedded with {settings.embedding_model}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=64, help="rows per encode/commit batch")
    args = ap.parse_args()
    asyncio.run(main(args.batch))
