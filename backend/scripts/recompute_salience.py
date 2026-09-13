"""Periodic: recompute salience for all feedback patterns and archive the
low-salience, stale ones.

Salience decays with age (recency half-life), so a pattern that was useful
months ago but hasn't recurred or scored a hit drifts down. This sweep:

1. Recomputes salience for every non-archived pattern from its current
   hit/miss/occurrence/last_seen.
2. Archives (status='archived') patterns whose salience < feedback_min_salience
   AND whose last_seen_at is older than --stale-days (default 90). The 90-day
   recency floor still applies; salience only lets us archive a pattern EARLY
   once the closed loop shows it stopped helping. Archiving is reversible via
   the management UI.

Run from cron or by hand (from backend/ with the venv active):
    python -m scripts.recompute_salience
    python -m scripts.recompute_salience --stale-days 60 --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from app.config import settings
from app.models.base import async_session
from app.models.feedback_pattern import FeedbackPattern
from app.services.feedback_patterns import clear_feedback_cache, compute_salience


async def main(stale_days: int, dry_run: bool) -> None:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=stale_days)
    async with async_session() as db:
        rows = list(
            (
                await db.execute(
                    select(FeedbackPattern).where(FeedbackPattern.status != "archived")
                )
            ).scalars().all()
        )
        recomputed = archived = 0
        for p in rows:
            p.salience = compute_salience(
                hit_count=p.hit_count or 0,
                miss_count=p.miss_count or 0,
                occurrence_count=p.occurrence_count or 1,
                last_seen_at=p.last_seen_at,
                now=now,
            )
            recomputed += 1
            last_seen = p.last_seen_at
            if last_seen is not None and last_seen.tzinfo is None:
                last_seen = last_seen.replace(tzinfo=timezone.utc)
            stale = last_seen is not None and last_seen < cutoff
            if p.salience < settings.feedback_min_salience and stale:
                archived += 1
                if not dry_run:
                    p.status = "archived"
                    p.last_scored_at = now
        if not dry_run:
            await db.commit()
            clear_feedback_cache()

    verb = "would archive" if dry_run else "archived"
    print(
        f"recompute_salience: recomputed {recomputed}, {verb} {archived} "
        f"(stale > {stale_days}d AND salience < {settings.feedback_min_salience})"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stale-days", type=int, default=90)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    asyncio.run(main(args.stale_days, args.dry_run))
