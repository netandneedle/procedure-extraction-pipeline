"""Seed the example table from runs a human already reviewed.

The corrected-example channel starts empty, and the only corrections that
belong in it have already happened: three sources reviewed by hand at all four
gates. Everything else in the queue was run unattended, so its "corrections"
are an AI reviewer agreeing with itself — the exact provenance that produced 63
patterns of which roughly one in seven was wrong.

Reads each source's stored checkpoint, recomputes the deltas the synthesizer
would have seen, and persists them. `record_examples` applies the same
human-attended gate filter it applies live, so a source with some gates on
`auto` contributes only its human gates.

Idempotent: `persist_examples` dedups on `dedup_key`, so a second run bumps
occurrence counts rather than duplicating rows. That is worth knowing before
re-running — the counts are read as "how often this correction recurred", and
a needless re-run inflates them.

USAGE (inside the api container)
    python -m scripts.backfill_feedback_examples --dry-run
    python -m scripts.backfill_feedback_examples
    python -m scripts.backfill_feedback_examples --threads <uuid>,<uuid>
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from sqlalchemy import text as sql_text

from app.graph.checkpointer import get_checkpointer
from app.models.base import async_session
from app.nodes.llm.feedback_synthesis import _compute_deltas
from app.services.feedback_examples import example_rows_from_deltas, record_examples

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

RULE = "=" * 78


async def candidates(only_threads: list[str] | None) -> list[dict]:
    async with async_session() as db:
        rows = (await db.execute(sql_text(
            "SELECT id, thread_id, title FROM sources "
            "WHERE thread_id IS NOT NULL ORDER BY created_at"
        ))).all()
    cp = await get_checkpointer()
    out = []
    for sid, thread_id, title in rows:
        tid = str(thread_id)
        if only_threads:
            if tid not in only_threads:
                continue
        elif not (title or "").startswith("[HUMAN"):
            continue
        try:
            snap = await cp.aget({"configurable": {"thread_id": tid}})
        except Exception as e:  # noqa: BLE001
            print(f"  ! {tid[:8]} unreadable ({type(e).__name__})", file=sys.stderr)
            continue
        if snap is None:
            continue
        values = dict(snap.get("channel_values", {}))
        # The checkpoint carries the source_id already, but a stored run that
        # predates that field would silently write examples with no provenance.
        values.setdefault("source_id", str(sid))
        values.setdefault("title", title or "")
        out.append({"title": title or tid, "values": values})
    return out


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", default="")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would be written and touch nothing")
    args = ap.parse_args()
    only = [t.strip() for t in args.threads.split(",") if t.strip()] or None

    srcs = await candidates(only)
    if not srcs:
        print("no human-reviewed sources found", file=sys.stderr)
        return

    print(f"{RULE}\n{'dry run — ' if args.dry_run else ''}"
          f"{len(srcs)} source(s)\n{RULE}")
    total = 0
    for s in srcs:
        deltas = _compute_deltas(s["values"])
        rows = example_rows_from_deltas(s["values"], deltas)
        total += len(rows)
        print(f"\n  {s['title'][:60]}  ->  {len(rows)} example(s)")
        for r in rows:
            print(f"    [{r['area']}/{r['action']}] "
                  f"{json.dumps(r['before'], default=str)[:90]}")
            if r.get("rationale"):
                print(f"        said: {r['rationale'][:80]}")
        if not args.dry_run and rows:
            print("   ", await record_examples(s["values"], deltas))
    print(f"\n{RULE}\n  {total} example(s) "
          f"{'would be' if args.dry_run else ''} written\n{RULE}")


if __name__ == "__main__":
    asyncio.run(main())
