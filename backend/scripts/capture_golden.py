"""Capture a real pipeline run's state into a golden fixture.

WHY:
The hand-written fixtures in tests/conftest.py are a thin subset of what a
real run produces, and that gap ships bugs. `x_chain_label`/`x_chain_root`
are written by the serializer for multi-chain sources; no fixture draft
carried them, so every one of ~1000 tests passed while a live run failed.
Golden fixtures close that gap: real state, replayed
deterministically, no LLM calls and no API cost.

WHAT IT CAPTURES:
Only the state keys the deterministic tail actually reads — derived from the
nodes themselves, not from docstrings (which were already out of date). Two
consequences worth stating plainly:

  1. `parsed_text` is read by NO deterministic node, so the vendor document
     never enters the fixture at all.
  2. Verbatim-quote fields inside what we do keep (`source_excerpt`,
     `context_snippet`) are redacted, because those are literal extracts from
     the source report.

That matters here: `data/test-sources/*` is gitignored precisely so vendor
PDFs stay out of the repo. A fixture that embedded their prose would quietly
undo that decision. What remains is structure — which fields exist, with what
shapes and combinations — which is the part that catches contract bugs.

USAGE:
    python scripts/capture_golden.py --thread <uuid> --name <slug>
    python scripts/capture_golden.py --thread <uuid> --name <slug> --step 9

Run it from inside the api container (it needs the checkpointer):
    docker exec -e PYTHONPATH=/app -w /app <api> python scripts/capture_golden.py ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.graph.checkpointer import get_checkpointer  # noqa: E402

# State keys the deterministic tail reads, unioned across normalize,
# serialize_stix, validate_bundle and distribute. Kept as an explicit
# allowlist rather than a redaction blocklist: anything not needed simply
# never lands in the fixture.
CAPTURE_KEYS = (
    "bundle_corrections",
    "chunk_conditions",
    "chunk_operators",
    "chunks",
    "detection_rules",
    "drafts",
    "gate1_approved_draft_ids",
    "gates_enabled",
    "is_sequential",
    "metadata",
    "normalized_drafts",
    "raw_content_path",
    "source_id",
    "source_reliability",
    "source_type",
    "title",
    "validated_entities",
)

# Fields holding text quoted verbatim from the source document.
REDACT_FIELDS = ("source_excerpt", "context_snippet")
REDACTION = "[redacted: verbatim source text]"

# The node whose output we want as the replay input. Step 9 = after gate_1,
# i.e. the last state before the deterministic tail begins.
DEFAULT_AFTER_NODE = "gate_1"


def _redact(value):
    """Recursively blank verbatim-quote fields, preserving structure."""
    if isinstance(value, dict):
        return {
            k: (REDACTION if k in REDACT_FIELDS and isinstance(v, str) and v else _redact(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


def _shape(value) -> str:
    if isinstance(value, list):
        return f"list[{len(value)}]"
    if isinstance(value, dict):
        return f"dict[{len(value)}]"
    if isinstance(value, str):
        return f"str[{len(value)}]"
    return type(value).__name__


async def capture(thread_id: str, name: str, after_node: str, out_root: Path) -> Path:
    cp = await get_checkpointer()
    cfg = {"configurable": {"thread_id": thread_id}}

    chosen = None
    async for tup in cp.alist(cfg, limit=200):
        cv = tup.checkpoint.get("channel_values", {})
        if cv.get("current_node") == after_node:
            step = (tup.metadata or {}).get("step", -1)
            if chosen is None or step > chosen[0]:
                chosen = (step, cv)

    if chosen is None:
        raise SystemExit(
            f"no checkpoint found with current_node={after_node!r} on thread "
            f"{thread_id}. Run with --step to inspect, or pick another node."
        )

    step, cv = chosen
    state = {k: _redact(cv[k]) for k in CAPTURE_KEYS if k in cv}

    missing = [k for k in CAPTURE_KEYS if k not in cv]
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    payload_path = out_dir / f"after_{after_node}.json"
    payload_path.write_text(json.dumps(state, indent=2, default=str))

    manifest = {
        "name": name,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "thread_id": thread_id,
        "after_node": after_node,
        "checkpoint_step": step,
        "title": cv.get("title", ""),
        "keys_captured": sorted(state),
        "keys_absent_at_this_step": missing,
        "redacted_fields": list(REDACT_FIELDS),
        "note": (
            "Allowlisted to the state keys the deterministic tail reads. "
            "parsed_text is deliberately absent — no deterministic node reads "
            "it, so the source document never enters the repo."
        ),
        "shapes": {k: _shape(v) for k, v in sorted(state.items())},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"captured step={step} (after {after_node}) -> {payload_path}")
    print(f"  size: {payload_path.stat().st_size / 1024:.0f} KB")
    for k, v in sorted(state.items()):
        print(f"    {k:<28} {_shape(v)}")
    if missing:
        print(f"  not present at this step: {missing}")
    return payload_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thread", required=True, help="LangGraph thread_id of a completed run")
    ap.add_argument("--name", required=True, help="fixture directory name, e.g. cascading-shadows")
    ap.add_argument("--after-node", default=DEFAULT_AFTER_NODE,
                    help=f"capture the state produced by this node (default: {DEFAULT_AFTER_NODE})")
    ap.add_argument("--out", default=None, help="fixture root (default: <repo>/tests/golden)")
    ap.add_argument("--step", action="store_true", help="list checkpoints and exit")
    args = ap.parse_args()

    if args.step:
        asyncio.run(_list_steps(args.thread))
        return

    out_root = Path(args.out) if args.out else Path(__file__).resolve().parents[2] / "tests" / "golden"
    asyncio.run(capture(args.thread, args.name, args.after_node, out_root))


async def _list_steps(thread_id: str) -> None:
    cp = await get_checkpointer()
    rows = []
    async for tup in cp.alist({"configurable": {"thread_id": thread_id}}, limit=200):
        cv = tup.checkpoint.get("channel_values", {})
        rows.append(((tup.metadata or {}).get("step", -1), cv.get("current_node"), cv.get("status")))
    for step, node, status in sorted(rows):
        print(f"  step={step:<4} node={node} status={status}")


if __name__ == "__main__":
    main()
