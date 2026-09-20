"""Re-run the chunker against a source's checkpointed parsed_text and print
the chain shape it produced — roots, labels, predecessors, shared segments,
tactic-order warnings.

Built for the SHARED SEGMENT RULE (an exploit kit several campaigns enter):
the question is whether the chunker now emits the hourglass natively —
campaign lures as chain roots, one labelled shared segment, tails labelled
per campaign — instead of making the kit the sole entry point.

One LLM call (the chunker), no state written. Usage (inside the api
container, thread_id == source_id for pipeline-created sources):

    docker compose exec -w /app api python -m scripts.dry_run_chunker \
        --thread-id 6ed3fb67-03a9-49b3-acfe-3923e6e658b1
    python -m scripts.dry_run_chunker --state-json /tmp/bluemoon_state.json

The chunker's prompt is part of the LLM cache key, so a prompt change is a
cache miss and a repeat run with the same prompt replays the cached answer.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from collections import defaultdict
from pathlib import Path

from app.graph.checkpointer import get_checkpointer
from app.nodes.llm.chunking import chunk_behaviors

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("app.nodes.llm.chunking").setLevel(logging.INFO)


async def _load_state(thread_id: str) -> dict:
    saver = await get_checkpointer()
    snap = await saver.aget({"configurable": {"thread_id": thread_id}})
    if not snap:
        raise SystemExit(f"No checkpoint for thread_id={thread_id}")
    return dict(snap["channel_values"])


def _shape(chunks: list[dict]) -> list[str]:
    by_id = {c["chunk_id"]: c for c in chunks}
    preds_of: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        for t in c.get("precedes_ids") or []:
            if t in by_id:
                preds_of[t].append(c["chunk_id"])
    roots = [c["chunk_id"] for c in chunks if not preds_of.get(c["chunk_id"])]
    declared = [c["chunk_id"] for c in chunks if c.get("chain_root")]
    reach: dict[str, set[str]] = defaultdict(set)
    for r in roots:
        stack, seen = [r], set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            reach[n].add(r)
            stack.extend(by_id[n].get("precedes_ids") or [])
    shared = [
        cid for cid in by_id
        if len(preds_of.get(cid, [])) >= 2
        and len(set().union(*(reach[p] for p in preds_of[cid]))) >= 2
    ]
    labels = sorted({c.get("chain_label") or "" for c in chunks})
    warned = [c["chunk_id"] for c in chunks if c.get("flow_warnings")]
    return [
        f"chunks: {len(chunks)}   edges: {sum(len(c.get('precedes_ids') or []) for c in chunks)}",
        f"entry points (no predecessors): {roots}",
        f"declared chain roots: {declared}",
        f"chain labels: {labels}",
        f"shared entries (predecessors from >=2 entry points): {shared}",
        f"tactic-order warnings on: {warned}",
    ]


async def main(thread_id: str | None, state_json: str | None) -> None:
    if state_json:
        import json
        state = json.loads(Path(state_json).read_text())
    else:
        state = await _load_state(thread_id)
    parsed_text = state.get("parsed_text") or ""
    if not parsed_text:
        raise SystemExit("checkpoint has no parsed_text")
    is_sequential = bool(state.get("is_sequential", True))
    print(f"parsed_text: {len(parsed_text):,} chars | is_sequential={is_sequential} "
          f"| prior chunks in checkpoint: {len(state.get('chunks') or [])}")

    run_state = {
        "parsed_text": parsed_text,
        "validated_entities": state.get("validated_entities") or state.get("entities") or [],
        "metadata": state.get("metadata") or {},
        "is_sequential": is_sequential,
        "source_id": state.get("source_id"),
    }
    result = await chunk_behaviors(run_state)
    chunks = result.get("chunks") or []
    print()
    for c in chunks:
        tactics = (c.get("context") or {}).get("tactics") or []
        print(f"[{c['chunk_id']}] seq={c.get('sequence_index')} "
              f"root={bool(c.get('chain_root'))} label={c.get('chain_label')!r}")
        print(f"    after={c.get('predecessor_indices')} precedes={c.get('precedes_ids')} "
              f"branch={bool(c.get('branch_point'))} converge={bool(c.get('convergence_point'))} "
              f"tactics={tactics[:3]}")
        print(f"    {(c.get('text') or '')[:150]}")
        for w in c.get("flow_warnings") or []:
            print(f"    WARNING: {w}")
    print()
    print("\n".join(_shape(chunks)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--thread-id", help="LangGraph thread to read parsed_text from")
    ap.add_argument("--state-json", help="a JSON file with parsed_text / validated_entities / "
                    "metadata / is_sequential, e.g. exported from another instance's checkpoint")
    args = ap.parse_args()
    if not args.thread_id and not args.state_json:
        ap.error("one of --thread-id or --state-json is required")
    asyncio.run(main(args.thread_id, args.state_json))
