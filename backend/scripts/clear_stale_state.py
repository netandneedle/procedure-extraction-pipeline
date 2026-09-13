"""One-off: clear stale `error` and `bundle_corrections` from the
LangGraph checkpoint of an in-flight thread, then sync the source row.

Used when a retry of a previously-failed source carries forward the
prior run's error/corrections via the runner's aget_state sync.
"""

import asyncio
import sys

from app.graph.checkpointer import get_checkpointer
from app.graph.pipeline import compile_pipeline
from app.models.base import async_session
from app.services import queue as queue_service
import uuid


async def main(source_id: str) -> None:
    checkpointer = await get_checkpointer()
    graph = compile_pipeline(checkpointer=checkpointer)
    config = {"configurable": {"thread_id": source_id}}
    state = await graph.aget_state(config)
    if True:
        if not state or not state.values:
            print(f"No state for thread {source_id}")
            return
        before = {
            "error": state.values.get("error"),
            "bundle_corrections": len(state.values.get("bundle_corrections") or []),
        }
        print(f"Before: {before}")
        await graph.aupdate_state(
            config,
            {"error": None, "bundle_corrections": []},
        )
        # Sync the source row too so the chip refreshes on the Kanban.
        async with async_session() as db:
            await queue_service.update_status(
                db, uuid.UUID(source_id),
                state.values.get("status", "parsing"),
                error=None,
                bundle_corrections=[],
            )
        new_state = await graph.aget_state(config)
        after = {
            "error": new_state.values.get("error"),
            "bundle_corrections": len(new_state.values.get("bundle_corrections") or []),
        }
        print(f"After:  {after}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1]))
