"""WebSocket endpoint for real-time pipeline status.

Clients connect to /ws/pipeline/{thread_id} to receive JSON events
as the pipeline progresses through nodes and gates.

Event format:
    {
        "type": "status_change" | "gate_arrived" | "completed" | "failed" | "heartbeat",
        "thread_id": "...",
        "data": { ... }
    }

Status change data mirrors PipelineStatusResponse fields.
Gate arrived data includes gate_id and pending review counts.

Architecture:
    - ConnectionManager: maps thread_id -> set of WebSocket connections
    - Pipeline background task calls manager.broadcast() at each node transition
    - Heartbeat: server sends ping every 30s; client responds with pong
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter()


class ConnectionManager:
    """Track active WebSocket connections by pipeline thread_id.

    Thread-safe via asyncio (single event loop). Multiple clients
    can watch the same pipeline.
    """

    def __init__(self):
        self._connections: dict[str, set[WebSocket]] = defaultdict(set)

    async def connect(self, thread_id: str, ws: WebSocket) -> None:
        await ws.accept()
        self._connections[thread_id].add(ws)
        logger.info("ws: client connected for thread %s (%d total)",
                     thread_id, len(self._connections[thread_id]))

    def disconnect(self, thread_id: str, ws: WebSocket) -> None:
        self._connections[thread_id].discard(ws)
        if not self._connections[thread_id]:
            del self._connections[thread_id]
        logger.info("ws: client disconnected from thread %s", thread_id)

    async def broadcast(self, thread_id: str, event: dict) -> None:
        """Send an event to all clients watching a thread.

        Silently drops connections that have closed.

        Iterates a *snapshot* of the connection set, not the live set:
        ``await ws.send_json`` yields the event loop, and a client
        connecting (``connect`` -> ``.add``) or disconnecting
        (``disconnect`` -> ``.discard``) mid-broadcast mutates the same
        set. Iterating the live set then raises "Set changed size during
        iteration" — which previously escaped the pipeline runner's
        astream loop and flipped an already-completed run to ``failed``.
        """
        dead: list[WebSocket] = []
        for ws in list(self._connections.get(thread_id, ())):
            try:
                await ws.send_json(event)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(thread_id, ws)

    def has_subscribers(self, thread_id: str) -> bool:
        return bool(self._connections.get(thread_id))


# Singleton — imported by pipeline.py to broadcast events
manager = ConnectionManager()

HEARTBEAT_INTERVAL_S = 30


@router.websocket("/ws/pipeline/{thread_id}")
async def pipeline_ws(ws: WebSocket, thread_id: str):
    """WebSocket connection for pipeline status updates.

    Sends events as JSON. Keeps the connection alive with periodic
    heartbeat pings. Closes cleanly on client disconnect.
    """
    await manager.connect(thread_id, ws)

    try:
        # Send initial connected event
        await ws.send_json({
            "type": "connected",
            "thread_id": thread_id,
            "data": {"message": "Subscribed to pipeline events"},
        })

        # Keep connection alive: listen for client messages + heartbeat
        while True:
            try:
                # Wait for client message or heartbeat timeout
                msg = await asyncio.wait_for(
                    ws.receive_text(),
                    timeout=HEARTBEAT_INTERVAL_S,
                )
                # Handle client pong or any message
                if msg == "pong":
                    continue
                # Future: could accept commands here
            except asyncio.TimeoutError:
                # No message received — send heartbeat ping
                try:
                    await ws.send_json({
                        "type": "heartbeat",
                        "thread_id": thread_id,
                        "data": {},
                    })
                except Exception:
                    break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("ws: unexpected error for thread %s: %s", thread_id, e)
    finally:
        manager.disconnect(thread_id, ws)


def build_status_event(thread_id: str, state: dict) -> dict:
    """Build a status_change event from pipeline state."""
    status = state.get("status", "unknown")

    # Determine event type from status
    if status in ("completed",):
        event_type = "completed"
    elif status in ("failed",):
        event_type = "failed"
    elif status in ("gate_0", "gate_1", "gate_2"):
        event_type = "gate_arrived"
    else:
        event_type = "status_change"

    return {
        "type": event_type,
        "thread_id": thread_id,
        "data": {
            "status": status,
            "current_node": state.get("current_node"),
            "error": state.get("error"),
            "entity_count": len(state.get("entities", [])),
            "chunk_count": len(state.get("chunks", [])),
            "draft_count": len(state.get("drafts", [])),
            "objects_written": state.get("objects_written", 0),
            "persistence_errors": state.get("persistence_errors") or [],
            "bundle_corrections": state.get("bundle_corrections") or [],
            "gate_id": _extract_gate_id(status),
        },
    }


def _extract_gate_id(status: str) -> int | None:
    """Extract gate number from status string like 'gate_0'."""
    if status and status.startswith("gate_"):
        try:
            return int(status.split("_")[1])
        except (IndexError, ValueError):
            pass
    return None
