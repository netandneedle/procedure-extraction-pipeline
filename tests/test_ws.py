"""WebSocket endpoint tests.

Tests the /ws/pipeline/{thread_id} WebSocket endpoint and the
ConnectionManager broadcast system. Uses Starlette's WebSocket
test client for in-process testing (no real network).
"""

import json
import pytest

from fastapi.testclient import TestClient

from app.main import app
from app.api.routes.ws import (
    ConnectionManager,
    build_status_event,
    _extract_gate_id,
)


# =============================================================================
# Unit tests: event building helpers
# =============================================================================

class TestBuildStatusEvent:
    """Tests for build_status_event()."""

    def test_status_change_event(self):
        """Normal node transition produces status_change type."""
        state = {
            "status": "parsing",
            "current_node": "parse_and_validate",
            "error": None,
            "entities": [],
            "chunks": [],
            "drafts": [],
            "objects_written": 0,
        }
        event = build_status_event("thread-1", state)
        assert event["type"] == "status_change"
        assert event["thread_id"] == "thread-1"
        assert event["data"]["status"] == "parsing"
        assert event["data"]["current_node"] == "parse_and_validate"
        assert event["data"]["gate_id"] is None

    def test_gate_arrived_event(self):
        """Gate status produces gate_arrived type with gate_id."""
        state = {"status": "gate_0", "current_node": "gate_0",
                 "entities": [1, 2, 3], "chunks": [], "drafts": [],
                 "objects_written": 0}
        event = build_status_event("thread-2", state)
        assert event["type"] == "gate_arrived"
        assert event["data"]["gate_id"] == 0
        assert event["data"]["entity_count"] == 3

    def test_completed_event(self):
        """Completed status produces completed type."""
        state = {"status": "completed", "current_node": "distribute",
                 "entities": [], "chunks": [], "drafts": [],
                 "objects_written": 42}
        event = build_status_event("thread-3", state)
        assert event["type"] == "completed"
        assert event["data"]["objects_written"] == 42

    def test_failed_event(self):
        """Failed status produces failed type with error."""
        state = {"status": "failed", "current_node": "serialize_stix",
                 "error": "Schema validation failed",
                 "entities": [], "chunks": [], "drafts": [],
                 "objects_written": 0}
        event = build_status_event("thread-4", state)
        assert event["type"] == "failed"
        assert event["data"]["error"] == "Schema validation failed"

    def test_missing_collections_default_to_zero(self):
        """State missing entity/chunk/draft lists defaults counts to 0."""
        state = {"status": "parsing", "current_node": "parse"}
        event = build_status_event("t", state)
        assert event["data"]["entity_count"] == 0
        assert event["data"]["chunk_count"] == 0
        assert event["data"]["draft_count"] == 0


class TestExtractGateId:
    """Tests for _extract_gate_id()."""

    def test_gate_0(self):
        assert _extract_gate_id("gate_0") == 0

    def test_gate_1(self):
        assert _extract_gate_id("gate_1") == 1

    def test_gate_2(self):
        assert _extract_gate_id("gate_2") == 2

    def test_non_gate_status(self):
        assert _extract_gate_id("parsing") is None

    def test_empty_string(self):
        assert _extract_gate_id("") is None

    def test_malformed_gate(self):
        assert _extract_gate_id("gate_abc") is None


# =============================================================================
# Unit tests: ConnectionManager
# =============================================================================

class TestConnectionManager:
    """Tests for ConnectionManager lifecycle and broadcast."""

    def test_has_subscribers_initially_false(self):
        mgr = ConnectionManager()
        assert mgr.has_subscribers("thread-1") is False

    @pytest.mark.asyncio
    async def test_connect_and_disconnect(self):
        """Connect adds subscriber, disconnect removes it."""
        mgr = ConnectionManager()
        ws = MockWebSocket()
        await mgr.connect("thread-1", ws)
        assert mgr.has_subscribers("thread-1") is True
        mgr.disconnect("thread-1", ws)
        assert mgr.has_subscribers("thread-1") is False

    @pytest.mark.asyncio
    async def test_broadcast_sends_to_all(self):
        """Broadcast delivers event to every connected client."""
        mgr = ConnectionManager()
        ws1 = MockWebSocket()
        ws2 = MockWebSocket()
        await mgr.connect("thread-1", ws1)
        await mgr.connect("thread-1", ws2)
        event = {"type": "test", "data": {}}
        await mgr.broadcast("thread-1", event)
        assert ws1.sent == [event]
        assert ws2.sent == [event]

    @pytest.mark.asyncio
    async def test_broadcast_to_empty_thread_is_noop(self):
        """Broadcasting to a thread with no subscribers does nothing."""
        mgr = ConnectionManager()
        await mgr.broadcast("nobody", {"type": "test"})
        # No error raised

    @pytest.mark.asyncio
    async def test_broadcast_drops_dead_connections(self):
        """Dead connections are removed on failed send."""
        mgr = ConnectionManager()
        ws_good = MockWebSocket()
        ws_dead = MockWebSocket(fail_on_send=True)
        await mgr.connect("thread-1", ws_good)
        await mgr.connect("thread-1", ws_dead)
        await mgr.broadcast("thread-1", {"type": "test"})
        # Dead connection should have been removed
        assert len(mgr._connections["thread-1"]) == 1
        assert ws_good in mgr._connections["thread-1"]

    @pytest.mark.asyncio
    async def test_broadcast_tolerates_connection_added_mid_send(self):
        """A client connecting mid-broadcast must not crash the broadcast.

        Regression: broadcast iterated the *live* connection set while
        `await send_json` yielded the event loop. A client connecting
        (`.add`) or disconnecting (`.discard`) during that await mutated
        the set being iterated, raising "Set changed size during
        iteration". That exception escaped the pipeline runner and
        flipped an already-completed run to `failed`.
        """
        mgr = ConnectionManager()

        class MutatingWebSocket:
            """On send, simulates another client connecting concurrently."""

            def __init__(self):
                self.sent = []

            async def accept(self):
                pass

            async def send_json(self, data):
                # Mutate the same set broadcast is iterating, as a real
                # connect/disconnect would during the await.
                mgr._connections["thread-1"].add(MockWebSocket())
                self.sent.append(data)

        for _ in range(3):
            await mgr.connect("thread-1", MutatingWebSocket())

        # Must NOT raise RuntimeError: Set changed size during iteration.
        await mgr.broadcast("thread-1", {"type": "test"})

    @pytest.mark.asyncio
    async def test_broadcast_tolerates_disconnect_mid_send(self):
        """A client disconnecting mid-broadcast must not crash the broadcast."""
        mgr = ConnectionManager()

        class DisconnectingWebSocket:
            def __init__(self):
                self.sent = []

            async def accept(self):
                pass

            async def send_json(self, data):
                # Drop an arbitrary *other* live connection mid-iteration.
                others = [c for c in mgr._connections["thread-1"] if c is not self]
                if others:
                    mgr._connections["thread-1"].discard(others[0])
                self.sent.append(data)

        for _ in range(4):
            await mgr.connect("thread-1", DisconnectingWebSocket())

        await mgr.broadcast("thread-1", {"type": "test"})

    @pytest.mark.asyncio
    async def test_multiple_threads_isolated(self):
        """Events for one thread don't leak to another."""
        mgr = ConnectionManager()
        ws_a = MockWebSocket()
        ws_b = MockWebSocket()
        await mgr.connect("thread-a", ws_a)
        await mgr.connect("thread-b", ws_b)
        await mgr.broadcast("thread-a", {"type": "for_a"})
        assert ws_a.sent == [{"type": "for_a"}]
        assert ws_b.sent == []


class MockWebSocket:
    """Lightweight mock for testing ConnectionManager without FastAPI."""

    def __init__(self, fail_on_send=False):
        self.sent = []
        self.accepted = False
        self._fail_on_send = fail_on_send

    async def accept(self):
        self.accepted = True

    async def send_json(self, data):
        if self._fail_on_send:
            raise RuntimeError("connection closed")
        self.sent.append(data)


# =============================================================================
# Integration tests: WebSocket endpoint via TestClient
# =============================================================================

class TestWebSocketEndpoint:
    """Integration tests for the /ws/pipeline/{thread_id} endpoint."""

    def test_connect_receives_connected_event(self):
        """Client receives a 'connected' event immediately after handshake."""
        client = TestClient(app)
        with client.websocket_connect("/ws/pipeline/test-thread-1") as ws:
            data = ws.receive_json()
            assert data["type"] == "connected"
            assert data["thread_id"] == "test-thread-1"

    def test_heartbeat_response(self):
        """Server sends heartbeat, client responds with pong."""
        # This tests the client -> server pong path
        client = TestClient(app)
        with client.websocket_connect("/ws/pipeline/test-thread-2") as ws:
            # Read the connected event
            ws.receive_json()
            # Send a pong (simulating client responding to heartbeat)
            ws.send_text("pong")
            # Connection should stay alive (no error)

    def test_different_threads_are_independent(self):
        """Two clients on different threads get independent events."""
        client = TestClient(app)
        with client.websocket_connect("/ws/pipeline/thread-x") as ws1:
            msg1 = ws1.receive_json()
            assert msg1["thread_id"] == "thread-x"
        with client.websocket_connect("/ws/pipeline/thread-y") as ws2:
            msg2 = ws2.receive_json()
            assert msg2["thread_id"] == "thread-y"
