"""API endpoint tests.

Tests the FastAPI routes with mocked dependencies (no real PostgreSQL
or LangGraph checkpointer needed). Validates:
- Request/response shapes
- Status code correctness
- Dependency injection wiring
- Gate state validation (409 when pipeline not at expected gate)
- Source queue CRUD operations
- Pipeline run triggering
"""

import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Patch settings before importing app (avoids real DB connection)
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.graph.state import (
    Entity,
    EntityType,
    GateAction,
    PipelineStatus,
    ProcedureDraft,
    TechniqueMapping,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def mock_db():
    """Mock AsyncSession for database operations."""
    session = AsyncMock()
    return session


@pytest.fixture
def mock_source():
    """A realistic Source model mock."""
    source_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    source = MagicMock()
    source.id = source_id
    source.source_type = "markdown"
    source.title = "ActiveMQ Exploitation Report"
    source.raw_content_path = "/data/reports/activemq.md"
    source.status = "queued"
    source.channel = "manual"
    source.gates_enabled = True
    source.gate_modes = "review"
    source.claimed_by = None
    source.claimed_at = None
    source.metadata_ = {"author": "Test Author", "tlp": "white"}
    source.source_reliability = 75
    source.thread_id = None
    source.error = None
    source.created_at = now
    source.updated_at = now
    # Allow Pydantic from_attributes to work
    source.__class__.__name__ = "Source"

    # Map attribute access for model_validate(from_attributes=True)
    # by setting a proper dict representation
    def _getattr(name):
        attrs = {
            "id": source_id,
            "source_type": "markdown",
            "title": "ActiveMQ Exploitation Report",
            "raw_content_path": "/data/reports/activemq.md",
            "status": "queued",
            "channel": "manual",
            "gates_enabled": True,
            "claimed_by": None,
            "claimed_at": None,
            "metadata": {"author": "Test Author", "tlp": "white"},
            "source_reliability": 75,
            "thread_id": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
        }
        return attrs.get(name)

    return source


@pytest.fixture
def mock_graph():
    """Mock compiled LangGraph pipeline."""
    graph = AsyncMock()
    return graph


@pytest.fixture
def sample_pipeline_state():
    """Pipeline state at gate_0 with entities to review."""
    return {
        # Gate submit endpoints call uuid.UUID(source_id) to type the
        # queue row update, so the fixture must carry a real UUID string.
        "source_id": "00000000-0000-4000-8000-0000000000aa",
        "status": "gate_0",
        "current_node": "gate_0",
        "gates_enabled": True,
        "metadata": {"author": "Test Author"},
        "parse_warnings": [],
        "entities": [
            asdict(Entity(
                entity_id="ent-001",
                entity_type=EntityType.MALWARE.value,
                value="LockBit 3.0",
                confidence=0.92,
            )),
            asdict(Entity(
                entity_id="ent-002",
                entity_type=EntityType.TOOL.value,
                value="certutil.exe",
                confidence=0.88,
            )),
        ],
        "chunks": [],
        "drafts": [],
        "validated_entities": [],
        "error": None,
        "objects_written": 0,
    }


# =============================================================================
# Source Queue Tests
# =============================================================================

class TestSourceQueueRoutes:
    """Test /api/sources endpoints."""

    def test_list_sources_empty(self):
        """GET /api/sources returns empty list when no sources exist."""
        with patch("app.api.routes.source_queue.queue_service") as mock_qs:
            mock_qs.list_sources = AsyncMock(return_value=([], 0))

            from app.main import app
            from app.api.dependencies import get_db
            app.dependency_overrides[get_db] = lambda: AsyncMock()

            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/api/sources/")

            assert response.status_code == 200
            data = response.json()
            assert data["sources"] == []
            assert data["total"] == 0

            app.dependency_overrides.clear()

    def test_list_sources_with_status_filter(self):
        """GET /api/sources?status=queued passes filter to service."""
        with patch("app.api.routes.source_queue.queue_service") as mock_qs:
            mock_qs.list_sources = AsyncMock(return_value=([], 0))

            from app.main import app
            from app.api.dependencies import get_db
            app.dependency_overrides[get_db] = lambda: AsyncMock()

            client = TestClient(app, raise_server_exceptions=False)
            response = client.get("/api/sources/?status=queued")

            assert response.status_code == 200
            mock_qs.list_sources.assert_called_once()
            call_kwargs = mock_qs.list_sources.call_args
            assert call_kwargs[1]["status"] == "queued" or call_kwargs[0][1] == "queued"

            app.dependency_overrides.clear()

    def test_create_source(self):
        """POST /api/sources creates a new source."""
        source_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        fake_source = MagicMock()
        fake_source.id = source_id
        fake_source.source_type = "markdown"
        fake_source.title = "Test Report"
        fake_source.raw_content_path = "/data/test.md"
        fake_source.status = "queued"
        fake_source.channel = "manual"
        fake_source.gates_enabled = {"entities": True, "chunks": True, "procedures": True, "bundle": True}
        fake_source.gate_modes = "review"
        fake_source.claimed_by = None
        fake_source.claimed_at = None
        fake_source.metadata_ = {}  # Pydantic AliasPath reads from metadata_
        fake_source.source_reliability = 50
        fake_source.sequentiality = "auto"
        fake_source.extract_figures = True
        fake_source.thread_id = None
        fake_source.error = None
        fake_source.created_at = now
        fake_source.updated_at = now

        with patch("app.api.routes.source_queue.queue_service") as mock_qs:
            mock_qs.create_source = AsyncMock(return_value=fake_source)

            from app.main import app
            from app.api.dependencies import get_db
            app.dependency_overrides[get_db] = lambda: AsyncMock()

            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/api/sources/", json={
                "source_type": "markdown",
                "title": "Test Report",
                "raw_content_path": "/data/test.md",
            })

            assert response.status_code == 201
            data = response.json()
            assert data["source_type"] == "markdown"
            assert data["title"] == "Test Report"
            assert data["status"] == "queued"

            app.dependency_overrides.clear()

    def test_create_source_missing_required_field(self):
        """POST /api/sources without required fields returns 422."""
        from app.main import app
        from app.api.dependencies import get_db
        app.dependency_overrides[get_db] = lambda: AsyncMock()

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/api/sources/", json={
            "title": "Missing source_type and path",
        })

        assert response.status_code == 422
        app.dependency_overrides.clear()

    def test_get_source_not_found(self):
        """GET /api/sources/{id} returns 404 for missing source."""
        with patch("app.api.routes.source_queue.queue_service") as mock_qs:
            mock_qs.get_source = AsyncMock(return_value=None)

            from app.main import app
            from app.api.dependencies import get_db
            app.dependency_overrides[get_db] = lambda: AsyncMock()

            client = TestClient(app, raise_server_exceptions=False)
            response = client.get(f"/api/sources/{uuid.uuid4()}")

            assert response.status_code == 404
            app.dependency_overrides.clear()

    def test_claim_source_conflict(self):
        """PATCH /api/sources/{id}/claim returns 409 if already claimed."""
        with patch("app.api.routes.source_queue.queue_service") as mock_qs:
            mock_qs.claim_source = AsyncMock(
                side_effect=ValueError("Source already claimed by alice@example.com")
            )

            from app.main import app
            from app.api.dependencies import get_db
            app.dependency_overrides[get_db] = lambda: AsyncMock()

            client = TestClient(app, raise_server_exceptions=False)
            response = client.patch(
                f"/api/sources/{uuid.uuid4()}/claim",
                json={"analyst": "bob@example.com"},
            )

            assert response.status_code == 409
            assert "already claimed" in response.json()["detail"]
            app.dependency_overrides.clear()


# =============================================================================
# Gate Tests
# =============================================================================

class TestGateRoutes:
    """Test /api/gates endpoints."""

    def test_get_pending_gate0(self, sample_pipeline_state):
        """GET /gates/{thread}/0/pending returns entities when at gate_0."""
        state_snapshot = MagicMock()
        state_snapshot.values = sample_pipeline_state

        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=state_snapshot)

        from app.main import app
        from app.api.dependencies import get_graph
        app.dependency_overrides[get_graph] = lambda: mock_graph

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/gates/00000000-0000-4000-8000-000000000001/0/pending")

        assert response.status_code == 200
        data = response.json()
        assert data["gate_id"] == 0
        assert len(data["items"]) == 2
        assert data["items"][0]["entity_id"] == "ent-001"
        assert "metadata" in data["context"]

        app.dependency_overrides.clear()

    def test_get_pending_wrong_gate_status(self, sample_pipeline_state):
        """GET /gates/{thread}/1/pending returns 409 when pipeline is at gate_0."""
        state_snapshot = MagicMock()
        state_snapshot.values = sample_pipeline_state  # status is gate_0

        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=state_snapshot)

        from app.main import app
        from app.api.dependencies import get_graph
        app.dependency_overrides[get_graph] = lambda: mock_graph

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/gates/00000000-0000-4000-8000-000000000001/1/pending")

        assert response.status_code == 409
        assert "gate_0" in response.json()["detail"]

        app.dependency_overrides.clear()

    def test_get_pending_invalid_gate_id(self):
        """GET /gates/{thread}/5/pending returns 400 for invalid gate_id."""
        mock_graph = AsyncMock()

        from app.main import app
        from app.api.dependencies import get_graph
        app.dependency_overrides[get_graph] = lambda: mock_graph

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/gates/00000000-0000-4000-8000-000000000001/5/pending")

        assert response.status_code == 400
        assert "Invalid gate_id" in response.json()["detail"]

        app.dependency_overrides.clear()

    def test_get_pending_no_state(self):
        """GET /gates/{thread}/0/pending returns 404 when no state exists."""
        state_snapshot = MagicMock()
        state_snapshot.values = None

        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=state_snapshot)

        from app.main import app
        from app.api.dependencies import get_graph
        app.dependency_overrides[get_graph] = lambda: mock_graph

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/gates/00000000-0000-4000-8000-000000000001/0/pending")

        assert response.status_code == 404

        app.dependency_overrides.clear()

    def test_submit_gate0(self, sample_pipeline_state):
        """POST /gates/{thread}/0/submit writes reviews, flips queue
        to resuming_from_gate_0, and returns 202 immediately.

        The background task resumes the graph; the endpoint does not
        wait for it. The body only asserts the inline state change and
        the outgoing queue write; the full resume flow is exercised in
        test_e2e.py.
        """
        pre_state = MagicMock()
        pre_state.values = sample_pipeline_state

        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=pre_state)
        mock_graph.aupdate_state = AsyncMock()

        # astream is only reached by the background task; fake it so
        # the task completes harmlessly if TestClient runs it inline.
        async def fake_stream(*args, **kwargs):
            yield {"chunk_behaviors": {"status": "chunking"}}
        mock_graph.astream = fake_stream

        with patch("app.api.routes.gates.queue_service") as mock_qs, \
             patch("app.api.routes.pipeline.queue_service") as mock_qs_pipeline:
            mock_qs.update_status = AsyncMock()
            mock_qs_pipeline.update_status = AsyncMock()

            from app.main import app
            from app.api.dependencies import get_graph
            app.dependency_overrides[get_graph] = lambda: mock_graph

            client = TestClient(app, raise_server_exceptions=False)
            response = client.post(
                "/api/gates/00000000-0000-4000-8000-000000000001/0/submit",
                json={
                    "reviews": [
                        {"entity_id": "ent-001", "action": "approve"},
                        {"entity_id": "ent-002", "action": "remove", "rationale": "Not relevant"},
                    ],
                },
            )

            assert response.status_code == 202
            data = response.json()
            assert data["gate_id"] == 0
            assert data["status"] == "resuming_from_gate_0"
            assert data["next_node"] == "resuming"

            # Reviews were written to state before the background task fired.
            mock_graph.aupdate_state.assert_called_once()
            call_args = mock_graph.aupdate_state.call_args
            reviews = call_args[0][1]["gate0_reviews"]
            assert len(reviews) == 2
            assert reviews[0]["entity_id"] == "ent-001"
            assert reviews[0]["action"] == "approve"

            # Queue was flipped to resuming_from_gate_0 synchronously so the
            # next 5s poll tick hides the "pending review" card affordance.
            mock_qs.update_status.assert_any_await(
                mock_qs.update_status.call_args.args[0],
                uuid.UUID(sample_pipeline_state["source_id"]),
                "resuming_from_gate_0",
            )

            app.dependency_overrides.clear()

    def test_submit_gate0_wrong_status(self):
        """POST /gates/{thread}/0/submit returns 409 when not at gate_0."""
        state_snapshot = MagicMock()
        state_snapshot.values = {"status": "chunking", "current_node": "chunk_behaviors"}

        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=state_snapshot)

        from app.main import app
        from app.api.dependencies import get_graph
        app.dependency_overrides[get_graph] = lambda: mock_graph

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post("/api/gates/00000000-0000-4000-8000-000000000001/0/submit", json={
            "reviews": [{"entity_id": "ent-001", "action": "approve"}],
        })

        assert response.status_code == 409

        app.dependency_overrides.clear()

    def test_submit_gate0_stale_checkpoint(self, sample_pipeline_state):
        """POST /gates/{thread}/0/submit returns 409 when the client's
        checkpoint_id is stale.

        This protects against the C8 double-submit race: a second submit
        arriving with the original checkpoint_id sees that the checkpointer
        has advanced and rejects without clobbering the first submit's
        review payload.
        """
        pre_state = MagicMock()
        pre_state.values = sample_pipeline_state
        # Simulate the checkpoint having advanced — current is a NEW id, the
        # client's body forwards an OLDER id from its GET /pending call.
        pre_state.config = {"configurable": {"checkpoint_id": "cp-new-002"}}

        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=pre_state)
        mock_graph.aupdate_state = AsyncMock()

        from app.main import app
        from app.api.dependencies import get_graph
        app.dependency_overrides[get_graph] = lambda: mock_graph

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/gates/00000000-0000-4000-8000-000000000001/0/submit",
            json={
                "reviews": [{"entity_id": "ent-001", "action": "approve"}],
                "checkpoint_id": "cp-stale-001",
            },
        )

        assert response.status_code == 409
        assert "stale" in response.json()["detail"].lower()
        # Stale submit must NOT advance state.
        mock_graph.aupdate_state.assert_not_called()

        app.dependency_overrides.clear()

    def test_submit_gate0_matching_checkpoint_accepted(self, sample_pipeline_state):
        """POST /gates/{thread}/0/submit accepts when checkpoint_id matches."""
        pre_state = MagicMock()
        pre_state.values = sample_pipeline_state
        pre_state.config = {"configurable": {"checkpoint_id": "cp-fresh-001"}}

        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=pre_state)
        mock_graph.aupdate_state = AsyncMock()

        async def fake_stream(*args, **kwargs):
            yield {"chunk_behaviors": {"status": "chunking"}}
        mock_graph.astream = fake_stream

        with patch("app.api.routes.gates.queue_service") as mock_qs, \
             patch("app.api.routes.pipeline.queue_service") as mock_qs_pipeline:
            mock_qs.update_status = AsyncMock()
            mock_qs_pipeline.update_status = AsyncMock()

            from app.main import app
            from app.api.dependencies import get_graph
            app.dependency_overrides[get_graph] = lambda: mock_graph

            client = TestClient(app, raise_server_exceptions=False)
            response = client.post(
                "/api/gates/00000000-0000-4000-8000-000000000001/0/submit",
                json={
                    "reviews": [{"entity_id": "ent-001", "action": "approve"}],
                    "checkpoint_id": "cp-fresh-001",
                },
            )

            assert response.status_code == 202
            mock_graph.aupdate_state.assert_called_once()

            app.dependency_overrides.clear()

    def test_submit_gate0_no_checkpoint_id_back_compat(self, sample_pipeline_state):
        """Submits without checkpoint_id still work (back-compat for non-UI clients)."""
        pre_state = MagicMock()
        pre_state.values = sample_pipeline_state
        pre_state.config = {"configurable": {"checkpoint_id": "cp-current-001"}}

        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=pre_state)
        mock_graph.aupdate_state = AsyncMock()

        async def fake_stream(*args, **kwargs):
            yield {"chunk_behaviors": {"status": "chunking"}}
        mock_graph.astream = fake_stream

        with patch("app.api.routes.gates.queue_service") as mock_qs, \
             patch("app.api.routes.pipeline.queue_service") as mock_qs_pipeline:
            mock_qs.update_status = AsyncMock()
            mock_qs_pipeline.update_status = AsyncMock()

            from app.main import app
            from app.api.dependencies import get_graph
            app.dependency_overrides[get_graph] = lambda: mock_graph

            client = TestClient(app, raise_server_exceptions=False)
            # No checkpoint_id field in body — opt-out of the race guard.
            response = client.post(
                "/api/gates/00000000-0000-4000-8000-000000000001/0/submit",
                json={"reviews": [{"entity_id": "ent-001", "action": "approve"}]},
            )

            assert response.status_code == 202
            mock_graph.aupdate_state.assert_called_once()

            app.dependency_overrides.clear()

    def test_submit_gate1_with_rejection(self):
        """POST /gates/{thread}/1/submit with rejection routing.

        Fire-and-forget: endpoint returns 202 with resuming_from_gate_1.
        The resume (and its rejection routing) runs as a background task.
        """
        pre_state = MagicMock()
        pre_state.values = {
            # Valid UUID required by uuid.UUID(source_id) in the endpoint.
            "source_id": "00000000-0000-4000-8000-0000000000bb",
            "status": "gate_1",
            "current_node": "gate_1",
            "drafts": [
                {"draft_id": "dft-001"},
                {"draft_id": "dft-002"},
            ],
        }

        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=pre_state)
        mock_graph.aupdate_state = AsyncMock()

        async def fake_stream(*args, **kwargs):
            yield {"extract_techniques": {"status": "extracting_techniques"}}
        mock_graph.astream = fake_stream

        with patch("app.api.routes.gates.queue_service") as mock_qs, \
             patch("app.api.routes.pipeline.queue_service") as mock_qs_pipeline:
            mock_qs.update_status = AsyncMock()
            mock_qs_pipeline.update_status = AsyncMock()

            from app.main import app
            from app.api.dependencies import get_graph
            app.dependency_overrides[get_graph] = lambda: mock_graph

            client = TestClient(app, raise_server_exceptions=False)
            response = client.post(
                "/api/gates/00000000-0000-4000-8000-000000000001/1/submit",
                json={
                    "reviews": [
                        {"draft_id": "dft-001", "action": "approve"},
                        {
                            "draft_id": "dft-002",
                            "action": "reject",
                            "reject_reason": "wrong_technique",
                            "rationale": "Should be T1059.003 not T1059.001",
                        },
                    ],
                },
            )

            assert response.status_code == 202
            data = response.json()
            assert data["gate_id"] == 1
            assert data["status"] == "resuming_from_gate_1"
            assert data["next_node"] == "resuming"

            app.dependency_overrides.clear()

    def test_submit_gate2_approve(self):
        """POST /gates/{thread}/2/submit batch approve.

        Fire-and-forget: endpoint returns 202 with resuming_from_gate_2.
        """
        pre_state = MagicMock()
        pre_state.values = {
            "source_id": "00000000-0000-4000-8000-0000000000cc",
            "status": "gate_2",
            "current_node": "gate_2",
        }

        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=pre_state)
        mock_graph.aupdate_state = AsyncMock()

        async def fake_stream(*args, **kwargs):
            yield {"serialize_stix": {"status": "serializing"}}
        mock_graph.astream = fake_stream

        with patch("app.api.routes.gates.queue_service") as mock_qs, \
             patch("app.api.routes.pipeline.queue_service") as mock_qs_pipeline:
            mock_qs.update_status = AsyncMock()
            mock_qs_pipeline.update_status = AsyncMock()

            from app.main import app
            from app.api.dependencies import get_graph
            app.dependency_overrides[get_graph] = lambda: mock_graph

            client = TestClient(app, raise_server_exceptions=False)
            response = client.post(
                "/api/gates/00000000-0000-4000-8000-000000000001/2/submit",
                json={"approved": True, "feedback": None},
            )

            assert response.status_code == 202
            data = response.json()
            assert data["gate_id"] == 2
            assert data["status"] == "resuming_from_gate_2"
            assert data["next_node"] == "resuming"

            # Verify gate2_review was written correctly
            call_args = mock_graph.aupdate_state.call_args
            review = call_args[0][1]["gate2_review"]
            assert review["approved"] is True

            app.dependency_overrides.clear()

    def test_submit_gate2_reject_with_feedback(self):
        """POST /gates/{thread}/2/submit with rejection routes back to normalize."""
        pre_state = MagicMock()
        pre_state.values = {
            "source_id": "00000000-0000-4000-8000-0000000000dd",
            "status": "gate_2",
            "current_node": "gate_2",
        }

        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=pre_state)
        mock_graph.aupdate_state = AsyncMock()

        async def fake_stream(*args, **kwargs):
            yield {"normalize": {"status": "normalizing"}}
        mock_graph.astream = fake_stream

        with patch("app.api.routes.gates.queue_service") as mock_qs, \
             patch("app.api.routes.pipeline.queue_service") as mock_qs_pipeline:
            mock_qs.update_status = AsyncMock()
            mock_qs_pipeline.update_status = AsyncMock()

            from app.main import app
            from app.api.dependencies import get_graph
            app.dependency_overrides[get_graph] = lambda: mock_graph

            client = TestClient(app, raise_server_exceptions=False)
            response = client.post(
                "/api/gates/00000000-0000-4000-8000-000000000001/2/submit",
                json={
                    "approved": False,
                    "feedback": "Missing relationship between LockBit and ActiveMQ exploit",
                },
            )

            assert response.status_code == 202
            data = response.json()
            assert data["status"] == "resuming_from_gate_2"

            app.dependency_overrides.clear()

    # ── Chunk-review gate routes ──────────────────────────────────────

    def _make_chunks_state(self):
        return {
            "source_id": "00000000-0000-4000-8000-0000000000aa",
            "status": "gate_chunks",
            "current_node": "gate_chunks",
            "gates_enabled": {"entities": True, "chunks": True, "procedures": True, "bundle": True},
            "parsed_text": "APT29 ran certutil to download a payload. Then it executed the payload.",
            "validated_entities": [],
            "classified_sections": [],
            "chunks": [
                {
                    "chunk_id": "ch-1", "text": "Downloaded payload via certutil",
                    "source_excerpt": "APT29 ran certutil to download a payload.",
                    "source_span": [0, 41], "sequence_index": 1,
                    "predecessor_indices": [], "precedes_ids": ["ch-2"],
                    "branch_point": False, "convergence_point": False,
                    "behavioral_confidence": 0.9, "context": {}, "source_location": {},
                },
                {
                    "chunk_id": "ch-2", "text": "Executed payload",
                    "source_excerpt": "Then it executed the payload.",
                    "source_span": [42, 71], "sequence_index": 2,
                    "predecessor_indices": [1], "precedes_ids": [],
                    "branch_point": False, "convergence_point": False,
                    "behavioral_confidence": 0.85, "context": {}, "source_location": {},
                },
            ],
        }

    def test_get_pending_chunks(self):
        state_snapshot = MagicMock()
        state_snapshot.values = self._make_chunks_state()

        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=state_snapshot)

        from app.main import app
        from app.api.dependencies import get_graph
        app.dependency_overrides[get_graph] = lambda: mock_graph

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/gates/00000000-0000-4000-8000-000000000001/chunks/pending")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "gate_chunks"
        assert len(data["chunks"]) == 2
        assert data["chunks"][0]["source_excerpt"].startswith("APT29")
        assert data["parsed_text"].startswith("APT29")

        app.dependency_overrides.clear()

    def test_get_pending_chunks_wrong_status(self):
        state_snapshot = MagicMock()
        state_snapshot.values = {"status": "gate_0", "current_node": "gate_0"}

        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=state_snapshot)

        from app.main import app
        from app.api.dependencies import get_graph
        app.dependency_overrides[get_graph] = lambda: mock_graph

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/gates/00000000-0000-4000-8000-000000000001/chunks/pending")

        assert response.status_code == 409
        assert "gate_chunks" in response.json()["detail"]

        app.dependency_overrides.clear()

    def test_submit_chunks_approve_with_edits(self):
        pre_state = MagicMock()
        pre_state.values = self._make_chunks_state()

        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=pre_state)
        mock_graph.aupdate_state = AsyncMock()

        async def fake_stream(*args, **kwargs):
            yield {"extract_techniques": {"status": "extracting_techniques"}}
        mock_graph.astream = fake_stream

        with patch("app.api.routes.gates.queue_service") as mock_qs, \
             patch("app.api.routes.pipeline.queue_service") as mock_qs_pipeline:
            mock_qs.update_status = AsyncMock()
            mock_qs_pipeline.update_status = AsyncMock()

            from app.main import app
            from app.api.dependencies import get_graph
            app.dependency_overrides[get_graph] = lambda: mock_graph

            client = TestClient(app, raise_server_exceptions=False)
            response = client.post(
                "/api/gates/00000000-0000-4000-8000-000000000001/chunks/submit",
                json={
                    "decisions": [
                        {"chunk_id": "ch-1", "action": "approve"},
                        {"chunk_id": "ch-2", "action": "edit",
                         "edits": {"text": "Executed dropped binary",
                                   "behavioral_confidence": 0.95,
                                   "chunk_id": "should-be-stripped"}},
                    ],
                    "edges": [{"action": "remove", "from": "ch-1", "to": "ch-2"}],
                },
            )

            assert response.status_code == 202
            data = response.json()
            assert data["status"] == "resuming_from_gate_chunks"
            assert data["next_node"] == "resuming"

            mock_graph.aupdate_state.assert_called_once()
            payload = mock_graph.aupdate_state.call_args[0][1]["chunk_reviews"]
            assert payload["decisions"][1]["action"] == "edit"
            # chunk_id stripped by the edits validator (whitelist-only).
            assert "chunk_id" not in payload["decisions"][1]["edits"]
            # behavioral_confidence (whitelisted) survives.
            assert payload["decisions"][1]["edits"]["behavioral_confidence"] == 0.95
            # Wire format uses JSON alias "from"; the route serializes with
            # by_alias=False so the dict checkpointed into state uses the
            # Python-safe field name "from_" instead of the reserved keyword
            # (see route comment). The gate processor reads "from_".
            assert payload["edges"][0]["from_"] == "ch-1"
            assert payload["edges"][0]["to"] == "ch-2"

            app.dependency_overrides.clear()

    def test_submit_chunks_reject(self):
        pre_state = MagicMock()
        pre_state.values = self._make_chunks_state()

        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=pre_state)
        mock_graph.aupdate_state = AsyncMock()

        async def fake_stream(*args, **kwargs):
            yield {"chunk_behaviors": {"status": "chunking"}}
        mock_graph.astream = fake_stream

        with patch("app.api.routes.gates.queue_service") as mock_qs, \
             patch("app.api.routes.pipeline.queue_service") as mock_qs_pipeline:
            mock_qs.update_status = AsyncMock()
            mock_qs_pipeline.update_status = AsyncMock()

            from app.main import app
            from app.api.dependencies import get_graph
            app.dependency_overrides[get_graph] = lambda: mock_graph

            client = TestClient(app, raise_server_exceptions=False)
            response = client.post(
                "/api/gates/00000000-0000-4000-8000-000000000001/chunks/submit",
                json={
                    "reject": {
                        "reason": "missed_procedures",
                        "comments": "Source mentions VSS deletion that was omitted",
                    },
                },
            )

            assert response.status_code == 202
            data = response.json()
            assert data["status"] == "resuming_from_gate_chunks"

            payload = mock_graph.aupdate_state.call_args[0][1]["chunk_reviews"]
            assert payload["reject"]["reason"] == "missed_procedures"
            assert "VSS deletion" in payload["reject"]["comments"]

            app.dependency_overrides.clear()

    def test_submit_chunks_wrong_status(self):
        state_snapshot = MagicMock()
        state_snapshot.values = {"status": "gate_0", "current_node": "gate_0"}

        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=state_snapshot)

        from app.main import app
        from app.api.dependencies import get_graph
        app.dependency_overrides[get_graph] = lambda: mock_graph

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/gates/00000000-0000-4000-8000-000000000001/chunks/submit",
            json={"decisions": []},
        )

        assert response.status_code == 409
        app.dependency_overrides.clear()

    def test_submit_chunks_invalid_reject_reason(self):
        pre_state = MagicMock()
        pre_state.values = self._make_chunks_state()

        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=pre_state)

        from app.main import app
        from app.api.dependencies import get_graph
        app.dependency_overrides[get_graph] = lambda: mock_graph

        client = TestClient(app, raise_server_exceptions=False)
        response = client.post(
            "/api/gates/00000000-0000-4000-8000-000000000001/chunks/submit",
            json={"reject": {"reason": "nonsense_reason", "comments": ""}},
        )
        # Pydantic Literal validation rejects with 422.
        assert response.status_code == 422
        app.dependency_overrides.clear()


# =============================================================================
# Pipeline Tests
# =============================================================================

class TestDisplayedStatus:
    """_displayed_status derives the Kanban status from the UPCOMING node so a
    long-running node shows its own status, not its predecessor's."""

    def _fn(self):
        from app.api.routes.pipeline import _displayed_status
        return _displayed_status

    def test_upcoming_long_node_wins_over_self(self):
        # After parse completes (self still 'parsing'), extract_figures is next
        # -> show 'extracting_figures' while it runs, not 'parsing'.
        assert self._fn()("parsing", "extract_figures") == "extracting_figures"

    def test_figures_to_entities(self):
        assert self._fn()("extracting_figures", "extract_entities") == "extracting_entities"

    def test_resuming_status_is_preserved(self):
        # A gate's transient resume status must survive even though the next
        # node (normalize) is a known long-running stage — it intentionally
        # keeps the card in the gate column while downstream restarts.
        assert self._fn()("resuming_from_gate_1", "normalize") == "resuming_from_gate_1"

    def test_unknown_next_node_falls_back_to_self(self):
        # Upcoming node not in the map (a gate) -> keep the node's
        # self-reported status.
        assert self._fn()("extracting_entities", "gate_0") == "extracting_entities"

    def test_no_next_node_falls_back_to_self(self):
        # End of run: state.next is empty -> next_node None -> self status.
        assert self._fn()("distributing", None) == "distributing"

    def test_synthesize_feedback_shows_while_it_runs(self):
        # distribute writes COMPLETED but synthesize_feedback (an LLM call)
        # still has to run. The card must not read "Complete" during it.
        assert self._fn()("completed", "synthesize_feedback") == "synthesizing_feedback"

    def test_completed_wins_once_synthesis_finishes(self):
        # synthesize_feedback is the last node; nothing follows it.
        assert self._fn()("completed", None) == "completed"

    def test_failed_is_never_masked_by_an_upcoming_node(self):
        # The distribute -> synthesize_feedback edge is unconditional, so a
        # run distribute just failed still has a next node. FAILED is
        # terminal and must survive.
        assert self._fn()("failed", "synthesize_feedback") == "failed"


class TestPipelineRoutes:
    """Test /api/pipeline endpoints."""

    def test_start_pipeline(self):
        """POST /api/pipeline/run starts background task and returns 202."""
        source_id = uuid.uuid4()
        now = datetime.now(timezone.utc)

        fake_source = MagicMock()
        fake_source.id = source_id
        fake_source.status = "queued"
        fake_source.channel = "manual"
        fake_source.source_type = "markdown"
        fake_source.raw_content_path = "/data/test.md"
        fake_source.metadata_ = {"author": "Test"}
        fake_source.source_reliability = 75
        fake_source.gates_enabled = True
        fake_source.gate_modes = "review"

        with patch("app.api.routes.pipeline.queue_service") as mock_qs:
            mock_qs.get_source = AsyncMock(return_value=fake_source)
            mock_qs.set_thread_id = AsyncMock(return_value=fake_source)
            mock_qs.clear_error = AsyncMock(return_value=fake_source)
            mock_qs.update_status = AsyncMock(return_value=fake_source)

            mock_graph = AsyncMock()

            from app.main import app
            from app.api.dependencies import get_db, get_graph
            app.dependency_overrides[get_db] = lambda: AsyncMock()
            app.dependency_overrides[get_graph] = lambda: mock_graph

            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/api/pipeline/run", json={
                "source_id": str(source_id),
            })

            assert response.status_code == 202
            data = response.json()
            assert data["thread_id"] == str(source_id)
            assert data["status"] == "parsing"
            # clear_error must run before update_status so a retry doesn't
            # display a stale red badge alongside an in-progress status.
            mock_qs.clear_error.assert_awaited_once()
            assert mock_qs.clear_error.await_args.args[1] == source_id

            app.dependency_overrides.clear()

    def test_start_pipeline_resets_stale_correction_state(self):
        """initial_state must null the per-run correction fields.

        thread_id == source.id is reused across runs, and LangGraph merges
        the input over the prior checkpoint — any field omitted here keeps
        its old value. Without these resets, a re-queued source resurrects
        run-1's gate1_correction_log (double-counted by synthesize_feedback,
        shown as 'captured this run') and a stranded technique_rerun_feedback
        injects phantom-chunk guidance into run-2's first extraction pass.
        """
        source_id = uuid.uuid4()

        fake_source = MagicMock()
        fake_source.id = source_id
        fake_source.status = "queued"
        fake_source.channel = "manual"
        fake_source.source_type = "markdown"
        fake_source.raw_content_path = "/data/test.md"
        fake_source.metadata_ = {}
        fake_source.source_reliability = 75
        fake_source.gates_enabled = True
        fake_source.gate_modes = "review"

        with patch("app.api.routes.pipeline.queue_service") as mock_qs, \
             patch("app.api.routes.pipeline._run_pipeline") as mock_run, \
             patch("app.api.routes.pipeline.launch_pipeline_task") as mock_launch:
            mock_qs.get_source = AsyncMock(return_value=fake_source)
            mock_qs.set_thread_id = AsyncMock(return_value=fake_source)
            mock_qs.clear_error = AsyncMock(return_value=fake_source)
            mock_qs.update_status = AsyncMock(return_value=fake_source)
            mock_run.return_value = MagicMock()  # never awaited; launch is mocked

            from app.main import app
            from app.api.dependencies import get_db, get_graph
            app.dependency_overrides[get_db] = lambda: AsyncMock()
            app.dependency_overrides[get_graph] = lambda: AsyncMock()

            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/api/pipeline/run", json={
                "source_id": str(source_id),
            })

            assert response.status_code == 202
            mock_launch.assert_called_once()
            initial_state = mock_run.call_args.args[2]
            assert initial_state["error"] is None
            assert initial_state["bundle_corrections"] == []
            assert initial_state["gate1_correction_log"] == []
            assert initial_state["technique_rerun_feedback"] is None

            app.dependency_overrides.clear()

    def test_start_pipeline_source_not_found(self):
        """POST /api/pipeline/run returns 404 for missing source."""
        with patch("app.api.routes.pipeline.queue_service") as mock_qs:
            mock_qs.get_source = AsyncMock(return_value=None)

            mock_graph = AsyncMock()

            from app.main import app
            from app.api.dependencies import get_db, get_graph
            app.dependency_overrides[get_db] = lambda: AsyncMock()
            app.dependency_overrides[get_graph] = lambda: mock_graph

            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/api/pipeline/run", json={
                "source_id": str(uuid.uuid4()),
            })

            assert response.status_code == 404

            app.dependency_overrides.clear()

    def test_start_pipeline_rejects_source_parked_at_a_gate(self):
        """A source mid-review must not be restarted from the top.

        Restarting would discard analyst decisions already applied to its
        checkpoint, so only 'queued' and 'failed' are restartable.
        """
        fake_source = MagicMock()
        fake_source.status = "gate_chunks"

        with patch("app.api.routes.pipeline.queue_service") as mock_qs:
            mock_qs.get_source = AsyncMock(return_value=fake_source)

            from app.main import app
            from app.api.dependencies import get_db, get_graph
            app.dependency_overrides[get_db] = lambda: AsyncMock()
            app.dependency_overrides[get_graph] = lambda: AsyncMock()

            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/api/pipeline/run", json={
                "source_id": str(uuid.uuid4()),
            })

            assert response.status_code == 409
            assert "gate_chunks" in response.json()["detail"]

            app.dependency_overrides.clear()

    def test_start_pipeline_accepts_failed_source(self):
        """The Retry button on a Failed card posts here — it must not 409.

        Regression: the guard admitted only 'queued', so every card in the
        Failed column offered a Retry button that returned
        "Source is 'failed', must be 'queued' to start a run."
        """
        fake_source = MagicMock()
        fake_source.status = "failed"
        fake_source.id = uuid.uuid4()
        fake_source.gates_enabled = {
            "entities": True, "chunks": True,
            "procedures": True, "bundle": True,
        }
        fake_source.gate_modes = "review"

        with patch("app.api.routes.pipeline.queue_service") as mock_qs:
            mock_qs.get_source = AsyncMock(return_value=fake_source)
            mock_qs.set_thread_id = AsyncMock()
            mock_qs.clear_error = AsyncMock()
            mock_qs.update_status = AsyncMock()

            from app.main import app
            from app.api.dependencies import get_db, get_graph
            app.dependency_overrides[get_db] = lambda: AsyncMock()
            app.dependency_overrides[get_graph] = lambda: AsyncMock()

            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/api/pipeline/run", json={
                "source_id": str(fake_source.id),
            })

            assert response.status_code == 202
            # The stale error must be wiped so the retry doesn't render the
            # previous failure's red badge alongside an in-progress status.
            mock_qs.clear_error.assert_awaited_once()

            app.dependency_overrides.clear()

    def test_start_pipeline_wrong_status(self):
        """POST /api/pipeline/run returns 409 if source not queued."""
        fake_source = MagicMock()
        fake_source.status = "gate_0"

        with patch("app.api.routes.pipeline.queue_service") as mock_qs:
            mock_qs.get_source = AsyncMock(return_value=fake_source)

            mock_graph = AsyncMock()

            from app.main import app
            from app.api.dependencies import get_db, get_graph
            app.dependency_overrides[get_db] = lambda: AsyncMock()
            app.dependency_overrides[get_graph] = lambda: mock_graph

            client = TestClient(app, raise_server_exceptions=False)
            response = client.post("/api/pipeline/run", json={
                "source_id": str(uuid.uuid4()),
            })

            assert response.status_code == 409
            assert "must be 'queued'" in response.json()["detail"]

            app.dependency_overrides.clear()

    async def test_clear_error_resets_error_field(self):
        """clear_error sets source.error back to None when a prior run left a message."""
        from app.services import queue as queue_service

        source_id = uuid.uuid4()
        fake_source = MagicMock()
        fake_source.id = source_id
        fake_source.error = "Parser could not decode file"

        fake_db = AsyncMock()

        with patch.object(queue_service, "get_source", AsyncMock(return_value=fake_source)):
            returned = await queue_service.clear_error(fake_db, source_id)

        assert returned is fake_source
        assert fake_source.error is None
        fake_db.commit.assert_awaited()
        fake_db.refresh.assert_awaited()

    async def test_clear_error_noop_when_already_clean(self):
        """clear_error skips the commit when source.error is already None."""
        from app.services import queue as queue_service

        source_id = uuid.uuid4()
        fake_source = MagicMock()
        fake_source.id = source_id
        fake_source.error = None
        # clear_error also touches bundle_corrections; the test fixture
        # must set it falsy explicitly, otherwise MagicMock's auto-attr
        # returns a truthy MagicMock and the dirty branch fires.
        fake_source.bundle_corrections = []
        # Same for the run counts clear_error resets on a re-run.
        for field in queue_service.RUN_COUNT_FIELDS:
            setattr(fake_source, field, None)

        fake_db = AsyncMock()

        with patch.object(queue_service, "get_source", AsyncMock(return_value=fake_source)):
            returned = await queue_service.clear_error(fake_db, source_id)

        assert returned is fake_source
        assert fake_source.error is None
        # No DB write should happen when there's nothing to clear.
        fake_db.commit.assert_not_awaited()

    async def test_clear_error_source_missing(self):
        """clear_error returns None when no source row exists."""
        from app.services import queue as queue_service

        fake_db = AsyncMock()

        with patch.object(queue_service, "get_source", AsyncMock(return_value=None)):
            returned = await queue_service.clear_error(fake_db, uuid.uuid4())

        assert returned is None
        fake_db.commit.assert_not_awaited()

    async def test_run_pipeline_forwards_state_error_to_update_status(self):
        """Astream loop must forward state['error'] to queue_service.update_status.

        Regression guard: deterministic nodes (parse,
        distribute, etc.) set state['error'] without raising. Before the
        fix, the outer try/except caught nothing and Source.error stayed
        NULL even though status flipped to 'failed'. This test verifies
        the astream-loop bridge.
        """
        from app.api.routes import pipeline as pipeline_route

        failing_state = MagicMock()
        failing_state.values = {
            "status": "failed",
            "error": "Source file not found: /data/missing.md",
            "persistence_errors": None,
        }
        failing_state.next = None  # astream ended, no pending gate

        async def fake_astream(initial_state, config):
            # A single yielded event is enough to enter the update branch.
            yield {"parse_and_validate": {"status": "failed"}}

        mock_graph = MagicMock()
        mock_graph.astream = fake_astream
        mock_graph.aget_state = AsyncMock(return_value=failing_state)

        captured: list[dict] = []

        async def fake_update_status(
            db, source_id, status,
            error=None, persistence_errors=None, bundle_corrections=None,
            run_counts=None,
        ):
            captured.append({
                "source_id": source_id,
                "status": status,
                "error": error,
                "persistence_errors": persistence_errors,
                "bundle_corrections": bundle_corrections,
            })
            return None

        # async_session() is used as an async context manager; patch it to
        # return a trivial AsyncMock session.
        fake_session_cm = AsyncMock()
        fake_session_cm.__aenter__ = AsyncMock(return_value=AsyncMock())
        fake_session_cm.__aexit__ = AsyncMock(return_value=None)

        source_id = uuid.uuid4()

        with patch.object(pipeline_route.queue_service, "update_status", fake_update_status), \
             patch.object(pipeline_route, "async_session", return_value=fake_session_cm), \
             patch.object(pipeline_route.ws_manager, "has_subscribers", return_value=False):
            await pipeline_route._run_pipeline(mock_graph, str(source_id), {}, source_id)

        # At least one update_status call must have carried the error string.
        error_calls = [c for c in captured if c["error"] == "Source file not found: /data/missing.md"]
        assert len(error_calls) >= 1, f"Expected error to be forwarded, got: {captured}"
        assert error_calls[0]["status"] == "failed"
        assert error_calls[0]["source_id"] == source_id

    def test_get_pipeline_status(self, sample_pipeline_state):
        """GET /api/pipeline/status/{thread_id} returns current state."""
        state_snapshot = MagicMock()
        state_snapshot.values = sample_pipeline_state

        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=state_snapshot)

        from app.main import app
        from app.api.dependencies import get_graph
        app.dependency_overrides[get_graph] = lambda: mock_graph

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/pipeline/status/00000000-0000-4000-8000-000000000001")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "gate_0"
        assert data["current_node"] == "gate_0"
        assert data["entity_count"] == 2
        assert data["gates_enabled"] == {"entities": True, "chunks": True, "procedures": True, "bundle": True}

        app.dependency_overrides.clear()

    def test_get_pipeline_status_not_found(self):
        """GET /api/pipeline/status/{thread_id} returns 404 when no state."""
        state_snapshot = MagicMock()
        state_snapshot.values = None

        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=state_snapshot)

        from app.main import app
        from app.api.dependencies import get_graph
        app.dependency_overrides[get_graph] = lambda: mock_graph

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/pipeline/status/00000000-0000-4000-8000-000000000099")

        assert response.status_code == 404

        app.dependency_overrides.clear()


# =============================================================================
# Pydantic Schema Validation Tests
# =============================================================================

class TestSchemaValidation:
    """Test Pydantic request model validation."""

    def test_gate0_review_item_valid(self):
        """Gate0ReviewItem accepts valid review."""
        from app.schemas.api import Gate0ReviewItem
        item = Gate0ReviewItem(
            entity_id="ent-001",
            action="approve",
        )
        assert item.action == "approve"
        assert item.edited_value is None

    def test_gate2_rel_type_matches_the_bundle_spelling(self):
        """The serializer emits `has-observable`; the whitelist must agree.

        For a while the whitelist (and both reviewer copies of it) spelled
        it `has_observable`. The canvas offered the bundle's spelling, so
        picking it 422'd the whole submission; the reviewer's spelling would
        have serialized an SRO type distribution.py cannot map.
        """
        from pydantic import ValidationError
        from app.schemas.api import Gate2ReviewItem
        ok = Gate2ReviewItem(rel_id="rel-1", action="edit", edited_rel_type="has-observable")
        assert ok.edited_rel_type == "has-observable"
        with pytest.raises(ValidationError, match="Invalid relationship type"):
            Gate2ReviewItem(rel_id="rel-1", action="edit", edited_rel_type="has_observable")

    def test_gate0_review_item_with_edit(self):
        """Gate0ReviewItem accepts edit with corrected value."""
        from app.schemas.api import Gate0ReviewItem
        item = Gate0ReviewItem(
            entity_id="ent-001",
            action="edit",
            edited_value="APT29",
            edited_type="intrusion_set",
            rationale="Corrected from LockBit to APT29",
        )
        assert item.edited_value == "APT29"
        assert item.edited_type == "intrusion_set"

    def test_gate0_review_item_accepts_edited_role(self):
        """Gate0ReviewItem.edited_role lets the analyst correct
        organization_role / location_role at gate review."""
        from app.schemas.api import Gate0ReviewItem
        item = Gate0ReviewItem(
            entity_id="ent-org",
            action="edit",
            edited_type="organization",
            edited_role="author",
        )
        assert item.edited_role == "author"

    def test_gate0_review_item_default_edited_role_none(self):
        """edited_role defaults to None — existing callers continue to
        work without the new field."""
        from app.schemas.api import Gate0ReviewItem
        item = Gate0ReviewItem(entity_id="ent-001", action="approve")
        assert item.edited_role is None

    def test_gate1_review_item_with_rejection(self):
        """Gate1ReviewItem accepts rejection with reason."""
        from app.schemas.api import Gate1ReviewItem
        item = Gate1ReviewItem(
            draft_id="dft-001",
            action="reject",
            reject_reason="wrong_technique",
            rationale="Should map to T1059.003",
        )
        assert item.reject_reason == "wrong_technique"

    def test_gate1_review_item_with_edits(self):
        """Gate1ReviewItem accepts edit with field corrections."""
        from app.schemas.api import Gate1ReviewItem
        item = Gate1ReviewItem(
            draft_id="dft-001",
            action="edit",
            analyst_edits={
                "name": "Exploit ActiveMQ via ClassInfo deserialization",
                "confidence": 92,
            },
        )
        assert item.analyst_edits["confidence"] == 92

    def test_gate1_submit_default_promotions_empty(self):
        """Gate1Submit's promotions field defaults to []. Existing callers
        that send only `reviews` continue to work without code changes."""
        from app.schemas.api import Gate1Submit
        body = Gate1Submit(reviews=[])
        assert body.promotions == []

    def test_gate1_submit_accepts_promotions(self):
        """Gate1Submit accepts a promotions list of {chunk_id, technique_id}."""
        from app.schemas.api import Gate1PromotionItem, Gate1Submit
        body = Gate1Submit(
            reviews=[],
            promotions=[
                Gate1PromotionItem(chunk_id="chk-001", technique_id="T1204.004"),
                Gate1PromotionItem(chunk_id="chk-002", technique_id="T1059.001"),
            ],
        )
        assert len(body.promotions) == 2
        assert body.promotions[0].chunk_id == "chk-001"
        assert body.promotions[0].technique_id == "T1204.004"

    def test_gate1_promotion_item_requires_both_fields(self):
        """Gate1PromotionItem rejects missing chunk_id or technique_id."""
        from app.schemas.api import Gate1PromotionItem
        from pydantic import ValidationError
        import pytest as _pt
        with _pt.raises(ValidationError):
            Gate1PromotionItem(chunk_id="chk-001")  # missing technique_id
        with _pt.raises(ValidationError):
            Gate1PromotionItem(technique_id="T1204.004")  # missing chunk_id

    def test_denylist_terms_model_cleans(self):
        """DenylistTermsModel dedups values case-insensitively and validates T-IDs."""
        from app.schemas.api import DenylistTermsModel
        m = DenylistTermsModel(
            values=["  A ", "a", "Bee"],
            technique_ids=["t1059", "T1059.001", "junk", "T1486"],
        )
        assert m.values == ["A", "Bee"]
        assert m.technique_ids == ["T1059", "T1059.001", "T1486"]

    def test_denylist_terms_drops_control_chars(self):
        """Control-char values are silently dropped (cleaning delegates to the
        service's normalize_denylist_terms — single source of truth, drop not
        raise). The char never reaches logs/storage either way."""
        from app.schemas.api import DenylistTermsModel
        m = DenylistTermsModel(values=["bad\x01value", "clean"])
        assert m.values == ["clean"]

    def test_promote_request_denylist_terms_optional(self):
        """Promote request: denylist_terms defaults None (prompt) and parses (denylist)."""
        from app.schemas.api import FeedbackPatternPromoteRequest
        prompt_req = FeedbackPatternPromoteRequest(action="prompt", by="analyst")
        assert prompt_req.denylist_terms is None
        deny_req = FeedbackPatternPromoteRequest(
            action="denylist", by="analyst",
            denylist_terms={"values": ["info@cert.example"], "technique_ids": ["T1204.004"]},
        )
        assert deny_req.denylist_terms.values == ["info@cert.example"]
        assert deny_req.denylist_terms.technique_ids == ["T1204.004"]

    def test_gate2_submit_valid(self):
        """Gate2Submit accepts binary decision."""
        from app.schemas.api import Gate2Submit
        submit = Gate2Submit(approved=True)
        assert submit.approved is True
        assert submit.feedback is None

    def test_gate2_submit_rejection_with_feedback(self):
        """Gate2Submit accepts rejection with feedback."""
        from app.schemas.api import Gate2Submit
        submit = Gate2Submit(
            approved=False,
            feedback="Relationships between malware and infrastructure are incorrect",
        )
        assert submit.approved is False
        assert "infrastructure" in submit.feedback

    def test_source_create_validation(self):
        """SourceCreate validates required fields."""
        from app.schemas.api import SourceCreate
        source = SourceCreate(
            source_type="markdown",
            raw_content_path="/data/test.md",
        )
        assert source.title == "Untitled Source"
        assert source.gates_enabled == {"entities": True, "chunks": True, "procedures": True, "bundle": True}
        assert source.source_reliability == 50
        assert source.sequentiality == "auto"

    def test_source_create_accepts_sequentiality_yes_no_auto(self):
        """SourceCreate accepts the three sequentiality values."""
        from app.schemas.api import SourceCreate
        for val in ("yes", "no", "auto"):
            source = SourceCreate(
                source_type="markdown",
                raw_content_path="/data/test.md",
                sequentiality=val,
            )
            assert source.sequentiality == val

    def test_source_create_rejects_invalid_sequentiality(self):
        """SourceCreate rejects sequentiality values outside the Literal set."""
        from app.schemas.api import SourceCreate
        with pytest.raises(Exception):  # ValidationError
            SourceCreate(
                source_type="markdown",
                raw_content_path="/data/test.md",
                sequentiality="maybe",
            )

    def test_source_create_reliability_bounds(self):
        """SourceCreate enforces 0-100 range on reliability."""
        from app.schemas.api import SourceCreate
        with pytest.raises(Exception):  # ValidationError
            SourceCreate(
                source_type="pdf",
                raw_content_path="/data/test.pdf",
                source_reliability=150,
            )

    def test_pipeline_run_request(self):
        """PipelineRunRequest validates source_id as UUID."""
        from app.schemas.api import PipelineRunRequest
        req = PipelineRunRequest(source_id=uuid.uuid4())
        assert isinstance(req.source_id, uuid.UUID)

    def test_bundle_rename_request_strips_whitespace(self):
        """BundleRenameRequest trims leading/trailing whitespace."""
        from app.schemas.api import BundleRenameRequest
        req = BundleRenameRequest(title="   ActiveMQ → LockBit   ")
        assert req.title == "ActiveMQ → LockBit"

    def test_bundle_rename_request_rejects_empty(self):
        """BundleRenameRequest rejects whitespace-only titles."""
        from app.schemas.api import BundleRenameRequest
        with pytest.raises(Exception):  # ValidationError
            BundleRenameRequest(title="   ")

    def test_bundle_rename_request_rejects_control_chars(self):
        """BundleRenameRequest rejects ASCII control characters (log-injection guard)."""
        from app.schemas.api import BundleRenameRequest
        # Newline is the most common injection vector in log viewers.
        with pytest.raises(Exception):
            BundleRenameRequest(title="Bundle\nINJECTED")
        # 0x7F (DEL) should also be rejected.
        with pytest.raises(Exception):
            BundleRenameRequest(title="Bundle\x7f")
        # Null byte
        with pytest.raises(Exception):
            BundleRenameRequest(title="Bundle\x00null")

    def test_bundle_rename_request_max_length(self):
        """BundleRenameRequest caps title at 512 characters after stripping."""
        from app.schemas.api import BundleRenameRequest
        # 512 is allowed
        ok = BundleRenameRequest(title="a" * 512)
        assert len(ok.title) == 512
        # 513 is not
        with pytest.raises(Exception):
            BundleRenameRequest(title="a" * 513)


# =============================================================================
# Bundle Explorer Tests
# =============================================================================

class TestBundleRoutes:
    """Test /api/bundles endpoints (PATCH rename, DELETE with cascade)."""

    def _fake_bundle(self, *, source_id=None, title="Demo Bundle"):
        """Build a CompletedBundle-shaped MagicMock for route handlers to serialize."""
        bundle_id = uuid.uuid4()
        now = datetime.now(timezone.utc)
        b = MagicMock()
        b.id = bundle_id
        b.source_id = source_id
        b.title = title
        b.object_count = 42
        b.relationship_count = 7
        b.procedure_count = 5
        b.source_file_name = "report.md"
        b.source_file_type = "text/markdown"
        b.persistence_errors = []
        b.metadata_ = {"tlp": "white"}
        b.completed_at = now
        return b

    def test_rename_bundle_updates_title(self):
        """PATCH /api/bundles/{id} returns 200 with updated metadata."""
        source_id = uuid.uuid4()
        updated = self._fake_bundle(source_id=source_id, title="Renamed Bundle")
        updated_id = updated.id

        with patch("app.api.routes.bundles.bundle_store") as mock_store:
            mock_store.rename_bundle = AsyncMock(return_value=updated)

            from app.main import app
            from app.api.dependencies import get_db
            app.dependency_overrides[get_db] = lambda: AsyncMock()

            client = TestClient(app, raise_server_exceptions=False)
            response = client.patch(
                f"/api/bundles/{updated_id}",
                json={"title": "Renamed Bundle"},
            )

            assert response.status_code == 200
            data = response.json()
            assert data["title"] == "Renamed Bundle"
            assert data["id"] == str(updated_id)
            assert data["source_id"] == str(source_id)
            # Schema-stripped title should reach the service layer.
            call_args = mock_store.rename_bundle.await_args
            assert call_args.args[2] == "Renamed Bundle"

            app.dependency_overrides.clear()

    def test_rename_bundle_strips_whitespace_before_service(self):
        """PATCH strips title via the schema validator before calling the service."""
        updated = self._fake_bundle(title="ActiveMQ")

        with patch("app.api.routes.bundles.bundle_store") as mock_store:
            mock_store.rename_bundle = AsyncMock(return_value=updated)

            from app.main import app
            from app.api.dependencies import get_db
            app.dependency_overrides[get_db] = lambda: AsyncMock()

            client = TestClient(app, raise_server_exceptions=False)
            response = client.patch(
                f"/api/bundles/{updated.id}",
                json={"title": "   ActiveMQ   "},
            )

            assert response.status_code == 200
            # Stripped title reaches the service, not the padded version.
            assert mock_store.rename_bundle.await_args.args[2] == "ActiveMQ"

            app.dependency_overrides.clear()

    def test_rename_bundle_not_found(self):
        """PATCH returns 404 when the bundle id doesn't exist."""
        with patch("app.api.routes.bundles.bundle_store") as mock_store:
            mock_store.rename_bundle = AsyncMock(return_value=None)

            from app.main import app
            from app.api.dependencies import get_db
            app.dependency_overrides[get_db] = lambda: AsyncMock()

            client = TestClient(app, raise_server_exceptions=False)
            response = client.patch(
                f"/api/bundles/{uuid.uuid4()}",
                json={"title": "New Title"},
            )

            assert response.status_code == 404
            assert response.json()["detail"] == "Bundle not found"

            app.dependency_overrides.clear()

    def test_rename_bundle_rejects_empty_title(self):
        """PATCH returns 422 for whitespace-only title (schema validation)."""
        from app.main import app
        from app.api.dependencies import get_db
        app.dependency_overrides[get_db] = lambda: AsyncMock()

        client = TestClient(app, raise_server_exceptions=False)
        response = client.patch(
            f"/api/bundles/{uuid.uuid4()}",
            json={"title": "   "},
        )

        assert response.status_code == 422

        app.dependency_overrides.clear()

    def test_rename_bundle_rejects_control_chars(self):
        """PATCH returns 422 when title contains ASCII control chars."""
        from app.main import app
        from app.api.dependencies import get_db
        app.dependency_overrides[get_db] = lambda: AsyncMock()

        client = TestClient(app, raise_server_exceptions=False)
        response = client.patch(
            f"/api/bundles/{uuid.uuid4()}",
            # Newline is the primary log-injection vector.
            json={"title": "Legit Title\nINJECTED FAKE LINE"},
        )

        assert response.status_code == 422

        app.dependency_overrides.clear()

    def test_rename_bundle_rejects_too_long(self):
        """PATCH returns 422 when title exceeds 512 characters."""
        from app.main import app
        from app.api.dependencies import get_db
        app.dependency_overrides[get_db] = lambda: AsyncMock()

        client = TestClient(app, raise_server_exceptions=False)
        response = client.patch(
            f"/api/bundles/{uuid.uuid4()}",
            json={"title": "a" * 513},
        )

        assert response.status_code == 422

        app.dependency_overrides.clear()

    def test_rename_bundle_rejects_malformed_uuid(self):
        """PATCH returns 422 when the path id isn't a valid UUID."""
        from app.main import app
        from app.api.dependencies import get_db
        app.dependency_overrides[get_db] = lambda: AsyncMock()

        client = TestClient(app, raise_server_exceptions=False)
        response = client.patch(
            "/api/bundles/not-a-uuid",
            json={"title": "Anything"},
        )

        assert response.status_code == 422

        app.dependency_overrides.clear()

    def test_delete_bundle_cascades_to_source(self):
        """DELETE removes the bundle and cascades to the source queue row."""
        source_id = uuid.uuid4()
        bundle_id = uuid.uuid4()

        # Fake async_session context manager for the cascade delete_source call.
        fake_session_cm = AsyncMock()
        fake_session_cm.__aenter__ = AsyncMock(return_value=AsyncMock())
        fake_session_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("app.api.routes.bundles.bundle_store") as mock_store, \
             patch("app.api.routes.bundles.queue_service") as mock_qs, \
             patch("app.api.routes.bundles.async_session", return_value=fake_session_cm):
            mock_store.delete_bundle = AsyncMock(return_value=(True, source_id))
            mock_qs.delete_source = AsyncMock(return_value=True)

            from app.main import app
            from app.api.dependencies import get_db
            app.dependency_overrides[get_db] = lambda: AsyncMock()

            client = TestClient(app, raise_server_exceptions=False)
            response = client.delete(f"/api/bundles/{bundle_id}")

            assert response.status_code == 204
            mock_store.delete_bundle.assert_awaited_once()
            # Cascade fired with the captured source_id.
            mock_qs.delete_source.assert_awaited_once()
            assert mock_qs.delete_source.await_args.args[1] == source_id

            app.dependency_overrides.clear()

    def test_delete_bundle_skips_cascade_when_source_id_null(self):
        """DELETE doesn't call delete_source when bundle had no linked source."""
        bundle_id = uuid.uuid4()

        with patch("app.api.routes.bundles.bundle_store") as mock_store, \
             patch("app.api.routes.bundles.queue_service") as mock_qs:
            mock_store.delete_bundle = AsyncMock(return_value=(True, None))
            mock_qs.delete_source = AsyncMock(return_value=True)

            from app.main import app
            from app.api.dependencies import get_db
            app.dependency_overrides[get_db] = lambda: AsyncMock()

            client = TestClient(app, raise_server_exceptions=False)
            response = client.delete(f"/api/bundles/{bundle_id}")

            assert response.status_code == 204
            mock_qs.delete_source.assert_not_awaited()

            app.dependency_overrides.clear()

    def test_delete_bundle_not_found(self):
        """DELETE returns 404 when the bundle id doesn't exist."""
        with patch("app.api.routes.bundles.bundle_store") as mock_store, \
             patch("app.api.routes.bundles.queue_service") as mock_qs:
            mock_store.delete_bundle = AsyncMock(return_value=(False, None))
            mock_qs.delete_source = AsyncMock()

            from app.main import app
            from app.api.dependencies import get_db
            app.dependency_overrides[get_db] = lambda: AsyncMock()

            client = TestClient(app, raise_server_exceptions=False)
            response = client.delete(f"/api/bundles/{uuid.uuid4()}")

            assert response.status_code == 404
            assert response.json()["detail"] == "Bundle not found"
            # Cascade must not fire when the bundle wasn't found.
            mock_qs.delete_source.assert_not_awaited()

            app.dependency_overrides.clear()

    def test_delete_bundle_cascade_failure_is_logged_not_raised(self):
        """Cascade delete_source raising must not 500 the DELETE — bundle is already gone."""
        source_id = uuid.uuid4()
        bundle_id = uuid.uuid4()

        fake_session_cm = AsyncMock()
        fake_session_cm.__aenter__ = AsyncMock(return_value=AsyncMock())
        fake_session_cm.__aexit__ = AsyncMock(return_value=None)

        with patch("app.api.routes.bundles.bundle_store") as mock_store, \
             patch("app.api.routes.bundles.queue_service") as mock_qs, \
             patch("app.api.routes.bundles.async_session", return_value=fake_session_cm), \
             patch("app.api.routes.bundles.logger") as mock_logger:
            mock_store.delete_bundle = AsyncMock(return_value=(True, source_id))
            mock_qs.delete_source = AsyncMock(side_effect=RuntimeError("db connection dropped"))

            from app.main import app
            from app.api.dependencies import get_db
            app.dependency_overrides[get_db] = lambda: AsyncMock()

            client = TestClient(app, raise_server_exceptions=False)
            response = client.delete(f"/api/bundles/{bundle_id}")

            # Bundle was removed; DELETE should still return 204.
            assert response.status_code == 204
            # Warning logged so an orphan source row is traceable.
            mock_logger.warning.assert_called_once()
            warning_args = mock_logger.warning.call_args.args
            # The format args carry source_id / bundle_id for grep-ability.
            assert source_id in warning_args
            assert bundle_id in warning_args

            app.dependency_overrides.clear()

    def test_delete_bundle_rejects_malformed_uuid(self):
        """DELETE returns 422 when the path id isn't a valid UUID."""
        from app.main import app
        from app.api.dependencies import get_db
        app.dependency_overrides[get_db] = lambda: AsyncMock()

        client = TestClient(app, raise_server_exceptions=False)
        response = client.delete("/api/bundles/not-a-uuid")

        assert response.status_code == 422

        app.dependency_overrides.clear()

    # ── Service-layer tests (no HTTP round-trip) ───────────────────

    async def test_bundle_store_rename_returns_none_when_missing(self):
        """rename_bundle returns None when no row matches the id (no commit)."""
        from app.services import bundle_store

        fake_db = AsyncMock()

        with patch.object(bundle_store, "get_bundle", AsyncMock(return_value=None)):
            returned = await bundle_store.rename_bundle(
                fake_db, uuid.uuid4(), "New Title"
            )

        assert returned is None
        fake_db.commit.assert_not_awaited()

    async def test_bundle_store_rename_assigns_title_and_commits(self):
        """rename_bundle sets title on the row and commits."""
        from app.services import bundle_store

        fake_bundle = MagicMock()
        fake_bundle.title = "Old Title"

        fake_db = AsyncMock()

        with patch.object(bundle_store, "get_bundle", AsyncMock(return_value=fake_bundle)):
            returned = await bundle_store.rename_bundle(
                fake_db, uuid.uuid4(), "Fresh Title"
            )

        assert returned is fake_bundle
        assert fake_bundle.title == "Fresh Title"
        fake_db.commit.assert_awaited_once()
        fake_db.refresh.assert_awaited_once_with(fake_bundle)

    async def test_bundle_store_delete_returns_source_id_before_delete(self):
        """delete_bundle captures source_id BEFORE db.delete (ORM expires the attr after)."""
        from app.services import bundle_store

        source_id = uuid.uuid4()
        fake_bundle = MagicMock()
        fake_bundle.source_id = source_id

        fake_db = AsyncMock()

        with patch.object(bundle_store, "get_bundle", AsyncMock(return_value=fake_bundle)):
            deleted, returned_source_id = await bundle_store.delete_bundle(
                fake_db, uuid.uuid4()
            )

        assert deleted is True
        assert returned_source_id == source_id
        fake_db.delete.assert_awaited_once_with(fake_bundle)
        fake_db.commit.assert_awaited_once()

    async def test_bundle_store_delete_returns_false_when_missing(self):
        """delete_bundle returns (False, None) when no row matches."""
        from app.services import bundle_store

        fake_db = AsyncMock()

        with patch.object(bundle_store, "get_bundle", AsyncMock(return_value=None)):
            deleted, source_id = await bundle_store.delete_bundle(
                fake_db, uuid.uuid4()
            )

        assert deleted is False
        assert source_id is None
        fake_db.delete.assert_not_awaited()
        fake_db.commit.assert_not_awaited()

    # ── queue_service.delete_source: checkpoint flush ───────────────

    async def test_delete_source_flushes_checkpoint_when_thread_id_set(self):
        """delete_source must call adelete_thread on the checkpointer so
        the LangGraph checkpoint rows don't accumulate after the source
        row is gone. Regression."""
        from app.services import queue as queue_service

        source_id = uuid.uuid4()
        thread_id = uuid.uuid4()
        fake_source = MagicMock()
        fake_source.id = source_id
        fake_source.thread_id = thread_id
        fake_source.raw_content_path = ""  # skip file-delete branch

        fake_db = AsyncMock()
        fake_checkpointer = MagicMock()
        fake_checkpointer.adelete_thread = AsyncMock()

        with patch.object(queue_service, "get_source", AsyncMock(return_value=fake_source)), \
             patch(
                 "app.graph.checkpointer.get_checkpointer",
                 AsyncMock(return_value=fake_checkpointer),
             ):
            deleted = await queue_service.delete_source(fake_db, source_id)

        assert deleted is True
        fake_db.delete.assert_awaited_once_with(fake_source)
        fake_db.commit.assert_awaited()
        fake_checkpointer.adelete_thread.assert_awaited_once_with(str(thread_id))

    async def test_delete_source_skips_checkpoint_when_thread_id_none(self):
        """Sources that never started have no checkpoint to flush; the
        checkpointer must not be touched (avoids a needless PG round-trip
        and a noisy log line)."""
        from app.services import queue as queue_service

        source_id = uuid.uuid4()
        fake_source = MagicMock()
        fake_source.id = source_id
        fake_source.thread_id = None
        fake_source.raw_content_path = ""

        fake_db = AsyncMock()
        fake_checkpointer = MagicMock()
        fake_checkpointer.adelete_thread = AsyncMock()

        with patch.object(queue_service, "get_source", AsyncMock(return_value=fake_source)), \
             patch(
                 "app.graph.checkpointer.get_checkpointer",
                 AsyncMock(return_value=fake_checkpointer),
             ) as mock_get_cp:
            deleted = await queue_service.delete_source(fake_db, source_id)

        assert deleted is True
        mock_get_cp.assert_not_awaited()
        fake_checkpointer.adelete_thread.assert_not_awaited()

    async def test_delete_source_checkpoint_failure_does_not_raise(self):
        """If the checkpointer raises on adelete_thread, the source delete
        must still succeed — the user-visible row is already gone and we'd
        otherwise surface a 5xx after a successful commit."""
        from app.services import queue as queue_service

        source_id = uuid.uuid4()
        thread_id = uuid.uuid4()
        fake_source = MagicMock()
        fake_source.id = source_id
        fake_source.thread_id = thread_id
        fake_source.raw_content_path = ""

        fake_db = AsyncMock()
        fake_checkpointer = MagicMock()
        fake_checkpointer.adelete_thread = AsyncMock(side_effect=RuntimeError("PG down"))

        with patch.object(queue_service, "get_source", AsyncMock(return_value=fake_source)), \
             patch(
                 "app.graph.checkpointer.get_checkpointer",
                 AsyncMock(return_value=fake_checkpointer),
             ):
            deleted = await queue_service.delete_source(fake_db, source_id)

        # Source row delete succeeded; checkpoint flush failure swallowed.
        assert deleted is True
        fake_db.commit.assert_awaited()


# =============================================================================
# Health check
# =============================================================================

class TestHealthCheck:
    """Test health endpoint (no dependencies needed)."""

    def test_health(self):
        """GET /health returns ok."""
        from app.main import app

        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# =============================================================================
# Feedback-pattern management routes (Phase 3)
# =============================================================================

class TestFeedbackPatternRoutes:
    """Test /api/feedback-patterns endpoints (list / promote / dismiss / edit)."""

    def _fake_pattern(self, *, status="active", category="wrong_technique"):
        from app.models.feedback_pattern import FeedbackPattern
        return FeedbackPattern(
            id=uuid.uuid4(),
            category=category,
            pattern="PowerShell over-picked in malware-capability sections",
            status=status,
            occurrence_count=5,
            hit_count=2,
            miss_count=1,
            salience=0.82,
            applies_to={"technique_ids": ["T1059.001"]},
            concepts=["powershell"],
        )

    def _client(self):
        from app.main import app
        from app.api.dependencies import get_db
        app.dependency_overrides[get_db] = lambda: AsyncMock()
        return app, TestClient(app, raise_server_exceptions=False)

    def test_list_patterns(self):
        row = self._fake_pattern()
        with patch("app.services.feedback_patterns.list_patterns",
                   new=AsyncMock(return_value=([row], 1))) as m:
            app, client = self._client()
            resp = client.get("/api/feedback-patterns/?category=wrong_technique&min_salience=0.5")
            assert resp.status_code == 200
            data = resp.json()
            assert data["total"] == 1
            assert data["patterns"][0]["category"] == "wrong_technique"
            assert data["patterns"][0]["salience"] == 0.82
            # filters reach the service
            kw = m.await_args.kwargs
            assert kw["category"] == "wrong_technique"
            assert kw["min_salience"] == 0.5
            app.dependency_overrides.clear()

    def test_captured_skips_malformed_checkpoint(self):
        """One checkpoint that READS fine but PARSES badly must be skipped,
        not 500 the whole list (the docstring's best-effort promise).

        Regression: captured_corrections(snap.values) used to sit outside
        the per-source try, so a mis-shaped field (e.g. a non-dict entry in
        validated_entities from a legacy/seed checkpoint) raised
        AttributeError and killed the captured panel for every source.
        """
        def _fake_src(title, thread_id):
            s = MagicMock()
            s.id = uuid.uuid4()
            s.title = title
            s.status = "gate_1"
            s.thread_id = thread_id
            return s

        bad = _fake_src("malformed checkpoint", "thread-bad")
        good = _fake_src("healthy source", "thread-good")

        snapshots = {
            # Non-dict entity entry → AttributeError inside _entity_deltas.
            "thread-bad": MagicMock(values={"validated_entities": ["just-a-string"]}),
            "thread-good": MagicMock(values={"gate1_correction_log": [{
                "draft_id": "d1", "draft_name": "Deliver lure", "chunk_id": "c1",
                "action": "reject", "reject_reason": "wrong_technique",
                "rationale": "", "rejected_techniques": [],
                "corrected_techniques": [],
            }]}),
        }

        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(
            side_effect=lambda config: snapshots[config["configurable"]["thread_id"]]
        )

        with patch("app.api.routes.feedback_patterns.queue_service") as mock_qs:
            mock_qs.list_sources = AsyncMock(return_value=([bad, good], 2))

            from app.main import app
            from app.api.dependencies import get_db, get_graph
            app.dependency_overrides[get_db] = lambda: AsyncMock()
            app.dependency_overrides[get_graph] = lambda: mock_graph

            client = TestClient(app, raise_server_exceptions=False)
            resp = client.get("/api/feedback-patterns/captured")

            assert resp.status_code == 200
            data = resp.json()
            # Only the healthy source survives; the bad one is skipped.
            assert [s["title"] for s in data["sources"]] == ["healthy source"]
            assert data["total"] == 1

            app.dependency_overrides.clear()

    def test_captured_memoizes_failed_sources(self):
        """A failed source's checkpoint is frozen — repeated polls must serve
        it from the process-local memo instead of re-deserializing the
        checkpoint every 8 seconds forever."""
        from app.api.routes import feedback_patterns as fp_routes
        fp_routes._terminal_capture_cache.clear()

        src = MagicMock()
        src.id = uuid.uuid4()
        src.title = "died mid-run"
        src.status = "failed"
        src.thread_id = "thread-dead"

        snap = MagicMock(values={"gate1_correction_log": [{
            "draft_id": "d1", "draft_name": "n", "chunk_id": "c1",
            "action": "reject", "reject_reason": "wrong_technique",
            "rationale": "", "rejected_techniques": [],
            "corrected_techniques": [],
        }]})
        mock_graph = AsyncMock()
        mock_graph.aget_state = AsyncMock(return_value=snap)

        with patch("app.api.routes.feedback_patterns.queue_service") as mock_qs:
            mock_qs.list_sources = AsyncMock(return_value=([src], 1))

            from app.main import app
            from app.api.dependencies import get_db, get_graph
            app.dependency_overrides[get_db] = lambda: AsyncMock()
            app.dependency_overrides[get_graph] = lambda: mock_graph

            client = TestClient(app, raise_server_exceptions=False)
            first = client.get("/api/feedback-patterns/captured")
            second = client.get("/api/feedback-patterns/captured")

            assert first.status_code == second.status_code == 200
            # Identical payloads, but only ONE checkpoint read.
            assert first.json() == second.json()
            assert first.json()["total"] == 1
            assert mock_graph.aget_state.await_count == 1

            app.dependency_overrides.clear()
            fp_routes._terminal_capture_cache.clear()

    def test_promote_pattern_sets_status(self):
        row = self._fake_pattern(status="promoted_to_prompt")
        with patch("app.services.feedback_patterns.promote_pattern",
                   new=AsyncMock(return_value=row)) as m:
            app, client = self._client()
            resp = client.patch(
                f"/api/feedback-patterns/{row.id}/promote",
                json={"action": "prompt", "by": "analyst"},
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "promoted_to_prompt"
            assert m.await_args.kwargs["action"] == "prompt"
            assert m.await_args.kwargs["by"] == "analyst"
            app.dependency_overrides.clear()

    def test_promote_pattern_404(self):
        with patch("app.services.feedback_patterns.promote_pattern",
                   new=AsyncMock(return_value=None)):
            app, client = self._client()
            resp = client.patch(
                f"/api/feedback-patterns/{uuid.uuid4()}/promote",
                json={"action": "denylist", "by": "analyst"},
            )
            assert resp.status_code == 404
            app.dependency_overrides.clear()

    def test_promote_invalid_action_422(self):
        app, client = self._client()
        resp = client.patch(
            f"/api/feedback-patterns/{uuid.uuid4()}/promote",
            json={"action": "bogus", "by": "analyst"},
        )
        assert resp.status_code == 422
        app.dependency_overrides.clear()

    def test_promote_control_char_by_422(self):
        app, client = self._client()
        resp = client.patch(
            f"/api/feedback-patterns/{uuid.uuid4()}/promote",
            json={"action": "prompt", "by": "a\x00b"},
        )
        assert resp.status_code == 422
        app.dependency_overrides.clear()

    def test_dismiss_pattern(self):
        row = self._fake_pattern(status="dismissed")
        with patch("app.services.feedback_patterns.dismiss_pattern",
                   new=AsyncMock(return_value=row)):
            app, client = self._client()
            resp = client.patch(f"/api/feedback-patterns/{row.id}/dismiss")
            assert resp.status_code == 200
            assert resp.json()["status"] == "dismissed"
            app.dependency_overrides.clear()

    def test_dismiss_pattern_404(self):
        with patch("app.services.feedback_patterns.dismiss_pattern",
                   new=AsyncMock(return_value=None)):
            app, client = self._client()
            resp = client.patch(f"/api/feedback-patterns/{uuid.uuid4()}/dismiss")
            assert resp.status_code == 404
            app.dependency_overrides.clear()

    def test_edit_pattern(self):
        row = self._fake_pattern()
        with patch("app.services.feedback_patterns.update_pattern",
                   new=AsyncMock(return_value=row)) as m:
            app, client = self._client()
            resp = client.patch(
                f"/api/feedback-patterns/{row.id}",
                json={"pattern": "A clearly long enough generalizable rule"},
            )
            assert resp.status_code == 200
            assert m.await_args.kwargs["pattern"] == "A clearly long enough generalizable rule"
            app.dependency_overrides.clear()

    def test_edit_pattern_too_short_422(self):
        app, client = self._client()
        resp = client.patch(
            f"/api/feedback-patterns/{uuid.uuid4()}",
            json={"pattern": "short"},
        )
        assert resp.status_code == 422
        app.dependency_overrides.clear()


class TestFeedbackPatternSchemas:
    """Validation on the Phase 3 request schemas."""

    def test_promote_request_accepts_valid(self):
        from app.schemas.api import FeedbackPatternPromoteRequest
        req = FeedbackPatternPromoteRequest(action="denylist", by="  analyst  ")
        assert req.action == "denylist"
        assert req.by == "analyst"  # stripped

    def test_promote_request_rejects_bad_action(self):
        from app.schemas.api import FeedbackPatternPromoteRequest
        with pytest.raises(Exception):
            FeedbackPatternPromoteRequest(action="nope", by="analyst")

    def test_promote_request_rejects_empty_by(self):
        from app.schemas.api import FeedbackPatternPromoteRequest
        with pytest.raises(Exception):
            FeedbackPatternPromoteRequest(action="prompt", by="   ")

    def test_edit_request_all_optional(self):
        from app.schemas.api import FeedbackPatternEditRequest
        req = FeedbackPatternEditRequest()
        assert req.pattern is None and req.category is None

    def test_edit_request_pattern_length(self):
        from app.schemas.api import FeedbackPatternEditRequest
        with pytest.raises(Exception):
            FeedbackPatternEditRequest(pattern="short")


class TestRunCounts:
    """The card's counts come from the Source row now, not only the WS event."""

    def test_run_counts_are_null_before_their_stage(self):
        from app.api.routes.pipeline import _run_counts
        assert _run_counts({}) == {"entity_count": None, "draft_count": None, "objects_written": None}
        assert _run_counts(None)["entity_count"] is None

    def test_run_counts_follow_state(self):
        from app.api.routes.pipeline import _run_counts
        counts = _run_counts({"entities": [1, 2, 3], "drafts": [1], "objects_written": 95})
        assert counts == {"entity_count": 3, "draft_count": 1, "objects_written": 95}

    def test_source_response_carries_the_counts(self):
        from app.schemas.api import SourceResponse
        assert {"entity_count", "draft_count", "objects_written"} <= set(SourceResponse.model_fields)
