"""Graph EXECUTION tests — driving the pipeline, not just describing it.

WHY THIS FILE EXISTS:
tests/test_integration.py asserts the graph's SHAPE (nodes exist, edges
exist, routers return the right string). Nothing drove the compiled graph
through a realistic gate configuration. That gap has a name: the
disabled-gate auto-skip regression, which needed a real source, real LLM
calls and a live analyst submission to surface, because:

  - `interrupt_before` is static and set at compile time, so the graph
    pauses before EVERY gate;
  - a per-source `gates_enabled` dict may mark some of them off;
  - the gate node self-skips when it RUNS, but the interrupt fires before it
    runs — so without a runner-side loop the graph sits forever at a gate no
    analyst will ever action.

Every existing test uses gates all-on or all-off. The regression was a MIXED
config (`chunks: True, procedures: False`), which nothing covered.

Two levels here:

  1. TestRunnerAutoSkipLoop drives `stream_with_sync` against a scripted fake
     graph. No Postgres, no LLM, no real graph — it isolates the auto-skip
     decision, which is the thing that broke.
  2. TestCompiledGraphExecution drives the REAL compiled graph with an
     in-memory checkpointer and mocked LLM calls, so gate pauses, resumes,
     mixed configs and rejection routing are exercised end to end without
     needing Postgres. tests/test_e2e.py covers the same ground against a
     real AsyncPostgresSaver, but skips silently when Postgres is absent —
     these run everywhere, always.
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.api.routes import pipeline as pipeline_routes
from app.graph.pipeline import compile_pipeline
from app.graph.state import GateAction, PipelineStatus

from tests.test_e2e import (
    _mock_chunks,
    _mock_classify,
    _mock_drafts,
    _mock_entities,
    _mock_propose_techniques,
    _mock_techniques,
)


# =====================================================================
# Level 1 — the runner's auto-skip loop, isolated
# =====================================================================


class _FakeState:
    """Stands in for a LangGraph StateSnapshot."""

    def __init__(self, values: dict, next_nodes: tuple[str, ...]):
        self.values = values
        self.next = next_nodes


class _ScriptedGraph:
    """A graph whose pauses are scripted.

    Each entry in `pauses` is the `state.next` tuple the runner will observe
    after one astream pass. An empty tuple means the run finished. Records
    every astream input so a test can tell a fresh start (dict) from a resume
    (None) — that distinction IS the auto-skip behavior.
    """

    def __init__(self, values: dict, pauses: list[tuple[str, ...]]):
        self._values = values
        self._pauses = list(pauses)
        self._current: tuple[str, ...] = ()
        self.astream_inputs: list[object] = []

    async def astream(self, graph_input, config):  # noqa: ARG002
        self.astream_inputs.append(graph_input)
        self._current = self._pauses.pop(0) if self._pauses else ()
        for event in ():  # pragma: no cover - shape only
            yield event

    async def aget_state(self, config):  # noqa: ARG002
        return _FakeState(self._values, self._current)


@asynccontextmanager
async def _fake_session():
    yield MagicMock()


@pytest.fixture
def runner_env():
    """Patch the runner's DB + WebSocket dependencies."""
    with patch.object(pipeline_routes, "async_session", _fake_session), \
         patch.object(pipeline_routes, "queue_service") as qs, \
         patch.object(pipeline_routes, "ws_manager") as ws:
        qs.update_status = AsyncMock()
        ws.has_subscribers.return_value = False
        yield qs


async def _drive(graph, runner_env) -> list:  # noqa: ARG001
    await pipeline_routes.stream_with_sync(
        graph, str(uuid.uuid4()), uuid.uuid4(), {"seed": True}
    )
    return graph.astream_inputs


class TestRunnerAutoSkipLoop:
    """The loop must resume past a disabled gate and stop at an enabled one."""

    ALL_ON = {"entities": True, "chunks": True, "procedures": True, "bundle": True}
    ALL_OFF = {k: False for k in ALL_ON}

    async def test_pauses_at_an_enabled_gate(self, runner_env):
        graph = _ScriptedGraph({"gates_enabled": self.ALL_ON}, [("gate_0",)])
        inputs = await _drive(graph, runner_env)
        assert inputs == [{"seed": True}], (
            "an enabled gate must pause for the analyst, not be resumed past"
        )

    async def test_resumes_past_a_disabled_gate(self, runner_env):
        """The regression. graph_input=None is how the runner tells LangGraph
        to execute the gate node so it can self-skip."""
        graph = _ScriptedGraph({"gates_enabled": self.ALL_OFF}, [("gate_0",), ()])
        inputs = await _drive(graph, runner_env)
        assert inputs == [{"seed": True}, None], (
            "a disabled gate must be resumed past with graph_input=None"
        )

    async def test_sweeps_through_a_run_of_disabled_gates(self, runner_env):
        graph = _ScriptedGraph(
            {"gates_enabled": self.ALL_OFF},
            [("gate_0",), ("gate_chunks",), ("gate_1",), ("gate_2",), ()],
        )
        inputs = await _drive(graph, runner_env)
        assert inputs == [{"seed": True}, None, None, None, None]

    async def test_mixed_config_skips_only_the_disabled_ones(self, runner_env):
        """`chunks` on, everything else off — the exact shape that hung a live
        source. It must sweep past gate_0 and stop at gate_chunks."""
        mixed = {"entities": False, "chunks": True, "procedures": False, "bundle": False}
        graph = _ScriptedGraph({"gates_enabled": mixed}, [("gate_0",), ("gate_chunks",)])
        inputs = await _drive(graph, runner_env)
        assert inputs == [{"seed": True}, None], (
            "should resume past disabled gate_0, then hold at enabled gate_chunks"
        )

    async def test_legacy_bool_gates_enabled_still_auto_skips(self, runner_env):
        """In-flight checkpoints predating the per-gate dict carry a bare
        bool. is_gate_enabled tolerates it; the loop must too."""
        graph = _ScriptedGraph({"gates_enabled": False}, [("gate_1",), ()])
        inputs = await _drive(graph, runner_env)
        assert inputs == [{"seed": True}, None]

    async def test_finished_run_does_not_loop(self, runner_env):
        graph = _ScriptedGraph({"gates_enabled": self.ALL_ON}, [()])
        inputs = await _drive(graph, runner_env)
        assert inputs == [{"seed": True}]

    async def test_pause_at_a_non_gate_node_is_not_auto_skipped(self, runner_env):
        """Only gate nodes are eligible. Anything else pausing is unexpected
        and must not be resumed past blindly."""
        graph = _ScriptedGraph({"gates_enabled": self.ALL_OFF}, [("normalize",)])
        inputs = await _drive(graph, runner_env)
        assert inputs == [{"seed": True}]

    async def test_every_gate_node_is_recognised_by_the_loop(self):
        """Anti-vacuity + coverage: the loop's node->key map must know every
        gate in the registry, or that gate can never be auto-skipped."""
        from app.api.routes._gate_registry import GATES

        mapped = pipeline_routes._GATE_NODE_TO_KEY
        assert len(mapped) >= 4
        for gate in GATES:
            assert gate.node_name in mapped, (
                f"{gate.node_name} is missing from _GATE_NODE_TO_KEY — a "
                f"source with that gate disabled would hang forever"
            )
            assert mapped[gate.node_name] == gate.state_key


# =====================================================================
# Level 2 — the real compiled graph, driven by the real runner
# =====================================================================


def _llm_dispatcher():
    """One stand-in for every call_llm, dispatching on the requested tool.

    Payload builders are reused from tests/test_e2e.py; only the dispatch is
    local. Chunk ids are assigned dynamically by the chunker, so the
    technique and draft payloads are keyed off ids scraped from the prompt.
    """
    import re

    captured: list[str] = []

    def _dispatch(*args, **kwargs):
        tool_choice = kwargs.get("tool_choice", {}) or {}
        tool = tool_choice.get("name", "") if isinstance(tool_choice, dict) else ""

        if tool == "extract_entities":
            return _mock_entities()
        if tool == "classify_sections":
            return _mock_classify()
        if tool == "chunk_behaviors":
            return _mock_chunks()

        messages = kwargs.get("messages", args[1] if len(args) > 1 else [])
        prompt = "".join(m.get("content", "") for m in messages if isinstance(m, dict))
        ids = re.findall(r"chk-[a-f0-9]+", prompt) or captured
        if ids and not captured:
            captured.extend(ids[:3])
        ids = (ids or ["chk-001", "chk-002", "chk-003"])[:3]

        if tool == "propose_techniques":
            return _mock_propose_techniques(ids)
        if tool == "extract_techniques":
            return _mock_techniques(ids)
        if tool == "draft_procedures":
            return _mock_drafts(captured[:3] or ids)
        return _mock_classify()

    return _dispatch


@pytest.fixture
def llm_stub():
    dispatch = _llm_dispatcher()
    targets = [
        "app.nodes.llm.entity_extraction.call_llm",
        "app.nodes.llm.chunking.call_llm",
        "app.nodes.llm.technique_extraction.call_llm",
        "app.nodes.llm.drafting.call_llm",
        "app.nodes.llm.feedback_synthesis.call_llm",
    ]
    patches = [patch(t, new_callable=AsyncMock, side_effect=dispatch) for t in targets]
    for p in patches:
        p.start()
    try:
        yield
    finally:
        for p in patches:
            p.stop()


@pytest.fixture
def graph():
    """The real compiled pipeline, with the PRODUCTION interrupt list.

    Deliberately does not override interrupt_before. tests/test_e2e.py pins
    ["gate_0", "gate_1", "gate_2"], which predates gate_chunks — the same
    stale-list mistake that once let gate_chunks run inline in main.py. Using
    the default keeps this honest as gates are added.
    """
    from langgraph.checkpoint.memory import InMemorySaver

    return compile_pipeline(checkpointer=InMemorySaver())


SAMPLE = (
    "LockBit 3.0 actors exploited CVE-2023-46604 in Apache ActiveMQ for initial "
    "access. They then used certutil.exe to download a web shell from 203.0.113.10. "
    "PowerShell downloaded a Cobalt Strike beacon for command and control. Mimikatz "
    "harvested credentials before LockBit ransomware was deployed network-wide."
)


def _seed(gates_enabled: dict | bool) -> dict:
    return {
        "source_id": f"exec-{uuid.uuid4().hex[:8]}",
        "channel": "manual",
        "source_type": "free_text",
        "raw_content_path": SAMPLE,
        "title": "graph execution test",
        "metadata": {},
        "source_reliability": 85,
        "gates_enabled": gates_enabled,
        "extract_figures": False,
        "sequentiality": "auto",
        "status": "parsing",
        "current_node": "parse_and_validate",
    }


async def _drive_real(graph, runner_env, gates_enabled):
    """Run the real graph through the real runner. Returns (final_state,
    every status the runner persisted)."""
    thread_id = str(uuid.uuid4())
    await pipeline_routes.stream_with_sync(
        graph, thread_id, uuid.uuid4(), _seed(gates_enabled)
    )
    statuses = [c.args[2] for c in runner_env.update_status.call_args_list if len(c.args) > 2]
    state = await graph.aget_state({"configurable": {"thread_id": thread_id}})
    return state, statuses


class TestCompiledGraphExecution:
    """The real graph, the real runner, no Postgres and no LLM."""

    async def test_default_interrupts_cover_every_registered_gate(self, graph):
        """A gate missing from interrupt_before runs inline and is never
        reviewed. That shipped once: gate_chunks auto-approved invisibly
        because main.py pinned a list written before it existed."""
        from app.api.routes._gate_registry import GATES

        interrupts = set(getattr(graph, "interrupt_before_nodes", ()) or ())
        assert interrupts, (
            "could not read interrupt_before_nodes off the compiled graph — the "
            "attribute moved, and this test would otherwise pass vacuously"
        )
        assert len(GATES) >= 4
        for gate in GATES:
            assert gate.node_name in interrupts, (
                f"{gate.node_name} is not in the default interrupt_before — it "
                f"would execute inline and auto-approve without review"
            )

    async def test_all_gates_off_runs_to_completion(self, graph, llm_stub, runner_env):
        state, statuses = await _drive_real(
            graph, runner_env,
            {"entities": False, "chunks": False, "procedures": False, "bundle": False},
        )
        assert not state.next, f"run did not finish; paused at {state.next}"
        assert state.values.get("status") in (
            PipelineStatus.COMPLETED.value, PipelineStatus.FAILED.value
        )
        assert "stix_bundle" in state.values, "never reached serialization"

    async def test_all_gates_on_pauses_at_the_first_gate(self, graph, llm_stub, runner_env):
        state, statuses = await _drive_real(
            graph, runner_env,
            {"entities": True, "chunks": True, "procedures": True, "bundle": True},
        )
        assert state.next == ("gate_0",), f"expected a pause at gate_0, got {state.next}"
        assert statuses[-1] == PipelineStatus.GATE_0.value

    async def test_mixed_config_sweeps_past_disabled_and_holds_at_enabled(
        self, graph, llm_stub, runner_env
    ):
        """The shape that hung a live source: entities off, chunks on. The
        runner must sweep past gate_0 and stop at gate_chunks."""
        state, statuses = await _drive_real(
            graph, runner_env,
            {"entities": False, "chunks": True, "procedures": False, "bundle": False},
        )
        assert state.next == ("gate_chunks",), (
            f"expected a pause at gate_chunks, got {state.next}"
        )
        assert PipelineStatus.GATE_0.value not in statuses, (
            "gate_0 is disabled — it must never be surfaced as a review state"
        )
        # Anti-vacuity: reaching gate_chunks means parse -> figures -> entities
        # -> gate_0 (skipped) -> chunking all actually ran.
        assert state.values.get("validated_entities"), "entity extraction never ran"
        assert state.values.get("chunks"), "chunking never ran"

    async def test_a_late_enabled_gate_is_reached(self, graph, llm_stub, runner_env):
        """Only the bundle gate on: everything earlier must auto-skip, which
        means the sweep has to work several times in a row."""
        state, statuses = await _drive_real(
            graph, runner_env,
            {"entities": False, "chunks": False, "procedures": False, "bundle": True},
        )
        assert state.next == ("gate_2",), f"expected a pause at gate_2, got {state.next}"


class TestRunnerLoopIsBounded:
    """A gate that never advances must fail the source, not spin forever.

    Found by mutation-testing the loop: making gate_2 ignore
    its disabled flag turned a rejection into a cycle — resume, reject, route
    back, interrupt, resume — and the unbounded `while True` hung the test
    runner outright, writing a DB status update per iteration.
    """

    ALL_OFF = {"entities": False, "chunks": False, "procedures": False, "bundle": False}

    async def test_a_gate_that_never_advances_fails_the_source(self, runner_env):
        """Scripted graph that pauses at the same disabled gate forever."""
        graph = _ScriptedGraph(
            {"gates_enabled": self.ALL_OFF},
            [("gate_2",)] * 500,  # far more than the bound
        )
        await _drive(graph, runner_env)

        limit = pipeline_routes._MAX_GATE_AUTO_SKIPS
        assert len(graph.astream_inputs) <= limit + 2, (
            f"loop ran {len(graph.astream_inputs)} times against a bound of "
            f"{limit} — it is not actually bounded"
        )

        statuses = [c.args[2] for c in runner_env.update_status.call_args_list
                    if len(c.args) > 2]
        assert statuses[-1] == PipelineStatus.FAILED.value, (
            "a stalled run must surface as failed, not sit silently"
        )
        err = runner_env.update_status.call_args_list[-1].kwargs.get("error", "")
        assert "stalled" in err and "gate_2" in err, (
            f"the error must name the symptom and the gate; got {err!r}"
        )

    async def test_the_bound_allows_a_normal_full_sweep(self, runner_env):
        """Guards against setting the bound too tight: skipping every gate in
        one run is legitimate and must not trip it."""
        from app.api.routes._gate_registry import GATES

        pauses = [(g.node_name,) for g in GATES] + [()]
        graph = _ScriptedGraph({"gates_enabled": self.ALL_OFF}, pauses)
        await _drive(graph, runner_env)

        statuses = [c.args[2] for c in runner_env.update_status.call_args_list
                    if len(c.args) > 2]
        assert PipelineStatus.FAILED.value not in statuses, (
            "a legitimate all-gates-disabled sweep tripped the stall guard"
        )
        assert len(graph.astream_inputs) == len(GATES) + 1
