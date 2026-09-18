"""Cross-layer contract tests.

WHY THIS FILE EXISTS:
Every bug that has escaped this project's ~1000 tests lived *between* two
layers, not inside one. The tests verify each layer in isolation; nothing
checked that the layers agreed with each other. The record:

  - `source_excerpt` was added to the chunk tool schema but not to ChunkItem.
    Found only by a live LLM call.
  - `precondition` did the same later and hard-failed a whole 40-chunk run
    on a real source.
  - `x_observable_refs`, then `x_source_provenance`, then
    `x_chain_label`/`x_chain_root` were written onto x-procedure objects that
    the schema (additionalProperties:false) never declared. The last pair
    failed a live pipeline run.
  - `PipelineStatus.SYNTHESIZING_FEEDBACK` was declared, never emitted, and
    mapped by no frontend component.

Each is the same shape: two places that must agree, and no test that they do.
These are static introspection tests — no LLM, no fixtures, milliseconds — so
the mismatch fails in CI instead of on a real report.

The pattern was established by TestChunkToolSchemaMatchesValidator in
test_llm_nodes.py; this file generalizes it to every remaining surface.
"""

import pytest

from app.nodes.llm import tool_models as tm
from app.nodes.llm.chunking import CHUNK_BEHAVIORS_TOOL, CLASSIFY_SECTIONS_TOOL
from app.nodes.llm.drafting import DRAFT_PROCEDURES_TOOL
from app.nodes.llm.entity_extraction import EXTRACT_ENTITIES_TOOL
from app.nodes.llm.feedback_synthesis import (
    ATTRIBUTE_CORRECTIONS_TOOL,
    SYNTHESIZE_FEEDBACK_TOOL,
)
from app.nodes.llm.figure_extraction import EXTRACT_FIGURE_TOOL
from app.nodes.llm.technique_extraction import (
    EXTRACT_TECHNIQUES_TOOL,
    PROPOSE_TECHNIQUES_TOOL,
)


# ── helpers to walk a tool schema ────────────────────────────────────

def _props(tool):
    """Top-level properties the LLM is told to emit."""
    return tool["input_schema"]["properties"]


def _items(tool, key):
    """Per-element properties of an array-valued tool field."""
    return _props(tool)[key]["items"]["properties"]


def _nested_obj(tool, array_key, obj_key):
    """Properties of an object-valued field inside an array element."""
    return _props(tool)[array_key]["items"]["properties"][obj_key]["properties"]


def _nested_items(tool, array_key, inner_key):
    """Per-element properties of an array nested inside an array element."""
    return _props(tool)[array_key]["items"]["properties"][inner_key]["items"]["properties"]


_CT = "chunk_techniques"

# (label, schema properties, Pydantic model that must accept them).
# Adding a tool schema without adding a row here is the gap this file exists
# to close — keep them in step.
SCHEMA_MODEL_PAIRS = [
    ("EXTRACT_ENTITIES_TOOL.entities[]",
     _items(EXTRACT_ENTITIES_TOOL, "entities"), tm.EntityItem),
    ("EXTRACT_ENTITIES_TOOL.detection_rules[]",
     _items(EXTRACT_ENTITIES_TOOL, "detection_rules"), tm.DetectionRuleItem),
    ("EXTRACT_ENTITIES_TOOL",
     _props(EXTRACT_ENTITIES_TOOL), tm.ExtractEntitiesOutput),

    ("DRAFT_PROCEDURES_TOOL.drafts[]",
     _items(DRAFT_PROCEDURES_TOOL, "drafts"), tm.DraftItem),
    ("DRAFT_PROCEDURES_TOOL",
     _props(DRAFT_PROCEDURES_TOOL), tm.DraftProceduresOutput),

    ("CLASSIFY_SECTIONS_TOOL.sections[]",
     _items(CLASSIFY_SECTIONS_TOOL, "sections"), tm.SectionItem),
    ("CLASSIFY_SECTIONS_TOOL",
     _props(CLASSIFY_SECTIONS_TOOL), tm.ClassifySectionsOutput),

    ("CHUNK_BEHAVIORS_TOOL.chunks[]",
     _items(CHUNK_BEHAVIORS_TOOL, "chunks"), tm.ChunkItem),
    ("CHUNK_BEHAVIORS_TOOL.chunks[].context",
     _nested_obj(CHUNK_BEHAVIORS_TOOL, "chunks", "context"), tm.ChunkContext),
    ("CHUNK_BEHAVIORS_TOOL.chunks[].precondition",
     _nested_obj(CHUNK_BEHAVIORS_TOOL, "chunks", "precondition"), tm.PreconditionItem),
    ("CHUNK_BEHAVIORS_TOOL",
     _props(CHUNK_BEHAVIORS_TOOL), tm.ChunkBehaviorsOutput),

    ("EXTRACT_FIGURE_TOOL",
     _props(EXTRACT_FIGURE_TOOL), tm.ExtractFigureOutput),

    ("SYNTHESIZE_FEEDBACK_TOOL.patterns[]",
     _items(SYNTHESIZE_FEEDBACK_TOOL, "patterns"), tm.FeedbackPatternItem),
    ("SYNTHESIZE_FEEDBACK_TOOL",
     _props(SYNTHESIZE_FEEDBACK_TOOL), tm.SynthesizeFeedbackOutput),

    ("ATTRIBUTE_CORRECTIONS_TOOL.attributions[]",
     _items(ATTRIBUTE_CORRECTIONS_TOOL, "attributions"), tm.CorrectionAttributionItem),
    ("ATTRIBUTE_CORRECTIONS_TOOL",
     _props(ATTRIBUTE_CORRECTIONS_TOOL), tm.AttributeCorrectionsOutput),

    ("PROPOSE_TECHNIQUES_TOOL.chunk_proposals[]",
     _items(PROPOSE_TECHNIQUES_TOOL, "chunk_proposals"), tm.ProposeTechniquesItem),
    ("PROPOSE_TECHNIQUES_TOOL",
     _props(PROPOSE_TECHNIQUES_TOOL), tm.ProposeTechniquesOutput),

    ("EXTRACT_TECHNIQUES_TOOL.chunk_techniques[]",
     _items(EXTRACT_TECHNIQUES_TOOL, _CT), tm.ChunkTechniqueMapping),
    ("EXTRACT_TECHNIQUES_TOOL.chunk_techniques[].techniques[]",
     _nested_items(EXTRACT_TECHNIQUES_TOOL, _CT, "techniques"), tm.TechniqueItem),
    ("EXTRACT_TECHNIQUES_TOOL.chunk_techniques[].verbatim_match_decisions[]",
     _nested_items(EXTRACT_TECHNIQUES_TOOL, _CT, "verbatim_match_decisions"),
     tm.VerbatimMatchDecision),
    ("EXTRACT_TECHNIQUES_TOOL",
     _props(EXTRACT_TECHNIQUES_TOOL), tm.ExtractTechniquesOutput),
]

_IDS = [label for label, _, _ in SCHEMA_MODEL_PAIRS]


class TestToolSchemaMatchesValidator:
    """Every field the LLM is told to emit must validate, and every field a
    validator declares must be reachable.

    All models derive from _StrictBase (extra='forbid'), so an advertised
    field the model doesn't know is not a warning — it hard-fails the run the
    moment the LLM complies. That is exactly how `precondition` killed a live
    40-chunk source.
    """

    @pytest.mark.parametrize("label,props,model", SCHEMA_MODEL_PAIRS, ids=_IDS)
    def test_every_advertised_field_is_accepted(self, label, props, model):
        missing = sorted(set(props) - set(model.model_fields))
        assert not missing, (
            f"{label}: the tool schema advertises {missing} but "
            f"{model.__name__} forbids it. extra='forbid' means the run "
            f"hard-fails as soon as the LLM emits it."
        )

    @pytest.mark.parametrize("label,props,model", SCHEMA_MODEL_PAIRS, ids=_IDS)
    def test_no_validator_field_is_unreachable(self, label, props, model):
        orphans = sorted(set(model.model_fields) - set(props))
        assert not orphans, (
            f"{label}: {model.__name__} declares {orphans} that the tool "
            f"schema never asks for. The LLM cannot populate it — it is dead "
            f"weight, a typo, or a schema field someone forgot to add."
        )


class TestEveryToolSchemaIsCovered:
    """Guards the guard: a new tool schema must gain a row above."""

    def test_all_tool_schemas_appear_in_the_pair_table(self):
        import importlib
        import pkgutil

        import app.nodes.llm as llm_pkg

        declared = set()
        for mod in pkgutil.iter_modules(llm_pkg.__path__):
            module = importlib.import_module(f"app.nodes.llm.{mod.name}")
            for name, value in vars(module).items():
                if (
                    name.endswith("_TOOL")
                    and isinstance(value, dict)
                    and "input_schema" in value
                ):
                    declared.add(name)

        # Anti-vacuity: if the scan finds nothing the assertion below passes
        # while checking nothing at all.
        assert len(declared) >= 8, (
            f"tool-schema scan found only {sorted(declared)} — the scan is "
            f"broken, so this test would pass without checking anything"
        )

        covered = {label.split(".")[0] for label in _IDS}
        uncovered = sorted(declared - covered)
        assert not uncovered, (
            f"tool schema(s) {uncovered} have no contract test. Add a row to "
            f"SCHEMA_MODEL_PAIRS pairing each with the model that accepts it."
        )


# =====================================================================
# Backend <-> frontend contracts
# =====================================================================

import re
from pathlib import Path

_FRONTEND = Path(__file__).resolve().parents[1] / "frontend" / "src"


def _read(rel: str) -> str:
    return (_FRONTEND / rel).read_text()


def _object_keys(source: str, const_name: str) -> set[str]:
    """Keys of a top-level `const NAME = { ... };` object literal.

    Deliberately a regex rather than a JS parse: this only has to read a
    few flat literals, and a parser would be a dependency for no gain.
    """
    m = re.search(rf"const {const_name} = \{{(.*?)\n\}};", source, re.S)
    assert m, f"could not locate `const {const_name} = {{...}}` — did it move or get renamed?"
    return set(re.findall(r"^\s{2}([A-Za-z_][A-Za-z_0-9]*):", m.group(1), re.M))


def _quoted_strings(source: str, pattern: str) -> set[str]:
    """Every quoted string inside each region matching `pattern`."""
    out: set[str] = set()
    for block in re.findall(pattern, source, re.S):
        out.update(re.findall(r'"([a-z_0-9]+)"', block))
    return out


def _backend_statuses() -> set[str]:
    from app.graph.state import PipelineStatus
    return {s.value for s in PipelineStatus}


# Statuses a source is never *left* sitting in, so they need no live
# WebSocket subscription. Everything else must be in ACTIVE_STATUSES or the
# card silently stops updating mid-run.
_TERMINAL_OR_PRE_RUN = {"queued", "completed", "failed"}


class TestPipelineStatusReachesTheUI:
    """Every backend status must have a home in the frontend.

    Regression: PipelineStatus.SYNTHESIZING_FEEDBACK was declared in the enum
    and mapped by nothing — no Kanban column, no label, no color, absent from
    ACTIVE_STATUSES. Nothing failed, because nothing checked.

    Second regression: the frontend kept FIVE hand-listed copies
    of the status set across four files, this test scanned three of them, and
    the two it could not see (StatsBar's processing set, the card's progress
    map) drifted for months. The frontend now has ONE table,
    lib/pipelineStatus.js, and every component derives from it — so this test
    scans that table (plus the gate entries it pulls from lib/gates.js) and
    nothing else. Adding a status is one row there.
    """

    @staticmethod
    def _table_source() -> str:
        return _read("lib/pipelineStatus.js")

    @staticmethod
    def _literal_entries() -> dict[str, str]:
        """status -> the object literal it is declared in (non-gate rows)."""
        out: dict[str, str] = {}
        for block in re.findall(r"\{[^{}]*?status: \"[a-z_0-9]+\"[^{}]*\}", TestPipelineStatusReachesTheUI._table_source()):
            m = re.search(r'status: "([a-z_0-9]+)"', block)
            out[m.group(1)] = block
        return out

    @staticmethod
    def _gate_entries() -> dict[str, str]:
        """status -> column, for the gate pauses and their resuming_* twins,
        which the table derives from lib/gates.js rather than listing."""
        gates = _read("lib/gates.js")
        out: dict[str, str] = {}
        for block in re.findall(r"\{\n    status: \"[a-z_0-9]+\",.*?\n  \},", gates, re.S):
            status = re.search(r'status: "([a-z_0-9]+)"', block).group(1)
            column = re.search(r'column: "([a-z_]+)"', block)
            nxt = re.search(r'nextStatus: "([a-z_0-9]+)"', block)
            assert column and nxt, f"gate {status} in lib/gates.js lacks column/nextStatus"
            out[status] = column.group(1)
            out[nxt.group(1)] = column.group(1)
        return out

    def _table_statuses(self) -> set[str]:
        return set(self._literal_entries()) | set(self._gate_entries())

    @staticmethod
    def _kanban_columns() -> set[str]:
        return set(re.findall(r'\n    id: "([a-z_]+)",', _read("components/KanbanBoard.jsx")))

    def test_scan_finds_the_table(self):
        """Anti-vacuity: these regexes are the whole test. If they stop
        matching, every assertion below passes while checking nothing."""
        assert len(_backend_statuses()) >= 15
        assert len(self._literal_entries()) >= 15
        assert len(self._gate_entries()) == 8   # 4 gates x (pause + resuming)
        assert len(self._kanban_columns()) == 7
        assert "ACTIVE_STATUSES" in _read("App.jsx")          # derives, not lists
        assert "PROCESSING_STATUSES" in _read("components/StatsBar.jsx")
        assert "statusesInColumn" in _read("components/KanbanBoard.jsx")

    def test_every_backend_status_is_in_the_table(self):
        missing = sorted(_backend_statuses() - self._table_statuses())
        assert not missing, (
            f"status(es) {missing} have no row in lib/pipelineStatus.js — a "
            f"source in that state vanishes from the board, shows a raw enum "
            f"value on its card, and is not counted anywhere."
        )

    def test_every_row_lands_in_a_real_kanban_column(self):
        columns = self._kanban_columns()
        for status, block in self._literal_entries().items():
            m = re.search(r'column: "([a-z_]+)"', block)
            assert m, f"status {status} has no column in lib/pipelineStatus.js"
            assert m.group(1) in columns, f"status {status} -> column {m.group(1)!r} that KanbanBoard does not render"
        for status, column in self._gate_entries().items():
            assert column in columns, f"gate status {status} -> column {column!r} that KanbanBoard does not render"

    def test_every_row_has_a_human_label(self):
        for status, block in self._literal_entries().items():
            assert re.search(r'label: "[^"]+"', block), (
                f"status {status} has no label — the card shows a raw enum value"
            )
        # Gate rows take statusLabel from lib/gates.js, which every gate declares.
        gates = _read("lib/gates.js")
        assert gates.count("statusLabel:") == 4

    def test_only_terminal_states_leave_the_websocket(self):
        """ACTIVE_STATUSES is derived as every row without `terminal: true`.
        The rows so marked must be exactly the states a source is LEFT in."""
        terminal = {
            s for s, block in self._literal_entries().items()
            if re.search(r"terminal: true", block)
        }
        assert terminal == _TERMINAL_OR_PRE_RUN, (
            f"terminal rows {sorted(terminal)} != {sorted(_TERMINAL_OR_PRE_RUN)} — "
            f"a mis-marked row either drops live updates mid-run or keeps a "
            f"WebSocket open on a finished source."
        )

    def test_frontend_invents_no_status_the_backend_cannot_emit(self):
        """A row for a status that no longer exists is dead config — and
        usually the leftover of a rename that half-landed."""
        unknown = sorted(self._table_statuses() - _backend_statuses())
        assert not unknown, (
            f"lib/pipelineStatus.js lists status(es) {unknown} that "
            f"PipelineStatus cannot produce."
        )


class TestFeedbackCategoryAreasAgree:
    """The Feedback tab colors a pattern by its gate area, mirrored from the
    backend's _AREA_FOR_CATEGORY. Two copies; this keeps them one."""

    def test_frontend_category_area_matches_backend(self):
        from app.nodes.llm.feedback_synthesis import _AREA_FOR_CATEGORY
        src = _read("components/FeedbackPatternsView.jsx")
        m = re.search(r"const CATEGORY_AREA = \{(.*?)\n\};", src, re.S)
        assert m, "could not locate CATEGORY_AREA in FeedbackPatternsView.jsx"
        js = dict(re.findall(r'([a-z_]+): "([a-z]+)"', m.group(1)))
        assert js == dict(_AREA_FOR_CATEGORY), (
            f"frontend CATEGORY_AREA {js} != backend _AREA_FOR_CATEGORY"
        )



class TestGateRegistriesAgree:
    """The Python gate registry and lib/gates.js describe the same gates.

    They are edited independently, and a mismatch is silent: the frontend
    posts to a route the backend doesn't expect, or shows a gate that never
    pauses.
    """

    @staticmethod
    def _js_gates() -> list[dict]:
        src = _read("lib/gates.js")
        m = re.search(r"export const GATES = \[(.*?)\n\];", src, re.S)
        assert m, "could not locate `export const GATES = [...]` in lib/gates.js"
        gates = []
        for block in re.findall(r"\{(.*?)\n  \}", m.group(1), re.S):
            g = dict(re.findall(r'(\w+): "([^"]*)"', block))
            route = re.search(r"routeId: (\d+|null)", block)
            if route:
                g["routeId"] = route.group(1)
            gates.append(g)
        return gates

    def test_scan_finds_the_gates(self):
        from app.api.routes._gate_registry import GATES
        assert len(GATES) >= 4
        assert len(self._js_gates()) >= 4, "gates.js scan found nothing"

    def test_gate_enable_keys_match(self):
        from app.graph.state import GATE_KEYS
        js_keys = {g["enableKey"] for g in self._js_gates() if "enableKey" in g}
        assert js_keys == set(GATE_KEYS), (
            f"gates_enabled keys disagree — backend {sorted(GATE_KEYS)} vs "
            f"frontend {sorted(js_keys)}. The modal would send a dict the API "
            f"cannot honor."
        )

    def test_gate_statuses_match(self):
        from app.api.routes._gate_registry import GATES
        py = {g.expected_status for g in GATES}
        js = {g["status"] for g in self._js_gates() if "status" in g}
        assert py == js, (
            f"gate pause-statuses disagree — backend {sorted(py)} vs frontend "
            f"{sorted(js)}."
        )

    def test_resuming_statuses_are_derivable(self):
        """SourceCard/lib derive `resuming_from_<status>`; the backend writes
        its own string. They must be the same string."""
        from app.api.routes._gate_registry import GATES
        for g in GATES:
            assert g.resuming_status == f"resuming_from_{g.expected_status}", (
                f"{g.node_name}: resuming status {g.resuming_status!r} is not "
                f"resuming_from_{g.expected_status} — the frontend derives the "
                f"latter and would never match."
            )


# =====================================================================
# Serializer <-> Neo4j distribution contract
# =====================================================================


def _serializer_source() -> str:
    return (
        Path(__file__).resolve().parents[1]
        / "backend" / "app" / "nodes" / "deterministic" / "serialization.py"
    ).read_text()


class TestSerializerOutputIsDistributable:
    """Everything the serializer emits must have a Neo4j mapping.

    An unmapped STIX type is written with no label, and an unmapped
    relationship_type is dropped — so the bundle looks fine while the graph
    silently loses nodes and edges. This has happened: one pass
    had to backfill DetectionStrategy, Analytic, DataComponent, DataSource,
    Report, MarkingDefinition and CourseOfAction labels, plus the
    uses-data-component / has-observable / subtechnique-of / authored-by /
    derived-from / duplicate-of / located-at relationship mappings.
    """

    @staticmethod
    def _label_map() -> dict:
        import app.nodes.deterministic.distribution as dist
        for value in vars(dist).values():
            if isinstance(value, dict) and value.get("x-procedure") == "Procedure":
                return value
        raise AssertionError("could not find the STIX-type -> Neo4j label map")

    @staticmethod
    def _emitted_relationship_types() -> set[str]:
        """relationship_type literals passed to _make_sro()."""
        import ast

        tree = ast.parse(_serializer_source())
        rels: set[str] = set()
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "_make_sro"):
                continue
            if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                rels.add(node.args[1].value)
            for kw in node.keywords:
                if kw.arg in ("relationship_type", "rel_type") and isinstance(kw.value, ast.Constant):
                    rels.add(kw.value.value)
        return rels

    @staticmethod
    def _emitted_object_types() -> set[str]:
        """Every STIX type the serializer can put in a bundle.

        Three sources: the entity-type map, literal `"type": "..."` object
        builders, and the ATT&CK detection-chain stubs.
        """
        from app.nodes.deterministic.serialization import _ENTITY_TO_STIX_TYPE

        src = _serializer_source()
        types = set(_ENTITY_TO_STIX_TYPE.values())
        types |= set(re.findall(r'"type": "([a-z][a-z0-9-]+)"', src))
        types |= set(re.findall(r'"(x-mitre-[a-z-]+)"', src))
        # Envelope / edge types are not nodes; extension-definition is bundle
        # metadata that distribution skips on purpose (see extension_definitions).
        return types - {"bundle", "relationship", "extension-definition"}

    def test_scan_finds_something(self):
        """Anti-vacuity: empty sets would make both assertions pass."""
        assert len(self._emitted_relationship_types()) >= 5
        assert len(self._emitted_object_types()) >= 15
        assert len(self._label_map()) >= 20

    def test_every_emitted_object_type_has_a_neo4j_label(self):
        missing = sorted(self._emitted_object_types() - set(self._label_map()))
        assert not missing, (
            f"STIX type(s) {missing} are emitted but have no Neo4j label — "
            f"they land in the graph unlabelled and become unqueryable."
        )

    def test_every_emitted_relationship_has_a_neo4j_mapping(self):
        from app.nodes.deterministic.distribution import _REL_TYPE_TO_NEO4J

        missing = sorted(self._emitted_relationship_types() - set(_REL_TYPE_TO_NEO4J))
        assert not missing, (
            f"relationship_type(s) {missing} are emitted but have no Neo4j "
            f"mapping — those edges are silently dropped on distribution."
        )

    def test_ipv6_is_mapped_even_though_the_entity_map_says_ipv4(self):
        """_entity_to_stix swaps ipv4-addr for ipv6-addr at runtime, so the
        static map under-reports. Pin the dynamic case explicitly."""
        assert "ipv6-addr" in self._label_map()


# ── AI gate reviewer ─────────────────────────────────────────────────
#
# The reviewer introduces four places that must agree about which gates have
# an AI reviewer. Getting them out of step does not fail loudly: a gate with a
# reviewer but no status writes an unmapped status, and Kanban columns
# enumerate statuses with no fallback — so the card vanishes from the board
# rather than merely looking wrong.

class TestReviewerRegistriesAgree:
    def test_reviewers_and_statuses_cover_the_same_gates(self):
        from app.api.routes.pipeline import _REVIEWING_STATUS
        from app.services.reviewer.gate_reviewers import REVIEWERS

        assert set(_REVIEWING_STATUS) == set(REVIEWERS), (
            "Every gate with an AI reviewer needs a reviewing status, and "
            "vice versa. A reviewer without a status writes a status no "
            "Kanban column lists, which removes the card from the board."
        )

    def test_reviewing_statuses_are_declared_pipeline_statuses(self):
        from app.api.routes.pipeline import _REVIEWING_STATUS
        from app.graph.state import PipelineStatus

        declared = {s.value for s in PipelineStatus}
        for gate_key, status in _REVIEWING_STATUS.items():
            assert status in declared, (
                f"gate '{gate_key}' maps to status '{status}', which is not a "
                f"PipelineStatus value"
            )

    def test_outcome_differs_cover_every_reviewer(self):
        """A gate with a reviewer but no differ records no agreement data.

        That failure is invisible: the review still works, the analyst still
        submits, and the measurement that decides whether the gate can ever
        run unattended just never accumulates.
        """
        from app.services.reviewer.outcomes import OUTCOME_DIFFERS
        from app.services.reviewer.gate_reviewers import REVIEWERS

        assert set(OUTCOME_DIFFERS) == set(REVIEWERS)

    def test_gate1_reject_reasons_are_routable(self):
        """The reviewer must never emit a reject_reason the gate cannot route.

        A reject_reason the gate does not recognize falls through to
        `_compute_rejection_routing` and routes the rerun somewhere the
        reviewer did not intend — bad_chunk_boundary re-chunks the whole
        source, everything else re-maps techniques.

        This was an equality check. It is now a SUBSET check, deliberately:
        `not_a_procedure` and `duplicate` were withdrawn from the reviewer's
        choices because no re-extraction can fix them (the chunk text is
        unchanged, so the identical draft returns), but they stay in
        Gate1RejectReason so stored checkpoints, the correction log and the
        human UI keep working. Subset is the property that was ever load-
        bearing; equality was a proxy for it.

        Tool schema and Pydantic Literal must still match each other exactly —
        a mismatch there lets the model emit something validation rejects.
        """
        import typing

        from app.graph.state import Gate1RejectReason
        from app.services.reviewer.gate_reviewers import REVIEW_PROCEDURES_TOOL
        from app.services.reviewer.models import RejectReason, RemoveReason

        enum = {e.value for e in Gate1RejectReason}
        reviewer_reasons = set(typing.get_args(RejectReason))
        assert reviewer_reasons <= enum, (
            "the reviewer can pick a reason the gate cannot route: "
            f"{sorted(reviewer_reasons - enum)}"
        )
        schema = REVIEW_PROCEDURES_TOOL["input_schema"]["properties"]["drafts"]
        assert set(schema["items"]["properties"]["reject_reason"]["enum"]) == (
            reviewer_reasons
        )

        # Nothing withdrawn from reject may simply vanish — it has to be
        # expressible as a removal instead, or the reviewer loses the ability
        # to report it at all.
        withdrawn = enum - reviewer_reasons
        assert withdrawn <= set(typing.get_args(RemoveReason)), (
            f"withdrawn from reject but unavailable as a remove: "
            f"{sorted(withdrawn - set(typing.get_args(RemoveReason)))}"
        )

    def test_gate1_remove_reasons_match_the_tool_schema(self):
        """Same exact-match requirement as reject, for the removal verb."""
        import typing

        from app.services.reviewer.gate_reviewers import REVIEW_PROCEDURES_TOOL
        from app.services.reviewer.models import RemoveReason

        schema = REVIEW_PROCEDURES_TOOL["input_schema"]["properties"]["drafts"]
        assert set(schema["items"]["properties"]["remove_reason"]["enum"]) == (
            set(typing.get_args(RemoveReason))
        )

    def test_chunk_rerun_reasons_match_the_enum(self):
        """Tool schema, Pydantic Literal, and the enum must be one list.

        A reason the gate does not recognize falls through to
        `ChunkGateRejectReason.OTHER`, which strips the hint the re-chunk was
        supposed to carry — the rerun happens, costs a full chunking pass,
        and arrives with no guidance about what was wrong.
        """
        import typing

        from app.graph.state import ChunkGateRejectReason
        from app.services.reviewer.gate_reviewers import REVIEW_CHUNKS_TOOL
        from app.services.reviewer.models import ChunkRerunReason

        enum = {e.value for e in ChunkGateRejectReason}
        assert set(typing.get_args(ChunkRerunReason)) == enum
        schema = REVIEW_CHUNKS_TOOL["input_schema"]["properties"]["reject"]
        assert set(schema["properties"]["reason"]["enum"]) == enum

    def test_chunk_actions_are_actions_the_gate_accepts(self):
        """An action the gate does not implement is an unapplicable
        recommendation — the same defect as Gate 1's dropped `edited_name`,
        which spent the analyst's attention on a change no control could
        make."""
        import typing

        from app.schemas.api import _VALID_CHUNK_ACTIONS, _VALID_EDGE_ACTIONS
        from app.services.reviewer.models import (
            ChunkEdgeRecommendation,
            ChunkRecommendation,
        )

        chunk_actions = set(
            typing.get_args(ChunkRecommendation.model_fields["action"].annotation)
        )
        assert chunk_actions <= set(_VALID_CHUNK_ACTIONS), (
            f"reviewer may propose {sorted(chunk_actions - set(_VALID_CHUNK_ACTIONS))}, "
            f"which ChunkDecisionItem would reject"
        )
        edge_actions = set(
            typing.get_args(ChunkEdgeRecommendation.model_fields["action"].annotation)
        )
        assert edge_actions <= set(_VALID_EDGE_ACTIONS)

    def test_chunk_edits_stay_inside_the_gates_whitelist(self):
        """`edited_*` fields map onto ChunkDecisionItem.edits keys, and the
        gate silently drops anything outside its whitelist. A recommendation
        naming a field it filters would apply in the UI and vanish on
        submit — worse than being refused, because it looks like it worked.
        """
        from app.schemas.api import _CHUNK_EDITABLE_FIELDS
        from app.services.reviewer.models import ChunkRecommendation

        edited = {
            name[len("edited_"):]
            for name in ChunkRecommendation.model_fields
            if name.startswith("edited_")
        }
        assert edited, "no editable fields declared"
        assert edited <= set(_CHUNK_EDITABLE_FIELDS), (
            f"reviewer may edit {sorted(edited - set(_CHUNK_EDITABLE_FIELDS))}, "
            f"which gate_chunks would filter out"
        )

    def test_reviewer_feedback_categories_exist(self):
        """Each gate's pinned-rule categories must be categories the
        flywheel actually files patterns under. A typo here is silent: the
        query returns nothing and the analyst's confirmed rules quietly stop
        reaching that gate."""
        from app.nodes.llm.feedback_synthesis import _AREA_FOR_CATEGORY
        from app.services.reviewer.gate_reviewers import REVIEWERS

        for key, reviewer in REVIEWERS.items():
            unknown = set(reviewer.feedback_categories) - set(_AREA_FOR_CATEGORY)
            assert not unknown, f"gate '{key}' names unknown categories: {sorted(unknown)}"

    def test_reviewer_gate_keys_are_real_gate_keys(self):
        from app.graph.state import GATE_KEYS
        from app.services.reviewer.gate_reviewers import REVIEWERS

        assert set(REVIEWERS) <= set(GATE_KEYS)

    def test_gate0_add_entity_dropdown_offers_only_real_entity_types(self):
        """The Add-Entity dropdown must match the backend EntityType enum.

        It used to carry 34 options of which 18 were STIX SCO type names
        ("ipv4-addr", "file", "domain-name") that are not EntityType values —
        and "ipv4-addr" was the default. Such an entity reaches
        validated_entities, then serialization finds no STIX mapping, logs a
        warning, and drops it. The analyst's addition never reaches the
        bundle and nothing tells them.

        Harmless while added entities were being discarded upstream. Not
        harmless once that channel works.
        """
        import re
        from pathlib import Path

        from app.graph.state import EntityType

        jsx = (
            Path(__file__).resolve().parents[1]
            / "frontend" / "src" / "components" / "Gate0Review.jsx"
        ).read_text()
        block = re.search(r"const ENTITY_TYPES = \[(.*?)\];", jsx, re.S)
        assert block, "ENTITY_TYPES list not found in Gate0Review.jsx"
        offered = set(re.findall(r'"([^"]+)"', block.group(1)))
        valid = {e.value for e in EntityType}
        assert offered <= valid, (
            f"dropdown offers non-EntityType values: {sorted(offered - valid)}"
        )

        # The default must be a real type too — it is what an analyst gets if
        # they add an entity without touching the selector.
        default = re.search(r'useState\(\{ type: "([^"]+)"', jsx)
        assert default, "add-entity default not found"
        assert default.group(1) in valid, (
            f"add-entity form defaults to {default.group(1)!r}, "
            f"which is not an EntityType"
        )

    def test_reviewer_tool_schema_matches_its_validator(self):
        """The reviewer tool and Gate0Recommendations must agree.

        Same failure shape as `source_excerpt` and `precondition` before it:
        a field added to a tool schema but not to the model that validates it
        surfaces only on a live LLM call, because that is the first time the
        model actually emits it. The reviewer's `sector` field was added
        exactly this way.
        """
        from app.services.reviewer.gate_reviewers import REVIEW_ENTITIES_TOOL
        from app.services.reviewer.models import (
            AddedEntityRecommendation,
            EntityRecommendation,
            Gate0Recommendations,
        )

        props = REVIEW_ENTITIES_TOOL["input_schema"]["properties"]
        assert set(props) <= set(Gate0Recommendations.model_fields), (
            "tool emits a top-level field the validator forbids"
        )

        from app.services.reviewer.gate_reviewers import REVIEW_PROCEDURES_TOOL
        from app.services.reviewer.models import (
            DraftRecommendation,
            Gate1Recommendations,
            TechniquePromotionRecommendation,
        )

        g1 = REVIEW_PROCEDURES_TOOL["input_schema"]["properties"]
        assert set(g1) <= set(Gate1Recommendations.model_fields)

        from app.services.reviewer.gate_reviewers import REVIEW_CHUNKS_TOOL
        from app.services.reviewer.models import (
            AddedChunkRecommendation,
            ChunkEdgeRecommendation,
            ChunkRecommendation,
            ChunkRerunRecommendation,
            GateChunksRecommendations,
        )

        gc = REVIEW_CHUNKS_TOOL["input_schema"]["properties"]
        assert set(gc) <= set(GateChunksRecommendations.model_fields)
        # `reject` is a single object, not a list — the only recommendation
        # channel that is, so it needs checking by hand rather than by the
        # items[] loop below.
        assert set(gc["reject"]["properties"]) <= set(
            ChunkRerunRecommendation.model_fields
        )
        assert set(gc["reject"].get("required", [])) <= set(
            ChunkRerunRecommendation.model_fields
        )

        for key, model, tool in (
            ("entities", EntityRecommendation, REVIEW_ENTITIES_TOOL),
            ("added_entities", AddedEntityRecommendation, REVIEW_ENTITIES_TOOL),
            ("chunks", ChunkRecommendation, REVIEW_CHUNKS_TOOL),
            ("added_chunks", AddedChunkRecommendation, REVIEW_CHUNKS_TOOL),
            ("edges", ChunkEdgeRecommendation, REVIEW_CHUNKS_TOOL),
            ("drafts", DraftRecommendation, REVIEW_PROCEDURES_TOOL),
            ("promotions", TechniquePromotionRecommendation, REVIEW_PROCEDURES_TOOL),
        ):
            props = tool["input_schema"]["properties"]
            emitted = set(props[key]["items"]["properties"])
            declared = set(model.model_fields)
            assert emitted <= declared, (
                f"{key}: tool emits {sorted(emitted - declared)} which "
                f"{model.__name__} (extra='forbid') would reject"
            )
            required = set(props[key]["items"].get("required", []))
            assert required <= declared, f"{key}: required field not on the model"

    def test_the_agreement_readout_understands_every_differ(self):
        """Feed each differ's real output through the aggregator.

        The readout walks `outcome.agreement.items[]` and groups by `kind` and
        `confidence`. Those keys are produced by the differs, in another
        module. A fifth gate whose differ emits an item shaped differently
        would not error — it would silently contribute nothing, and the dial
        that decides whether a gate can run unattended would quietly stop
        counting one of them.

        So this runs the actual differs rather than fixtures, and asserts every
        item they emit is one the aggregator scores.
        """
        from app.services.reviewer.outcomes import OUTCOME_DIFFERS
        from app.services.reviewer.agreement import aggregate

        rec = {
            "confidence": "high", "rationale": "r", "evidence_quote": "",
            "quote_unsupported": False,
        }
        # One recommendation per channel each differ reads, so no `kind` it can
        # emit goes unexercised.
        payloads = {
            "entities": {
                "entities": [{"entity_id": "e1", "action": "remove", **rec}],
                "added_entities": [{"value": "v", "entity_type": "tool", **rec}],
            },
            "chunks": {
                "chunks": [{"chunk_id": "c1", "action": "drop", **rec}],
                "added_chunks": [{"text": "t", **rec}],
                "edges": [{"action": "add", "from_chunk_id": "c1",
                           "to_chunk_id": "c2", **rec}],
                "reject": {"reason": "bad_flow", **rec},
            },
            "procedures": {
                "drafts": [{"draft_id": "d1", "action": "edit",
                            "remove_technique_ids": ["T1059"], **rec}],
                "promotions": [{"chunk_id": "c1", "technique_id": "T1105", **rec}],
            },
            "bundle": {
                "relationships": [{"rel_id": "relp_1", "action": "remove", **rec}],
            },
        }
        assert set(payloads) == set(OUTCOME_DIFFERS), (
            "a gate gained a differ but this test still exercises the old set"
        )

        import types

        for gate_key, payload in payloads.items():
            differ = OUTCOME_DIFFERS[gate_key]
            extras = {} if gate_key == "chunks" else []
            outcome = differ(payload, [], extras)
            items = outcome["agreement"]["items"]
            assert items, f"{gate_key}: differ produced nothing to score"

            row = types.SimpleNamespace(
                gate_key=gate_key, source_id="s", status="ok", outcome=outcome,
            )
            gate = aggregate([row])["gates"][0]

            scoreable = [i for i in items if not i.get("moot")]
            assert gate["total"] == len(scoreable), (
                f"{gate_key}: differ emitted {len(scoreable)} scoreable items, "
                f"the readout counted {gate['total']}"
            )
            # Every kind reached a bucket, and none landed in the catch-all.
            assert "unknown" not in gate["by_kind"], (
                f"{gate_key}: an item arrived with no `kind`"
            )
            assert set(gate["by_kind"]) == {i["kind"] for i in scoreable}
            # Every item carries the reasoning the overrides list renders.
            assert all("rationale" in i for i in items), (
                f"{gate_key}: items lack `rationale`, so an override would "
                f"show the analyst a disagreement with no reason attached"
            )

    def test_frontend_reject_reasons_match_the_enum(self):
        """The chunk canvas's reject dropdown must offer the real enum.

        It used to only have to match for the analyst's own choice, which is
        always one of the rendered options — drift would have been invisible
        but harmless. The AI pre-fill changed that: the reviewer's `reason`
        goes straight into the select's value, so a value the list does not
        carry renders blank and submits something nobody chose. The reason is
        what routes the re-chunk's guidance, and a re-chunk discards the whole
        pass, so getting it wrong is expensive.

        Same shape as the Gate 0 entity-type dropdown test above.
        """
        import re
        from pathlib import Path

        from app.graph.state import ChunkGateRejectReason

        jsx = (
            Path(__file__).resolve().parents[1]
            / "frontend" / "src" / "components" / "ChunkReviewCanvas.jsx"
        ).read_text()
        block = re.search(r"const REJECT_REASONS = \[(.*?)\];", jsx, re.S)
        assert block, "REJECT_REASONS list not found in ChunkReviewCanvas.jsx"
        offered = set(re.findall(r'value: "([^"]+)"', block.group(1)))
        assert offered == {e.value for e in ChunkGateRejectReason}

    def test_frontend_marks_the_same_gates_as_ai_reviewed(self):
        """lib/gates.js `aiReviewer: true` must match the backend registry.

        The frontend flag controls what the Add Source modal OFFERS. If it
        offers assist for a gate the backend has no reviewer for, the setting
        silently does nothing — a config that lies is worse than one that is
        missing.
        """
        import re
        from pathlib import Path

        from app.services.reviewer.gate_reviewers import REVIEWERS

        gates_js = (
            Path(__file__).resolve().parents[1]
            / "frontend" / "src" / "lib" / "gates.js"
        ).read_text()

        # Each GATES entry is a brace-delimited block; find the enableKey of
        # every block that also sets aiReviewer: true.
        marked = {
            m.group(1)
            for block in re.findall(r"\{[^{}]*\}", gates_js, re.S)
            if re.search(r"aiReviewer:\s*true", block)
            for m in [re.search(r'enableKey:\s*"([^"]+)"', block)]
            if m
        }
        assert marked == set(REVIEWERS), (
            f"frontend marks {sorted(marked)} as AI-reviewed, backend "
            f"implements {sorted(REVIEWERS)}"
        )


def _code_only(src: str) -> str:
    """Source with comments and docstrings removed.

    Searching raw source would let a COMMENT mentioning a field satisfy the
    check. That is not hypothetical: the first version of this test passed
    against the very bug it was written for, because `apply_procedures` kept
    a comment naming `remove_technique_ids` after the code that read it was
    gone. A test a mention can satisfy is measuring documentation.
    """
    import io
    import tokenize

    out = []
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING:
            body = tok.string.lstrip("rbfuRBFU")
            if body[:1] and body.startswith(body[0] * 3):
                continue  # docstring / triple-quoted block
        out.append(tok.string)
    return "\n".join(out)


class TestEveryRecommendationFieldIsActionable:
    """A field the reviewer can fill that no converter reads is dead weight.

    The existing direction — everything a converter EMITS must be accepted by
    the submit schema — is only half the contract. It says nothing about a
    field the reviewer populates and the converter silently ignores.

    That half cost a real unattended run: the reviewer asked to drop four
    techniques via `remove_technique_ids`, `apply_procedures`' whitelist did
    not include it, and the request vanished. Nothing failed. The only symptom
    was an auto outcome reading 8/12 — the agent recorded as disagreeing with
    itself, buried in a table nobody was watching.

    Static and crude on purpose: it asks whether the field NAME appears in
    apply.py at all. That cannot prove the field is used correctly, but it
    does prove somebody thought about it, which is the failure here.
    """

    # Fields that are legitimately not read by apply.py, each with a reason.
    EXEMPT = {
        # Written by the grounding check for audit, never forwarded to a gate.
        "confidence", "evidence_quote", "quote_source_support",
        "quote_unsupported",
        # Folded into `value` by enforce_sector_vocabulary before the
        # recommendation is ever stored, so apply.py only ever sees `value`.
        "sector",
        # Advisory prose for the analyst's benefit; no gate channel carries it.
        "overall_notes", "comments",
    }

    # The opening read is the reviewer's account of the report, shown to the
    # analyst and replayed into every later turn. It is not a gate decision,
    # so no converter should touch it. Exempted as a whole model rather than
    # field-by-field so it does not need editing when a field is added.
    EXEMPT_MODELS = {"InitialRead"}

    def test_no_recommendation_field_is_silently_dropped(self):
        import typing
        from pathlib import Path

        from pydantic import BaseModel

        from app.services.reviewer.gate_reviewers import REVIEWERS

        source = Path("backend/app/services/reviewer/apply.py")
        if not source.exists():  # running from backend/
            source = Path("app/services/reviewer/apply.py")
        text = _code_only(source.read_text())

        def item_models(model, seen=None):
            """Every nested recommendation model reachable from a container."""
            seen = seen if seen is not None else set()
            for field in model.model_fields.values():
                for arg in (field.annotation, *typing.get_args(field.annotation)):
                    if (isinstance(arg, type) and issubclass(arg, BaseModel)
                            and arg not in seen):
                        seen.add(arg)
                        item_models(arg, seen)
            return seen

        missing = {}
        for gate_key, reviewer in REVIEWERS.items():
            for m in item_models(reviewer.output_model):
                if m.__name__ in self.EXEMPT_MODELS:
                    continue
                for name in m.model_fields:
                    if name in self.EXEMPT or name in text:
                        continue
                    missing.setdefault(gate_key, []).append(f"{m.__name__}.{name}")

        assert not missing, (
            "recommendation fields no converter in apply.py reads — the "
            f"reviewer can fill these and nothing will happen: {missing}"
        )


class TestFeedbackCategoriesAreReadable:
    """Every category the synthesizer can emit must be read by some node.

    The synthesizer picks a pattern's category freely; each LLM node fetches a
    fixed tuple of categories. Nothing connected the two, and the failure is
    silent in the worst way: a real analyst correction is captured, persisted,
    embedded, and then never surfaced anywhere, with nothing logged.

    That is not hypothetical. `other` was in the emittable enum and in no
    node's tuple, so its two patterns had 0 lifetime surfacings — and one of
    them was `promoted_to_prompt`, an analyst-vetted "always apply" rule that
    reached nothing. `other` is also the *fallback* when the LLM omits a
    category (`category = (raw.get("category") or "other")`), so any malformed
    emission landed in the same black hole.
    """

    @staticmethod
    def _emittable() -> set[str]:
        from app.nodes.llm.feedback_synthesis import SYNTHESIZE_FEEDBACK_TOOL

        schema = SYNTHESIZE_FEEDBACK_TOOL["input_schema"]
        patterns = schema["properties"]["patterns"]
        return set(patterns["items"]["properties"]["category"]["enum"])

    @staticmethod
    def _readable() -> set[str]:
        from app.nodes.llm.chunking import _CHUNK_FEEDBACK_CATEGORIES
        from app.nodes.llm.drafting import _DRAFT_FEEDBACK_CATEGORIES
        from app.nodes.llm.entity_extraction import _ENTITY_FEEDBACK_CATEGORIES
        from app.nodes.llm.technique_extraction import _TECHNIQUE_FEEDBACK_CATEGORIES

        return (
            set(_ENTITY_FEEDBACK_CATEGORIES)
            | set(_CHUNK_FEEDBACK_CATEGORIES)
            | set(_TECHNIQUE_FEEDBACK_CATEGORIES)
            | set(_DRAFT_FEEDBACK_CATEGORIES)
        )

    def test_scan_finds_both_registries(self):
        assert len(self._emittable()) >= 10
        assert len(self._readable()) >= 10

    def test_every_emittable_category_is_read_by_some_node(self):
        unreachable = self._emittable() - self._readable()
        assert not unreachable, (
            f"categories the synthesizer can emit that NO node reads: "
            f"{sorted(unreachable)} — patterns filed here are persisted and "
            f"then never surfaced. Add each to a node tuple, or remove it from "
            f"SYNTHESIZE_FEEDBACK_TOOL's category enum."
        )

    def test_the_omitted_category_fallback_is_readable(self):
        """`other` is written by code, not just chosen by the LLM."""
        assert "other" in self._readable()

    def test_no_node_asks_for_a_category_that_cannot_exist(self):
        """The reverse drift: a node fetching a category nothing can produce
        is dead weight in the query and usually a typo or a rename."""
        orphaned = self._readable() - self._emittable()
        assert not orphaned, (
            f"node tuples request categories the synthesizer cannot emit: "
            f"{sorted(orphaned)}"
        )


class TestProcedureNamesDescribeTheAdversary:
    """A procedure name must carry the adversary's verb, not the vendor's.

    The Zerologon draft from one ransomware run came out "Discuss Zerologon as Speculative
    Initial Access Vector via CVE-2020-1472" on two separate runs. `Discuss` is
    what the report's AUTHORS do; the procedure is not about them. The naming
    convention is `[Verb] [Object] via [Tool/Method]`, verb-first and
    imperative.

    This got fixable once x_procedure_type could express `hypothetical`: the
    uncertainty now has a structured home, so the name neither hedges nor
    appends "(Unconfirmed)". A prompt that reintroduces either is drift.
    """

    REPORTING_VERBS = (
        "Discuss", "Report", "Note", "Describe",
        "Propose", "Identify", "Assess",
    )

    def test_naming_rule_forbids_reporting_verbs(self):
        from app.nodes.llm.drafting import SYSTEM_PROMPT

        assert "ADVERSARY'S action" in SYSTEM_PROMPT
        for verb in self.REPORTING_VERBS:
            assert verb in SYSTEM_PROMPT, (
                f"the naming rule no longer names {verb!r} as a forbidden "
                f"reporting verb"
            )

    def test_naming_rule_survives_the_hypothetical_case(self):
        """The one case where hedging in the name is most tempting."""
        from app.nodes.llm.drafting import SYSTEM_PROMPT

        assert "hypothetical" in SYSTEM_PROMPT
        assert "(Unconfirmed)" in SYSTEM_PROMPT

    def test_tool_schema_carries_the_same_rule(self):
        """The prompt and the per-field description are edited independently."""
        from app.nodes.llm.drafting import DRAFT_PROCEDURES_TOOL

        desc = (
            DRAFT_PROCEDURES_TOOL["input_schema"]["properties"]["drafts"]
            ["items"]["properties"]["name"]["description"]
        )
        assert "ADVERSARY'S action" in desc
        assert "Discuss" in desc


class TestSectionClassifierKnowsArtifactLists:
    """Hunting guidance that lists artifacts is indicator data, not detection logic.

    On one report the "threat detection and hunting" section was a bulleted
    list of a process tree, scheduled-task names, a mutex and a registry key,
    with no rule syntax, and the classifier filed it as detection_logic, which
    entity extraction skips on purpose (rule listings once typed six defender
    products as adversary tools). Every indicator in the list was lost and the
    AI reviewer proposed them all back at Gate 0. The definitions now draw the
    line at rule syntax, not at the heading. This pins the wording, because a
    prompt edit that reads as a tidy-up can reopen the loss.
    """

    def test_indicator_data_names_the_artifact_kinds(self):
        from app.nodes.llm.chunking import CLASSIFY_SYSTEM_PROMPT

        for term in ("mutex", "registry key", "scheduled-task", "process tree"):
            assert term in CLASSIFY_SYSTEM_PROMPT, term

    def test_detection_logic_is_scoped_to_rule_syntax_not_headings(self):
        from app.nodes.llm.chunking import CLASSIFY_SYSTEM_PROMPT

        assert "hunting" in CLASSIFY_SYSTEM_PROMPT.lower()
        assert "rule syntax" in CLASSIFY_SYSTEM_PROMPT
        assert "A heading does not make a section detection_logic" in CLASSIFY_SYSTEM_PROMPT

    def test_entity_extraction_still_skips_detection_logic(self):
        """The guard the definitions protect: detection_logic stays excluded
        from entity extraction, indicator_data stays included."""
        from app.nodes.llm.entity_extraction import _ENTITY_EXCLUDED_SECTIONS

        assert "detection_logic" in _ENTITY_EXCLUDED_SECTIONS
        assert "indicator_data" not in _ENTITY_EXCLUDED_SECTIONS


class TestVendoredRegionVocabIsIntact:
    """The vendored region-ov had two entries with a missing comma, silently
    merging four values into two, plus one absent outright — 26 values where
    STIX 2.1 defines 29. No runtime effect today (the `region` property is a
    bare string and the ov is unused), but anything that later validates
    against it would reject legitimate values.
    """

    SPEC = {
        "africa", "eastern-africa", "middle-africa", "northern-africa",
        "southern-africa", "western-africa", "americas",
        "latin-america-caribbean", "south-america", "caribbean",
        "central-america", "northern-america", "asia", "central-asia",
        "eastern-asia", "southern-asia", "south-eastern-asia", "western-asia",
        "europe", "eastern-europe", "northern-europe", "southern-europe",
        "western-europe", "oceania", "antarctica", "australia-new-zealand",
        "melanesia", "micronesia", "polynesia",
    }

    @staticmethod
    def _ov() -> list[str]:
        import json
        from pathlib import Path

        import app
        path = (
            Path(app.__file__).parent / "schemas" / "stix21" / "sdos" / "location.json"
        )
        return json.loads(path.read_text())["definitions"]["region-ov"]["enum"]

    def test_no_value_contains_a_space(self):
        """A space is the signature of the missing-comma corruption."""
        assert [v for v in self._ov() if " " in v] == []

    def test_matches_the_stix_21_vocabulary_exactly(self):
        assert set(self._ov()) == self.SPEC
