"""Unit tests for the distribute node."""

import pytest
import pytest_asyncio
from dataclasses import asdict

from app.graph.state import PipelineStatus
from app.nodes.deterministic.distribution import (
    CATALOGUE_OWNED_TYPES,
    PIPELINE_MARKER,
    distribute,
    _build_cypher_statements,
    _build_describes_queries,
    _build_node_query,
    _build_rel_query,
    _execute_writes,
    _STIX_TYPE_TO_LABEL,
    _REL_TYPE_TO_NEO4J,
)


# ── distribute node function ──────────────────────────────────────


class TestDistribute:
    """Tests for the top-level LangGraph node function."""

    @pytest.mark.asyncio
    async def test_valid_bundle_succeeds(self):
        """Valid bundle with passing validation produces success."""
        state = {
            "stix_bundle": {
                "type": "bundle",
                "id": "bundle--test",
                "objects": [
                    {
                        "type": "identity",
                        "id": "identity--001",
                        "created": "2024-01-01T00:00:00Z",
                        "modified": "2024-01-01T00:00:00Z",
                        "name": "Test Author",
                    },
                    {
                        "type": "intrusion-set",
                        "id": "intrusion-set--001",
                        "created": "2024-01-01T00:00:00Z",
                        "modified": "2024-01-01T00:00:00Z",
                        "name": "APT29",
                    },
                ],
            },
            "validation_results": {
                "schema": True,
                "reference_integrity": True,
                "attack_flow": True,
            },
        }
        result = await distribute(state)

        assert result["neo4j_write_status"] == "success"
        assert result["objects_written"] == 2
        assert result["status"] == PipelineStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_failed_validation_blocks_write(self):
        """If any validation failed, distribute refuses to write."""
        state = {
            "stix_bundle": {
                "type": "bundle",
                "id": "bundle--test",
                "objects": [{"type": "malware", "id": "malware--001"}],
            },
            "validation_results": {
                "schema": True,
                "reference_integrity": False,
                "attack_flow": True,
            },
        }
        result = await distribute(state)

        assert result["status"] == PipelineStatus.FAILED.value
        assert "reference_integrity" in result["neo4j_write_status"]
        assert result["objects_written"] == 0

    @pytest.mark.asyncio
    async def test_empty_bundle(self):
        """Empty bundle produces zero writes but doesn't fail."""
        state = {
            "stix_bundle": {"type": "bundle", "id": "bundle--test", "objects": []},
            "validation_results": {"schema": True, "reference_integrity": True, "attack_flow": True},
        }
        result = await distribute(state)

        assert result["objects_written"] == 0
        assert result["status"] == PipelineStatus.COMPLETED.value

    @pytest.mark.asyncio
    async def test_mixed_objects_counted(self):
        """Both nodes and relationships are counted."""
        state = {
            "stix_bundle": {
                "type": "bundle",
                "id": "bundle--test",
                "objects": [
                    {"type": "malware", "id": "malware--001", "name": "Test", "created": "2024-01-01", "modified": "2024-01-01"},
                    {"type": "tool", "id": "tool--001", "name": "certutil", "created": "2024-01-01", "modified": "2024-01-01"},
                    {
                        "type": "relationship",
                        "id": "relationship--001",
                        "relationship_type": "uses",
                        "source_ref": "malware--001",
                        "target_ref": "tool--001",
                    },
                ],
            },
            "validation_results": {"schema": True, "reference_integrity": True, "attack_flow": True},
        }
        result = await distribute(state)
        assert result["objects_written"] == 3

    @pytest.mark.asyncio
    async def test_all_validation_false_blocks(self):
        """All validations failing still blocks write."""
        state = {
            "stix_bundle": {"type": "bundle", "id": "bundle--test", "objects": []},
            "validation_results": {"schema": False, "reference_integrity": False, "attack_flow": False},
        }
        result = await distribute(state)
        assert result["status"] == PipelineStatus.FAILED.value


# ── _build_cypher_statements ──────────────────────────────────────


class TestResolveDisplayTitle:
    """One ladder for the bundle row, the Report SDO and the flow name."""

    def test_state_title_wins(self):
        from app.graph.state import resolve_display_title

        state = {"title": " Example Report ", "metadata": {"title": "meta"}}
        assert resolve_display_title(state, "d") == "Example Report"

    def test_placeholder_falls_through_to_metadata_then_default(self):
        from app.graph.state import resolve_display_title

        assert resolve_display_title(
            {"title": "Untitled Source", "metadata": {"title": "meta"}}, "d",
        ) == "meta"
        assert resolve_display_title({"title": "Untitled Source"}, "d") == "d"
        assert resolve_display_title({"metadata": None}, "d") == "d"

    @pytest.mark.asyncio
    async def test_bundle_row_gets_the_state_title(self):
        """distribute keeps passing the resolved title to save_bundle."""
        from unittest.mock import AsyncMock, MagicMock, patch

        state = {
            "title": "Royal ransomware advisory",
            "metadata": {},
            "stix_bundle": {"type": "bundle", "id": "bundle--t", "objects": [
                {"type": "identity", "id": "identity--001",
                 "created": "2024-01-01T00:00:00Z", "modified": "2024-01-01T00:00:00Z",
                 "name": "Test Author"},
            ]},
            "validation_results": {"schema": True, "reference_integrity": True,
                                   "attack_flow": True},
        }
        session = MagicMock()
        session.__aenter__ = AsyncMock(return_value=session)
        session.__aexit__ = AsyncMock(return_value=False)
        with patch("app.nodes.deterministic.distribution.async_session", return_value=session), \
             patch("app.nodes.deterministic.distribution.bundle_store.save_bundle",
                   new=AsyncMock()) as save:
            await distribute(state)
        assert save.await_args.kwargs["title"] == "Royal ransomware advisory"


class TestBuildCypherStatements:
    """Tests for Cypher statement generation."""

    def test_separates_nodes_and_rels(self):
        """Nodes and relationships are split into separate lists."""
        objects = [
            {"type": "malware", "id": "malware--001", "name": "Test"},
            {
                "type": "relationship",
                "id": "relationship--001",
                "relationship_type": "uses",
                "source_ref": "malware--001",
                "target_ref": "tool--001",
            },
        ]
        node_q, rel_q, describes_q = _build_cypher_statements(objects)
        assert len(node_q) == 1
        assert len(rel_q) == 1
        assert describes_q == []          # no Report SDO in this bundle

    def test_unknown_type_skipped(self):
        """Objects with unmapped STIX types are skipped."""
        objects = [
            {"type": "some-future-type", "id": "some-future-type--001", "name": "X"},
        ]
        node_q, rel_q, describes_q = _build_cypher_statements(objects)
        assert len(node_q) == 0
        assert len(rel_q) == 0
        assert len(describes_q) == 0


# ── _build_node_query ─────────────────────────────────────────────


class TestBuildNodeQuery:
    """Tests for individual node Cypher query building."""

    def test_malware_query(self):
        obj = {
            "type": "malware",
            "spec_version": "2.1",
            "id": "malware--abc",
            "name": "Cobalt Strike",
            "is_family": True,
            "created": "2024-01-01",
            "modified": "2024-01-01",
        }
        query = _build_node_query(obj)
        assert query is not None
        assert "MERGE" in query["query"]
        assert "Malware" in query["query"]
        assert query["params"]["stix_id"] == "malware--abc"
        # Properties travel as a map, not as interpolated key names.
        assert query["params"]["props"]["name"] == "Cobalt Strike"
        assert query["params"]["stix_type"] == "malware"

    def test_sco_query(self):
        obj = {
            "type": "ipv4-addr",
            "spec_version": "2.1",
            "id": "ipv4-addr--abc",
            "value": "203.0.113.10",
        }
        query = _build_node_query(obj)
        assert query is not None
        assert "IPv4Addr" in query["query"]
        assert query["params"]["props"]["value"] == "203.0.113.10"

    def test_procedure_query(self):
        obj = {
            "type": "x-procedure",
            "spec_version": "2.1",
            "id": "x-procedure--abc",
            "name": "Test Procedure",
            "created": "2024-01-01",
            "modified": "2024-01-01",
            "x_platforms": ["windows::server"],
        }
        query = _build_node_query(obj)
        assert query is not None
        assert "Procedure" in query["query"]

    def test_no_id_returns_none(self):
        obj = {"type": "malware", "name": "Test"}
        query = _build_node_query(obj)
        assert query is None

    def test_dict_properties_serialized(self):
        """Dict properties are JSON-serialized for Neo4j."""
        obj = {
            "type": "malware",
            "id": "malware--abc",
            "name": "Test",
            "hashes": {"SHA-256": "abc123"},
        }
        query = _build_node_query(obj)
        # hashes should be JSON string, not dict
        assert isinstance(query["params"]["props"]["hashes"], str)

    def test_skip_properties_excluded(self):
        """type, spec_version, id are not in SET clause params."""
        obj = {
            "type": "tool",
            "spec_version": "2.1",
            "id": "tool--abc",
            "name": "certutil",
        }
        query = _build_node_query(obj)
        # spec_version and type are meta-fields, not graph properties
        assert "spec_version" not in query["params"]["props"]
        assert "type" not in query["params"]["props"]


# ── _build_rel_query ──────────────────────────────────────────────


class TestBuildRelQuery:
    """Tests for relationship Cypher query building."""

    def test_uses_relationship(self):
        obj = {
            "type": "relationship",
            "id": "relationship--abc",
            "relationship_type": "uses",
            "source_ref": "intrusion-set--001",
            "target_ref": "x-procedure--001",
            "created": "2024-01-01",
            "modified": "2024-01-01",
        }
        query = _build_rel_query(obj)
        assert query is not None
        assert "USES" in query["query"]
        assert "MATCH" in query["query"]
        assert "MERGE" in query["query"]
        assert query["params"]["source_ref"] == "intrusion-set--001"

    def test_precedes_relationship(self):
        obj = {
            "type": "relationship",
            "id": "relationship--xyz",
            "relationship_type": "precedes",
            "source_ref": "x-procedure--001",
            "target_ref": "x-procedure--002",
        }
        query = _build_rel_query(obj)
        assert "PRECEDES" in query["query"]

    def test_missing_refs_returns_none(self):
        obj = {
            "type": "relationship",
            "id": "relationship--abc",
            "relationship_type": "uses",
            "source_ref": "",
            "target_ref": "",
        }
        query = _build_rel_query(obj)
        assert query is None

    def test_unknown_rel_type_uppercased(self):
        """Unknown relationship types are uppercased with hyphens replaced."""
        obj = {
            "type": "relationship",
            "id": "relationship--abc",
            "relationship_type": "derived-from",
            "source_ref": "a--001",
            "target_ref": "b--001",
        }
        query = _build_rel_query(obj)
        assert "DERIVED_FROM" in query["query"]


# ── Label/type mappings ───────────────────────────────────────────


class TestMappings:
    """Tests for STIX-to-Neo4j mapping coverage."""

    def test_all_sdo_types_mapped(self):
        """Key SDO types have Neo4j labels."""
        expected = [
            "identity", "intrusion-set", "threat-actor", "malware", "tool",
            "campaign", "vulnerability", "location", "indicator",
            "infrastructure", "x-procedure",
        ]
        for t in expected:
            assert t in _STIX_TYPE_TO_LABEL, f"Missing label for {t}"

    def test_all_sco_types_mapped(self):
        """Key SCO types have Neo4j labels."""
        expected = [
            "ipv4-addr", "ipv6-addr", "domain-name", "url", "email-addr",
            "file", "windows-registry-key", "mutex", "software", "user-account",
        ]
        for t in expected:
            assert t in _STIX_TYPE_TO_LABEL, f"Missing label for {t}"

    def test_relationship_types_mapped(self):
        """Key relationship types are mapped."""
        expected = ["uses", "precedes", "attributed-to", "indicates"]
        for t in expected:
            assert t in _REL_TYPE_TO_NEO4J, f"Missing mapping for {t}"


# ── The catalogue must survive a write ────────────────────────────


class TestCatalogueIsNotOverwritten:
    """The ATT&CK catalogue is the one thing in the graph with no undo.

    The serializer re-emits catalogue objects so the bundle stands alone — one
    real bundle carried 241 of them — and `MERGE ... SET` would have replaced
    ATT&CK's own created/modified/name/description with a pipeline run's
    values. For T1074.001 that meant overwriting `created 2020-03-13` with the
    run timestamp.
    """

    @pytest.mark.parametrize("stix_type", sorted(CATALOGUE_OWNED_TYPES))
    def test_catalogue_owned_types_emit_no_node_query(self, stix_type):
        objects = [{"type": stix_type, "id": f"{stix_type}--abc", "name": "T"}]
        node_q, _rel_q, _desc_q = _build_cypher_statements(objects)
        assert node_q == []

    def test_their_relationships_are_still_written(self):
        """The node is the catalogue's; the edge to it is ours."""
        objects = [
            {"type": "attack-pattern", "id": "attack-pattern--abc", "name": "T"},
            {"type": "relationship", "id": "relationship--1",
             "relationship_type": "uses",
             "source_ref": "x-procedure--1", "target_ref": "attack-pattern--abc"},
        ]
        node_q, rel_q, _ = _build_cypher_statements(objects)
        assert node_q == [] and len(rel_q) == 1

    def test_an_existing_node_is_only_updated_if_we_created_it(self):
        q = _build_node_query({"type": "tool", "id": "tool--abc", "name": "x"})
        assert "ON CREATE SET" in q["query"]
        assert "ON MATCH SET" in q["query"]
        # The ON MATCH branch is guarded by the marker, so a node this
        # pipeline did not create is left alone.
        on_match = q["query"].split("ON MATCH SET")[1]
        assert "CASE WHEN n.x_ingested_by = $marker" in on_match
        assert q["params"]["marker"] == PIPELINE_MARKER

    def test_created_nodes_are_marked_as_ours(self):
        q = _build_node_query({"type": "tool", "id": "tool--abc", "name": "x"})
        assert "n.x_ingested_by = $marker" in q["query"].split("ON MATCH")[0]


# ── Graph shape parity with the catalogue loader ──────────────────


class TestNodesMatchTheCatalogueShape:
    def test_dual_label_so_stix_object_queries_and_the_index_reach_us(self):
        """`stix_id`'s index is scoped to :STIXObject. A single-labelled node
        is both invisible to those queries and unable to use it."""
        q = _build_node_query({"type": "tool", "id": "tool--abc", "name": "x"})
        assert "MERGE (n:Tool:STIXObject {stix_id: $stix_id})" in q["query"]

    def test_stix_type_is_written_like_the_loader_writes_it(self):
        q = _build_node_query({"type": "x-procedure", "id": "x-procedure--a",
                               "name": "P"})
        assert q["params"]["stix_type"] == "x-procedure"
        assert "n.stix_type = $stix_type" in q["query"]

    def test_relationship_endpoints_match_on_the_indexed_label(self):
        q = _build_rel_query({
            "type": "relationship", "id": "relationship--1",
            "relationship_type": "precedes",
            "source_ref": "x-procedure--1", "target_ref": "x-procedure--2",
        })
        assert "(a:STIXObject {stix_id: $source_ref})" in q["query"]
        assert "(b:STIXObject {stix_id: $target_ref})" in q["query"]

    def test_property_names_are_data_not_query_text(self):
        """`n += $props` means a property key can no longer reach the Cypher
        string at all, so key interpolation is not an injection surface."""
        q = _build_node_query({"type": "tool", "id": "tool--abc",
                               "name": "x", "description": "y"})
        assert "n += $props" in q["query"]
        assert "description" not in q["query"]
        assert q["params"]["props"]["description"] == "y"


# ── The technique edge ────────────────────────────────────────────


class TestTechniqueEdgeIsNamed:
    """`USES` is overloaded: 18,000+ catalogue edges plus procedure->tool and
    procedure->malware. The clustering query joins two procedures through a
    shared endpoint, so under `USES` it reads "both used PsExec" as technique
    overlap."""

    def test_procedure_to_technique_becomes_implements_technique(self):
        q = _build_rel_query({
            "type": "relationship", "id": "relationship--1",
            "relationship_type": "uses",
            "source_ref": "x-procedure--1", "target_ref": "attack-pattern--2",
        })
        assert "IMPLEMENTS_TECHNIQUE" in q["query"]
        assert ":USES" not in q["query"]

    def test_procedure_to_tool_stays_uses(self):
        q = _build_rel_query({
            "type": "relationship", "id": "relationship--1",
            "relationship_type": "uses",
            "source_ref": "x-procedure--1", "target_ref": "tool--2",
        })
        assert "r:USES" in q["query"]
        assert "IMPLEMENTS_TECHNIQUE" not in q["query"]

    def test_actor_to_technique_stays_uses(self):
        """Only the procedure's own linkage is renamed — an intrusion-set
        using a technique is the catalogue's existing USES semantics."""
        q = _build_rel_query({
            "type": "relationship", "id": "relationship--1",
            "relationship_type": "uses",
            "source_ref": "intrusion-set--1", "target_ref": "attack-pattern--2",
        })
        assert "r:USES" in q["query"]


# ── Provenance ────────────────────────────────────────────────────


class TestDescribesEdges:
    REPORT = {
        "type": "report", "id": "report--r1", "name": "Vendor writeup",
        "object_refs": ["x-procedure--1", "attack-pattern--2", "tool--3",
                        "relationship--4", "x-procedure--absent"],
    }

    def _objects(self):
        return [
            self.REPORT,
            {"type": "x-procedure", "id": "x-procedure--1", "name": "P"},
            {"type": "attack-pattern", "id": "attack-pattern--2", "name": "T"},
            {"type": "tool", "id": "tool--3", "name": "certutil"},
        ]

    def test_one_edge_per_contributed_object(self):
        qs = _build_describes_queries(self._objects())
        refs = {q["params"]["object_id"] for q in qs}
        # relationship refs are edges, not nodes; the absent ref is not in the
        # bundle so there is nothing to point at.
        assert refs == {"x-procedure--1", "attack-pattern--2", "tool--3"}

    def test_catalogue_objects_are_described_too(self):
        """A report does describe the techniques it covers, even though we do
        not own those nodes. This is why an undo has to filter on
        x_ingested_by rather than on DESCRIBES alone."""
        qs = _build_describes_queries(self._objects())
        assert any(q["params"]["object_id"] == "attack-pattern--2" for q in qs)

    def test_no_report_means_no_edges(self):
        assert _build_describes_queries(
            [{"type": "tool", "id": "tool--3", "name": "x"}]) == []

    def test_a_report_does_not_describe_itself(self):
        qs = _build_describes_queries([
            {"type": "report", "id": "report--r1",
             "object_refs": ["report--r1", "tool--3"]},
            {"type": "tool", "id": "tool--3", "name": "x"},
        ])
        assert [q["params"]["object_id"] for q in qs] == ["tool--3"]


# ── Dangling relationships ────────────────────────────────────────


class TestDanglingEdgesAreCounted:
    """`MATCH a, b MERGE (a)-[r]->(b)` with a missing endpoint returns zero
    rows and does nothing. No error, no warning, the transaction commits. The
    edge is simply absent, and nothing anywhere records it."""

    @staticmethod
    def _rel(i):
        return {"query": "MATCH ... RETURN 1",
                "params": {"stix_id": f"relationship--{i}",
                           "source_ref": "x-procedure--1",
                           "target_ref": f"attack-pattern--{i}"}}

    async def test_a_zero_row_edge_is_reported_not_counted_as_written(self, monkeypatch):
        from app.nodes.deterministic import distribution as dist
        monkeypatch.setattr(dist.settings, "neo4j_writes_enabled", True)

        async def fake_nodes(queries):
            return len(queries)

        async def fake_counted(queries):
            # second edge finds no endpoint
            return [1 if i != 1 else 0 for i in range(len(queries))]

        monkeypatch.setattr(dist, "write_transaction", fake_nodes)
        monkeypatch.setattr(dist, "write_transaction_counted", fake_counted)

        result = await _execute_writes([], [self._rel(0), self._rel(1)], [])
        assert result["written"] == 1
        assert result["dropped"] == 1
        assert "attack-pattern--1" in result["dropped_detail"][0]

    async def test_all_landed_reports_nothing(self, monkeypatch):
        from app.nodes.deterministic import distribution as dist
        monkeypatch.setattr(dist.settings, "neo4j_writes_enabled", True)

        async def fake_counted(queries):
            return [1] * len(queries)

        monkeypatch.setattr(dist, "write_transaction_counted", fake_counted)
        result = await _execute_writes([], [self._rel(0)], [])
        assert result == {"written": 1, "dropped": 0, "dropped_detail": []}

    async def test_dry_run_executes_nothing(self, monkeypatch):
        from app.nodes.deterministic import distribution as dist
        monkeypatch.setattr(dist.settings, "neo4j_writes_enabled", False)
        called = []
        monkeypatch.setattr(dist, "write_transaction_counted",
                            lambda q: called.append(q))
        result = await _execute_writes([], [self._rel(0)], [self._rel(1)])
        assert called == []
        assert result["written"] == 2 and result["dropped"] == 0


class TestCatalogueInternalEdgesAreNotDuplicated:
    """ATT&CK's own detection chain is re-emitted by the serializer so the
    bundle stands alone. Writing it creates edges parallel to the catalogue's
    (MERGE keys on stix_id, which the catalogue's edges lack) — and undo
    cannot reach them, because DETACH DELETE only touches edges on a node
    being deleted and both endpoints belong to ATT&CK. One bundle left 446."""

    @pytest.mark.parametrize("rel,src,tgt", [
        ("detects", "x-mitre-detection-strategy--1", "attack-pattern--2"),
        ("has-analytic", "x-mitre-detection-strategy--1", "x-mitre-analytic--2"),
        ("uses-data-component", "x-mitre-analytic--1", "x-mitre-data-component--2"),
    ])
    def test_catalogue_to_catalogue_edges_are_skipped(self, rel, src, tgt):
        assert _build_rel_query({
            "type": "relationship", "id": "relationship--1",
            "relationship_type": rel, "source_ref": src, "target_ref": tgt,
        }) is None

    def test_an_edge_with_one_catalogue_endpoint_is_still_written(self):
        """procedure -> technique is ours to make; only the technique is not."""
        q = _build_rel_query({
            "type": "relationship", "id": "relationship--1",
            "relationship_type": "uses",
            "source_ref": "x-procedure--1", "target_ref": "attack-pattern--2",
        })
        assert q is not None and "IMPLEMENTS_TECHNIQUE" in q["query"]


# ── bundle metadata is never written ─────────────────────────────


class TestBundleMetadataIsNotWritten:
    """Extension definitions and their author identities are about the object
    types, not the intrusion. They get no node, no warning, and stay outside
    the undo boundary (the Report never lists them)."""

    def test_definitions_and_their_authors_get_no_node_query(self, caplog):
        from app.nodes.deterministic.extension_definitions import (
            attack_flow_author_identity,
            attack_flow_extension_definition,
            x_procedure_author_identity,
            x_procedure_extension_definition,
        )
        objects = [
            x_procedure_author_identity(),
            x_procedure_extension_definition(),
            attack_flow_author_identity(),
            attack_flow_extension_definition(),
            {"type": "malware", "id": "malware--001", "name": "Real content"},
        ]
        with caplog.at_level("WARNING"):
            node_q, rel_q, describes_q = _build_cypher_statements(objects)
        assert len(node_q) == 1
        assert node_q[0]["params"]["stix_id"] == "malware--001" or "malware--001" in str(node_q[0])
        assert rel_q == [] and describes_q == []
        assert "No Neo4j label" not in caplog.text

    def test_an_ordinary_identity_is_still_written(self):
        objects = [{"type": "identity", "id": "identity--001", "name": "Report author"}]
        node_q, _, _ = _build_cypher_statements(objects)
        assert len(node_q) == 1
