"""Tests for full STIX 2.1 schema validation (app.services.stix_schema).

This is the machinery that makes x_procedure_v3.json and the OASIS schemas
load-bearing at runtime. Before it existed, `_validate_schema` was a
hand-rolled required-fields check and two contract violations lived in the
serializer undetected for months.
"""

import pytest

from app.services import stix_schema


VALID_IDENTITY = {
    "type": "identity",
    "spec_version": "2.1",
    "id": "identity--11111111-1111-4111-8111-111111111111",
    "created": "2024-01-01T00:00:00.000Z",
    "modified": "2024-01-01T00:00:00.000Z",
    "name": "Acme CTI",
}

VALID_PROCEDURE = {
    "type": "x-procedure",
    "spec_version": "2.1",
    "id": "x-procedure--22222222-2222-4222-8222-222222222222",
    "created": "2024-01-01T00:00:00.000Z",
    "modified": "2024-01-01T00:00:00.000Z",
    "name": "Delete Volume Shadow Copies via WMIC",
}


class TestSchemaCorpus:

    def test_corpus_loads(self):
        assert stix_schema.is_available(), (
            "vendored OASIS schemas failed to load — Phase 2 validation would "
            "silently degrade to the hand-rolled structural checks"
        )

    def test_indexes_stock_and_custom_types(self):
        by_type = stix_schema._get()["by_type"]
        # A spread of what the serializer actually emits.
        for t in ("identity", "relationship", "process", "malware", "report",
                  "windows-registry-key", "ipv4-addr", "vulnerability"):
            assert t in by_type, f"no schema indexed for emitted type {t!r}"
        assert "x-procedure" in by_type, "our own custom object must be checked too"


class TestStockTypeValidation:

    def test_valid_object_passes(self):
        assert stix_schema.validate_object(VALID_IDENTITY) == []

    def test_missing_required_field_is_caught(self):
        obj = {k: v for k, v in VALID_IDENTITY.items() if k != "name"}
        errors = stix_schema.validate_object(obj)
        assert any("name" in e for e in errors), errors

    def test_malformed_identifier_is_caught(self):
        """The placeholder-style IDs that used to sail through."""
        errors = stix_schema.validate_object({**VALID_IDENTITY, "id": "identity--src"})
        assert errors, "an id without a UUID suffix is not a valid STIX identifier"

    def test_timestamp_without_millisecond_precision_is_caught(self):
        errors = stix_schema.validate_object(
            {**VALID_IDENTITY, "created": "2024-01-01T00:00:00Z"}
        )
        assert errors

    def test_spec_legal_custom_property_is_allowed(self):
        """STIX 2.1 §11.3 permits x_-prefixed custom properties on stock
        types. The serializer emits x_source_id on identity and x_exe_name
        on process; neither may be treated as an error."""
        assert stix_schema.validate_object(
            {**VALID_IDENTITY, "x_source_id": "src-001"}
        ) == []


class TestProcedureValidation:

    def test_valid_procedure_passes(self):
        assert stix_schema.validate_object(VALID_PROCEDURE) == []

    def test_undeclared_property_is_caught(self):
        """x-procedure is additionalProperties:false — this is the exact
        drift (x_observable_refs) that motivated the whole exercise."""
        errors = stix_schema.validate_object(
            {**VALID_PROCEDURE, "x_observable_refs": ["file--3333"]}
        )
        assert any("x_observable_refs" in e for e in errors), errors

    def test_fabricated_technique_ref_is_caught(self):
        """The other drift: `attack-pattern--T1190` is not a STIX id."""
        errors = stix_schema.validate_object(
            {**VALID_PROCEDURE, "x_technique_refs": ["attack-pattern--T1190"]}
        )
        assert errors

    def test_declared_provenance_enum_is_enforced(self):
        assert stix_schema.validate_object(
            {**VALID_PROCEDURE, "x_source_provenance": "figure"}
        ) == []
        assert stix_schema.validate_object(
            {**VALID_PROCEDURE, "x_source_provenance": "invented"}
        )


class TestSkippedTypes:

    @pytest.mark.parametrize("t", ["attack-flow", "attack-operator", "attack-condition"])
    def test_attack_flow_extension_types_are_not_schema_checked(self, t):
        """CTID Attack Flow objects aren't core STIX; OASIS ships no schema.
        They must not be reported as violations just for existing."""
        assert stix_schema.validate_object({"type": t, "id": f"{t}--x"}) == []

    def test_unknown_custom_type_is_skipped(self):
        assert stix_schema.validate_object({"type": "x-made-up", "id": "x-made-up--1"}) == []

    def test_object_without_type_is_skipped(self):
        assert stix_schema.validate_object({"id": "whatever"}) == []


class TestDegradedMode:

    def test_unavailable_corpus_reports_no_errors(self, monkeypatch):
        """A packaging fault must not fail every bundle. Bundle validation is
        all-or-nothing and hard-fails the source, so a broken validator would
        take the whole pipeline down; Phase 1 stays the floor instead."""
        monkeypatch.setattr(stix_schema, "_state", {"ok": False})
        try:
            assert stix_schema.validate_object({**VALID_PROCEDURE, "bogus": 1}) == []
        finally:
            stix_schema.reset_cache()

    def test_bundle_helper_aggregates(self):
        errors = stix_schema.validate_bundle_objects([
            VALID_IDENTITY,
            {**VALID_PROCEDURE, "x_observable_refs": ["file--3333"]},
        ])
        assert len(errors) >= 1
        assert all(isinstance(e, str) for e in errors)


class TestExtensionDefinitionValidation:
    """The definitions the serializer embeds are checked like any other object."""

    def test_extension_definition_is_indexed(self):
        assert "extension-definition" in stix_schema._get()["by_type"]

    def test_published_definitions_pass(self):
        from app.nodes.deterministic.extension_definitions import (
            attack_flow_extension_definition,
            x_procedure_extension_definition,
        )
        for d in (x_procedure_extension_definition(), attack_flow_extension_definition()):
            assert stix_schema.validate_object(d) == [], d["id"]

    def test_definition_missing_schema_is_caught(self):
        from app.nodes.deterministic.extension_definitions import x_procedure_extension_definition
        d = x_procedure_extension_definition()
        del d["schema"]
        errors = stix_schema.validate_object(d)
        assert any("schema" in e for e in errors), errors

    def test_procedure_extension_declaration_shape_is_enforced(self):
        from app.nodes.deterministic.extension_definitions import (
            X_PROCEDURE_EXTENSION_ID,
            extension_declaration,
        )
        assert stix_schema.validate_object(
            {**VALID_PROCEDURE, "extensions": extension_declaration(X_PROCEDURE_EXTENSION_ID)}
        ) == []
        # Keys must be extension-definition ids; values must say new-sdo.
        assert stix_schema.validate_object(
            {**VALID_PROCEDURE, "extensions": {"made-up-ext": {"extension_type": "new-sdo"}}}
        )
        assert stix_schema.validate_object(
            {**VALID_PROCEDURE, "extensions": {X_PROCEDURE_EXTENSION_ID: {"extension_type": "new-sco"}}}
        )
