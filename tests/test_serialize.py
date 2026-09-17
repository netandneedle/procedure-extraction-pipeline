"""Unit tests for the serialize_stix node."""

import re

from unittest.mock import AsyncMock, patch

import pytest
from dataclasses import asdict

from app.graph.state import (
    EntityType,
    GateAction,
    NormalizedDraft,
    PipelineStatus,
)
from app.nodes.deterministic.serialization import (
    _build_process_sco,
    serialize_stix,
    _entity_to_stix,
    _draft_to_procedure,
    _build_relationships,
    _detect_hash_type,
    _validate_bundle,
    _create_source_identity,
    X_PROCEDURE_TYPE,
    _SCO_TYPES,
)


# ── serialize_stix node function ──────────────────────────────────


class TestSerializeStix:
    """Tests for the top-level LangGraph node function."""

    async def test_full_state_produces_bundle(self, base_state):
        """Full pipeline state produces a valid STIX bundle."""
        # Build normalized_drafts from sample drafts
        normalized = [
            asdict(NormalizedDraft(
                draft_id=d["draft_id"],
                composite_confidence=80,
                confidence_breakdown={},
                standardized_names={},
            ))
            for d in base_state["drafts"]
        ]
        state = {**base_state, "normalized_drafts": normalized}

        result = await serialize_stix(state)

        assert result["status"] == PipelineStatus.SERIALIZING.value
        assert result["current_node"] == "serialize_stix"
        assert "stix_bundle" in result

        bundle = result["stix_bundle"]
        assert bundle["type"] == "bundle"
        assert len(bundle["objects"]) > 0

    async def test_raw_command_lines_become_process_scos(self, base_state):
        """A draft carrying raw_command_lines yields one Process SCO per
        command and a populated x_components_refs on its procedure.

        Every other test hands the serializer a draft with the commands
        already in place; this pins the full-node path, which is the one
        that shipped bundles with x_components_refs empty on 8 of 9
        procedures when drafting stopped supplying them.
        """
        normalized = [
            asdict(NormalizedDraft(draft_id=d["draft_id"], composite_confidence=80))
            for d in base_state["drafts"]
        ]
        state = {**base_state, "normalized_drafts": normalized}
        result = await serialize_stix(state)
        objects = result["stix_bundle"]["objects"]

        with_cmds = [d for d in base_state["drafts"] if d.get("raw_command_lines")]
        assert with_cmds, "fixture must carry at least one draft with commands"
        processes = {o["id"]: o for o in objects if o["type"] == "process"}
        assert len(processes) == sum(len(d["raw_command_lines"]) for d in with_cmds)

        procs = {o["name"]: o for o in objects if o["type"] == X_PROCEDURE_TYPE}
        for d in with_cmds:
            refs = procs[d["name"]]["x_components_refs"]
            assert len(refs) == len(d["raw_command_lines"])
            assert [processes[r]["command_line"] for r in refs] == d["raw_command_lines"]
        for d in base_state["drafts"]:
            if not d.get("raw_command_lines") and d["name"] in procs:
                assert not procs[d["name"]].get("x_components_refs")

    async def test_bundle_has_source_identity(self, base_state):
        """Bundle always contains a source identity SDO."""
        normalized = [
            asdict(NormalizedDraft(draft_id=d["draft_id"], composite_confidence=80))
            for d in base_state["drafts"]
        ]
        state = {**base_state, "normalized_drafts": normalized}
        result = await serialize_stix(state)

        identities = [
            o for o in result["stix_bundle"]["objects"]
            if o["type"] == "identity" and o.get("x_source_id")
        ]
        assert len(identities) == 1
        assert identities[0]["name"] == "Test Author"

    async def test_bundle_has_procedures(self, base_state):
        """Bundle contains x-procedure SDOs for each approved draft."""
        normalized = [
            asdict(NormalizedDraft(draft_id=d["draft_id"], composite_confidence=80))
            for d in base_state["drafts"]
        ]
        state = {**base_state, "normalized_drafts": normalized}
        result = await serialize_stix(state)

        procedures = [
            o for o in result["stix_bundle"]["objects"]
            if o["type"] == X_PROCEDURE_TYPE
        ]
        assert len(procedures) == 3

    async def test_removed_entities_excluded(self, base_state):
        """Entities with gate_action='remove' are not in the bundle."""
        # Mark one entity as removed
        entities = list(base_state["validated_entities"])
        entities[0] = {**entities[0], "gate_action": "remove"}

        normalized = [
            asdict(NormalizedDraft(draft_id=d["draft_id"], composite_confidence=80))
            for d in base_state["drafts"]
        ]
        state = {**base_state, "validated_entities": entities, "normalized_drafts": normalized}
        result = await serialize_stix(state)

        # The removed entity's value should not appear as a named object
        removed_value = base_state["validated_entities"][0]["value"]
        named_objects = [
            o for o in result["stix_bundle"]["objects"]
            if o.get("name") == removed_value
        ]
        assert len(named_objects) == 0

    async def test_validation_runs(self, base_state):
        """Validation results dict is populated."""
        normalized = [
            asdict(NormalizedDraft(draft_id=d["draft_id"], composite_confidence=80))
            for d in base_state["drafts"]
        ]
        state = {**base_state, "normalized_drafts": normalized}
        result = await serialize_stix(state)

        vr = result["validation_results"]
        assert "schema" in vr
        assert "reference_integrity" in vr
        assert "attack_flow" in vr

    async def test_empty_drafts_still_produces_bundle(self, base_state):
        """Even with no drafts, bundle has entity objects."""
        state = {**base_state, "normalized_drafts": [], "drafts": []}
        result = await serialize_stix(state)

        assert result["stix_bundle"]["type"] == "bundle"
        # Should still have source identity + entity objects
        assert len(result["stix_bundle"]["objects"]) > 0

    async def test_attack_flow_object_emitted_with_start_refs(self, base_state):
        """When 2+ procedures exist, the bundle gets an attack-flow SDO
        whose start_refs lists every chain_root procedure (or every
        topological root when none are explicitly marked)."""
        # Mark draft #2 as a chain root so start_refs is deterministic.
        drafts = [dict(d) for d in base_state["drafts"]]
        drafts[1]["chain_root"] = True
        drafts[1]["chain_label"] = "Veeam intrusion"

        normalized = [
            asdict(NormalizedDraft(draft_id=d["draft_id"], composite_confidence=80))
            for d in drafts
        ]
        state = {**base_state, "drafts": drafts, "normalized_drafts": normalized}
        result = await serialize_stix(state)
        bundle = result["stix_bundle"]

        flows = [o for o in bundle["objects"] if o["type"] == "attack-flow"]
        assert len(flows) == 1, "Expected exactly one attack-flow"
        flow = flows[0]
        assert flow["start_refs"], "attack-flow.start_refs should be non-empty"

        # Each start_ref must resolve to an x-procedure in the bundle.
        proc_ids = {
            o["id"] for o in bundle["objects"] if o["type"] == "x-procedure"
        }
        for ref in flow["start_refs"]:
            assert ref in proc_ids, f"start_ref {ref} not in bundle"

        # Flow membership is expressed via start_refs + PRECEDES SROs, NOT
        # via an embedded back-pointer. No procedure carries x_flow_ref
        # (removed in v0.5.0-draft).
        procs = [o for o in bundle["objects"] if o["type"] == "x-procedure"]
        for p in procs:
            assert "x_flow_ref" not in p

        # The chain_root draft's procedure has x_chain_root=True and
        # x_chain_label populated.
        chain_root_procs = [p for p in procs if p.get("x_chain_root")]
        assert len(chain_root_procs) == 1
        assert chain_root_procs[0].get("x_chain_label") == "Veeam intrusion"

    async def test_no_attack_flow_when_single_procedure(self, base_state):
        """A bundle with only one procedure shouldn't emit an attack-flow."""
        drafts = [base_state["drafts"][0]]
        normalized = [
            asdict(NormalizedDraft(draft_id=drafts[0]["draft_id"], composite_confidence=80))
        ]
        state = {
            **base_state, "drafts": drafts, "normalized_drafts": normalized,
            "gate1_approved_draft_ids": [drafts[0]["draft_id"]],
        }
        result = await serialize_stix(state)
        flows = [o for o in result["stix_bundle"]["objects"] if o["type"] == "attack-flow"]
        assert flows == []


    async def test_omitted_procedure_is_recorded_not_just_logged(self, base_state):
        """A dropped procedure must reach the analyst, not only the log.

        serialize_stix runs AFTER the bundle gate, so an omission here can
        never be noticed at a prompt — the analyst already approved these
        procedures. On a ransomware run this would have silently
        discarded the one procedure the analyst rejected a whole chunking pass
        to capture.
        """
        drafts = [dict(d) for d in base_state["drafts"]]
        drafts[0]["techniques"] = []  # unmappable: no resolved technique
        normalized = [
            asdict(NormalizedDraft(draft_id=d["draft_id"], composite_confidence=80))
            for d in drafts
        ]
        state = {**base_state, "drafts": drafts, "normalized_drafts": normalized}

        result = await serialize_stix(state)

        recorded = [
            c for c in (result.get("bundle_corrections") or [])
            if c.get("rule") == "procedure_omitted_no_technique"
        ]
        assert len(recorded) == 1, result.get("bundle_corrections")
        assert recorded[0]["severity"] == "warn"
        assert recorded[0]["ref_field"] == "x_technique_refs"
        assert recorded[0]["ref_id"] == drafts[0]["draft_id"]

        # And the procedure really is absent — the correction describes a
        # loss that happened, not one that was averted.
        procs = [
            o for o in result["stix_bundle"]["objects"]
            if o["type"] == X_PROCEDURE_TYPE
        ]
        assert len(procs) == len(drafts) - 1

    async def test_no_omission_correction_when_all_drafts_map(self, base_state):
        normalized = [
            asdict(NormalizedDraft(draft_id=d["draft_id"], composite_confidence=80))
            for d in base_state["drafts"]
        ]
        state = {**base_state, "normalized_drafts": normalized}

        result = await serialize_stix(state)

        assert result.get("bundle_corrections") == []


# ── _entity_to_stix ──────────────────────────────────────────────


class TestEntityToStix:
    """Tests for entity-to-STIX object conversion."""

    def test_intrusion_set_to_intrusion_set(self):
        entity = {
            "entity_id": "e1",
            "entity_type": EntityType.INTRUSION_SET.value,
            "value": "APT29",
            "gate_action": GateAction.APPROVE.value,
        }
        obj = _entity_to_stix(entity)
        assert obj is not None
        assert obj["type"] == "intrusion-set"
        assert obj["name"] == "APT29"
        assert "created" in obj

    def test_threat_actor_to_threat_actor(self):
        entity = {
            "entity_id": "e1b",
            "entity_type": EntityType.THREAT_ACTOR.value,
            "value": "SVR",
        }
        obj = _entity_to_stix(entity)
        assert obj["type"] == "threat-actor"
        assert obj["name"] == "SVR"
        assert "threat_actor_types" in obj

    def test_region_goes_to_region_not_country(self):
        """Every location used to serialize as Location.country, so "EMEA"
        shipped as a country. STIX has a separate `region` property for this.

        The canonical six (glossary: "region (canonical set)") are business
        regions, not STIX region-ov values — EMEA alone spans three M49 values —
        but `region` is an open vocabulary, so the acronym validates.
        """
        for value in ("EMEA", "APAC", "Asia-Pacific", "Eastern Europe"):
            obj = _entity_to_stix({
                "entity_id": "loc-r", "entity_type": EntityType.LOCATION.value,
                "value": value,
            })
            assert obj["type"] == "location"
            assert obj.get("region") == value, value
            assert "country" not in obj, value

    def test_country_still_goes_to_country(self):
        for value in ("Turkey", "Madagascar", "South Korea"):
            obj = _entity_to_stix({
                "entity_id": "loc-c", "entity_type": EntityType.LOCATION.value,
                "value": value,
            })
            assert obj.get("country") == value, value
            assert "region" not in obj, value

    def test_subregion_is_not_widened_to_a_canonical_acronym(self):
        """"South Asia" is not reliably "APAC" — definitions differ on whether
        India and Pakistan are inside. Widening a targeting claim the source did
        not make is worse than keeping its own word."""
        obj = _entity_to_stix({
            "entity_id": "loc-s", "entity_type": EntityType.LOCATION.value,
            "value": "South Asia",
        })
        assert obj["region"] == "South Asia"

    def test_location_object_passes_schema_validation(self):
        from app.services import stix_schema

        obj = _entity_to_stix({
            "entity_id": "loc-v", "entity_type": EntityType.LOCATION.value,
            "value": "EMEA",
        })
        assert not stix_schema.validate_bundle_objects([obj])

    def test_infrastructure_carries_a_type_placeholder(self):
        """An Infrastructure SDO with only a name validates but says nothing.

        `infrastructure_types` is optional, so omitting it passed schema
        validation while losing the C2-vs-staging-vs-phishing distinction these
        entities are extracted for. `threat_actor_types` above set the
        precedent. minItems is 1, so [] would not validate — "unknown" is the
        spec's own placeholder (infrastructure-type-ov, STIX 2.1 §10.12).
        """
        entity = {
            "entity_id": "e1c",
            "entity_type": EntityType.INFRASTRUCTURE.value,
            "value": "staging server 203.0.113.10",
        }
        obj = _entity_to_stix(entity)
        assert obj["type"] == "infrastructure"
        assert obj["infrastructure_types"] == ["unknown"]

    def test_infrastructure_object_passes_schema_validation(self):
        """The open vocabulary is not enforced, but minItems: 1 is."""
        from app.services import stix_schema

        entity = {
            "entity_id": "e1d",
            "entity_type": EntityType.INFRASTRUCTURE.value,
            "value": "bulletproof host",
        }
        obj = _entity_to_stix(entity)
        errors = stix_schema.validate_bundle_objects([obj])
        assert not errors, errors

    def test_malware_to_malware(self):
        entity = {
            "entity_id": "e2",
            "entity_type": EntityType.MALWARE.value,
            "value": "Cobalt Strike",
        }
        obj = _entity_to_stix(entity)
        assert obj["type"] == "malware"
        assert obj["is_family"] is True

    def test_ioc_ip_to_ipv4_sco(self):
        entity = {
            "entity_id": "e3",
            "entity_type": EntityType.IOC_IP.value,
            "value": "203.0.113.10",
        }
        obj = _entity_to_stix(entity)
        assert obj["type"] == "ipv4-addr"
        assert obj["value"] == "203.0.113.10"
        assert "created" not in obj

    def test_ioc_ip_ipv6_detected(self):
        """IPv6 addresses are parsed, not guessed at from a colon."""
        entity = {
            "entity_id": "e3b",
            "entity_type": EntityType.IOC_IP.value,
            "value": "2001:db8::1",
        }
        obj = _entity_to_stix(entity)
        assert obj["type"] == "ipv6-addr"
        assert obj["value"] == "2001:db8::1"

    def test_ipv4_with_port_is_not_mistaken_for_ipv6(self):
        """The bug that cost a whole bundle.

        Detection used to be `":" in value`, which is true of every host:port
        indicator, so 198.51.100.167:2967 was emitted as an `ipv6-addr`. It is
        not a valid IPv6 address, so schema validation rejected it — and
        because bundle validation is all-or-nothing, that ONE observable
        destroyed the entire bundle (one malware-analysis report).
        """
        obj = _entity_to_stix({
            "entity_id": "e-port",
            "entity_type": EntityType.IOC_IP.value,
            "value": "198.51.100.167:2967",
        })
        assert obj["type"] == "ipv4-addr"
        # The port is dropped: STIX ipv4-addr has no field for it, and a port
        # belongs on a network-traffic SCO.
        assert obj["value"] == "198.51.100.167"

    def test_bracketed_ipv6_with_port_keeps_the_address(self):
        obj = _entity_to_stix({
            "entity_id": "e-v6port",
            "entity_type": EntityType.IOC_IP.value,
            "value": "[2001:db8::1]:443",
        })
        assert obj["type"] == "ipv6-addr"
        assert obj["value"] == "2001:db8::1"

    def test_unparseable_ip_is_skipped_not_emitted(self):
        """One bad indicator must not poison the bundle it travels in.

        Returning None drops the entity; emitting a malformed SCO fails
        validation for every object in the bundle.
        """
        for junk in ("not-an-ip", "999.1.1.1", "evil.com:8080", ""):
            assert _entity_to_stix({
                "entity_id": "e-junk",
                "entity_type": EntityType.IOC_IP.value,
                "value": junk,
            }) is None, f"{junk!r} should have been skipped"

    def test_ioc_hash_to_file_sco(self):
        entity = {
            "entity_id": "e4",
            "entity_type": EntityType.IOC_HASH.value,
            "value": "a" * 64,
        }
        obj = _entity_to_stix(entity)
        assert obj["type"] == "file"
        assert "SHA-256" in obj["hashes"]

    def test_edited_value_used(self):
        entity = {
            "entity_id": "e5",
            "entity_type": EntityType.INTRUSION_SET.value,
            "value": "APT 29",
            "edited_value": "APT29",
            "edit_rationale": "Removed space",
        }
        obj = _entity_to_stix(entity)
        assert obj["name"] == "APT29"

    def test_unknown_type_returns_none(self):
        entity = {
            "entity_id": "e6",
            "entity_type": "some_unknown_type",
            "value": "test",
        }
        obj = _entity_to_stix(entity)
        assert obj is None

    def test_ioc_domain_to_sco(self):
        entity = {
            "entity_id": "e7",
            "entity_type": EntityType.IOC_DOMAIN.value,
            "value": "evil.example.com",
        }
        obj = _entity_to_stix(entity)
        assert obj["type"] == "domain-name"
        assert obj["value"] == "evil.example.com"

    def test_vulnerability_to_sdo(self):
        entity = {
            "entity_id": "e8",
            "entity_type": EntityType.VULNERABILITY.value,
            "value": "CVE-2023-46604",
        }
        obj = _entity_to_stix(entity)
        assert obj["type"] == "vulnerability"
        assert obj["name"] == "CVE-2023-46604"
        assert obj["external_references"][0]["external_id"] == "CVE-2023-46604"

    def test_organization_to_identity(self):
        entity = {
            "entity_id": "e9",
            "entity_type": EntityType.ORGANIZATION.value,
            "value": "Acme Corp",
            "organization_role": "victim",
        }
        obj = _entity_to_stix(entity)
        assert obj["type"] == "identity"
        assert obj["identity_class"] == "organization"
        assert obj["name"] == "Acme Corp"

    def test_location_to_location(self):
        entity = {
            "entity_id": "e10",
            "entity_type": EntityType.LOCATION.value,
            "value": "United States",
        }
        obj = _entity_to_stix(entity)
        assert obj["type"] == "location"
        assert obj["country"] == "United States"

    def test_mutex_to_sco(self):
        entity = {
            "entity_id": "e11",
            "entity_type": EntityType.IOC_MUTEX.value,
            "value": "Global\\MyMutex123",
        }
        obj = _entity_to_stix(entity)
        assert obj["type"] == "mutex"
        assert obj["name"] == "Global\\MyMutex123"

    def test_software_to_sco(self):
        entity = {
            "entity_id": "e12",
            "entity_type": EntityType.SOFTWARE.value,
            "value": "Apache ActiveMQ 5.15.0",
        }
        obj = _entity_to_stix(entity)
        assert obj["type"] == "software"
        assert obj["name"] == "Apache ActiveMQ 5.15.0"

    def test_user_account_to_sco(self):
        entity = {
            "entity_id": "e13",
            "entity_type": EntityType.USER_ACCOUNT.value,
            "value": "svc_backup",
        }
        obj = _entity_to_stix(entity)
        assert obj["type"] == "user-account"
        assert obj["account_login"] == "svc_backup"


# ── _draft_to_procedure ──────────────────────────────────────────


class TestDraftToProcedure:
    """Tests for draft-to-x-procedure conversion."""

    def test_basic_procedure(self, sample_drafts):
        draft = sample_drafts[0]
        ndraft = asdict(NormalizedDraft(draft_id="dft-001", composite_confidence=85))

        proc = _draft_to_procedure(draft, ndraft, "identity--source-001")

        assert proc["type"] == X_PROCEDURE_TYPE
        assert proc["spec_version"] == "2.1"
        assert proc["name"] == draft["name"]
        assert proc["confidence"] == 85
        assert proc["created_by_ref"] == "identity--source-001"
        assert proc["x_procedure_type"] == "reporting"

    def test_command_lines_preserved(self, sample_drafts):
        """Draft with raw_command_lines gets converted to x_components_refs."""
        draft = sample_drafts[1]  # dft-002 has raw_command_lines
        ndraft = asdict(NormalizedDraft(draft_id="dft-002", composite_confidence=79))

        proc = _draft_to_procedure(draft, ndraft, "identity--source-001")

        # In the draft, raw_command_lines are present, but they are converted
        # to Process SCOs during serialization (in serialize_stix), not in
        # _draft_to_procedure. The primary command is the first entry in
        # x_components_refs (x_command_ref was removed in v0.5.0-draft).
        assert proc["type"] == X_PROCEDURE_TYPE
        assert proc["spec_version"] == "2.1"

    def test_sequencing_fields(self, sample_drafts):
        """Sequencing is NOT embedded on the procedure (no x_effect_refs /
        x_flow_ref / x_sequence_index); it materializes as PRECEDES SROs and
        the attack-flow object at the bundle level."""
        draft = sample_drafts[2]  # dft-003 has predecessor_indices=[2]
        ndraft = asdict(NormalizedDraft(draft_id="dft-003", composite_confidence=72))

        proc = _draft_to_procedure(draft, ndraft, "identity--source-001")

        # The procedure carries no embedded flow scaffolding.
        assert "x_sequence_index" not in proc
        assert "x_predecessor_indices" not in proc
        assert "x_effect_refs" not in proc
        assert "x_flow_ref" not in proc
        assert proc["type"] == X_PROCEDURE_TYPE

    def test_unresolved_technique_is_omitted_not_fabricated(self, sample_drafts):
        """An unresolved technique must not become `attack-pattern--T1190`.

        That form is not a valid STIX identifier and resolves to nothing.
        Reference-integrity validation never caught it because it waves
        through any ref prefixed `attack-pattern--`.
        """
        # Own the unresolved state rather than borrowing it. sample_drafts
        # now carries resolved stix_ids (production resolves every id before
        # drafting, and serialize_stix omits a procedure that has none), so
        # depending on the shared fixture for "unresolved" made this test
        # hostage to that fixture — which is exactly what its own guard
        # caught when the fixture changed.
        draft = {
            **sample_drafts[0],
            "techniques": [
                {**t, "stix_id": None} for t in sample_drafts[0]["techniques"]
            ],
        }
        assert draft["techniques"], "fixture must have techniques to omit"
        assert not any(t.get("stix_id") for t in draft["techniques"]), (
            "this test must run against UNresolved techniques"
        )
        ndraft = asdict(NormalizedDraft(draft_id="dft-001", composite_confidence=85))

        proc = _draft_to_procedure(draft, ndraft, "identity--source-001")

        for ref in proc.get("x_technique_refs", []):
            assert not re.match(r"^attack-pattern--T\d{4}", ref), (
                f"fabricated technique ref leaked into the bundle: {ref}"
            )
        # Nothing resolved, so the key is absent rather than holding junk.
        assert "x_technique_refs" not in proc

    def test_resolved_technique_is_kept(self, sample_drafts):
        """The drop only applies to unresolved techniques — a real UUID
        still rides through untouched."""
        draft = {**sample_drafts[0]}
        real_id = "attack-pattern--3f886f2a-874f-4333-b794-aa6075009b1c"
        draft["techniques"] = [
            {**draft["techniques"][0], "stix_id": real_id},
            {**draft["techniques"][0], "technique_id": "T9999", "stix_id": None},
        ]
        ndraft = asdict(NormalizedDraft(draft_id="dft-001", composite_confidence=85))

        proc = _draft_to_procedure(draft, ndraft, "identity--source-001")

        assert proc["x_technique_refs"] == [real_id]

    def test_procedure_type_field(self, sample_drafts):
        """Draft with procedure_type field includes x_procedure_type."""
        draft = sample_drafts[0]  # dft-001 has procedure_type="reporting"
        ndraft = asdict(NormalizedDraft(draft_id="dft-001", composite_confidence=85))

        proc = _draft_to_procedure(draft, ndraft, "identity--source-001")

        # x_detail_gap was removed in v0.5.0-draft
        # Instead, x_procedure_type is now set (defaults to "reporting")
        assert proc.get("x_procedure_type") == "reporting"
        assert "x_detail_gap" not in proc

    def test_hypothetical_procedure_type_reaches_the_bundle(self, sample_drafts):
        """A vendor's "the actors could ALSO have used X" must not serialize
        identically to "the actors did".

        procedure_type was hardcoded "reporting" at drafting, so the schema's
        `hypothetical` value was unreachable and every hedged vector shipped as
        a confirmed one. The Zerologon procedure from one ransomware run is the live case: the
        chunker marked it behavioral_confidence 0.40 against a median of 1.00,
        the analyst raised it to 0.85 after rewriting it, and all of that
        flattened at this boundary.
        """
        draft = {**sample_drafts[0], "procedure_type": "hypothetical"}
        ndraft = asdict(NormalizedDraft(draft_id="dft-001", composite_confidence=85))

        proc = _draft_to_procedure(draft, ndraft, "identity--source-001")

        assert proc["x_procedure_type"] == "hypothetical"

    def test_procedure_type_defaults_when_draft_omits_it(self, sample_drafts):
        """Cached LLM outputs predating the field must still serialize."""
        draft = {k: v for k, v in sample_drafts[0].items() if k != "procedure_type"}
        ndraft = asdict(NormalizedDraft(draft_id="dft-001", composite_confidence=85))

        proc = _draft_to_procedure(draft, ndraft, "identity--source-001")

        assert proc["x_procedure_type"] == "reporting"

    def test_platforms_set(self, sample_drafts):
        """Platforms are mapped to x_platforms."""
        draft = sample_drafts[0]  # has ["linux::server", "windows::server"]
        ndraft = asdict(NormalizedDraft(draft_id="dft-001", composite_confidence=85))

        proc = _draft_to_procedure(draft, ndraft, "identity--source-001")
        assert "x_platforms" in proc
        assert "linux::server" in proc["x_platforms"]


# ── _detect_hash_type ─────────────────────────────────────────────


class TestDetectHashType:
    """Tests for hash type detection by length."""

    def test_md5(self):
        assert _detect_hash_type("a" * 32) == "MD5"

    def test_sha1(self):
        assert _detect_hash_type("a" * 40) == "SHA-1"

    def test_sha256(self):
        assert _detect_hash_type("a" * 64) == "SHA-256"

    def test_sha512(self):
        assert _detect_hash_type("a" * 128) == "SHA-512"

    def test_unknown_returns_none(self):
        assert _detect_hash_type("abc") is None


# ── _validate_bundle ──────────────────────────────────────────────


class TestValidateBundle:
    """Tests for bundle validation."""

    def test_empty_bundle_fails_schema(self):
        bundle = {"type": "bundle", "id": "bundle--test", "objects": []}
        results, errors = _validate_bundle(bundle)
        assert results["schema"] is False
        assert any("no objects" in e.lower() for e in errors)

    def test_missing_type_fails(self):
        bundle = {
            "type": "bundle",
            "id": "bundle--test",
            "objects": [{"id": "malware--abc", "created": "2024-01-01", "modified": "2024-01-01"}],
        }
        results, errors = _validate_bundle(bundle)
        assert results["schema"] is False

    def test_relationship_missing_fields_fails(self):
        bundle = {
            "type": "bundle",
            "id": "bundle--test",
            "objects": [{
                "type": "relationship",
                "id": "relationship--abc",
                # missing source_ref, target_ref, relationship_type
            }],
        }
        results, errors = _validate_bundle(bundle)
        assert results["schema"] is False

    def test_dangling_ref_fails_integrity(self):
        """A created_by_ref pointing to a non-existent ID fails reference check."""
        bundle = {
            "type": "bundle",
            "id": "bundle--test",
            "objects": [{
                "type": "x-procedure",
                "id": "x-procedure--001",
                "created": "2024-01-01T00:00:00Z",
                "modified": "2024-01-01T00:00:00Z",
                "created_by_ref": "identity--does-not-exist",
            }],
        }
        results, errors = _validate_bundle(bundle)
        assert results["reference_integrity"] is False

    def test_attack_flow_cycle_detected(self):
        """A cyclic PRECEDES graph is caught."""
        bundle = {
            "type": "bundle",
            "id": "bundle--test",
            "objects": [
                {
                    "type": X_PROCEDURE_TYPE,
                    "id": "x-procedure--a",
                    "created": "2024-01-01T00:00:00Z",
                    "modified": "2024-01-01T00:00:00Z",
                },
                {
                    "type": X_PROCEDURE_TYPE,
                    "id": "x-procedure--b",
                    "created": "2024-01-01T00:00:00Z",
                    "modified": "2024-01-01T00:00:00Z",
                },
                {
                    "type": "relationship",
                    "id": "relationship--p1",
                    "created": "2024-01-01T00:00:00Z",
                    "modified": "2024-01-01T00:00:00Z",
                    "relationship_type": "precedes",
                    "source_ref": "x-procedure--a",
                    "target_ref": "x-procedure--b",
                },
                {
                    "type": "relationship",
                    "id": "relationship--p2",
                    "created": "2024-01-01T00:00:00Z",
                    "modified": "2024-01-01T00:00:00Z",
                    "relationship_type": "precedes",
                    "source_ref": "x-procedure--b",
                    "target_ref": "x-procedure--a",
                },
            ],
        }
        results, errors = _validate_bundle(bundle)
        assert results["attack_flow"] is False
        assert any("cycle" in e.lower() for e in errors)

    def test_multiple_attack_flow_objects_detected(self):
        """More than one attack-flow object in a bundle is caught."""
        bundle = {
            "type": "bundle",
            "id": "bundle--test",
            "objects": [
                {
                    "type": X_PROCEDURE_TYPE,
                    "id": "x-procedure--a",
                    "created": "2024-01-01T00:00:00Z",
                    "modified": "2024-01-01T00:00:00Z",
                },
                {
                    "type": X_PROCEDURE_TYPE,
                    "id": "x-procedure--b",
                    "created": "2024-01-01T00:00:00Z",
                    "modified": "2024-01-01T00:00:00Z",
                },
                {
                    "type": "attack-flow",
                    "id": "attack-flow--flow-001",
                    "created": "2024-01-01T00:00:00Z",
                    "modified": "2024-01-01T00:00:00Z",
                    "name": "Flow 1",
                    "start_refs": ["x-procedure--a"],
                },
                {
                    "type": "attack-flow",
                    "id": "attack-flow--flow-002",
                    "created": "2024-01-01T00:00:00Z",
                    "modified": "2024-01-01T00:00:00Z",
                    "name": "Flow 2",
                    "start_refs": ["x-procedure--b"],
                },
            ],
        }
        results, errors = _validate_bundle(bundle)
        assert results["attack_flow"] is False
        assert any("attack-flow" in e.lower() for e in errors)

    def test_valid_linear_flow(self):
        """A simple 1->2->3 flow using PRECEDES SROs passes validation.

        IDs are real UUIDs, not placeholders: this asserts the bundle is
        VALID, and since _validate_schema gained full JSON Schema
        validation a placeholder like "identity--src" is genuinely
        invalid STIX. The fixture has to be what it claims to be.
        """
        identity = {
            "type": "identity",
            "id": "identity--11111111-1111-4111-8111-111111111111",
            "spec_version": "2.1",
            "created": "2024-01-01T00:00:00.000Z",
            "modified": "2024-01-01T00:00:00.000Z",
            "name": "Test",
        }

        # Create supporting objects
        attack_pattern = {
            "type": "attack-pattern",
            "id": "attack-pattern--22222222-2222-4222-8222-222222222222",
            "spec_version": "2.1",
            "created": "2024-01-01T00:00:00.000Z",
            "modified": "2024-01-01T00:00:00.000Z",
            "name": "Test Technique",
        }
        log_source = {
            "type": "x-log-source",
            "id": "x-log-source--33333333-3333-4333-8333-333333333333",
            "created": "2024-01-01T00:00:00.000Z",
            "modified": "2024-01-01T00:00:00.000Z",
            "name": "Test Log Source",
        }
        process_sco = {
            "type": "process",
            "id": "process--44444444-4444-4444-8444-444444444444",
            "command_line": "test.exe",
        }

        procs = []
        proc_ids = [f"x-procedure--5555555{i}-5555-4555-8555-555555555555" for i in range(1, 4)]

        for i in range(3):
            p = {
                "type": X_PROCEDURE_TYPE,
                "spec_version": "2.1",
                "id": proc_ids[i],
                "created": "2024-01-01T00:00:00.000Z",
                "modified": "2024-01-01T00:00:00.000Z",
                "name": f"Step {i+1}",
                "created_by_ref": "identity--11111111-1111-4111-8111-111111111111",
                # Add required tuple fields to pass validation
                "x_technique_refs": ["attack-pattern--22222222-2222-4222-8222-222222222222"],
                "x_log_source_refs": ["x-log-source--33333333-3333-4333-8333-333333333333"],
                "x_components_refs": ["process--44444444-4444-4444-8444-444444444444"],
            }
            procs.append(p)

        # Each procedure (except last) precedes the next via a PRECEDES SRO.
        precedes_sros = [
            {
                "type": "relationship",
                "spec_version": "2.1",
                "id": f"relationship--6666666{i}-6666-4666-8666-666666666666",
                "created": "2024-01-01T00:00:00.000Z",
                "modified": "2024-01-01T00:00:00.000Z",
                "relationship_type": "precedes",
                "source_ref": proc_ids[i],
                "target_ref": proc_ids[i + 1],
            }
            for i in range(2)
        ]

        bundle = {
            "type": "bundle",
            "id": "bundle--77777777-7777-4777-8777-777777777777",
            "objects": (
                [identity, attack_pattern, log_source, process_sco]
                + procs + precedes_sros
            ),
        }
        results, errors = _validate_bundle(bundle)
        assert results["schema"] is True
        assert results["reference_integrity"] is True
        assert results["attack_flow"] is True
        # Filter out tuple_semantics warnings (they're expected for low-confidence)
        actual_errors = [e for e in errors if "Tuple semantics" not in e]
        assert len(actual_errors) == 0


class TestProcedureMatchesDeclaredSchema:
    """The one check that would have caught the x_observable_refs drift.

    x_procedure_v3.json is the declared contract and sets
    `additionalProperties: false`, but nothing loads it at runtime —
    `_validate_schema` is still the hand-rolled required-fields check its
    own docstring flags as pre-"Phase 2". That gap let the serializer embed
    `x_observable_refs`, a field the schema has never allowed, for months.

    These tests close it from the other side: if anyone puts a field on an
    x-procedure that the schema doesn't declare, this fails now instead of
    the day real JSON Schema validation is switched on (at which point
    all-or-nothing validation would route every affected source to Failed).
    """

    def _schema(self):
        import json
        from pathlib import Path
        schema_path = (
            Path(__file__).resolve().parents[1]
            / "backend" / "app" / "schemas" / "x_procedure_v3.json"
        )
        return json.loads(schema_path.read_text())

    # Undeclared properties waived for now. Keyed by exact property NAME —
    # a coarser key (just "some additionalProperties error") would let a
    # reintroduced x_observable_refs slip through, which defeats the point.
    # Empty is the goal state; every entry here is a known open gap.
    ALLOWED_UNDECLARED: set[str] = set()

    # Non-additionalProperties violations, as (json-path head, keyword).
    # Empty is the goal state; every entry is a known open gap.
    KNOWN_GAPS: set[tuple[str, str]] = set()

    def test_every_field_the_serializer_writes_is_declared(self):
        """Static ratchet over the whole class of bug, not one instance.

        The fixture-driven tests below only see fields the FIXTURE happens to
        produce. That gap is real and it bit: x_chain_label / x_chain_root are
        written for multi-chain sources, no fixture draft carried them, and
        they sailed through every test until a live run failed on them.

        This scans the serializer's own source for `procedure["..."] = ` style
        writes and requires each key to be declared in the schema — so a new
        field is caught the moment it's written, regardless of fixtures.

        SCOPE: serializer writes only. Mutations applied AFTER serialization —
        bundle_validator's auto-fix passes — are out of reach of a static scan
        of this module, and one of them (`procedure_tactics_rederived`) is a
        live example of the same bug class. Those are covered behaviorally by
        the golden-replay tests, which re-validate the bundle after
        validate_bundle has had its turn.
        """
        import ast
        from pathlib import Path

        src_path = (
            Path(__file__).resolve().parents[1]
            / "backend" / "app" / "nodes" / "deterministic" / "serialization.py"
        )
        tree = ast.parse(src_path.read_text())

        # Names that hold an x-procedure dict in this module.
        proc_names = {"procedure", "procedure_obj"}
        written: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for target in node.targets:
                if (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.value, ast.Name)
                    and target.value.id in proc_names
                    and isinstance(target.slice, ast.Constant)
                    and isinstance(target.slice.value, str)
                ):
                    written.add(target.slice.value)

        assert written, "found no procedure field writes — did the scan break?"

        declared = set(self._schema().get("properties", {}))
        undeclared = sorted(written - declared)
        assert not undeclared, (
            "serialization.py writes x-procedure field(s) the schema does not "
            "declare, and the schema is additionalProperties:false — bundles "
            f"carrying them fail validation: {undeclared}"
        )

    async def test_no_new_schema_violations_on_serialized_procedures(self, base_state):
        jsonschema = pytest.importorskip("jsonschema")

        normalized = [
            asdict(NormalizedDraft(
                draft_id=d["draft_id"],
                composite_confidence=80,
                confidence_breakdown={},
                standardized_names={},
            ))
            for d in base_state["drafts"]
        ]
        result = await serialize_stix({**base_state, "normalized_drafts": normalized})

        procedures = [
            o for o in result["stix_bundle"]["objects"]
            if o.get("type") == X_PROCEDURE_TYPE
        ]
        assert procedures, "fixture produced no procedures to check"

        schema = self._schema()
        declared = set(schema.get("properties", {}))
        validator = jsonschema.Draft7Validator(schema)

        unexpected = []
        for proc in procedures:
            # Undeclared properties, checked by name so a specific field
            # can be waived without blanket-waiving the whole keyword.
            for prop in sorted(set(proc) - declared - self.ALLOWED_UNDECLARED):
                unexpected.append(
                    f"{proc.get('id')}: undeclared property {prop!r} "
                    f"(schema is additionalProperties:false)"
                )
            for err in validator.iter_errors(proc):
                if err.validator == "additionalProperties":
                    continue  # handled by name above
                path = "/".join(str(x) for x in err.absolute_path) or "(root)"
                head = path.split("/")[0] or "(root)"
                if (head, err.validator) in self.KNOWN_GAPS:
                    continue
                unexpected.append(f"{proc.get('id')}: [{err.validator}] {path} — {err.message}")

        assert not unexpected, (
            "New x-procedure schema violation(s). x_procedure_v3.json is "
            "additionalProperties:false and is the declared contract:\n  "
            + "\n  ".join(unexpected)
        )

    @staticmethod
    def _ioc_linked_state(base_state):
        """State that actually drives the IoC-linking pass.

        base_state ships `chunks: []`, so the pass short-circuits before it
        can touch a procedure — asserting on it without this would pass
        vacuously. Here chunk chk-002 carries artifacts whose values match
        seeded entities (the certutil tool and the 203.0.113.10 IP), which
        is exactly the condition that used to produce x_observable_refs.
        """
        chunks = [{
            "chunk_id": "chk-002",
            "text": "The actor used certutil.exe to pull a web shell from 203.0.113.10.",
            "artifacts": {
                "ioc_ip": ["203.0.113.10"],
                "tool": ["certutil.exe"],
            },
        }]
        normalized = [
            asdict(NormalizedDraft(
                draft_id=d["draft_id"],
                composite_confidence=80,
                confidence_breakdown={},
                standardized_names={},
            ))
            for d in base_state["drafts"]
        ]
        return {**base_state, "chunks": chunks, "normalized_drafts": normalized}

    async def test_ioc_linking_pass_actually_fires(self, base_state):
        """Guards the guard: if this stops finding has-observable SROs, the
        test below has gone vacuous and proves nothing."""
        result = await serialize_stix(self._ioc_linked_state(base_state))
        objects = result["stix_bundle"]["objects"]

        has_obs = [
            o for o in objects
            if o.get("type") == "relationship"
            and o.get("relationship_type") == "has-observable"
        ]
        assert has_obs, "IoC-linking pass did not fire — fixture no longer exercises it"

    async def test_observables_ride_sros_not_the_procedure(self, base_state):
        """Observables a procedure TOUCHES belong on has-observable SROs.

        Embedding them as x_observable_refs is what broke the schema. It is
        also not the same thing as x_components_refs (what the procedure is
        MADE of) — components are the process tree of its commands — so the
        fix is the SRO, not a merge into components.
        """
        result = await serialize_stix(self._ioc_linked_state(base_state))
        objects = result["stix_bundle"]["objects"]

        procedures = [o for o in objects if o.get("type") == X_PROCEDURE_TYPE]
        assert procedures
        for proc in procedures:
            assert "x_observable_refs" not in proc
            assert "x_asset_refs" not in proc

    async def test_no_new_schema_violations_when_iocs_are_linked(self, base_state):
        """The IoC-linking path, which is the one that used to break the
        schema, run through the same ratchet."""
        jsonschema = pytest.importorskip("jsonschema")

        result = await serialize_stix(self._ioc_linked_state(base_state))
        schema = self._schema()
        declared = set(schema.get("properties", {}))

        unexpected = []
        for proc in result["stix_bundle"]["objects"]:
            if proc.get("type") != X_PROCEDURE_TYPE:
                continue
            for prop in sorted(set(proc) - declared - self.ALLOWED_UNDECLARED):
                unexpected.append(f"{proc.get('id')}: undeclared property {prop!r}")
        assert not unexpected, "\n  ".join(unexpected)

    def test_schema_rejects_the_field_that_used_to_be_embedded(self):
        """Guards the premise: this only matters because the schema is
        closed. If additionalProperties ever opens up, this test says so."""
        jsonschema = pytest.importorskip("jsonschema")

        schema = self._schema()
        assert schema.get("additionalProperties") is False

        proc = {
            "type": X_PROCEDURE_TYPE,
            "spec_version": "2.1",
            "id": "x-procedure--11111111-1111-1111-1111-111111111111",
            "created": "2026-01-01T00:00:00.000Z",
            "modified": "2026-01-01T00:00:00.000Z",
            "name": "Delete Volume Shadow Copies via WMIC",
        }
        jsonschema.validate(proc, schema)  # baseline is clean

        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {**proc, "x_observable_refs": ["process--2222"]}, schema
            )


class TestEveryEntityTypeSerializes:
    """No EntityType may be silently dropped at serialization.

    `_entity_to_stix` returns None for any type missing from
    `_ENTITY_TO_STIX_TYPE`, logging a warning nobody reads. `ioc_process_name`
    sat in that hole: the entity prompt devotes two paragraphs to extracting
    it, the analyst reviews it at gate_0, and then it produced no object at
    all. On one campaign report that lost Qt6Core.dll — the sideloaded DLL, the payoff
    of the whole tar-extraction step — from the bundle entirely.
    """

    # Types with a deliberate reason not to map to a standalone object.
    EXEMPT = {
        # Command lines become Process SCOs via draft.raw_command_lines,
        # carrying command_line + image_ref. A bare string SCO would duplicate
        # that with less structure.
        EntityType.IOC_COMMAND_LINE.value,
    }

    def test_every_entity_type_maps_or_is_explicitly_exempt(self):
        from app.nodes.deterministic.serialization import _ENTITY_TO_STIX_TYPE

        unmapped = {
            e.value for e in EntityType
            if e.value not in _ENTITY_TO_STIX_TYPE and e.value not in self.EXEMPT
        }
        assert unmapped == set(), (
            f"these EntityTypes serialize to nothing: {sorted(unmapped)}. "
            "Either map them or add them to EXEMPT with a reason."
        )

    def test_process_name_becomes_a_named_file_sco(self):
        """A file SCO with neither name nor hashes is invalid STIX."""
        obj = _entity_to_stix({
            "entity_id": "ent-1",
            "entity_type": EntityType.IOC_PROCESS_NAME.value,
            "value": "Qt6Core.dll",
        })
        assert obj is not None
        assert obj["type"] == "file"
        assert obj["name"] == "Qt6Core.dll"

    def test_process_name_is_a_file_not_a_process(self):
        """STIX Process is a live execution instance, not a filename.

        A DLL is loaded into a process and can never be one, which is the
        clearest case for the File typing.
        """
        obj = _entity_to_stix({
            "entity_id": "ent-1",
            "entity_type": EntityType.IOC_PROCESS_NAME.value,
            "value": "curl.exe",
        })
        assert obj["type"] == "file"
        assert "command_line" not in obj


class TestProcessImageRef:
    """Process SCOs link to their binary via image_ref (STIX-standard)."""

    def test_image_ref_resolves_when_the_binary_is_a_known_file(self):
        index = {"certutil.exe": "file--11111111-1111-4111-8111-111111111111"}
        sco = _build_process_sco("certutil.exe -urlcache -f http://x/a", index)
        assert sco["image_ref"] == index["certutil.exe"]
        assert "x_exe_name" not in sco

    def test_falls_back_to_a_label_when_the_binary_is_unknown(self):
        """Better a non-standard label than losing the binary name."""
        sco = _build_process_sco("mystery.exe --run", {})
        assert "image_ref" not in sco
        assert sco["x_exe_name"] == "mystery.exe"

    def test_bare_command_name_matches_the_exe_file(self):
        """Commands say `tar`; the File SCO is `tar.exe`.

        Caught in end-to-end validation — the unit tests passed because they
        used matching names on both sides.
        """
        index = {"tar.exe": "file--33333333-3333-4333-8333-333333333333"}
        sco = _build_process_sco("tar -xf a.pdf -C out", index)
        assert sco["image_ref"] == index["tar.exe"]

    def test_path_prefixes_are_stripped_before_matching(self):
        index = {"tar.exe": "file--22222222-2222-4222-8222-222222222222"}
        sco = _build_process_sco(r"C:\Windows\System32\tar.exe -xf a.pdf", index)
        assert sco["image_ref"] == index["tar.exe"]


class TestSectorVocabulary:
    """Identity sectors stay inside STIX industry-sector-ov."""

    def test_prompt_vocabulary_comes_from_the_schema(self):
        """The hand-maintained prose list had drifted from the real values."""
        from app.nodes.llm.entity_extraction import SYSTEM_PROMPT
        from app.services.stix_schema import industry_sector_vocab

        line = next(
            l for l in SYSTEM_PROMPT.split("\n")
            if l.startswith("- victim_sector")
        )
        vocab = industry_sector_vocab()
        assert vocab, "schema vocabulary must load"
        assert "financial-services" in line and "defense" in line
        # The specific drift the audit found: British spellings and values
        # that are not STIX at all. `sectors` is not enum-checked at
        # validation, so these would have shipped silently.
        for drifted in ("defence", "maritime", "real-estate"):
            assert drifted not in line

    def test_invalid_sector_is_coerced_and_the_original_preserved(self):
        from app.nodes.deterministic.bundle_validator import (
            _coerce_invalid_sectors,
        )

        obj = {
            "type": "identity", "id": "identity--1",
            "name": "Victim", "sectors": ["legal-services"],
        }
        corrections = _coerce_invalid_sectors([obj])
        assert obj["sectors"] == ["commercial"]
        # The source's own wording survives even though the vocabulary
        # cannot express it — one campaign report named "legal & professional
        # services" victims and that targeting was being lost outright.
        assert "sector:legal-services" in obj["labels"]
        assert corrections[0]["rule"] == "invalid_sector_coerced"

    def test_british_spelling_is_normalised(self):
        from app.nodes.deterministic.bundle_validator import (
            _coerce_invalid_sectors,
        )

        obj = {"type": "identity", "id": "identity--1", "sectors": ["defence"]}
        _coerce_invalid_sectors([obj])
        assert obj["sectors"] == ["defense"]

    def test_valid_sectors_are_untouched(self):
        from app.nodes.deterministic.bundle_validator import (
            _coerce_invalid_sectors,
        )

        obj = {
            "type": "identity", "id": "identity--1",
            "sectors": ["government-national", "technology"],
        }
        assert _coerce_invalid_sectors([obj]) == []
        assert obj["sectors"] == ["government-national", "technology"]

    def test_unmappable_sector_is_dropped_with_a_warning(self):
        """Better no sector than a non-standard one in a shared bundle."""
        from app.nodes.deterministic.bundle_validator import (
            _coerce_invalid_sectors,
        )

        obj = {"type": "identity", "id": "identity--1", "sectors": ["kangaroos"]}
        corrections = _coerce_invalid_sectors([obj])
        assert "sectors" not in obj
        assert corrections[0]["rule"] == "invalid_sector_dropped"
        assert corrections[0]["severity"] == "warn"


class TestReportName:
    """The Report SDO and the attack-flow object carry the source title.

    `title` is its own state key (Source.title); `metadata` never holds
    one. Reading only `metadata["title"]` named every real run's Report
    "Untitled CTI Report" — the three reports in the graph on 2026-09-12
    were indistinguishable by name — while the CompletedBundle row next
    to it had the right title.
    """

    @staticmethod
    def _normalized(base_state):
        return [
            asdict(NormalizedDraft(draft_id=d["draft_id"], composite_confidence=80))
            for d in base_state["drafts"]
        ]

    @staticmethod
    def _names(result):
        objs = result["stix_bundle"]["objects"]
        report = next(o for o in objs if o["type"] == "report")
        flow = next((o for o in objs if o["type"] == "attack-flow"), None)
        return report["name"], (flow["name"] if flow else None)

    async def test_state_title_names_report_and_flow(self, base_state):
        title = "[HUMAN-C] Espionage RAT report — cross-vendor transfer test"
        state = {**base_state, "normalized_drafts": self._normalized(base_state),
                 "title": title, "metadata": {"author": "Example Vendor"}}
        report_name, flow_name = self._names(await serialize_stix(state))
        assert report_name == title
        assert flow_name == title

    async def test_state_title_wins_over_metadata_title(self, base_state):
        """Matches CompletedBundle, so the row and the SDO cannot diverge."""
        state = {**base_state, "normalized_drafts": self._normalized(base_state),
                 "title": "From the queue", "metadata": {"title": "From metadata"}}
        report_name, _ = self._names(await serialize_stix(state))
        assert report_name == "From the queue"

    async def test_placeholder_title_falls_through_to_metadata(self, base_state):
        state = {**base_state, "normalized_drafts": self._normalized(base_state),
                 "title": "Untitled Source", "metadata": {"title": "From metadata"}}
        report_name, _ = self._names(await serialize_stix(state))
        assert report_name == "From metadata"

    async def test_no_title_anywhere_keeps_the_defaults(self, base_state):
        state = {**base_state, "normalized_drafts": self._normalized(base_state),
                 "title": "Untitled Source", "metadata": {}}
        report_name, flow_name = self._names(await serialize_stix(state))
        assert report_name == "Untitled CTI Report"
        assert flow_name in (None, "Extracted attack flow")


class TestDetectionRulesAreNotIndicators:
    """Detection rules stay out of the bundle as Indicators.

    `.cursorrules`: "IOCs are SCOs, NOT indicators. Detection rules are a
    separate pipeline." Serialization was emitting one Indicator per detection
    rule with `pattern` set to the rule's content and `pattern_type` to its
    type — so a vendor rule NAME shipped as a Sigma pattern that nothing can
    parse. One campaign bundle carried five.
    """

    async def test_no_indicator_sdos_are_emitted(self, sample_entities, sample_drafts):
        state = {
            "gates_enabled": True, "metadata": {"title": "T"},
            "source_reliability": 85,
            "validated_entities": sample_entities, "drafts": sample_drafts,
            "gate1_approved_draft_ids": [d["draft_id"] for d in sample_drafts],
            "detection_rules": [{
                "rule_id": "rule-1", "rule_type": "sigma",
                "rule_content": "Okta Admin Console Access Failure",
                "description": "Google SecOps rule name under Okta rule pack",
            }],
        }
        with patch(
            "app.nodes.deterministic.serialization.run_query",
            new=AsyncMock(return_value=[]),
        ):
            bundle = (await serialize_stix(state))["stix_bundle"]

        assert [o for o in bundle["objects"] if o["type"] == "indicator"] == []

    async def test_the_reference_is_preserved_on_the_report(
        self, sample_entities, sample_drafts,
    ):
        """Dropping the Indicator must not lose the provenance."""
        state = {
            "gates_enabled": True, "metadata": {"title": "T"},
            "source_reliability": 85,
            "validated_entities": sample_entities, "drafts": sample_drafts,
            "gate1_approved_draft_ids": [d["draft_id"] for d in sample_drafts],
            "detection_rules": [{
                "rule_id": "rule-1", "rule_type": "sigma",
                "rule_content": "Okta Admin Console Access Failure",
                "description": "Google SecOps rule name under Okta rule pack",
            }],
        }
        with patch(
            "app.nodes.deterministic.serialization.run_query",
            new=AsyncMock(return_value=[]),
        ):
            bundle = (await serialize_stix(state))["stix_bundle"]

        report = next(o for o in bundle["objects"] if o["type"] == "report")
        refs = report.get("external_references", [])
        assert any("detection-rule" in r.get("source_name", "") for r in refs)


class TestRuleBodyGuard:
    """Rule names must not be accepted as rule content."""

    @pytest.mark.parametrize("name", [
        "Okta Admin Console Access Failure",
        "Okta Suspicious Actions from Anonymized IP",
        # Reads as a boolean expression but is a title — matching the English
        # word "or" was the first version's mistake.
        "O365 SharePoint Bulk File Access or Download via PowerShell",
    ])
    def test_rule_names_are_rejected(self, name):
        from app.nodes.llm.entity_extraction import _process_detection_rules

        assert _process_detection_rules(
            [{"rule_type": "sigma", "rule_content": name}]
        ) == []

    @pytest.mark.parametrize("body", [
        "title: Suspicious\ndetection:\n  sel:\n    EventID: 4625",
        'process.name == "certutil.exe"',
        'rule t { strings: $a = "x" condition: $a }',
        "index=main sourcetype=WinEventLog | stats count",
    ])
    def test_real_rule_bodies_are_kept(self, body):
        from app.nodes.llm.entity_extraction import _process_detection_rules

        assert len(_process_detection_rules(
            [{"rule_type": "sigma", "rule_content": body}]
        )) == 1


class TestOneJunkRuleTypeIsNotFatal:
    """An unrecognised `rule_type` costs that rule, not the whole extraction.

    `DetectionRuleItem.rule_type` used to be typed as the `DetectionRuleType`
    enum, so a single rule tagged `ioc_command_line` — an ENTITY type the model
    confused for a rule type — failed Pydantic validation for the ENTIRE
    `extract_entities` tool output. Every entity in a hundred-entity report,
    lost to one bad line.

    `_process_detection_rules` already handled it correctly: log and skip. Two
    layers disagreed about the blast radius and the fatal one won. Same
    principle as commit 6c4cfe2, "one odd context key must not cost a whole
    pass".
    """

    SIGMA = "title: x\ndetection:\n  sel:\n    EventID: 4625\n  condition: sel"

    def test_tool_output_survives_an_unknown_rule_type(self):
        from app.nodes.llm.tool_models import ExtractEntitiesOutput

        out = ExtractEntitiesOutput.model_validate({
            "entities": [],
            "detection_rules": [
                {"rule_type": "ioc_command_line", "rule_content": "powershell -enc AAA"},
                {"rule_type": "sigma", "rule_content": self.SIGMA},
            ],
            "is_sequential": True,
            "sequentiality_rationale": "narrative",
        })
        assert len(out.detection_rules) == 2

    def test_post_processor_drops_only_the_unknown_one(self):
        from app.nodes.llm.entity_extraction import _process_detection_rules

        kept = _process_detection_rules([
            {"rule_type": "ioc_command_line", "rule_content": "powershell -enc AAA"},
            {"rule_type": "sigma", "rule_content": self.SIGMA},
        ])
        assert [r["rule_type"] for r in kept] == ["sigma"]


class TestRegistryHiveExpansion:
    """STIX 2.1 rejects abbreviated registry hives.

    windows-registry-key's `key` carries a NEGATIVE pattern on
    ^HKLM|HKCC|HKCR|HKCU|HKU, so the abbreviation every CTI report actually
    writes is schema-invalid. One ransomware run hard-failed on three ordinary
    Terminal Server keys because of it.
    """

    def test_abbreviated_hive_is_expanded(self):
        from app.nodes.deterministic.serialization import _expand_registry_hive

        assert _expand_registry_hive(r"HKLM\SYSTEM\CurrentControlSet") == (
            r"HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet"
        )
        assert _expand_registry_hive(r"hkcu\Software\Run") == (
            r"HKEY_CURRENT_USER\Software\Run"
        )

    def test_bare_hive_with_no_subkey_is_expanded(self):
        from app.nodes.deterministic.serialization import _expand_registry_hive

        assert _expand_registry_hive("HKLM") == "HKEY_LOCAL_MACHINE"

    def test_full_hive_name_passes_through(self):
        from app.nodes.deterministic.serialization import _expand_registry_hive

        full = r"HKEY_LOCAL_MACHINE\SECURITY"
        assert _expand_registry_hive(full) == full

    def test_lookalike_prefix_is_left_alone(self):
        """Only a whole first path segment counts — not a mere prefix."""
        from app.nodes.deterministic.serialization import _expand_registry_hive

        assert _expand_registry_hive(r"HKLMX\Software") == r"HKLMX\Software"
        assert _expand_registry_hive("") == ""

    def test_built_sco_carries_the_expanded_key(self):
        """The value that actually reaches the bundle, not just the helper."""
        from app.graph.state import EntityType
        from app.nodes.deterministic.serialization import _build_sco

        obj = _build_sco(
            "windows-registry-key",
            "windows-registry-key--00000000-0000-4000-8000-000000000000",
            r"HKLM\SYSTEM\CurrentControlSet\Control\Terminal Server",
            EntityType.IOC_REGISTRY_KEY.value,
            "2026-01-01T00:00:00.000Z",
        )
        assert obj["key"].startswith("HKEY_LOCAL_MACHINE")


class TestUnmappableProceduresAreOmitted:
    """A draft with no resolved ATT&CK technique cannot be a valid x-procedure.

    x_technique_refs is required by the schema, so emitting one hard-fails
    bundle validation and loses every other procedure with it — which is how
    both ransomware runs died. The proportionate response is to omit that one
    procedure, loudly, and ship the rest.
    """

    def _state(self, drafts, normalized):
        return {
            "drafts": drafts,
            "normalized_drafts": normalized,
            "validated_entities": [],
            "detection_rules": [],
            "chunks": [],
            "is_sequential": False,
        }

    async def test_draft_without_resolved_technique_is_dropped(self, caplog):
        import logging
        from app.nodes.deterministic.serialization import serialize_stix

        drafts = [
            {"draft_id": "d1", "chunk_id": "c1", "name": "Good one",
             "techniques": [{"technique_id": "T1059.001", "stix_id": "attack-pattern--x"}]},
            {"draft_id": "d2", "chunk_id": "c2", "name": "Unmappable one",
             "techniques": []},
        ]
        normalized = [
            {"draft_id": "d1", "name": "Good one", "confidence": 80},
            {"draft_id": "d2", "name": "Unmappable one", "confidence": 80},
        ]
        with caplog.at_level(logging.WARNING):
            out = await serialize_stix(self._state(drafts, normalized))

        objects = out["stix_bundle"]["objects"]
        procs = [o for o in objects if o.get("type") == "x-procedure"]
        assert len(procs) == 1, "the unmappable draft must not be emitted"
        assert procs[0]["name"] == "Good one"
        assert all("x_technique_refs" in p for p in procs)
        assert "Unmappable one" in caplog.text, "the omission must be named, not silent"

    async def test_draft_whose_techniques_are_all_unresolved_is_dropped(self):
        """Techniques present but with no STIX id resolve to no refs at all."""
        from app.nodes.deterministic.serialization import serialize_stix

        drafts = [{"draft_id": "d1", "chunk_id": "c1", "name": "Unresolved",
                   "techniques": [{"technique_id": "T9999", "stix_id": None}]}]
        normalized = [{"draft_id": "d1", "name": "Unresolved", "confidence": 80}]
        out = await serialize_stix(self._state(drafts, normalized))
        procs = [o for o in out["stix_bundle"]["objects"] if o.get("type") == "x-procedure"]
        assert procs == []


# ── extension definitions ─────────────────────────────────────────

from app.nodes.deterministic.extension_definitions import (  # noqa: E402
    ATTACK_FLOW_AUTHOR_IDENTITY_ID,
    ATTACK_FLOW_EXTENSION_ID,
    X_PROCEDURE_AUTHOR_IDENTITY_ID,
    X_PROCEDURE_EXTENSION_ID,
    bundle_meta_ids,
)
from app.services import stix_schema  # noqa: E402


class TestExtensionDefinitions:
    """A custom type is only a STIX 2.1 extension if the bundle carries the
    extension-definition it declares. Every procedure declares ours; every
    attack-* object declares CTID's; both definitions ride in the bundle
    with their author identity, exactly as CTID's own example bundle does."""

    @staticmethod
    def _by_type(bundle: dict, t: str) -> list[dict]:
        return [o for o in bundle["objects"] if o["type"] == t]

    async def _multi(self, base_state) -> dict:
        normalized = [
            asdict(NormalizedDraft(draft_id=d["draft_id"], composite_confidence=80))
            for d in base_state["drafts"]
        ]
        result = await serialize_stix({**base_state, "normalized_drafts": normalized})
        return result["stix_bundle"]

    async def test_every_procedure_declares_the_extension(self, base_state):
        bundle = await self._multi(base_state)
        procs = self._by_type(bundle, X_PROCEDURE_TYPE)
        assert procs
        for p in procs:
            assert p["extensions"] == {X_PROCEDURE_EXTENSION_ID: {"extension_type": "new-sdo"}}

    async def test_bundle_embeds_the_x_procedure_definition_and_author_once(self, base_state):
        bundle = await self._multi(base_state)
        defs = [o for o in self._by_type(bundle, "extension-definition")
                if o["id"] == X_PROCEDURE_EXTENSION_ID]
        assert len(defs) == 1
        assert defs[0]["extension_types"] == ["new-sdo"]
        assert defs[0]["created_by_ref"] == X_PROCEDURE_AUTHOR_IDENTITY_ID
        authors = [o for o in self._by_type(bundle, "identity")
                   if o["id"] == X_PROCEDURE_AUTHOR_IDENTITY_ID]
        assert len(authors) == 1
        assert authors[0]["identity_class"] == "individual"

    async def test_attack_flow_objects_share_the_one_ctid_definition(self, base_state):
        bundle = await self._multi(base_state)
        assert self._by_type(bundle, "attack-flow"), "fixture should yield a flow"
        flow_objs = [o for o in bundle["objects"]
                     if o["type"] in ("attack-flow", "attack-operator", "attack-condition")]
        for o in flow_objs:
            assert o["extensions"] == {ATTACK_FLOW_EXTENSION_ID: {"extension_type": "new-sdo"}}, o["type"]
        defs = [o for o in self._by_type(bundle, "extension-definition")
                if o["id"] == ATTACK_FLOW_EXTENSION_ID]
        assert len(defs) == 1
        assert defs[0]["version"] == "2.0.0"
        assert [o for o in self._by_type(bundle, "identity")
                if o["id"] == ATTACK_FLOW_AUTHOR_IDENTITY_ID]

    async def test_single_procedure_bundle_carries_no_attack_flow_definition(self, base_state):
        drafts = [base_state["drafts"][0]]
        normalized = [asdict(NormalizedDraft(draft_id=drafts[0]["draft_id"], composite_confidence=80))]
        state = {
            **base_state, "drafts": drafts, "normalized_drafts": normalized,
            "gate1_approved_draft_ids": [drafts[0]["draft_id"]],
        }
        bundle = (await serialize_stix(state))["stix_bundle"]
        ids = {o["id"] for o in bundle["objects"]}
        assert X_PROCEDURE_EXTENSION_ID in ids
        assert ATTACK_FLOW_EXTENSION_ID not in ids
        assert ATTACK_FLOW_AUTHOR_IDENTITY_ID not in ids

    async def test_report_does_not_describe_bundle_metadata(self, base_state):
        bundle = await self._multi(base_state)
        report = self._by_type(bundle, "report")[0]
        refs = set(report["object_refs"])
        assert not (refs & bundle_meta_ids(bundle["objects"]))
        # It still describes the content: every procedure and the source identity.
        assert all(p["id"] in refs for p in self._by_type(bundle, X_PROCEDURE_TYPE))
        assert any(o["id"] in refs for o in self._by_type(bundle, "identity") if o.get("x_source_id"))

    async def test_definitions_pass_the_oasis_schema(self, base_state):
        bundle = await self._multi(base_state)
        defs = self._by_type(bundle, "extension-definition")
        assert len(defs) == 2
        for d in defs:
            assert stix_schema.validate_object(d) == [], d["id"]

    def test_no_unpublished_attack_flow_ids_survive_in_source(self):
        """The operator and condition builders once cited ids CTID never
        published; Attack Flow 2.0.0 has exactly one definition."""
        import inspect
        from app.nodes.deterministic import attack_conditions, attack_operators, serialization
        src = "".join(inspect.getsource(m) for m in (attack_operators, attack_conditions, serialization))
        assert "677b4ce7" not in src
        assert "53cf6cc8" not in src
