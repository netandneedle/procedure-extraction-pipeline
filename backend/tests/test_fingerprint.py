"""Tests for app.utils.fingerprint — the canonical x_fingerprint formula.

The single most important property is: normalization-time stamp ==
validator-time recompute on the same procedure. When they diverge, the
validator emits `fingerprint_recomputed` corrections on every bundle.

This module is the shared helper used by both call sites; tests here
freeze the behavior so future refactors of either site can't drift.
"""

from __future__ import annotations

from app.utils.fingerprint import (
    compute_fingerprint_for_draft,
    compute_fingerprint_for_stix_obj,
)


# Mirror of serialize.py's draft → STIX projection for the inputs the
# fingerprint reads. Lets us assert round-trip equivalence without
# requiring the full serializer.
def _project_draft_to_stix(draft: dict) -> dict:
    refs: list[str] = []
    for t in draft.get("techniques", []) or []:
        stix_id = t.get("stix_id")
        if stix_id:
            refs.append(stix_id)
        else:
            tid = t.get("technique_id", "")
            if tid:
                refs.append(f"attack-pattern--{tid}")

    # Serializer prefers draft.kill_chain_phases when explicitly set;
    # else derives from draft.techniques[*].tactic.
    kcp = draft.get("kill_chain_phases") or []
    if not kcp:
        seen: set[str] = set()
        derived: list[dict] = []
        for t in draft.get("techniques", []) or []:
            tactic = t.get("tactic")
            if tactic and tactic not in seen:
                seen.add(tactic)
                derived.append(
                    {"kill_chain_name": "mitre-attack", "phase_name": tactic}
                )
        kcp = derived

    return {
        "x_technique_refs": refs,
        "x_platforms": list(draft.get("platforms", []) or []),
        "kill_chain_phases": kcp,
    }


class TestFormulaStability:
    def test_deterministic(self):
        draft = {
            "techniques": [
                {"stix_id": "attack-pattern--aaa", "tactic": "execution"},
                {"stix_id": "attack-pattern--bbb", "tactic": "persistence"},
            ],
            "platforms": ["Windows", "Linux"],
        }
        assert (
            compute_fingerprint_for_draft(draft)
            == compute_fingerprint_for_draft(draft)
        )

    def test_order_independent_techniques(self):
        a = {
            "techniques": [
                {"stix_id": "attack-pattern--aaa", "tactic": "execution"},
                {"stix_id": "attack-pattern--bbb", "tactic": "persistence"},
            ],
            "platforms": ["Windows"],
        }
        b = {
            "techniques": [
                {"stix_id": "attack-pattern--bbb", "tactic": "persistence"},
                {"stix_id": "attack-pattern--aaa", "tactic": "execution"},
            ],
            "platforms": ["Windows"],
        }
        assert compute_fingerprint_for_draft(a) == compute_fingerprint_for_draft(b)

    def test_order_independent_platforms(self):
        a = {
            "techniques": [{"stix_id": "attack-pattern--aaa", "tactic": "execution"}],
            "platforms": ["Windows", "Linux"],
        }
        b = {
            "techniques": [{"stix_id": "attack-pattern--aaa", "tactic": "execution"}],
            "platforms": ["Linux", "Windows"],
        }
        assert compute_fingerprint_for_draft(a) == compute_fingerprint_for_draft(b)


class TestDraftStixRoundTrip:
    """The normalization stamp must equal what the validator would recompute
    against the serializer's emit. If these diverge, `fingerprint_recomputed`
    fires on every bundle — the project's S57 milestone.
    """

    def test_round_trip_basic(self):
        draft = {
            "techniques": [
                {"stix_id": "attack-pattern--aaa", "tactic": "execution"},
                {"stix_id": "attack-pattern--bbb", "tactic": "persistence"},
            ],
            "platforms": ["Windows"],
        }
        from_draft = compute_fingerprint_for_draft(draft)
        from_stix = compute_fingerprint_for_stix_obj(_project_draft_to_stix(draft))
        assert from_draft == from_stix

    def test_round_trip_with_explicit_kill_chain_phases(self):
        """When draft explicitly sets kill_chain_phases (e.g. drafting node
        emits them), the serializer uses that list verbatim. The normalizer
        used to derive tactics from draft.techniques[*].tactic, which drifted
        when kill_chain_phases said something different. Both call sites now
        prefer explicit kill_chain_phases when present — this guards the fix.
        """
        draft = {
            "techniques": [
                {"stix_id": "attack-pattern--aaa", "tactic": "execution"},
                # Tactic in the technique row that ISN'T on kill_chain_phases:
                {"stix_id": "attack-pattern--bbb", "tactic": "persistence"},
            ],
            "platforms": ["Windows"],
            # Explicit kill_chain_phases that lists only one of the two tactics.
            # The serializer would emit this list (not the derived one).
            "kill_chain_phases": [
                {"kill_chain_name": "mitre-attack", "phase_name": "execution"},
            ],
        }
        from_draft = compute_fingerprint_for_draft(draft)
        from_stix = compute_fingerprint_for_stix_obj(_project_draft_to_stix(draft))
        assert from_draft == from_stix

    def test_round_trip_unresolved_technique(self):
        """When a technique has no stix_id, both sides fabricate
        `attack-pattern--<technique_id>` so the hash still matches."""
        draft = {
            "techniques": [{"technique_id": "T1059.001", "tactic": "execution"}],
            "platforms": ["Windows"],
        }
        from_draft = compute_fingerprint_for_draft(draft)
        from_stix = compute_fingerprint_for_stix_obj(_project_draft_to_stix(draft))
        assert from_draft == from_stix

    def test_round_trip_empty(self):
        draft = {"techniques": [], "platforms": []}
        from_draft = compute_fingerprint_for_draft(draft)
        from_stix = compute_fingerprint_for_stix_obj(_project_draft_to_stix(draft))
        assert from_draft == from_stix


class TestNormalizationStampMatchesValidator:
    """Cross-module integration — the actual call sites must agree, not
    just the shared helper. Imports the same functions both nodes use
    in production."""

    def test_full_pipeline_stamp_matches_recompute(self):
        from app.nodes.deterministic.bundle_validator import _recompute_fingerprints
        from app.nodes.deterministic.normalization import _compute_fingerprint

        draft = {
            "techniques": [
                {"stix_id": "attack-pattern--aaa", "tactic": "execution"},
                {"stix_id": "attack-pattern--bbb", "tactic": "persistence"},
            ],
            "platforms": ["Windows", "Linux"],
            "kill_chain_phases": [
                {"kill_chain_name": "mitre-attack", "phase_name": "execution"},
                {"kill_chain_name": "mitre-attack", "phase_name": "persistence"},
            ],
        }
        stamp = _compute_fingerprint(draft)

        # Project to STIX shape and run the validator's pass.
        proc = {
            "type": "x-procedure",
            "id": "x-procedure--00000000-0000-4000-8000-000000000000",
            "x_fingerprint": stamp,
            **_project_draft_to_stix(draft),
        }
        corrections = _recompute_fingerprints([proc])
        assert corrections == [], (
            "fingerprint_recomputed should NOT fire when the normalizer's "
            "stamp matches what the validator would recompute."
        )
