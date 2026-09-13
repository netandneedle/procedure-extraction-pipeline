"""Behavioral fingerprint for x-procedure objects.

Single source of truth for the x_fingerprint formula. Imported by
both `normalization._compute_fingerprint` (stamps the fingerprint
onto the draft before serialization) and
`bundle_validator._recompute_fingerprints` (recomputes against the
serialized object and corrects any drift).

The two call sites historically drifted: normalization read tactics
from `draft.techniques[*].tactic`; the validator read them from
`obj.kill_chain_phases[*].phase_name`. When the drafting node set
`kill_chain_phases` explicitly, the serializer used that list and the
validator's recompute disagreed with the normalizer's stamp, so
`fingerprint_recomputed` fired on every procedure on every bundle.
Routing both call sites through this module keeps the formula
canonical.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


_ATTACK_PREFIX = "attack-pattern--"
_MITRE_ATTACK_KILL_CHAIN = "mitre-attack"


def _technique_refs_from_draft(techniques: Iterable[dict]) -> list[str]:
    """Extract canonical technique refs from a draft's techniques list.

    Prefers the resolved STIX UUID (matches what the serializer writes
    to x_technique_refs); falls back to an attack-pattern-- prefix on the
    technique_id for an unresolved entry. The serializer omits unresolved
    techniques rather than emitting that form, so a draft with one hashes
    differently from its serialized object; the validator's recompute
    corrects the stamp in that case.
    """
    refs: list[str] = []
    for t in techniques or []:
        stix_id = t.get("stix_id")
        if stix_id:
            refs.append(stix_id)
            continue
        tid = t.get("technique_id", "")
        if tid:
            refs.append(f"{_ATTACK_PREFIX}{tid}")
    return refs


def _tactics_from_draft(draft: dict) -> list[str]:
    """Resolve the tactic set from a draft using the same precedence
    the serializer applies when emitting kill_chain_phases:
      1. Explicit `draft.kill_chain_phases` entries when present.
      2. Otherwise derive from `draft.techniques[*].tactic`.

    Returns a deduped list (set is collapsed before sort downstream).
    """
    seen: set[str] = set()
    kcp = draft.get("kill_chain_phases") or []
    for phase in kcp:
        if not isinstance(phase, dict):
            continue
        if phase.get("kill_chain_name") != _MITRE_ATTACK_KILL_CHAIN:
            continue
        name = phase.get("phase_name")
        if name:
            seen.add(name)
    if seen:
        return list(seen)
    for t in draft.get("techniques") or []:
        tactic = t.get("tactic")
        if tactic:
            seen.add(tactic)
    return list(seen)


def _tactics_from_stix_obj(obj: dict) -> list[str]:
    """Resolve the tactic set from a serialized x-procedure STIX object."""
    return [
        phase.get("phase_name", "")
        for phase in (obj.get("kill_chain_phases") or [])
        if isinstance(phase, dict)
        and phase.get("kill_chain_name") == _MITRE_ATTACK_KILL_CHAIN
        and phase.get("phase_name")
    ]


def _hash_signature(
    technique_refs: Iterable[str],
    platforms: Iterable[str],
    tactics: Iterable[str],
) -> str:
    """Canonical hash: sha256(sorted refs | sorted platforms | sorted tactics)[:32].

    Order-independent (each tuple is sorted). 128 bits is enough to
    avoid collisions at the scale we generate procedures (< 10^4 per
    bundle, < 10^9 cumulative across the archive).
    """
    parts = [
        "|".join(sorted(technique_refs)),
        "|".join(sorted(platforms)),
        "|".join(sorted(set(tactics))),
    ]
    signature = "::".join(parts)
    return hashlib.sha256(signature.encode("utf-8")).hexdigest()[:32]


def compute_fingerprint_for_draft(draft: dict) -> str:
    """Compute x_fingerprint at normalization time, before serialization.

    Matches what the serializer would write + what the validator would
    recompute. Stamping this onto the draft prevents the validator
    from emitting `fingerprint_recomputed` corrections on fresh runs.
    """
    technique_refs = _technique_refs_from_draft(draft.get("techniques", []))
    platforms = draft.get("platforms", []) or []
    tactics = _tactics_from_draft(draft)
    return _hash_signature(technique_refs, platforms, tactics)


def compute_fingerprint_for_stix_obj(obj: dict) -> str:
    """Recompute x_fingerprint from a serialized x-procedure STIX object.

    Used by the validator to detect/correct stale fingerprints. Must
    yield the same hash that `compute_fingerprint_for_draft` produced
    upstream when the serializer's emit path is lossless on the inputs.
    """
    technique_refs = obj.get("x_technique_refs", []) or []
    platforms = obj.get("x_platforms", []) or []
    tactics = _tactics_from_stix_obj(obj)
    return _hash_signature(technique_refs, platforms, tactics)
