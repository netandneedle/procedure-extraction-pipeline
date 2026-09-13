"""Smoke tests for app.services.attack_data.

Verifies the direct-JSON STIX adapter loads the configured bundle and
returns the flat-dict catalogue shape the rest of the pipeline expects.
Skips automatically if the STIX file isn't on disk (so CI without the
bundle doesn't fail).

Run inside the api container:
    python -m pytest tests/test_attack_data.py -v
"""

from __future__ import annotations

import os

import pytest

from app.config import settings


_STIX_PATH = settings.attack_stix_path
_HAS_STIX = os.path.exists(_STIX_PATH)


# Skip the entire module if the STIX bundle isn't present.
pytestmark = pytest.mark.skipif(
    not _HAS_STIX,
    reason=f"STIX bundle not found at {_STIX_PATH}; run inside the api "
           f"container with the ./data:/data bind mount, or configure "
           f"settings.attack_stix_path.",
)


@pytest.fixture(scope="module")
def techniques():
    """Module-scoped fixture: load the catalogue once for all tests."""
    from app.services.attack_data import AttackData
    db = AttackData(_STIX_PATH)
    return db.all_techniques()


def test_catalogue_loads_with_substantial_size(techniques):
    """ATT&CK Enterprise should have well over 500 technique entries."""
    assert len(techniques) > 500, (
        f"only {len(techniques)} techniques loaded; expected >500. "
        f"STIX bundle may be malformed."
    )


def test_each_entry_has_required_fields(techniques):
    """Every entry should match the flat-dict shape used by callers."""
    required = {"external_id", "name", "stix_id", "description",
                "tactics", "platforms", "revoked", "deprecated"}
    for t in techniques[:10]:  # spot-check first 10; if shape is consistent it'll hold
        assert required.issubset(t.keys()), (
            f"missing fields: {required - set(t.keys())} "
            f"in entry {t.get('external_id', '<no-id>')}"
        )


def test_field_types_are_correct(techniques):
    """Type-check the flat-dict shape on a couple of entries."""
    for t in techniques[:5]:
        assert isinstance(t["external_id"], str)
        assert isinstance(t["name"], str)
        assert isinstance(t["stix_id"], str)
        assert isinstance(t["description"], str)
        assert isinstance(t["tactics"], list)
        assert isinstance(t["platforms"], list)
        assert isinstance(t["revoked"], bool)
        assert isinstance(t["deprecated"], bool)


def test_external_ids_are_attack_format(techniques):
    """external_id should match the ATT&CK T-number format."""
    import re
    pattern = re.compile(r"^T\d{4}(\.\d{3})?$")
    bad = [t["external_id"] for t in techniques if not pattern.match(t["external_id"])]
    assert not bad, f"non-conforming IDs found: {bad[:5]}"


def test_v19_specific_ids_present(techniques):
    """The v19 ATT&CK release introduced T1685, T1686, T1687, T1688,
    T1689, T1690, T1682, T1683 (and subs). If we're loading a v19+
    bundle, at least the parent IDs should be present."""
    ids = {t["external_id"] for t in techniques}
    v19_parents = {"T1685", "T1686", "T1687", "T1688", "T1689", "T1690", "T1682", "T1683", "T1684"}
    found = v19_parents & ids
    assert found, (
        f"none of the v19 parent IDs ({sorted(v19_parents)}) found in catalogue. "
        f"STIX bundle may be v18 or earlier."
    )


def test_revocation_metadata_populated(techniques):
    """v19 should have revoked entries (the T1562 family was retired).
    Non-empty revoked count proves the revoked flag is actually being
    read, not just defaulted to False."""
    revoked = [t for t in techniques if t["revoked"]]
    assert len(revoked) > 0, "expected non-zero revoked entries"


def test_t1059_001_powershell_is_present_and_well_formed(techniques):
    """Sanity check on a stable, ubiquitous technique. T1059.001 (PowerShell)
    has been in ATT&CK since the early days; it should always be present
    with a non-empty description and 'execution' tactic."""
    by_id = {t["external_id"]: t for t in techniques}
    powershell = by_id.get("T1059.001")
    assert powershell is not None, "T1059.001 (PowerShell) should always exist"
    assert powershell["name"], "PowerShell entry should have a name"
    assert "execution" in powershell["tactics"], (
        f"PowerShell should be tagged with execution tactic; got {powershell['tactics']}"
    )
    assert powershell["description"], "PowerShell description should not be empty"


def test_singleton_returns_same_instance():
    """get_attack_data() should return the cached singleton on repeat calls."""
    from app.services.attack_data import get_attack_data
    a = get_attack_data()
    b = get_attack_data()
    assert a is b, "get_attack_data should return the cached singleton"


# =============================================================================
# Revoked-by redirect + deprecated flag (post-validation helpers, task #52)
# =============================================================================
#
# These methods let _resolve_stix_ids transparently redirect revoked picks
# (e.g. T1562.001 from training knowledge) to their v19 replacement (T1685),
# and keep deprecated picks with a flag rather than dropping them.


@pytest.fixture(scope="module")
def db():
    """Fresh AttackData instance for redirect/deprecated tests. Separate
    from the singleton so tests don't depend on get_attack_data() call order."""
    from app.services.attack_data import AttackData
    return AttackData(_STIX_PATH)


def test_revoked_by_target_known_v18_to_v19_redirect(db):
    """T1562.001 (Disable or Modify Tools, v18) was revoked in v19 in favor
    of T1685 (or T1685.x). The exact target may shift between releases, but
    revoked_by_target should produce SOME T-number, not None."""
    target = db.revoked_by_target("T1562.001")
    assert target is not None, (
        "T1562.001 was revoked in v19; revoked_by_target should return a "
        "replacement T-number"
    )
    # Sanity: target should be a well-formed T-number
    import re
    assert re.match(r"^T\d{4}(\.\d{3})?$", target), (
        f"redirect target {target!r} is not a valid T-number"
    )


def test_revoked_by_target_returns_none_for_active_technique(db):
    """T1059.001 (PowerShell) is active and never revoked. Should return None."""
    assert db.revoked_by_target("T1059.001") is None


def test_revoked_by_target_returns_none_for_unknown_tid(db):
    """A nonexistent T-number should produce None, not raise."""
    assert db.revoked_by_target("T9999") is None
    assert db.revoked_by_target("T1234.567") is None


def test_revoked_by_target_returns_none_for_empty_tid(db):
    """Defensive: empty string should not raise, just return None."""
    assert db.revoked_by_target("") is None


def test_is_deprecated_active_technique_returns_false(db):
    """T1059.001 (PowerShell) is active, not deprecated."""
    assert db.is_deprecated("T1059.001") is False


def test_is_deprecated_unknown_tid_returns_false(db):
    """Unknown T-numbers are not deprecated (they're hallucinated). Should
    return False, not raise."""
    assert db.is_deprecated("T9999") is False
    assert db.is_deprecated("") is False


def test_is_deprecated_at_least_one_deprecated_in_v19(db):
    """v19 catalogue retains some deprecated entries. Find one and confirm
    is_deprecated() returns True for it."""
    deprecated_tids = [
        t["external_id"] for t in db.all_techniques() if t["deprecated"]
    ]
    if not deprecated_tids:
        pytest.skip("v19 bundle has no deprecated techniques (unexpected)")
    sample = deprecated_tids[0]
    assert db.is_deprecated(sample) is True, (
        f"is_deprecated({sample!r}) should be True; the technique is in "
        f"all_techniques() with deprecated=True"
    )


def test_get_technique_record_resolves_revoked_techniques(db):
    """get_technique_record should return the full record for a revoked
    technique (indexed counterpart of all_techniques())."""
    revoked_tids = [
        t["external_id"] for t in db.all_techniques() if t["revoked"]
    ]
    if not revoked_tids:
        pytest.skip("v19 bundle has no revoked techniques (unexpected)")
    sample = revoked_tids[0]
    rec = db.get_technique_record(sample)
    assert rec is not None
    assert rec["external_id"] == sample
    assert rec["revoked"] is True


def test_revoked_map_built_once_per_instance(db):
    """The revoked-by map is built lazily on first call and cached. A second
    call shouldn't rebuild — verifiable via the private _revoked_by_map state."""
    # Trigger build
    db.revoked_by_target("T1059.001")
    assert db._revoked_by_map is not None
    cached = db._revoked_by_map
    # Second call must reuse the same dict object
    db.revoked_by_target("T1003.001")
    assert db._revoked_by_map is cached


def test_revoked_map_substantial_size_for_v19(db):
    """v19 introduced large-scale revocations (T1562 family alone is 13
    revocations). The map should have well over 10 entries on a v19 bundle."""
    db.revoked_by_target("T1562.001")  # trigger build
    assert db._revoked_by_map is not None
    assert len(db._revoked_by_map) >= 10, (
        f"only {len(db._revoked_by_map)} revoked-by entries; expected >=10 "
        f"on a v19 bundle"
    )


# =============================================================================
# validate_technique_ids — used by the C+A+D propose-then-pick flow
# =============================================================================
# The "reason + propose" LLM call emits T-IDs from training memory; before
# they enter the candidate pool we validate against the v19 catalogue. The
# helper handles five outcomes: kept_active, redirected, dropped_hallucinated,
# dropped_revoked_no_redirect, dropped_deprecated.


def test_validate_technique_ids_keeps_active(db):
    """An active T-ID passes through unchanged."""
    valid, audit = db.validate_technique_ids(["T1059.001"])
    assert valid == {"T1059.001"}
    assert len(audit) == 1
    assert audit[0]["original_id"] == "T1059.001"
    assert audit[0]["outcome"] == "kept_active"


def test_validate_technique_ids_drops_hallucinations(db):
    """Nonexistent T-IDs are dropped with 'dropped_hallucinated' outcome."""
    valid, audit = db.validate_technique_ids(["T9999", "T1234.567"])
    assert valid == set()
    assert all(a["outcome"] == "dropped_hallucinated" for a in audit)
    assert {a["original_id"] for a in audit} == {"T9999", "T1234.567"}


def test_validate_technique_ids_redirects_revoked(db):
    """Revoked T-IDs with a usable redirect target return the new T-ID."""
    valid, audit = db.validate_technique_ids(["T1562.001"])
    # T1562.001 was revoked in v19. The redirect target should be active
    # and end up in the valid set.
    assert audit[0]["outcome"] == "redirected"
    assert audit[0]["original_id"] == "T1562.001"
    assert audit[0]["result_id"] in valid
    # Original is NOT in valid set (only the redirect target is).
    assert "T1562.001" not in valid


def test_validate_technique_ids_drops_deprecated(db):
    """Deprecated T-IDs are dropped from the candidate pool. Post-pick
    handling (_resolve_stix_ids) still flags them if the LLM picks one
    from training memory anyway."""
    deprecated_tids = [
        t["external_id"] for t in db.all_techniques() if t.get("deprecated")
    ]
    if not deprecated_tids:
        pytest.skip("v19 bundle has no deprecated techniques (unexpected)")
    sample = deprecated_tids[0]
    valid, audit = db.validate_technique_ids([sample])
    assert sample not in valid
    assert audit[0]["outcome"] == "dropped_deprecated"


def test_validate_technique_ids_handles_empty_and_whitespace(db):
    """Empty strings and whitespace-only inputs drop as hallucinations
    rather than raising."""
    valid, audit = db.validate_technique_ids(["", "  ", "T1059.001"])
    assert valid == {"T1059.001"}
    outcomes = [a["outcome"] for a in audit]
    assert outcomes.count("dropped_hallucinated") == 2
    assert outcomes.count("kept_active") == 1


def test_validate_technique_ids_mixed_batch(db):
    """A batch with active + revoked + hallucinated entries returns the
    expected union of usable IDs."""
    valid, audit = db.validate_technique_ids([
        "T1059.001",   # active -> kept
        "T1562.001",   # revoked -> redirected
        "T9999",       # hallucinated -> dropped
    ])
    # T1059.001 stays. T1562.001 is replaced by its redirect target.
    # Both should be present in valid; T9999 should not.
    assert "T1059.001" in valid
    assert "T9999" not in valid
    # Exactly one redirect target was added (T1562.001's replacement)
    redirect_audits = [a for a in audit if a["outcome"] == "redirected"]
    assert len(redirect_audits) == 1
    assert redirect_audits[0]["result_id"] in valid


def test_validate_technique_ids_empty_input(db):
    """Empty input yields empty set + empty audit, no errors."""
    valid, audit = db.validate_technique_ids([])
    assert valid == set()
    assert audit == []
