"""Tests for app.services.technique_pattern_brands.

The brand-expansion module bridges the brand-vs-mechanism source ambiguity
in CTI reports. Source reports often name a brand (ClickFix, MFA fatigue,
EvilProxy) without spelling out the underlying mechanism, so the LLM
can't pick the right technique without help. This module's substring
match → T-ID lookup is the deterministic backstop.

Tests cover:
  - Single brand match → expected T-IDs
  - Case-insensitive matching
  - Word-boundary handling (matches inside compound words like
    "CLICKFIX-flavored", does NOT match inside arbitrary substrings)
  - Multiple brands in same chunk
  - Empty / no-match inputs
  - Cross-chunk aggregation with audit log
"""

from __future__ import annotations

from app.services.technique_pattern_brands import (
    find_brand_techniques,
    find_brand_techniques_across_chunks,
)


# -----------------------------------------------------------------------------
# find_brand_techniques (single-chunk scan)
# -----------------------------------------------------------------------------


def test_clickfix_maps_to_t1204_004():
    """The anchor case the module was built for: ClickFix → T1204.004
    (Malicious Copy and Paste). If this regresses, the module's primary
    purpose is broken."""
    out = find_brand_techniques("The actor deployed a ClickFix fake captcha.")
    assert out == {"clickfix": ["T1204.004"]}


def test_clickfix_case_insensitive():
    """Source reports use varied casing — CLICKFIX, ClickFix, clickfix.
    All should match."""
    for variant in ("CLICKFIX", "ClickFix", "clickfix", "Clickfix"):
        out = find_brand_techniques(f"deployed {variant} on victim sites")
        assert "clickfix" in out, f"failed for casing {variant!r}"
        assert out["clickfix"] == ["T1204.004"]


def test_clickfix_matches_in_compound_words():
    """Word-boundary anchoring is loose: 'CLICKFIX-flavored' should match
    because the boundary sees a non-alphanumeric character on either side."""
    out = find_brand_techniques("a CLICKFIX-flavored campaign targeted users")
    assert out == {"clickfix": ["T1204.004"]}


def test_clickfix_does_not_match_random_substring():
    """The brand keyword is a discrete word, not a partial match. A word
    like 'antiClickfixing' should not register because the keyword sits
    inside an alphabetic run (the word-boundary regex blocks adjacency
    to alphanumerics on either side)."""
    out = find_brand_techniques("the antiClickfixing tool was deployed")
    assert out == {}


def test_mfa_fatigue_maps_to_t1621():
    out = find_brand_techniques("attackers used MFA fatigue to bypass auth")
    assert out == {"mfa fatigue": ["T1621"]}


def test_mfa_bombing_maps_to_t1621():
    """MFA bombing is a synonym for MFA fatigue — same target technique."""
    out = find_brand_techniques("an MFA bombing attack overwhelmed the user")
    assert out == {"mfa bombing": ["T1621"]}


def test_evilproxy_maps_to_t1557():
    out = find_brand_techniques("the EvilProxy phishing kit was used")
    assert out == {"evilproxy": ["T1557"]}


def test_aitm_phrase_maps_to_t1557():
    """AitM phrase forms — 'adversary-in-the-middle', 'AitM phishing'."""
    out = find_brand_techniques("an adversary-in-the-middle attack captured creds")
    assert "adversary-in-the-middle" in out
    assert out["adversary-in-the-middle"] == ["T1557"]


def test_drive_by_compromise_maps_to_t1189():
    out = find_brand_techniques("a drive-by compromise on the watering-hole site")
    assert "drive-by compromise" in out
    assert out["drive-by compromise"] == ["T1189"]


def test_watering_hole_maps_to_t1189():
    out = find_brand_techniques("the actor poisoned a watering hole frequented by execs")
    assert out == {"watering hole": ["T1189"]}


def test_multiple_brands_in_one_chunk():
    """A single chunk can mention multiple brands. Each should appear
    in the output dict with its own T-ID list."""
    text = (
        "The campaign combined a ClickFix lure with EvilProxy "
        "credential capture against MFA fatigue victims."
    )
    out = find_brand_techniques(text)
    assert "clickfix" in out
    assert "evilproxy" in out
    assert "mfa fatigue" in out
    assert out["clickfix"] == ["T1204.004"]
    assert out["evilproxy"] == ["T1557"]
    assert out["mfa fatigue"] == ["T1621"]


def test_repeated_brand_dedups():
    """If a brand is mentioned multiple times in a chunk, the output
    should still have one entry for it (not a list of duplicates)."""
    text = "ClickFix... again ClickFix... a third ClickFix mention."
    out = find_brand_techniques(text)
    assert list(out.keys()) == ["clickfix"]
    assert out["clickfix"] == ["T1204.004"]


def test_empty_text_returns_empty_dict():
    assert find_brand_techniques("") == {}


def test_no_brands_returns_empty_dict():
    out = find_brand_techniques(
        "The actor used PowerShell to download a payload via certutil."
    )
    assert out == {}


def test_none_input_returns_empty_dict():
    """Defensive: None / falsy text shouldn't raise."""
    assert find_brand_techniques(None) == {}  # type: ignore[arg-type]


# -----------------------------------------------------------------------------
# find_brand_techniques_across_chunks (aggregator)
# -----------------------------------------------------------------------------


def test_aggregator_unions_t_ids_across_chunks():
    chunks = [
        {"chunk_id": "c1", "text": "ClickFix campaign"},
        {"chunk_id": "c2", "text": "MFA fatigue attempts"},
    ]
    tids, audit = find_brand_techniques_across_chunks(chunks)
    assert tids == {"T1204.004", "T1621"}
    assert len(audit) == 2
    chunk_ids = {a["chunk_id"] for a in audit}
    assert chunk_ids == {"c1", "c2"}


def test_aggregator_audit_records_chunk_brand_techniques():
    """The audit log must preserve which chunk surfaced which brand —
    that's what makes the log useful for debugging picks back to their
    brand-expansion source."""
    chunks = [{"chunk_id": "c1", "text": "ClickFix lure"}]
    _, audit = find_brand_techniques_across_chunks(chunks)
    assert audit == [{
        "chunk_id": "c1",
        "brand": "clickfix",
        "techniques": ["T1204.004"],
    }]


def test_aggregator_falls_back_to_source_excerpt():
    """When chunk text is empty, fall back to source_excerpt — the
    chunker's verbatim quote can carry brand mentions even when the
    chunk's narrative text doesn't."""
    chunks = [{
        "chunk_id": "c1",
        "text": "",
        "source_excerpt": "Operators ran a ClickFix campaign.",
    }]
    tids, audit = find_brand_techniques_across_chunks(chunks)
    assert "T1204.004" in tids
    assert audit[0]["brand"] == "clickfix"


def test_aggregator_empty_chunks_returns_empty():
    tids, audit = find_brand_techniques_across_chunks([])
    assert tids == set()
    assert audit == []


def test_aggregator_chunks_without_brands_return_empty():
    chunks = [
        {"chunk_id": "c1", "text": "PowerShell downloaded a payload."},
        {"chunk_id": "c2", "text": "Registry persistence under HKCU\\Run."},
    ]
    tids, audit = find_brand_techniques_across_chunks(chunks)
    assert tids == set()
    assert audit == []


def test_same_brand_in_multiple_chunks_records_each():
    """If two chunks both mention ClickFix, the audit log gets two
    entries (one per chunk) but the union T-ID set still has just
    T1204.004 once."""
    chunks = [
        {"chunk_id": "c1", "text": "ClickFix step 1"},
        {"chunk_id": "c2", "text": "ClickFix step 2"},
    ]
    tids, audit = find_brand_techniques_across_chunks(chunks)
    assert tids == {"T1204.004"}
    assert len(audit) == 2
    assert all(a["brand"] == "clickfix" for a in audit)


# ---------------------------------------------------------------------------
# brand strength
# ---------------------------------------------------------------------------


def test_clickfix_is_definitional():
    """ClickFix without a copy-paste step isn't ClickFix — strong enough to
    act on when the picker chooses a different T1204 sub-technique."""
    from app.services.technique_pattern_brands import DEFINITIONAL, brand_strength
    assert brand_strength("clickfix") == DEFINITIONAL


def test_watering_hole_is_suggestive():
    """A watering hole is a targeting choice; delivery could be drive-by,
    supply chain, or a lure. It corroborates, never overrides."""
    from app.services.technique_pattern_brands import SUGGESTIVE, brand_strength
    assert brand_strength("watering hole") == SUGGESTIVE


def test_unknown_brand_defaults_to_suggestive():
    """A future map entry must opt in to definitional treatment rather than
    inherit the stronger behavior by omission."""
    from app.services.technique_pattern_brands import SUGGESTIVE, brand_strength
    assert brand_strength("some-new-brand") == SUGGESTIVE
    assert brand_strength("") == SUGGESTIVE
    assert brand_strength(None) == SUGGESTIVE


def test_strength_lookup_is_case_insensitive():
    from app.services.technique_pattern_brands import DEFINITIONAL, brand_strength
    assert brand_strength("ClickFix") == DEFINITIONAL
    assert brand_strength("CLICKFIX") == DEFINITIONAL


def test_every_mapped_brand_has_an_explicit_strength():
    """Guards the map against silent drift: adding a brand without deciding
    its strength would leave it quietly suggestive."""
    from app.services.technique_pattern_brands import (
        _BRAND_STRENGTH, _BRAND_TO_TECHNIQUES,
    )
    assert set(_BRAND_STRENGTH) == set(_BRAND_TO_TECHNIQUES)
