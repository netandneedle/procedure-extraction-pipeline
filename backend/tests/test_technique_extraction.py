"""Unit tests for app.nodes.llm.technique_extraction post-processor.

Focus on verbatim-match handling in _process_technique_mappings:
how confirmed/rejected decisions become synthesized TechniqueItems
with locked confidence and provenance, and how LLM-additional
techniques get de-duplicated against verbatim matches.

Pure-function tests; no LLM / DB / STIX dependencies.

    docker compose exec -T api python -m pytest tests/test_technique_extraction.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Provide a dummy API key so app.config import doesn't error.
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-dummy")

# Ensure backend is importable when running outside the api container.
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.nodes.llm.technique_extraction import (  # noqa: E402
    _format_chunks_for_prompt,
    _process_technique_mappings,
    _recalibrate_confidence,
)


# =============================================================================
# Helpers
# =============================================================================


def _chunk(cid):
    return {"chunk_id": cid, "text": "chunk text"}


def _verbatim(tid, substring="tool.exe -arg", actor="APT1"):
    return {
        "technique_id": tid,
        "matched_substring": substring,
        "source_actor_name": actor,
        "source_actor_type": "intrusion-set",
    }


def _llm_technique(tid, confidence=0.7, name="", tactic="execution", rationale="LLM said so"):
    return {
        "technique_id": tid,
        "technique_name": name,
        "tactic": tactic,
        "confidence": confidence,
        "rationale": rationale,
    }


def _decision(tid, decision, reason=None):
    d = {"technique_id": tid, "decision": decision}
    if reason:
        d["rejection_reason"] = reason
    return d


def _catalogue_entry(name="PowerShell", tactics=None):
    return {
        "name": name,
        "stix_id": "attack-pattern--abc",
        "description": "...",
        "tactics": tactics or ["execution"],
        "platforms": ["Windows"],
    }


# =============================================================================
# Confirmed verbatim match → 0.95 + provenance verbatim_match
# =============================================================================


def test_confirmed_verbatim_match_locks_high_confidence():
    raw_mappings = [{
        "chunk_id": "chk-1",
        "techniques": [],
        "verbatim_match_decisions": [
            _decision("T1059.001", "confirm"),
        ],
    }]
    chunks = [_chunk("chk-1")]
    verbatim_matches_by_chunk = {
        "chk-1": [_verbatim("T1059.001", "powershell.exe -enc")],
    }
    technique_lookup = {"T1059.001": _catalogue_entry("PowerShell")}

    result = _process_technique_mappings(
        raw_mappings, chunks,
        verbatim_matches_by_chunk=verbatim_matches_by_chunk,
        technique_lookup=technique_lookup,
    )

    assert "chk-1" in result
    items = result["chk-1"]
    assert len(items) == 1
    item = items[0]
    assert item["technique_id"] == "T1059.001"
    assert item["confidence"] == 0.95
    assert item["provenance"] == "verbatim_match"
    assert item["matched_substring"] == "powershell.exe -enc"
    assert item["source_actor_name"] == "APT1"
    assert "Verbatim match to MITRE procedure example" in item["rationale"]
    assert item["technique_name"] == "PowerShell"  # filled from catalogue
    assert item["tactic"] == "execution"            # filled from catalogue first tactic


# =============================================================================
# Rejected verbatim match → 0.3 + provenance verbatim_match_rejected
# =============================================================================


def test_rejected_verbatim_match_demotes_with_reason():
    raw_mappings = [{
        "chunk_id": "chk-1",
        "techniques": [],
        "verbatim_match_decisions": [
            _decision("T1059.001", "reject", "Defender mentioned the tool, not attacker."),
        ],
    }]
    chunks = [_chunk("chk-1")]
    verbatim_matches_by_chunk = {"chk-1": [_verbatim("T1059.001")]}
    technique_lookup = {"T1059.001": _catalogue_entry()}

    result = _process_technique_mappings(
        raw_mappings, chunks,
        verbatim_matches_by_chunk=verbatim_matches_by_chunk,
        technique_lookup=technique_lookup,
    )

    items = result["chk-1"]
    assert len(items) == 1
    item = items[0]
    assert item["confidence"] == 0.3
    assert item["provenance"] == "verbatim_match_rejected"
    assert "REJECTED" in item["rationale"]
    assert "Defender mentioned the tool" in item["rationale"]
    assert item["rejection_reason"] == "Defender mentioned the tool, not attacker."


def test_rejected_with_no_reason_uses_fallback():
    raw_mappings = [{
        "chunk_id": "chk-1",
        "techniques": [],
        "verbatim_match_decisions": [
            _decision("T1059.001", "reject"),  # no rejection_reason
        ],
    }]
    chunks = [_chunk("chk-1")]
    verbatim_matches_by_chunk = {"chk-1": [_verbatim("T1059.001")]}
    technique_lookup = {"T1059.001": _catalogue_entry()}

    result = _process_technique_mappings(
        raw_mappings, chunks,
        verbatim_matches_by_chunk=verbatim_matches_by_chunk,
        technique_lookup=technique_lookup,
    )

    items = result["chk-1"]
    assert items[0]["confidence"] == 0.3
    assert "no reason given" in items[0]["rationale"]


# =============================================================================
# LLM-additional technique → provenance llm
# =============================================================================


def test_llm_additional_technique_marked_with_llm_provenance():
    raw_mappings = [{
        "chunk_id": "chk-1",
        "techniques": [_llm_technique("T1140", confidence=0.6)],
        "verbatim_match_decisions": [],
    }]
    chunks = [_chunk("chk-1")]

    result = _process_technique_mappings(raw_mappings, chunks)

    items = result["chk-1"]
    assert len(items) == 1
    assert items[0]["technique_id"] == "T1140"
    assert items[0]["provenance"] == "llm"
    assert items[0]["confidence"] == 0.6


# =============================================================================
# Duplicate detection
# =============================================================================


def test_llm_duplicate_of_verbatim_match_is_skipped():
    """If LLM lists a verbatim-matched technique in the techniques array
    (despite prompt instructions), the LLM duplicate is skipped and the
    verbatim entry (locked confidence) wins."""
    raw_mappings = [{
        "chunk_id": "chk-1",
        "techniques": [
            _llm_technique("T1059.001", confidence=0.8, rationale="LLM also picked this"),
            _llm_technique("T1140", confidence=0.6),
        ],
        "verbatim_match_decisions": [_decision("T1059.001", "confirm")],
    }]
    chunks = [_chunk("chk-1")]
    verbatim_matches_by_chunk = {"chk-1": [_verbatim("T1059.001")]}
    technique_lookup = {"T1059.001": _catalogue_entry()}

    result = _process_technique_mappings(
        raw_mappings, chunks,
        verbatim_matches_by_chunk=verbatim_matches_by_chunk,
        technique_lookup=technique_lookup,
    )

    items = result["chk-1"]
    technique_ids = [item["technique_id"] for item in items]
    assert technique_ids.count("T1059.001") == 1
    assert "T1140" in technique_ids
    t1059 = next(i for i in items if i["technique_id"] == "T1059.001")
    assert t1059["provenance"] == "verbatim_match"
    assert t1059["confidence"] == 0.95


# =============================================================================
# Defensive cases
# =============================================================================


def test_orphan_decision_for_undetected_technique_skipped():
    """LLM emits a decision for a technique_id that wasn't in the
    DETECTED VERBATIM MATCHES block. Defensive skip."""
    raw_mappings = [{
        "chunk_id": "chk-1",
        "techniques": [],
        "verbatim_match_decisions": [_decision("T9999", "confirm")],
    }]
    chunks = [_chunk("chk-1")]
    verbatim_matches_by_chunk = {"chk-1": [_verbatim("T1059.001")]}
    technique_lookup = {
        "T1059.001": _catalogue_entry(),
        "T9999": _catalogue_entry("Bogus"),
    }

    result = _process_technique_mappings(
        raw_mappings, chunks,
        verbatim_matches_by_chunk=verbatim_matches_by_chunk,
        technique_lookup=technique_lookup,
    )

    # Orphan decision dropped; no real techniques means no chunk entry
    assert result.get("chk-1", []) == []


def test_invalid_technique_id_format_skipped():
    raw_mappings = [{
        "chunk_id": "chk-1",
        "techniques": [
            _llm_technique("not-a-tid", confidence=0.5),
            _llm_technique("T1059.001", confidence=0.5),
        ],
    }]
    chunks = [_chunk("chk-1")]
    result = _process_technique_mappings(raw_mappings, chunks)
    items = result["chk-1"]
    assert len(items) == 1
    assert items[0]["technique_id"] == "T1059.001"


def test_unknown_chunk_id_skipped():
    raw_mappings = [
        {"chunk_id": "chk-real", "techniques": [_llm_technique("T1059.001")]},
        {"chunk_id": "chk-ghost", "techniques": [_llm_technique("T1140")]},
    ]
    chunks = [_chunk("chk-real")]
    result = _process_technique_mappings(raw_mappings, chunks)
    assert "chk-real" in result
    assert "chk-ghost" not in result


# =============================================================================
# Backward compat: pre-Phase-2 callers (no verbatim args)
# =============================================================================


def test_no_verbatim_matches_works_without_kwargs():
    """Pre-Phase-2 call sites that don't pass verbatim_matches_by_chunk
    or technique_lookup still get the same shape as before. Backward
    compat for any caller (test, script) that hasn't been updated."""
    raw_mappings = [{
        "chunk_id": "chk-1",
        "techniques": [_llm_technique("T1059.001", confidence=0.7)],
    }]
    chunks = [_chunk("chk-1")]

    result = _process_technique_mappings(raw_mappings, chunks)

    assert "chk-1" in result
    items = result["chk-1"]
    assert len(items) == 1
    assert items[0]["technique_id"] == "T1059.001"
    assert items[0]["provenance"] == "llm"
    assert items[0]["confidence"] == 0.7


# =============================================================================
# Missing-decisions warning: LLM omitted decisions for a chunk that had matches
# =============================================================================


def test_missing_decisions_warning_logged(caplog):
    """If the LLM emits no verbatim_match_decisions for a chunk that
    DID have detected matches, the matches are dropped — but a warning
    must be logged so the silent drop is observable in production."""
    raw_mappings = [{
        "chunk_id": "chk-1",
        "techniques": [],
        # verbatim_match_decisions intentionally omitted
    }]
    chunks = [_chunk("chk-1")]
    verbatim_matches_by_chunk = {
        "chk-1": [_verbatim("T1059.001"), _verbatim("T1140")],
    }
    technique_lookup = {
        "T1059.001": _catalogue_entry(),
        "T1140": _catalogue_entry("Deobfuscate/Decode"),
    }

    import logging
    with caplog.at_level(logging.WARNING):
        _process_technique_mappings(
            raw_mappings, chunks,
            verbatim_matches_by_chunk=verbatim_matches_by_chunk,
            technique_lookup=technique_lookup,
        )

    assert any(
        "chk-1" in r.message and "no decisions" in r.message
        for r in caplog.records if r.levelno == logging.WARNING
    ), f"expected missing-decisions warning, got: {[r.message for r in caplog.records]}"


# =============================================================================
# _recalibrate_confidence: verbatim entries are skipped
# =============================================================================


def test_recalibrate_skips_verbatim_match_entries():
    """Verbatim matches lock confidence at extraction time. Recalibration
    rules (sub-technique promotion, overlap modulation, rationale cap)
    must NOT touch entries with provenance starting with 'verbatim_match'.
    Otherwise the locked 0.95 / 0.3 audit guarantee is broken."""
    chunks_by_id = {"chk-1": "totally unrelated chunk text about widgets"}
    technique_lookup = {
        "T1059.001": {
            "name": "PowerShell",
            "stix_id": "attack-pattern--abc",
            "description": "PowerShell is a powerful interactive shell.",
            "tactics": ["execution"],
            "platforms": ["Windows"],
        },
    }

    # Confirmed verbatim entry: locked at 0.95
    confirmed = {
        "technique_id": "T1059.001",
        "technique_name": "PowerShell",
        "tactic": "execution",
        "confidence": 0.95,
        "rationale": "Verbatim match to MITRE procedure example",
        "stix_id": "attack-pattern--abc",
        "provenance": "verbatim_match",
        "matched_substring": "powershell.exe -enc",
        "source_actor_name": "APT1",
    }
    # Rejected verbatim entry: locked at 0.3
    rejected = {
        "technique_id": "T1059.001",
        "technique_name": "PowerShell",
        "tactic": "execution",
        "confidence": 0.3,
        "rationale": "Verbatim match REJECTED by LLM: defender mention",
        "stix_id": "attack-pattern--abc",
        "provenance": "verbatim_match_rejected",
        "matched_substring": "powershell.exe -enc",
        "source_actor_name": "APT1",
        "rejection_reason": "defender mention",
    }
    technique_mappings = {"chk-1": [confirmed, rejected]}

    result = _recalibrate_confidence(
        technique_mappings, technique_lookup, chunks_by_id,
    )

    items = result["chk-1"]
    # Confidences unchanged
    assert items[0]["confidence"] == 0.95
    assert items[1]["confidence"] == 0.3
    # No raw-confidence shadow added (the recalibrate path didn't run)
    assert "confidence_raw" not in items[0]
    assert "confidence_raw" not in items[1]
    # Provenance preserved
    assert items[0]["provenance"] == "verbatim_match"
    assert items[1]["provenance"] == "verbatim_match_rejected"


def test_recalibrate_skips_overlap_modulation_for_definite_bucket():
    """C+A+D bucket-aware skip: when the LLM committed to 'definite' AND
    the source_quote is verbatim in the chunk, Rules 3 + 4 (overlap
    modulation, rationale grounding cap) are skipped. The bucket plus
    quote already evidence-grounded the pick; recalibration's vocabulary
    modulation would be double-jeopardy."""
    # Chunk vocabulary deliberately diverges from MITRE's description
    # (they say "exploit", chunk says "drove kernel"). Old behavior
    # would drag confidence down via overlap modulation.
    chunks_by_id = {"chk-1": "the operator drove kernel control through CVE-2026-x"}
    technique_lookup = {
        "T1190": {
            "name": "Exploit Public-Facing Application",
            "stix_id": "attack-pattern--t1190",
            "description": (
                "Exploit weaknesses in internet-facing software to gain "
                "initial access. Targets web applications and databases."
            ),
            "tactics": ["initial-access"],
            "platforms": ["Linux", "Windows"],
        },
    }
    # Definite bucket pick with confidence 0.95. No verbatim name match
    # ("Exploit Public-Facing Application" not in chunk text), so Rule 2
    # doesn't fire. Rule 3 would normally drag adjusted down to ~0.65
    # since overlap is near-zero.
    pick = {
        "technique_id": "T1190",
        "technique_name": "Exploit Public-Facing Application",
        "tactic": "initial-access",
        "confidence": 0.95,
        "confidence_bucket": "definite",
        "source_quote": "drove kernel control",
        "rationale": "the chunk demonstrates exploitation of an internet-facing system",
        "stix_id": "attack-pattern--t1190",
        "provenance": "llm",
    }
    result = _recalibrate_confidence({"chk-1": [pick]}, technique_lookup, chunks_by_id)
    # confidence preserved at 0.95 because bucket is 'definite'
    assert result["chk-1"][0]["confidence"] == 0.95


def test_recalibrate_skips_overlap_modulation_for_probable_bucket():
    """Same skip applies for 'probable' bucket — both bucket levels are
    bundle-bound and shouldn't be modulated by description overlap."""
    chunks_by_id = {"chk-1": "the actor used powershell to drop the loader"}
    technique_lookup = {
        "T1059.001": {
            "name": "PowerShell",
            "stix_id": "attack-pattern--t1059001",
            "description": (
                "PowerShell is an interactive command-line interface and "
                "scripting environment included in the Windows operating system."
            ),
            "tactics": ["execution"],
            "platforms": ["Windows"],
        },
    }
    pick = {
        "technique_id": "T1059.001",
        "technique_name": "PowerShell",
        "tactic": "execution",
        "confidence": 0.75,
        "confidence_bucket": "probable",
        "source_quote": "used powershell to drop the loader",
        "rationale": "powershell explicitly named in the source quote",
        "stix_id": "attack-pattern--t1059001",
        "provenance": "llm",
    }
    result = _recalibrate_confidence({"chk-1": [pick]}, technique_lookup, chunks_by_id)
    # Rule 2 (verbatim name anchor) WILL fire because "powershell" is in chunk
    # and adjusted < 0.85 — so confidence floors to 0.85, not preserves at 0.75.
    # The point of this test is that Rule 3's overlap modulation does NOT
    # subsequently drag it back down.
    assert result["chk-1"][0]["confidence"] == 0.85


def test_recalibrate_still_modulates_possible_bucket():
    """'possible' bucket picks STILL flow through recalibration. They're
    weak by design and the deterministic guardrails are appropriate."""
    chunks_by_id = {"chk-1": "the operator drove kernel control through CVE-2026-x"}
    technique_lookup = {
        "T1190": {
            "name": "Exploit Public-Facing Application",
            "stix_id": "attack-pattern--t1190",
            "description": (
                "Exploit weaknesses in internet-facing software to gain "
                "initial access. Targets web applications and databases."
            ),
            "tactics": ["initial-access"],
            "platforms": ["Linux", "Windows"],
        },
    }
    pick = {
        "technique_id": "T1190",
        "technique_name": "Exploit Public-Facing Application",
        "tactic": "initial-access",
        "confidence": 0.95,
        "confidence_bucket": "possible",
        "source_quote": "drove kernel control",
        "rationale": "the chunk demonstrates exploitation of an internet-facing system",
        "stix_id": "attack-pattern--t1190",
        "provenance": "llm",
    }
    result = _recalibrate_confidence({"chk-1": [pick]}, technique_lookup, chunks_by_id)
    # 'possible' bucket: Rules 3 + 4 still apply; chunk vocabulary diverges
    # from description, so overlap modulation drags adjusted down. Exact
    # value depends on tokenization — assert direction, not magnitude.
    assert result["chk-1"][0]["confidence"] < 0.95


def test_recalibrate_still_applies_to_llm_entries_alongside_verbatim():
    """Skipping verbatim entries must not skip LLM entries in the same
    chunk. Verifies the per-entry skip is per-entry, not per-chunk."""
    chunks_by_id = {"chk-1": "the attacker ran powershell.exe -enc on the host"}
    technique_lookup = {
        "T1059.001": {
            "name": "PowerShell",
            "stix_id": "attack-pattern--abc",
            "description": "PowerShell is an interactive shell on Windows hosts.",
            "tactics": ["execution"],
            "platforms": ["Windows"],
        },
    }
    verbatim = {
        "technique_id": "T1059.001",
        "technique_name": "PowerShell",
        "tactic": "execution",
        "confidence": 0.95,
        "rationale": "Verbatim match",
        "stix_id": "attack-pattern--abc",
        "provenance": "verbatim_match",
        "matched_substring": "powershell.exe -enc",
        "source_actor_name": "APT1",
    }
    llm_pick = {
        "technique_id": "T1059.001",
        "technique_name": "PowerShell",
        "tactic": "execution",
        "confidence": 0.4,
        "rationale": "powershell host attacker",  # grounded in chunk
        "stix_id": "attack-pattern--abc",
        "provenance": "llm",
    }
    technique_mappings = {"chk-1": [verbatim, llm_pick]}

    result = _recalibrate_confidence(
        technique_mappings, technique_lookup, chunks_by_id,
    )
    items = result["chk-1"]

    # Verbatim entry untouched
    assert items[0]["confidence"] == 0.95
    assert "confidence_raw" not in items[0]
    # LLM entry got the verbatim-name-anchor floor (PowerShell appears in chunk)
    # and/or overlap modulation — its confidence should have changed
    assert items[1].get("confidence_raw") == 0.4
    assert items[1]["confidence"] != 0.4


# =============================================================================
# Sub-technique auto-promotion threshold
#
# The promote rule has TWO thresholds:
#   - Relative gap of 0.05 (sub_overlap > parent_overlap + 0.05)
#   - Absolute floor of 0.20 (sub_overlap >= 0.20)
#
# Without the absolute floor, the rule misfired during a synthetic dry run:
# T1218 (parent, overlap 0.05) was promoted to T1218.008 Odbcconf (wrong sub,
# overlap 0.10) — relative gap was satisfied but both numbers were too low
# to trust. The floor ensures promotion only fires when there's real evidence
# of vocabulary alignment, not when both are near-zero noise.
# =============================================================================


def test_sub_promotion_skipped_when_both_overlaps_below_floor():
    """Both parent (0.05-ish) and sub (0.10-ish) below the 0.20 absolute
    floor — promotion must NOT fire even though the 0.05 gap is met.
    Regression test for the T1218.008 Odbcconf misfire pattern."""
    # Chunk vocabulary deliberately diverges from MITRE descriptions so
    # both overlaps are near-zero. The matched sub has marginally better
    # overlap due to one extra incidental token, but neither is high
    # enough in absolute terms to justify promoting.
    chunks_by_id = {"chk-cert": "operator drove certutil binary download"}
    technique_lookup = {
        "T1218": {
            "name": "System Binary Proxy Execution",
            "stix_id": "ap--1218",
            "description": "Adversaries may bypass process and signature-based defenses by proxying execution of malicious content with signed, or otherwise trusted, binaries.",
            "tactics": ["defense-evasion"],
            "platforms": ["Windows"],
        },
        "T1218.008": {
            "name": "Odbcconf",
            "stix_id": "ap--1218008",
            "description": "Adversaries may abuse odbcconf.exe to proxy execution of malicious payloads. odbcconf.exe is a Windows-native utility for managing Open Database Connectivity.",
            "tactics": ["defense-evasion"],
            "platforms": ["Windows"],
        },
    }
    pick = {
        "technique_id": "T1218",
        "technique_name": "System Binary Proxy Execution",
        "tactic": "defense-evasion",
        "confidence": 0.75,
        "confidence_bucket": "probable",
        "source_quote": "drove certutil binary download",
        "rationale": "x",
        "stix_id": "ap--1218",
        "provenance": "llm",
    }
    result = _recalibrate_confidence(
        {"chk-cert": [pick]}, technique_lookup, chunks_by_id,
    )
    # Promotion did NOT fire — pick is still the parent T1218.
    assert result["chk-cert"][0]["technique_id"] == "T1218"


def test_sub_promotion_fires_when_sub_overlap_above_floor():
    """Sub_overlap clears the 0.20 absolute floor AND beats the parent
    by more than 0.05 — promotion SHOULD fire. Sanity-check that the
    rule still works when the evidence is real."""
    # Chunk vocabulary aligns strongly with the sub's description; the
    # parent's description is generic.
    chunks_by_id = {"chk-ps": (
        "the actor used powershell scripting interactive command line "
        "interface to invoke encoded base64 payload on the windows host"
    )}
    technique_lookup = {
        "T1059": {
            "name": "Command and Scripting Interpreter",
            "stix_id": "ap--1059",
            "description": "Adversaries may abuse command and scripting interpreters to execute commands.",
            "tactics": ["execution"],
            "platforms": ["Linux", "Windows", "macOS"],
        },
        "T1059.001": {
            "name": "PowerShell",
            "stix_id": "ap--1059001",
            "description": "Adversaries may abuse PowerShell, an interactive command-line interface and scripting environment included in the Windows operating system. PowerShell can invoke encoded base64 payload commands.",
            "tactics": ["execution"],
            "platforms": ["Windows"],
        },
    }
    pick = {
        "technique_id": "T1059",
        "technique_name": "Command and Scripting Interpreter",
        "tactic": "execution",
        "confidence": 0.7,
        "confidence_bucket": "probable",
        "source_quote": "used powershell scripting interactive command line",
        "rationale": "x",
        "stix_id": "ap--1059",
        "provenance": "llm",
    }
    result = _recalibrate_confidence(
        {"chk-ps": [pick]}, technique_lookup, chunks_by_id,
    )
    # Promotion fired — pick is now the sub.
    assert result["chk-ps"][0]["technique_id"] == "T1059.001"


# =============================================================================
# _format_chunks_for_prompt: verbatim block rendering
# =============================================================================


def test_format_chunks_for_prompt_renders_verbatim_block():
    """The DETECTED VERBATIM MATCHES block is the LLM's contract for
    confirm/reject. Lock its rendering so tweaks to whitespace or
    surrounding text don't silently break LLM parsing."""
    chunks = [{
        "chunk_id": "chk-1",
        "text": "The actor ran powershell.exe -enc on the box.",
        "sequence_index": 1,
    }]
    verbatim_matches_by_chunk = {
        "chk-1": [{
            "technique_id": "T1059.001",
            "matched_substring": "powershell.exe -enc",
            "source_actor_name": "APT1",
            "source_actor_type": "intrusion-set",
        }],
    }
    out = _format_chunks_for_prompt(chunks, verbatim_matches_by_chunk)

    assert "DETECTED VERBATIM MATCHES" in out
    assert "confirm or reject each in your response" in out
    assert "T1059.001" in out
    assert "powershell.exe -enc" in out
    assert "APT1 (intrusion-set)" in out


def test_format_chunks_for_prompt_omits_block_when_no_matches():
    """Chunks without matches must NOT get an empty DETECTED VERBATIM
    MATCHES header — that would tell the LLM to emit decisions for
    nothing."""
    chunks = [{
        "chunk_id": "chk-1",
        "text": "Plain prose with no operative strings.",
        "sequence_index": 1,
    }]
    out = _format_chunks_for_prompt(chunks, verbatim_matches_by_chunk={})
    assert "DETECTED VERBATIM MATCHES" not in out


def test_format_chunks_for_prompt_handles_missing_actor_type():
    """If source_actor_type is empty, render just the actor name (no
    parenthesized type), not 'APT1 ()' which looks like a bug."""
    chunks = [{
        "chunk_id": "chk-1",
        "text": "Test chunk.",
        "sequence_index": 1,
    }]
    verbatim_matches_by_chunk = {
        "chk-1": [{
            "technique_id": "T1059.001",
            "matched_substring": "tool.exe -arg",
            "source_actor_name": "APT1",
            "source_actor_type": "",
        }],
    }
    out = _format_chunks_for_prompt(chunks, verbatim_matches_by_chunk)
    assert "from APT1)" in out
    assert "APT1 ()" not in out


# =============================================================================
# _resolve_stix_ids: revoked-by redirect + deprecated flag (task #52)
# =============================================================================
#
# Test the four resolution paths in _resolve_stix_ids:
#   1. Active passthrough  — direct lookup wins
#   2. Revoked redirect    — follow revoked-by to active replacement
#   3. Deprecated kept     — flag instead of dropping
#   4. Unresolved warning  — truly unknown ID, log + drop


from unittest.mock import MagicMock  # noqa: E402

from app.nodes.llm.technique_extraction import _resolve_stix_ids  # noqa: E402


def _active_lookup():
    """Catalogue with two active v19 techniques. Mirrors the shape produced
    by _load_technique_catalogue() for the test."""
    return {
        "T1059.001": {
            "name": "PowerShell",
            "stix_id": "attack-pattern--pwsh",
            "description": "...",
            "tactics": ["execution"],
            "platforms": ["Windows"],
        },
        "T1685": {
            "name": "Disable or Modify Tools",
            "stix_id": "attack-pattern--disable-tools",
            "description": "...",
            "tactics": ["defense-evasion"],
            "platforms": ["Windows", "Linux", "macOS"],
        },
    }


def _make_mock_attack_data(redirect_map=None, deprecated_records=None):
    """Build a MagicMock AttackData substitute for _resolve_stix_ids.

    redirect_map: {old_tid: new_tid} for revoked_by_target lookups
    deprecated_records: {tid: {name, stix_id}}; the mock auto-injects
        deprecated=True so rec.get("deprecated") matches the path-3 check
        in the resolver.
    """
    redirect_map = redirect_map or {}
    deprecated_records = deprecated_records or {}
    # Real records carry the deprecated flag; mock keyed-by-tid lookups
    # need it too, since the resolver now reads it directly.
    record_index = {
        tid: {**rec, "deprecated": True} for tid, rec in deprecated_records.items()
    }

    db = MagicMock()
    db.revoked_by_target = MagicMock(side_effect=lambda tid: redirect_map.get(tid))
    db.is_deprecated = MagicMock(side_effect=lambda tid: tid in deprecated_records)
    db.get_technique_record = MagicMock(side_effect=lambda tid: record_index.get(tid))
    return db


def test_resolve_stix_ids_active_passthrough(monkeypatch):
    """Path 1: LLM picked an active technique; resolver populates stix_id
    from the catalogue and corrects the name."""
    db = _make_mock_attack_data()
    monkeypatch.setattr(
        "app.services.attack_data.get_attack_data",
        lambda: db,
    )

    mappings = {"chk-1": [{
        "technique_id": "T1059.001",
        "technique_name": "wrong name",
        "stix_id": None,
    }]}
    result, warnings = _resolve_stix_ids(mappings, _active_lookup())

    assert warnings == []
    item = result["chk-1"][0]
    assert item["stix_id"] == "attack-pattern--pwsh"
    assert item["technique_name"] == "PowerShell"  # name corrected
    assert "redirected_from" not in item
    assert "is_deprecated" not in item
    # Direct hit shouldn't even consult the redirect map
    db.revoked_by_target.assert_not_called()


def test_resolve_stix_ids_redirects_revoked_to_active_replacement(monkeypatch):
    """Path 2: LLM picked a revoked v18 ID; resolver follows revoked-by to
    the v19 replacement, rewrites the mapping, and records original on
    redirected_from for audit."""
    db = _make_mock_attack_data(redirect_map={"T1562.001": "T1685"})
    monkeypatch.setattr(
        "app.services.attack_data.get_attack_data",
        lambda: db,
    )

    mappings = {"chk-1": [{
        "technique_id": "T1562.001",
        "technique_name": "Disable or Modify Tools",
        "stix_id": None,
    }]}
    result, warnings = _resolve_stix_ids(mappings, _active_lookup())

    assert warnings == []
    item = result["chk-1"][0]
    assert item["technique_id"] == "T1685"
    assert item["technique_name"] == "Disable or Modify Tools"
    assert item["stix_id"] == "attack-pattern--disable-tools"
    assert item["redirected_from"] == "T1562.001"
    assert "is_deprecated" not in item


def test_resolve_stix_ids_keeps_deprecated_with_flag(monkeypatch):
    """Path 3: LLM picked a deprecated technique. No redirect target available,
    but we keep the mapping with is_deprecated=True so the analyst can
    decide at gate review."""
    deprecated_records = {
        "T1497.003": {
            "name": "Time Based Evasion",
            "stix_id": "attack-pattern--time-evasion",
        },
    }
    db = _make_mock_attack_data(deprecated_records=deprecated_records)
    monkeypatch.setattr(
        "app.services.attack_data.get_attack_data",
        lambda: db,
    )

    mappings = {"chk-1": [{
        "technique_id": "T1497.003",
        "technique_name": "Time Based Evasion",
        "stix_id": None,
    }]}
    result, warnings = _resolve_stix_ids(mappings, _active_lookup())

    assert warnings == []
    item = result["chk-1"][0]
    assert item["technique_id"] == "T1497.003"  # unchanged
    assert item["technique_name"] == "Time Based Evasion"
    assert item["stix_id"] == "attack-pattern--time-evasion"
    assert item["is_deprecated"] is True
    assert "redirected_from" not in item


def test_resolve_stix_ids_warns_truly_unresolved(monkeypatch):
    """Path 4: LLM picked a hallucinated T-number not in active catalogue,
    not revoked, not deprecated. Warning logged, no stix_id populated."""
    db = _make_mock_attack_data()  # empty redirect + deprecated
    monkeypatch.setattr(
        "app.services.attack_data.get_attack_data",
        lambda: db,
    )

    mappings = {"chk-1": [{
        "technique_id": "T9999",
        "technique_name": "Bogus",
        "stix_id": None,
    }]}
    result, warnings = _resolve_stix_ids(mappings, _active_lookup())

    assert len(warnings) == 1
    assert "T9999" in warnings[0]
    item = result["chk-1"][0]
    # Mapping retained but stix_id stays None — downstream STIX serialization
    # will skip entries without stix_id
    assert item["stix_id"] is None


def test_resolve_stix_ids_redirect_target_not_active_falls_through(monkeypatch):
    """Edge case: revoked-by points at a target that itself isn't active.
    Single-hop policy: don't chase further. Falls through to deprecated
    or warning path."""
    db = _make_mock_attack_data(
        redirect_map={"T1OLD": "T1NEWBUT_NOT_ACTIVE"},
    )
    monkeypatch.setattr(
        "app.services.attack_data.get_attack_data",
        lambda: db,
    )

    mappings = {"chk-1": [{
        "technique_id": "T1OLD",
        "technique_name": "Old Technique",
        "stix_id": None,
    }]}
    result, warnings = _resolve_stix_ids(mappings, _active_lookup())

    # Redirect target wasn't active, original isn't deprecated -> warning.
    assert len(warnings) == 1
    assert "T1OLD" in warnings[0]
    assert result["chk-1"][0]["stix_id"] is None


def test_resolve_stix_ids_handles_attack_data_unavailable(monkeypatch):
    """If attack_data fails to load (e.g. STIX bundle missing in some test
    env), resolver degrades gracefully: active passthrough still works,
    everything else falls to the unresolved-warning path."""
    def _raise(*args, **kwargs):
        raise FileNotFoundError("STIX bundle not found")
    monkeypatch.setattr(
        "app.services.attack_data.get_attack_data",
        _raise,
    )

    mappings = {"chk-1": [
        {"technique_id": "T1059.001", "technique_name": "PowerShell", "stix_id": None},
        {"technique_id": "T1562.001", "technique_name": "old", "stix_id": None},
    ]}
    result, warnings = _resolve_stix_ids(mappings, _active_lookup())

    # Active still resolves
    items = result["chk-1"]
    pwsh = next(i for i in items if i["technique_id"] == "T1059.001")
    assert pwsh["stix_id"] == "attack-pattern--pwsh"

    # Revoked falls to warning (no attack_data to consult for redirect)
    revoked = next(i for i in items if i["technique_id"] == "T1562.001")
    assert revoked["stix_id"] is None
    assert any("T1562.001" in w for w in warnings)


# =============================================================================
# Denylist enforcement (_apply_technique_denylist) — deterministic guardrail
# =============================================================================

import pytest  # noqa: E402
from unittest.mock import AsyncMock, patch as _patch  # noqa: E402

from app.nodes.llm import technique_extraction as _te  # noqa: E402


async def test_denylisted_bundle_pick_demoted_to_review_flagged():
    # A denylisted bundle pick must not auto-ship — it's pulled from the bundle
    # and demoted into the review lane, flagged, so the analyst can still
    # promote it at Gate 1 (design decision B).
    bundle = {"chk-1": [{"technique_id": "T1059.001"}, {"technique_id": "T1204.004"}]}
    review = {}
    dl = {"values": {}, "technique_ids": {"T1204.004": {"pattern_id": "p1", "pattern": "noise"}}}
    with _patch.object(_te, "load_denylist", new=AsyncMock(return_value=dl)):
        flagged = await _te._apply_technique_denylist(bundle, review)
    assert flagged == 1
    # Non-denylisted pick stays in the bundle.
    assert bundle == {"chk-1": [{"technique_id": "T1059.001"}]}
    # Denylisted pick moved into review, flagged.
    moved = review["chk-1"][0]
    assert moved["technique_id"] == "T1204.004"
    assert moved["denylisted"] is True
    assert moved["denylist_pattern_id"] == "p1"


async def test_denylisted_review_pick_flagged_not_dropped():
    # A denylisted pick already in the review lane stays put, just flagged.
    bundle = {}
    review = {"chk-2": [{"technique_id": "T1204.004"}]}
    dl = {"values": {}, "technique_ids": {"T1204.004": {"pattern_id": "p1", "pattern": "noise"}}}
    with _patch.object(_te, "load_denylist", new=AsyncMock(return_value=dl)):
        flagged = await _te._apply_technique_denylist(bundle, review)
    assert flagged == 1
    assert review["chk-2"][0]["technique_id"] == "T1204.004"
    assert review["chk-2"][0]["denylisted"] is True


async def test_apply_technique_denylist_case_insensitive():
    bundle = {"chk-1": [{"technique_id": "t1204.004"}]}
    review = {}
    dl = {"values": {}, "technique_ids": {"T1204.004": {"pattern_id": "p1", "pattern": "noise"}}}
    with _patch.object(_te, "load_denylist", new=AsyncMock(return_value=dl)):
        flagged = await _te._apply_technique_denylist(bundle, review)
    assert flagged == 1
    assert bundle == {}  # emptied chunk key removed
    assert review["chk-1"][0]["technique_id"] == "t1204.004"
    assert review["chk-1"][0]["denylisted"] is True


async def test_apply_technique_denylist_noop_when_empty():
    bundle = {"chk-1": [{"technique_id": "T1059.001"}]}
    review = {}
    with _patch.object(
        _te, "load_denylist",
        new=AsyncMock(return_value={"values": {}, "technique_ids": {}}),
    ):
        flagged = await _te._apply_technique_denylist(bundle, review)
    assert flagged == 0
    assert bundle == {"chk-1": [{"technique_id": "T1059.001"}]}
    assert review == {}


# ---------------------------------------------------------------------------
# Duplicate technique picks (one ransomware run)
# ---------------------------------------------------------------------------

class TestDedupePicks:
    """One chunk must not carry the same technique twice.

    The duplicate is not merely redundant. It reaches Gate 1, where a
    reviewer that spots it can only say remove_technique_ids=[<id>] — which
    removes BOTH copies and leaves a procedure with no techniques at all,
    hard-failing the x-procedure schema. That is how one ransomware run died.
    """

    def test_duplicate_technique_collapses_to_one(self):
        from app.nodes.llm.technique_extraction import _dedupe_picks

        picks = [
            {"technique_id": "T1482", "confidence_bucket": "definite", "confidence": 0.9},
            {"technique_id": "T1482", "confidence_bucket": "definite", "confidence": 0.9},
        ]
        assert [p["technique_id"] for p in _dedupe_picks(picks)] == ["T1482"]

    def test_strongest_entry_survives(self):
        from app.nodes.llm.technique_extraction import _dedupe_picks

        picks = [
            {"technique_id": "T1059", "confidence_bucket": "possible", "confidence": 0.3},
            {"technique_id": "T1059", "confidence_bucket": "definite", "confidence": 0.95},
        ]
        kept = _dedupe_picks(picks)
        assert len(kept) == 1
        assert kept[0]["confidence_bucket"] == "definite"

    def test_order_follows_first_occurrence(self):
        """Bundle technique order stays stable across runs."""
        from app.nodes.llm.technique_extraction import _dedupe_picks

        picks = [
            {"technique_id": "T1003", "confidence_bucket": "probable", "confidence": 0.7},
            {"technique_id": "T1482", "confidence_bucket": "possible", "confidence": 0.2},
            {"technique_id": "T1003", "confidence_bucket": "definite", "confidence": 0.9},
        ]
        kept = _dedupe_picks(picks)
        assert [p["technique_id"] for p in kept] == ["T1003", "T1482"]
        assert kept[0]["confidence"] == 0.9

    def test_distinct_techniques_are_untouched(self):
        from app.nodes.llm.technique_extraction import _dedupe_picks

        picks = [
            {"technique_id": "T1059", "confidence_bucket": "definite", "confidence": 0.9},
            {"technique_id": "T1482", "confidence_bucket": "probable", "confidence": 0.7},
        ]
        assert len(_dedupe_picks(picks)) == 2

    def test_split_by_bucket_dedupes(self):
        """The dedup has to be on the path the node actually takes."""
        from app.nodes.llm.technique_extraction import _split_by_bucket

        bundle, review = _split_by_bucket({
            "chk-1": [
                {"technique_id": "T1482", "confidence_bucket": "definite", "confidence": 0.9},
                {"technique_id": "T1482", "confidence_bucket": "definite", "confidence": 0.9},
            ],
        })
        assert [p["technique_id"] for p in bundle["chk-1"]] == ["T1482"]
        assert review == {}


class TestPickerCoverageIsReconciled:
    """A chunk the pick step never answers for must not vanish in silence.

    The loop that builds technique_mappings is driven entirely by what the
    LLM returned, so an omitted chunk produces no key, no error and no trace.
    On one ransomware run the picker answered for 28 of 32 chunks; the four it
    dropped had correct propose-step objectives, and the pipeline only
    noticed three nodes later, as a schema error pointing at the wrong thing.

    A chunk that loses only SOME of its techniques this way ships quietly
    under-mapped forever. This warning is the only thing that can say so.
    """

    def _chunks(self):
        return [{"chunk_id": "chk-aaa"}, {"chunk_id": "chk-bbb"}]

    def test_missing_chunk_is_named_in_a_warning(self, caplog):
        import logging
        from app.nodes.llm.technique_extraction import _process_technique_mappings

        raw = [{"chunk_id": "chk-aaa", "techniques": [
            {"technique_id": "T1059.001", "confidence": 0.9,
             "confidence_bucket": "definite", "source_quote": "q"},
        ]}]
        with caplog.at_level(logging.WARNING):
            out = _process_technique_mappings(raw, self._chunks())

        assert "chk-aaa" in out
        assert "chk-bbb" not in out
        assert "chk-bbb" in caplog.text, (
            "the chunk the picker skipped must be named"
        )

    def test_full_coverage_stays_quiet(self, caplog):
        import logging
        from app.nodes.llm.technique_extraction import _process_technique_mappings

        raw = [
            {"chunk_id": c["chunk_id"], "techniques": [
                {"technique_id": "T1059.001", "confidence": 0.9,
                 "confidence_bucket": "definite", "source_quote": "q"},
            ]}
            for c in self._chunks()
        ]
        with caplog.at_level(logging.WARNING):
            _process_technique_mappings(raw, self._chunks())

        assert "returned no mapping" not in caplog.text
