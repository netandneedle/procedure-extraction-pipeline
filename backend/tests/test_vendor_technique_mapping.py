"""Tests for reading the vendor's own ATT&CK mapping table.

The table is the report authors' considered judgement about what they
observed, and until this module existed nothing in the pipeline read it —
on one campaign report it names `T1204.004 Malicious Copy and Paste` outright while
the picker spent two runs landing on neighbouring sub-techniques.
"""

from __future__ import annotations

from app.services.vendor_technique_mapping import extract_vendor_technique_ids


# Shaped like a vendor's ATT&CK mapping table as Docling renders it, ligature
# damage and all ("Certicates" is what Docling produces for "Certificates").
# Technique names are ATT&CK's own; the campaign id is synthetic.
REAL_TABLE = """## MITRE ATT&CK Mapping

The following techniques are associated with Campaign 00.001 based on
activity observed at the time of publication.

## Resource Development

- T1588 Obtain Capabilities
- T1588.003 Code Signing Certicates

## Execution

- T1129 Shared Modules
- T1204 User Execution
- T1204.004 Malicious Copy and Paste
"""


def _section(text, classification="technique_reference", section_id="sec-1"):
    return {
        "section_id": section_id,
        "text": text,
        "classification": classification,
    }


def test_parses_real_vendor_table():
    tids, audit = extract_vendor_technique_ids([_section(REAL_TABLE)])
    assert tids == {
        "T1588", "T1588.003", "T1129", "T1204", "T1204.004",
    }
    assert len(audit) == 1
    assert audit[0]["section_id"] == "sec-1"
    assert audit[0]["technique_ids"] == sorted(tids)


def test_parent_and_subtechnique_both_captured():
    """`T1204` and `T1204.004` are distinct claims and both are kept — the
    reconciliation step needs the sub-technique to spot sibling mismatches."""
    tids, _ = extract_vendor_technique_ids([_section("- T1204\n- T1204.004")])
    assert tids == {"T1204", "T1204.004"}


def test_only_technique_reference_sections_are_scanned():
    """Narrative prose mentioning a T-ID is not the vendor's mapping — it's
    usually the report quoting someone else's analysis."""
    sections = [
        _section("- T1204.004 Malicious Copy and Paste"),
        _section("the actor used T1059.001 PowerShell",
                 classification="behavioral_narrative", section_id="sec-2"),
        _section("T1078 Valid Accounts", classification="contextual",
                 section_id="sec-3"),
    ]
    tids, audit = extract_vendor_technique_ids(sections)
    assert tids == {"T1204.004"}
    assert [a["section_id"] for a in audit] == ["sec-1"]


def test_near_miss_ids_ignored():
    """Only the T####[.###] shape counts."""
    tids, _ = extract_vendor_technique_ids(
        [_section("NT12345 and T99 and TA0002 and 1204 are not techniques")],
    )
    assert tids == set()


def test_tactic_ids_not_captured():
    """TA-prefixed tactic IDs share the table but are not techniques."""
    tids, _ = extract_vendor_technique_ids(
        [_section("## TA0002 Execution\n- T1204.004 Malicious Copy and Paste")],
    )
    assert tids == {"T1204.004"}


def test_section_without_ids_contributes_no_audit_entry():
    """One campaign report has a second technique_reference section — the vendor's
    internal taxonomy — that names no ATT&CK IDs at all."""
    sections = [
        _section("## Vendor Techniques\n- ClickFix Related-Lures\n- Download using cURL"),
        _section("- T1204.004", section_id="sec-2"),
    ]
    tids, audit = extract_vendor_technique_ids(sections)
    assert tids == {"T1204.004"}
    assert [a["section_id"] for a in audit] == ["sec-2"]


def test_no_mapping_table_is_normal():
    """Plenty of reports ship without one; it's a bonus signal, never a
    requirement."""
    assert extract_vendor_technique_ids([]) == (set(), [])
    assert extract_vendor_technique_ids(None) == (set(), [])
    assert extract_vendor_technique_ids(
        [_section("prose", classification="behavioral_narrative")],
    ) == (set(), [])


def test_missing_text_key_is_tolerated():
    """Sections come from LangGraph checkpoints, not only validated models."""
    tids, audit = extract_vendor_technique_ids([{"classification": "technique_reference"}])
    assert tids == set()
    assert audit == []
