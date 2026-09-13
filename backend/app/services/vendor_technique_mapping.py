"""Read the vendor's own ATT&CK mapping out of the source report.

Why this exists
---------------
Most vendor CTI reports end with a "MITRE ATT&CK Mapping" table listing the
techniques the authors say they observed. That is a considered judgement by
analysts who had the full incident in front of them — the closest thing to a
ground truth the pipeline will ever be handed.

The section classifier already labels those tables `technique_reference`, and
until now **nothing read them**. On one vendor report the table says, in as many
words, `T1204.004 Malicious Copy and Paste` — the exact technique the picker
spent two runs failing to land on.

What this is not
----------------
Not an answer key to copy. The table says what the vendor concluded about the
whole report; it carries no per-chunk attribution, so it cannot tell you WHICH
procedure a technique belongs to. Blindly injecting all ~17 of that report's
listed techniques into each of its 8 chunks would produce 130-odd review-lane
entries and bury the analyst.

So the mapping is used two ways, both modest:
  - it annotates the candidate pool, so the pick step can see that the report's
    own authors named a technique;
  - it corroborates a per-chunk signal (a brand hit) that already has chunk
    attribution.

It never drives an injection on its own. See `_reconcile_curated_knowledge`
in `nodes/llm/technique_extraction.py`.

A caveat worth knowing: `technique_reference` also collects malware capability
deep-dives, so an occasional T-ID here describes what a family CAN do rather
than what was observed. Since the mapping only annotates and corroborates,
that costs a slightly generous pool label and nothing more.
"""

from __future__ import annotations

import logging

from app.utils.text import TECHNIQUE_ID_SCAN_RE

logger = logging.getLogger(__name__)

# Only these section classes are scanned. `technique_reference` is where the
# classifier puts ATT&CK tables (see SectionClassification in graph/state.py).
_MAPPING_SECTIONS = frozenset({"technique_reference"})


def extract_vendor_technique_ids(
    classified_sections: list[dict] | None,
) -> tuple[set[str], list[dict]]:
    """Pull every ATT&CK technique ID the report itself names.

    Args:
        classified_sections: ClassifiedSection dicts from PipelineState.

    Returns:
        `(tids, audit)` — the union of T-IDs found, and one audit entry per
        contributing section (`{section_id, technique_ids}`) so a later log
        line can say which part of the report supplied which ID. Both empty
        when the report has no mapping table, which is common: it is a bonus
        signal, never a requirement.
    """
    tids: set[str] = set()
    audit: list[dict] = []
    for section in classified_sections or []:
        if section.get("classification") not in _MAPPING_SECTIONS:
            continue
        found = set(TECHNIQUE_ID_SCAN_RE.findall(section.get("text") or ""))
        if not found:
            continue
        tids |= found
        audit.append({
            "section_id": section.get("section_id", ""),
            "technique_ids": sorted(found),
        })
    return tids, audit
