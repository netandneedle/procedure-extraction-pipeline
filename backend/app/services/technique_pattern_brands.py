"""Brand-name → ATT&CK technique-ID expansion for the C+A+D extract step.

Why this exists
---------------
Some CTI brand/pattern names (ClickFix, MFA fatigue, EvilProxy AitM, etc.)
appear in source reports as the LABEL for an underlying technique pattern,
without the source spelling out the mechanism. The pipeline's policy is
to NOT extract these brand names as entities (they're "technique patterns,
not malware/tools" — see the NOT MALWARE, NOT TOOLS rule in the
entity-extraction prompt), but their technique mapping
still needs to land. The pick step's source-quote requirement means the
LLM can only pick a technique if it has chunk-text evidence for it, so
when a chunk says "deployed a CLICKFIX fake captcha" without describing
"press Win+R and paste a clipboard payload," the LLM fishes for the
closest T1204 sub and lands on T1204.001 / T1204.002 instead of the
correct T1204.004 (Malicious Copy and Paste).

This module bridges that gap deterministically. It scans chunk text for
known brand-name substrings and maps them to ATT&CK technique IDs that
the propose+validate+union step injects into the unified candidate
pool, so the pick step's LLM call sees the right technique as a
candidate. The pick step's normal evidence checks still apply: the
brand name itself is a verbatim chunk substring, so the LLM can use it
as the source_quote without violating the auto-cap rule.

Out of scope
------------
- Prompt-side brand awareness (would mix deterministic and LLM logic).
- Chunker-side expansion (chunker shouldn't need to know technique IDs).
- Adding brands as entities at Gate 0 (explicitly forbidden by the
  entity-extraction policy on technique-pattern names).
"""

from __future__ import annotations

import re


# Map of known CTI brand / pattern names to ATT&CK technique IDs.
#
# Conservative starter set: only entries where the brand → technique
# mapping is well-validated. Speculative mappings (e.g. Browser-in-the-
# Browser → T1056.003 Web Portal Capture) are intentionally omitted; add
# only after a careful technique-mapping decision.
#
# Substring-match keys are lowercase. Each key maps to a list of T-IDs
# (multiple when the brand integrates more than one technique — e.g.,
# AitM / EvilProxy is both "Adversary-in-the-Middle" and a credential-
# capture technique depending on the deployment).
#
# Add entries via documented mappings only. When in doubt, leave it
# out — false-positive technique candidates are worse than missed ones
# because they pollute every chunk that mentions the substring.
_BRAND_TO_TECHNIQUES: dict[str, list[str]] = {
    "clickfix": ["T1204.004"],          # User Execution: Malicious Copy and Paste
    "mfa fatigue": ["T1621"],           # MFA Request Generation
    "mfa bombing": ["T1621"],
    "push fatigue": ["T1621"],
    "evilproxy": ["T1557"],             # Adversary-in-the-Middle
    "adversary-in-the-middle": ["T1557"],
    "aitm phishing": ["T1557"],
    "drive-by compromise": ["T1189"],   # Drive-by Compromise
    "drive-by attack": ["T1189"],
    "watering hole": ["T1189"],
}


# How firmly a brand implies its technique.
#
#   definitional — the brand IS the technique. "ClickFix" without a
#       copy-paste step isn't ClickFix; "MFA fatigue" without repeated push
#       prompts isn't MFA fatigue. Strong enough to act on when the model
#       picks a different sibling.
#   suggestive — the brand describes a strategy that USUALLY uses the
#       technique but doesn't entail it. A watering hole is a targeting
#       choice; the delivery could be drive-by, supply chain, or a lure.
#       Corroborates a pick, never overrides one.
#
# Only `definitional` entries drive review-lane injection. Anything not
# listed here defaults to `suggestive`, so a future map addition has to
# opt in to the stronger treatment rather than inherit it silently.
DEFINITIONAL = "definitional"
SUGGESTIVE = "suggestive"

_BRAND_STRENGTH: dict[str, str] = {
    "clickfix": DEFINITIONAL,
    "mfa fatigue": DEFINITIONAL,
    "mfa bombing": DEFINITIONAL,
    "push fatigue": DEFINITIONAL,
    "evilproxy": DEFINITIONAL,
    "adversary-in-the-middle": DEFINITIONAL,
    "aitm phishing": DEFINITIONAL,
    "drive-by compromise": DEFINITIONAL,
    "drive-by attack": DEFINITIONAL,
    "watering hole": SUGGESTIVE,
}


def brand_strength(brand: str) -> str:
    """How firmly `brand` implies its technique — see `_BRAND_STRENGTH`.

    Unknown or unlisted brands are `SUGGESTIVE`: a new map entry must opt
    in to definitional treatment deliberately.
    """
    return _BRAND_STRENGTH.get((brand or "").lower(), SUGGESTIVE)


# Pre-compile a single regex matching any brand keyword as a word-boundary
# substring. Word boundaries here are loose — ClickFix should match in
# "CLICKFIX-flavored" too — so we anchor only on either side using non-
# alphanumeric or string-edge.
_BRAND_PATTERN = re.compile(
    r"(?<![a-z0-9])(" + "|".join(re.escape(b) for b in _BRAND_TO_TECHNIQUES) + r")(?![a-z0-9])",
    re.IGNORECASE,
)


def find_brand_techniques(chunk_text: str) -> dict[str, list[str]]:
    """Scan chunk text for known brand mentions and return the mapped T-IDs.

    Returns a dict keyed by the matched brand keyword (lowercase, as it
    appears in the map) with the list of T-IDs as the value. The dict
    shape preserves attribution so callers can log "brand X → T-IDs Y"
    rather than just dumping a flat T-ID list.

    Multiple matches in the same chunk text are deduplicated by brand
    key; one mention per brand per chunk is sufficient.

    Args:
        chunk_text: The chunk's body text.

    Returns:
        {brand_keyword: [tid, ...]} for every recognized brand mention.
        Empty dict when nothing matches or when chunk_text is falsy.
    """
    if not chunk_text:
        return {}
    found: dict[str, list[str]] = {}
    for match in _BRAND_PATTERN.finditer(chunk_text):
        brand = match.group(1).lower()
        if brand in _BRAND_TO_TECHNIQUES:
            found.setdefault(brand, _BRAND_TO_TECHNIQUES[brand])
    return found


def find_brand_techniques_across_chunks(
    chunks: list[dict],
) -> tuple[set[str], list[dict]]:
    """Aggregate brand-expansion across a chunk list.

    Walks every chunk's `text` (and `source_excerpt` as a fallback when
    text is empty), runs `find_brand_techniques`, and returns:
      - The union of T-IDs to add to the candidate pool.
      - An audit log of {chunk_id, brand, techniques} entries for
        observability when callers want to log which chunk triggered
        which expansion.

    Args:
        chunks: list of chunk dicts (must have `text` or `source_excerpt`).

    Returns:
        (tids, audit) where tids is a set of T-IDs and audit is a list
        of dicts.
    """
    tids: set[str] = set()
    audit: list[dict] = []
    for chunk in chunks:
        text = chunk.get("text") or chunk.get("source_excerpt") or ""
        per_chunk = find_brand_techniques(text)
        if not per_chunk:
            continue
        for brand, brand_tids in per_chunk.items():
            tids.update(brand_tids)
            audit.append({
                "chunk_id": chunk.get("chunk_id", ""),
                "brand": brand,
                "techniques": list(brand_tids),
            })
    return tids, audit
