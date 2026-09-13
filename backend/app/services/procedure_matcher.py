"""Verbatim procedure-example matcher for cross-source consistency.

Builds a reverse index from MITRE ATT&CK procedure examples to operative
substrings (commands, paths, CVEs, URLs, etc.). At extraction time scans
each chunk for these substrings to produce deterministic high-prior
technique candidates BEFORE the LLM call. The LLM then confirms or
rejects each candidate, instead of picking technique IDs from scratch.

Filter settings (the middle of three strictness levels that were tried):
    - Substring min length: 8 characters
    - Substring must contain at least one of: . \\ / - : or a digit
    - Max 3 techniques per substring (more = too ambiguous, dropped)
    - Normalized to lowercase + single-spaced for case/whitespace
      insensitivity

The index is built once per process via get_index() and reused across
all pipeline runs.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# =============================================================================
# Filter constants (middle settings — see module docstring)
# =============================================================================

MIN_LENGTH = 8
ALLOWED_PUNCT = ".\\/-:"
MAX_TECHNIQUES_PER_STRING = 3


# =============================================================================
# Operative-substring extraction patterns
# =============================================================================

_BACKTICK_PATTERN = re.compile(r"`([^`]+)`")
_CODE_BLOCK_PATTERN = re.compile(r"```([^`]+)```")
_CODE_TAG_PATTERN = re.compile(r"<code>([^<]+)</code>", re.IGNORECASE)
_WIN_PATH_PATTERN = re.compile(r"[A-Za-z]:\\[\\\w\.\- ]+")
_UNC_PATH_PATTERN = re.compile(r"\\\\[\\\w\.\-]+")
_UNIX_PATH_PATTERN = re.compile(r"/[a-zA-Z][\w\.\-/]+")
_CVE_PATTERN = re.compile(r"CVE-\d{4}-\d{4,}", re.IGNORECASE)
_URL_PATTERN = re.compile(r"https?://[^\s'\"<>`]+", re.IGNORECASE)
_TOOL_WITH_ARGS_PATTERN = re.compile(
    r"\b\w+\.(?:exe|dll|ps1|sh|py|bat|com|cmd|vbs|js)\s+[\-/][^\s`'\"]+"
    r"(?:\s+[\-/]?[^\s`'\"]+)*",
    re.IGNORECASE,
)

# T-IDs (T1059.001 etc.) intentionally NOT extracted: MITRE's procedure
# descriptions cite other techniques as cross-references, which would
# create high-frequency false matches on any external report that also
# cites ATT&CK. Chunks that explicitly mention T-IDs are already boosted
# in TokenOverlapRetriever.find_candidates (app.services.technique_retriever)
# via the verbatim ID match (+0.5 to overlap score).

_EXTRACTORS = [
    _CODE_BLOCK_PATTERN, _BACKTICK_PATTERN, _CODE_TAG_PATTERN,
    _WIN_PATH_PATTERN, _UNC_PATH_PATTERN, _UNIX_PATH_PATTERN,
    _CVE_PATTERN,
    _URL_PATTERN, _TOOL_WITH_ARGS_PATTERN,
]

# Blacklist: substrings containing any of these tokens are dropped after
# extraction. Captures MITRE's editorial scaffolding (cross-references
# to other ATT&CK pages) which appears in every procedure description
# and would otherwise produce noisy matches against every external CTI
# report that links to MITRE.
#
# Includes both the URL host and the path fragments that survive when
# the host gets stripped by an over-eager regex (Unix-path pattern
# catches "/techniques/t1059" out of a longer MITRE URL).
_SUBSTRING_BLACKLIST = (
    "attack.mitre.org",
    "/techniques/t",   # ATT&CK technique cross-reference paths
    "/software/s",     # ATT&CK software cross-reference paths
    "/groups/g",       # ATT&CK group cross-reference paths
    "/campaigns/c",    # ATT&CK campaign cross-reference paths
    "/tactics/ta",     # ATT&CK tactic cross-reference paths
)

_WS_RE = re.compile(r"\s+")


# =============================================================================
# Helpers
# =============================================================================

def _normalize(s: str) -> str:
    """Lowercase + collapse whitespace runs to single space + strip."""
    return _WS_RE.sub(" ", s).strip().lower()


def _passes_filter(s: str) -> bool:
    """Check normalized substring against length + char-class filter."""
    if len(s) < MIN_LENGTH:
        return False
    if not any(c in ALLOWED_PUNCT or c.isdigit() for c in s):
        return False
    if any(token in s for token in _SUBSTRING_BLACKLIST):
        return False
    return True


# =============================================================================
# Public API
# =============================================================================

def extract_operative_strings(description: str) -> list[str]:
    """Extract normalized operative substrings from a procedure-example
    description. Returns sorted unique normalized strings that pass the
    filter rules."""
    if not description:
        return []
    found: set[str] = set()
    for pattern in _EXTRACTORS:
        for raw_match in pattern.findall(description):
            # findall returns strings (single capture group) or tuples
            # (multi-group). Take the longest tuple element if tuple.
            if isinstance(raw_match, tuple):
                raw_match = max(raw_match, key=len) if raw_match else ""
            normalized = _normalize(raw_match)
            if _passes_filter(normalized):
                found.add(normalized)
    return sorted(found)


def build_index(attack_data: Any) -> dict[str, list[dict[str, str]]]:
    """Build the verbatim-match index from MITRE procedure examples.

    Returns:
        {normalized_substring: [{technique_id, source_actor_name,
        source_actor_type}, ...]}

    Substrings mapping to MORE than MAX_TECHNIQUES_PER_STRING distinct
    techniques are dropped (too ambiguous to be a useful fingerprint).
    Within a substring's list, entries are deduped by
    (technique_id, source_actor_name, source_actor_type).
    """
    # Use a set during construction to dedupe entries within each substring
    raw_index: dict[str, set[tuple[str, str, str]]] = {}

    for example in attack_data.get_procedure_examples():
        substrings = extract_operative_strings(example["description"])
        if not substrings:
            continue
        entry = (
            example["technique_id"],
            example["source_actor_name"],
            example["source_actor_type"],
        )
        for substr in substrings:
            raw_index.setdefault(substr, set()).add(entry)

    # Apply max-techniques cap
    filtered: dict[str, list[dict[str, str]]] = {}
    dropped_ambiguous = 0
    for substr, entries_set in raw_index.items():
        unique_techniques = {e[0] for e in entries_set}
        if len(unique_techniques) > MAX_TECHNIQUES_PER_STRING:
            dropped_ambiguous += 1
            continue
        filtered[substr] = [
            {"technique_id": tid, "source_actor_name": name, "source_actor_type": atype}
            for (tid, name, atype) in sorted(entries_set)
        ]

    logger.info(
        "procedure_matcher: built index with %d unique substrings "
        "(dropped %d ambiguous strings mapping to >%d techniques)",
        len(filtered), dropped_ambiguous, MAX_TECHNIQUES_PER_STRING,
    )
    return filtered


def match_chunk(
    chunk_text: str,
    index: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    """Scan a chunk for substrings present in the index.

    Returns deduped match dicts:
        {technique_id, matched_substring, source_actor_name, source_actor_type}

    Dedup is by (technique_id, matched_substring) so a chunk that
    contains the same substring twice against the same technique gets
    one entry, but the SAME substring matching multiple distinct
    techniques (e.g., a tool used in different ways) returns one entry
    per technique.
    """
    if not chunk_text:
        return []
    normalized = _normalize(chunk_text)
    seen: set[tuple[str, str]] = set()
    matches: list[dict[str, Any]] = []
    for substr, entries in index.items():
        if substr in normalized:
            for entry in entries:
                key = (entry["technique_id"], substr)
                if key in seen:
                    continue
                seen.add(key)
                matches.append({
                    "technique_id": entry["technique_id"],
                    "matched_substring": substr,
                    "source_actor_name": entry["source_actor_name"],
                    "source_actor_type": entry["source_actor_type"],
                })
    return matches


# =============================================================================
# Module-level singleton
# =============================================================================

_index_cache: dict[str, list[dict[str, str]]] | None = None


def get_index() -> dict[str, list[dict[str, str]]]:
    """Return the process-wide verbatim-match index. Lazy-built on first call.

    Reuses the AttackData singleton from app.services.attack_data, so
    the STIX bundle is parsed exactly once across the LLM catalogue +
    the matcher index.
    """
    global _index_cache
    if _index_cache is None:
        from app.services.attack_data import get_attack_data
        _index_cache = build_index(get_attack_data())
    return _index_cache
