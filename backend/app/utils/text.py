"""Text utility functions shared across the LLM pipeline.

Tokenization + similarity + grounding helpers used by both the
candidate retriever (technique_retriever.TokenOverlapRetriever) and
the post-LLM confidence recalibration in technique_extraction. Lives
in its own module so neither caller depends on the other (avoids
circular imports).
"""

from __future__ import annotations

import json
import re
from typing import Any, NamedTuple

_WORD_PATTERN = re.compile(r"[a-z]{3,}")
_GROUNDING_WORD_PATTERN = re.compile(r"[a-z]{4,}")

# Canonical ATT&CK technique-ID shape: T#### optionally with a .### sub-id.
# Single source of truth — imported by technique_extraction (LLM-output
# validation) and feedback_patterns / the denylist schema (term validation),
# which uppercase before matching, so no IGNORECASE flag is needed here.
TECHNIQUE_ID_RE = re.compile(r"^T\d{4}(\.\d{3})?$")

# The same shape, unanchored, for pulling T-IDs OUT of prose — e.g. the
# vendor's own ATT&CK mapping table in a `technique_reference` section.
# Kept beside TECHNIQUE_ID_RE so the two shapes can't drift; use that one to
# validate a whole string, this one to scan.
TECHNIQUE_ID_SCAN_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")


def join_technique_ids(techniques: list | None) -> str:
    """Comma-join technique_id values from a list of {technique_id, ...} dicts.

    Shared by the Gate 1 correction-log consumers (synthesis digest, rerun
    prompt, captured-corrections panel). Defensive against non-dict entries
    and missing ids — these lists come from LangGraph checkpoints and seed
    scripts, not only Pydantic-validated payloads.
    """
    return ", ".join(
        t["technique_id"]
        for t in (techniques or [])
        if isinstance(t, dict) and t.get("technique_id")
    )


def tokenize(text: str) -> set[str]:
    """Lowercase split into alpha tokens, min 3 chars. Cheap and deterministic.

    Used as the basis for token_overlap scoring in candidate retrieval.
    """
    return {w for w in _WORD_PATTERN.findall(text.lower())}


def token_overlap(text_a: str, text_b: str) -> float:
    """Jaccard token overlap between two strings. Returns 0.0-1.0.

    Switched from min-denominator to union-denominator (Jaccard) to reduce
    score volatility at the top-K cutoff: short technique descriptions
    matching long chunks no longer get disproportionate boosts, so
    techniques near the K-th-place boundary stop flipping in/out based
    on text-length quirks.
    """
    tokens_a = tokenize(text_a)
    tokens_b = tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def grounding_tokens(text: str) -> set[str]:
    """4+ char alpha tokens used to check if a rationale references chunk
    content.

    Wider min length than tokenize() so most English glue words (the,
    and, are, was, has, had, can, may, but) are filtered without an
    explicit stopword list. Used by the post-LLM confidence recalibration
    Rule 4 (rationale grounding cap) in technique_extraction.
    """
    return {w for w in _GROUNDING_WORD_PATTERN.findall(text.lower())}


class LenientParse(NamedTuple):
    """Result of ``loads_json_lenient``: the value plus what it took to get it.

    ``repaired`` is True when the escape repair was needed; ``discarded`` is the
    number of trailing characters dropped by the prefix salvage (0 when the
    whole string parsed), with ``tail`` holding the first few of them so the
    caller can log what was thrown away.
    """

    value: Any
    repaired: bool = False
    discarded: int = 0
    tail: str = ""


def loads_json_lenient(s: str) -> LenientParse:
    """``json.loads`` with the two repairs model output has actually needed.

    Three rungs, tried in order, each one more forgiving than the last:

      1. plain ``json.loads`` — the common case, and the only one for valid
         input (the escape repair is the identity on valid JSON, so applying
         it unconditionally would change nothing but cost an O(n) pass).
      2. ``repair_invalid_json_escapes`` then ``json.loads`` — the nested
         single-escape bug: a Windows path ``C:\\Windows`` inside a stringified
         value arrives as ``\\W``, which is not a legal escape.
      3. ``raw_decode`` prefix salvage — take the first complete JSON value
         and report how much trailing junk followed it. This is the shape that
         killed one run: a well-formed 16-element array followed by
         one stray ``}``. Only the prefix is trusted; the caller logs the
         discard size, because a large tail means the prefix may be wrong too.

    Both the adapter (for stringified collection fields) and any provider whose
    tool arguments arrive as a JSON string go through this ONE ladder, so a
    fix for the next kind of junk lands in both places at once.

    Raises ValueError when no rung produces a value.
    """
    try:
        return LenientParse(json.loads(s))
    except ValueError as exc:
        first_error = exc  # `exc` itself is unbound once this block exits
    try:
        return LenientParse(json.loads(repair_invalid_json_escapes(s)), repaired=True)
    except ValueError:
        pass
    stripped = s.lstrip()
    try:
        value, end = json.JSONDecoder().raw_decode(stripped)
    except ValueError:
        raise ValueError(str(first_error)) from first_error
    discarded = len(stripped) - end
    if discarded <= 0:  # pragma: no cover — raw_decode succeeding here implies a tail
        raise ValueError(str(first_error)) from first_error
    return LenientParse(value, discarded=discarded, tail=stripped[end:end + 40])


def repair_invalid_json_escapes(s: str) -> str:
    """Repair odd-length backslash runs that precede invalid JSON escape chars.

    JSON consumes backslashes in pairs. An odd run leaves a trailing ``\\``
    starting a new escape; if the next char isn't one of ``" \\ / b f n r t u``,
    json.loads raises 'Invalid \\escape'. This is the nested-string
    double-escape bug (seen on Claude; structurally possible on any
    model that emits a JSON string inside a JSON string): a Windows path ``C:\\Windows`` ends up as ``\\W`` in the
    inner string. Inserting one more ``\\`` makes it ``\\\\W`` which parses to
    literal ``\\W``.

    Only touches odd runs followed by an INVALID escape char — even runs and
    odd runs followed by a legal escape are left alone so we don't corrupt
    legitimate sequences like ``\\n`` or ``\\"``.
    """
    valid_escape_chars = set('"\\/bfnrtu')
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] != "\\":
            out.append(s[i])
            i += 1
            continue
        j = i
        while j < n and s[j] == "\\":
            j += 1
        run_len = j - i
        if run_len % 2 == 1 and j < n and s[j] not in valid_escape_chars:
            out.append("\\" * (run_len + 1))
        else:
            out.append("\\" * run_len)
        i = j
    return "".join(out)
