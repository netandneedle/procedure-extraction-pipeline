"""Verify that a quote is actually supported by the source report.

Extracted from `app.nodes.llm.technique_extraction`, where this check was
built to stop a specific failure: `T1204.004` once shipped at bucket
'definite' with confidence 0.95, quoting "copying and pasting an
attacker-supplied command into the Windows Run dialog" — a phrase that
occurs zero times in the report it claimed to cite.

The check lives here rather than in that module because it is not
technique-specific. Any LLM asked to justify a decision with evidence can
invent the evidence, and every such caller needs the same instrument. The
gate reviewer (`app.services.reviewer`) is the second caller.

Token support, not substring, is the load-bearing design choice — see
`quote_support`.
"""

from __future__ import annotations

from collections.abc import Mapping

from app.utils.text import grounding_tokens

# Section classifications held OUT of the grounding corpus.
#
# `technique_reference` is the vendor's own ATT&CK mapping table — the answer
# key. A model allowed to quote it could "prove" any technique by citing the
# report's summary of techniques rather than the narrative that witnesses
# them. Measured effect of including it: the known-fabricated quote's support
# rose from 0.27 to 0.38, real movement toward the pass line from text that
# proves nothing.
#
# `detection_logic` is excluded for the same reason: rule bodies name
# techniques without witnessing them.
NON_GROUNDING_SECTIONS = frozenset({"technique_reference", "detection_logic"})

# Minimum fraction of a quote's distinctive words that must also occur in the
# report before the quote counts as source-grounded.
#
# Derived from measurement, not taste. Across two runs on one vendor report (26
# quotes), support against the report's narrative sections was:
#   0.25, 0.27   <- the two known fabrications
#   0.67 .. 1.00 <- every legitimate paraphrase
# Nothing landed between 0.27 and 0.67, so 0.5 sits mid-gap with ~1.9x margin
# below and ~1.3x above. Re-derive with scripts/measure_pick_stability.py if
# the paraphrasing style of the quoting node changes.
QUOTE_SUPPORT_THRESHOLD = 0.5

# Below this many distinct 4+ char words the ratio is too coarse to mean
# anything (a 2-word quote can only score 0, 0.5 or 1.0).
QUOTE_MIN_TOKENS = 4


def build_source_grounding_tokens(state: Mapping) -> set[str]:
    """Distinctive words of the report, for verifying quotes against it.

    Prefers `classified_sections` so the vendor's ATT&CK table can be held
    out (see `NON_GROUNDING_SECTIONS`). Falls back to raw `parsed_text` when
    the sections aren't in state — still far better than checking a quote
    against LLM-written prose, which is unfalsifiable by construction.

    Returns an empty set when there's no text at all. Callers treat that as
    "check disabled", which fails OPEN deliberately: a source we cannot read
    should not have every quote rejected.
    """
    sections = state.get("classified_sections") or []
    if sections:
        text = "\n".join(
            s.get("text") or ""
            for s in sections
            if s.get("classification") not in NON_GROUNDING_SECTIONS
        )
    else:
        text = state.get("parsed_text") or ""
    if not text.strip():
        return set()
    return grounding_tokens(text)


def quote_support(quote: str, source_tokens: set[str]) -> float | None:
    """Fraction of the quote's distinctive words that occur in the report.

    Token support rather than substring matching, deliberately. Text that
    quotes a report is usually a rewrite of it — chunk prose is a rewrite by
    design — so quotes are rarely verbatim-in-report even when honest. A
    substring test would flag nearly every faithful paraphrase and be
    switched off within a day. A faithful paraphrase reuses the report's
    nouns; an invention does not.

    Returns None when the check does not apply: no source corpus, or a quote
    too short for the ratio to carry information. None means "no opinion" —
    never treat it as a failure.
    """
    if not source_tokens:
        return None
    tokens = grounding_tokens(quote or "")
    if len(tokens) < QUOTE_MIN_TOKENS:
        return None
    return len(tokens & source_tokens) / len(tokens)


def is_quote_grounded(quote: str, source_tokens: set[str]) -> tuple[bool, float | None]:
    """(grounded, support) for a quote. Ungrounded only on a real failure.

    `grounded` is True when the check doesn't apply (support is None), so a
    caller that can't evaluate a quote doesn't punish it. Only a computed
    support score below the threshold returns False.
    """
    support = quote_support(quote, source_tokens)
    if support is None:
        return True, None
    return support >= QUOTE_SUPPORT_THRESHOLD, support
