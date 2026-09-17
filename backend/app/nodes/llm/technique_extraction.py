"""extract_techniques node: Stage 3 of the extraction pipeline.

LLM node. Maps behavioral chunks to MITRE ATT&CK techniques.

PROCESS:
    1. Load full ATT&CK technique catalogue via app.services.attack_data
       (T-numbers, names, descriptions, tactics, STIX UUIDs, platforms)
    2. Filter out revoked/deprecated techniques
    3. Inject technique reference (id | name | tactics) into LLM prompt;
       run verbatim procedure-example matcher to surface deterministic
       candidates per chunk
    4. LLM assigns technique(s) per chunk and confirms/rejects each
       verbatim match
    5. Post-extraction: resolve T-numbers to real STIX UUIDs via lookup
    6. Per-technique confidence scoring against ATT&CK descriptions

DESIGN DECISIONS:
    - app.services.attack_data parses the STIX 2.1 bundle directly via
      stdlib json (no third-party STIX library). Catalogue version is
      determined by which STIX file is configured (defaults to v19.2).
      Includes descriptions, revoked/deprecated status, platforms, and
      real STIX UUIDs.
    - Descriptions are NOT sent to the LLM (Path A token reduction): the
      prompt-bound reference is `id | name | tactics` only, which fits
      free-tier rate limits. The LLM uses its ATT&CK training knowledge
      to map behavior; verbatim matches anchor cited operative strings,
      and post-extraction resolution catches hallucinated IDs.
      Descriptions remain in technique_lookup for local scoring
      (overlap modulation, sub-technique promotion).
    - Revoked (132) and deprecated (12) techniques are excluded from the
      reference list, preventing the LLM from assigning dead techniques.
    - One LLM call processes ALL chunks together (batch) for consistency.
    - Post-extraction resolution populates stix_id on each mapping with
      the real UUID. Downstream nodes use real UUIDs directly.
    - Neo4j is still used for log source enrichment (DataComponents) in
      the serialization stage, but not for technique grounding.

READS: chunks, validated_entities
WRITES: technique_mappings, status, current_node
"""

from __future__ import annotations

import logging

from app.graph.state import (
    EntityType,
    GateAction,
    PipelineState,
    PipelineStatus,
)
from app.nodes.llm._definitions import PROCEDURE_DEFINITION
from app.nodes.llm.llm_adapter import call_llm
from app.nodes.llm.tool_models import (
    ExtractTechniquesOutput,
    ProposeTechniquesOutput,
)
from app.services.feedback_examples import relevant_examples_cached
from app.services.feedback_patterns import (
    denylist_match_technique,
    load_denylist,
    relevant_addendum_cached,
)
from app.services.grounding import (
    NON_GROUNDING_SECTIONS,
    QUOTE_MIN_TOKENS,
    QUOTE_SUPPORT_THRESHOLD,
    build_source_grounding_tokens,
)
from app.services.procedure_matcher import get_index, match_chunk
from app.services.technique_pattern_brands import (
    DEFINITIONAL,
    brand_strength,
    find_brand_techniques_across_chunks,
)
from app.services.technique_retriever import get_retriever
from app.services.vendor_technique_mapping import extract_vendor_technique_ids
from app.utils.text import (
    TECHNIQUE_ID_RE,
    grounding_tokens,
    join_technique_ids,
    token_overlap,
)

logger = logging.getLogger(__name__)


# Feedback-pattern categories relevant to technique-mapping judgments.
# Read at prompt-build time and prepended as an analyst-feedback addendum.
_TECHNIQUE_FEEDBACK_CATEGORIES = (
    "missing_tactic",
    "wrong_technique",
    "thin_initial_access",
    "orphan_ioc",
    # Catch-all, and the fallback when the synthesizer omits a category.
    # Every node reads it — see the note in entity_extraction.
    # Pinned by tests/test_contracts.py.
    "other",
)


async def _fetch_feedback_addendum(
    state, *, node: str = "extract_techniques",
) -> str:
    """Analyst feedback patterns RELEVANT TO THIS SOURCE, as a prompt addendum.
    Source-relative hybrid retrieval, TTL-cached — see relevant_addendum_cached.
    Best-effort: returns "" on any failure.

    This node fetches TWICE, and ``node`` distinguishes the two entries in the
    surfacing ledger. The reason is in the pick-step call site: retrieval's
    largest scoring term is +0.5 for a shared technique ID, and no technique ID
    exists yet when the propose step asks.
    """
    return await relevant_addendum_cached(
        state, categories=_TECHNIQUE_FEEDBACK_CATEGORIES,
        node=node, limit=15,
    )


async def _fetch_feedback_examples(state, *, node: str = "extract_techniques") -> str:
    """Past analyst corrections most similar to this source, as demonstrations.

    A separate channel from `_fetch_feedback_addendum` on purpose: the rules
    are LLM-written generalizations and the examples are records, they fail in
    different ways, and keeping the fetches apart is what lets an ablation arm
    vary one without the other. Best-effort: "" on any failure.
    """
    return await relevant_examples_cached(
        state, areas=("techniques",), node=node,
    )


# =============================================================================
# ATT&CK technique catalogue (via app.services.attack_data, direct JSON parse)
# =============================================================================

# Module-level cache: loaded once per process, reused across pipeline runs.
_technique_lookup: dict[str, dict] | None = None
_reference_text: str | None = None


def _load_technique_catalogue() -> tuple[dict[str, dict], str]:
    """Load ATT&CK technique catalogue from app.services.attack_data.

    Filters out revoked and deprecated techniques. Caches result at
    module level so subsequent pipeline runs don't re-parse the JSON.

    Returns:
        (technique_lookup, reference_text) where:
        - technique_lookup: {T-number: {name, stix_id, description,
          tactics, platforms}} for resolution and grounding
        - reference_text: formatted string for injection into LLM prompt
    """
    global _technique_lookup, _reference_text

    if _technique_lookup is not None and _reference_text is not None:
        logger.info(
            "extract_techniques: using cached catalogue (%d techniques)",
            len(_technique_lookup),
        )
        return _technique_lookup, _reference_text

    try:
        from app.services.attack_data import get_attack_data
        db = get_attack_data()
        all_techniques = db.all_techniques()
    except Exception as e:
        logger.warning(
            "extract_techniques: attack_data load failed (%s: %s), "
            "proceeding without grounding",
            type(e).__name__, e,
        )
        return {}, ""

    if not all_techniques:
        logger.warning("extract_techniques: catalogue returned 0 techniques")
        return {}, ""

    technique_lookup: dict[str, dict] = {}
    lines: list[str] = []
    skipped_revoked = 0
    skipped_deprecated = 0

    for tech in all_techniques:
        # Skip revoked and deprecated techniques
        if tech.get("revoked"):
            skipped_revoked += 1
            continue
        if tech.get("deprecated"):
            skipped_deprecated += 1
            continue

        tid = tech.get("external_id", "")
        if not tid:
            continue

        name = tech.get("name", "")
        stix_id = tech.get("stix_id", "")
        description = tech.get("description", "")
        tactics = tech.get("tactics", [])
        platforms = tech.get("platforms", [])

        technique_lookup[tid] = {
            "name": name,
            "stix_id": stix_id,
            "description": description,
            "tactics": tactics,
            "platforms": platforms,
        }

        tactic_str = ", ".join(tactics) if tactics else "n/a"
        # Descriptions are NOT included in the prompt (would push the
        # full catalogue past per-minute rate limits on free tiers).
        # LLM uses its training knowledge of each technique; the verbatim
        # matcher provides deterministic anchors for cited operative
        # strings, and post-extraction _resolve_stix_ids catches any
        # hallucinated IDs. Descriptions stay in technique_lookup for
        # local scoring (overlap modulation, sub-technique promotion).
        lines.append(f"{tid} | {name} | {tactic_str}")

    logger.info(
        "extract_techniques: loaded %d active techniques "
        "(skipped %d revoked, %d deprecated)",
        len(technique_lookup), skipped_revoked, skipped_deprecated,
    )

    reference_text = (
        "\n\nATT&CK TECHNIQUE REFERENCE (revoked/deprecated excluded):\n"
        "Use ONLY technique IDs from this list. Use your ATT&CK knowledge "
        "of what each technique represents to map chunk behavior. If a "
        "behavior doesn't match any listed technique, use the closest "
        "parent technique with lower confidence and note the gap in "
        "rationale.\n"
        "Format: technique_id | name | tactics\n"
        + "\n".join(lines)
    )

    # Cache for reuse
    _technique_lookup = technique_lookup
    _reference_text = reference_text

    return technique_lookup, reference_text


def _resolve_stix_ids(
    technique_mappings: dict[str, list[dict]],
    technique_lookup: dict[str, dict],
) -> tuple[dict[str, list[dict]], list[str]]:
    """Resolve T-numbers to real STIX UUIDs using ATT&CK catalogue.

    Populates the stix_id field on each TechniqueMapping. Corrects
    technique names to match canonical ATT&CK names. Resolution order:

        1. Active lookup — direct hit in the (filtered) catalogue.
        2. Revoked-with-redirect — the LLM picked a stale v18 T-number from
           training knowledge; follow the STIX 'revoked-by' relationship
           to the v19 replacement and rewrite the mapping in place. Original
           pick recorded on 'redirected_from' for audit.
        3. Deprecated — no replacement available, but keep the mapping with
           is_deprecated=True so the analyst can decide at gate review
           rather than silently dropping a behavior the LLM identified.
        4. Truly unresolved — log a warning, leave stix_id unset, downstream
           STIX serialization will skip the entry.

    Returns:
        (updated_mappings, warnings) where warnings lists unresolved T-numbers.
    """
    # Lazy import: attack_data wraps a ~50 MB JSON parse. Keeping the import
    # here matches the existing _load_technique_catalogue lazy pattern (try/
    # except wrapping), so a missing STIX bundle doesn't kill module import.
    try:
        from app.services.attack_data import get_attack_data
        db = get_attack_data()
    except Exception as e:
        logger.warning(
            "extract_techniques: attack_data unavailable for redirect lookup "
            "(%s: %s); skipping revoked/deprecated handling",
            type(e).__name__, e,
        )
        db = None

    warnings: list[str] = []
    redirects: list[str] = []
    deprecated_kept: list[str] = []

    for chunk_id, techniques in technique_mappings.items():
        for t in techniques:
            tid = t.get("technique_id", "")

            # Path 1: active technique, direct lookup
            lookup = technique_lookup.get(tid)
            if lookup:
                t["stix_id"] = lookup["stix_id"]
                if lookup["name"] and t.get("technique_name") != lookup["name"]:
                    t["technique_name"] = lookup["name"]
                continue

            # Paths 2+3 require attack_data; skip if unavailable
            if db is None:
                warnings.append(
                    f"chunk {chunk_id}: technique {tid} not found in catalogue "
                    f"(hallucinated or revoked/deprecated; attack_data unavailable)"
                )
                continue

            # Path 2: revoked technique with valid redirect target
            redirect_target = db.revoked_by_target(tid)
            if redirect_target:
                target_lookup = technique_lookup.get(redirect_target)
                if target_lookup:
                    t["redirected_from"] = tid
                    t["technique_id"] = redirect_target
                    t["technique_name"] = target_lookup["name"]
                    t["stix_id"] = target_lookup["stix_id"]
                    redirects.append(f"chunk {chunk_id}: {tid} -> {redirect_target}")
                    continue
                # Redirect target itself isn't active (revoked-to-revoked
                # chain or revoked-to-deprecated). Single-hop policy: don't
                # chase further; fall through to deprecated/warning paths.

            # Path 3: deprecated but not revoked (or revoked target is
            # deprecated and we landed here). Keep with the flag.
            rec = db.get_technique_record(tid)
            if rec and rec.get("deprecated"):
                t["technique_name"] = rec.get("name", t.get("technique_name", ""))
                t["stix_id"] = rec.get("stix_id", "")
                t["is_deprecated"] = True
                deprecated_kept.append(f"chunk {chunk_id}: {tid}")
                continue

            # Path 4: truly unresolved (hallucinated or stale ID we can't
            # redirect)
            warnings.append(
                f"chunk {chunk_id}: technique {tid} not found in catalogue "
                f"(hallucinated or revoked without redirect target)"
            )

    if redirects:
        logger.info(
            "extract_techniques: redirected %d revoked picks via revoked-by: %s",
            len(redirects), "; ".join(redirects),
        )
    if deprecated_kept:
        logger.warning(
            "extract_techniques: kept %d deprecated picks with is_deprecated flag "
            "(analyst review at gate): %s",
            len(deprecated_kept), "; ".join(deprecated_kept),
        )
    if warnings:
        logger.warning(
            "extract_techniques: %d unresolved technique IDs: %s",
            len(warnings), "; ".join(warnings),
        )

    return technique_mappings, warnings


# =============================================================================
# Deterministic pre-filter
# =============================================================================

# Token-overlap pre-filter logic lives in app.services.technique_retriever
# (TokenOverlapRetriever). Kept out of this module so retriever variants
# (token_overlap, embedding) can be swapped without touching the LLM node,
# and so neither module needs a lazy import to break a circular dep.


# =============================================================================
# Tactic auto-correction
# =============================================================================

def _auto_correct_tactics(
    technique_mappings: dict[str, list[dict]],
    technique_lookup: dict[str, dict],
) -> dict[str, list[dict]]:
    """Deterministic tactic correction for single-tactic techniques.

    If a technique only belongs to one tactic in the ATT&CK catalogue,
    override whatever the LLM assigned with the canonical tactic.
    For multi-tactic techniques, leave the LLM's assignment (it chose
    based on context).

    Args:
        technique_mappings: {chunk_id: [TechniqueMapping, ...]}
        technique_lookup: full catalogue dict

    Returns:
        Updated technique_mappings with corrected tactics.
    """
    corrections = 0

    for chunk_id, techniques in technique_mappings.items():
        for t in techniques:
            tid = t.get("technique_id", "")
            entry = technique_lookup.get(tid)
            if not entry:
                continue

            tactics = entry.get("tactics", [])
            if len(tactics) == 1:
                canonical_tactic = tactics[0]
                llm_tactic = t.get("tactic", "")
                if llm_tactic != canonical_tactic:
                    logger.info(
                        "tactic_correct: chunk %s, %s: %s -> %s (single-tactic technique)",
                        chunk_id, tid, llm_tactic, canonical_tactic,
                    )
                    t["tactic"] = canonical_tactic
                    corrections += 1

    if corrections:
        logger.info("tactic_correct: corrected %d tactic assignments", corrections)

    return technique_mappings


# =============================================================================
# Confidence recalibration
# =============================================================================

# Token utilities (token_overlap, grounding_tokens) live in app.utils.text
# so both this module and technique_retriever can use them without circular
# imports.


def _recalibrate_confidence(
    technique_mappings: dict[str, list[dict]],
    technique_lookup: dict[str, dict],
    chunks_by_id: dict[str, str],
) -> dict[str, list[dict]]:
    """Deterministic post-extraction confidence recalibration.

    Adjusts LLM confidence scores using observable signals and promotes
    parent techniques to more specific sub-techniques when warranted.

    Rules applied (in order):
        1. Sub-technique auto-promotion: if LLM assigned a parent (e.g. T1059)
           but a sub-technique's description has better overlap with the chunk,
           swap in the sub-technique.
        2. Verbatim anchor: if technique name or T-number appears in chunk text,
           floor confidence at 0.85.
        3. Keyword overlap modulation: token overlap between chunk and technique
           description adjusts confidence toward the overlap signal.
        4. Rationale quality cap: weak/missing rationale caps at 0.5.

    Args:
        technique_mappings: {chunk_id: [TechniqueMapping, ...]}
        technique_lookup: catalogue dict from _load_technique_catalogue()
        chunks_by_id: {chunk_id: chunk_text}

    Returns:
        Updated technique_mappings with adjusted confidence scores.
    """
    # Pre-build parent -> sub-technique index for promotion checks
    subtechnique_index: dict[str, list[str]] = {}
    for tid in technique_lookup:
        if "." in tid:
            parent = tid.split(".")[0]
            subtechnique_index.setdefault(parent, []).append(tid)

    adjustments = 0

    for chunk_id, techniques in technique_mappings.items():
        chunk_text = chunks_by_id.get(chunk_id, "")
        chunk_text_lower = chunk_text.lower()

        for t in techniques:
            # Verbatim matches lock confidence at extraction time (0.95
            # confirmed, 0.3 rejected); recalibration rules don't apply.
            if t.get("provenance", "").startswith("verbatim_match"):
                continue
            tid = t.get("technique_id", "")
            raw_confidence = t.get("confidence", 0.5)
            adjusted = raw_confidence
            reasons: list[str] = []

            cat_entry = technique_lookup.get(tid, {})
            tech_description = cat_entry.get("description", "")

            # -----------------------------------------------------------------
            # Rule 1: Sub-technique auto-promotion
            # Only applies to parent techniques (no dot in ID).
            #
            # Two thresholds for promotion (BOTH must hold):
            #   - Relative gap: sub_overlap > parent_overlap + 0.05.
            #     The sub must beat the parent by a meaningful margin.
            #   - Absolute floor: sub_overlap >= 0.20.
            #     The sub's overlap with the chunk must be non-trivial in
            #     ABSOLUTE terms. Without this floor, the rule misfired
            #     during the C+A+D synthetic dry run: certutil download
            #     chunk had T1218 (parent) overlap=0.05 and T1218.008
            #     (Odbcconf, wrong sub) overlap=0.10. The relative gap was
            #     exactly 0.05 → promotion fired → wrong sub. The 0.20
            #     floor ensures we only promote when there's real evidence
            #     in description-vocabulary overlap, not when both are
            #     near-zero noise.
            #
            # Notes:
            # - C+A+D's propose step already prefers sub-techniques, so the
            #   LLM increasingly picks the right sub directly. This rule is
            #   now a backstop for cases where the LLM still picks a parent
            #   despite the prompt guidance — keeping it tight reduces
            #   misfires without giving up real wins.
            # - Token overlap thresholds vary across data; if future tuning
            #   surfaces a regression, consider also requiring the parent to
            #   appear ALONE in the candidate pool (no manually-picked subs).
            # -----------------------------------------------------------------
            _SUB_PROMOTE_GAP = 0.05
            _SUB_PROMOTE_ABS_FLOOR = 0.20
            if "." not in tid and tid in subtechnique_index:
                parent_overlap = token_overlap(chunk_text, tech_description)
                best_sub = None
                best_sub_overlap = parent_overlap

                for sub_tid in subtechnique_index[tid]:
                    sub_entry = technique_lookup.get(sub_tid, {})
                    sub_desc = sub_entry.get("description", "")
                    sub_overlap = token_overlap(chunk_text, sub_desc)

                    if (
                        sub_overlap > best_sub_overlap + _SUB_PROMOTE_GAP
                        and sub_overlap >= _SUB_PROMOTE_ABS_FLOOR
                    ):
                        best_sub = sub_tid
                        best_sub_overlap = sub_overlap

                if best_sub:
                    sub_entry = technique_lookup[best_sub]
                    old_tid = tid
                    t["technique_id"] = best_sub
                    t["technique_name"] = sub_entry["name"]
                    t["stix_id"] = sub_entry["stix_id"]
                    # Update local refs for subsequent rules
                    tid = best_sub
                    cat_entry = sub_entry
                    tech_description = sub_entry.get("description", "")
                    reasons.append(
                        f"promoted {old_tid} -> {best_sub} "
                        f"(overlap {parent_overlap:.2f} -> {best_sub_overlap:.2f})"
                    )

            # -----------------------------------------------------------------
            # Rule 2: Verbatim anchor
            # Technique name or ID appears literally in chunk text
            # -----------------------------------------------------------------
            tech_name = cat_entry.get("name", "")
            name_in_text = tech_name and tech_name.lower() in chunk_text_lower
            id_in_text = tid.lower() in chunk_text_lower

            if (name_in_text or id_in_text) and adjusted < 0.85:
                adjusted = 0.85
                anchor = "name" if name_in_text else "ID"
                reasons.append(f"verbatim {anchor} anchor -> floor 0.85")

            # -----------------------------------------------------------------
            # Bucket-aware skip: when the LLM committed to 'definite' or
            # 'probable' AND the source_quote is verbatim in the chunk,
            # the pick has already passed two evidence checks (calibrated
            # bucket + literal substring). Rules 3 + 4 below would
            # double-jeopardy these by dragging the numeric down whenever
            # the chunk's vocabulary differs from MITRE's technique
            # description — a near-universal vocabulary divergence in real
            # CTI. So the post-C+A+D contract is: trust the bucket on
            # verbatim-grounded picks, skip overlap and grounding caps.
            #
            # 'possible' picks still flow through Rules 3 + 4 — they're
            # weak by design and the deterministic recalibration is the
            # appropriate guardrail there.
            #
            # `_apply_source_quote_cap` already downgrades any pick whose
            # source_quote isn't verbatim in the chunk to 'possible', so
            # this branch implicitly requires verbatim grounding.
            bucket = t.get("confidence_bucket", "probable")
            skip_modulation = bucket in ("definite", "probable")

            # -----------------------------------------------------------------
            # Rule 3: Keyword overlap modulation
            # Blend LLM confidence with description overlap signal
            # -----------------------------------------------------------------
            if tech_description and not skip_modulation:
                overlap = token_overlap(chunk_text, tech_description)
                # Weighted blend: 70% LLM score, 30% overlap signal
                # This nudges without overriding the LLM entirely
                blended = (adjusted * 0.7) + (overlap * 0.3)
                if abs(blended - adjusted) > 0.01:
                    reasons.append(
                        f"overlap modulation {adjusted:.2f} -> {blended:.2f} "
                        f"(overlap={overlap:.2f})"
                    )
                    adjusted = blended

            # -----------------------------------------------------------------
            # Rule 4: Rationale grounding cap
            # Replaces a prior length-based cap that punished terse-but-
            # grounded rationales ("certutil") and let verbose-but-empty
            # ones pass ("strongly aligned with this technique"). New rule:
            # rationale must share at least one 4+ char alpha token with
            # the chunk text. Length is no longer a signal.
            # -----------------------------------------------------------------
            rationale = (t.get("rationale", "") or "").strip()
            if adjusted > 0.5 and not skip_modulation:
                if not (grounding_tokens(rationale) & grounding_tokens(chunk_text)):
                    reasons.append("rationale not grounded in chunk text -> cap 0.50")
                    adjusted = 0.5

            # Apply final adjusted score (clamp 0-1)
            adjusted = max(0.0, min(1.0, round(adjusted, 2)))

            if adjusted != raw_confidence or reasons:
                t["confidence"] = adjusted
                t["confidence_raw"] = raw_confidence
                adjustments += 1
                if reasons:
                    logger.info(
                        "recalibrate: chunk %s, %s: %s",
                        chunk_id, tid, "; ".join(reasons),
                    )

    logger.info(
        "recalibrate: adjusted %d/%d technique assignments",
        adjustments,
        sum(len(v) for v in technique_mappings.values()),
    )

    return technique_mappings


# =============================================================================
# Procedure definition — shared anchor (see app.nodes.llm._definitions)
# =============================================================================

# Both LLM calls in this module (propose + pick) anchor on the same procedure
# definition the chunker uses. Sourced from a shared module so any revision
# propagates everywhere consistently.
_PROCEDURE_DEFINITION = PROCEDURE_DEFINITION + (
    "\nEach behavioral chunk you process represents one procedure. Your job "
    "is to identify the techniques that integrate to fulfill the chunk's "
    "objective — not techniques associated with its surrounding context, the "
    "campaign, or the broader engagement.\n"
)


# =============================================================================
# Tool definitions
# =============================================================================

PROPOSE_TECHNIQUES_TOOL = {
    "name": "propose_techniques",
    "description": (
        "Describe each chunk's adversary action in ATT&CK terms and propose "
        "candidate technique IDs from your training knowledge. This is the "
        "FIRST step of a two-step flow; you will pick from a validated, "
        "unioned candidate pool in the second step."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "chunk_proposals": {
                "type": "array",
                "description": "One entry per input chunk.",
                "items": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {
                            "type": "string",
                            "description": "The chunk_id being analyzed.",
                        },
                        "behavior_description": {
                            "type": "string",
                            "description": (
                                "1-2 sentences in ATT&CK-flavored prose describing "
                                "what the adversary did in this chunk. Reference the "
                                "tactic phase, the technique class, and the artifacts "
                                "(commands / binaries / observables). Use plain "
                                "English; do NOT use technique IDs in this field."
                            ),
                        },
                        "objective": {
                            "type": "string",
                            "description": (
                                "The procedure's specific adversarial objective in "
                                "ONE sentence. This is the north-star that scopes "
                                "the procedure: techniques BELONG if they serve this "
                                "objective; techniques DRIFT if they serve a different "
                                "one. State it as the goal, not the action. "
                                "Examples: 'Execute attacker code on the victim host "
                                "by tricking the user into pasting and running a "
                                "payload via a fake CAPTCHA lure.' "
                                "'Establish per-user persistence on the host by "
                                "registering the loader to auto-execute at logon.' "
                                "'Download the second-stage loader to disk using a "
                                "signed Microsoft binary to evade unsigned-binary "
                                "heuristics.'"
                            ),
                        },
                        "tactics": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "ALL ATT&CK tactics this procedure spans. A single "
                                "procedure routinely spans multiple tactics (e.g., a "
                                "LOLBin download spans command-and-control + "
                                "defense-evasion). Pick from: reconnaissance, "
                                "resource-development, initial-access, execution, "
                                "persistence, privilege-escalation, defense-evasion, "
                                "credential-access, discovery, lateral-movement, "
                                "collection, command-and-control, exfiltration, "
                                "impact."
                            ),
                        },
                        "proposed_techniques": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "ATT&CK technique IDs (T#### or T####.###) that "
                                "INTEGRATE to fulfill the objective above, ordered "
                                "most-specific-and-confident first. Prefer sub-"
                                "techniques over parents. Multi-tactic procedures "
                                "typically need 2-5 techniques to compose the "
                                "implementation; single-tactic ones may need only 1. "
                                "Don't worry about whether your IDs match the "
                                "deployed catalogue version — invalid ones will be "
                                "filtered downstream."
                            ),
                        },
                    },
                    "required": [
                        "chunk_id", "behavior_description",
                        "objective", "tactics", "proposed_techniques",
                    ],
                },
            },
        },
        "required": ["chunk_proposals"],
    },
}


EXTRACT_TECHNIQUES_TOOL = {
    "name": "extract_techniques",
    "description": (
        "Map each behavioral chunk to one or more MITRE ATT&CK techniques. "
        "Use real ATT&CK technique IDs (e.g., T1059.001, T1190). "
        "Prefer sub-techniques over parent techniques when specific."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "chunk_techniques": {
                "type": "array",
                "description": "One entry per chunk, mapping it to ATT&CK techniques.",
                "items": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {
                            "type": "string",
                            "description": "The chunk_id being mapped.",
                        },
                        "techniques": {
                            "type": "array",
                            "description": "ATT&CK techniques this chunk maps to.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "technique_id": {
                                        "type": "string",
                                        "description": (
                                            "ATT&CK technique ID. Format: T####  or "
                                            "T####.### for sub-techniques. Must be a "
                                            "real ATT&CK ID."
                                        ),
                                    },
                                    "technique_name": {
                                        "type": "string",
                                        "description": "Human-readable technique name.",
                                    },
                                    "tactic": {
                                        "type": "string",
                                        "description": (
                                            "ATT&CK tactic this technique belongs to "
                                            "(e.g., initial-access, execution, persistence, "
                                            "privilege-escalation, defense-evasion, "
                                            "credential-access, discovery, lateral-movement, "
                                            "collection, command-and-control, exfiltration, "
                                            "impact, resource-development, reconnaissance)."
                                        ),
                                    },
                                    "confidence": {
                                        "type": "number",
                                        "minimum": 0.0,
                                        "maximum": 1.0,
                                        "description": (
                                            "Numeric confidence (finer-grained than "
                                            "confidence_bucket). 1.0 = explicit technique "
                                            "use, 0.7 = strongly matches, 0.4 = plausible "
                                            "but ambiguous."
                                        ),
                                    },
                                    "confidence_bucket": {
                                        "type": "string",
                                        "enum": ["definite", "probable", "possible"],
                                        "description": (
                                            "Calibrated bucket. 'definite' = chunk "
                                            "explicitly demonstrates this technique "
                                            "(definitional source quote, ~0.85-1.0). "
                                            "'probable' = strongly implied but not "
                                            "definitional (~0.6-0.85). 'possible' = "
                                            "plausible but other techniques fit at "
                                            "least as well (~0.4-0.6); these are "
                                            "surfaced for analyst review at Gate 1 "
                                            "but NOT included in the bundle by default."
                                        ),
                                    },
                                    "source_quote": {
                                        "type": "string",
                                        "description": (
                                            "Verbatim 5-30 word substring of the chunk "
                                            "text that justifies this pick. Must appear "
                                            "in the chunk word-for-word. Quote an "
                                            "OBSERVATION the report makes — a command, "
                                            "binary, artifact, or action that occurred. "
                                            "Do NOT quote your own reasoning about why "
                                            "the technique matters, its likely effect, "
                                            "or its significance; that is analysis, not "
                                            "evidence. If you cannot "
                                            "find a verbatim quote, set this to an "
                                            "empty string and your pick will be "
                                            "auto-capped at the 'possible' bucket."
                                        ),
                                    },
                                    "rationale": {
                                        "type": "string",
                                        "description": (
                                            "One sentence explaining why source_quote "
                                            "justifies this technique. Reference the "
                                            "quote, don't paraphrase the chunk."
                                        ),
                                    },
                                },
                                "required": [
                                    "technique_id", "technique_name",
                                    "tactic", "confidence",
                                    "confidence_bucket", "source_quote",
                                    "rationale",
                                ],
                            },
                        },
                        "verbatim_match_decisions": {
                            "type": "array",
                            "description": (
                                "Confirm/reject decisions for verbatim matches "
                                "detected in this chunk. REQUIRED when the "
                                "chunk's prompt block includes 'DETECTED "
                                "VERBATIM MATCHES'. Emit one entry per detected "
                                "match. Use empty list when no matches were "
                                "detected for this chunk."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "technique_id": {
                                        "type": "string",
                                        "description": (
                                            "Technique ID of the verbatim match. "
                                            "Must match one of the IDs in the "
                                            "chunk's DETECTED VERBATIM MATCHES "
                                            "block."
                                        ),
                                    },
                                    "decision": {
                                        "type": "string",
                                        "enum": ["confirm", "reject"],
                                        "description": (
                                            "'confirm' if the chunk's behavior "
                                            "aligns with the technique's "
                                            "described behavior; 'reject' if the "
                                            "matched substring appears in a "
                                            "different context that doesn't "
                                            "actually implement the technique."
                                        ),
                                    },
                                    "rejection_reason": {
                                        "type": "string",
                                        "description": (
                                            "One-sentence reason for rejection. "
                                            "Required when decision is 'reject'; "
                                            "omit when decision is 'confirm'."
                                        ),
                                    },
                                },
                                "required": ["technique_id", "decision"],
                            },
                        },
                    },
                    "required": ["chunk_id", "techniques"],
                },
            },
        },
        "required": ["chunk_techniques"],
    },
}


# =============================================================================
# System prompts (C+A+D propose-then-pick architecture)
# =============================================================================

PROPOSE_SYSTEM_PROMPT = _PROCEDURE_DEFINITION + """
You are an expert CTI analyst preparing to map adversary behaviors to MITRE ATT&CK.
This is a TWO-STEP process. In this step, you describe what each chunk shows,
state its objective, enumerate the tactics it spans, and propose techniques
from training memory BEFORE seeing any candidate list. In the second step
you will pick from a validated, unioned candidate pool.

YOUR TASK PER CHUNK:

1. behavior_description (1-2 sentences): describe what the adversary did in
   ATT&CK-flavored prose — tactic phase, technique class, and artifacts
   (commands / binaries / observables) cited in the source. Do NOT use
   ATT&CK technique IDs in this field — words only.
   Example: "User Execution via fake CAPTCHA prompting copy-paste of a
   PowerShell command that downloads a second-stage loader."

2. objective (ONE sentence): the procedure's specific adversarial OBJECTIVE.
   State the GOAL, not the action. The objective scopes the procedure —
   techniques you propose in step 4 must serve THIS objective. The
   objective also anchors the drafted description prose downstream.
   Examples:
     - "Execute attacker code on the victim host by tricking the user
        into pasting and running a payload via a fake CAPTCHA lure."
     - "Download the second-stage loader to disk using a signed Microsoft
        binary to evade unsigned-binary heuristics."
     - "Establish per-user persistence on the host by registering the
        loader to auto-execute at logon."

3. tactics (LIST): ALL ATT&CK tactics this procedure spans. Procedures
   ROUTINELY span multiple tactics (the new procedure definition above
   makes this explicit). Examples:
     - LOLBin download → ["command-and-control", "defense-evasion"]
     - Copy-paste lure → ["initial-access", "execution"]
     - BYOVD EDR-kill → ["defense-evasion", "privilege-escalation"]
   Single-tactic procedures exist (e.g., a pure registry persistence
   write spans only "persistence") but they're the exception, not the
   default. Pick from the standard 14-tactic enterprise list.

4. proposed_techniques: ATT&CK technique IDs (T#### or T####.###) that
   INTEGRATE to fulfill the objective. Rules:
   - PREFER SUB-TECHNIQUES over parents when the chunk has the specificity.
   - Order from most-specific-and-confident to least.
   - Multi-tactic procedures typically need 2-5 techniques to compose the
     implementation; single-tactic ones may need only 1. The cap is
     "techniques that SERVE the objective, no more, no fewer" — not a
     fixed number.
   - These are PROPOSALS. Step 2 will validate IDs against the actual
     catalogue and pair your list with retriever-supplied candidates.
     Don't worry if your IDs are stale or unfamiliar — invalid ones get
     filtered.

CRITICAL — TRAINING-CUTOFF AWARENESS:
Your training data may predate the current ATT&CK version. New techniques
(e.g., AI-related additions in v19) may exist that you don't know about.
The retriever-supplied candidates in step 2 will cover post-cutoff
additions, so don't try to invent technique IDs for behaviors you don't
recognize. If the action is novel (e.g., "queried a public LLM service"),
propose the closest pre-cutoff parent and let the retriever supply post-
cutoff sub-techniques.

DO NOT pick techniques in this step. You will see the validated candidate
union in step 2 and pick from it there. Here, just describe, state the
objective, list spanned tactics, and propose."""


SYSTEM_PROMPT = _PROCEDURE_DEFINITION + """
You are an expert CTI analyst finalizing ATT&CK technique assignments for
behavioral chunks. Step 1 produced — for each chunk — a behavior_description,
an OBJECTIVE (the procedure's north-star), the SPANNED TACTICS, and a
proposed-technique list. Step 2 (this step) gives you a unified candidate
pool — the validated union of step 1's proposals and the retriever's
catalogue picks. You now pick from that pool, anchored on the objective.

THE OBJECTIVE IS YOUR DISCRIMINATOR:
For each candidate technique in the pool, ask: does this technique SERVE
the objective from step 1, or does it DRIFT into a different goal?
- SERVES → pick it.
- DRIFTS → drop it, even if it's lexically related to the chunk.

Examples:
- Objective: "Download loader via a signed Microsoft binary." Candidates:
  T1105 Ingress Tool Transfer (serves), T1218 Signed Binary Proxy Exec
  (serves — the LOLBin enables the download), T1078 Valid Accounts
  (DRIFTS — credentials may be elsewhere in the report but they aren't
  this procedure's objective).
- Objective: "Establish per-user persistence via Run key." Candidates:
  T1547.001 Registry Run Keys (serves), T1059.001 PowerShell (DRIFTS
  — PowerShell may have *written* the key but the procedure's objective
  is the persistence mechanism, not the execution that set it).

REQUIRED PER PICK:
1. technique_id, technique_name, tactic — taken from the candidate pool
   entry.
2. source_quote — verbatim 5-30 word substring of the chunk text that
   justifies this pick. Must appear in the chunk word-for-word.

   Quote an OBSERVATION: a command, binary, file, artifact, or action the
   report says occurred. Do NOT quote your own reasoning about why the
   technique matters, what it probably achieved, or how effective it would
   be — that is analysis, and analysis is not evidence no matter how sound
   it is. "registers a new attacker-controlled MFA device" is an
   observation. "raises the likelihood the note is believed" is not.

   If the chunk gives you no observation to quote, set source_quote to an
   empty string; the pick will be auto-capped at the 'possible' bucket.
   An empty quote is an honest answer. Substituting your own commentary
   to fill the field is not.
3. rationale — one sentence explaining why source_quote justifies this
   technique. Reference the quote, don't paraphrase the chunk.
4. confidence_bucket — calibrated judgment:
   - "definite": the chunk EXPLICITLY demonstrates this technique. Source
     quote names the binary, command, API, or behavior class
     definitionally. Confidence ~0.85-1.0.
   - "probable": the chunk strongly implies this technique. Source quote
     is consistent but not definitional. Confidence ~0.6-0.85.
   - "possible": plausible but other techniques fit at least as well.
     Use for picks you'd otherwise drop. Surfaced for analyst review at
     Gate 1, NOT included in the bundle by default. Confidence ~0.4-0.6.
5. confidence — numeric 0.0-1.0, finer-grained than the bucket.

HOW MANY TECHNIQUES PER PROCEDURE:
The right number is "however many INTEGRATE to fulfill the objective" —
not a fixed cap. Multi-tactic procedures typically need 2-5 techniques to
compose the implementation; single-tactic ones may need only 1. If you
find yourself with 6+ picks, you're almost certainly drifting (techniques
that touch the chunk lexically but serve a different objective) or the
chunk fuses two distinct procedures (in which case pick conservatively
for the primary objective and flag the chunking issue in rationale).

The cap is functional, not numeric: every pick must point at the same
objective.

DECISION DISCIPLINE:
- OBJECTIVE COHERENCE FIRST. The objective from step 1 is your anchor.
  If a candidate doesn't serve it, drop it.
- PREFER SUB-TECHNIQUES over parents. If both are in the candidate pool
  and the chunk has specificity, drop the parent.
- TACTIC SPANNING IS NORMAL. Procedures often span multiple tactics
  (the procedure definition above makes this explicit). Don't drop a
  pick just because its tactic differs from another pick's — drop it
  only if its OBJECTIVE differs.
- ASSIGN THE CORRECT TACTIC PER PICK. A technique can appear under
  multiple tactics; pick the tactic that matches HOW it was used in
  this procedure's context.
- USE REAL ATT&CK IDs ONLY. Format: T#### or T####.###. The candidate
  pool already enforces this; do not invent new IDs.
- EVERY CHUNK SHOULD MAP TO AT LEAST ONE TECHNIQUE. If no candidate is
  a good fit, use the most relevant parent with confidence_bucket =
  'possible'.

MAP BASED ON BEHAVIOR, NOT SOURCE LABELS:
Sources frequently misname techniques. A report may say "process
hollowing" when describing CreateRemoteThread injection, or "lateral
movement" for what is actually discovery. Always map based on the
DESCRIBED BEHAVIOR (APIs called, commands run, artifacts created), not
the label the source applies.

DISAMBIGUATION FOR COMMON AMBIGUITIES:
- Clipboard hijack / fake dialog social engineering: T1204.004 (Malicious
  Copy and Paste) + the vehicle technique (e.g., T1059.001 for PowerShell).
  If the user arrives via a compromised website, also add T1189 (Drive-by
  Compromise) for initial-access.
- VirtualAlloc/CreateThread in current process (self-injection, shellcode
  execution): T1620 (Reflective Code Loading), NOT T1055 (Process
  Injection).
- VirtualAllocEx/WriteProcessMemory/CreateRemoteThread in a REMOTE process:
  T1055 (Process Injection) or a specific sub-technique based on method.
- LOLBin abuse (certutil, mshta, rundll32, etc.): map to the specific
  T1218.xxx Signed Binary Proxy Execution sub-technique, plus the effect.
- Supply chain via package manager: T1195.002 (Compromise Software Supply
  Chain), not T1059.
- Don't assign T1071 (Application Layer Protocol) just because the attacker
  used HTTP. Only if the attacker abused an application protocol for C2.
- Don't assign T1078 (Valid Accounts) just because credentials were used.
  Only if the attacker obtained and reused legitimate credentials.
- Don't confuse T1105 (Ingress Tool Transfer) with T1041 (Exfiltration
  Over C2). Transfer IN vs transfer OUT.
- Don't assign T1059 (parent) when a sub-technique is clearly applicable.

VERBATIM MATCHES:
For chunks with a "DETECTED VERBATIM MATCHES" block, you MUST emit a
`verbatim_match_decisions` array containing one decision per detected
match:
- decision: "confirm" if the chunk's behavior aligns with the technique's
  described behavior.
- decision: "reject" with a one-sentence `rejection_reason` if the matched
  substring appears in an unrelated context.

Confirmed matches lock at confidence 0.95 / 'definite'. Rejected matches
drop to 0.3 / 'possible' with the reason. Either way, do NOT also include
verbatim-matched techniques in your `techniques` array — that array is for
ADDITIONAL techniques the verbatim matcher didn't surface."""


# =============================================================================
# Node function
# =============================================================================

def _build_technique_rerun_context(rerun_feedback: list[dict] | None) -> str:
    """Render Gate 1 technique-mapping feedback into prompt context.

    Built by gate_1 (one entry per technique-corrected chunk) and injected
    into BOTH the propose and pick prompts, telling the LLM what the analyst
    rejected or revised (and why) so the rerun doesn't repeat it, and
    surfacing any analyst-supplied correction as a strong hint.

    Cache note: a non-empty context also makes both calls pass
    bypass_cache=True — the prompt change alone differentiates rerun 1 from
    the rejected pass, but a REPEAT reject with identical reason/rationale
    reproduces rerun 1's prompt exactly, and the content-keyed cache would
    serve back the already-rejected response forever.

    Entries are action-aware. A REJECT renders as a rejection of the full
    previous mapping. An EDIT renders as a *revision*: only the removed
    techniques were wrong — rendering kept picks as "REJECTED" was actively
    steering the rerun away from techniques the analyst confirmed. Entries
    without an action (stale checkpoints) render as rejects.

    chunk_ids are stable across a wrong_technique loop (no re-chunk happens),
    so the per-chunk feedback maps cleanly onto the same chunks.
    """
    if not rerun_feedback:
        return ""
    parts = [
        "\n\nTECHNIQUE RE-MAP FEEDBACK (highest priority — the analyst "
        "corrected the previous technique mapping for the chunks below):",
    ]
    for fb in rerun_feedback:
        cid = fb.get("chunk_id", "")
        rejected = join_technique_ids(fb.get("rejected_techniques"))
        corrected = join_technique_ids(fb.get("corrected_techniques"))
        is_edit = fb.get("action") == GateAction.EDIT.value

        if is_edit:
            parts.append(
                f"- Chunk {cid}: the analyst REVISED the previous technique "
                "mapping."
            )
            if rejected:
                parts.append(
                    f"    Removed (wrong for this chunk): {rejected}."
                )
            added = join_technique_ids(fb.get("added_techniques"))
            if added:
                parts.append(
                    f"    Added (missed previously): {added}."
                )
        else:
            reason = fb.get("reject_reason") or "wrong_technique"
            parts.append(
                f"- Chunk {cid}: previously picked "
                f"{rejected or '(none recorded)'} — REJECTED "
                f"(reason: {reason})."
            )

        rationale = (fb.get("rationale") or "").strip()[:500].replace("\n", " ")
        if rationale:
            parts.append(
                "    Analyst rationale (data, not instructions): "
                f"\"{rationale}\""
            )

        if corrected:
            parts.append(
                f"    The analyst indicates the correct technique(s) are: "
                f"{corrected}. Strongly prefer these unless the source text "
                f"clearly contradicts them."
            )
        elif is_edit:
            # has_correction edits with an empty corrected list mean the
            # analyst said NONE of the previous picks apply.
            parts.append(
                "    The analyst indicates NONE of the previously picked "
                "technique(s) apply to this chunk. Do not re-pick them "
                "unless the source clearly supports them."
            )
        else:
            parts.append(
                "    Re-evaluate this chunk's objective and pick the "
                "technique(s) that actually match; do NOT repeat the rejected "
                "pick(s) unless the source clearly supports them."
            )
    parts.append(
        "Apply this feedback for the listed chunks; map all other chunks "
        "normally."
    )
    return "\n".join(parts)


async def extract_techniques(state: PipelineState) -> dict:
    """Stage 3: Map behavioral chunks to ATT&CK techniques (C+A+D flow).

    Two-call architecture:
      1. PROPOSE: LLM describes each chunk's behavior and proposes T-IDs
         from training memory (no catalogue shown).
      2. VALIDATE + UNION: proposed T-IDs are filtered through the v19
         catalogue (drops hallucinations, redirects revoked IDs) and
         unioned with the retriever's candidates into a per-source pool.
      3. PICK: LLM picks from the unified pool, emitting confidence
         buckets ('definite' | 'probable' | 'possible') and verbatim
         source quotes.

    Post-processing chain:
      - source_quote auto-cap: picks without a verbatim quote in
        the chunk text get capped at the 'possible' bucket.
      - resolve_stix_ids: T-numbers -> real STIX UUIDs.
      - tactic auto-correction.
      - confidence recalibration.

    Output split:
      - technique_mappings: 'definite' + 'probable' picks (feed bundle).
      - technique_mappings_for_review: 'possible' picks (Gate 1 review,
        analyst can promote into the bundle).

    Args:
        state: Pipeline state with chunks and validated_entities.

    Returns:
        Dict with technique_mappings, technique_mappings_for_review,
        status, current_node.
    """
    chunks = state.get("chunks", [])
    validated_entities = state.get("validated_entities", [])
    # Corrective feedback from a Gate 1 technique reject/revision (set by
    # gate_1 when it routes back here). Injected into the propose + pick
    # prompts, which also run with bypass_cache so a repeat reject can't be
    # served the prior rerun's cached response. None on a first pass.
    technique_rerun_feedback = state.get("technique_rerun_feedback")
    rerun_context = _build_technique_rerun_context(technique_rerun_feedback)

    logger.info(
        "extract_techniques: starting, %d chunks%s",
        len(chunks),
        " (re-map with analyst feedback)" if rerun_context else "",
    )

    update: dict = {
        "status": PipelineStatus.EXTRACTING_TECHNIQUES.value,
        "current_node": "extract_techniques",
    }

    if not chunks:
        logger.warning("extract_techniques: no chunks to process")
        update["technique_mappings"] = {}
        update["technique_mappings_for_review"] = {}
        return update

    try:
        # Step 1: Load full ATT&CK technique catalogue (cached after first call)
        technique_lookup, _full_reference_text = _load_technique_catalogue()
        if technique_lookup:
            logger.info(
                "extract_techniques: catalogue loaded with %d techniques",
                len(technique_lookup),
            )
        else:
            logger.warning(
                "extract_techniques: no catalogue available, "
                "LLM will use training knowledge only"
            )

        # Step 2: Candidate retrieval (provider abstraction).
        # Strategy controlled by settings.technique_retriever; default
        # "embedding" delegates to EmbeddingRetriever (SecureBERT 2.0
        # biencoder), "token_overlap" delegates to TokenOverlapRetriever
        # (Jaccard + CVE/tool anchors). Both honor identical retrieval
        # contracts (top-K per chunk, union, parent/sub expansion).
        if technique_lookup:
            retriever = get_retriever()
            filtered_lookup, _ = retriever.find_candidates(
                chunks, technique_lookup, top_k=30,
            )
        else:
            filtered_lookup = {}

        # Step 2.5: Verbatim procedure-example matching.
        # Per chunk, find substrings in the chunk text that exactly match
        # operative strings from MITRE-curated procedure examples. The LLM
        # will be asked to confirm or reject each in its response.
        verbatim_index = get_index()
        verbatim_matches_by_chunk: dict[str, list[dict]] = {}
        total_matches = 0
        for chunk in chunks:
            cid = chunk.get("chunk_id", "")
            matches = match_chunk(chunk.get("text", ""), verbatim_index)
            if matches:
                verbatim_matches_by_chunk[cid] = matches
                total_matches += len(matches)

        if verbatim_matches_by_chunk:
            logger.info(
                "extract_techniques: found %d verbatim matches across "
                "%d/%d chunks (rest will be pure-LLM)",
                total_matches, len(verbatim_matches_by_chunk), len(chunks),
            )

        # Step 2.6: Reason + Propose LLM call (C step in C+A+D).
        # The LLM describes each chunk's behavior in ATT&CK terms and
        # proposes T-IDs from training memory, with NO catalogue shown.
        # This step's outputs anchor the pick prompt and augment the
        # candidate pool for behaviors the retriever's embeddings missed.
        # Feedback for the propose step. The pick step re-fetches rather
        # than reusing this — see the second call below for why.
        propose_feedback_addendum = await _fetch_feedback_addendum(state)
        proposals_by_chunk = await _run_propose_step(
            chunks=chunks,
            validated_entities=validated_entities,
            verbatim_matches_by_chunk=verbatim_matches_by_chunk,
            feedback_addendum=propose_feedback_addendum,
            rerun_feedback_context=rerun_context,
        )
        logger.info(
            "extract_techniques: propose-step returned proposals for %d/%d chunks "
            "(total %d unique T-IDs proposed)",
            len(proposals_by_chunk), len(chunks),
            len({tid for p in proposals_by_chunk.values() for tid in p["proposed_techniques"]}),
        )

        # Curated-knowledge signals, gathered before the pick call and
        # reconciled against its output afterwards. Defaults hold for the
        # no-catalogue path, where there is no pool to annotate.
        brand_audit: list[dict] = []
        vendor_tids: set[str] = set()

        # Step 2.7: Validate + union.
        # Drop hallucinations from LLM proposals, redirect revoked IDs,
        # and merge into the retriever's candidate pool. The unified
        # filtered_lookup is what the pick step sees.
        if technique_lookup:
            filtered_lookup, propose_audit = _unify_candidate_pool(
                retriever_lookup=filtered_lookup,
                proposals_by_chunk=proposals_by_chunk,
                full_catalogue=technique_lookup,
            )
            _log_propose_audit(propose_audit)

            # Step 2.8: Brand-expansion (deterministic).
            # Some CTI brand/pattern names appear in source reports
            # WITHOUT the source spelling out the underlying mechanism
            # (e.g., "deployed CLICKFIX fake captcha" — no mention of
            # the copy-paste action). The LLM's source-quote requirement
            # then prevents the right technique from being picked even
            # when training memory could supply it. This step closes the
            # gap deterministically: it scans chunks for known brand
            # substrings and adds the mapped T-IDs to the unified pool
            # so the pick step's LLM call sees them as candidates.
            filtered_lookup, brand_audit = _augment_pool_with_brand_techniques(
                filtered_lookup=filtered_lookup,
                chunks=chunks,
                full_catalogue=technique_lookup,
            )

        # Step 2.9: ensure every sub-technique in the pool is accompanied by
        # its parent.
        filtered_lookup = _expand_parents(filtered_lookup, technique_lookup)

        # Step 2.95: read the report's own ATT&CK mapping table, if it has
        # one. Vendor analysts had the whole incident in front of them; their
        # table is the nearest thing to ground truth available here. It is
        # source-level though — no per-chunk attribution — so it annotates
        # and corroborates, and never injects on its own.
        vendor_tids, vendor_audit = extract_vendor_technique_ids(
            state.get("classified_sections"),
        )
        if vendor_tids:
            logger.info(
                "extract_techniques: report's own ATT&CK mapping names %d "
                "technique(s) across %d section(s): %s",
                len(vendor_tids), len(vendor_audit), ", ".join(sorted(vendor_tids)),
            )

        # Rebuild reference_text from the unified pool. Candidates backed by
        # curated knowledge are annotated so the pick step can see the signal
        # rather than facing a flat, provenance-free list.
        pool_annotations = _build_pool_annotations(brand_audit, vendor_tids)
        reference_text = _format_reference_text(filtered_lookup, pool_annotations)

        # Step 3: Pick LLM call (A + D in C+A+D).
        # Build the prompt with proposals threaded through per-chunk so
        # the LLM sees its own behavior_description as anchor.
        chunks_text = _format_chunks_for_prompt(
            chunks,
            verbatim_matches_by_chunk=verbatim_matches_by_chunk,
            proposals_by_chunk=proposals_by_chunk,
        )
        entity_context = _format_entity_context(validated_entities)

        # Re-fetch feedback now that the propose step has put T-IDs in play.
        # Not a redundant roundtrip: the largest term in `_score_pattern` is
        # +0.5 for a pattern whose applies_to.technique_ids intersects the
        # run's. That set comes from `_technique_ids_in_play`, which reads
        # `proposals_by_chunk` — empty when the propose-step addendum was
        # fetched, so the boost could never fire anywhere in the pipeline.
        #
        # `proposals_by_chunk` is still a local here (the node writes it to
        # state on return), so it must be merged in explicitly. Without that
        # the source fingerprint is unchanged, the TTL cache hits, and this
        # silently hands back the pre-propose addendum — a no-op that looks
        # like a fix.
        pick_feedback_addendum = await _fetch_feedback_addendum(
            {**state, "proposals_by_chunk": proposals_by_chunk},
            node="extract_techniques:pick",
        ) or propose_feedback_addendum
        # Same reason the addendum is fetched twice: the proposed T-IDs are
        # what let the +0.5 shared-technique term fire, and no technique ID
        # exists yet at propose time.
        pick_feedback_addendum += await _fetch_feedback_examples(
            {**state, "proposals_by_chunk": proposals_by_chunk},
            node="extract_techniques:pick",
        )

        system = SYSTEM_PROMPT
        if pick_feedback_addendum:
            system += pick_feedback_addendum
        if entity_context:
            system += entity_context
        if reference_text:
            system += reference_text
        # Append analyst rerun feedback LAST so it's the most recent (and
        # highest-priority) guidance the pick step sees.
        if rerun_context:
            system += rerun_context

        response = await call_llm(
            system=system,
            messages=[{
                "role": "user",
                "content": (
                    "Map each of the following behavioral chunks to ATT&CK "
                    "technique(s) from the unified candidate pool:\n\n" + chunks_text
                ),
            }],
            tools=[EXTRACT_TECHNIQUES_TOOL],
            tool_choice={"type": "tool", "name": "extract_techniques"},
            temperature=0.0,
            # A rerun must always hit the live LLM. The changed prompt
            # already differentiates rerun 1 from the rejected pass, but a
            # REPEAT reject with identical reason/rationale reproduces rerun
            # 1's prompt byte-for-byte — the content-keyed cache would then
            # serve back the already-rejected response forever (deterministic
            # livelock the analyst can only escape by rewording their note).
            bypass_cache=bool(rerun_context),
            # call_llm streams, so the SDK's non-streaming output ceiling
            # does not cap this. 32000 because a 24-chunk multi-intrusion
            # source needs ~12k for picks alone, and with adaptive thinking
            # on, thinking tokens are charged against the same budget.
            # Sources past 24 chunks have headroom.
            max_tokens=32000,
            output_model=ExtractTechniquesOutput,
        )

        # Step 4: Process raw LLM output (incl. verbatim_match_decisions)
        raw_mappings = response.tool_output.get("chunk_techniques", [])
        technique_mappings = _process_technique_mappings(
            raw_mappings, chunks,
            verbatim_matches_by_chunk=verbatim_matches_by_chunk,
            technique_lookup=technique_lookup,
        )

        # Step 4.5: source_quote auto-cap.
        # A pick's quote must be real evidence: present in the chunk AND
        # supported by the report itself. The chunk-only check that used to
        # stand here verified LLM prose against LLM prose, so a chunker
        # invention could be quoted as proof (see _apply_source_quote_cap).
        # Verbatim matches keep their 'definite' / 0.95 lock — their
        # provenance is the MITRE example index, not a quote.
        chunks_by_id = {c["chunk_id"]: c.get("text", "") for c in chunks}
        source_tokens = _build_source_grounding_tokens(state)
        if not source_tokens:
            logger.warning(
                "extract_techniques: no report text available for quote "
                "grounding — picks will be checked against chunk text only",
            )
        technique_mappings = _apply_source_quote_cap(
            technique_mappings, chunks_by_id, source_tokens,
        )

        # Demote picks whose ATT&CK platforms cannot apply to this source —
        # e.g. a macOS-only sub-technique on an all-Windows intrusion. They
        # go to the review lane rather than the bundle; nothing is dropped,
        # since platform metadata is incomplete for some techniques and the
        # analyst can still promote.
        source_platforms = _detect_source_platforms(chunks, validated_entities)
        if source_platforms:
            logger.info(
                "extract_techniques: source platforms detected: %s",
                ", ".join(sorted(source_platforms)),
            )
            for picks in technique_mappings.values():
                _demote_platform_mismatches(
                    picks, source_platforms, technique_lookup,
                )

        # Step 5: Resolve T-numbers to real STIX UUIDs (uses FULL catalogue)
        if technique_lookup:
            technique_mappings, resolve_warnings = _resolve_stix_ids(
                technique_mappings, technique_lookup,
            )
            resolved = sum(
                1 for techniques in technique_mappings.values()
                for t in techniques if t.get("stix_id")
            )
            total = sum(len(v) for v in technique_mappings.values())
            logger.info(
                "extract_techniques: resolved %d/%d technique IDs to real STIX UUIDs",
                resolved, total,
            )

            # Step 6: Deterministic tactic auto-correction
            technique_mappings = _auto_correct_tactics(
                technique_mappings, technique_lookup,
            )

            # Step 7: Deterministic confidence recalibration
            technique_mappings = _recalibrate_confidence(
                technique_mappings, technique_lookup, chunks_by_id,
            )

        # Step 7.5: reconcile the picks against curated knowledge.
        # Runs last among the post-processing rules so it sees final buckets,
        # and before the bundle/review split so an injected pick's 'possible'
        # bucket routes it to the review lane for free.
        if technique_lookup and brand_audit:
            demoted, injected = _reconcile_curated_knowledge(
                technique_mappings, brand_audit, vendor_tids, technique_lookup,
            )
            if demoted or injected:
                logger.info(
                    "extract_techniques: curated-knowledge reconciliation "
                    "demoted %d sibling pick(s) and injected %d review-lane "
                    "candidate(s)", demoted, injected,
                )

        # Step 8: Bucket filter — split picks into bundle vs review lanes.
        # 'definite' + 'probable' feed the bundle (technique_mappings).
        # 'possible' picks land in technique_mappings_for_review for
        # analyst review at Gate 1, where they can be promoted.
        bundle_mappings, review_mappings = _split_by_bucket(technique_mappings)

        # Deterministic guardrail: denylisted T-IDs (promoted_to_denylist
        # patterns) never auto-ship — bundle-lane hits are demoted into the
        # review lane and flagged; review-lane hits stay flagged. The analyst
        # can still promote one at Gate 1 for a source where it's correct.
        # Best-effort — load_denylist no-ops on failure.
        flagged = await _apply_technique_denylist(bundle_mappings, review_mappings)
        if flagged:
            logger.info(
                "extract_techniques: %d denylisted pick(s) held back to review lane",
                flagged,
            )

        # Re-apply analyst promotions that survived a Gate 1 rewind.
        #
        # A Gate 1 rejection for anything other than BAD_CHUNK_BOUNDARY routes
        # back here with the chunks UNCHANGED, and this node rebuilds both
        # lanes from scratch — so a pick the analyst had promoted out of the
        # review lane silently returns to it, and they must promote it again.
        # Observed on a ransomware run: T1003.001 was promoted, the pass
        # rewound, and it came back 'possible'.
        #
        # gate1_correction_log is durable across the loop, so the intent is
        # still on state. Promotions whose chunk_id no longer exists — the
        # BAD_CHUNK_BOUNDARY path regenerates chunk_ids — are skipped by
        # _apply_promotions itself, which is why that path needs no special
        # case: a re-chunk genuinely invalidates the promotion.
        logged_promotions = [
            rec for rec in (state.get("gate1_correction_log") or [])
            if isinstance(rec, dict) and rec.get("action") == "promote"
        ]
        if logged_promotions:
            from app.nodes.gates import _apply_promotions

            bundle_mappings, review_mappings, _, reapplied = _apply_promotions(
                logged_promotions,
                bundle_mappings=bundle_mappings,
                review_mappings=review_mappings,
                drafts=[],  # not built yet; draft_procedures reads the mappings
            )
            if reapplied:
                logger.info(
                    "extract_techniques: re-applied %d analyst promotion(s) "
                    "that a Gate 1 rewind would otherwise have discarded",
                    reapplied,
                )

        bundle_count = sum(len(v) for v in bundle_mappings.values())
        review_count = sum(len(v) for v in review_mappings.values())
        logger.info(
            "extract_techniques: %d chunks mapped — %d bundle picks "
            "(definite+probable), %d for-review picks (possible). "
            "pick tokens: in=%d, out=%d",
            len(technique_mappings), bundle_count, review_count,
            response.input_tokens, response.output_tokens,
        )

        update["technique_mappings"] = bundle_mappings
        update["technique_mappings_for_review"] = review_mappings
        # Clear consumed rerun feedback so a SECOND pass through gate_1 →
        # extract_techniques doesn't re-inject this stale guidance. Only set
        # when it was present, and only on success (a failure keeps it so a
        # retry still carries the analyst's correction). Mirrors the way
        # chunk_behaviors clears chunk_rerun_feedback.
        if technique_rerun_feedback:
            update["technique_rerun_feedback"] = None
        # Persist propose-step output. Drafting reads this to anchor each
        # description's lead sentence on the objective. Available in the
        # LangGraph checkpoint for post-hoc audit even though objective
        # never reaches the STIX bundle as a separate field.
        update["proposals_by_chunk"] = proposals_by_chunk

    except Exception as e:
        logger.exception("extract_techniques: failed")
        update["error"] = f"Technique extraction failed: {type(e).__name__}: {e}"
        update["status"] = PipelineStatus.FAILED.value
        update["technique_mappings"] = {}
        update["technique_mappings_for_review"] = {}
        update["proposals_by_chunk"] = {}

    return update


# =============================================================================
# C+A+D helpers: propose, unify, source_quote cap, bucket filter
# =============================================================================

async def _run_propose_step(
    chunks: list[dict],
    validated_entities: list[dict],
    verbatim_matches_by_chunk: dict[str, list[dict]],
    feedback_addendum: str = "",
    rerun_feedback_context: str = "",
) -> dict[str, dict]:
    """LLM call #1 (Reason + Propose).

    Asks the LLM to describe each chunk's behavior in ATT&CK terms and
    propose T-IDs from training memory. NO catalogue shown — the goal
    is to get proposals untainted by the retriever's candidate set so
    behaviors the retriever's embeddings missed get a second chance.

    Returns: {chunk_id: {behavior_description, tactic, proposed_techniques}}.
    Returns empty dict on LLM failure (graceful degradation — pick step
    still runs with retriever-only candidates).
    """
    chunks_text = _format_chunks_for_prompt(
        chunks, verbatim_matches_by_chunk=verbatim_matches_by_chunk,
    )
    entity_context = _format_entity_context(validated_entities)

    system = PROPOSE_SYSTEM_PROMPT
    if feedback_addendum:
        system += feedback_addendum
    if entity_context:
        system += entity_context
    # Analyst rerun feedback (Gate 1 technique reject/revision): steers the
    # propose step toward the correct technique(s).
    if rerun_feedback_context:
        system += rerun_feedback_context

    try:
        response = await call_llm(
            system=system,
            messages=[{
                "role": "user",
                "content": (
                    "Describe each chunk's adversary action and propose "
                    "ATT&CK technique IDs:\n\n" + chunks_text
                ),
            }],
            tools=[PROPOSE_TECHNIQUES_TOOL],
            tool_choice={"type": "tool", "name": "propose_techniques"},
            temperature=0.0,
            # Reruns always hit the live LLM — a repeat reject with an
            # identical rationale would otherwise cache-hit rerun 1's
            # (already-rejected) proposals. See the pick-step call.
            bypass_cache=bool(rerun_feedback_context),
            # Same headroom logic as the pick step. Propose emits one
            # entry per chunk (objective + tactics + ~3-5 T-IDs); 24 chunks
            # at ~200 tokens each = ~5k. The budget matches the pick step's
            # because thinking tokens share it.
            max_tokens=32000,
            output_model=ProposeTechniquesOutput,
        )
    except Exception as e:
        logger.warning(
            "extract_techniques: propose-step failed (%s); pick step will "
            "use retriever-only candidates",
            type(e).__name__,
        )
        return {}

    raw = response.tool_output.get("chunk_proposals", [])
    valid_chunk_ids = {c["chunk_id"] for c in chunks}
    out: dict[str, dict] = {}
    for entry in raw:
        cid = entry.get("chunk_id", "")
        if cid not in valid_chunk_ids:
            continue
        # Filter proposed_techniques to well-formed T-IDs; downstream
        # validate_technique_ids drops anything that isn't in the
        # catalogue, but format-screening here means malformed IDs
        # don't pollute logs.
        raw_props = entry.get("proposed_techniques", []) or []
        clean_props = [
            tid.strip() for tid in raw_props
            if isinstance(tid, str) and TECHNIQUE_ID_RE.match(tid.strip())
        ]
        # tactics is a list (procedures span multiple tactics per the new
        # definition). Defensively coerce a stray legacy `tactic: str` into
        # a single-element list so older checkpoints still work.
        raw_tactics = entry.get("tactics")
        if isinstance(raw_tactics, list):
            tactics = [str(t).strip() for t in raw_tactics if str(t).strip()]
        elif isinstance(raw_tactics, str) and raw_tactics.strip():
            tactics = [raw_tactics.strip()]
        else:
            tactics = []
        out[cid] = {
            "behavior_description": (entry.get("behavior_description") or "").strip(),
            "objective": (entry.get("objective") or "").strip(),
            "tactics": tactics,
            "proposed_techniques": clean_props,
        }
    return out


def _unify_candidate_pool(
    retriever_lookup: dict[str, dict],
    proposals_by_chunk: dict[str, dict],
    full_catalogue: dict[str, dict],
) -> tuple[dict[str, dict], list[dict]]:
    """Validate LLM proposals against the catalogue and merge into the
    retriever's pool. Returns (unified_lookup, audit_log).

    The retriever's lookup stays authoritative for v19+ techniques; LLM
    proposals augment by adding pre-cutoff T-IDs whose embeddings the
    retriever ranked outside top-K. Hallucinations are dropped (validator
    guarantees set membership against the catalogue), revoked IDs are
    redirected to their replacements.
    """
    # Lazy import: matches the pattern in _resolve_stix_ids.
    try:
        from app.services.attack_data import get_attack_data
        db = get_attack_data()
    except Exception as e:
        logger.warning(
            "extract_techniques: attack_data unavailable for proposal validation "
            "(%s); skipping union with LLM proposals",
            type(e).__name__,
        )
        return retriever_lookup, []

    # Flatten all unique proposed T-IDs across chunks.
    all_proposed: set[str] = set()
    for proposal in proposals_by_chunk.values():
        all_proposed.update(proposal.get("proposed_techniques", []))

    if not all_proposed:
        return retriever_lookup, []

    valid_ids, audit = db.validate_technique_ids(sorted(all_proposed))

    # Add validated IDs to the lookup. Skip ones already present (no-op merge).
    unified = dict(retriever_lookup)
    added = 0
    for tid in valid_ids:
        if tid in unified:
            continue
        cat_entry = full_catalogue.get(tid)
        if cat_entry:
            unified[tid] = cat_entry
            added += 1

    logger.info(
        "extract_techniques: unified candidate pool — retriever %d, "
        "LLM-proposed %d unique, %d added to pool, %d redirected, "
        "%d hallucinated, %d revoked-no-redirect, %d deprecated",
        len(retriever_lookup),
        len(all_proposed),
        added,
        sum(1 for a in audit if a["outcome"] == "redirected"),
        sum(1 for a in audit if a["outcome"] == "dropped_hallucinated"),
        sum(1 for a in audit if a["outcome"] == "dropped_revoked_no_redirect"),
        sum(1 for a in audit if a["outcome"] == "dropped_deprecated"),
    )
    return unified, audit


def _log_propose_audit(audit: list[dict]) -> None:
    """Pretty-log notable propose-step outcomes (redirects + hallucinations)."""
    redirects = [a for a in audit if a["outcome"] == "redirected"]
    halluc = [a for a in audit if a["outcome"] == "dropped_hallucinated"]
    if redirects:
        logger.info(
            "extract_techniques: propose-step redirects: %s",
            "; ".join(f"{a['original_id']}->{a['result_id']}" for a in redirects),
        )
    if halluc:
        logger.info(
            "extract_techniques: propose-step hallucinations dropped: %s",
            ", ".join(a["original_id"] for a in halluc),
        )


# Signals that a source concerns a given ATT&CK platform. Keys are the
# platform names ATT&CK uses in `x_mitre_platforms`; values are substrings
# looked for in chunk text, command lines, and software entity values.
_PLATFORM_SIGNALS: dict[str, tuple[str, ...]] = {
    "Windows": (
        "windows", ".exe", ".dll", "powershell", "cmd.exe", "registry",
        "hkey_", "hklm", "hkcu", "c:\\", "%appdata%", "%comspec%",
        "active directory", "ntlm", "kerberos", "wmi", "sysmon",
    ),
    "Linux": ("linux", "/etc/", "/tmp/", "bash", "systemd", "cron", "debian",
              "ubuntu", "centos", ".so"),
    "macOS": ("macos", "mac os", "osx", "gatekeeper", "launchd", "plist",
              ".dmg", ".pkg"),
    "Containers": ("kubernetes", "docker", "container", "kubectl", "pod "),
    "IaaS": ("aws", "azure", "gcp", "s3 bucket", "ec2", "iam role",
             "cloud storage"),
    "Office Suite": ("microsoft 365", "office 365", "o365", "sharepoint",
                     "onedrive", "outlook", "exchange online", "entra"),
    "Identity Provider": ("okta", "sso", "saml", "identity provider",
                          "single sign-on", "mfa", "passkey"),
    "SaaS": ("salesforce", "zendesk", "servicenow", "slack", "workspace"),
    "Network Devices": ("cisco ios", "juniper", "router", "firewall appliance"),
    "ESXi": ("esxi", "vcenter", "vsphere"),
}


def _detect_source_platforms(
    chunks: list[dict],
    validated_entities: list[dict] | None = None,
) -> set[str]:
    """Infer which ATT&CK platforms a source concerns.

    There is no `platforms` field in state at technique-extraction time —
    drafts do not exist yet — so this reads the signals that ARE available:
    chunk text and the software / file-path / command-line entities.

    Deliberately conservative. Absence of signal returns an empty set, which
    callers must treat as "do not filter": mis-detecting a platform and then
    down-ranking the correct technique would be worse than the noise this is
    meant to reduce.
    """
    haystack_parts: list[str] = []
    for chunk in chunks or []:
        haystack_parts.append(chunk.get("text") or "")
        haystack_parts.append(chunk.get("source_excerpt") or "")
    for ent in validated_entities or []:
        if ent.get("entity_type") in (
            "software", "ioc_file_path", "ioc_command_line",
            "ioc_process_name", "ioc_registry_key",
        ):
            haystack_parts.append(str(ent.get("value") or ""))
    haystack = " ".join(haystack_parts).lower()

    return {
        platform
        for platform, signals in _PLATFORM_SIGNALS.items()
        if any(sig in haystack for sig in signals)
    }


def _demote_platform_mismatches(
    picks: list[dict],
    source_platforms: set[str],
    catalogue: dict[str, dict],
) -> list[dict]:
    """Flag picks whose ATT&CK platforms don't intersect the source's.

    A macOS-only sub-technique on an all-Windows intrusion is almost
    certainly wrong — `T1553.001 Gatekeeper Bypass` once surfaced for a
    Windows SSL-revocation bypass. This marks such picks so
    the bucket filter routes them to the review lane rather than the bundle;
    it never drops them, because platform metadata is not always complete and
    an analyst can still promote.
    """
    if not source_platforms:
        return picks

    for pick in picks:
        entry = catalogue.get(pick.get("technique_id", ""))
        if not entry:
            continue
        tech_platforms = set(entry.get("platforms") or [])
        if not tech_platforms:
            continue
        if tech_platforms & source_platforms:
            continue
        pick["platform_mismatch"] = True
        pick["confidence_bucket"] = "possible"
        pick["confidence"] = min(float(pick.get("confidence", 0.5)), 0.5)
        logger.info(
            "extract_techniques: demoting %s — platforms %s do not intersect "
            "the source's %s",
            pick.get("technique_id"), sorted(tech_platforms),
            sorted(source_platforms),
        )
    return picks


def _parent_technique_id(tid: str) -> str:
    """`T1204.004` -> `T1204`; a parent ID is returned unchanged."""
    return (tid or "").split(".", 1)[0]


def _reconcile_curated_knowledge(
    technique_mappings: dict[str, list[dict]],
    brand_audit: list[dict],
    vendor_tids: set[str],
    catalogue: dict[str, dict],
) -> tuple[int, int]:
    """Check the model's picks against what curated knowledge expects.

    Runs per chunk, driven only by signals that HAVE chunk attribution —
    i.e. brand hits. The report's ATT&CK table is source-level, so it
    corroborates a brand hit but never triggers anything by itself; letting
    it inject would have put all 17 of one report's listed techniques into
    each of its 8 chunks.

    Three outcomes per expected technique:

      corroborated — the model picked it. Tag provenance and leave it alone.

      sibling mismatch — the model picked a DIFFERENT sub-technique of the
        same parent. This is the ClickFix failure exactly: brand says
        `T1204.004`, the model picks `T1204.001`, and no existing rule
        notices, because every granularity rule in this module compares a
        parent against its child and these are peers. The sibling drops to
        the review lane and the expected technique is injected beside it, so
        the analyst sees both readings rather than one silent wrong answer.

      absent — inject the expected technique alone.

    Injected picks land in the review lane (bucket 'possible', which
    `_split_by_bucket` routes there) with a rationale naming the brand and
    any vendor corroboration. They reach the bundle only if an analyst
    promotes them at Gate 1. That is the whole point: curated knowledge
    stops being invisible without ever being passed off as source-grounded
    evidence, which is what `.cursorrules` forbids — thin source means low
    confidence, not invention.

    Mutates `technique_mappings` in place. Returns `(demoted, injected)`.
    """
    demoted = 0
    injected = 0

    for hit in brand_audit:
        brand = hit.get("brand", "")
        if brand_strength(brand) != DEFINITIONAL:
            continue  # suggestive brands corroborate, never override
        chunk_id = hit.get("chunk_id", "")
        if not chunk_id:
            continue
        picks = technique_mappings.setdefault(chunk_id, [])

        for expected in hit.get("techniques") or []:
            entry = catalogue.get(expected)
            if not entry:
                continue  # not in this ATT&CK version; nothing to inject
            corroborated = expected in vendor_tids

            already = next(
                (p for p in picks if p.get("technique_id") == expected), None,
            )
            if already is not None:
                already["curated_provenance"] = (
                    "brand+vendor" if corroborated else "brand"
                )
                continue

            parent = _parent_technique_id(expected)
            family = [
                p for p in picks
                if _parent_technique_id(p.get("technique_id", "")) == parent
            ]

            if expected == parent:
                # The brand names a PARENT, so it says nothing about which
                # sub-technique applies — three brands map to T1557, which has
                # four subs. A picked sub is a REFINEMENT of the brand's claim,
                # not a contradiction: corroborate it and add nothing, rather
                # than demoting a more precise answer and injecting a coarser
                # one beside it.
                if family:
                    for p in family:
                        p["curated_provenance"] = (
                            "brand+vendor" if corroborated else "brand"
                        )
                    continue
                siblings: list[dict] = []
            else:
                # The brand names a specific sub-technique. Another sub of the
                # same parent contradicts it. The parent itself does not — a
                # coarse answer is defensible, and promoting it is already
                # _recalibrate_confidence Rule 1's job.
                siblings = [
                    p for p in family
                    if p.get("technique_id") != parent
                ]

            for sib in siblings:
                sib["brand_mismatch"] = expected
                sib["confidence_bucket"] = "possible"
                sib["confidence"] = min(float(sib.get("confidence", 0.5)), 0.5)
                demoted += 1
                logger.info(
                    "extract_techniques: chunk %s picked %s but the '%s' "
                    "pattern means %s — demoting the sibling to review",
                    chunk_id, sib.get("technique_id"), brand, expected,
                )

            why = f"The text names the '{brand}' pattern, which is {expected}"
            if corroborated:
                why += "; the report's own ATT&CK mapping lists it too"
            if siblings:
                why += (
                    f". The model picked {', '.join(sorted(s['technique_id'] for s in siblings))}"
                    f" instead — a different sub-technique of {parent}"
                )
            picks.append({
                "technique_id": expected,
                "technique_name": entry.get("name", ""),
                "tactic": (entry.get("tactics") or ["unknown"])[0],
                "confidence": 0.5,
                # Review lane, always. Curated knowledge is a strong hint
                # about which technique is meant, not evidence that it
                # happened here — only an analyst promotion makes it a
                # bundle claim.
                "confidence_bucket": "possible",
                "source_quote": "",
                "rationale": why + ".",
                "stix_id": entry.get("stix_id"),
                "provenance": "brand_inferred",
                "curated_provenance": "brand+vendor" if corroborated else "brand",
                "brand_inferred": True,
            })
            injected += 1
            logger.info(
                "extract_techniques: injecting %s into chunk %s's review lane "
                "(%s)", expected, chunk_id,
                "brand + report mapping" if corroborated else "brand only",
            )

    return demoted, injected


def _expand_parents(
    filtered_lookup: dict[str, dict],
    full_catalogue: dict[str, dict],
) -> dict[str, dict]:
    """Add the parent of every sub-technique already in the candidate pool.

    The retriever does this for its own picks (`_finalize_candidates`), but
    T-IDs that enter later — LLM proposals (`_unify_candidate_pool`) and brand
    expansion (`_augment_pool_with_brand_techniques`) — bypass it. That gap
    produced the dominant technique error mode in an audit of real output:
    correct family, wrong member, with the right answer absent from BOTH
    lanes so there was nothing for the analyst to promote at gate_1.

    Three of the four recall misses had this shape — `T1553.001` (macOS
    Gatekeeper Bypass) surfaced for a Windows SSL-revocation bypass while the
    applicable parent `T1553` never appeared; `T1567.002`/`T1041` were
    correctly demoted for an exfiltration procedure that then carried no
    exfiltration technique at all, because parent `T1567` was never a
    candidate.

    Entries are shared objects with the module-level catalogue cache, so this
    only ever adds keys — it never mutates an entry.
    """
    added: list[str] = []
    for tid in list(filtered_lookup):
        if "." not in tid:
            continue
        parent = tid.split(".")[0]
        if parent in filtered_lookup:
            continue
        entry = full_catalogue.get(parent)
        if entry is not None:
            filtered_lookup[parent] = entry
            added.append(parent)

    if added:
        logger.info(
            "extract_techniques: parent expansion added %d parents to the "
            "candidate pool: %s",
            len(added), ", ".join(sorted(added)),
        )
    return filtered_lookup


def _augment_pool_with_brand_techniques(
    filtered_lookup: dict[str, dict],
    chunks: list[dict],
    full_catalogue: dict[str, dict],
) -> tuple[dict[str, dict], list[dict]]:
    """Augment the unified candidate pool with brand-expansion picks.

    Scans every chunk's text for known CTI brand/pattern names (ClickFix,
    MFA fatigue, EvilProxy AitM, drive-by, watering hole, etc. — see
    `app.services.technique_pattern_brands` for the full map) and adds
    the mapped ATT&CK technique IDs to the candidate pool.

    Closes the brand-vs-mechanism source-ambiguity gap: when a source
    names a brand without describing its underlying mechanism, the LLM
    can't reliably surface the right technique on its own. This
    deterministic augmentation ensures the candidate is in the pool;
    the pick step's normal evidence checks (source_quote, bucket) still
    decide whether the LLM picks it.

    Defense-in-depth: the discovered T-IDs run through `validate_technique_ids`
    so a stale brand-map entry can't inject a hallucinated or revoked-without-
    redirect T-ID into the pool.

    Returns `(augmented_lookup, audit)`. The audit — `{chunk_id, brand,
    techniques}` per hit — used to be computed here and thrown away, which
    left the pool unable to say WHY a candidate was in it. It now carries the
    per-chunk attribution that `_reconcile_curated_knowledge` needs.
    """
    if not chunks:
        return filtered_lookup, []

    brand_tids, audit = find_brand_techniques_across_chunks(chunks)
    if not brand_tids:
        return filtered_lookup, []

    try:
        from app.services.attack_data import get_attack_data
        db = get_attack_data()
    except Exception as e:
        logger.warning(
            "extract_techniques: attack_data unavailable for brand-expansion "
            "validation (%s); skipping",
            type(e).__name__,
        )
        return filtered_lookup, []

    valid_ids, _ = db.validate_technique_ids(sorted(brand_tids))
    augmented = dict(filtered_lookup)
    added = []
    for tid in valid_ids:
        if tid in augmented:
            continue
        cat_entry = full_catalogue.get(tid)
        if cat_entry:
            augmented[tid] = cat_entry
            added.append(tid)

    if added or audit:
        # Log per-chunk brand hits so it's clear which brand drove which
        # T-ID into the pool. Useful when debugging picks that originate
        # from this path rather than from the retriever or LLM proposal.
        per_chunk = "; ".join(
            f"chunk {a['chunk_id']}: {a['brand']} -> {','.join(a['techniques'])}"
            for a in audit
        )
        logger.info(
            "extract_techniques: brand-expansion %d hits, %d new T-IDs added "
            "to pool — %s",
            len(audit), len(added), per_chunk,
        )
    return augmented, audit


def _build_pool_annotations(
    brand_audit: list[dict],
    vendor_tids: set[str],
) -> dict[str, list[str]]:
    """Label pool candidates that curated knowledge vouches for.

    Returns `{technique_id: [label, ...]}`, e.g.
    `{"T1204.004": ["named by the ClickFix lure pattern", "listed in the
    report's own ATT&CK mapping"]}`.

    Without this the pool is a flat `id | name | tactics` list in which a
    brand-injected candidate is indistinguishable from a retriever hit — the
    signal was computed and then hidden from the only party that could act
    on it. Annotating also changes the prompt text, which changes the cache
    key: the pool used to render byte-identically with and without brand
    expansion, so a cached wrong pick could survive the fix meant to correct
    it.
    """
    annotations: dict[str, list[str]] = {}
    for hit in brand_audit:
        brand = hit.get("brand", "")
        for tid in hit.get("techniques") or []:
            label = f"named by the '{brand}' pattern"
            annotations.setdefault(tid, [])
            if label not in annotations[tid]:
                annotations[tid].append(label)
    for tid in sorted(vendor_tids):
        annotations.setdefault(tid, []).append(
            "listed in the report's own ATT&CK mapping",
        )
    return annotations


def _format_reference_text(
    lookup: dict[str, dict],
    annotations: dict[str, list[str]] | None = None,
) -> str:
    """Format the unified candidate pool for the pick prompt.

    Identical line format to `_load_technique_catalogue` and the retrievers
    so the prompt block reads the same regardless of whether the pool came
    from retriever alone or was augmented by LLM proposals — plus a trailing
    bracket on any candidate curated knowledge vouches for (see
    `_build_pool_annotations`).
    """
    if not lookup:
        return ""
    annotations = annotations or {}
    lines = []
    for tid in sorted(lookup):
        entry = lookup[tid]
        name = entry.get("name", "")
        tactics = entry.get("tactics", [])
        tactic_str = ", ".join(tactics) if tactics else "n/a"
        line = f"{tid} | {name} | {tactic_str}"
        if annotations.get(tid):
            line += f"  [{'; '.join(annotations[tid])}]"
        lines.append(line)
    header = (
        "\n\nUNIFIED CANDIDATE POOL (retriever + validated LLM proposals; "
        "revoked/deprecated excluded):\n"
        "Pick technique IDs ONLY from this list. Format: id | name | tactics\n"
    )
    if annotations:
        header += (
            "A trailing [bracket] means an outside source vouches for that "
            "candidate — a known attack-pattern name in the text, or the "
            "report's own ATT&CK mapping table. Treat it as a strong hint "
            "about WHICH technique is meant, especially when choosing between "
            "sub-techniques of the same parent. It is NOT evidence on its "
            "own: you still need a source_quote from the chunk, and if the "
            "chunk gives you nothing to quote, say so rather than reaching "
            "for a neighboring sub-technique.\n"
        )
    return header + "\n".join(lines)


# The source-grounding check now lives in app.services.grounding — the gate
# reviewer needs the same instrument, and any LLM asked to justify a decision
# with evidence can invent that evidence. Re-exported under the original
# private names so this module's callers and tests are unaffected.
_NON_GROUNDING_SECTIONS = NON_GROUNDING_SECTIONS
_build_source_grounding_tokens = build_source_grounding_tokens
_QUOTE_SUPPORT_THRESHOLD = QUOTE_SUPPORT_THRESHOLD
_QUOTE_MIN_TOKENS = QUOTE_MIN_TOKENS


def _apply_source_quote_cap(
    technique_mappings: dict[str, list[dict]],
    chunks_by_id: dict[str, str],
    source_tokens: set[str] | None = None,
) -> dict[str, list[dict]]:
    """Auto-cap picks at 'possible' when the source_quote isn't real evidence.

    Three checks, each downgrading to the 'possible' bucket:
      - Empty source_quote -> the LLM admitted it had no quote.
      - Quote absent from the chunk text -> the LLM quoted nothing at all.
      - Quote present in the chunk but NOT supported by the report itself.

    The third check is the one that matters. The first two verify the quote
    against `chunk.text`, which is *LLM-written prose* — so a chunker
    invention becomes unfalsifiable evidence for a pick. That is exactly how
    `T1204.004` once shipped at 'definite' 0.95 quoting "copying and pasting
    an attacker-supplied command", a phrase that appears zero times in the
    source report.

    The check is deliberately token-support, not substring. Chunk text is a
    rewrite of the source by design, so quotes are verbatim-in-chunk by
    construction and rarely verbatim-in-report; a substring test would flag
    nearly every honest paraphrase and be switched off within a day. Instead
    the quote's distinctive words must have `_QUOTE_SUPPORT_THRESHOLD`
    support in the report. A faithful paraphrase reuses the report's nouns;
    an invention does not.

    `source_tokens` should come from `_build_source_grounding_tokens`, which
    excludes the vendor's own ATT&CK mapping table — otherwise a model could
    quote the answer key and call it evidence. Passing None skips the check
    entirely (the pre-existing two-argument behavior).

    Verbatim_match-provenanced picks are exempt throughout: their grounding
    is the MITRE procedure-example index, not a chunk quote.

    Confidence numeric is also capped at 0.6 when the bucket is downgraded,
    so downstream recalibration doesn't promote a 'possible' back into
    bundle-territory by accident.
    """
    capped = 0
    for chunk_id, techniques in technique_mappings.items():
        chunk_text = chunks_by_id.get(chunk_id, "")
        for t in techniques:
            if t.get("provenance", "").startswith("verbatim_match"):
                continue
            quote = (t.get("source_quote") or "").strip()
            current_bucket = t.get("confidence_bucket", "probable")
            if current_bucket == "possible":
                continue  # already capped
            should_cap = False
            if not quote:
                should_cap = True
                reason = "empty source_quote"
            elif quote not in chunk_text:
                should_cap = True
                reason = "source_quote not verbatim in chunk"
            elif source_tokens:
                quote_tokens = grounding_tokens(quote)
                if len(quote_tokens) >= _QUOTE_MIN_TOKENS:
                    support = len(quote_tokens & source_tokens) / len(quote_tokens)
                    if support < _QUOTE_SUPPORT_THRESHOLD:
                        should_cap = True
                        reason = (
                            f"quote unsupported by report "
                            f"(token support {support:.2f} < "
                            f"{_QUOTE_SUPPORT_THRESHOLD})"
                        )
                        t["quote_unsupported_by_source"] = True
                        t["quote_source_support"] = round(support, 2)
            if should_cap:
                t["confidence_bucket"] = "possible"
                t["confidence"] = min(t.get("confidence", 0.6), 0.6)
                t["bucket_capped"] = reason
                capped += 1
                logger.info(
                    "source_quote cap: chunk %s, %s capped to 'possible' (%s)",
                    chunk_id, t.get("technique_id"), reason,
                )
    if capped:
        logger.info(
            "extract_techniques: source_quote auto-cap downgraded %d picks to 'possible'",
            capped,
        )
    return technique_mappings


_BUCKET_RANK = {"definite": 3, "probable": 2, "possible": 1}


def _dedupe_picks(techniques: list[dict]) -> list[dict]:
    """Collapse repeat picks of the same technique on one chunk.

    A technique can be picked twice for one chunk — most often when the
    verbatim procedure-example matcher fires on two separate example strings
    that both map to it. Two identical entries buy nothing: x_technique_refs
    is a list of refs to the same attack-pattern either way.

    They are not harmless, though. The duplicate is visible at Gate 1, and a
    reviewer that notices spends a recommendation asking for it to be
    dropped — which it can only express as remove_technique_ids=[<id>]. That
    removes BOTH copies, and a procedure with zero techniques hard-fails the
    schema. That is exactly how one run died: T1482 twice, "drop the
    duplicate", nothing left. Collapsing here means the ambiguity never
    reaches the gate.

    The survivor is the strongest entry — highest bucket, then highest
    numeric confidence — so no evidence is traded away for the dedup. Order
    is otherwise preserved: the survivor sits at the first occurrence's
    position, which keeps the bundle's technique order stable across runs.
    """
    best: dict[str, dict] = {}
    order: list[str] = []
    for t in techniques:
        tid = str(t.get("technique_id", "")).strip().upper()
        if not tid:
            continue
        if tid not in best:
            best[tid] = t
            order.append(tid)
            continue
        incumbent = best[tid]
        challenger_rank = (
            _BUCKET_RANK.get(t.get("confidence_bucket", "probable"), 0),
            float(t.get("confidence") or 0.0),
        )
        incumbent_rank = (
            _BUCKET_RANK.get(incumbent.get("confidence_bucket", "probable"), 0),
            float(incumbent.get("confidence") or 0.0),
        )
        if challenger_rank > incumbent_rank:
            best[tid] = t
    if len(order) != len(techniques):
        logger.info(
            "extract_techniques: collapsed %d duplicate technique pick(s)",
            len(techniques) - len(order),
        )
    return [best[tid] for tid in order]


def _split_by_bucket(
    technique_mappings: dict[str, list[dict]],
) -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Split picks into bundle (definite + probable) vs review (possible).

    Returns (bundle_mappings, review_mappings). Both are keyed by chunk_id;
    chunks whose only picks are 'possible' appear only in review_mappings.
    The review lane feeds Gate 1, where the analyst can promote possibles
    into the bundle.
    """
    bundle: dict[str, list[dict]] = {}
    review: dict[str, list[dict]] = {}
    for chunk_id, techniques in technique_mappings.items():
        for t in _dedupe_picks(techniques):
            bucket = t.get("confidence_bucket", "probable")
            if bucket == "possible":
                review.setdefault(chunk_id, []).append(t)
            else:
                bundle.setdefault(chunk_id, []).append(t)
    return bundle, review


def _flag_denylisted_pick(pick: dict, info: dict) -> None:
    """Mark a technique pick as denylisted (analyst-promoted guardrail) so the
    Gate 1 review lane can badge it. Mirrors the entity-tag shape."""
    pick["denylisted"] = True
    pick["denylist_pattern_id"] = info.get("pattern_id")
    pick["denylist_reason"] = (
        f"Matches analyst denylist (pattern {info.get('pattern_id')}): "
        f"{(info.get('pattern') or '')[:160]}"
    )


async def _apply_technique_denylist(
    bundle_mappings: dict[str, list[dict]],
    review_mappings: dict[str, list[dict]],
) -> int:
    """Enforce the technique denylist: a denylisted T-ID never auto-ships, but
    stays visible + promotable.

    - Bundle lane: a denylisted pick is pulled out (it must not auto-ship) and
      DEMOTED into the review lane, flagged ``denylisted``, so the analyst can
      still inspect it and promote it at Gate 1 for a source where it's
      genuinely correct.
    - Review lane: a denylisted pick stays put, just flagged ``denylisted``.

    The denylist is global, so a hard drop would over-reach on the minority of
    sources where the technique is legitimately implied; demote-and-flag keeps
    the guardrail (never auto-included) without removing the analyst's override.
    Mutates both dicts in place. Best-effort (load_denylist no-ops on failure).
    Returns the count of picks flagged/demoted.
    """
    denylist = await load_denylist()
    if not denylist.get("technique_ids"):
        return 0
    affected = 0

    # Review lane: flag in place — keep visible + promotable, don't drop.
    for picks in review_mappings.values():
        for t in picks:
            info = denylist_match_technique(t.get("technique_id", ""), denylist)
            if info:
                _flag_denylisted_pick(t, info)
                affected += 1

    # Bundle lane: pull denylisted picks out (no auto-ship) and demote them
    # into the review lane, flagged, so they remain recoverable for this source.
    for chunk_id in list(bundle_mappings.keys()):
        kept = []
        for t in bundle_mappings[chunk_id]:
            info = denylist_match_technique(t.get("technique_id", ""), denylist)
            if info:
                _flag_denylisted_pick(t, info)
                review_mappings.setdefault(chunk_id, []).append(t)
                affected += 1
                logger.info(
                    "denylist demote->review: %s in chunk %s (pattern %s)",
                    t.get("technique_id"), chunk_id, info["pattern_id"],
                )
            else:
                kept.append(t)
        if kept:
            bundle_mappings[chunk_id] = kept
        else:
            del bundle_mappings[chunk_id]
    return affected


# =============================================================================
# Prompt formatting
# =============================================================================

def _format_chunks_for_prompt(
    chunks: list[dict],
    verbatim_matches_by_chunk: dict[str, list[dict]] | None = None,
    proposals_by_chunk: dict[str, dict] | None = None,
) -> str:
    """Format chunks into a readable text block for the LLM prompt.

    If verbatim_matches_by_chunk is provided, each chunk that has matches
    gets an additional 'DETECTED VERBATIM MATCHES' block listing each
    matched (technique_id, substring, source_actor) for the LLM to
    confirm or reject in its response.

    If proposals_by_chunk is provided (C+A+D pick step), each chunk gets
    a 'PROPOSED BEHAVIOR' block carrying the propose step's behavior
    description + tactic. This anchors the pick step on the LLM's own
    earlier reasoning rather than letting candidate-list bias take over.
    """
    verbatim_matches_by_chunk = verbatim_matches_by_chunk or {}
    proposals_by_chunk = proposals_by_chunk or {}
    parts = []
    for chunk in chunks:
        ctx = chunk.get("context", {})
        ctx_str = ""
        if ctx:
            ctx_parts = []
            if ctx.get("actor"):
                ctx_parts.append(f"Actor: {ctx['actor']}")
            if ctx.get("malware"):
                ctx_parts.append(f"Malware: {', '.join(ctx['malware'])}")
            if ctx.get("tools"):
                ctx_parts.append(f"Tools: {', '.join(ctx['tools'])}")
            if ctx_parts:
                ctx_str = f"\n  Context: {' | '.join(ctx_parts)}"

        cid = chunk.get("chunk_id", "")

        # Build propose-step block (C+A+D) when we have a proposal for this chunk.
        proposal = proposals_by_chunk.get(cid, {})
        propose_str = ""
        if proposal:
            beh = proposal.get("behavior_description", "")
            objective = proposal.get("objective", "")
            # tactics is the list shape; tolerate the legacy singular `tactic`
            # for any in-flight checkpoint that predates the multi-tactic switch.
            tactics = proposal.get("tactics") or (
                [proposal["tactic"]] if proposal.get("tactic") else []
            )
            props = proposal.get("proposed_techniques", [])
            propose_lines = []
            if objective:
                # Objective comes FIRST: it's the discriminator the pick step
                # is meant to anchor on. Keep it visible above the description.
                propose_lines.append(f"  PROPOSED OBJECTIVE: {objective}")
            if beh:
                propose_lines.append(f"  PROPOSED BEHAVIOR (step 1): {beh}")
            if tactics:
                propose_lines.append(
                    f"  PROPOSED TACTICS (procedure may span multiple): "
                    f"{', '.join(tactics)}"
                )
            if props:
                propose_lines.append(
                    f"  PROPOSED TECHNIQUES (step 1, may be filtered): {', '.join(props)}"
                )
            if propose_lines:
                propose_str = "\n" + "\n".join(propose_lines)

        # Build verbatim-match block for this chunk if any matches exist
        chunk_matches = verbatim_matches_by_chunk.get(cid, [])
        match_str = ""
        if chunk_matches:
            match_lines = []
            for m in chunk_matches:
                actor = m.get("source_actor_name", "") or "unknown"
                actor_type = m.get("source_actor_type", "")
                actor_label = f"{actor} ({actor_type})" if actor_type else actor
                match_lines.append(
                    f"  - {m['technique_id']}: matched substring "
                    f"\"{m['matched_substring']}\" (from {actor_label})"
                )
            match_str = (
                "\n  DETECTED VERBATIM MATCHES "
                "(confirm or reject each in your response):\n"
                + "\n".join(match_lines)
            )

        parts.append(
            f"CHUNK [{chunk['chunk_id']}] (seq={chunk.get('sequence_index', '?')}):\n"
            f"  {chunk['text']}{ctx_str}{propose_str}{match_str}\n"
        )
    return "\n".join(parts)


def _format_entity_context(entities: list[dict]) -> str:
    """Summarize entities for technique mapping context."""
    actors = []
    malware = []
    tools = []

    campaigns = []
    intrusion_sets = []
    vulnerabilities = []
    infrastructure = []

    _type_map = {
        EntityType.THREAT_ACTOR.value: actors,
        EntityType.MALWARE.value: malware,
        EntityType.TOOL.value: tools,
        EntityType.CAMPAIGN.value: campaigns,
        EntityType.INTRUSION_SET.value: intrusion_sets,
        EntityType.VULNERABILITY.value: vulnerabilities,
        EntityType.INFRASTRUCTURE.value: infrastructure,
    }

    for e in entities:
        if e.get("gate_action") == GateAction.REMOVE.value:
            continue
        val = e.get("edited_value") or e.get("value", "")
        etype = e.get("entity_type", "")
        target_list = _type_map.get(etype)
        if target_list is not None and val:
            target_list.append(val)

    has_any = any([actors, malware, tools, campaigns, intrusion_sets,
                   vulnerabilities, infrastructure])
    if not has_any:
        return ""

    parts = ["\n\nKNOWN ENTITIES (from earlier pipeline stages):"]
    if intrusion_sets:
        parts.append(f"  Intrusion sets: {', '.join(intrusion_sets)}")
    if actors:
        parts.append(f"  Threat actors: {', '.join(actors)}")
    if malware:
        parts.append(f"  Malware: {', '.join(malware)}")
    if tools:
        parts.append(f"  Tools: {', '.join(tools)}")
    if campaigns:
        parts.append(f"  Campaigns: {', '.join(campaigns)}")
    if vulnerabilities:
        parts.append(f"  Vulnerabilities: {', '.join(vulnerabilities)}")
    if infrastructure:
        parts.append(f"  Infrastructure: {', '.join(infrastructure)}")
    return "\n".join(parts)


# =============================================================================
# Post-processing
# =============================================================================


def _process_technique_mappings(
    raw_mappings: list[dict],
    chunks: list[dict],
    verbatim_matches_by_chunk: dict[str, list[dict]] | None = None,
    technique_lookup: dict[str, dict] | None = None,
) -> dict[str, list[dict]]:
    """Convert raw LLM output to technique_mappings dict.

    If verbatim_matches_by_chunk is provided, processes
    verbatim_match_decisions from the LLM response: confirmed matches
    get locked-in confidence 0.95 with provenance 'verbatim_match';
    rejected matches get 0.3 with the rejection reason and provenance
    'verbatim_match_rejected'. LLM-discovered additional techniques
    (in the 'techniques' array) are appended, excluding any that
    duplicate verbatim-matched IDs.

    technique_lookup is used to pre-fill the tactic and technique_name
    on verbatim entries (first catalogue tactic for the technique).

    Returns:
        {chunk_id: [TechniqueMapping dict, ...]}
    """
    verbatim_matches_by_chunk = verbatim_matches_by_chunk or {}
    technique_lookup = technique_lookup or {}
    valid_chunk_ids = {c["chunk_id"] for c in chunks}
    technique_mappings: dict[str, list[dict]] = {}

    confirmed_count = 0
    rejected_count = 0
    orphan_decisions = 0

    for raw in raw_mappings:
        chunk_id = raw.get("chunk_id", "")
        if chunk_id not in valid_chunk_ids:
            logger.warning(
                "extract_techniques: skipping mapping for unknown chunk '%s'",
                chunk_id,
            )
            continue

        original_matches = verbatim_matches_by_chunk.get(chunk_id, [])
        original_by_tid = {m["technique_id"]: m for m in original_matches}

        techniques: list[dict] = []
        verbatim_tids: set[str] = set()

        raw_decisions = raw.get("verbatim_match_decisions", [])
        if original_matches and not raw_decisions:
            # LLM ignored the REQUIRED instruction and emitted no decisions
            # for a chunk that had detected matches. The matches would
            # otherwise be silently dropped — surface it instead.
            logger.warning(
                "extract_techniques: chunk %s had %d verbatim matches but "
                "LLM emitted no decisions; matches dropped",
                chunk_id, len(original_matches),
            )

        # ----- Verbatim matches: synthesized TechniqueItems with locked confidence
        for decision in raw_decisions:
            tid = decision.get("technique_id", "").strip()
            if not TECHNIQUE_ID_RE.match(tid):
                continue
            original = original_by_tid.get(tid)
            if not original:
                # LLM emitted a decision for a technique that wasn't in the
                # detected matches — defensive skip.
                orphan_decisions += 1
                continue
            verbatim_tids.add(tid)

            cat_entry = technique_lookup.get(tid, {})
            cat_tactics = cat_entry.get("tactics", [])
            default_tactic = cat_tactics[0] if cat_tactics else ""

            decision_value = decision.get("decision", "")
            if decision_value == "confirm":
                # Verbatim confirms lock at 'definite' / 0.95.
                techniques.append({
                    "technique_id": tid,
                    "technique_name": cat_entry.get("name", ""),
                    "tactic": default_tactic,
                    "confidence": 0.95,
                    "confidence_bucket": "definite",
                    "source_quote": original["matched_substring"],
                    "rationale": (
                        f"Verbatim match to MITRE procedure example "
                        f"({original['source_actor_name']}): matched substring "
                        f"\"{original['matched_substring']}\""
                    ),
                    "stix_id": None,
                    "provenance": "verbatim_match",
                    "matched_substring": original["matched_substring"],
                    "source_actor_name": original["source_actor_name"],
                })
                confirmed_count += 1
            elif decision_value == "reject":
                # Rejected verbatims drop to 'possible' / 0.3 — analyst can
                # still see them at Gate 1 review.
                rejection_reason = (decision.get("rejection_reason") or "").strip()
                techniques.append({
                    "technique_id": tid,
                    "technique_name": cat_entry.get("name", ""),
                    "tactic": default_tactic,
                    "confidence": 0.3,
                    "confidence_bucket": "possible",
                    "source_quote": original["matched_substring"],
                    "rationale": (
                        f"Verbatim match REJECTED by LLM: "
                        f"{rejection_reason or 'no reason given'}. "
                        f"Original substring: \"{original['matched_substring']}\" "
                        f"(from {original['source_actor_name']})"
                    ),
                    "stix_id": None,
                    "provenance": "verbatim_match_rejected",
                    "matched_substring": original["matched_substring"],
                    "source_actor_name": original["source_actor_name"],
                    "rejection_reason": rejection_reason,
                })
                rejected_count += 1

        # ----- LLM-discovered additional techniques (the 'techniques' array)
        for t in raw.get("techniques", []):
            technique_id = t.get("technique_id", "").strip()

            if not TECHNIQUE_ID_RE.match(technique_id):
                logger.warning(
                    "extract_techniques: invalid technique ID '%s' for chunk '%s', skipping",
                    technique_id, chunk_id,
                )
                continue

            if technique_id in verbatim_tids:
                # LLM duplicated a verbatim-matched technique; the
                # verbatim entry (locked confidence) already covers it.
                continue

            confidence = max(0.0, min(1.0, float(t.get("confidence", 0.5))))
            # Pick step (C+A+D) emits these as required fields. Defaults
            # land them in the safe lane if the LLM omits them: 'probable'
            # bucket + empty quote (auto-cap will downgrade to 'possible').
            bucket = t.get("confidence_bucket", "probable")
            if bucket not in ("definite", "probable", "possible"):
                bucket = "probable"
            source_quote = (t.get("source_quote") or "").strip()

            techniques.append({
                "technique_id": technique_id,
                "technique_name": t.get("technique_name", ""),
                "tactic": t.get("tactic", ""),
                "confidence": confidence,
                "confidence_bucket": bucket,
                "source_quote": source_quote,
                "rationale": t.get("rationale", ""),
                "stix_id": None,  # Resolved later from catalogue
                "provenance": "llm",
            })

        if techniques:
            technique_mappings[chunk_id] = techniques

    if verbatim_matches_by_chunk:
        logger.info(
            "extract_techniques: verbatim outcomes — %d confirmed, %d rejected, "
            "%d orphan-decisions",
            confirmed_count, rejected_count, orphan_decisions,
        )

    # Coverage reconciliation. The loop above is driven entirely by what the
    # LLM returned, so a chunk the model simply omits never appears — no key,
    # no error, no trace. That is a silent loss in the node that decides what
    # the bundle says about a behavior.
    #
    # On one run the picker answered for 28 of 32 chunks and dropped
    # four, whose propose-step objectives had been correct and specific
    # (Veeam exploitation, Veeam credential harvest, tunneled C2, cloud
    # exfiltration). Nothing said so. The pipeline hard-failed three nodes
    # later on a schema error that pointed at the procedure, not the picker.
    #
    # Note the asymmetry this repairs: a chunk that lost SOME of its
    # techniques this way ships quietly under-mapped and nobody ever finds
    # out. The crash only happened because the loss reached zero. This
    # warning is the only place that can say it happened.
    missing = sorted(valid_chunk_ids - set(technique_mappings))
    if missing:
        logger.warning(
            "extract_techniques: the pick step returned no mapping for %d of "
            "%d chunk(s) — %s. These chunks reached the picker and it did not "
            "answer for them; their behaviors carry no techniques.",
            len(missing), len(valid_chunk_ids), ", ".join(missing),
        )

    return technique_mappings
