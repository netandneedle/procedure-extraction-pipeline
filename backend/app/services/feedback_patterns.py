"""Feedback patterns service: persist + fetch helpers for the
FeedbackPattern table.

Responsibilities:
1. `persist_pattern`: upsert-style write. On match: bump occurrence_count +
   last_seen_at. On miss: insert.
2. Retrieval for LLM nodes at prompt-build time: `relevant_addendum_cached`
   (source-relative, hybrid-ranked, TTL-cached) over `fetch_feedback_patterns`
   (the plain query helper: category, status, min_occurrence_count, max age).
3. The denylist guardrail behind promoted_to_denylist, and the management
   surface (list / promote / dismiss / edit).

Dedup is two-tier: exact (category, pattern) text match, then semantic —
embedding cosine above `feedback_dedup_threshold` AND a lexical-overlap floor
within the same category — so paraphrased recurrences merge without a
distinct rule being swallowed.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Mapping

from sqlalchemy import and_, delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.base import async_session
from app.models.feedback_pattern import FeedbackPattern
from app.models.feedback_surfacing import FeedbackPatternSurfacing
from app.services import pattern_embedding
from app.services.technique_retriever import _extract_anchor_keywords
from app.utils.refang import refang
from app.utils.text import TECHNIQUE_ID_RE, token_overlap

logger = logging.getLogger(__name__)


# Statuses consumed by LLM nodes at prompt-build time. Active + promoted_*
# (promoted ones are de-facto rules so the LLM benefits from continuing to see
# them). Used by the deprecated category-dump path + the default fetch.
_CONSUMABLE_STATUSES = ("active", "promoted_to_denylist", "promoted_to_prompt")

# Relevance-first retrieval splits these: the relevance POOL (ranked + top-N
# cut) is active + denylist patterns, while promoted_to_prompt patterns are
# PINNED — always injected for their category, exempt from the top-N cut and
# the age cutoff (an analyst-confirmed permanent rule shouldn't expire just
# because the LLM started obeying it and corrections stopped).
_RELEVANCE_STATUSES = ("active", "promoted_to_denylist")
_PINNED_STATUS = "promoted_to_prompt"
# Bound on pinned rules injected per node, in case promotions pile up. Generous
# — promotion is a deliberate analyst act, so this is a backstop, not a budget.
_MAX_PINNED_PATTERNS = 25


# Dedup-candidate scan cap for the semantic path. The same-category set is
# usually tiny; this just bounds the worst case.
_SEMANTIC_DEDUP_SCAN_LIMIT = 300


def _embedding_text(pattern: str, applies_to: dict | None) -> str:
    """Text we embed for a pattern: the prose plus a compact applies_to render
    so structurally-similar patterns (same techniques/tactics/genre) cluster."""
    applies_to = applies_to or {}
    bits: list[str] = [pattern]
    extra = list(applies_to.get("technique_ids") or [])
    extra += list(applies_to.get("tactics") or [])
    genre = applies_to.get("source_genre")
    if genre:
        extra.append(str(genre))
    if extra:
        bits.append(" ".join(str(x) for x in extra))
    return " ".join(bits)


def _union_applies_to(a: dict | None, b: dict | None) -> dict:
    """Union the list-valued retrieval keys; keep the first non-empty genre."""
    a = a or {}
    b = b or {}
    out: dict = {}
    for key in ("technique_ids", "entity_types", "tactics"):
        merged = list(dict.fromkeys((a.get(key) or []) + (b.get(key) or [])))
        if merged:
            out[key] = merged
    genre = a.get("source_genre") or b.get("source_genre")
    if genre:
        out["source_genre"] = genre
    return out


def _union_concepts(a: list | None, b: list | None) -> list:
    return list(dict.fromkeys((a or []) + (b or [])))


def _enrich_recurrence(
    existing: FeedbackPattern,
    *,
    applies_to: dict | None,
    concepts: list | None,
    embedding: list[float] | None,
    now: datetime,
) -> None:
    """Apply a recurrence to an existing row: bump count, merge structured
    keys, and backfill the embedding if the row was missing one. Reassigns
    JSON attributes (SQLAlchemy doesn't track in-place mutation)."""
    existing.occurrence_count += 1
    existing.last_seen_at = now
    if applies_to:
        existing.applies_to = _union_applies_to(existing.applies_to, applies_to)
    if concepts:
        existing.concepts = _union_concepts(existing.concepts, concepts)
    if embedding is not None and not existing.embedding:
        existing.embedding = embedding
        existing.embedding_model = settings.embedding_model


async def persist_pattern(
    db: AsyncSession,
    *,
    source_id: uuid.UUID | None,
    category: str,
    pattern: str,
    evidence: dict | None = None,
    applies_to: dict | None = None,
    concepts: list | None = None,
) -> tuple[FeedbackPattern, bool]:
    """Upsert a feedback pattern with two-tier dedup.

    1. Exact path: same (category, pattern) text → bump occurrence_count,
       merge structured keys, backfill embedding. Returns (row, False).
    2. Semantic path: on exact-miss, if we can embed the pattern, compare
       against same-category embedded rows; cosine >= the dedup threshold
       treats it as a recurrence (bump + merge). Returns (matched, False).
    3. Otherwise insert a new row (with embedding when available). (row, True).

    Status filter is intentionally omitted on dedup so a previously-dismissed
    pattern's recurrence still bumps the counter (signals it keeps mattering).
    """
    now = datetime.now(timezone.utc)
    embedding = pattern_embedding.embed_text(_embedding_text(pattern, applies_to))

    # --- Tier 1: exact text match within category --------------------------
    stmt = select(FeedbackPattern).where(
        and_(
            FeedbackPattern.category == category,
            FeedbackPattern.pattern == pattern,
        )
    )
    existing = (await db.execute(stmt)).scalar_one_or_none()
    if existing is not None:
        _enrich_recurrence(
            existing, applies_to=applies_to, concepts=concepts,
            embedding=embedding, now=now,
        )
        await db.commit()
        await db.refresh(existing)
        return existing, False

    # --- Tier 2: semantic near-duplicate within category -------------------
    if embedding is not None:
        cand_stmt = (
            select(FeedbackPattern)
            .where(
                and_(
                    FeedbackPattern.category == category,
                    FeedbackPattern.embedding.isnot(None),
                )
            )
            .order_by(FeedbackPattern.occurrence_count.desc())
            .limit(_SEMANTIC_DEDUP_SCAN_LIMIT)
        )
        candidates = list((await db.execute(cand_stmt)).scalars().all())
        best_row: FeedbackPattern | None = None
        best_sim = 0.0
        best_overlap = 0.0
        for cand in candidates:
            sim = pattern_embedding.cosine(embedding, cand.embedding)
            if sim > best_sim:
                best_sim = sim
                best_row = cand
                best_overlap = token_overlap(pattern, cand.pattern or "")
        # Both bars, deliberately. Cosine on rule-shaped prose cannot separate
        # a duplicate from a distinct rule on its own — see the calibration
        # note on feedback_dedup_threshold in config.py. A false merge is the
        # expensive error here: it silently deletes one rule's guidance, while
        # a missed merge only leaves a near-duplicate in the corpus. So the
        # conjunction is biased toward not merging.
        merges = (
            best_row is not None
            and best_sim >= settings.feedback_dedup_threshold
            and best_overlap >= settings.feedback_dedup_lexical_floor
        )
        if best_row is not None and not merges and best_sim >= settings.feedback_dedup_threshold:
            logger.info(
                "persist_pattern: near-miss NOT merged (sim=%.3f, overlap=%.3f "
                "< %.2f) [%s] vs %s",
                best_sim, best_overlap, settings.feedback_dedup_lexical_floor,
                category, best_row.id,
            )
        if merges:
            logger.info(
                "persist_pattern: semantic dedup (sim=%.3f, overlap=%.3f) "
                "[%s] merged into %s",
                best_sim, best_overlap, category, best_row.id,
            )
            _enrich_recurrence(
                best_row, applies_to=applies_to, concepts=concepts,
                embedding=embedding, now=now,
            )
            await db.commit()
            await db.refresh(best_row)
            return best_row, False

    # --- Tier 3: insert new ------------------------------------------------
    row = FeedbackPattern(
        source_id=source_id,
        category=category,
        pattern=pattern,
        evidence=evidence or {},
        applies_to=applies_to or {},
        concepts=concepts or [],
        embedding=embedding,
        embedding_model=settings.embedding_model if embedding is not None else None,
        status="active",
        occurrence_count=1,
        last_seen_at=now,
    )
    db.add(row)
    await db.commit()
    await db.refresh(row)
    return row, True


async def fetch_feedback_patterns(
    db: AsyncSession,
    *,
    categories: list[str] | None = None,
    statuses: tuple[str, ...] = _CONSUMABLE_STATUSES,
    min_occurrence_count: int = 1,
    max_age_days: int | None = 90,
    limit: int = 20,
) -> list[FeedbackPattern]:
    """Fetch recent feedback patterns matching the criteria, ordered by
    occurrence_count desc, then last_seen_at desc.

    Default behavior (no category filter): return up to 20 highest-occurrence
    active patterns from the last 90 days. Suitable for "show me what we've
    learned" UI surfacing. LLM nodes typically pass a categories filter.
    """
    stmt = select(FeedbackPattern).where(
        FeedbackPattern.status.in_(statuses),
        FeedbackPattern.occurrence_count >= min_occurrence_count,
    )
    if categories:
        stmt = stmt.where(FeedbackPattern.category.in_(categories))
    if max_age_days is not None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
        stmt = stmt.where(FeedbackPattern.last_seen_at >= cutoff)

    stmt = stmt.order_by(
        FeedbackPattern.occurrence_count.desc(),
        FeedbackPattern.last_seen_at.desc(),
    ).limit(limit)

    result = await db.execute(stmt)
    return list(result.scalars().all())


def _render_pattern_section(
    header: str, patterns: list[FeedbackPattern], *, with_count: bool,
) -> list[str]:
    """Render one category-grouped block of patterns. ``with_count`` shows the
    occurrence tally (advisory list) vs. omitting it (pinned permanent rules,
    which read as firm rules rather than tallied observations)."""
    by_category: dict[str, list[FeedbackPattern]] = {}
    for p in patterns:
        by_category.setdefault(p.category, []).append(p)
    parts = [header]
    for category in sorted(by_category):
        parts.append(f"\n  [{category}]")
        for p in by_category[category]:
            if with_count:
                parts.append(f"    - (seen {p.occurrence_count}x) {p.pattern}")
            else:
                parts.append(f"    - {p.pattern}")
    return parts


def format_patterns_for_prompt(
    patterns: list[FeedbackPattern],
    pinned: list[FeedbackPattern] | None = None,
) -> str:
    """Render patterns as a system-prompt addendum.

    Used by LLM nodes (entity_extraction, chunk_behaviors, etc.) to prepend
    feedback context. ``pinned`` (promoted_to_prompt rules) render first under
    a PERMANENT RULES header — analyst-confirmed, always-apply guidance, kept
    distinct from the salience-ranked advisory list so the LLM weights them as
    firm rules. Returns "" when there's nothing, so callers can unconditionally
    concatenate.
    """
    pinned = pinned or []
    if not patterns and not pinned:
        return ""

    parts: list[str] = []
    if pinned:
        parts += _render_pattern_section(
            "\n\nPERMANENT RULES (analyst-confirmed — always apply):",
            pinned, with_count=False,
        )
    if patterns:
        parts += _render_pattern_section(
            "\n\nRECENT ANALYST FEEDBACK (apply these corrections in this run):",
            patterns, with_count=True,
        )
    return "\n".join(parts)


async def pinned_rules_addendum(
    db: AsyncSession, *, categories: tuple[str, ...],
) -> str:
    """The analyst's PERMANENT RULES for these categories, prompt-ready.

    Pinned rules ONLY — no relevance-ranked advisory patterns. Written for the
    AI gate reviewer (app.services.reviewer), where that distinction is the
    whole point:

      * A pinned rule is confirmed analyst policy. A reviewer that contradicts
        policy is a bug, and forces the analyst to re-make a correction the
        flywheel exists to capture once.
      * The auto-synthesised advisory patterns are the EXTRACTOR's soft priors.
        Withholding them keeps the reviewer an independent check — able to
        catch the extractor faithfully following a pattern that is itself
        wrong. Sharing them would make the two agree by construction, exactly
        where a second opinion is worth paying for.

    Best-effort: returns "" on any failure, so a review is never blocked by
    the flywheel being unavailable. Same contract as
    ``relevant_addendum_cached``.
    """
    if not categories:
        return ""
    try:
        pinned = await fetch_feedback_patterns(
            db,
            categories=list(categories),
            statuses=(_PINNED_STATUS,),
            max_age_days=None,   # permanent — never age out
            limit=_MAX_PINNED_PATTERNS,
        )
        return format_patterns_for_prompt([], pinned=pinned)
    except Exception:  # noqa: BLE001 — advisory context, never fatal
        logger.warning(
            "pinned_rules_addendum failed for categories %s", categories,
            exc_info=True,
        )
        return ""


# Process-level TTL cache for the FORMATTED addendum, keyed by category set
# + limit. FeedbackPattern only changes when feedback_synthesis runs (after
# a pipeline completes), so within a run the addendum is stable. Without
# this, every LLM node hit postgres independently — entity, chunk,
# technique, drafting = N+1 per run, more on reruns. Best-effort: a
# stale-by-up-to-TTL addendum is
# harmless (it's prompt guidance, not correctness), and a fresh run picks
# up new patterns once the TTL lapses.
_ADDENDUM_CACHE: dict[tuple, tuple[float, str]] = {}
_ADDENDUM_TTL_SECONDS = 60.0


def _pattern_anchor_keywords(row: FeedbackPattern) -> set[str]:
    """CVE/tool anchors present in a pattern's text + structured keys."""
    applies = row.applies_to or {}
    extra = " ".join(
        str(x) for x in (
            list(applies.get("technique_ids") or [])
            + list(applies.get("tactics") or [])
        )
    )
    return _extract_anchor_keywords(f"{row.pattern or ''} {extra}")


def _score_pattern(
    row: FeedbackPattern,
    source_vec: list[float] | None,
    anchors: dict,
) -> float:
    """Hybrid relevance score for a candidate pattern against the source.

    cosine(source, pattern)
      + 0.3  shared CVE/tool lexical anchor
      + 0.5  shared applies_to.technique_id (strong: the run maps that T-ID)
      + 0.2  shared entity_type
      + 0.2  shared tactic
      + w * salience            (NULL salience contributes 0)
    Boost magnitudes mirror technique_retriever's anchors, sized for cosine's
    tighter [~0.2, ~0.7] range. Patterns with NULL embedding score lexical-only.
    """
    score = 0.0
    if source_vec is not None and row.embedding:
        score += pattern_embedding.cosine(source_vec, row.embedding)

    if anchors.get("keywords") and (_pattern_anchor_keywords(row) & anchors["keywords"]):
        score += 0.3

    applies = row.applies_to or {}
    if anchors.get("technique_ids") and (
        set(applies.get("technique_ids") or []) & anchors["technique_ids"]
    ):
        score += 0.5
    if anchors.get("entity_types") and (
        set(applies.get("entity_types") or []) & anchors["entity_types"]
    ):
        score += 0.2
    if anchors.get("tactics") and (
        set(applies.get("tactics") or []) & anchors["tactics"]
    ):
        score += 0.2

    score += settings.feedback_salience_weight * (row.salience or 0.0)
    return score


async def relevant_feedback_addendum(
    state: Mapping,
    *,
    categories: tuple[str, ...],
    limit: int = 15,
) -> tuple[str, list[str]]:
    """Hybrid, source-relative feedback addendum.

    Two streams, both category-scoped:
    - Relevance pool (active + denylist): pre-filtered by category/status/recency
      (up to ``feedback_candidate_ceiling``), ranked against an embedding of the
      current source, top-``limit`` kept.
    - Pinned (promoted_to_prompt): analyst-confirmed permanent rules — ALWAYS
      included, exempt from the top-N cut and the age cutoff, rendered under a
      PERMANENT RULES header.

    Returns ``(addendum_text, surfaced_pattern_ids)``. ``surfaced_pattern_ids``
    (pinned + top) is consumed by the closed-loop ledger. Best-effort:
    returns ``("", [])`` on any failure.
    """
    try:
        source_vec = pattern_embedding.embed_text(
            pattern_embedding.build_source_representation(state)
        )
        anchors = pattern_embedding.extract_source_anchors(state)
        async with async_session() as db:
            candidates = await fetch_feedback_patterns(
                db,
                categories=list(categories),
                statuses=_RELEVANCE_STATUSES,
                limit=settings.feedback_candidate_ceiling,
            )
            pinned = await fetch_feedback_patterns(
                db,
                categories=list(categories),
                statuses=(_PINNED_STATUS,),
                max_age_days=None,  # permanent — never age out
                limit=_MAX_PINNED_PATTERNS,
            )
            if not candidates and not pinned:
                return "", []
            ranked = sorted(
                candidates,
                key=lambda r: (
                    _score_pattern(r, source_vec, anchors),
                    r.occurrence_count,
                    r.last_seen_at,
                ),
                reverse=True,
            )
            top = ranked[:limit]
            text = format_patterns_for_prompt(top, pinned=pinned)
            # Pinned rules are surfaced too, so the closed loop still scores
            # them hit/miss (a permanent rule that keeps getting re-corrected
            # is signal its phrasing needs work).
            surfaced_ids = [str(r.id) for r in pinned] + [str(r.id) for r in top]
        return text, surfaced_ids
    except Exception as e:  # noqa: BLE001 — best-effort, never block the run
        logger.warning(
            "relevant_feedback_addendum failed (proceeding without): %s", e,
        )
        return "", []


# Source-relative addendum cache. Keyed by (source_fingerprint, categories,
# limit) so different sources get different relevant sets — a (categories,
# limit) key alone would serve the first source's set to all. Within a
# node's invocation the inputs are stable, so a rerun/resume of the same
# node hits the cache rather than re-querying postgres. Stores
# (timestamp, (text, surfaced_ids)).
_RELEVANT_CACHE: dict[tuple, tuple[float, tuple[str, list[str]]]] = {}
_RELEVANT_TTL_SECONDS = 60.0


def _source_fingerprint(state: Mapping) -> str:
    """Stable-per-source, distinct-across-sources hash for the cache key.

    Backbone is the parsed_text head (identical across a run's nodes, unique
    per source, present at every stage); title + entity values + technique IDs
    add distinction. A change to any of these (e.g. a rerun after more entities
    were validated) correctly busts the entry.
    """
    title = state.get("title") or (state.get("metadata") or {}).get("title") or ""
    head = (state.get("parsed_text") or "")[:2000]
    ents = state.get("validated_entities") or state.get("entities") or []
    vals = sorted(str((e or {}).get("value", "")) for e in ents if (e or {}).get("value"))
    tids = sorted(pattern_embedding._technique_ids_in_play(state))
    raw = "||".join([str(title), head, *vals, *tids])
    return hashlib.sha1(raw.encode("utf-8", "ignore")).hexdigest()[:16]


async def relevant_addendum_cached(
    state: Mapping,
    *,
    categories: tuple[str, ...],
    node: str,
    limit: int = 15,
) -> str:
    """TTL-cached wrapper over ``relevant_feedback_addendum``.

    Returns just the addendum text (the node concatenates it onto its system
    prompt). ``node`` identifies the calling node for the surfacing ledger
    (recorded on every call — hit OR miss — so all 4 nodes log).

    ``node`` is also part of the CACHE KEY, and that is load-bearing rather
    than tidy. `extract_techniques` fetches twice — once before its propose
    step and once after — precisely so the second fetch sees the T-IDs the
    first could not. Keying on the fingerprint alone made that correctness
    depend on the propose step happening to surface a technique the vendor's
    own mapping table had not already named; when it did not, the fingerprint
    was unchanged, the second fetch hit this cache, and the fix silently became
    a no-op that still looked applied.
    """
    key = (_source_fingerprint(state), tuple(sorted(categories)), limit, node)
    now = time.monotonic()
    hit = _RELEVANT_CACHE.get(key)
    if hit is not None and now - hit[0] < _RELEVANT_TTL_SECONDS:
        # Cache hit (e.g. a rerun of the same node within the TTL). The
        # surfacing for this (source, node) was already recorded on the miss
        # that populated the cache; recording again would double-count, and
        # the synthesis scorer dedups by pattern_id anyway.
        return hit[1][0]

    text, surfaced_ids = await relevant_feedback_addendum(
        state, categories=categories, limit=limit,
    )
    _RELEVANT_CACHE[key] = (now, (text, surfaced_ids))
    # Record the surfacing ledger (closed loop). Best-effort, on the miss
    # path only. Every distinct `node` label has its own cache key, so each
    # fetch point misses on its first call and all of them get logged.
    await _record_surfacings(state, node=node, surfaced_ids=surfaced_ids)
    return text


async def _record_surfacings(
    state: Mapping, *, node: str, surfaced_ids: list[str],
) -> None:
    """Write one FeedbackPatternSurfacing row per surfaced pattern.

    Best-effort: skips when there's no source_id to attribute to or nothing
    was surfaced, and swallows any DB error so it never blocks the run.
    """
    if not surfaced_ids:
        return
    source_id_str = state.get("source_id") or ""
    try:
        source_uuid = uuid.UUID(source_id_str) if source_id_str else None
    except (ValueError, AttributeError, TypeError):
        source_uuid = None
    if source_uuid is None:
        return  # can't attribute the outcome to a run
    try:
        async with async_session() as db:
            for pid in surfaced_ids:
                try:
                    pattern_uuid = uuid.UUID(str(pid))
                except (ValueError, AttributeError, TypeError):
                    continue
                db.add(FeedbackPatternSurfacing(
                    pattern_id=pattern_uuid,
                    source_id=source_uuid,
                    node=node,
                ))
            await db.commit()
    except Exception as e:  # noqa: BLE001 — best-effort, never block the run
        logger.warning("_record_surfacings failed (proceeding): %s", e)


async def delete_surfacings_for_source(db: AsyncSession, source_id) -> int:
    """Remove the surfacing ledger rows attributed to one source.

    ``source_id`` is a plain UUID column, not a foreign key, so deleting a
    source leaves its surfacings behind — at one point 131 of the 140 source
    ids in the ledger pointed at sources that no longer existed. Those
    rows cannot be joined back to anything: their per-node breakdown and
    their outcome are unattributable, and they inflate every count taken
    over the table.

    The pattern counters are NOT rewound. They are the durable aggregate and
    the evidence was real when it was recorded; only the row-level detail,
    which is now unjoinable, goes away.

    Returns the number of rows removed.
    """
    result = await db.execute(
        delete(FeedbackPatternSurfacing).where(
            FeedbackPatternSurfacing.source_id == source_id
        )
    )
    return int(result.rowcount or 0)


def clear_feedback_cache() -> None:
    """Force-evict both addendum caches.

    Called by the promote/dismiss/edit routes so analyst actions
    reflect in prompts without waiting for the 60s TTL to lapse.
    """
    _ADDENDUM_CACHE.clear()
    _RELEVANT_CACHE.clear()
    _DENYLIST_CACHE.clear()


# =============================================================================
# Salience (closed loop): does the pattern actually help?
# =============================================================================

def compute_salience(
    *,
    hit_count: int,
    miss_count: int,
    occurrence_count: int,
    last_seen_at: datetime | None,
    now: datetime | None = None,
) -> float:
    """Composite usefulness score: hit-rate x recency decay x occurrence weight.

        hit_rate = (hit + alpha) / (hit + miss + alpha + beta)   # mild prior
        recency  = 0.5 ** (age_days / half_life)
        occ_w    = log(1 + occurrence_count)
        salience = hit_rate * recency * occ_w

    A MISS means the pattern is still relevant but INEFFECTIVE AS PHRASED — not
    useless. So a miss-heavy pattern's salience drops (it becomes a demote /
    rephrase candidate) rather than being deleted. The alpha/beta prior biases
    fresh, never-scored patterns toward "assume mildly useful" so they get a
    fair chance to be surfaced before the loop has any signal on them.
    """
    alpha = settings.feedback_salience_alpha
    beta = settings.feedback_salience_beta
    hit_rate = (hit_count + alpha) / (hit_count + miss_count + alpha + beta)

    if last_seen_at is not None:
        ref = now or datetime.now(timezone.utc)
        last_seen = (
            last_seen_at.replace(tzinfo=timezone.utc)
            if last_seen_at.tzinfo is None else last_seen_at
        )
        age_days = max(0.0, (ref - last_seen).total_seconds() / 86400.0)
        half_life = max(1e-6, settings.feedback_salience_half_life_days)
        recency = 0.5 ** (age_days / half_life)
    else:
        recency = 1.0

    occ_w = math.log(1 + max(0, occurrence_count))
    return hit_rate * recency * occ_w


# =============================================================================
# Denylist (deterministic guardrail): the teeth behind promoted_to_denylist
# =============================================================================
#
# A pattern promoted to 'denylist' carries concrete terms — literal entity
# values and/or ATT&CK technique IDs — that future runs enforce
# DETERMINISTICALLY (not as LLM-advisory prompt text). extract_entities tags
# matching entities and gate_0 auto-removes them; extract_techniques drops
# denylisted T-ID picks. Loaded once per run per node, TTL-cached globally
# (the denylist is source-independent, unlike the relevance addendum).

_DENYLIST_STATUS = "promoted_to_denylist"
_MAX_DENYLIST_ITEMS = 200
_MAX_DENYLIST_VALUE_LEN = 256


def normalize_denylist_terms(raw: Mapping | None) -> dict:
    """Coerce/clean a {values, technique_ids, entity_types} dict into shape.

    - values: stripped, non-empty, <= 256 chars, control-char-free,
      deduped case-insensitively (original casing kept for display).
    - technique_ids: uppercased, validated against T#### / T####.### shape,
      deduped.
    - entity_types: lowercased, deduped — scope the `values` matches to these
      entity types. Empty => a value matches any entity type (the default).
    Anything malformed is dropped silently — this is defense-in-depth behind
    the Pydantic schema, and is also called directly by tests.
    """
    raw = raw or {}
    values: list[str] = []
    seen_v: set[str] = set()
    for v in list(raw.get("values") or [])[:_MAX_DENYLIST_ITEMS]:
        s = str(v).strip()
        if not s or len(s) > _MAX_DENYLIST_VALUE_LEN:
            continue
        if any(ord(c) < 0x20 or ord(c) == 0x7F for c in s):
            continue
        key = s.lower()
        if key in seen_v:
            continue
        seen_v.add(key)
        values.append(s)

    tids: list[str] = []
    seen_t: set[str] = set()
    for t in list(raw.get("technique_ids") or [])[:_MAX_DENYLIST_ITEMS]:
        s = str(t).strip().upper()
        if not TECHNIQUE_ID_RE.match(s) or s in seen_t:
            continue
        seen_t.add(s)
        tids.append(s)

    etypes: list[str] = []
    seen_e: set[str] = set()
    for e in list(raw.get("entity_types") or [])[:_MAX_DENYLIST_ITEMS]:
        s = str(e).strip().lower()
        if not s or len(s) > 64 or s in seen_e:
            continue
        if any(ord(c) < 0x20 or ord(c) == 0x7F for c in s):
            continue
        seen_e.add(s)
        etypes.append(s)

    return {"values": values, "technique_ids": tids, "entity_types": etypes}


# Global TTL cache for the aggregated denylist. Keyed by a single constant
# (the denylist is the same for every source). Cleared by clear_feedback_cache
# on promote/dismiss/edit so changes take effect without waiting out the TTL.
_DENYLIST_CACHE: dict[str, tuple[float, dict]] = {}
_DENYLIST_CACHE_KEY = "denylist"
_DENYLIST_TTL_SECONDS = 60.0


async def load_denylist() -> dict:
    """Aggregate every promoted_to_denylist pattern's terms into a lookup.

    Returns ``{"values": {value_lower: info}, "technique_ids": {TID: info}}``
    where ``info`` is ``{pattern_id, pattern, category}`` for the rationale
    shown to the analyst. TTL-cached, best-effort: returns a FRESH empty-maps
    dict on any error so callers can enforce unconditionally without guarding.

    The error result is deliberately NOT cached: a transient DB blip retries
    on the next call so enforcement resumes the instant the DB recovers
    (fail-open-but-recover-fast), rather than being suppressed for a full TTL.
    A persistent outage that breaks this query also breaks the Postgres
    checkpointer/queue, so the extra reads are moot.
    """
    now = time.monotonic()
    hit = _DENYLIST_CACHE.get(_DENYLIST_CACHE_KEY)
    if hit is not None and now - hit[0] < _DENYLIST_TTL_SECONDS:
        return hit[1]
    try:
        async with async_session() as db:
            rows = list((await db.execute(
                select(FeedbackPattern).where(
                    FeedbackPattern.status == _DENYLIST_STATUS
                )
            )).scalars().all())
    except Exception as e:  # noqa: BLE001 — best-effort, never block the run
        logger.warning("load_denylist failed (proceeding without): %s", e)
        # Fresh dict, never the cached/shared one — a caller mutating the
        # result must not corrupt a process-global sentinel.
        return {"values": {}, "technique_ids": {}}

    values: dict[str, dict] = {}
    tids: dict[str, dict] = {}
    for r in rows:
        terms = normalize_denylist_terms(r.denylist_terms or {})
        info = {"pattern_id": str(r.id), "pattern": r.pattern, "category": r.category}
        # entity_types scope for THIS pattern's values. None => match any type.
        pat_scope = set(terms["entity_types"]) or None
        for v in terms["values"]:
            # Refang the key so a denylist entered in defanged form
            # ("1[.]2[.]3[.]4", "evil[.]com") matches the refanged entity
            # value that extract_entities produces. Idempotent on clean values.
            key = refang(v).lower()
            existing = values.get(key)
            if existing is None:
                values[key] = {
                    **info,
                    "entity_types": set(pat_scope) if pat_scope else None,
                }
            elif existing["entity_types"] is None or pat_scope is None:
                # Any unscoped contributor widens the value to all types.
                existing["entity_types"] = None
            else:
                existing["entity_types"] |= pat_scope
        for t in terms["technique_ids"]:
            tids.setdefault(t, info)

    result = {"values": values, "technique_ids": tids}
    _DENYLIST_CACHE[_DENYLIST_CACHE_KEY] = (now, result)
    return result


def denylist_match_value(
    value: str, denylist: Mapping, entity_type: str | None = None,
) -> dict | None:
    """Return the blocking pattern info if ``value`` is denylisted, else None.

    Match is case-insensitive on the trimmed value (exact, not substring —
    deterministic and safe; substring over-blocks). The query is refanged so a
    defanged input matches the refanged keys (load_denylist refangs both sides).

    Type scoping: if the matched entry carries an ``entity_types`` scope, the
    value only blocks entities of those types — so denylisting "google" as a
    noisy org won't drop a google.com domain. An unscoped entry (the default)
    matches any type. A scoped entry with no ``entity_type`` supplied does NOT
    match (conservative — we won't remove what we can't verify).
    """
    if not value:
        return None
    entry = (denylist.get("values") or {}).get(refang(str(value).strip()).lower())
    if entry is None:
        return None
    allowed = entry.get("entity_types")  # None/empty => any type
    if not allowed:
        return entry
    if entity_type and str(entity_type).strip().lower() in allowed:
        return entry
    return None


def denylist_match_technique(technique_id: str, denylist: Mapping) -> dict | None:
    """Return the blocking pattern info if ``technique_id`` is denylisted."""
    if not technique_id:
        return None
    return (denylist.get("technique_ids") or {}).get(str(technique_id).strip().upper())


# =============================================================================
# Management surface: list / promote / dismiss / edit
# =============================================================================

# action keyword -> the status the pattern transitions to.
_PROMOTE_ACTIONS = {
    "prompt": "promoted_to_prompt",
    "denylist": "promoted_to_denylist",
}


async def list_patterns(
    db: AsyncSession,
    *,
    category: str | None = None,
    status: str | None = None,
    min_salience: float | None = None,
    search: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[FeedbackPattern], int]:
    """Paginated listing for the management UI. All filters optional.

    Ordered by salience desc (NULLs last), then occurrence, then recency —
    most-useful-first. Returns (rows, total_matching).
    """
    conds = []
    if category:
        conds.append(FeedbackPattern.category == category)
    if status:
        conds.append(FeedbackPattern.status == status)
    if min_salience is not None:
        conds.append(FeedbackPattern.salience >= min_salience)
    if search:
        conds.append(FeedbackPattern.pattern.ilike(f"%{search}%"))

    base = select(FeedbackPattern)
    count_q = select(func.count()).select_from(FeedbackPattern)
    if conds:
        base = base.where(and_(*conds))
        count_q = count_q.where(and_(*conds))

    total = (await db.execute(count_q)).scalar_one()
    rows = list((await db.execute(
        base.order_by(
            FeedbackPattern.salience.desc().nullslast(),
            FeedbackPattern.occurrence_count.desc(),
            FeedbackPattern.last_seen_at.desc(),
        ).limit(limit).offset(offset)
    )).scalars().all())
    return rows, total


async def _get_pattern(db: AsyncSession, pattern_id: uuid.UUID) -> FeedbackPattern | None:
    return (
        await db.execute(select(FeedbackPattern).where(FeedbackPattern.id == pattern_id))
    ).scalar_one_or_none()


async def promote_pattern(
    db: AsyncSession,
    pattern_id: uuid.UUID,
    *,
    action: str,
    by: str,
    denylist_terms: Mapping | None = None,
) -> FeedbackPattern | None:
    """Promote a pattern to a permanent guardrail, wiring the promoted_* fields.

    ``action`` is 'prompt' or 'denylist'. For 'denylist', ``denylist_terms``
    ({values, technique_ids}, analyst-confirmed) become the deterministic
    guardrail enforced by extract_entities / gate_0 / extract_techniques. A
    denylist promotion with no usable terms is allowed but logs a warning —
    it then behaves like 'prompt' (advisory only, nothing enforced).
    """
    target = _PROMOTE_ACTIONS.get(action)
    if target is None:
        raise ValueError(f"unknown promote action: {action!r}")
    row = await _get_pattern(db, pattern_id)
    if row is None:
        return None
    row.status = target
    row.promoted_at = datetime.now(timezone.utc)
    row.promoted_by = by

    if action == "denylist":
        terms = normalize_denylist_terms(denylist_terms)
        row.denylist_terms = terms
        enforced = terms["values"] + terms["technique_ids"]
        if enforced:
            # Human-readable audit of what's actually blocked (col is 255).
            row.promoted_action = ("denylist:" + ", ".join(enforced))[:255]
        else:
            row.promoted_action = "denylist:(no enforceable terms — advisory only)"
            logger.warning(
                "promote_pattern: denylist promotion of %s has no enforceable "
                "terms; it will be advisory only", pattern_id,
            )
    else:
        # 'prompt' promotions don't carry denylist terms; clear any stale ones.
        row.denylist_terms = {}
        row.promoted_action = f"{action}:{row.category}"

    await db.commit()
    await db.refresh(row)
    clear_feedback_cache()
    return row


async def dismiss_pattern(db: AsyncSession, pattern_id: uuid.UUID) -> FeedbackPattern | None:
    """Mark a pattern dismissed (no longer consumed at prompt-build time)."""
    row = await _get_pattern(db, pattern_id)
    if row is None:
        return None
    row.status = "dismissed"
    await db.commit()
    await db.refresh(row)
    clear_feedback_cache()
    return row


async def update_pattern(
    db: AsyncSession,
    pattern_id: uuid.UUID,
    *,
    pattern: str | None = None,
    category: str | None = None,
    applies_to: dict | None = None,
    concepts: list | None = None,
) -> FeedbackPattern | None:
    """Edit a pattern's text / category / structured keys. Re-embeds when the
    text or applies_to changed so semantic ranking stays consistent."""
    row = await _get_pattern(db, pattern_id)
    if row is None:
        return None
    changed_text = False
    if pattern is not None:
        row.pattern = pattern
        changed_text = True
    if category is not None:
        row.category = category
    if applies_to is not None:
        row.applies_to = applies_to
        changed_text = True
    if concepts is not None:
        row.concepts = concepts
    if changed_text:
        vec = pattern_embedding.embed_text(_embedding_text(row.pattern, row.applies_to))
        if vec is not None:
            row.embedding = vec
            row.embedding_model = settings.embedding_model
    await db.commit()
    await db.refresh(row)
    clear_feedback_cache()
    return row


def pattern_to_dict(row: FeedbackPattern) -> dict:
    """Serialize a FeedbackPattern row for the management API (no embedding —
    it's a large float array of no use to the UI)."""
    return {
        "id": str(row.id),
        "source_id": str(row.source_id) if row.source_id else None,
        "category": row.category,
        "pattern": row.pattern,
        "status": row.status,
        "occurrence_count": row.occurrence_count,
        "hit_count": row.hit_count or 0,
        "miss_count": row.miss_count or 0,
        "salience": row.salience,
        "applies_to": row.applies_to or {},
        "concepts": row.concepts or [],
        "evidence": row.evidence or {},
        "denylist_terms": row.denylist_terms or {},
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "last_seen_at": row.last_seen_at.isoformat() if row.last_seen_at else None,
        "last_scored_at": row.last_scored_at.isoformat() if row.last_scored_at else None,
        "promoted_at": row.promoted_at.isoformat() if row.promoted_at else None,
        "promoted_by": row.promoted_by,
        "promoted_action": row.promoted_action,
    }
