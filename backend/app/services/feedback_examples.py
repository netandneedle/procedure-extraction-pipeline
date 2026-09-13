"""Retrieval of past analyst corrections, as demonstrations.

The flywheel's other half (`feedback_patterns`) asks an LLM to write a general
rule from each correction and injects the rule. Two measurements say that step
is where the errors come from: a hand review dropped ~1 rule in 7 as wrong, and
a calibration found 19 of the 25 rules that fired against real output
never once agreed with the analyst. The rules that failed were advisory
generalisations with implicit conditions; the idea was wrong, not the wording.

This module skips the generalisation. It keeps the correction itself and shows
the model the two or three most similar past ones. An example cannot be a wrong
generalisation — it can only be irrelevant, and ranking by relevance is
machinery that already exists here.

Both channels run. Nothing about the rules is removed by this module, and the
two are fetched separately so an ablation arm can vary one without the other.

Best-effort throughout: every public function degrades to "" / [] rather than
raising. A feedback channel must never be the reason a run fails.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from datetime import datetime, timedelta, timezone
from typing import Mapping

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.graph.state import gate_mode, is_gate_enabled
from app.models.base import async_session
from app.models.feedback_example import FeedbackExample
from app.services import pattern_embedding
from app.services.technique_retriever import _extract_anchor_keywords

logger = logging.getLogger(__name__)

# Gate area -> the gates_enabled / gate_modes key that governs it. An example
# is only written when a HUMAN stood at that gate: a disabled gate produces no
# judgement, and a gate left on `auto` produces the AI reviewer's own opinion.
# Every one of the original 63 patterns was synthesized from the latter, which
# is the specific mistake this table exists not to repeat.
_GATE_KEY_FOR_AREA = {
    "entities": "entities",
    "chunks": "chunks",
    "techniques": "procedures",
}

# Which delta bucket feeds which area.
_AREA_FOR_DELTA_KEY = {
    "gate_0_entities": "entities",
    "gate_chunks": "chunks",
    "gate_1_procedures": "techniques",
}

_ACTIVE = "active"
_MAX_RATIONALE = 400
_MAX_CONTEXT = 300


# ---------------------------------------------------------------- building

def _tid_name(t) -> tuple[str, str]:
    if not isinstance(t, dict):
        return "", ""
    return t.get("technique_id") or "", t.get("technique_name") or ""


def _digest(*parts: str) -> str:
    raw = "|".join(p.strip().lower() for p in parts if p)
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _entity_examples(delta: dict) -> list[dict]:
    value = (delta.get("value") or "").strip()
    etype = (delta.get("entity_type") or "").strip()
    if not value:
        return []
    action = delta.get("action") or "remove"
    after: dict = {}
    if delta.get("edited_value") or delta.get("edited_type"):
        after = {"value": delta.get("edited_value") or value,
                 "entity_type": delta.get("edited_type") or etype}
    return [{
        "area": "entities",
        "action": action,
        "before": {"value": value, "entity_type": etype,
                   "confidence": delta.get("llm_confidence")},
        "after": after,
        "rationale": (delta.get("edit_rationale") or "")[:_MAX_RATIONALE],
        "context_snippet": (delta.get("context_snippet") or "")[:_MAX_CONTEXT],
        "applies_to": {"entity_types": [etype] if etype else []},
        # Type + value, so the same false positive from two vendors collapses.
        "dedup_key": f"entities|{action}|{_digest(etype, value)}",
    }]


def _chunk_examples(delta: dict) -> list[dict]:
    kind = delta.get("kind")
    if kind == "wholesale_reject":
        # The whole pass was thrown away. There is no item to demonstrate, and
        # the reason ("missed_procedures") is a complaint about an absence, so
        # the example is the reason itself.
        reason = delta.get("reason") or ""
        comments = (delta.get("comments") or "")[:_MAX_RATIONALE]
        if not reason and not comments:
            return []
        return [{
            "area": "chunks", "action": "reject",
            "before": {"chunking_pass": "rejected wholesale"},
            "after": {}, "rationale": f"{reason}: {comments}".strip(": "),
            "context_snippet": "", "applies_to": {},
            "dedup_key": f"chunks|reject|{_digest(reason, comments)}",
        }]
    if kind == "analyst_added":
        text = (delta.get("text") or "")[:_MAX_CONTEXT]
        if not text:
            return []
        return [{
            "area": "chunks", "action": "add",
            "before": {"chunk": "(the chunker did not produce one here)"},
            "after": {"text": text},
            "rationale": "", "context_snippet": (delta.get("source_excerpt") or "")[:_MAX_CONTEXT],
            "applies_to": {},
            "dedup_key": f"chunks|add|{_digest(text)}",
        }]
    text = (delta.get("original_text") or "")[:_MAX_CONTEXT]
    action = delta.get("action") or ""
    if not text or not action:
        return []
    edits = delta.get("edits") or {}
    return [{
        "area": "chunks", "action": action,
        "before": {"text": text},
        "after": {k: str(v)[:_MAX_CONTEXT] for k, v in edits.items()} if isinstance(edits, dict) else {},
        "rationale": (delta.get("rationale") or "")[:_MAX_RATIONALE],
        "context_snippet": "", "applies_to": {},
        "dedup_key": f"chunks|{action}|{_digest(text)}",
    }]


def _technique_examples(delta: dict) -> list[dict]:
    """One example per corrected technique, not one per draft.

    A draft where the analyst removed two techniques is two demonstrations —
    they are separately transferable, and keying them together would make both
    undiscoverable to a source that matches only one.
    """
    out: list[dict] = []
    if delta.get("kind") == "promotion":
        tid = delta.get("technique_id") or ""
        if not tid:
            return []
        return [{
            "area": "techniques", "action": "promote",
            "before": {"technique": tid,
                       "lane": "held back in the review lane"},
            "after": {"technique": tid, "lane": "included in the bundle"},
            "rationale": "", "context_snippet": "",
            "applies_to": {"technique_ids": [tid]},
            "dedup_key": f"techniques|promote|{_digest(tid)}",
        }]

    name = (delta.get("original_name") or "").strip()
    kept = ", ".join(
        f"{tid}" for tid, _ in (_tid_name(t) for t in delta.get("corrected_techniques") or []) if tid
    )
    rationale = (delta.get("feedback") or "")[:_MAX_RATIONALE]
    reason = delta.get("reject_reason")
    if reason and reason not in rationale:
        rationale = f"{reason}. {rationale}".strip()

    for t in delta.get("rejected_techniques") or []:
        tid, tname = _tid_name(t)
        if not tid:
            continue
        out.append({
            "area": "techniques", "action": "remove",
            "before": {"technique": tid, "technique_name": tname,
                       "on_procedure": name},
            "after": {"kept_techniques": kept} if kept else {},
            "rationale": rationale, "context_snippet": "",
            "applies_to": {"technique_ids": [tid]},
            "dedup_key": f"techniques|remove|{_digest(tid, name)}",
        })
    for t in delta.get("added_techniques") or []:
        tid, tname = _tid_name(t)
        if not tid:
            continue
        out.append({
            "area": "techniques", "action": "add",
            "before": {"on_procedure": name,
                       "technique": "(not picked)"},
            "after": {"technique": tid, "technique_name": tname},
            "rationale": rationale, "context_snippet": "",
            "applies_to": {"technique_ids": [tid]},
            "dedup_key": f"techniques|add|{_digest(tid, name)}",
        })
    # A reject with no per-technique detail is still a demonstration: the whole
    # draft was wrong, and the reason is the lesson.
    if not out and delta.get("action") in ("reject", "edit") and (name or rationale):
        out.append({
            "area": "techniques", "action": delta.get("action"),
            "before": {"procedure": name}, "after": {},
            "rationale": rationale, "context_snippet": "",
            "applies_to": {},
            "dedup_key": f"techniques|{delta.get('action')}|{_digest(name, rationale)}",
        })
    return out


_BUILDER = {
    "entities": _entity_examples,
    "chunks": _chunk_examples,
    "techniques": _technique_examples,
}


def example_rows_from_deltas(state: Mapping, deltas: Mapping) -> list[dict]:
    """Turn this run's analyst deltas into example rows.

    Skips any area whose gate a human did not actually attend. That check is
    the whole reason this table can be trusted where the pattern table could
    not: `_score_surfacings` records UNSCORED for the same condition, and the
    original corpus was built without it.
    """
    rows: list[dict] = []
    seen: set[str] = set()
    title = str(state.get("title") or "")[:256]
    sid = state.get("source_id") or None
    for delta_key, area in _AREA_FOR_DELTA_KEY.items():
        gate_key = _GATE_KEY_FOR_AREA[area]
        if not is_gate_enabled(state, gate_key):
            continue
        if gate_mode(state, gate_key) == "auto":
            continue
        for d in deltas.get(delta_key) or []:
            if not isinstance(d, dict):
                continue
            try:
                built = _BUILDER[area](d)
            except Exception as e:  # noqa: BLE001 — one odd delta must not
                logger.warning("example build failed for %s: %s", area, e)
                continue
            for row in built:
                # Collapse repeats WITHIN one run. The occurrence count is read
                # as "this correction recurred across reports", and a rewind
                # makes the analyst redo the same decision on the same source —
                # One report's T1003.001 promotion appears twice in its deltas for
                # exactly that reason. Counting it twice would claim
                # cross-source evidence that does not exist.
                key = row.get("dedup_key") or ""
                if key and key in seen:
                    continue
                if key:
                    seen.add(key)
                row["source_id"] = sid
                row["source_title"] = title
                rows.append(row)
    return rows


# --------------------------------------------------------------- persisting

def _embedding_text(row: Mapping) -> str:
    """What the example is ABOUT, for retrieval.

    The analyst's rationale and the source context carry most of the
    transferable meaning; the raw before/after values carry the specifics that
    the anchor boosts already handle. Both go in.
    """
    bits = [row.get("area", ""), row.get("action", "")]
    for blob in (row.get("before"), row.get("after")):
        if isinstance(blob, dict):
            bits += [str(v) for v in blob.values() if v]
    bits.append(row.get("rationale") or "")
    bits.append(row.get("context_snippet") or "")
    return " ".join(b for b in bits if b).strip()


async def persist_examples(db: AsyncSession, rows: list[dict]) -> list[FeedbackExample]:
    """Insert, or bump the count on an identical prior correction.

    Exact dedup only. Two near-identical corrections from different reports are
    two pieces of evidence that the error is general, and the occurrence count
    is that signal — collapsing them semantically, as the pattern table does,
    would throw it away.
    """
    out: list[FeedbackExample] = []
    now = datetime.now(timezone.utc)
    for row in rows:
        key = row.get("dedup_key") or ""
        existing = None
        if key:
            existing = (await db.execute(
                select(FeedbackExample).where(FeedbackExample.dedup_key == key)
            )).scalars().first()
        if existing is not None:
            existing.occurrence_count = (existing.occurrence_count or 1) + 1
            existing.last_seen_at = now
            if not existing.rationale and row.get("rationale"):
                existing.rationale = row["rationale"]
            out.append(existing)
            continue
        text = _embedding_text(row)
        vec = pattern_embedding.embed_text(text) if text else None
        sid = row.get("source_id")
        try:
            sid = uuid.UUID(str(sid)) if sid else None
        except (ValueError, TypeError, AttributeError):
            sid = None
        rec = FeedbackExample(
            source_id=sid,
            source_title=row.get("source_title") or "",
            area=row.get("area") or "",
            action=row.get("action") or "",
            before=row.get("before") or {},
            after=row.get("after") or {},
            rationale=row.get("rationale") or "",
            context_snippet=row.get("context_snippet") or "",
            embedding=vec,
            embedding_model=settings.embedding_model if vec else None,
            applies_to=row.get("applies_to") or {},
            dedup_key=key,
            status=_ACTIVE,
            created_at=now,
            last_seen_at=now,
        )
        db.add(rec)
        out.append(rec)
    await db.flush()
    return out


async def record_examples(state: Mapping, deltas: Mapping) -> dict:
    """Build and persist this run's examples. Never raises."""
    try:
        rows = example_rows_from_deltas(state, deltas)
        if not rows:
            return {"built": 0, "persisted": 0}
        async with async_session() as db:
            recs = await persist_examples(db, rows)
            await db.commit()
        clear_example_cache()
        logger.info("feedback examples: recorded %d correction(s)", len(recs))
        return {"built": len(rows), "persisted": len(recs)}
    except Exception as e:  # noqa: BLE001 — the bundle has already shipped
        logger.warning("record_examples failed (non-fatal): %s", e)
        return {"built": 0, "persisted": 0,
                "error": f"{type(e).__name__}: {e}"}


# --------------------------------------------------------------- retrieving

def _example_anchor_keywords(row: FeedbackExample) -> set[str]:
    blob = " ".join([
        str(row.rationale or ""), str(row.context_snippet or ""),
        " ".join(str(v) for v in (row.before or {}).values()),
        " ".join(str(v) for v in (row.after or {}).values()),
    ])
    return _extract_anchor_keywords(blob)


def _score_example(row: FeedbackExample, source_vec, anchors: dict) -> float:
    """Same shape as `_score_pattern`, minus salience.

    There is no hit/miss ledger for examples and there should not be one: an
    example is a record, so "was it right" is not a question about it. Whether
    it was USEFUL is a question about retrieval, which is what the ablation
    arm is for.
    """
    score = 0.0
    if source_vec is not None and row.embedding:
        score += pattern_embedding.cosine(source_vec, row.embedding)
    if anchors.get("keywords") and (_example_anchor_keywords(row) & anchors["keywords"]):
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
    return score


def _render(row: FeedbackExample) -> list[str]:
    def _one(blob: Mapping) -> str:
        # Values are stored whole and truncated only here: an entity value can
        # be a 200-character command line, and six of those would crowd out the
        # task the prompt is actually for.
        return "; ".join(f"{k}={str(v)[:160]}"
                         for k, v in (blob or {}).items() if v not in (None, ""))

    # Titles in this queue carry an em-dash suffix describing why the run was
    # made ("— first run with an analyst at the gates"). That is provenance for
    # the operator, noise in a prompt, and it pushes the actual report name out
    # of the truncation window.
    src = (row.source_title or "an earlier report").split(" — ")[0].strip()
    lines = [f"\n  [{row.area}] {src[:70]}"]
    before = _one(row.before)
    if before:
        lines.append(f"    produced   {before}")
    lines.append(f"    analyst    {row.action}")
    after = _one(row.after)
    if after:
        lines.append(f"    result     {after}")
    if row.rationale:
        lines.append(f'    said       "{row.rationale[:220]}"')
    if (row.occurrence_count or 1) > 1:
        lines.append(f"    (the same correction has been made {row.occurrence_count}x)")
    return lines


def format_examples_for_prompt(rows: list[FeedbackExample]) -> str:
    """Render examples as a system-prompt addendum. "" when there are none."""
    if not rows:
        return ""
    parts = [
        "\n\nPAST ANALYST CORRECTIONS ON SIMILAR REPORTS",
        "  These are records of what a human actually changed, not rules. Judge",
        "  whether the situation in front of you resembles one of them; if it",
        "  does not, they do not apply.",
    ]
    for row in rows:
        parts += _render(row)
    return "\n".join(parts)


async def fetch_examples(
    db: AsyncSession, *, areas: tuple[str, ...],
    statuses: tuple[str, ...] = (_ACTIVE,),
    max_age_days: int | None = None,
    limit: int = 80,
) -> list[FeedbackExample]:
    q = select(FeedbackExample).where(
        FeedbackExample.area.in_(list(areas)),
        FeedbackExample.status.in_(list(statuses)),
    )
    if max_age_days is not None:
        q = q.where(FeedbackExample.last_seen_at
                    >= datetime.now(timezone.utc) - timedelta(days=max_age_days))
    q = q.order_by(FeedbackExample.last_seen_at.desc()).limit(limit)
    return list((await db.execute(q)).scalars().all())


async def relevant_examples(
    state: Mapping, *, areas: tuple[str, ...], limit: int,
) -> tuple[str, list[str]]:
    """Top-`limit` past corrections most relevant to this source.

    Returns (addendum_text, example_ids). Best-effort: ("", []) on any failure.
    """
    try:
        source_vec = pattern_embedding.embed_text(
            pattern_embedding.build_source_representation(state)
        )
        anchors = pattern_embedding.extract_source_anchors(state)
        async with async_session() as db:
            candidates = await fetch_examples(
                db, areas=areas,
                limit=settings.feedback_example_candidate_ceiling,
            )
        if not candidates:
            return "", []
        ranked = sorted(
            candidates,
            key=lambda r: (_score_example(r, source_vec, anchors),
                           r.occurrence_count, r.last_seen_at),
            reverse=True,
        )
        top = ranked[:limit]
        return format_examples_for_prompt(top), [str(r.id) for r in top]
    except Exception as e:  # noqa: BLE001 — never block a run
        logger.warning("relevant_examples failed (proceeding without): %s", e)
        return "", []


_EXAMPLE_CACHE: dict[tuple, tuple[float, str]] = {}
_EXAMPLE_TTL_SECONDS = 60.0


def clear_example_cache() -> None:
    _EXAMPLE_CACHE.clear()


async def relevant_examples_cached(
    state: Mapping, *, areas: tuple[str, ...], node: str, limit: int | None = None,
) -> str:
    """TTL-cached wrapper. Keyed on the source fingerprint, so two sources
    never share a set — the bug the pattern cache shipped with first."""
    if not settings.feedback_examples_enabled:
        return ""
    limit = settings.feedback_examples_limit if limit is None else limit
    # Imported here rather than at module scope: _source_fingerprint is the
    # pattern module's private helper and the two modules are otherwise
    # independent. Sharing the fingerprint matters more than the purity —
    # a second definition would drift.
    from app.services.feedback_patterns import _source_fingerprint

    key = (_source_fingerprint(state), tuple(sorted(areas)), limit)
    now = time.monotonic()
    hit = _EXAMPLE_CACHE.get(key)
    if hit and (now - hit[0]) < _EXAMPLE_TTL_SECONDS:
        return hit[1]
    text, _ids = await relevant_examples(state, areas=areas, limit=limit)
    _EXAMPLE_CACHE[key] = (now, text)
    logger.info("feedback examples: %d chars for node=%s areas=%s",
                len(text), node, ",".join(areas))
    return text
