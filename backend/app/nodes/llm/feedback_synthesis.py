"""synthesize_feedback node: post-distribute analyst-feedback synthesis.

WHY THIS NODE EXISTS:
The pipeline already captures rich per-source analyst feedback at every
gate (rationales, reject_reason enums, edited fields, promotions). But
that feedback dies with the source — Gate 0 rationales become entity
metadata, Gate 1 reject reasons feed a one-shot retry, Gate 2 decisions
become relationship metadata. Nothing aggregates across sources, nothing
updates prompts, nothing builds heuristic guardrails. The next similar
report comes in and the LLM rediscovers every false positive from scratch.

This node turns each pipeline run into structured training signal. It:
1. Reads the LLM-output ↔ analyst-decision deltas across all four gates
   (entities, chunks, procedures + techniques, relationships)
2. Asks the extraction model to extract structured patterns from the deltas
3. Persists each pattern to the FeedbackPattern table (deduped by
   category + pattern text, then semantically; recurrence bumps
   occurrence_count)

The patterns then feed back into future runs via relevant_addendum_cached,
called at LLM-prompt-build time in upstream nodes.

PLACEMENT:
This node runs AFTER distribute (so the bundle has shipped before the
synthesizer touches anything; failures here can't block delivery).
Conditional skip: when no analyst review actually happened (every key in
the per-gate gates_enabled dict False), we have no signal to synthesize.

READS: entities, validated_entities, chunks, chunk_decisions,
       gate1_correction_log (durable; falls back to gate1_decisions),
       gate1_promotions, gate2_reviews, drafts,
       technique_mappings_for_review, source_id
WRITES: status, current_node, feedback_synthesis, feedback_pattern_ids
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy import and_, select

from app.config import settings
from app.graph.state import (
    GateAction,
    PipelineState,
    PipelineStatus,
    gate_mode,
    is_gate_enabled,
)
from app.models.base import async_session
from app.models.feedback_pattern import FeedbackPattern
from app.models.feedback_surfacing import FeedbackPatternSurfacing
from app.nodes.llm.llm_adapter import call_llm
from app.nodes.llm.tool_models import (
    AttributeCorrectionsOutput,
    SynthesizeFeedbackOutput,
)
from app.services import pattern_embedding
from app.services.feedback_examples import record_examples
from app.services.feedback_patterns import compute_salience, persist_pattern
from app.utils.text import join_technique_ids

logger = logging.getLogger(__name__)


# =============================================================================
# Tool definition
# =============================================================================

SYNTHESIZE_FEEDBACK_TOOL = {
    "name": "synthesize_feedback",
    "description": (
        "Read the analyst-decision deltas from a completed pipeline run "
        "and emit structured feedback patterns. Each pattern captures a "
        "generalizable lesson: a category of LLM output the analyst "
        "consistently corrected. Patterns are persisted to a knowledge "
        "base and consulted on future runs to prevent the same "
        "false-positive / mis-classification class from recurring."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "patterns": {
                "type": "array",
                "description": (
                    "Patterns extracted from this run's analyst decisions. "
                    "Empty when the analyst had no notable disagreements "
                    "with the LLM output. Bias toward FEW HIGH-QUALITY "
                    "patterns rather than many noisy ones."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": [
                                "defender_ioc", "brand_as_malware",
                                "false_positive_entity", "over_chunked",
                                "under_chunked", "missing_tactic",
                                "missing_procedure", "wrong_predecessor",
                                "parallel_capability_misordered",
                                "mis_attribution", "wrong_technique",
                                "thin_initial_access", "orphan_ioc",
                                "artifact_loss", "wrong_relationship",
                                "other",
                            ],
                            "description": "Pattern category.",
                        },
                        "pattern": {
                            "type": "string",
                            "minLength": 10,
                            "maxLength": 500,
                            "description": (
                                "Short, searchable description of the "
                                "pattern. Phrase as a generalizable rule, "
                                "NOT a one-off observation. "
                                "GOOD: 'Email addresses extracted from "
                                "figures classified as letterheads/contact "
                                "cards are defender contact info, not "
                                "adversary IoCs.' "
                                "BAD: 'info@cert.example was rejected as IoC.'"
                            ),
                        },
                        "evidence": {
                            "type": "object",
                            "description": (
                                "Per-category evidence dict. Free-form "
                                "JSON; include the specific instance that "
                                "triggered the pattern (entity_id, chunk_id, "
                                "rationale text, etc.) for audit."
                            ),
                        },
                        "confidence": {
                            "type": "number",
                            "minimum": 0.0,
                            "maximum": 1.0,
                            "description": (
                                "Confidence the pattern is real and "
                                "generalizable. >0.7 = clear signal; "
                                "<0.5 = likely one-off noise (consumers "
                                "filter these out)."
                            ),
                        },
                        "applies_to": {
                            "type": "object",
                            "description": (
                                "Structured retrieval keys so future runs "
                                "surface this pattern ONLY for relevant "
                                "sources. Derive from the deltas. Omit a key "
                                "if not applicable; do NOT invent values."
                            ),
                            "properties": {
                                "technique_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "ATT&CK T-IDs this pattern concerns, e.g. ['T1059.001'].",
                                },
                                "entity_types": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Entity types involved, e.g. ['malware','tool','ipv4'].",
                                },
                                "tactics": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "ATT&CK tactic shortnames, e.g. ['execution','impact'].",
                                },
                                "source_genre": {
                                    "type": "string",
                                    "enum": [
                                        "incident_report", "actor_profile",
                                        "malware_analysis", "campaign_writeup",
                                        "other",
                                    ],
                                    "description": "Kind of source this pattern is most relevant to.",
                                },
                            },
                        },
                        "concepts": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Short concept tags for search/display, e.g. "
                                "['letterhead', 'figure-extraction', 'social-engineering']."
                            ),
                        },
                    },
                    "required": ["category", "pattern", "evidence", "confidence"],
                },
            },
        },
        "required": ["patterns"],
    },
}


SYSTEM_PROMPT = """You are a quality-feedback synthesizer for a CTI extraction pipeline.

YOUR TASK:
Read the analyst's decisions from a completed pipeline run and identify GENERALIZABLE PATTERNS the LLM stages got wrong (or right against early skepticism). Emit structured patterns via the synthesize_feedback tool. The patterns persist into a knowledge base that future runs consult at prompt-build time, so a recurring false-positive class gets fixed once instead of relitigated every run.

WHAT YOU RECEIVE:
A structured digest of the pipeline run, organized by gate:
- Gate 0 (entities): LLM-extracted entities + which the analyst rejected/edited + their rationales.
- Gate 1 (chunks): LLM-extracted chunks + which the analyst dropped/edited/added + reject reasons.
- Gate 1 (procedures+techniques): LLM-drafted procedures + analyst edits/promotions/rejections + reject_reason enum values.
- Gate 2 (relationships): LLM-emitted relationships + analyst removes/adds/edits.

WHAT MAKES A GOOD PATTERN:
- GENERALIZABLE: phrased as a rule that applies beyond this one source.
- CATEGORY-SCOPED: maps to a specific category (defender_ioc, over_chunked, etc.) so future LLM nodes can fetch the right patterns for their task.
- EVIDENCE-BACKED: includes the specific decision(s) that justify the pattern.
- ACTIONABLE: future LLM runs can use it as a guardrail.

GOOD vs BAD:
- BAD: "info@cert.example was rejected as IoC."  (one-off observation; not generalizable)
- GOOD: "Email addresses extracted from figures classified as letterheads or government CERT contact cards are defender contact info, not adversary IoCs."  (generalizable rule)

- BAD: "Chunk #3 had wrong predecessors."  (no actionable pattern)
- GOOD: "When a source describes alternative infection chains observed across DIFFERENT victims, the chunker tends to mis-link them as parallel branches on a single host's kill chain. They should be sibling sub-DAGs sharing only an actor/campaign attribution, not a kill-chain root."  (generalizable + actionable)

- BAD: "The analyst rejected technique T1059.001."  (no reasoning)
- GOOD: "T1059.001 (PowerShell) tends to be over-picked when the source mentions 'powershell' in a malware-capability section but no observed PowerShell execution. Should be filtered when the term appears only inside a malware capability description."

STRUCTURED KEYS (applies_to + concepts):
For each pattern, also emit `applies_to` — the structured keys that tell future runs WHEN this pattern is relevant, so it's surfaced only for matching sources instead of dumped onto every run. Derive them from the deltas you were given (which already carry entity_type, technique_id, and tactic):
- `technique_ids`: the ATT&CK T-IDs the pattern concerns (e.g. a wrong-technique pattern about PowerShell over-picking → ["T1059.001"]).
- `entity_types`: entity types involved (e.g. a defender_ioc pattern about emails → ["email"]).
- `tactics`: relevant ATT&CK tactic shortnames.
- `source_genre`: the kind of source this is most relevant to.
Omit any key you can't ground in the deltas — do NOT invent T-IDs or types. Also emit `concepts`: 1-4 short free-text tags for search/display (e.g. ["letterhead", "figure-extraction"]).

CONSERVATIVE BIAS:
Bias toward FEW HIGH-QUALITY patterns rather than many noisy ones. If you can only justify confidence < 0.5, prefer to emit nothing. The knowledge base is more useful with 5 high-confidence rules than 50 weak observations.

DO NOT EMIT:
- Patterns with no analyst signal (e.g., "the LLM did fine" — that's not a pattern)
- Pattern phrasings that cite a specific entity_id, chunk_id, or value (those go in evidence, not in the pattern text)
- Patterns about the analyst (e.g., "the analyst tends to approve quickly") — focus on LLM behavior

ALSO RESPECT:
- Empty patterns array is a valid response. If the analyst's decisions don't show any meaningful pattern, return [].
"""


# =============================================================================
# Node function
# =============================================================================

async def synthesize_feedback(state: PipelineState) -> dict:
    """Synthesize feedback patterns from the run's analyst decisions
    and persist them to the FeedbackPattern table.

    Robust to missing data and LLM failures: never raises, always returns
    a populated `feedback_synthesis` dict so the pipeline can advance to
    COMPLETED regardless.
    """
    # Pipeline is COMPLETED after this node regardless of SYNTHESIS outcome —
    # this is the last node before END, and failures inside don't block
    # end-of-run.
    #
    # An UPSTREAM failure is different. The distribute -> synthesize_feedback
    # edge is unconditional, so a run distribute already marked FAILED still
    # arrives here; writing COMPLETED would flip a failed run green on the
    # Kanban while state["error"] still holds the reason. Preserve it.
    upstream_failed = state.get("status") == PipelineStatus.FAILED.value
    update: dict = {
        "status": (
            PipelineStatus.FAILED.value if upstream_failed
            else PipelineStatus.COMPLETED.value
        ),
        "current_node": "synthesize_feedback",
        "feedback_synthesis": {"status": "started"},
        "feedback_pattern_ids": [],
    }

    # Skip path: no analyst review actually happened across any gate.
    # Without analyst signal, there's nothing to synthesize.
    if not _had_real_reviews(state):
        logger.info("synthesize_feedback: no analyst reviews detected; skipping")
        # Still close out this run's surfacings. They were previously left
        # NULL forever, which is why 81% of the ledger was unreadable: a row
        # nobody had processed looked exactly like a row we had decided not
        # to score. With no reviews every outcome here is UNSCORED, which is
        # the honest record.
        update["feedback_synthesis"] = {
            "status": "skipped_no_reviews",
            "patterns_emitted": 0,
            "surfacing_scoring": await _score_surfacings(state, {}),
        }
        return update

    try:
        deltas = _compute_deltas(state)

        # Closed loop: score the patterns surfaced during THIS run before
        # anything else. Runs even when there are no deltas — zero analyst
        # corrections means every surfaced pattern was a HIT (the guardrail
        # held). Best-effort: never raises.
        surfacing_scoring = await _score_surfacings(state, deltas)

        # Keep the corrections themselves, not only the rules an LLM writes
        # about them. This runs BEFORE the synthesis call and does not depend
        # on it: the generalization step is the one measured to produce wrong
        # guidance (1 rule in 7 dropped by hand; 19 of 25 never agreeing with
        # the analyst when checked), so the record must not be hostage to it.
        example_recording = await record_examples(state, deltas)

        if not _deltas_have_signal(deltas):
            logger.info("synthesize_feedback: zero analyst disagreements; skipping LLM call")
            update["feedback_synthesis"] = {
                "status": "skipped_no_deltas",
                "patterns_emitted": 0,
                "surfacing_scoring": surfacing_scoring,
                "example_recording": example_recording,
            }
            return update

        response = await call_llm(
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": _format_deltas_for_synthesis(deltas),
            }],
            tools=[SYNTHESIZE_FEEDBACK_TOOL],
            tool_choice={"type": "tool", "name": "synthesize_feedback"},
            temperature=0.0,
            output_model=SynthesizeFeedbackOutput,
        )

        patterns = response.tool_output.get("patterns", []) if response.tool_output else []
        logger.info(
            "synthesize_feedback: LLM emitted %d pattern(s); persisting",
            len(patterns),
        )

        # Persist each pattern (dedup by category + pattern text).
        source_id_str = state.get("source_id", "") or ""
        try:
            source_uuid = uuid.UUID(source_id_str) if source_id_str else None
        except (ValueError, AttributeError):
            source_uuid = None

        pattern_ids: list[str] = []
        async with async_session() as db:
            for raw in patterns:
                category = (raw.get("category") or "other").strip()
                text = (raw.get("pattern") or "").strip()
                evidence = raw.get("evidence") or {}
                confidence = float(raw.get("confidence", 0.0) or 0.0)
                applies_to = raw.get("applies_to") or {}
                concepts = raw.get("concepts") or []

                # Quality filter: drop low-confidence patterns at the
                # persist boundary so the knowledge base stays clean.
                if confidence < 0.5:
                    logger.info(
                        "synthesize_feedback: dropping low-conf pattern (%.2f) "
                        "[%s]: %s",
                        confidence, category, text[:80],
                    )
                    continue
                if not text or len(text) < 10:
                    continue

                # Tuck confidence into evidence so consumers can see it.
                evidence_with_conf = dict(evidence)
                evidence_with_conf.setdefault("synthesizer_confidence", confidence)

                row, created = await persist_pattern(
                    db,
                    source_id=source_uuid,
                    category=category,
                    pattern=text,
                    evidence=evidence_with_conf,
                    applies_to=applies_to if isinstance(applies_to, dict) else {},
                    concepts=concepts if isinstance(concepts, list) else [],
                )
                if created:
                    pattern_ids.append(str(row.id))
                    logger.info(
                        "synthesize_feedback: NEW pattern [%s] %s",
                        category, text[:80],
                    )
                else:
                    logger.info(
                        "synthesize_feedback: bumped existing [%s] (now x%d) %s",
                        category, row.occurrence_count, text[:80],
                    )

        update["feedback_pattern_ids"] = pattern_ids
        update["feedback_synthesis"] = {
            "status": "ok",
            "patterns_emitted_total": len(patterns),
            "patterns_persisted_new": len(pattern_ids),
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "surfacing_scoring": surfacing_scoring,
            "example_recording": example_recording,
        }

    except Exception as e:
        logger.exception("synthesize_feedback: failed (non-fatal)")
        update["feedback_synthesis"] = {
            "status": "failed",
            "error": f"{type(e).__name__}: {e}",
        }
        # Don't re-raise — bundle has already shipped, feedback is bonus.

    return update


# =============================================================================
# Helpers: did the analyst actually review anything?
# =============================================================================

def _had_real_reviews(state: PipelineState) -> bool:
    """True if any gate captured non-empty analyst input.

    When all gates are auto-skipped (gates_enabled all False), the
    review fields are empty and there's nothing to synthesize.

    Checks BOTH the transient review channels (gate0_reviews, chunk_reviews,
    gate1_reviews, gate1_promotions — present only in the window before the
    consuming gate clears them) AND the durable evidence that survives to
    synthesis time: the per-gate correction logs and the delta producers'
    durable sources (validated_entities gate_actions, chunk_decisions).
    Before the durable checks, a run whose only corrections were Gate 0
    entity edits or chunk drops/edits returned False here and skipped
    synthesis entirely — the flywheel silently lost those gates' signal.
    """
    # Transient channels (cheap, and the only signal mid-superstep).
    if state.get("gate0_reviews"):
        return True
    if state.get("chunk_reviews"):
        return True
    if state.get("gate1_reviews") or state.get("gate1_promotions"):
        return True
    # Durable: the per-gate correction logs survive loops and clears.
    if state.get("gate1_correction_log"):
        return True
    if state.get("chunk_correction_log"):
        return True
    # gate_2 never clears its review channel, so it's durable as-is.
    if state.get("gate2_reviews"):
        return True
    # Durable fallbacks via the delta producers themselves: non-approve
    # entity gate_actions on validated_entities (denylist auto-removals
    # already filtered) and legacy chunk_decisions shapes. Reusing the
    # producers keeps this check and _compute_deltas in lockstep.
    if _entity_deltas(state) or _chunk_deltas(state):
        return True
    return False


# =============================================================================
# Helpers: compute LLM-output ↔ analyst-decision deltas
# =============================================================================

_REMOVE = GateAction.REMOVE.value
_REJECT = GateAction.REJECT.value
_EDIT = GateAction.EDIT.value
_APPROVE = GateAction.APPROVE.value


def _compute_deltas(state: PipelineState) -> dict[str, list[dict]]:
    """Build a structured per-gate delta dict for the LLM to reason over."""
    return {
        "gate_0_entities": _entity_deltas(state),
        "gate_chunks": _chunk_deltas(state),
        "gate_1_procedures": _procedure_deltas(state),
        "gate_2_relationships": _relationship_deltas(state),
    }


def _deltas_have_signal(deltas: dict[str, list[dict]]) -> bool:
    """True iff at least one gate has at least one non-trivial delta."""
    return any(v for v in deltas.values())


def captured_corrections(state: PipelineState) -> list[dict]:
    """Flatten the in-flight analyst corrections in a checkpoint state into
    UI-friendly items: {gate, action, summary, detail}.

    Reuses _compute_deltas, so the "captured this run" view the Feedback tab
    shows BEFORE synthesis matches exactly what synthesize_feedback will
    consume at run completion. Powers GET /api/feedback-patterns/captured so
    an analyst sees their reject/edit/discard the moment they submit it,
    instead of waiting for the run to finish (and synthesis to maybe emit a
    pattern). Read-only; never raises on odd shapes.
    """
    deltas = _compute_deltas(state)
    items: list[dict] = []

    for e in deltas.get("gate_0_entities", []):
        val = e.get("value", "")
        summary = f"{e.get('action', '')} {e.get('entity_type', '')} {val!r}".strip()
        if e.get("edited_type") or e.get("edited_value"):
            summary += f"  → type={e.get('edited_type') or '—'} value={e.get('edited_value') or '—'}"
        items.append({
            "gate": "entities", "action": e.get("action", ""),
            "summary": summary, "detail": e.get("edit_rationale") or "",
        })

    for c in deltas.get("gate_chunks", []):
        kind = c.get("kind") or c.get("action") or ""
        if c.get("chunk_id"):
            summary = f"{kind} chunk {c.get('chunk_id')}"
        elif kind == "wholesale_reject":
            summary = f"rejected all chunks ({c.get('reason', '')})"
        elif kind == "analyst_added":
            summary = "added a chunk the LLM missed"
        else:
            summary = kind
        items.append({
            "gate": "chunks", "action": kind, "summary": summary.strip(),
            "detail": c.get("rationale") or c.get("comments") or c.get("text") or "",
        })

    for p in deltas.get("gate_1_procedures", []):
        kind = p.get("kind") or p.get("action") or ""
        if p.get("technique_id"):  # promotion
            summary = f"promoted technique {p.get('technique_id')}"
            detail = ""
        else:
            name = p.get("original_name", "")
            rr = p.get("reject_reason")
            summary = f"{kind} {name!r}" + (f" — {rr}" if rr else "")
            # Per-action labels (mirrors _format_deltas_for_synthesis): an
            # edit's rejected_techniques holds only the removed ones, and its
            # corrected list (kept + added) is summarized by removed/added.
            is_edit = kind == GateAction.EDIT.value
            rej = join_technique_ids(p.get("rejected_techniques"))
            add = join_technique_ids(p.get("added_techniques"))
            cor = join_technique_ids(p.get("corrected_techniques"))
            detail = p.get("feedback") or ""
            if rej:
                detail = (detail + f"  {'removed' if is_edit else 'rejected'}: {rej}").strip()
            if add:
                detail = (detail + f"  added: {add}").strip()
            if cor and not is_edit:
                detail = (detail + f"  → correct: {cor}").strip()
        items.append({
            "gate": "techniques", "action": kind, "summary": summary.strip(),
            "detail": detail,
        })

    for r in deltas.get("gate_2_relationships", []):
        kind = r.get("kind") or r.get("action") or ""
        if kind == "analyst_added":
            summary = (
                f"added {r.get('source_name', '')} "
                f"--{r.get('relationship_type', '')}--> {r.get('target_name', '')}"
            )
        elif kind == "wholesale_reject":
            summary = "rejected all relationships"
        else:
            summary = f"{kind} relationship {r.get('rel_id', '')}"
        items.append({
            "gate": "relationships", "action": kind, "summary": summary.strip(),
            "detail": r.get("rationale") or r.get("feedback") or "",
        })

    return items


def _entity_deltas(state: PipelineState) -> list[dict]:
    """Entities the analyst rejected/edited at Gate 0.

    Original entities live in state['entities']; the analyst's decisions
    are written onto state['validated_entities'] via gate_0's processor
    (gate_action / edited_value / edited_type / edit_rationale).
    """
    deltas: list[dict] = []
    validated = state.get("validated_entities", []) or []
    for e in validated:
        action = e.get("gate_action")
        if not action or action == _APPROVE:
            continue
        # Denylist removals are machine-originated (the guardrail auto-removes
        # at gate_0, even on gates-disabled runs) — not analyst corrections.
        # Feeding them back would echo an already-promoted pattern into
        # synthesis and show phantom "analyst corrections" in the captured
        # panel. An analyst EDIT on a denylisted entity is still real signal,
        # so only the remove action is filtered.
        if e.get("denylisted") and action == _REMOVE:
            continue
        deltas.append({
            "entity_id": e.get("entity_id", ""),
            "value": e.get("value", ""),
            "entity_type": e.get("entity_type", ""),
            "llm_confidence": e.get("confidence", 0.0),
            "context_snippet": (e.get("context_snippet") or "")[:300],
            "action": action,
            "edited_value": e.get("edited_value"),
            "edited_type": e.get("edited_type"),
            "edit_rationale": e.get("edit_rationale"),
        })
    return deltas


def _chunk_deltas(state: PipelineState) -> list[dict]:
    """Chunk-level analyst signal at gate_chunks.

    Sources from the durable chunk_correction_log (every non-approve signal
    across every pass, stored by gate_chunks in this exact delta shape with
    text snapshotted at gate time). This is what lets a wholesale reject, an
    analyst-added chunk, or a pass-1 drop/edit survive to synthesis — the
    transient channels they originally rode on (chunk_rerun_feedback,
    chunk_reviews) are cleared after consumption, and chunk_decisions is
    last-write-wins across re-chunk loops. Falls back to assembling from
    those transient channels for checkpoints predating the ledger.
    """
    log = state.get("chunk_correction_log", []) or []
    if log:
        return [dict(rec) for rec in log if isinstance(rec, dict)]

    # Back-compat: checkpoints written before chunk_correction_log existed.
    deltas: list[dict] = []
    decisions = state.get("chunk_decisions", []) or []
    chunks = state.get("chunks", []) or []
    by_id = {c.get("chunk_id", ""): c for c in chunks}
    for d in decisions:
        action = d.get("action")
        if not action or action == _APPROVE:
            continue
        cid = d.get("chunk_id", "")
        original = by_id.get(cid, {})
        deltas.append({
            "chunk_id": cid,
            "original_text": (original.get("text", "") or "")[:300],
            "action": action,
            "edits": d.get("edits"),
            "rationale": d.get("rationale"),
        })

    # Reject path: the whole chunking output was rejected with a reason.
    rerun = state.get("chunk_rerun_feedback")
    if rerun:
        deltas.append({
            "kind": "wholesale_reject",
            "reason": rerun.get("reason"),
            "comments": (rerun.get("comments") or "")[:500],
        })

    # Analyst-added chunks (signal that the LLM missed something).
    chunk_reviews = state.get("chunk_reviews") or {}
    if isinstance(chunk_reviews, dict):
        added = chunk_reviews.get("added_chunks") or []
        for a in added:
            deltas.append({
                "kind": "analyst_added",
                "text": (a.get("text", "") or "")[:300],
                "source_excerpt": (a.get("source_excerpt") or "")[:200],
            })
    return deltas


def _procedure_deltas(state: PipelineState) -> list[dict]:
    """Procedure draft edits + technique promotions at Gate 1.

    Sources from the durable gate1_correction_log (all non-approve decisions
    across every pass, with the rejected + corrected techniques carried on
    each entry). This is what lets a reject that triggered a re-extract loop
    still reach the flywheel — gate1_decisions is last-write-wins and only
    reflects the final (post-loop, typically all-approve) pass. Falls back to
    gate1_decisions for in-flight checkpoints predating the correction log.
    """
    deltas: list[dict] = []
    log = state.get("gate1_correction_log", []) or []
    if log:
        for rec in log:
            if rec.get("action") == "promote":
                # Durable promotion record (the transient gate1_promotions
                # list is cleared in the same gate_1 update that applies it).
                deltas.append({
                    "kind": "promotion",
                    "chunk_id": rec.get("chunk_id"),
                    "technique_id": rec.get("technique_id"),
                })
                continue
            deltas.append({
                "draft_id": rec.get("draft_id", ""),
                "original_name": rec.get("draft_name", ""),
                "chunk_id": rec.get("chunk_id", ""),
                "action": rec.get("action"),
                "reject_reason": rec.get("reject_reason"),
                "feedback": (rec.get("rationale") or "")[:400],
                "rejected_techniques": rec.get("rejected_techniques", []),
                "corrected_techniques": rec.get("corrected_techniques", []),
                "added_techniques": rec.get("added_techniques", []),
            })
    else:
        # Back-compat: checkpoints written before gate1_correction_log existed.
        drafts = state.get("drafts", []) or []
        by_id = {d.get("draft_id", ""): d for d in drafts}
        for d in state.get("gate1_decisions", []) or []:
            action = d.get("action")
            if not action or action == _APPROVE:
                continue
            did = d.get("draft_id", "")
            orig = by_id.get(did, {})
            deltas.append({
                "draft_id": did,
                "original_name": orig.get("name", ""),
                "action": action,
                "reject_reason": d.get("reason"),
                "feedback": (d.get("feedback") or "")[:400],
            })
    promotions = state.get("gate1_promotions", []) or []
    for p in promotions:
        deltas.append({
            "kind": "promotion",
            "chunk_id": p.get("chunk_id"),
            "technique_id": p.get("technique_id"),
        })
    return deltas


def _relationship_deltas(state: PipelineState) -> list[dict]:
    """Relationship-level edits at Gate 2."""
    deltas: list[dict] = []
    reviews = state.get("gate2_reviews", []) or []
    for r in reviews:
        action = r.get("action")
        if not action or action == _APPROVE:
            continue
        deltas.append({
            "rel_id": r.get("rel_id", ""),
            "action": action,
            "rationale": (r.get("rationale") or "")[:300],
        })

    # Wholesale reject path
    g2 = state.get("gate2_decision") or {}
    if isinstance(g2, dict) and g2.get("approved") is False:
        deltas.append({
            "kind": "wholesale_reject",
            "feedback": (g2.get("feedback") or "")[:400],
        })

    # Analyst-added relationships
    added = state.get("gate2_added_rels", []) or []
    for a in added:
        deltas.append({
            "kind": "analyst_added",
            "relationship_type": a.get("relationship_type"),
            "source_name": a.get("source_name"),
            "target_name": a.get("target_name"),
        })
    return deltas


def _format_deltas_for_synthesis(deltas: dict[str, list[dict]]) -> str:
    """Render deltas as a human-readable digest for the LLM to reason over."""
    parts: list[str] = ["PIPELINE RUN DELTAS:"]

    e_deltas = deltas.get("gate_0_entities", [])
    if e_deltas:
        parts.append(f"\nGate 0 — Entity decisions ({len(e_deltas)} non-approve):")
        for e in e_deltas:
            line = (
                f"  [{e.get('action')}] {e.get('entity_type')}={e.get('value')!r}"
                f"  llm_conf={e.get('llm_confidence', 0):.2f}"
            )
            parts.append(line)
            if e.get("edited_type") or e.get("edited_value"):
                parts.append(
                    f"      edits: type={e.get('edited_type')} value={e.get('edited_value')}"
                )
            if e.get("edit_rationale"):
                parts.append(f"      rationale: {e.get('edit_rationale')}")
            if e.get("context_snippet"):
                parts.append(f"      context: {e.get('context_snippet')}")
    else:
        parts.append("\nGate 0 — Entity decisions: (no non-approve actions)")

    c_deltas = deltas.get("gate_chunks", [])
    if c_deltas:
        parts.append(f"\ngate_chunks — Chunk decisions ({len(c_deltas)} entries):")
        for c in c_deltas:
            kind = c.get("kind") or c.get("action")
            # A wholesale_reject carries no chunk_id — it rejects the whole
            # pass. Rendering "chunk_id=?" made an absent value look like a
            # lost one.
            cid = c.get("chunk_id")
            parts.append(
                f"  [{kind}] chunk_id={cid}" if cid
                else f"  [{kind}] (entire chunking pass)"
            )
            # The reason code is the only unambiguous statement of INTENT a
            # chunk delta carries. Without it the synthesizer sees a bare
            # rejection plus prose describing what the SOURCE said, and has to
            # infer whether the analyst wanted that content added or removed.
            # It once inferred "removed" from a `missed_procedures`
            # reject — the analyst had rejected the pass to get a procedure
            # ADDED — and wrote a rule whose diagnosis was the reverse of the
            # correction that produced it. Gate 1's block below has always
            # passed its reject_reason; this block never did.
            if c.get("reason"):
                parts.append(f"      reason: {c.get('reason')}")
            # An analyst edit is the most precise signal a chunk gate produces —
            # it says what "correct" looks like, not merely that something was
            # wrong. The fields were carried in the delta and dropped here, so
            # a rewritten chunk reached the synthesizer as a bare [edit] line.
            if c.get("edits"):
                rendered = ", ".join(
                    f"{k}={str(v)[:200]!r}" for k, v in c["edits"].items()
                )
                parts.append(f"      analyst edits: {rendered}")
            if c.get("rationale"):
                parts.append(f"      rationale: {c.get('rationale')}")
            if c.get("comments"):
                parts.append(f"      comments: {c.get('comments')}")
            if c.get("text"):
                parts.append(f"      analyst-added text: {c.get('text')}")
    else:
        parts.append("\ngate_chunks: (no non-approve actions)")

    p_deltas = deltas.get("gate_1_procedures", [])
    if p_deltas:
        parts.append(f"\nGate 1 — Procedure/technique decisions ({len(p_deltas)} entries):")
        for p in p_deltas:
            kind = p.get("kind") or p.get("action")
            # A promotion is keyed by CHUNK, not draft (_procedure_deltas emits
            # only kind/chunk_id/technique_id for it), so every promotion
            # rendered "draft_id=?" while discarding the id it actually had.
            if kind == "promotion":
                parts.append(f"  [{kind}] chunk_id={p.get('chunk_id') or '?'}")
            else:
                parts.append(f"  [{kind}] draft_id={p.get('draft_id') or '?'}")
            if p.get("reject_reason"):
                parts.append(f"      reject_reason: {p.get('reject_reason')}")
            if p.get("feedback"):
                parts.append(f"      feedback: {p.get('feedback')}")
            # Action-aware labels: an EDIT's rejected_techniques holds only
            # the techniques the analyst REMOVED (per-action producer
            # semantics in gates._correction_record) — labeling those
            # "LLM-picked (rejected)" on an edit, or rendering kept picks as
            # rejected, was teaching the synthesizer false corrections.
            is_edit = kind == GateAction.EDIT.value
            rej = join_technique_ids(p.get("rejected_techniques"))
            if rej:
                label = "analyst removed" if is_edit else "LLM-picked (rejected)"
                parts.append(f"      {label}: {rej}")
            add = join_technique_ids(p.get("added_techniques"))
            if add:
                parts.append(f"      analyst added: {add}")
            cor = join_technique_ids(p.get("corrected_techniques"))
            if cor and not is_edit:
                # For rejects the corrected list is the analyst's full target
                # mapping; for edits removed/added above already say it.
                parts.append(f"      analyst-corrected to: {cor}")
            if p.get("technique_id"):
                parts.append(f"      promoted technique: {p.get('technique_id')}")
    else:
        parts.append("\nGate 1: (no non-approve actions)")

    r_deltas = deltas.get("gate_2_relationships", [])
    if r_deltas:
        parts.append(f"\nGate 2 — Relationship decisions ({len(r_deltas)} entries):")
        for r in r_deltas:
            kind = r.get("kind") or r.get("action")
            parts.append(
                f"  [{kind}] rel_id={r.get('rel_id', '?')} type={r.get('relationship_type', '?')}"
            )
            if r.get("rationale"):
                parts.append(f"      rationale: {r.get('rationale')}")
            if r.get("feedback"):
                parts.append(f"      feedback: {r.get('feedback')}")
    else:
        parts.append("\nGate 2: (no non-approve actions)")

    parts.append(
        "\n\nExtract generalizable patterns the next pipeline run should "
        "consult. Bias toward few high-confidence rules. Empty patterns "
        "list is a valid response."
    )
    return "\n".join(parts)


# =============================================================================
# Closed loop: score this run's surfacings (hit/miss) → salience
# =============================================================================
#
# A pattern SURFACED into a node's prompt this run is a MISS if the analyst
# still made the correction it warns about (relevant but ineffective as
# phrased), and a HIT otherwise (the guardrail held). We map each pattern's
# category to the gate area its corrections show up in, then check whether any
# delta in that area matches the pattern by shared structured anchor or
# embedding cosine. Outcomes update hit/miss counts + recompute salience.

# category -> the gate "area" its corrections surface in.
_AREA_FOR_CATEGORY = {
    "defender_ioc": "entities",
    "brand_as_malware": "entities",
    "false_positive_entity": "entities",
    "mis_attribution": "entities",
    "over_chunked": "chunks",
    "under_chunked": "chunks",
    "missing_procedure": "chunks",
    "wrong_predecessor": "chunks",
    "parallel_capability_misordered": "chunks",
    "artifact_loss": "chunks",
    "missing_tactic": "techniques",
    "wrong_technique": "techniques",
    "thin_initial_access": "techniques",
    "orphan_ioc": "techniques",
    "wrong_relationship": "relationships",
}
# area -> the key in the _compute_deltas() dict.
# Which gate's review produces evidence about a pattern in this area. Used to
# decide whether a surfacing can be scored at all: a gate that was disabled,
# or was decided by the AI reviewer unattended, yields no human judgment, and
# "nobody corrected anything" is then not evidence the pattern held.
_GATE_KEY_FOR_AREA = {
    "entities": "entities",
    "chunks": "chunks",
    "techniques": "procedures",
    "relationships": "bundle",
}

_DELTA_KEY_FOR_AREA = {
    "entities": "gate_0_entities",
    "chunks": "gate_chunks",
    "techniques": "gate_1_procedures",
    "relationships": "gate_2_relationships",
}

# Salient string fields across all delta shapes, concatenated into the text we
# embed to compare against a surfaced pattern.
_DELTA_TEXT_FIELDS = (
    "value", "entity_type", "edited_type", "edited_value", "edit_rationale",
    "original_text", "rationale", "comments", "reason", "text", "source_excerpt",
    "original_name", "reject_reason", "feedback", "technique_id",
    "relationship_type", "source_name", "target_name",
)


# Technique-list fields that represent an ACTUAL correction: rejected (full
# mapping on a reject; removed-only on an edit) and added (analyst additions).
# corrected_techniques is deliberately NOT here — for an edit it is the
# analyst's full final list, kept techniques included, and folding kept
# techniques into match text/anchors was MISS-scoring patterns whose advice
# the analyst followed (salience decay of working patterns).
_DELTA_CORRECTION_TECHNIQUE_KEYS = ("rejected_techniques", "added_techniques")

# The same two, kept apart. A correction has a DIRECTION, and the deterministic
# matcher was blind to it: on one espionage-RAT run the analyst PROMOTED T1070.004, and
# two rules that merely listed T1070.004 among the ids they were learned from
# took a miss for it — one of them a rule about deletion on a ransomware crew's
# own leak server, in an espionage report that has no leak server.
_DELTA_REMOVED_TECHNIQUE_KEYS = ("rejected_techniques",)
_DELTA_ADDED_TECHNIQUE_KEYS = ("added_techniques",)


def _delta_text(d: dict) -> str:
    base = " ".join(str(d[k]) for k in _DELTA_TEXT_FIELDS if d.get(k))
    # Procedure deltas carry techniques as list-of-dict, not the flat
    # technique_id field — fold the ids + names of the actually-corrected
    # ones into the text so embedding-based hit/miss scoring can match a
    # wrong_technique pattern.
    extra: list[str] = []
    for key in _DELTA_CORRECTION_TECHNIQUE_KEYS:
        for t in d.get(key) or []:
            if isinstance(t, dict):
                if t.get("technique_id"):
                    extra.append(t["technique_id"])
                if t.get("technique_name"):
                    extra.append(t["technique_name"])
    return (base + " " + " ".join(extra)).strip() if extra else base


def _collect_tids(d: dict, keys: tuple[str, ...]) -> set[str]:
    out: set[str] = set()
    for key in keys:
        for t in d.get(key) or []:
            if isinstance(t, dict) and t.get("technique_id"):
                out.add(t["technique_id"])
    return out


def _delta_anchors(d: dict) -> dict:
    removed = _collect_tids(d, _DELTA_REMOVED_TECHNIQUE_KEYS)
    added = _collect_tids(d, _DELTA_ADDED_TECHNIQUE_KEYS)
    # A flat technique_id only ever rides a promotion delta (_procedure_deltas),
    # which is the analyst putting a technique BACK.
    if d.get("technique_id"):
        added.add(d["technique_id"])
    etypes = {d[k] for k in ("entity_type", "edited_type") if d.get(k)}
    return {
        "removed_technique_ids": removed,
        "added_technique_ids": added,
        "technique_ids": removed | added,
        "entity_types": etypes,
    }


def _deltas_by_area(deltas: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Pre-embed each delta's text once, grouped by gate area."""
    out: dict[str, list[dict]] = {}
    for area, dkey in _DELTA_KEY_FOR_AREA.items():
        reprs = []
        for d in deltas.get(dkey, []) or []:
            text = _delta_text(d)
            reprs.append({
                "raw": d,
                "text": text,
                "anchors": _delta_anchors(d),
                "vec": pattern_embedding.embed_text(text) if text else None,
            })
        out[area] = reprs
    return out


def _pattern_was_recorrected(pattern: FeedbackPattern, area_deltas: dict) -> bool:
    """Deterministic fallback for miss attribution, used only when the
    adjudicator could not run.

    Deliberately conservative — it under-reports misses rather than
    misdirecting them. A miss that is never recorded leaves salience
    uninformative; a miss recorded against the wrong rule actively decays a
    working rule and promotes a broken one, and salience feeds retrieval
    ranking, so the error compounds into what gets surfaced next run.

    Two signals the earlier version trusted are gone:

    * `entity_types` overlap on its own. `vulnerability` and `ioc_domain` are
      buckets of hundreds of entities, so every rule tagged with a common type
      was charged for every correction of that type. On one espionage-RAT run a rule
      about ATT&CK T-IDs written inline in prose being mistyped as
      vulnerabilities took a miss because the analyst removed a redacted CVE
      placeholder — same bucket, unrelated error.
    * A shared technique id on a correction that ADDED that technique. Only a
      removal is evidence that a "do not include this" rule failed to hold.
      Most of the corpus is over-inclusion rules; the omission categories are
      left to the adjudicator, which can read the direction out of the rule
      text instead of guessing it from the category name.
    """
    area = _AREA_FOR_CATEGORY.get(pattern.category)
    if area is None:
        candidates = [r for rs in area_deltas.values() for r in rs]
    else:
        candidates = area_deltas.get(area, [])
    if not candidates:
        return False

    pat_emb = pattern.embedding
    pat_at = pattern.applies_to or {}
    pat_tids = set(pat_at.get("technique_ids") or [])
    for r in candidates:
        anc = r["anchors"]
        if pat_tids and (pat_tids & anc["removed_technique_ids"]):
            return True
        if pat_emb and r["vec"] and (
            pattern_embedding.cosine(pat_emb, r["vec"]) >= settings.feedback_match_threshold
        ):
            return True
    return False


# =============================================================================
# Miss attribution
# =============================================================================
#
# WHY THIS IS AN LLM CALL AND NOT A LOOKUP:
# Attribution asks "is this correction an instance of the error this rule
# describes?" — a judgment about a rule and an event. The only deterministic
# handle available is "do they name the same ATT&CK technique", which is
# neither necessary nor sufficient.
#
# Not necessary: on one espionage-RAT run the analyst removed T1036.008 (Masquerade
# File Type) from a curl reconnaissance draft — a textbook instance of the rule
# "techniques appear over-attributed when a step only generically implies a
# capability". That rule was surfaced at that exact step and scored a HIT,
# because its `applies_to.technique_ids` are T1059.003 / T1021.001 / T1068 —
# the ids it happened to be LEARNED from, which is provenance, not scope.
#
# Not sufficient: the same run charged two rules with a miss for a technique
# the analyst PROMOTED, on the strength of one shared id.
#
# `applies_to` is doing two incompatible jobs. For retrieval, "learned from
# similar techniques" is a fine relevance proxy. For attribution it is wrong,
# and attribution feeds salience, which feeds retrieval — so the error
# compounds run over run.

ATTRIBUTE_CORRECTIONS_TOOL = {
    "name": "attribute_corrections",
    "description": (
        "Given the guardrail rules that were shown to the pipeline during a "
        "run, and the corrections the analyst then made anyway, say which "
        "rule each correction is an instance of. A rule is only credited "
        "with a correction when following that rule would have prevented it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "attributions": {
                "type": "array",
                "description": "One entry per correction, in any order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "correction": {
                            "type": "integer",
                            "description": "The correction's number.",
                        },
                        "rules": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": (
                                "Numbers of the rules this correction is an "
                                "instance of. Usually empty."
                            ),
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "One sentence: why those, or why none.",
                        },
                    },
                    "required": ["correction", "rules", "reasoning"],
                },
            },
        },
        "required": ["attributions"],
    },
}


ATTRIBUTION_SYSTEM_PROMPT = """You are auditing a guardrail system.

Before this pipeline run, a set of RULES was injected into the prompts of the
model that produced the output. An analyst then reviewed that output and made
CORRECTIONS. Your job is to decide which corrections each rule failed to
prevent, so the rule can be scored.

THE TEST, applied to one rule and one correction:

    Had the model actually followed this rule, would the analyst have had no
    reason to make this correction?

Only "yes" attributes the correction to the rule.

WHAT THIS IS NOT:
- Not topical similarity. A rule and a correction can both be about
  PowerShell, or both be about a domain name, and still have nothing to do
  with each other.
- Not a shared ATT&CK technique id. Rules carry the ids they were learned
  from; that a correction names one of them is a coincidence unless the rule's
  own claim covers this correction.
- Not "the rule is about roughly this area". Every rule shown to you was
  surfaced in the relevant area already. That is why the list is not empty; it
  is not evidence.

WHAT IT IS:
- A rule stated generally ("techniques are over-attributed when a step only
  implies a capability") DOES cover a specific instance of that error, even
  when the rule names different techniques than the correction does. Match on
  the claim, not on the identifiers.
- A rule whose conditions are absent from this source cannot be an instance.
  A rule about a ransomware crew's leak server does not apply to an espionage
  report with no leak server, whatever ids it lists.

WORKED EXAMPLE — the mistake to avoid:

    RULE   "Entities that appear only in a report's own remediation or
            hardening advice describe the defender's response, not the
            adversary's tooling."
    CORRECTION  removed the entity "CrowdStrike Falcon" (tool), no rationale
            given, from a report whose closing section recommends deploying it

    WRONG: "the rule does not mention CrowdStrike" — the rule names no entity
    at all, because it is about a KIND of error.
    RIGHT: attribute it. The rule describes exactly this error, so following
    it would have prevented the correction.

Ask of every rule: what error does it claim happens, and is this correction an
instance of that error? A rule that names no technique can still be the right
answer; a rule that names this exact technique can still be the wrong one.

Most corrections are instances of NO rule — the analyst found something no
rule had anticipated. An empty `rules` list is the normal answer and is what
lets a new rule be written for the gap. Return one entry per correction.
"""


def _format_rules_for_attribution(patterns: list) -> str:
    # `applies_to.technique_ids` is deliberately NOT shown. It records the ids
    # a rule was learned from, and putting it in front of the adjudicator
    # invites exactly the id-matching this call exists to replace. Any id the
    # rule actually claims something about is in its text.
    return "\n".join(
        f"[{i}] ({p.category}) {(p.pattern or '').strip()}"
        for i, p in enumerate(patterns)
    )


_ATTRIBUTION_SKIP_KEYS = {"entity_id", "draft_id", "chunk_id", "llm_confidence"}
_ATTRIBUTION_TECHNIQUE_KEYS = _DELTA_CORRECTION_TECHNIQUE_KEYS + ("corrected_techniques",)


def _name_technique_list(v) -> str:
    """`T1036.008 (Masquerade File Type)` — the name is what makes a correction
    legible as an instance of a rule stated in prose."""
    out = []
    for t in v or []:
        if not isinstance(t, dict):
            continue
        tid = t.get("technique_id") or ""
        nm = t.get("technique_name") or ""
        out.append(f"{tid} ({nm})" if tid and nm else (tid or nm))
    return ", ".join(x for x in out if x)


def _format_corrections_for_attribution(corrections: list[tuple[str, dict]]) -> str:
    lines = []
    for i, (area, d) in enumerate(corrections):
        fields = []
        for k, v in d.items():
            if k in _ATTRIBUTION_SKIP_KEYS or not v:
                continue
            if k in _ATTRIBUTION_TECHNIQUE_KEYS:
                rendered = _name_technique_list(v)
                if rendered:
                    fields.append(f"{k}={rendered}")
                continue
            fields.append(f"{k}={str(v)[:300]}")
        lines.append(f"[{i}] gate={area} " + "; ".join(fields))
    return "\n".join(lines)


async def _adjudicate_attribution(
    patterns: list, area_deltas: dict
) -> set | None:
    """Ask which surfaced rules each correction is an instance of.

    Returns the set of pattern ids that at least one correction was attributed
    to, or None when the call could not be made or trusted — in which case the
    caller falls back to `_pattern_was_recorrected`.
    """
    if not patterns:
        return set()
    areas = {_AREA_FOR_CATEGORY.get(p.category) for p in patterns}
    corrections = [
        (area, r["raw"])
        for area, reprs in area_deltas.items()
        if area in areas
        for r in reprs
    ]
    if not corrections:
        return set()

    try:
        response = await call_llm(
            system=ATTRIBUTION_SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": (
                    "RULES SHOWN TO THE MODEL DURING THIS RUN\n"
                    + _format_rules_for_attribution(patterns)
                    + "\n\nCORRECTIONS THE ANALYST MADE\n"
                    + _format_corrections_for_attribution(corrections)
                ),
            }],
            tools=[ATTRIBUTE_CORRECTIONS_TOOL],
            tool_choice={"type": "tool", "name": "attribute_corrections"},
            temperature=0.0,
            output_model=AttributeCorrectionsOutput,
        )
    except Exception as e:  # noqa: BLE001 — best-effort; caller has a fallback
        logger.warning("miss attribution call failed, falling back: %s", e)
        return None

    items = (response.tool_output or {}).get("attributions") or []
    attributed: set = set()
    for item in items:
        for idx in item.get("rules") or []:
            if isinstance(idx, int) and 0 <= idx < len(patterns):
                attributed.add(patterns[idx].id)
                logger.info(
                    "miss attributed: correction %s -> rule [%d] %s (%s)",
                    item.get("correction"), idx,
                    (patterns[idx].pattern or "")[:70],
                    (item.get("reasoning") or "")[:120],
                )
    return attributed


# Recorded instead of NULL so "we looked and there was no evidence" is
# distinguishable from "we never got here" — the 1202 NULL rows once found in
# the ledger were the latter, and were indistinguishable from the former.
UNSCORED = "unscored"


def _is_scoreable(state: PipelineState, pattern) -> bool:
    """True when this run can say anything at all about this pattern.

    A hit requires a human to have actually looked at the gate the pattern
    belongs to. See _outcome_for for why.
    """
    area = _AREA_FOR_CATEGORY.get(pattern.category)
    if area is None:
        return False
    gate_key = _GATE_KEY_FOR_AREA.get(area)
    if gate_key is None:
        return False
    if not is_gate_enabled(state, gate_key):
        return False
    return gate_mode(state, gate_key) != "auto"


def _outcome_for(
    state: PipelineState,
    pattern,
    area_deltas: dict,
    attributed: set | None = None,
) -> str:
    """hit / miss / unscored for one surfaced pattern.

    The old rule was "MISS if a correction landed in this gate's area, HIT
    otherwise" — so a pattern scored a hit whenever nobody corrected anything.
    That is also what happens when the gate was switched off, when the AI
    reviewer ran it unattended, and when the pattern had nothing to do with
    the source. Three of those four are not evidence the guardrail held, and
    pooling them produced an 84% hit rate that mostly measured the absence of
    review.

    So a hit now requires a human to have actually looked at that gate.
    Everything else is UNSCORED: recorded, and excluded from the counters.
    """
    if not _is_scoreable(state, pattern):
        # Unknown category, gate never ran, or the AI decided it unattended.
        # None of those is a human judging the output, so neither a hit nor a
        # miss is evidence of anything.
        return UNSCORED
    if attributed is not None:
        return "miss" if pattern.id in attributed else "hit"
    return "miss" if _pattern_was_recorrected(pattern, area_deltas) else "hit"


async def _score_surfacings(state: PipelineState, deltas: dict) -> dict:
    """Score this run's unscored surfacings as hit/miss and update salience.

    Best-effort: own session, swallows errors, never raises (the bundle has
    shipped; this is bonus signal). Returns a small summary dict.
    """
    result = {"scored": 0, "hits": 0, "misses": 0, "unscored": 0}
    source_id_str = state.get("source_id", "") or ""
    try:
        source_uuid = uuid.UUID(source_id_str) if source_id_str else None
    except (ValueError, AttributeError, TypeError):
        source_uuid = None
    if source_uuid is None:
        return result

    try:
        async with async_session() as db:
            rows = list((await db.execute(
                select(FeedbackPatternSurfacing).where(and_(
                    FeedbackPatternSurfacing.source_id == source_uuid,
                    FeedbackPatternSurfacing.outcome.is_(None),
                ))
            )).scalars().all())
            if not rows:
                return result

            # Dedup by pattern_id (multiple nodes / reruns may have logged the
            # same pattern); score the pattern once, close out all its rows.
            by_pattern: dict = {}
            for r in rows:
                by_pattern.setdefault(r.pattern_id, []).append(r)

            pattern_rows = list((await db.execute(
                select(FeedbackPattern).where(
                    FeedbackPattern.id.in_(list(by_pattern.keys()))
                )
            )).scalars().all())
            patterns_by_id = {p.id: p for p in pattern_rows}

            area_deltas = _deltas_by_area(deltas)
            now = datetime.now(timezone.utc)

            # Attribution runs once over the whole run, not per pattern: the
            # question "which rule is this correction an instance of" is only
            # answerable with every candidate rule in view at the same time.
            scoreable = [
                p for p in patterns_by_id.values() if _is_scoreable(state, p)
            ]
            attributed = await _adjudicate_attribution(scoreable, area_deltas)
            if attributed is None:
                logger.info(
                    "miss attribution unavailable; using the conservative "
                    "deterministic fallback for %d pattern(s)", len(scoreable),
                )

            for pid, srows in by_pattern.items():
                pattern = patterns_by_id.get(pid)
                if pattern is None:
                    # Deleted since it was surfaced. Nothing to evaluate, and
                    # crediting it as a hit inflated the ledger with rows that
                    # can never be traced back to a pattern.
                    outcome = UNSCORED
                else:
                    outcome = _outcome_for(state, pattern, area_deltas, attributed)
                    if outcome == "miss":
                        pattern.miss_count = (pattern.miss_count or 0) + 1
                    elif outcome == "hit":
                        pattern.hit_count = (pattern.hit_count or 0) + 1
                    if outcome in ("hit", "miss"):
                        # Counters move once per (pattern, source): four nodes
                        # surfacing one pattern for one source is one review
                        # event, not four. The per-row stamp below still
                        # records which node surfaced it.
                        pattern.salience = compute_salience(
                            hit_count=pattern.hit_count or 0,
                            miss_count=pattern.miss_count or 0,
                            occurrence_count=pattern.occurrence_count or 1,
                            last_seen_at=pattern.last_seen_at,
                            now=now,
                        )
                        pattern.last_scored_at = now
                for r in srows:
                    r.outcome = outcome
                    r.scored_at = now
                result["scored"] += 1
                if outcome == "miss":
                    result["misses"] += 1
                elif outcome == "hit":
                    result["hits"] += 1
                else:
                    result["unscored"] += 1

            await db.commit()
        logger.info(
            "synthesize_feedback: closed out %d surfaced pattern(s) — "
            "%d hit, %d miss, %d unscored (no human judgment at that gate)",
            result["scored"], result["hits"], result["misses"],
            result["unscored"],
        )
        return result
    except Exception as e:  # noqa: BLE001 — best-effort, never block end-of-run
        logger.warning("_score_surfacings failed (non-fatal): %s", e)
        return {"status": "error", "error": f"{type(e).__name__}: {e}"}
