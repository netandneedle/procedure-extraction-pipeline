"""Gate nodes: Human review checkpoints in the extraction pipeline.

Gate nodes are where the graph pauses for analyst review. The flow is:

    1. Upstream node finishes (e.g., extract_entities)
    2. Graph hits interrupt_before on the gate node -> freezes
    3. API serves review data to the frontend
    4. Analyst submits decisions
    5. API writes raw decisions to state via graph.update_state()
    6. Graph resumes -> gate node runs
    7. Gate node processes raw decisions into downstream state fields

GATE 0 (entity review):
    - Analyst reviews extracted entities: approve, edit, or remove each one
    - Gate produces validated_entities for downstream consumption

GATE (chunks) (chunk review):
    - Analyst reviews the behavioral chunks before technique extraction:
      approve / edit / drop / merge / add chunks, edit precedes edges,
      override operator kinds and conditions
    - Reject re-runs chunk_behaviors with the analyst's feedback

GATE 1 (procedure review):
    - Analyst reviews draft procedures + technique mappings
    - Can approve, reject (with reason), edit, or remove each draft
    - Gate determines rejection routing:
        BAD_CHUNK_BOUNDARY -> re-chunk (highest priority)
        other rejections   -> re-map techniques
        no rejections      -> proceed to normalize

GATE 2 (relationship review):
    - Analyst reviews normalized drafts + entity relationships
    - Per-relationship approve / edit / remove / add (preferred), or a
      batch approve/reject
    - Only a batch reject routes back to normalize with feedback

AUTO-SKIP:
    Each gate has its own enable/disable flag in state["gates_enabled"]
    (a dict keyed by 'entities', 'chunks', 'procedures', 'bundle'; read it
    with is_gate_enabled()). When a gate's flag is False, that gate
    auto-approves all items and continues without pausing. Other gates run
    normally.

READS: gate{N}_reviews (raw analyst input), entities/drafts (upstream data)
WRITES: validated_entities, gate1_decisions, gate2_decision, etc.
"""

from __future__ import annotations

import copy
import hashlib
import logging
import uuid

from app.nodes.deterministic.attack_operators import splice_absent
from app.graph.state import (
    CHUNK_EDITABLE_FIELDS,
    ChunkGateRejectReason,
    ChunkProblemType,
    EntityType,
    GateAction,
    Gate1RejectReason,
    PipelineState,
    PipelineStatus,
    is_gate_enabled,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Gate (chunks): Chunk validation
# =============================================================================
#
# Sits between chunk_behaviors and extract_techniques. Three behaviors:
#   1. Auto-skip when gates_enabled["chunks"] is False -> approve all chunks.
#   2. Reject -> set chunks_rejection_routing="chunk_behaviors" so the graph
#      loops back, with chunk_rerun_feedback carrying analyst comments for
#      the chunk_behaviors retry prompt.
#   3. Approve (default) -> apply per-chunk decisions (approve/edit/drop) +
#      added chunks + edge mutations, then proceed to extract_techniques.
#
# chunk_reviews submission shape (set by API while gate is paused):
#   {
#     "decisions": [{chunk_id, action: approve|edit|drop, edits?: dict}, ...],
#     "added_chunks": [{text, source_excerpt, ...}, ...],   # analyst additions
#     "edges": [{action: add|remove, from: chunk_id, to: chunk_id}, ...],
#     "reject": {"reason": ChunkGateRejectReason, "comments": str} | None,
#   }
# When `reject` is present everything else is ignored — the gate routes back
# to chunk_behaviors with the comments attached.

# Whitelist of chunk fields the analyst can edit in-place. Shared with the
# API validator — see CHUNK_EDITABLE_FIELDS in app.graph.state. chunk_id /
# sequence_index / predecessor_indices / precedes_ids / source_span are
# managed by the gate processor (or derived) — not directly editable.
_CHUNK_EDITABLE_FIELDS = CHUNK_EDITABLE_FIELDS


def _next_sequence_index(chunks: list[dict]) -> int:
    """Highest existing sequence_index + 1, with sane fallback."""
    if not chunks:
        return 1
    return max(int(c.get("sequence_index", 0) or 0) for c in chunks) + 1


def _sync_predecessor_indices(chunks: list[dict]) -> None:
    """Rebuild every chunk's predecessor_indices from the final precedes_ids.

    The gate's edge, drop-rewire and merge handling all mutate `precedes_ids`
    only, but drafting copies `predecessor_indices` onto each draft, and the
    technique-gate flow editor, the bundle-gate relationship preview and the
    attack-flow start_refs fallback all read that copy. Until this ran, every
    edge the analyst drew here reached the bundle (the serializer walks
    `precedes_ids`) but not the two review surfaces after it — they kept
    describing the chunker's original graph. This is the exact reverse of the
    chunker's own inversion in `_finalize_chunks`. Assigns fresh lists: the
    approve pass-through appends the caller's dicts, so no shared list may be
    appended to in place.
    """
    id_to_seq: dict[str, int] = {}
    for chunk in chunks:
        cid = chunk.get("chunk_id")
        seq = chunk.get("sequence_index")
        if cid and isinstance(seq, int) and not isinstance(seq, bool):
            id_to_seq[cid] = seq
    preds: dict[str, set[int]] = {
        c["chunk_id"]: set() for c in chunks if c.get("chunk_id")
    }
    for chunk in chunks:
        src_seq = id_to_seq.get(chunk.get("chunk_id"))
        if src_seq is None:
            continue
        for tgt in chunk.get("precedes_ids") or []:
            if tgt in preds:
                preds[tgt].add(src_seq)
    for chunk in chunks:
        cid = chunk.get("chunk_id")
        if cid in preds:
            chunk["predecessor_indices"] = sorted(preds[cid])


def _apply_chunk_edits(chunk: dict, edits: dict) -> dict:
    """Return a copy of chunk with whitelisted fields overwritten."""
    out = copy.deepcopy(chunk)
    for k, v in edits.items():
        if k in _CHUNK_EDITABLE_FIELDS:
            out[k] = v
    return out


def _build_added_chunk(raw: dict, seq_idx: int) -> dict:
    """Construct a fresh chunk dict from analyst-supplied fields. Mirrors
    the shape produced by chunking._postprocess_raw_chunks so downstream
    nodes see uniform data.

    Uses a uuid4-based chunk_id rather than the deterministic
    `_new_chunk_id(seq_idx, text)` scheme: an analyst who drops chunk N
    then adds a new chunk with the same text would otherwise get the
    freed seq_idx reassigned (via _next_sequence_index) and collide with
    the dropped chunk's ID, so edges referencing the dropped chunk would
    silently resolve to the new one.
    Analyst-added chunks are new content, so there's no downstream LLM
    cache hit to preserve by keeping the ID deterministic.
    """
    text = (raw.get("text") or "").strip()
    return {
        "chunk_id": f"chk-{uuid.uuid4().hex[:12]}",
        "text": text,
        "context": raw.get("context", {}),
        "sequence_index": seq_idx,
        "predecessor_indices": [],
        "branch_point": bool(raw.get("branch_point", False)),
        "convergence_point": bool(raw.get("convergence_point", False)),
        "behavioral_confidence": max(0.0, min(1.0, float(raw.get("behavioral_confidence", 0.7)))),
        "source_location": {},
        "source_excerpt": (raw.get("source_excerpt") or "").strip(),
        "source_span": None,  # Could derive if parsed_text were re-passed; defer to next chunk_behaviors run.
        "precedes_ids": [],
    }


def _append_chunk_corrections(state: PipelineState, entries: list[dict]) -> list[dict]:
    """Read-modify-write append to the durable chunk_correction_log.

    Mirrors gate1_correction_log: the transient channels this gate writes
    (chunk_rerun_feedback, chunk_reviews) are cleared after consumption, and
    chunk_decisions is last-write-wins across re-chunk loops — so without
    this append-only ledger, a wholesale reject, an analyst-added chunk, or
    a pass-1 drop/edit never reaches synthesize_feedback at run completion.
    Entries are stored in the exact delta shape _chunk_deltas emits (text
    snapshotted at gate time, since chunks regenerate on loops).
    """
    return list(state.get("chunk_correction_log", []) or []) + entries


def _sequentiality_override(state: PipelineState, review: dict) -> dict:
    """Analyst override of `is_sequential`, submitted with the gate review.

    Entity extraction sets the flag once and nothing downstream re-derives
    it, so without this the analyst's only lever on a misread source was to
    re-ingest it. Returns the state keys to overwrite, or {} when the review
    carries no boolean. The rationale keeps the auto-detect reasoning and
    appends the override, so the later gate reviewers and the bundle's
    provenance see both.
    """
    flag = review.get("is_sequential")
    if not isinstance(flag, bool):
        return {}
    base = (state.get("sequentiality_rationale") or "").strip()
    note = (
        f"Overridden to {'sequential' if flag else 'non-sequential'} "
        "by the analyst at the procedure gate."
    )
    logger.info(
        "gate_chunks: analyst overrode is_sequential %s -> %s",
        state.get("is_sequential"), flag,
    )
    return {
        "is_sequential": flag,
        "sequentiality_rationale": f"{base} | {note}" if base else note,
    }


def gate_chunks(state: PipelineState) -> dict:
    """Process chunk review decisions from the analyst.

    See module-level comment for the chunk_reviews submission shape and
    routing semantics. Returns a state update dict with refreshed `chunks`,
    `chunk_decisions`, `chunks_approved_ids`, `chunks_rejection_routing`,
    optional `chunk_rerun_feedback`, plus the standard status/current_node.
    An `is_sequential` boolean on the review overrides the auto-detected
    flag on both the approve and reject paths (see _sequentiality_override).
    Every non-approve analyst signal is also appended to the durable
    chunk_correction_log (see _append_chunk_corrections).
    """
    chunks = list(state.get("chunks", []) or [])

    # Path 1: gate disabled -> auto-approve everything, no edits.
    if not is_gate_enabled(state, "chunks"):
        decisions = [
            {"chunk_id": c.get("chunk_id", ""), "action": GateAction.APPROVE.value}
            for c in chunks
        ]
        logger.info("gate_chunks auto-skip: approved %d chunks", len(decisions))
        return {
            "chunks": chunks,
            "chunk_decisions": decisions,
            "chunks_approved_ids": [d["chunk_id"] for d in decisions],
            "chunks_rejection_routing": None,
            "status": PipelineStatus.RESUMING_FROM_GATE_CHUNKS.value,
            "current_node": "gate_chunks",
        }

    review = state.get("chunk_reviews") or {}

    # Path 2: reject -> route back to chunk_behaviors with feedback.
    reject = review.get("reject")
    if reject:
        reason = reject.get("reason", ChunkGateRejectReason.OTHER.value)
        comments = (reject.get("comments") or "").strip()
        logger.info(
            "gate_chunks reject: routing back to chunk_behaviors (reason=%s, comments_chars=%d)",
            reason, len(comments),
        )
        return {
            "chunks": chunks,  # Keep prior chunks visible; chunk_behaviors will overwrite.
            "chunk_decisions": [],
            "chunks_approved_ids": [],
            "chunks_rejection_routing": "chunk_behaviors",
            "chunk_rerun_feedback": {"reason": reason, "comments": comments},
            # The rerun chunker's prompt depends on is_sequential, so a flip
            # submitted with the reject must reach it.
            **_sequentiality_override(state, review),
            # Durable copy: chunk_behaviors clears chunk_rerun_feedback after
            # consuming it, so this ledger entry is what reaches the flywheel.
            "chunk_correction_log": _append_chunk_corrections(state, [{
                "kind": "wholesale_reject",
                "reason": reason,
                "comments": comments[:500],
            }]),
            # Clear consumed reviews so the SECOND pass through gate_chunks
            # (after chunk_behaviors re-runs) doesn't see this stale reject
            # and re-route — would otherwise create an infinite-loop on the
            # rerun-chunking path.
            "chunk_reviews": {},
            "status": PipelineStatus.RESUMING_FROM_GATE_CHUNKS.value,
            "current_node": "gate_chunks",
        }

    # Path 3: approve with optional in-place edits.
    decisions_raw = review.get("decisions", []) or []
    added_chunks_raw = review.get("added_chunks", []) or []
    edges = review.get("edges", []) or []

    decision_map = {
        d["chunk_id"]: d for d in decisions_raw
        if isinstance(d, dict) and "chunk_id" in d
    }

    decisions: list[dict] = []
    new_chunks: list[dict] = []
    # Durable ledger entries for this pass, in the exact delta shape
    # _chunk_deltas emits. Text is snapshotted HERE because a later re-chunk
    # loop regenerates the chunks list — by synthesis time the chunk a
    # pass-1 decision referred to may no longer exist.
    correction_entries: list[dict] = []
    dropped_count = 0
    # chunk_id -> the successors it pointed at, captured before it is dropped.
    dropped_successors: dict[str, list[str]] = {}
    edited_count = 0

    # Expand `merge` into the primitives the loop below already handles:
    # the survivor gets an edit carrying the combined text, and each absorbed
    # chunk gets a drop (whose edges the splice pass then rewires onto the
    # survivor). Doing it here keeps merge a single analyst gesture without
    # duplicating the edit/drop machinery.
    chunk_by_id_pre = {c.get("chunk_id", ""): c for c in chunks}
    merged_away: set[str] = set()
    for cid, decision in list(decision_map.items()):
        if decision.get("action") != "merge":
            continue
        survivor = chunk_by_id_pre.get(cid)
        absorb_ids = [
            a for a in (decision.get("merge_with") or [])
            if a in chunk_by_id_pre and a != cid
        ]
        if not survivor or not absorb_ids:
            logger.warning(
                "gate_chunks: merge on %s has no resolvable partners; ignoring",
                cid,
            )
            decision_map[cid] = {"chunk_id": cid, "action": GateAction.APPROVE.value}
            continue

        absorbed = [chunk_by_id_pre[a] for a in absorb_ids]
        # An explicit `edits.text` wins — the analyst may have written the
        # merged wording themselves. Otherwise concatenate in sequence order.
        supplied = (decision.get("edits") or {}).get("text")
        ordered = sorted(
            [survivor, *absorbed],
            key=lambda c: c.get("sequence_index", 0),
        )
        merged_text = supplied or " ".join(
            (c.get("text") or "").strip() for c in ordered
            if (c.get("text") or "").strip()
        )
        merged_excerpt = (decision.get("edits") or {}).get("source_excerpt") or next(
            ((c.get("source_excerpt") or "").strip() for c in ordered
             if (c.get("source_excerpt") or "").strip()),
            "",
        )
        decision_map[cid] = {
            "chunk_id": cid,
            "action": GateAction.EDIT.value,
            "edits": {"text": merged_text, "source_excerpt": merged_excerpt},
            "rationale": decision.get("rationale") or (
                f"merged {', '.join(absorb_ids)} into this procedure"
            ),
        }
        for a in absorb_ids:
            merged_away.add(a)
            decision_map[a] = {
                "chunk_id": a,
                "action": GateAction.REMOVE.value,
                "rationale": f"merged into {cid}",
            }
        logger.info(
            "gate_chunks: merging %s into %s", ", ".join(absorb_ids), cid,
        )

    for chunk in chunks:
        cid = chunk.get("chunk_id", "")
        decision = decision_map.get(cid)

        if decision is None:
            # No explicit decision -> safe default is approve (analyst saw it,
            # chose not to act). Mirrors gate_0 / gate_1 behavior.
            new_chunks.append(chunk)
            decisions.append({"chunk_id": cid, "action": GateAction.APPROVE.value})
            continue

        action = decision.get("action", GateAction.APPROVE.value)
        if action == GateAction.REMOVE.value or action == "drop":
            dropped_count += 1
            # Remember what this chunk pointed at so the chain can be spliced
            # back together below rather than severed.
            dropped_successors[cid] = list(chunk.get("precedes_ids") or [])
            decisions.append({"chunk_id": cid, "action": GateAction.REMOVE.value})
            correction_entries.append({
                "chunk_id": cid,
                "original_text": (chunk.get("text", "") or "")[:300],
                "action": GateAction.REMOVE.value,
                "edits": None,
                "rationale": decision.get("rationale"),
            })
            continue
        if action == GateAction.EDIT.value:
            edits = decision.get("edits", {}) or {}
            kept_edits = {
                k: v for k, v in edits.items() if k in _CHUNK_EDITABLE_FIELDS
            }
            correction_entries.append({
                "chunk_id": cid,
                # Snapshot BEFORE applying so the ledger carries the LLM
                # original, not the analyst's replacement.
                "original_text": (chunk.get("text", "") or "")[:300],
                "action": GateAction.EDIT.value,
                "edits": kept_edits,
                "rationale": decision.get("rationale"),
            })
            edited = _apply_chunk_edits(chunk, edits)
            new_chunks.append(edited)
            edited_count += 1
            decisions.append({
                "chunk_id": cid, "action": GateAction.EDIT.value,
                "edits": kept_edits,
            })
            continue
        # Default and APPROVE both pass through.
        new_chunks.append(chunk)
        decisions.append({"chunk_id": cid, "action": GateAction.APPROVE.value})

    # Add analyst-supplied new chunks (assign new sequence_index + chunk_id).
    next_seq = _next_sequence_index(new_chunks)
    for i, raw in enumerate(added_chunks_raw):
        if not isinstance(raw, dict):
            continue
        added = _build_added_chunk(raw, next_seq + i)
        new_chunks.append(added)
        decisions.append({
            "chunk_id": added["chunk_id"], "action": "add",
        })
        # "The LLM missed this" is among the strongest chunking signals; the
        # transient chunk_reviews carrying it is cleared below.
        correction_entries.append({
            "kind": "analyst_added",
            "text": (raw.get("text", "") or "")[:300],
            "source_excerpt": (raw.get("source_excerpt") or "")[:200],
        })

    # Rewire around dropped chunks BEFORE applying analyst edge mutations, so
    # an explicit edit always wins over the automatic splice.
    #
    # Dropping a chunk used to leave two problems behind: survivors kept a
    # dangling `precedes_ids` entry pointing at the removed chunk, and the
    # chain was severed rather than bridged — predecessor and successor were
    # simply disconnected, so `_build_attack_flow` saw the successor as a new
    # root. Downstream code filters unresolvable targets, so it was latent
    # rather than fatal, but it meant the analyst had to repair the edge by
    # hand after every drop.
    chunk_by_id = {c["chunk_id"]: c for c in new_chunks}
    if dropped_successors:
        # Shared with the serializer via splice_absent. This used to be an
        # inline pass that kept a successor only when it had itself survived,
        # so two consecutive drops still severed the chain; the shared helper
        # walks through removed chunks transitively and fixes that here too.
        successors = {c["chunk_id"]: list(c.get("precedes_ids") or [])
                      for c in new_chunks}
        successors.update(dropped_successors)
        spliced, bridged, dropped_edges = splice_absent(
            successors, set(chunk_by_id),
        )
        for chunk in new_chunks:
            chunk["precedes_ids"] = spliced.get(chunk["chunk_id"], [])
        logger.info(
            "gate_chunks: rewired the flow around %d dropped chunk(s) "
            "(%d edge(s) bridged, %d dropped)",
            len(dropped_successors), bridged, dropped_edges,
        )

    edge_changes = 0
    for e in edges:
        if not isinstance(e, dict):
            continue
        # Prefer the Python-safe `from_` key (current route emits this);
        # fall back to legacy `from` for any in-flight checkpoint dumped
        # before the route switched to `from_`. See ChunkEdgeMutation in
        # schemas/api.py.
        src = e.get("from_") or e.get("from")
        dst = e.get("to")
        action = e.get("action")
        if not src or not dst or src not in chunk_by_id:
            continue
        # The target was never validated, so an "add" pointing at a dropped
        # chunk appended a dangling reference (previously accepted as a known
        # limitation). Removals still apply either way — pruning a stale edge
        # must keep working even when its target is gone.
        if action == "add" and dst not in chunk_by_id:
            logger.info(
                "gate_chunks: ignoring edge add %s -> %s (target not present)",
                src, dst,
            )
            continue
        precedes = chunk_by_id[src].setdefault("precedes_ids", [])
        if action == "add" and dst not in precedes:
            precedes.append(dst)
            edge_changes += 1
        elif action == "remove" and dst in precedes:
            precedes.remove(dst)
            edge_changes += 1

    # Operator-kind overrides. Persist as partial
    # dicts (kind only) keyed by operator_id. normalize merges these
    # into the freshly-inferred operator dict via infer_operators'
    # existing_operators contract — overrides whose operator_id no
    # longer exists post-edit silently drop. Carry forward any prior
    # operator overrides not touched in this submission.
    operator_overrides_raw = review.get("operator_overrides", []) or []
    operator_kinds: dict[str, dict] = dict(state.get("chunk_operators", {}) or {})
    for ovr in operator_overrides_raw:
        if not isinstance(ovr, dict):
            continue
        op_id = ovr.get("operator_id")
        kind = ovr.get("kind")
        if not op_id or kind not in {"AND", "OR", "XOR"}:
            continue
        existing = operator_kinds.get(op_id, {})
        operator_kinds[op_id] = {**existing, "kind": kind}

    # Condition edits. Keyed by chunk_id (the
    # condition's anchor). action="set" replaces / creates the
    # condition; action="clear" removes it. Re-validation of
    # on_true_ids / on_false_ids against current precedes_ids
    # happens at extract_conditions time (called from normalize and
    # the gate GET); the gate processor just persists the analyst's
    # raw intent into state.chunk_conditions.
    condition_edits_raw = review.get("condition_edits", []) or []
    chunk_conditions: dict[str, dict] = dict(state.get("chunk_conditions", {}) or {})
    for edit in condition_edits_raw:
        if not isinstance(edit, dict):
            continue
        chunk_id = edit.get("chunk_id")
        action = edit.get("action")
        if not chunk_id or action not in {"set", "clear"}:
            continue
        if action == "clear":
            chunk_conditions.pop(chunk_id, None)
            continue
        description = (edit.get("description") or "").strip()
        if not description:
            # Drop "set" with empty description — matches the
            # _finalize_chunks bias-toward-null rule.
            logger.warning(
                "gate_chunks: condition_edit on chunk %s has empty "
                "description; ignoring set action",
                chunk_id,
            )
            continue
        on_true_ids = [i for i in (edit.get("on_true_ids") or []) if isinstance(i, str)]
        on_false_ids = [i for i in (edit.get("on_false_ids") or []) if isinstance(i, str)]
        pattern = (edit.get("pattern") or "").strip() or None
        pattern_type = edit.get("pattern_type")
        if pattern_type not in {"stix", "regex", "plain"}:
            pattern_type = None
        if not pattern:
            pattern_type = None
        chunk_conditions[chunk_id] = {
            "description": description,
            "pattern": pattern,
            "pattern_type": pattern_type,
            "on_true_ids": on_true_ids,
            "on_false_ids": on_false_ids,
        }

    # Every mutation above touched precedes_ids only; bring the LLM-shaped
    # copy in line so drafting and the later gates see the same graph.
    _sync_predecessor_indices(new_chunks)

    approved_ids = [c["chunk_id"] for c in new_chunks]
    logger.info(
        "gate_chunks approve: %d -> %d chunks (edits=%d, drops=%d, adds=%d, edge_changes=%d, op_overrides=%d, cond_edits=%d)",
        len(chunks), len(new_chunks), edited_count, dropped_count,
        len(added_chunks_raw), edge_changes, len(operator_overrides_raw),
        len(condition_edits_raw),
    )

    return {
        "chunks": new_chunks,
        "chunk_decisions": decisions,
        "chunks_approved_ids": approved_ids,
        "chunks_rejection_routing": None,
        "chunk_operators": operator_kinds,
        "chunk_conditions": chunk_conditions,
        **_sequentiality_override(state, review),
        # Durable copy of this pass's non-approve signal (chunk_decisions is
        # last-write-wins across re-chunk loops; chunk_reviews is cleared).
        "chunk_correction_log": _append_chunk_corrections(state, correction_entries),
        # Clear consumed reviews — see reject path above for rationale.
        "chunk_reviews": {},
        "status": PipelineStatus.RESUMING_FROM_GATE_CHUNKS.value,
        "current_node": "gate_chunks",
    }


# =============================================================================
# Gate 0: Entity review
# =============================================================================

def gate_0(state: PipelineState) -> dict:
    """Process entity review decisions from the analyst.

    If the entity gate is disabled, auto-approves all entities.
    Otherwise, applies analyst decisions from gate0_reviews to produce
    validated_entities.

    Each review in gate0_reviews is a dict:
        entity_id: str          (required)
        action: str             (required, GateAction value)
        edited_value: str|None  (for edit actions)
        edited_type: str|None   (for type corrections)
        rationale: str|None     (analyst's reasoning)

    Entities without a matching review are auto-approved (safe default:
    the analyst saw them and chose not to act, meaning they're fine).
    """
    entities = state.get("entities", [])

    if not is_gate_enabled(state, "entities"):
        # Auto-approve: copy all entities with gate_action=approve
        validated = _auto_approve_entities(entities)
        logger.info("Gate 0 auto-skip: approved %d entities", len(validated))
        return {
            "validated_entities": validated,
            "status": PipelineStatus.RESUMING_FROM_GATE_0.value,
            "current_node": "gate_0",
        }

    reviews = state.get("gate0_reviews", [])
    review_map = {r["entity_id"]: r for r in reviews if "entity_id" in r}

    validated = []
    removed_count = 0
    edited_count = 0

    for entity in entities:
        eid = entity.get("entity_id", "")
        review = review_map.get(eid)

        if review is None:
            # Denylisted entity the analyst didn't touch -> auto-remove
            # (deterministic guardrail from a promoted_to_denylist pattern).
            # An explicit review below would override this (analyst keep).
            if entity.get("denylisted"):
                logger.info(
                    "Gate 0: denylist auto-removed %s=%r (pattern %s)",
                    entity.get("entity_type"), entity.get("value"),
                    entity.get("denylist_pattern_id"),
                )
                updated = _apply_entity_action(
                    entity, GateAction.REMOVE.value,
                    rationale=entity.get("denylist_reason"),
                )
                validated.append(updated)
                removed_count += 1
                continue
            # Sub-threshold confidence the analyst didn't touch -> auto-remove.
            # Same shape as the denylist above and equally overridable: an
            # explicit review wins. The model flagged these as guesses and
            # nothing downstream was acting on the signal — a 0.3 entity
            # serialized identically to a 1.0 one.
            if entity.get("low_confidence"):
                logger.info(
                    "Gate 0: low-confidence auto-removed %s=%r (%s)",
                    entity.get("entity_type"), entity.get("value"),
                    entity.get("low_confidence_reason"),
                )
                updated = _apply_entity_action(
                    entity, GateAction.REMOVE.value,
                    rationale=entity.get("low_confidence_reason"),
                )
                validated.append(updated)
                removed_count += 1
                continue
            # No explicit review -> auto-approve
            updated = _apply_entity_action(entity, GateAction.APPROVE.value)
            validated.append(updated)
            continue

        action = review.get("action", GateAction.APPROVE.value)

        if not _is_valid_gate_action(action):
            logger.warning(
                "Gate 0: invalid action '%s' for entity %s, defaulting to approve",
                action, eid,
            )
            action = GateAction.APPROVE.value

        if action == GateAction.REMOVE.value:
            # Mark as removed but still include in validated_entities
            # for audit trail. Downstream nodes filter on gate_action.
            updated = _apply_entity_action(entity, action, rationale=review.get("rationale"))
            validated.append(updated)
            removed_count += 1
        elif action == GateAction.EDIT.value:
            updated = _apply_entity_edit(entity, review)
            validated.append(updated)
            edited_count += 1
        else:
            # approve or reject (reject at Gate 0 = just approve with note)
            updated = _apply_entity_action(entity, action, rationale=review.get("rationale"))
            validated.append(updated)

    logger.info(
        "Gate 0: %d entities processed (%d edited, %d removed, %d approved)",
        len(validated), edited_count, removed_count,
        len(validated) - edited_count - removed_count,
    )

    # Entities the analyst added because extraction missed them. Appended
    # after the review loop so they cannot collide with a decision, and
    # pre-approved because the act of adding one IS the review.
    added_count = 0
    for raw in state.get("gate0_added_entities", []) or []:
        if not isinstance(raw, dict) or not (raw.get("value") or "").strip():
            continue
        validated.append(_build_added_entity(raw))
        added_count += 1
    if added_count:
        logger.info("Gate 0: analyst added %d entities", added_count)

    return {
        "validated_entities": validated,
        # Clear consumed reviews so a future rerun loop doesn't pick up
        # this stale set (matches the gate_chunks / gate_1 hygiene).
        "gate0_reviews": [],
        "gate0_added_entities": [],
        "status": PipelineStatus.RESUMING_FROM_GATE_0.value,
        "current_node": "gate_0",
    }


_ADDED_ENTITY_FIELDS = (
    "value", "entity_type", "organization_role", "location_role",
)


def _build_added_entity(raw: dict) -> dict:
    """Build a validated-entity dict from an analyst's addition.

    Mirrors `_build_added_chunk`. The entity enters pre-approved — the
    analyst adding it IS the review — and is stamped `analyst_added` so its
    provenance is never confused with an extractor output. A fresh uuid4 id
    avoids colliding with the extractor's `ent-` ids.
    """
    entity: dict = {
        "entity_id": f"ent-{uuid.uuid4().hex[:8]}",
        "value": (raw.get("value") or "").strip(),
        "entity_type": (raw.get("entity_type") or "").strip(),
        "confidence": max(0.0, min(1.0, float(raw.get("confidence", 1.0)))),
        "context_snippet": "",
        "gate_action": GateAction.APPROVE.value,
        "edited_value": None,
        "edited_type": None,
        "edit_rationale": raw.get("rationale"),
        "analyst_added": True,
    }
    for field in ("organization_role", "location_role"):
        if raw.get(field):
            entity[field] = raw[field]
    if entity["entity_type"] == EntityType.INTRUSION_SET.value:
        # No sponsorship claim for an analyst-added cluster: the serializer
        # attributes only from this list (or the single-actor fallback).
        entity["attributed_to"] = []
    return entity


def _auto_approve_entities(entities: list[dict]) -> list[dict]:
    """Auto-approve all entities (disabled-gate path).

    Denylisted and sub-threshold-confidence entities are auto-removed even
    when the gate is disabled — both are deterministic guardrails rather than
    review prompts, so they fire whether or not the entity gate is active.
    """
    out: list[dict] = []
    for e in entities:
        if e.get("denylisted"):
            # Gate disabled => no analyst review surface, so log the silent
            # removal (a too-broad denylist could drop a legit entity here
            # with no UI trace — this is the breadcrumb).
            logger.info(
                "Gate 0 (disabled): denylist auto-removed %s=%r (pattern %s)",
                e.get("entity_type"), e.get("value"), e.get("denylist_pattern_id"),
            )
            out.append(_apply_entity_action(
                e, GateAction.REMOVE.value, rationale=e.get("denylist_reason"),
            ))
        elif e.get("low_confidence"):
            # Same reasoning as the denylist above: no review surface here,
            # so leave a breadcrumb for the silent removal.
            logger.info(
                "Gate 0 (disabled): low-confidence auto-removed %s=%r (%s)",
                e.get("entity_type"), e.get("value"),
                e.get("low_confidence_reason"),
            )
            out.append(_apply_entity_action(
                e, GateAction.REMOVE.value,
                rationale=e.get("low_confidence_reason"),
            ))
        else:
            out.append(_apply_entity_action(e, GateAction.APPROVE.value))
    return out


def _apply_entity_action(
    entity: dict,
    action: str,
    rationale: str | None = None,
) -> dict:
    """Apply a gate action to an entity, returning a new dict."""
    updated = copy.deepcopy(entity)
    updated["gate_action"] = action
    if rationale:
        updated["edit_rationale"] = rationale
    return updated


_VALID_ORG_ROLES = {"victim", "sponsor", "publisher", "author", "other"}
_VALID_LOCATION_ROLES = {"victim", "origin", "context"}


def _apply_entity_edit(entity: dict, review: dict) -> dict:
    """Apply an edit review to an entity.

    Handles edits for value, type, and role. Role edits route based on the
    entity's effective type — organization picks update organization_role,
    location picks update location_role, others ignore the role edit. The
    role enum is also validated; an unknown value drops the edit silently
    rather than corrupting state.
    """
    updated = copy.deepcopy(entity)
    updated["gate_action"] = GateAction.EDIT.value

    if review.get("edited_value") is not None:
        updated["edited_value"] = review["edited_value"]
    if review.get("edited_type") is not None:
        updated["edited_type"] = review["edited_type"]
    edited_role = review.get("edited_role")
    if edited_role:
        # Effective type after edit: prefer the analyst's correction so
        # "type=organization, edited_type=location, edited_role=origin"
        # writes location_role rather than organization_role.
        effective_type = (
            updated.get("edited_type")
            or updated.get("entity_type")
            or ""
        )
        if effective_type == "organization" and edited_role in _VALID_ORG_ROLES:
            updated["organization_role"] = edited_role
        elif effective_type == "location" and edited_role in _VALID_LOCATION_ROLES:
            updated["location_role"] = edited_role
        # Silently ignore role edits on other entity types or for invalid
        # role values; the analyst's intent is unclear and logging on
        # every gate-0 submit would be noisy.
    if review.get("rationale"):
        updated["edit_rationale"] = review["rationale"]

    return updated


def _is_valid_gate_action(action: str) -> bool:
    """Check if a string is a valid GateAction value."""
    return action in {a.value for a in GateAction}


# =============================================================================
# Gate 1: Procedure review
# =============================================================================

def gate_1(state: PipelineState) -> dict:
    """Process procedure review decisions from the analyst.

    If the procedure gate is disabled, auto-approves all drafts.
    Otherwise, applies analyst decisions from gate1_reviews to produce
    gate1_decisions, gate1_approved_draft_ids, and gate1_rejection_routing.

    Each review in gate1_reviews is a dict:
        draft_id: str                (required)
        action: str                  (required, GateAction value)
        reject_reason: str|None      (Gate1RejectReason value, for reject)
        analyst_edits: dict|None     ({field_name: new_value} for edit)
        rationale: str|None          (analyst's reasoning)

    Rejection routing priority:
        BAD_CHUNK_BOUNDARY -> "chunk_behaviors" (highest)
        any other reject   -> "extract_techniques"
        no rejections      -> None (proceed to normalize)
    """
    drafts = state.get("drafts", [])

    if not is_gate_enabled(state, "procedures"):
        decisions, approved_ids = _auto_approve_drafts(drafts)
        # Auto-promote for-review picks for drafts that would otherwise
        # ship with empty x_technique_refs. Covers the case where
        # every pick for a chunk landed in the `possible` bucket and
        # routed to the analyst-promote lane. Without this, unattended
        # runs lose those chunks' techniques and the validator hard-fails
        # on x_procedure_missing_required_field.
        review_mappings = {
            k: list(v) for k, v in
            (state.get("technique_mappings_for_review") or {}).items()
        }
        bundle_mappings = {
            k: list(v) for k, v in
            (state.get("technique_mappings") or {}).items()
        }
        auto_promotions = _auto_promote_orphans(
            drafts, bundle_mappings, review_mappings,
        )
        result = {
            "gate1_decisions": decisions,
            "gate1_approved_draft_ids": approved_ids,
            "gate1_rejection_routing": None,
            "status": PipelineStatus.RESUMING_FROM_GATE_1.value,
            "current_node": "gate_1",
        }
        if auto_promotions:
            bundle_mappings, review_mappings, drafts, applied = _apply_promotions(
                auto_promotions,
                bundle_mappings=bundle_mappings,
                review_mappings=review_mappings,
                drafts=drafts,
            )
            result["technique_mappings"] = bundle_mappings
            result["technique_mappings_for_review"] = review_mappings
            result["drafts"] = drafts
            logger.info(
                "Gate 1 auto-skip: approved %d drafts; auto-promoted %d "
                "possible-bucket pick(s) to fill empty technique mappings",
                len(approved_ids), applied,
            )
        else:
            logger.info("Gate 1 auto-skip: approved %d drafts", len(approved_ids))
        return result

    reviews = state.get("gate1_reviews", [])
    # Drafts whose sequencing / chain fields the analyst edited; their chunks
    # are re-synced below so the bundle follows the edit.
    sequencing_edited: set[str] = set()
    review_map = {r["draft_id"]: r for r in reviews if "draft_id" in r}

    decisions = []
    approved_ids = []
    reject_reasons = []
    # Self-contained audit of every non-approve decision THIS pass. Appended
    # to the durable gate1_correction_log so reject feedback survives the
    # loop-and-overwrite and reaches synthesize_feedback at run completion.
    correction_records: list[dict] = []

    for draft in drafts:
        did = draft.get("draft_id", "")
        review = review_map.get(did)

        if review is None:
            # No explicit review -> auto-approve
            decisions.append(_build_decision(did, GateAction.APPROVE.value))
            approved_ids.append(did)
            continue

        action = review.get("action", GateAction.APPROVE.value)

        if not _is_valid_gate_action(action):
            logger.warning(
                "Gate 1: invalid action '%s' for draft %s, defaulting to approve",
                action, did,
            )
            action = GateAction.APPROVE.value

        decision = _build_decision(
            draft_id=did,
            action=action,
            reason=review.get("reject_reason"),
            edits=review.get("analyst_edits"),
            rationale=review.get("rationale"),
        )
        decisions.append(decision)

        if action == GateAction.APPROVE.value:
            approved_ids.append(did)
        elif action == GateAction.EDIT.value:
            # Edited drafts are also approved (with modifications).
            approved_ids.append(did)
            # Log BEFORE applying so rejected_techniques captures the LLM
            # output (the "before"), not the applied edit.
            correction_records.append(_correction_record(draft, review, action))
            _apply_draft_edits(draft, review)
            if set((review.get("analyst_edits") or {}).keys()) & _SEQUENCING_EDIT_FIELDS:
                sequencing_edited.add(did)
        elif action == GateAction.REJECT.value:
            correction_records.append(_correction_record(draft, review, action))
            reason = review.get("reject_reason", "")
            # A reject ALWAYS routes backward — it must never silently
            # advance the bundle, so a missing reason still loops. There is
            # deliberately no apply-and-proceed shortcut for a reject that
            # carries an inline technique correction (the UI never sends
            # one; the verdict model makes "approve + edited techniques"
            # the fix path): the correction instead rides into the rerun
            # feedback as a strong hint, keeping reject semantics uniform.
            reject_reasons.append(reason or "unspecified")
        elif action == GateAction.REMOVE.value:
            correction_records.append(_correction_record(draft, review, action))
        # REMOVE: not added to approved_ids, draft is dropped

    # Determine routing from rejections.
    routing = _compute_rejection_routing(reject_reasons)

    removed = sum(1 for d in decisions if d["action"] == GateAction.REMOVE.value)
    rejected = sum(1 for d in decisions if d["action"] == GateAction.REJECT.value)
    logger.info(
        "Gate 1: %d drafts processed (%d approved, %d rejected, %d removed). Routing: %s",
        len(decisions), len(approved_ids), rejected, removed,
        routing or "normalize",
    )

    result = {
        "gate1_decisions": decisions,
        "gate1_approved_draft_ids": approved_ids,
        "gate1_rejection_routing": routing,
        # Clear consumed reviews so a rerun loop (gate_1 reject →
        # chunk_behaviors → ... → gate_1 again) doesn't reprocess this
        # stale set when LangGraph schedules gate_1 the second time.
        "gate1_reviews": [],
        # Append (read-modify-write) so rejects from a prior pass survive the
        # last-write-wins overwrite that erases gate1_decisions. Consumed by
        # synthesize_feedback at run completion.
        "gate1_correction_log": (
            list(state.get("gate1_correction_log", []) or []) + correction_records
        ),
        "status": PipelineStatus.RESUMING_FROM_GATE_1.value,
        "current_node": "gate_1",
    }

    # Corrective feedback for a wrong_technique re-extract: pass the analyst's
    # rejection rationale + which T-IDs were wrong (and any correction) into
    # extract_techniques so the rerun is informed. extract_techniques also
    # bypasses the LLM cache when this is present, so even a repeat reject
    # with identical wording gets a fresh extraction.
    if routing == "extract_techniques":
        result["technique_rerun_feedback"] = _build_technique_rerun_feedback(
            correction_records,
        )

    # When routing back to chunk_behaviors, build structured chunk feedback
    # and preserve technique overrides from approved/edited drafts.
    if routing == "chunk_behaviors":
        chunks = state.get("chunks", [])
        chunk_lookup = {c.get("chunk_id", ""): c for c in chunks}

        result["chunk_feedback"] = _build_chunk_feedback(
            reviews, review_map, drafts, chunk_lookup,
        )
        result["previous_technique_overrides"] = _build_technique_overrides(
            drafts, review_map, chunk_lookup,
        )

    # Apply analyst promotions of possible-bucket picks from the review lane.
    # Promotions move a pick from technique_mappings_for_review into the
    # active technique_mappings AND add it to the matching draft's techniques
    # list so downstream serialization carries the promoted technique. Done
    # last so it survives even when a draft was edited above (the in-place
    # mutation here updates the same dict objects).
    bundle = dict(state.get("technique_mappings", {}) or {})
    review_lane = dict(state.get("technique_mappings_for_review", {}) or {})
    updated_drafts = drafts
    mutated = False

    promotions = state.get("gate1_promotions", []) or []
    if promotions:
        bundle, review_lane, updated_drafts, applied = _apply_promotions(
            promotions=promotions,
            bundle_mappings=bundle,
            review_mappings=review_lane,
            drafts=updated_drafts,
        )
        # Clear consumed promotions so a stale list doesn't reapply on a
        # second pass (e.g. when the analyst rejects + re-runs Gate 1).
        result["gate1_promotions"] = []
        # Log each promotion durably IN THE SAME UPDATE that clears the
        # transient list — without this, promotions never reached
        # synthesize_feedback or the captured panel (the same clear-after-
        # consumption erasure the correction log fixes for rejects). Only
        # ANALYST promotions are logged; the safety-net auto-promote below
        # is machine-originated and carries no analyst signal.
        result["gate1_correction_log"] = result["gate1_correction_log"] + [
            {
                "action": "promote",
                "chunk_id": p.get("chunk_id", ""),
                "technique_id": p.get("technique_id", ""),
            }
            for p in promotions
            if isinstance(p, dict) and p.get("technique_id")
        ]
        logger.info(
            "Gate 1: applied %d analyst promotions (possible -> probable)",
            applied,
        )
        mutated = True

    # Safety-net auto-promote for orphan drafts the analyst approved without
    # explicitly promoting any review-lane pick. Same logic as the auto-skip
    # path: a draft is "orphaned" when its chunk has no entries in
    # technique_mappings but does have entries in
    # technique_mappings_for_review. Without this, those drafts ship with
    # empty x_technique_refs and the validator hard-fails on
    # x_procedure_missing_required_field. Restricted to approved drafts so
    # rejected/removed drafts don't get a phantom technique.
    approved_set = set(approved_ids)
    approved_drafts = [d for d in updated_drafts if d.get("draft_id", "") in approved_set]
    safety_promotions = _auto_promote_orphans(
        approved_drafts, bundle, review_lane,
    )
    if safety_promotions:
        bundle, review_lane, updated_drafts, applied = _apply_promotions(
            promotions=safety_promotions,
            bundle_mappings=bundle,
            review_mappings=review_lane,
            drafts=updated_drafts,
        )
        logger.info(
            "Gate 1: auto-promoted %d possible-bucket pick(s) to fill empty "
            "technique mappings on analyst-approved orphan drafts",
            applied,
        )
        mutated = True

    if mutated:
        result["technique_mappings"] = bundle
        result["technique_mappings_for_review"] = review_lane
        result["drafts"] = updated_drafts

    if sequencing_edited and state.get("chunks"):
        result["chunks"] = _mirror_sequencing_onto_chunks(
            list(state.get("chunks") or []), drafts, set(approved_ids),
        )
        logger.info(
            "Gate 1: mirrored sequencing/chain edits from %d draft(s) onto %d chunks",
            len(sequencing_edited), len(result["chunks"]),
        )

    return result


def _apply_promotions(
    promotions: list[dict],
    bundle_mappings: dict[str, list[dict]],
    review_mappings: dict[str, list[dict]],
    drafts: list[dict],
) -> tuple[dict[str, list[dict]], dict[str, list[dict]], list[dict], int]:
    """Move analyst-promoted possible-bucket picks into the bundle.

    For each `{chunk_id, technique_id}` promotion:
      - Find the matching pick in `review_mappings[chunk_id]`.
      - Flip its bucket to 'probable', floor confidence at 0.7, set
        `analyst_promoted=True`.
      - Move it into `bundle_mappings[chunk_id]` (skipping if a duplicate
        T-ID is already present so the analyst can't double-add).
      - Find the draft whose `chunk_id` matches and append the technique
        to its `techniques` list (same dedup safety).

    Returns the updated mappings, drafts, and applied-count for logging.
    Promotions whose chunk_id or technique_id don't resolve are dropped
    silently with a warning — the gate has already passed validation, no
    point hard-failing on a stale UI submission.

    All returned containers are new objects; the input state's drafts /
    mappings are not mutated. LangGraph checkpointers can cache state
    references between supersteps, so in-place mutation risks leaking
    updates into checkpoint versions that were supposed to be frozen.
    """
    # Track per-chunk new technique entries to append at draft-rebuild time.
    pending_appends: dict[str, list[dict]] = {}
    # Work on shallow copies of the mapping dicts; values themselves are
    # rebuilt as new lists when touched (see below).
    bundle_mappings = dict(bundle_mappings)
    review_mappings = dict(review_mappings)

    applied = 0
    for promo in promotions:
        chunk_id = promo.get("chunk_id", "")
        tid = promo.get("technique_id", "")
        if not chunk_id or not tid:
            logger.warning("Gate 1 promotion: skipping invalid entry %r", promo)
            continue

        review_picks = review_mappings.get(chunk_id, [])
        match_idx = next(
            (i for i, p in enumerate(review_picks)
             if p.get("technique_id") == tid),
            None,
        )
        if match_idx is None:
            logger.warning(
                "Gate 1 promotion: chunk %s has no review-lane pick for %s; skipping",
                chunk_id, tid,
            )
            continue

        pick = dict(review_picks[match_idx])
        pick["confidence_bucket"] = "probable"
        pick["confidence"] = max(pick.get("confidence", 0.7), 0.7)
        pick["analyst_promoted"] = True

        # Move out of review lane.
        new_review = review_picks[:match_idx] + review_picks[match_idx + 1:]
        if new_review:
            review_mappings[chunk_id] = new_review
        else:
            review_mappings.pop(chunk_id, None)

        # Add to bundle (dedup by technique_id).
        bundle_picks = list(bundle_mappings.get(chunk_id, []))
        if not any(p.get("technique_id") == tid for p in bundle_picks):
            bundle_picks.append(pick)
            bundle_mappings[chunk_id] = bundle_picks

        # Queue a draft technique-list append. We rebuild drafts as a new
        # list below so state.drafts is never mutated in place.
        pending_appends.setdefault(chunk_id, []).append({
            "technique_id": pick.get("technique_id"),
            "technique_name": pick.get("technique_name", ""),
            "tactic": pick.get("tactic", ""),
            "confidence": pick.get("confidence", 0.7),
            "rationale": pick.get("rationale", ""),
            "stix_id": pick.get("stix_id"),
            "provenance": "analyst_promoted",
        })

        applied += 1

    # Rebuild drafts as a new list. Only drafts whose chunk_id has pending
    # appends get a fresh dict; others pass through by reference (they're
    # unchanged, so identity reuse is safe).
    if pending_appends:
        touched_chunks = set(pending_appends)
        rebuilt: list[dict] = []
        seen_chunks: set[str] = set()
        for draft in drafts:
            cid = draft.get("chunk_id", "")
            if cid in touched_chunks:
                seen_chunks.add(cid)
                existing = list(draft.get("techniques", []) or [])
                existing_tids = {t.get("technique_id") for t in existing}
                for new_t in pending_appends[cid]:
                    if new_t.get("technique_id") not in existing_tids:
                        existing.append(new_t)
                        existing_tids.add(new_t.get("technique_id"))
                rebuilt.append({**draft, "techniques": existing})
            else:
                rebuilt.append(draft)
        # Only a real mismatch is worth warning about. An empty `drafts` means
        # the caller has no drafts YET — extract_techniques replays promotions
        # after a Gate 1 rewind, before draft_procedures runs — and drafting
        # builds its techniques from technique_mappings, which this has already
        # updated. Warning there reported a problem that does not exist, once
        # per promotion, on every rewind.
        if drafts:
            for cid in touched_chunks - seen_chunks:
                logger.warning(
                    "Gate 1 promotion: no draft found for chunk %s; picks added "
                    "to technique_mappings but not to any draft",
                    cid,
                )
        drafts = rebuilt

    return bundle_mappings, review_mappings, drafts, applied


def _auto_promote_orphans(
    drafts: list[dict],
    bundle_mappings: dict[str, list[dict]],
    review_mappings: dict[str, list[dict]],
) -> list[dict]:
    """Build a promotion list that fills empty bundle_mappings from
    the for-review lane.

    A draft is "orphaned" when its chunk has no entries in
    bundle_mappings (i.e. extract_techniques bucketed every pick as
    `possible` and routed them to the review lane). For each such
    chunk, promote the highest-confidence for-review pick — that
    becomes the draft's sole bundle technique on auto-skip.

    Returns a list of `{chunk_id, technique_id}` promotion dicts
    consumable by _apply_promotions. Empty list when nothing needs
    promoting (the common case when the LLM picked at least one
    definite/probable per chunk).
    """
    promotions: list[dict] = []
    seen_chunks: set[str] = set()
    for draft in drafts:
        chunk_id = draft.get("chunk_id", "")
        if not chunk_id or chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)
        # Skip if the chunk already has a definite/probable pick.
        if bundle_mappings.get(chunk_id):
            continue
        review_picks = review_mappings.get(chunk_id, [])
        if not review_picks:
            continue
        # Highest-confidence pick wins (numeric confidence — falls back
        # to 0 when missing so the comparison stays stable).
        best = max(
            review_picks,
            key=lambda p: float(p.get("confidence") or 0.0),
        )
        tid = best.get("technique_id")
        if tid:
            promotions.append({"chunk_id": chunk_id, "technique_id": tid})
    return promotions


def _auto_approve_drafts(drafts: list[dict]) -> tuple[list[dict], list[str]]:
    """Auto-approve all drafts (disabled-gate path)."""
    decisions = []
    approved_ids = []
    for draft in drafts:
        did = draft.get("draft_id", "")
        decisions.append(_build_decision(did, GateAction.APPROVE.value))
        approved_ids.append(did)
    return decisions, approved_ids


def _build_decision(
    draft_id: str,
    action: str,
    reason: str | None = None,
    edits: dict | None = None,
    rationale: str | None = None,
) -> dict:
    """Build a gate1_decisions entry."""
    decision = {
        "draft_id": draft_id,
        "action": action,
    }
    if reason:
        decision["reason"] = reason
    if edits:
        decision["edits"] = edits
    if rationale:
        decision["rationale"] = rationale
    return decision


_SEQUENCING_EDIT_FIELDS = frozenset({
    "sequence_index", "predecessor_indices", "chain_root", "chain_label",
})


def _mirror_sequencing_onto_chunks(
    chunks: list[dict], drafts: list[dict], approved_ids: set[str],
) -> list[dict]:
    """Copy approved drafts' sequencing and chain fields back onto the chunks.

    The technique gate's flow editor edits sequence_index / predecessor_indices
    (and chain_root / chain_label) on DRAFTS. normalize reads those for the
    relationship preview and the start_refs fallback, but the bundle's PRECEDES
    edges and operators are routed from the chunks' precedes_ids — so without
    this mirror a reorder showed in every review surface and never reached the
    bundle. Mirroring also carries the edit across a WRONG_TECHNIQUE rerun,
    which re-drafts from state["chunks"].

    Sequence indices come from the approved drafts, not the chunks: the flow
    editor renumbers drafts 1..N and remaps predecessors through the new
    numbering, so the chunks' old sequence_index must not drive the inversion.
    predecessor_indices are sanitized here because nothing upstream checks
    them: ints only, no self-reference, no unknown sequence. Chunks without an
    approved draft (removed, rejected) are returned untouched; the caller's
    dicts are never mutated.
    """
    approved = [
        d for d in drafts
        if d.get("draft_id") in approved_ids and d.get("chunk_id")
    ]
    seq_to_chunk_id: dict[int, str] = {}
    for d in approved:
        seq = d.get("sequence_index")
        if isinstance(seq, int) and not isinstance(seq, bool):
            seq_to_chunk_id[seq] = d["chunk_id"]
    draft_by_chunk = {d["chunk_id"]: d for d in approved}

    updated: list[dict] = []
    preds_by_chunk: dict[str, list[int]] = {}
    for chunk in chunks:
        cid = chunk.get("chunk_id")
        draft = draft_by_chunk.get(cid)
        if draft is None:
            updated.append(chunk)
            continue
        out = dict(chunk)
        seq = draft.get("sequence_index")
        if isinstance(seq, int) and not isinstance(seq, bool):
            out["sequence_index"] = seq
        preds = sorted({
            p for p in (draft.get("predecessor_indices") or [])
            if isinstance(p, int) and not isinstance(p, bool)
            and p in seq_to_chunk_id and p != seq
        })
        out["predecessor_indices"] = preds
        for key in ("chain_root", "chain_label"):
            if key in draft:
                out[key] = draft[key]
        updated.append(out)
        preds_by_chunk[cid] = preds

    # Invert the sanitized predecessors into forward edges; only chunks with
    # an approved draft get a rebuilt precedes_ids.
    forward: dict[str, list[str]] = {cid: [] for cid in preds_by_chunk}
    for out in updated:
        cid = out.get("chunk_id")
        for p in preds_by_chunk.get(cid, []):
            src = seq_to_chunk_id[p]
            if src in forward and cid not in forward[src]:
                forward[src].append(cid)
    for out in updated:
        cid = out.get("chunk_id")
        if cid in forward:
            out["precedes_ids"] = forward[cid]
    return updated


def _apply_draft_edits(draft: dict, review: dict) -> None:
    """Apply analyst edits to a draft in-place.

    The analyst_edits dict maps field names to new values.
    Only known ProcedureDraft fields are applied.
    """
    edits = review.get("analyst_edits")
    if not edits:
        return

    # Fields the analyst can edit at Gate 1.
    # raw_command_lines: verbatim commands extracted from source (staging data).
    # The serializer converts these to Process SCOs at bundle time.
    # techniques: analyst can add/remove/edit technique mappings inline.
    # sequence_index / predecessor_indices: the Gate 2 flow editor's
    # reorders. normalize reads both off the approved drafts for the
    # relationship preview and the start_refs fallback — but the bundle's
    # PRECEDES edges and operators are routed from the CHUNKS' precedes_ids,
    # so gate_1 mirrors these edits back onto state.chunks (see
    # _mirror_sequencing_onto_chunks) or the reorder would show in the
    # review and vanish from the bundle. branch_point / convergence_point
    # ride along as the same editor's flags; chain_root / chain_label are
    # the chain-separation flags the serializer reads for start_refs and
    # x_chain_label, mirrored the same way.
    _EDITABLE_FIELDS = {
        "name", "description", "platforms", "raw_command_lines",
        "confidence", "first_observed", "last_observed",
        "techniques",
        "sequence_index", "predecessor_indices",
        "branch_point", "convergence_point",
        "chain_root", "chain_label",
    }

    for field_name, new_value in edits.items():
        if field_name in _EDITABLE_FIELDS:
            draft[field_name] = new_value
        else:
            logger.warning("Gate 1: ignoring edit to non-editable field '%s'", field_name)

    # When techniques are edited, rebuild kill_chain_phases from the new
    # technique list so serialization stays consistent.
    if "techniques" in edits:
        techniques = edits["techniques"]
        seen_phases = set()
        kill_chain_phases = []
        for t in techniques:
            tactic = t.get("tactic", "")
            if tactic and tactic not in seen_phases:
                seen_phases.add(tactic)
                kill_chain_phases.append({
                    "kill_chain_name": "mitre-attack",
                    "phase_name": tactic,
                })
        draft["kill_chain_phases"] = kill_chain_phases
        logger.info(
            "Gate 1: rebuilt kill_chain_phases (%d phases) from edited techniques",
            len(kill_chain_phases),
        )

    draft["gate_action"] = GateAction.EDIT.value
    draft["analyst_edits"] = edits
    if review.get("rationale"):
        draft["analyst_rationale"] = review["rationale"]


def _slim_techniques(techniques: list | None) -> list[dict]:
    """Compact a draft/edit technique list to {id, name, tactic} entries.

    Used for the durable correction log + the re-extract rerun feedback,
    where we only need the technique identity (not the full pick payload)
    to (a) teach the flywheel what was wrong/right and (b) hint the LLM on
    rerun. Defensive against non-dict entries.
    """
    out: list[dict] = []
    for t in techniques or []:
        if not isinstance(t, dict):
            continue
        out.append({
            "technique_id": t.get("technique_id", ""),
            "technique_name": t.get("technique_name", ""),
            "tactic": t.get("tactic", ""),
        })
    return out


def _correction_record(draft: dict, review: dict, action: str) -> dict:
    """Build one self-contained Gate 1 correction-log entry.

    Self-contained on purpose: draft_ids are regenerated on every
    re-extract loop, so by the time synthesize_feedback runs at completion
    the original draft no longer exists. We snapshot draft_name, chunk_id,
    and the technique signal here so the lesson survives the loop. Call
    BEFORE _apply_draft_edits so the snapshot reflects the LLM output, not
    the applied edit.

    Technique fields are per-action honest — downstream consumers (the
    synthesis digest, MISS-scoring anchors, the rerun prompt, the captured
    panel) take them at face value, so a technique must appear in
    rejected/added ONLY if the analyst actually corrected it:

    - REJECT: rejected_techniques = the FULL original list (the analyst
      rejected the mapping wholesale); corrected/added = the inline fix,
      when one was supplied.
    - EDIT: rejected_techniques = only the techniques the analyst REMOVED
      (original minus final); added_techniques = the ones they ADDED. A
      technique the analyst kept appears in neither — anchoring kept
      techniques as "corrections" was MISS-scoring patterns whose advice
      the analyst followed.
    - REMOVE: no technique signal at all. Discarding a procedure is not a
      mapping correction, and any analyst_edits riding along on a remove
      (non-UI clients) are ignored rather than logged as a phantom
      "analyst corrected techniques to X" on a draft they deleted.

    has_correction is change-detected: an analyst_edits.techniques list
    identical to the LLM's output (delete + re-add the same chip) is NOT a
    correction. An EXPLICIT empty list is — it means "none of these
    techniques apply", the strongest negative signal — so has_correction
    can be True while corrected_techniques is [].
    """
    original = _slim_techniques(draft.get("techniques", []))
    edits = review.get("analyst_edits") or {}

    corrected_raw = None
    if action != GateAction.REMOVE.value and "techniques" in edits:
        corrected_raw = edits.get("techniques")

    corrected = _slim_techniques(corrected_raw)
    orig_ids = {t["technique_id"] for t in original}
    corr_ids = {t["technique_id"] for t in corrected}
    has_correction = corrected_raw is not None and orig_ids != corr_ids

    removed = (
        [t for t in original if t["technique_id"] not in corr_ids]
        if has_correction else []
    )
    added = (
        [t for t in corrected if t["technique_id"] not in orig_ids]
        if has_correction else []
    )
    if action == GateAction.REJECT.value:
        rejected = original
    else:
        rejected = removed

    return {
        "draft_id": draft.get("draft_id", ""),
        "draft_name": draft.get("name", ""),
        "chunk_id": draft.get("chunk_id", ""),
        "action": action,
        "reject_reason": review.get("reject_reason", ""),
        # Why a draft was dropped is as useful to the pattern learner as why
        # one was sent back — "duplicate" recurring across sources is a
        # chunking signal, not noise.
        "remove_reason": review.get("remove_reason", ""),
        "rationale": review.get("rationale", ""),
        "rejected_techniques": rejected,
        "corrected_techniques": corrected if has_correction else [],
        "added_techniques": added,
        "has_correction": has_correction,
    }


def _build_technique_rerun_feedback(
    correction_records: list[dict],
) -> list[dict]:
    """Build the per-chunk corrective feedback for a wrong_technique re-extract.

    Only technique-level signal routes to extract_techniques: non-chunk
    rejects (bare or corrected) and technique edits. The chunk_behaviors
    route preserves edits separately via _build_technique_overrides, so a
    BAD_CHUNK_BOUNDARY reject is excluded here. Removes carry no
    technique-mapping signal and are also excluded.

    extract_techniques injects these into its propose + pick prompts, which
    busts the LLM cache (the prompt changes) so the rerun is corrective
    rather than a byte-identical repeat of the rejected pass.
    """
    feedback: list[dict] = []
    for rec in correction_records:
        action = rec.get("action")
        reason = rec.get("reject_reason", "")
        if reason == Gate1RejectReason.BAD_CHUNK_BOUNDARY.value:
            continue
        # Edits matter only when techniques actually changed; bare-name
        # edits carry no technique-mapping signal.
        if action == GateAction.EDIT.value and not rec.get("has_correction"):
            continue
        if action not in (GateAction.REJECT.value, GateAction.EDIT.value):
            continue
        chunk_id = rec.get("chunk_id", "")
        if not chunk_id:
            continue
        feedback.append({
            "chunk_id": chunk_id,
            # action lets the prompt renderer distinguish a rejection
            # ("previously picked X — REJECTED") from a revision ("analyst
            # revised the mapping") — without it, an edit's kept techniques
            # were being rendered as rejected picks.
            "action": action,
            "reject_reason": reason,
            "rationale": rec.get("rationale", ""),
            "rejected_techniques": rec.get("rejected_techniques", []),
            "corrected_techniques": rec.get("corrected_techniques", []),
            "added_techniques": rec.get("added_techniques", []),
        })
    return feedback


def _compute_rejection_routing(reject_reasons: list[str]) -> str | None:
    """Determine where to route after Gate 1 based on rejection reasons.

    Priority order (highest first):
    1. BAD_CHUNK_BOUNDARY -> "chunk_behaviors" (chunking was wrong,
       so technique extraction is also suspect)
    2. Any other rejection -> "extract_techniques" (technique mapping
       was wrong but chunks are fine)
    3. No rejections -> None (proceed to normalize)
    """
    if not reject_reasons:
        return None

    # Check for chunking issues first (highest priority)
    if Gate1RejectReason.BAD_CHUNK_BOUNDARY.value in reject_reasons:
        return "chunk_behaviors"

    # Any other rejection routes to technique re-extraction
    return "extract_techniques"


def _chunk_text_hash(text: str) -> str:
    """Stable fingerprint of chunk text for matching after re-chunking.

    Uses first 20 hex chars of SHA-256 (80 bits). Not cryptographic,
    just a lookup key for re-applying technique overrides to similar chunks.
    80 bits gives negligible collision probability for realistic chunk counts.
    """
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()[:20]


def _build_chunk_feedback(
    reviews: list[dict],
    review_map: dict[str, dict],
    drafts: list[dict],
    chunk_lookup: dict[str, dict],
) -> list[dict]:
    """Build structured chunk feedback from BAD_CHUNK_BOUNDARY rejections.

    Each BAD_CHUNK_BOUNDARY review can optionally include:
        chunk_problem: ChunkProblemType value
        related_draft_id: which other draft overlaps/should merge
        chunk_guidance: free-text describing where the boundary should be

    These are passed through from the frontend via analyst_edits.
    """
    feedback = []
    draft_lookup = {d.get("draft_id", ""): d for d in drafts}

    for review in reviews:
        if review.get("reject_reason") != Gate1RejectReason.BAD_CHUNK_BOUNDARY.value:
            continue

        did = review.get("draft_id", "")
        draft = draft_lookup.get(did)
        if not draft:
            logger.warning("Gate 1 chunk feedback: unknown draft_id '%s', skipping", did)
            continue
        chunk_id = draft.get("chunk_id", "")
        chunk = chunk_lookup.get(chunk_id)
        if not chunk:
            logger.warning("Gate 1 chunk feedback: no chunk for draft '%s' (chunk_id='%s')", did, chunk_id)
            chunk = {}

        # Extract structured chunk feedback from analyst_edits
        edits = review.get("analyst_edits") or {}

        entry = {
            "draft_id": did,
            "chunk_id": chunk_id,
            "chunk_text": chunk.get("text", ""),
            "problem": edits.get("chunk_problem", ChunkProblemType.WRONG_BOUNDARY.value),
            "related_draft_id": edits.get("related_draft_id"),
            "guidance": edits.get("chunk_guidance", review.get("rationale", "")),
        }

        # If a related draft is specified, include its chunk text for context
        related_did = entry["related_draft_id"]
        if related_did:
            related_draft = draft_lookup.get(related_did, {})
            related_chunk_id = related_draft.get("chunk_id", "")
            related_chunk = chunk_lookup.get(related_chunk_id, {})
            entry["related_chunk_text"] = related_chunk.get("text", "")

        feedback.append(entry)

    logger.info("Gate 1: built %d chunk feedback entries", len(feedback))
    return feedback


def _build_technique_overrides(
    drafts: list[dict],
    review_map: dict[str, dict],
    chunk_lookup: dict[str, dict],
) -> list[dict]:
    """Preserve analyst technique edits so they survive a re-chunk cycle.

    For every draft where the analyst edited techniques (action=edit with
    techniques in analyst_edits) OR approved with modified techniques,
    save the technique list keyed by a hash of the source chunk text.

    After re-chunking produces new chunks, the drafting node can match
    new chunks against these hashes to re-apply the analyst's corrections.
    """
    overrides = []

    for draft in drafts:
        did = draft.get("draft_id", "")
        review = review_map.get(did)
        chunk_id = draft.get("chunk_id", "")
        chunk = chunk_lookup.get(chunk_id, {})
        chunk_text = chunk.get("text", "")

        if not chunk_text:
            continue

        # Check if this draft has technique edits
        techniques = None
        if review:
            edits = review.get("analyst_edits") or {}
            if "techniques" in edits:
                techniques = edits["techniques"]

        # Also check if techniques were applied in-place by _apply_draft_edits
        if techniques is None and draft.get("gate_action") == GateAction.EDIT.value:
            draft_edits = draft.get("analyst_edits") or {}
            if "techniques" in draft_edits:
                techniques = draft_edits["techniques"]

        if techniques is None:
            continue

        overrides.append({
            "chunk_text_hash": _chunk_text_hash(chunk_text),
            "chunk_text_preview": chunk_text[:200],  # For debugging
            "techniques": techniques,
            "draft_name": draft.get("name", ""),
            "rationale": (review or {}).get("rationale", ""),
        })

    logger.info(
        "Gate 1: preserved %d technique overrides for re-chunk cycle",
        len(overrides),
    )
    return overrides


# =============================================================================
# Gate 2: Relationship review
# =============================================================================

def gate_2(state: PipelineState) -> dict:
    """Process relationship review decisions from the analyst.

    Supports two modes:
    1. Per-relationship decisions via gate2_reviews (list of per-rel actions)
    2. Binary batch decision via gate2_review (legacy: approve/reject all)

    Per-relationship mode (preferred):
        gate2_reviews is a list of dicts:
            rel_id: str              (matches relationship_preview[].id)
            action: str              (approve | edit | remove)
            edited_rel_type: str     (if edited)
            edited_source: str       (if edited)
            edited_target: str       (if edited)
            rationale: str           (why, especially for edits/removes)

    Binary mode (batch reject):
        gate2_review is a dict:
            approved: bool
            feedback: str|None

    The serializer filters its SRO list against gate2_removed_rel_ids,
    translating preview ids to content keys via relationship_preview.
    (gate2_added_rels is consumed only by the feedback synthesizer today —
    analyst-added relationships are not yet materialized as SROs.)
    """
    if not is_gate_enabled(state, "bundle"):
        logger.info("Gate 2 auto-skip: approved")
        return {
            "gate2_decision": {"approved": True, "feedback": None},
            "gate2_approved_rel_ids": [
                r.get("id", "") for r in state.get("relationship_preview", [])
            ],
            "gate2_removed_rel_ids": [],
            "gate2_added_rels": [],
            "gate2_edited_rels": [],
            "status": PipelineStatus.RESUMING_FROM_GATE_2.value,
            "current_node": "gate_2",
        }

    # Check for per-relationship reviews first (new mode).
    # Use `is not None` rather than truthiness: an empty list is a
    # legitimate signal that "the analyst submitted per-rel mode but
    # made no changes" (approve all unmentioned). Falling through to
    # batch mode on empty would set approved=False by default and
    # erroneously remove every relationship — which the
    # BundleReviewCanvas's "Approve as-is" path otherwise hit.
    per_rel_reviews = state.get("gate2_reviews")
    if per_rel_reviews is not None:
        return _process_per_relationship_reviews(per_rel_reviews, state)

    # Fallback: binary batch decision (legacy mode)
    review = state.get("gate2_review", {})
    approved = review.get("approved", False)
    feedback = review.get("feedback")

    decision = {
        "approved": approved,
        "feedback": feedback,
    }

    all_rel_ids = [r.get("id", "") for r in state.get("relationship_preview", [])]

    if approved:
        logger.info("Gate 2: batch approved (%d relationships)", len(all_rel_ids))
        return {
            "gate2_decision": decision,
            "gate2_approved_rel_ids": all_rel_ids,
            "gate2_removed_rel_ids": [],
            "gate2_added_rels": [],
            "gate2_edited_rels": [],
            "status": PipelineStatus.RESUMING_FROM_GATE_2.value,
            "current_node": "gate_2",
        }
    else:
        logger.info("Gate 2: batch rejected. Feedback: %s", feedback or "(none)")
        return {
            "gate2_decision": decision,
            "gate2_approved_rel_ids": [],
            "gate2_removed_rel_ids": all_rel_ids,
            "gate2_added_rels": [],
            "gate2_edited_rels": [],
            "status": PipelineStatus.RESUMING_FROM_GATE_2.value,
            "current_node": "gate_2",
        }


def _process_per_relationship_reviews(
    reviews: list[dict], state: PipelineState,
) -> dict:
    """Process per-relationship analyst decisions.

    Builds approved/removed/added lists from the review actions.

    Default behavior for unmentioned relationships: APPROVE. The
    BundleReviewCanvas (Gate 3 — Bundle Review) only emits review
    entries for rels the analyst actively touched (remove/edit/add),
    so the natural reading of an empty or partial reviews list is
    "leave the rest alone, approve them as-is." This matches the
    auto-skip path which puts every rel_id into approved_ids.
    """
    removed_ids: list[str] = []
    added_rels: list[dict] = []
    edited_rels: list[dict] = []
    feedback_parts: list[str] = []

    # Read-only view of the preview. An edit is recorded as {original,
    # edited} rather than applied in place: the serializer removes the
    # original's key and emits the edited row, so the preview stays the
    # record of what normalize derived and nothing re-keys under the
    # analyst's feet.
    preview_by_id = {
        r.get("id", ""): r for r in state.get("relationship_preview", [])
    }

    # Track explicit-approve / explicit-edit IDs so we can compute the
    # "unmentioned" set later.
    explicitly_handled: set[str] = set()

    for review in reviews:
        rel_id = review.get("rel_id", "")
        action = review.get("action", "approve")
        rationale = (review.get("rationale") or "").strip()
        # Truncate rationale to prevent oversized state entries
        if len(rationale) > 500:
            rationale = rationale[:500]

        if rel_id.startswith("added_"):
            # Analyst-added relationship (not in original preview). The
            # endpoint types come from the canvas; the serializer resolves
            # (name, type) to a STIX id and skips what it cannot name.
            added_rels.append({
                "relationship_type": review.get("edited_rel_type", "uses"),
                "source_name": review.get("edited_source", ""),
                "target_name": review.get("edited_target", ""),
                "source_type": review.get("edited_source_type") or "",
                "target_type": review.get("edited_target_type") or "",
                "rationale": rationale,
            })
            logger.info("Gate 2: analyst added relationship: %s -> %s -> %s",
                        review.get("edited_source", ""),
                        review.get("edited_rel_type", ""),
                        review.get("edited_target", ""))
            continue

        if action == "approve":
            explicitly_handled.add(rel_id)
        elif action == "remove":
            removed_ids.append(rel_id)
            explicitly_handled.add(rel_id)
            if rationale:
                preview = preview_by_id.get(rel_id, {})
                feedback_parts.append(
                    f"[remove] {preview.get('source_name', '')} -> "
                    f"{preview.get('relationship_type', '')} -> "
                    f"{preview.get('target_name', '')}: {rationale}"
                )
        elif action == "edit":
            # Edited relationships are approved with modifications: the
            # serializer drops the ORIGINAL row's key and emits the edited
            # row. Endpoint types default to the original's — the canvas
            # only ever edits the verb by hand, and the reviewer's edits
            # carry names without types.
            explicitly_handled.add(rel_id)
            preview = preview_by_id.get(rel_id)
            if preview:
                edited = dict(preview)
                if review.get("edited_rel_type"):
                    edited["relationship_type"] = review["edited_rel_type"]
                if review.get("edited_source"):
                    edited["source_name"] = review["edited_source"]
                if review.get("edited_target"):
                    edited["target_name"] = review["edited_target"]
                if review.get("edited_source_type"):
                    edited["source_type"] = review["edited_source_type"]
                if review.get("edited_target_type"):
                    edited["target_type"] = review["edited_target_type"]
                edited["rationale"] = rationale
                edited_rels.append({
                    "rel_id": rel_id,
                    "original": dict(preview),
                    "edited": edited,
                })

    # Default-approve: every rel from the preview that wasn't explicitly
    # removed gets approved. This matches the auto-skip path and the
    # BundleReviewCanvas's natural shape (only changed rels appear in
    # the reviews list). A rel that was edited above also lands in
    # approved_ids here, with the edits applied to preview_by_id.
    approved_ids: list[str] = [
        rid for rid in preview_by_id.keys() if rid not in set(removed_ids)
    ]

    # A removal is a REFINEMENT of the bundle, not a rejection of it.
    #
    # This previously read `all_approved = len(removed_ids) == 0`, which made
    # any single removal route the graph back to `normalize` — where the
    # preview was re-derived from scratch and the analyst's decisions were
    # discarded, so the run re-paused at gate_2 with the same 37 edges and no
    # way through except approving everything. Per-relationship mode always
    # proceeds now; only an explicit batch `{"approved": false, "feedback":...}`
    # rejects the bundle and loops back.
    all_approved = True
    feedback = "\n".join(feedback_parts) if feedback_parts else None

    logger.info(
        "Gate 2: %d approved, %d removed, %d added, %d edited",
        len(approved_ids), len(removed_ids), len(added_rels), len(edited_rels),
    )

    result = {
        "gate2_decision": {"approved": all_approved, "feedback": feedback},
        "gate2_approved_rel_ids": approved_ids,
        "gate2_removed_rel_ids": removed_ids,
        "gate2_added_rels": added_rels,
        "gate2_edited_rels": edited_rels,
        # Clear the consumed channel, matching gate_0 and gate_chunks. Left
        # set, a follow-up BATCH submission would be shadowed: the dispatch
        # checks `gate2_reviews is not None` first and would take the per-rel
        # branch again using stale decisions.
        "gate2_reviews": None,
        "status": PipelineStatus.RESUMING_FROM_GATE_2.value,
        "current_node": "gate_2",
    }

    return result
