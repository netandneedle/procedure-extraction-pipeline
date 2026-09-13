"""Attack Flow v2.0.0 operator inference + SDO construction.

Currently single-purpose: turn the chunk DAG's branch_point / convergence_point
geometry into `attack-operator` SDOs (AND/OR/XOR) so the serialized bundle
expresses control-flow structure instead of flattening converges/branches
into pairwise procedure→procedure precedes SROs.

Design:
  * Geometry-derived defaults: ≥2 predecessors → AND (converge); ≥2
    successors → OR (branch). XOR is never inferred — it's analyst-marked
    only, via the operator-kind override at gate_chunks.
  * Stable operator STIX IDs: derived from (role, sorted input chunk_ids,
    sorted output chunk_ids) so identical geometry across re-runs yields
    the same `attack-operator--<hash>` id. Re-chunking that changes the
    geometry changes the id (intentional — the operator is bound to the
    specific chunk graph, not abstractly "the converge at this position").
  * Operator inference does NOT run when `is_sequential=False`. Catalog
    sources ship as a flat collection without flow scaffolding (same gate
    as PRECEDES SRO emission in the serializer).

This module handles geometry only. Analyst overrides arrive as
`state["chunk_operators"][op_id]["kind"]` written at gate_chunks and are
honoured through `existing_operators`.
"""
from __future__ import annotations

import hashlib
import logging
import uuid
from collections import defaultdict
from typing import Literal

from app.nodes.deterministic.extension_definitions import (
    ATTACK_FLOW_EXTENSION_ID,
    extension_declaration,
)

logger = logging.getLogger(__name__)


OperatorKind = Literal["AND", "OR", "XOR"]
OperatorRole = Literal["converge", "branch"]


def infer_operators(
    chunks: list[dict],
    is_sequential: bool,
    existing_operators: dict | None = None,
    condition_anchors: set[str] | None = None,
) -> dict[str, dict]:
    """Walk the chunk DAG and emit one entry per branch/converge point.

    Returns a dict keyed by operator_id (a stable 12-hex-char hash) with
    entries shaped:
        {
            "kind": "AND" | "OR" | "XOR",
            "role": "converge" | "branch",
            "anchor_chunk_id": str,        # The chunk that is the branch/converge point.
            "input_chunk_ids": [str, ...], # For converge: the predecessors.
                                           # For branch: [anchor_chunk_id].
            "output_chunk_ids": [str, ...],# For converge: [anchor_chunk_id].
                                           # For branch: the successors.
        }

    `existing_operators` (when provided) lets the analyst's gate_chunks
    overrides survive a re-run of normalization. Only the `kind` field is
    preserved from the existing entry; everything else is re-derived from
    current geometry. Operators with the same operator_id as an entry in
    `existing_operators` inherit the override; operators with no match
    get the geometry default.
    """
    if not is_sequential:
        return {}

    chunks_by_id = {c.get("chunk_id"): c for c in chunks if c.get("chunk_id")}
    if not chunks_by_id:
        return {}

    # Build a forward-edges map: chunk_id -> sorted list of successor chunk_ids.
    # precedes_ids is the chunker's forward-edge field (see Chunk in state.py).
    successors: dict[str, list[str]] = defaultdict(list)
    for chunk in chunks:
        src = chunk.get("chunk_id")
        if not src:
            continue
        for tgt in chunk.get("precedes_ids", []) or []:
            if tgt in chunks_by_id and tgt not in successors[src]:
                successors[src].append(tgt)

    # Inverse map: chunk_id -> sorted list of predecessor chunk_ids.
    predecessors: dict[str, list[str]] = defaultdict(list)
    for src, tgts in successors.items():
        for tgt in tgts:
            if src not in predecessors[tgt]:
                predecessors[tgt].append(src)

    operators: dict[str, dict] = {}
    existing_kind_by_id = {
        op_id: meta.get("kind")
        for op_id, meta in (existing_operators or {}).items()
        if meta.get("kind") in {"AND", "OR", "XOR"}
    }

    # Converge operators: any chunk with ≥2 predecessors gets an AND.
    for chunk_id, preds in predecessors.items():
        if len(preds) < 2:
            continue
        sorted_preds = sorted(preds)
        op_id = _operator_id("converge", sorted_preds, [chunk_id])
        kind = existing_kind_by_id.get(op_id, "AND")
        operators[op_id] = {
            "kind": kind,
            "role": "converge",
            "anchor_chunk_id": chunk_id,
            "input_chunk_ids": sorted_preds,
            "output_chunk_ids": [chunk_id],
        }

    # Branch operators: any chunk with ≥2 successors gets an OR — UNLESS
    # the chunk anchors an attack-condition, in which case the condition
    # replaces the branch operator semantically.
    suppress = set(condition_anchors or ())
    for chunk_id, succs in successors.items():
        if len(succs) < 2:
            continue
        if chunk_id in suppress:
            continue
        sorted_succs = sorted(succs)
        op_id = _operator_id("branch", [chunk_id], sorted_succs)
        kind = existing_kind_by_id.get(op_id, "OR")
        operators[op_id] = {
            "kind": kind,
            "role": "branch",
            "anchor_chunk_id": chunk_id,
            "input_chunk_ids": [chunk_id],
            "output_chunk_ids": sorted_succs,
        }

    if operators:
        logger.info(
            "infer_operators: %d operators (%d converge, %d branch)",
            len(operators),
            sum(1 for o in operators.values() if o["role"] == "converge"),
            sum(1 for o in operators.values() if o["role"] == "branch"),
        )
    return operators


def _operator_id(role: str, inputs: list[str], outputs: list[str]) -> str:
    """Stable 12-hex-char id from (role, sorted inputs, sorted outputs).

    Same geometry → same id across re-runs. Re-chunking that changes any
    of these inputs produces a different id — the operator is bound to
    the specific chunk graph, not abstractly to "the converge at this
    position." Analyst overrides keyed by operator_id therefore orphan
    when geometry shifts; re-keying is not implemented.
    """
    h = hashlib.sha256()
    h.update(role.encode("utf-8"))
    h.update(b"|")
    h.update(",".join(inputs).encode("utf-8"))
    h.update(b"|")
    h.update(",".join(outputs).encode("utf-8"))
    return h.hexdigest()[:12]


def build_attack_operator_sdos(
    chunk_operators: dict[str, dict],
    chunk_to_procedure_stix_id: dict[str, str],
    source_identity_id: str,
    now_iso: str,
    spec_version: str,
) -> tuple[list[dict], dict[str, str]]:
    """Materialize attack-operator SDOs from inferred operators.

    Returns:
        (sdos, op_id_to_stix_id) — `sdos` is the list of attack-operator
        objects ready to append to the bundle; `op_id_to_stix_id` maps
        the internal operator_id (used in chunk_operators) to the
        `attack-operator--<uuid>` STIX id.

    Operators whose anchor or any input/output chunk doesn't resolve to
    a procedure STIX id are SKIPPED with a logged warning — they would
    produce dangling effect_refs that the validator would reject. This
    can happen when an analyst drops a chunk between gate_chunks and
    gate_1.
    """
    sdos: list[dict] = []
    op_id_to_stix_id: dict[str, str] = {}

    for op_id, meta in chunk_operators.items():
        # Resolve effect_refs (outputs) to procedure STIX ids.
        effect_refs: list[str] = []
        for out_chunk_id in meta["output_chunk_ids"]:
            stix_id = chunk_to_procedure_stix_id.get(out_chunk_id)
            if stix_id:
                effect_refs.append(stix_id)

        # A branch with <2 resolved effect_refs is degenerate (the
        # downstream of the branch collapsed to a single output, which
        # means there's nothing to "branch" anymore). Skip and let
        # precedes routing fall back to direct procedure→procedure
        # edges for surviving outputs.
        if len(effect_refs) < 2 and meta["role"] == "branch":
            logger.warning(
                "build_attack_operator_sdos: branch operator %s has only "
                "%d resolved effect_refs, skipping (chunks: %s)",
                op_id, len(effect_refs), meta["output_chunk_ids"],
            )
            continue
        # For converge operators, effect_refs is always a single
        # post-converge procedure. The 2+ inputs flow into the operator,
        # which then flows into 1 output. So 1 effect_ref is correct.
        if not effect_refs:
            logger.warning(
                "build_attack_operator_sdos: operator %s has no resolved "
                "effect_refs, skipping (output chunks: %s)",
                op_id, meta["output_chunk_ids"],
            )
            continue

        stix_id = f"attack-operator--{uuid.uuid4()}"
        op_id_to_stix_id[op_id] = stix_id
        sdos.append({
            "type": "attack-operator",
            "spec_version": spec_version,
            "id": stix_id,
            "created": now_iso,
            "modified": now_iso,
            "created_by_ref": source_identity_id,
            "operator": meta["kind"],
            "effect_refs": effect_refs,
            # Attack Flow 2.0.0 has ONE extension definition covering every
            # attack-* type; the serializer embeds it once per bundle.
            "extensions": extension_declaration(ATTACK_FLOW_EXTENSION_ID),
        })

    return sdos, op_id_to_stix_id


def splice_absent(
    successors: dict[str, list[str]],
    present_ids: set[str],
) -> tuple[dict[str, list[str]], int, int]:
    """Bridge the flow around chunks that have no procedure in the bundle.

    A chunk can vanish between the chunk DAG and the bundle for several
    reasons: the analyst removed its draft at Gate 1, Gate 1 rejected it, or
    serialize_stix omitted it for having no ATT&CK technique. Whatever the
    cause, the edges through it used to be dropped outright — so A -> B -> C
    with B gone left A and C disconnected, and _build_attack_flow then read C
    as a fresh chain root. Nothing logged it (on one run, 3 of 21
    edges were lost this way, in silence).

    Bridging is the honest reading of the source: the report still says A
    happens before C. Only the intermediate step is unrepresentable.

    Transitive on purpose. The chunk gate's own splice (gates.py) keeps a
    successor only when it survived, so two consecutive removals still lose
    the link; walking through them fixes that for both callers.

    Returns (spliced_successors, bridged, dropped) where `bridged` counts
    edges rerouted onto a further-downstream chunk and `dropped` counts edges
    that led only into removed chunks with no surviving successor at all.
    """
    bridged = dropped = 0

    def _resolve(chunk_id: str, seen: set[str]) -> list[str]:
        """Nearest surviving descendants of an absent chunk."""
        out: list[str] = []
        for nxt in successors.get(chunk_id, []):
            if nxt in seen:
                # Cycle through removed chunks. The precedes DAG is meant to
                # be acyclic and the validator hard-fails if it is not, but
                # this walk must terminate regardless of what it is handed.
                continue
            seen.add(nxt)
            if nxt in present_ids:
                out.append(nxt)
            else:
                out.extend(_resolve(nxt, seen))
        return out

    spliced: dict[str, list[str]] = {}
    for src, targets in successors.items():
        if src not in present_ids:
            continue  # src itself is gone; its inbound edges get bridged instead
        kept: list[str] = []
        for tgt in targets:
            if tgt in present_ids:
                kept.append(tgt)
                continue
            replacements = _resolve(tgt, {tgt})
            if replacements:
                bridged += 1
                kept.extend(replacements)
            else:
                dropped += 1
        # dict.fromkeys dedups while preserving order — two removed chunks can
        # bridge onto the same survivor.
        spliced[src] = [t for t in dict.fromkeys(kept) if t != src]

    return spliced, bridged, dropped


def route_precedes_through_operators(
    chunks: list[dict],
    chunk_operators: dict[str, dict],
    op_id_to_stix_id: dict[str, str],
    chunk_to_procedure_stix_id: dict[str, str],
    chunk_conditions: dict[str, dict] | None = None,
    cond_anchor_to_stix_id: dict[str, str] | None = None,
) -> list[tuple[str, str]]:
    """Build the final precedes-SRO edge list, inserting operators.

    Walks the chunk DAG's forward edges (chunk.precedes_ids) and for each
    edge A→B decides where to route the SROs:

      * If A is a branch point (OR/XOR) AND B is a convergence point (AND):
        emit A_proc → op_branch → op_converge → B_proc (three edges).
        Both the branch-feed and converge-feed edges are deduplicated
        across all A→B traversals that touch the same operators.
      * If A is a branch point only: emit A_proc → op_branch (once) and
        op_branch → B_proc.
      * If B is a convergence point only: emit A_proc → op_converge (once)
        and op_converge → B_proc.
      * Otherwise: emit A_proc → B_proc direct.

    Returns a deduplicated list of (source_stix_id, target_stix_id) tuples.
    Caller wraps each into a `precedes` Relationship SDO.

    Edges involving chunks with no resolved procedure STIX id are silently
    dropped (those procedures didn't make it into the bundle — gate_1
    rejected the draft, etc.).
    """
    branch_at: dict[str, str] = {}   # chunk_id -> operator_id (branch role only)
    converge_at: dict[str, str] = {} # chunk_id -> operator_id (converge role only)
    for op_id, meta in chunk_operators.items():
        if op_id not in op_id_to_stix_id:
            # Operator was skipped by build_attack_operator_sdos (e.g.
            # insufficient resolved effect_refs). Route around it as if
            # it didn't exist.
            continue
        anchor = meta["anchor_chunk_id"]
        if meta["role"] == "converge":
            converge_at[anchor] = op_id
        else:
            branch_at[anchor] = op_id

    # Condition routing — same shape as branch operators on the source side
    # (chunk → cond → targets), but the condition's anchor takes precedence
    # over any OR-branch operator at the same chunk (infer_operators already
    # suppressed the branch operator there, so branch_at won't have an entry
    # for a condition anchor — this lookup is just to materialize the
    # source-side replacement).
    cond_at: dict[str, str] = {} # chunk_id -> attack-condition STIX id
    if chunk_conditions and cond_anchor_to_stix_id:
        for anchor in chunk_conditions:
            stix_id = cond_anchor_to_stix_id.get(anchor)
            if stix_id:
                cond_at[anchor] = stix_id

    edges: set[tuple[str, str]] = set()

    def _resolve(chunk_id: str) -> str | None:
        return chunk_to_procedure_stix_id.get(chunk_id)

    # Bridge around chunks with no procedure BEFORE walking the DAG. Every
    # reason a chunk can go missing converges here — removed at Gate 1,
    # rejected there, or omitted by serialize_stix for having no ATT&CK
    # technique — so this is the one place that can keep the chain intact for
    # all of them. Without it the edge is simply skipped and the chain breaks
    # in silence.
    raw_successors = {
        c["chunk_id"]: list(c.get("precedes_ids") or [])
        for c in chunks if c.get("chunk_id")
    }
    spliced, bridged, dropped = splice_absent(
        raw_successors, set(chunk_to_procedure_stix_id),
    )
    if bridged or dropped:
        logger.info(
            "route_precedes: %d edge(s) bridged around chunks with no "
            "procedure, %d dropped for having no surviving successor",
            bridged, dropped,
        )

    for chunk in chunks:
        src_chunk = chunk.get("chunk_id")
        if not src_chunk:
            continue
        src_proc = _resolve(src_chunk)
        if not src_proc:
            continue

        for tgt_chunk in spliced.get(src_chunk, []):
            tgt_proc = _resolve(tgt_chunk)
            if not tgt_proc:
                continue

            # Source-side hop: condition wins over branch operator (the
            # condition replaces the branch). branch_at and cond_at should
            # never both contain src_chunk because infer_operators
            # suppresses OR-branch at condition anchors, but cond first
            # is the safer ordering.
            src_cond = cond_at.get(src_chunk)
            src_branch_op = branch_at.get(src_chunk)
            tgt_converge_op = converge_at.get(tgt_chunk)

            if src_cond:
                src_side_id = src_cond
                edges.add((src_proc, src_cond))
            elif src_branch_op:
                src_side_id = op_id_to_stix_id[src_branch_op]
                edges.add((src_proc, src_side_id))
            else:
                src_side_id = src_proc

            tgt_side_id = op_id_to_stix_id[tgt_converge_op] if tgt_converge_op else tgt_proc

            # Middle edge — condition→procedure, condition→operator,
            # operator→operator, operator→procedure, or
            # procedure→procedure depending on which sides have what.
            edges.add((src_side_id, tgt_side_id))

            # Feed-out edge from the target convergence operator to the
            # target procedure.
            if tgt_converge_op:
                edges.add((op_id_to_stix_id[tgt_converge_op], tgt_proc))

    return sorted(edges)
