"""Per-gate reviewers: what the AI reviewer is shown, and what it may return.

One `GateReviewer` per gate. Each knows how to render that gate's state into
a user turn and declares the tool schema + Pydantic model for its output.

Adding a gate is adding one record to `REVIEWERS`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.graph.state import EntityType
from app.services.stix_schema import industry_sector_vocab
from app.services.reviewer.models import (
    Gate0Recommendations,
    Gate1Recommendations,
    Gate2Recommendations,
    GateChunksRecommendations,
)
from app.services.vendor_technique_mapping import extract_vendor_technique_ids

# The report is sent in full so the reviewer sees what the extractor's
# windowed passes could not. Capped defensively — a runaway parse should
# degrade the review, not fail the request. The reviewer is TOLD when it has
# been truncated, because a model that doesn't know it is reading a fragment
# will confidently report what is missing from it.
MAX_REPORT_CHARS = 120_000

# Confidence values the shared recommendation fields accept.
_CONFIDENCE_ENUM = ["high", "medium", "low"]

# The entity vocabulary, straight from the enum the pipeline validates
# against. Handed to the model as a JSON-schema enum rather than described
# in prose: on the first live run the reviewer proposed entity_type
# "attack_pattern" for CLICKFIX — a reasonable-sounding STIX type that is
# not an EntityType, and which the addition channel would have carried
# straight into validated_entities. Deriving the list here means it cannot
# drift from the enum, and the model cannot invent a member of it.
_ENTITY_TYPE_ENUM = [e.value for e in EntityType]

# The STIX industry-sector-ov vocabulary, from the same source the serializer
# and the bundle validator use.
#
# The reviewer needs this for the same reason the extractor does, and the
# first live run showed what happens without it: asked to review a vendor report,
# the reviewer recommended adding victim_sector "legal & professional
# services" — verbatim from the report, quote support 1.0, entirely
# well-evidenced. The extractor had deliberately NOT emitted it, because its
# own prompt names that exact string as the worked example of a sector with
# no vocabulary value. The analyst took the recommendation, and the bundle
# validator silently dropped the sector from the Identity.
#
# The lesson generalizes past sectors: a reviewer holding less context than
# the stage it reviews will "correct" deliberate decisions into mistakes, and
# do it with real evidence attached. Any constrained vocabulary the extractor
# is given has to reach the reviewer too.
_SECTOR_VOCAB = list(industry_sector_vocab())

_SHARED_REC_PROPS: dict[str, Any] = {
    "confidence": {
        "type": "string",
        "enum": _CONFIDENCE_ENUM,
        "description": (
            "high = the report states this plainly. medium = well-supported "
            "inference. low = worth a look, but you are unsure. The analyst "
            "can bulk-accept HIGH without opening it, so treat high as a "
            "claim on their trust."
        ),
    },
    "rationale": {
        "type": "string",
        "description": "Why. One or two concrete sentences referencing the source.",
    },
    "evidence_quote": {
        "type": "string",
        "description": (
            "VERBATIM span from the report supporting this — copy the "
            "characters, do not paraphrase. Empty string if you have none. "
            "Quotes are checked against the report; an unsupported quote "
            "downgrades your recommendation to low confidence."
        ),
    },
}

_INITIAL_READ_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": (
        "Your opening read of the report. Emit this ONLY on your first turn "
        "for a source; omit it on later gates."
    ),
    "properties": {
        "summary": {
            "type": "string",
            "description": "What this report describes. 3-6 sentences, plain language.",
        },
        "actors": {
            "type": "array", "items": {"type": "string"},
            "description": "Named actors, groups, or campaigns. [] if none named.",
        },
        "attack_chain": {
            "type": "array", "items": {"type": "string"},
            "description": "The attack in order, one short phrase per step, as the report tells it.",
        },
        "thin_areas": {
            "type": "array", "items": {"type": "string"},
            "description": (
                "Where the report is vague, second-hand, or asserts without "
                "evidence — where extraction is most likely to invent."
            ),
        },
        "notes_for_later_gates": {
            "type": "string",
            "description": "Anything you want to remember when reviewing the extraction.",
        },
    },
    "required": ["summary", "actors", "attack_chain", "thin_areas"],
}

REVIEW_ENTITIES_TOOL: dict[str, Any] = {
    "name": "review_entities",
    "description": (
        "Submit your review of the extracted entities. Recommend a change "
        "only where you have a reason; approve what is right."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "initial_read": _INITIAL_READ_SCHEMA,
            "entities": {
                "type": "array",
                "description": (
                    "One entry per entity you want to change, plus any you "
                    "want to explicitly endorse. Entities you omit are left "
                    "as extracted."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "entity_id": {"type": "string"},
                        "action": {
                            "type": "string",
                            "enum": ["approve", "edit", "remove"],
                            "description": (
                                "approve = correct as extracted. edit = right "
                                "thing, wrong value/type/role. remove = should "
                                "not be an entity at all."
                            ),
                        },
                        "edited_value": {
                            "type": "string",
                            "description": "Corrected value, with action=edit.",
                        },
                        "edited_type": {
                            "type": "string",
                            "enum": _ENTITY_TYPE_ENUM,
                            "description": "Corrected EntityType value, with action=edit.",
                        },
                        "edited_role": {
                            "type": "string",
                            "description": (
                                "Corrected organization_role "
                                "(victim/sponsor/publisher/author/other) or "
                                "location_role (victim/origin/context)."
                            ),
                        },
                        **_SHARED_REC_PROPS,
                    },
                    "required": ["entity_id", "action", "confidence", "rationale"],
                },
            },
            "added_entities": {
                "type": "array",
                "description": (
                    "Entities the extractor MISSED. Only add what the report "
                    "states — a missing entity is a recall problem, an "
                    "invented one is a correctness problem."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "entity_type": {
                            "type": "string",
                            "enum": _ENTITY_TYPE_ENUM,
                            "description": (
                                "Must be one of these exactly. There is no "
                                "type for technique patterns or behaviors — "
                                "if what you want to add is not one of these, "
                                "it does not belong in the entity list at all."
                            ),
                        },
                        "organization_role": {"type": "string"},
                        "location_role": {"type": "string"},
                        "sector": {
                            "type": "string",
                            "enum": _SECTOR_VOCAB,
                            "description": (
                                "REQUIRED when entity_type is victim_sector. "
                                "Must be a STIX industry-sector-ov value. If "
                                "the report names a sector with no value on "
                                "this list, pick the closest one — a value "
                                "off this list is dropped from the bundle "
                                "silently, so proposing the report's exact "
                                "wording loses the sector entirely."
                            ),
                        },
                        **_SHARED_REC_PROPS,
                    },
                    "required": ["value", "entity_type", "confidence", "rationale"],
                },
            },
            "overall_notes": {
                "type": "string",
                "description": "What you want the analyst to know before they review.",
            },
        },
        # `initial_read` is REQUIRED, not optional.
        #
        # It was optional, and on a live run the model simply left it out.
        # Nothing failed: the entity review was fine, `GET /brief` 404'd, and
        # the source ran to completion without ever having an opening read.
        #
        # That is a silent loss of the two things the brief exists for. It is
        # turn one of the transcript, so the chunk gate loses the attack chain
        # it was supposed to compare the decomposition against — the whole
        # reason that gate is worth a stateful reviewer. And it is the
        # analyst's correction surface, so the cheapest place in the system to
        # fix a misread stops existing for that source.
        #
        # Safe to require: this tool is only ever used at the entities gate,
        # and `run_reviewer` discards a second read if one is already stored,
        # so a re-run of that gate cannot overwrite the analyst's edits.
        "required": ["initial_read", "entities", "added_entities"],
    },
}


def _fmt_entity(e: dict) -> str:
    """One entity, compact, with everything a reviewer needs to judge it."""
    bits = [
        f"  [{e.get('entity_id', '?')}] {e.get('entity_type', '?')}"
        f" = {e.get('value', '')!r}",
    ]
    conf = e.get("confidence")
    if isinstance(conf, (int, float)):
        bits.append(f"conf={conf:.2f}")
    for role_key in ("organization_role", "location_role"):
        if e.get(role_key):
            bits.append(f"{role_key}={e[role_key]}")
    if e.get("denylisted"):
        bits.append(f"DENYLISTED ({e.get('denylist_reason', 'analyst denylist')})")
    line = "  ".join(bits)
    ctx = (e.get("source_location") or {}).get("context")
    if ctx:
        line += f"\n      context: {str(ctx)[:300]}"
    return line


def build_entities_turn(state: dict) -> str:
    """The Gate 0 user turn: the extracted entities, in context."""
    entities = state.get("entities") or []
    lines = [
        "GATE 0 — ENTITY REVIEW",
        "",
        f"The extraction stage found {len(entities)} entities in this report.",
        "Review them. For each one you want to change, return an entry with an",
        "action and a reason. Entities you say nothing about are kept as-is,",
        "so silence is assent — only stay silent where you actually agree.",
        "",
        "Look for, in rough order of how often each goes wrong:",
        "  - Technique PATTERNS classified as malware or tools (ClickFix and",
        "    friends). These propagate into the bundle as fake malware SDOs.",
        "  - The wrong specificity: a tool named where a family belongs, a",
        "    generic binary recorded as bespoke malware.",
        "  - author vs publisher on organizations; victim vs origin on locations.",
        "  - Values that are artifacts of the document rather than the incident",
        "    (page furniture, the vendor's own product names, defanging debris).",
        "  - Entities the report states plainly that are missing entirely.",
        "",
        "EXTRACTED ENTITIES:",
    ]
    if entities:
        lines.extend(_fmt_entity(e) for e in entities)
    else:
        lines.append("  (none — the extractor found nothing, which is itself worth a look)")

    warnings = state.get("parse_warnings") or []
    if warnings:
        lines += ["", "PARSER WARNINGS (the text you are reading may be imperfect):"]
        lines += [f"  - {w}" for w in warnings[:20]]

    seq = state.get("sequentiality_rationale") or ""
    if seq:
        lines += [
            "",
            "SEQUENTIALITY — the pipeline classified this source as "
            f"{'SEQUENTIAL' if state.get('is_sequential', True) else 'NON-SEQUENTIAL'}:",
            f"  {seq}",
            "  If that reading is wrong, say so in overall_notes — it changes",
            "  whether the bundle gets attack-flow sequencing at all.",
        ]
    return "\n".join(lines)


def build_report_turn(state: dict) -> str:
    """Turn one: the report itself. Sent once; later gates replay it."""
    text = state.get("parsed_text") or ""
    truncated = len(text) > MAX_REPORT_CHARS
    if truncated:
        text = text[:MAX_REPORT_CHARS]
    header = [
        "SOURCE REPORT",
        "",
        f"Title: {state.get('title') or '(untitled)'}",
    ]
    meta = state.get("metadata") or {}
    if isinstance(meta, dict) and meta.get("publisher"):
        header.append(f"Publisher: {meta['publisher']}")
    if truncated:
        header += [
            "",
            f"NOTE: truncated to the first {MAX_REPORT_CHARS:,} characters. "
            "Do not report content as missing on the strength of this excerpt "
            "alone — you are not seeing all of it.",
        ]
    header += ["", "---", "", text]
    return "\n".join(header)


@dataclass(frozen=True)
class GateReviewer:
    gate_key: str
    tool: dict[str, Any]
    output_model: type
    build_turn: Callable[[dict], str]
    # Feedback-pattern categories whose PINNED rules this gate's reviewer
    # needs — the analyst's confirmed policy for the decisions it is about to
    # make. Declared here rather than imported from the extraction nodes:
    # those constants are module-private, and importing them would pull the
    # heavy technique-extraction module into the reviewer for four strings.
    #
    # Mirrors the node-side tuples; pinned by a contract test so the two
    # cannot drift.
    feedback_categories: tuple[str, ...] = ()





# =============================================================================
# Gate chunks — the decomposition itself
# =============================================================================

# Mirrors app.graph.state.ChunkGateRejectReason; pinned by a contract test.
_CHUNK_RERUN_REASONS = [
    "missed_procedures", "over_chunked", "under_chunked",
    "bad_boundaries", "bad_descriptions", "bad_flow", "other",
]

REVIEW_CHUNKS_TOOL: dict[str, Any] = {
    "name": "review_chunks",
    "description": (
        "Submit your review of how the report was decomposed into "
        "procedures. Speak where you want a change; the rest ship as chunked."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "chunks": {
                "type": "array",
                "description": (
                    "One entry per chunk you want to change, plus any you "
                    "want to explicitly endorse. Omitted chunks are kept."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string"},
                        "action": {
                            "type": "string",
                            "enum": ["approve", "edit", "drop", "merge"],
                        },
                        "merge_with": {
                            "type": "array", "items": {"type": "string"},
                            "description": (
                                "With action=merge: chunk_ids absorbed INTO "
                                "this one. Their flow edges rewire onto this "
                                "chunk; they stop existing."
                            ),
                        },
                        "edited_text": {
                            "type": "string",
                            "description": (
                                "Corrected narrative, with action=edit. Use "
                                "this to remove a claim the report does not "
                                "make, not to embellish one it does."
                            ),
                        },
                        "edited_source_excerpt": {
                            "type": "string",
                            "description": (
                                "Corrected excerpt, with action=edit. Must be "
                                "VERBATIM from the report — this field is what "
                                "anchors the chunk back to its source."
                            ),
                        },
                        **_SHARED_REC_PROPS,
                    },
                    "required": ["chunk_id", "action", "confidence", "rationale"],
                },
            },
            "added_chunks": {
                "type": "array",
                "description": (
                    "Procedures the report describes that no chunk covers. "
                    "They arrive disconnected from the flow — do not also "
                    "propose edges for them."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The behavior, 1-2 sentences, as the report tells it.",
                        },
                        "source_excerpt": {
                            "type": "string",
                            "description": "VERBATIM span from the report this comes from.",
                        },
                        **_SHARED_REC_PROPS,
                    },
                    "required": ["text", "confidence", "rationale"],
                },
            },
            "edges": {
                "type": "array",
                "description": (
                    "Corrections to the order. Only where the report itself "
                    "states or plainly implies the ordering."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["add", "remove"]},
                        "from_chunk_id": {"type": "string", "description": "Happens first."},
                        "to_chunk_id": {"type": "string", "description": "Follows it."},
                        **_SHARED_REC_PROPS,
                    },
                    "required": [
                        "action", "from_chunk_id", "to_chunk_id",
                        "confidence", "rationale",
                    ],
                },
            },
            "reject": {
                "type": "object",
                "description": (
                    "Re-chunk the whole report. Only when the decomposition "
                    "is wrong AS A WHOLE — it discards every chunk, and every "
                    "other recommendation in this payload along with them."
                ),
                "properties": {
                    "reason": {
                        "type": "string",
                        "enum": _CHUNK_RERUN_REASONS,
                        "description": "What was systemically wrong.",
                    },
                    "comments": {
                        "type": "string",
                        "description": (
                            "Guidance for the re-chunk. Concrete: name the "
                            "chunks that were wrong and what the right "
                            "boundaries would be."
                        ),
                    },
                    **_SHARED_REC_PROPS,
                },
                "required": ["reason", "confidence", "rationale"],
            },
            "overall_notes": {"type": "string"},
        },
        "required": ["chunks"],
    },
}


def _chunk_flow_summary(chunks: list[dict]) -> list[str]:
    """Shape of the precedes DAG, in the terms the analyst argues about.

    Computed rather than described because the reviewer cannot see it: each
    chunk carries its own forward edges, and "this chunk is unreachable" or
    "there are three disconnected pieces" is a property of the whole set. A
    model asked to hold 15 adjacency lists in its head and derive that will
    sometimes get it wrong, and it is cheap to just be right here.
    """
    ids = [c.get("chunk_id", "") for c in chunks if c.get("chunk_id")]
    id_set = set(ids)
    indeg = {cid: 0 for cid in ids}
    adj: dict[str, set[str]] = {cid: set() for cid in ids}
    edges = 0
    for c in chunks:
        src = c.get("chunk_id", "")
        if src not in id_set:
            continue
        for tgt in c.get("precedes_ids") or []:
            if tgt not in id_set:
                continue
            indeg[tgt] += 1
            adj[src].add(tgt)
            adj[tgt].add(src)
            edges += 1

    # Undirected components: how many separate pieces the flow is in.
    seen: set[str] = set()
    components: list[list[str]] = []
    for cid in ids:
        if cid in seen:
            continue
        stack, group = [cid], []
        seen.add(cid)
        while stack:
            node = stack.pop()
            group.append(node)
            for nbr in adj[node]:
                if nbr not in seen:
                    seen.add(nbr)
                    stack.append(nbr)
        components.append(group)

    roots = [cid for cid in ids if indeg[cid] == 0]
    declared_roots = {
        c.get("chunk_id", "") for c in chunks if c.get("chain_root")
    }
    # The lowest-sequence chunk is the primary chain's start and is NOT
    # marked chain_root — the chunker only marks ADDITIONAL chains, and
    # `_finalize_chunks` leaves chunk 1 implicit. Counting it as undeclared
    # would report the one entry point every source legitimately has.
    first = min(
        chunks, key=lambda c: c.get("sequence_index", 0), default=None,
    )
    if first is not None:
        declared_roots.add(first.get("chunk_id", ""))

    lines = [
        f"FLOW SHAPE: {len(ids)} chunks, {edges} precedes edges, "
        f"{len(components)} connected "
        + ("piece" if len(components) == 1 else "pieces") + ".",
    ]
    if roots:
        lines.append(
            "  Entry points (nothing precedes them): " + ", ".join(roots)
        )
    undeclared = [r for r in roots if r not in declared_roots]
    if len(components) > 1 and undeclared:
        lines += [
            "  A separate piece is correct when the report describes a",
            "  SEPARATE intrusion, and a defect when one attack was split",
            "  into fragments. These start a piece of their own without",
            "  being marked as a new chain: " + ", ".join(undeclared),
        ]
    return lines


def _fmt_chunk(c: dict) -> list[str]:
    """One chunk, with the provenance the pipeline computed about it."""
    cid = c.get("chunk_id", "?")
    head = f"CHUNK {cid}  (#{c.get('sequence_index', '?')})"
    if c.get("chain_root"):
        label = c.get("chain_label") or "unnamed chain"
        head += f"  [starts a separate chain: {label}]"
    elif c.get("chain_label"):
        head += f"  [chain: {c['chain_label']}]"
    lines = [head, f"  {(c.get('text') or '').strip()[:900]}"]

    meta = [f"conf={c.get('behavioral_confidence', 0):.2f}"]
    prov = c.get("source_provenance") or "paraphrased"
    if prov == "paraphrased":
        # The analogue of an unsupported technique quote: nothing in the
        # report matched this excerpt literally, so the chunk's evidence has
        # been through the model's own wording at least once.
        meta.append("NO VERBATIM ANCHOR — the excerpt is paraphrased")
    else:
        meta.append(f"evidence={prov}")
    if c.get("branch_point"):
        meta.append("branch")
    if c.get("convergence_point"):
        meta.append("converge")
    lines.append("  " + " | ".join(meta))

    excerpt = (c.get("source_excerpt") or "").strip()
    if excerpt:
        lines.append(f'  cites: "{excerpt[:400]}"')
    succ = [s for s in (c.get("precedes_ids") or [])]
    lines.append("  precedes: " + (", ".join(succ) if succ else "(nothing)"))

    artifacts = c.get("artifacts") or {}
    if isinstance(artifacts, dict) and artifacts:
        shown = "; ".join(
            f"{k}: {', '.join(str(v)[:80] for v in (vals or [])[:4])}"
            for k, vals in list(artifacts.items())[:6] if vals
        )
        if shown:
            lines.append(f"  artifacts: {shown}")

    cond = c.get("precondition")
    if isinstance(cond, dict) and cond.get("description"):
        lines.append(f"  runtime condition: {cond['description'][:200]}")
    return lines


def build_chunks_turn(state: dict) -> str:
    """The chunk-gate user turn: the decomposition, and how it hangs together.

    This is the gate where the transcript pays best. When the entity gate also
    ran in assist mode, the reviewer already wrote down the attack chain as
    the report tells it, BEFORE seeing any chunking — so the comparison here
    is against a read it committed to while unanchored, which is a stronger
    position than reading the report and the chunks together and calling the
    result independent. When this is its first gate there is no such read, so
    the prompt asks it to build one before looking at the chunks; the same
    anchoring risk is why the order is stated rather than assumed.

    Two things are deliberately not offered: the AND/OR/XOR operator kinds
    and the attack-condition partitions. Both are derived from the flow
    geometry, and the geometry is exactly what `edges` changes — a payload
    that moves an edge and pins an operator id in the same breath is
    proposing two things that cannot both hold. The prompt says they are out
    of scope so their absence does not read as approval.
    """
    chunks = state.get("chunks") or []
    is_sequential = state.get("is_sequential", True)

    lines = [
        # Deliberately unnumbered. The other three turns use the Python
        # identifiers (gate_0/gate_1/gate_2), and this gate's user-facing
        # number is 1 — which is gate_1's internal number. Either numbering
        # puts two turns labeled "GATE 1" in one transcript, so this one
        # goes by name.
        "PROCEDURE REVIEW — HOW THE REPORT WAS SPLIT UP",
        "",
        f"The chunker split this report into {len(chunks)} procedures.",
        "",
        "This is the gate that decides what the bundle contains. Every later",
        "stage runs PER CHUNK: techniques are mapped to a chunk, a draft is",
        "written per chunk, a procedure object is emitted per chunk. A missed",
        "procedure here is missing from the bundle and nothing downstream can",
        "recover it; a chunk covering two objectives produces one procedure",
        "that is honestly neither.",
        "",
        "A procedure is a discrete, repeatable technical implementation that",
        "integrates one or more techniques to fulfill ONE adversarial",
        "objective, as an atomic event in the attack. The test for a boundary",
        "is the OBJECTIVE, not the action count: several actions serving one",
        "goal are one procedure; one action serving two goals is two.",
        "",
        "What goes wrong here, in order:",
        "  - A procedure the report describes that no chunk covers. If you",
        "    wrote down the attack chain at an earlier gate, walk it against",
        "    these chunks step by step. If this is your first gate, build",
        "    that list from the report BEFORE you read the chunks — read them",
        "    first and their decomposition becomes the chain you compare to.",
        "  - Two chunks serving one objective. Merge them.",
        "  - A chunk that is not a procedure at all: the vendor's detection",
        "    advice, a mitigation, background on the actor, a restatement of",
        "    the report's own conclusions. Drop it.",
        "  - A chunk whose excerpt has no verbatim anchor in the report. That",
        "    is flagged below. It does not make the chunk wrong — but it is",
        "    where invented detail enters, so read the text against the report.",
        "  - Order that the report does not support. Co-occurrence in a",
        "    paragraph is not sequence, and neither is the order a vendor",
        "    chose to narrate things in.",
        "",
        "You CANNOT split a chunk here — the gate has no split. If a chunk",
        "covers two procedures, say so with reject/under_chunked rather than",
        "proposing to drop it and retype both halves from memory.",
        "",
        "Not yours at this gate: the AND/OR/XOR branch operators and the",
        "attack-condition partitions. They are derived from the flow geometry",
        "your `edges` would change, so they are settled after you, not by you.",
    ]

    if is_sequential:
        lines += [
            "",
            "The pipeline read this source as SEQUENTIAL, so the chunks carry",
            "flow edges and the bundle will ship attack-flow sequencing.",
        ]
    else:
        lines += [
            "",
            "The pipeline read this source as NON-SEQUENTIAL — a catalogue of",
            "procedures rather than one narrated intrusion. Disconnected",
            "chunks are the CORRECT output here, and the bundle will ship no",
            "sequencing at all. Do not add edges to 'tidy up' the graph. If",
            "that reading is wrong, say so in overall_notes: it is a",
            "source-level call, not something an edge fixes.",
        ]
    rationale = (state.get("sequentiality_rationale") or "").strip()
    if rationale:
        lines.append(f"  Its reasoning: {rationale}")

    lines += ["", *_chunk_flow_summary(chunks), "", "=" * 66, ""]

    if not chunks:
        lines.append("(no chunks — the chunker produced nothing, which is itself the finding)")
        return "\n".join(lines)

    for c in chunks:
        lines += _fmt_chunk(c)
        lines.append("")
    return "\n".join(lines)


# =============================================================================
# Gate 1 — procedures + techniques
# =============================================================================

_REJECT_REASONS = [
    "wrong_technique", "hallucinated", "too_vague",
    "wrong_technique_granularity", "bad_chunk_boundary",
]

# Mirrors RemoveReason in models.py. Kept separate from _REJECT_REASONS
# because a reject re-runs extraction and a remove does not: "not a
# procedure" and "duplicate" have no re-run that could fix them.
_REMOVE_REASONS = ["not_a_procedure", "duplicate", "other"]

REVIEW_PROCEDURES_TOOL: dict[str, Any] = {
    "name": "review_procedures",
    "description": (
        "Submit your review of the drafted procedures and their technique "
        "mappings. Speak where you want a change; approve what is right."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "drafts": {
                "type": "array",
                "description": (
                    "One entry per draft you want to change, plus any you "
                    "want to explicitly endorse. Drafts you omit ship as-is."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "draft_id": {"type": "string"},
                        "action": {
                            "type": "string",
                            "enum": ["approve", "edit", "reject", "remove"],
                            "description": (
                                "approve = ships as-is. edit = ships with "
                                "your corrections. reject = send back for "
                                "re-extraction. remove = drop it entirely. "
                                "Prefer edit over reject when the problem is "
                                "one bad technique on an otherwise sound "
                                "procedure — a reject discards the good work "
                                "with the bad. Prefer remove over reject when "
                                "the draft simply should not exist — it is "
                                "not a procedure, or it duplicates another: "
                                "a reject re-runs extraction over the SAME "
                                "chunk text, so the identical draft comes "
                                "back and nothing is fixed. Reject only when "
                                "a fresh extraction pass could plausibly "
                                "produce a better answer."
                            ),
                        },
                        "reject_reason": {
                            "type": "string",
                            "enum": _REJECT_REASONS,
                            "description": (
                                "Required with action=reject. This picks the "
                                "re-run route: bad_chunk_boundary re-chunks "
                                "the whole source, everything else re-maps "
                                "techniques. Choose accordingly."
                            ),
                        },
                        "remove_reason": {
                            "type": "string",
                            "enum": _REMOVE_REASONS,
                            "description": (
                                "Required with action=remove. Say why it is "
                                "being dropped — the reason feeds the pattern "
                                "learner, and a removal without one teaches "
                                "it nothing."
                            ),
                        },
                        "remove_technique_ids": {
                            "type": "array", "items": {"type": "string"},
                            "description": (
                                "ATT&CK IDs to drop from this draft, with "
                                "action=edit."
                            ),
                        },
                        **_SHARED_REC_PROPS,
                    },
                    "required": ["draft_id", "action", "confidence", "rationale"],
                },
            },
            "promotions": {
                "type": "array",
                "description": (
                    "Possible-bucket picks from the review lane that belong "
                    "in the bundle after all."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {"type": "string"},
                        "technique_id": {"type": "string"},
                        **_SHARED_REC_PROPS,
                    },
                    "required": ["chunk_id", "technique_id", "confidence", "rationale"],
                },
            },
            "overall_notes": {"type": "string"},
        },
        "required": ["drafts", "promotions"],
    },
}


def _fmt_technique(t: dict, vendor_tids: set[str]) -> str:
    """One technique pick, with every signal the pipeline computed about it.

    This is the reviewer's real advantage over the extractor. The picker saw
    a candidate pool and a chunk; it never saw its own quote scored against
    the report, whether the report's own ATT&CK table corroborates the pick,
    or that a guardrail already demoted it. All of that exists in state and
    none of it was ever shown to anything that could act on it.
    """
    tid = t.get("technique_id", "?")
    bits = [f"      {tid} {t.get('technique_name', '')} [{t.get('tactic', '?')}]"]
    meta = [f"conf={t.get('confidence', 0):.2f}", f"bucket={t.get('confidence_bucket', '?')}"]
    if t.get("provenance"):
        meta.append(f"via={t['provenance']}")

    # Corroboration by the report's own ATT&CK mapping table. Deliberately
    # NOT treated as proof: a vendor table naming a technique is a claim, not
    # a witnessed behavior, which is why that section is excluded from the
    # quote-grounding corpus. It is a strong hint in both directions.
    parent = tid.split(".")[0]
    if tid in vendor_tids:
        meta.append("REPORT'S OWN TABLE lists this")
    elif parent in vendor_tids:
        meta.append(f"report's table lists the parent {parent}")

    if t.get("quote_unsupported_by_source"):
        meta.append(
            f"QUOTE UNSUPPORTED BY THE REPORT (support "
            f"{t.get('quote_source_support', '?')})"
        )
    if t.get("bucket_capped"):
        meta.append(f"auto-demoted: {t['bucket_capped']}")
    if t.get("denylisted"):
        meta.append("DENYLISTED by an analyst rule")
    if t.get("analyst_promoted"):
        meta.append("promoted by an analyst on an earlier pass")
    bits.append("        " + " | ".join(meta))

    quote = (t.get("source_quote") or "").strip()
    if quote:
        bits.append(f'        quote: "{quote[:200]}"')
    rationale = (t.get("rationale") or "").strip()
    if rationale:
        bits.append(f"        picker said: {rationale[:200]}")
    return "\n".join(bits)


def build_procedures_turn(state: dict) -> str:
    """The Gate 1 user turn: drafts, their techniques, and the evidence."""
    drafts = state.get("drafts") or []
    mappings = state.get("technique_mappings") or {}
    review_lane = state.get("technique_mappings_for_review") or {}
    chunks_by_id = {
        c.get("chunk_id"): c for c in (state.get("chunks") or []) if c.get("chunk_id")
    }
    proposals = state.get("proposals_by_chunk") or {}

    vendor_tids: set[str] = set()
    try:
        vendor_tids, _ = extract_vendor_technique_ids(
            state.get("classified_sections") or []
        )
    except Exception:  # noqa: BLE001 — a missing table must not fail the review
        vendor_tids = set()

    lines = [
        "GATE 1 — PROCEDURE AND TECHNIQUE REVIEW",
        "",
        f"{len(drafts)} procedures were drafted from this report.",
        "",
        "Each draft below carries its techniques and everything the pipeline",
        "already computed about them — quote grounding, guardrail demotions,",
        "and whether the report's own ATT&CK table corroborates the pick. The",
        "stage that CHOSE these techniques saw none of that. You do.",
        "",
        "What goes wrong here most often, in order:",
        "  - A technique that is real in the report but does not serve THIS",
        "    procedure's objective. It drifted in from a neighboring",
        "    sentence. Remove it; do not reject the draft over it.",
        "  - Wrong granularity: a sub-technique asserted where only the parent",
        "    is evidenced, or a parent left where the report names the",
        "    specific mechanism.",
        "  - A pick resting on a quote the report does not support. Those are",
        "    flagged below. Treat the reasoning as unverified, not as wrong.",
        "  - A description that asserts more than the report does. You cannot",
        "    rewrite wording here — there is no editing surface for it — so",
        "    reject with `too_vague` when it is bad enough to matter, and",
        "    otherwise say so in overall_notes rather than inventing a",
        "    correction nobody can apply.",
        "",
        "Prefer `edit` with remove_technique_ids over `reject`. A reject",
        "re-runs extraction and throws away the parts that were right.",
    ]

    if vendor_tids:
        lines += [
            "",
            f"THE REPORT'S OWN ATT&CK TABLE names {len(vendor_tids)} techniques: "
            + ", ".join(sorted(vendor_tids)),
            "  Corroboration, not proof. A vendor naming a technique in a",
            "  summary table is a claim; the narrative is what witnesses it.",
            "  Useful both ways: a pick the table also names is more likely",
            "  right, and a table entry nothing picked may be a real miss.",
        ]

    lines += ["", "=" * 66, ""]

    for d in drafts:
        did = d.get("draft_id", "?")
        cid = d.get("chunk_id", "")
        lines.append(f"DRAFT {did}  (chunk {cid})")
        lines.append(f"  name: {d.get('name', '')}")
        desc = (d.get("description") or "").strip()
        if desc:
            lines.append(f"  description: {desc[:700]}")
        lines.append(f"  confidence: {d.get('confidence', '?')}")
        if d.get("platforms"):
            lines.append(f"  platforms: {', '.join(d['platforms'])}")
        for key in ("command_lines", "raw_command_lines"):
            if d.get(key):
                lines.append(f"  {key}:")
                lines += [f"      {c[:200]}" for c in d[key][:6]]
        if d.get("detail_gap"):
            lines.append("  detail_gap: the drafter flagged this as thin on specifics")

        obj = (proposals.get(cid) or {}).get("objective")
        if obj:
            lines.append(f"  declared objective: {obj}")

        chunk = chunks_by_id.get(cid) or {}
        if chunk.get("text"):
            lines.append(f"  chunk text: {chunk['text'][:600]}")
        if chunk.get("source_excerpt"):
            lines.append(f'  chunk cites: "{chunk["source_excerpt"][:400]}"')

        picks = mappings.get(cid) or []
        lines.append(f"  TECHNIQUES IN THE BUNDLE ({len(picks)}):")
        lines += [_fmt_technique(t, vendor_tids) for t in picks] or ["      (none)"]

        lane = review_lane.get(cid) or []
        if lane:
            lines.append(f"  HELD FOR REVIEW — not in the bundle ({len(lane)}):")
            lines.append("      Promote any that belong. They were parked for")
            lines.append("      weak evidence, which is a default, not a verdict.")
            lines += [_fmt_technique(t, vendor_tids) for t in lane]
        lines.append("")

    return "\n".join(lines)


# =============================================================================
# Gate 2 — bundle relationships
# =============================================================================

_RELATIONSHIP_TYPES = [
    "uses", "targets", "precedes", "indicates",
    "has-observable", "attributed-to", "mitigates", "detects",
    "exploits", "component-of",
]

REVIEW_BUNDLE_TOOL: dict[str, Any] = {
    "name": "review_relationships",
    "description": (
        "Submit your review of the bundle's relationships. Speak where you "
        "want a change; the rest ship as derived."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "relationships": {
                "type": "array",
                "description": (
                    "One entry per relationship you want to change, plus any "
                    "you want to explicitly endorse. Omitted ones ship as-is."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "rel_id": {"type": "string"},
                        "action": {
                            "type": "string",
                            "enum": ["approve", "edit", "remove"],
                        },
                        "edited_rel_type": {
                            "type": "string", "enum": _RELATIONSHIP_TYPES,
                            "description": "Corrected type, with action=edit.",
                        },
                        "edited_source": {"type": "string"},
                        "edited_target": {"type": "string"},
                        **_SHARED_REC_PROPS,
                    },
                    "required": ["rel_id", "action", "confidence", "rationale"],
                },
            },
            "overall_notes": {"type": "string"},
        },
        "required": ["relationships"],
    },
}


def build_bundle_turn(state: dict) -> str:
    """The Gate 2 user turn: the reviewable relationships, grouped.

    Two deliberate omissions, both stated in the prompt so the reviewer knows
    what it is NOT seeing:

      * Relationships marked `reviewable: False` — the x-procedure ->
        attack-pattern technique mappings. This same reviewer already judged
        those at Gate 1 and the analyst has already ruled on them; showing
        them again invites it to relitigate a settled decision, and costs
        tokens proportional to the largest group in the bundle.
      * Nothing else. Everything reviewable is shown, because a missing edge
        is as much a defect as a wrong one and it cannot be spotted from a
        filtered list.

    Grouped by source procedure rather than listed flat. ~87 flat rows is
    exactly the shape that makes an analyst skim, and a reviewer reading a
    flat list has the same problem.
    """
    preview = state.get("relationship_preview") or []
    reviewable = [r for r in preview if r.get("reviewable")]
    inherent = len(preview) - len(reviewable)

    lines = [
        "GATE 2 — BUNDLE RELATIONSHIP REVIEW",
        "",
        f"The bundle has {len(preview)} relationships. {len(reviewable)} are",
        "open for review below.",
    ]
    if inherent:
        lines += [
            f"The other {inherent} "
            + ("is a technique mapping" if inherent == 1 else "are technique mappings")
            + " (procedure -> attack-pattern)",
            "derived from decisions already made at Gate 1. They are not shown",
            "and are not yours to change here.",
        ]
    lines += [
        "",
        "YOUR ADVANTAGE HERE IS MEMORY, NOT STRUCTURE.",
        "A schema check can find a malformed edge. You reviewed the entities",
        "and the procedures that these edges connect, so you can find an edge",
        "that contradicts what was decided earlier — a `uses` pointing at a",
        "tool you recommended removing, an `attributed-to` naming an actor the",
        "analyst struck, a `precedes` implying an order the chunking did not",
        "support. That is the failure a structural checker cannot see, and it",
        "is why this gate is worth your attention rather than a lint rule.",
        "",
        "Also worth checking:",
        "  - Direction. The procedure is the actor and the tool is the",
        "    instrument: procedure -> tool, never the reverse.",
        "  - An edge to an entity the report never connects to this procedure.",
        "    Co-occurrence in a paragraph is not a relationship.",
        "  - `targets` naming an organization that is the report's publisher or",
        "    author rather than a victim.",
        "",
        "=" * 66,
        "",
    ]

    if not reviewable:
        lines.append("(no reviewable relationships)")
        return "\n".join(lines)

    by_source: dict[str, list[dict]] = {}
    for rel in reviewable:
        by_source.setdefault(rel.get("source_name", "(unnamed)"), []).append(rel)

    for source_name, rels in by_source.items():
        stype = rels[0].get("source_type", "")
        lines.append(f"{source_name}  [{stype}]")
        for rel in rels:
            lines.append(
                f"    [{rel.get('id', '?')}] --{rel.get('relationship_type', '?')}--> "
                f"{rel.get('target_name', '?')}  [{rel.get('target_type', '?')}]"
            )
        lines.append("")
    return "\n".join(lines)


REVIEWERS: dict[str, GateReviewer] = {
    "entities": GateReviewer(
        gate_key="entities",
        tool=REVIEW_ENTITIES_TOOL,
        output_model=Gate0Recommendations,
        build_turn=build_entities_turn,
        # Mirrors entity_extraction._ENTITY_FEEDBACK_CATEGORIES.
        feedback_categories=(
            "defender_ioc", "brand_as_malware",
            "false_positive_entity", "mis_attribution",
        ),
    ),
    "chunks": GateReviewer(
        gate_key="chunks",
        tool=REVIEW_CHUNKS_TOOL,
        output_model=GateChunksRecommendations,
        build_turn=build_chunks_turn,
        # Mirrors chunking._CHUNK_FEEDBACK_CATEGORIES.
        feedback_categories=(
            "over_chunked", "under_chunked", "missing_procedure",
            "wrong_predecessor", "parallel_capability_misordered",
            "thin_initial_access", "artifact_loss",
        ),
    ),
    "procedures": GateReviewer(
        gate_key="procedures",
        tool=REVIEW_PROCEDURES_TOOL,
        output_model=Gate1Recommendations,
        build_turn=build_procedures_turn,
        # Gate 1 judges drafts AND their technique mappings, so it needs the
        # union of what technique_extraction and drafting each ask for.
        feedback_categories=(
            "missing_tactic", "wrong_technique",
            "thin_initial_access", "orphan_ioc",
            "artifact_loss", "wrong_relationship", "missing_procedure",
        ),
    ),
    "bundle": GateReviewer(
        gate_key="bundle",
        tool=REVIEW_BUNDLE_TOOL,
        output_model=Gate2Recommendations,
        build_turn=build_bundle_turn,
        # wrong_relationship is the only category mapped to the
        # `relationships` area. mis_attribution is filed under entities but is
        # about attribution correctness, which is exactly what an
        # attributed-to edge encodes.
        feedback_categories=("wrong_relationship", "mis_attribution"),
    ),
}
