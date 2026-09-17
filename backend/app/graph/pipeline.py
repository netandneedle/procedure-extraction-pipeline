"""LangGraph pipeline graph: nodes, edges, interrupts, and conditional routing.

HOW THIS FILE WORKS:
- build_pipeline() creates a StateGraph from PipelineState.
- Each node is a Python function that receives the full state and returns
  a partial dict of fields to update.
- Edges connect nodes sequentially. Conditional edges branch based on
  state values (gate rejection routing, hard-fail guards).
- interrupt_before on the four gate nodes makes LangGraph freeze execution
  and persist state to PostgreSQL. The graph resumes when the API writes the
  analyst's review into state via graph.update_state() and streams again.

NODE TYPES (sixteen nodes, in pipeline order):
- Deterministic (no model call): parse_and_validate, normalize,
  serialize_stix, validate_bundle, distribute.
- LLM (through app.nodes.llm.llm_adapter, provider-neutral):
  extract_figures, classify_sections, extract_entities, chunk_behaviors,
  extract_techniques, draft_procedures, synthesize_feedback.
- Gates: gate_0 (entities), gate_chunks (chunks, i.e. the procedures),
  gate_1 (technique mappings on the drafts), gate_2 (the bundle). The graph
  pauses before each; the API writes the raw decisions to state; the gate
  node then runs and applies them (validation, edits, routing). Each gate
  auto-approves when its key in state["gates_enabled"] is False — see
  is_gate_enabled() in app.graph.state; the runner in api/routes/pipeline.py
  resumes past such gates without an analyst prompt.

CONDITIONAL ROUTING (only at gates and hard-fail guards):
- After parse_and_validate: status == failed -> END (a missing or
  unparseable file must not flow on as empty text: every later node skips
  on "no parsed_text" and the run surfaces at Gate 0 with zero entities and
  a Submit button); otherwise -> extract_figures.
- After chunk_behaviors: status == failed -> END (zero chunks or a
  validation failure must not flow into a gate that would auto-approve
  the empty list); otherwise -> gate_chunks.
- After gate_chunks: reject -> chunk_behaviors; otherwise -> extract_techniques.
- After gate_1: a BAD_CHUNK_BOUNDARY rejection -> chunk_behaviors; any other
  rejection -> extract_techniques; otherwise -> normalize. Bad chunking wins,
  because wrong chunks make the technique mapping wrong too.
- After gate_2: approved -> serialize_stix; rejected -> normalize.
- After validate_bundle: hard-fail -> END (the source is marked failed and
  distribute never runs); otherwise -> distribute -> synthesize_feedback.
"""

from langgraph.graph import END, StateGraph

from app.graph.state import (
    GateAction,
    Gate1RejectReason,
    PipelineState,
    PipelineStatus,
)

# Deterministic nodes
from app.nodes.deterministic.parse import parse_and_validate
from app.nodes.deterministic.normalization import normalize
from app.nodes.deterministic.serialization import serialize_stix
from app.nodes.deterministic.bundle_validator import validate_bundle
from app.nodes.deterministic.distribution import distribute

# LLM nodes
from app.nodes.llm.entity_extraction import extract_entities
from app.nodes.llm.feedback_synthesis import synthesize_feedback
from app.nodes.llm.figure_extraction import extract_figures
from app.nodes.llm.chunking import chunk_behaviors, classify_sections
from app.nodes.llm.technique_extraction import extract_techniques
from app.nodes.llm.drafting import draft_procedures

# Gate nodes
from app.nodes.gates import gate_0, gate_1, gate_2, gate_chunks


# =============================================================================
# Conditional edge routing functions
#
# These inspect state after a gate node runs and return the name of the
# next node to execute. LangGraph calls these to decide the edge.
# =============================================================================

def route_after_parse(state: PipelineState) -> str:
    """Route after parse_and_validate.

    parse_and_validate catches its own errors (file not found, empty or
    unsupported content) and returns status=failed + error rather than
    raising. Without this guard that update flowed into extract_figures,
    classify_sections and extract_entities — each of which logs "no
    parsed_text" and returns — and the run paused at gate_0 showing zero
    entities and an Approve button. Seen live 2026-09-12 when a container
    recreate dropped the uploads directory: the Retry landed the analyst
    at an empty gate instead of a Failed card.
    """
    if state.get("status") == PipelineStatus.FAILED.value:
        return "__end__"
    return "extract_figures"


def route_after_chunk_behaviors(state: PipelineState) -> str:
    """Route after the chunk_behaviors node.

    - Hard-fail (status == FAILED): route to END. chunk_behaviors catches
      its own exceptions (LLMValidationError, the zero-chunks RuntimeError)
      and sets status=failed + error on the state update rather than
      raising. Without this guard the failed update flows into gate_chunks,
      which auto-approves the empty chunk list and OVERWRITES the failed
      status — the error is silently swallowed and the source can even
      reach "completed" with no chunks. Mirrors route_after_validate's
      hard-fail-to-END pattern.
    - Otherwise: proceed to gate_chunks for review.
    """
    if state.get("status") == PipelineStatus.FAILED.value:
        return "__end__"
    return "gate_chunks"


def route_after_gate_chunks(state: PipelineState) -> str:
    """Route after gate_chunks based on per-chunk rejection decisions.

    - chunks_rejection_routing == "chunk_behaviors": re-run chunking with
      analyst feedback (e.g. analyst flagged bad boundaries).
    - otherwise: proceed to extract_techniques.
    """
    if state.get("chunks_rejection_routing") == "chunk_behaviors":
        return "chunk_behaviors"
    return "extract_techniques"


def route_after_gate_1(state: PipelineState) -> str:
    """Route after Gate 1 based on rejection decisions.

    Routing logic:
    - If any draft was rejected with BAD_CHUNK_BOUNDARY, route back to
      chunk_behaviors to re-chunk with analyst feedback.
    - If any draft was rejected for other reasons (WRONG_TECHNIQUE, etc.),
      route back to extract_techniques to re-map.
    - If all drafts approved (or mix of approve/remove with no rejects),
      proceed to normalize.

    Priority: BAD_CHUNK_BOUNDARY > other rejections > approve.
    Rationale: If chunking is wrong, technique extraction is also wrong,
    so we fix the earlier stage first.
    """
    routing = state.get("gate1_rejection_routing")

    if routing == "chunk_behaviors":
        return "chunk_behaviors"
    elif routing == "extract_techniques":
        return "extract_techniques"
    else:
        return "normalize"


def route_after_gate_2(state: PipelineState) -> str:
    """Route after Gate 2 based on approval decision.

    - Approved: proceed to serialize_stix
    - Rejected: route back to normalize with feedback
    """
    decision = state.get("gate2_decision", {})

    if decision.get("approved", False):
        return "serialize_stix"
    else:
        return "normalize"


def route_after_validate(state: PipelineState) -> str:
    """Route after the validate_bundle node based on hard-fail outcome.

    - Hard-fail (bundle_validation_failed=True): route to END, skipping
      distribute and synthesize_feedback. The source lands in the Failed
      column with structured corrections in state["bundle_corrections"]
      for analyst inspection. There is no automated re-route to an earlier
      gate: fixing a hard-fail means re-running the source, which the
      analyst drives from the Kanban card.
    - Otherwise: distribute. The bundle is structurally sound, persistence
      can proceed.
    """
    if state.get("bundle_validation_failed", False):
        return "__end__"
    return "distribute"


# =============================================================================
# Graph construction
# =============================================================================

def build_pipeline() -> StateGraph:
    """Build and compile the extraction pipeline graph.

    The graph follows this flow:

        parse_and_validate
              | \
              |  status == failed -> END
              |
        extract_figures           <-- vision LLM pass over each figure
              |                       (skipped when Source.extract_figures=False
              |                        or source_type lacks images)
              |
        classify_sections         <-- which sections carry behavior
              |
        extract_entities
              |
          [gate_0]                <-- interrupt: analyst reviews entities
              |
        chunk_behaviors
              | \\                    status == failed -> END
              |
          [gate_chunks]           <-- interrupt: analyst reviews chunks
              | \\                    reject -> chunk_behaviors (re-chunk)
              |  \\                   approve -> extract_techniques
              |
        extract_techniques
              |
        draft_procedures
              |
          [gate_1]                <-- interrupt: analyst reviews procedures
           /  |  \\
          /   |   \\                conditional routing based on
         /    |    \\               rejection reason
    chunk_   extract_   normalize
    behaviors techniques     |
                          [gate_2]    <-- interrupt: analyst reviews relationships
                           /    \\
                          /      \\    conditional: approve/reject
                  normalize    serialize_stix
                                    |
                              validate_bundle   <-- check-and-correct;
                                /     \\              hard-fail routes to END
                          distribute   END           (skipping distribute)
                              |
                    synthesize_feedback     <-- post-run: read all gate
                              |                       decisions, emit
                              |                       FeedbackPattern rows
                              |                       into postgres for
                              |                       cross-source learning
                             END

    Returns:
        A compiled StateGraph ready to be invoked with a PipelineState.
    """
    graph = StateGraph(PipelineState)

    # ── Register nodes ───────────────────────────────────────────────
    graph.add_node("parse_and_validate", parse_and_validate)
    graph.add_node("extract_figures", extract_figures)
    graph.add_node("classify_sections", classify_sections)
    graph.add_node("extract_entities", extract_entities)
    graph.add_node("gate_0", gate_0)
    graph.add_node("chunk_behaviors", chunk_behaviors)
    graph.add_node("gate_chunks", gate_chunks)
    graph.add_node("extract_techniques", extract_techniques)
    graph.add_node("draft_procedures", draft_procedures)
    graph.add_node("gate_1", gate_1)
    graph.add_node("normalize", normalize)
    graph.add_node("gate_2", gate_2)
    graph.add_node("serialize_stix", serialize_stix)
    graph.add_node("validate_bundle", validate_bundle)
    graph.add_node("distribute", distribute)
    graph.add_node("synthesize_feedback", synthesize_feedback)

    # ── Entry point ──────────────────────────────────────────────────
    graph.set_entry_point("parse_and_validate")

    # ── Conditional edge: parse_and_validate -> (extract_figures | END) ──
    # END on a failed parse; see route_after_parse.
    graph.add_conditional_edges(
        "parse_and_validate",
        route_after_parse,
        {
            "extract_figures": "extract_figures",
            "__end__": END,
        },
    )

    # ── Sequential edges (simple A -> B) ─────────────────────────────
    # extract_figures -> extract_entities chain. extract_figures runs a
    # vision LLM pass over each figure (when Source.extract_figures=True)
    # and inlines the extracted text into parsed_text before downstream
    # nodes see it.
    # classify_sections runs here — after extract_figures (which rewrites
    # parsed_text, so section line-ranges must be computed against the final
    # text) and before extract_entities (which needs the labels to skip
    # remediation and detection sections).
    graph.add_edge("extract_figures", "classify_sections")
    graph.add_edge("classify_sections", "extract_entities")
    graph.add_edge("extract_entities", "gate_0")
    graph.add_edge("gate_0", "chunk_behaviors")
    graph.add_edge("extract_techniques", "draft_procedures")
    graph.add_edge("draft_procedures", "gate_1")

    # ── Conditional edge: chunk_behaviors -> (gate_chunks | END) ──────────
    # END on hard-fail so a swallowed chunker error doesn't get overwritten
    # by gate_chunks auto-approval (Batch H category B).
    graph.add_conditional_edges(
        "chunk_behaviors",
        route_after_chunk_behaviors,
        {
            "gate_chunks": "gate_chunks",
            "__end__": END,
        },
    )

    # ── Conditional edge: gate_chunks -> (chunk_behaviors | extract_techniques) ─
    graph.add_conditional_edges(
        "gate_chunks",
        route_after_gate_chunks,
        {
            "chunk_behaviors": "chunk_behaviors",
            "extract_techniques": "extract_techniques",
        },
    )

    # ── Conditional edge: Gate 1 -> (chunk_behaviors | extract_techniques | normalize)
    graph.add_conditional_edges(
        "gate_1",
        route_after_gate_1,
        {
            "chunk_behaviors": "chunk_behaviors",
            "extract_techniques": "extract_techniques",
            "normalize": "normalize",
        },
    )

    # ── Sequential edge: normalize -> gate_2 ─────────────────────────
    graph.add_edge("normalize", "gate_2")

    # ── Conditional edge: Gate 2 -> (normalize | serialize_stix)
    graph.add_conditional_edges(
        "gate_2",
        route_after_gate_2,
        {
            "normalize": "normalize",
            "serialize_stix": "serialize_stix",
        },
    )

    # ── Sequential edges: serialization -> validate_bundle ──────────
    # validate_bundle does check-and-correct on the assembled bundle:
    # recovery for dangling refs, auto-fix for SRO direction / dedup /
    # fingerprint / role enums, hard-fail for schema issues / brand-as-
    # malware / unrecoverable refs / cycles. On hard-fail it routes
    # straight to END (skipping distribute) so failed bundles don't
    # land in Neo4j or the bundle store.
    graph.add_edge("serialize_stix", "validate_bundle")
    graph.add_conditional_edges(
        "validate_bundle",
        route_after_validate,
        {
            "distribute": "distribute",
            "__end__": END,
        },
    )

    # synthesize_feedback runs AFTER distribute so the bundle has shipped
    # before the synthesizer touches anything; failures inside synthesize_feedback
    # are caught and logged but never block END.
    graph.add_edge("distribute", "synthesize_feedback")
    graph.add_edge("synthesize_feedback", END)

    return graph


def compile_pipeline(**kwargs):
    """Build the graph and compile it with optional config.

    This is the function the API layer calls. Pass a checkpointer
    to enable state persistence:

        from app.graph.checkpointer import get_checkpointer

        checkpointer = get_checkpointer()
        app = compile_pipeline(checkpointer=checkpointer)

    The default `interrupt_before` list pauses before every analyst gate
    (gate_0, gate_chunks, gate_1, gate_2). Override only if you need a
    different pause set — main.py uses the default.

    The interrupt_before list tells LangGraph to freeze the graph
    BEFORE entering those nodes. The checkpointer persists the
    frozen state to PostgreSQL. When the API calls
    app.update_state(thread_id, new_values), the graph thaws
    and continues from where it paused.

    Args:
        **kwargs: Passed to graph.compile(). Common options:
            checkpointer: AsyncPostgresSaver instance
            interrupt_before: list of node names to pause before

    Returns:
        A compiled LangGraph application ready for invocation.
    """
    graph = build_pipeline()

    # Default: always interrupt before gate nodes
    if "interrupt_before" not in kwargs:
        kwargs["interrupt_before"] = ["gate_0", "gate_chunks", "gate_1", "gate_2"]

    return graph.compile(**kwargs)
