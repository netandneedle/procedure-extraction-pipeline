"""draft_procedures node: Stage 4 of the extraction pipeline.

LLM node. Assembles procedure candidates from chunks + technique mappings.
Each chunk becomes one ProcedureDraft targeting the x-procedure v0.5.0 schema.

WHAT THIS NODE DOES:
    1. For each chunk, combine the chunk text, its technique mappings,
       entity context, and metadata into a ProcedureDraft
    2. Generate a structured procedure name: "[Verb] [Object] via [Tool/Method]"
    3. Generate a three-part description:
       a) Action summary (what happened)
       b) Mechanism (how it happened, tools used)
       c) Observation guidance (what a defender would see)
    4. Map platforms using OpenTide vocabulary
    5. Extract verbatim command lines (from the source excerpt or the Gate 0
       ioc_command_line entities, never fabricate; a deterministic check
       drops anything not found in the source)
    6. Inherit sequencing from the chunk (sequence_index, predecessors, etc.)
    7. Set initial confidence from behavioral_confidence

DESIGN DECISIONS:
    - One LLM call per batch (all chunks at once) for consistency
    - The LLM generates name, description, platforms, and identifies
      command lines. It does NOT generate IOCs, detection rules, or
      modify technique mappings. Those come from other stages.
    - Command lines are COPIED from the source, never generated; the
      grounding check enforces it.
      Stored as raw_command_lines (staging data). The serializer
      converts them to Process SCOs and populates
      components_refs with STIX IDs.
    - Temporal fields (first_observed, etc.) come from metadata or
      chunk context, not LLM generation.
    - Sequencing fields (sequence_index, predecessor_indices) are
      inherited from chunks. The normalize node inverts them into
      draft-level effect_refs (forward edges); the serializer turns those
      into PRECEDES SROs. No flow_ref — removed in v0.5.0-draft.

READS: chunks, technique_mappings, validated_entities, metadata
WRITES: drafts, status, current_node
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from difflib import SequenceMatcher

from app.graph.state import (
    PipelineState,
    PipelineStatus,
)
from app.nodes.llm.llm_adapter import call_llm
from app.nodes.llm.tool_models import DraftProceduresOutput
from app.services.feedback_examples import relevant_examples_cached
from app.services.feedback_patterns import relevant_addendum_cached
from app.utils.refang import refang

logger = logging.getLogger(__name__)


# Feedback-pattern categories relevant to drafting judgments. Read at
# prompt-build time and prepended as an analyst-feedback addendum.
_DRAFT_FEEDBACK_CATEGORIES = (
    "artifact_loss",
    "orphan_ioc",
    "wrong_relationship",
    "missing_procedure",
    "thin_initial_access",
    # Catch-all, and the fallback when the synthesizer omits a category.
    # Every node reads it — see the note in entity_extraction.
    # Pinned by tests/test_contracts.py.
    "other",
)


async def _fetch_feedback_addendum(state) -> str:
    """Analyst feedback patterns RELEVANT TO THIS SOURCE, as a prompt addendum.
    Source-relative hybrid retrieval, TTL-cached — see relevant_addendum_cached.
    Best-effort: returns "" on any failure."""
    return await relevant_addendum_cached(
        state, categories=_DRAFT_FEEDBACK_CATEGORIES,
        node="draft_procedures", limit=15,
    )


async def _fetch_feedback_examples(state) -> str:
    """Past analyst corrections most similar to this source, as demonstrations.

    A separate channel from `_fetch_feedback_addendum` on purpose: the rules
    are LLM-written generalizations and the examples are records, they fail in
    different ways, and keeping the fetches apart is what lets an ablation arm
    vary one without the other. Best-effort: "" on any failure.
    """
    return await relevant_examples_cached(
        state, areas=("techniques",), node="draft_procedures",
    )


# =============================================================================
# Tool definitions
# =============================================================================

DRAFT_PROCEDURES_TOOL = {
    "name": "draft_procedures",
    "description": (
        "Create structured procedure drafts from behavioral chunks and their "
        "ATT&CK technique mappings. Each chunk becomes one procedure draft."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "drafts": {
                "type": "array",
                "description": "One procedure draft per chunk.",
                "items": {
                    "type": "object",
                    "properties": {
                        "chunk_id": {
                            "type": "string",
                            "description": "The chunk_id this draft is based on.",
                        },
                        "name": {
                            "type": "string",
                            "description": (
                                "Procedure name following the pattern: "
                                "'[Verb] [Object] via [Tool/Method]'. "
                                "Examples: 'Download Web Shell via certutil', "
                                "'Execute Payload via PowerShell', "
                                "'Exploit ActiveMQ via CVE-2023-46604'. "
                                "The verb is ALWAYS the ADVERSARY'S action — "
                                "never a reporting verb (Discuss, Report, Note, "
                                "Describe, Propose, Identify, Assess), which "
                                "describes the vendor rather than the intrusion. "
                                "This holds when procedure_type is "
                                "'hypothetical': that field carries the "
                                "uncertainty, so the name neither hedges nor "
                                "appends qualifiers like '(Unconfirmed)'."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "Three-part description:\n"
                                "1. Action summary: What the adversary did (1 sentence)\n"
                                "2. Mechanism: How they did it, what tools/methods (1-2 sentences)\n"
                                "3. Observation: What a defender would observe (1 sentence)"
                            ),
                        },
                        "platforms": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Target platforms using OpenTide Threat Surface vocabulary "
                                "(title case, :: delimiter). Common values:\n"
                                "  Windows, Windows::Server, Windows::Desktop\n"
                                "  Linux, Linux::Ubuntu, Linux::RHEL, Linux::Debian\n"
                                "  macOS\n"
                                "  Mobile, Mobile::Android, Mobile::iOS\n"
                                "  Container Runtime, Container Runtime::Docker\n"
                                "  Cloud (use for cloud-targeting behaviors)\n"
                                "Use the broadest applicable category unless the source "
                                "explicitly identifies a specific sub-type (e.g., use "
                                "'Linux' unless the source names Ubuntu or RHEL)."
                            ),
                        },
                        "command_lines": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Command lines copied VERBATIM from the chunk's SOURCE "
                                "EXCERPT or from its CAPTURED COMMAND LINES list (the "
                                "analyst-validated ioc_command_line entities). The chunk "
                                "description is a paraphrase and is NOT a source of "
                                "commands. Do NOT generate, reconstruct, or fabricate. "
                                "Include only the commands THIS procedure executes; if "
                                "neither the excerpt nor the captured list shows one, "
                                "return an empty array."
                            ),
                        },
                        "tools_used": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Names of TOOL entities (from the entity context shown "
                                "for this chunk) that THIS procedure ACTUALLY uses. "
                                "Pull from the chunk's stated tools — not every tool "
                                "in the source. Empty array if the procedure doesn't "
                                "leverage any extracted tool. The serializer uses this "
                                "to emit per-procedure procedure→tool USES SROs; "
                                "fan-out to every tool in the source is a known prior "
                                "bug we are explicitly fixing."
                            ),
                        },
                        "malware_used": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Names of MALWARE entities (from the entity context) "
                                "that THIS procedure ACTUALLY deploys / executes / "
                                "embeds. Pull from the chunk's stated malware. Empty "
                                "array if not applicable. Same per-procedure filter "
                                "principle as tools_used."
                            ),
                        },
                        "attributed_actors": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Names of INTRUSION_SET entities (from the entity "
                                "context) that the source attributes THIS procedure "
                                "to. Usually the one actor the report is about — "
                                "include it even when the sentence says 'the group' "
                                "or 'the actor' rather than naming them. "
                                "CRITICAL: reports often name OTHER actors for "
                                "CONTRAST or comparison ('unlike X', 'similar to "
                                "techniques used by Y', 'previously attributed to "
                                "Z'). Do NOT list those — attributing a procedure to "
                                "an actor the report explicitly distinguishes is a "
                                "false-attribution error, the most damaging mistake "
                                "in threat intelligence. Empty array if the source "
                                "does not attribute this behavior to any named actor."
                            ),
                        },
                        "detail_gap": {
                            "type": "boolean",
                            "description": (
                                "True if the source provided sparse detail for this "
                                "action. Indicates the procedure may need enrichment "
                                "from additional sources."
                            ),
                        },
                        "procedure_type": {
                            "type": "string",
                            "enum": ["reporting", "hypothetical"],
                            "description": (
                                "'reporting' (default) — the source states the "
                                "adversary DID this. Use for anything observed, "
                                "detected, or described as having happened.\n"
                                "'hypothetical' — the source's AUTHORS propose this "
                                "as a possible or additional vector rather than a "
                                "confirmed one ('we also identified another potential "
                                "access vector', 'the actors could also have used X'). "
                                "The behavior is real enough to model, but the source "
                                "does not claim it happened in THIS intrusion.\n"
                                "Judge by who is speaking and how certain they are, "
                                "NOT by how much detail there is — a thinly-described "
                                "but confirmed action is still 'reporting'. When the "
                                "source hedges only the ATTRIBUTION or the tooling but "
                                "not whether it occurred, that is still 'reporting'."
                            ),
                        },
                    },
                    "required": [
                        "chunk_id", "name", "description",
                        "platforms", "command_lines",
                        "tools_used", "malware_used", "attributed_actors",
                        "detail_gap", "procedure_type",
                    ],
                },
            },
        },
        "required": ["drafts"],
    },
}


# =============================================================================
# System prompt
# =============================================================================

SYSTEM_PROMPT = """You are an expert CTI analyst creating structured procedure drafts from behavioral chunks.

YOUR TASK:
For each chunk and its ATT&CK technique mapping(s), create a procedure draft that follows the x-procedure v0.5.0 schema.

PROCEDURE NAMING:
Use the pattern: "[Verb] [Object] via [Tool/Method]"
- Verb: active verb describing the action (Download, Execute, Deploy, Exploit, Enumerate, Exfiltrate, etc.)
- Object: what was acted on (Web Shell, Payload, Credentials, Registry Key, etc.)
- Tool/Method: how it was done (certutil, PowerShell, CVE-2023-46604, RDP, etc.)
Examples:
  "Download Web Shell via certutil"
  "Execute Reconnaissance Commands via cmd.exe"
  "Exploit Apache ActiveMQ via CVE-2023-46604"
  "Deploy Cobalt Strike Beacon via DLL Sideloading"

The verb is ALWAYS the ADVERSARY'S action, never the report author's. Never
name a procedure with a reporting verb — Discuss, Report, Note, Describe,
Propose, Identify, Assess, Highlight, Observe — those describe what the
VENDOR did, and the procedure is not about them. Name the behavior the
adversary performed or would perform.

This applies unchanged when procedure_type is "hypothetical". The uncertainty
is carried by that field, so the NAME does not hedge and must not editorialise:
  WRONG: "Discuss Zerologon as Speculative Initial Access Vector via CVE-2020-1472"
  RIGHT: "Exploit Netlogon Elevation of Privilege via CVE-2020-1472"
Do not append qualifiers like "(Unconfirmed)", "(Possible)" or "(Speculative)"
to the name for the same reason.

DESCRIPTION FORMAT:
Write a four-part description as flowing prose (no headers, no bullets):

1. Objective sentence (REQUIRED to be the FIRST sentence): If the chunk
   block includes "OBJECTIVE (REQUIRED to lead description): <text>",
   COPY THAT SENTENCE VERBATIM as the description's opening. It states
   the procedure's specific adversarial objective — the north-star that
   scopes the procedure. Per this pipeline's procedure definition, the
   procedure object does NOT have a separate `objective` STIX field;
   the objective survives only as the description's lead sentence, so
   the prose must carry it intact.
   If no OBJECTIVE block is present (legacy path), write a one-sentence
   action summary instead.

2. Mechanism (1-2 sentences): How the adversary fulfilled the objective —
   tools, commands, methods, sequence of operations. Include specific
   details from the chunk text.

3. Observation (1 sentence): What a defender monitoring the environment
   would observe. Focus on observable artifacts: network traffic, file
   writes, process execution, registry modifications.

The procedure definition for this pipeline:
  "A procedure is a discrete, repeatable technical implementation that
   integrates one or more techniques, often spanning multiple tactics,
   to fulfill a specific adversarial objective as an atomic event within
   an attack sequence."
The objective sentence anchors that definition in every drafted procedure.

PLATFORM ASSIGNMENT:
- Use OpenTide Threat Surface vocabulary (title case, :: delimiter)
- Valid top-level: Windows, Linux, macOS, Mobile, Container Runtime, Cloud, Embedded, BSD
- Valid sub-types: Windows::Server, Windows::Desktop, Linux::Ubuntu, Linux::RHEL, Linux::Debian, Mobile::Android, Mobile::iOS, Container Runtime::Docker
- Only assign platforms explicitly supported by the behavior described
- Use the BROADEST applicable category by default. Only use sub-types when the source EXPLICITLY names the sub-type (e.g., "Ubuntu server" -> Linux::Ubuntu, "Windows Server 2019" -> Windows::Server::2019)
- If the source just says "Linux" or "Linux server" without naming a distro, use "Linux"
- If the source says "Windows" without specifying server/desktop, use "Windows"

COMMAND LINES:
- The chunk description is a paraphrase written by the chunker. It is NOT a source of commands.
- Copy commands VERBATIM from the chunk's SOURCE EXCERPT, or from its CAPTURED COMMAND LINES list (analyst-validated ioc_command_line entities, also listed globally under VALIDATED ENTITIES).
- A captured command belongs to a procedure only when THAT procedure executes it. Do not attach every captured command to every procedure.
- Never generate, reconstruct, complete, or fabricate command lines. A command that appears nowhere in the source will be dropped.
- If neither the excerpt nor the captured list shows a command for this procedure, return an empty array.

PER-PROCEDURE TOOL / MALWARE ATTRIBUTION:
The chunk block lists the tools and malware entities extracted from the source globally. tools_used and malware_used must capture only the entities THIS procedure actually leverages.
- Read the chunk text + the chunk's stated tools/malware (the chunk's own context).
- A procedure that fires `certutil` to download a payload uses the certutil tool — even if the source elsewhere mentions Rclone for a different procedure.
- A procedure that drops a malware loader uses that malware family.
- Default to the chunk's stated tools/malware. Cross-reference with the chunk's source_excerpt for confirmation. If the chunk doesn't actually leverage an extracted entity, EMIT EMPTY ARRAYS rather than dumping the global list.
- Match by entity NAME (the value the entity extractor captured). The pipeline resolves names to STIX IDs at serialization.
- Worked example: a 14-procedure ransomware source extracted 10 tools globally (NetScan, Rclone, AnyDesk, ...). Only the exfiltration procedure's tools_used should contain "Rclone"; the password-spray procedure's tools_used should contain "net.exe" / "PowerShell" — NOT every tool in the source.

PER-PROCEDURE ACTOR ATTRIBUTION:
attributed_actors names the INTRUSION_SET entities the source attributes THIS procedure to.
- In a single-actor report that is simply the actor, on every procedure — include them even when the sentence says "the group", "the actor", or "they" instead of the name. Attribution follows the report's meaning, not the literal string.
- Reports frequently name OTHER actors for CONTRAST: "unlike OtherGroup", "similar to techniques used by X", "previously attributed to Y", "co-opted the Z brand". Those actors must NOT appear in attributed_actors.
- Worked example: a report on UNC0001 states "the vendor assesses that the operations are independent" of OtherGroup / UNC0002, which appear only for comparison. Every procedure's attributed_actors is ["UNC0001"] — never the other two. Listing them asserted that OtherGroup used all ten of UNC0001's procedures, the exact opposite of what the report says.
- False attribution is the most damaging error in threat intelligence. When genuinely unsure whether the source attributes a procedure to an actor, emit an empty array and let the pipeline's single-actor fallback decide.

TECHNIQUE ASSIGNMENT ON PROCEDURES vs MALWARE:
Techniques that describe what a malware family CAN do (its capabilities, modules, features) do NOT belong on individual procedure drafts. They belong on the Malware SDO as uses relationships. Only assign techniques that describe what HAPPENED in the observed attack chain to procedure drafts.

Example: A StealC procedure "Inject StealC into svchost.exe via Process Injection" gets T1055 (what happened). StealC's credential theft capability (T1555.003) goes on the Malware SDO, NOT on this procedure, unless the chunk specifically describes credential theft executing during this intrusion.

DETAIL GAP:
- Set to true if the chunk is sparse (e.g., "the actor established persistence" without specifics)
- Set to false if the chunk has concrete details (tool names, command lines, specific methods)"""


# =============================================================================
# Node function
# =============================================================================

async def draft_procedures(state: PipelineState) -> dict:
    """Stage 4: Create procedure drafts from chunks + techniques.

    Each chunk becomes one ProcedureDraft with v0.5.0 schema fields.

    Args:
        state: Pipeline state with chunks, technique_mappings,
               validated_entities, and metadata.

    Returns:
        Dict with drafts, status, current_node.
    """
    chunks = state.get("chunks", [])
    technique_mappings = state.get("technique_mappings", {})
    validated_entities = state.get("validated_entities", [])
    metadata = state.get("metadata", {})
    parsed_text = state.get("parsed_text", "") or ""
    # proposals_by_chunk carries the per-chunk objective (and behavior_description
    # / spanned tactics) emitted by the technique-extraction propose step. The
    # objective anchors the description's lead sentence per the new procedure
    # definition. Empty dict if extract_techniques didn't run or the propose
    # call failed — in that case, drafting falls back to the prior format
    # without an objective lead.
    proposals_by_chunk = state.get("proposals_by_chunk", {}) or {}

    logger.info("draft_procedures: starting, %d chunks", len(chunks))

    update: dict = {
        "status": PipelineStatus.DRAFTING.value,
        "current_node": "draft_procedures",
    }

    if not chunks:
        logger.warning("draft_procedures: no chunks to draft")
        update["drafts"] = []
        return update

    try:
        # Build the prompt with chunks + their technique mappings + objectives
        prompt_text = _format_chunks_with_techniques(
            chunks, technique_mappings, proposals_by_chunk,
            validated_entities=validated_entities, parsed_text=parsed_text,
        )
        entity_context = _format_entity_context(validated_entities)

        feedback_addendum = await _fetch_feedback_addendum(state)
        feedback_addendum += await _fetch_feedback_examples(state)
        system = SYSTEM_PROMPT
        if feedback_addendum:
            system += feedback_addendum
        if entity_context:
            system += entity_context

        response = await call_llm(
            system=system,
            messages=[{
                "role": "user",
                "content": (
                    "Create procedure drafts for each of the following "
                    "chunks and their ATT&CK technique mappings:\n\n"
                    + prompt_text
                ),
            }],
            tools=[DRAFT_PROCEDURES_TOOL],
            tool_choice={"type": "tool", "name": "draft_procedures"},
            temperature=0.0,
            # The extraction model spends thinking tokens out of the same
            # max_tokens budget; 8192 left little room for a multi-chunk
            # draft pass on top.
            max_tokens=24000,
            output_model=DraftProceduresOutput,
        )

        raw_drafts = response.tool_output.get("drafts", [])
        drafts = _process_drafts(
            raw_drafts, chunks, technique_mappings, metadata,
            validated_entities=validated_entities, parsed_text=parsed_text,
        )

        # Re-apply analyst technique overrides from a previous Gate 1 review
        # that was routed back to re-chunking. Matches by chunk text hash.
        previous_overrides = state.get("previous_technique_overrides", [])
        if previous_overrides:
            applied = _apply_technique_overrides(drafts, chunks, previous_overrides)
            logger.info(
                "draft_procedures: re-applied %d/%d technique overrides from previous review",
                applied, len(previous_overrides),
            )

        logger.info(
            "draft_procedures: created %d drafts. tokens: in=%d, out=%d",
            len(drafts), response.input_tokens, response.output_tokens,
        )

        update["drafts"] = drafts
        # Clear overrides after consuming them
        if previous_overrides:
            update["previous_technique_overrides"] = []

    except Exception as e:
        logger.exception("draft_procedures: failed")
        update["error"] = f"Procedure drafting failed: {type(e).__name__}: {e}"
        update["status"] = PipelineStatus.FAILED.value
        update["drafts"] = []

    return update


# =============================================================================
# Prompt formatting
# =============================================================================

def _format_chunks_with_techniques(
    chunks: list[dict],
    technique_mappings: dict[str, list[dict]],
    proposals_by_chunk: dict[str, dict] | None = None,
    *,
    validated_entities: list[dict] | None = None,
    parsed_text: str = "",
) -> str:
    """Format chunks with their technique mappings for the LLM prompt.

    When proposals_by_chunk is provided (post-C+A+D), each chunk also
    surfaces the OBJECTIVE the propose step committed to. The drafting
    LLM is required (per system prompt) to lead the description with this
    objective sentence so the procedure object's description carries the
    objective inline (no separate STIX field).

    Each chunk also carries the two verbatim things the model is allowed to
    copy command lines from: its `source_excerpt`, and the Gate 0
    `ioc_command_line` entities that fall inside the chunk's text, excerpt,
    or `source_span` window of `parsed_text`. `chunk["text"]` alone is a
    1-3 sentence paraphrase, so a prompt that shows only that and demands
    verbatim commands gets an empty array on nearly every chunk — which is
    how bundles shipped with an empty `x_components_refs` on 8 of 9
    procedures.
    """
    proposals_by_chunk = proposals_by_chunk or {}
    captured = _captured_command_lines(validated_entities or [])
    parts = []
    for chunk in chunks:
        chunk_id = chunk["chunk_id"]
        techniques = technique_mappings.get(chunk_id, [])

        tech_strs = []
        for t in techniques:
            tech_strs.append(
                f"    - {t.get('technique_id', '?')} {t.get('technique_name', '')} "
                f"({t.get('tactic', '')}, confidence={t.get('confidence', 0):.1f})"
            )
        tech_block = "\n".join(tech_strs) if tech_strs else "    (no techniques mapped)"

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

        # Inject objective + spanned tactics from the propose step. Per the
        # new procedure definition, the procedure's description must lead
        # with the objective sentence; this surfaces it to the drafting LLM.
        proposal = proposals_by_chunk.get(chunk_id, {})
        objective_str = ""
        if proposal:
            objective = (proposal.get("objective") or "").strip()
            tactics = proposal.get("tactics") or []
            objective_lines = []
            if objective:
                objective_lines.append(f"  OBJECTIVE (REQUIRED to lead description): {objective}")
            if tactics:
                objective_lines.append(
                    f"  TACTICS SPANNED: {', '.join(tactics)}"
                )
            if objective_lines:
                objective_str = "\n" + "\n".join(objective_lines)

        excerpt = " ".join((chunk.get("source_excerpt") or "").split())
        excerpt_str = f"\n  SOURCE EXCERPT (verbatim): {excerpt}" if excerpt else ""
        chunk_cmds = _commands_for_chunk(chunk, captured, parsed_text)
        cmd_str = ""
        if chunk_cmds:
            cmd_str = "\n  CAPTURED COMMAND LINES (verbatim, analyst-validated):\n" + "\n".join(
                f"    - {c}" for c in chunk_cmds
            )

        parts.append(
            f"CHUNK [{chunk_id}] (seq={chunk.get('sequence_index', '?')}):\n"
            f"  {chunk['text']}{ctx_str}{objective_str}{excerpt_str}{cmd_str}\n"
            f"  Techniques:\n{tech_block}\n"
        )

    return "\n".join(parts)


# Characters of parsed_text kept around a chunk's source_span when deciding
# which captured command lines belong to it. Excerpts are 2-3 sentences and
# a report's command block usually follows the sentence that introduces it,
# so the window leans forward.
_SPAN_WINDOW_BEFORE = 500
_SPAN_WINDOW_AFTER = 1500


def _normalize_command(cmd: str) -> str:
    """Whitespace-collapsed, refanged, case-folded form for comparison.

    PDF extraction breaks long commands across lines, so runs of whitespace
    compare equal. Both sides are refanged because reports defang the URLs
    inside commands (`files.example[.]net`) and the model may or may not keep
    the brackets. Case is folded because the check exists to drop the
    clearly-fabricated, not to adjudicate `Cmd.exe` vs `cmd.exe`.
    """
    return refang(" ".join((cmd or "").split())).lower()


# A command that is not a verbatim substring of the source can still be in
# it. Figure transcriptions carry OCR noise — one espionage-RAT report's command
# block came through as `cur1 -o ...` and `content-1ength` — and the model
# returns the command a human would read. Anchored near-match: align the
# command on any of its longer tokens that the source contains, compare the
# window that alignment implies, keep the best ratio. On real output the
# OCR case scored 0.986, a command with an invented flag 0.911, and an
# analog built from a different filename 0.0 (no token anchors at all).
_NEAR_MATCH_THRESHOLD = 0.95
_NEAR_MATCH_MIN_TOKEN = 6
_NEAR_MATCH_MAX_ANCHORS = 40


def _near_match_ratio(cmd: str, source: str) -> float:
    """Best similarity between `cmd` and any aligned window of `source`.

    Both arguments are already `_normalize_command`-ed. Returns 0.0 when
    no token of the command long enough to anchor on occurs in the source.
    """
    if not cmd or not source:
        return 0.0
    best = 0.0
    tokens = sorted(
        {t for t in cmd.split() if len(t) >= _NEAR_MATCH_MIN_TOKEN},
        key=len, reverse=True,
    )
    anchors = 0
    for tok in tokens:
        offset = cmd.index(tok)
        start = 0
        while anchors < _NEAR_MATCH_MAX_ANCHORS:
            i = source.find(tok, start)
            if i < 0:
                break
            anchors += 1
            w_start = max(0, i - offset)
            window = source[w_start: w_start + len(cmd)]
            ratio = SequenceMatcher(None, cmd, window, autojunk=False).ratio()
            if ratio > best:
                best = ratio
                if best >= 0.999:
                    return best
            start = i + 1
    return best


def _captured_command_lines(validated_entities: list[dict]) -> list[str]:
    """The Gate 0 `ioc_command_line` entities the analyst kept, verbatim."""
    out: list[str] = []
    seen: set[str] = set()
    for ent in validated_entities:
        if ent.get("entity_type") != "ioc_command_line":
            continue
        if ent.get("gate_action") == "remove":
            continue
        value = (ent.get("edited_value") or ent.get("value") or "").strip()
        key = _normalize_command(value)
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _chunk_source_window(chunk: dict, parsed_text: str) -> str:
    """The slice of parsed_text around the chunk's source_span, or ''."""
    span = chunk.get("source_span")
    if not parsed_text or not span or len(span) != 2:
        return ""
    try:
        start, end = int(span[0]), int(span[1])
    except (TypeError, ValueError):
        return ""
    if start < 0 or end <= start:
        return ""
    return parsed_text[max(0, start - _SPAN_WINDOW_BEFORE): end + _SPAN_WINDOW_AFTER]


def _commands_for_chunk(chunk: dict, captured: list[str], parsed_text: str) -> list[str]:
    """Captured command lines that appear in this chunk's text, excerpt, or
    source window. Unmatched commands stay visible to the model in the
    global VALIDATED ENTITIES block, so an attribution the window misses
    can still be made — it just isn't suggested per chunk."""
    if not captured:
        return []
    haystack = _normalize_command(" ".join([
        chunk.get("text") or "",
        chunk.get("source_excerpt") or "",
        _chunk_source_window(chunk, parsed_text),
    ]))
    return [c for c in captured if _normalize_command(c) in haystack]


def _ground_command_lines(
    command_lines: list,
    chunk: dict,
    parsed_text: str,
    captured: list[str],
    draft_name: str,
) -> list[str]:
    """Keep only command lines that exist in the source.

    The domain rule is "no fabricated command lines": a thin source gets low
    confidence, never an invented command. The prompt says so, but a prompt
    is advice. This is the check. A command survives when its normalized
    form is a substring of the parsed source text (or of the chunk's own
    text / excerpt, for replayed states that carry no parsed_text), equals
    a captured `ioc_command_line` entity, or near-matches an aligned window
    of the source above `_NEAR_MATCH_THRESHOLD` (OCR noise in a figure
    transcription, logged at INFO so the tolerance is visible). Everything
    else is dropped with a warning naming the procedure, so a drop is
    visible in the run log rather than silent.
    """
    source = _normalize_command(" ".join([
        parsed_text or "",
        chunk.get("text") or "",
        chunk.get("source_excerpt") or "",
    ]))
    captured_keys = {_normalize_command(c) for c in captured}

    kept: list[str] = []
    seen: set[str] = set()
    for cmd in command_lines or []:
        if not isinstance(cmd, str) or not cmd.strip():
            continue
        key = _normalize_command(cmd)
        if key in seen:
            continue
        if key in captured_keys or key in source:
            seen.add(key)
            kept.append(cmd.strip())
            continue
        ratio = _near_match_ratio(key, source)
        if ratio >= _NEAR_MATCH_THRESHOLD:
            seen.add(key)
            kept.append(cmd.strip())
            logger.info(
                "draft_procedures: kept command line by near match "
                "(ratio=%.3f) for procedure %r: %r",
                ratio, draft_name, cmd,
            )
        else:
            logger.warning(
                "draft_procedures: dropping command line not found in the "
                "source from procedure %r: %r",
                draft_name, cmd,
            )
    return kept


def _format_entity_context(entities: list[dict]) -> str:
    """Summarize validated entities for procedure drafting context."""
    by_type: dict[str, list[str]] = {}
    for e in entities:
        if e.get("gate_action") == "remove":
            continue
        etype = e.get("entity_type", "")
        value = e.get("edited_value") or e.get("value", "")
        if value:
            by_type.setdefault(etype, []).append(value)

    if not by_type:
        return ""

    parts = ["\n\nVALIDATED ENTITIES (use canonical names from this list):"]
    for etype, values in sorted(by_type.items()):
        parts.append(f"  {etype}: {', '.join(values)}")
    return "\n".join(parts)


# =============================================================================
# Post-processing
# =============================================================================

def _filter_to_evidenced(
    names: list,
    chunk: dict,
    raw_draft: dict,
    kind: str,
) -> list[str]:
    """Keep only tool/malware names the chunk actually evidences.

    The prompt already asks for per-procedure attribution ("not every tool in
    the source"), but the LLM still attaches tooling the procedure never
    touches. In one audit `PowerShell` was attributed to two procedures on a
    report whose observed events contain no PowerShell command at all — it
    appears only in the report's title and summary — and `mshta` was
    attached to the download procedure when mshta is the executor in a
    different one. Unlike the relationship-preview fan-out, these are backed
    by real `tools_used` values, so they reach the bundle as `uses` SROs:
    false tooling claims about a threat actor.

    Same shape as `_ground_command_lines`, which does this for command
    lines: the name must appear in the chunk text, its source excerpt, or
    one of the procedure's command lines. Matching is substring-on-lowercase, which is
    deliberately permissive — the goal is to drop the clearly-unevidenced,
    not to adjudicate borderline cases.
    """
    haystack = " ".join([
        chunk.get("text") or "",
        chunk.get("source_excerpt") or "",
        " ".join(raw_draft.get("command_lines", []) or []),
        raw_draft.get("description") or "",
    ]).lower()

    kept: list[str] = []
    for name in names:
        if not isinstance(name, str) or not name.strip():
            continue
        cleaned = name.strip()
        if cleaned.lower() in haystack:
            kept.append(cleaned)
        else:
            logger.info(
                "draft_procedures: dropping unevidenced %s %r from procedure "
                "%r — not present in the chunk text, excerpt, or commands",
                kind, cleaned, raw_draft.get("name", "?"),
            )
    return kept


def _process_drafts(
    raw_drafts: list[dict],
    chunks: list[dict],
    technique_mappings: dict[str, list[dict]],
    metadata: dict,
    validated_entities: list[dict] | None = None,
    parsed_text: str = "",
) -> list[dict]:
    """Convert raw LLM output to ProcedureDraft-shaped dicts.

    Merges LLM-generated fields (name, description, platforms, command_lines)
    with pipeline-tracked fields (techniques, sequencing, temporal data).

    When `validated_entities` is supplied, also detects CVE references in
    the chunk + draft text and populates `vulnerability_refs` with the
    matching entity IDs. Without this, every procedure ships with empty
    x_vulnerability_refs even when it explicitly mentions a CVE.
    """
    # Build chunk lookup
    chunk_lookup = {c["chunk_id"]: c for c in chunks}
    valid_chunk_ids = set(chunk_lookup.keys())
    captured_commands = _captured_command_lines(validated_entities or [])

    # CVE detection lookup: build a {normalized_cve: entity_id} map
    # from validated_entities so each draft can match its mentions
    # against the bundle's Vulnerability SDOs. Normalized form upper-
    # cases the prefix and strips defangs.
    cve_to_entity_id: dict[str, str] = {}
    if validated_entities:
        for ent in validated_entities:
            etype = ent.get("entity_type", "")
            if etype != "vulnerability":
                continue
            value = (ent.get("edited_value") or ent.get("value") or "").strip()
            normalized = _normalize_cve(value)
            if normalized:
                cve_to_entity_id[normalized] = ent.get("entity_id", "")

    drafts: list[dict] = []

    for raw in raw_drafts:
        chunk_id = raw.get("chunk_id", "")
        if chunk_id not in valid_chunk_ids:
            logger.warning("draft_procedures: unknown chunk_id '%s', skipping", chunk_id)
            continue

        chunk = chunk_lookup[chunk_id]

        # Get technique mappings for this chunk
        techniques = technique_mappings.get(chunk_id, [])

        # Build kill chain phases from techniques
        kill_chain_phases = []
        seen_phases = set()
        for t in techniques:
            tactic = t.get("tactic", "")
            if tactic and tactic not in seen_phases:
                kill_chain_phases.append({
                    "kill_chain_name": "mitre-attack",
                    "phase_name": tactic,
                })
                seen_phases.add(tactic)

        draft = {
            "draft_id": f"dft-{uuid.uuid4().hex[:8]}",
            "chunk_id": chunk_id,

            # LLM-generated fields
            "name": raw.get("name", ""),
            "description": raw.get("description", ""),
            "platforms": _normalize_platforms(
                raw.get("platforms", []),
                chunk.get("text", ""),
            ),
            "detail_gap": raw.get("detail_gap", False),

            # Per-procedure entity attribution. Names from the chunk's
            # stated tools/malware; the
            # serializer resolves to STIX IDs via the entity registry. Drives
            # per-procedure procedure→tool / procedure→malware USES SROs in
            # place of the prior global N×M fan-out.
            "tools_used": _filter_to_evidenced(
                raw.get("tools_used", []), chunk, raw, "tool",
            ),
            "malware_used": _filter_to_evidenced(
                raw.get("malware_used", []), chunk, raw, "malware",
            ),
            # Per-procedure attribution. Unlike tools/malware this is NOT
            # evidence-filtered against the chunk text: reports routinely
            # refer to the actor as "the group" or "the actor" rather than by
            # name, so requiring the literal name in the chunk would discard
            # correct attributions. The serializer's guard handles the rest.
            "attributed_actors": [
                a.strip() for a in raw.get("attributed_actors", []) or []
                if isinstance(a, str) and a.strip()
            ],

            # Raw command lines quoted by the LLM, grounded against the
            # source before they are kept. These are staging data, NOT STIX
            # fields. The serializer converts them to Process SCOs and
            # populates components_refs with the resulting STIX IDs.
            "raw_command_lines": _ground_command_lines(
                raw.get("command_lines", []), chunk, parsed_text,
                captured_commands, raw.get("name", "?"),
            ),

            # v0.5.0-draft STIX reference fields.
            # These start empty and are populated during serialization
            # when raw data (command lines, observables) is converted
            # to proper SCOs with STIX IDs.
            # There is no command_ref: the x_command_ref STIX property it
            # fed was retired in v0.5.0-draft and must not come back. The
            # primary command is simply the first entry in components_refs.
            "components_refs": [],      # Ordered SCO IDs (component sequence)
            "log_source_refs": [],      # x-log-source IDs (detection mapping)

            # x_procedure_type. Hardcoding "reporting" would make a vendor's
            # "the actors could also have used X" serialize identically to
            # "the actors did" — on a schema whose whole point is the
            # evidentiary layer. The drafting LLM classifies it from the
            # chunk (see the tool schema); "reporting" stays the default for
            # cached outputs and for anything it does not mark.
            "procedure_type": raw.get("procedure_type", "reporting"),

            # From technique extraction (not LLM-generated here)
            "techniques": techniques,
            "kill_chain_phases": kill_chain_phases,

            # Confidence from chunk (0-100 integer)
            "confidence": int(chunk.get("behavioral_confidence", 0.5) * 100),

            # Temporal fields from metadata (if available)
            "first_observed": metadata.get("first_observed"),
            "last_observed": metadata.get("last_observed"),

            # Source references (populated at serialization with Identity IDs)
            "source_refs": [],
            # Vulnerability references — entity IDs of CVE entities the
            # serializer resolves to vulnerability--<UUID> STIX IDs at
            # bundle-assembly time. Populated below by scanning the chunk's
            # text + this draft's name/description for CVE patterns.
            "vulnerability_refs": _detect_vulnerability_refs(
                chunk, raw, cve_to_entity_id,
            ) if cve_to_entity_id else [],

            # ATT&CK Flow sequencing (inherited from chunk).
            # The normalize node inverts these into draft-level effect_refs
            # (forward edges); the serializer turns those into PRECEDES SROs
            # and the attack-flow object's start_refs.
            "sequence_index": chunk.get("sequence_index", 0),
            "predecessor_indices": chunk.get("predecessor_indices", []),
            "effect_refs": [],          # Populated by normalize node

            # Quality metadata (internal, not serialized to STIX)
            "source_location": chunk.get("source_location", {}),
            # Propagate the chunk's source-fidelity category so the
            # serializer can emit x_source_provenance on the procedure.
            # See state.Chunk.source_provenance for taxonomy.
            "source_provenance": chunk.get("source_provenance", "paraphrased"),
            # Chain-separation passthrough. Serializer rolls chain_root
            # drafts into attack-flow.start_refs and embeds chain_label
            # on the procedure as x_chain_label.
            "chain_root": bool(chunk.get("chain_root", False)),
            "chain_label": chunk.get("chain_label", "") or "",

            # Gate 1 fields (set by analyst later)
            "gate_action": None,
            "reject_reason": None,
            "analyst_edits": None,
            "analyst_rationale": None,
        }

        drafts.append(draft)

    return drafts


# =============================================================================
# Platform normalization
# =============================================================================

# Valid OpenTide Threat Surface top-level platforms (OS stage).
# Sub-types are validated by prefix matching against these roots.
_VALID_PLATFORM_ROOTS = {
    "Windows", "Linux", "macOS", "Mobile", "Container Runtime",
    "BSD", "Solaris", "ChromeOS", "Embedded", "Miscellaneous",
}

# Common LLM mistakes -> correct OpenTide values.
# Keys are lowercase for case-insensitive matching.
_PLATFORM_ALIASES = {
    # Case fixes
    "windows": "Windows",
    "linux": "Linux",
    "macos": "macOS",
    "mobile": "Mobile",
    "android": "Mobile::Android",
    "ios": "Mobile::iOS",
    "docker": "Container Runtime::Docker",
    "containers": "Container Runtime",
    "container runtime": "Container Runtime",
    # Invalid sub-types the LLM invents
    "linux::server": "Linux",
    "linux::workstation": "Linux",
    "windows::server": "Windows::Server",
    "windows::desktop": "Windows::Desktop",
    "windows::workstation": "Windows::Desktop",
    # Cloud (not an OS-stage entry but commonly assigned)
    "cloud": "Cloud",
    "cloud::aws": "Cloud::AWS",
    "cloud::azure": "Cloud::Azure",
    "cloud::gcp": "Cloud::Google Cloud",
    # Network (not in OpenTide, map to Infrastructure)
    "network": "Infrastructure",
}


def _normalize_platforms(platforms: list[str], chunk_text: str) -> list[str]:
    """Validate and normalize LLM-generated platforms against OpenTide vocab.

    Rules:
    1. Fix casing (lowercase -> title case via alias map)
    2. Correct invalid sub-types (linux::server -> Linux)
    3. Deduplicate
    4. If a sub-type is present alongside its parent, keep only the sub-type
       (e.g., [Linux, Linux::Ubuntu] -> [Linux::Ubuntu])
    """
    normalized = []
    seen = set()

    for raw in platforms:
        if not isinstance(raw, str) or not raw.strip():
            continue
        # Try alias lookup (case-insensitive)
        platform = _PLATFORM_ALIASES.get(raw.lower().strip(), raw.strip())

        # Validate: must start with a known root
        root = platform.split("::")[0]
        if root not in _VALID_PLATFORM_ROOTS and root not in ("Cloud", "Infrastructure"):
            logger.warning("draft_procedures: unknown platform '%s' (from LLM '%s'), dropping", platform, raw)
            continue

        if platform not in seen:
            seen.add(platform)
            normalized.append(platform)

    # Deduplicate: if both parent and child present, keep child only
    # e.g., [Linux, Linux::Ubuntu] -> [Linux::Ubuntu]
    final = []
    for p in normalized:
        # Check if any other entry is a more specific child of this one
        has_child = any(
            other != p and other.startswith(p + "::")
            for other in normalized
        )
        if not has_child:
            final.append(p)

    return final


# =============================================================================
# Technique override re-application
# =============================================================================

def _chunk_text_hash(text: str) -> str:
    """Same hash function as gates.py for matching chunk text."""
    return hashlib.sha256(text.strip().lower().encode("utf-8")).hexdigest()[:20]


def _apply_technique_overrides(
    drafts: list[dict],
    chunks: list[dict],
    overrides: list[dict],
) -> int:
    """Re-apply analyst technique corrections from a previous Gate 1 review.

    After re-chunking, chunk_ids and draft_ids change but the text content
    may be similar. We match by chunk text hash (exact match on normalized
    text) and apply the saved technique list to the new draft.

    Returns the number of overrides successfully applied.
    """
    if not overrides:
        return 0

    # Build override lookup by hash
    override_map = {o["chunk_text_hash"]: o for o in overrides}

    # Build chunk lookup
    chunk_lookup = {c["chunk_id"]: c for c in chunks}

    applied = 0
    for draft in drafts:
        chunk_id = draft.get("chunk_id", "")
        chunk = chunk_lookup.get(chunk_id, {})
        chunk_text = chunk.get("text", "")
        if not chunk_text:
            continue

        text_hash = _chunk_text_hash(chunk_text)
        override = override_map.get(text_hash)
        if not override:
            continue

        # Apply the analyst's technique list
        techniques = override["techniques"]
        draft["techniques"] = techniques

        # Rebuild kill_chain_phases from overridden techniques
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

        # Mark that this draft carries analyst-reviewed techniques
        draft["analyst_rationale"] = (
            "Techniques re-applied from previous review"
            + (f": {override.get('rationale', '')}" if override.get("rationale") else "")
        )

        applied += 1

    return applied


# =============================================================================
# CVE detection
# =============================================================================
#
# Without this pass, drafts ship with vulnerability_refs=[] even when the
# chunk text and the draft's own name/description explicitly mention a CVE
# that exists as a Vulnerability SDO in the bundle. The serializer then
# leaves x_vulnerability_refs absent on every procedure and the
# Explorer/Flow view shows "Vulnerabilities: —" for procedures whose
# entire raison d'être is exploiting a specific CVE (e.g. a ransomware family's
# CVE-2025-53770 SharePoint exploit chain).

import re

# CVE pattern: CVE-YYYY-N{4,7}. Case-insensitive; stripped of common
# defangs (CVE[-]2023-... / CVE-2023[.]27532) before matching.
_CVE_RE = re.compile(r"\bCVE[\s\-_]*(\d{4})[\s\-_]*(\d{4,7})\b", re.IGNORECASE)


def _normalize_cve(value: str) -> str:
    """Canonicalize a CVE-ish string to 'CVE-YYYY-NNNN' form.

    Handles defanged variants (CVE[-]2023-46604, CVE 2023 46604, etc.).
    Returns the empty string when no CVE pattern is present.
    """
    if not value:
        return ""
    m = _CVE_RE.search(value)
    if not m:
        return ""
    return f"CVE-{m.group(1)}-{m.group(2)}"


def _detect_vulnerability_refs(
    chunk: dict, raw_draft: dict, cve_to_entity_id: dict[str, str],
) -> list[str]:
    """Return entity IDs of every Vulnerability whose CVE appears in the
    chunk's text/source_excerpt or the draft's name/description.

    The serializer then resolves these entity IDs to vulnerability--<UUID>
    STIX IDs via id_registry at bundle-assembly time. Order is preserved
    (first-mention wins) and duplicates are deduped.
    """
    if not cve_to_entity_id:
        return []
    haystacks: list[str] = []
    if chunk:
        if chunk.get("text"):
            haystacks.append(chunk["text"])
        if chunk.get("source_excerpt"):
            haystacks.append(chunk["source_excerpt"])
    if raw_draft:
        if raw_draft.get("name"):
            haystacks.append(raw_draft["name"])
        if raw_draft.get("description"):
            haystacks.append(raw_draft["description"])
    if not haystacks:
        return []

    seen: set[str] = set()
    refs: list[str] = []
    for blob in haystacks:
        for match in _CVE_RE.finditer(blob):
            normalized = f"CVE-{match.group(1)}-{match.group(2)}"
            entity_id = cve_to_entity_id.get(normalized)
            if entity_id and entity_id not in seen:
                seen.add(entity_id)
                refs.append(entity_id)
    return refs
