"""Pydantic models for validating Claude tool_use output.

WHY THIS EXISTS:
Claude occasionally returns tool_use input that doesn't match the JSON
schema we gave it. Common failure modes:
  - bare string in a list of objects (the bug that inspired this module)
  - missing a required field
  - wrong type for a scalar (string where int was asked)
  - extra fields the schema didn't declare

Without validation, every node's post-processor calls raw.get(...) and
hopes for the best. Instead the adapter validates tool_output against
these models and either returns a validated object or raises
LLMValidationError, which the pipeline turns into a failed run.

MODEL NAMING:
Each tool has a top-level Output model matching the tool's input_schema.
Nested item models are suffixed Item (EntityItem, ChunkItem, etc.)
so the public type surface is obvious when imported.

EXTRA='FORBID' BY DEFAULT:
These models reject unknown fields. That's stricter than the JSON schema
(which is open by default), but it's exactly the behavior we want:
if Claude invents a field we didn't ask for, something's weird and we'd
rather know. If you need to allow an experimental field, flip extra
to 'ignore' on that specific model.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.graph.state import (
    EntityType,
    SectionClassification,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Shared base: strict extra='forbid' for all tool models
# =============================================================================


class _StrictBase(BaseModel):
    """Base for every tool_use model. Reject unknown fields, no arbitrary
    coercion of scalars from collections."""

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# extract_entities (entity_extraction.py)
# =============================================================================


class EntityItem(_StrictBase):
    value: str
    entity_type: EntityType
    confidence: float = Field(ge=0.0, le=1.0)
    context_snippet: str | None = None
    organization_role: str | None = None
    location_role: str | None = None
    # Defaulted so cached responses predating the field still validate.
    attributed_to: list[str] = Field(default_factory=list)


class DetectionRuleItem(_StrictBase):
    # `str`, not `DetectionRuleType`, and that is deliberate. As an enum this
    # field was FATAL to the whole tool call: one rule tagged `ioc_command_line`
    # -- an ENTITY type the model confused for a rule type -- failed validation
    # for the entire `extract_entities` output, losing every entity in a
    # hundred-entity report to one bad line.
    #
    # `_process_detection_rules` in entity_extraction.py already does the right
    # thing with an unrecognised value: log it and skip that rule. Two layers
    # disagreed about how much one junk rule should cost, and the fatal one won.
    # Same principle as ChunkContext below: one odd key must not cost a whole
    # pass.
    #
    # The general fix for this class now lives in the adapter: a malformed
    # LIST ITEM is dropped on its own (see _salvage_list_items in
    # llm_adapter.py), so one bad rule or entity costs that row, not the
    # pass. This field stays `str` regardless -- the enum bought nothing
    # that _process_detection_rules does not already do.
    rule_type: str
    rule_content: str
    description: str | None = None


class ExtractEntitiesOutput(_StrictBase):
    """Output of the extract_entities tool."""

    entities: list[EntityItem]
    detection_rules: list[DetectionRuleItem]
    # Defaults so older cached LLM responses (predating the SOURCE STRUCTURE
    # block) keep validating. New runs always populate these per the
    # tool_schema 'required' list.
    is_sequential: bool = True
    sequentiality_rationale: str = ""


# =============================================================================
# extract_figure (figure_extraction.py) — single-figure vision call
# =============================================================================


class ExtractFigureOutput(_StrictBase):
    """Output of the per-figure vision extraction tool."""

    figure_type: Literal["diagram", "screenshot", "decorative", "other"]
    extracted_text: str = ""
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    rationale: str = ""


# =============================================================================
# synthesize_feedback (feedback_synthesis.py)
# =============================================================================


# Canonical taxonomy of feedback pattern categories. Stored in postgres as
# String (not enum) so the taxonomy can grow without schema migrations, but
# typed here as a Literal so the LLM emits known values.
FeedbackCategory = Literal[
    "defender_ioc",              # IoCs that are actually defender/CERT/sandbox infrastructure
    "brand_as_malware",          # Technique-pattern brand names (ClickFix, MFA fatigue) miscoded as malware/tool
    "false_positive_entity",     # Other entity-level false positives at Gate 0
    "over_chunked",              # Chunker emitted N chunks where 1 was right (analyst merged)
    "under_chunked",             # Chunker emitted 1 chunk where N were right (analyst split)
    "missing_tactic",            # Source had a tactic the chunker didn't represent
    "missing_procedure",         # Specific procedure absent from chunks (analyst added)
    "wrong_predecessor",         # Sequencing edges the analyst corrected
    "parallel_capability_misordered",  # Forced-into-sequence what should be parallel
    "mis_attribution",           # Cluster vs campaign vs intrusion-set confusion
    "wrong_technique",           # Technique pick at Gate 1 the analyst rejected
    "thin_initial_access",       # IA chunk with no exploit/CVE/product entity link
    "orphan_ioc",                # IoC table entries never linked to consuming procedure
    "artifact_loss",             # Chunk paraphrased away verbatim artifacts
    "wrong_relationship",        # Relationship at Gate 2 the analyst removed
    "other",
]


class FeedbackPatternItem(_StrictBase):
    """One pattern emitted by the synthesizer."""

    category: FeedbackCategory
    pattern: str = Field(
        min_length=10,
        max_length=500,
        description="Short, searchable description of the pattern.",
    )
    evidence: dict = Field(
        default_factory=dict,
        description="Per-category structured evidence supporting the pattern.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Synthesizer's confidence the pattern is real / generalizable. "
            "Low confidence (<0.5) = likely a one-off noise; consumers can "
            "filter on this."
        ),
    )
    # Structured retrieval keys (relevance-first flywheel). Default-empty so
    # cached/legacy tool outputs that predate these fields still validate.
    applies_to: dict = Field(
        default_factory=dict,
        description=(
            "Structured retrieval keys: technique_ids, entity_types, tactics, "
            "source_genre. Used to surface this pattern only for relevant sources."
        ),
    )
    concepts: list[str] = Field(
        default_factory=list,
        description="Short concept tags for search/display.",
    )


class SynthesizeFeedbackOutput(_StrictBase):
    """Output of the synthesize_feedback tool. May be an empty list when
    the analyst had no notable disagreements with the LLM output."""

    patterns: list[FeedbackPatternItem]


class CorrectionAttributionItem(_StrictBase):
    """Which surfaced rules, if any, one analyst correction is an instance of."""

    correction: int = Field(
        description="0-based index of the correction, as numbered in the prompt.",
    )
    rules: list[int] = Field(
        default_factory=list,
        description=(
            "0-based indices of the rules this correction is an instance of. "
            "Usually empty — most corrections are unrelated to any surfaced rule."
        ),
    )
    reasoning: str = Field(
        default="",
        description="One sentence. Why those rules, or why none.",
    )


class AttributeCorrectionsOutput(_StrictBase):
    """Output of the attribute_corrections tool: the miss ledger's attribution
    step. Empty `attributions` is legal and means nothing was attributable."""

    attributions: list[CorrectionAttributionItem] = Field(default_factory=list)


# =============================================================================
# classify_sections (chunking.py)
# =============================================================================


class SectionItem(_StrictBase):
    """One classified section, addressed by line range rather than by text.

    The classifier used to echo each section's full text back, which made its
    output scale with input length and blew the max_tokens ceiling on long
    sources (a 97k-char source needs ~25k output tokens just to repeat
    itself). It now returns 1-based inclusive line ranges into the numbered
    text it was shown, and chunking.py slices the original lines. Output is
    a fixed ~20 tokens per section regardless of source length.
    """

    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    classification: SectionClassification
    classification_confidence: float = Field(ge=0.0, le=1.0)


class ClassifySectionsOutput(_StrictBase):
    sections: list[SectionItem]


# =============================================================================
# chunk_behaviors (chunking.py)
# =============================================================================


class ChunkContext(BaseModel):
    """Entity references attached to a chunk. All fields optional because
    the schema marks none of them as required; Claude may emit a bare
    empty object.

    NOT strict, unlike its siblings, and the sibling right below it says why:
    `artifacts` is "free-form ... new categories are tolerated". `context` is
    the same kind of thing — descriptive metadata that colors a chunk — and
    it was the only one that could kill a run over it.

    On a ransomware run, one chunk of sixteen carried `target_software`
    instead of `target`. Strict validation rejected the whole batch, the
    retry came back with a malformed payload, and the run failed at chunking
    having produced sixteen perfectly good chunks. An unknown descriptive key
    is worth dropping, never worth a run.

    Dropped keys are logged rather than ignored: unrecognised keys are how
    you find out the prompt and the model have drifted apart, and silence is
    how that goes unnoticed for months.
    """

    model_config = ConfigDict(extra="ignore")

    @model_validator(mode="before")
    @classmethod
    def _log_unknown_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            unknown = set(data) - set(cls.model_fields)
            if unknown:
                logger.info(
                    "ChunkContext: dropping unrecognised key(s) %s — tolerated, "
                    "but the chunker prompt and this model may have drifted",
                    sorted(unknown),
                )
        return data

    actor: str | None = None
    malware: list[str] | None = None
    tools: list[str] | None = None
    target: str | None = None
    # Per-chunk ATT&CK tactic shortnames the chunk's objective spans.
    # Empty for cached LLM outputs predating the field; new runs always
    # populate per the chunker prompt (see chunking.py CHUNK_SYSTEM_PROMPT).
    tactics: list[str] = Field(default_factory=list)


class PreconditionItem(_StrictBase):
    """Attack-Flow attack-condition signal emitted by the chunker.

    Every field except the container itself is optional HERE even though the
    tool schema marks `description` required, and `pattern_type` is a bare str
    rather than a Literal. That is deliberate: `_finalize_chunks` already
    normalizes this structure — it drops preconditions with a blank
    description, coerces an out-of-enum `pattern_type` to "plain", prunes
    non-successor branch targets, and drops the whole precondition when
    neither branch resolves. Validating more strictly than the consumer would
    turn a recoverable one-field normalization into a hard failure of the
    entire chunking run (a single malformed precondition on chunk 7 of 40
    would discard all 40 chunks).

    Unknown SUBFIELDS are still rejected via _StrictBase. That strictness is
    what surfaced this model's own absence — the chunker had been emitting
    `precondition` while ChunkItem forbade it, failing the run outright rather
    than silently discarding attack-condition data.
    """

    description: str = ""
    pattern: str | None = None
    pattern_type: str | None = None
    on_true_indices: list[int] = Field(default_factory=list)
    on_false_indices: list[int] = Field(default_factory=list)


class ChunkItem(_StrictBase):
    text: str
    # source_excerpt: verbatim 2-3 sentences from the source justifying the
    # chunk. Required by the prompt.
    # Default to empty string so older cached LLM outputs without this field
    # don't fail validation; new runs always populate it.
    source_excerpt: str = ""
    context: ChunkContext | None = None
    sequence_index: int = Field(ge=1)
    predecessor_indices: list[int] = Field(default_factory=list)
    branch_point: bool = False
    convergence_point: bool = False
    behavioral_confidence: float = Field(ge=0.0, le=1.0)
    # Per-chunk artifacts captured at chunk-time. Free-form dict keyed by
    # category — the chunker prompt
    # enumerates the recognized categories; new categories are tolerated.
    # Default empty so cached LLM outputs predating this field still validate.
    artifacts: dict[str, list[str]] = Field(default_factory=dict)
    # Chain-separation fields. Set by the chunker when a chunk begins a
    # NEW attack chain in a multi-intrusion source (see CHAIN ROOT RULE
    # in CHUNK_SYSTEM_PROMPT). Defaults keep cached outputs valid.
    chain_root: bool = False
    chain_label: str = ""
    # Optional Attack-Flow runtime check gating downstream flow. The chunker
    # prompt asks for this whenever the source describes a conditional ("if
    # domain-joined, kerberoasting; else NTLM relay"); state.Chunk carries it
    # and _finalize_chunks resolves its indices to chunk_ids. Default None =
    # the common case (no conditional structure in the source).
    precondition: PreconditionItem | None = None


class ChunkBehaviorsOutput(_StrictBase):
    chunks: list[ChunkItem]


# =============================================================================
# extract_techniques (technique_extraction.py)
# =============================================================================


class TechniqueItem(_StrictBase):
    technique_id: str
    technique_name: str
    tactic: str
    confidence: float = Field(ge=0.0, le=1.0)
    # C+A+D pick-step output: bucket is the LLM's calibrated judgment;
    # `confidence` (numeric) is its continuous score. Bucket drives bundle
    # inclusion (definite/probable land in the bundle, possible is review-only
    # at Gate 1). The two should correlate but aren't auto-derived from each
    # other — recalibration adjusts numeric confidence, bucket stays as
    # emitted by the LLM.
    confidence_bucket: Literal["definite", "probable", "possible"] = "probable"
    # Verbatim 5-30 word substring of the chunk text justifying this pick.
    # Empty string allowed when the LLM cannot find a verbatim quote (the
    # caller auto-caps such picks at the 'possible' bucket).
    source_quote: str = ""
    rationale: str


class ProposeTechniquesItem(_StrictBase):
    """Per-chunk output of the C+A+D 'reason + propose' LLM call.

    The LLM describes the chunk's behavior in ATT&CK-flavored prose,
    states the procedure's adversarial objective (transient — folded
    into the description prose at drafting time, NOT a STIX field),
    enumerates ALL ATT&CK tactics the procedure spans, and proposes
    raw technique IDs from training memory. The proposed_techniques
    list is then validated against the v19 catalogue (drops
    hallucinations, redirects revoked), unioned with the retriever-
    supplied candidates, and fed to the pick step.

    `tactics` is a list — the procedure definition explicitly notes
    that procedures often span multiple tactics (e.g., LOLBin download
    spans command-and-control + defense-evasion).

    `objective` is the discriminator the pick step uses to decide
    whether a candidate technique BELONGS (serves the objective) or
    DRIFTS (serves a different objective). It also anchors the drafted
    description's lead sentence.
    """

    chunk_id: str
    behavior_description: str
    objective: str
    tactics: list[str] = Field(default_factory=list)
    proposed_techniques: list[str] = Field(default_factory=list)


class ProposeTechniquesOutput(_StrictBase):
    chunk_proposals: list[ProposeTechniquesItem]


class VerbatimMatchDecision(_StrictBase):
    """LLM's confirm-or-reject decision on a verbatim match surfaced by
    app.services.procedure_matcher.

    The matcher detects substrings in chunk text that exactly match
    operative strings from MITRE-curated procedure examples. Each
    detected match is presented to the LLM as a candidate; the LLM
    confirms (chunk's behavior aligns with the technique) or rejects
    (matched substring appears in a different context that doesn't
    actually implement the technique). Confirmed matches lock at
    confidence 0.95; rejected matches drop to 0.3 with the reason
    in the audit trail.
    """

    technique_id: str
    decision: Literal["confirm", "reject"]
    rejection_reason: str | None = None  # required by prompt convention when decision == "reject"


class ChunkTechniqueMapping(_StrictBase):
    chunk_id: str
    techniques: list[TechniqueItem]
    verbatim_match_decisions: list[VerbatimMatchDecision] = Field(default_factory=list)


class ExtractTechniquesOutput(_StrictBase):
    chunk_techniques: list[ChunkTechniqueMapping]


# =============================================================================
# draft_procedures (drafting.py)
# =============================================================================


class DraftItem(_StrictBase):
    chunk_id: str
    name: str
    description: str
    platforms: list[str]
    command_lines: list[str]
    detail_gap: bool
    # Per-procedure entity attribution. Drafting LLM populates with entity
    # NAMES drawn from the entity context for tools/malware THIS procedure
    # actually uses. Default empty so cached LLM outputs predating these
    # fields still validate; new runs populate them and the serializer uses
    # them to filter procedure→tool / procedure→malware USES SROs (rather
    # than a global N×M fan-out).
    tools_used: list[str] = Field(default_factory=list)
    malware_used: list[str] = Field(default_factory=list)
    # Default-empty for the same reason as tools_used/malware_used: cached
    # LLM outputs predating the field must still validate.
    attributed_actors: list[str] = Field(default_factory=list)
    # x_procedure_type. Defaults to "reporting" so cached outputs predating the
    # field still validate — and because that is the right answer for the
    # overwhelming majority of drafts. "observed" is deliberately NOT offered:
    # it means internal case/IR work, and this pipeline only ingests external
    # vendor reporting, so exposing it would only invite a wrong pick.
    procedure_type: Literal["reporting", "hypothetical"] = "reporting"


class DraftProceduresOutput(_StrictBase):
    drafts: list[DraftItem]


# =============================================================================
# Tool-name registry (convenience for tests and observability)
# =============================================================================

TOOL_OUTPUT_MODELS: dict[str, type[_StrictBase]] = {
    "extract_entities": ExtractEntitiesOutput,
    "classify_sections": ClassifySectionsOutput,
    "chunk_behaviors": ChunkBehaviorsOutput,
    "extract_techniques": ExtractTechniquesOutput,
    "draft_procedures": DraftProceduresOutput,
}
