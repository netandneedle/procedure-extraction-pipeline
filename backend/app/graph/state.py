"""LangGraph state schema for the extraction pipeline.

This is THE foundational file. Every node reads from and writes to this state.
The state flows through the entire pipeline graph, carrying ingestion records,
chunks, drafts, entity data, and gate decisions.

HOW LANGGRAPH STATE WORKS:
- The PipelineState TypedDict defines every field available to every node.
- When a node runs, it receives the full state and returns a partial dict
  of only the fields it wants to update.
- LangGraph merges those updates into the state (overwrites by default).
- The checkpointer persists the full state to PostgreSQL at every step,
  enabling pause/resume at human gates.

Example:
    def my_node(state: PipelineState) -> dict:
        text = state["parsed_text"]       # Read any field
        result = do_work(text)
        return {"my_output": result}      # Update only what changed
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, TypedDict


# =============================================================================
# Per-gate enable/disable
# =============================================================================
#
# Each key maps to one gate node. Names are semantic, not positional —
# they survive gate renumbering, and a new gate adds a key without
# breaking existing data.

GATE_KEYS: tuple[str, ...] = ("entities", "chunks", "procedures", "bundle")
DEFAULT_GATES: dict[str, bool] = {k: True for k in GATE_KEYS}
# 'entities' gates gate_0, 'chunks' gates gate_chunks (the chunk-validation
# gate between chunk_behaviors and extract_techniques), 'procedures' gates
# gate_1 (the post-draft technique review), and 'bundle' gates gate_2.


def normalize_gates(value: object) -> dict[str, bool]:
    """Coerce input into a {gate_key: bool} dict.

    Accepts:
        - None  : returns DEFAULT_GATES (all enabled).
        - bool  : legacy form. Expands to all-true or all-false.
        - dict  : keys must be a subset of GATE_KEYS. Missing keys default True.

    Raises ValueError on unknown keys or non-bool values.
    """
    if value is None:
        return dict(DEFAULT_GATES)
    if isinstance(value, bool):
        return {k: value for k in GATE_KEYS}
    if isinstance(value, Mapping):
        unknown = set(value) - set(GATE_KEYS)
        if unknown:
            raise ValueError(f"unknown gate keys: {sorted(unknown)}")
        result = dict(DEFAULT_GATES)
        for k, v in value.items():
            if not isinstance(v, bool):
                raise ValueError(
                    f"gate '{k}' must be bool, got {type(v).__name__}"
                )
            result[k] = v
        return result
    raise ValueError(
        f"gates_enabled must be bool or dict, got {type(value).__name__}"
    )


def is_gate_enabled(state: Mapping, key: str) -> bool:
    """Return True if the named gate should pause for analyst review.

    Reads ``state["gates_enabled"]``. Tolerant of legacy bool checkpoints
    that predate the dict migration: a bare bool is treated as a uniform
    setting across all gates.
    """
    raw = state.get("gates_enabled", True)
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, Mapping):
        return bool(raw.get(key, True))
    return True


# How a gate that IS enabled gets reviewed.
#
#   review  — a human reviews it. Today's behaviour, and the default.
#   assist  — the AI reviewer recommends; a human still decides and submits.
#   auto    — the AI reviewer decides and the pipeline advances unattended.
#
# This is deliberately a SEPARATE field from `gates_enabled` rather than a
# widening of it. Two reasons, both load-bearing:
#   1. `normalize_gates` raises on non-bool values, so string modes cannot
#      ride in that dict without loosening a validator that exists to catch
#      exactly this class of mistake.
#   2. Every mode string is TRUTHY in Python. Any code path that read
#      `gates_enabled[key]` directly instead of going through
#      `is_gate_enabled` would silently treat a disabled gate as enabled —
#      a safety gate switched off by a refactor, with no error.
#
# Keeping them orthogonal also means mode only has meaning when the gate is
# on, which is the actual semantics: a disabled gate has nobody to review it,
# human or otherwise.
GATE_MODES: tuple[str, ...] = ("review", "assist", "auto")
DEFAULT_GATE_MODE = "review"
DEFAULT_GATE_MODES: dict[str, str] = {k: DEFAULT_GATE_MODE for k in GATE_KEYS}


UNTITLED_SOURCE_SENTINEL = "Untitled Source"


def resolve_display_title(state: Mapping, default: str) -> str:
    """The title a run's outputs should carry, or `default`.

    `state["title"]` is Source.title, seeded by start_pipeline. It is a
    separate key from `state["metadata"]` (author, publication_date, ...),
    and no code writes a `title` into metadata — so anything that reads only
    `metadata.get("title")` sees nothing on every real run. That is how the
    Report SDO, the attack-flow object and the Neo4j Report node all came
    out named "Untitled CTI Report" while the bundle row next to them had
    the right title. One ladder for all of them:

        state.title  (unless it is the "Untitled Source" placeholder the
                      API fills in when the analyst set none)
        metadata.title  (legacy; still honoured for hand-built states)
        default
    """
    state_title = (state.get("title") or "").strip()
    if state_title and state_title != UNTITLED_SOURCE_SENTINEL:
        return state_title
    metadata = state.get("metadata") or {}
    return (metadata.get("title") or "").strip() or default


def normalize_gate_modes(value: object) -> dict[str, str]:
    """Coerce input into a {gate_key: mode} dict.

    Accepts:
        - None : returns DEFAULT_GATE_MODES (every gate human-reviewed).
        - str  : uniform mode across all gates.
        - dict : keys must be a subset of GATE_KEYS. Missing keys default
                 to "review".

    Raises ValueError on unknown keys or unknown mode values. Mirrors
    `normalize_gates` so the two read the same at every call site.
    """
    if value is None:
        return dict(DEFAULT_GATE_MODES)
    if isinstance(value, str):
        if value not in GATE_MODES:
            raise ValueError(
                f"unknown gate mode: {value!r}. Must be one of {GATE_MODES}"
            )
        return {k: value for k in GATE_KEYS}
    if isinstance(value, Mapping):
        unknown = set(value) - set(GATE_KEYS)
        if unknown:
            raise ValueError(f"unknown gate keys: {sorted(unknown)}")
        result = dict(DEFAULT_GATE_MODES)
        for k, v in value.items():
            if not isinstance(v, str) or v not in GATE_MODES:
                raise ValueError(
                    f"gate '{k}' mode must be one of {GATE_MODES}, got {v!r}"
                )
            result[k] = v
        return result
    raise ValueError(
        f"gate_modes must be str or dict, got {type(value).__name__}"
    )


def gate_mode(state: Mapping, key: str) -> str:
    """Return the review mode for the named gate.

    Reads ``state["gate_modes"]``. Absent or malformed -> "review", which is
    exactly the behaviour every checkpoint written before this field existed
    already has. Back-compat is by construction here, not by a shim: there is
    no legacy shape to tolerate because the field is new.

    Says nothing about whether the gate is enabled — ask `is_gate_enabled`
    for that. A disabled gate has no reviewer of any kind.
    """
    raw = state.get("gate_modes")
    if isinstance(raw, Mapping):
        mode = raw.get(key)
        if isinstance(mode, str) and mode in GATE_MODES:
            return mode
    return DEFAULT_GATE_MODE


# =============================================================================
# Enums
# =============================================================================

class SourceType(str, Enum):
    """Supported source types for ingestion."""
    PDF = "pdf"
    HTML = "html"
    DOCX = "docx"
    TWEET_URL = "tweet_url"       # Tweet URL; parser fetches structured content
    IMAGE = "image"               # Generic image (OCR fallback)
    MARKDOWN = "markdown"         # Markdown files (.md)
    FREE_TEXT = "free_text"       # Pasted content, emails, notes
    STIX_BUNDLE = "stix_bundle"   # Existing STIX for re-ingestion


class Channel(str, Enum):
    """How the source entered the pipeline."""
    MANUAL = "manual"
    AUTOMATED = "automated"


class PipelineStatus(str, Enum):
    """Pipeline progress status, also used for Kanban board columns.

    RESUMING_FROM_GATE_N is a transient status the gate node writes
    after processing analyst decisions. It exists so the Kanban card
    stays in the correct gate column while the pipeline restarts,
    without letting the gate node accidentally re-write "gate_N"
    (which would bounce the card back to the review state). The
    downstream node (chunk_behaviors, extract_techniques, normalize,
    serialize) overwrites it almost immediately with its own
    processing status.
    """
    QUEUED = "queued"
    PARSING = "parsing"
    EXTRACTING_FIGURES = "extracting_figures"
    CLASSIFYING_SECTIONS = "classifying_sections"
    EXTRACTING_ENTITIES = "extracting_entities"
    # The AI reviewer is reading the report and the extracted entities. Its own
    # status rather than a shared "ai_reviewing" because Kanban columns
    # enumerate statuses with no fallback — one status can only sit in one
    # column, and this card belongs in Entity Review. Every gate has its own
    # value for the same reason; a declared-but-never-emitted status is its own
    # defect, since a card with such a status would vanish from the board.
    REVIEWING_ENTITIES = "reviewing_entities"
    GATE_0 = "gate_0"
    RESUMING_FROM_GATE_0 = "resuming_from_gate_0"
    CHUNKING = "chunking"
    # See REVIEWING_ENTITIES for why each gate gets its own value.
    REVIEWING_CHUNKS = "reviewing_chunks"
    GATE_CHUNKS = "gate_chunks"
    RESUMING_FROM_GATE_CHUNKS = "resuming_from_gate_chunks"
    EXTRACTING_TECHNIQUES = "extracting_techniques"
    DRAFTING = "drafting"
    # See REVIEWING_ENTITIES for why each gate gets its own value.
    REVIEWING_PROCEDURES = "reviewing_procedures"
    GATE_1 = "gate_1"
    RESUMING_FROM_GATE_1 = "resuming_from_gate_1"
    NORMALIZING = "normalizing"
    # See REVIEWING_ENTITIES for why each gate gets its own value.
    REVIEWING_BUNDLE = "reviewing_bundle"
    GATE_2 = "gate_2"
    RESUMING_FROM_GATE_2 = "resuming_from_gate_2"
    SERIALIZING = "serializing"
    DISTRIBUTING = "distributing"
    SYNTHESIZING_FEEDBACK = "synthesizing_feedback"
    COMPLETED = "completed"
    FAILED = "failed"


class SectionClassification(str, Enum):
    """Content classification by function, not report structure.

    The classifier is a RECALL tool: when in doubt, label as
    BEHAVIORAL_NARRATIVE. The downstream chunker is the precision tool
    that isolates actual adversary actions from surrounding context.
    Better to over-send to the chunker than miss a procedure.
    """
    BEHAVIORAL_NARRATIVE = "behavioral_narrative"   # Describes adversary actions (-> chunking)
    INDICATOR_DATA = "indicator_data"               # IOCs, hashes, IPs, domains
    DETECTION_LOGIC = "detection_logic"             # Sigma, YARA, detection rules
    TECHNIQUE_REFERENCE = "technique_reference"     # ATT&CK mappings, technique tables
    CONTEXTUAL = "contextual"                       # Background, summaries, commentary, opinions
    METADATA = "metadata"                           # Dates, authors, TLP, headers, footers
    UNCLASSIFIED = "unclassified"                   # Doesn't fit above categories


class EntityType(str, Enum):
    """Types of entities extracted from source material.

    Naming rules enforced at extraction:
    - INTRUSION_SET: preserve source name verbatim (UNC4899, APT29, SCATTERED SPIDER).
      Never rename across vendor nomenclatures.
    - THREAT_ACTOR: the real-world actor/org behind intrusion sets (SVR, GRU).
    - MALWARE / TOOL: prefer ATT&CK canonical name, fallback to Malpedia.
    - VICTIM_SECTOR: use STIX industry-sector-ov vocabulary.
    - LOCATION: country/region, not stuffed into identity.
    """
    # SDO-producing types
    INTRUSION_SET = "intrusion_set"      # Activity cluster (APT29, UNC4899, Storm-0558)
    THREAT_ACTOR = "threat_actor"        # Real-world actor behind intrusion sets (SVR, GRU, IRGC)
    MALWARE = "malware"                  # Malware families and variants
    TOOL = "tool"                        # Offensive tools, LOLBins, dual-use utilities
    CAMPAIGN = "campaign"                # Named operations (Operation Aurora)
    VULNERABILITY = "vulnerability"      # CVE references (CVE-2023-46604)
    ORGANIZATION = "organization"        # Victim companies, government agencies, sponsors
    LOCATION = "location"                # Countries, regions, cities (replaces victim_geo)
    VICTIM_SECTOR = "victim_sector"      # Industry sector (STIX industry-sector-ov)
    INFRASTRUCTURE = "infrastructure"    # C2 servers, staging, redirectors

    # SCO-producing types (observables attached to procedures)
    IOC_HASH = "ioc_hash"
    IOC_IP = "ioc_ip"                    # IPv4 or IPv6; version detected at serialization
    IOC_DOMAIN = "ioc_domain"
    IOC_URL = "ioc_url"
    IOC_EMAIL = "ioc_email"
    IOC_FILE_PATH = "ioc_file_path"
    IOC_REGISTRY_KEY = "ioc_registry_key"
    IOC_MUTEX = "ioc_mutex"              # Malware mutexes and named pipes
    IOC_COMMAND_LINE = "ioc_command_line"  # Verbatim command strings (powershell -enc ..., wmic ...)
    IOC_PROCESS_NAME = "ioc_process_name"  # Concrete executable filenames (powershell.exe, evil.dll)
    SOFTWARE = "software"                # Targeted product/version (Apache ActiveMQ 5.15.0)
    USER_ACCOUNT = "user_account"        # Compromised accounts


class GateAction(str, Enum):
    """Analyst actions at validation gates."""
    APPROVE = "approve"
    REJECT = "reject"
    EDIT = "edit"
    REMOVE = "remove"


class Gate1RejectReason(str, Enum):
    """Structured rejection reasons specific to Gate 1."""
    WRONG_TECHNIQUE = "wrong_technique"
    HALLUCINATED = "hallucinated"
    TOO_VAGUE = "too_vague"
    DUPLICATE = "duplicate"
    NOT_A_PROCEDURE = "not_a_procedure"
    WRONG_TECHNIQUE_GRANULARITY = "wrong_technique_granularity"
    BAD_CHUNK_BOUNDARY = "bad_chunk_boundary"


class ChunkProblemType(str, Enum):
    """Structured problem types for BAD_CHUNK_BOUNDARY feedback."""
    OVERLAP = "overlap"              # Two chunks describe the same content
    SPLIT_NEEDED = "split_needed"    # One chunk covers multiple distinct behaviors
    MERGE_NEEDED = "merge_needed"    # Two chunks should be combined
    WRONG_BOUNDARY = "wrong_boundary"  # Boundary between chunks is in the wrong place


class ChunkGateRejectReason(str, Enum):
    """High-level rejection reasons at the chunk-review gate (Gate 1 in the
    user-facing numbering). Each value is a hint to chunk_behaviors on the
    rerun about WHY the previous output was wrong.

    Distinct from ChunkProblemType (which describes per-chunk-pair issues
    fed into the existing fine-grained chunk_feedback channel).
    """
    MISSED_PROCEDURES = "missed_procedures"      # Important behaviors not chunked
    OVER_CHUNKED = "over_chunked"                # Too many tiny chunks
    UNDER_CHUNKED = "under_chunked"              # Broad chunks that should be split
    BAD_BOUNDARIES = "bad_boundaries"            # Boundaries don't align with behavioral transitions
    BAD_DESCRIPTIONS = "bad_descriptions"        # Descriptions are wrong / hallucinated
    BAD_FLOW = "bad_flow"                        # Sequencing is wildly wrong
    OTHER = "other"                              # Free-form via comments only


class DetectionRuleType(str, Enum):
    """Types of detection rules the pipeline can preserve from sources.

    Maps to the pattern_type field in STIX Indicator SDOs.
    Only rules found verbatim in the source are preserved.
    """
    SIGMA = "sigma"                 # Sigma detection rules
    YARA = "yara"                   # YARA malware detection rules
    SNORT = "snort"                 # Snort IDS rules
    SURICATA = "suricata"           # Suricata IDS rules
    KQL = "kql"                     # Kusto Query Language (Microsoft Sentinel/Defender)
    CQL = "cql"                     # CrowdStrike Query Language
    SPL = "spl"                     # Splunk Processing Language
    STIX_PATTERN = "stix-pattern"   # Native STIX pattern language


# =============================================================================
# Nested data models
#
# These use @dataclass instead of Pydantic so they serialize cleanly with
# LangGraph's checkpointer (which uses JSON serialization under the hood).
# =============================================================================

@dataclass
class SourceLocation:
    """Pointer back to a specific location in the source material."""
    page: int | None = None
    paragraph: int | None = None
    char_offset: int | None = None
    source_url: str | None = None


@dataclass
class ClassifiedSection:
    """A section of source content with its classification (Stage 2b).

    All source content is retained regardless of classification.
    Only 'behavioral_narrative' sections are forwarded to chunking.
    Everything else stays here for audit and analyst reference.
    """
    section_id: str = ""
    text: str = ""
    classification: str = ""  # SectionClassification value
    classification_confidence: float = 0.0
    source_location: dict = field(default_factory=dict)


@dataclass
class Entity:
    """An extracted entity from the source material (Stage 2a).

    Entities are non-procedure structured intelligence: IOCs, actors,
    malware, victims, infrastructure, campaigns.

    IOC-type entities (hashes, IPs, domains, etc.) become Observable SCOs
    at serialization, linked to the procedures they appear in. They are
    NOT promoted to Indicator SDOs. They are evidence attached to a
    procedure, not standalone detection objects.
    """
    entity_id: str = ""
    entity_type: str = ""  # EntityType value
    value: str = ""
    confidence: float = 0.0
    source_location: dict = field(default_factory=dict)
    # Gate 0 sets these:
    gate_action: str | None = None       # GateAction value
    edited_value: str | None = None      # Corrected value (if edited)
    edited_type: str | None = None       # Corrected entity type (if misclassified)
    edit_rationale: str | None = None    # Why the analyst made the change or removal


@dataclass
class DetectionRule:
    """A detection rule found verbatim in the source material.

    NOT generated by the pipeline. Only preserved when the source author
    included it (e.g., a Sigma rule in a threat report appendix).
    Becomes an Indicator SDO at serialization.
    """
    rule_id: str = ""
    rule_type: str = ""       # DetectionRuleType value
    rule_content: str = ""    # Verbatim rule text from source
    description: str = ""     # Author's description if provided
    source_location: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A discrete behavioral chunk from source material (Stage 2b).

    Each chunk maps roughly 1:1 to a procedure. Chunks carry sequencing
    data for ATT&CK Flow (supports branching/convergence via DAG).

    Source-linkage fields:
        source_excerpt -- verbatim text from the source (2-3 sentences) that
            justifies this chunk. Used at the chunk-review gate so the analyst
            doesn't have to scroll the source to validate a chunk.
        source_span -- byte offsets into PipelineState["parsed_text"] derived
            from source_excerpt at postprocess time. None when the excerpt
            couldn't be located (e.g. LLM paraphrased instead of quoting).
        precedes_ids -- forward edges in chunk_id space. Inverted from
            predecessor_indices and made stable across splits/merges. The gate
            review processor and the canvas editor read this; predecessor_indices
            is kept for backward compat and as the LLM's emission shape (it
            doesn't know chunk_ids yet at emission time).
    """
    chunk_id: str = ""
    text: str = ""  # 1-2 sentence narrative of adversary behavior
    context: dict = field(default_factory=dict)  # Actor, campaign, target, malware refs
    sequence_index: int = 0  # Position in attack narrative (1-based)
    predecessor_indices: list[int] = field(default_factory=list)  # Which chunks come before
    branch_point: bool = False  # Multiple successors diverge here
    convergence_point: bool = False  # Multiple predecessors rejoin here
    behavioral_confidence: float = 0.0  # 0.0-1.0
    source_location: dict = field(default_factory=dict)
    source_excerpt: str = ""  # Verbatim source text justifying this chunk
    source_span: tuple[int, int] | None = None  # Byte offsets into parsed_text
    # Provenance category for the chunk's source evidence. Computed from
    # where source_span lands in parsed_text relative to figure markers
    # and code fences. Values:
    #   "prose"        — native PDF prose (high source fidelity)
    #   "code"         — fenced code block (verbatim, no LLM interpretation)
    #   "figure"       — vision-LLM-transcribed figure block
    #   "paraphrased"  — no verbatim anchor (LLM rewrote source enough that
    #                    parsed_text.find(excerpt) failed). Content still
    #                    grounded in source — just not literally quoted.
    # Surfaces in the UI as a badge so analysts know how much LLM
    # interpretation the chunk's evidence required.
    source_provenance: str = "paraphrased"
    # Chain-separation fields. When the source describes multiple distinct
    # attack chains by the same actor (e.g. "in a separate earlier attack
    # chain..."), the chunker marks the first chunk of the new chain with
    # chain_root=True so the orphan-link backstop in _finalize_chunks
    # respects the disconnection. chain_label is a short LLM-emitted label
    # ("Veeam intrusion", "April 2026 incident") used by the chunk-review
    # canvas to chip-tag the chain. The serializer rolls chain_roots into
    # attack-flow.start_refs and propagates the label to procedures.
    chain_root: bool = False
    chain_label: str = ""
    precedes_ids: list[str] = field(default_factory=list)  # Forward edges in chunk_id space
    # Per-chunk artifacts captured at chunk-time. Categories like
    # "registry_keys", "c2_domains", "c2_ips", "file_hashes", "file_paths",
    # "urls", "process_names", "mutexes". Values are VERBATIM substrings of the
    # source. Drives procedure→observable has-observable SROs at serialization
    # (the IoC-linking pass) so analysts can answer "what
    # observables does this procedure touch" without traversing through globals.
    artifacts: dict = field(default_factory=dict)  # {category: [verbatim_value, ...]}
    # Optional Attack-Flow precondition (attack-condition SDO source signal).
    # When the source describes a runtime check that gates flow ("if domain-
    # joined, kerberoasting; else NTLM relay"), the chunker emits a
    # precondition dict with the description + a partition of this chunk's
    # downstream successors. Shape after _finalize_chunks:
    #   {
    #     "description": str,
    #     "pattern": str | None,         # optional STIX pattern / regex / plain
    #     "pattern_type": str | None,    # "stix" | "regex" | "plain"
    #     "on_true_indices":  [seq_idx, ...],  # LLM emission shape (sequence-index)
    #     "on_false_indices": [seq_idx, ...],
    #     "on_true_ids":  [chunk_id, ...],     # post-finalize, chunk_id space
    #     "on_false_ids": [chunk_id, ...],
    #   }
    # None when the chunker found no conditional structure (the common case).
    # Bias toward null over fabrication — the chunker prompt is strict
    # about this.
    precondition: dict | None = None


@dataclass
class TechniqueMapping:
    """A technique assignment for a chunk (Stage 3).

    One chunk can map to multiple techniques (x_technique_refs is an array).
    Each technique has its own confidence score.
    """
    technique_id: str = ""  # e.g., "T1059.001"
    technique_name: str = ""  # e.g., "PowerShell"
    tactic: str = ""  # e.g., "execution"
    confidence: float = 0.0  # Per-technique confidence (0.0-1.0)
    stix_id: str | None = None  # Resolved by extract_techniques against the ATT&CK catalogue


@dataclass
class ProcedureDraft:
    """A structured procedure draft targeting v0.5.0-draft fields (Stage 4).

    This is the procedure candidate that the analyst reviews at Gate 1.
    Maps directly to x-procedure schema fields.
    """
    draft_id: str = ""
    chunk_id: str = ""  # Links back to the source chunk

    # Core fields (map to v0.5.0 schema)
    name: str = ""  # "[Verb] [Object] via [Tool/Method]"
    description: str = ""  # Three-part: objective sentence + mechanism + observation
    techniques: list[TechniqueMapping] = field(default_factory=list)
    kill_chain_phases: list[dict] = field(default_factory=list)  # [{kill_chain_name, phase_name}]
    platforms: list[str] = field(default_factory=list)  # OpenTide vocab, :: delimited
    # Verbatim from source ONLY. Note: the runtime dict carries this under
    # the key `raw_command_lines` (see drafting._assemble_draft); the
    # dataclass attribute name kept for back-compat with test fixtures
    # that build drafts via asdict(ProcedureDraft(...)). Downstream code
    # (normalize, serialize, gate edit whitelist) reads `raw_command_lines`.
    command_lines: list[str] = field(default_factory=list)
    confidence: int = 0  # 0-100, from behavioral_confidence initially

    # Temporal fields
    first_observed: str | None = None  # ISO 8601
    last_observed: str | None = None
    # No execution_start / execution_end: they fed the x_execution_start /
    # x_execution_end STIX properties, which were pruned from x-procedure in
    # v0.5.0-draft, and had no producer and no reader.

    # Reference fields
    source_refs: list[str] = field(default_factory=list)  # Identity/report STIX IDs
    vulnerability_refs: list[str] = field(default_factory=list)  # CVE STIX IDs

    # Per-procedure entity attribution. Drafting LLM populates these from
    # the entity context to tell the serializer which tools/malware THIS
    # procedure actually uses. Without them the serializer fanned out every
    # procedure to every extracted tool/malware (the Rclone-attached-to-
    # everything bug). Values
    # are entity NAMES (matched against validated_entities by value); the
    # serializer resolves them to STIX IDs via the entity registry.
    tools_used: list[str] = field(default_factory=list)
    malware_used: list[str] = field(default_factory=list)

    # Sequencing (inherited from chunk, used for ATT&CK Flow at Stage 6a)
    sequence_index: int = 0
    predecessor_indices: list[int] = field(default_factory=list)
    branch_point: bool = False
    convergence_point: bool = False

    # Quality metadata
    source_location: dict = field(default_factory=dict)
    detail_gap: bool = False  # True if source was sparse
    # Source-fidelity category propagated from the underlying chunk(s).
    # Values: "prose" | "code" | "figure" | "paraphrased" | "hybrid".
    # Single-chunk drafts inherit the chunk's value verbatim. Multi-chunk
    # drafts, if ever produced, collapse to "hybrid" when chunks disagree.
    source_provenance: str = "paraphrased"
    # Chain-separation passthrough. See Chunk.chain_root / chain_label.
    # The serializer rolls chain_root drafts into attack-flow.start_refs
    # and embeds chain_label on the procedure as x_chain_label.
    chain_root: bool = False
    chain_label: str = ""

    # Gate 1 sets these:
    gate_action: str | None = None           # GateAction value (approve, reject, edit, remove)
    reject_reason: str | None = None         # Gate1RejectReason value (if rejected)
    analyst_edits: dict | None = None        # {field_name: new_value} for edited fields only
    analyst_rationale: str | None = None     # Why (feeds the feedback flywheel)


@dataclass
class CorrelationResult:
    """Correlation match from the procedure archive (Stage 5.3).

    Every procedure is a distinct record (no dedup). Correlation identifies
    similarity for analytical value, never merge/discard.
    """
    archive_procedure_id: str = ""
    overlap_type: str = ""  # same_technique, same_tool, same_actor, temporal_cluster
    similarity_score: float = 0.0


@dataclass
class NormalizedDraft:
    """A procedure draft after normalization (Stage 5).

    Carries the original draft plus normalization additions:
    standardized names, composite confidence, and correlation results.
    """
    draft_id: str = ""  # Same as ProcedureDraft.draft_id
    composite_confidence: int = 0  # 0-100, replaces raw confidence
    confidence_breakdown: dict = field(default_factory=dict)  # {source_reliability, context_completeness, behavioral_confidence}
    standardized_names: dict = field(default_factory=dict)  # Original -> canonical mappings applied
    correlations: list[CorrelationResult] = field(default_factory=list)
    enrichment: dict = field(default_factory=dict)  # Related techniques, mitigations, detection strategies from Neo4j
    fingerprint: str = ""  # Behavioral fingerprint computed from technique + platform + command data


@dataclass
class Gate2Decision:
    """Analyst decision on procedure-entity relationships (Gate 2)."""
    approved: bool = False
    feedback: str | None = None  # Free-text if rejected


# =============================================================================
# Pipeline State
#
# This is the TypedDict that LangGraph uses. Every field is accessible to
# every node. The checkpointer serializes this to PostgreSQL.
#
# Fields are organized by pipeline stage. Each section notes which node
# writes to it and which nodes read from it.
# =============================================================================

class PipelineState(TypedDict, total=False):
    """Complete pipeline state flowing through the LangGraph graph.

    LangGraph persists this to PostgreSQL via the checkpointer at every
    step. When the graph hits a gate (interrupt), the state freezes.
    When the analyst submits their review, the state thaws and the
    graph continues from where it paused.

    total=False means all fields are optional. This is important because
    early nodes don't populate later fields. The parse_and_validate node
    doesn't know about procedure drafts yet. Each node checks for the
    fields it needs and writes the fields it produces.
    """

    # ── Source identification (written by queue, read by all) ──────────
    source_id: str
    channel: str  # Channel value
    source_type: str  # SourceType value
    raw_content_path: str  # File path or blob reference (not the content itself)
    title: str  # Display title from Source.title; propagated to CompletedBundle.title
    metadata: dict  # author, publication_date, threat_actor, campaign, malware_family, source_url
    source_reliability: int  # 0-100, set at ingestion
    gates_enabled: dict[str, bool]  # GATE_KEYS → bool. See normalize_gates() / is_gate_enabled().
    # GATE_KEYS → "review" | "assist" | "auto". Only meaningful for gates
    # that are ENABLED; a disabled gate has no reviewer. Absent on every
    # checkpoint written before this field existed, which gate_mode() reads
    # as "review" — i.e. unchanged behaviour. See normalize_gate_modes().
    gate_modes: dict[str, str]

    # Analyst's pre-flight answer to "does this source include sequential
    # information?" Values: "yes" | "no" | "auto". Written from Source.sequentiality
    # at start_pipeline. Read by entity_extraction (which short-circuits the
    # LLM classifier when "yes"/"no"). The resolved boolean lives in
    # is_sequential below.
    sequentiality: str  # "yes" | "no" | "auto"

    # Resolved boolean for downstream consumers. True iff the source describes
    # a chronologically ordered sequence of attacker actions. Written by
    # entity_extraction after consulting `sequentiality` and (in "auto" mode)
    # the LLM. Read by chunk_behaviors (gates the orphan-link backstop) and
    # serialize_stix (gates PRECEDES SRO emission).
    is_sequential: bool

    # One-sentence explanation citing the source phrasing that drove the
    # is_sequential decision. Used for analyst inspection at the chunk-review
    # gate header. Empty for "yes"/"no" overrides ("Analyst override at
    # upload" placeholder).
    sequentiality_rationale: str

    # Whether the figure_extraction node should run a vision-LLM pass over
    # each figure in the source. Default True per Source.extract_figures.
    # Analyst can disable at upload for sources known to be all-prose.
    extract_figures: bool

    # Per-figure audit list emitted by figure_extraction. Each entry:
    #   {figure_id, page, caption, figure_type, extracted_text,
    #    confidence, rationale, status}
    # `status` is "extracted" | "skipped_decorative" | "skipped_too_small"
    # | "failed". `figure_type` is "diagram" | "screenshot" | "decorative"
    # | "other". Used for Gate 0 chip + post-hoc audit; the actual figure-
    # derived text lives inline in parsed_text (placeholders replaced).
    # `extracted_text` here is a capped PREVIEW (first 200 chars) — the
    # full transcription is in parsed_text, so the audit doesn't re-carry
    # tens of KB into every checkpoint. Extracted entries also carry
    # `extracted_text_len` (full length of the original transcription).
    extracted_figures: list[dict]

    # ── Pipeline control (updated by every node) ──────────────────────
    status: str  # PipelineStatus value, drives Kanban board columns
    current_node: str  # Which graph node is active
    error: str | None  # Error message if pipeline fails

    # ── Stage 1c: Parse and validate ──────────────────────────────────
    # Written by: parse_and_validate
    # Read by: extract_entities, chunk_behaviors
    parsed_text: str  # Full extracted text from source
    parse_warnings: list[str]  # e.g., "OCR quality low", "type mismatch detected"

    # ── Stage 2a: Entity extraction ───────────────────────────────────
    # Written by: extract_entities
    # Read by: gate_0, chunk_behaviors, extract_techniques, draft_procedures
    entities: list[dict]  # List of Entity dicts
    # Also extracted by extract_entities when source contains verbatim rules.
    # These are NOT generated by the pipeline. Only preserved from source.
    detection_rules: list[dict]  # List of DetectionRule dicts (source-provided only)

    # ── Gate 0: Entity review ─────────────────────────────────────────
    # Written by: API via update_state() while graph is paused
    # Read by: gate_0 node
    gate0_reviews: list[dict]  # Raw analyst decisions: [{entity_id, action, edited_value, edited_type, rationale}]
    # Entities the analyst added at gate_0 because extraction missed them.
    # Read by: gate_0 node. Mirrors chunk_reviews.added_chunks.
    gate0_added_entities: list[dict]
    # Written by: gate_0 node (processes gate0_reviews)
    # Read by: chunk_behaviors, extract_techniques, draft_procedures, serialize_stix
    validated_entities: list[dict]  # Entities after analyst approve/edit/remove

    # ── Stage 2b: Behavioral chunking ─────────────────────────────────
    # Written by: chunk_behaviors
    # Read by: gate_chunks, extract_techniques
    classified_sections: list[dict]  # List of ClassifiedSection dicts (all content)
    chunks: list[dict]  # List of Chunk dicts (procedure-relevant only)

    # ── Attack Flow operators (AND/OR/XOR) ────────────────────────────
    # Written by: normalize (geometry-derived defaults).
    # Mutated by: gate_chunks (analyst kind overrides).
    # Read by: serialize_stix (emit attack-operator SDOs + route precedes).
    #
    # Keyed by a stable 12-hex operator_id derived from
    # (role, sorted input chunk_ids, sorted output chunk_ids) — see
    # nodes/deterministic/attack_operators.py. Same chunk geometry across
    # re-runs yields the same operator_id, so analyst overrides survive
    # normalize re-execution. Re-chunking that changes geometry produces
    # new ids and the prior overrides orphan (re-keying is not
    # implemented).
    #
    # Each entry shape:
    #   {"kind": "AND"|"OR"|"XOR",
    #    "role": "converge"|"branch",
    #    "anchor_chunk_id": str,
    #    "input_chunk_ids": [str, ...],
    #    "output_chunk_ids": [str, ...]}
    chunk_operators: dict[str, dict]

    # ── Attack Flow conditions ────────────────────────────────────────
    # Written by: normalize (via extract_conditions in attack_conditions.py).
    # Mutated by: gate_chunks (analyst description / partition overrides).
    # Read by: serialize_stix (emit attack-condition SDOs + route precedes).
    #
    # Keyed by the anchor chunk_id (the chunk whose precondition this is).
    # Each entry shape:
    #   {"description": str,
    #    "pattern": str | None,
    #    "pattern_type": "stix" | "regex" | "plain" | None,
    #    "on_true_ids":  [chunk_id, ...],
    #    "on_false_ids": [chunk_id, ...]}
    chunk_conditions: dict[str, dict]

    # ── Gate (chunks): Chunk validation ───────────────────────────────
    # Sits between chunk_behaviors and extract_techniques. The analyst can
    # approve / edit / drop / merge chunks, add chunks the LLM missed, add or
    # remove precedes edges, override operator kinds and conditions, or
    # reject the pass to re-run chunk_behaviors with feedback.
    #
    # Written by: API via update_state() while graph is paused
    # Read by: gate_chunks node
    chunk_reviews: dict | None           # Raw analyst submission (see ChunkReviewSubmission below)
    # Written by: gate_chunks node
    # Read by: extract_techniques (on approve), chunk_behaviors (on rerun)
    chunk_decisions: list[dict]          # Processed per-chunk decisions
    chunks_approved_ids: list[str]       # IDs of approved chunks
    chunks_rejection_routing: str | None # "chunk_behaviors" or None
    # Written by: gate_chunks on reject; consumed by chunk_behaviors on next run.
    # High-level "redo it all" reason + comments. Distinct from the existing
    # chunk_feedback field which carries per-chunk-pair structured problems.
    chunk_rerun_feedback: dict | None    # {"reason": str, "comments": str} | None
    # Append-only durable audit of EVERY non-approve gate_chunks signal
    # across ALL passes: wholesale rejects (chunk_rerun_feedback is cleared
    # by chunk_behaviors after consumption), per-chunk drops/edits
    # (chunk_decisions is last-write-wins across re-chunk loops), and
    # analyst-added chunks (chunk_reviews is cleared by this gate). Entries
    # are stored in the exact delta shape _chunk_deltas emits, with chunk
    # text snapshotted at gate time (chunks regenerate on loops). Mirrors
    # gate1_correction_log. Written by: gate_chunks (read-modify-write
    # append). Read by: synthesize_feedback + GET /feedback-patterns/captured.
    chunk_correction_log: list[dict]

    # ── Stage 3: Technique extraction ─────────────────────────────────
    # Written by: extract_techniques
    # Read by: draft_procedures
    # Stored as a dict mapping chunk_id -> list of TechniqueMapping dicts.
    #
    # The propose→retrieve→pick step splits picks by confidence_bucket:
    #   technique_mappings           -> 'definite' + 'probable' picks; feed
    #                                   drafting and the bundle by default.
    #   technique_mappings_for_review -> 'possible' picks; surfaced at Gate 1
    #                                   (Technique Review) as a low-confidence
    #                                   lane the analyst can promote into
    #                                   the bundle.
    technique_mappings: dict[str, list[dict]]
    technique_mappings_for_review: dict[str, list[dict]]

    # Per-chunk output of the propose-step LLM call. Keyed by chunk_id; each
    # entry has {behavior_description, objective, tactics, proposed_techniques}.
    # The objective is a transient, pipeline-internal concept — read by the
    # pick step (compose-vs-drift discriminator) and by drafting (anchors
    # the description's lead sentence). Persisted in the LangGraph
    # checkpoint for post-hoc audit but NOT serialized as a STIX field.
    proposals_by_chunk: dict[str, dict]

    # ── Stage 4: Procedure drafting ───────────────────────────────────
    # Written by: draft_procedures
    # Read by: gate_1, normalize
    drafts: list[dict]  # List of ProcedureDraft dicts

    # ── Gate 1: Technique + procedure review ──────────────────────────
    # Written by: API via update_state() while graph is paused
    # Read by: gate_1 node
    gate1_reviews: list[dict]  # Raw analyst decisions: [{draft_id, action, reject_reason, analyst_edits, rationale}]
    # Promotions: analyst-approved moves of possible-bucket picks from
    # technique_mappings_for_review into the active technique_mappings.
    # Each entry: {chunk_id, technique_id}. Empty list when the analyst
    # didn't promote anything.
    gate1_promotions: list[dict]
    # Written by: gate_1 node (processes gate1_reviews)
    # Read by: normalize (approved drafts), chunk_behaviors or extract_techniques (rejected)
    gate1_decisions: list[dict]  # Per-draft: {draft_id, action, reason, feedback}
    gate1_approved_draft_ids: list[str]  # Convenience: IDs of approved drafts
    gate1_rejection_routing: str | None  # "chunk_behaviors" or "extract_techniques" or None
    # Append-only audit of EVERY non-approve Gate 1 decision across ALL
    # passes (reject/edit/remove). gate1_decisions above is last-write-wins,
    # so a reject that triggers a re-extract loop is erased when gate_1 runs
    # again on the post-loop (all-approve) pass — the reject feedback would
    # never reach the flywheel. This accumulator survives the loop+overwrite
    # so synthesize_feedback can learn from rejections at run completion.
    # Each entry is self-contained (carries draft_name/chunk_id/rejected +
    # corrected techniques) because draft_ids are regenerated each re-extract.
    # Also carries {action: "promote", chunk_id, technique_id} records for
    # analyst promotions — the transient gate1_promotions list is cleared in
    # the same gate_1 update that applies it, so the log is the only copy
    # that survives to synthesis.
    # Written by: gate_1 (read-modify-write append). Read by: synthesize_feedback.
    gate1_correction_log: list[dict]
    # Corrective feedback for a wrong_technique re-extract. Built by gate_1
    # when routing == "extract_techniques"; consumed by extract_techniques to
    # inject analyst guidance into the propose + pick prompts (which also run
    # with bypass_cache so even a repeat reject with identical wording gets a
    # fresh extraction). Cleared after consumption. Mirrors
    # chunk_rerun_feedback. Each entry: {chunk_id, action, reject_reason,
    # rationale, rejected_techniques, corrected_techniques, added_techniques}
    technique_rerun_feedback: list[dict] | None

    # ── Chunk feedback (written by gate_1, read by chunk_behaviors on retry) ─
    # Structured analyst feedback when BAD_CHUNK_BOUNDARY is selected.
    # Each entry: {draft_id, chunk_id, problem, related_draft_id?, guidance}
    chunk_feedback: list[dict]
    # Technique overrides the analyst applied before the re-chunk routing.
    # Keyed by a text fingerprint so they can be re-applied to new drafts
    # after re-chunking produces fresh chunk_ids/draft_ids.
    # Each entry: {chunk_text_hash, techniques, rationale?}
    previous_technique_overrides: list[dict]

    # ── Stage 5: Normalization ────────────────────────────────────────
    # Written by: normalize
    # Read by: gate_2, serialize_stix
    normalized_drafts: list[dict]  # List of NormalizedDraft dicts
    # Written by: normalize
    # Read by: gate_2 (served as items to the frontend for relationship review)
    # Each entry: {id, relationship_type, source_name, target_name, source_type, target_type}
    relationship_preview: list[dict]

    # ── Gate 2: Relationship review ───────────────────────────────────
    # Written by: API via update_state() while graph is paused
    # Read by: gate_2 node
    gate2_review: dict  # Batch mode (legacy): {approved: bool, feedback: str|None}. Unused in per-relationship mode.
    gate2_reviews: list[dict]  # Per-relationship mode (preferred): [{rel_id, action, edited fields, rationale}]
    # Written by: gate_2 node (processes gate2_review / gate2_reviews)
    # Read by: serialize_stix
    gate2_decision: dict  # Gate2Decision dict
    gate2_approved_rel_ids: list[str]  # IDs of approved relationships
    gate2_removed_rel_ids: list[str]  # IDs of relationships the analyst removed
    gate2_added_rels: list[dict]  # Relationships the analyst added manually

    # ── Stage 6: STIX serialization ───────────────────────────────────
    # Written by: serialize_stix
    # Read by: distribute
    stix_bundle: dict  # The complete STIX 2.1 bundle (JSON)
    validation_results: dict  # {schema, reference_integrity, attack_flow, tuple_semantics: bool}
    validation_errors: list[str]  # Any validation failures

    # ── Stage 6b: Bundle validation (validate_bundle node) ────────────
    # Written by: validate_bundle
    # Read by: gate_2 / bundle response API; consumed by analyst at gate_2
    # surfacing and surfaced on completed bundles for post-hoc audit.
    #
    # Each entry is a structured correction record:
    #   {rule, severity, ref_id?, recovered_from?, holder_id?,
    #    holder_type?, before?, after?, message}
    # severity ∈ {"auto_fix", "repaired", "warn", "hard_fail"}.
    # "repaired" entries cover for upstream bugs (e.g. dangling ref
    # recovered from normalize output) and are louder than "auto_fix"
    # (housekeeping like SRO direction flips, dedup, fingerprint recompute).
    # "hard_fail" entries are present when the validator fails the source;
    # they describe what needs fixing before a re-run.
    bundle_corrections: list[dict]
    # When the validator hard-fails, it sets this to True so route_after_validate
    # ends the run (the source lands in Failed; distribute never runs).
    bundle_validation_failed: bool

    # ── Distribution ──────────────────────────────────────────────────
    # Written by: distribute
    neo4j_write_status: str | None  # "success" or error message
    objects_written: int  # Count of STIX objects written to Neo4j
    persistence_errors: list[str]  # Non-fatal persistence failures (bundle store, source file read) surfaced to UI

    # ── Feedback synthesis (post-distribute) ──────────────────────────
    # Written by: synthesize_feedback
    # Read by: nothing in-pipeline; the audit trail keys downstream queries
    # against the FeedbackPattern table. Feedback synthesis runs after
    # distribute and emits one or more FeedbackPattern rows into postgres
    # capturing analyst-decision patterns (e.g., "letterhead emails
    # rejected as IoCs", "chunker over-chunks credential-access steps")
    # so future runs can consult them at LLM-prompt-build time.
    feedback_synthesis: dict  # {patterns_emitted, llm_tokens_used, status, error?}
    feedback_pattern_ids: list[str]  # IDs of newly emitted FeedbackPattern rows
