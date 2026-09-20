"""Pydantic models for API request/response validation.

These are the API contract. Frontend sends requests matching these shapes,
backend validates them, and responses conform to these schemas.

Separate from the LangGraph state dataclasses (state.py) which are internal.
The API schemas translate between HTTP and pipeline state.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import AliasPath, BaseModel, Field, field_validator, model_validator

from app.graph.state import (
    CHUNK_EDITABLE_FIELDS,
    DEFAULT_GATE_MODES,
    DEFAULT_GATES,
    ChunkProblemType,
    EntityType,
    GateAction,
    Gate1RejectReason,
    PipelineStatus,
    SourceType,
    normalize_gate_modes,
    normalize_gates,
)

# Valid values derived from enums (used in Literal types and validators)
_VALID_SOURCE_TYPES = tuple(e.value for e in SourceType)
_VALID_STATUSES = tuple(e.value for e in PipelineStatus)
_VALID_GATE_ACTIONS = tuple(e.value for e in GateAction)
_VALID_REJECT_REASONS = tuple(e.value for e in Gate1RejectReason)
_VALID_CHUNK_PROBLEMS = tuple(e.value for e in ChunkProblemType)

# Allowed base directories for source content (path traversal guard).
# Resolved at import time so we compare apples-to-apples after symlink
# resolution (e.g. on macOS `/tmp` is a symlink to `/private/tmp`, so a
# raw startswith() on `/tmp/pipeline/` would fail every check).
_ALLOWED_PATH_PREFIXES = ("/data/", "/tmp/pipeline/", "/app/uploads/")
_ALLOWED_PATH_PREFIXES_RESOLVED = tuple(
    str(Path(p).resolve()) for p in _ALLOWED_PATH_PREFIXES
)
_ALLOWED_URL_SCHEMES = ("https://", "http://", "s3://", "gs://")

# Metadata size limits
_METADATA_MAX_KEYS = 50
_METADATA_MAX_SERIALIZED_BYTES = 32_768  # 32 KB
_METADATA_MAX_DEPTH = 6  # nesting depth (top-level dict = depth 1)
_METADATA_MAX_LEAF_CHARS = 4_096  # per string-leaf cap


def _validate_content_path(v: str) -> str:
    """Validate raw_content_path against path traversal and allowed locations.

    Defense in depth, not perimeter:
      1. URL schemes: pass through (separate validation path).
      2. NUL bytes + backslashes: outright reject (NUL-truncation tricks
         in legacy C string handling; backslashes are path separators on
         Windows tooling and have no business in a Unix path).
      3. Literal `..`: reject before resolve() so we never even try
         to follow an explicit traversal.
      4. Path.resolve(): collapses `.`, `..`, and follows symlinks for
         any existing component. The resolved string is then compared
         against the resolved allowed prefixes — this catches the
         attack the prior validator missed (a symlink under an allowed
         prefix that points at e.g. `/etc/passwd`).
    """
    v = v.strip()
    if not v:
        raise ValueError("raw_content_path must not be empty")

    # URL paths are allowed with approved schemes — out of filesystem scope.
    if any(v.startswith(scheme) for scheme in _ALLOWED_URL_SCHEMES):
        return v

    if "\x00" in v:
        raise ValueError("raw_content_path must not contain NUL bytes")
    if "\\" in v:
        raise ValueError(
            "raw_content_path must use forward slashes only ('\\' not allowed)"
        )
    if ".." in v:
        raise ValueError(
            "Path traversal sequences ('..') are not allowed in raw_content_path"
        )

    # Path.resolve() yields an absolute, normalized path with all
    # existing symlinks resolved. Non-existent tail components are
    # appended literally — that's fine because the *prefix* is what we
    # enforce, and the prefix has to exist (it's the upload dir).
    try:
        resolved = str(Path(v).resolve())
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Could not resolve raw_content_path: {exc}")

    if not any(
        resolved == prefix or resolved.startswith(prefix.rstrip("/") + "/")
        for prefix in _ALLOWED_PATH_PREFIXES_RESOLVED
    ):
        raise ValueError(
            f"raw_content_path must resolve under one of "
            f"{_ALLOWED_PATH_PREFIXES} or be a URL "
            f"(https://, http://, s3://, gs://). "
            f"Resolved to '{resolved}'."
        )

    return v


def _check_metadata_depth(value: Any, depth: int) -> None:
    """Recursively enforce nesting depth + per-leaf string length.

    `_resolve_stix_ids` and other downstream code walk nested structures
    recursively; a pathologically deep object could blow the stack or
    stall the walk. Bound both depth and per-leaf size so analyst-supplied
    metadata can't smuggle a runaway structure past the top-level key /
    byte caps.
    """
    if depth > _METADATA_MAX_DEPTH:
        raise ValueError(
            f"metadata nesting exceeds max depth of {_METADATA_MAX_DEPTH}"
        )
    if isinstance(value, dict):
        for val in value.values():
            _check_metadata_depth(val, depth + 1)
    elif isinstance(value, list):
        for item in value:
            _check_metadata_depth(item, depth + 1)
    elif isinstance(value, str) and len(value) > _METADATA_MAX_LEAF_CHARS:
        raise ValueError(
            f"metadata string value exceeds {_METADATA_MAX_LEAF_CHARS} chars"
        )


def _validate_metadata(v: dict[str, Any]) -> dict[str, Any]:
    """Enforce size and depth limits on metadata."""
    if len(v) > _METADATA_MAX_KEYS:
        raise ValueError(f"metadata must have at most {_METADATA_MAX_KEYS} keys, got {len(v)}")

    serialized = json.dumps(v)
    if len(serialized.encode("utf-8")) > _METADATA_MAX_SERIALIZED_BYTES:
        raise ValueError(
            f"metadata must be under {_METADATA_MAX_SERIALIZED_BYTES} bytes when serialized, "
            f"got {len(serialized.encode('utf-8'))}"
        )

    # Recursive depth + per-leaf length (the top-level caps above don't
    # catch a single deeply-nested chain).
    _check_metadata_depth(v, depth=1)

    return v


# =============================================================================
# Source Queue schemas
# =============================================================================

class SourceCreate(BaseModel):
    """Request body for creating a new source in the queue."""
    source_type: Literal[_VALID_SOURCE_TYPES] = Field(
        ...,
        description="SourceType enum value: pdf, html, markdown, free_text, etc.",
    )
    title: str = Field(
        default="Untitled Source",
        max_length=512,
        description="Display name for Kanban card",
    )
    raw_content_path: str = Field(
        ...,
        description="File path or blob reference to source content",
    )
    channel: Literal["manual", "automated"] = Field(
        default="manual",
        description="How the source entered: manual or automated",
    )
    gates_enabled: dict[str, bool] = Field(
        default_factory=lambda: dict(DEFAULT_GATES),
        description=(
            "Per-gate enable/disable. Keys: entities, procedures, bundle. "
            "Each value defaults to True. Accepts a bare bool for legacy "
            "callers (expanded to all-keys). Unknown keys are rejected."
        ),
    )
    gate_modes: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_GATE_MODES),
        description=(
            "Per-gate review mode, meaningful only where the gate is enabled. "
            '"review" = a human reviews it (default). "assist" = the AI '
            'reviewer recommends and a human decides. "auto" = the AI '
            "reviewer decides unattended. A gate whose reviewer has not "
            "shipped falls through to human review whatever this says."
        ),
    )
    source_reliability: int = Field(
        default=50,
        ge=0,
        le=100,
        description="Source reliability score 0-100",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Flexible metadata: author, publication_date, campaign, tlp, source_url",
    )
    sequentiality: Literal["yes", "no", "auto"] = Field(
        default="auto",
        description=(
            "Does the source include sequential information? 'yes' = single "
            "intrusion narrative with chronological ordering; 'no' = catalog/"
            "profile of procedures without intrinsic ordering; 'auto' defers "
            "to the entity_extraction classifier. Gates the chunker's "
            "orphan-link backstop and the serializer's PRECEDES SROs."
        ),
    )
    extract_figures: bool = Field(
        default=True,
        description=(
            "Run a vision-LLM pass over each figure in the source and inline "
            "the extracted text into parsed_text. Default True. Disable for "
            "sources known to be all-prose (no diagrams or command-line "
            "screenshots)."
        ),
    )

    @field_validator("raw_content_path")
    @classmethod
    def check_content_path(cls, v: str) -> str:
        return _validate_content_path(v)

    @field_validator("metadata")
    @classmethod
    def check_metadata(cls, v: dict[str, Any]) -> dict[str, Any]:
        return _validate_metadata(v)

    @field_validator("gates_enabled", mode="before")
    @classmethod
    def check_gates(cls, v: object) -> dict[str, bool]:
        return normalize_gates(v)

    @field_validator("gate_modes", mode="before")
    @classmethod
    def check_gate_modes(cls, v: object) -> dict[str, str]:
        return normalize_gate_modes(v)


class SourceResponse(BaseModel):
    """Response body for a source record."""
    id: uuid.UUID
    source_type: str
    title: str
    raw_content_path: str
    status: str
    channel: str
    gates_enabled: dict[str, bool]
    gate_modes: dict[str, str] = Field(
        default_factory=lambda: dict(DEFAULT_GATE_MODES),
    )
    claimed_by: str | None
    claimed_at: datetime | None
    metadata: dict[str, Any] = Field(validation_alias=AliasPath("metadata_"))
    source_reliability: int
    thread_id: uuid.UUID | None
    error: str | None
    persistence_errors: list[str] = Field(default_factory=list)
    bundle_corrections: list[dict] = Field(default_factory=list)
    sequentiality: Literal["yes", "no", "auto"] = "auto"
    extract_figures: bool = True
    # Run counts, NULL until the producing stage has run (see models/source.py).
    entity_count: int | None = None
    draft_count: int | None = None
    objects_written: int | None = None
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}

    @field_validator("gates_enabled", mode="before")
    @classmethod
    def coerce_gates(cls, v: object) -> dict[str, bool]:
        return normalize_gates(v)

    @field_validator("gate_modes", mode="before")
    @classmethod
    def coerce_gate_modes(cls, v: object) -> dict[str, str]:
        # Rows predating the column read as None -> all "review". Mirrors
        # coerce_gates so both fields tolerate the same shapes.
        return normalize_gate_modes(v)


class SourceListResponse(BaseModel):
    """Response body for listing sources."""
    sources: list[SourceResponse]
    total: int


class SourceClaim(BaseModel):
    """Request body for claiming a source."""
    analyst: str = Field(
        ...,
        max_length=128,
        description="Analyst email or username",
    )


class SourceStatusUpdate(BaseModel):
    """Request body for updating source status (Kanban drag-and-drop)."""
    status: Literal[_VALID_STATUSES] = Field(
        ...,
        description="New PipelineStatus value",
    )


# =============================================================================
# Gate schemas
# =============================================================================

class Gate0ReviewItem(BaseModel):
    """A single entity review decision for Gate 0."""
    entity_id: str
    action: Literal[_VALID_GATE_ACTIONS] = Field(
        ...,
        description="GateAction value: approve, reject, edit, remove",
    )
    edited_value: str | None = Field(
        default=None,
        description="Corrected value (required if action=edit)",
    )
    edited_type: str | None = Field(
        default=None,
        description="Corrected EntityType value (optional with action=edit)",
    )
    edited_role: str | None = Field(
        default=None,
        description=(
            "Corrected organization_role (victim/sponsor/publisher/author/other) "
            "or location_role (victim/origin/context). Optional with action=edit. "
            "The gate_0 processor routes this to the right field based on the "
            "entity's type — organization picks go to organization_role, "
            "location picks go to location_role, others ignore it."
        ),
    )
    rationale: str | None = Field(
        default=None,
        description="Why the analyst made this decision",
    )


class AddedEntityItem(BaseModel):
    """An entity the analyst adds that the extractor missed.

    Gate 0 previously had no addition channel at all — only approve / reject /
    edit / remove on entities the LLM had already produced — so a recall miss
    was unrecoverable. That is asymmetric with gate_chunks (added_chunks) and
    gate_1 (promotions), and the only workaround was hijacking an unrelated
    entity via `edit`, which corrupts its provenance (audit finding H3).
    """
    value: str = Field(..., min_length=1, max_length=1024)
    entity_type: str = Field(..., description="EntityType value")
    organization_role: str | None = None
    location_role: str | None = None
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    rationale: str | None = Field(default=None, max_length=500)

    @field_validator("entity_type")
    @classmethod
    def check_entity_type(cls, v: str) -> str:
        """Reject a type the pipeline has no mapping for.

        `_build_added_entity` copies entity_type through untouched, and
        serialization drops any entity whose type has no STIX mapping with
        nothing but a log warning — so an unvalidated type means the
        analyst's addition disappears between the gate and the bundle,
        silently. Better to 422 at the boundary where someone can see it.
        """
        valid = {e.value for e in EntityType}
        if v not in valid:
            raise ValueError(
                f"Invalid entity_type: {v!r}. Must be one of {sorted(valid)}"
            )
        return v


class Gate0Submit(BaseModel):
    """Full review submission for Gate 0 (entity review)."""
    reviews: list[Gate0ReviewItem] = Field(
        ...,
        description="Per-entity decisions. Unreviewed entities default to approve.",
    )
    added_entities: list[AddedEntityItem] = Field(
        default_factory=list,
        description=(
            "Entities the analyst adds because the extractor missed them. "
            "They enter validated_entities pre-approved and stamped "
            "analyst_added=True."
        ),
    )
    checkpoint_id: str | None = Field(
        default=None,
        description=(
            "Optimistic-concurrency token. The frontend forwards the "
            "checkpoint_id it received from GET /pending; the submit "
            "handler rejects (409) if the LangGraph checkpoint has "
            "advanced in the meantime — preventing two analysts (or a "
            "double-click) from clobbering each other's reviews."
        ),
    )


class Gate1ReviewItem(BaseModel):
    """A single procedure draft review decision for Gate 1."""
    draft_id: str
    action: Literal[_VALID_GATE_ACTIONS] = Field(
        ...,
        description="GateAction value: approve, reject, edit, remove",
    )
    reject_reason: Literal[_VALID_REJECT_REASONS] | None = Field(
        default=None,
        description="Gate1RejectReason value (required if action=reject)",
    )
    # Free-form rather than an enum: the human UI has always been able to
    # remove a draft without justifying it, and narrowing that here would
    # reject submissions the analyst can legitimately make. The AI reviewer is
    # constrained instead, by RemoveReason in services/reviewer/models.py.
    remove_reason: str | None = Field(
        default=None, max_length=64,
        description="Why the draft was dropped (with action=remove)",
    )
    analyst_edits: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Field edits. Applied only under action=edit; see "
            "_apply_draft_edits in app.nodes.gates for the whitelist."
        ),
    )
    rationale: str | None = Field(
        default=None,
        description="Why the analyst made this decision",
    )


class Gate1PromotionItem(BaseModel):
    """A single promotion of a possible-bucket technique pick into the bundle.

    The C+A+D pick step splits picks by confidence_bucket: definite + probable
    flow into the bundle, possible lands in `technique_mappings_for_review`
    for analyst inspection. At Gate 1 the analyst can promote a possible pick
    by submitting a `{chunk_id, technique_id}` here. The gate_1 node then:
      - Moves the pick from technique_mappings_for_review[chunk_id] into
        technique_mappings[chunk_id], with the bucket flipped to 'probable'.
      - Adds the technique to the matching draft's `techniques` list so
        downstream serialization carries it as both `x_technique_refs` and
        a `procedure --uses--> attack-pattern` SRO.
      - Annotates the pick with `analyst_promoted=True` for audit.
    """
    chunk_id: str = Field(..., description="The chunk_id whose review-lane pick is being promoted.")
    technique_id: str = Field(..., description="ATT&CK technique ID from the review-lane entry.")


class Gate1Submit(BaseModel):
    """Full review submission for Gate 1 (procedure + technique review)."""
    reviews: list[Gate1ReviewItem] = Field(
        ...,
        description="Per-draft decisions. Unreviewed drafts default to approve.",
    )
    promotions: list[Gate1PromotionItem] = Field(
        default_factory=list,
        description=(
            "Promote possible-bucket techniques into the bundle. Each entry "
            "moves a pick from technique_mappings_for_review into the active "
            "technique_mappings + adds it to the matching draft's techniques."
        ),
    )
    checkpoint_id: str | None = Field(
        default=None,
        description="Optimistic-concurrency token; see Gate0Submit.checkpoint_id.",
    )


_VALID_RELATIONSHIP_TYPES = (
    "uses", "targets", "precedes", "indicates",
    "has-observable", "attributed-to", "mitigates", "detects",
    "exploits", "component-of",
)


class Gate2ReviewItem(BaseModel):
    """A single relationship review decision for Gate 2."""
    rel_id: str
    action: Literal["approve", "edit", "remove"] = Field(
        ...,
        description="Action: approve, edit, or remove",
    )
    edited_rel_type: str | None = Field(
        default=None,
        description="Corrected relationship type (if action=edit)",
    )
    edited_source: str | None = Field(
        default=None,
        max_length=512,
        description="Corrected source name (if action=edit)",
    )
    edited_target: str | None = Field(
        default=None,
        max_length=512,
        description="Corrected target name (if action=edit)",
    )
    # STIX types of the endpoints. Required in practice for an added row
    # (the canvas knows both); an edit that omits them keeps the original
    # row's types. Without a type, "Cobalt Strike" is ambiguous between the
    # malware and the tool node the same name can carry.
    edited_source_type: str | None = Field(
        default=None, max_length=64, pattern=r"^[a-z0-9-]+$",
        description="STIX type of the source endpoint (e.g. intrusion-set, x-procedure)",
    )
    edited_target_type: str | None = Field(
        default=None, max_length=64, pattern=r"^[a-z0-9-]+$",
        description="STIX type of the target endpoint (e.g. identity, tool)",
    )
    rationale: str | None = Field(
        default=None,
        max_length=500,
        description="Why the analyst made this decision",
    )

    @field_validator("edited_rel_type")
    @classmethod
    def validate_rel_type(cls, v: str | None) -> str | None:
        if v is not None and v not in _VALID_RELATIONSHIP_TYPES:
            raise ValueError(f"Invalid relationship type: {v}. Must be one of {_VALID_RELATIONSHIP_TYPES}")
        return v


class Gate2Submit(BaseModel):
    """Review submission for Gate 2 (relationship review).

    Supports two modes:
    1. Per-relationship: provide reviews list (preferred)
    2. Batch: provide approved + feedback (legacy, used for batch reject)
    """
    reviews: list[Gate2ReviewItem] | None = Field(
        default=None,
        description="Per-relationship decisions. If provided, approved/feedback are ignored.",
    )
    approved: bool = Field(
        default=True,
        description="Batch mode: True to approve all, False to reject all",
    )
    feedback: str | None = Field(
        default=None,
        description="Batch mode: free-text feedback (recommended if rejecting)",
    )
    checkpoint_id: str | None = Field(
        default=None,
        description="Optimistic-concurrency token; see Gate0Submit.checkpoint_id.",
    )


# =============================================================================
# Chunk-review gate schemas
# =============================================================================
#
# Submission shape mirrors gate_chunks's expected `chunk_reviews` dict.
# Path: /api/gates/{thread_id}/chunks/{pending|submit}.

_VALID_CHUNK_ACTIONS = ("approve", "edit", "drop", "merge")
_VALID_EDGE_ACTIONS = ("add", "remove")
_VALID_OPERATOR_KINDS = ("AND", "OR", "XOR")
_VALID_CHUNK_REJECT_REASONS = (
    "missed_procedures", "over_chunked", "under_chunked",
    "bad_boundaries", "bad_descriptions", "bad_flow", "other",
)
# The same object gate_chunks filters on — see CHUNK_EDITABLE_FIELDS in
# app.graph.state for why there is exactly one.
_CHUNK_EDITABLE_FIELDS = CHUNK_EDITABLE_FIELDS


class ChunkDecisionItem(BaseModel):
    """A single per-chunk decision at the chunk-review gate."""
    chunk_id: str
    action: Literal[_VALID_CHUNK_ACTIONS]
    edits: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Field overrides applied when action=edit. Whitelisted to: "
            f"{', '.join(_CHUNK_EDITABLE_FIELDS)}. Unknown keys are silently dropped."
        ),
    )
    merge_with: list[str] = Field(
        default_factory=list,
        description=(
            "With action=merge: chunk_ids absorbed INTO this one. They are "
            "dropped and their flow edges rewired onto the survivor. Supply "
            "`edits.text` to control the merged wording; otherwise the texts "
            "are concatenated in sequence order."
        ),
    )
    rationale: str | None = Field(default=None, max_length=500)

    @field_validator("edits")
    @classmethod
    def filter_edits_keys(cls, v: dict | None) -> dict | None:
        if v is None:
            return v
        # Mirror gate_chunks's filter so the API is honest about what gets applied.
        return {k: val for k, val in v.items() if k in _CHUNK_EDITABLE_FIELDS}


class AddedChunkItem(BaseModel):
    """A new chunk the analyst added that the LLM missed."""
    text: str = Field(..., min_length=1, max_length=4000)
    source_excerpt: str = Field(default="", max_length=4000)
    context: dict[str, Any] = Field(default_factory=dict)
    behavioral_confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    branch_point: bool = False
    convergence_point: bool = False


class ChunkEdgeMutation(BaseModel):
    """An edge add/remove on the chunk-flow DAG.

    JSON uses `from`/`to` (matching the gate processor's expected shape);
    we expose `from_` in code because `from` is reserved.
    """
    action: Literal[_VALID_EDGE_ACTIONS]
    from_: str = Field(..., alias="from")
    to: str

    model_config = {"populate_by_name": True}


class ChunkGateRejectItem(BaseModel):
    """High-level rejection — re-runs chunk_behaviors with these comments."""
    reason: Literal[_VALID_CHUNK_REJECT_REASONS]
    comments: str = Field(default="", max_length=2000)


class ConditionEditItem(BaseModel):
    """Analyst edit to a chunk's attack-condition.

    Two actions:
      * `"set"` — create or update the condition on the given chunk.
        description is required; pattern + pattern_type optional but
        must be paired (or both omitted). on_true_ids + on_false_ids
        partition the chunk's downstream successors; the gate processor
        re-validates against precedes_ids and silently prunes invalid
        refs (same contract as _finalize_chunks).
      * `"clear"` — drop any existing condition on the chunk. All other
        fields are ignored.
    """
    chunk_id: str
    action: Literal["set", "clear"]
    description: str | None = None
    pattern: str | None = None
    pattern_type: Literal["stix", "regex", "plain"] | None = None
    on_true_ids: list[str] = Field(default_factory=list)
    on_false_ids: list[str] = Field(default_factory=list)


class OperatorOverrideItem(BaseModel):
    """Analyst override of an attack-operator's kind.

    The `operator_id` is the 12-hex stable id surfaced in the
    GET chunks/pending response (computed live from current chunk
    geometry via `infer_operators`). Overrides for operators whose
    geometry no longer exists post-edit are silently dropped at
    normalize time — see attack_operators.infer_operators's
    existing_operators merge semantics.
    """
    operator_id: str
    kind: Literal[_VALID_OPERATOR_KINDS]


class ChunkGateSubmit(BaseModel):
    """Full submission for the chunk-review gate.

    All fields optional. If `reject` is set, the gate processor ignores
    decisions/added_chunks/edges/operator_overrides and routes back to
    chunk_behaviors. `is_sequential` is honoured on BOTH paths: a reject
    re-runs the chunker, whose prompt depends on the flag.
    """
    decisions: list[ChunkDecisionItem] = Field(default_factory=list)
    added_chunks: list[AddedChunkItem] = Field(default_factory=list)
    edges: list[ChunkEdgeMutation] = Field(default_factory=list)
    operator_overrides: list[OperatorOverrideItem] = Field(default_factory=list)
    condition_edits: list[ConditionEditItem] = Field(default_factory=list)
    reject: ChunkGateRejectItem | None = None
    is_sequential: bool | None = Field(
        default=None,
        description=(
            "Analyst override of the auto-detected sequentiality. None keeps "
            "the detected value. False ships the bundle without PRECEDES "
            "edges, operators or conditions; True enables all three."
        ),
    )
    checkpoint_id: str | None = Field(
        default=None,
        description="Optimistic-concurrency token; see Gate0Submit.checkpoint_id.",
    )


class ChunkGateReviewPayload(BaseModel):
    """Response body when fetching the chunk-review gate's pending data.

    Carries everything the canvas needs to render: chunks (with their new
    source_excerpt / source_span / precedes_ids fields), the full
    parsed_text for source-pane highlighting, and the validated entities
    for context badges.
    """
    thread_id: str
    source_id: str
    status: str
    chunks: list[dict[str, Any]]
    parsed_text: str
    validated_entities: list[dict[str, Any]] = Field(default_factory=list)
    classified_sections: list[dict[str, Any]] = Field(default_factory=list)
    # Sequentiality classification surfaced as a header chip — explains
    # why the canvas may have many disconnected components (catalog mode)
    # vs a single connected DAG (sequential mode).
    is_sequential: bool = True
    sequentiality_rationale: str = ""
    # Attack-Flow operators inferred live from current chunk geometry,
    # with any analyst kind overrides from a prior submission merged in.
    # Keyed by operator_id — same id the canvas dropdown writes into
    # operator_overrides on submit. Empty when is_sequential=False
    # (operators don't apply to catalog sources).
    chunk_operators: dict[str, dict] = Field(default_factory=dict)
    # Attack-Flow conditions surfaced from current chunks (post-finalize),
    # merged with any analyst edits already in state. Keyed by anchor
    # chunk_id — the canvas reads to render description + partition,
    # writes ConditionEditItem on change. Empty when is_sequential=False.
    chunk_conditions: dict[str, dict] = Field(default_factory=dict)
    checkpoint_id: str | None = Field(
        default=None,
        description=(
            "LangGraph checkpoint id at fetch time. The frontend echoes "
            "this on /submit so the backend can reject stale reviews."
        ),
    )


class GateReviewPayload(BaseModel):
    """Response body when fetching pending gate review data.

    The items field contains the relevant objects for the analyst to review.
    Shape depends on gate_id:
    - Gate 0: list of Entity dicts
    - Gate 1: list of ProcedureDraft dicts + technique_mappings
    - Gate 2: list of relationship preview dicts (source_name, relationship_type, target_name)
    """
    gate_id: int
    thread_id: str
    source_id: str
    status: str
    items: list[dict[str, Any]] = Field(
        description="Objects for analyst review (entities, drafts, or normalized drafts)",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context: metadata, technique_mappings, etc.",
    )
    checkpoint_id: str | None = Field(
        default=None,
        description=(
            "LangGraph checkpoint id at fetch time. The frontend echoes "
            "this on /submit so the backend can reject stale reviews."
        ),
    )


class GateSubmitResponse(BaseModel):
    """Response after submitting a gate review."""
    thread_id: str
    gate_id: int
    status: str = Field(
        description="New pipeline status after resuming",
    )
    next_node: str = Field(
        description="Which node the pipeline will execute next",
    )


# =============================================================================
# Pipeline schemas
# =============================================================================

class PipelineRunRequest(BaseModel):
    """Request body for triggering a pipeline run."""
    source_id: uuid.UUID = Field(
        ...,
        description="Source ID from the queue",
    )
    gates_enabled: dict[str, bool] | None = Field(
        default=None,
        description=(
            "Override gates_enabled from source record (optional). "
            "Accepts a {gate_key: bool} dict or a bare bool (legacy)."
        ),
    )

    @field_validator("gates_enabled", mode="before")
    @classmethod
    def check_gates(cls, v: object) -> dict[str, bool] | None:
        if v is None:
            return None
        return normalize_gates(v)


class PipelineRunResponse(BaseModel):
    """Response after starting a pipeline run."""
    thread_id: str
    source_id: str
    status: str


class PipelineStatusResponse(BaseModel):
    """Response for pipeline status check."""
    thread_id: str
    source_id: str
    status: str
    current_node: str | None
    error: str | None
    gates_enabled: dict[str, bool]
    # Summary counts for progress indication
    entity_count: int = 0
    chunk_count: int = 0
    draft_count: int = 0
    objects_written: int = 0
    # Non-fatal persistence failures (bundle store / source file). Empty
    # when everything wrote cleanly. UI surfaces these as a warning so
    # the analyst knows the bundle landed in Neo4j but a side write
    # (source file attachment, bundle_store row) didn't.
    persistence_errors: list[str] = []
    # Structured corrections + warnings from validate_bundle (Stage 6b).
    # Each entry: {rule, severity, message, ...}. severity ∈
    # {auto_fix, repaired, warn, hard_fail}. Empty when everything
    # validated cleanly without any fixes.
    bundle_corrections: list[dict] = []

    @field_validator("gates_enabled", mode="before")
    @classmethod
    def coerce_gates(cls, v: object) -> dict[str, bool]:
        return normalize_gates(v)


# =============================================================================
# Bundle Explorer schemas
# =============================================================================

class BundleRenameRequest(BaseModel):
    """Request body for renaming a completed bundle.

    Title is stripped, must be 1-512 chars after strip, and must not
    contain control characters (which render as weird glyphs in the
    Explorer card and can break log parsing).
    """
    title: str = Field(
        ...,
        min_length=1,
        max_length=512,
        description="New display title for the bundle",
    )

    @field_validator("title")
    @classmethod
    def _strip_and_check(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("title must not be empty or whitespace-only")
        # Reject ASCII control chars (0x00-0x1F, 0x7F) except none are
        # meaningful in a title. Keeps logs clean and prevents terminal
        # escape injection via log viewers.
        if any(ord(c) < 0x20 or ord(c) == 0x7F for c in v):
            raise ValueError("title must not contain control characters")
        if len(v) > 512:
            raise ValueError("title must be at most 512 characters after stripping")
        return v


class DenylistTermsModel(BaseModel):
    """Concrete things a denylist promotion blocks deterministically.

    `values` are literal entity values (case-insensitive exact match, removed
    at Gate 0). `technique_ids` are ATT&CK T-IDs dropped from technique picks.
    `entity_types` optionally scopes the `values` matches to those entity types
    (empty => any type), so e.g. denylisting "google" as a noisy org won't drop
    a google.com domain. The analyst confirms/edits these at promote time
    (pre-filled from the pattern's evidence/applies_to). All optional; an empty
    values+technique_ids pair enforces nothing (the promotion is then advisory).
    """
    values: list[str] = Field(default_factory=list, max_length=200)
    technique_ids: list[str] = Field(default_factory=list, max_length=200)
    entity_types: list[str] = Field(default_factory=list, max_length=200)

    @model_validator(mode="after")
    def _normalize(self) -> "DenylistTermsModel":
        # Single source of truth: delegate cleaning to the service's
        # normalize_denylist_terms so the API and the persisted/enforced form
        # apply identical rules (values trimmed + case-insensitive deduped,
        # control-char/over-long/empty dropped, technique_ids upper+shape-
        # validated, entity_types lowercased). Malformed terms are dropped, not
        # rejected — a control char never reaches logs/storage either way.
        # Lazy import avoids any schema<->service import cycle at module load.
        from app.services.feedback_patterns import normalize_denylist_terms
        cleaned = normalize_denylist_terms({
            "values": self.values,
            "technique_ids": self.technique_ids,
            "entity_types": self.entity_types,
        })
        self.values = cleaned["values"]
        self.technique_ids = cleaned["technique_ids"]
        self.entity_types = cleaned["entity_types"]
        return self


class FeedbackPatternPromoteRequest(BaseModel):
    """Promote a feedback pattern to a permanent guardrail.

    'prompt'   -> status promoted_to_prompt   (kept in every relevant prompt)
    'denylist' -> status promoted_to_denylist (a deterministic guardrail)
    `by` records who promoted it (audit). Control chars rejected like titles.
    `denylist_terms` carries the analyst-confirmed values/T-IDs to block;
    ignored for the 'prompt' action.
    """
    action: Literal["prompt", "denylist"]
    by: str = Field(..., min_length=1, max_length=128, description="Analyst who promoted (email/username)")
    denylist_terms: DenylistTermsModel | None = None

    @field_validator("by")
    @classmethod
    def _clean_by(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("by must not be empty or whitespace-only")
        if any(ord(c) < 0x20 or ord(c) == 0x7F for c in v):
            raise ValueError("by must not contain control characters")
        return v


class FeedbackPatternEditRequest(BaseModel):
    """Edit a feedback pattern's text / category / structured keys.

    All fields optional — only the provided ones are changed. Editing the
    pattern text or applies_to triggers a re-embed in the service layer.
    """
    pattern: str | None = Field(None, min_length=10, max_length=500)
    category: str | None = Field(None, min_length=1, max_length=64)
    applies_to: dict | None = None
    concepts: list[str] | None = None

    @field_validator("pattern")
    @classmethod
    def _clean_pattern(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        if any(ord(c) < 0x20 or ord(c) == 0x7F for c in v if c not in "\n\t"):
            raise ValueError("pattern must not contain control characters")
        return v


# =============================================================================
# Error schemas
# =============================================================================

# =============================================================================
# AI gate reviewer
# =============================================================================


class ReviewerBriefUpdate(BaseModel):
    """Analyst's correction to the reviewer's opening read of the report.

    The brief is turn one of the reviewer's transcript, so a correction here
    propagates to every downstream gate rather than having to be repeated at
    each one. That is the whole reason it is editable.
    """
    summary: str = Field(..., min_length=1, max_length=4000)
    actors: list[str] = Field(default_factory=list, max_length=50)
    attack_chain: list[str] = Field(default_factory=list, max_length=100)
    thin_areas: list[str] = Field(default_factory=list, max_length=50)
    notes_for_later_gates: str = Field(default="", max_length=4000)

    @field_validator("summary", "notes_for_later_gates")
    @classmethod
    def no_control_chars(cls, v: str) -> str:
        # Same log-injection guard as BundleRenameRequest.
        if any(ord(c) < 32 and c not in "\n\r\t" for c in v):
            raise ValueError("control characters are not allowed")
        return v.strip()


class ReviewerRecommendationResponse(BaseModel):
    """One reviewer turn, as the assist UI reads it."""
    id: uuid.UUID
    source_id: uuid.UUID
    gate_key: str
    pass_number: int
    model: str
    status: str
    payload: dict[str, Any] = Field(default_factory=dict)
    agent_notes: str = ""
    error: str | None = None
    created_at: datetime
    outcome: dict[str, Any] | None = None
    outcome_at: datetime | None = None

    model_config = {"from_attributes": True}


class ReviewerAgreementResponse(BaseModel):
    """Per-gate agreement across every reviewed source.

    Loosely typed on purpose. The per-gate dicts carry nested tallies keyed by
    confidence tier and by item kind, and the kinds come from the differs — a
    new gate adds new ones. Pinning that shape here would mean editing two
    files to add a gate and would let a mismatch fail the response rather than
    show the new numbers.

    Note what is absent: there is no overall agreement rate. Four gates with
    different costs of error do not average into a meaningful figure, and no
    "ready for autopilot" verdict is emitted — see
    app.services.reviewer.agreement.
    """
    sources_reviewed: int
    gates: list[dict[str, Any]] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Standard error response."""
    detail: str
    error_code: str | None = None
