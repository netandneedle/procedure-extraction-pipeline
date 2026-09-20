"""Pydantic output models for the AI gate reviewer.

Each gate reviewer returns a set of recommendations shaped like that gate's
submit schema, plus three fields the submit schema does not carry:

  confidence      — drives the confidence-gated "accept all" in the UI.
  rationale       — required. A recommendation without a reason is not
                    reviewable, and an unreviewable recommendation is just
                    an unattended edit wearing a costume.
  evidence_quote  — verbatim from the source, "" when the reviewer has none.
                    Run through app.services.grounding before the analyst
                    ever sees it; an unsupported quote is forced to "low".

Strict (`extra="forbid"`) for the same reason as app.nodes.llm.tool_models:
if the model invents a field we didn't ask for, we want to know.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Confidence = Literal["high", "medium", "low"]


class _StrictBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class _Recommendation(_StrictBase):
    """Fields every gate's recommendation carries."""

    confidence: Confidence = Field(
        description=(
            "high = the source states this plainly and you would stake the "
            "bundle on it. medium = well-supported inference. low = worth "
            "the analyst's eye but you are not sure."
        ),
    )
    rationale: str = Field(
        min_length=1,
        description="Why. One or two sentences, concrete, referencing the source.",
    )
    evidence_quote: str = Field(
        default="",
        description=(
            "Verbatim span from the report supporting this. Empty string if "
            "you have none — an empty quote is honest, an invented one is "
            "not, and invented quotes are detected and downgraded."
        ),
    )

    # Written by the grounding check after the model returns, never by the
    # model. Present so the assist UI can explain a forced downgrade.
    quote_source_support: float | None = None
    quote_unsupported: bool = False


class InitialRead(_StrictBase):
    """The reviewer's opening read of the report — turn one of the transcript.

    Visible to the analyst above Gate 0 and editable there. Because it is the
    first turn, correcting it propagates to every downstream gate, which makes
    it the cheapest place in the system to fix a misread.
    """

    summary: str = Field(
        min_length=1,
        description="What this report describes. 3-6 sentences, plain language.",
    )
    actors: list[str] = Field(
        default_factory=list,
        description="Named threat actors, groups, or campaigns. [] if none named.",
    )
    attack_chain: list[str] = Field(
        default_factory=list,
        description="The attack in order, one short phrase per step, as the report tells it.",
    )
    thin_areas: list[str] = Field(
        default_factory=list,
        description=(
            "Where the report is vague, second-hand, or asserts without "
            "evidence. This is where extraction is most likely to invent."
        ),
    )
    notes_for_later_gates: str = Field(
        default="",
        description="Anything you want to remember when reviewing the extraction.",
    )


class EntityRecommendation(_Recommendation):
    """One entity decision at Gate 0. Mirrors Gate0ReviewItem."""

    entity_id: str
    action: Literal["approve", "edit", "remove"]
    edited_value: str | None = None
    edited_type: str | None = None
    edited_role: str | None = None


class AddedEntityRecommendation(_Recommendation):
    """An entity the extractor missed. Mirrors AddedEntityItem."""

    value: str = Field(min_length=1, max_length=1024)
    entity_type: str
    organization_role: str | None = None
    location_role: str | None = None
    # STIX industry-sector-ov value, required when entity_type is
    # victim_sector. Folded into `value` by enforce_sector_vocabulary before
    # the recommendation is stored — the gate's addition channel carries the
    # sector in `value`, and a value off the vocabulary is silently dropped
    # from the bundle by the validator.
    sector: str | None = None


# A `<parameter name="...">` wrapper the model sometimes leaks into a field
# value — its own tool-call serialization showing through. Stripped rather
# than tolerated: the brief is shown to the analyst and is the text every
# later gate replays, so markup in it is noise in both places.
_PARAM_WRAPPER_OPEN = re.compile(r'^\s*<parameter\s+name="[^"]*"\s*>')
_PARAM_WRAPPER_CLOSE = re.compile(r"</parameter>\s*$")


def _strip_parameter_wrapper(text: str) -> str:
    return _PARAM_WRAPPER_CLOSE.sub("", _PARAM_WRAPPER_OPEN.sub("", text)).strip()


# Field names that belong INSIDE initial_read. Module-level rather than a
# class attribute because Pydantic claims leading-underscore class attributes
# as private model fields, which makes them unusable from a validator.
_INITIAL_READ_FIELDS = (
    "summary", "actors", "attack_chain", "thin_areas", "notes_for_later_gates",
)


class Gate0Recommendations(_StrictBase):
    """Full Gate 0 review output."""

    @model_validator(mode="before")
    @classmethod
    def _lift_flattened_initial_read(cls, data):
        """Accept the opening read flattened to the top level.

        Measured, not hypothetical: on both live runs the model emitted
        `initial_read` AND copies of its sub-fields at the top level. With
        extra='forbid' that fails validation, and the adapter's retry means
        EVERY reviewer call was costing two Opus requests — a silent 2x on
        the most expensive call in the pipeline.

        Lifting rather than dropping, because the flat form may one day
        arrive on its own; dropping would then discard the whole read while
        looking like a success. A nested value wins over a flat one of the
        same name — the nested shape is what the schema asked for, so if both
        are present it is the more deliberate answer.
        """
        if not isinstance(data, dict):
            return data
        nested = data.get("initial_read")
        stray = {k: data[k] for k in _INITIAL_READ_FIELDS if k in data}
        if not stray and not isinstance(nested, str):
            return data
        data = dict(data)
        if isinstance(nested, str):
            # `initial_read` carrying the summary PROSE, with the rest of the
            # read hoisted to the top level. Confirmed on a live run, not
            # inferred: the stored value was
            #
            #   '<parameter name="summary">Google Threat Intelligence Group
            #    reports Campaign 00.001, attributed to ...'
            #
            # — the model's own tool-call serialization leaking into the
            # field, with good prose after it. `summary` is then at neither
            # level, `initial_read.summary:missing`, and the whole call
            # retries: a second Opus generation on the most expensive request
            # in the pipeline.
            nested = {"summary": _strip_parameter_wrapper(nested)}
        merged = stray if not isinstance(nested, dict) else {**stray, **nested}
        data["initial_read"] = merged
        for key in stray:
            data.pop(key, None)
        return data

    initial_read: InitialRead | None = Field(
        default=None,
        description="Only on the first turn for a source; omit on later passes.",
    )
    entities: list[EntityRecommendation] = Field(default_factory=list)
    added_entities: list[AddedEntityRecommendation] = Field(default_factory=list)
    overall_notes: str = Field(
        default="",
        description="What you want the analyst to know before they review.",
    )


# =============================================================================
# Gate 1 — procedures + techniques
# =============================================================================

# Mirrors app.graph.state.Gate1RejectReason. Pinned by a contract test rather
# than imported, because a tool-schema enum has to be a literal list anyway
# and silent drift between the two is exactly what that test exists to catch.
RejectReason = Literal[
    "wrong_technique",
    "hallucinated",
    "too_vague",
    "wrong_technique_granularity",
    "bad_chunk_boundary",
]

# Why a draft should be dropped rather than re-extracted.
#
# `not_a_procedure` and `duplicate` used to live in RejectReason, and that was
# the whole problem. A reject routes back to extract_techniques
# (_compute_rejection_routing), and re-running technique extraction cannot fix
# a draft that is not a procedure or that duplicates another — the chunk text
# is unchanged, so the identical draft returns and is rejected again. Three
# unattended runs on one ransomware report all stopped here, each time on a reject whose own
# rationale argued for deletion ("its concrete instances are already captured
# by dft-2e5693ee and dft-79f130db").
#
# They are removal reasons, so they belong to the removal verb.
RemoveReason = Literal[
    "not_a_procedure",
    "duplicate",
    "other",
]


class DraftRecommendation(_Recommendation):
    """One procedure-draft decision at Gate 1.

    Edits are STRUCTURED rather than a free-form `analyst_edits` dict. The
    gate accepts arbitrary field writes there, and handing an LLM an
    unbounded write surface into the draft invites exactly the class of
    problem the entity-type enum was added to stop. Named fields also let the
    UI show the analyst precisely what would change.

    `remove_technique_ids` rather than a rewritten technique list, for the
    same reason: asking the model to re-emit every mapping risks it dropping
    `stix_id` or inventing a tactic. Removal is the operation that is
    actually wanted, and it cannot corrupt what it does not touch.
    """

    draft_id: str
    action: Literal["approve", "edit", "reject", "remove"] = Field(
        description=(
            "approve = ships as-is. edit = ships with your corrections. "
            "reject = send back for re-extraction (needs reject_reason). "
            "remove = drop this draft from the bundle entirely."
        ),
    )
    reject_reason: RejectReason | None = Field(
        default=None,
        description="Required when action=reject. Picks the re-run route.",
    )
    remove_reason: RemoveReason | None = Field(
        default=None,
        description=(
            "Why this draft should be dropped, with action=remove. Give one: "
            "a removal without a reason teaches the pattern learner nothing."
        ),
    )
    remove_technique_ids: list[str] = Field(
        default_factory=list,
        description=(
            "ATT&CK IDs to drop from this draft, with action=edit. Use this "
            "for a technique that does not serve the procedure's objective, "
            "rather than rejecting the whole draft over one bad mapping."
        ),
    )
    # No edited_name / edited_description on purpose. Gate1Review has no
    # editing surface for either — `analyst_edits` is initialized to null and
    # no control ever writes to it — so a recommendation about wording would
    # be unactionable, and an unactionable recommendation is worse than none:
    # it spends the analyst's attention and teaches them the panel wastes it.
    #
    # The review dimension survives via the reject path: `too_vague` covers a
    # description that asserts more than the report does. Revisit if a draft
    # editor is ever built.


class TechniquePromotionRecommendation(_Recommendation):
    """Promote a possible-bucket pick out of the review lane into the bundle."""

    chunk_id: str
    technique_id: str


class Gate1Recommendations(_StrictBase):
    """Full Gate 1 review output."""

    drafts: list[DraftRecommendation] = Field(default_factory=list)
    promotions: list[TechniquePromotionRecommendation] = Field(default_factory=list)
    overall_notes: str = Field(
        default="",
        description="What you want the analyst to know before they review.",
    )


# =============================================================================
# Gate 2 — bundle relationships
# =============================================================================

# Mirrors _VALID_RELATIONSHIP_TYPES in app.schemas.api; pinned by a contract
# test. A type outside this set is rejected by Gate2ReviewItem's validator,
# so proposing one would make the recommendation unapplicable.
RelationshipType = Literal[
    "uses", "targets", "precedes", "indicates",
    "has-observable", "attributed-to", "mitigates", "detects",
    "exploits", "component-of",
]


class RelationshipRecommendation(_Recommendation):
    """One relationship decision at Gate 2. Mirrors Gate2ReviewItem.

    All three actions are actionable in BundleReviewCanvas, which already
    emits approve/edit/remove and carries the three edited_* fields — unlike
    Gate 1, where a wording edit had no UI to land in.
    """

    rel_id: str = Field(
        description="The `id` from the relationship list you were shown.",
    )
    action: Literal["approve", "edit", "remove"] = Field(
        description=(
            "approve = correct as derived. edit = right relationship, wrong "
            "type or endpoint. remove = this edge should not exist."
        ),
    )
    edited_rel_type: RelationshipType | None = Field(
        default=None,
        description="Corrected relationship type, with action=edit.",
    )
    edited_source: str | None = Field(
        default=None,
        description="Corrected source name, with action=edit.",
    )
    edited_target: str | None = Field(
        default=None,
        description="Corrected target name, with action=edit.",
    )


class Gate2Recommendations(_StrictBase):
    """Full Gate 2 review output."""

    relationships: list[RelationshipRecommendation] = Field(default_factory=list)
    overall_notes: str = Field(
        default="",
        description="What you want the analyst to know before they review.",
    )


# =============================================================================
# Gate chunks — the decomposition itself
#
# Named for the Python identifier (`gate_chunks`), not the user-facing "Gate
# 1", for the same reason the node is: the positional numbering shifted when
# this gate was inserted and the identifiers deliberately did not follow.
# =============================================================================

# Mirrors app.graph.state.ChunkGateRejectReason. Pinned by a contract test,
# same as RejectReason above.
ChunkRerunReason = Literal[
    "missed_procedures",
    "over_chunked",
    "under_chunked",
    "bad_boundaries",
    "bad_descriptions",
    "bad_flow",
    "other",
]


class ChunkRecommendation(_Recommendation):
    """One chunk decision at the chunk gate. Mirrors ChunkDecisionItem.

    Edits are two NAMED fields rather than the gate's free-form `edits`
    dict, for the reason DraftRecommendation gives: an unbounded write
    surface into a chunk is the shape of problem the entity-type enum was
    added to stop, and named fields let the UI show the analyst exactly what
    would change.

    The gate's whitelist is wider than this (`context`,
    `behavioral_confidence`, `branch_point`, `convergence_point` are also
    editable). They are withheld on purpose: the confidence is the chunker's
    own estimate and a second opinion on it changes nothing downstream, and
    the branch/converge flags describe geometry that `edges` already
    expresses — offering both invites a payload where they disagree.
    `chain_root` / `chain_label` ARE offered: they are not geometry, they are
    the chunker's claim about which intrusion a chunk belongs to and where a
    chain starts, and a shared segment mis-rooted as a chain (an exploit kit
    made the sole entry point with the campaigns hanging off it) is exactly
    the defect a reviewer can see and an edge alone cannot fix.

    There is deliberately no `split`. The gate has no split primitive; the
    documented workaround is drop-then-add, which for a reviewer means
    proposing to delete evidence-bearing chunks and re-type them from
    scratch. `reject` with `under_chunked` is the honest way to say a chunk
    covers two procedures, and the prompt says so.
    """

    chunk_id: str
    action: Literal["approve", "edit", "drop", "merge"] = Field(
        description=(
            "approve = this chunk is a sound procedure. edit = keep it with "
            "your corrections. drop = it is not a procedure, or duplicates "
            "another. merge = absorb the chunks in merge_with into this one."
        ),
    )
    merge_with: list[str] = Field(
        default_factory=list,
        description=(
            "With action=merge: chunk_ids absorbed INTO this one. They are "
            "dropped and their flow edges rewired onto the survivor."
        ),
    )
    edited_text: str | None = Field(
        default=None,
        description="Corrected chunk narrative, with action=edit.",
    )
    edited_source_excerpt: str | None = Field(
        default=None,
        description="Corrected verbatim source excerpt, with action=edit.",
    )
    edited_chain_root: bool | None = Field(
        default=None,
        description=(
            "With action=edit: whether this chunk BEGINS a chain (an entry "
            "point such as a lure). False on a shared capability that "
            "several chains enter."
        ),
    )
    edited_chain_label: str | None = Field(
        default=None,
        description=(
            "With action=edit: the chain this chunk belongs to — a campaign, "
            "or the name of a shared capability."
        ),
    )


class AddedChunkRecommendation(_Recommendation):
    """A procedure the chunker missed. Mirrors AddedChunkItem.

    Lands with `source_span: None` and no flow edges — the gate cannot
    resolve an edge against a chunk whose id it has not allocated yet. So an
    added chunk arrives disconnected, and the prompt says so rather than
    letting the reviewer propose edges that would be silently dropped.
    """

    text: str = Field(min_length=1, max_length=4000)
    source_excerpt: str = Field(
        default="", max_length=4000,
        description="Verbatim span from the report this procedure comes from.",
    )


class ChunkEdgeRecommendation(_Recommendation):
    """One precedes-edge mutation on the chunk DAG. Mirrors ChunkEdgeMutation.

    `from_chunk_id` / `to_chunk_id` rather than the wire shape's `from`/`to`:
    `from` is a reserved word, and the two names are unambiguous to a model
    in a way that a bare `from` is not.
    """

    action: Literal["add", "remove"]
    from_chunk_id: str = Field(description="The chunk that happens first.")
    to_chunk_id: str = Field(description="The chunk that follows it.")


class ChunkRerunRecommendation(_Recommendation):
    """Send the whole decomposition back to be re-chunked.

    Kept because it is the only way to express the systemic findings — a
    source chunked at the wrong granularity throughout, or sequenced wrongly
    throughout. Those are not a list of per-chunk edits, and pretending
    otherwise would leave the reviewer no way to say the true thing.

    Expensive, and the prompt says so: a rerun discards every chunk and every
    edit made on this pass.
    """

    reason: ChunkRerunReason = Field(
        description="Why the whole pass was wrong. Routes the rerun's guidance.",
    )
    comments: str = Field(
        default="", max_length=2000,
        description="Specific guidance for the re-chunk. The more concrete, the better.",
    )


class GateChunksRecommendations(_StrictBase):
    """Full chunk-gate review output."""

    chunks: list[ChunkRecommendation] = Field(default_factory=list)
    added_chunks: list[AddedChunkRecommendation] = Field(default_factory=list)
    edges: list[ChunkEdgeRecommendation] = Field(default_factory=list)
    reject: ChunkRerunRecommendation | None = Field(
        default=None,
        description=(
            "Only when the decomposition is wrong as a whole. Discards "
            "everything, including your own per-chunk recommendations."
        ),
    )
    overall_notes: str = Field(
        default="",
        description="What you want the analyst to know before they review.",
    )
