"""Stable identity for a previewed relationship.

`normalize` shows the analyst a `relationship_preview`; `serialize_stix` later
builds the real SROs from scratch with fresh STIX UUIDs. Nothing carries an id
across that boundary, so a Gate 2 decision ("remove this edge") needs a key
both sides can compute independently.

The key is the human-readable tuple the preview already carries and the
serializer can reconstruct: source name, relationship type, target name, plus
both STIX types to disambiguate the `uses` collisions (procedure->tool vs
procedure->malware vs intrusion-set->procedure all share the verb).

Why not the old positional `relp_N` counter: it renumbered on every
`normalize` re-run, so a recorded removal silently re-mapped to a *different*
relationship the next time through the loop. Content-derived ids are stable
across re-derivation, which is what makes an analyst decision durable.
"""

from __future__ import annotations

import hashlib

RelationshipKey = tuple[str, str, str, str, str]


def _norm(value: object) -> str:
    return (str(value) if value is not None else "").strip().lower()


def relationship_key(
    source_name: object,
    relationship_type: object,
    target_name: object,
    source_type: object = "",
    target_type: object = "",
) -> RelationshipKey:
    """Build the stable key from its five parts."""
    return (
        _norm(source_name),
        _norm(relationship_type),
        _norm(target_name),
        _norm(source_type),
        _norm(target_type),
    )


def preview_relationship_key(preview: dict) -> RelationshipKey:
    """Key a `relationship_preview` entry."""
    return relationship_key(
        preview.get("source_name"),
        preview.get("relationship_type"),
        preview.get("target_name"),
        preview.get("source_type"),
        preview.get("target_type"),
    )


def preview_relationship_id(preview: dict) -> str:
    """Content-derived, re-derivation-stable id for a preview entry."""
    joined = "|".join(preview_relationship_key(preview))
    digest = hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]
    return f"relp_{digest}"
