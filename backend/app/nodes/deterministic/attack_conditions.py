"""Attack Flow v2.0.0 attack-condition inference + SDO construction.

Mirror of attack_operators.py for the condition signal. Where operators
emerge from chunk DAG geometry (branch/converge counts), conditions
come from an LLM-emitted `precondition` field on chunks — the chunker
extracts source prose like "if domain-joined, kerberoasting; else NTLM
relay" into a `{description, pattern?, pattern_type?, on_true_ids,
on_false_ids}` dict at finalize time.

This module turns those preconditions into `attack-condition` SDOs and
provides the anchor mapping the serializer needs to route precedes
through them. Composition with operators is handled in
attack_operators.route_precedes_through_operators (which accepts
chunk_conditions and prefers the condition over the OR-branch operator
at any chunk that anchors both).

Design notes:
  * Stable STIX ID: sha256("condition|" + anchor_chunk_id + "|" +
    description.strip())[:12], wrapped as attack-condition--<hash>.
    Mutating the description changes the id (same as renaming a chunk
    changes its derived ids). This is intentional — the analyst's
    overridden description IS a content change.
  * `is_sequential=False` skips condition emission entirely — catalog
    sources don't get flow scaffolding.
  * Conditions with no resolved targets on either branch are skipped
    with a warning. _finalize_chunks already prunes these, so this is a
    defense-in-depth check for analyst-edited bundles.
"""
from __future__ import annotations

import hashlib
import logging
import uuid

from app.nodes.deterministic.extension_definitions import (
    ATTACK_FLOW_EXTENSION_ID,
    extension_declaration,
)

logger = logging.getLogger(__name__)


def extract_conditions(
    chunks: list[dict],
    is_sequential: bool,
) -> dict[str, dict]:
    """Pull precondition entries off chunks, keyed by anchor chunk_id.

    Returns a dict shaped:
        {
            "<anchor_chunk_id>": {
                "description": str,
                "pattern": str | None,
                "pattern_type": str | None,
                "on_true_ids":  [chunk_id, ...],
                "on_false_ids": [chunk_id, ...],
            }
        }

    `is_sequential=False` returns empty (matches the gate used for
    operators + precedes SROs in the serializer).

    Chunks with no precondition, or with both sides empty (which
    _finalize_chunks should already have pruned), are silently skipped.
    """
    if not is_sequential:
        return {}

    out: dict[str, dict] = {}
    for chunk in chunks:
        pre = chunk.get("precondition")
        if not isinstance(pre, dict):
            continue
        anchor = chunk.get("chunk_id")
        if not anchor:
            continue
        description = (pre.get("description") or "").strip()
        if not description:
            continue
        on_true_ids = list(pre.get("on_true_ids", []) or [])
        on_false_ids = list(pre.get("on_false_ids", []) or [])
        if not on_true_ids and not on_false_ids:
            continue
        out[anchor] = {
            "description": description,
            "pattern": pre.get("pattern"),
            "pattern_type": pre.get("pattern_type"),
            "on_true_ids": on_true_ids,
            "on_false_ids": on_false_ids,
        }

    if out:
        logger.info("extract_conditions: %d conditions extracted", len(out))
    return out


def build_attack_condition_sdos(
    chunk_conditions: dict[str, dict],
    chunk_to_procedure_stix_id: dict[str, str],
    source_identity_id: str,
    now_iso: str,
    spec_version: str,
) -> tuple[list[dict], dict[str, str]]:
    """Materialize attack-condition SDOs from extracted conditions.

    Returns:
        (sdos, anchor_to_stix_id) — sdos is the list of attack-condition
        objects ready to append to the bundle; anchor_to_stix_id maps
        the anchor chunk_id to the attack-condition--<uuid> STIX id.

    Conditions with no resolved on_true_refs AND no resolved on_false_refs
    after chunk-id → procedure-id resolution are SKIPPED (logged warning).
    Conditions with at least one resolved ref on either side are kept;
    the empty side stays empty in the SDO.
    """
    sdos: list[dict] = []
    anchor_to_stix_id: dict[str, str] = {}

    for anchor, meta in chunk_conditions.items():
        on_true_refs = [
            chunk_to_procedure_stix_id[i]
            for i in meta["on_true_ids"]
            if i in chunk_to_procedure_stix_id
        ]
        on_false_refs = [
            chunk_to_procedure_stix_id[i]
            for i in meta["on_false_ids"]
            if i in chunk_to_procedure_stix_id
        ]
        if not on_true_refs and not on_false_refs:
            logger.warning(
                "build_attack_condition_sdos: condition at %s has no "
                "resolved refs on either branch, skipping",
                anchor,
            )
            continue

        stix_id = f"attack-condition--{uuid.uuid4()}"
        anchor_to_stix_id[anchor] = stix_id
        obj: dict = {
            "type": "attack-condition",
            "spec_version": spec_version,
            "id": stix_id,
            "created": now_iso,
            "modified": now_iso,
            "created_by_ref": source_identity_id,
            "description": meta["description"],
            "on_true_refs": on_true_refs,
            "on_false_refs": on_false_refs,
            # Attack Flow 2.0.0 has ONE extension definition covering every
            # attack-* type; the serializer embeds it once per bundle.
            "extensions": extension_declaration(ATTACK_FLOW_EXTENSION_ID),
        }
        # Optional pattern + pattern_type pair, emitted only when both are
        # set. _finalize_chunks already coupled the two; defense-in-depth
        # check here for hand-edited bundles.
        if meta.get("pattern") and meta.get("pattern_type") in {"stix", "regex", "plain"}:
            obj["pattern"] = meta["pattern"]
            obj["pattern_type"] = meta["pattern_type"]
        sdos.append(obj)

    return sdos, anchor_to_stix_id


def _condition_stix_id_hash(anchor: str, description: str) -> str:
    """Stable 12-hex hash of (anchor_chunk_id, description).

    Currently not used by build_attack_condition_sdos (it allocates a
    fresh uuid per call so re-runs get a fresh STIX id). Reserved for
    a future enhancement where condition STIX ids need to be
    stable across re-runs for analyst-referenced state. Documented here
    so the path is obvious when needed.
    """
    h = hashlib.sha256()
    h.update(b"condition|")
    h.update(anchor.encode("utf-8"))
    h.update(b"|")
    h.update(description.encode("utf-8"))
    return h.hexdigest()[:12]
