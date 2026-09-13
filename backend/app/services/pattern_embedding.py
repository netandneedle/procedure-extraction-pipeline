"""Shared SecureBERT embedding handle for the feedback flywheel.

Reuses the SAME model as ``technique_retriever.EmbeddingRetriever``
(``settings.embedding_model`` — cisco-ai/SecureBERT2.0-biencoder) but as a
thin process-singleton. We deliberately do NOT import ``EmbeddingRetriever``
itself: it carries technique-index state and a per-instance cache that's
irrelevant here. This module just needs the generic encode handle.

Everything is BEST-EFFORT. If sentence-transformers isn't installed or the
model fails to load/encode, the functions return ``None`` / ``0.0`` and the
callers (persist-time embedding, retrieval-time ranking, backfill) degrade to
lexical-only behavior. Embedding a feedback pattern is prompt guidance, not
correctness — it must never block a pipeline run.

Also hosts the per-source representation + anchor extraction used by the
relevance retriever (kept here, free of any ORM import, so both persist-time
and retrieval-time code share one place).
"""

from __future__ import annotations

import logging
from typing import Mapping

import numpy as np

from app.config import settings
from app.services.technique_retriever import _extract_anchor_keywords
from app.services.vendor_technique_mapping import extract_vendor_technique_ids

logger = logging.getLogger(__name__)

_MODEL = None  # lazy process-singleton
_MODEL_LOAD_FAILED = False

# Bound the representation we embed so encode stays fast and within the model's
# attention zone regardless of source length. SecureBERT (ModernBERT-based)
# tolerates long inputs, but the relevance signal saturates well before this.
_MAX_REPR_CHARS = 4000
_MAX_CHUNKS_IN_REPR = 40
_MAX_ENTITIES_IN_REPR = 60


# =============================================================================
# Model handle (lazy, best-effort)
# =============================================================================

def _get_model():
    """Lazy-load the SentenceTransformer once per process.

    Returns ``None`` (logging once) if sentence-transformers isn't installed
    or the model fails to load. Callers treat ``None`` as "skip embedding".
    """
    global _MODEL, _MODEL_LOAD_FAILED
    if _MODEL is not None:
        return _MODEL
    if _MODEL_LOAD_FAILED:
        return None
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.warning(
            "pattern_embedding: sentence-transformers not installed; "
            "feedback retrieval falls back to lexical-only scoring",
        )
        _MODEL_LOAD_FAILED = True
        return None
    try:
        logger.info(
            "pattern_embedding: loading model %s (first call only)",
            settings.embedding_model,
        )
        _MODEL = SentenceTransformer(settings.embedding_model)
    except Exception as e:  # noqa: BLE001 — best-effort, never block the run
        logger.warning(
            "pattern_embedding: model load failed (%s); lexical-only fallback", e,
        )
        _MODEL_LOAD_FAILED = True
        return None
    return _MODEL


def embed_text(text: str) -> list[float] | None:
    """Embed one string into an L2-normalized float list. ``None`` on empty/failure."""
    if not text or not text.strip():
        return None
    model = _get_model()
    if model is None:
        return None
    try:
        vec = model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False,
        )[0]
        return np.asarray(vec, dtype=np.float32).tolist()
    except Exception as e:  # noqa: BLE001
        logger.warning("pattern_embedding.embed_text failed: %s", e)
        return None


def embed_texts(texts: list[str]) -> np.ndarray | None:
    """Batch-embed (used by the backfill script). Returns an (n, d) float32
    array of L2-normalized rows, or ``None`` on failure."""
    if not texts:
        return None
    model = _get_model()
    if model is None:
        return None
    try:
        arr = model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False,
        )
        return np.asarray(arr, dtype=np.float32)
    except Exception as e:  # noqa: BLE001
        logger.warning("pattern_embedding.embed_texts failed: %s", e)
        return None


def cosine(a, b) -> float:
    """Cosine similarity of two already-L2-normalized vectors (= dot product).

    Tolerant: returns 0.0 on ``None`` / empty / mismatched-length input.
    """
    if not a or not b or len(a) != len(b):
        return 0.0
    try:
        return float(
            np.dot(np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)),
        )
    except Exception:  # noqa: BLE001
        return 0.0


# =============================================================================
# Per-source representation + anchors (used by the relevance retriever)
# =============================================================================

def _entities_for_repr(state: Mapping) -> list[dict]:
    """Prefer analyst-validated entities; fall back to raw extracted ones."""
    validated = state.get("validated_entities") or []
    if validated:
        return validated
    return state.get("entities") or []


def _technique_ids_in_play(state: Mapping) -> set[str]:
    """All technique IDs the run has touched.

    Four sources: confirmed mappings, the review lane, the propose step's raw
    proposals, and the ATT&CK table the vendor printed in the report itself.

    That last one is what makes this usable EARLY. The first three are all
    populated by `extract_techniques`, so before that node runs this returned
    the empty set — and the +0.5 shared-technique boost in `_score_pattern`,
    the largest term in the formula, could never fire at `extract_entities` or
    `chunk_behaviors`. `classify_sections` runs before both, so a report with a
    mapping table has named its techniques by then.
    """
    tids: set[str] = set()
    for bucket in ("technique_mappings", "technique_mappings_for_review"):
        mapping = state.get(bucket) or {}
        if isinstance(mapping, Mapping):
            for picks in mapping.values():
                for p in picks or []:
                    tid = (p or {}).get("technique_id")
                    if tid:
                        tids.add(tid)
    proposals = state.get("proposals_by_chunk") or {}
    if isinstance(proposals, Mapping):
        for prop in proposals.values():
            for tid in (prop or {}).get("proposed_techniques", []) or []:
                if tid:
                    tids.add(tid)
    # The vendor's own ATT&CK mapping table, via the section classifier.
    # Scoped to `technique_reference` sections, so this is the report's
    # considered list rather than every T-ID mentioned in passing.
    vendor_tids, _ = extract_vendor_technique_ids(
        state.get("classified_sections") or []
    )
    tids |= vendor_tids
    return tids


def _tactics_in_play(state: Mapping) -> set[str]:
    tactics: set[str] = set()
    for bucket in ("technique_mappings", "technique_mappings_for_review"):
        mapping = state.get(bucket) or {}
        if isinstance(mapping, Mapping):
            for picks in mapping.values():
                for p in picks or []:
                    t = (p or {}).get("tactic")
                    if t:
                        tactics.add(t)
    proposals = state.get("proposals_by_chunk") or {}
    if isinstance(proposals, Mapping):
        for prop in proposals.values():
            for t in (prop or {}).get("tactics", []) or []:
                if t:
                    tactics.add(t)
    return tactics


def build_source_representation(state: Mapping) -> str:
    """Build one text blob describing what THIS source is about, for embedding.

    Composed (when available) from: title + key metadata (actor/campaign/
    malware) + the behavioral chunk narratives + entity ``type:value`` pairs +
    the technique IDs in play. Falls back to a head slice of ``parsed_text``
    when chunks aren't produced yet (entity/chunk stages). Bounded to
    ``_MAX_REPR_CHARS`` so encode stays cheap.
    """
    parts: list[str] = []

    title = state.get("title") or ""
    meta = state.get("metadata") or {}
    if not title and isinstance(meta, Mapping):
        title = meta.get("title") or ""
    if title:
        parts.append(str(title))

    if isinstance(meta, Mapping):
        for key in ("threat_actor", "campaign", "malware_family"):
            val = meta.get(key)
            if val:
                parts.append(str(val))

    chunks = state.get("chunks") or []
    chunk_texts = [
        (c or {}).get("text", "") for c in chunks[:_MAX_CHUNKS_IN_REPR]
    ]
    chunk_texts = [t for t in chunk_texts if t]
    if chunk_texts:
        parts.append(" ".join(chunk_texts))
    else:
        # No chunks yet (entity/chunk stages) — anchor on the source head.
        parsed = state.get("parsed_text") or ""
        if parsed:
            parts.append(parsed[:_MAX_REPR_CHARS])

    entities = _entities_for_repr(state)
    if entities:
        ent_bits = []
        for e in entities[:_MAX_ENTITIES_IN_REPR]:
            val = (e or {}).get("value", "")
            etype = (e or {}).get("entity_type", "")
            if val:
                ent_bits.append(f"{etype}:{val}" if etype else val)
        if ent_bits:
            parts.append(" ".join(ent_bits))

    tids = _technique_ids_in_play(state)
    if tids:
        parts.append(" ".join(sorted(tids)))

    repr_text = "\n".join(p for p in parts if p)
    return repr_text[:_MAX_REPR_CHARS]


def extract_source_anchors(state: Mapping) -> dict:
    """Structured + lexical anchors for the hybrid retrieval boost.

    Returns ``{keywords, technique_ids, entity_types, tactics}`` (all sets):
      - keywords: CVE/tool anchors found in the source representation
        (reuses technique_retriever's ``_extract_anchor_keywords``).
      - technique_ids: mappings + proposals + the vendor's own ATT&CK table.
      - tactics: pulled from mappings.
      - entity_types: the EntityType values present on the source.

    Not every anchor can exist at every stage, and one of them never can:

      - ``entity_types`` is empty at ``extract_entities`` and always will be.
        That node's job is to produce the entities, so nothing can name their
        types beforehand. This is a property of the pipeline, not a gap to
        fill — do not synthesise a guess here to make the term look alive.
      - ``tactics`` is empty until ``extract_techniques`` writes its mappings.
      - ``technique_ids`` used to share that limitation; it no longer does,
        because the vendor mapping table arrives with ``classify_sections``.
        A report without such a table still yields nothing, which is correct.
    """
    repr_text = build_source_representation(state)
    keywords = _extract_anchor_keywords(repr_text) if repr_text else set()

    entity_types: set[str] = set()
    for e in _entities_for_repr(state):
        etype = (e or {}).get("entity_type")
        if etype:
            entity_types.add(etype)

    return {
        "keywords": keywords,
        "technique_ids": _technique_ids_in_play(state),
        "entity_types": entity_types,
        "tactics": _tactics_in_play(state),
    }
