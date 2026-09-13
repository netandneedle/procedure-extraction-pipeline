"""Technique candidate retriever — provider abstraction.

Retrievers rank techniques by relevance to a set of chunks and return
the top-K candidates per chunk (union across all chunks), plus a
formatted reference text for the LLM prompt.

Two implementations:
  TokenOverlapRetriever — Jaccard token overlap + CVE/tool anchors
                          (zero infrastructure)
  EmbeddingRetriever    — semantic retrieval via sentence-transformers
                          + cisco-ai/SecureBERT2.0-biencoder

Picked at runtime via settings.technique_retriever ("token_overlap" |
"embedding"). Default: "embedding"; "token_overlap" is the opt-out for
deployments that cannot carry the model download.

Both retrievers honor the same retrieval contract (top-K per chunk,
union across chunks, parent/sub expansion, ALWAYS_INCLUDE floor) plus
the same anchor-keyword + name + ID precision boosts. The only
difference is how they score base relevance: Jaccard token overlap vs
cosine similarity over SecureBERT 2.0 embeddings.
"""

from __future__ import annotations

import logging
import re
from typing import Protocol

import numpy as np

from app.config import settings
from app.utils.text import token_overlap

logger = logging.getLogger(__name__)


# =============================================================================
# Anchor keyword tables (shared by both retrievers)
# =============================================================================

# CVE pattern, common LOLBins, and offensive tool names for keyword anchoring
_CVE_PATTERN = re.compile(r"CVE-\d{4}-\d{4,}")
_KNOWN_TOOLS = {
    # LOLBins
    "certutil", "mshta", "rundll32", "regsvr32", "wmic", "cscript", "wscript",
    "bitsadmin", "powershell", "cmd", "schtasks", "at", "sc", "net", "reg",
    "msiexec", "installutil", "cmstp",
    # Offensive tools
    "mimikatz", "cobalt strike", "beacon", "meterpreter", "psexec",
    "bloodhound", "sharphound", "rubeus", "impacket", "lazagne",
    "procdump", "ntdsutil", "secretsdump",
    # Shells / interpreters
    "bash", "python", "perl", "vbscript", "javascript", "jscript",
    "java", "msbuild",
    # Commercial AI / LLM products. Anchors v19+ techniques like T1682
    # (Query Public AI) and T1683 (Generate Content) when chunks reference
    # specific products by name. Substring matching, so "gpt-4 was used"
    # extracts both "gpt-4" and would also match a "gpt" entry — kept
    # specific to avoid false positives on "encrypted" etc.
    "chatgpt", "gpt-3", "gpt-4", "gpt-5", "claude", "gemini",
    "copilot", "llama", "mistral", "deepseek", "grok",
}

# High-frequency techniques that should always be in the candidate pool
# when any chunk exists (these are commonly co-occurring and easy to miss
# with pure token overlap because their descriptions are generic)
_ALWAYS_INCLUDE = {
    "T1059", "T1059.001", "T1059.003", "T1059.004", "T1059.005",
    "T1059.006", "T1059.007",  # Command and Scripting Interpreter family
    "T1105",  # Ingress Tool Transfer
    "T1071.001",  # Web Protocols
    "T1082",  # System Information Discovery
    "T1083",  # File and Directory Discovery
    "T1057",  # Process Discovery
    "T1027",  # Obfuscated Files or Information
    "T1070",  # Indicator Removal
}


def _extract_anchor_keywords(chunk_text: str) -> set[str]:
    """Extract CVEs, tool names, and command names from chunk text.

    These are used to boost techniques whose descriptions mention
    the same terms, improving pre-filter recall for specific behaviors.
    """
    text_lower = chunk_text.lower()
    keywords: set[str] = set()

    # CVE IDs
    for match in _CVE_PATTERN.finditer(chunk_text):
        keywords.add(match.group().lower())

    # Known tool names (check as substrings in the chunk)
    for tool in _KNOWN_TOOLS:
        if tool in text_lower:
            keywords.add(tool)

    return keywords


# =============================================================================
# Shared scoring + finalization helpers (used by both retrievers)
# =============================================================================


def _boost_score(
    tid: str,
    entry: dict,
    chunk_text_lower: str,
    anchor_keywords: set[str],
    boost_scale: float = 1.0,
) -> float:
    """Return additive precision boost for a (technique, chunk) pair.

    Three rules, applied identically by both retrievers so a swap to
    embedding mode preserves the precision anchors that protect against
    semantic drift on tool names and CVE IDs:

      +0.3 if any anchor keyword (CVE/tool) appears in the technique
           name+description (capped at one boost per technique)
      +0.5 if the technique name appears verbatim in the chunk text
      +0.5 if the technique ID (e.g. "T1059.001") appears in the chunk

    boost_scale multiplies the final boost. Default 1.0 preserves
    historical Jaccard tuning. EmbeddingRetriever may want a smaller
    scale (cosine sits in a tighter [~0.2, ~0.7] range, so the same
    +0.5 magnitude can dominate the cosine signal). boost_scale=0.0
    disables boosts entirely — useful for isolating retriever-only
    behavior in A/B tests.
    """
    boost = 0.0
    name = entry.get("name", "")
    desc = entry.get("description", "")

    if anchor_keywords:
        score_text_lower = f"{name} {desc}".lower()
        for kw in anchor_keywords:
            if kw in score_text_lower:
                boost += 0.3
                break  # one anchor boost per technique is enough

    if name and name.lower() in chunk_text_lower:
        boost += 0.5

    if tid.lower() in chunk_text_lower:
        boost += 0.5

    return boost * boost_scale


def _finalize_candidates(
    candidate_tids: set[str],
    technique_lookup: dict[str, dict],
    chunks_count: int,
    top_k: int,
    retriever_label: str,
) -> tuple[dict[str, dict], str]:
    """Expand parent/sub pairs, build filtered_lookup, format reference_text.

    Both retrievers feed the same downstream contract: prompt-ready text
    listing candidate techniques + a filtered lookup keyed by T-number.
    """
    # Expand parent/sub-technique pairs AFTER all chunks are processed:
    # if a sub-technique made the cut, include the parent, and vice versa.
    parent_sub_additions: set[str] = set()
    for tid in list(candidate_tids):
        if "." in tid:
            parent = tid.split(".")[0]
            if parent in technique_lookup and parent not in candidate_tids:
                parent_sub_additions.add(parent)
        else:
            # Include sub-techniques of selected parents
            for other_tid in technique_lookup:
                if other_tid.startswith(tid + ".") and other_tid not in candidate_tids:
                    parent_sub_additions.add(other_tid)
    candidate_tids |= parent_sub_additions

    # Build filtered lookup and reference text
    filtered_lookup: dict[str, dict] = {}
    lines: list[str] = []

    for tid in sorted(candidate_tids):
        entry = technique_lookup.get(tid)
        if not entry:
            continue
        filtered_lookup[tid] = entry

        tactics = entry.get("tactics", [])
        tactic_str = ", ".join(tactics) if tactics else "n/a"
        # Descriptions kept in technique_lookup for local scoring (Rule 3
        # overlap modulation, sub-technique promotion) but not sent to LLM.
        lines.append(f"{tid} | {entry['name']} | {tactic_str}")

    reference_text = (
        "\n\nATT&CK TECHNIQUE CANDIDATES (pre-filtered, "
        f"{len(filtered_lookup)} of {len(technique_lookup)} total):\n"
        "These are the most relevant techniques for the chunks below. "
        "Use ONLY technique IDs from this list. Use your ATT&CK knowledge "
        "of what each technique represents to map chunk behavior. If a "
        "behavior doesn't match any listed technique, use the closest "
        "parent technique with lower confidence and note the gap in "
        "rationale.\n"
        "Format: technique_id | name | tactics\n"
        + "\n".join(lines)
    )

    logger.info(
        "prefilter (%s): selected %d candidate techniques from %d total "
        "(%d chunks, top_k=%d, %d always-include, %d parent/sub expansions)",
        retriever_label,
        len(filtered_lookup), len(technique_lookup), chunks_count, top_k,
        len(_ALWAYS_INCLUDE & set(technique_lookup.keys())),
        len(parent_sub_additions),
    )

    return filtered_lookup, reference_text


# =============================================================================
# Provider Protocol
# =============================================================================


class TechniqueRetriever(Protocol):
    """Interface for technique candidate retrieval.

    Implementations rank techniques by relevance to chunks and return
    the top-K candidates per chunk (union across all chunks), plus a
    formatted reference text for the LLM prompt.

    Returns:
        (filtered_lookup, reference_text) where:
          filtered_lookup: subset of technique_lookup keyed by tid
          reference_text: prompt-ready string describing candidates
    """

    def find_candidates(
        self,
        chunks: list[dict],
        technique_lookup: dict[str, dict],
        top_k: int = 30,
    ) -> tuple[dict[str, dict], str]:
        ...


# =============================================================================
# TokenOverlapRetriever — Jaccard + CVE/tool anchors
# =============================================================================


class TokenOverlapRetriever:
    """Deterministic pre-filter using Jaccard token overlap with CVE/tool
    anchor boosts and parent/sub-technique expansion. Zero infrastructure;
    runs in-process with no model load. The opt-out alternative to the
    default EmbeddingRetriever.

    boost_scale (default 1.0) multiplies the precision boosts (+0.3
    anchor / +0.5 name / +0.5 ID). Lower values down-weight precision
    anchors relative to base token overlap — useful for A/B isolation.
    """

    def __init__(self, boost_scale: float = 1.0) -> None:
        self._boost_scale = boost_scale

    def find_candidates(
        self,
        chunks: list[dict],
        technique_lookup: dict[str, dict],
        top_k: int = 30,
    ) -> tuple[dict[str, dict], str]:
        if not technique_lookup:
            return {}, ""

        # Collect candidate T-numbers across all chunks (union)
        candidate_tids: set[str] = set()

        # Always include high-frequency techniques
        for tid in _ALWAYS_INCLUDE:
            if tid in technique_lookup:
                candidate_tids.add(tid)

        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            if not chunk_text:
                continue

            anchor_keywords = _extract_anchor_keywords(chunk_text)
            chunk_text_lower = chunk_text.lower()

            # Score every technique against this chunk
            scores: list[tuple[str, float]] = []

            for tid, entry in technique_lookup.items():
                desc = entry.get("description", "")
                name = entry.get("name", "")
                # Base score: token overlap with name + description
                base = token_overlap(chunk_text, f"{name} {desc}")
                boost = _boost_score(
                    tid, entry, chunk_text_lower, anchor_keywords,
                    boost_scale=self._boost_scale,
                )
                scores.append((tid, base + boost))

            # Take top-K by score; secondary sort by technique_id for
            # deterministic tie-breaking when scores match (otherwise dict
            # iteration order leaks into top-K membership at the boundary).
            scores.sort(key=lambda x: (-x[1], x[0]))
            for tid, _score in scores[:top_k]:
                candidate_tids.add(tid)

        return _finalize_candidates(
            candidate_tids, technique_lookup, len(chunks), top_k,
            "token_overlap",
        )


# =============================================================================
# EmbeddingRetriever — semantic retrieval via SecureBERT 2.0 biencoder
# =============================================================================


class EmbeddingRetriever:
    """Semantic retriever using sentence-transformers + cisco-ai/SecureBERT2.0-biencoder.

    Encodes the technique catalogue once per process (cached on this
    instance), encodes all chunks per call, computes cosine similarity
    via a single matmul on L2-normalized vectors, then applies the same
    anchor/name/ID boosts as TokenOverlapRetriever for precision parity.

    Retrieval semantics are identical to TokenOverlapRetriever:
        - top-K per chunk
        - union across chunks
        - parent/sub-technique expansion
        - ALWAYS_INCLUDE floor

    Only the base score differs: cosine similarity vs Jaccard. Boost
    magnitudes (+0.3, +0.5, +0.5) are tuned for token_overlap's [0,1]
    range; cosine sits in roughly [-1, 1] but practically [0.2, 0.7] for
    decent matches. Boosts still serve their purpose because top-K only
    cares about ordering. An A/B found that lowering boost_scale regressed
    recall — the boosts at full strength pull the right candidates into
    top-K — so 1.0 is the tuned value.

    sentence_transformers is imported lazily inside _ensure_model so
    token_overlap-only deployments don't have to install torch.
    """

    def __init__(
        self,
        model_name: str | None = None,
        boost_scale: float = 1.0,
    ) -> None:
        self._model_name = model_name or settings.embedding_model
        self._boost_scale = boost_scale
        self._model = None  # lazy
        # Cache key: (id(catalogue), len(catalogue)). The id() guards
        # against in-place mutation; len() guards against the rare case
        # where two distinct dicts get assigned the same id by Python's
        # allocator. Catalogue is module-cached by _load_technique_catalogue
        # so in practice we embed once per process.
        self._index_cache_key: tuple[int, int] | None = None
        self._technique_tids: list[str] = []
        self._technique_embeddings: np.ndarray | None = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "EmbeddingRetriever requires sentence-transformers. "
                "Install with: pip install sentence-transformers"
            ) from e
        logger.info(
            "embedding_retriever: loading model %s (first call only)",
            self._model_name,
        )
        self._model = SentenceTransformer(self._model_name)

    def _ensure_technique_index(self, technique_lookup: dict[str, dict]) -> None:
        cache_key = (id(technique_lookup), len(technique_lookup))
        if cache_key == self._index_cache_key:
            return

        self._ensure_model()

        # Sort tids for stable matrix-row ordering (deterministic top-K
        # tie-breaking on identical scores).
        tids = sorted(technique_lookup.keys())
        texts = [
            f"{technique_lookup[t].get('name', '')}. "
            f"{technique_lookup[t].get('description', '')}"
            for t in tids
        ]

        logger.info(
            "embedding_retriever: encoding %d techniques", len(tids),
        )
        embeds = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self._technique_tids = tids
        self._technique_embeddings = np.asarray(embeds, dtype=np.float32)
        self._index_cache_key = cache_key

    def find_candidates(
        self,
        chunks: list[dict],
        technique_lookup: dict[str, dict],
        top_k: int = 30,
    ) -> tuple[dict[str, dict], str]:
        if not technique_lookup:
            return {}, ""

        self._ensure_technique_index(technique_lookup)
        assert self._technique_embeddings is not None  # set by _ensure_technique_index

        candidate_tids: set[str] = set()

        # Always include high-frequency techniques
        for tid in _ALWAYS_INCLUDE:
            if tid in technique_lookup:
                candidate_tids.add(tid)

        # Filter to chunks that have text; preserve original ordering for
        # row-to-chunk mapping.
        text_chunks = [c for c in chunks if c.get("text", "")]
        if text_chunks:
            chunk_texts = [c["text"] for c in text_chunks]
            chunk_embeds = self._model.encode(
                chunk_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            chunk_embeds = np.asarray(chunk_embeds, dtype=np.float32)
            # Cosine similarity = matmul of L2-normalized vectors.
            # Shape: (n_chunks, n_techniques)
            sims = chunk_embeds @ self._technique_embeddings.T

            for i, chunk in enumerate(text_chunks):
                chunk_text = chunk["text"]
                chunk_text_lower = chunk_text.lower()
                anchor_keywords = _extract_anchor_keywords(chunk_text)

                # Apply precision boosts on top of cosine. Boost magnitudes
                # are additive and shared with TokenOverlapRetriever, scaled
                # by self._boost_scale (lower for embedding compresses
                # boosts to better match cosine's tighter score range).
                row = sims[i].copy()
                for j, tid in enumerate(self._technique_tids):
                    boost = _boost_score(
                        tid, technique_lookup[tid],
                        chunk_text_lower, anchor_keywords,
                        boost_scale=self._boost_scale,
                    )
                    if boost:
                        row[j] += boost

                # Top-K with deterministic tie-breaking by tid (the tids
                # list is sorted, so a stable argsort on -row is enough).
                order = np.argsort(-row, kind="stable")[:top_k]
                for j in order:
                    candidate_tids.add(self._technique_tids[j])

        return _finalize_candidates(
            candidate_tids, technique_lookup, len(chunks), top_k,
            "embedding",
        )


# =============================================================================
# Factory
# =============================================================================


# Retriever instances, keyed by the settings that determine their identity.
# EmbeddingRetriever caches BOTH the SentenceTransformer handle and the encoded
# technique matrix on the instance, so the instance must be reused: a factory
# that hands out a FRESH instance on every call makes every extract_techniques
# run reload the model and re-encode all ~700 techniques. That is a ~2-minute
# CPU burn which blocks the whole API (the encode is synchronous and holds the
# event loop), and it repeats on every gate rewind — one run paid it
# three times.
#
# Keyed by (mode, model_name) rather than mode alone so a settings change
# yields a new instance rather than a stale one — which is also what lets the
# factory tests patch embedding_model and still get what they asked for.
_RETRIEVERS: dict[tuple[str, str | None], TechniqueRetriever] = {}


def reset_retriever_cache() -> None:
    """Drop memoized retrievers (frees the model handle + technique matrix)."""
    _RETRIEVERS.clear()


def get_retriever() -> TechniqueRetriever:
    """Return the configured retriever, memoized per (mode, model).

    Reads settings.technique_retriever; the default is "embedding"
    (app/config.py, chosen after an A/B against token_overlap).
    """
    mode = getattr(settings, "technique_retriever", "embedding")
    if mode not in ("token_overlap", "embedding"):
        raise ValueError(
            f"Unknown technique_retriever mode: {mode!r}. "
            f"Expected 'token_overlap' or 'embedding'."
        )

    model_name = getattr(settings, "embedding_model", None) if mode == "embedding" else None
    key = (mode, model_name)
    cached = _RETRIEVERS.get(key)
    if cached is not None:
        return cached

    retriever: TechniqueRetriever = (
        TokenOverlapRetriever() if mode == "token_overlap" else EmbeddingRetriever()
    )
    _RETRIEVERS[key] = retriever
    return retriever
