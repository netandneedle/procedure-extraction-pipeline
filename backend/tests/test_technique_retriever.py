"""Unit tests for app.services.technique_retriever.

Covers:
  - TokenOverlapRetriever behavior parity after the helper-extraction
    refactor (no behavior change)
  - EmbeddingRetriever with a mock SentenceTransformer (real SecureBERT
    2.0 isn't downloaded — keeping tests fast and offline-safe)
  - Three explicit regressions for the self-critique concerns from
    that refactor:
      1. normalize_embeddings=True is actually passed to model.encode
      2. catalogue cache invalidates when a different lookup dict comes in
      3. end-to-end smoke: EmbeddingRetriever returns the expected shape
         (filtered_lookup keyed by tid, prompt-ready reference_text)
  - Factory dispatch via settings

    docker compose exec -T api python -m pytest tests/test_technique_retriever.py -v
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Provide a dummy API key so app.config import doesn't error.
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-dummy")

# Ensure backend is importable when running outside the api container.
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from app.services.technique_retriever import (  # noqa: E402
    EmbeddingRetriever,
    TokenOverlapRetriever,
    _ALWAYS_INCLUDE,
    _boost_score,
    _extract_anchor_keywords,
    _finalize_candidates,
    get_retriever,
)


# =============================================================================
# Fake sentence_transformers module
# =============================================================================
#
# sentence-transformers is a heavyweight optional dep — production installs
# that opt into embedding mode pull torch (~2 GB). We don't want CI/test
# runs to require that, so we register a fake module in sys.modules and
# point each test's mock model into it. This mirrors how _ensure_model
# does `from sentence_transformers import SentenceTransformer` under the
# hood: when the import lands, it gets our fake.


@pytest.fixture
def fake_st_module(monkeypatch):
    """Inject a fake sentence_transformers module that returns a MagicMock
    SentenceTransformer class. Yields the class so individual tests can
    set its return_value to a specific mock model."""
    fake_module = types.ModuleType("sentence_transformers")
    fake_class = MagicMock(name="SentenceTransformer")
    fake_module.SentenceTransformer = fake_class
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_module)
    yield fake_class


# =============================================================================
# Test fixtures
# =============================================================================


def _entry(name, description="", tactics=None, platforms=None):
    return {
        "name": name,
        "stix_id": f"attack-pattern--{name.lower().replace(' ', '-')}",
        "description": description,
        "tactics": tactics or ["execution"],
        "platforms": platforms or ["Windows"],
    }


def _small_catalogue():
    """Tiny catalogue for shape assertions. No ALWAYS_INCLUDE overlap."""
    return {
        "T9001": _entry("Foo Technique", "Adversaries do foo via bar."),
        "T9002": _entry("Baz Technique", "Adversaries do baz via qux."),
        "T9001.001": _entry("Foo Sub-A", "Foo sub-technique A."),
        "T9001.002": _entry("Foo Sub-B", "Foo sub-technique B."),
    }


def _chunk(cid, text):
    return {"chunk_id": cid, "text": text}


# =============================================================================
# Anchor keyword extraction (regression for the moved helper)
# =============================================================================


def test_extract_anchor_keywords_picks_up_cves_and_tools():
    text = (
        "Attacker exploited CVE-2024-1234 then ran certutil.exe to "
        "download a payload. Mimikatz was used for credential dumping."
    )
    kws = _extract_anchor_keywords(text)
    assert "cve-2024-1234" in kws
    assert "certutil" in kws
    assert "mimikatz" in kws


def test_extract_anchor_keywords_empty_on_clean_text():
    assert _extract_anchor_keywords("benign text with no IOCs") == set()


# =============================================================================
# _boost_score: shared precision boost rules
# =============================================================================


def test_boost_score_anchor_keyword_boost():
    entry = _entry("Tool Use", "Adversaries abuse certutil to download files.")
    boost = _boost_score("T1140", entry, "attacker ran certutil", {"certutil"})
    # Only the anchor matches, not name-in-chunk, not ID-in-chunk
    assert boost == pytest.approx(0.3)


def test_boost_score_name_in_chunk_boost():
    entry = _entry("PowerShell", "Description text.")
    # Name appears in chunk text. No anchors, no ID match.
    boost = _boost_score(
        "T1059.001", entry, "attacker used powershell extensively", set(),
    )
    assert boost == pytest.approx(0.5)


def test_boost_score_id_in_chunk_boost():
    entry = _entry("Some Technique", "Some description.")
    # ID appears in chunk. No name match, no anchors.
    boost = _boost_score(
        "T1059.001", entry, "tagged as t1059.001 by analyst", set(),
    )
    assert boost == pytest.approx(0.5)


def test_boost_score_stacks_all_three():
    entry = _entry("PowerShell", "Adversaries abuse mimikatz on PowerShell.")
    boost = _boost_score(
        "T1059.001", entry,
        "attacker used powershell mimikatz t1059.001",
        {"mimikatz"},
    )
    # 0.3 (anchor) + 0.5 (name) + 0.5 (id) = 1.3
    assert boost == pytest.approx(1.3)


def test_boost_score_anchor_capped_at_one_match():
    """Multiple anchor matches in the description still only boost +0.3 once."""
    entry = _entry("Tools", "Both certutil and mimikatz are abused.")
    boost = _boost_score(
        "T9999", entry, "no name no id here", {"certutil", "mimikatz"},
    )
    assert boost == pytest.approx(0.3)


def test_boost_score_scale_halves_boost():
    entry = _entry("PowerShell", "Adversaries abuse mimikatz on PowerShell.")
    full = _boost_score(
        "T1059.001", entry,
        "attacker used powershell mimikatz t1059.001",
        {"mimikatz"},
        boost_scale=1.0,
    )
    half = _boost_score(
        "T1059.001", entry,
        "attacker used powershell mimikatz t1059.001",
        {"mimikatz"},
        boost_scale=0.5,
    )
    assert full == pytest.approx(1.3)
    assert half == pytest.approx(0.65)


def test_boost_score_scale_zero_disables_boosts():
    """boost_scale=0.0 must zero out all three boost rules — useful for
    isolating retriever-only behavior in A/B tests."""
    entry = _entry("PowerShell", "Adversaries abuse mimikatz on PowerShell.")
    boost = _boost_score(
        "T1059.001", entry,
        "attacker used powershell mimikatz t1059.001",
        {"mimikatz"},
        boost_scale=0.0,
    )
    assert boost == 0.0


# =============================================================================
# _finalize_candidates: parent/sub expansion, reference text shape
# =============================================================================


def test_finalize_candidates_expands_parent_when_only_sub_picked():
    catalogue = _small_catalogue()
    candidate_tids = {"T9001.001"}  # only the sub
    filtered, ref = _finalize_candidates(
        candidate_tids, catalogue, chunks_count=1, top_k=30,
        retriever_label="test",
    )
    assert "T9001" in filtered  # parent auto-included
    assert "T9001.001" in filtered


def test_finalize_candidates_expands_subs_when_only_parent_picked():
    catalogue = _small_catalogue()
    candidate_tids = {"T9001"}
    filtered, ref = _finalize_candidates(
        candidate_tids, catalogue, chunks_count=1, top_k=30,
        retriever_label="test",
    )
    # All known subs of T9001 pulled in
    assert "T9001.001" in filtered
    assert "T9001.002" in filtered


def test_finalize_candidates_reference_text_format():
    catalogue = _small_catalogue()
    candidate_tids = {"T9001"}
    _, ref = _finalize_candidates(
        candidate_tids, catalogue, chunks_count=2, top_k=30,
        retriever_label="test",
    )
    assert "ATT&CK TECHNIQUE CANDIDATES" in ref
    assert "T9001 | Foo Technique | execution" in ref
    # Subs got expanded, so they should be listed too
    assert "T9001.001 | Foo Sub-A" in ref


# =============================================================================
# TokenOverlapRetriever: behavior parity with pre-refactor implementation
# =============================================================================


def test_token_overlap_returns_empty_on_empty_catalogue():
    chunks = [_chunk("c1", "anything")]
    filtered, ref = TokenOverlapRetriever().find_candidates(chunks, {}, top_k=30)
    assert filtered == {}
    assert ref == ""


def test_token_overlap_includes_always_include_when_present():
    catalogue = {
        "T1059": _entry("Command and Scripting Interpreter", "execution stuff"),
        "T9999": _entry("Random", "unrelated"),
    }
    chunks = [_chunk("c1", "totally unrelated to anything in catalogue")]
    filtered, _ref = TokenOverlapRetriever().find_candidates(
        chunks, catalogue, top_k=1,
    )
    # T1059 is in _ALWAYS_INCLUDE so it's pulled in regardless of overlap
    assert "T1059" in filtered
    assert "T1059" in _ALWAYS_INCLUDE  # sanity check on the constant


def test_token_overlap_picks_up_lexically_overlapping_technique():
    catalogue = {
        "T9001": _entry("Phishing Spearphishing Attachment",
                        "adversary sends spearphishing attachment to victim"),
        "T9002": _entry("Encrypted Channel",
                        "adversary uses encrypted protocols for communication"),
    }
    chunks = [_chunk("c1", "victim received a spearphishing attachment via email")]
    filtered, _ref = TokenOverlapRetriever().find_candidates(
        chunks, catalogue, top_k=1,
    )
    assert "T9001" in filtered


# =============================================================================
# EmbeddingRetriever: tested with a mock SentenceTransformer.
# Real SecureBERT 2.0 download (~1.5 GB) is intentionally avoided; live
# verification belongs in a manual smoke run after rebuild.
# =============================================================================


def _make_mock_model(tech_vectors=None, chunk_vectors=None):
    """Build a MagicMock SentenceTransformer.

    tech_vectors / chunk_vectors are dicts {text_substring: vector}
    used to route encode() calls deterministically. Falls back to a
    default vector for unmapped inputs so tests can focus on whichever
    cases matter.
    """
    tech_vectors = tech_vectors or {}
    chunk_vectors = chunk_vectors or {}
    default_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def encode(texts, normalize_embeddings=False, show_progress_bar=False):
        # CRITICAL: the retriever MUST pass normalize_embeddings=True.
        # We assert it here so any regression is caught.
        assert normalize_embeddings is True, (
            "EmbeddingRetriever must call encode(..., normalize_embeddings=True) "
            "so cosine similarity reduces to a dot product"
        )
        out = []
        for t in texts:
            mapped = None
            for needle, vec in {**tech_vectors, **chunk_vectors}.items():
                if needle in t:
                    mapped = vec
                    break
            out.append(mapped if mapped is not None else default_vec)
        return np.asarray(out, dtype=np.float32)

    mock = MagicMock()
    mock.encode = MagicMock(side_effect=encode)
    return mock


def test_embedding_retriever_passes_normalize_embeddings_true(fake_st_module):
    """Self-critique #1: normalize_embeddings=True is actually passed.

    Cosine similarity reduces to a single matmul ONLY when both technique
    and chunk vectors are L2-normalized. If we ever forget the flag, the
    "cosine similarity" math silently becomes dot-product-of-arbitrary-norms,
    which is not a similarity metric and breaks ranking.
    """
    fake_st_module.return_value = _make_mock_model()
    retriever = EmbeddingRetriever()
    catalogue = _small_catalogue()
    chunks = [_chunk("c1", "anything")]

    retriever.find_candidates(chunks, catalogue, top_k=2)

    # Assertion lives inside the mock's encode() side_effect, but also
    # confirm the call site actually invoked encode at least twice
    # (once for techniques, once for chunks).
    assert retriever._model.encode.call_count >= 2
    for call in retriever._model.encode.call_args_list:
        kwargs = call.kwargs
        assert kwargs.get("normalize_embeddings") is True


def test_embedding_retriever_caches_technique_index_per_catalogue(fake_st_module):
    """Self-critique #2: catalogue cache invalidates when lookup changes.

    The first find_candidates call should encode the catalogue. A second
    call with the SAME catalogue object should NOT re-encode. A third
    call with a DIFFERENT catalogue object should re-encode.
    """
    fake_st_module.return_value = _make_mock_model()
    retriever = EmbeddingRetriever()
    catalogue_a = _small_catalogue()
    catalogue_b = {
        "T8001": _entry("Different Technique", "different description"),
    }

    chunks = [_chunk("c1", "test text")]

    # 1st call — encodes catalogue_a (4 entries) + 1 chunk = 2 encode calls
    retriever.find_candidates(chunks, catalogue_a, top_k=2)
    encode_calls_after_first = retriever._model.encode.call_count

    # 2nd call — same catalogue_a, only chunk encoding happens (1 more)
    retriever.find_candidates(chunks, catalogue_a, top_k=2)
    encode_calls_after_second = retriever._model.encode.call_count
    assert encode_calls_after_second == encode_calls_after_first + 1, (
        "cache hit on identical catalogue should skip technique encoding"
    )

    # 3rd call — different catalogue_b, must re-encode techniques + chunk (2 more)
    retriever.find_candidates(chunks, catalogue_b, top_k=2)
    encode_calls_after_third = retriever._model.encode.call_count
    assert encode_calls_after_third == encode_calls_after_second + 2, (
        "different catalogue should trigger technique re-encoding"
    )


def test_embedding_retriever_end_to_end_shape_smoke(fake_st_module):
    """Self-critique #3: end-to-end produces the expected shape.

    Without a real model we can't verify ranking quality, but we CAN
    verify the contract: filtered_lookup keyed by tid, reference_text
    listing the expected entries, ALWAYS_INCLUDE honored, parent/sub
    expansion working, downstream prompt-ready string format.
    """
    # Make T9001 highly similar to chunk text (vec [1,0,0]),
    # T9002 perpendicular (vec [0,1,0]) so it scores ~0.
    tech_vecs = {
        "Foo Technique": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "Foo Sub-A":     np.array([0.99, 0.14, 0.0], dtype=np.float32),
        "Foo Sub-B":     np.array([0.99, 0.14, 0.0], dtype=np.float32),
        "Baz Technique": np.array([0.0, 1.0, 0.0], dtype=np.float32),
    }
    chunk_vecs = {
        "phishing email":  np.array([1.0, 0.0, 0.0], dtype=np.float32),
    }
    fake_st_module.return_value = _make_mock_model(tech_vecs, chunk_vecs)

    retriever = EmbeddingRetriever()
    catalogue = _small_catalogue()
    chunks = [_chunk("c1", "victim opened phishing email attachment")]

    filtered, ref = retriever.find_candidates(chunks, catalogue, top_k=2)

    # Top-2 by cosine should include T9001 (highest similarity).
    assert "T9001" in filtered
    # Parent/sub expansion: subs of T9001 get pulled in as well.
    assert "T9001.001" in filtered
    assert "T9001.002" in filtered

    # Reference text format
    assert "ATT&CK TECHNIQUE CANDIDATES" in ref
    assert "T9001 | Foo Technique | execution" in ref
    assert ref.startswith("\n\n")  # downstream prompt assembly expects this


def test_embedding_retriever_returns_empty_on_empty_catalogue(fake_st_module):
    fake_st_module.return_value = _make_mock_model()
    retriever = EmbeddingRetriever()
    chunks = [_chunk("c1", "anything")]
    filtered, ref = retriever.find_candidates(chunks, {}, top_k=30)
    assert filtered == {}
    assert ref == ""
    # Empty catalogue should short-circuit before model load
    assert retriever._model is None


def test_embedding_retriever_skips_empty_text_chunks(fake_st_module):
    fake_st_module.return_value = _make_mock_model()
    retriever = EmbeddingRetriever()
    catalogue = _small_catalogue()
    chunks = [
        _chunk("c1", ""),  # empty
        _chunk("c2", "real chunk text"),
    ]
    filtered, _ref = retriever.find_candidates(chunks, catalogue, top_k=2)
    # Should still produce candidates from the non-empty chunk
    assert len(filtered) > 0


def test_embedding_retriever_raises_clear_error_when_st_missing(monkeypatch):
    """If sentence-transformers isn't installed, the import inside
    _ensure_model raises ImportError with a fix-it message."""
    retriever = EmbeddingRetriever()
    catalogue = _small_catalogue()
    chunks = [_chunk("c1", "test")]

    # Simulate the import failing by ensuring the module is not in sys.modules
    monkeypatch.delitem(sys.modules, "sentence_transformers", raising=False)
    # And block fresh imports by registering None (Python idiom for "import fails")
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)

    with pytest.raises(ImportError) as exc_info:
        retriever.find_candidates(chunks, catalogue, top_k=2)
    assert "sentence-transformers" in str(exc_info.value)
    assert "pip install" in str(exc_info.value)


# =============================================================================
# get_retriever() factory
# =============================================================================


def test_factory_returns_token_overlap_by_default():
    with patch("app.services.technique_retriever.settings") as mock_settings:
        mock_settings.technique_retriever = "token_overlap"
        r = get_retriever()
        assert isinstance(r, TokenOverlapRetriever)


def test_factory_returns_embedding_retriever_when_configured():
    with patch("app.services.technique_retriever.settings") as mock_settings:
        mock_settings.technique_retriever = "embedding"
        mock_settings.embedding_model = "test-model"
        r = get_retriever()
        assert isinstance(r, EmbeddingRetriever)
        assert r._model_name == "test-model"
        # Lazy: model not loaded yet
        assert r._model is None


def test_factory_raises_on_unknown_mode():
    with patch("app.services.technique_retriever.settings") as mock_settings:
        mock_settings.technique_retriever = "bogus_mode"
        with pytest.raises(ValueError) as exc_info:
            get_retriever()
        assert "bogus_mode" in str(exc_info.value)


def test_factory_memoizes_so_the_technique_index_is_encoded_once():
    """A fresh instance per call re-encoded ~700 techniques every run.

    EmbeddingRetriever caches the model handle and the encoded matrix on the
    instance, so returning a new one each time defeated both. The encode is
    synchronous and holds the event loop for ~2 minutes, and it repeated on
    every gate rewind — one ransomware run paid it three times.
    """
    from app.services.technique_retriever import reset_retriever_cache

    reset_retriever_cache()
    with patch("app.services.technique_retriever.settings") as mock_settings:
        mock_settings.technique_retriever = "embedding"
        mock_settings.embedding_model = "test-model"
        assert get_retriever() is get_retriever()
    reset_retriever_cache()


def test_factory_does_not_serve_a_stale_instance_after_a_settings_change():
    """Keyed by (mode, model): changing either must yield a new retriever."""
    from app.services.technique_retriever import reset_retriever_cache

    reset_retriever_cache()
    with patch("app.services.technique_retriever.settings") as mock_settings:
        mock_settings.technique_retriever = "embedding"
        mock_settings.embedding_model = "model-a"
        first = get_retriever()
        mock_settings.embedding_model = "model-b"
        second = get_retriever()
    assert first is not second
    assert first._model_name == "model-a"
    assert second._model_name == "model-b"

    reset_retriever_cache()
    with patch("app.services.technique_retriever.settings") as mock_settings:
        mock_settings.technique_retriever = "token_overlap"
        tok = get_retriever()
        mock_settings.technique_retriever = "embedding"
        mock_settings.embedding_model = "model-a"
        emb = get_retriever()
    assert isinstance(tok, TokenOverlapRetriever)
    assert isinstance(emb, EmbeddingRetriever)
    reset_retriever_cache()
