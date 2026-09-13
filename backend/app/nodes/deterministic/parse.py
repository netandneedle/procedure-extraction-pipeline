"""parse_and_validate node: Stage 1c of the extraction pipeline.

This is the first node the pipeline hits after a source is dequeued.
It converts raw source material into clean text for downstream
processing.

WHAT THIS NODE DOES:
1. Reads the source file from raw_content_path
2. Dispatches to the correct parser based on source_type
3. Validates the output (non-empty, minimum length)
4. Stashes rendered figures to disk for the extract_figures node
5. Returns parsed_text and any parser warnings

FIGURE STASH:
When the source can carry figures and the analyst left figure extraction
on, the Docling conversion runs with image generation enabled and the
rendered bitmaps are written to the figure stash. This exists so the
downstream extract_figures node doesn't have to convert the same document
a second time just to get its images (~110-120s of duplicated work). The
stash write is best-effort: if it fails, extract_figures falls back to
re-parsing and the source still completes.

Parser architecture:
- DoclingParser: ML-powered document understanding for PDF, HTML, DOCX,
  IMAGE. Handles layout detection, table structure, OCR, reading order.
- TextParser: Lightweight pass-through for plain text and markdown
  (already extractable formats that don't need ML conversion).

WHAT THIS NODE DOES NOT DO:
- No LLM calls (purely deterministic)
- No entity extraction (that's Stage 2a)
- No content classification (that's Stage 2b)
- No threat intel validation (future enhancement)
"""

from __future__ import annotations

import logging

from app.graph.state import PipelineState, PipelineStatus, SourceType
from app.nodes.deterministic.parsers.base import ParseResult
from app.nodes.deterministic.parsers.docling_parser import get_shared_parser
from app.nodes.deterministic.parsers.text_parser import TextParser
from app.services import figure_stash

logger = logging.getLogger(__name__)

# Minimum word count for parsed output. Below this, we warn but don't fail.
# A very short source might still be valid (e.g., a tweet or brief advisory).
MIN_WORD_COUNT_WARNING = 50

# Shared parser instances (stateless, instantiate once at module load).
# DoclingParser lazy-loads its ML models on first use. The Docling instance
# is process-wide (see get_shared_parser) so the figure_extraction fallback
# reuses the same cached converters instead of reloading the models.
_docling = get_shared_parser()
_text = TextParser()

# Parser registry: maps SourceType values to parser instances.
_PARSERS = {
    SourceType.PDF.value: _docling,
    SourceType.HTML.value: _docling,
    SourceType.DOCX.value: _docling,
    SourceType.IMAGE.value: _docling,
    SourceType.MARKDOWN.value: _text,
    SourceType.FREE_TEXT.value: _text,
}

# Source types that are not yet implemented
_NOT_IMPLEMENTED = {
    SourceType.TWEET_URL.value,
    SourceType.STIX_BUNDLE.value,
}


def parse_and_validate(state: PipelineState) -> dict:
    """Stage 1c: Convert raw source to clean text.

    LangGraph node function. Receives full pipeline state, returns
    partial update dict with parsed text and warnings.

    Args:
        state: Current pipeline state. Must contain:
            - raw_content_path: Path to the source file or raw content
            - source_type: SourceType value determining which parser to use

    Returns:
        Dict with:
            - parsed_text: Extracted text
            - parse_warnings: List of warning strings
            - status: Updated to PARSING
            - current_node: "parse_and_validate"
            - error: Set if parsing fails entirely
    """
    source_type = state.get("source_type", "")
    raw_path = state.get("raw_content_path", "")
    source_id = state.get("source_id", "") or ""

    # Render figures during this conversion when the downstream vision pass
    # will want them. Doing it here is what lets extract_figures skip its own
    # full Docling conversion of the same file.
    want_images = (
        bool(state.get("extract_figures", True))
        and source_type in figure_stash.FIGURE_BEARING_TYPES
    )

    logger.info(
        "parse_and_validate: source_type=%s, path=%s, with_images=%s",
        source_type, raw_path, want_images,
    )

    # Base update (always set regardless of success/failure)
    update: dict = {
        "status": PipelineStatus.PARSING.value,
        "current_node": "parse_and_validate",
    }

    try:
        result = _dispatch_parser(source_type, raw_path, with_images=want_images)

        update["parsed_text"] = result.text
        update["parse_warnings"] = result.warnings

        _stash_figures(source_id, result, want_images)

        # Warn on very short output
        if result.word_count < MIN_WORD_COUNT_WARNING:
            update["parse_warnings"] = result.warnings + [
                f"Parsed text is short ({result.word_count} words). "
                "Source may be incomplete or low-quality."
            ]

        logger.info(
            "parse_and_validate: success. words=%d, warnings=%d, pages=%s",
            result.word_count, len(result.warnings), result.page_count,
        )

    except FileNotFoundError as e:
        logger.error("parse_and_validate: file not found: %s", e)
        update["error"] = f"Source file not found: {e}"
        update["status"] = PipelineStatus.FAILED.value
        update["parsed_text"] = ""
        update["parse_warnings"] = [str(e)]

    except ValueError as e:
        logger.error("parse_and_validate: validation error: %s", e)
        update["error"] = f"Parse validation failed: {e}"
        update["status"] = PipelineStatus.FAILED.value
        update["parsed_text"] = ""
        update["parse_warnings"] = [str(e)]

    except NotImplementedError as e:
        logger.error("parse_and_validate: not implemented: %s", e)
        update["error"] = str(e)
        update["status"] = PipelineStatus.FAILED.value
        update["parsed_text"] = ""
        update["parse_warnings"] = [str(e)]

    except Exception as e:
        logger.exception("parse_and_validate: unexpected error")
        update["error"] = f"Unexpected parse error: {type(e).__name__}: {e}"
        update["status"] = PipelineStatus.FAILED.value
        update["parsed_text"] = ""
        update["parse_warnings"] = [str(e)]

    return update


def _dispatch_parser(
    source_type: str,
    raw_path: str,
    *,
    with_images: bool = False,
) -> ParseResult:
    """Route to the correct parser based on source_type.

    `with_images` only reaches DoclingParser; the text parsers have no
    figures to render and ignore it.
    """
    if not raw_path:
        raise ValueError("raw_content_path is empty")

    # Check for not-yet-implemented types
    if source_type in _NOT_IMPLEMENTED:
        raise NotImplementedError(
            f"Parser for source_type '{source_type}' is not yet implemented. "
            f"Supported types: {', '.join(_PARSERS.keys())}"
        )

    # Look up parser in registry
    parser = _PARSERS.get(source_type)
    if parser is None:
        raise ValueError(
            f"Unknown source_type: '{source_type}'. "
            f"Supported types: {', '.join(_PARSERS.keys())}"
        )

    if with_images and parser is _docling:
        return parser.parse(raw_path, with_images=True)
    return parser.parse(raw_path)


def _stash_figures(source_id: str, result: ParseResult, want_images: bool) -> None:
    """Persist rendered figures for the extract_figures node.

    Best-effort, and defensively so: the parse has already succeeded by the
    time we get here, and the stash is only a cache. Anything that goes
    wrong costs the downstream node a re-parse — it must never turn a good
    parse into a failed source, so the whole thing is swallowed rather than
    left to the node's outer handler.

    When figures were NOT requested we clear instead, so a source re-run
    with figure extraction switched off doesn't leave a stale stash behind
    for a later run to pick up.
    """
    if not source_id:
        return
    try:
        if not want_images:
            figure_stash.clear(source_id)
            return
        n = figure_stash.save(source_id, result.pictures)
        if n:
            logger.info(
                "parse_and_validate: stashed %d figure(s) for source %s "
                "(extract_figures will skip its Docling re-parse)",
                n, source_id,
            )
    except Exception as e:  # noqa: BLE001 — a cache write must not fail the parse
        logger.warning(
            "parse_and_validate: could not stash figures for source %s: %s "
            "(extract_figures will fall back to re-parsing)",
            source_id, e,
        )
