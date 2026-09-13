"""figure_extraction node: vision-LLM pass over PDF figures.

WHY THIS NODE EXISTS:
A figure-coverage measurement across 14 vendor reports showed Docling's
text export drops the *content* of every figure — 221/221 figures had
`<!-- image -->` placeholders in markdown but zero had text annotations.
Attack-chain diagrams, command-line screenshots, registry shots, and other
figure-only content were silently lost.

This node closes the gap. It picks up the figures `parse_and_validate`
rendered into the figure stash, sends each picture to the extraction model
with a vision-aware prompt that classifies the figure (diagram / screenshot
/ decorative / other) and extracts its text content, then replaces the
corresponding `<!-- image -->` placeholder in `parsed_text` with a
delimited block.

Downstream (entity_extraction, chunk_behaviors, etc.) sees figure-derived
text as part of the body without any code changes.

DESIGN DECISIONS:
    1. Always-process by default — accuracy over cost.
    2. Single multimodal call per figure: classify + extract together,
       capped at settings.figure_extraction_max_tokens so one dense image
       can't run away to the adapter's global output default.
    3. Bounded-concurrency processing: figures fan out via asyncio.gather
       behind a semaphore of settings.figure_extraction_concurrency, so a
       figure-heavy source isn't serialized one vision call at a time.
       (With sequential processing a single 2.5-min runaway figure once
       blocked the whole node.)
    4. On per-figure failure: log warning, leave placeholder in place,
       continue. Don't fail the whole source for one bad figure.
    5. Source-level toggle (Source.extract_figures) lets analysts skip
       this node for all-prose sources.
    6. Figures come from the stash `parse_and_validate` wrote, not from a
       second Docling conversion. Re-parsing is kept only as a fallback
       for sources parsed before the stash existed (or whose stash write
       failed). See WHY THE STASH below.

TAXONOMY NOTE:
The node lives in nodes/llm/ because it makes LLM calls. Loading figures is
a deterministic prelude inside the node, not a separate step — keeping the
graph topology simple.

WHY THE STASH:
Docling only hands over figure bitmaps when converted with
`generate_picture_images=True`, which is incompatible with the cached
text-only converter. Running a whole second conversion here doubled
ingestion time (~110-120s on a mid-size PDF, measured on CPU). So
parse_and_validate converts once with images on and writes them to
app.services.figure_stash; this node reads them back. The fallback keeps
old checkpoints working, and logs loudly when it fires so a systematically
missing stash is visible rather than silently slow.

READS: parsed_text, raw_content_path, source_type, extract_figures, source_id
WRITES: parsed_text (placeholders replaced), extracted_figures
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from app.config import settings
from app.graph.state import PipelineState, PipelineStatus
from app.nodes.deterministic.parsers.docling_parser import (
    get_shared_parser,
    pictures_to_b64,
)
from app.nodes.llm.llm_adapter import call_llm
from app.nodes.llm.providers import image_block, text_block
from app.nodes.llm.tool_models import ExtractFigureOutput
from app.services import figure_stash

logger = logging.getLogger(__name__)


# Source types where figure extraction is meaningful. Defined in the stash
# service so parse_and_validate can consult it without importing an LLM node.
_FIGURE_BEARING_TYPES = figure_stash.FIGURE_BEARING_TYPES

# Images smaller than this on either dimension are likely icons/decorations
# and not worth a vision call. Conservative threshold — better to send a
# small attack-chain diagram than miss it.
_MIN_IMAGE_DIMENSION_PX = 100

# Max chars of extracted figure text kept in the per-figure AUDIT entry.
# The full text already lives inline in parsed_text; the audit only needs
# a preview for the Gate 0 chip / debugging.
_AUDIT_TEXT_PREVIEW_CHARS = 200


# =============================================================================
# Tool definition
# =============================================================================

EXTRACT_FIGURE_TOOL = {
    "name": "extract_figure",
    "description": (
        "Classify a figure from a CTI source and extract its text content "
        "in one structured response. Use 'diagram' for attack-chain "
        "summaries / kill-chain visualizations / network flow diagrams; "
        "'screenshot' for command-line / registry / log / file-explorer "
        "captures showing verbatim text; 'decorative' for logos, icons, "
        "and brand graphics with no operational content; 'other' for "
        "anything else (org charts, geographical maps, etc.)."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "figure_type": {
                "type": "string",
                "enum": ["diagram", "screenshot", "decorative", "other"],
                "description": (
                    "Classification of the figure. 'diagram' = structured "
                    "kill-chain or attack-flow visualization with grouped "
                    "tactics + procedure callouts. 'screenshot' = raster "
                    "capture of a terminal, registry editor, file explorer, "
                    "log viewer, or any UI showing verbatim text. "
                    "'decorative' = logos, icons, brand graphics, divider "
                    "art, page-margin imagery — anything with no CTI value. "
                    "'other' = visualization that doesn't fit the above "
                    "(org charts, maps, charts/graphs, abstract illustrations)."
                ),
            },
            "extracted_text": {
                "type": "string",
                "description": (
                    "Text content extracted from the figure, formatted "
                    "appropriately for the figure_type:\n"
                    "- diagram: a numbered list of procedures in kill-chain "
                    "  order, each formatted as 'Tactic: Procedure Name — "
                    "  description'. Capture every procedure callout visible.\n"
                    "- screenshot: VERBATIM transcription of the visible "
                    "  text, preserving line breaks and any visible quoting "
                    "  (paths, command flags, registry keys).\n"
                    "- decorative: empty string.\n"
                    "- other: a 1-3 sentence factual description of what "
                    "  the figure shows (no speculation).\n"
                    "Do NOT invent content. If unsure of a character "
                    "(e.g., blurry text), use [unclear] markers. Do NOT "
                    "summarize — extract everything visible."
                ),
            },
            "confidence": {
                "type": "number",
                "minimum": 0.0,
                "maximum": 1.0,
                "description": (
                    "Confidence in the extraction. 1.0 = clear figure, all "
                    "text legible, classification unambiguous. 0.5 = some "
                    "text unclear or classification borderline. 0.0-0.3 = "
                    "low-quality image or genuine ambiguity."
                ),
            },
            "rationale": {
                "type": "string",
                "description": (
                    "One short sentence explaining the figure_type "
                    "decision. Example: 'Numbered kill-chain diagram "
                    "showing 9 procedures grouped by ATT&CK tactic.'"
                ),
            },
        },
        "required": ["figure_type", "extracted_text", "confidence", "rationale"],
    },
}


SYSTEM_PROMPT = """You are a CTI ingestion assistant. Your job is to classify a figure from a threat intelligence report and extract its text content using the extract_figure tool.

CLASSIFICATION GUIDANCE:
- diagram: structured visualization showing the *flow* of an intrusion. Common shapes: kill-chain summary with tactic boxes (Initial Access → Execution → Persistence → ...), arrows connecting procedures, tactic groupings with per-procedure callouts. The information is in the structure: which procedures, which tactics, what order.
- screenshot: raster capture of a UI element. Common shapes: terminal/command prompt with visible commands, registry editor with key paths, file explorer with paths, log viewer, code editor, browser address bar. The information is in the text: verbatim commands, paths, registry keys, log entries.
- decorative: zero CTI signal. Examples: vendor logos, page-divider graphics, decorative icons, branded headers. Skip these.
- other: visualization that has SOME content but doesn't fit the above. Examples: geographical heatmaps showing target regions, org charts naming subsidiaries, time-series graphs of attacks-per-month. Provide a brief factual description.

EXTRACTION GUIDANCE:
- For diagrams: output a numbered list. Each entry: 'Tactic: Procedure Name — one-line description'. Walk left-to-right, top-to-bottom in the order the diagram presents. Capture EVERY procedure callout, even small ones. If the diagram shows arrows for sequencing, the order of your list IS the kill-chain order.
- For screenshots: transcribe text VERBATIM. Preserve case, punctuation, slashes, quoting. If a command line spans multiple lines visually, preserve the wrapping. Use [unclear] for any character you cannot read confidently.
- For decorative figures: extracted_text is empty. The pipeline drops these.
- For other: 1-3 factual sentences. No speculation about adversary intent — just what the figure depicts.

CRITICAL RULES:
- DO NOT INVENT CONTENT. If a procedure isn't visible in the figure, don't list it.
- DO NOT SUMMARIZE. Extract everything visible. Better to over-extract than miss content.
- DO NOT INFER. If a screenshot shows a partial command, transcribe what's visible — don't complete it from training memory.
- If the figure is too blurry or low-quality to read, set confidence < 0.3 and explain in rationale.
"""


# =============================================================================
# Node function
# =============================================================================

async def extract_figures(state: PipelineState) -> dict:
    """Vision-LLM pass over the source's figures.

    Loads the pictures from the figure stash (re-parsing with Docling only
    when no stash exists), sends each to the vision-capable extraction model
    with the extract_figure tool, and replaces `<!-- image -->` placeholders
    in parsed_text with the extracted content (or omits them for decorative
    figures).

    Robust to per-figure failures: logs a warning and continues on any
    individual figure error rather than failing the source.

    READS: parsed_text, raw_content_path, source_type, extract_figures
    WRITES: parsed_text (placeholders replaced), extracted_figures, status,
    current_node
    """
    parsed_text = state.get("parsed_text", "") or ""
    raw_path = state.get("raw_content_path", "") or ""
    source_type = state.get("source_type", "") or ""
    source_id = state.get("source_id", "") or ""
    enabled = bool(state.get("extract_figures", True))

    update: dict = {
        "status": PipelineStatus.EXTRACTING_FIGURES.value,
        "current_node": "extract_figures",
        "extracted_figures": [],
    }

    # Skip paths — keep parsed_text as-is.
    if not enabled:
        logger.info("extract_figures: disabled by Source.extract_figures=False; skipping")
        return update
    if source_type not in _FIGURE_BEARING_TYPES:
        logger.info(
            "extract_figures: source_type=%s has no figures; skipping",
            source_type,
        )
        return update
    if not parsed_text:
        logger.info("extract_figures: empty parsed_text; skipping")
        return update
    if "<!-- image -->" not in parsed_text:
        logger.info("extract_figures: no image placeholders in parsed_text; skipping")
        return update

    try:
        pictures = _load_pictures(source_id, raw_path)
    except Exception as e:
        logger.exception("extract_figures: could not obtain figures")
        update["extracted_figures"] = [{
            "figure_id": "",
            "page": None,
            "caption": "",
            "figure_type": "",
            "extracted_text": "",
            "confidence": 0.0,
            "rationale": f"figure load failed: {type(e).__name__}: {e}",
            "status": "failed",
        }]
        return update

    if not pictures:
        logger.info("extract_figures: 0 pictures available; skipping")
        return update

    logger.info("extract_figures: processing %d pictures", len(pictures))

    # Fan out the per-figure vision calls behind a semaphore. gather() returns
    # results in INPUT order regardless of completion order, so `replacements`
    # stays positionally aligned with the `<!-- image -->` placeholders even
    # though the calls finish out of order. _process_one_figure never raises;
    # return_exceptions=True is belt-and-suspenders so a leaked error fails
    # one figure soft instead of aborting the whole gather.
    concurrency = max(1, int(settings.figure_extraction_concurrency))
    semaphore = asyncio.Semaphore(concurrency)
    results = await asyncio.gather(
        *(_process_one_figure(i, pic, semaphore) for i, pic in enumerate(pictures)),
        return_exceptions=True,
    )

    extracted: list[dict] = []
    replacements: list[str] = []  # parallel to pictures; what to replace each placeholder with
    for i, res in enumerate(results):
        if isinstance(res, BaseException):
            figure_id = f"fig-{i+1}"
            logger.warning(
                "extract_figures: figure %s raised through gather: %r",
                figure_id, res,
            )
            extracted.append({
                "figure_id": figure_id,
                "page": None,
                "caption": "",
                "figure_type": "",
                "extracted_text": "",
                "confidence": 0.0,
                "rationale": f"Unexpected error: {type(res).__name__}",
                "status": "failed",
            })
            replacements.append("<!-- image -->")  # leave placeholder for visibility
            continue
        entry, replacement = res
        extracted.append(entry)
        replacements.append(replacement)

    # Now replace the Nth `<!-- image -->` in parsed_text with replacements[N].
    new_text = _replace_placeholders_in_order(parsed_text, replacements)
    update["parsed_text"] = new_text
    update["extracted_figures"] = extracted

    n_done = sum(1 for e in extracted if e["status"] == "extracted")
    n_skipped = sum(1 for e in extracted if e["status"].startswith("skipped"))
    n_failed = sum(1 for e in extracted if e["status"] == "failed")
    logger.info(
        "extract_figures: extracted=%d skipped=%d failed=%d  parsed_text: %d -> %d chars",
        n_done, n_skipped, n_failed, len(parsed_text), len(new_text),
    )
    return update


# =============================================================================
# Helpers
# =============================================================================


async def _process_one_figure(
    index: int,
    pic: dict,
    semaphore: asyncio.Semaphore,
) -> tuple[dict, str]:
    """Process one figure → (audit_entry, placeholder_replacement).

    NEVER raises: every failure path returns a `status="failed"` audit entry
    with the `<!-- image -->` placeholder retained, so one bad figure can't
    abort the gather() over the whole set. The cheap skip paths (too-small,
    no image bytes) return without acquiring `semaphore` — only the actual
    vision call is rate-limited, so decorative/tiny figures don't consume a
    concurrency slot.
    """
    figure_id = f"fig-{index+1}"
    caption = pic.get("caption", "")
    page = pic.get("page")
    image_b64 = pic.get("image_b64", "")
    media_type = pic.get("media_type", "image/png")
    width = pic.get("width", 0)
    height = pic.get("height", 0)

    # Skip too-small images (likely icons / page decorations) — no vision call.
    if width and height and (width < _MIN_IMAGE_DIMENSION_PX or height < _MIN_IMAGE_DIMENSION_PX):
        return (
            {
                "figure_id": figure_id,
                "page": page,
                "caption": caption,
                "figure_type": "decorative",
                "extracted_text": "",
                "confidence": 1.0,
                "rationale": f"Skipped: image too small ({width}x{height} < {_MIN_IMAGE_DIMENSION_PX}px)",
                "status": "skipped_too_small",
            },
            "",  # remove placeholder, no inline content
        )

    if not image_b64:
        return (
            {
                "figure_id": figure_id,
                "page": page,
                "caption": caption,
                "figure_type": "",
                "extracted_text": "",
                "confidence": 0.0,
                "rationale": "Skipped: Docling did not produce image bytes",
                "status": "failed",
            },
            "<!-- image -->",  # leave placeholder
        )

    # Vision LLM call — bounded by the shared semaphore.
    try:
        async with semaphore:
            result = await _classify_and_extract(image_b64, media_type, caption)
    except Exception as e:
        logger.warning(
            "extract_figures: vision call failed for figure %s: %s",
            figure_id, e,
        )
        return (
            {
                "figure_id": figure_id,
                "page": page,
                "caption": caption,
                "figure_type": "",
                "extracted_text": "",
                "confidence": 0.0,
                "rationale": f"Vision call failed: {type(e).__name__}",
                "status": "failed",
            },
            "<!-- image -->",  # leave placeholder for visibility
        )

    figure_type = result.figure_type
    extracted_text = result.extracted_text or ""
    confidence = float(result.confidence)
    rationale = result.rationale or ""

    if figure_type == "decorative":
        return (
            {
                "figure_id": figure_id,
                "page": page,
                "caption": caption,
                "figure_type": figure_type,
                "extracted_text": "",
                "confidence": confidence,
                "rationale": rationale,
                "status": "skipped_decorative",
            },
            "",  # decorative: drop placeholder, no inline content
        )

    # Format the extracted block. Markers preserve provenance for downstream
    # classifier / chunker; analysts see the figure source at gate_chunks.
    block = _format_figure_block(figure_id, page, caption, figure_type, extracted_text)
    # Audit carries a capped PREVIEW, not the full transcription. The full text
    # is already inlined into parsed_text via the [FIGURE...] block, so
    # duplicating it here would re-serialize tens of KB per figure into every
    # PostgresSaver checkpoint (20+ per run).
    entry = {
        "figure_id": figure_id,
        "page": page,
        "caption": caption,
        "figure_type": figure_type,
        "extracted_text": extracted_text[:_AUDIT_TEXT_PREVIEW_CHARS],
        "extracted_text_len": len(extracted_text),
        "confidence": confidence,
        "rationale": rationale,
        "status": "extracted",
    }
    return entry, block


def _load_pictures(source_id: str, raw_path: str) -> list[dict]:
    """Get this source's figures, preferring the stash parse_and_validate
    wrote over a fresh Docling conversion.

    A present-but-empty stash is a real answer ("that parse found no
    figures"), so it short-circuits too — only a MISSING stash falls through
    to the slow path.
    """
    stashed = figure_stash.load(source_id)
    if stashed is not None:
        logger.info(
            "extract_figures: loaded %d figure(s) from stash for source %s "
            "(skipped Docling re-parse)",
            len(stashed), source_id,
        )
        return stashed

    logger.warning(
        "extract_figures: no figure stash for source %r — re-parsing %s with "
        "Docling (slow path, ~2min). Expected only for sources parsed before "
        "the stash existed or when the stash write failed.",
        source_id, raw_path,
    )
    return _extract_pictures(raw_path)


def _extract_pictures(raw_path: str) -> list[dict]:
    """Fallback: re-convert the source with Docling image generation on and
    return {caption, page, image_b64, media_type, width, height} in document
    order.

    Order matches `<!-- image -->` placeholders in the markdown export
    (Docling emits both from the same traversal). Goes through the shared
    parser so it reuses the cached image-capable converter rather than
    reloading Docling's ML models on every call.
    """
    result = get_shared_parser().parse(raw_path, with_images=True)
    return pictures_to_b64(result.pictures)


async def _classify_and_extract(
    image_b64: str,
    media_type: str,
    caption: str,
) -> ExtractFigureOutput:
    """One vision call per figure: classify + extract together."""
    # Neutral blocks, not Anthropic-shaped ones — the provider translates.
    # This used to build `{"type": "image", "source": {...}}` inline, which
    # was the only vendor-specific request code outside the adapter.
    user_content: list[dict[str, Any]] = [
        image_block(media_type=media_type, data=image_b64),
        text_block(
            f"Caption (from the source layout, may be empty): "
            f"{caption!r}\n\n"
            "Use the extract_figure tool to classify this figure and "
            "extract its text content."
        ),
    ]

    response = await call_llm(
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        tools=[EXTRACT_FIGURE_TOOL],
        tool_choice={"type": "tool", "name": "extract_figure"},
        # Cap output so a single dense/ambiguous image can't run the model to
        # the adapter's global default (one figure once took ~2.5 min that
        # way). An over-tight cap that truncates the tool call surfaces
        # as a validation error → this figure fails soft, placeholder retained.
        max_tokens=settings.figure_extraction_max_tokens,
        temperature=0.0,
        output_model=ExtractFigureOutput,
    )

    if response.validated is not None:
        return response.validated
    # Fall back to building from the raw dict (validation off path)
    return ExtractFigureOutput(
        figure_type=response.tool_output.get("figure_type", "other"),
        extracted_text=response.tool_output.get("extracted_text", "") or "",
        confidence=float(response.tool_output.get("confidence", 0.0) or 0.0),
        rationale=response.tool_output.get("rationale", "") or "",
    )


def _format_figure_block(
    figure_id: str,
    page: int | None,
    caption: str,
    figure_type: str,
    text: str,
) -> str:
    """Format an extracted figure's content as a delimited block to inline
    in parsed_text. Markers preserve provenance so downstream consumers
    (and analysts at gate_chunks) can see the figure source."""
    page_str = f"page {page}" if page is not None else "page ?"
    cap_str = f', "{caption}"' if caption else ""
    header = f"[FIGURE {figure_id} — {figure_type}, {page_str}{cap_str}]"
    footer = f"[/FIGURE {figure_id}]"
    body = (text or "").strip()
    return f"\n\n{header}\n{body}\n{footer}\n\n"


def _replace_placeholders_in_order(text: str, replacements: list[str]) -> str:
    """Replace each `<!-- image -->` in `text` with replacements[i], in
    order. If `replacements` is shorter than the number of placeholders,
    leave the remaining placeholders intact. If longer, ignore extras."""
    placeholder = "<!-- image -->"
    parts = text.split(placeholder)
    if len(parts) == 1:
        return text  # no placeholders
    out = [parts[0]]
    for i, segment in enumerate(parts[1:]):
        if i < len(replacements):
            out.append(replacements[i])
        else:
            out.append(placeholder)
        out.append(segment)
    return "".join(out)
