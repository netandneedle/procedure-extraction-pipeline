"""Docling-based parser for structured document understanding.

Docling (57k+ stars) provides ML-powered document parsing with:
- Layout detection and reading order analysis (handles multi-column layouts)
- ML table structure recognition (merged cells, borderless tables)
- OCR for scanned PDFs and images
- Header/footer separation via furniture tree (no fixed-margin heuristics)
- Unified DoclingDocument representation (Pydantic)

Handles: PDF, DOCX, PPTX, XLSX, HTML, images (PNG, TIFF, JPEG)

The DoclingDocument intermediate representation is preserved in ParseResult
for downstream nodes that want structure-aware processing (e.g., weighting
entities found in tables higher than entities in prose).
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path
from typing import Any

from app.nodes.deterministic.parsers.base import BaseParser, ParseResult

logger = logging.getLogger(__name__)

# Scale at which figures are rendered when with_images=True. 1.5x keeps
# screenshot text legible to the vision model without blowing up memory.
PICTURE_IMAGE_SCALE = 1.5

# Largest dimension we keep for a rendered figure. Vision APIs resize
# server-side anyway; capping here bounds both the on-disk stash and
# the per-figure token cost.
MAX_IMAGE_DIMENSION_PX = 1568


class DoclingParser(BaseParser):
    """Parse documents using Docling's ML-powered conversion pipeline.

    Converters are built lazily and cached PER VARIANT — the plain
    text-only converter and the image-generating one have incompatible
    pipeline options, so they can't share an instance. Building a converter
    loads ML models, which is why caching both matters: the figure pass
    would otherwise pay full model-load cost on every call.

    Models are downloaded on first invocation and cached locally.
    """

    def __init__(self):
        # Keyed by with_images. Two entries at most.
        self._converters: dict[bool, Any] = {}

    def reset(self) -> None:
        """Drop cached converters. Used by tests to un-stick a mocked
        DocumentConverter between cases."""
        self._converters = {}

    def _get_converter(self, with_images: bool = False):
        """Lazy-init the converter for this variant (avoids model download
        at import time).

        Args:
            with_images: When True, configure the PDF pipeline to render
                figure bitmaps so `doc.pictures[*].get_image(doc)` works.

        Returns:
            DocumentConverter instance (imported lazily to avoid hard dependency).
        """
        cached = self._converters.get(with_images)
        if cached is not None:
            return cached

        from docling.document_converter import DocumentConverter
        logger.info(
            "DoclingParser: initializing DocumentConverter "
            "(with_images=%s; first call, models may download)",
            with_images,
        )

        if with_images:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import PdfFormatOption

            pipeline_opts = PdfPipelineOptions()
            pipeline_opts.images_scale = PICTURE_IMAGE_SCALE
            pipeline_opts.generate_picture_images = True
            converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts),
                }
            )
        else:
            converter = DocumentConverter()

        self._converters[with_images] = converter
        logger.info("DoclingParser: DocumentConverter ready (with_images=%s)", with_images)
        return converter

    def parse(self, source: str | Path, *, with_images: bool = False) -> ParseResult:
        """Parse a document file into text using Docling.

        Args:
            source: Path to a document file (PDF, DOCX, HTML, image, etc.)
            with_images: When True, also render every figure and return them
                in ParseResult.pictures. Costs extra memory during the parse
                but saves a whole second Docling pass downstream — the
                figure_extraction node reads these from the stash instead of
                re-converting the document.

        Returns:
            ParseResult with Markdown-formatted text, warnings,
            page count, the raw DoclingDocument in structured_doc, and
            (when with_images) rendered figures in pictures.

        Raises:
            FileNotFoundError: If the source file does not exist.
            ValueError: If Docling produces empty output.
        """
        path = self._validate_file_exists(source)
        warnings: list[str] = []

        converter = self._get_converter(with_images)

        try:
            result = converter.convert(str(path))
            doc = result.document
        except Exception as e:
            raise ValueError(
                f"Docling conversion failed for {path.name}: "
                f"{type(e).__name__}: {e}"
            ) from e

        # Export to Markdown (preserves headings, tables, lists, code blocks)
        markdown_text = doc.export_to_markdown()

        if not markdown_text or not markdown_text.strip():
            raise ValueError(
                f"Docling produced empty output for {path.name}. "
                "File may be corrupted, password-protected, or contain "
                "only unsupported content."
            )

        # Extract page count if available
        page_count = self._get_page_count(doc)

        # Detect potential quality issues
        self._check_quality(doc, markdown_text, warnings, path)

        # Normalize whitespace
        markdown_text = self._normalize_ligatures(markdown_text)
        markdown_text, watermark_warning = self._strip_tracking_watermark(
            markdown_text,
        )
        if watermark_warning:
            warnings.append(watermark_warning)
        markdown_text = self._normalize_whitespace(markdown_text)

        pictures = extract_pictures_from_doc(doc) if with_images else []

        return ParseResult(
            text=markdown_text,
            warnings=warnings,
            page_count=page_count,
            structured_doc=doc,
            pictures=pictures,
        )

    def _get_page_count(self, doc: Any) -> int | None:
        """Extract page count from DoclingDocument if available."""
        try:
            if hasattr(doc, "pages") and doc.pages:
                return len(doc.pages)
        except Exception:
            pass
        return None

    def _check_quality(
        self,
        doc: Any,
        text: str,
        warnings: list[str],
        path: Path,
    ) -> None:
        """Add warnings for potential quality issues."""
        word_count = len(text.split())

        # Very short output from a document file is suspicious
        if word_count < 30:
            warnings.append(
                f"Very short extraction ({word_count} words) from {path.name}. "
                "Document may be mostly images/scans with limited OCR results."
            )

        # Check for tables in the document
        try:
            if hasattr(doc, "tables"):
                table_count = len(doc.tables) if doc.tables else 0
                if table_count > 0:
                    logger.info(
                        "DoclingParser: extracted %d table(s) from %s",
                        table_count, path.name,
                    )
        except Exception:
            pass

        # Check for furniture (headers/footers detected and separated)
        try:
            if hasattr(doc, "furniture") and doc.furniture:
                furniture_items = getattr(doc.furniture, "children", [])
                if furniture_items:
                    logger.debug(
                        "DoclingParser: %d furniture items separated (headers/footers)",
                        len(furniture_items),
                    )
        except Exception:
            pass


# =============================================================================
# Figure rendering
# =============================================================================


def extract_pictures_from_doc(doc: Any) -> list[dict]:
    """Harvest rendered figures from a DoclingDocument, in document order.

    Returns a list of
    ``{caption, page, image_bytes, media_type, width, height}``. Order
    matches the `<!-- image -->` placeholders in `doc.export_to_markdown()`
    (Docling emits both from the same traversal), which is what lets
    downstream code replace placeholders positionally.

    Only meaningful when the document was converted with
    `generate_picture_images=True` — otherwise `get_image()` returns None
    and every entry carries empty `image_bytes`.

    Never raises: a document without a usable `pictures` collection yields
    an empty list, and a single unreadable figure yields an entry with
    empty bytes rather than aborting the harvest.
    """
    try:
        raw_pictures = list(getattr(doc, "pictures", None) or [])
    except TypeError:
        # Not iterable (e.g. a mock or an unexpected Docling shape).
        return []

    pictures: list[dict] = []
    for pic in raw_pictures:
        caption_parts = []
        try:
            for cap_ref in getattr(pic, "captions", None) or []:
                cap_obj = cap_ref.resolve(doc) if hasattr(cap_ref, "resolve") else cap_ref
                text = getattr(cap_obj, "text", None)
                if text:
                    caption_parts.append(text.strip())
        except (TypeError, AttributeError):
            pass
        caption = " | ".join(caption_parts)

        page_no = None
        try:
            for prov in getattr(pic, "prov", None) or []:
                page_no = getattr(prov, "page_no", None)
                break
        except (TypeError, AttributeError):
            pass

        image_bytes = b""
        width = 0
        height = 0
        try:
            pil_image = pic.get_image(doc)
            if pil_image is not None:
                # Resize if too large (bounds stash size and vision token cost)
                if max(pil_image.size) > MAX_IMAGE_DIMENSION_PX:
                    pil_image.thumbnail(
                        (MAX_IMAGE_DIMENSION_PX, MAX_IMAGE_DIMENSION_PX)
                    )
                width, height = pil_image.size
                buf = io.BytesIO()
                pil_image.save(buf, format="PNG")
                image_bytes = buf.getvalue()
        except Exception as e:  # noqa: BLE001 — one bad figure must not kill the parse
            logger.warning("extract_pictures_from_doc: failed to render a figure: %s", e)

        pictures.append({
            "caption": caption,
            "page": page_no,
            "image_bytes": image_bytes,
            "media_type": "image/png",
            "width": width,
            "height": height,
        })

    return pictures


def pictures_to_b64(pictures: list[dict]) -> list[dict]:
    """Convert `image_bytes` entries to the `image_b64` shape the vision
    node consumes. Single definition of that shape, shared by the stash
    loader and the re-parse fallback."""
    out: list[dict] = []
    for pic in pictures:
        raw = pic.get("image_bytes") or b""
        out.append({
            "caption": pic.get("caption", ""),
            "page": pic.get("page"),
            "image_b64": base64.b64encode(raw).decode("ascii") if raw else "",
            "media_type": pic.get("media_type", "image/png"),
            "width": pic.get("width", 0),
            "height": pic.get("height", 0),
        })
    return out


# Process-wide parser instance. Shared so the (expensive) converter cache is
# built once: parse_and_validate and the figure_extraction fallback both go
# through the same instance rather than each loading Docling's ML models.
_SHARED_PARSER: DoclingParser | None = None


def get_shared_parser() -> DoclingParser:
    """Return the process-wide DoclingParser."""
    global _SHARED_PARSER
    if _SHARED_PARSER is None:
        _SHARED_PARSER = DoclingParser()
    return _SHARED_PARSER
