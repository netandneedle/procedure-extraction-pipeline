"""Base parser interface and shared result type.

Every parser implements the same contract:
    parse(source_path_or_content) -> ParseResult

ParseResult carries the extracted text plus any warnings the parser
generated (OCR quality issues, encoding problems, missing content, etc.).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


@dataclass
class ParseResult:
    """Output from any parser.

    Attributes:
        text: The extracted plain text from the source.
        warnings: Non-fatal issues encountered during parsing.
            Examples: "OCR quality low on pages 3-5",
            "Encoding detected as windows-1252, converted to UTF-8",
            "Table on page 7 may have lost column alignment".
        page_count: Number of pages (for PDF), None for other formats.
        word_count: Approximate word count of the extracted text.
        structured_doc: Optional structured document representation from
            Docling (DoclingDocument). Carries typed elements (TextItem,
            TableItem, PictureItem), bounding boxes, and hierarchy.
            Available when parsed by DoclingParser; None for text-only parsers.
            Downstream nodes can use this for structure-aware extraction
            (e.g., weighting entities from tables vs. prose).
        pictures: Rendered figures, in document order, as
            {caption, page, image_bytes, media_type, width, height}.
            Only populated when the caller asked for images
            (DoclingParser.parse(..., with_images=True)); empty otherwise.
            Order matches the `<!-- image -->` placeholders in `text`, which
            is what lets the figure stash replace them positionally.
    """
    text: str = ""
    warnings: list[str] = field(default_factory=list)
    page_count: int | None = None
    word_count: int = 0
    structured_doc: Any = None
    pictures: list[dict] = field(default_factory=list)

    def __post_init__(self):
        """Calculate word count from text if not set."""
        if self.text and self.word_count == 0:
            self.word_count = len(self.text.split())


class BaseParser(ABC):
    """Abstract base class for all source parsers.

    Subclasses implement parse() for their specific format.
    The parse_and_validate node dispatches to the correct parser
    based on source_type.
    """

    @abstractmethod
    def parse(self, source: str | Path) -> ParseResult:
        """Parse a source file or raw content into plain text.

        Args:
            source: Either a file path (str or Path) to the source
                material, or raw string content (for free text).

        Returns:
            ParseResult with extracted text and any warnings.

        Raises:
            FileNotFoundError: If source is a path that doesn't exist.
            ValueError: If source content is empty or unparseable.
        """
        ...

    def _validate_file_exists(self, path: str | Path) -> Path:
        """Check that a file path exists and return as Path object."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Source file not found: {p}")
        if not p.is_file():
            raise ValueError(f"Source path is not a file: {p}")
        return p

    # Typographic ligatures. PDF text layers frequently encode these as
    # single codepoints, and Docling emits them verbatim — so "The first"
    # arrives as "Therst" and "financially" as "nancially" once the glyph
    # is dropped by the font mapping. Left alone they corrupt the text the
    # LLM reasons over, corrupt every source_excerpt derived from it, and
    # break the exact-match `parsed_text.find(excerpt)` that source_span
    # anchoring relies on.
    _LIGATURES = str.maketrans({
        "\ufb00": "ff", "\ufb01": "fi", "\ufb02": "fl",
        "\ufb03": "ffi", "\ufb04": "ffl", "\ufb05": "st", "\ufb06": "st",
    })

    # A trailing run of base64-ish characters with no whitespace, sitting in
    # the footer region. Vendor CTI PDFs use these as per-download tracking
    # watermarks; the one seen decoded to the report id plus a
    # user-id hash plus the SUBSCRIBING CUSTOMER'S DOMAIN. Since bundles
    # exist to be shared, that is a data-leak vector, not just noise.
    _WATERMARK_RE = re.compile(r"\n\s*[A-Za-z0-9+/=_-]{80,}\s*$")

    # Words whose "fi"/"fl" glyph the PDF text layer DROPPED rather than
    # encoded. Docling never emits U+FB01 for these fonts — the letters are
    # simply absent from the extracted text — so expansion has nothing to
    # work on and the damage has to be repaired lexically.
    #
    # Curated rather than dictionary-driven on purpose: a generic
    # "insert fi/fl anywhere and see if it becomes a word" pass needs a
    # lexicon we do not ship, and would rewrite legitimate strings
    # (hostnames, hashes, command fragments). This list covers the words
    # that actually occur in CTI prose; anything else is reported as a
    # warning instead of being guessed at.
    _DROPPED_LIGATURE_WORDS = {
        "identied": "identified", "identies": "identifies",
        "identy": "identify", "unidentied": "unidentified",
        "nancially": "financially", "nancial": "financial",
        "condential": "confidential", "condence": "confidence",
        "condent": "confident", "conguration": "configuration",
        "congured": "configured", "congure": "configure",
        "modied": "modified", "modies": "modifies",
        "veried": "verified", "verication": "verification",
        "classied": "classified", "notied": "notified",
        "specic": "specific", "specically": "specically",
        "signicant": "significant", "signicantly": "significantly",
        "articial": "artificial", "benecial": "beneficial",
        "prole": "profile", "proles": "profiles",
        "le": "file", "les": "files", "lename": "filename",
        "lesystem": "filesystem", "lepath": "filepath",
        "exltrate": "exfiltrate", "exltration": "exfiltration",
        "ltering": "filtering", "ltered": "filtered", "lter": "filter",
        "rst": "first", "therst": "the first",
        "agged": "flagged", "agging": "flagging",
        "reected": "reflected", "reect": "reflect",
        "conict": "conflict", "inuence": "influence",
        "briey": "briefly", "chiey": "chiefly",
    }

    # Residual suspicious tokens: a lowercase run containing no f where one
    # of "fi"/"fl" plausibly belongs is hard to detect generically, so we
    # only flag the shape we know — a word that becomes a known word when
    # "fi" is reinserted is already handled above.
    _LIGATURE_WORD_RE = re.compile(r"\b[A-Za-z]{2,}\b")

    def _normalize_ligatures(self, text: str) -> str:
        """Repair typographic ligatures lost in PDF text extraction.

        Two distinct failures, both producing the same symptom:

        1. The extractor emits the ligature CODEPOINT (U+FB01 "ﬁ"). Expanding
           it is a straight translate.
        2. The extractor DROPS the glyph, so "identified" arrives as
           "identied" and "The first" as "Therst". Nothing is there to
           expand — the letters must be put back.

        Case 2 is what Docling does on some vendor PDFs, and it is the one
        that matters: it corrupts the text the LLM reasons over, corrupts
        every source_excerpt derived from it, and breaks the exact-match
        find() that source_span anchoring depends on.
        """
        text = text.translate(self._LIGATURES)

        def _repair(match: "re.Match[str]") -> str:
            word = match.group()
            fixed = self._DROPPED_LIGATURE_WORDS.get(word.lower())
            if fixed is None:
                return word
            # Short entries ("le" for "file", "rst" for "first") are the ones
            # that actually occur in this damage, but they are also the ones
            # that could corrupt a path or identifier — `C:\le\x`,
            # `a-rst-b`. Only repair them when they sit in plain prose.
            if len(word) <= 3:
                start, end = match.start(), match.end()
                before = text[start - 1] if start else " "
                after = text[end] if end < len(text) else " "
                if before in "\\/.-_=:" or after in "\\/.-_=:":
                    return word
            # Preserve the original capitalization of the first letter.
            if word[0].isupper():
                return fixed[0].upper() + fixed[1:]
            return fixed

        return self._LIGATURE_WORD_RE.sub(_repair, text)

    def _strip_tracking_watermark(self, text: str) -> tuple[str, str | None]:
        """Remove a trailing base64-ish tracking watermark.

        Returns ``(text, warning)`` — ``warning`` is None when nothing was
        stripped, so callers can surface the removal in ``parse_warnings``
        rather than silently mutating the document.
        """
        match = self._WATERMARK_RE.search(text)
        if not match:
            return text, None
        blob = match.group().strip()
        return (
            text[: match.start()].rstrip(),
            f"stripped {len(blob)}-char trailing tracking watermark "
            f"(starts {blob[:16]!r})",
        )

    def _normalize_whitespace(self, text: str) -> str:
        """Clean up whitespace without losing paragraph structure.

        - Collapses runs of 3+ newlines into 2 (preserves paragraphs)
        - Strips trailing whitespace from lines
        - Strips leading/trailing whitespace from the full text
        """
        import re
        # Strip trailing whitespace per line
        text = re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)
        # Collapse 3+ newlines into double newline (paragraph break)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
