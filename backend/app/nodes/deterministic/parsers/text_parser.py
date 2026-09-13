"""Text parser for plain text and markdown sources.

Handles: free text (pasted content, notes, terminal output) and
markdown files (.md). Both are already extractable text formats
that don't need ML-powered conversion.

Uses chardet (v7, 2.6k stars) for encoding detection with
language identification. v7 is 47x faster than v6.

Markdown is passed through as-is (no stripping of formatting).
Markdown syntax is human-readable text, and downstream LLM nodes
handle it natively. Code blocks, tables, and formatting are all
preserved verbatim since they often contain command lines and IOCs.
"""

from __future__ import annotations

from pathlib import Path

from app.nodes.deterministic.parsers.base import BaseParser, ParseResult


class TextParser(BaseParser):
    """Process plain text and markdown input with encoding detection.

    Handles both SourceType.FREE_TEXT and SourceType.MARKDOWN.
    For files: detects encoding, reads content, normalizes whitespace.
    For raw strings: normalizes whitespace.
    """

    # Minimum content length to consider valid input
    MIN_CONTENT_LENGTH = 10

    def parse(self, source: str | Path) -> ParseResult:
        """Parse text or markdown from a file or raw string.

        Args:
            source: Path to a text/markdown file, or raw string content.

        Returns:
            ParseResult with normalized text and warnings.

        Raises:
            ValueError: If content is empty or too short.
            FileNotFoundError: If source is a path that doesn't exist.
        """
        warnings: list[str] = []
        text = self._load_text(source, warnings)

        if not text or len(text.strip()) < self.MIN_CONTENT_LENGTH:
            raise ValueError(
                f"Text content too short ({len(text.strip())} chars). "
                f"Minimum {self.MIN_CONTENT_LENGTH} characters required."
            )

        text = self._normalize_ligatures(text)
        text, watermark_warning = self._strip_tracking_watermark(text)
        if watermark_warning:
            warnings.append(watermark_warning)
        text = self._normalize_whitespace(text)

        return ParseResult(text=text, warnings=warnings)

    def _load_text(self, source: str | Path, warnings: list[str]) -> str:
        """Load text from file or accept raw string.

        File detection heuristic: if the source string is short
        (under 260 chars) and contains no newlines, treat it as a
        potential file path. Otherwise it's raw content.
        """
        # Detect raw string early: if the source is long or contains
        # newlines/spaces, it's content not a file path.
        if isinstance(source, str) and (len(source) > 260 or "\n" in source):
            return source

        path = Path(source) if not isinstance(source, Path) else source

        try:
            is_file = path.exists() and path.is_file()
        except OSError:
            # Path too long or invalid characters: treat as raw string
            return str(source)

        if is_file:
            raw_bytes = path.read_bytes()

            if not raw_bytes:
                raise ValueError(f"File is empty: {path}")

            # Encoding detection
            try:
                import chardet
                detected = chardet.detect(raw_bytes)
                encoding = detected.get("encoding", "utf-8") or "utf-8"
                confidence = detected.get("confidence", 0)
                language = detected.get("language")

                if confidence < 0.7:
                    warnings.append(
                        f"Low encoding confidence ({confidence:.0%}). "
                        f"Detected: {encoding}. Using UTF-8."
                    )
                    encoding = "utf-8"
                elif encoding.lower() != "utf-8" and encoding.lower() != "ascii":
                    warnings.append(
                        f"Encoding detected as {encoding} "
                        f"(confidence: {confidence:.0%}), converted to UTF-8."
                    )

                if language:
                    warnings.append(f"Language detected: {language}")

            except ImportError:
                encoding = "utf-8"

            return raw_bytes.decode(encoding, errors="replace")

        # Raw string input
        if isinstance(source, str):
            return source

        raise FileNotFoundError(f"Text source not found: {source}")
