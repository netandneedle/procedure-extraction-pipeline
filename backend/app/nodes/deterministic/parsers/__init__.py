"""Source format parsers for the parse_and_validate node.

Two parser classes:
- DoclingParser: ML-powered document understanding (PDF, HTML, DOCX, IMAGE)
- TextParser: Lightweight pass-through (plain text, markdown)
"""

from app.nodes.deterministic.parsers.base import BaseParser, ParseResult
from app.nodes.deterministic.parsers.docling_parser import DoclingParser
from app.nodes.deterministic.parsers.text_parser import TextParser

__all__ = [
    "BaseParser",
    "ParseResult",
    "DoclingParser",
    "TextParser",
]
