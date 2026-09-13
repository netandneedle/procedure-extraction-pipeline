"""Tests for the figure stash — the on-disk handoff that lets
extract_figures skip a second full Docling conversion.

Covers three seams:
  1. The stash service itself (save / load / clear, and its guards).
  2. parse_and_validate writing the stash.
  3. extract_figures reading it, and falling back when it's absent.
"""

import base64
import io
import json
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from app.config import settings
from app.graph.state import SourceType
from app.nodes.deterministic.parse import parse_and_validate
from app.nodes.deterministic.parsers.docling_parser import (
    MAX_IMAGE_DIMENSION_PX,
    extract_pictures_from_doc,
)
from app.nodes.llm.figure_extraction import extract_figures
from app.services import figure_stash


# ── Fixtures / helpers ───────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolated_upload_dir(tmp_path, monkeypatch):
    """Point the stash at a temp dir so tests never touch /tmp/pipeline."""
    monkeypatch.setattr(settings, "upload_dir", str(tmp_path / "uploads"))
    yield


def _png_bytes(width=40, height=30, color=(200, 30, 30)):
    """A real PNG so the b64 round-trip is exercised for real."""
    buf = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buf, format="PNG")
    return buf.getvalue()


def _picture(caption="Attack chain", page=3, raw=None, width=40, height=30):
    """A picture dict in the `image_bytes` shape the stash consumes."""
    return {
        "caption": caption,
        "page": page,
        "image_bytes": _png_bytes(width, height) if raw is None else raw,
        "media_type": "image/png",
        "width": width,
        "height": height,
    }


# ── Service: save / load round-trip ──────────────────────────────────


class TestFigureStashRoundTrip:

    def test_save_then_load_preserves_order_and_metadata(self):
        pics = [
            _picture(caption="first", page=1),
            _picture(caption="second", page=7),
            _picture(caption="third", page=9),
        ]
        assert figure_stash.save("src-1", pics) == 3

        loaded = figure_stash.load("src-1")
        assert loaded is not None
        assert [p["caption"] for p in loaded] == ["first", "second", "third"]
        assert [p["page"] for p in loaded] == [1, 7, 9]

    def test_loaded_image_b64_decodes_to_the_original_bytes(self):
        raw = _png_bytes(color=(1, 2, 3))
        figure_stash.save("src-1", [_picture(raw=raw)])

        loaded = figure_stash.load("src-1")
        assert base64.b64decode(loaded[0]["image_b64"]) == raw

    def test_load_returns_none_when_no_stash_exists(self):
        """None (not []) is the signal to fall back to a re-parse."""
        assert figure_stash.load("never-stashed") is None

    def test_save_replaces_a_stale_stash(self):
        figure_stash.save("src-1", [_picture(caption=f"old-{i}") for i in range(4)])
        figure_stash.save("src-1", [_picture(caption="new")])

        loaded = figure_stash.load("src-1")
        assert [p["caption"] for p in loaded] == ["new"]
        # The old PNGs are gone too, not just unreferenced.
        assert sorted(p.name for p in figure_stash.stash_dir("src-1").iterdir()) == [
            "fig-001.png", "manifest.json",
        ]

    def test_clear_removes_the_stash(self):
        figure_stash.save("src-1", [_picture()])
        figure_stash.clear("src-1")
        assert figure_stash.load("src-1") is None

    def test_clear_is_a_noop_when_absent(self):
        figure_stash.clear("never-stashed")  # must not raise


# ── Service: guards and degraded inputs ──────────────────────────────


class TestFigureStashGuards:

    @pytest.mark.parametrize("bad_id", [
        "../../etc",
        "a/b",
        "..",
        "",
        "with space",
        "x" * 65,
    ])
    def test_unsafe_source_id_is_refused(self, bad_id):
        """source_id becomes a directory name; it must not escape the root."""
        assert figure_stash.save(bad_id, [_picture()]) == 0
        assert figure_stash.load(bad_id) is None
        assert figure_stash.stash_dir(bad_id) is None

    def test_traversal_attempt_writes_nothing_outside_the_root(self):
        figure_stash.save("../escape", [_picture()])
        root = figure_stash.stash_root()
        assert not (root.parent / "escape").exists()

    def test_empty_picture_list_stashes_nothing(self):
        assert figure_stash.save("src-1", []) == 0
        assert figure_stash.load("src-1") is None

    def test_picture_without_bytes_round_trips_as_empty_b64(self):
        """A figure Docling couldn't render is kept as a positional
        placeholder — dropping it would misalign every later figure."""
        figure_stash.save("src-1", [
            _picture(caption="renders fine"),
            _picture(caption="no bytes", raw=b""),
        ])
        loaded = figure_stash.load("src-1")
        assert len(loaded) == 2
        assert loaded[1]["image_b64"] == ""
        assert loaded[1]["caption"] == "no bytes"

    def test_missing_png_on_disk_degrades_to_empty_b64(self):
        figure_stash.save("src-1", [_picture(), _picture(caption="second")])
        (figure_stash.stash_dir("src-1") / "fig-001.png").unlink()

        loaded = figure_stash.load("src-1")
        assert len(loaded) == 2
        assert loaded[0]["image_b64"] == ""
        assert loaded[1]["image_b64"] != ""

    def test_corrupt_manifest_reads_as_no_stash(self):
        figure_stash.save("src-1", [_picture()])
        (figure_stash.stash_dir("src-1") / "manifest.json").write_text("{not json")
        assert figure_stash.load("src-1") is None

    def test_malformed_manifest_shape_reads_as_no_stash(self):
        figure_stash.save("src-1", [_picture()])
        (figure_stash.stash_dir("src-1") / "manifest.json").write_text(
            json.dumps({"version": 1, "pictures": "not-a-list"})
        )
        assert figure_stash.load("src-1") is None


# ── extract_pictures_from_doc ────────────────────────────────────────


def _mock_pic(image=None, caption=None, page=None):
    pic = MagicMock()
    pic.get_image.return_value = image
    if caption is None:
        pic.captions = []
    else:
        cap = MagicMock()
        cap.resolve.return_value = MagicMock(text=caption)
        pic.captions = [cap]
    if page is None:
        pic.prov = []
    else:
        pic.prov = [MagicMock(page_no=page)]
    return pic


class TestExtractPicturesFromDoc:

    def test_harvests_caption_page_and_bytes(self):
        doc = MagicMock()
        doc.pictures = [_mock_pic(Image.new("RGB", (50, 40)), caption="Fig 1", page=4)]

        pics = extract_pictures_from_doc(doc)

        assert len(pics) == 1
        assert pics[0]["caption"] == "Fig 1"
        assert pics[0]["page"] == 4
        assert pics[0]["width"] == 50 and pics[0]["height"] == 40
        assert pics[0]["image_bytes"].startswith(b"\x89PNG")

    def test_oversized_image_is_downscaled(self):
        big = Image.new("RGB", (MAX_IMAGE_DIMENSION_PX * 2, MAX_IMAGE_DIMENSION_PX))
        doc = MagicMock()
        doc.pictures = [_mock_pic(big)]

        pics = extract_pictures_from_doc(doc)

        assert max(pics[0]["width"], pics[0]["height"]) <= MAX_IMAGE_DIMENSION_PX

    def test_unrenderable_figure_keeps_its_slot(self):
        """Order is load-bearing: a failed render must not shift later
        figures onto the wrong `<!-- image -->` placeholder."""
        doc = MagicMock()
        broken = _mock_pic()
        broken.get_image.side_effect = RuntimeError("no raster")
        doc.pictures = [broken, _mock_pic(Image.new("RGB", (20, 20)), caption="ok")]

        pics = extract_pictures_from_doc(doc)

        assert len(pics) == 2
        assert pics[0]["image_bytes"] == b""
        assert pics[1]["caption"] == "ok"

    def test_document_without_pictures_yields_empty_list(self):
        doc = MagicMock()
        doc.pictures = None
        assert extract_pictures_from_doc(doc) == []


# ── parse_and_validate writes the stash ──────────────────────────────


def _mock_docling_result(markdown, pictures=()):
    doc = MagicMock()
    doc.export_to_markdown.return_value = markdown
    doc.pages = [MagicMock()]
    doc.tables = []
    doc.furniture = MagicMock()
    doc.furniture.children = []
    doc.pictures = list(pictures)
    result = MagicMock()
    result.document = doc
    return result


@pytest.fixture(autouse=True)
def _reset_shared_docling():
    from app.nodes.deterministic.parse import _docling
    _docling.reset()
    yield
    _docling.reset()


class TestParseStashesFigures:

    @patch("docling.document_converter.DocumentConverter")
    def test_pdf_parse_renders_and_stashes_figures(self, mock_cls, tmp_path):
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        mock_cls.return_value.convert.return_value = _mock_docling_result(
            "Prose about LockBit exploiting ActiveMQ. <!-- image --> More prose here.",
            pictures=[_mock_pic(Image.new("RGB", (50, 40)), caption="Chain", page=2)],
        )

        result = parse_and_validate({
            "source_id": "src-parse-1",
            "source_type": SourceType.PDF.value,
            "raw_content_path": str(pdf),
            "extract_figures": True,
        })

        assert "error" not in result
        stashed = figure_stash.load("src-parse-1")
        assert stashed is not None and len(stashed) == 1
        assert stashed[0]["caption"] == "Chain"

    @patch("docling.document_converter.DocumentConverter")
    def test_extract_figures_disabled_skips_image_generation(self, mock_cls, tmp_path):
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        mock_cls.return_value.convert.return_value = _mock_docling_result(
            "Prose about LockBit exploiting ActiveMQ over many words here.",
            pictures=[_mock_pic(Image.new("RGB", (50, 40)))],
        )

        parse_and_validate({
            "source_id": "src-parse-2",
            "source_type": SourceType.PDF.value,
            "raw_content_path": str(pdf),
            "extract_figures": False,
        })

        assert figure_stash.load("src-parse-2") is None

    @patch("docling.document_converter.DocumentConverter")
    def test_disabled_run_clears_a_stale_stash(self, mock_cls, tmp_path):
        """Re-running a source with figures switched off must not leave the
        previous run's stash for a later run to pick up."""
        figure_stash.save("src-parse-3", [_picture(caption="from an older run")])
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        mock_cls.return_value.convert.return_value = _mock_docling_result(
            "Prose about LockBit exploiting ActiveMQ over many words here."
        )

        parse_and_validate({
            "source_id": "src-parse-3",
            "source_type": SourceType.PDF.value,
            "raw_content_path": str(pdf),
            "extract_figures": False,
        })

        assert figure_stash.load("src-parse-3") is None

    def test_text_source_stashes_nothing(self, tmp_path):
        txt = tmp_path / "note.md"
        txt.write_text("# Report\n\n" + "APT28 used PowerShell for execution. " * 10)

        parse_and_validate({
            "source_id": "src-parse-4",
            "source_type": SourceType.MARKDOWN.value,
            "raw_content_path": str(txt),
            "extract_figures": True,
        })

        assert figure_stash.load("src-parse-4") is None

    @patch("docling.document_converter.DocumentConverter")
    def test_parse_survives_an_unwritable_stash(self, mock_cls, tmp_path, monkeypatch):
        """A stash failure costs a downstream re-parse, never the source."""
        monkeypatch.setattr(
            figure_stash, "save", MagicMock(side_effect=OSError("disk full"))
        )
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        mock_cls.return_value.convert.return_value = _mock_docling_result(
            "Prose about LockBit exploiting ActiveMQ over many words here. <!-- image -->",
            pictures=[_mock_pic(Image.new("RGB", (50, 40)))],
        )

        result = parse_and_validate({
            "source_id": "src-parse-5",
            "source_type": SourceType.PDF.value,
            "raw_content_path": str(pdf),
            "extract_figures": True,
        })

        assert result["status"] != "failed"
        assert "LockBit" in result["parsed_text"]


# ── extract_figures reads the stash ──────────────────────────────────


class TestExtractFiguresUsesStash:

    @pytest.mark.asyncio
    @patch("app.nodes.llm.figure_extraction._classify_and_extract")
    @patch("app.nodes.llm.figure_extraction._extract_pictures")
    async def test_stashed_figures_skip_the_docling_reparse(self, mock_reparse, mock_vision):
        from app.nodes.llm.tool_models import ExtractFigureOutput

        figure_stash.save("src-fig-1", [_picture(caption="Attack chain", width=800, height=600)])
        mock_vision.return_value = ExtractFigureOutput(
            figure_type="diagram",
            extracted_text="1. Initial Access\n2. Execution",
            confidence=0.9,
            rationale="Kill-chain diagram.",
        )

        result = await extract_figures({
            "source_id": "src-fig-1",
            "parsed_text": "Prose <!-- image --> more prose.",
            "raw_content_path": "/data/test.pdf",
            "source_type": "pdf",
            "extract_figures": True,
        })

        mock_reparse.assert_not_called()
        assert "<!-- image -->" not in result["parsed_text"]
        assert "1. Initial Access" in result["parsed_text"]
        assert result["extracted_figures"][0]["status"] == "extracted"

    @pytest.mark.asyncio
    @patch("app.nodes.llm.figure_extraction._classify_and_extract")
    @patch("app.nodes.llm.figure_extraction._extract_pictures")
    async def test_missing_stash_falls_back_to_reparse(self, mock_reparse, mock_vision):
        """Sources parsed before the stash existed must still work."""
        from app.nodes.llm.tool_models import ExtractFigureOutput

        mock_reparse.return_value = [{
            "caption": "From re-parse",
            "page": 1,
            "image_b64": base64.b64encode(_png_bytes()).decode(),
            "media_type": "image/png",
            "width": 800,
            "height": 600,
        }]
        mock_vision.return_value = ExtractFigureOutput(
            figure_type="screenshot",
            extracted_text="C:\\> whoami",
            confidence=0.8,
            rationale="Terminal capture.",
        )

        result = await extract_figures({
            "source_id": "src-never-stashed",
            "parsed_text": "Prose <!-- image --> more prose.",
            "raw_content_path": "/data/test.pdf",
            "source_type": "pdf",
            "extract_figures": True,
        })

        mock_reparse.assert_called_once_with("/data/test.pdf")
        assert "C:\\> whoami" in result["parsed_text"]

    @pytest.mark.asyncio
    @patch("app.nodes.llm.figure_extraction._extract_pictures")
    async def test_stash_order_maps_onto_placeholder_order(self, mock_reparse):
        """Nth stashed figure must land on the Nth placeholder."""
        from app.nodes.llm.tool_models import ExtractFigureOutput

        figure_stash.save("src-fig-2", [
            _picture(caption="one", width=800, height=600),
            _picture(caption="two", width=800, height=600),
        ])

        async def _by_caption(image_b64, media_type, caption):
            return ExtractFigureOutput(
                figure_type="diagram",
                extracted_text=f"content-for-{caption}",
                confidence=0.9,
                rationale="r",
            )

        with patch(
            "app.nodes.llm.figure_extraction._classify_and_extract",
            side_effect=_by_caption,
        ):
            result = await extract_figures({
                "source_id": "src-fig-2",
                "parsed_text": "A <!-- image --> B <!-- image --> C",
                "raw_content_path": "/data/test.pdf",
                "source_type": "pdf",
                "extract_figures": True,
            })

        mock_reparse.assert_not_called()
        text = result["parsed_text"]
        assert text.index("content-for-one") < text.index("content-for-two")
        assert text.index("A ") < text.index("content-for-one") < text.index(" B ")


# ── Real-world key shape + reclamation ───────────────────────────────


class TestFigureStashLifecycle:

    def test_uuid_source_id_is_accepted(self):
        """Pipeline source_ids are UUIDs — the safety regex must not
        reject the only key shape production actually uses."""
        source_id = "75a78ada-40e4-48dc-bade-5010505cc139"
        assert figure_stash.save(source_id, [_picture()]) == 1
        loaded = figure_stash.load(source_id)
        assert loaded is not None and len(loaded) == 1

    @pytest.mark.asyncio
    async def test_delete_source_reclaims_the_stash(self):
        """The stash is keyed by source_id, so deleting the source is the
        only place it gets reclaimed."""
        import uuid as _uuid
        from unittest.mock import AsyncMock

        from app.services import queue as queue_service

        source_id = _uuid.uuid4()
        figure_stash.save(str(source_id), [_picture()])
        assert figure_stash.load(str(source_id)) is not None

        row = MagicMock()
        row.thread_id = None            # skip the checkpoint flush
        row.raw_content_path = ""       # nothing to unlink
        db = AsyncMock()

        with patch.object(
            queue_service, "get_source", AsyncMock(return_value=row)
        ):
            assert await queue_service.delete_source(db, source_id) is True

        assert figure_stash.load(str(source_id)) is None

    @pytest.mark.asyncio
    async def test_delete_source_reclaims_the_surfacing_ledger(self):
        """Feedback surfacings are a plain UUID association with no cascade,
        so deleting the source is the only place they get reclaimed.

        Skipping it is not cosmetic: at one audit, 131 of the 140 source ids
        in `feedback_pattern_surfacings` pointed at sources that no longer
        existed, so 94% of the ledger could not be joined back to anything
        while still counting toward every total taken over the table.
        """
        import uuid as _uuid
        from unittest.mock import AsyncMock

        from app.services import feedback_patterns as fp
        from app.services import queue as queue_service

        source_id = _uuid.uuid4()
        row = MagicMock()
        row.thread_id = None
        row.raw_content_path = ""
        db = AsyncMock()

        reclaim = AsyncMock(return_value=3)
        with patch.object(queue_service, "get_source", AsyncMock(return_value=row)), \
             patch.object(fp, "delete_surfacings_for_source", reclaim):
            assert await queue_service.delete_source(db, source_id) is True

        reclaim.assert_awaited_once()
        assert reclaim.await_args.args[1] == source_id

    @pytest.mark.asyncio
    async def test_delete_source_survives_a_ledger_cleanup_failure(self):
        """A bookkeeping failure must not block the delete the caller asked
        for — the same contract the reviewer-row cleanup above follows."""
        import uuid as _uuid
        from unittest.mock import AsyncMock

        from app.services import feedback_patterns as fp
        from app.services import queue as queue_service

        source_id = _uuid.uuid4()
        row = MagicMock()
        row.thread_id = None
        row.raw_content_path = ""
        db = AsyncMock()

        with patch.object(queue_service, "get_source", AsyncMock(return_value=row)), \
             patch.object(fp, "delete_surfacings_for_source",
                          AsyncMock(side_effect=RuntimeError("db down"))):
            assert await queue_service.delete_source(db, source_id) is True
