"""On-disk stash of rendered PDF figures, keyed by source_id.

WHY THIS EXISTS:
Docling has to rasterize a document to hand us figure bitmaps, and that
needs `PdfPipelineOptions(generate_picture_images=True)` — pipeline options
the plain text converter doesn't carry. Before this module existed, the
`extract_figures` node therefore ran a SECOND full Docling conversion of the
same file purely to get the images: ~110-120s of duplicated CPU work on
every figure-bearing source.

The stash removes that. `parse_and_validate` now converts once with images
enabled and writes the rendered figures here; `extract_figures` loads them
back instead of re-converting. The node keeps its re-parse path as a
fallback, so a source parsed before this existed (or one whose stash write
failed) still works — just slowly.

LAYOUT:
    {upload_dir}/figures/{source_id}/
        manifest.json     — ordered metadata, one entry per figure
        fig-001.png       — rendered bitmap, already resized
        fig-002.png
        ...

ORDER IS LOAD-BEARING. `extract_figures` replaces the Nth `<!-- image -->`
placeholder in parsed_text with the Nth figure's extracted content, so the
manifest list must stay in the same document order the markdown export used.
Both come from the same conversion now, which is strictly more consistent
than the old two-parse arrangement.

LIFETIME:
The stash is written by parse_and_validate and read by extract_figures a
few nodes later, so its real job is a within-run handoff. It outlives the
run because that costs nothing and buys one genuine case: a graph RESUMED
from its checkpoint restarts at the next pending node, so a run interrupted
between the two nodes (an API restart mid-run, say) picks the figures back
up instead of re-converting. A fresh pipeline run always re-parses and
overwrites — it needs the text anyway — so the stash is not a re-run
optimization.

It is cleared when the source is deleted (`queue.delete_source`), when a
fresh parse overwrites it, and when a source is re-parsed with figure
extraction turned off.

Every operation here is best-effort: a stash that can't be written or read
degrades to the node's re-parse path, never to a failed source.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
from pathlib import Path

from app.config import settings
from app.graph.state import SourceType
from app.nodes.deterministic.parsers.docling_parser import pictures_to_b64

logger = logging.getLogger(__name__)

# Source types where figure extraction is meaningful. Plain markdown / free
# text don't have figures; STIX bundles and tweet URLs aren't supported.
# Lives here rather than in the node so the parse side can consult it
# without importing an LLM node.
FIGURE_BEARING_TYPES = {
    SourceType.PDF.value,
    SourceType.HTML.value,
    SourceType.DOCX.value,
    SourceType.IMAGE.value,
}

MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1

# source_id becomes a directory name, so it must not be able to escape the
# stash root. Pipeline source_ids are UUIDs; tests use short slugs. Anything
# with a separator, a dot-segment, or exotic characters is rejected outright
# rather than sanitized — a caller passing one is a bug, not a user input.
_SAFE_SOURCE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def stash_root() -> Path:
    """Root directory holding every source's figure stash."""
    return Path(settings.upload_dir) / "figures"


def stash_dir(source_id: str) -> Path | None:
    """Directory for one source's figures, or None if source_id is unusable."""
    if not source_id or not _SAFE_SOURCE_ID.match(source_id):
        return None
    return stash_root() / source_id


def save(source_id: str, pictures: list[dict]) -> int:
    """Write `pictures` (the `image_bytes` shape from
    `extract_pictures_from_doc`) to this source's stash, replacing anything
    already there.

    Returns the number of figures written; 0 means nothing was stashed
    (unusable source_id, empty input, or a write failure). Never raises —
    the caller's parse already succeeded and must not fail over a cache miss.
    """
    target = stash_dir(source_id)
    if target is None:
        logger.warning("figure_stash: refusing to stash under unsafe source_id %r", source_id)
        return 0
    if not pictures:
        return 0

    try:
        clear(source_id)
        target.mkdir(parents=True, exist_ok=True)

        entries: list[dict] = []
        for i, pic in enumerate(pictures):
            filename = f"fig-{i + 1:03d}.png"
            raw = pic.get("image_bytes") or b""
            if raw:
                (target / filename).write_bytes(raw)
            entries.append({
                "file": filename if raw else "",
                "caption": pic.get("caption", ""),
                "page": pic.get("page"),
                "media_type": pic.get("media_type", "image/png"),
                "width": pic.get("width", 0),
                "height": pic.get("height", 0),
            })

        manifest = {
            "version": MANIFEST_VERSION,
            "source_id": source_id,
            "pictures": entries,
        }
        (target / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2))
    except OSError as e:
        logger.warning("figure_stash: save failed for source %s: %s", source_id, e)
        return 0

    logger.info("figure_stash: stashed %d figure(s) for source %s", len(entries), source_id)
    return len(entries)


def load(source_id: str) -> list[dict] | None:
    """Load this source's stashed figures in the `image_b64` shape the
    vision node consumes.

    Returns None when there is no usable stash — the caller should fall back
    to re-parsing. Returns a (possibly empty) list when the stash exists.
    A figure whose PNG is missing comes back with empty `image_b64`, which
    the node already treats as a soft per-figure failure.
    """
    target = stash_dir(source_id)
    if target is None:
        return None

    manifest_path = target / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as e:
        if manifest_path.exists():
            logger.warning("figure_stash: unreadable manifest for source %s: %s", source_id, e)
        return None

    entries = manifest.get("pictures")
    if not isinstance(entries, list):
        logger.warning("figure_stash: malformed manifest for source %s", source_id)
        return None

    raw_pictures: list[dict] = []
    for entry in entries:
        filename = entry.get("file") or ""
        image_bytes = b""
        if filename:
            try:
                image_bytes = (target / filename).read_bytes()
            except OSError as e:
                logger.warning(
                    "figure_stash: missing figure %s for source %s: %s",
                    filename, source_id, e,
                )
        raw_pictures.append({
            "caption": entry.get("caption", ""),
            "page": entry.get("page"),
            "image_bytes": image_bytes,
            "media_type": entry.get("media_type", "image/png"),
            "width": entry.get("width", 0),
            "height": entry.get("height", 0),
        })

    return pictures_to_b64(raw_pictures)


def clear(source_id: str) -> None:
    """Remove this source's stash. No-op when absent. Never raises."""
    target = stash_dir(source_id)
    if target is None:
        return
    try:
        shutil.rmtree(target)
    except FileNotFoundError:
        pass
    except OSError as e:
        logger.warning("figure_stash: failed to clear stash for source %s: %s", source_id, e)


__all__ = [
    "FIGURE_BEARING_TYPES",
    "clear",
    "load",
    "save",
    "stash_dir",
    "stash_root",
]
