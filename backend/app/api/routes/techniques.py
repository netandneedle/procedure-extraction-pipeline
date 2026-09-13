"""Technique catalogue endpoint.

Serves the full ATT&CK technique catalogue (excluding revoked/deprecated)
for the technique editor's search/autocomplete at the technique-review gate.

The catalogue is loaded once from app.services.attack_data (direct JSON
parse of the STIX 2.1 bundle at settings.attack_stix_path) and cached
at module level. Subsequent requests return the cached list instantly.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

logger = logging.getLogger(__name__)

router = APIRouter()

# Module-level cache
_catalogue_cache: list[dict] | None = None


def _load_catalogue() -> list[dict]:
    """Load and cache the full ATT&CK technique catalogue."""
    global _catalogue_cache

    if _catalogue_cache is not None:
        return _catalogue_cache

    try:
        from app.services.attack_data import get_attack_data
        db = get_attack_data()
        all_techniques = db.all_techniques()
    except Exception as e:
        logger.warning(
            "techniques: attack_data load failed (%s: %s)",
            type(e).__name__, e,
        )
        return []

    catalogue = []
    for tech in all_techniques:
        if tech.get("revoked") or tech.get("deprecated"):
            continue
        tid = tech.get("external_id", "")
        if not tid:
            continue
        catalogue.append({
            "technique_id": tid,
            "name": tech.get("name", ""),
            "stix_id": tech.get("stix_id", ""),
            "description": tech.get("description", ""),
            "tactics": tech.get("tactics", []),
            "platforms": tech.get("platforms", []),
        })

    catalogue.sort(key=lambda t: t["technique_id"])
    _catalogue_cache = catalogue

    logger.info("techniques: cached %d active techniques", len(catalogue))
    return catalogue


@router.get(
    "/",
    summary="Get full ATT&CK technique catalogue",
    response_model=list[dict],
)
async def get_techniques():
    """Return the full active ATT&CK technique catalogue.

    Used by the technique editor at the technique-review gate.
    Excludes revoked and deprecated techniques. Cached after first load.
    """
    return _load_catalogue()
