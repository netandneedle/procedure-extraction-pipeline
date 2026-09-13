"""Flat-dict facade over an ATT&CK STIX 2.1 bundle, loaded as JSON.

Deliberately NOT backed by mitreattack-python.MitreAttackData. That pulls
in stix2 + stix2-patterns + antlr 4.13, which conflicts at runtime with
docling's omegaconf (omegaconf ships ATN v3 grammar files; antlr 4.13
expects v4). Direct JSON parsing instead — fewer deps, no version
conflict, simpler.

If we ever need mitreattack-python's richer helpers (Navigator layers,
diff_stix, group/campaign relationship traversal), they'll live in a
separate optional module that's loaded only when those features are
invoked, and we'll deal with the antlr conflict then. For our current
usage (technique catalogue + procedure-example matcher) the stdlib
json module is sufficient.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from app.config import settings

logger = logging.getLogger(__name__)


class AttackData:
    """Loads an ATT&CK STIX 2.1 bundle as JSON and exposes a flat-dict
    catalogue + procedure examples.

    Construction parses the full ~50 MB JSON once and indexes objects
    by id for fast ref resolution. Callers should reuse a single
    instance via get_attack_data().
    """

    def __init__(self, stix_filepath: str) -> None:
        self._stix_filepath = stix_filepath
        with Path(stix_filepath).open(encoding="utf-8") as f:
            self._bundle = json.load(f)
        # Index objects by stix_id for O(1) ref resolution
        self._objects_by_id: dict[str, dict] = {
            obj.get("id"): obj
            for obj in self._bundle.get("objects", [])
            if obj.get("id")
        }
        self._techniques_cache: list[dict[str, Any]] | None = None
        self._procedure_examples_cache: list[dict[str, Any]] | None = None
        # Built lazily on first call to revoked_by_target / is_deprecated /
        # _get_technique_record. Indexes ALL techniques (active + revoked
        # + deprecated) by external_id, plus the revoked-by redirect map.
        self._techniques_by_tid: dict[str, dict[str, Any]] | None = None
        self._revoked_by_map: dict[str, str] | None = None

    def all_techniques(self) -> list[dict[str, Any]]:
        """Return all techniques as flat dicts. Cached after first call.

        Returned shape:
            {external_id, name, stix_id, description, tactics, platforms,
             revoked, deprecated}
        """
        if self._techniques_cache is not None:
            return self._techniques_cache

        result: list[dict[str, Any]] = []
        for obj in self._bundle.get("objects", []):
            if obj.get("type") != "attack-pattern":
                continue
            external_id = self._first_attack_external_id(obj)
            if not external_id:
                continue
            result.append({
                "external_id": external_id,
                "name": obj.get("name", ""),
                "stix_id": obj.get("id", ""),
                "description": obj.get("description", ""),
                "tactics": self._tactics(obj),
                "platforms": list(obj.get("x_mitre_platforms", []) or []),
                "revoked": bool(obj.get("revoked", False)),
                "deprecated": bool(obj.get("x_mitre_deprecated", False)),
            })

        logger.info(
            "attack_data: loaded %d techniques from %s",
            len(result), self._stix_filepath,
        )
        self._techniques_cache = result
        return result

    def get_procedure_examples(self) -> list[dict[str, Any]]:
        """Return all procedure examples across the catalogue.

        Walks all 'uses' relationships in the bundle, resolves source
        and target refs, and returns flat dicts:

            {technique_id, technique_stix_id, source_actor_name,
             source_actor_type, description}

        Skips relationships with empty descriptions or non-technique
        targets (e.g. actor-uses-malware edges, which we don't surface
        as procedure examples).
        """
        if self._procedure_examples_cache is not None:
            return self._procedure_examples_cache

        # Pre-build target_ref -> external_id lookup so per-relationship
        # resolution is O(1).
        techniques_by_stix: dict[str, str] = {}
        for obj in self._bundle.get("objects", []):
            if obj.get("type") != "attack-pattern":
                continue
            ext_id = self._first_attack_external_id(obj)
            if ext_id:
                techniques_by_stix[obj.get("id", "")] = ext_id

        result: list[dict[str, Any]] = []
        skipped_no_description = 0
        skipped_unknown_target = 0
        skipped_unknown_source = 0

        for obj in self._bundle.get("objects", []):
            if obj.get("type") != "relationship":
                continue
            if obj.get("relationship_type") != "uses":
                continue

            target_ref = obj.get("target_ref", "")
            technique_id = techniques_by_stix.get(target_ref)
            if not technique_id:
                skipped_unknown_target += 1
                continue  # not a technique target (e.g. actor uses malware)

            description = obj.get("description", "") or ""
            if not description.strip():
                skipped_no_description += 1
                continue

            source_ref = obj.get("source_ref", "")
            source_obj = self._objects_by_id.get(source_ref)
            if not source_obj:
                skipped_unknown_source += 1
                continue

            result.append({
                "technique_id": technique_id,
                "technique_stix_id": target_ref,
                "source_actor_name": source_obj.get("name", ""),
                "source_actor_type": source_obj.get("type", ""),
                "description": description,
            })

        logger.info(
            "attack_data: loaded %d procedure examples "
            "(skipped: %d no-description, %d non-technique-target, %d unknown-source)",
            len(result), skipped_no_description,
            skipped_unknown_target, skipped_unknown_source,
        )
        self._procedure_examples_cache = result
        return result

    # =========================================================================
    # Revoked / deprecated post-validation helpers
    # =========================================================================
    #
    # MITRE periodically revokes or deprecates techniques: revoked = renamed
    # or replaced (has a `revoked-by` relationship pointing at the new ID);
    # deprecated = no longer recommended but no replacement (analyst handles
    # at gate review). The LLM may pick a stale T-number from training
    # knowledge even when our prompt only lists active techniques, because
    # extract_techniques relies on training memory rather than full-catalogue
    # injection. These helpers let _resolve_stix_ids redirect or flag those
    # picks instead of silently dropping them.

    def revoked_by_target(self, tid: str) -> str | None:
        """If `tid` is a revoked technique, return the T-number that
        replaces it via STIX `revoked-by` relationship.

        Returns None for active or unknown tids, or for revoked tids with
        no recorded replacement target. Single-hop only — does not chase
        chains of redirects (real ATT&CK shouldn't ship those, and chasing
        adds complexity for an edge case).
        """
        self._ensure_revoked_map_built()
        assert self._revoked_by_map is not None
        return self._revoked_by_map.get(tid)

    def is_deprecated(self, tid: str) -> bool:
        """True if `tid` is deprecated (x_mitre_deprecated=True). Deprecated
        techniques have no replacement; they survive in the catalogue but
        analysts should flag them at review.
        """
        entry = self.get_technique_record(tid)
        return bool(entry and entry.get("deprecated"))

    def validate_technique_ids(
        self, ids: list[str]
    ) -> tuple[set[str], list[dict[str, Any]]]:
        """Filter LLM-proposed T-IDs against the v19 catalogue before they
        enter the candidate pool.

        Use case: the C+A+D technique-extraction flow has a "reason +
        propose" LLM call that emits T-IDs from training memory. Those
        IDs may be hallucinated (don't exist), stale (revoked since the
        model's training cutoff), or deprecated. This helper turns a
        raw proposal list into a usable candidate set + an audit log.

        Resolution per ID:
            1. Active in catalogue            -> kept ('kept_active')
            2. Revoked WITH redirect target   -> swapped for the redirect
                                                target ('redirected')
            3. Revoked WITHOUT redirect       -> dropped
                                                ('dropped_revoked_no_redirect')
            4. Deprecated                     -> dropped from candidate pool
                                                ('dropped_deprecated')
            5. Not in catalogue at all        -> dropped ('dropped_hallucinated')

        Deprecated picks are dropped here rather than kept-with-flag because
        the candidate pool feeds the LLM's pick step; we don't want
        deprecated IDs back in front of the LLM. The existing
        `_resolve_stix_ids` post-processing path still flags deprecated
        picks when the LLM picks them despite the pool exclusion (training
        memory leakage), so analyst-visibility on dead-end picks is
        preserved at gate review.

        Args:
            ids: T-IDs to validate (e.g., ['T1059.001', 'T9999', 'T1086']).

        Returns:
            (valid_ids, audit) where:
            - valid_ids: set[str] of T-IDs safe to add to the candidate
              pool. Includes redirected targets (post-redirect IDs, not
              originals).
            - audit: list of {original_id, outcome, result_id?, note?}
              dicts, one per input. Useful for logging and observability.
        """
        valid: set[str] = set()
        audit: list[dict[str, Any]] = []

        for raw_id in ids:
            tid = (raw_id or "").strip()
            if not tid:
                audit.append({
                    "original_id": raw_id,
                    "outcome": "dropped_hallucinated",
                    "note": "empty or whitespace",
                })
                continue

            record = self.get_technique_record(tid)

            # Path 5: not in catalogue at all
            if record is None:
                audit.append({
                    "original_id": tid,
                    "outcome": "dropped_hallucinated",
                    "note": "not present in catalogue",
                })
                continue

            # Path 1: active
            is_revoked = bool(record.get("revoked"))
            is_dep = bool(record.get("deprecated"))
            if not is_revoked and not is_dep:
                valid.add(tid)
                audit.append({"original_id": tid, "outcome": "kept_active"})
                continue

            # Path 2: revoked WITH redirect. Follow the redirect CHAIN
            # (ATT&CK occasionally chains revokes: T_old -> T_mid -> T_new)
            # until we land on an active technique, guarding against cycles
            # and runaway chains.
            if is_revoked:
                final = None
                seen_redirects: set[str] = {tid}
                cursor = self.revoked_by_target(tid)
                hops = 0
                while cursor and cursor not in seen_redirects and hops < 10:
                    seen_redirects.add(cursor)
                    hops += 1
                    target_record = self.get_technique_record(cursor)
                    if target_record is None:
                        break  # dangling redirect target
                    if not target_record.get("revoked") \
                            and not target_record.get("deprecated"):
                        final = cursor  # landed on an active technique
                        break
                    if target_record.get("deprecated"):
                        break  # chain ends in a deprecated tech — unusable
                    # target itself revoked: keep following.
                    cursor = self.revoked_by_target(cursor)
                if final:
                    valid.add(final)
                    audit.append({
                        "original_id": tid,
                        "outcome": "redirected",
                        "result_id": final,
                    })
                    continue
                # Path 3: revoked but no usable redirect (incl. cyclic or
                # chain that never reaches an active technique).
                audit.append({
                    "original_id": tid,
                    "outcome": "dropped_revoked_no_redirect",
                })
                continue

            # Path 4: deprecated (drops from pool but is_deprecated may
            # surface later if LLM picks it from training memory anyway)
            audit.append({
                "original_id": tid,
                "outcome": "dropped_deprecated",
            })

        return valid, audit

    def get_technique_record(self, tid: str) -> dict[str, Any] | None:
        """Return the full technique record (active, revoked, OR deprecated)
        for `tid`, or None if no such T-number exists. O(1) keyed-by-tid
        version of all_techniques(); returns the same record dicts but
        looked up directly rather than scanned. Caller can read fields like
        `deprecated`, `revoked`, `name`, `stix_id` without a second call.
        """
        self._ensure_techniques_by_tid_built()
        assert self._techniques_by_tid is not None
        return self._techniques_by_tid.get(tid)

    def _ensure_techniques_by_tid_built(self) -> None:
        if self._techniques_by_tid is not None:
            return
        index: dict[str, dict[str, Any]] = {}
        for entry in self.all_techniques():
            tid = entry.get("external_id")
            if tid:
                index[tid] = entry
        self._techniques_by_tid = index

    def _ensure_revoked_map_built(self) -> None:
        if self._revoked_by_map is not None:
            return
        # Build stix_id -> external_id reverse lookup. Walks attack-pattern
        # objects directly (rather than all_techniques()) so revoked entries
        # whose external_id might be filtered out elsewhere still resolve.
        stix_uuid_to_tid: dict[str, str] = {}
        for obj in self._bundle.get("objects", []):
            if obj.get("type") != "attack-pattern":
                continue
            ext_id = self._first_attack_external_id(obj)
            stix_uuid = obj.get("id", "")
            if ext_id and stix_uuid:
                stix_uuid_to_tid[stix_uuid] = ext_id

        redirects: dict[str, str] = {}
        for obj in self._bundle.get("objects", []):
            if obj.get("type") != "relationship":
                continue
            if obj.get("relationship_type") != "revoked-by":
                continue
            old_tid = stix_uuid_to_tid.get(obj.get("source_ref", ""))
            new_tid = stix_uuid_to_tid.get(obj.get("target_ref", ""))
            if old_tid and new_tid:
                redirects[old_tid] = new_tid

        logger.info(
            "attack_data: built revoked-by map (%d redirects) from %s",
            len(redirects), self._stix_filepath,
        )
        self._revoked_by_map = redirects

    # =========================================================================
    # Internal helpers
    # =========================================================================

    @staticmethod
    def _first_attack_external_id(obj: dict[str, Any]) -> str:
        """Return the first 'mitre-attack' external_id (T-number) on the
        object, or empty string if none."""
        for ref in obj.get("external_references", []) or []:
            if ref.get("source_name") == "mitre-attack":
                ext_id = ref.get("external_id")
                if ext_id:
                    return ext_id
        return ""

    @staticmethod
    def _tactics(obj: dict[str, Any]) -> list[str]:
        """Extract tactic short_names from kill_chain_phases. Filters to
        the mitre-attack kill chain (some bundles include others)."""
        return [
            phase.get("phase_name", "")
            for phase in obj.get("kill_chain_phases", []) or []
            if phase.get("kill_chain_name") == "mitre-attack"
            and phase.get("phase_name")
        ]


# =============================================================================
# Module-level singleton
# =============================================================================

_instance: AttackData | None = None


def get_attack_data() -> AttackData:
    """Return the process-wide AttackData singleton.

    First call loads + parses the STIX bundle (slow, a few hundred ms);
    subsequent calls are free.
    """
    global _instance
    if _instance is None:
        _instance = AttackData(settings.attack_stix_path)
    return _instance
