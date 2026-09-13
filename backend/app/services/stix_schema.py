"""Full STIX 2.1 schema validation for assembled bundles.

WHY THIS EXISTS:
With no real schema loaded, `serialization._validate_schema` was only a
hand-rolled required-fields check, and two contract violations lived in the
serializer undetected for months (`x_observable_refs` embedded on procedures
despite `additionalProperties: false`, and fabricated `attack-pattern--T1190`
refs that aren't valid STIX identifiers). This module closes that gap: the
schemas become load-bearing instead of decorative.

WHY NOT THE `stix2` LIBRARY:
`stix2` pulls `stix2-patterns`, which requires `antlr4-python3-runtime~=4.13`;
docling's `omegaconf` pins antlr at `4.9.3`. Installing it upgrades antlr and
breaks the PDF parser — the same conflict that got `mitreattack-python`
removed (see backend/requirements.txt). Validating the vendored OASIS
schemas with `jsonschema` needs no new transitive dependencies.

ROUTING (see `validate_object`):
    x-procedure                    -> app/schemas/x_procedure_v3.json
    stock STIX types               -> app/schemas/stix21/{sdos,sros,observables}
    attack-flow / -operator /
      -condition                   -> not validated here (CTID extension
                                      objects; OASIS publishes no schema).
                                      The structural checks in
                                      serialization._validate_schema still
                                      apply to them.
    anything else custom (x-*)     -> skipped

DEGRADED MODE:
If the schema corpus can't be loaded, `validate_object` returns no errors
and logs once at ERROR. A packaging mistake must not fail every bundle —
bundle validation is all-or-nothing and hard-fails the source, so a broken
validator would take the whole pipeline down. The hand-rolled structural
checks remain as the floor.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SCHEMA_ROOT = Path(__file__).resolve().parents[1] / "schemas"
_STIX21_ROOT = _SCHEMA_ROOT / "stix21"
_X_PROCEDURE_SCHEMA = _SCHEMA_ROOT / "x_procedure_v3.json"

# Custom object types we deliberately do not schema-check here. ATT&CK Flow
# is a CTID extension, not core STIX, so OASIS ships no schema for it.
UNVALIDATED_CUSTOM_TYPES = frozenset({
    "attack-flow", "attack-operator", "attack-condition",
})

X_PROCEDURE_TYPE = "x-procedure"

# Built once, then reused. Loading parses 57 JSON files and builds a
# reference registry; doing that per bundle would be wasteful.
_lock = threading.Lock()
_state: dict[str, Any] | None = None


def _build() -> dict[str, Any]:
    """Load every schema and index it by STIX type. Never raises."""
    result: dict[str, Any] = {"by_type": {}, "registry": None, "ok": False}
    try:
        import jsonschema
        from referencing import Registry, Resource
        from referencing.jsonschema import DRAFT202012

        # Every schema is registered by its $id so the cross-file $refs in
        # the OASIS corpus (common/ -> sdos/ -> observables/) resolve.
        resources = []
        for path in _STIX21_ROOT.rglob("*.json"):
            try:
                doc = json.loads(path.read_text())
            except (OSError, ValueError) as e:
                logger.warning("stix_schema: skipping unreadable %s: %s", path.name, e)
                continue
            if isinstance(doc, dict) and doc.get("$id"):
                resources.append(
                    (doc["$id"], Resource(contents=doc, specification=DRAFT202012))
                )

        by_type: dict[str, dict] = {}
        for sub in ("sdos", "sros", "observables"):
            for path in (_STIX21_ROOT / sub).rglob("*.json"):
                try:
                    by_type[path.stem] = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue
        # common/ holds shared fragments (core, properties, vocabularies),
        # not object schemas — except extension-definition, which is a real
        # meta object the serializer embeds and so must be checked too.
        try:
            by_type["extension-definition"] = json.loads(
                (_STIX21_ROOT / "common" / "extension-definition.json").read_text()
            )
        except (OSError, ValueError) as e:
            logger.error("stix_schema: extension-definition schema unreadable: %s", e)

        try:
            by_type[X_PROCEDURE_TYPE] = json.loads(_X_PROCEDURE_SCHEMA.read_text())
        except (OSError, ValueError) as e:
            logger.error("stix_schema: x-procedure schema unreadable: %s", e)

        if not by_type:
            logger.error(
                "stix_schema: no schemas loaded from %s — schema validation "
                "degrades to the hand-rolled structural checks", _STIX21_ROOT,
            )
            return result

        result["by_type"] = by_type
        result["registry"] = Registry().with_resources(resources)
        result["validator_cls"] = jsonschema.Draft202012Validator
        result["ok"] = True
        logger.info(
            "stix_schema: loaded %d type schemas (%d refs registered)",
            len(by_type), len(resources),
        )
    except ImportError as e:
        logger.error(
            "stix_schema: jsonschema/referencing unavailable (%s) — schema "
            "validation degrades to the hand-rolled structural checks", e,
        )
    except Exception as e:  # noqa: BLE001 — a broken validator must not fail every bundle
        logger.exception("stix_schema: failed to build schema index: %s", e)
    return result


def _get() -> dict[str, Any]:
    global _state
    if _state is None:
        with _lock:
            if _state is None:
                _state = _build()
    return _state


def is_available() -> bool:
    """True when the schema corpus loaded and validation is really running."""
    return bool(_get().get("ok"))


def reset_cache() -> None:
    """Drop the loaded corpus. For tests."""
    global _state
    with _lock:
        _state = None


def validate_object(obj: dict) -> list[str]:
    """Validate one STIX object against its schema.

    Returns a list of human-readable error strings; empty means valid (or
    that this type is deliberately not schema-checked). Never raises.
    """
    state = _get()
    if not state.get("ok"):
        return []

    obj_type = obj.get("type", "")
    if not obj_type or obj_type in UNVALIDATED_CUSTOM_TYPES:
        return []

    schema = state["by_type"].get(obj_type)
    if schema is None:
        # Unknown custom type — nothing to check it against. Stock types all
        # have schemas, so this only fires for x-* types we don't own.
        if not obj_type.startswith("x-"):
            logger.warning(
                "stix_schema: no schema for stock STIX type %r — not validated",
                obj_type,
            )
        return []

    obj_id = obj.get("id", "<no id>")
    try:
        validator = state["validator_cls"](schema, registry=state["registry"])
        errors = []
        for err in validator.iter_errors(obj):
            path = "/".join(str(p) for p in err.absolute_path) or "(root)"
            errors.append(f"{obj_id}: {path}: {err.message}")
        return errors
    except Exception as e:  # noqa: BLE001 — see DEGRADED MODE
        logger.warning(
            "stix_schema: validator error on %s (%s) — treating as valid",
            obj_id, e,
        )
        return []


def validate_bundle_objects(objects: list[dict]) -> list[str]:
    """Validate every object in a bundle. Returns all error strings."""
    errors: list[str] = []
    for obj in objects:
        errors.extend(validate_object(obj))
    return errors


def industry_sector_vocab() -> tuple[str, ...]:
    """The STIX 2.1 `industry-sector-ov` values, read from the bundled schema.

    The vocabulary is declared in `sdos/identity.json` under
    `definitions.industry-sector-ov`, but the `sectors` property never $refs
    it — it is `items: {type: string}` — which is why out-of-vocabulary values
    validate cleanly. Reading the definition directly gives one source of
    truth for both the extraction prompt and the bundle-validator coercion,
    instead of the hand-maintained prose list that had drifted from it
    (`defence` vs `defense`, `pharmaceutical` vs `pharmaceuticals`, plus
    `maritime`/`media`/`real-estate` which are not STIX values at all).
    """
    try:
        doc = json.loads((_STIX21_ROOT / "sdos" / "identity.json").read_text())
        values = doc["definitions"]["industry-sector-ov"]["enum"]
        return tuple(str(v) for v in values)
    except (OSError, ValueError, KeyError, TypeError):  # pragma: no cover
        logger.warning(
            "stix_schema: could not read industry-sector-ov from identity.json",
        )
        return ()
