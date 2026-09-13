"""STIX 2.1 extension definitions the serializer embeds in every bundle.

A custom STIX 2.1 object is only an *extension* if a bundle can resolve the
`extension-definition` it declares. Without the definition object, a strict
consumer sees a pre-2.1-style custom type and may drop it. This module is the
one place those definitions live: the x-procedure definition (ours) and the
CTID Attack Flow 2.0.0 definition (theirs, reproduced verbatim), plus the
identity each one names in `created_by_ref`.

Two facts drive the shape of this file:

  * Attack Flow 2.0.0 uses ONE definition for all five of its object types
    (attack-flow, attack-action, attack-asset, attack-condition,
    attack-operator). Every attack-* object we emit declares the same id.
  * A definition is a published artifact, not a per-run object, so its
    timestamps are fixed. Re-running the pipeline must not mint a "new"
    version of the same extension.

Definitions and their author identities are bundle metadata: the Report SDO
does not list them in `object_refs`, `distribute` never writes them to Neo4j,
and the Explorer does not render them. `bundle_meta_ids` is the shared test.
"""
from __future__ import annotations

X_PROCEDURE_EXTENSION_ID = "extension-definition--b422519e-c47a-439d-9195-0f16b94fa889"
X_PROCEDURE_EXTENSION_VERSION = "0.5.0-draft"  # the version x_procedure_v3.json declares
X_PROCEDURE_AUTHOR_IDENTITY_ID = "identity--f431f809-377b-45e0-aa1c-6a4751cae5ff"

# https://github.com/center-for-threat-informed-defense/attack-flow/blob/main/stix/attack-flow-extension-2.0.0.json
ATTACK_FLOW_EXTENSION_ID = "extension-definition--fb9c968a-745b-4ade-9b25-c324172197f4"
ATTACK_FLOW_AUTHOR_IDENTITY_ID = "identity--d673f8cb-c168-42da-8ed4-0cb26725f86c"

_REPO_URL = "https://github.com/netandneedle/procedure-extraction-pipeline"
_SCHEMA_URL = (
    "https://raw.githubusercontent.com/netandneedle/procedure-extraction-pipeline/"
    "main/backend/app/schemas/x_procedure_v3.json"
)
_X_PROCEDURE_DEFINED = "2026-09-12T00:00:00.000Z"
_AUTHOR_CREATED = "2026-02-09T00:00:00.000Z"
_ATTACK_FLOW_STAMP = "2022-08-02T19:34:35.143Z"

ATTACK_FLOW_TYPES = frozenset({"attack-flow", "attack-operator", "attack-condition"})


def extension_declaration(extension_id: str) -> dict:
    """The `extensions` value an object carries to declare a new-sdo extension."""
    return {extension_id: {"extension_type": "new-sdo"}}


def x_procedure_extension_definition() -> dict:
    return {
        "type": "extension-definition",
        "spec_version": "2.1",
        "id": X_PROCEDURE_EXTENSION_ID,
        "created_by_ref": X_PROCEDURE_AUTHOR_IDENTITY_ID,
        "created": _X_PROCEDURE_DEFINED,
        "modified": _X_PROCEDURE_DEFINED,
        "name": "x-procedure",
        "description": (
            "Defines the x-procedure SDO: a discrete, repeatable technical "
            "implementation of one or more ATT&CK techniques, formalized as the "
            "tuple P = {AP, LS, <C>} of attack patterns, log sources and ordered "
            "component observables. Every x-procedure is a unique observation; "
            "behaviourally equivalent procedures are grouped at query time by "
            "x_fingerprint, never merged at creation."
        ),
        "schema": _SCHEMA_URL,
        "version": X_PROCEDURE_EXTENSION_VERSION,
        "extension_types": ["new-sdo"],
        "external_references": [
            {
                "source_name": "procedure-extraction-pipeline",
                "description": "Reference implementation: the pipeline that emits this object",
                "url": _REPO_URL,
            },
            {
                "source_name": "x-procedure documentation",
                "description": "Every property, the fingerprint, relationships and sequencing",
                "url": f"{_REPO_URL}/blob/main/docs/X_PROCEDURE.md",
            },
        ],
    }


def x_procedure_author_identity() -> dict:
    return {
        "type": "identity",
        "spec_version": "2.1",
        "id": X_PROCEDURE_AUTHOR_IDENTITY_ID,
        "created": _AUTHOR_CREATED,
        "modified": _X_PROCEDURE_DEFINED,
        "name": "Sherman Chu",
        "identity_class": "individual",
        "description": "Author of the x-procedure STIX 2.1 extension.",
    }


def attack_flow_extension_definition() -> dict:
    """CTID's published definition, verbatim (including its timestamps)."""
    return {
        "type": "extension-definition",
        "id": ATTACK_FLOW_EXTENSION_ID,
        "spec_version": "2.1",
        "name": "Attack Flow",
        "description": "Extends STIX 2.1 with features to create Attack Flows.",
        "created": _ATTACK_FLOW_STAMP,
        "modified": _ATTACK_FLOW_STAMP,
        "created_by_ref": ATTACK_FLOW_AUTHOR_IDENTITY_ID,
        "schema": (
            "https://center-for-threat-informed-defense.github.io/attack-flow/"
            "stix/attack-flow-schema-2.0.0.json"
        ),
        "version": "2.0.0",
        "extension_types": ["new-sdo"],
        "external_references": [
            {
                "source_name": "Documentation",
                "description": "Documentation for Attack Flow",
                "url": "https://center-for-threat-informed-defense.github.io/attack-flow",
            },
            {
                "source_name": "GitHub",
                "description": "Source code repository for Attack Flow",
                "url": "https://github.com/center-for-threat-informed-defense/attack-flow",
            },
        ],
    }


def attack_flow_author_identity() -> dict:
    return {
        "type": "identity",
        "spec_version": "2.1",
        "id": ATTACK_FLOW_AUTHOR_IDENTITY_ID,
        "created_by_ref": ATTACK_FLOW_AUTHOR_IDENTITY_ID,
        "created": _ATTACK_FLOW_STAMP,
        "modified": _ATTACK_FLOW_STAMP,
        "name": "MITRE Center for Threat-Informed Defense",
        "identity_class": "organization",
    }


def bundle_meta_ids(objects: list[dict]) -> set[str]:
    """Ids of every extension-definition in `objects` plus the identities they
    name as author. These are bundle metadata, not report content."""
    ids: set[str] = set()
    for obj in objects:
        if obj.get("type") != "extension-definition" or not obj.get("id"):
            continue
        ids.add(obj["id"])
        author = obj.get("created_by_ref")
        if isinstance(author, str) and author:
            ids.add(author)
    return ids
