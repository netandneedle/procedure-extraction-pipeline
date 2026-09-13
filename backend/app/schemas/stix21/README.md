# STIX 2.1 JSON Schemas (vendored)

Source: https://github.com/oasis-open/cti-stix2-json-schemas (`schemas/`)
License: BSD-3-Clause, © OASIS Open — see [LICENSE](LICENSE).

## Why these are vendored rather than pip-installed

The obvious way to validate STIX would be the `stix2` library. It is not
usable here: `stix2` depends on `stix2-patterns`, which requires
`antlr4-python3-runtime~=4.13`, while docling's `omegaconf` pins it at
`4.9.3`. That is the *same* conflict that got `mitreattack-python` removed
(see the note in `backend/requirements.txt`) — installing `stix2` upgrades
antlr and breaks the PDF parser, i.e. the whole ingestion path.

Validating the raw schemas with `jsonschema` (already present, and now a
declared dependency) gets full STIX 2.1 structural validation with no new
transitive dependencies at all.

## What is and isn't covered

`app.services.stix_schema` routes each object by `type`:

- **Stock STIX types** (`sdos/`, `sros/`, `observables/`) → the OASIS schema
  for that type. 22 emitted types verified against these.
- **`x-procedure`** → `../x_procedure_v3.json`, which is stricter
  (`additionalProperties: false`).
- **ATT&CK Flow objects** (`attack-flow`, `attack-operator`,
  `attack-condition`) → no schema here. They are CTID Attack Flow extension
  objects, not core STIX, so OASIS publishes nothing for them. The
  hand-rolled structural checks in `serialization._validate_schema` still
  cover them.

## Updating

Re-copy `schemas/` from the upstream repo. Cross-file `$ref`s resolve by
`$id`, so keep the directory layout intact — do not flatten it.
