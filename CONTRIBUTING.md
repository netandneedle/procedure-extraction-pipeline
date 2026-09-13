# Contributing

Thanks for looking. This is a small project with strong opinions; the rules
below are the ones that bite newcomers, and every one of them is enforced by a
test somewhere.

## Setting up

Follow the README Quick Start. For development you also want a host-side
Python environment for the backend suite and the loader script:

```bash
python3.12 -m venv backend/.venv && source backend/.venv/bin/activate
pip install -r backend/requirements.txt
pip install pre-commit && pre-commit install      # gitleaks + hygiene hooks
```

## Running the tests

There are two pytest suites with two configs, and a plain `pytest` runs only
one of them:

```bash
pytest -q -rs                    # tests/        (integration + e2e)
cd backend && pytest -q -rs      # backend/tests/ (unit)
cd frontend && npm test -- --run && npm run build
```

Both suites must pass with zero failures. There is no baseline of expected
failures to subtract; a single red test is a regression. `-rs` prints the
skips: the end-to-end tests skip when Postgres is not reachable, and a suite
that skips them is not the same as a suite that ran them.

The suites never call a model, never write to Neo4j, and never download the
embedding model; all three are pinned off in the two `conftest.py` files.

## Where changes go

**Adding a field to the pipeline.** Three places must agree, and a field added
to only two of them fails at runtime on the first real model call:

1. the tool schema the model is told to emit (`backend/app/nodes/llm/*.py`),
2. the Pydantic validator that accepts it (`backend/app/nodes/llm/tool_models.py`),
3. the state dataclass or `PipelineState` field it flows into (`backend/app/graph/state.py`).

**Adding a pipeline status.** One row in `frontend/src/lib/pipelineStatus.js`.
Every component derives from that table; `tests/test_contracts.py` checks it
against the backend enum in both directions.

**Adding a gate reviewer.** In the same change: the reviewer, its
`PipelineStatus` value, its `_REVIEWING_STATUS` entry, its row in the status
table, and its outcome differ. A reviewer with no differ records no agreement
data; a status with no column makes the card vanish from the board.

**Touching the LLM cache key.** `llm_adapter._build_cache_key` hashes the whole
request. The cache holds real, paid-for responses, and two compatibility
shims keep old keys reachable (the provider key is omitted for Anthropic;
image blocks hash in their legacy shape). Changing either invalidates every
cached response. If that is deliberate, bump `LLM_CACHE_VERSION` in the same
change. `backend/tests/test_llm_providers.py` pins the key format.

**Prompts.** The system prompt is part of the cache key, so editing a prompt
already invalidates its cache. Worked examples inside prompts must be
synthetic: `example.com`, RFC 5737 addresses, made-up hashes. Never paste a
real indicator or a real organisation's contact details into a prompt.

## Domain rules

These are product decisions, not style. Changes that violate them will be
declined.

- **Indicators are STIX Cyber Observables, not Indicators.** Detection rules
  are a different pipeline.
- **No fabricated command lines.** A thin source produces a low-confidence
  procedure, never an invented command. If the report does not contain it,
  the bundle does not either.
- **Bundle validation is all-or-nothing.** Schema, reference integrity and
  Attack Flow integrity all pass or the run fails. Do not add a "warn and
  continue" path.
- **The ATT&CK catalogue is read-only.** The distributor never writes an
  `attack-pattern` or any other catalogue-owned object to the graph, and never
  writes an edge between two catalogue objects. Every node it does write
  carries `x_ingested_by = 'pipeline'`; that marker is what the undo relies on.
- **The reviewer recommends; a human decides.** Nothing the AI reviewer
  produces is applied without an analyst action except in the explicit
  unattended mode, and unattended decisions never count toward the agreement
  readout.

## Style

- Backend: type hints and Pydantic models everywhere; `async` for anything
  that does I/O; all Cypher lives in `backend/app/services/neo4j.py` and the
  distributor, never inline in a node.
- Frontend: functional React, Tailwind classes, no new global state library.
  One definition per UI term lives in `frontend/src/lib/glossary.js`; a
  tooltip that redefines a term fails the static scan.
- Delete dead code rather than commenting it out. Comments say why, not when:
  no dates, ticket numbers or review references.
- Conventional commits (`fix(frontend): ...`, `feat(llm): ...`), one logical
  change per commit.

## Secrets

`pre-commit` runs gitleaks on every commit. Real keys live in `.env`
(gitignored); `.env.example` is the template and must never hold a real
value. If a key does land in a commit, revoke it first and rewrite history
second.

## Licensing of contributions

By submitting a pull request you agree that your contribution is licensed
under the project's Apache-2.0 license. There is no CLA.

## Reporting a vulnerability

See [SECURITY.md](SECURITY.md).
