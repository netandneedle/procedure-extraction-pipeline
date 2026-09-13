# CLAUDE.md

Automated ATT&CK procedure extraction: threat reports in, validated STIX 2.1
bundles with `x-procedure` objects out, with four human review gates. Python
(FastAPI + LangGraph) backend, React frontend, Postgres + Neo4j.

@docs/ARCHITECTURE.md
@CONTRIBUTING.md

Two test suites, two configs: `pytest` at the repo root runs `tests/` only;
`cd backend && pytest` runs `backend/tests/`. Both must pass. There is no
lint or typecheck step.
