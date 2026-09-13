# Security

## Threat model, in one paragraph

This is a **single-user, localhost-only** tool. It has **no authentication and
no rate limiting** on any API route, and it is designed that way: it runs on an
analyst's own machine, every published port is bound to `127.0.0.1`, and the
only client is a browser on the same machine. Do not expose it to a network.
If you put it behind a reverse proxy on a LAN or the internet, you have
removed the only boundary it has, and anyone who can reach port 8000 can
upload files, run LLM calls on your API key, and read or delete everything in
its databases.

## What is protected, and how

- **Secrets stay in process config.** LLM API keys and database passwords are
  read from environment variables (`.env`, gitignored). They are never written
  to the pipeline state, the LangGraph checkpoint, or the Neo4j graph.
- **Databases require a password you chose.** `docker compose up` refuses to
  start until `POSTGRES_PASSWORD` and `NEO4J_PASSWORD` are set. There are no
  defaults to forget to change.
- **Uploads are bounded.** 50 MB cap by default (`UPLOAD_MAX_BYTES`),
  streamed in 1 MB chunks, extension allowlist, and stored under a directory
  the delete path is confined to.
- **Every identifier is a UUID.** Sources, bundles, threads and patterns
  cannot be enumerated by guessing sequential IDs.
- **Graph writes are sanitised and reversible.** Cypher labels come from a
  fixed table, relationship types are pattern-checked, and every property is
  passed as a parameter. Each report's contribution can be undone with
  `docker compose exec -w /app api python -m scripts.undo_graph_source --report-id <id>`.
- **Commits are scanned.** Pre-commit runs gitleaks against
  `.gitleaks.toml`, and CI runs it again on every push.
- **The dev server is loopback too.** Vite serves the UI on `127.0.0.1:5173`
  and proxies to the API; exposing it (`host: true` in `vite.config.js`)
  exposes the unauthenticated API behind it.

## What is *not* protected — read this

- **Prompt injection from the source material.** This tool exists to ingest
  reports about adversary behaviour, which means it feeds adversary-adjacent
  text to a language model and writes the model's output to a database. A
  report crafted to steer the extractor or the AI gate reviewer is a real
  possibility. The mitigations are structural, not complete: every quoted
  piece of evidence is checked against the source text
  (`backend/app/services/grounding.py`), an unsupported quote drops the
  recommendation to low confidence, and **a human decides at every gate**.
  Do not run gates in unattended `auto` mode on sources you do not trust.
- **No authentication.** Stated above; repeated here because it is the
  finding that matters most if the deployment assumption breaks.
- **Validation errors echo paths.** A rejected `raw_content_path` returns the
  resolved filesystem path in the error body. On localhost this tells you
  nothing you did not already know; it is noted for completeness.

## Reporting a vulnerability

Email **sherman@netandneedle.com** with a description and, if you have one, a
reproduction. You will get an acknowledgement within a week. Please do not
open a public issue for anything that could be exploited before it is fixed.
