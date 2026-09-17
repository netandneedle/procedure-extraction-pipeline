# Software bill of materials

Everything this pipeline is built from, pulls in, or calls out to. A package
manifest covers less than half of it: two of the largest components are
models downloaded on first use, the ATT&CK catalogue is fetched by a script,
and the runtime talks to exactly one model API. So this document lists all of
it, in one place, with the license and the point at which each component
enters the system.

The human-readable tables below are the curated view. The last section says
how to regenerate machine-readable files for a scanner.

**How this was produced (2026-09-12).** Versions are the ones resolved inside
the API image built from `backend/Dockerfile` on that date (`docker run --rm
--entrypoint pip <image> freeze`), and inside `frontend/package-lock.json`.
`requirements.txt` declares floors, not pins, so a build on a later date
resolves newer versions; the floor column is the contract, the resolved column
is a snapshot. Licenses from `pip-licenses` and `npx license-checker`, checked
against each project's own repository where the two disagreed.

## 1. Runtime images

| Image | Role | License |
|---|---|---|
| `python:3.12-slim` | Base of the API image (`backend/Dockerfile`); adds `libpq-dev`, `gcc`, `libxcb1`, `libgl1`, `libglib2.0-0` for the Postgres driver and Docling | PSF (Python), Debian packages under their own licenses |
| `postgres:16-alpine` | Source queue, pipeline checkpoints, LLM response cache, bundles, learned rules | PostgreSQL License |
| `neo4j:5-community` | ATT&CK catalogue and, when writes are enabled, the procedure graph | GPLv3 (the server, run unmodified as a separate service; nothing links against it) |
| APOC plugin | Pulled by the Neo4j image on first boot (`NEO4J_PLUGINS: '["apoc"]'`); used by the loader and the log-source query | Apache 2.0 |

## 2. Python, direct dependencies

Declared in `backend/requirements.txt`. Test dependencies ship in the image on
purpose so the container can run the suite.

| Package | Floor | Resolved | License | Purpose |
|---|---|---|---|---|
| fastapi | ≥0.115.0 | 0.141.1 | MIT | HTTP API |
| uvicorn[standard] | ≥0.34.0 | 0.52.4 | BSD-3-Clause | ASGI server |
| pydantic | ≥2.10.0 | 2.13.5 | MIT | Every schema and model-output validator |
| pydantic-settings | ≥2.7.0 | 2.15.0 | MIT | `Settings` from environment and `.env` |
| langgraph | ≥0.3.0 | 1.2.11 | MIT | The pipeline state machine |
| langgraph-checkpoint-postgres | ≥2.0.0 | 3.1.2 | MIT | Checkpoints every node's state to Postgres |
| anthropic | ≥0.42.0 | 1.4.0 | MIT | Default model provider SDK |
| openai | ≥2.0.0 | 3.10.0 | Apache-2.0 | OpenAI-compatible provider SDK |
| asyncpg | ≥0.30.0 | 0.31.0 | Apache-2.0 | Async Postgres driver (SQLAlchemy) |
| sqlalchemy[asyncio] | ≥2.0.0 | 2.0.52 | MIT | ORM for the queue, bundles, rules |
| psycopg[binary] | ≥3.2.0 | 3.3.5 | LGPL-3.0 | Postgres driver used by the LangGraph checkpointer |
| neo4j | ≥5.27.0 | 6.3.0 | Apache-2.0 | Graph driver |
| jsonschema | ≥4.20.0 | 4.26.0 | MIT | STIX and `x-procedure` schema validation |
| docling | ≥2.86.0 | 2.126.0 | MIT | PDF, HTML and DOCX to Markdown, figure rendering |
| chardet | ≥5.0.0 | 7.6.0 | 0BSD / LGPL-2.1 (see note) | Text-encoding detection for plain-text sources |
| python-dotenv | ≥1.0.0 | 1.2.3 | BSD-3-Clause | `.env` loading |
| httpx | ≥0.28.0 | 0.28.1 | BSD-3-Clause | HTTP client (health check, tests) |
| python-multipart | ≥0.0.18 | 0.0.32 | Apache-2.0 | File upload parsing |
| websockets | ≥14.0 | 16.1.1 | BSD-3-Clause | Live status to the UI |
| sentence-transformers | ≥3.0.0 | 6.0.1 | Apache-2.0 | Runs the technique-retrieval embedding model |
| numpy | ≥1.26.0 | 2.5.3 | BSD-3-Clause | Cosine similarity over stored embeddings |
| pytest | ≥8.0.0 | 9.1.1 | MIT | Tests |
| pytest-asyncio | ≥0.23.0 | 1.4.0 | Apache-2.0 | Tests |

Two LGPL packages are in the tree: `psycopg` (this table) and `python-bidi`
(a Docling dependency). Both are used as ordinary libraries through their
public interfaces and are not modified; that use is what the LGPL permits
without further obligation. `chardet` reports `0BSD` in its package metadata;
older releases were LGPL-2.1, so both are listed. No GPL or AGPL package is
present in the Python tree.

## 3. Python, transitive dependencies worth knowing about

Not declared directly, but they decide image size and behavior.

| Package | Resolved | License | Why it is here |
|---|---|---|---|
| torch | 2.14.0 | BSD-3-Clause | Backing tensor library for the embedding model; most of the image's size |
| transformers | 5.16.1 | Apache-2.0 | Model loading for `sentence-transformers` and Docling |
| huggingface-hub | 1.30.0 | Apache-2.0 | Downloads and caches the models in §5 |
| safetensors | 0.8.0 | Apache-2.0 | Model weight format |
| docling-core / docling-parse | 2.95.0 / 7.18.0 | MIT | Docling's document model and PDF parser |
| pillow | 12.3.0 | MIT-CMU | Figure images |
| langchain-core | 1.6.2 | MIT | A `langgraph` dependency; not used directly |
| langgraph-checkpoint | 4.2.0 | MIT | Checkpointer base |
| pydantic-core | 2.46.5 | MIT | Pydantic's engine |

The full resolved list is 172 packages; `pip freeze` inside the image is the
authoritative record for any given build.

## 4. Node, direct dependencies

Declared in `frontend/package.json`, resolved in `package-lock.json`.

| Package | Range | Resolved | License | Purpose |
|---|---|---|---|---|
| react, react-dom | ^18.3.0 | 18.3.1 | MIT | UI |
| @xyflow/react | ^12.10.2 | 12.10.2 | MIT | The three graph canvases |
| @hello-pangea/dnd | ^17.0.0 | 17.0.0 | Apache-2.0 | Kanban drag and drop |
| @mdi/js | ^7.4.47 | 7.4.47 | Apache-2.0 | Icon paths (attributed in `frontend/THIRD_PARTY.md`) |
| axios | ^1.7.0 | 1.20.0 | MIT | API client |
| dagre | ^0.8.5 | 0.8.5 | MIT | Graph layout |
| vite (dev) | ^6.0.0 | 6.4.3 | MIT | Build and dev server |
| vitest (dev) | ^4.1.11 | 4.1.11 | MIT | Tests |
| tailwindcss, @tailwindcss/vite (dev) | ^4.0.0 | 4.2.2 | MIT | Styling |
| @vitejs/plugin-react (dev) | ^4.3.0 | 4.7.0 | MIT | React fast refresh |

`npx license-checker --summary --production` over the resolved tree: MIT 55,
ISC 8, Apache-2.0 2, BSD-3-Clause 1, plus the app itself. The production
bundle contains no runtime call to any third-party host.

## 5. Models downloaded at runtime

None of these ship in the repository or the image. They are fetched from the
Hugging Face Hub the first time they are needed and cached in the `hf-cache`
Docker volume (`/root/.cache/huggingface` in the container), so they survive
image rebuilds and are downloaded once per machine.

| Model | Size on disk | License | When it downloads | Purpose |
|---|---|---|---|---|
| `cisco-ai/SecureBERT2.0-biencoder` | ~570 MB | Apache-2.0 | First technique-extraction call, with the default `TECHNIQUE_RETRIEVER=embedding`; never with `token_overlap` | Ranks the ATT&CK catalogue against each chunk; also embeds learned rules for the feedback loop |
| `docling-project/docling-models` | ~340 MB | MIT | First document conversion | Docling's layout and table-structure models |
| `docling-project/docling-layout-heron` | ~165 MB | MIT | First document conversion | Docling's page-layout model |

Sizes are what the cache held after one full run on 2026-09-09. The
SecureBERT figure is smaller than the "~1.5 GB" some older comments quote;
that number was the full repository, the cache keeps only the weights it
loads.

## 6. Data downloaded at runtime

| Data | Size | Source | License / terms | How it enters |
|---|---|---|---|---|
| ATT&CK Enterprise v19.2, STIX 2.1 bundle | 54 MB | `mitre-attack/attack-stix-data` on GitHub | [ATT&CK Terms of Use](https://attack.mitre.org/resources/legal-and-branding/terms-of-use/); ATT&CK® is a registered trademark of The MITRE Corporation | `scripts/fetch_attack.sh` writes it to `data/attack/`; the API reads it directly; `scripts/load_attack.py` loads it into Neo4j. The version is pinned by `attack_stix_path` in `backend/app/config.py` and must match the graph. |

## 7. Vendored in the repository

| Component | Location | License | Notes |
|---|---|---|---|
| OASIS STIX 2.1 JSON schemas | `backend/app/schemas/stix21/` (18 common, 18 observable, 19 SDO, 2 SRO files) | BSD-3-Clause, © OASIS Open | From `oasis-open/cti-stix2-json-schemas`. Vendored because the `stix2` library's `antlr4` pin conflicts with Docling's; the directory README explains. |
| `x-procedure` schema | `backend/app/schemas/x_procedure_v3.json` | Apache-2.0 (this project) | See [X_PROCEDURE.md](X_PROCEDURE.md) |
| IBM Plex Sans, IBM Plex Mono | `frontend/public/fonts/` (woff2, Google Fonts builds, unmodified) | SIL Open Font License 1.1 | Self-hosted so the UI makes no request to a font CDN |
| CISA advisory AA23-061A | `data/samples/` | United States Government work, public domain | The Quick Start sample; see the README beside it |
| Pipeline diagram | `docs/pipeline-flow.mmd` / `.png` | Apache-2.0 (this project) | |

## 8. External services the running system calls

| Service | When | What leaves the machine |
|---|---|---|
| Anthropic API, or the OpenAI-compatible endpoint named by `OPENAI_BASE_URL` | Every model call, for whichever provider `LLM_PROVIDER` selects; only one is ever configured | The report text, the extracted state the prompt needs, and figure images when figure extraction is on. Keys come from `.env` and are never written to state, checkpoints or the graph. |
| Hugging Face Hub (`huggingface.co`) | First run only, for the models in §5 | Nothing but the download request |
| GitHub raw content | Only when `scripts/fetch_attack.sh` runs | Nothing but the download request |
| LangSmith | Only if `LANGCHAIN_TRACING_V2=true`, which is off by default | Traces of every model call |

Nothing else. The frontend proxies to the local API only, the fonts are local,
and no telemetry of any kind is sent. The [SECURITY.md](../SECURITY.md) threat
model assumes every listener is on loopback.

## 9. Regenerating a machine-readable SBOM

The tables above are hand-curated so they can say *why* a component is present.
For a scanner, generate CycloneDX from the same manifests:

```bash
# Python: from the resolved environment, not the floors
docker run --rm --entrypoint pip <api-image> freeze > /tmp/frozen.txt
pip install cyclonedx-bom && cyclonedx-py requirements /tmp/frozen.txt -o sbom-backend.cdx.json

# Node
cd frontend && npm sbom --sbom-format cyclonedx --omit dev > ../sbom-frontend.cdx.json
```

Neither file is committed. If they are ever added, regenerate both on every
dependency change, since a stale SBOM is worse than none.
