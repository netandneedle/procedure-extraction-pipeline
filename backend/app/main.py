"""FastAPI application entry point.

Startup sequence:
1. Initialize PostgreSQL tables (Source model)
2. Create LangGraph checkpointer (PostgresSaver)
3. Compile pipeline graph with checkpointer + interrupt config
4. Store graph and checkpointer for dependency injection

Shutdown:
1. Dispose database engine
2. Close Neo4j driver
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import inspect as sa_inspect, text
from sqlalchemy.schema import CreateColumn

# Configure root logger so pipeline node logs (INFO+) reach the console.
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-7s %(name)s: %(message)s",
)

from app.api.dependencies import set_checkpointer, set_graph
from app.api.routes import (
    bundles,
    feedback_patterns,
    gates,
    pipeline,
    reviewer,
    source_queue,
    techniques,
    ws,
)
from app.graph.checkpointer import get_checkpointer, close_checkpointer
from app.graph.pipeline import compile_pipeline
from app.models.base import Base, engine
from app.services.neo4j import close as close_neo4j


def _check_model_config() -> None:
    """Warn about model configurations that fail silently rather than loudly.

    Both cases below produce a pipeline that runs fine and reports numbers that
    mean less than they appear to. Neither is worth refusing to boot over —
    browsing existing bundles does not need a working LLM config — so these are
    warnings, and the same problems surface again at the first LLM call.
    """
    from app.config import settings
    from app.nodes.llm import llm_adapter
    from app.nodes.llm.providers import get_provider

    log = logging.getLogger("startup.models")

    # Both roles resolve through the SAME function call_llm and the reviewer
    # runner use, so what is checked here is what actually runs. The reviewer
    # used to be logged but never validated: a typo in REVIEWER_PROVIDER, or
    # the Anthropic default REVIEWER_MODEL under LLM_PROVIDER=openai, booted
    # clean and then silently turned every assist-mode gate into plain human
    # review (run_reviewer fails soft by design).
    # `_target` suffix: `reviewer` is the routes module imported above.
    extraction_target = llm_adapter.resolve_target("extraction")
    reviewer_target = llm_adapter.resolve_target("reviewer")

    # 1. A reviewer identical to the extractor agrees with itself. The
    #    agreement readout then reports a healthy-looking number that measures
    #    nothing — the failure mode the whole readout exists to avoid.
    if (
        reviewer_target.provider == extraction_target.provider
        and reviewer_target.model == extraction_target.model
    ):
        log.warning(
            "Gate reviewer and extraction are the same model (%s on %s). The "
            "reviewer is meant to bring judgment extraction did not, so it "
            "will largely agree with itself and the Reviewer tab's agreement "
            "rates will not mean what they appear to. Set REVIEWER_MODEL to a "
            "stronger, different model.",
            extraction_target.model, extraction_target.provider,
        )

    # 2. Per role: the provider must exist (and its SDK must import), and
    #    LLM_EFFORT on a model with no reasoning knob is inert — it was read,
    #    accepted, and silently discarded. A provider's own foreign-model-ID
    #    warning fires here too, at boot, instead of at the first gate.
    efforts: dict[str, str] = {}
    for target in (extraction_target, reviewer_target):
        try:
            provider = get_provider(target.provider)
        except (ValueError, ImportError) as exc:
            log.error("%s %s calls will fail until this is fixed.", exc, target.role)
            continue
        params = provider.resolve_params(
            target.model, temperature=0.0, effort=target.effort,
        )
        efforts[target.role] = params.effort or "n/a"
        if params.effort is None and target.effort:
            log.warning(
                "LLM_EFFORT=%r has no effect for %s: %s on %s exposes no "
                "reasoning-effort knob. The setting is being discarded, not "
                "applied.",
                target.effort, target.role, target.model, target.provider,
            )

    # 3. The configured provider's key. Without it every model call fails,
    #    but only at the first call — after a run has already parsed the
    #    document — with the same error this line gives at boot. The check
    #    stays a log line, not a refusal to boot, for the reason in the
    #    docstring: browsing bundles needs no key.
    _KEY_FOR_PROVIDER = {
        "anthropic": settings.anthropic_api_key,
        "openai": settings.openai_api_key,
    }
    for provider_name in sorted({extraction_target.provider, reviewer_target.provider}):
        if provider_name in _KEY_FOR_PROVIDER and not _KEY_FOR_PROVIDER[provider_name]:
            log.error(
                "%s_API_KEY is not set but the %s provider is configured for %s. "
                "Every model call on it will fail until it is set.",
                provider_name.upper(), provider_name,
                " and ".join(
                    t.role for t in (extraction_target, reviewer_target)
                    if t.provider == provider_name
                ),
            )

    log.info(
        "LLM config: extraction=%s/%s (effort=%s) reviewer=%s/%s (effort=%s)",
        extraction_target.provider, extraction_target.model,
        efforts.get("extraction", "n/a"),
        reviewer_target.provider, reviewer_target.model,
        efforts.get("reviewer", "n/a"),
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle."""
    # ── Startup ──────────────────────────────────────────────────────
    _check_model_config()

    # Create tables that don't exist yet. There is no separate migration
    # runner: a fresh database gets its whole schema here, and the loop below
    # adds columns to tables that predate them.
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # Auto-migrate: add any columns defined in models but missing from
    # existing tables. create_all only creates new tables; it won't
    # ALTER existing ones. This loop inspects each model table and
    # issues ALTER TABLE ADD COLUMN for any gaps, using SQLAlchemy's
    # DDL compiler so types, defaults, and nullability render correctly.
    migrate_logger = logging.getLogger("startup.migrate")

    def _run_auto_migrate(sync_conn):
        insp = sa_inspect(sync_conn)
        dialect = sync_conn.dialect
        for table in Base.metadata.sorted_tables:
            if not insp.has_table(table.name):
                continue
            existing = {c["name"] for c in insp.get_columns(table.name)}
            for col in table.columns:
                if col.name not in existing:
                    # Use CreateColumn to render the column definition
                    # (type, NULL/NOT NULL, DEFAULT) via the dialect
                    # compiler, then wrap in ALTER TABLE.
                    col_def = CreateColumn(col.copy()).compile(dialect=dialect)
                    stmt = f"ALTER TABLE {table.name} ADD COLUMN IF NOT EXISTS {col_def}"
                    migrate_logger.info("auto-migrate: %s", stmt)
                    sync_conn.execute(text(stmt))

    async with engine.begin() as conn:
        await conn.run_sync(_run_auto_migrate)

    # Initialize LangGraph checkpointer
    checkpointer = await get_checkpointer()
    set_checkpointer(checkpointer)

    # Compile pipeline with checkpointer. The gate-interrupt list lives in
    # compile_pipeline() so adding a new gate is a single-file change.
    graph = compile_pipeline(checkpointer=checkpointer)
    set_graph(graph)

    yield

    # ── Shutdown ─────────────────────────────────────────────────────
    from app.nodes.llm.providers import close_providers

    await close_providers()
    await close_checkpointer()
    await engine.dispose()
    await close_neo4j()


app = FastAPI(
    title="Procedure Extraction Pipeline",
    description="ATT&CK procedure extraction from threat intelligence to validated STIX 2.1 bundles",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS is pinned to the two local dev-server origins. Keep it that way.
# allow_credentials=True plus a broad origin list (or "*") would let any page
# a browser visits call this API with the user's cookies — and there is no
# authentication on any route, so "the user's cookies" is the only thing
# standing between a malicious web page and this pipeline. If the origin
# list ever needs to grow, drop allow_credentials or add auth first.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(source_queue.router, prefix="/api/sources", tags=["Source Queue"])
app.include_router(bundles.router, prefix="/api/bundles", tags=["Bundles"])
app.include_router(feedback_patterns.router, prefix="/api/feedback-patterns", tags=["Feedback Patterns"])
app.include_router(gates.router, prefix="/api/gates", tags=["Gates"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["Pipeline"])
app.include_router(reviewer.router, prefix="/api/reviewer", tags=["AI Reviewer"])
app.include_router(techniques.router, prefix="/api/techniques", tags=["Techniques"])
app.include_router(ws.router, tags=["WebSocket"])


@app.get("/health")
async def health():
    return {"status": "ok"}
