from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # PostgreSQL. Docker Compose sets DATABASE_URL from POSTGRES_* in the
    # root .env. Outside Docker the backend reads backend/.env and this
    # default is only a local-dev placeholder: set DATABASE_URL there.
    database_url: str = "postgresql+asyncpg://pipeline:pipeline_dev@localhost:5432/pipeline"

    # Neo4j. Compose passes NEO4J_PASSWORD through; outside Docker set it in
    # backend/.env. There is no working default password on purpose.
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = ""

    # Neo4j write control. False keeps the graph read-only (the ATT&CK
    # catalogue stays untouched and `distribute` is a no-op); True lets a
    # completed bundle land in the graph, reversibly per report via
    # backend/scripts/undo_graph_source.py.
    neo4j_writes_enabled: bool = False

    # LLM keys. Keys stay process-level config (env / backend/.env) — there is
    # no per-user key storage, because there are no users. Only the key for the
    # provider(s) actually in use needs to be set.
    openai_api_key: str = ""
    anthropic_api_key: str = ""
    # Only for an Anthropic key created at the ORGANIZATION level (not inside
    # a workspace): the API rejects such keys with 400 "This API key is not
    # scoped to a workspace" unless every request names one. A key created
    # inside a workspace needs nothing here. Value: the workspace id from
    # https://console.anthropic.com/settings/workspaces (wrkspc_...).
    anthropic_workspace_id: str = ""
    # Point the "openai" provider at an OpenAI-COMPATIBLE endpoint instead of
    # api.openai.com — Azure OpenAI, OpenRouter, Together, Groq, vLLM, Ollama,
    # LM Studio. Empty means the SDK default. This is why the openai provider
    # is built on /v1/chat/completions rather than the Responses API.
    openai_base_url: str = ""

    # --- Model selection -----------------------------------------------------
    #
    # This pipeline runs TWO tiers on purpose, and an operator bringing their
    # own key must set both:
    #
    #   llm_model      extraction. Called ~9x per source (entities, figures,
    #                  chunking, technique propose+pick, drafting, feedback
    #                  synthesis). This is where the bill lands. A mid-frontier
    #                  model is the right tier; the top tier did not measurably
    #                  improve extraction when compared head to head.
    #   reviewer_model the AI gate reviewer. Called once per enabled gate, and
    #                  should be a STRONGER and DIFFERENT model. See the note
    #                  on reviewer_model below for why sameness is corrosive.
    #
    # Deliberately not a cross-vendor equivalence table: model lineups move
    # faster than this repo does. The requirement is a property (mid vs top
    # reasoning tier, distinct from each other), not a specific ID.
    llm_provider: str = "anthropic"     # see app.nodes.llm.providers.KNOWN_PROVIDERS
    # LLM_MODEL is the only model variable, for every provider. Per-vendor
    # variables (ANTHROPIC_MODEL and friends) were removed: they overrode the
    # model for EVERY provider, so a stale export sent a Claude ID to OpenAI,
    # and the startup same-model check could not see them.
    llm_model: str = "claude-sonnet-5"
    # Reasoning effort on models that expose the knob. "high" is the API
    # default and right for most work; "max" trades cost for the hardest
    # cases. Inert on models without a reasoning tier.
    llm_effort: str = "high"

    # Bump to invalidate every cached LLM response at once. A real env var
    # beats .env (pydantic-settings' default precedence), so a single run can
    # be re-sampled with `LLM_CACHE_VERSION=x uvicorn ...` without editing the
    # file. There is intentionally no TTL — see the comment in llm_adapter.
    llm_cache_version: str = "1"

    # LangSmith (optional)
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "procedure-extraction-pipeline"

    # Uploads. Must fall under one of _ALLOWED_PATH_PREFIXES in schemas/api.py;
    # /tmp exists on the host too, so this default works inside and outside
    # Docker. In Compose this directory is the `uploads` named volume, so
    # files survive container recreates.
    upload_dir: str = "/tmp/pipeline/uploads"
    upload_max_bytes: int = 50 * 1024 * 1024  # 50 MB

    # ATT&CK STIX bundle path (read by app.services.attack_data via stdlib
    # json). Repo root data/ is bind-mounted into the api container as /data,
    # so this default is a CONTAINER path: outside Docker set ATTACK_STIX_PATH
    # to <repo>/data/attack/enterprise-attack-19.2.json (scripts/fetch_attack.sh
    # downloads it).
    #
    # This and the Neo4j catalogue MUST move together. The graph is loaded from
    # the same file by scripts/load_attack.py, and if the two drift the
    # pipeline can map a technique the graph has no node for — which surfaces
    # as a relationship whose endpoint does not exist. The graph records what
    # it holds on its (:AttackCollection) node; check that before changing this.
    attack_stix_path: str = "/data/attack/enterprise-attack-19.2.json"

    # Technique candidate retriever (app.services.technique_retriever):
    #   "embedding"     — semantic retrieval via sentence-transformers +
    #                     cisco-ai/SecureBERT2.0-biencoder (default).
    #                     Catches lexically-distant cases (e.g. T1684.001
    #                     Impersonation from "impersonating IT support")
    #                     that token_overlap misses. Requires
    #                     sentence-transformers + a ~570 MB SecureBERT 2.0
    #                     download on first run.
    #   "token_overlap" — Jaccard scoring + CVE/tool anchors. Zero infra,
    #                     ~5x faster per call. Use when sentence-transformers
    #                     isn't available or model download is undesirable.
    # Head to head the two tie on straightforward cases; embedding wins the
    # lexically-distant ones, which is why it is the default. Override via
    # the TECHNIQUE_RETRIEVER env var.
    technique_retriever: str = "embedding"
    embedding_model: str = "cisco-ai/SecureBERT2.0-biencoder"

    # Feedback flywheel (app.services.feedback_patterns + pattern_embedding).
    # Relevance-first retrieval replaces the old category-dump: patterns are
    # embedded with the SAME SecureBERT model as technique_retriever and
    # ranked against a per-source representation (cosine + lexical anchors +
    # structured-key overlap + salience).
    # Dedup is a CONJUNCTION, and the cosine bar is calibrated to the genre it
    # is applied to. SecureBERT scores ~0.91 for two paraphrases of one threat
    # BEHAVIOR, but feedback patterns are instructional RULES, where its scale
    # is compressed: the most similar pair of real patterns scored ~0.70, so a
    # 0.90 bar was unreachable and no merge ever fired. Cosine alone cannot
    # separate the classes either (a true duplicate and a distinct pair both
    # scored ~0.58), so token overlap is required alongside it: on hand-labeled
    # pairs the duplicates ran >= 0.228 overlap and the distinct pairs <= 0.164,
    # and 0.20 sits in that gap. Both bars must be cleared. Re-calibrate both
    # if the embedding model changes.
    feedback_dedup_threshold: float = 0.55        # cosine floor
    feedback_dedup_lexical_floor: float = 0.20    # token-overlap floor, checked as well
    feedback_candidate_ceiling: int = 80     # SQL pre-filter size before in-process hybrid ranking
    feedback_salience_weight: float = 0.15   # weight of the salience term in the hybrid retrieval score (small: tune, don't dominate cosine)
    # Closed-loop salience: bayesian-smoothed hit-rate x recency decay x occurrence weight.
    feedback_salience_alpha: float = 2.0     # hit-rate prior numerator ("assume mildly useful")
    feedback_salience_beta: float = 1.0      # hit-rate prior denominator
    feedback_salience_half_life_days: float = 45.0
    feedback_match_threshold: float = 0.55   # cosine >= this counts a delta as matching a surfaced pattern (re-correction => MISS)
    feedback_min_salience: float = 0.05      # patterns below this AND stale get archived by recompute_salience.py

    # Corrected-example channel (app.services.feedback_examples). Separate
    # knobs from the pattern channel on purpose: the two are fetched
    # independently so an ablation arm can vary one without the other, and a
    # demonstration costs far more prompt than a one-line rule, so its top-N is
    # much smaller.
    feedback_examples_enabled: bool = True
    feedback_examples_limit: int = 6         # demonstrations injected per node
    feedback_example_candidate_ceiling: int = 120  # SQL pre-filter before in-process ranking

    # Figure extraction (app.nodes.llm.figure_extraction — vision LLM pass).
    # max_tokens caps each figure's output so a single dense image cannot run
    # away to the global 20k default (one figure took minutes). The budget has
    # to cover a full-screen verbatim transcription AND the model's reasoning,
    # because on models with adaptive thinking the thinking is charged against
    # max_tokens too. concurrency bounds simultaneous vision calls so a
    # figure-heavy source isn't serialized one call at a time.
    figure_extraction_max_tokens: int = 12000
    figure_extraction_concurrency: int = 5

    # Model for the AI gate reviewer (app.services.reviewer). Deliberately
    # NOT the pipeline's DEFAULT_MODEL: the reviewer's whole purpose is to
    # bring judgment the extraction stage did not, and re-reading the same
    # material with the same model mostly buys noise. Configurable so the
    # tier can be compared against the measured agreement rate on the
    # Reviewer tab.
    reviewer_model: str = "claude-opus-5"
    # Empty means "same provider as llm_provider". Set it to run the reviewer
    # on a different VENDOR than extraction — the strongest available form of
    # the independence argument above.
    reviewer_provider: str = ""

    # env_ignore_empty: a blank value — `LLM_EFFORT=` in a .env, or the empty
    # string docker-compose substitutes for an unset `${ANTHROPIC_API_KEY:-}` —
    # is treated as UNSET and falls through to the next source or the default.
    # Without it, pydantic-settings takes "" as a real value that beats the
    # file: an operator whose key lived only in backend/.env got "not set"
    # inside the container, a blank LLM_EFFORT went to the wire as
    # output_config.effort="" (400), and a blank LLM_CACHE_VERSION silently
    # re-keyed the whole response cache. Verified on pydantic-settings 2.13.1
    # that this covers blank dotenv lines as well as empty env vars.
    model_config = {"env_file": ".env", "extra": "ignore", "env_ignore_empty": True}


settings = Settings()
