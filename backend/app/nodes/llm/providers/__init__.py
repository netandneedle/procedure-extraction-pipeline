"""Provider abstraction behind ``call_llm``.

WHY THIS EXISTS:
``llm_adapter.call_llm`` is the single chokepoint every LLM node calls, but
until this package it was Anthropic-shaped end to end — it imported the SDK,
built the client, sent Anthropic's request shape, read Anthropic's response
blocks, and decided thinking-vs-temperature from a hard-coded ``claude-*``
prefix list. Bringing your own key for a different vendor meant rewriting the
adapter.

The split is deliberate and narrow:

  * The PROVIDER owns wire format only — client construction, request
    translation, response parsing, token-field names, and which knobs the
    model accepts.
  * ``call_llm`` keeps everything else — the response cache, Pydantic
    validation with bounded retry, the refusal retry, the stringified-field
    coercion, and token accounting. None of that is vendor-specific, and all
    of it is hard-won behavior that should not be duplicated per provider.

This mirrors ``app.services.technique_retriever``: a Protocol, one
implementation per backend, a config string selecting between them, and a
memoized factory.

ON THE NEUTRAL SHAPES:
``tools`` and ``tool_choice`` cross this boundary in Anthropic's shape —
``{"name", "description", "input_schema"}`` and
``{"type": "tool", "name": X}``. That is a historical accident, not a design
statement: those shapes are already baked into the ~10 tool-schema constants
across the LLM nodes, and adopting them as the neutral form means adding a
provider costs zero node edits. A non-Anthropic provider translates on the way
out. Do not read it as Anthropic leaking through the abstraction.

MESSAGE CONTENT:
``content`` is either a plain string or a list of neutral blocks built by
``text_block`` / ``image_block`` below. Providers translate those to their own
wire shape. This is the one place the neutral form is genuinely neutral rather
than Anthropic-by-default, because the vendors disagree outright: Anthropic
nests base64 under ``source``, OpenAI wants a ``data:`` URI under
``image_url``.

REQUIRED CAPABILITIES:
Every node in this pipeline passes ``tool_choice={"type": "tool", ...}``. A
provider that can offer tools but cannot FORCE a named one cannot run this
pipeline — check that before writing an implementation. Vision (image content
blocks) is required too, unless figure extraction is disabled per source.

ONE NAME, ONE SPELLING:
``resolve_provider_name`` is the only place a provider name is normalized, and
``LLMProvider.name`` is the only spelling that reaches the cache key.
``get_provider`` once tolerated ``"Anthropic"`` while the cache-key shim
compared against the exact string ``"anthropic"`` — a capitalized
``LLM_PROVIDER`` routed correctly and silently orphaned every historical
cache row.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

logger = logging.getLogger(__name__)


@dataclass
class ProviderResponse:
    """One completed model turn, normalized across vendors.

    Token fields use Anthropic's semantics because that is what the rest of
    the pipeline already records: ``input_tokens`` counts UNCACHED prompt
    tokens only, with cached prompt tokens reported separately. A provider
    whose API reports a single all-in prompt total must split it the same way
    (or report zero for the cache fields), otherwise the cost logging silently
    under- or over-reports.

    Attributes:
        tool_output: Parsed arguments of the tool call. ``{}`` when the model
            did not call a tool — the caller decides whether that is fatal.
        raw_text: Any free-text the model emitted alongside the tool call —
            including a vendor's refusal text, when it puts that in its own
            field. Used for diagnostics when a forced tool call does not
            arrive, so it must not be empty exactly when the model refused.
        model: The model ID the provider actually served, as reported by the
            API — not the one requested. These can differ when a vendor
            resolves an alias. Never None: it lands in a NOT NULL column, so a
            gateway that omits it must fall back to the requested ID.
        input_tokens: Uncached prompt tokens.
        output_tokens: Generated tokens.
        cache_creation_tokens: Prompt tokens written to the vendor's prompt
            cache. Zero when the vendor has no such concept.
        cache_read_tokens: Prompt tokens served from the vendor's prompt cache.
        stop_reason: Why generation stopped, normalized to Anthropic's
            vocabulary: ``tool_use`` | ``end_turn`` | ``max_tokens`` |
            ``stop_sequence`` | ``refusal``. Anthropic passes through
            unchanged, which matters because this value is persisted in the
            ``llm_cache`` table — existing rows must keep their meaning. A
            turn that carries a tool call is ``tool_use`` whatever the vendor
            called it.
    """

    tool_output: dict = field(default_factory=dict)
    raw_text: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    stop_reason: str = ""


@dataclass(frozen=True)
class ResolvedParams:
    """Per-model request knobs, decided by the provider.

    Split into two halves because they are consumed in different places and at
    different times:

        wire_kwargs — passed straight to the provider's own request call.
            Vendor-specific and never inspected by ``call_llm``.
        effort — the reasoning-effort level actually in force, or None when
            the model has no such knob. This one is NOT vendor-specific
            because it goes into the response cache key: re-tuning effort must
            not serve back answers reasoned at the previous level.
    """

    effort: str | None = None
    wire_kwargs: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    """What ``call_llm`` needs from a vendor.

    Implementations are memoized one-per-name by ``get_provider`` and must be
    safe to share across concurrent calls. The instance is where a provider
    keeps its SDK client: one connection pool per process, not one per call.
    """

    name: str

    def resolve_params(
        self,
        model: str,
        *,
        temperature: float,
        effort: str,
    ) -> ResolvedParams:
        """Decide which knobs this model accepts.

        Called once per ``call_llm``, BEFORE the cache lookup, because the
        resolved effort is part of the cache key.

        A model ID this provider does not recognize must not raise — it should
        warn once, naming the model, and fall back to the most conservative
        request it can make. A typo in ``LLM_MODEL`` should degrade loudly in
        the logs, not fail the run at the vendor with an opaque 400. An effort
        value the model cannot take is clamped the same way, via
        ``clamp_effort`` — that contract holds for EVERY provider, because
        ``.env.example`` promises it without naming one.
        """
        ...

    async def complete(
        self,
        *,
        model: str,
        system: str,
        messages: list[dict],
        tools: list[dict],
        tool_choice: dict | None,
        max_tokens: int,
        wire_kwargs: dict[str, Any],
    ) -> ProviderResponse:
        """Make one request and normalize the result.

        Raises whatever the vendor SDK raises on unrecoverable errors; the
        caller does not catch these. Transport-level retries belong here (the
        SDK usually provides them), while semantic retries — bad schema, a
        refused tool call — belong to ``call_llm``.
        """
        ...

    async def aclose(self) -> None:
        """Release the SDK client, if one was built. Called at app shutdown."""
        ...


# =============================================================================
# Neutral content blocks
# =============================================================================


def text_block(text: str) -> dict:
    """A neutral text content block."""
    return {"type": "text", "text": text}


def image_block(*, media_type: str, data: str) -> dict:
    """A neutral base64 image content block.

    Callers must not hand-roll vendor image blocks — that is how vision ended
    up hard-coded to Anthropic outside the adapter in the first place.
    """
    return {"type": "image", "media_type": media_type, "data": data}


def is_image_block(block: Any) -> bool:
    """True for a neutral image block (the shape ``image_block`` builds)."""
    return isinstance(block, dict) and block.get("type") == "image" and "data" in block


def legacy_image_block(block: dict) -> dict:
    """The pre-BYOK image shape: Anthropic's nested ``source`` form.

    Defined ONCE because two things depend on it being byte-identical: the
    Anthropic provider sends it on the wire, and the adapter's cache-key shim
    renders neutral blocks into it so that every figure extraction ever cached
    (284 of the 741 live rows, ~25 s of vision each) stays reachable. The two
    were once separate copies; the first field added to one and not the
    other would have keyed every vision call differently from how it was
    sent, and no test would have noticed.

    ``media_type`` defaults to ``""`` rather than ``image/png`` for the same
    reason — that is what the historical keys hashed. ``image_block`` always
    sets it, so the default only matters for a hand-built block.
    """
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": block.get("media_type", ""),
            "data": block["data"],
        },
    }


# =============================================================================
# Shared policy helpers — the pieces that must behave the same on every vendor
# =============================================================================

# Warnings that describe a configuration, not a call, are said once per
# process. resolve_params runs on every call_llm; without this, a documented
# setting like LLM_EFFORT=max on an OpenAI reasoning model would log the same
# WARNING for every one of the ~9 + N-figure calls per source, forever.
_WARNED: set[tuple] = set()


def warn_once(key: tuple, msg: str, *args: Any) -> None:
    """Log ``msg`` at WARNING the first time ``key`` is seen; drop repeats."""
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(msg, *args)


def clamp_effort(
    effort: str | None,
    *,
    valid: tuple[str, ...],
    fallback: str,
    provider: str,
    model: str,
) -> str:
    """Return ``effort`` if this provider accepts it, else ``fallback`` (warned once).

    Case- and whitespace-insensitive. ``.env.example`` promises "a value the
    provider doesn't accept is clamped to high with a warning rather than
    sent" — this is the one implementation of that promise, so it cannot be
    true for one vendor and false for another.
    """
    candidate = (effort or "").strip().lower()
    if candidate in valid:
        return candidate
    warn_once(
        ("effort", provider, model, candidate),
        "%s: effort=%r is not accepted for model %s (accepts %s); using %r. "
        "Set LLM_EFFORT to one of the accepted values to silence this.",
        provider, effort, model, ", ".join(valid), fallback,
    )
    return fallback


def resolve_provider_name(name: str | None = None) -> str:
    """The canonical provider key for ``name``; blank means ``settings.llm_provider``.

    This is the ONLY place a provider name is normalized. ``call_llm``, the
    reviewer runner, and the startup check all go through it, so "" (inherit
    the extraction provider), "Anthropic", and " anthropic " all mean the same
    thing everywhere — including in the cache key.
    """
    key = (name or "").strip().lower()
    if key:
        return key
    from app.config import settings

    return (settings.llm_provider or "").strip().lower()


# =============================================================================
# Factory
# =============================================================================


def _build_anthropic() -> LLMProvider:
    from app.nodes.llm.providers.anthropic_provider import AnthropicProvider

    return AnthropicProvider()


def _build_openai() -> LLMProvider:
    # Also covers OpenAI-compatible gateways (Azure, OpenRouter, vLLM,
    # Ollama, ...) via settings.openai_base_url.
    #
    # The provider module defers `import openai` to client construction, so
    # without this line a missing SDK would boot clean and raise at the first
    # LLM call. The startup check relies on get_provider raising ImportError
    # at boot; importing here keeps that true while still leaving the SDK
    # optional for an Anthropic-only deployment.
    import openai  # noqa: F401

    from app.nodes.llm.providers.openai_provider import OpenAIProvider

    return OpenAIProvider()


# Implementation modules are imported lazily by the factory so that an unused
# provider's SDK is never a hard import requirement — an Anthropic-only
# deployment must not need the OpenAI package installed. Each provider is
# named exactly once, here.
_FACTORIES: dict[str, Callable[[], LLMProvider]] = {
    "anthropic": _build_anthropic,
    "openai": _build_openai,
}

# Providers this package knows how to build.
KNOWN_PROVIDERS: tuple[str, ...] = tuple(_FACTORIES)

# Memoized one instance per provider name. The instance owns the SDK client
# (built lazily on first use), so this is what makes one connection pool per
# process rather than one per call.
_PROVIDERS: dict[str, LLMProvider] = {}


def get_provider(name: str | None = None) -> LLMProvider:
    """Return the provider implementation for ``name`` (blank = configured default).

    Raises ValueError on an unknown name rather than falling back to a
    default. A silent fallback would send a run to the wrong vendor — and bill
    the wrong key — on nothing worse than a typo in ``LLM_PROVIDER``.
    """
    key = resolve_provider_name(name)
    factory = _FACTORIES.get(key)
    if factory is None:
        raise ValueError(
            f"Unknown LLM provider {name!r}. "
            f"Known providers: {', '.join(KNOWN_PROVIDERS)}. "
            f"Set LLM_PROVIDER to one of those."
        )

    cached = _PROVIDERS.get(key)
    if cached is not None:
        return cached

    provider = factory()
    _PROVIDERS[key] = provider
    return provider


def reset_provider_cache() -> None:
    """Drop memoized providers (and their clients) and the once-only warnings.

    For tests: each test that patches a provider's ``get_client`` needs a
    fresh instance, or the previous test's mock client is served from the
    memo. Autouse fixtures in the provider test modules call this.
    """
    _PROVIDERS.clear()
    _WARNED.clear()


async def close_providers() -> None:
    """Close every memoized provider's SDK client. Called at app shutdown."""
    for provider in list(_PROVIDERS.values()):
        try:
            await provider.aclose()
        except Exception:  # noqa: BLE001 — shutdown must not fail on cleanup
            logger.warning("could not close provider %s", provider.name, exc_info=True)
    _PROVIDERS.clear()


__all__ = [
    "KNOWN_PROVIDERS",
    "LLMProvider",
    "ProviderResponse",
    "ResolvedParams",
    "clamp_effort",
    "close_providers",
    "get_provider",
    "image_block",
    "is_image_block",
    "legacy_image_block",
    "reset_provider_cache",
    "resolve_provider_name",
    "text_block",
    "warn_once",
]
