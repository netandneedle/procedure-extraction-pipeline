"""OpenAI provider — and, by consequence, most OpenAI-compatible endpoints.

WHY CHAT COMPLETIONS AND NOT THE RESPONSES API:
`/v1/chat/completions` is the de-facto standard interface. Azure OpenAI,
OpenRouter, Together, Groq, Fireworks, vLLM, Ollama and LM Studio all speak
it. Building on it means this single implementation plus an `openai_base_url`
setting covers all of them, which is a large multiplier on "bring your own
key" for one file. The Responses API is richer, but it is OpenAI-only, and
nothing this pipeline does needs it — we ask for one forced tool call and read
the arguments back.

That choice also drives two smaller ones below: the system prompt is sent as
role "system" rather than "developer", and `parallel_tool_calls` is not sent
at all. Both are things a third-party gateway is more likely to reject than to
need.

FIVE THINGS DIFFER FROM ANTHROPIC IN WAYS THAT BITE:

1. Tool arguments arrive as a JSON *string*, not a parsed object. So the
   lenient parse (`app.utils.text.loads_json_lenient`) runs on the main path
   here, where the adapter only needs it as a fallback for stringified fields.
   This pipeline's outputs are full of Windows paths like C:\\Windows, which
   is exactly the sequence that breaks naive json.loads.
2. `prompt_tokens` INCLUDES cached tokens; Anthropic's `input_tokens`
   excludes them. ProviderResponse uses Anthropic's semantics, so the cached
   count is subtracted back out. Getting this wrong makes prompt caching look
   free while it is still being billed.
3. Reasoning models take `max_completion_tokens` and reject `max_tokens`;
   non-reasoning models are the other way round on gateways that lag.
4. Reasoning models reject `temperature`, the same trade Anthropic makes for
   thinking-capable models.
5. Non-reasoning models have output caps far below the pipeline's 64k default
   budget, and OpenAI validates the budget up front — an over-cap request is a
   400 before a single token is generated. The budget is clamped per family.

MODEL IDS ON GATEWAYS: OpenRouter namespaces them (`openai/gpt-5`), Azure
uses deployment names. Every capability decision below is made on the LAST
path segment of the ID, so `openai/gpt-5` is recognised as gpt-5. A deployment
named something unrelated gets the conservative request and a one-time warning.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from app.nodes.llm.providers import (
    ProviderResponse,
    ResolvedParams,
    clamp_effort,
    is_image_block,
    warn_once,
)
from app.utils.text import loads_json_lenient

if TYPE_CHECKING:  # pragma: no cover — the SDK is a lazy import at runtime
    from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

# Transport-level retries, handed to the SDK. Matches the Anthropic provider.
MAX_RETRIES = 2

# Request timeout. This path is NON-streaming, so a whole generation has to
# finish inside one request. The SDK's default is 10 minutes, and it retries a
# timeout MAX_RETRIES times from scratch — a reasoning model given the 64k
# budget at effort=high could outrun 10 minutes, be aborted, and be re-billed
# twice before failing. Thirty minutes is what the Anthropic SDK's own
# non-streaming guard would allow for a budget this size.
TIMEOUT_SECONDS = 1800.0

# Model families that take `reasoning_effort`, reject `temperature`, and want
# `max_completion_tokens`. Matched against the ID's last path segment — and a
# "-chat" variant (gpt-5-chat-latest) is a non-reasoning model whatever its
# prefix says.
#
# UPDATE THIS when a new reasoning family ships. The symptom of a stale entry
# is a 400 naming `temperature` or `max_tokens`.
_REASONING_MODELS = ("gpt-5", "o1", "o3", "o4")

# Output-token caps for non-reasoning families, longest prefix first. OpenAI
# rejects `max_tokens` above the model's cap with a 400 ("max_tokens is too
# large ... supports at most N completion tokens"), and the pipeline's default
# budget is 64000, so every default-budget call would fail on these models
# without a clamp. A family missing from this table is sent the caller's
# budget unchanged; the symptom of a missing entry is that 400.
_OUTPUT_TOKEN_CAPS: tuple[tuple[str, int], ...] = (
    ("gpt-4.1", 32_768),
    ("gpt-4o", 16_384),
    ("gpt-4-turbo", 4_096),
    ("gpt-4", 8_192),
    ("gpt-3.5", 4_096),
)

# IDs that are unmistakably another vendor's. Sending one here is the single
# most likely BYOK mistake (LLM_PROVIDER=openai with LLM_MODEL left at the
# Anthropic default), and the vendor's 404 says nothing about the cause.
_FOREIGN_PREFIXES = ("claude-", "gemini-")

# What `reasoning_effort` accepts (verified against openai 2.31.0). Anthropic's
# "max" has no equivalent, so it is clamped rather than sent — an unknown
# effort is a 400, and silently dropping the caller's intent is worse than
# saying so once.
VALID_EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh")
_EFFORT_FALLBACK = "high"


def _model_family(model: str) -> str:
    """The ID with any gateway namespace stripped: ``openai/gpt-5`` -> ``gpt-5``."""
    return model.rsplit("/", 1)[-1].strip().lower()


def _is_reasoning_model(model: str) -> bool:
    family = _model_family(model)
    if "chat" in family:
        return False
    return any(family.startswith(pfx) for pfx in _REASONING_MODELS)


def _output_token_cap(model: str) -> int | None:
    family = _model_family(model)
    for prefix, cap in _OUTPUT_TOKEN_CAPS:
        if family.startswith(prefix):
            return cap
    return None


def get_client() -> AsyncOpenAI:
    """Create an async OpenAI client from environment or .env.

    `base_url` is what makes this provider serve OpenAI-compatible gateways
    too. Left unset, the SDK's own default (api.openai.com) applies.

    Built once per provider instance (see ``OpenAIProvider._client``), not per
    call — see the note on the Anthropic side.
    """
    from openai import AsyncOpenAI

    from app.config import settings

    api_key = os.environ.get("OPENAI_API_KEY") or settings.openai_api_key
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set, but LLM_PROVIDER=openai. "
            "Set it in your .env file or environment."
        )

    base_url = os.environ.get("OPENAI_BASE_URL") or settings.openai_base_url
    kwargs: dict[str, Any] = {
        "api_key": api_key,
        "max_retries": MAX_RETRIES,
        "timeout": TIMEOUT_SECONDS,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


def _to_openai_content(content: Any) -> Any:
    """Translate neutral content blocks into OpenAI's wire shape.

    Images become a `data:` URI under `image_url`, where Anthropic nests the
    same bytes under `source`. Text blocks happen to match already.
    """
    if not isinstance(content, list):
        return content

    translated: list[Any] = []
    for block in content:
        if is_image_block(block):
            media_type = block.get("media_type") or "image/png"
            translated.append({
                "type": "image_url",
                "image_url": {"url": f"data:{media_type};base64,{block['data']}"},
            })
        else:
            translated.append(block)
    return translated


def _to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    """Prepend the system prompt as a message; translate content blocks.

    Anthropic takes `system` as a top-level request parameter. OpenAI has no
    such parameter — the instruction is the first message. Role "system" (not
    "developer") because every OpenAI-compatible gateway understands it, and
    OpenAI maps it for reasoning models on its own.
    """
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    out.extend(
        {**m, "content": _to_openai_content(m.get("content"))} for m in messages
    )
    return out


def _to_openai_tools(tools: list[dict]) -> list[dict]:
    """Neutral `input_schema` shape -> OpenAI's nested `function` shape."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {}),
            },
        }
        for t in tools
    ]


def _to_openai_tool_choice(tool_choice: dict | None) -> dict | str | None:
    """Neutral tool_choice -> OpenAI's.

    The neutral (Anthropic-shaped) vocabulary is: ``tool`` + name (force this
    one), ``any`` (must call some tool), ``auto`` (may call one), ``none``
    (must not). Each has an exact OpenAI counterpart; the earlier version
    mapped every unnamed choice to "required", which inverted ``none``. Every
    in-tree caller forces a named tool, so this is latent, but the first
    tool-optional caller would have been silently over-forced on OpenAI.
    """
    if not tool_choice:
        return None
    name = tool_choice.get("name")
    if name:
        return {"type": "function", "function": {"name": name}}
    kind = tool_choice.get("type")
    if kind == "auto":
        return "auto"
    if kind == "none":
        return "none"
    return "required"


def _parse_tool_arguments(raw: str, tool_name: str) -> dict:
    """Parse the JSON string OpenAI returns as tool arguments.

    Unlike Anthropic — which hands back an already-parsed object — this is a
    string, so the lenient parse is on the main path. It is the same
    three-rung ladder the adapter uses for stringified fields, so a repair
    that lands in one lands in both. A payload that survives no rung returns
    {} and is treated by call_llm exactly like a refused tool call.
    """
    if not raw:
        return {}
    try:
        result = loads_json_lenient(raw)
    except ValueError as exc:
        logger.warning(
            "openai: could not parse '%s' arguments (%s). head=%r",
            tool_name, exc, raw[:120],
        )
        return {}
    if result.repaired:
        logger.warning(
            "openai: repaired invalid JSON escapes in '%s' arguments", tool_name,
        )
    if result.discarded:
        logger.warning(
            "openai: parsed a valid JSON prefix of '%s' arguments and discarded "
            "%d trailing char(s): %r",
            tool_name, result.discarded, result.tail,
        )
    # A tool call whose arguments are a JSON scalar or list is not something
    # any caller here can use; treat it as no tool call at all.
    return result.value if isinstance(result.value, dict) else {}


# OpenAI's finish_reason -> the normalized (Anthropic) vocabulary. Keeping one
# vocabulary matters because stop_reason is persisted in the llm_cache table.
_STOP_REASONS = {
    "tool_calls": "tool_use",
    "stop": "end_turn",
    "length": "max_tokens",
    "content_filter": "refusal",
    "function_call": "tool_use",
}


class OpenAIProvider:
    """Talks to /v1/chat/completions with a forced function call."""

    name = "openai"

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    def resolve_params(
        self,
        model: str,
        *,
        temperature: float,
        effort: str,
    ) -> ResolvedParams:
        family = _model_family(model)
        if family.startswith(_FOREIGN_PREFIXES):
            warn_once(
                ("foreign-model", self.name, model),
                "OpenAIProvider: %r does not look like an OpenAI model ID. Check "
                "LLM_PROVIDER/LLM_MODEL (and REVIEWER_PROVIDER/REVIEWER_MODEL) — "
                "sending it anyway; expect a 404 from the vendor.",
                model,
            )

        if _is_reasoning_model(model):
            resolved_effort = clamp_effort(
                effort, valid=VALID_EFFORTS, fallback=_EFFORT_FALLBACK,
                provider=self.name, model=model,
            )
            return ResolvedParams(
                effort=resolved_effort,
                # temperature is deliberately absent — reasoning models reject
                # it, the same trade the Anthropic provider makes.
                wire_kwargs={"reasoning_effort": resolved_effort},
            )

        return ResolvedParams(effort=None, wire_kwargs={"temperature": temperature})

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = get_client()
        return self._client

    async def aclose(self) -> None:
        client, self._client = self._client, None
        if client is not None:
            await client.close()

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
        client = self._get_client()

        # Reasoning models reject `max_tokens`; older gateways may not know
        # `max_completion_tokens`. Send whichever the model can take.
        budget_kwarg = (
            "max_completion_tokens" if _is_reasoning_model(model) else "max_tokens"
        )

        # See module docstring, point 5. Clamp rather than 400.
        cap = _output_token_cap(model)
        if cap is not None and max_tokens > cap:
            warn_once(
                ("max_tokens-cap", model, max_tokens),
                "openai: max_tokens=%d exceeds %s's output cap of %d; sending %d. "
                "Long outputs (classify_sections on a big source) may truncate.",
                max_tokens, model, cap, cap,
            )
            max_tokens = cap

        resolved_choice = _to_openai_tool_choice(tool_choice)

        response = await client.chat.completions.create(
            model=model,
            messages=_to_openai_messages(system, messages),
            tools=_to_openai_tools(tools),
            **{budget_kwarg: max_tokens},
            **wire_kwargs,
            **({"tool_choice": resolved_choice} if resolved_choice else {}),
        )

        choices = getattr(response, "choices", None) or []
        if not choices:
            # A lax gateway returning no choices at all is a malformed reply,
            # not "the model declined". Treating it as a refusal would spend
            # the refusal retry on a second identical request.
            raise ValueError(
                f"openai: {model} returned no choices (id={getattr(response, 'id', '?')})"
            )
        choice = choices[0]
        message = getattr(choice, "message", None)

        tool_output: dict = {}
        tool_calls = getattr(message, "tool_calls", None) or []
        if tool_calls:
            # One forced call is what we asked for; ignore any extras rather
            # than guessing how to merge them.
            call = tool_calls[0]
            tool_output = _parse_tool_arguments(
                call.function.arguments or "", call.function.name or "<unknown>",
            )

        # On a refusal OpenAI puts the text in `refusal` and leaves `content`
        # None — the one time raw_text matters most for diagnostics.
        raw_text = (
            getattr(message, "content", None)
            or getattr(message, "refusal", None)
            or ""
        )

        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        cached_tokens = getattr(details, "cached_tokens", 0) or 0

        finish_reason = getattr(choice, "finish_reason", "") or ""
        stop_reason = _STOP_REASONS.get(finish_reason, finish_reason)
        if tool_calls and stop_reason == "end_turn":
            # OpenAI reports finish_reason="stop" (not "tool_calls") when the
            # tool was FORCED — which is every call in this pipeline. The
            # normalized vocabulary says a turn carrying a tool call is
            # tool_use, and llm_cache rows must mean the same thing across
            # vendors.
            stop_reason = "tool_use"

        return ProviderResponse(
            tool_output=tool_output,
            raw_text=raw_text,
            # `or model`: a lax gateway can return model=None, and the value
            # lands in a NOT NULL column — the cache write would fail on every
            # call and every rerun would be re-billed.
            model=getattr(response, "model", None) or model,
            # Subtract so `input_tokens` means UNCACHED prompt tokens, as it
            # does on Anthropic. See the module docstring.
            input_tokens=max(0, prompt_tokens - cached_tokens),
            output_tokens=completion_tokens,
            # OpenAI's prompt cache is automatic and bills no write, so there
            # is nothing to report here — not a zero standing in for unknown.
            cache_creation_tokens=0,
            cache_read_tokens=cached_tokens,
            stop_reason=stop_reason,
        )
