"""Anthropic provider — the pipeline's default and reference implementation.

Everything here was lifted verbatim from ``llm_adapter.call_llm`` when the
provider seam was extracted. It is deliberately a move, not a rewrite: the
streaming choice, the capability gating, and the token-field handling each
encode a live production incident, and re-deriving them would re-earn the
same bugs.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import anthropic

from app.nodes.llm.providers import (
    ProviderResponse,
    ResolvedParams,
    clamp_effort,
    is_image_block,
    legacy_image_block,
    warn_once,
)

logger = logging.getLogger(__name__)

# Transport-level retries, handed to the SDK. Separate from the validation
# retries in call_llm: these cover connection resets and 429/5xx, and cost
# nothing extra when they don't fire.
MAX_RETRIES = 2

# Every Anthropic model ID starts with this. Used only to tell "an Anthropic
# model I have no capability entry for" (fine — fall back) from "not an
# Anthropic model at all" (a misconfiguration worth shouting about).
_MODEL_ID_PREFIX = "claude-"

# Model-capability gating, prefix-matched against the model ID.
# Adaptive thinking and output_config.effort are rejected (400) by Sonnet 4.5
# and earlier. An allowlist is the safe shape here: a model missing from it
# degrades quietly rather than fatally, because Sonnet 5 thinks adaptively
# even when the field is omitted and effort defaults to "high" on the server.
_ADAPTIVE_THINKING_MODELS = (
    "claude-fable-5", "claude-mythos-5",
    "claude-opus-5", "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6",
    "claude-sonnet-5", "claude-sonnet-4-6",
)

# What output_config.effort accepts. The API is a 400 on anything else — and
# on "" — so an unaccepted value is clamped here rather than sent. ("xhigh"
# sits between high and max on the wire but is absent from this SDK version's
# Literal type — 0.94.1 knows low/medium/high/max only.)
VALID_EFFORTS = ("low", "medium", "high", "max")
_EFFORT_FALLBACK = "high"


def get_client() -> anthropic.AsyncAnthropic:
    """Create an async Anthropic client from environment or .env.

    Checks os.environ first (for Docker / explicit export), then falls back
    to the Settings object which reads from backend/.env automatically.

    Returns the async SDK client so that ``await client.messages.create(...)``
    yields the event loop while waiting on the network — pipelines and HTTP
    requests can interleave instead of serializing behind every LLM call.

    Built once per provider instance (see ``AnthropicProvider._client``), not
    per call: each client owns an httpx connection pool, and a pool per
    attempt meant a fresh TLS handshake on every retry and nothing ever
    closing it.
    """
    from app.config import settings

    api_key = os.environ.get("ANTHROPIC_API_KEY") or settings.anthropic_api_key
    if not api_key:
        raise EnvironmentError(
            "ANTHROPIC_API_KEY not set. Required for LLM nodes. "
            "Set it in your .env file or environment."
        )
    workspace_id = (
        os.environ.get("ANTHROPIC_WORKSPACE_ID") or settings.anthropic_workspace_id
    ).strip()
    kwargs: dict[str, Any] = {"api_key": api_key, "max_retries": MAX_RETRIES}
    if workspace_id:
        # An organization-level key must name the workspace on every request;
        # a workspace-scoped key ignores the header. See Settings.
        kwargs["default_headers"] = {"anthropic-workspace-id": workspace_id}
    return anthropic.AsyncAnthropic(**kwargs)


def _extract_tool_output(response: Any) -> tuple[dict, str]:
    """Pull the tool_input dict and any free text from the Anthropic response.

    Returns a (tool_output, raw_text) tuple. tool_output may be {} if no
    tool_use block was emitted.
    """
    tool_output: dict = {}
    text_parts: list[str] = []
    for block in response.content:
        if block.type == "tool_use":
            tool_output = block.input
        elif block.type == "text":
            text_parts.append(block.text)
    return tool_output, "\n".join(text_parts)


def _to_anthropic_content(content: Any) -> Any:
    """Translate neutral content blocks into Anthropic's wire shape.

    Only images actually differ — Anthropic nests the payload under `source`,
    where the neutral form keeps it flat. Text blocks and plain-string content
    pass through untouched. The nested form is ``legacy_image_block`` — the
    same function the adapter's cache-key shim uses, so wire and key cannot
    drift apart.

    A block that already carries `source` is left alone rather than rewritten,
    so a caller that hands us a vendor-shaped block gets sent, not a KeyError
    from deep inside the provider.
    """
    if not isinstance(content, list):
        return content
    return [legacy_image_block(b) if is_image_block(b) else b for b in content]


def _to_anthropic_messages(messages: list[dict]) -> list[dict]:
    """Copy-on-translate so the caller's message list is never mutated."""
    return [{**m, "content": _to_anthropic_content(m.get("content"))} for m in messages]


class AnthropicProvider:
    """Talks to the Anthropic Messages API via tool_use."""

    name = "anthropic"

    def __init__(self) -> None:
        self._client: anthropic.AsyncAnthropic | None = None

    def resolve_params(
        self,
        model: str,
        *,
        temperature: float,
        effort: str,
    ) -> ResolvedParams:
        """Pick between adaptive thinking and temperature for this model.

        The two are mutually exclusive in practice. Both halves verified
        live: Sonnet 5 (with Opus 5/4.8/4.7 and Fable) removed
        `temperature` outright and 400s on any value, while Sonnet 4.6 and
        Opus 4.6 still accept it but 400 on anything other than 1 once
        thinking is on ("`temperature` may only be set to 1 when thinking is
        enabled or in adaptive mode"). So the pipeline's temperature=0.0
        convention is unsendable on every thinking-capable model. Determinism
        now rests on the prompts and the response cache — temperature=0 never
        guaranteed identical outputs anyway.
        """
        if any(model.startswith(pfx) for pfx in _ADAPTIVE_THINKING_MODELS):
            resolved_effort = clamp_effort(
                effort, valid=VALID_EFFORTS, fallback=_EFFORT_FALLBACK,
                provider=self.name, model=model,
            )
            return ResolvedParams(
                effort=resolved_effort,
                wire_kwargs={
                    "thinking": {"type": "adaptive"},
                    "output_config": {"effort": resolved_effort},
                },
            )

        if not model.startswith(_MODEL_ID_PREFIX):
            # Not an Anthropic model ID at all — almost certainly LLM_PROVIDER
            # and LLM_MODEL disagreeing, or a typo. The request will fail at
            # the vendor with a 404/400 that says nothing about the cause, so
            # say it here while there is still context.
            warn_once(
                ("foreign-model", self.name, model),
                "AnthropicProvider: %r does not look like an Anthropic model "
                "ID (expected a %r prefix). Check LLM_PROVIDER/LLM_MODEL — "
                "sending it anyway with pre-thinking defaults.",
                model, _MODEL_ID_PREFIX,
            )

        # Pre-4.6 Claude: no adaptive thinking, and temperature still works.
        return ResolvedParams(effort=None, wire_kwargs={"temperature": temperature})

    def _get_client(self) -> anthropic.AsyncAnthropic:
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

        # Stream rather than messages.create so max_tokens can use the
        # full output window. The SDK refuses large non-streaming budgets
        # because generation could outrun its 10-minute request cap — that
        # ceiling is what previously pinned callers to ~16-20k. We don't
        # consume the incremental events; get_final_message() reassembles
        # exactly the Message object messages.create would have returned,
        # so everything downstream of here is unchanged.
        async with client.messages.stream(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=_to_anthropic_messages(messages),
            tools=tools,
            **wire_kwargs,
            **({"tool_choice": tool_choice} if tool_choice else {}),
        ) as stream:
            response = await stream.get_final_message()

        tool_output, raw_text = _extract_tool_output(response)

        return ProviderResponse(
            tool_output=tool_output,
            raw_text=raw_text,
            model=response.model or model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            # getattr: these fields are absent on models/SDK versions without
            # prompt caching, and a missing attribute must not break a call.
            cache_creation_tokens=getattr(
                response.usage, "cache_creation_input_tokens", 0
            ) or 0,
            cache_read_tokens=getattr(
                response.usage, "cache_read_input_tokens", 0
            ) or 0,
            # Anthropic's vocabulary IS the normalized vocabulary — see
            # ProviderResponse.stop_reason. Passing through keeps values in
            # the existing llm_cache table meaningful.
            stop_reason=response.stop_reason,
        )
