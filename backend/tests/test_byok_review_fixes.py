"""Pins for a review of the BYOK provider layer.

Each test names the finding it closes. They live together rather than spread
across the provider test modules because the review found one root cause
behind most of them — no single place resolved (provider, model, effort) per
role — and the fixes are easier to read as a set.

    pytest tests/test_byok_review_fixes.py -v
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

from app.config import settings  # noqa: E402
from app.nodes.llm import llm_adapter  # noqa: E402
from app.nodes.llm.providers import (  # noqa: E402
    ProviderResponse,
    ResolvedParams,
    clamp_effort,
    get_provider,
    image_block,
    legacy_image_block,
    reset_provider_cache,
    resolve_provider_name,
)
from app.nodes.llm.providers import anthropic_provider, openai_provider  # noqa: E402
from app.utils.text import loads_json_lenient  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_providers():
    reset_provider_cache()
    yield
    reset_provider_cache()


# =============================================================================
# Provider name: one spelling everywhere (cache-key finding)
# =============================================================================


def test_resolve_provider_name_normalizes_and_defaults():
    with patch.object(settings, "llm_provider", "Anthropic"):
        assert resolve_provider_name(None) == "anthropic"
        assert resolve_provider_name("") == "anthropic"
    assert resolve_provider_name("  OpenAI ") == "openai"


def test_cache_key_ignores_provider_case_and_whitespace():
    """LLM_PROVIDER=Anthropic routed correctly but orphaned every cached row."""
    common = dict(
        system="s",
        messages=[{"role": "user", "content": "c"}],
        tools=[{"name": "t", "description": "d", "input_schema": {}}],
        tool_choice={"type": "tool", "name": "t"},
        model="claude-sonnet-5",
        temperature=0.0,
        max_tokens=100,
        output_model=None,
        effort="high",
    )
    canonical = llm_adapter._build_cache_key(**common, provider="anthropic")
    assert llm_adapter._build_cache_key(**common, provider="Anthropic") == canonical
    assert llm_adapter._build_cache_key(**common, provider=" anthropic ") == canonical


async def test_call_llm_keys_the_cache_on_the_providers_own_name():
    """The provider's `.name`, not the raw setting, reaches the cache key."""
    seen: dict = {}

    def fake_build(*args, **kwargs):
        seen["provider"] = args[-1] if args else kwargs.get("provider")
        return "k" * 64

    async def no_cache(_key):
        return None

    provider = MagicMock()
    provider.name = "anthropic"
    provider.resolve_params.return_value = ResolvedParams()
    provider.complete = AsyncMock(return_value=ProviderResponse(
        tool_output={"a": 1}, stop_reason="tool_use", model="m",
    ))

    with patch.object(llm_adapter, "get_provider", return_value=provider), \
         patch.object(llm_adapter, "_build_cache_key", side_effect=fake_build), \
         patch.object(llm_adapter, "get_cached", side_effect=no_cache), \
         patch.object(llm_adapter, "write_cache", new=AsyncMock()):
        await llm_adapter.call_llm(
            system="s", messages=[{"role": "user", "content": "c"}],
            tools=[{"name": "t", "input_schema": {}}],
            tool_choice={"type": "tool", "name": "t"},
            provider=" Anthropic ",
        )

    assert seen["provider"] == "anthropic"


# =============================================================================
# One resolver per role (ANTHROPIC_MODEL / REVIEWER_PROVIDER findings)
# =============================================================================


def test_extraction_target_reads_the_rebindable_module_globals():
    with patch.object(llm_adapter, "DEFAULT_MODEL", "claude-x"), \
         patch.object(llm_adapter, "DEFAULT_EFFORT", "max"), \
         patch.object(settings, "llm_provider", "Anthropic"):
        t = llm_adapter.resolve_target("extraction")
    assert (t.provider, t.model, t.effort) == ("anthropic", "claude-x", "max")


def test_reviewer_target_inherits_provider_when_blank():
    with patch.object(settings, "llm_provider", "openai"), \
         patch.object(settings, "reviewer_provider", ""), \
         patch.object(settings, "reviewer_model", "gpt-5"):
        t = llm_adapter.resolve_target("reviewer")
    assert (t.provider, t.model) == ("openai", "gpt-5")


def test_reviewer_target_honours_an_explicit_vendor():
    with patch.object(settings, "llm_provider", "anthropic"), \
         patch.object(settings, "reviewer_provider", " OpenAI "), \
         patch.object(settings, "reviewer_model", "gpt-5"):
        assert llm_adapter.resolve_target("reviewer").provider == "openai"


def test_legacy_anthropic_model_env_no_longer_overrides():
    """A stale export used to send a Claude ID to every provider."""
    with patch.dict(os.environ, {"ANTHROPIC_MODEL": "claude-stale"}), \
         patch.object(llm_adapter, "DEFAULT_MODEL", "gpt-5"):
        assert llm_adapter.resolve_target("extraction").model == "gpt-5"


# =============================================================================
# Blank values mean "unset" (compose empty-key / blank LLM_EFFORT findings)
# =============================================================================


def test_settings_treat_blank_env_and_dotenv_as_unset(tmp_path):
    """The empty string compose substitutes must not shadow backend/.env."""
    from pydantic_settings import BaseSettings

    dotenv = tmp_path / ".env"
    dotenv.write_text("LLM_EFFORT=\nLLM_CACHE_VERSION=\nANTHROPIC_API_KEY=fromfile\n")

    class S(BaseSettings):
        llm_effort: str = "high"
        llm_cache_version: str = "1"
        anthropic_api_key: str = ""
        model_config = {**settings.model_config, "env_file": str(dotenv)}

    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": ""}):
        s = S()

    assert s.llm_effort == "high"
    assert s.llm_cache_version == "1"
    assert s.anthropic_api_key == "fromfile"


def test_project_settings_enable_env_ignore_empty():
    assert settings.model_config.get("env_ignore_empty") is True


# =============================================================================
# Effort validation on every provider (Anthropic clamp finding)
# =============================================================================


def test_clamp_effort_accepts_valid_and_warns_once(caplog):
    with caplog.at_level("WARNING"):
        assert clamp_effort("HIGH ", valid=("high",), fallback="high",
                            provider="p", model="m") == "high"
        assert clamp_effort("", valid=("high",), fallback="high",
                            provider="p", model="m") == "high"
        assert clamp_effort("", valid=("high",), fallback="high",
                            provider="p", model="m") == "high"
    assert caplog.text.count("effort=''") == 1


@pytest.mark.parametrize("effort", ["", "xhigh", "minimal", "hgih"])
def test_anthropic_never_sends_an_unaccepted_effort(effort, caplog):
    with caplog.at_level("WARNING"):
        params = anthropic_provider.AnthropicProvider().resolve_params(
            "claude-sonnet-5", temperature=0.0, effort=effort,
        )
    assert params.effort == "high"
    assert params.wire_kwargs["output_config"] == {"effort": "high"}
    assert "not accepted" in caplog.text


def test_anthropic_accepts_max():
    params = anthropic_provider.AnthropicProvider().resolve_params(
        "claude-sonnet-5", temperature=0.0, effort="max",
    )
    assert params.wire_kwargs["output_config"] == {"effort": "max"}


# =============================================================================
# OpenAI: gateway IDs, output caps, tool_choice, refusal text, lax gateways
# =============================================================================


@pytest.mark.parametrize("model", [
    "openai/gpt-5", "OpenAI/o3-mini", "gpt-5", "o4-mini",
])
def test_reasoning_detection_strips_gateway_namespaces(model):
    assert openai_provider._is_reasoning_model(model) is True


@pytest.mark.parametrize("model", ["gpt-5-chat-latest", "gpt-4o", "openai/gpt-4.1"])
def test_chat_and_classic_families_are_not_reasoning_models(model):
    assert openai_provider._is_reasoning_model(model) is False


def _fake_completion(**kw):
    return SimpleNamespace(
        id="cmpl-1",
        model=kw.get("model", "gpt-x"),
        choices=[SimpleNamespace(
            message=SimpleNamespace(
                tool_calls=kw.get("tool_calls"),
                content=kw.get("content"),
                refusal=kw.get("refusal"),
            ),
            finish_reason=kw.get("finish_reason", "stop"),
        )],
        usage=None,
    )


async def _run(model="gpt-4o", *, completion=None, **overrides):
    completion = completion or _fake_completion()
    client = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=completion)
    kwargs = dict(
        model=model, system="s", messages=[{"role": "user", "content": "c"}],
        tools=[{"name": "t", "input_schema": {}}],
        tool_choice={"type": "tool", "name": "t"},
        max_tokens=64000, wire_kwargs={},
    )
    kwargs.update(overrides)
    with patch.object(openai_provider, "get_client", return_value=client):
        resp = await openai_provider.OpenAIProvider().complete(**kwargs)
    return resp, client.chat.completions.create.call_args.kwargs


async def test_default_budget_is_clamped_to_the_models_output_cap(caplog):
    """64000 to gpt-4o was a 400 before a single token was generated."""
    with caplog.at_level("WARNING"):
        _, sent = await _run("gpt-4o")
    assert sent["max_tokens"] == 16_384
    assert "output cap" in caplog.text


async def test_budget_under_the_cap_is_sent_unchanged():
    _, sent = await _run("gpt-4.1", max_tokens=8000)
    assert sent["max_tokens"] == 8000


async def test_unknown_family_gets_the_callers_budget():
    _, sent = await _run("some-gateway-model")
    assert sent["max_tokens"] == 64000


@pytest.mark.parametrize("neutral,expected", [
    ({"type": "auto"}, "auto"),
    ({"type": "none"}, "none"),
    ({"type": "any"}, "required"),
    ({"type": "tool", "name": "t"}, {"type": "function", "function": {"name": "t"}}),
])
def test_tool_choice_is_mapped_three_ways(neutral, expected):
    assert openai_provider._to_openai_tool_choice(neutral) == expected


async def test_refusal_text_reaches_raw_text():
    """On a refusal OpenAI sets `refusal` and leaves `content` None."""
    resp, _ = await _run(completion=_fake_completion(
        content=None, refusal="I can't help with that.", finish_reason="stop",
    ))
    assert resp.tool_output == {}
    assert resp.raw_text == "I can't help with that."


async def test_null_model_from_a_lax_gateway_falls_back_to_the_requested_id():
    resp, _ = await _run("gpt-4o", completion=_fake_completion(model=None))
    assert resp.model == "gpt-4o"


async def test_empty_choices_is_an_error_not_a_refusal():
    bad = SimpleNamespace(id="cmpl-0", model="gpt-4o", choices=[], usage=None)
    with pytest.raises(ValueError, match="no choices"):
        await _run("gpt-4o", completion=bad)


def test_openai_warns_once_on_a_foreign_model_id(caplog):
    p = openai_provider.OpenAIProvider()
    with caplog.at_level("WARNING"):
        p.resolve_params("claude-opus-5", temperature=0.0, effort="high")
        p.resolve_params("claude-opus-5", temperature=0.0, effort="high")
    assert caplog.text.count("does not look like an OpenAI model") == 1


def test_openai_client_gets_a_timeout_sized_to_the_budget():
    with patch.object(openai_provider, "TIMEOUT_SECONDS", 1234.0):
        client = openai_provider.get_client()
    assert client.timeout == 1234.0


# =============================================================================
# Client lifecycle: one per provider instance, closed at shutdown
# =============================================================================


async def test_client_is_built_once_per_provider_and_closed():
    fake = MagicMock()
    fake.close = AsyncMock()
    fake.messages.stream = MagicMock(side_effect=RuntimeError("stop here"))

    with patch.object(anthropic_provider, "get_client", return_value=fake) as gc:
        provider = get_provider("anthropic")
        for _ in range(3):
            with pytest.raises(RuntimeError):
                await provider.complete(
                    model="claude-sonnet-5", system="s", messages=[], tools=[],
                    tool_choice=None, max_tokens=10, wire_kwargs={},
                )
        assert gc.call_count == 1

        from app.nodes.llm.providers import close_providers
        await close_providers()

    fake.close.assert_awaited_once()


# =============================================================================
# One JSON ladder, shared (refusal-shape finding)
# =============================================================================


def test_lenient_parse_rungs():
    assert loads_json_lenient('{"a": 1}') == (({"a": 1}), False, 0, "")

    repaired = loads_json_lenient('{"p": "C:\\Windows"}')
    assert repaired.value == {"p": "C:\\Windows"} and repaired.repaired

    salvaged = loads_json_lenient('[1, 2, 3]}')
    assert salvaged.value == [1, 2, 3]
    assert salvaged.discarded == 1 and salvaged.tail == "}"

    with pytest.raises(ValueError):
        loads_json_lenient("not json at all")


def test_openai_arguments_get_the_prefix_salvage_rung():
    """That shape used to be a refusal on the OpenAI path."""
    assert openai_provider._parse_tool_arguments('{"a": [1, 2]}}', "t") == {"a": [1, 2]}


# =============================================================================
# Cache shim and wire form are the same function
# =============================================================================


def test_cache_shim_and_anthropic_wire_share_one_image_translation():
    neutral = [{"role": "user", "content": [image_block(media_type="image/png", data="Zm9v")]}]
    assert (
        llm_adapter._canonical_cache_messages(neutral)
        == anthropic_provider._to_anthropic_messages(neutral)
        == [{"role": "user", "content": [legacy_image_block(neutral[0]["content"][0])]}]
    )
