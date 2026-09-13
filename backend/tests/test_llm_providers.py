"""Unit tests for the LLM provider seam (app.nodes.llm.providers).

Runs without any network — the Anthropic SDK client is patched.

    pytest tests/test_llm_providers.py -v
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-not-real")

from app.nodes.llm import llm_adapter  # noqa: E402
from app.nodes.llm.providers import (  # noqa: E402
    ProviderResponse,
    get_provider,
    reset_provider_cache,
)
from app.nodes.llm.providers import anthropic_provider  # noqa: E402

# =============================================================================
# 1. Cache-key continuity — the refactor must not cold-start the cache
# =============================================================================


def test_anthropic_cache_key_is_unchanged_by_the_provider_refactor():
    """The Anthropic cache key must be byte-identical to the pre-refactor one.

    This digest was computed from the algorithm as it stood at commit 10cb31f,
    BEFORE app.nodes.llm.providers existed. The llm_cache table holds thousands
    of entries from real source runs; any change to the payload shape silently
    invalidates all of them and re-bills every cached call.

    This caught a real bug during the refactor: writing `"provider": None` into
    the payload for Anthropic still serializes as `"provider": null` and
    changes the hash. The key has to be ABSENT, not null.
    """
    expected = "94338977f9bbf9b4a4c17cdf96e7dfc42ae0528904838b5846c4c9d85626fe49"

    with patch.object(llm_adapter, "CACHE_VERSION", "1"):
        key = llm_adapter._build_cache_key(
            system="You are a CTI analyst.",
            messages=[{"role": "user", "content": "report text"}],
            tools=[{
                "name": "widget_tool",
                "description": "d",
                "input_schema": {"type": "object"},
            }],
            tool_choice={"type": "tool", "name": "widget_tool"},
            model="claude-sonnet-5",
            temperature=0.0,
            max_tokens=64000,
            output_model=SimpleNamespace(__name__="WidgetOutput"),
            effort="high",
            provider="anthropic",
        )

    assert key == expected


def test_non_anthropic_provider_gets_a_distinct_key_space():
    """A second vendor must not collide with Anthropic's cached answers."""
    common = dict(
        system="s",
        messages=[{"role": "user", "content": "c"}],
        tools=[{"name": "t", "description": "d", "input_schema": {}}],
        tool_choice={"type": "tool", "name": "t"},
        model="some-shared-model-id",
        temperature=0.0,
        max_tokens=100,
        output_model=None,
        effort=None,
    )
    anthropic_key = llm_adapter._build_cache_key(**common, provider="anthropic")
    other_key = llm_adapter._build_cache_key(**common, provider="someone-else")

    assert anthropic_key != other_key


# =============================================================================
# 2. Factory
# =============================================================================


def test_get_provider_returns_anthropic():
    provider = get_provider("anthropic")
    assert provider.name == "anthropic"


def test_get_provider_is_memoized():
    """Providers are shared across calls, so they must be stateless."""
    assert get_provider("anthropic") is get_provider("anthropic")


def test_get_provider_normalizes_case_and_whitespace():
    assert get_provider("  ANTHROPIC ").name == "anthropic"


def test_get_provider_rejects_unknown_name():
    """No silent fallback — that would bill the wrong key on a typo."""
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        get_provider("definitely-not-a-provider")


def test_reset_provider_cache_drops_instances():
    first = get_provider("anthropic")
    reset_provider_cache()
    assert get_provider("anthropic") is not first


# =============================================================================
# 3. Capability resolution
# =============================================================================


def test_thinking_model_gets_adaptive_thinking_and_no_temperature():
    """temperature is unsendable on thinking-capable models (400 on any value)."""
    params = anthropic_provider.AnthropicProvider().resolve_params(
        "claude-sonnet-5", temperature=0.0, effort="high",
    )

    assert params.effort == "high"
    assert params.wire_kwargs["thinking"] == {"type": "adaptive"}
    assert params.wire_kwargs["output_config"] == {"effort": "high"}
    assert "temperature" not in params.wire_kwargs


def test_effort_is_threaded_through_to_the_wire_and_the_cache_key():
    params = anthropic_provider.AnthropicProvider().resolve_params(
        "claude-opus-5", temperature=0.0, effort="max",
    )
    assert params.effort == "max"
    assert params.wire_kwargs["output_config"] == {"effort": "max"}


def test_legacy_claude_model_gets_temperature_and_no_thinking():
    """Pre-4.6 Claude: adaptive thinking 400s, temperature still works."""
    params = anthropic_provider.AnthropicProvider().resolve_params(
        "claude-sonnet-4-20250514", temperature=0.0, effort="high",
    )

    assert params.effort is None
    assert params.wire_kwargs == {"temperature": 0.0}


def test_legacy_claude_model_does_not_warn(caplog):
    """A known-old Claude ID is a valid config, not a mistake."""
    with caplog.at_level("WARNING", logger=anthropic_provider.__name__):
        anthropic_provider.AnthropicProvider().resolve_params(
            "claude-3-5-sonnet-20241022", temperature=0.0, effort="high",
        )
    assert caplog.text == ""


def test_foreign_model_id_warns_but_still_sends(caplog):
    """LLM_PROVIDER/LLM_MODEL disagreeing should be loud here, not a 404 later."""
    with caplog.at_level("WARNING", logger=anthropic_provider.__name__):
        params = anthropic_provider.AnthropicProvider().resolve_params(
            "gpt-5", temperature=0.0, effort="high",
        )

    assert "gpt-5" in caplog.text
    assert "LLM_PROVIDER" in caplog.text
    # Degrades, does not raise — the request still goes out.
    assert params.wire_kwargs == {"temperature": 0.0}
    assert params.effort is None


# =============================================================================
# 4. Response normalization
# =============================================================================


def _fake_anthropic_message(*, tool_input=None, text="", with_cache_fields=True):
    """Build a stand-in for the SDK's Message object."""
    blocks = []
    if tool_input is not None:
        blocks.append(SimpleNamespace(type="tool_use", input=tool_input))
    if text:
        blocks.append(SimpleNamespace(type="text", text=text))

    usage_fields = {"input_tokens": 111, "output_tokens": 222}
    if with_cache_fields:
        usage_fields["cache_creation_input_tokens"] = 33
        usage_fields["cache_read_input_tokens"] = 44

    return SimpleNamespace(
        content=blocks,
        usage=SimpleNamespace(**usage_fields),
        model="claude-sonnet-5-20260101",
        stop_reason="tool_use",
    )


def _client_returning(message):
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=message)

    class _StreamManager:
        def __init__(self, **kwargs):
            self._kwargs = kwargs

        async def __aenter__(self):
            msg = await client.messages.create(**self._kwargs)
            stream = MagicMock()
            stream.get_final_message = AsyncMock(return_value=msg)
            return stream

        async def __aexit__(self, *exc_info):
            return False

    client.messages.stream = lambda **kwargs: _StreamManager(**kwargs)
    return client


async def test_complete_normalizes_a_tool_use_response():
    message = _fake_anthropic_message(tool_input={"widgets": ["a"]}, text="thinking out loud")
    client = _client_returning(message)

    with patch.object(anthropic_provider, "get_client", return_value=client):
        resp = await anthropic_provider.AnthropicProvider().complete(
            model="claude-sonnet-5",
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"name": "t", "description": "d", "input_schema": {}}],
            tool_choice={"type": "tool", "name": "t"},
            max_tokens=1000,
            wire_kwargs={"thinking": {"type": "adaptive"}},
        )

    assert isinstance(resp, ProviderResponse)
    assert resp.tool_output == {"widgets": ["a"]}
    assert resp.raw_text == "thinking out loud"
    # The model the API SERVED, not the alias we asked for.
    assert resp.model == "claude-sonnet-5-20260101"
    assert resp.input_tokens == 111
    assert resp.output_tokens == 222
    assert resp.cache_creation_tokens == 33
    assert resp.cache_read_tokens == 44
    assert resp.stop_reason == "tool_use"


async def test_complete_passes_wire_kwargs_and_tool_choice_through():
    client = _client_returning(_fake_anthropic_message(tool_input={}))

    with patch.object(anthropic_provider, "get_client", return_value=client):
        await anthropic_provider.AnthropicProvider().complete(
            model="claude-sonnet-5",
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"name": "t", "description": "d", "input_schema": {}}],
            tool_choice={"type": "tool", "name": "t"},
            max_tokens=1234,
            wire_kwargs={"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
        )

    sent = client.messages.create.call_args.kwargs
    assert sent["model"] == "claude-sonnet-5"
    assert sent["max_tokens"] == 1234
    assert sent["system"] == "sys"
    assert sent["thinking"] == {"type": "adaptive"}
    assert sent["output_config"] == {"effort": "high"}
    assert sent["tool_choice"] == {"type": "tool", "name": "t"}


async def test_complete_omits_tool_choice_when_none():
    """Sending tool_choice=None is not the same as omitting it."""
    client = _client_returning(_fake_anthropic_message(tool_input={}))

    with patch.object(anthropic_provider, "get_client", return_value=client):
        await anthropic_provider.AnthropicProvider().complete(
            model="claude-sonnet-5",
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"name": "t", "description": "d", "input_schema": {}}],
            tool_choice=None,
            max_tokens=100,
            wire_kwargs={},
        )

    assert "tool_choice" not in client.messages.create.call_args.kwargs


async def test_complete_tolerates_missing_prompt_cache_fields():
    """Older models/SDKs omit the cache counters; that must not break a call."""
    message = _fake_anthropic_message(tool_input={"a": 1}, with_cache_fields=False)
    client = _client_returning(message)

    with patch.object(anthropic_provider, "get_client", return_value=client):
        resp = await anthropic_provider.AnthropicProvider().complete(
            model="claude-sonnet-4-20250514",
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"name": "t", "description": "d", "input_schema": {}}],
            tool_choice=None,
            max_tokens=100,
            wire_kwargs={"temperature": 0.0},
        )

    assert resp.cache_creation_tokens == 0
    assert resp.cache_read_tokens == 0


async def test_complete_returns_empty_tool_output_on_a_text_only_turn():
    """A refusal is the caller's problem to retry, not an exception here."""
    message = _fake_anthropic_message(tool_input=None, text="I can't help with that.")
    message.stop_reason = "refusal"
    client = _client_returning(message)

    with patch.object(anthropic_provider, "get_client", return_value=client):
        resp = await anthropic_provider.AnthropicProvider().complete(
            model="claude-sonnet-5",
            system="sys",
            messages=[{"role": "user", "content": "hi"}],
            tools=[{"name": "t", "description": "d", "input_schema": {}}],
            tool_choice={"type": "tool", "name": "t"},
            max_tokens=100,
            wire_kwargs={},
        )

    assert resp.tool_output == {}
    assert resp.stop_reason == "refusal"
    assert "can't help" in resp.raw_text


# =============================================================================
# 5. Provider selection through call_llm
# =============================================================================


async def test_call_llm_rejects_an_unknown_provider():
    with pytest.raises(ValueError, match="Unknown LLM provider"):
        await llm_adapter.call_llm(
            system="s",
            messages=[{"role": "user", "content": "c"}],
            tools=[{"name": "t", "description": "d", "input_schema": {}}],
            provider="nope",
        )


# =============================================================================
# 6. Neutral content blocks -> Anthropic wire shape
# =============================================================================


def test_image_block_is_translated_to_anthropic_source_shape():
    from app.nodes.llm.providers import image_block

    out = anthropic_provider._to_anthropic_content(
        [image_block(media_type="image/png", data="BASE64")]
    )

    assert out == [{
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "BASE64"},
    }]


def test_text_blocks_and_string_content_pass_through():
    from app.nodes.llm.providers import text_block

    assert anthropic_provider._to_anthropic_content("plain string") == "plain string"
    assert anthropic_provider._to_anthropic_content([text_block("hi")]) == [
        {"type": "text", "text": "hi"}
    ]


def test_already_vendor_shaped_image_is_left_alone():
    """Tolerate a hand-rolled block rather than KeyError inside the provider."""
    vendor = {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "X"}}
    assert anthropic_provider._to_anthropic_content([vendor]) == [vendor]


def test_translation_does_not_mutate_the_callers_messages():
    """call_llm retries reuse the caller's list; mutation would corrupt attempt 2."""
    from app.nodes.llm.providers import image_block

    original = [{"role": "user", "content": [image_block(media_type="image/png", data="B")]}]
    snapshot = [{"role": "user", "content": [{"type": "image", "media_type": "image/png", "data": "B"}]}]

    anthropic_provider._to_anthropic_messages(original)

    assert original == snapshot


async def test_complete_translates_images_on_the_way_out():
    client = _client_returning(_fake_anthropic_message(tool_input={"ok": 1}))
    from app.nodes.llm.providers import image_block

    with patch.object(anthropic_provider, "get_client", return_value=client):
        await anthropic_provider.AnthropicProvider().complete(
            model="claude-sonnet-5",
            system="sys",
            messages=[{"role": "user", "content": [image_block(media_type="image/jpeg", data="Q")]}],
            tools=[{"name": "t", "description": "d", "input_schema": {}}],
            tool_choice={"type": "tool", "name": "t"},
            max_tokens=100,
            wire_kwargs={},
        )

    sent = client.messages.create.call_args.kwargs["messages"]
    assert sent[0]["content"][0]["source"]["media_type"] == "image/jpeg"
    assert sent[0]["content"][0]["source"]["data"] == "Q"


def test_neutral_and_legacy_image_messages_share_one_cache_key():
    """Vision cache entries must survive the neutral-block migration.

    284 of the 741 live llm_cache rows are figure extractions — the most
    expensive calls in the pipeline. Hashing the neutral shape directly would
    have orphaned every one of them.
    """
    from app.nodes.llm.providers import image_block, text_block

    neutral = [{
        "role": "user",
        "content": [image_block(media_type="image/png", data="B64"), text_block("caption")],
    }]
    legacy = [{
        "role": "user",
        "content": [
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "B64"}},
            {"type": "text", "text": "caption"},
        ],
    }]
    common = dict(
        system="s",
        tools=[{"name": "extract_figure", "description": "d", "input_schema": {}}],
        tool_choice={"type": "tool", "name": "extract_figure"},
        model="claude-sonnet-5",
        temperature=0.0,
        max_tokens=12000,
        output_model=None,
        effort="high",
        provider="anthropic",
    )

    assert (
        llm_adapter._build_cache_key(messages=neutral, **common)
        == llm_adapter._build_cache_key(messages=legacy, **common)
    )


# =============================================================================
# 7. Startup config checks (app.main._check_model_config)
# =============================================================================


def test_startup_warns_when_reviewer_matches_extraction(caplog):
    """Same model on both tiers makes the agreement readout meaningless."""
    from app.config import settings
    from app.main import _check_model_config

    with patch.object(settings, "llm_provider", "anthropic"), \
         patch.object(settings, "reviewer_provider", ""), \
         patch.object(settings, "reviewer_model", "claude-sonnet-5"), \
         patch.object(llm_adapter, "DEFAULT_MODEL", "claude-sonnet-5"), \
         caplog.at_level("WARNING", logger="startup.models"):
        _check_model_config()

    assert "same model" in caplog.text
    assert "REVIEWER_MODEL" in caplog.text


def test_startup_is_quiet_on_a_correct_two_tier_config(caplog):
    from app.config import settings
    from app.main import _check_model_config

    with patch.object(settings, "anthropic_api_key", "sk-ant-present"), \
         patch.object(settings, "llm_provider", "anthropic"), \
         patch.object(settings, "reviewer_provider", ""), \
         patch.object(settings, "reviewer_model", "claude-opus-5"), \
         patch.object(llm_adapter, "DEFAULT_MODEL", "claude-sonnet-5"), \
         patch.object(llm_adapter, "DEFAULT_EFFORT", "high"), \
         caplog.at_level("WARNING", logger="startup.models"):
        _check_model_config()

    assert caplog.text == ""


def test_startup_does_not_warn_when_only_the_vendor_differs(caplog):
    """Same model ID on two vendors is a real (if odd) two-tier config."""
    from app.config import settings
    from app.main import _check_model_config

    with patch.object(settings, "llm_provider", "anthropic"), \
         patch.object(settings, "reviewer_provider", "openai"), \
         patch.object(settings, "reviewer_model", "claude-sonnet-5"), \
         patch.object(llm_adapter, "DEFAULT_MODEL", "claude-sonnet-5"), \
         caplog.at_level("WARNING", logger="startup.models"):
        _check_model_config()

    assert "same model" not in caplog.text


def test_startup_warns_that_effort_is_inert_on_a_non_reasoning_model(caplog):
    """LLM_EFFORT was read, accepted, and silently discarded."""
    from app.config import settings
    from app.main import _check_model_config

    with patch.object(settings, "llm_provider", "anthropic"), \
         patch.object(settings, "reviewer_provider", ""), \
         patch.object(settings, "reviewer_model", "claude-opus-5"), \
         patch.object(llm_adapter, "DEFAULT_MODEL", "claude-sonnet-4-20250514"), \
         patch.object(llm_adapter, "DEFAULT_EFFORT", "high"), \
         caplog.at_level("WARNING", logger="startup.models"):
        _check_model_config()

    assert "LLM_EFFORT" in caplog.text
    assert "no effect" in caplog.text


def test_startup_logs_an_error_but_does_not_raise_on_a_bad_provider(caplog):
    """A config typo must not stop the API booting — bundles are still browsable."""
    from app.config import settings
    from app.main import _check_model_config

    with patch.object(settings, "llm_provider", "not-a-vendor"), \
         patch.object(settings, "reviewer_provider", ""), \
         caplog.at_level("ERROR", logger="startup.models"):
        _check_model_config()  # must not raise

    assert "Unknown LLM provider" in caplog.text


def test_startup_validates_the_reviewer_provider_too(caplog):
    """A typo in REVIEWER_PROVIDER used to boot clean.

    run_reviewer fails soft by design, so the first sign was every
    assist-mode gate silently becoming plain human review.
    """
    from app.config import settings
    from app.main import _check_model_config

    with patch.object(settings, "llm_provider", "anthropic"), \
         patch.object(settings, "reviewer_provider", "openia"), \
         patch.object(settings, "reviewer_model", "claude-opus-5"), \
         caplog.at_level("ERROR", logger="startup.models"):
        _check_model_config()  # must not raise

    assert "Unknown LLM provider 'openia'" in caplog.text
    assert "reviewer" in caplog.text


def test_startup_names_the_missing_key_for_the_configured_provider(caplog):
    """Found on the first live OpenAI run before release: the provider, its
    SDK and the effort knob were all validated at boot, and the run then
    parsed a whole document before failing at its first model call with
    "OPENAI_API_KEY not set". The key is checkable at boot; say so there."""
    from app.config import settings
    from app.main import _check_model_config

    with patch.object(settings, "llm_provider", "openai"), \
         patch.object(settings, "llm_model", "gpt-5-mini"), \
         patch.object(settings, "reviewer_provider", ""), \
         patch.object(settings, "reviewer_model", "gpt-5"), \
         patch.object(settings, "openai_api_key", ""), \
         patch.object(settings, "anthropic_api_key", "sk-ant-present"), \
         caplog.at_level("ERROR", logger="startup.models"):
        _check_model_config()  # must not raise

    assert "OPENAI_API_KEY is not set" in caplog.text
    assert "extraction and reviewer" in caplog.text
    assert "ANTHROPIC_API_KEY" not in caplog.text


@pytest.fixture(autouse=True)
def _fresh_providers():
    """Fresh provider instance (and once-only warning set) per test."""
    from app.nodes.llm.providers import reset_provider_cache

    reset_provider_cache()
    yield
    reset_provider_cache()


def test_get_client_sends_workspace_header_only_when_configured(monkeypatch):
    """An organization-level key needs anthropic-workspace-id on every
    request; a workspace-scoped key must not get a stray header."""
    from app.nodes.llm.providers import anthropic_provider as ap

    captured: dict = {}

    class FakeClient:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(ap.anthropic, "AsyncAnthropic", FakeClient)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

    from app.config import settings

    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    monkeypatch.setattr(settings, "anthropic_workspace_id", "")
    ap.get_client()
    assert "default_headers" not in captured

    captured.clear()
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_01TEST")
    ap.get_client()
    assert captured["default_headers"] == {"anthropic-workspace-id": "wrkspc_01TEST"}
