"""Unit tests for the OpenAI provider.

Runs without any network — the SDK client is patched. These tests are the
whole safety net for the translation matrix, since nothing else in the suite
exercises a non-Anthropic wire shape.

    pytest tests/test_openai_provider.py -v
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "test-key-not-real")

from app.nodes.llm.providers import image_block, reset_provider_cache, text_block  # noqa: E402
from app.nodes.llm.providers import openai_provider  # noqa: E402
from app.nodes.llm.providers.openai_provider import OpenAIProvider  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_providers():
    """Fresh provider instance (and once-only warning set) per test."""
    reset_provider_cache()
    yield
    reset_provider_cache()

TOOL = {
    "name": "widget_tool",
    "description": "Emit widgets.",
    "input_schema": {"type": "object", "properties": {"a": {"type": "string"}}},
}
CHOICE = {"type": "tool", "name": "widget_tool"}


# =============================================================================
# Fakes
# =============================================================================


def _fake_completion(
    *,
    arguments: str | None = '{"a": "x"}',
    name: str = "widget_tool",
    content: str = "",
    finish_reason: str = "tool_calls",
    prompt_tokens: int = 100,
    completion_tokens: int = 20,
    cached_tokens: int = 30,
    with_usage_details: bool = True,
):
    tool_calls = None
    if arguments is not None:
        tool_calls = [SimpleNamespace(
            function=SimpleNamespace(name=name, arguments=arguments)
        )]

    details = (
        SimpleNamespace(cached_tokens=cached_tokens) if with_usage_details else None
    )
    return SimpleNamespace(
        choices=[SimpleNamespace(
            message=SimpleNamespace(tool_calls=tool_calls, content=content),
            finish_reason=finish_reason,
        )],
        usage=SimpleNamespace(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            prompt_tokens_details=details,
        ),
        model="gpt-5-2026-01-01",
    )


def _client_returning(completion):
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=completion)
    return client


async def _run(**overrides):
    """Drive complete() and return (ProviderResponse, sent_kwargs)."""
    completion = overrides.pop("completion", None) or _fake_completion()
    client = _client_returning(completion)
    call = dict(
        model="gpt-5",
        system="sys prompt",
        messages=[{"role": "user", "content": "hello"}],
        tools=[TOOL],
        tool_choice=CHOICE,
        max_tokens=1000,
        wire_kwargs={"reasoning_effort": "high"},
    )
    call.update(overrides)
    with patch.object(openai_provider, "get_client", return_value=client):
        resp = await OpenAIProvider().complete(**call)
    return resp, client.chat.completions.create.call_args.kwargs


# =============================================================================
# 1. Capability resolution
# =============================================================================


def test_reasoning_model_gets_effort_and_no_temperature():
    params = OpenAIProvider().resolve_params("gpt-5", temperature=0.0, effort="high")
    assert params.effort == "high"
    assert params.wire_kwargs == {"reasoning_effort": "high"}
    assert "temperature" not in params.wire_kwargs


@pytest.mark.parametrize("model", ["gpt-5", "gpt-5-mini", "o1", "o3-mini", "o4-mini"])
def test_reasoning_families_are_all_recognized(model):
    assert OpenAIProvider().resolve_params(model, temperature=0.0, effort="low").effort == "low"


def test_non_reasoning_model_gets_temperature():
    params = OpenAIProvider().resolve_params("gpt-4o", temperature=0.0, effort="high")
    assert params.effort is None
    assert params.wire_kwargs == {"temperature": 0.0}


def test_anthropic_only_effort_is_clamped_not_sent(caplog):
    """'max' is valid on Anthropic and a 400 on OpenAI. Clamp, and say so."""
    with caplog.at_level("WARNING", logger=openai_provider.__name__):
        params = OpenAIProvider().resolve_params("gpt-5", temperature=0.0, effort="max")

    assert params.effort == "high"
    assert params.wire_kwargs == {"reasoning_effort": "high"}
    assert "max" in caplog.text


# =============================================================================
# 2. Request translation
# =============================================================================


async def test_system_prompt_becomes_the_first_message():
    """OpenAI has no top-level `system` parameter; Anthropic does."""
    _, sent = await _run()
    assert "system" not in sent
    assert sent["messages"][0] == {"role": "system", "content": "sys prompt"}
    assert sent["messages"][1]["content"] == "hello"


async def test_tool_schema_is_translated_to_the_function_shape():
    _, sent = await _run()
    assert sent["tools"] == [{
        "type": "function",
        "function": {
            "name": "widget_tool",
            "description": "Emit widgets.",
            "parameters": TOOL["input_schema"],
        },
    }]


async def test_forced_tool_choice_is_translated():
    _, sent = await _run()
    assert sent["tool_choice"] == {
        "type": "function", "function": {"name": "widget_tool"},
    }


async def test_unnamed_tool_choice_becomes_required_not_auto():
    """Every node depends on a tool actually being called."""
    _, sent = await _run(tool_choice={"type": "any"})
    assert sent["tool_choice"] == "required"


async def test_absent_tool_choice_is_omitted():
    _, sent = await _run(tool_choice=None)
    assert "tool_choice" not in sent


async def test_reasoning_model_gets_max_completion_tokens():
    """Reasoning models reject max_tokens outright."""
    _, sent = await _run(model="gpt-5")
    assert sent["max_completion_tokens"] == 1000
    assert "max_tokens" not in sent


async def test_non_reasoning_model_gets_max_tokens():
    """The widely-supported spelling, for third-party gateways."""
    _, sent = await _run(model="gpt-4o", wire_kwargs={"temperature": 0.0})
    assert sent["max_tokens"] == 1000
    assert "max_completion_tokens" not in sent


async def test_image_block_becomes_a_data_uri():
    _, sent = await _run(
        messages=[{"role": "user", "content": [
            image_block(media_type="image/jpeg", data="QUJD"),
            text_block("caption"),
        ]}],
    )
    content = sent["messages"][1]["content"]
    assert content[0] == {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64,QUJD"},
    }
    assert content[1] == {"type": "text", "text": "caption"}


async def test_translation_does_not_mutate_the_callers_messages():
    original = [{"role": "user", "content": [image_block(media_type="image/png", data="B")]}]
    snapshot = [{"role": "user", "content": [
        {"type": "image", "media_type": "image/png", "data": "B"}
    ]}]
    await _run(messages=original)
    assert original == snapshot


# =============================================================================
# 3. Response normalization
# =============================================================================


async def test_tool_arguments_json_string_is_parsed():
    resp, _ = await _run(completion=_fake_completion(arguments='{"a": "x", "n": 2}'))
    assert resp.tool_output == {"a": "x", "n": 2}
    assert resp.stop_reason == "tool_use"
    assert resp.model == "gpt-5-2026-01-01"


async def test_windows_paths_survive_the_escape_repair():
    """The exact failure this pipeline hits — command lines full of C:\\Windows.

    Anthropic hands back a parsed object; OpenAI hands back a string, so the
    repair is on the main path here, not a fallback.
    """
    bad = '{"cmd": "C:\\Windows\\System32\\cmd.exe"}'
    resp, _ = await _run(completion=_fake_completion(arguments=bad))
    assert resp.tool_output["cmd"] == r"C:\Windows\System32\cmd.exe"


async def test_unparseable_arguments_degrade_to_no_tool_call():
    """call_llm already retries an absent tool call; reuse that path."""
    resp, _ = await _run(completion=_fake_completion(arguments="{not json at all"))
    assert resp.tool_output == {}


async def test_non_object_arguments_degrade_to_no_tool_call():
    resp, _ = await _run(completion=_fake_completion(arguments='["a", "b"]'))
    assert resp.tool_output == {}


async def test_text_only_turn_returns_empty_tool_output():
    resp, _ = await _run(completion=_fake_completion(
        arguments=None, content="I can't help with that.", finish_reason="content_filter",
    ))
    assert resp.tool_output == {}
    assert resp.raw_text == "I can't help with that."
    assert resp.stop_reason == "refusal"


@pytest.mark.parametrize("finish,expected", [
    ("tool_calls", "tool_use"),
    ("stop", "end_turn"),
    ("length", "max_tokens"),
    ("content_filter", "refusal"),
    ("something_new", "something_new"),
])
async def test_finish_reason_is_normalized(finish, expected):
    """stop_reason is persisted in llm_cache, so one vocabulary across vendors."""
    resp, _ = await _run(
        completion=_fake_completion(arguments=None, finish_reason=finish),
    )
    assert resp.stop_reason == expected


async def test_forced_tool_call_with_finish_stop_is_tool_use():
    """OpenAI reports finish_reason="stop" when the tool was FORCED.

    That is every call in this pipeline. The normalized vocabulary says a turn
    carrying a tool call is tool_use, whatever the vendor called it, or the
    llm_cache rows would split by vendor.
    """
    resp, _ = await _run(completion=_fake_completion(finish_reason="stop"))
    assert resp.tool_output == {"a": "x"}
    assert resp.stop_reason == "tool_use"


# =============================================================================
# 4. Token accounting — the silent-cost bug
# =============================================================================


async def test_cached_tokens_are_subtracted_from_input_tokens():
    """OpenAI's prompt_tokens INCLUDES cached; Anthropic's input_tokens excludes.

    ProviderResponse uses Anthropic's semantics. Reporting the raw
    prompt_tokens would double-count every cached token in the cost log.
    """
    resp, _ = await _run(completion=_fake_completion(
        prompt_tokens=100, completion_tokens=20, cached_tokens=30,
    ))
    assert resp.input_tokens == 70
    assert resp.cache_read_tokens == 30
    assert resp.output_tokens == 20
    # OpenAI bills no cache write, so there is genuinely nothing to report.
    assert resp.cache_creation_tokens == 0


async def test_missing_usage_details_are_tolerated():
    """Gateways commonly omit prompt_tokens_details entirely."""
    resp, _ = await _run(completion=_fake_completion(
        prompt_tokens=50, with_usage_details=False,
    ))
    assert resp.input_tokens == 50
    assert resp.cache_read_tokens == 0


async def test_input_tokens_never_goes_negative():
    """Defensive: a gateway reporting cached > prompt must not underflow."""
    resp, _ = await _run(completion=_fake_completion(
        prompt_tokens=10, cached_tokens=99,
    ))
    assert resp.input_tokens == 0


# =============================================================================
# 5. Client construction
# =============================================================================


def test_base_url_is_passed_through_for_compatible_gateways():
    with patch.dict(os.environ, {"OPENAI_BASE_URL": "https://gateway.example/v1"}):
        client = openai_provider.get_client()
    assert "gateway.example" in str(client.base_url)


def test_missing_key_raises_a_named_error():
    from app.config import settings

    with patch.dict(os.environ, {"OPENAI_API_KEY": ""}), \
         patch.object(settings, "openai_api_key", ""):
        with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
            openai_provider.get_client()
