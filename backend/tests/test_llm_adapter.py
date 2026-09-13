"""Unit tests for app.nodes.llm.llm_adapter.call_llm.

Covers the Pydantic-validation + retry behavior.
Runs without any network — anthropic.Anthropic is patched.

Run:
    pytest tests/test_llm_adapter.py -v
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel, ConfigDict, Field

# Ensure backend is importable
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

# Provide a dummy API key so get_client() doesn't throw in CI.
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-test-dummy")

from app.nodes.llm.errors import LLMValidationError  # noqa: E402
from app.nodes.llm import llm_adapter  # noqa: E402
from app.nodes.llm.llm_adapter import (  # noqa: E402
    LLMResponse,
    _coerce_json_string_fields,
    _expects_collection,
    call_llm,
)
from app.nodes.llm.providers import reset_provider_cache  # noqa: E402
from app.utils.text import repair_invalid_json_escapes as _repair_invalid_json_escapes  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_providers():
    """Each test gets a fresh provider instance.

    Providers now cache their SDK client on the instance; without this, the
    first test's patched get_client() would be served to every later test.
    """
    reset_provider_cache()
    yield
    reset_provider_cache()


# =============================================================================
# Shared fixtures / helpers
# =============================================================================


class _Strict(BaseModel):
    """Minimal strict model mirroring the tool_models._StrictBase pattern."""

    model_config = ConfigDict(extra="forbid")


class _WidgetOutput(_Strict):
    """A toy Pydantic model used as output_model across tests."""

    widgets: list[str]
    count: int = Field(ge=0)


def _make_anthropic_response(
    tool_input: dict | None,
    *,
    model: str = "claude-test",
    stop_reason: str = "tool_use",
    input_tokens: int = 10,
    output_tokens: int = 20,
    text: str = "",
) -> SimpleNamespace:
    """Build a fake anthropic response matching what call_llm consumes."""
    blocks: list[SimpleNamespace] = []
    if text:
        blocks.append(SimpleNamespace(type="text", text=text))
    if tool_input is not None:
        blocks.append(
            SimpleNamespace(
                type="tool_use",
                input=tool_input,
                name="widget_tool",
                id="toolu_test",
            )
        )
    return SimpleNamespace(
        content=blocks,
        model=model,
        stop_reason=stop_reason,
        usage=SimpleNamespace(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        ),
    )


def _default_call_args() -> dict:
    """Baseline kwargs for call_llm used across tests."""
    return {
        "system": "you are a widget factory",
        "messages": [{"role": "user", "content": "make widgets"}],
        "tools": [
            {
                "name": "widget_tool",
                "description": "Produce widgets.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "widgets": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "count": {"type": "integer"},
                    },
                    "required": ["widgets", "count"],
                },
            }
        ],
        "tool_choice": {"type": "tool", "name": "widget_tool"},
    }


@pytest.fixture
def fake_client():
    """Patch the Anthropic provider's get_client() to return a MagicMock.

    The test body configures client.messages.create.side_effect to drive
    call/response behavior. messages.create is an AsyncMock because the
    real AsyncAnthropic returns a coroutine; tests `await` it.
    """
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock()

    # call_llm streams rather than calling messages.create directly (it needs
    # the full output window, which the SDK won't grant a non-streaming
    # request). Bridge stream -> create so every test below still drives
    # responses through `create.side_effect` / `.return_value` and still reads
    # per-attempt arguments off `create.call_args_list`.
    class _StreamManager:
        def __init__(self, **kwargs):
            self._kwargs = kwargs

        async def __aenter__(self):
            message = await client.messages.create(**self._kwargs)
            stream = MagicMock()
            stream.get_final_message = AsyncMock(return_value=message)
            return stream

        async def __aexit__(self, *exc_info):
            return False

    client.messages.stream = lambda **kwargs: _StreamManager(**kwargs)

    # Patched on the PROVIDER, not the adapter: client construction moved
    # there when the provider seam was extracted. Patching
    # llm_adapter would appear to work and silently hit the network.
    from app.nodes.llm.providers import anthropic_provider

    with patch.object(anthropic_provider, "get_client", return_value=client):
        yield client


# =============================================================================
# 1. Happy path with output_model
# =============================================================================


async def test_call_llm_returns_validated_when_output_model_provided(fake_client):
    """With output_model set and a valid tool_output, returns the Pydantic
    instance AND the raw dict. Single API call, attempts=1."""
    good_output = {"widgets": ["alpha", "beta"], "count": 2}
    fake_client.messages.create.return_value = _make_anthropic_response(good_output)

    resp = await call_llm(**_default_call_args(), output_model=_WidgetOutput)

    assert fake_client.messages.create.call_count == 1
    assert isinstance(resp, LLMResponse)
    assert resp.attempts == 1
    assert resp.tool_output == good_output
    assert isinstance(resp.validated, _WidgetOutput)
    assert resp.validated.widgets == ["alpha", "beta"]
    assert resp.validated.count == 2
    assert resp.input_tokens == 10
    assert resp.output_tokens == 20
    assert resp.stop_reason == "tool_use"


# =============================================================================
# 2. Retry on validation error then succeeds
# =============================================================================


async def test_call_llm_retries_on_validation_error_and_succeeds(fake_client):
    """First response has an invalid shape (count negative). Adapter sends
    a correction message and the second call returns a good payload."""
    bad = {"widgets": ["x"], "count": -1}  # violates ge=0
    good = {"widgets": ["x", "y"], "count": 2}
    fake_client.messages.create.side_effect = [
        _make_anthropic_response(bad, input_tokens=5, output_tokens=15),
        _make_anthropic_response(good, input_tokens=7, output_tokens=9),
    ]

    resp = await call_llm(
        **_default_call_args(),
        output_model=_WidgetOutput,
        validation_retries=1,
    )

    assert fake_client.messages.create.call_count == 2
    assert resp.attempts == 2
    assert resp.tool_output == good
    assert isinstance(resp.validated, _WidgetOutput)
    # Token counts should accumulate across attempts.
    assert resp.input_tokens == 5 + 7
    assert resp.output_tokens == 15 + 9


# =============================================================================
# 3. Retry budget exhausted → LLMValidationError
# =============================================================================


async def test_call_llm_raises_llm_validation_error_after_retries_exhausted(fake_client):
    """Both attempts return invalid shape → LLMValidationError with
    attempts=2 and the first errors populated."""
    bad_a = {"widgets": "not-a-list", "count": 1}  # wrong type
    bad_b = {"widgets": [1, 2], "count": "nine"}  # still wrong
    fake_client.messages.create.side_effect = [
        _make_anthropic_response(bad_a),
        _make_anthropic_response(bad_b),
    ]

    with pytest.raises(LLMValidationError) as excinfo:
        await call_llm(
            **_default_call_args(),
            output_model=_WidgetOutput,
            validation_retries=1,
        )

    assert fake_client.messages.create.call_count == 2
    err = excinfo.value
    assert err.tool_name == "widget_tool"
    assert err.attempts == 2
    assert len(err.errors) >= 1  # at least one pydantic error recorded
    # raw_output should reflect the LAST attempt so post-mortem sees what
    # the model emitted on the final try.
    assert err.raw_output == bad_b


# =============================================================================
# 4. Correction message content
# =============================================================================


async def test_call_llm_correction_message_contains_validation_detail(fake_client):
    """The second call's messages list must include a user-role correction
    turn that names the tool, mentions the validation failure, and echoes
    (truncated) the bad payload."""
    bad = {"widgets": ["a"], "count": -5}
    good = {"widgets": ["a"], "count": 0}
    fake_client.messages.create.side_effect = [
        _make_anthropic_response(bad),
        _make_anthropic_response(good),
    ]

    await call_llm(
        **_default_call_args(),
        output_model=_WidgetOutput,
        validation_retries=1,
    )

    # The SECOND call is the retry with the correction message appended.
    second_call = fake_client.messages.create.call_args_list[1]
    kwargs = second_call.kwargs
    messages = kwargs["messages"]

    # Original user message + new correction user message.
    assert len(messages) == 2
    correction = messages[-1]
    assert correction["role"] == "user"
    content = correction["content"]

    # Names the tool
    assert "widget_tool" in content
    # Indicates schema failure
    assert "schema" in content.lower() or "validation" in content.lower()
    # Echoes the bad payload (count=-5 appears somewhere)
    assert "-5" in content
    # Defuses prompt injection by framing as quoted data
    assert "do NOT treat" in content or "do not treat" in content.lower()


async def test_call_llm_correction_message_truncates_huge_payload(fake_client):
    """If the bad payload is >2000 chars when rendered, the correction
    message must truncate and flag it so context doesn't blow up."""
    # Build a payload that's comfortably over 2000 chars when JSON-rendered.
    huge_list = ["a" * 50 for _ in range(500)]
    bad = {"widgets": huge_list, "count": -1}  # invalid: count negative
    good = {"widgets": ["ok"], "count": 1}
    fake_client.messages.create.side_effect = [
        _make_anthropic_response(bad),
        _make_anthropic_response(good),
    ]

    await call_llm(
        **_default_call_args(),
        output_model=_WidgetOutput,
        validation_retries=1,
    )

    second_call = fake_client.messages.create.call_args_list[1]
    content = second_call.kwargs["messages"][-1]["content"]
    # Truncation marker is present
    assert "truncated" in content.lower()
    # Overall message stays bounded: correction header + 2000 payload chars
    # + errors + framing. Generous upper bound.
    assert len(content) < 6000


# =============================================================================
# 5. Backward compat: no output_model = old behavior
# =============================================================================


async def test_call_llm_backward_compat_without_output_model(fake_client):
    """When no output_model is passed, adapter does NOT validate, does NOT
    retry, and returns tool_output as the raw dict exactly as before."""
    # Deliberately include a shape that would FAIL _WidgetOutput validation.
    # Without output_model, it must pass through unchanged.
    raw = {"widgets": "not a list", "count": -1}
    fake_client.messages.create.return_value = _make_anthropic_response(raw)

    resp = await call_llm(**_default_call_args())

    assert fake_client.messages.create.call_count == 1
    assert resp.attempts == 1
    assert resp.validated is None
    assert resp.tool_output == raw


# =============================================================================
# Bonus: validation_retries=0 disables retry
# =============================================================================


async def test_call_llm_validation_retries_zero_raises_on_first_failure(fake_client):
    """validation_retries=0 → exactly one API call, LLMValidationError raised
    with attempts=1. Useful for callers that want fail-fast behavior."""
    bad = {"widgets": "nope", "count": 0}
    fake_client.messages.create.return_value = _make_anthropic_response(bad)

    with pytest.raises(LLMValidationError) as excinfo:
        await call_llm(
            **_default_call_args(),
            output_model=_WidgetOutput,
            validation_retries=0,
        )

    assert fake_client.messages.create.call_count == 1
    assert excinfo.value.attempts == 1


# =============================================================================
# Bonus: tool_choice present but no tool_use block in response → ValueError
# =============================================================================


async def test_call_llm_raises_when_tool_forced_but_no_tool_use_returned(fake_client):
    """Forced a tool, got text only, and every retry did the same: raise.

    Quietly returning empty tool_output would push the failure downstream to
    whichever node dereferences it. The message names the tool, the stop
    reason, and how many attempts were spent.
    """
    fake_client.messages.create.return_value = _make_anthropic_response(
        tool_input=None,
        stop_reason="end_turn",
        text="Sorry, I cannot do that.",
    )

    with pytest.raises(ValueError) as excinfo:
        await call_llm(**_default_call_args(), validation_retries=1)

    msg = str(excinfo.value)
    assert "widget_tool" in msg
    assert "end_turn" in msg
    assert fake_client.messages.create.await_count == 2, (
        "a refusal must be retried before it kills the call"
    )


async def test_call_llm_retries_a_refusal_and_succeeds(fake_client):
    """A refusal is transient, so it must not cost the whole node.

    Two of three IDENTICAL `extract_techniques` runs over the
    one actor-profile report came back `stop_reason='refusal'` and the third
    succeeded, dropping that source out of an ablation arm. The adapter was
    raising straight out of the retry loop, so the retry machinery that
    already existed for validation errors never engaged.
    """
    fake_client.messages.create.side_effect = [
        _make_anthropic_response(tool_input=None, stop_reason="refusal"),
        _make_anthropic_response({"widgets": ["a"], "count": 1}),
    ]

    resp = await call_llm(**_default_call_args(), validation_retries=1)

    assert resp.tool_output == {"widgets": ["a"], "count": 1}
    assert fake_client.messages.create.await_count == 2


async def test_call_llm_refusal_respects_zero_retries(fake_client):
    """validation_retries=0 still means one attempt, refusal included."""
    fake_client.messages.create.return_value = _make_anthropic_response(
        tool_input=None, stop_reason="refusal",
    )

    with pytest.raises(ValueError):
        await call_llm(**_default_call_args(), validation_retries=0)

    assert fake_client.messages.create.await_count == 1


# =============================================================================
# Stringified-collection coercion (a model emitted collections as JSON strings)
# =============================================================================


class _CoerceModel(_Strict):
    """Model with mixed collection / scalar fields for coercion tests."""

    widgets: list[str]
    metadata: dict[str, str] | None = None
    note: str | None = None
    count: int = Field(ge=0)


def test_expects_collection_detects_bare_and_generic_and_optional():
    """The type-inspection helper covers bare/generic/Optional/Union."""
    assert _expects_collection(list) is True
    assert _expects_collection(dict) is True
    assert _expects_collection(list[str]) is True
    assert _expects_collection(dict[str, int]) is True
    assert _expects_collection(list[str] | None) is True
    assert _expects_collection(str) is False
    assert _expects_collection(int) is False
    assert _expects_collection(str | None) is False


def test_coerce_leaves_well_formed_dict_untouched():
    """Already-valid dict should be returned as-is (identity, not copy)."""
    raw = {"widgets": ["a", "b"], "count": 2}
    result = _coerce_json_string_fields(raw, _CoerceModel)
    assert result is raw


def test_coerce_parses_stringified_list_in_collection_field():
    """The observed bug: whole list arrives as JSON string."""
    raw = {"widgets": '["a", "b", "c"]', "count": 3}
    result = _coerce_json_string_fields(raw, _CoerceModel)
    assert result is not raw  # new dict
    assert result["widgets"] == ["a", "b", "c"]
    assert result["count"] == 3


def test_coerce_parses_stringified_dict_in_optional_dict_field():
    """Optional[dict] fields are also in scope for coercion."""
    raw = {"widgets": [], "metadata": '{"k": "v"}', "count": 0}
    result = _coerce_json_string_fields(raw, _CoerceModel)
    assert result["metadata"] == {"k": "v"}


def test_coerce_ignores_string_fields_even_if_starts_with_bracket():
    """A field declared as str must not be coerced, even if its value looks
    like JSON. Prevents spurious transforms on legitimate prose."""
    raw = {"widgets": [], "note": "[this is a bracketed note]", "count": 0}
    result = _coerce_json_string_fields(raw, _CoerceModel)
    # note is a str field → untouched; nothing coerced → identity.
    assert result is raw
    assert result["note"] == "[this is a bracketed note]"


def test_coerce_leaves_malformed_json_string_for_pydantic_to_reject():
    """If a collection field is a string that starts with [ or { but isn't
    valid JSON, leave it alone so pydantic's clear error fires."""
    raw = {"widgets": "[unterminated, bad", "count": 0}
    result = _coerce_json_string_fields(raw, _CoerceModel)
    # No successful coercion → original returned untouched.
    assert result is raw
    assert result["widgets"] == "[unterminated, bad"


async def test_call_llm_coerces_stringified_list_without_retry(fake_client):
    """End-to-end: a stringified `widgets` field validates on the FIRST call
    thanks to coercion — no retry, no correction turn."""
    stringified = {"widgets": '["alpha", "beta"]', "count": 2}
    fake_client.messages.create.return_value = _make_anthropic_response(stringified)

    resp = await call_llm(**_default_call_args(), output_model=_WidgetOutput)

    assert fake_client.messages.create.call_count == 1
    assert resp.attempts == 1
    assert isinstance(resp.validated, _WidgetOutput)
    assert resp.validated.widgets == ["alpha", "beta"]
    # The returned tool_output carries the coerced (list) shape, not the
    # original string, so downstream callers don't re-parse.
    assert resp.tool_output["widgets"] == ["alpha", "beta"]


# =============================================================================
# Invalid-escape repair (nested-JSON single-escape bug)
# =============================================================================


class _DraftItem(_Strict):
    """Mirror of tool_models.DraftItem — exercises the real drafting shape."""

    chunk_id: str
    name: str
    description: str
    platforms: list[str]
    command_lines: list[str]
    detail_gap: bool


class _DraftProceduresOutput(_Strict):
    drafts: list[_DraftItem]


def test_repair_leaves_valid_escapes_alone():
    """Legal JSON escapes (``\\n``, ``\\"``, ``\\\\``, ``\\u00e9``) must not be
    disturbed."""
    legal = r'line1\nline2 quote \" backslash \\ unicode \u00e9'
    assert _repair_invalid_json_escapes(legal) == legal


def test_repair_fixes_odd_run_before_invalid_char():
    """``\\W`` (invalid escape) becomes ``\\\\W`` so it parses to literal
    ``\\W``."""
    import json as _json

    # Raw string that would fail json.loads: it's one string token containing
    # a literal invalid \W escape.
    bad = r'"C:\Windows\System32"'
    with pytest.raises(_json.JSONDecodeError):
        _json.loads(bad)
    repaired = _repair_invalid_json_escapes(bad)
    # After repair, the string parses and the literal backslashes survive.
    assert _json.loads(repaired) == r"C:\Windows\System32"


def test_repair_preserves_even_runs():
    """``\\\\`` is already a legal escaped backslash — don't touch it even if
    followed by what would otherwise be an invalid char."""
    # Raw token: "a\\Z" — an even (2) backslash run followed by Z. Legal JSON:
    # parses to "a\\Z" → literal a\Z.
    legal = r'"a\\Z"'
    import json as _json

    assert _json.loads(legal) == r"a\Z"
    assert _repair_invalid_json_escapes(legal) == legal


def test_repair_handles_multiple_bad_escapes_in_one_string():
    """Two invalid escapes in the same token should both be fixed.

    Uses capital-letter path segments (``\\U``, ``\\D``, ``\\A``) so every
    backslash precedes an invalid escape char — avoids accidentally hitting
    ``\\t``/``\\n``/``\\b`` which are legal JSON escapes."""
    import json as _json

    bad = r'"path1=C:\Users\Admin path2=D:\Data\Archive"'
    with pytest.raises(_json.JSONDecodeError):
        _json.loads(bad)
    repaired = _repair_invalid_json_escapes(bad)
    assert _json.loads(repaired) == r"path1=C:\Users\Admin path2=D:\Data\Archive"


def test_repair_mixed_valid_and_invalid_escapes_in_one_string():
    """Repair must fix only the invalid escapes and leave valid ones alone.

    Production drafts often have a description with `\\n` newlines AND a
    command_line with a Windows path `\\Windows`. Naive repair could corrupt
    the legal escapes. We want `\\n` preserved and `\\W` fixed."""
    import json as _json

    # Raw JSON token: "step1\nrun C:\Windows\System32\cmd"
    # \n is legal (newline), \W \S \c are all invalid.
    bad = r'"step1\nrun C:\Windows\System32\cmd"'
    with pytest.raises(_json.JSONDecodeError):
        _json.loads(bad)
    repaired = _repair_invalid_json_escapes(bad)
    parsed = _json.loads(repaired)
    # Newline survives as a real newline; backslash path survives as literal
    # backslashes.
    assert parsed == "step1\nrun C:\\Windows\\System32\\cmd"


def test_repair_trailing_backslash_at_end_of_string():
    """A trailing backslash (no following char) is an odd run of 1 with j==n;
    the condition `j < n` guards this — we leave it for json.loads to error
    on rather than silently inserting a backslash."""
    s = "abc\\"
    # No change — we can't tell the fix without a following char.
    assert _repair_invalid_json_escapes(s) == "abc\\"


def test_coerce_repairs_stringified_drafts_with_windows_paths():
    """The production failure mode: `drafts` arrives as a JSON string whose
    inner string values contain single-escaped Windows paths that make
    json.loads choke with 'Invalid \\escape'. The repair should rescue it."""
    # A stringified drafts array. The inner `command_lines` string contains
    # literal `\W`, `\S`, `\c` — all invalid JSON escapes that trip json.loads.
    # This mirrors what Claude emits on the extract_techniques / drafting bug.
    # Note: we deliberately avoid path chars like `\t`, `\b`, `\n`, `\r`, `\f`,
    # `\u` because those are LEGAL JSON escapes and wouldn't exercise repair.
    stringified = (
        '[{"chunk_id":"c1","name":"Run payload","description":"exec",'
        r'"platforms":["Windows"],"command_lines":["C:\Windows\System32\cmd.exe"],'
        '"detail_gap":false}]'
    )
    raw = {"drafts": stringified}

    result = _coerce_json_string_fields(raw, _DraftProceduresOutput)

    assert result is not raw  # coercion happened
    assert isinstance(result["drafts"], list)
    assert len(result["drafts"]) == 1
    draft = result["drafts"][0]
    assert draft["chunk_id"] == "c1"
    # The literal backslashes survived the repair — command line still points
    # at the Windows binary.
    assert draft["command_lines"] == [r"C:\Windows\System32\cmd.exe"]
    # Pydantic accepts the coerced shape.
    parsed = _DraftProceduresOutput.model_validate(result)
    assert parsed.drafts[0].command_lines == [r"C:\Windows\System32\cmd.exe"]


def test_coerce_repair_falls_through_when_input_is_unsalvageable():
    """If the repair can't fix the string (e.g. truly truncated JSON), the
    coerce helper must fall through and leave the raw dict untouched so
    pydantic's normal error path fires."""
    # Missing closing bracket — no escape trick can rescue this.
    raw = {"drafts": '[{"chunk_id":"c1","name":"broken'}
    result = _coerce_json_string_fields(raw, _DraftProceduresOutput)
    assert result is raw
    assert result["drafts"] == '[{"chunk_id":"c1","name":"broken'


async def test_call_llm_coercion_failure_still_sends_original_in_correction(fake_client):
    """When coercion can't rescue the payload (malformed JSON string), the
    correction message echoes the ORIGINAL string so Claude sees what it
    actually sent, not a partially-repaired shape."""
    bad = {"widgets": "[unterminated", "count": 0}
    good = {"widgets": ["ok"], "count": 1}
    fake_client.messages.create.side_effect = [
        _make_anthropic_response(bad),
        _make_anthropic_response(good),
    ]

    await call_llm(
        **_default_call_args(),
        output_model=_WidgetOutput,
        validation_retries=1,
    )

    second_call = fake_client.messages.create.call_args_list[1]
    content = second_call.kwargs["messages"][-1]["content"]
    # Original stringified value (with the opening bracket) appears in the
    # echoed preview; the coercion did NOT silently mutate what Claude saw.
    assert "[unterminated" in content


# =============================================================================
# LLM cache layer
# =============================================================================


@pytest.fixture(autouse=True)
def fake_cache():
    """Mock get_cached and write_cache where they're imported into the
    llm_adapter module. Reads/writes route through an in-fixture dict so
    multiple call_llm invocations in a single test see consistent state.

    autouse=True so EVERY test in this module runs against an isolated
    in-memory cache. Without this, tests that don't explicitly request
    the fixture would read/write the real llm_cache table inside the
    container, polluting both the production cache AND each other's
    state (a test's first call_llm would write, a sibling test's first
    call_llm would hit that cached entry and never call the API).

    Tests that need to inspect cache state (the 6 cache-specific tests)
    request the fixture by name; tests that don't care just get the
    patches applied silently.
    """
    store: dict[str, dict] = {}

    def fake_get(key):
        return store.get(key)

    def fake_write(key, payload):
        store.setdefault(key, dict(payload))

    with patch.object(llm_adapter, "get_cached", side_effect=fake_get) as get_mock, \
         patch.object(llm_adapter, "write_cache", side_effect=fake_write) as write_mock:
        yield SimpleNamespace(
            get_cached=get_mock,
            write_cache=write_mock,
            store=store,
        )


async def test_call_llm_writes_cache_on_first_success(fake_client, fake_cache):
    """First successful call writes one cache entry with the response."""
    good = {"widgets": ["a"], "count": 1}
    fake_client.messages.create.return_value = _make_anthropic_response(good)

    resp = await call_llm(**_default_call_args(), output_model=_WidgetOutput)

    assert resp.cached is False
    assert fake_cache.write_cache.call_count == 1
    assert len(fake_cache.store) == 1
    cached_payload = next(iter(fake_cache.store.values()))
    assert cached_payload["tool_output"] == good
    assert cached_payload["attempts"] == 1


async def test_call_llm_returns_cached_on_repeat(fake_client, fake_cache):
    """Second call with identical input returns cached, skips API entirely."""
    good = {"widgets": ["alpha"], "count": 1}
    fake_client.messages.create.return_value = _make_anthropic_response(good)

    resp1 = await call_llm(**_default_call_args(), output_model=_WidgetOutput)
    assert resp1.cached is False
    assert fake_client.messages.create.call_count == 1

    resp2 = await call_llm(**_default_call_args(), output_model=_WidgetOutput)
    assert resp2.cached is True
    assert fake_client.messages.create.call_count == 1  # unchanged
    assert resp2.tool_output == good
    assert isinstance(resp2.validated, _WidgetOutput)


async def test_call_llm_bypass_cache_skips_lookup_and_write(fake_client, fake_cache):
    """bypass_cache=True hits API, ignores cache entirely (no read, no write).
    Re-extract is ephemeral; cache state stays untouched."""
    good = {"widgets": ["a"], "count": 1}
    fake_client.messages.create.return_value = _make_anthropic_response(good)

    await call_llm(**_default_call_args(), output_model=_WidgetOutput, bypass_cache=True)
    assert fake_cache.get_cached.call_count == 0
    assert fake_cache.write_cache.call_count == 0

    await call_llm(**_default_call_args(), output_model=_WidgetOutput, bypass_cache=True)
    assert fake_client.messages.create.call_count == 2
    assert fake_cache.get_cached.call_count == 0
    assert fake_cache.write_cache.call_count == 0


async def test_call_llm_validation_failure_does_not_cache(fake_client, fake_cache):
    """When all retries fail, no cache row is written (no cache poisoning)."""
    bad_a = {"widgets": "not-a-list", "count": 1}
    bad_b = {"widgets": [1, 2], "count": "nine"}
    fake_client.messages.create.side_effect = [
        _make_anthropic_response(bad_a),
        _make_anthropic_response(bad_b),
    ]

    with pytest.raises(LLMValidationError):
        await call_llm(**_default_call_args(), output_model=_WidgetOutput, validation_retries=1)

    assert fake_client.messages.create.call_count == 2
    assert fake_cache.write_cache.call_count == 0
    assert len(fake_cache.store) == 0


async def test_call_llm_validation_retry_caches_final_output(fake_client, fake_cache):
    """First attempt validation-fails, retry succeeds. Cache holds the
    retry's output. A subsequent identical input returns cached and skips
    the failed first attempt entirely."""
    bad = {"widgets": "not-a-list", "count": 1}
    good = {"widgets": ["x"], "count": 1}
    fake_client.messages.create.side_effect = [
        _make_anthropic_response(bad),
        _make_anthropic_response(good),
    ]

    resp1 = await call_llm(**_default_call_args(), output_model=_WidgetOutput, validation_retries=1)
    assert resp1.cached is False
    assert resp1.attempts == 2
    assert resp1.tool_output == good
    assert len(fake_cache.store) == 1
    assert next(iter(fake_cache.store.values()))["tool_output"] == good

    fake_client.messages.create.reset_mock()
    fake_client.messages.create.side_effect = []  # would StopIteration if called
    resp2 = await call_llm(**_default_call_args(), output_model=_WidgetOutput, validation_retries=1)
    assert resp2.cached is True
    assert resp2.tool_output == good
    assert fake_client.messages.create.call_count == 0


async def test_call_llm_cache_key_changes_with_system_prompt(fake_client, fake_cache):
    """Different system prompts produce different cache keys.

    Proxy for catalogue-version-invalidation: when the ATT&CK catalogue
    text in the system prompt changes (e.g. v18 -> v19), cache keys change
    and old entries become unreachable.
    """
    good = {"widgets": ["a"], "count": 1}
    fake_client.messages.create.return_value = _make_anthropic_response(good)

    args_v1 = _default_call_args()
    args_v1["system"] = "you are a widget factory v1"
    args_v2 = _default_call_args()
    args_v2["system"] = "you are a widget factory v2"

    await call_llm(**args_v1, output_model=_WidgetOutput)
    await call_llm(**args_v2, output_model=_WidgetOutput)

    assert len(fake_cache.store) == 2
    assert fake_client.messages.create.call_count == 2


async def test_cache_version_bump_invalidates_every_entry(fake_client, fake_cache):
    """The lever for changes the prompt text can't express.

    A catalogue upgrade that only rewrites technique DESCRIPTIONS leaves the
    candidate pool (`id | name | tactics`) byte-identical, so the key above
    would not budge — while the descriptions still drive
    `_recalibrate_confidence`. Bumping CACHE_VERSION re-samples everything.
    """
    from app.nodes.llm import llm_adapter

    good = {"widgets": ["a"], "count": 1}
    fake_client.messages.create.return_value = _make_anthropic_response(good)

    resp1 = await call_llm(**_default_call_args(), output_model=_WidgetOutput)
    assert resp1.cached is False
    resp2 = await call_llm(**_default_call_args(), output_model=_WidgetOutput)
    assert resp2.cached is True  # same version -> served from cache

    original = llm_adapter.CACHE_VERSION
    try:
        llm_adapter.CACHE_VERSION = "2"
        resp3 = await call_llm(**_default_call_args(), output_model=_WidgetOutput)
    finally:
        llm_adapter.CACHE_VERSION = original

    assert resp3.cached is False
    assert len(fake_cache.store) == 2
    assert fake_client.messages.create.call_count == 2


async def test_cache_version_comes_from_settings():
    """Deployments pin their own version without a code change.

    The value is read from Settings (LLM_CACHE_VERSION via env or .env — a real
    env var wins, and a blank value falls through to the default rather than
    re-keying the whole cache). The adapter no longer reads os.environ itself.
    """
    import importlib
    from unittest.mock import patch

    from app.config import settings
    from app.nodes.llm import llm_adapter

    with patch.object(settings, "llm_cache_version", "cti-2026-08"):
        reloaded = importlib.reload(llm_adapter)
        try:
            assert reloaded.CACHE_VERSION == "cti-2026-08"
        finally:
            importlib.reload(llm_adapter)


async def test_truncated_tool_call_is_not_retried(fake_client, fake_cache):
    """A tool call cut off by max_tokens must not spend the refusal retry.

    It arrives as partial JSON, parses to {}, and looks exactly like a refusal
    — but re-sending the identical request with the identical budget
    truncates again. Two full generations billed for a deterministic failure.
    """
    fake_client.messages.create.return_value = _make_anthropic_response(
        tool_input=None, stop_reason="max_tokens",
    )

    with pytest.raises(ValueError, match="max_tokens"):
        await call_llm(**_default_call_args(), validation_retries=1)

    assert fake_client.messages.create.call_count == 1
