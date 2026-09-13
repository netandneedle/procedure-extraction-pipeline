"""Refang IOCs — convert defanged forms back to canonical fanged values.

CTI reports routinely defang IOCs to keep them inert in shared documents:
    domain[.]com         → domain.com
    192[.]168[.]1[.]1    → 192.168.1.1
    hxxps://evil.com     → https://evil.com
    user[at]domain.com   → user@domain.com

Defanged values cause two downstream problems:
1. STIX SCOs require canonical forms (`domain.com`, not `domain[.]com`).
2. The IoC-linking pass matches `chunk.artifacts` values against entity
   values; if either side is defanged and the other isn't, no match.

Refang at extraction (entity-side) AND at artifact normalization (chunk-side)
guarantees both ends agree without coupling the chunker to the entity
extractor's output shape.

Coverage: the patterns below cover the defang styles most commonly seen in
vendor reports (Mandiant, Trend Micro, Talos, Unit42, FBI/CISA advisories).
Edge cases (unicode lookalikes, base64-defanged URLs) are not handled here;
add patterns as real CTI corpora surface them.

Idempotent — calling refang on an already-fanged value returns it unchanged.
"""

from __future__ import annotations

import re

# Bracketed-dot defang: `domain[.]com`, `192[.]168[.]1[.]1`. The
# canonical and most common form. Also handles `(.)`, `{.}`.
_BRACKETED_DOT_RE = re.compile(r"[\[\(\{]\s*\.\s*[\]\)\}]")

# Bracketed-at defang: `user[at]domain.com`, `user(at)domain.com`,
# `user [at] domain.com`. Case-insensitive. Word-boundary on the right
# to avoid mangling phrases like "[ATTRIBUTION]".
_BRACKETED_AT_RE = re.compile(r"[\[\(\{]\s*at\s*[\]\)\}]", re.IGNORECASE)

# Bracketed-colon defang for ports: `evil[.]com[:]443` → `evil.com:443`.
_BRACKETED_COLON_RE = re.compile(r"[\[\(\{]\s*:\s*[\]\)\}]")

# Bracketed-slash defang for URL paths: `evil[.]com[/]path` → `evil.com/path`.
_BRACKETED_SLASH_RE = re.compile(r"[\[\(\{]\s*/\s*[\]\)\}]")

# URL scheme defang: `hxxp://`, `hxxps://`, `hXXp://`, `hxxtp://` (rare).
# Anchored on a word boundary so `hxxp` substrings without `://` aren't touched.
_HXXP_RE = re.compile(r"\bhxx(t?ps?)://", re.IGNORECASE)

# `fxp://` defang for FTP (less common but real).
_FXP_RE = re.compile(r"\bfxp://", re.IGNORECASE)

# Backslash-dot defang: `domain\.com` → `domain.com`. Note: only inside
# values that already look IOC-shaped; we shouldn't touch regex-style
# escape sequences in arbitrary text. Caller controls scope.
_BACKSLASH_DOT_RE = re.compile(r"\\\.")


def refang(value: str) -> str:
    """Refang a single IOC value. Idempotent.

    Handles the most common defang patterns seen in vendor CTI reports.
    Returns the original value (with whitespace stripped) when it's
    already in canonical form.
    """
    if not isinstance(value, str):
        return value
    out = value.strip()
    if not out:
        return out

    # URL scheme refangs first — they may overlap with bracket patterns.
    # group(1) is the trailing `(t?ps?)`: `p`, `ps`, `tp`, or `tps`. Build
    # the canonical scheme by mapping `p`/`ps`→`http(s)`, `tp`/`tps`→`http(s)`.
    def _scheme_sub(m: re.Match) -> str:
        suffix = m.group(1).lower()
        # Normalize: 'tp'→'p', 'tps'→'ps' (i.e. drop the leading 't').
        if suffix.startswith("t"):
            suffix = suffix[1:]
        return f"http{'s' if suffix.endswith('s') else ''}://"
    out = _HXXP_RE.sub(_scheme_sub, out)
    out = _FXP_RE.sub("ftp://", out)

    # Bracketed-character defangs (most common).
    out = _BRACKETED_DOT_RE.sub(".", out)
    out = _BRACKETED_AT_RE.sub("@", out)
    out = _BRACKETED_COLON_RE.sub(":", out)
    out = _BRACKETED_SLASH_RE.sub("/", out)

    # Backslash-dot refang for paths/domains. Cheap to apply broadly because
    # we already filtered for non-empty IOC-shaped values upstream.
    out = _BACKSLASH_DOT_RE.sub(".", out)

    return out


def refang_artifacts(
    artifacts: dict[str, list[str]],
) -> dict[str, list[str]]:
    """Refang every value in a chunk-artifacts dict, preserving categories
    and per-category dedup-after-refang ordering.
    """
    if not artifacts:
        return {}
    out: dict[str, list[str]] = {}
    for category, values in artifacts.items():
        if not isinstance(values, list):
            continue
        seen: set[str] = set()
        canonical: list[str] = []
        for v in values:
            refanged = refang(v) if isinstance(v, str) else None
            if not refanged or refanged in seen:
                continue
            seen.add(refanged)
            canonical.append(refanged)
        if canonical:
            out[category] = canonical
    return out
