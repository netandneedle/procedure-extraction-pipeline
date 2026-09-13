"""Aggregate reviewer outcomes across sources — the readout assist mode exists for.

`outcomes.py` writes one diff per gate submit: what the reviewer recommended, what
the analyst actually did, item by item. This reads those back across every source and
answers the question the whole assist-first design was built to answer — *how often
does the analyst take this gate's advice?* — because that is what decides whether a
gate is ever safe to run unattended.

Pure functions over already-fetched rows, like `outcomes.py`, so the arithmetic is
testable without a database.

IN PYTHON, NOT SQL, deliberately. The shape needs per-item grouping by confidence tier
and moot-filtering, and a `jsonb_array_elements` query would encode the payload's shape
in a second place — the exact drift class tests/test_contracts.py exists to catch.
Volume is four rows per source.

FIVE WAYS THIS COULD LIE, each guarded below:

  1. A gate where the reviewer recommended nothing has `rate: None`, not 0%.
  2. `moot` items — a chunk-gate re-chunk discards three whole channels — are excluded.
     `diff_chunk_gate` already keeps them out of its `total`, but `items[]` still
     carries them, so anything walking items has to filter.
  3. Rates are POOLED, never averaged. Averaging would weigh a source with two
     recommendations the same as one with forty.
  4. No rate is ever emitted without the count it came from. 100% of 3 is not evidence.
  5. Rows the agent applied unattended (`outcome["auto"]`) are excluded entirely.
     They record the reviewer agreeing with itself; counting them would drive
     every rate to 100% and make this readout useless exactly when autopilot
     makes it most necessary.

And one thing it deliberately does not do: there is no "ready for autopilot" verdict
here, and no blended all-gates rate. The threshold is not a property of the number — a
wrong entity removal is recoverable at the next gate, a wrong chunk drop deletes a
procedure from the bundle with nothing downstream able to notice. Collapsing four gates
with different costs of error into one figure would launder a product decision into
arithmetic.
"""

from __future__ import annotations

from typing import Any, Iterable

from app.models.reviewer import INITIAL_READ, STATUS_FAILED

# Display order — the pipeline's gate order, not alphabetical.
GATE_ORDER: tuple[str, ...] = ("entities", "chunks", "procedures", "bundle")

# The tiers the reviewer calibrates against. Always all three, even at zero, so the
# UI renders a stable shape rather than a table that changes columns with the data.
CONFIDENCE_TIERS: tuple[str, ...] = ("high", "medium", "low")


def _rate(agreed: int, total: int) -> float | None:
    """Agreement rate, or None when there is nothing to rate.

    None rather than 0.0 on an empty denominator: "agreed with all zero of its
    recommendations" is not a 0% agreement rate, it is an absence of evidence, and
    the two must not render the same.
    """
    return round(agreed / total, 3) if total else None


def _tally() -> dict[str, Any]:
    return {"agreed": 0, "total": 0, "rate": None}


def _add(bucket: dict[str, Any], agreed: bool) -> None:
    bucket["total"] += 1
    bucket["agreed"] += 1 if agreed else 0


def _finalise(bucket: dict[str, Any]) -> dict[str, Any]:
    bucket["rate"] = _rate(bucket["agreed"], bucket["total"])
    return bucket


def scored_items(outcome: Any) -> list[dict[str, Any]]:
    """The items in one outcome that count toward agreement.

    Drops `moot` entries. At the chunk gate a re-chunk makes the decisions, adds and
    edges channels unreachable — the analyst never got to act on them. Counting those
    as disagreement would report a collapse that did not happen; counting them as
    agreement would invent consent.
    """
    if not isinstance(outcome, dict):
        return []
    items = (outcome.get("agreement") or {}).get("items") or []
    return [
        i for i in items
        if isinstance(i, dict) and not i.get("moot")
    ]


def aggregate(rows: Iterable[Any], *, max_overrides: int = 25) -> dict[str, Any]:
    """Per-gate agreement across every source that has been reviewed.

    `rows` are ReviewerRecommendation records (anything with `gate_key`, `source_id`,
    `status`, `outcome`); attribute access keeps this usable from tests without an ORM.

    Rows are expected newest-first, which is the order `store.load_outcomes` returns
    and the order the overrides list wants.
    """
    gates: dict[str, dict[str, Any]] = {}
    sources_reviewed: set[Any] = set()

    def gate_of(key: str) -> dict[str, Any]:
        if key not in gates:
            gates[key] = {
                "gate_key": key,
                # Distinct sources, and separately the number of REVIEWS — a gate can
                # be visited twice when a rejection routes back through it, and both
                # numbers matter: one is sample breadth, the other is sample size.
                "_sources": set(),
                "reviews": 0,
                "agreed": 0,
                "total": 0,
                "rate": None,
                "by_confidence": {t: _tally() for t in CONFIDENCE_TIERS},
                "by_kind": {},
                "unsupported_quotes": 0,
                "failures": 0,
                "overrides": [],
            }
        return gates[key]

    for row in rows:
        key = getattr(row, "gate_key", None)
        # The opening read is not a gate decision and carries no outcome by design.
        # Unknown keys are KEPT rather than dropped: a fifth gate should show up here
        # as soon as it ships, not vanish because this list predates it.
        if not key or key == INITIAL_READ:
            continue

        gate = gate_of(key)
        source_id = getattr(row, "source_id", None)

        if getattr(row, "status", None) == STATUS_FAILED:
            # A reviewer that errors is not a candidate for unattended operation
            # whatever its agreement rate, so failures are counted rather than
            # silently skipped. They carry no outcome to score.
            gate["failures"] += 1
            continue

        outcome = getattr(row, "outcome", None)

        if isinstance(outcome, dict) and outcome.get("auto"):
            # Applied unattended: the reviewer's recommendation submitted as-is
            # with no human in between. Scoring it would compare the agent to
            # itself and return 100%, turning the one instrument that could
            # justify autonomy into a mirror. Not a failure and not a review —
            # simply not evidence about agreement, so it is counted as neither.
            continue

        items = scored_items(outcome)
        if not isinstance(outcome, dict):
            # Recommended but never submitted — the source is still paused at this
            # gate, or was deleted mid-review. Not evidence either way.
            continue

        gate["reviews"] += 1
        if source_id is not None:
            gate["_sources"].add(source_id)
            sources_reviewed.add(source_id)

        for item in items:
            agreed = bool(item.get("agreed"))
            gate["total"] += 1
            gate["agreed"] += 1 if agreed else 0

            tier = item.get("confidence")
            if tier in gate["by_confidence"]:
                _add(gate["by_confidence"][tier], agreed)

            kind = item.get("kind") or "unknown"
            _add(gate["by_kind"].setdefault(kind, _tally()), agreed)

            if item.get("quote_unsupported"):
                gate["unsupported_quotes"] += 1

            if not agreed and len(gate["overrides"]) < max_overrides:
                # The actionable half. A rate says whether a gate is trusted; an
                # override says why it should not be — on a real source, with the
                # reviewer's reasoning and the analyst's actual answer side by side.
                gate["overrides"].append({
                    "source_id": str(source_id) if source_id else None,
                    "kind": kind,
                    "ref": item.get("ref"),
                    "recommended": item.get("recommended"),
                    "actual": item.get("actual"),
                    "confidence": tier,
                    "quote_unsupported": bool(item.get("quote_unsupported")),
                    "rationale": item.get("rationale") or "",
                })

    ordered = sorted(
        gates.values(),
        key=lambda g: (
            GATE_ORDER.index(g["gate_key"])
            if g["gate_key"] in GATE_ORDER else len(GATE_ORDER)
        ),
    )
    for gate in ordered:
        gate["sources"] = len(gate.pop("_sources"))
        gate["rate"] = _rate(gate["agreed"], gate["total"])
        for tier in gate["by_confidence"].values():
            _finalise(tier)
        for kind in gate["by_kind"].values():
            _finalise(kind)

    return {
        # A sample-size fact, not a blended verdict — see the module docstring for
        # why there is no all-gates agreement rate.
        "sources_reviewed": len(sources_reviewed),
        "gates": ordered,
    }
