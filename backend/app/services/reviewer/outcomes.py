"""Diff what the AI reviewer recommended against what the analyst submitted.

This is the point of the whole reviewer_recommendations table. Deciding
whether a gate is ever safe to run unattended means asking, across many
sources: how often did the analyst take the reviewer's advice? That question
has to be answerable by query, and it only has an answer if a human was in
the loop — which is why assist mode ships before autopilot, and why assist
mode is as much an instrument as it is a product.

An OVERRIDE is the valuable outcome, not the embarrassing one. Agreement
tells you the reviewer is redundant or right; disagreement tells you exactly
where it is wrong, on a real source, with the correct answer attached.
"""

from __future__ import annotations

from typing import Any


def _entity_key(value: str, entity_type: str) -> str:
    return f"{(entity_type or '').strip().lower()}|{(value or '').strip().lower()}"


def diff_entity_gate(
    payload: dict[str, Any],
    submitted_reviews: list[dict],
    submitted_added: list[dict],
) -> dict[str, Any]:
    """Compare Gate 0 recommendations against the analyst's submission.

    Scored over the RECOMMENDATIONS, not over every entity. The reviewer only
    speaks where it wants a change (or wants to endorse), while the UI submits
    a decision for every entity — scoring over all of them would drown the
    signal in agreement the reviewer never claimed credit for.

    Two item kinds:
      - `entity`   : the reviewer named an entity_id and an action. Agreed
                     when the analyst's action for that entity matches.
      - `addition` : the reviewer proposed an entity the extractor missed.
                     Agreed when the analyst actually added it.
    """
    by_id = {
        r.get("entity_id"): r
        for r in submitted_reviews
        if isinstance(r, dict) and r.get("entity_id")
    }
    added_keys = {
        _entity_key(a.get("value", ""), a.get("entity_type", ""))
        for a in submitted_added
        if isinstance(a, dict)
    }

    items: list[dict[str, Any]] = []

    for rec in payload.get("entities") or []:
        if not isinstance(rec, dict):
            continue
        eid = rec.get("entity_id")
        recommended = rec.get("action")
        actual_review = by_id.get(eid)
        # An entity the analyst didn't mention is left as extracted, which is
        # the same outcome as "approve" — silence is assent at this gate.
        actual = (actual_review or {}).get("action", "approve")
        items.append({
            "kind": "entity",
            "ref": eid,
            "recommended": recommended,
            "confidence": rec.get("confidence"),
            "quote_unsupported": bool(rec.get("quote_unsupported")),
            "rationale": rec.get("rationale") or "",
            "actual": actual,
            "agreed": recommended == actual,
        })

    for rec in payload.get("added_entities") or []:
        if not isinstance(rec, dict):
            continue
        key = _entity_key(rec.get("value", ""), rec.get("entity_type", ""))
        accepted = key in added_keys
        items.append({
            "kind": "addition",
            "ref": rec.get("value"),
            "recommended": "add",
            "confidence": rec.get("confidence"),
            "quote_unsupported": bool(rec.get("quote_unsupported")),
            "rationale": rec.get("rationale") or "",
            "actual": "add" if accepted else "not_added",
            "agreed": accepted,
        })

    agreed = sum(1 for i in items if i["agreed"])
    return {
        "gate_key": "entities",
        "submitted": {
            "reviews": submitted_reviews,
            "added_entities": submitted_added,
        },
        "agreement": {
            "total": len(items),
            "agreed": agreed,
            "overridden": len(items) - agreed,
            # Rate over recommendations only; None rather than 1.0 when the
            # reviewer recommended nothing, because "agreed with all zero of
            # its recommendations" is not evidence of anything.
            "rate": round(agreed / len(items), 3) if items else None,
            "items": items,
        },
    }


def diff_procedure_gate(
    payload: dict[str, Any],
    submitted_reviews: list[dict],
    submitted_promotions: list[dict],
) -> dict[str, Any]:
    """Compare Gate 1 recommendations against the analyst's submission.

    Scored over the recommendations, same as Gate 0 — the UI submits a
    decision for every draft, and counting those as agreement would drown the
    signal in verdicts the reviewer never claimed.

    Three item kinds:
      - `draft`     : agreed when the analyst's action matches.
      - `technique` : a recommended removal, agreed when the technique is
                      actually gone from the draft the analyst submitted.
      - `promotion` : agreed when the analyst promoted that pick.

    Technique removals are scored separately from the draft verdict on
    purpose. "Approve this draft but drop T1105" is two claims, and an
    analyst who keeps the draft while rejecting the removal has agreed with
    one and overridden the other. Collapsing them would hide the disagreement
    that matters most — the reviewer's technique judgement is the thing Gate 1
    exists to test.
    """
    by_draft = {
        r.get("draft_id"): r
        for r in submitted_reviews
        if isinstance(r, dict) and r.get("draft_id")
    }
    promoted = {
        (p.get("chunk_id"), p.get("technique_id"))
        for p in submitted_promotions
        if isinstance(p, dict)
    }

    items: list[dict[str, Any]] = []

    for rec in payload.get("drafts") or []:
        if not isinstance(rec, dict):
            continue
        did = rec.get("draft_id")
        actual_review = by_draft.get(did) or {}
        # A draft the analyst didn't mention ships as drafted, which is the
        # same outcome as approve.
        actual = actual_review.get("action", "approve")
        recommended = rec.get("action")
        items.append({
            "kind": "draft",
            "ref": did,
            "recommended": recommended,
            "confidence": rec.get("confidence"),
            "quote_unsupported": bool(rec.get("quote_unsupported")),
            "rationale": rec.get("rationale") or "",
            "actual": actual,
            "agreed": recommended == actual,
        })

        # Did the recommended technique removals actually happen?
        wanted = rec.get("remove_technique_ids") or []
        if not wanted:
            continue
        edits = actual_review.get("analyst_edits") or {}
        edited = edits.get("techniques")
        if edited is None:
            # The analyst didn't touch the technique list, so nothing was
            # removed — regardless of what they did with the draft itself.
            surviving = None
        else:
            surviving = {
                t.get("technique_id") for t in edited if isinstance(t, dict)
            }
        for tid in wanted:
            gone = surviving is not None and tid not in surviving
            items.append({
                "kind": "technique",
                "ref": f"{did}:{tid}",
                "recommended": "remove",
                "confidence": rec.get("confidence"),
                "quote_unsupported": bool(rec.get("quote_unsupported")),
                "rationale": rec.get("rationale") or "",
                "actual": "removed" if gone else "kept",
                "agreed": gone,
            })

    for rec in payload.get("promotions") or []:
        if not isinstance(rec, dict):
            continue
        key = (rec.get("chunk_id"), rec.get("technique_id"))
        taken = key in promoted
        items.append({
            "kind": "promotion",
            "ref": f"{key[0]}:{key[1]}",
            "recommended": "promote",
            "confidence": rec.get("confidence"),
            "quote_unsupported": bool(rec.get("quote_unsupported")),
            "rationale": rec.get("rationale") or "",
            "actual": "promoted" if taken else "left_in_review",
            "agreed": taken,
        })

    agreed = sum(1 for i in items if i["agreed"])
    return {
        "gate_key": "procedures",
        "submitted": {
            "reviews": submitted_reviews,
            "promotions": submitted_promotions,
        },
        "agreement": {
            "total": len(items),
            "agreed": agreed,
            "overridden": len(items) - agreed,
            "rate": round(agreed / len(items), 3) if items else None,
            "items": items,
        },
    }


def diff_bundle_gate(
    payload: dict[str, Any],
    submitted_reviews: list[dict],
    _unused: list[dict] | None = None,
) -> dict[str, Any]:
    """Compare Gate 2 recommendations against the analyst's submission.

    Third positional arg exists only so every differ shares one signature and
    `record_gate_outcome` can dispatch without special-casing; Gate 2 has
    no second channel the way Gate 0 has additions and Gate 1 promotions.

    Scored over the recommendations, same as the other gates. Note the caveat
    on `rel_id`: it is positional (`relp_N`) and renumbers whenever `normalize`
    re-runs, so a stored ref is only meaningful against the pass that produced
    it. That is fine for agreement — both sides are scored at submit time,
    against the same pass — but do not read a stored `relp_3` as durable.
    """
    by_id = {
        r.get("rel_id"): r
        for r in submitted_reviews
        if isinstance(r, dict) and r.get("rel_id")
    }

    items: list[dict[str, Any]] = []
    for rec in payload.get("relationships") or []:
        if not isinstance(rec, dict):
            continue
        rid = rec.get("rel_id")
        recommended = rec.get("action")
        # A relationship the analyst didn't mention ships as derived, which is
        # the same outcome as approve.
        actual = (by_id.get(rid) or {}).get("action", "approve")
        items.append({
            "kind": "relationship",
            "ref": rid,
            "recommended": recommended,
            "confidence": rec.get("confidence"),
            "quote_unsupported": bool(rec.get("quote_unsupported")),
            "rationale": rec.get("rationale") or "",
            "actual": actual,
            "agreed": recommended == actual,
        })

    agreed = sum(1 for i in items if i["agreed"])
    return {
        "gate_key": "bundle",
        "submitted": {"reviews": submitted_reviews},
        "agreement": {
            "total": len(items),
            "agreed": agreed,
            "overridden": len(items) - agreed,
            "rate": round(agreed / len(items), 3) if items else None,
            "items": items,
        },
    }


def _edge_key(frm: str, to: str) -> str:
    return f"{(frm or '').strip()}->{(to or '').strip()}"


def diff_chunk_gate(
    payload: dict[str, Any],
    submitted_decisions: list[dict],
    submitted_extras: dict[str, Any] | list[dict] | None = None,
) -> dict[str, Any]:
    """Compare chunk-gate recommendations against the analyst's submission.

    `submitted_extras` is a DICT here, not a list. This gate submits four
    channels — decisions, added_chunks, edges, and a wholesale reject — and
    the other three do not fit the single extra slot the entity and procedure
    gates use. Keeping the differ signature uniform matters more than keeping
    the argument type uniform: `record_gate_outcome` dispatches on
    gate_key alone and must not learn each gate's shape.

    Four item kinds:
      - `chunk`   : agreed when the analyst's action for that chunk matches.
      - `addition`: agreed when a chunk with that text was actually added.
      - `edge`    : agreed when the same mutation appears in the submission.
      - `rerun`   : agreed when the analyst also rejected. Scored on the
                    ACTION, not the reason — an analyst who re-chunks for a
                    different stated reason has still taken the advice, and
                    the reason is a hint to the chunker rather than a claim
                    the reviewer made about the source.

    A reject makes the other three moot: the gate ignores decisions, adds and
    edges entirely when `reject` is set. So when the analyst rejects, the
    remaining recommendations are recorded as `moot` rather than overridden —
    calling them disagreement would slander a reviewer whose advice was never
    reached, and calling them agreement would invent consent.
    """
    extras = submitted_extras if isinstance(submitted_extras, dict) else {}
    submitted_added = extras.get("added_chunks") or []
    submitted_edges = extras.get("edges") or []
    submitted_reject = extras.get("reject")
    rejected = bool(submitted_reject)

    by_id = {
        d.get("chunk_id"): d
        for d in submitted_decisions
        if isinstance(d, dict) and d.get("chunk_id")
    }
    added_texts = {
        (a.get("text") or "").strip().lower()
        for a in submitted_added
        if isinstance(a, dict)
    }
    submitted_edge_keys = {
        (e.get("action"), _edge_key(e.get("from") or e.get("from_"), e.get("to")))
        for e in submitted_edges
        if isinstance(e, dict)
    }

    items: list[dict[str, Any]] = []

    def _item(kind: str, ref: str, rec: dict, recommended: str, actual: str) -> dict:
        return {
            "kind": kind,
            "ref": ref,
            "recommended": recommended,
            "confidence": rec.get("confidence"),
            "quote_unsupported": bool(rec.get("quote_unsupported")),
            "rationale": rec.get("rationale") or "",
            "actual": "moot_rerun" if rejected else actual,
            "agreed": (not rejected) and recommended == actual,
            **({"moot": True} if rejected else {}),
        }

    for rec in payload.get("chunks") or []:
        if not isinstance(rec, dict):
            continue
        cid = rec.get("chunk_id")
        # A chunk the analyst didn't mention is kept as chunked, which is the
        # same outcome as approve — silence is assent, as at every gate.
        actual = (by_id.get(cid) or {}).get("action", "approve")
        items.append(_item("chunk", cid, rec, rec.get("action"), actual))

    for rec in payload.get("added_chunks") or []:
        if not isinstance(rec, dict):
            continue
        text = (rec.get("text") or "").strip()
        taken = text.lower() in added_texts
        items.append(_item(
            "addition", text[:80], rec, "add", "add" if taken else "not_added",
        ))

    for rec in payload.get("edges") or []:
        if not isinstance(rec, dict):
            continue
        action = rec.get("action")
        key = _edge_key(rec.get("from_chunk_id"), rec.get("to_chunk_id"))
        taken = (action, key) in submitted_edge_keys
        items.append(_item(
            "edge", f"{action} {key}", rec, action,
            action if taken else "not_applied",
        ))

    reject_rec = payload.get("reject")
    if isinstance(reject_rec, dict) and reject_rec.get("reason"):
        # Scored outside _item: a rerun recommendation is the one thing a
        # rerun does NOT make moot.
        items.append({
            "kind": "rerun",
            "ref": reject_rec.get("reason"),
            "recommended": "rerun",
            "confidence": reject_rec.get("confidence"),
            "quote_unsupported": bool(reject_rec.get("quote_unsupported")),
            "rationale": reject_rec.get("rationale") or "",
            "actual": "rerun" if rejected else "kept_this_pass",
            "agreed": rejected,
        })
    elif rejected:
        # The analyst re-chunked without being asked to. Not a scored
        # recommendation — the reviewer made none — but the single most
        # informative thing that can happen at this gate, and it would be
        # invisible if only recommendations were recorded.
        items.append({
            "kind": "rerun",
            "ref": (submitted_reject or {}).get("reason", "?"),
            "recommended": "none",
            "confidence": None,
            "quote_unsupported": False,
            "rationale": "",
            "actual": "rerun",
            "agreed": False,
        })

    scored = [i for i in items if not i.get("moot")]
    agreed = sum(1 for i in scored if i["agreed"])
    return {
        "gate_key": "chunks",
        "submitted": {
            "decisions": submitted_decisions,
            "added_chunks": submitted_added,
            "edges": submitted_edges,
            "reject": submitted_reject,
        },
        "agreement": {
            # Rate over what was actually reachable. A rerun moots most of a
            # review, and dividing by recommendations the analyst never got to
            # act on would report a collapse in agreement that did not happen.
            "total": len(scored),
            "agreed": agreed,
            "overridden": len(scored) - agreed,
            "moot": len(items) - len(scored),
            "rate": round(agreed / len(scored), 3) if scored else None,
            "items": items,
        },
    }


# gates_enabled key -> the differ that scores that gate's recommendations
# against what was actually submitted. Lives here rather than in the HTTP
# route because it is reviewer-domain, and because both submission paths need
# it: the analyst's POST and the autonomous runner. The route module cannot be
# the shared home — `gates.py` imports from `pipeline.py`, so `pipeline.py`
# importing back would be circular.
#
# A gate absent here records no outcome, which is correct for one with no
# reviewer, and wrong for one that has one — pinned by
# tests/test_contracts.py::TestReviewerRegistriesAgree.
OUTCOME_DIFFERS = {
    "entities": diff_entity_gate,
    "chunks": diff_chunk_gate,
    "procedures": diff_procedure_gate,
    "bundle": diff_bundle_gate,
}
