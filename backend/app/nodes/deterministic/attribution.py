"""Per-actor attribution rules, shared by the bundle-gate preview and the serializer.

`normalize` shows the analyst a relationship preview; `serialize_stix` builds
the real SROs. When the two derived actor edges by different rules the preview
fanned every intrusion set out to every procedure (128 rows on an eight-actor
report) while the serializer shipped the 14 the drafts actually named — so the
analyst reviewed 114 edges that could never exist. One rule, one module, both
callers.

Everything here works on the entity dicts and draft dicts as they sit in
pipeline state; the serializer maps the result to STIX ids afterwards.
"""

from __future__ import annotations


def entity_name(entity: dict) -> str:
    """Display name of an entity: the analyst's edit first, else the value."""
    return (entity.get("edited_value") or entity.get("value") or "").strip()


def actors_for_procedure(draft: dict, intrusion_sets: list[dict]) -> list[dict]:
    """The intrusion sets THIS procedure is attributed to, as entity dicts.

    Attribution used to be a cross product — every intrusion set linked to
    every procedure — and on a report whose entire point was that two actors
    are UNRELATED it asserted the opposite. Resolution order:

      1. the drafting LLM's per-procedure `attributed_actors`, matched
         case-insensitively against the approved intrusion sets' names — an
         actor the extractor never produced cannot be invented here;
      2. failing that, if the source has EXACTLY ONE intrusion set, that one —
         single-actor reports refer to "the group" far more often than by name;
      3. otherwise none. With several actors and no per-procedure signal,
         guessing is what created the bug; contrast actors stay in the bundle
         as context with no fabricated edges.
    """
    wanted = {
        n.strip().lower()
        for n in (draft.get("attributed_actors") or [])
        if isinstance(n, str) and n.strip()
    }
    if wanted:
        named: list[dict] = []
        seen: set[str] = set()
        for iset in intrusion_sets:
            name = entity_name(iset).lower()
            key = iset.get("entity_id") or name
            if name and name in wanted and key not in seen:
                named.append(iset)
                seen.add(key)
        if named:
            return named
    if len(intrusion_sets) == 1:
        return [intrusion_sets[0]]
    return []


def threat_actors_for_intrusion_set(
    iset: dict,
    intrusion_sets: list[dict],
    threat_actors: list[dict],
) -> list[dict]:
    """The threat actors THIS intrusion set is attributed to, as entity dicts.

    `intrusion-set --attributed-to--> threat-actor` used to be a cross
    product: every cluster to every sponsor in the source. On a report that
    tied one of four clusters to the MSS by indictment, the other three were
    attributed too — one of them explicitly unattributed by the report.

      1. the extractor's per-cluster `attributed_to` names, matched
         case-insensitively against the approved threat actors;
      2. failing that, when the source has exactly ONE intrusion set and
         exactly ONE threat actor, that pair — the single-actor report, where
         "the group" and its sponsor are the whole story and the model may
         well have returned an empty list;
      3. otherwise none.
    """
    wanted = {
        n.strip().lower()
        for n in (iset.get("attributed_to") or [])
        if isinstance(n, str) and n.strip()
    }
    if wanted:
        out: list[dict] = []
        seen: set[str] = set()
        for ta in threat_actors:
            name = entity_name(ta).lower()
            key = ta.get("entity_id") or name
            if name and name in wanted and key not in seen:
                out.append(ta)
                seen.add(key)
        if out:
            return out
    if len(intrusion_sets) == 1 and len(threat_actors) == 1:
        return [threat_actors[0]]
    return []


def has_maas_only(malware_entities: list[dict]) -> bool:
    """True when every malware entity is malware-as-a-service.

    Guards campaign -> intrusion-set attribution: MaaS malware is operated by
    many unrelated actors, so a low-confidence intrusion set gets no
    attribution edge. Note: nothing in extraction sets `is_maas` today — only
    tests do — so this currently always returns False on real runs. Kept as
    the one implementation both callers share; whether to wire or delete the
    flag is a separate decision.
    """
    return len(malware_entities) > 0 and all(
        bool(mw.get("is_maas", False)) for mw in malware_entities
    )
