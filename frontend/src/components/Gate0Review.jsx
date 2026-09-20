/**
 * Gate0Review — Entity review for Gate 0.
 *
 * Card-based layout (responsive within the slide-over panel).
 * Each entity is a card with type tag, value, action dropdown,
 * and expandable edit fields when action = "edit".
 *
 * Features:
 * - Bulk "Approve All" / "Remove All" actions
 * - Per-entity approve / edit / remove
 * - Inline edit for type + value when action = "edit"
 * - Rationale field for edit/remove
 * - "+ Add Entity" form for manually adding missed entities
 * - Decision tally + Submit button in footer
 * - Sort by type or value
 */
import { useState, useMemo, useCallback } from "react";
import ReviewerBrief from "./ReviewerBrief";
import SuggestionChip from "./SuggestionChip";
import InfoDot from "./InfoDot";
import { hint } from "../lib/glossary";
import {
  applyRecommendation,
  indexByEntity,
  isBulkAcceptable,
  pickRestoredDecision,
} from "../lib/reviewerSuggestions";

/** The pipeline's EntityType values, verbatim.
 *
 * Pinned to the backend enum by a contract test — this list used to carry 18
 * STIX SCO type names ("ipv4-addr", "file", "domain-name") that are NOT
 * EntityType values, and "ipv4-addr" was the DEFAULT. An entity added with
 * one of those reaches validated_entities, then serialization finds no STIX
 * mapping, logs a warning and drops it: the analyst's addition never reaches
 * the bundle and nothing tells them.
 *
 * That was inert while added entities were being discarded before they got
 * this far. It stopped being inert when that channel was fixed.
 */
const ENTITY_TYPES = [
  "intrusion_set", "threat_actor", "malware", "tool",
  "campaign", "vulnerability", "organization", "location",
  "victim_sector", "infrastructure", "ioc_hash", "ioc_ip",
  "ioc_domain", "ioc_url", "ioc_email", "ioc_file_path",
  "ioc_registry_key", "ioc_mutex", "ioc_command_line", "ioc_process_name",
  "software", "user_account",
];

/** Initial decision state for an entity. */
function initDecision(entity) {
  // Seed edited_role from whichever role field applies to the entity's
  // type. The edit panel uses this to render a dropdown that defaults
  // to the LLM's chosen role; the analyst can override before submit.
  const etype = entity.type ?? entity.entity_type ?? "";
  let initRole = "";
  if (etype === "organization") initRole = entity.organization_role ?? "";
  else if (etype === "location") initRole = entity.location_role ?? "";
  const denylisted = Boolean(entity.denylisted);
  return {
    entity_id: entity.id ?? entity.entity_id,
    // Denylisted entities default to REMOVE — the deterministic guardrail.
    // The submit sends a decision for every entity, so without this default
    // a denylisted entity would carry an explicit "approve" and silently
    // defeat the guardrail. The analyst overrides by flipping to approve.
    action: denylisted ? "remove" : "approve",
    denylisted,
    edited_value: entity.value ?? "",
    edited_type: etype,
    edited_role: initRole,
    rationale: denylisted ? (entity.denylist_reason ?? "Matches analyst denylist") : "",
  };
}

const ORG_ROLE_OPTIONS = ["victim", "sponsor", "publisher", "author", "other"];
const LOCATION_ROLE_OPTIONS = ["victim", "origin", "context"];

/** Tailwind classes for the role chip per role value. Each role gets its
 * own color so the analyst can scan a list of locations or organizations
 * and immediately spot anomalies. Designed against gruvbox dark.
 */
const ROLE_CHIP_STYLES = {
  // Locations
  victim: "bg-gb-red/15 text-gb-red",
  origin: "bg-gb-orange/15 text-gb-orange",
  context: "bg-gb-bg1 text-gb-fg4",
  // Organizations
  sponsor: "bg-gb-purple/15 text-gb-bright-purple",
  publisher: "bg-gb-aqua/15 text-gb-bright-aqua",
  author: "bg-gb-blue/15 text-gb-bright-blue",
  other: "bg-gb-bg1 text-gb-fg4",
};

/** Action badge colors. */
const ACTION_STYLES = {
  approve: "bg-gb-tag-reliability-bg text-gb-bright-green border-gb-green",
  edit: "bg-gb-tag-gate-bg text-gb-bright-orange border-gb-orange",
  remove: "bg-gb-bg1 text-gb-bright-red border-gb-red",
};

export default function Gate0Review({
  payload,
  onSubmit,
  submitting,
  // AI reviewer additions. All optional — absent means this gate ran in
  // plain review mode and the component behaves exactly as it always has.
  recommendations = null,
  brief = null,
  onSaveBrief = null,
  savingBrief = false,
}) {
  const entities = payload?.items ?? [];
  const context = payload?.context ?? {};

  // Per-entity decisions keyed by entity_id
  const [decisions, setDecisions] = useState(() => {
    const map = {};
    entities.forEach((e) => {
      const d = initDecision(e);
      map[d.entity_id] = d;
    });
    return map;
  });

  // Manually added entities
  const [addedEntities, setAddedEntities] = useState([]);

  // Which AI recommendations the analyst has applied. Tracked separately from
  // `decisions` so "applied" survives a later manual edit of the same entity
  // and so the chip can offer an undo.
  const [appliedRecs, setAppliedRecs] = useState(() => new Set());

  // Decision as it stood just BEFORE a recommendation was applied, so undo
  // restores what the analyst had rather than the extractor's original.
  // Rebuilding from the entity would discard any value, type, role or
  // rationale they set by hand — an undo that destroys unrelated work is
  // worse than no undo. Same rule as Gate 1's preApplyTechniques.
  const [preApplyDecisions, setPreApplyDecisions] = useState({});

  /** entity_id -> recommendation. Empty when there is no AI review. */
  const recsByEntity = useMemo(
    () => indexByEntity(recommendations),
    [recommendations],
  );

  /** Additions the reviewer proposed that the analyst has not added yet. */
  const suggestedAdditions = useMemo(() => {
    const already = new Set(
      addedEntities.map((e) => `${e.type}|${(e.value ?? "").toLowerCase()}`),
    );
    return (recommendations?.added_entities ?? []).filter(
      (r) => !already.has(`${r.entity_type}|${(r.value ?? "").toLowerCase()}`),
    );
  }, [recommendations, addedEntities]);

  /** How many recommendations a bulk accept would apply. */
  const bulkAcceptableCount = useMemo(
    () =>
      (recommendations?.entities ?? []).filter(isBulkAcceptable).length +
      (recommendations?.added_entities ?? []).filter(isBulkAcceptable).length,
    [recommendations],
  );

  // Sort state
  const [sort, setSort] = useState(null);

  // Add entity form
  const [showAddForm, setShowAddForm] = useState(false);
  const [newEntity, setNewEntity] = useState({ type: "ioc_ip", value: "", rationale: "" });

  /** Toggle sort. */
  const handleSort = useCallback((column) => {
    setSort((prev) => {
      if (prev?.column !== column) return { column, direction: "asc" };
      if (prev.direction === "asc") return { column, direction: "desc" };
      return null;
    });
  }, []);

  /** Sorted entity list. */
  const sortedEntities = useMemo(() => {
    const combined = [
      ...entities.map((e) => ({
        id: e.id ?? e.entity_id,
        type: e.type ?? e.entity_type ?? "",
        value: e.value ?? "",
        confidence: e.confidence ?? 0,
        source: e.source ?? e.extraction_method ?? "",
        denylisted: Boolean(e.denylisted),
        denylistReason: e.denylist_reason ?? "",
        // The role chips read these off the row. Left out of this mapping,
        // the chips could never render and the role-edit feature was dead.
        location_role: e.location_role ?? null,
        organization_role: e.organization_role ?? null,
        // Per-cluster sponsorship from the extractor; read-only here. The
        // serializer attributes an intrusion set only to these.
        attributed_to: Array.isArray(e.attributed_to) ? e.attributed_to : [],
        isAdded: false,
      })),
      ...addedEntities,
    ];

    if (!sort) return combined;

    const { column, direction } = sort;
    const mult = direction === "asc" ? 1 : -1;

    return [...combined].sort((a, b) => {
      let aVal = a[column] ?? "";
      let bVal = b[column] ?? "";
      if (column === "confidence") return (aVal - bVal) * mult;
      return String(aVal).localeCompare(String(bVal)) * mult;
    });
  }, [entities, addedEntities, sort]);

  /** Update a decision field. */
  const updateDecision = useCallback((entityId, field, value) => {
    setDecisions((prev) => ({
      ...prev,
      [entityId]: { ...prev[entityId], [field]: value },
    }));
  }, []);

  /** Bulk approve all. Denylisted entities are skipped so a one-click
   * "Approve All" can't silently un-block the deterministic guardrail —
   * the analyst must flip a denylisted entity to approve individually. */
  const handleApproveAll = useCallback(() => {
    setDecisions((prev) => {
      const next = { ...prev };
      Object.keys(next).forEach((id) => {
        if (next[id].denylisted) return;
        next[id] = { ...next[id], action: "approve" };
      });
      return next;
    });
  }, []);

  /** Apply one AI recommendation onto its entity's decision. */
  const handleApplyRec = useCallback((entityId) => {
    const rec = recsByEntity[entityId];
    if (!rec) return;
    setDecisions((prev) => {
      setPreApplyDecisions((snap) =>
        entityId in snap ? snap : { ...snap, [entityId]: prev[entityId] },
      );
      return { ...prev, [entityId]: applyRecommendation(prev[entityId], rec) };
    });
    setAppliedRecs((prev) => new Set(prev).add(entityId));
  }, [recsByEntity]);

  /** Undo an applied recommendation, restoring the analyst's own decision. */
  const handleDismissRec = useCallback((entityId) => {
    const entity = entities.find((e) => (e.id ?? e.entity_id) === entityId);
    setDecisions((prev) => ({
      ...prev,
      // No snapshot means nothing was ever applied to this entity; fall
      // back to a clean rebuild rather than leaving it as-is.
      [entityId]: pickRestoredDecision(
        preApplyDecisions[entityId],
        entity ? initDecision(entity) : prev[entityId],
      ),
    }));
    setPreApplyDecisions((snap) => {
      const next = { ...snap };
      delete next[entityId];
      return next;
    });
    setAppliedRecs((prev) => {
      const next = new Set(prev);
      next.delete(entityId);
      return next;
    });
  }, [entities, preApplyDecisions]);

  /** Add an entity the reviewer says the extractor missed. */
  const handleAddSuggested = useCallback((rec) => {
    const id = `added_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const entity = {
      id,
      type: rec.entity_type,
      value: rec.value,
      confidence: 100,
      // Provenance stays honest: this came from the AI reviewer with an
      // analyst's assent, which is not the same as an analyst spotting it.
      source: "ai_reviewer",
      isAdded: true,
    };
    setAddedEntities((prev) => [...prev, entity]);
    setDecisions((prev) => ({
      ...prev,
      [id]: {
        entity_id: id,
        action: "approve",
        edited_value: rec.value,
        edited_type: rec.entity_type,
        edited_role: rec.organization_role ?? rec.location_role ?? "",
        rationale: rec.rationale || "Added on AI reviewer recommendation",
      },
    }));
  }, []);

  /** Apply every HIGH-confidence recommendation in one click.
   *
   * Deliberately does NOT touch medium or low. If one click could accept
   * everything, assist mode would become autopilot wearing a human's badge —
   * and the override signal that decides whether autopilot is ever safe would
   * never be generated. The friction is the point. Same reasoning as
   * "Approve All" skipping denylisted entities. */
  const handleAcceptAI = useCallback(() => {
    const eligible = (recommendations?.entities ?? []).filter(isBulkAcceptable);
    setDecisions((prev) => {
      const next = { ...prev };
      eligible.forEach((rec) => {
        if (next[rec.entity_id]) {
          next[rec.entity_id] = applyRecommendation(next[rec.entity_id], rec);
        }
      });
      return next;
    });
    setAppliedRecs((prev) => {
      const next = new Set(prev);
      eligible.forEach((rec) => next.add(rec.entity_id));
      return next;
    });
    (recommendations?.added_entities ?? [])
      .filter(isBulkAcceptable)
      .forEach(handleAddSuggested);
  }, [recommendations, handleAddSuggested]);

  /** Add a new entity manually. */
  const handleAddEntity = useCallback(() => {
    if (!newEntity.value.trim()) return;
    const id = `added_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const entity = {
      id,
      type: newEntity.type,
      value: newEntity.value.trim(),
      confidence: 100,
      source: "analyst",
      isAdded: true,
    };
    setAddedEntities((prev) => [...prev, entity]);
    setDecisions((prev) => ({
      ...prev,
      [id]: {
        entity_id: id,
        action: "approve",
        edited_value: entity.value,
        edited_type: entity.type,
        rationale: newEntity.rationale.trim() || "Manually added by analyst",
      },
    }));
    setNewEntity({ type: "ioc_ip", value: "", rationale: "" });
    setShowAddForm(false);
  }, [newEntity]);

  /** Decision tally. */
  const tally = useMemo(() => {
    const counts = { approve: 0, edit: 0, remove: 0 };
    Object.values(decisions).forEach((d) => {
      counts[d.action] = (counts[d.action] || 0) + 1;
    });
    return counts;
  }, [decisions]);

  /** Submit all decisions.
   *
   * Newly added entities go in `added_entities`, NOT in `reviews`. The gate
   * processor walks the extractor's entity list and matches reviews by
   * entity_id, so a review carrying a synthetic `added_*` id matches nothing
   * and is silently discarded — which is exactly what used to happen to every
   * entity an analyst added here. `gate0_added_entities` is the channel the
   * backend actually reads for them.
   */
  const handleSubmit = useCallback(() => {
    const addedIds = new Set(addedEntities.map((e) => e.id));

    const reviews = Object.values(decisions)
      .filter((d) => !addedIds.has(d.entity_id))
      .map((d) => {
        const review = { entity_id: d.entity_id, action: d.action };
        if (d.action === "edit") {
          review.edited_value = d.edited_value;
          review.edited_type = d.edited_type;
          // Only send edited_role when the analyst picked a value AND the
          // type is one that has roles. Empty string means "leave the
          // role as the LLM emitted it" — backend treats null/missing
          // as no-change.
          if (d.edited_role && (d.edited_type === "organization" || d.edited_type === "location")) {
            review.edited_role = d.edited_role;
          }
        }
        if (d.rationale) review.rationale = d.rationale;
        return review;
      });

    // An entity added and then flipped to remove is simply not sent — there
    // is nothing to remove, it never reached the pipeline.
    const added_entities = addedEntities
      .filter((e) => decisions[e.id]?.action !== "remove")
      .map((e) => {
        const d = decisions[e.id] ?? {};
        const entity_type = d.edited_type || e.type;
        const item = { value: d.edited_value || e.value, entity_type };
        if (d.edited_role) {
          if (entity_type === "organization") item.organization_role = d.edited_role;
          else if (entity_type === "location") item.location_role = d.edited_role;
        }
        if (d.rationale) item.rationale = d.rationale;
        return item;
      });

    onSubmit({ reviews, added_entities });
  }, [decisions, addedEntities, onSubmit]);

  const sortArrow = (col) => {
    if (sort?.column !== col) return "";
    return sort.direction === "asc" ? " ↑" : " ↓";
  };

  return (
    <div className="flex flex-col gap-4">
      {/* The reviewer's read of the report, if this gate ran in assist mode.
          Renders nothing when there is no brief. */}
      {brief && onSaveBrief && (
        <ReviewerBrief brief={brief} onSave={onSaveBrief} saving={savingBrief} />
      )}

      {/* Context bar */}
      <div className="flex gap-4 text-[12px] text-gb-fg4 font-data items-center">
        <span>Entities: <strong className="text-gb-fg1">{sortedEntities.length}</strong></span>
        {context.parse_warnings?.length > 0 && (
          <span className="text-gb-bright-orange">
            {context.parse_warnings.length} parse warning{context.parse_warnings.length !== 1 ? "s" : ""}
          </span>
        )}
        {/* Sequentiality classification chip. Surfaces the auto-detect
            decision (or analyst override) so the analyst can sanity-check
            before approving entities. The rationale tooltip cites the
            source phrasing the LLM used. */}
        {typeof context.is_sequential === "boolean" && (
          <span
            className={`px-1.5 py-0.5 rounded font-medium ${
              context.is_sequential
                ? "bg-gb-bright-blue/15 text-gb-bright-blue border border-gb-bright-blue/40"
                : "bg-gb-bright-orange/15 text-gb-bright-orange border border-gb-bright-orange/40"
            }`}
            title={context.sequentiality_rationale || "No rationale available"}
          >
            Sequential: {context.is_sequential ? "yes" : "no"}
          </span>
        )}
      </div>

      {/* Bulk actions + sort */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-[11px] text-gb-fg4 font-data">Bulk:</span>
          <button
            onClick={handleApproveAll}
            className="px-2.5 py-1 rounded text-[11px] font-semibold font-data bg-gb-tag-reliability-bg text-gb-bright-green border border-gb-green hover:bg-gb-bg2 transition-colors"
          >
            Approve All
          </button>
          {/* Only HIGH-confidence recommendations. Medium and low must be
              opened individually — see handleAcceptAI. */}
          {bulkAcceptableCount > 0 && (
            <button
              onClick={handleAcceptAI}
              className="px-2.5 py-1 rounded text-[11px] font-semibold font-data bg-gb-purple/15 text-gb-bright-purple border border-gb-purple hover:bg-gb-bg2 transition-colors"
              title="Applies only the AI's high-confidence recommendations. Medium and low stay for you to judge."
            >
              🤖 Accept {bulkAcceptableCount} high-confidence
            </button>
          )}
        </div>
        <div className="flex items-center gap-1">
          <span className="text-[10px] text-gb-fg4 font-data">Sort:</span>
          <button
            onClick={() => handleSort("type")}
            className={`px-2 py-0.5 rounded text-[10px] font-data transition-colors ${
              sort?.column === "type"
                ? "text-gb-bright-yellow bg-gb-bg1"
                : "text-gb-fg4 hover:text-gb-fg1"
            }`}
          >
            Type{sortArrow("type")}
          </button>
          <button
            onClick={() => handleSort("value")}
            className={`px-2 py-0.5 rounded text-[10px] font-data transition-colors ${
              sort?.column === "value"
                ? "text-gb-bright-yellow bg-gb-bg1"
                : "text-gb-fg4 hover:text-gb-fg1"
            }`}
          >
            Value{sortArrow("value")}
          </button>
        </div>
      </div>

      {/* Entities the reviewer says the extractor missed. Separate from the
          card list because they do not exist yet — adding one is a different
          decision from judging one that is already there. */}
      {suggestedAdditions.length > 0 && (
        <div className="border border-gb-purple/40 bg-gb-purple/5 rounded-md px-3 py-2">
          <p className="text-[12px] font-semibold text-gb-bright-purple mb-1.5">
            🤖 AI says these were missed ({suggestedAdditions.length})
          </p>
          <div className="flex flex-col gap-2">
            {suggestedAdditions.map((rec, i) => (
              <div key={`${rec.entity_type}|${rec.value}|${i}`} className="flex flex-col">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="font-data text-[10px] px-1.5 py-0.5 rounded bg-gb-tag-type-bg text-gb-bright-blue border border-gb-blue">
                    {rec.entity_type}
                  </span>
                  <span className="text-[12px] text-gb-fg1 font-data break-all">{rec.value}</span>
                  <div className="flex-1" />
                  <button
                    onClick={() => handleAddSuggested(rec)}
                    className="text-[10px] font-data text-gb-fg4 hover:text-gb-bright-green transition-colors"
                  >
                    + add
                  </button>
                </div>
                {/* Evidence only — the "+ add" button above is the action. */}
                <SuggestionChip rec={rec} />
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Entity cards */}
      <div className="flex flex-col gap-2">
        {sortedEntities.map((entity) => {
          const d = decisions[entity.id];
          if (!d) return null;
          const isEdit = d.action === "edit";
          const isRemove = d.action === "remove";

          return (
            <div
              key={entity.id}
              className={`rounded-lg border p-3 transition-all ${
                isRemove
                  ? "border-gb-red/40 bg-gb-bg0-s opacity-60"
                  : isEdit
                    ? "border-gb-orange bg-gb-bg0-s"
                    : "border-gb-bg2 bg-gb-bg0-s"
              }`}
            >
              {/* Top row: type tag + value + action dropdown */}
              <div className="flex items-start gap-2">
                {/* Type tag */}
                <span className="shrink-0 font-data text-[10px] font-medium px-1.5 py-0.5 rounded bg-gb-tag-type-bg text-gb-bright-blue border border-gb-blue mt-0.5">
                  {isEdit ? d.edited_type : entity.type}
                </span>

                {/* Denylist badge — this entity matched an analyst denylist
                    pattern and defaults to remove. The tooltip carries the
                    specific reason AND what a denylist is: the reason alone
                    ("matches cert.example") does not tell a new analyst that this
                    is overridable, or that flipping it to approve affects only
                    this source. */}
                {entity.denylisted && (
                  <span
                    className="shrink-0 font-data text-[10px] font-medium px-1.5 py-0.5 rounded bg-gb-bright-red/15 text-gb-bright-red border border-gb-red mt-0.5"
                    title={
                      entity.denylistReason
                        ? `${entity.denylistReason}\n\n${hint("denylist")}`
                        : hint("denylist")
                    }
                  >
                    ⊘ denylist
                  </span>
                )}

                {/* Value */}
                <span
                  className={`flex-1 min-w-0 font-data text-[12px] break-all ${
                    isRemove
                      ? "line-through text-gb-bright-red"
                      : "text-gb-fg1"
                  }`}
                >
                  {isEdit ? d.edited_value : entity.value}
                </span>

                {/* Action dropdown */}
                <select
                  value={d.action}
                  onChange={(e) => updateDecision(entity.id, "action", e.target.value)}
                  className={`shrink-0 font-data text-[11px] rounded px-1.5 py-0.5 border cursor-pointer ${
                    ACTION_STYLES[d.action] || "bg-gb-bg1 text-gb-fg4 border-gb-bg2"
                  }`}
                >
                  <option value="approve">approve</option>
                  <option value="edit">edit</option>
                  <option value="remove">remove</option>
                </select>
              </div>

              {/* Metadata row: confidence + source + role badges */}
              {!isEdit && (
                <div className="flex gap-3 mt-1.5 ml-0.5 items-center">
                  {entity.confidence > 0 && (
                    <span className="text-[10px] font-data text-gb-fg4">
                      conf: {entity.confidence <= 1 ? Math.round(entity.confidence * 100) : entity.confidence}%
                    </span>
                  )}
                  {entity.source && (
                    <span className="text-[10px] font-data text-gb-fg4">
                      src: {entity.source}
                    </span>
                  )}
                  {(entity.type === "location" || entity.entity_type === "location") && entity.location_role && (
                    <span className={`text-[10px] font-data px-1.5 py-0.5 rounded ${ROLE_CHIP_STYLES[entity.location_role] ?? "bg-gb-bg1 text-gb-fg4"}`}>
                      role: {entity.location_role}
                    </span>
                  )}
                  {(entity.type === "organization" || entity.entity_type === "organization") && entity.organization_role && (
                    <span className={`text-[10px] font-data px-1.5 py-0.5 rounded ${ROLE_CHIP_STYLES[entity.organization_role] ?? "bg-gb-bg1 text-gb-fg4"}`}>
                      role: {entity.organization_role}
                    </span>
                  )}
                  {(entity.type === "intrusion_set" || entity.entity_type === "intrusion_set")
                    && Array.isArray(entity.attributed_to) && entity.attributed_to.length > 0 && (
                    <span
                      className="text-[10px] font-data px-1.5 py-0.5 rounded bg-gb-bg1 text-gb-fg4"
                      title="Threat actors the source explicitly attributes this cluster to. The bundle's attributed-to edges come from this list."
                    >
                      ↳ attributed to {entity.attributed_to.join(", ")}
                    </span>
                  )}
                </div>
              )}

              {/* AI reviewer's recommendation for this entity, if any. */}
              {recsByEntity[entity.id] && (
                <SuggestionChip
                  rec={recsByEntity[entity.id]}
                  applied={appliedRecs.has(entity.id)}
                  onApply={() => handleApplyRec(entity.id)}
                  onDismiss={() => handleDismissRec(entity.id)}
                />
              )}

              {/* Edit fields (expanded when action = "edit") */}
              {isEdit && (
                <div className="mt-2 pt-2 border-t border-gb-bg1 flex flex-col gap-2">
                  <div className="flex gap-2">
                    <div className="flex flex-col gap-0.5">
                      <label className="text-[9px] text-gb-fg4 font-data uppercase tracking-wider">
                        Type<InfoDot term="entity-type" />
                      </label>
                      <select
                        value={d.edited_type}
                        onChange={(e) => updateDecision(entity.id, "edited_type", e.target.value)}
                        className="bg-gb-bg1 text-gb-fg1 border border-gb-bg2 rounded px-2 py-1 text-[11px] font-data"
                      >
                        {ENTITY_TYPES.map((t) => (
                          <option key={t} value={t}>{t}</option>
                        ))}
                      </select>
                    </div>
                    <div className="flex flex-col gap-0.5 flex-1 min-w-0">
                      <label className="text-[9px] text-gb-fg4 font-data uppercase tracking-wider">Value</label>
                      <input
                        type="text"
                        value={d.edited_value}
                        onChange={(e) => updateDecision(entity.id, "edited_value", e.target.value)}
                        className="w-full bg-gb-bg1 text-gb-fg1 border border-gb-bg2 rounded px-2 py-1 text-[11px] font-data"
                      />
                    </div>
                    {/* Role dropdown — only relevant for organization or location.
                        Backend's gate_0 processor routes the value to the right
                        field based on edited_type, with enum validation. */}
                    {(d.edited_type === "organization" || d.edited_type === "location") && (
                      <div className="flex flex-col gap-0.5">
                        <label className="text-[9px] text-gb-fg4 font-data uppercase tracking-wider">
                          Role<InfoDot term="entity-role" />
                        </label>
                        <select
                          value={d.edited_role || ""}
                          onChange={(e) => updateDecision(entity.id, "edited_role", e.target.value)}
                          className="bg-gb-bg1 text-gb-fg1 border border-gb-bg2 rounded px-2 py-1 text-[11px] font-data"
                        >
                          <option value="">(unchanged)</option>
                          {(d.edited_type === "organization" ? ORG_ROLE_OPTIONS : LOCATION_ROLE_OPTIONS).map((r) => (
                            <option key={r} value={r}>{r}</option>
                          ))}
                        </select>
                      </div>
                    )}
                  </div>
                  <div className="flex flex-col gap-0.5">
                    <label className="text-[9px] text-gb-fg4 font-data uppercase tracking-wider">
                      Rationale<InfoDot term="entity-rationale" />
                    </label>
                    <input
                      type="text"
                      value={d.rationale}
                      onChange={(e) => updateDecision(entity.id, "rationale", e.target.value)}
                      placeholder="Why are you editing this entity?"
                      className="w-full bg-gb-bg1 text-gb-fg1 border border-gb-bg2 rounded px-2 py-1 text-[11px] font-data placeholder-gb-gray"
                    />
                  </div>
                </div>
              )}

              {/* Rationale for remove */}
              {isRemove && (
                <div className="mt-2 pt-2 border-t border-gb-bg1">
                  <input
                    type="text"
                    value={d.rationale}
                    onChange={(e) => updateDecision(entity.id, "rationale", e.target.value)}
                    placeholder="Why remove this entity?"
                    className="w-full bg-gb-bg1 text-gb-fg1 border border-gb-bg2 rounded px-2 py-1 text-[11px] font-data placeholder-gb-gray"
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>

      {/* Add Entity */}
      <div>
        <button
          onClick={() => setShowAddForm((v) => !v)}
          className="text-[12px] font-semibold text-gb-bright-aqua hover:text-gb-aqua transition-colors"
        >
          + Add Entity
        </button>
        {showAddForm && (
          <div className="mt-2 p-3 bg-gb-bg0-s border border-gb-bg2 rounded-lg flex flex-col gap-2">
            <div className="flex gap-2">
              <div className="flex flex-col gap-1">
                <label className="text-[10px] text-gb-fg4 font-data">Type</label>
                <select
                  value={newEntity.type}
                  onChange={(e) => setNewEntity((p) => ({ ...p, type: e.target.value }))}
                  className="bg-gb-bg1 text-gb-fg1 border border-gb-bg2 rounded px-2 py-1 text-[11px] font-data"
                >
                  {ENTITY_TYPES.map((t) => (
                    <option key={t} value={t}>{t}</option>
                  ))}
                </select>
              </div>
              <div className="flex flex-col gap-1 flex-1">
                <label className="text-[10px] text-gb-fg4 font-data">Value</label>
                <input
                  type="text"
                  value={newEntity.value}
                  onChange={(e) => setNewEntity((p) => ({ ...p, value: e.target.value }))}
                  placeholder="e.g. 192.168.1.100"
                  className="bg-gb-bg1 text-gb-fg1 border border-gb-bg2 rounded px-2 py-1 text-[11px] font-data placeholder-gb-gray"
                />
              </div>
            </div>
            <div className="flex gap-2 items-end">
              <input
                type="text"
                value={newEntity.rationale}
                onChange={(e) => setNewEntity((p) => ({ ...p, rationale: e.target.value }))}
                placeholder="Why is this entity missing?"
                className="flex-1 bg-gb-bg1 text-gb-fg1 border border-gb-bg2 rounded px-2 py-1 text-[11px] font-data placeholder-gb-gray"
              />
              <button
                onClick={handleAddEntity}
                disabled={!newEntity.value.trim()}
                className="px-3 py-1 rounded text-[11px] font-semibold bg-gb-bright-aqua text-gb-bg0-h hover:bg-gb-aqua transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
              >
                Add
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Footer: tally + submit */}
      <div className="flex items-center justify-between pt-3 border-t border-gb-bg2">
        <div className="flex gap-3 text-[11px] font-data">
          <span className="text-gb-bright-green"><strong>{tally.approve}</strong> approve</span>
          {tally.edit > 0 && <span className="text-gb-bright-orange"><strong>{tally.edit}</strong> edit</span>}
          {tally.remove > 0 && <span className="text-gb-bright-red"><strong>{tally.remove}</strong> remove</span>}
          {addedEntities.length > 0 && (
            <span className="text-gb-bright-aqua"><strong>{addedEntities.length}</strong> added</span>
          )}
        </div>
        <button
          onClick={handleSubmit}
          disabled={submitting}
          className="px-4 py-1.5 rounded-md text-[12px] font-semibold bg-gb-green text-gb-bg0-h hover:bg-gb-bright-green transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
        >
          {submitting ? "Submitting..." : "Submit Review"}
        </button>
      </div>
    </div>
  );
}
