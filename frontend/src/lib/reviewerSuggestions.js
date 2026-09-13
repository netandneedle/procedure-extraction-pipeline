/**
 * lib/reviewerSuggestions.js — pure logic for AI reviewer recommendations.
 *
 * Kept out of the components so the rules that decide what a bulk-accept
 * touches are unit-testable. That matters more here than usual: the whole
 * value of assist mode rests on the analyst actually reviewing, and the
 * bulk-accept filter is the only thing standing between "assisted review"
 * and "autopilot with extra clicks".
 */

/** Tailwind classes per confidence tier. */
export const CONFIDENCE_STYLES = {
  high: "bg-gb-green/15 text-gb-bright-green border-gb-green",
  medium: "bg-gb-yellow/15 text-gb-bright-yellow border-gb-yellow",
  low: "bg-gb-bg1 text-gb-fg4 border-gb-bg2",
};

/**
 * Index entity recommendations by entity_id.
 * @param {object|null} payload  Reviewer recommendation payload.
 * @returns {Record<string, object>}
 */
export function indexByEntity(payload) {
  const out = {};
  for (const rec of payload?.entities ?? []) {
    if (rec?.entity_id) out[rec.entity_id] = rec;
  }
  return out;
}

/**
 * Is this recommendation eligible for one-click bulk acceptance?
 *
 * Only HIGH confidence, and never one whose evidence quote the report does
 * not support. Both conditions matter and they are not the same condition:
 * the grounding check already forces an unsupported quote down to "low", so
 * the second test is belt-and-braces against a future path that sets
 * confidence without re-running grounding.
 *
 * Everything else must be opened individually. That friction is the feature —
 * if accepting everything were one click, the analyst's override signal (the
 * only evidence that says whether autopilot is ever safe) would never exist.
 */
export function isBulkAcceptable(rec) {
  if (!rec) return false;
  if (rec.quote_unsupported) return false;
  return rec.confidence === "high";
}

/**
 * Apply a recommendation onto an existing decision object.
 *
 * Returns a NEW decision; never mutates. Only fields the reviewer actually
 * specified are overlaid, so accepting an "edit" that names a value but not a
 * type leaves the type as extracted rather than blanking it.
 */
export function applyRecommendation(decision, rec) {
  if (!decision || !rec) return decision;
  const next = { ...decision, action: rec.action ?? decision.action };
  if (rec.edited_value) next.edited_value = rec.edited_value;
  if (rec.edited_type) next.edited_type = rec.edited_type;
  if (rec.edited_role) next.edited_role = rec.edited_role;
  // Never overwrite a rationale the analyst typed. Theirs is the more
  // authoritative reason for the decision, and silently replacing text
  // someone wrote is the surprising behaviour — the same data-loss class
  // already fixed for technique lists. Fill an empty field only.
  if (rec.rationale && !decision.rationale) next.rationale = rec.rationale;
  return next;
}

/**
 * Map a Gate 1 draft recommendation onto the review panel's verdict model.
 *
 * The panel does not think in the gate's action vocabulary; it thinks in four
 * verdicts, and `reject` splits between two of them by reject_reason. That
 * split is load-bearing: `rechunk` re-runs chunking for the whole source
 * while `remap` only re-runs technique extraction, so getting it wrong throws
 * away far more work than intended.
 *
 * @param {object} rec  DraftRecommendation
 * @returns {"approve"|"discard"|"remap"|"rechunk"}
 */
export function verdictForRecommendation(rec) {
  const action = rec?.action;
  if (action === "remove") return "discard";
  if (action === "reject") {
    return rec?.reject_reason === "bad_chunk_boundary" ? "rechunk" : "remap";
  }
  // approve and edit both ship the draft; edit differs only in carrying
  // technique removals, which are applied separately.
  return "approve";
}

/**
 * Index Gate 1 draft recommendations by draft_id.
 * @param {object|null} payload
 * @returns {Record<string, object>}
 */
export function indexByDraft(payload) {
  const out = {};
  for (const rec of payload?.drafts ?? []) {
    if (rec?.draft_id) out[rec.draft_id] = rec;
  }
  return out;
}

/**
 * Index Gate 1 promotion recommendations by `${chunk_id}|${technique_id}`,
 * the same key the panel already uses for its promoted set.
 * @param {object|null} payload
 * @returns {Record<string, object>}
 */
export function indexByPromotion(payload) {
  const out = {};
  for (const rec of payload?.promotions ?? []) {
    if (rec?.chunk_id && rec?.technique_id) {
      out[`${rec.chunk_id}|${rec.technique_id}`] = rec;
    }
  }
  return out;
}

/**
 * Which decision an undo should restore.
 *
 * Extracted so the branch is testable: the component-level handler is not
 * reachable from the pure-logic tests, and this is exactly where the bug
 * lived. Gate 0's undo used to skip the snapshot entirely and rebuild from
 * the extractor's entity, destroying any value, type, role or rationale the
 * analyst had set by hand — an undo that discards unrelated work.
 *
 * @param {object|undefined} snapshot  decision captured before applying
 * @param {object} fallback            clean rebuild, when nothing was applied
 * @returns {object}
 */
export function pickRestoredDecision(snapshot, fallback) {
  return snapshot === undefined || snapshot === null ? fallback : snapshot;
}

/**
 * Index chunk-gate recommendations by chunk_id.
 * @param {object|null} payload
 * @returns {Record<string, object>}
 */
export function indexByChunk(payload) {
  const out = {};
  for (const rec of payload?.chunks ?? []) {
    if (rec?.chunk_id) out[rec.chunk_id] = rec;
  }
  return out;
}

/**
 * Stable key for one edge recommendation.
 *
 * Includes the ACTION, not just the endpoints: "add A->B" and "remove A->B"
 * are opposite recommendations about the same pair, and keying on the pair
 * alone would let one silently stand in for the other.
 *
 * @param {object} rec  ChunkEdgeRecommendation
 * @returns {string}
 */
export function edgeRecKey(rec) {
  return `${rec?.action}|${rec?.from_chunk_id}->${rec?.to_chunk_id}`;
}

/**
 * Which edge recommendations are still worth showing.
 *
 * An edge the analyst's current graph already satisfies is not a
 * recommendation any more, it is a description. Filtering them keeps the
 * panel to things that would actually change something — the same reason the
 * submit payload only carries net mutations.
 *
 * @param {object|null} payload         reviewer payload
 * @param {Set<string>} liveEdgeKeys    `${from}->${to}` for every edge on the canvas now
 * @returns {object[]}
 */
export function pendingEdgeRecs(payload, liveEdgeKeys) {
  return (payload?.edges ?? []).filter((rec) => {
    const pair = `${rec?.from_chunk_id}->${rec?.to_chunk_id}`;
    const present = liveEdgeKeys.has(pair);
    return rec?.action === "add" ? !present : present;
  });
}

/**
 * Undo one applied chunk recommendation, restoring the analyst's own state.
 *
 * The chunk gate spreads a single decision across three places — the edits
 * map, the dropped set, and the merge groups — so "undo" is three restores
 * that must agree. Pure and exported because this is exactly where the bug
 * has lived twice before: an undo that CLEARS instead of restoring, throwing
 * away the wording, drop or merge the analyst had set by hand as the price
 * of declining an unrelated AI suggestion. A handler buried in the component
 * cannot be tested, and an untested undo is how that shipped.
 *
 * @param {string} chunkId
 * @param {object|undefined} snapshot  {edits, dropped, merge} captured before applying
 * @param {{edits: object, droppedIds: Set<string>, mergeGroups: object}} current
 * @returns {{edits: object, droppedIds: Set<string>, mergeGroups: object}} new values
 */
export function restoreChunkSnapshot(chunkId, snapshot, current) {
  const edits = { ...current.edits };
  if (snapshot?.edits) edits[chunkId] = snapshot.edits;
  else delete edits[chunkId];

  const droppedIds = new Set(current.droppedIds);
  if (snapshot?.dropped) droppedIds.add(chunkId);
  else droppedIds.delete(chunkId);

  const mergeGroups = { ...current.mergeGroups };
  if (snapshot?.merge) mergeGroups[chunkId] = snapshot.merge;
  else delete mergeGroups[chunkId];

  return { edits, droppedIds, mergeGroups };
}

/**
 * Format an agreement rate together with the count it came from.
 *
 * Both parts or neither — that is the whole point of this being one function.
 * A bare percentage is how an agreement readout misleads the person reading
 * it: "100%" reads as a settled fact, "100% (3/3)" reads as three data points.
 * The number decides whether a gate can run unattended, so the sample size is
 * not decoration.
 *
 * `pct` is null when there is nothing to rate — a gate where the reviewer
 * recommended nothing is an absence of evidence, not 0% agreement.
 *
 * @param {number} agreed
 * @param {number} total
 * @returns {{pct: number|null, n: string}}
 */
export function rateLabel(agreed, total) {
  if (!total) return { pct: null, n: "no data" };
  return { pct: Math.round((agreed / total) * 100), n: `${agreed}/${total}` };
}
