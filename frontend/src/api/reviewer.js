/**
 * AI gate reviewer API calls.
 * Maps to FastAPI routes at /api/reviewer/*.
 *
 * A 404 from these endpoints is the ORDINARY case, not an error: it means the
 * gate ran in plain review mode, or the reviewer has not reached it. Callers
 * get null and render the gate exactly as they always have.
 */
import client from "./client";

/** Swallow 404 (no recommendations) but let real failures surface. */
async function getOrNull(url) {
  try {
    const { data } = await client.get(url);
    return data;
  } catch (err) {
    if (err?.response?.status === 404) return null;
    throw err;
  }
}

/**
 * Latest AI recommendations for one gate.
 * @param {string} sourceId  UUID of the source
 * @param {string} gateKey   "entities" | "chunks" | "procedures" | "bundle"
 * @returns {Promise<object|null>}  Recommendation row, or null if none.
 */
export async function fetchRecommendations(sourceId, gateKey) {
  return getOrNull(`/reviewer/${sourceId}/${gateKey}`);
}

/**
 * The reviewer's opening read of the report.
 * @param {string} sourceId  UUID of the source
 * @returns {Promise<object|null>}
 */
export async function fetchBrief(sourceId) {
  return getOrNull(`/reviewer/${sourceId}/brief`);
}

/**
 * Correct the brief, then re-review the gate currently in progress.
 *
 * The brief is turn one of the reviewer's transcript, so a correction here
 * propagates to every later gate instead of needing to be repeated at each.
 *
 * @param {string} sourceId  UUID of the source
 * @param {object} brief     {summary, actors, attack_chain, thin_areas, notes_for_later_gates}
 * @returns {Promise<object>}  The updated brief row.
 */
export async function updateBrief(sourceId, brief) {
  const { data } = await client.post(`/reviewer/${sourceId}/brief`, brief);
  return data;
}

/**
 * Per-gate agreement between the reviewer and the analyst, across every source.
 *
 * The instrument assist mode exists to produce: how often each gate's advice
 * was actually taken, which is the only evidence that can say whether a gate
 * could ever run unattended.
 *
 * @returns {Promise<{sources_reviewed: number, gates: object[]}>}
 */
export async function fetchAgreement() {
  const { data } = await client.get("/reviewer/agreement");
  return data;
}
