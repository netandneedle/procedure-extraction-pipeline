/**
 * Gate review API calls.
 * Maps to FastAPI routes at /api/gates/*.
 */
import client from "./client";

/**
 * Fetch pending review data for a gate.
 * @param {string} threadId  UUID of the pipeline thread
 * @param {number} gateId    0 | 1 | 2
 * @returns {Promise<{gate_id, thread_id, source_id, status, items, context}>}
 */
export async function fetchPendingReview(threadId, gateId) {
  const { data } = await client.get(`/gates/${threadId}/${gateId}/pending`);
  return data;
}

/**
 * Submit Gate 0 review (entity decisions).
 *
 * Accepts either:
 *   - an array of reviews (legacy callers; wrapped into {reviews}).
 *   - an object {reviews, checkpoint_id?} matching the Gate0Submit schema.
 *
 * @param {string} threadId  UUID of the pipeline thread
 * @param {Array|object} body  Reviews array or {reviews, checkpoint_id?} object
 * @returns {Promise<{thread_id, gate_id, status, next_node}>}
 */
export async function submitGate0(threadId, body) {
  const payload = Array.isArray(body) ? { reviews: body } : body;
  const { data } = await client.post(`/gates/${threadId}/0/submit`, payload);
  return data;
}

/**
 * Submit Gate 1 review (procedure draft decisions + optional promotions).
 *
 * Accepts either:
 *   - an array of reviews (legacy callers; wrapped into {reviews}).
 *   - an object {reviews, promotions} where promotions is an array of
 *     {chunk_id, technique_id} for analyst-promoted possible-bucket picks
 *     (C+A+D review lane).
 *
 * @param {string} threadId  UUID of the pipeline thread
 * @param {Array|object} body  Reviews array or {reviews, promotions} object
 * @returns {Promise<{thread_id, gate_id, status, next_node}>}
 */
export async function submitGate1(threadId, body) {
  const payload = Array.isArray(body) ? { reviews: body } : body;
  const { data } = await client.post(`/gates/${threadId}/1/submit`, payload);
  return data;
}

/**
 * Submit Gate 2 review (relationship decisions).
 * @param {string} threadId  UUID of the pipeline thread
 * @param {object} payload   { approved: bool, feedback?: string }
 * @returns {Promise<{thread_id, gate_id, status, next_node}>}
 */
export async function submitGate2(threadId, payload) {
  const { data } = await client.post(`/gates/${threadId}/2/submit`, payload);
  return data;
}

/**
 * Fetch pending chunk-review data (string-keyed path, separate from int gates).
 * @param {string} threadId
 * @returns {Promise<{thread_id, source_id, status, chunks, parsed_text, validated_entities, classified_sections}>}
 */
export async function fetchPendingChunkReview(threadId) {
  const { data } = await client.get(`/gates/${threadId}/chunks/pending`);
  return data;
}

/**
 * Submit chunk-review decisions.
 * @param {string} threadId
 * @param {object} payload  { decisions, added_chunks, edges, reject? }
 * @returns {Promise<{thread_id, gate_id, status, next_node}>}
 */
export async function submitChunkReview(threadId, payload) {
  const { data } = await client.post(`/gates/${threadId}/chunks/submit`, payload);
  return data;
}
