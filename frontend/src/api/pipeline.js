/**
 * Pipeline control API calls.
 * Maps to FastAPI routes at /api/pipeline/*.
 */
import client from "./client";

/** Start a pipeline run for a source. */
export async function startPipeline(sourceId, gatesEnabled = null) {
  const payload = { source_id: sourceId };
  if (gatesEnabled !== null) payload.gates_enabled = gatesEnabled;
  const { data } = await client.post("/pipeline/run", payload);
  return data;
}

/** Get pipeline status for a thread. */
export async function getPipelineStatus(threadId) {
  const { data } = await client.get(`/pipeline/status/${threadId}`);
  return data;
}
