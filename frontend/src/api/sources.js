/**
 * Source queue API calls.
 * Maps to FastAPI routes at /api/sources/*.
 */
import client from "./client";

/** List sources with optional filters. */
export async function listSources({ status, claimedBy, limit = 50, offset = 0 } = {}) {
  const params = { limit, offset };
  if (status) params.status = status;
  if (claimedBy) params.claimed_by = claimedBy;
  const { data } = await client.get("/sources/", { params });
  return data;
}

/** Get a single source by ID. */
export async function getSource(sourceId) {
  const { data } = await client.get(`/sources/${sourceId}`);
  return data;
}

/** Create a new source in the queue. */
export async function createSource(payload) {
  const { data } = await client.post("/sources/", payload);
  return data;
}

/** Claim a source for an analyst. */
export async function claimSource(sourceId, analyst) {
  const { data } = await client.patch(`/sources/${sourceId}/claim`, { analyst });
  return data;
}

/** Update source status (Kanban drag-and-drop). */
export async function updateSourceStatus(sourceId, status) {
  const { data } = await client.patch(`/sources/${sourceId}/status`, { status });
  return data;
}

/**
 * Upload a file and get back the server-side path + auto-detected source_type.
 * Response: { path, source_type, filename }
 */
export async function uploadSourceFile(file) {
  const form = new FormData();
  form.append("file", file);
  const { data } = await client.post("/sources/upload", form, {
    headers: { "Content-Type": "multipart/form-data" },
  });
  return data;
}

/** Delete a source and any file it owns. */
export async function deleteSource(sourceId) {
  await client.delete(`/sources/${sourceId}`);
}
