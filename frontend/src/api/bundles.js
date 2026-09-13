/**
 * Bundle Explorer API calls.
 * Maps to FastAPI routes at /api/bundles/*.
 */
import client from "./client";

/** List completed bundles (metadata only, no file bytes). */
export async function listBundles({ limit = 50, offset = 0 } = {}) {
  const { data } = await client.get("/bundles/", { params: { limit, offset } });
  if (!data || !Array.isArray(data.bundles)) {
    throw new Error("listBundles: malformed response (expected { bundles: [...] })");
  }
  return { bundles: data.bundles, total: Number.isFinite(data.total) ? data.total : data.bundles.length };
}

/** Get a single bundle including full STIX JSON. */
export async function getBundle(bundleId) {
  const { data } = await client.get(`/bundles/${bundleId}`);
  if (!data || !data.bundle_json || !Array.isArray(data.bundle_json.objects)) {
    throw new Error("getBundle: malformed bundle (expected bundle_json.objects array)");
  }
  return data;
}

/**
 * Get the source file URL for a bundle.
 * Returns a URL string that can be used in an iframe src or fetch.
 */
export function getSourceFileUrl(bundleId) {
  return `/api/bundles/${bundleId}/source`;
}

/**
 * Rename a bundle. Title is stripped server-side; must be 1-512 chars
 * and free of ASCII control characters. Returns the updated metadata.
 */
export async function renameBundle(bundleId, title) {
  const { data } = await client.patch(`/bundles/${bundleId}`, { title });
  if (!data || typeof data.title !== "string") {
    throw new Error("renameBundle: malformed response (expected { title })");
  }
  return data;
}

/**
 * Delete a bundle. Cascades to the linked source queue row and any
 * uploaded source file under upload_dir. Neo4j is not touched.
 * Returns true on 204, throws on 404 / network error.
 */
export async function deleteBundle(bundleId) {
  await client.delete(`/bundles/${bundleId}`);
  return true;
}
