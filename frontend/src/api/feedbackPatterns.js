/**
 * Feedback-pattern management API calls.
 * Maps to FastAPI routes at /api/feedback-patterns/*.
 */
import client from "./client";

/** List feedback patterns (paginated, filterable). */
export async function listFeedbackPatterns({
  category = null,
  status = null,
  minSalience = null,
  search = null,
  limit = 50,
  offset = 0,
} = {}) {
  const params = { limit, offset };
  if (category) params.category = category;
  if (status) params.status = status;
  if (minSalience != null) params.min_salience = minSalience;
  if (search) params.search = search;
  const { data } = await client.get("/feedback-patterns/", { params });
  if (!data || !Array.isArray(data.patterns)) {
    throw new Error("listFeedbackPatterns: malformed response (expected { patterns: [...] })");
  }
  return {
    patterns: data.patterns,
    total: Number.isFinite(data.total) ? data.total : data.patterns.length,
  };
}

/**
 * Captured-but-not-yet-synthesized analyst corrections on still-running
 * sources. These become patterns when a run completes; this surfaces them
 * immediately so feedback isn't invisible mid-run.
 */
export async function listCapturedCorrections() {
  const { data } = await client.get("/feedback-patterns/captured");
  return {
    sources: Array.isArray(data?.sources) ? data.sources : [],
    total: Number.isFinite(data?.total) ? data.total : 0,
  };
}

/**
 * Promote a pattern to a permanent guardrail. action: "prompt" | "denylist".
 * For "denylist", pass denylistTerms = { values: [...], technique_ids: [...] }
 * (the analyst-confirmed concrete things to block). Ignored for "prompt".
 */
export async function promotePattern(patternId, action, by, denylistTerms = null) {
  const body = { action, by };
  if (action === "denylist" && denylistTerms) {
    body.denylist_terms = {
      values: denylistTerms.values || [],
      technique_ids: denylistTerms.technique_ids || [],
      entity_types: denylistTerms.entity_types || [],
    };
  }
  const { data } = await client.patch(`/feedback-patterns/${patternId}/promote`, body);
  if (!data || typeof data.status !== "string") {
    throw new Error("promotePattern: malformed response");
  }
  return data;
}

/** Dismiss a pattern (stop consuming it at prompt-build time). */
export async function dismissPattern(patternId) {
  const { data } = await client.patch(`/feedback-patterns/${patternId}/dismiss`, {});
  if (!data || typeof data.status !== "string") {
    throw new Error("dismissPattern: malformed response");
  }
  return data;
}

/** Edit a pattern's text / category / structured keys. Only provided fields change. */
export async function editPattern(patternId, fields) {
  const { data } = await client.patch(`/feedback-patterns/${patternId}`, fields);
  if (!data || typeof data.id !== "string") {
    throw new Error("editPattern: malformed response");
  }
  return data;
}
