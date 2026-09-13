/**
 * Technique catalogue API.
 * Fetches the full active ATT&CK technique list for search/autocomplete.
 */
import client from "./client";

let _cache = null;

/**
 * Fetch the full ATT&CK technique catalogue.
 * Cached in-memory after the first call.
 * @returns {Promise<Array<{technique_id, name, stix_id, description, tactics, platforms}>>}
 */
export async function fetchTechniques() {
  if (_cache) return _cache;
  const { data } = await client.get("/techniques/");
  _cache = data;
  return data;
}
