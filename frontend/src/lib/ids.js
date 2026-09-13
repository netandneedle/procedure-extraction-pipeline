/**
 * Client-side ids for things that do not exist on the server yet — an
 * analyst-added chunk or relationship before the gate submit allocates the
 * real one.
 *
 * crypto.randomUUID exists only in secure contexts (https, or localhost).
 * Served over plain http from a LAN or Tailscale address it is undefined,
 * and calling it inside a state updater threw during render and unmounted
 * the root. getRandomValues is available everywhere; the uuid is a
 * convenience, not a requirement — callers slice it for a short tag.
 */
export function newId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  const bytes = new Uint8Array(16);
  if (typeof crypto !== "undefined" && typeof crypto.getRandomValues === "function") {
    crypto.getRandomValues(bytes);
  } else {
    for (let i = 0; i < bytes.length; i += 1) bytes[i] = Math.floor(Math.random() * 256);
  }
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("");
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`;
}
