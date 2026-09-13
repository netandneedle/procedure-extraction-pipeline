/** Truncate to `n` characters with an ellipsis. Was copied byte-for-byte
 *  into two canvases; one place now. */
export function ellipsize(s, n) {
  if (!s) return "";
  return s.length > n ? `${s.slice(0, n)}…` : s;
}
