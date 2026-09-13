import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import StixNodeIcon from "./StixNodeIcon";

/**
 * Search-and-jump input used by BundleGraph (canvas Explorer),
 * BundleReviewCanvas (Gate 2), and BundleFlowView (Flow tab).
 *
 * Match strategy: case-insensitive substring on displayName, exact
 * match on STIX id, and prefix match on mitreId (so analysts can
 * type "T1059" to jump to PowerShell). Results are ranked: exact
 * id > mitre-prefix > displayName-prefix > substring. The top 8 are
 * shown in a dropdown below the input.
 *
 * Keyboard: type to live-filter; ArrowUp/Down to cycle highlight;
 * Enter to jump to the highlighted match; Escape to clear.
 *
 * Props:
 *   nodes      Array of { id, displayName, type, mitreId? }.
 *   onJump     Called with the matched node when the user picks one
 *              (Enter or click). The caller is responsible for any
 *              viewer-specific selection + transform.
 *   placeholder Optional input placeholder. Default "Find node…".
 *   className   Optional wrapper className.
 */
export default function NodeSearchBox({ nodes, onJump, placeholder = "Find node…", className = "" }) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const inputRef = useRef(null);
  const containerRef = useRef(null);

  const matches = useMemo(() => rankMatches(nodes, query), [nodes, query]);

  // Close on outside click.
  useEffect(() => {
    if (!open) return;
    const onDocClick = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) {
        setOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocClick);
    return () => document.removeEventListener("mousedown", onDocClick);
  }, [open]);

  // Reset active when matches change.
  useEffect(() => {
    setActive(0);
  }, [query]);

  const pick = useCallback(
    (node) => {
      onJump(node);
      setOpen(false);
      setQuery("");
      inputRef.current?.blur();
    },
    [onJump],
  );

  const onKeyDown = useCallback(
    (e) => {
      if (e.key === "ArrowDown") {
        e.preventDefault();
        setActive((a) => Math.min(matches.length - 1, a + 1));
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setActive((a) => Math.max(0, a - 1));
      } else if (e.key === "Enter") {
        if (matches[active]) {
          e.preventDefault();
          pick(matches[active]);
        }
      } else if (e.key === "Escape") {
        // Consumed here: the review panel closes on a document-level
        // Escape, which would discard every unsaved edit.
        e.stopPropagation();
        setQuery("");
        setOpen(false);
        inputRef.current?.blur();
      }
    },
    [matches, active, pick],
  );

  return (
    <div ref={containerRef} className={`relative ${className}`}>
      <input
        ref={inputRef}
        type="text"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(query.length > 0)}
        onKeyDown={onKeyDown}
        placeholder={placeholder}
        className="text-[11px] font-data px-2 py-1 w-44 rounded border border-gb-bg2 bg-gb-bg0 text-gb-fg1 placeholder:text-gb-fg4 focus:border-gb-bright-blue focus:outline-none"
      />
      {open && matches.length > 0 && (
        <div
          className="absolute top-full left-0 mt-0.5 w-72 max-h-72 overflow-y-auto rounded border border-gb-bg2 bg-gb-bg0 shadow-lg z-30"
        >
          {matches.map((m, i) => (
            <button
              key={m.id}
              type="button"
              onMouseEnter={() => setActive(i)}
              onMouseDown={(e) => {
                // mousedown (not click) so the input doesn't blur first
                // and dismiss the dropdown before we get the pick.
                e.preventDefault();
                pick(m);
              }}
              className={`w-full flex items-center gap-2 px-2 py-1 text-left transition-colors ${
                i === active ? "bg-gb-bg1" : "hover:bg-gb-bg1"
              }`}
            >
              <StixNodeIcon type={m.type} size={18} className="shrink-0" title={m.type} />
              <div className="min-w-0 flex-1">
                <div className="text-[11px] font-data text-gb-fg1 truncate" title={m.displayName}>
                  {m.mitreId ? `${m.mitreId}: ${m.displayName}` : m.displayName}
                </div>
                <div className="text-[9px] font-data text-gb-fg4 truncate">{m.type}</div>
              </div>
            </button>
          ))}
        </div>
      )}
      {open && query.length > 0 && matches.length === 0 && (
        <div className="absolute top-full left-0 mt-0.5 w-72 rounded border border-gb-bg2 bg-gb-bg0 px-2 py-1.5 text-[11px] font-data text-gb-fg4 z-30">
          No matches for "{query}"
        </div>
      )}
    </div>
  );
}


/**
 * Score and rank matches. Returns up to 8 best matches.
 * Scoring (higher wins):
 *   100  exact id match
 *    80  exact mitreId match
 *    60  displayName starts with query (case-insensitive)
 *    40  mitreId starts with query
 *    20  displayName contains query
 *     0  no match
 */
function rankMatches(nodes, query) {
  if (!query) return [];
  const q = query.trim().toLowerCase();
  if (!q) return [];
  const scored = [];
  for (const n of nodes) {
    const name = (n.displayName || "").toLowerCase();
    const mid = (n.mitreId || "").toLowerCase();
    const id = (n.id || "").toLowerCase();
    let score = 0;
    if (id === q) score = 100;
    else if (mid === q) score = 80;
    else if (name.startsWith(q)) score = 60;
    else if (mid && mid.startsWith(q)) score = 40;
    else if (name.includes(q)) score = 20;
    if (score > 0) scored.push({ ...n, _score: score });
  }
  scored.sort((a, b) => b._score - a._score || a.displayName.localeCompare(b.displayName));
  return scored.slice(0, 8);
}
