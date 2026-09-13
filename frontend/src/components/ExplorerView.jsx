/**
 * ExplorerView — browse completed STIX bundles with side-by-side
 * source document viewer and graph visualizer.
 *
 * Layout: left sidebar (completed source list) + split content area
 * (source doc viewer | graph visualizer).
 */
import { useState, useEffect, useCallback, useRef } from "react";
import { listBundles, getBundle, renameBundle, deleteBundle } from "../api/bundles";
import SourceDocViewer from "./SourceDocViewer";
import BundleGraph from "./BundleGraph";
import BundleFlowView from "./BundleFlowView";
import ConfirmModal from "./ConfirmModal";
import usePolling from "../hooks/usePolling";

const POLL_INTERVAL_MS = 30000; // Refresh bundle list every 30s

export default function ExplorerView() {
  const [bundles, setBundles] = useState([]);
  const [total, setTotal] = useState(0);
  // `loading` only reflects the FIRST fetch on this mount. Subsequent
  // polling refreshes update the list silently — we don't want the
  // spinner flashing every 30s while the analyst is browsing. Also
  // prevents a "stuck Loading..." UX if a re-render races with the
  // initial fetch's resolution.
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedId, setSelectedId] = useState(null);
  const [selectedBundle, setSelectedBundle] = useState(null);
  const [loadingBundle, setLoadingBundle] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [splitRatio, setSplitRatio] = useState(0.42);
  const [docCollapsed, setDocCollapsed] = useState(false);
  const splitRef = useRef(null);
  const draggingRef = useRef(false);
  // Mirrors draggingRef for rendering. The ref drives the move handler (no
  // re-render per pixel); this drives the drag overlay, which exists so the
  // pointer never hovers the PDF iframe mid-drag — belt to pointer capture's
  // braces — and so the col-resize cursor stays put instead of flickering to
  // whatever is underneath.
  const [splitDragging, setSplitDragging] = useState(false);

  // Inline rename state: which bundle is being edited + draft text
  const [editingId, setEditingId] = useState(null);
  const [editDraft, setEditDraft] = useState("");
  const [renameError, setRenameError] = useState(null);

  // Right-pane viewer mode: "graph" (existing canvas-based force-directed
  // explorer) vs. "flow" (procedures + precedes edges via React Flow).
  // Graph stays the default — it surfaces the full bundle, not just the
  // procedure spine. Analyst flips to Flow to walk the kill chain or
  // drill into a specific procedure's resolved fields.
  const [viewerMode, setViewerMode] = useState("graph");

  // Delete modal state: bundle staged for deletion (null = closed)
  const [deleteTarget, setDeleteTarget] = useState(null);
  const [actionError, setActionError] = useState(null);

  // ── Fetch bundle list ──────────────────────────────────────────
  // usePolling carries the two lessons this view learned the hard way: a
  // cancelled flag the callback checks before touching state (a fetch
  // resolving after the toggle back to Kanban must no-op), and
  // setTimeout-recursion so a slow listBundles cannot stack a second
  // request on top of the first the way setInterval did.
  const fetchBundles = useCallback(async (isCancelled) => {
    try {
      const data = await listBundles({ limit: 200 });
      if (isCancelled()) return;
      setBundles(data.bundles || []);
      setTotal(data.total || 0);
      setError(null);
    } catch (err) {
      if (isCancelled()) return;
      console.error("ExplorerView: fetchBundles failed", err);
      setError("Failed to load completed bundles");
    } finally {
      // Only the first fetch shows the spinner; later ticks find it
      // already false and React skips the no-op update.
      if (!isCancelled()) setLoading(false);
    }
  }, []);
  usePolling(fetchBundles, POLL_INTERVAL_MS);

  // ── Load full bundle when selection changes ────────────────────
  useEffect(() => {
    if (!selectedId) {
      setSelectedBundle(null);
      return;
    }
    let cancelled = false;
    setLoadingBundle(true);
    getBundle(selectedId)
      .then((data) => {
        if (!cancelled) setSelectedBundle(data);
      })
      .catch((err) => {
        console.error("ExplorerView: getBundle failed", err);
        if (!cancelled) setSelectedBundle(null);
      })
      .finally(() => {
        if (!cancelled) setLoadingBundle(false);
      });
    return () => { cancelled = true; };
  }, [selectedId]);

  // Auto-select first bundle on load
  useEffect(() => {
    if (!selectedId && bundles.length > 0) {
      setSelectedId(bundles[0].id);
    }
  }, [bundles, selectedId]);

  // ── Resizable split pane ───────────────────────────────────────

  // Pointer events with capture, NOT mouse events on `document`.
  //
  // The left pane renders the PDF in an <iframe> (SourceDocViewer), and an
  // iframe is a separate browsing context: while dragging, the moment the
  // cursor crossed into it the parent document stopped receiving mousemove
  // AND — the actual bug — never received the mouseup, because that fired
  // inside the iframe. draggingRef stayed true forever, so moving back over
  // the app resumed resizing with no button held. The only way to drag
  // successfully was to keep the cursor on the 10px handle the whole way.
  //
  // setPointerCapture retargets every subsequent event for this pointer to
  // the handle regardless of what sits underneath, so the iframe cannot
  // intercept them. It also fixes the same bug's other trigger: releasing the
  // button outside the browser window, where no mouseup ever arrives.
  //
  // lostpointercapture is the backstop — the browser fires it if capture is
  // broken for any reason (element removed, pointer cancelled), so the drag
  // can never be left stuck on.
  const handleSplitPointerDown = useCallback((e) => {
    e.preventDefault();
    e.currentTarget.setPointerCapture?.(e.pointerId);
    draggingRef.current = true;
    setSplitDragging(true);
  }, []);

  const handleSplitPointerMove = useCallback((e) => {
    if (!draggingRef.current || !splitRef.current) return;
    const rect = splitRef.current.getBoundingClientRect();
    const ratio = (e.clientX - rect.left) / rect.width;
    setSplitRatio(Math.max(0.2, Math.min(0.7, ratio)));
  }, []);

  const endSplitDrag = useCallback((e) => {
    if (!draggingRef.current) return;
    draggingRef.current = false;
    setSplitDragging(false);
    e?.currentTarget?.releasePointerCapture?.(e.pointerId);
  }, []);

  // No unmount cleanup needed any more: capture lives on the handle, so it
  // dies with the element instead of leaking listeners onto `document`.

  // ── Rename / delete handlers ───────────────────────────────────
  const startEditing = useCallback((bundle, e) => {
    e.stopPropagation();
    setEditingId(bundle.id);
    setEditDraft(bundle.title || "");
    setRenameError(null);
  }, []);

  const cancelEditing = useCallback(() => {
    setEditingId(null);
    setEditDraft("");
    setRenameError(null);
  }, []);

  const submitRename = useCallback(async (bundleId) => {
    const trimmed = editDraft.trim();
    if (!trimmed) {
      setRenameError("Title cannot be empty");
      return;
    }
    if (trimmed.length > 512) {
      setRenameError("Title must be 512 characters or fewer");
      return;
    }
    // Find current title: no-op when unchanged.
    const current = bundles.find((b) => b.id === bundleId)?.title || "";
    if (trimmed === current) {
      cancelEditing();
      return;
    }
    try {
      const updated = await renameBundle(bundleId, trimmed);
      setBundles((prev) =>
        prev.map((b) => (b.id === bundleId ? { ...b, title: updated.title } : b))
      );
      // If the selected bundle was renamed, keep its loaded detail in sync.
      setSelectedBundle((prev) =>
        prev && prev.id === bundleId ? { ...prev, title: updated.title } : prev
      );
      cancelEditing();
    } catch (err) {
      console.error("ExplorerView: renameBundle failed", err);
      const detail = err?.response?.data?.detail;
      setRenameError(
        typeof detail === "string" ? detail : "Rename failed. Try again."
      );
    }
  }, [editDraft, bundles, cancelEditing]);

  const openDeleteModal = useCallback((bundle, e) => {
    e.stopPropagation();
    setDeleteTarget(bundle);
    setActionError(null);
  }, []);

  const confirmDelete = useCallback(async () => {
    if (!deleteTarget) return;
    const targetId = deleteTarget.id;
    try {
      await deleteBundle(targetId);
      setBundles((prev) => prev.filter((b) => b.id !== targetId));
      // Clear selection if the deleted bundle was selected.
      setSelectedId((prev) => (prev === targetId ? null : prev));
      setSelectedBundle((prev) => (prev && prev.id === targetId ? null : prev));
      setTotal((prev) => Math.max(0, prev - 1));
      setDeleteTarget(null);
    } catch (err) {
      console.error("ExplorerView: deleteBundle failed", err);
      const detail = err?.response?.data?.detail;
      setActionError(
        typeof detail === "string" ? detail : "Delete failed. Try again."
      );
      // Rethrow so ConfirmModal doesn't auto-close on failure.
      throw err;
    }
  }, [deleteTarget]);

  // ── Filtered bundles ───────────────────────────────────────────
  const filtered = searchQuery
    ? bundles.filter((b) =>
        b.title.toLowerCase().includes(searchQuery.toLowerCase())
      )
    : bundles;

  // ── Source type badge ──────────────────────────────────────────
  const sourceTypeBadge = (fileType) => {
    if (!fileType) return null;
    if (fileType.includes("pdf")) return { label: "PDF", cls: "bg-[rgba(251,73,52,0.12)] text-gb-bright-red" };
    if (fileType.includes("html")) return { label: "HTML", cls: "bg-[rgba(131,165,152,0.12)] text-gb-bright-blue" };
    if (fileType.includes("markdown") || fileType.includes("text/plain")) return { label: "TEXT", cls: "bg-[rgba(184,187,38,0.12)] text-gb-bright-green" };
    return { label: fileType.split("/")[1]?.toUpperCase() || "FILE", cls: "bg-gb-bg1 text-gb-fg4" };
  };

  return (
    <div className="flex flex-1 min-h-0 relative">
      {/* ── Toast: action error banner ───────────────────────────── */}
      {/* z-[60] keeps this above the delete modal (z-50) so a cascade
          failure is still visible while the modal is open. */}
      {actionError && (
        <div
          role="alert"
          className="fixed top-14 left-1/2 -translate-x-1/2 z-[60] px-3.5 py-2 rounded-md bg-[rgba(251,73,52,0.12)] border border-gb-red text-gb-bright-red text-[12px] flex items-center gap-3 shadow-lg"
        >
          <span>{actionError}</span>
          <button
            type="button"
            onClick={() => setActionError(null)}
            className="text-gb-fg4 hover:text-gb-fg1"
            aria-label="Dismiss error"
          >
            ×
          </button>
        </div>
      )}

      {/* ── Left sidebar: completed bundles ──────────────────────── */}
      <aside className="w-[280px] min-w-[280px] bg-gb-bg0 border-r border-gb-bg2 flex flex-col">
        <div className="px-3.5 py-2.5 border-b border-gb-bg1 flex items-center justify-between">
          <h2 className="text-[12px] font-semibold text-gb-fg3 uppercase tracking-wider">
            Completed
          </h2>
          <span className="text-[11px] font-data text-gb-gray bg-gb-bg1 px-2 py-0.5 rounded-full">
            {total}
          </span>
        </div>

        <div className="px-3.5 py-2 border-b border-gb-bg1">
          <input
            type="text"
            placeholder="Search bundles..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="w-full text-[12px] px-2.5 py-1.5 bg-gb-bg0-s border border-gb-bg2 rounded-md text-gb-fg2 placeholder-gb-bg4 outline-none focus:border-gb-bright-blue font-ui"
          />
        </div>

        <div className="flex-1 overflow-y-auto px-2 py-1.5">
          {loading && (
            <div className="flex items-center justify-center py-8 text-gb-gray text-sm">
              Loading...
            </div>
          )}

          {error && (
            <div className="px-2 py-3 text-gb-bright-orange text-[12px]">
              {error}
              <button onClick={fetchBundles} className="ml-2 underline hover:text-gb-bright-yellow">
                Retry
              </button>
            </div>
          )}

          {!loading && filtered.length === 0 && (
            <div className="flex flex-col items-center justify-center py-12 text-gb-bg4 text-[12px] gap-1">
              <span className="text-xl opacity-40">📂</span>
              {bundles.length === 0
                ? "No completed bundles yet"
                : "No bundles match your search"}
            </div>
          )}

          {filtered.map((b) => {
            const badge = sourceTypeBadge(b.source_file_type);
            const isSelected = b.id === selectedId;
            const isEditing = editingId === b.id;
            const date = b.completed_at
              ? new Date(b.completed_at).toLocaleDateString("en-US", {
                  month: "short", day: "numeric", year: "numeric",
                })
              : "";
            return (
              <div
                key={b.id}
                onClick={() => !isEditing && setSelectedId(b.id)}
                className={`group relative px-3 py-2.5 rounded-lg mb-1 transition-all border ${
                  isEditing ? "cursor-default" : "cursor-pointer"
                } ${
                  isSelected
                    ? "bg-gb-bg1 border-gb-bright-orange shadow-[inset_3px_0_0_var(--color-gb-bright-orange)]"
                    : "border-transparent hover:bg-gb-bg0-s hover:border-gb-bg2"
                }`}
              >
                {isEditing ? (
                  <div className="mb-1" onClick={(e) => e.stopPropagation()}>
                    <input
                      type="text"
                      value={editDraft}
                      maxLength={512}
                      autoFocus
                      onChange={(e) => setEditDraft(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") submitRename(b.id);
                        else if (e.key === "Escape") cancelEditing();
                      }}
                      onBlur={() => submitRename(b.id)}
                      className="w-full text-[13px] font-semibold px-1.5 py-1 bg-gb-bg0-h border border-gb-bright-orange rounded-md text-gb-fg0 outline-none font-ui"
                    />
                    {renameError && (
                      <div className="text-[10px] text-gb-bright-red mt-1">
                        {renameError}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="flex items-start justify-between gap-1.5 mb-1">
                    <div className="text-[13px] font-semibold text-gb-fg1 truncate flex-1" title={b.title}>
                      {b.title}
                    </div>
                    <div className="flex items-center gap-0.5 opacity-0 group-hover:opacity-100 transition-opacity shrink-0">
                      <button
                        type="button"
                        onClick={(e) => startEditing(b, e)}
                        title="Rename"
                        aria-label={`Rename bundle ${b.title}`}
                        className="p-1 rounded hover:bg-gb-bg2 text-gb-fg4 hover:text-gb-bright-yellow transition-colors"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M17 3a2.85 2.83 0 1 1 4 4L7.5 20.5 2 22l1.5-5.5Z" />
                        </svg>
                      </button>
                      <button
                        type="button"
                        onClick={(e) => openDeleteModal(b, e)}
                        title="Delete"
                        aria-label={`Delete bundle ${b.title}`}
                        className="p-1 rounded hover:bg-gb-bg2 text-gb-fg4 hover:text-gb-bright-red transition-colors"
                      >
                        <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                          <path d="M3 6h18" />
                          <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6" />
                          <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                          <line x1="10" y1="11" x2="10" y2="17" />
                          <line x1="14" y1="11" x2="14" y2="17" />
                        </svg>
                      </button>
                    </div>
                  </div>
                )}
                <div className="flex items-center gap-2 text-[10px] font-data text-gb-gray">
                  {badge && (
                    <span className={`px-1.5 py-px rounded text-[9px] font-semibold uppercase ${badge.cls}`}>
                      {badge.label}
                    </span>
                  )}
                  <span>{date}</span>
                </div>
                <div className="flex gap-2.5 mt-1 text-[10px] font-data text-gb-fg4">
                  <span className="flex items-center gap-1">
                    <span className="w-[5px] h-[5px] rounded-full bg-gb-bright-orange" />
                    {b.procedure_count} procedures
                  </span>
                  <span className="flex items-center gap-1">
                    <span className="w-[5px] h-[5px] rounded-full bg-gb-bright-blue" />
                    {b.object_count} objects
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </aside>

      {/* ── Content area: split pane ─────────────────────────────── */}
      <div ref={splitRef} className="flex-1 flex min-h-0 min-w-0">
        {!selectedBundle && !loadingBundle && (
          <div className="flex-1 flex flex-col items-center justify-center text-gb-bg4 text-sm gap-2">
            <span className="text-3xl opacity-30">🔬</span>
            {bundles.length === 0
              ? "Complete a pipeline run to see bundles here"
              : "Select a bundle to explore"}
          </div>
        )}

        {loadingBundle && (
          <div className="flex-1 flex items-center justify-center text-gb-gray text-sm">
            Loading bundle...
          </div>
        )}

        {selectedBundle && !loadingBundle && (
          <>
            {/* Source Document Viewer (collapsible) */}
            {!docCollapsed && (
              <div style={{ flex: `0 0 ${splitRatio * 100}%` }} className="min-w-[280px] flex flex-col border-r border-gb-bg2 relative">
                <button
                  type="button"
                  onClick={() => setDocCollapsed(true)}
                  title="Hide document viewer"
                  className="absolute top-2 right-2 z-20 w-6 h-6 flex items-center justify-center rounded bg-gb-bg1 hover:bg-gb-bg2 text-gb-fg3 hover:text-gb-fg1 text-[14px] font-data border border-gb-bg2"
                >
                  «
                </button>
                <SourceDocViewer
                  bundleId={selectedBundle.id}
                  fileName={selectedBundle.source_file_name}
                  fileType={selectedBundle.source_file_type}
                />
              </div>
            )}

            {/* Resize handle (only when document is visible) */}
            {!docCollapsed && (
              <div
                onPointerDown={handleSplitPointerDown}
                onPointerMove={handleSplitPointerMove}
                onPointerUp={endSplitDrag}
                onPointerCancel={endSplitDrag}
                onLostPointerCapture={endSplitDrag}
                className="w-[10px] cursor-col-resize relative shrink-0 z-10 group hover:bg-gb-bg2/40 transition-colors touch-none"
                title="Drag to resize"
              >
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[3px] h-12 bg-gb-bg3 rounded-sm opacity-60 group-hover:opacity-100 group-hover:bg-gb-bright-aqua transition-all" />
              </div>
            )}

            {/* Drag overlay. Covers the viewport while the splitter is held so
                the cursor never sits over the PDF iframe, and so the
                col-resize cursor does not flicker to whatever is underneath.
                Pointer capture already routes the events correctly; this
                keeps the drag looking like one continuous gesture. */}
            {splitDragging && (
              <div className="fixed inset-0 z-40 cursor-col-resize select-none" />
            )}

            {/* Show-document tab when collapsed */}
            {docCollapsed && (
              <button
                type="button"
                onClick={() => setDocCollapsed(false)}
                title="Show document viewer"
                className="w-6 shrink-0 flex items-center justify-center bg-gb-bg1 hover:bg-gb-bg2 text-gb-fg3 hover:text-gb-fg1 text-[14px] font-data border-r border-gb-bg2 transition-colors"
              >
                »
              </button>
            )}

            {/* Graph / Flow Visualizer */}
            <div className="flex-1 min-w-[350px] flex flex-col relative">
              {/* Mode toggle. Position over the visualizer's top-left so it
                  doesn't compete with the BundleGraph layer-toggle bar
                  (which lives in its own header). */}
              <div className="absolute top-2 right-2 z-20 flex items-center gap-1 bg-gb-bg0-s border border-gb-bg2 rounded-md p-0.5 shadow-sm">
                {["graph", "flow"].map((mode) => (
                  <button
                    key={mode}
                    type="button"
                    onClick={() => setViewerMode(mode)}
                    className={`px-2.5 py-1 rounded text-[11px] font-medium transition-colors ${
                      viewerMode === mode
                        ? "bg-gb-bg1 text-gb-bright-yellow"
                        : "text-gb-fg4 hover:text-gb-fg1"
                    }`}
                  >
                    {mode === "graph" ? "Graph" : "Flow"}
                  </button>
                ))}
              </div>
              {viewerMode === "graph" && (
                <BundleGraph bundleJson={selectedBundle.bundle_json} />
              )}
              {viewerMode === "flow" && (
                <BundleFlowView bundle={selectedBundle.bundle_json} />
              )}
            </div>
          </>
        )}
      </div>

      {/* ── Delete confirmation modal (typed-confirm) ────────────── */}
      <ConfirmModal
        isOpen={!!deleteTarget}
        title="Delete bundle?"
        message={
          deleteTarget
            ? `Removes the completed bundle and its source queue entry. Neo4j graph data is NOT touched. This cannot be undone.`
            : ""
        }
        confirmLabel="Delete"
        danger
        requireTypedConfirm="DELETE"
        onConfirm={confirmDelete}
        onClose={() => setDeleteTarget(null)}
      />
    </div>
  );
}
