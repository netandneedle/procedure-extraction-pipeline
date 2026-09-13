/**
 * App — root component for the Procedure Extraction Pipeline frontend.
 *
 * Manages source state, API polling, and top-level layout.
 * Views: Source Queue (Kanban), Explorer, Feedback, Reviewer.
 */
import { useState, useEffect, useCallback, useRef, useMemo } from "react";
import KanbanBoard from "./components/KanbanBoard";
import AddSourceModal from "./components/AddSourceModal";
import ConfirmModal from "./components/ConfirmModal";
import GateReviewPanel from "./components/GateReviewPanel";
import StatsBar from "./components/StatsBar";
import useWebSocket from "./hooks/useWebSocket";
import { listSources, createSource, updateSourceStatus, deleteSource } from "./api/sources";
import { startPipeline } from "./api/pipeline";
import ExplorerView from "./components/ExplorerView";
import FeedbackPatternsView from "./components/FeedbackPatternsView";
import ReviewerAgreementView from "./components/ReviewerAgreementView";
import { hint } from "./lib/glossary";
import { GATE_ENABLE_KEYS } from "./lib/gates";
import { ACTIVE_STATUSES } from "./lib/pipelineStatus";

const POLL_INTERVAL_MS = 5000;
// While nothing is running, or a WebSocket is already delivering the
// transitions, the poll is only a safety net and runs this slowly. At 5 s
// it made 720 requests an hour per idle tab and re-rendered the whole board
// on each, with nothing to show for it.
const POLL_IDLE_INTERVAL_MS = 30000;
const POLL_MAX_INTERVAL_MS = 60000;
const POLL_BACKOFF_FACTOR = 2;

// Shift+click on Run: every gate off. Built from the same key list the
// backend validates against, so a new gate cannot be missed here.
const ALL_GATES_OFF = Object.fromEntries(GATE_ENABLE_KEYS.map((k) => [k, false]));

function App() {
  const [view, setView] = useState("kanban");
  const [sources, setSources] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [modalOpen, setModalOpen] = useState(false);
  const [reviewSource, setReviewSource] = useState(null);
  const [deleteCandidate, setDeleteCandidate] = useState(null);
  const pollTimerRef = useRef(null);
  const failCountRef = useRef(0);
  const fetchRef = useRef(null); // stable ref for fetchSources, used by WS handler
  // true when the poll is a safety net only (see POLL_IDLE_INTERVAL_MS).
  const pollIdleRef = useRef(false);

  /** Fetch sources from API. Updates the backoff counter; the polling
   *  loop reads it to schedule the next tick (see effect below).
   *  Using setTimeout-recursion instead of a replaceable setInterval
   *  avoids races where success and failure paths both schedule new
   *  intervals at the same time, which can spawn parallel pollers. */
  const fetchSources = useCallback(async () => {
    try {
      const data = await listSources({ limit: 200 });
      setSources(data.sources ?? data.items ?? data);
      setError(null);
      failCountRef.current = 0;
    } catch (err) {
      failCountRef.current += 1;
      setError("Failed to load sources. Is the API running?");
      console.error("fetchSources:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  /** Initial fetch + polling. A single timer chain: each tick schedules
   *  the next when it finishes. Backoff (5s -> 10s -> 20s -> 40s -> 60s
   *  cap) reads failCountRef as-of-the-tick. The cleanup clears the
   *  pending timer AND flags the chain stopped so any in-flight fetch
   *  doesn't reschedule after unmount.
   *
   *  "One chain" is enforced by a generation counter, the same guard
   *  useWebSocket uses. The visibility handler restarts the chain; if a
   *  fetch is mid-flight at that moment, clearing the timer ref does
   *  nothing (the timer already fired) and the old tick would reschedule
   *  itself when it resolved — two chains, doubled on every further
   *  flip. Bumping the generation orphans that continuation instead. */
  useEffect(() => {
    let stopped = false;
    let gen = 0;
    fetchRef.current = fetchSources;

    const computeDelay = () => {
      if (failCountRef.current === 0) {
        return pollIdleRef.current ? POLL_IDLE_INTERVAL_MS : POLL_INTERVAL_MS;
      }
      return Math.min(
        POLL_INTERVAL_MS * Math.pow(POLL_BACKOFF_FACTOR, failCountRef.current - 1),
        POLL_MAX_INTERVAL_MS,
      );
    };

    const tick = async () => {
      if (stopped) return;
      const myGen = gen;
      // A hidden tab fetches nothing; visibilitychange below resumes it
      // immediately, this timer is only the fallback.
      if (document.visibilityState === "hidden") {
        pollTimerRef.current = setTimeout(tick, POLL_MAX_INTERVAL_MS);
        return;
      }
      await fetchSources();
      // A newer chain took over while this fetch was in flight: its
      // response is still applied above, but it must not reschedule.
      if (stopped || myGen !== gen) return;
      pollTimerRef.current = setTimeout(tick, computeDelay());
    };

    const onVisibility = () => {
      if (stopped || document.visibilityState !== "visible") return;
      gen += 1;
      if (pollTimerRef.current) clearTimeout(pollTimerRef.current);
      tick();
    };
    document.addEventListener("visibilitychange", onVisibility);

    tick();

    return () => {
      stopped = true;
      document.removeEventListener("visibilitychange", onVisibility);
      if (pollTimerRef.current) {
        clearTimeout(pollTimerRef.current);
        pollTimerRef.current = null;
      }
    };
  }, [fetchSources]);

  // ── WebSocket: real-time updates for active pipelines ──────────
  // Find the source currently in a pipeline-active state.
  // Only return a thread_id once the backend has assigned one — falling
  // back to source.id opens a WS to a URL the server doesn't recognize
  // and burns a backoff cycle before the real thread_id lands.
  const activeThreadId = useMemo(() => {
    const active = sources.find(
      (s) => ACTIVE_STATUSES.has(s.status) && s.thread_id,
    );
    return active?.thread_id ?? null;
  }, [sources]);

  /** Handle real-time WebSocket events from the pipeline. */
  const handleWsEvent = useCallback((event) => {
    if (!event?.data?.status) return;
    const {
      status,
      current_node,
      error: wsError,
      entity_count,
      draft_count,
      objects_written,
      persistence_errors,
      bundle_corrections,
    } = event.data;
    const threadId = event.thread_id;

    // Update the matching source in-place for instant UI feedback. The
    // counts and audit lists are overlaid here so the card is right before
    // the next poll; the Source row carries all of them durably too
    // (queue.update_status writes run_counts), so a page load agrees.
    setSources((prev) =>
      prev.map((s) => {
        const sThread = s.thread_id ?? s.id;
        if (String(sThread) !== String(threadId)) return s;
        return {
          ...s,
          status,
          current_node: current_node ?? s.current_node,
          error: wsError ?? s.error,
          entity_count: typeof entity_count === "number" ? entity_count : s.entity_count,
          draft_count: typeof draft_count === "number" ? draft_count : s.draft_count,
          objects_written:
            typeof objects_written === "number" && objects_written > 0
              ? objects_written
              : s.objects_written,
          persistence_errors: Array.isArray(persistence_errors)
            ? persistence_errors
            : s.persistence_errors,
          bundle_corrections: Array.isArray(bundle_corrections)
            ? bundle_corrections
            : s.bundle_corrections,
        };
      })
    );

    // On terminal events, do a full refresh to pick up final state
    if (event.type === "completed" || event.type === "failed") {
      if (fetchRef.current) fetchRef.current();
    }
  }, []);

  const { connected: wsConnected } = useWebSocket(activeThreadId, handleWsEvent);

  // No source in flight, or a live WebSocket carrying its transitions:
  // the poll drops to the idle cadence. Read by the poll effect through a
  // ref so a change here does not restart the timer chain.
  useEffect(() => {
    pollIdleRef.current = wsConnected || !activeThreadId;
  }, [wsConnected, activeThreadId]);

  /** Handle Kanban drag-and-drop status change. */
  const handleStatusChange = useCallback(
    async (sourceId, newStatus) => {
      // Validate the source still exists locally before issuing the
      // optimistic update. If the analyst's drag landed after a WS
      // delete event reduced the list, the prior optimistic setSources
      // map would silently no-op and the 404 from the API would silently
      // revert via fetchSources — leaving the analyst confused about
      // why the card snapped back.
      setSources((prev) => {
        if (!prev.some((s) => s.id === sourceId)) {
          setError("That source no longer exists. The list has been refreshed.");
          // Defer the refetch so the setError above commits first.
          setTimeout(() => fetchSources(), 0);
          return prev;
        }
        return prev.map((s) => (s.id === sourceId ? { ...s, status: newStatus } : s));
      });
      try {
        await updateSourceStatus(sourceId, newStatus);
      } catch (err) {
        console.error("updateSourceStatus:", err);
        const status = err?.response?.status;
        if (status === 404) {
          setError("That source no longer exists. The list has been refreshed.");
        } else {
          setError(`Failed to update status (${status ?? "network"}). Reverted.`);
        }
        fetchSources();
      }
    },
    [fetchSources]
  );

  /** Handle new source creation. */
  const handleCreateSource = useCallback(
    async (payload) => {
      const created = await createSource(payload);
      // Start pipeline automatically if gates are disabled,
      // or just refresh the list
      fetchSources();
      return created;
    },
    [fetchSources]
  );

  /** Open the gate review slide-over panel for a source at a gate. */
  const handleGateReview = useCallback((source) => {
    setReviewSource(source);
  }, []);

  /** Start a pipeline run for a queued source.
   *  unattended=false (plain click): send NO gates override, so the per-gate
   *  choices made in Add Source govern the run. Sending `true` here used to
   *  expand server-side to "every gate on" and silently overrode them.
   *  unattended=true (Shift+click): an explicit all-off dict — every gate
   *  auto-approves. */
  const handleStartPipeline = useCallback(
    async (source, unattended = false) => {
      // Optimistic: flip the card into "parsing" immediately
      setSources((prev) =>
        prev.map((s) =>
          s.id === source.id ? { ...s, status: "parsing" } : s
        )
      );
      try {
        await startPipeline(source.id, unattended ? ALL_GATES_OFF : null);
        fetchSources();
      } catch (err) {
        console.error("startPipeline:", err);
        setError(
          `Failed to start pipeline for "${source.title}". ${err?.response?.data?.detail ?? err.message}`
        );
        fetchSources(); // revert
      }
    },
    [fetchSources]
  );

  /** Ask for confirmation before deleting a source. */
  const handleDeleteRequest = useCallback((source) => {
    setDeleteCandidate(source);
  }, []);

  /** Confirm delete: optimistic remove + API call + background refresh.
   *  On failure, revert by refetching from the server. */
  const handleDeleteConfirm = useCallback(async () => {
    if (!deleteCandidate) return;
    const target = deleteCandidate;
    // Optimistic: drop the card immediately.
    setSources((prev) => prev.filter((s) => s.id !== target.id));
    try {
      await deleteSource(target.id);
    } catch (err) {
      console.error("deleteSource:", err);
      setError(
        `Failed to delete "${target.title}". ${err?.response?.data?.detail ?? err.message}`
      );
    } finally {
      // Background refresh to catch any drift (e.g. failure revert).
      fetchSources();
    }
  }, [deleteCandidate, fetchSources]);

  /** Called after a gate review is submitted successfully.
   *  nextStatus: optimistic next pipeline status (e.g. "chunking" after gate_0). */
  const handleGateSubmitted = useCallback((result, nextStatus) => {
    // Optimistically flip the card to the next processing status
    if (reviewSource && nextStatus) {
      setSources((prev) =>
        prev.map((s) =>
          s.id === reviewSource.id ? { ...s, status: nextStatus } : s
        )
      );
    }
    // DON'T unmount or refetch here — the panel shows a 1.2s success flash.
    // Panel's onClose() will set reviewSource=null and trigger the refresh.
  }, [reviewSource]);

  return (
    <div className="flex flex-col h-screen bg-gb-bg0-h text-gb-fg1 font-ui">
      {/* Top nav */}
      <nav className="flex items-center justify-between px-6 py-2.5 bg-gb-bg0 border-b border-gb-bg2">
        <h1 className="text-[15px] font-semibold text-gb-fg0 tracking-tight">
          Procedure Extraction Pipeline
        </h1>
        <div className="flex items-center gap-3">
          {/* View switcher */}
          <div className="flex gap-1 bg-gb-bg1 rounded-md p-0.5">
            {/* "Feedback" and "Reviewer" are both about the AI and are easy to
                mix up — one is rules learned from your corrections, the other
                is how often you take the reviewer's advice. They carry a
                hint; the first two say what they are. An ⓘ glyph inside a tab
                button would be a click target that does not switch tabs, so
                these use the button's own tooltip instead. */}
            {[
              { key: "kanban", label: "Source Queue" },
              { key: "explorer", label: "Explorer" },
              { key: "patterns", label: "Feedback", hintTerm: "tab-feedback" },
              { key: "agreement", label: "Reviewer", hintTerm: "tab-reviewer" },
            ].map((v) => (
              <button
                key={v.key}
                onClick={() => setView(v.key)}
                title={v.hintTerm ? hint(v.hintTerm) : undefined}
                className={`px-3 py-1 rounded text-[12px] font-medium transition-colors ${
                  view === v.key
                    ? "bg-gb-bg0-s text-gb-bright-yellow"
                    : "text-gb-fg4 hover:text-gb-fg1"
                }`}
              >
                {v.label}
              </button>
            ))}
          </div>

          {/* Live indicator */}
          {wsConnected && (
            <span className="flex items-center gap-1.5 text-[11px] text-gb-green font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-gb-green animate-pulse" />
              LIVE
            </span>
          )}

          {/* Add source button */}
          {view === "kanban" && (
            <button
              onClick={() => setModalOpen(true)}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[12px] font-semibold bg-gb-green text-gb-bg0-h hover:bg-gb-bright-green transition-colors"
            >
              <span className="text-[14px] leading-none">+</span> Add Source
            </button>
          )}
        </div>
      </nav>

      {/* Stats bar */}
      {view === "kanban" && <StatsBar sources={sources} />}

      {/* Main content */}
      <main className="flex-1 flex flex-col min-h-0">
        {loading && (
          <div className="flex-1 flex items-center justify-center">
            <p className="text-gb-gray text-sm">Loading sources...</p>
          </div>
        )}

        {error && !loading && (
          <div className="mx-4 mt-3 px-4 py-2.5 rounded-lg bg-gb-tag-gate-bg border border-gb-orange text-gb-bright-orange text-[13px]">
            {error}
            <button
              onClick={fetchSources}
              className="ml-3 underline hover:text-gb-bright-yellow"
            >
              Retry
            </button>
          </div>
        )}

        {!loading && view === "kanban" && (
          <KanbanBoard
            sources={sources}
            onStatusChange={handleStatusChange}
            onGateReview={handleGateReview}
            onStartPipeline={handleStartPipeline}
            onDelete={handleDeleteRequest}
          />
        )}

        {!loading && view === "explorer" && <ExplorerView />}

        {!loading && view === "patterns" && <FeedbackPatternsView />}

        {!loading && view === "agreement" && <ReviewerAgreementView />}
      </main>

      {/* Add source modal */}
      <AddSourceModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        onSubmit={handleCreateSource}
      />

      {/* Delete confirmation */}
      <ConfirmModal
        isOpen={deleteCandidate !== null}
        title="Delete source?"
        message={
          deleteCandidate
            ? `"${deleteCandidate.title}" will be removed from the queue. If it was uploaded through this UI, the file will be deleted too. This cannot be undone.`
            : ""
        }
        confirmLabel="Delete"
        danger
        onConfirm={handleDeleteConfirm}
        onClose={() => setDeleteCandidate(null)}
      />

      {/* Gate review slide-over panel */}
      {reviewSource && (
        <GateReviewPanel
          source={reviewSource}
          onClose={() => setReviewSource(null)}
          onSubmitted={handleGateSubmitted}
        />
      )}
    </div>
  );
}

export default App;
