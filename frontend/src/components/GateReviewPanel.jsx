/**
 * GateReviewPanel — slide-over panel for analyst gate reviews.
 *
 * Slides in from the right, dims the Kanban behind it.
 * Routes to Gate0Review, ChunkReviewCanvas (gate_chunks), Gate1Review, or
 * BundleReviewCanvas (gate_2) based on source status.
 * Manages the fetch/submit lifecycle for all four gates.
 */
import { useState, useEffect, useCallback, useRef } from "react";
import {
  fetchPendingReview, fetchPendingChunkReview,
  submitGate0, submitGate1, submitGate2, submitChunkReview,
} from "../api/gates";
import { STATUS_TO_GATE, GATE_TITLES, GATE_NEXT_STATUS, GATES } from "../lib/gates";
import { fetchBrief, fetchRecommendations, updateBrief } from "../api/reviewer";
import Gate0Review from "./Gate0Review";
import Gate1Review from "./Gate1Review";
import BundleReviewCanvas from "./BundleReviewCanvas";
import ChunkReviewCanvas from "./ChunkReviewCanvas";

// Flash copy per rerun target — must match what the backend actually loops
// back to ("Rerunning chunking" on a techniques re-map was lying).
const RERUN_FLASH = {
  chunking: {
    title: "Rerunning chunking",
    sub: "Re-chunking with your feedback...",
  },
  extracting_techniques: {
    title: "Re-mapping techniques",
    sub: "Re-extracting techniques with your feedback...",
  },
};

export default function GateReviewPanel({ source, onClose, onSubmitted }) {
  const [payload, setPayload] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState(false);
  const panelRef = useRef(null);

  const gateId = STATUS_TO_GATE[source?.status];
  const threadId = source?.thread_id;

  /** Fetch pending review data when panel opens. */
  useEffect(() => {
    if (threadId == null || gateId == null) return;
    let canceled = false;

    setLoading(true);
    setError(null);
    const fetcher = gateId === "chunks"
      ? fetchPendingChunkReview(threadId)
      : fetchPendingReview(threadId, gateId);
    fetcher
      .then((data) => {
        if (!canceled) setPayload(data);
      })
      .catch((err) => {
        if (!canceled) {
          const detail = err?.response?.data?.detail;
          setError(typeof detail === "string" ? detail : String(detail ?? "Failed to load review data."));
        }
      })
      .finally(() => {
        if (!canceled) setLoading(false);
      });

    return () => { canceled = true; };
  }, [threadId, gateId]);

  /** Fetch AI reviewer recommendations alongside the gate payload.
   *
   * Separate effect, and deliberately non-blocking: a 404 (this gate had no
   * AI review) or an outright failure must leave the gate fully usable. The
   * reviewer is an assistant, not a dependency.
   */
  useEffect(() => {
    const sourceId = source?.id;
    const gate = GATES.find((g) => g.routeId === gateId);
    if (!sourceId || !gate?.aiReviewer) {
      setRecommendations(null);
      setBrief(null);
      return;
    }
    let canceled = false;
    Promise.all([
      fetchRecommendations(sourceId, gate.enableKey).catch(() => null),
      fetchBrief(sourceId).catch(() => null),
    ]).then(([recs, briefRow]) => {
      if (canceled) return;
      setRecommendations(recs?.payload ?? null);
      setBrief(briefRow);
    });
    return () => { canceled = true; };
  }, [source?.id, gateId]);

  /** Save a corrected brief, then pull the re-run recommendations. */
  const handleSaveBrief = useCallback(async (corrected) => {
    const sourceId = source?.id;
    const gate = GATES.find((g) => g.routeId === gateId);
    if (!sourceId || !gate?.aiReviewer) return;
    setSavingBrief(true);
    try {
      const updated = await updateBrief(sourceId, corrected);
      setBrief(updated);
      // The backend re-reviews the current gate with the corrected read, so
      // refetch rather than leave pre-correction advice on screen.
      const recs = await fetchRecommendations(sourceId, gate.enableKey).catch(() => null);
      setRecommendations(recs?.payload ?? null);
    } finally {
      setSavingBrief(false);
    }
  }, [source?.id, gateId]);

  /** Close on Escape key — but not when the analyst is inside a text field
   *  (NodeSearchBox clears on Escape, textareas expect it) or a modal is
   *  open. Closing the panel unmounts every unsaved decision, so an Escape
   *  meant for a search box must never reach here. */
  useEffect(() => {
    const handleKey = (e) => {
      if (e.key !== "Escape") return;
      const t = e.target;
      if (t?.closest?.('input, textarea, select, [contenteditable="true"], [role="dialog"]')) return;
      onClose();
    };
    document.addEventListener("keydown", handleKey);
    return () => document.removeEventListener("keydown", handleKey);
  }, [onClose]);

  // Which pipeline stage this submission loops back to (null = no rerun,
  // it advances). Drives both the optimistic Kanban status and the flash
  // copy — a Gate 1 re-map loops to extract_techniques, not chunking, and
  // the card/flash should say so.
  const [rerunTarget, setRerunTarget] = useState(null);

  // AI reviewer state. All null when the gate ran in plain review mode —
  // the child components then render exactly as they always have.
  const [recommendations, setRecommendations] = useState(null);
  const [brief, setBrief] = useState(null);
  const [savingBrief, setSavingBrief] = useState(false);

  /** Submit handler — routes to the correct gate endpoint.
   *
   *  The submit endpoints return 202 quickly after writing analyst decisions
   *  into LangGraph state and scheduling the pipeline as a background task,
   *  so we can await the request without holding the UI. On 4xx/5xx the
   *  panel stays open with the error surfaced AND the gate child stays
   *  mounted — its local state holds every unsaved decision, so unmounting
   *  it on error (which the render conditions once did) threw the work away
   *  and then invited a retry of nothing.
   *
   *  Gate components may pass a second arg { willRerun, rerunTarget } to
   *  signal that the submission loops back and to which stage. rerunTarget
   *  defaults to "chunking" (correct for the chunks gate's reject, the only
   *  caller that predates the field). */
  const handleSubmit = useCallback(
    async (reviewData, opts) => {
      if (!threadId) return;
      setSubmitting(true);
      setError(null);

      const rerun = opts?.willRerun ?? false;
      const target = rerun ? (opts?.rerunTarget ?? "chunking") : null;
      setRerunTarget(target);

      // Pick the right submit function. The chunks gate uses a string sentinel
      // ("chunks") and a separate endpoint; falling through to submitGate2 was
      // an early-3d bug that mis-routed chunk-review submissions.
      const submitFn =
        gateId === "chunks" ? submitChunkReview
        : gateId === 0 ? submitGate0
        : gateId === 1 ? submitGate1
        : submitGate2;

      // Attach the checkpoint_id we received at GET time so the backend can
      // reject stale submits (e.g. a double-click or two analysts reviewing
      // the same source). Gate 0 still emits a bare reviews array from its
      // child component; everything else emits an object — handle both.
      const checkpointId = payload?.checkpoint_id ?? null;
      const body = Array.isArray(reviewData)
        ? { reviews: reviewData, checkpoint_id: checkpointId }
        : { ...reviewData, checkpoint_id: checkpointId };

      try {
        await submitFn(threadId, body);
      } catch (err) {
        const status = err?.response?.status;
        const detail = err?.response?.data?.detail;
        const msg = typeof detail === "string"
          ? detail
          : (detail ? JSON.stringify(detail) : err?.message || "Submit failed.");
        const prefix = status ? `Submit failed (${status}): ` : "Submit failed: ";
        setError(prefix + msg);
        setSubmitting(false);
        setRerunTarget(null);
        return;
      }

      // Success: flash + update Kanban card. On a rerun the card moves to
      // the stage the pipeline actually loops back to.
      const nextStatus = rerun ? target : GATE_NEXT_STATUS[gateId];
      setSubmitting(false);
      setSubmitSuccess(true);
      onSubmitted?.(null, nextStatus);
      setTimeout(() => {
        onClose();
      }, 1200);
    },
    // payload is read for checkpoint_id: leaving it out of the deps kept a
    // callback built while payload was still null, which sent
    // checkpoint_id: null and disabled the backend's stale-submit guard.
    [threadId, gateId, payload, onSubmitted, onClose]
  );

  if (!source) return null;

  return (
    <>
      {/* Backdrop: dims the Kanban */}
      <div
        className="fixed inset-0 bg-black/50 z-40 transition-opacity"
        onClick={onClose}
      />

      {/* Slide-over panel */}
      <div
        ref={panelRef}
        className={`fixed top-0 right-0 h-full max-w-[95vw] z-50 flex flex-col bg-gb-bg0-h border-l border-gb-bg2 shadow-2xl animate-slide-in ${
          gateId === "chunks" || gateId === 2 ? "w-[1400px]" : "w-[680px]"
        }`}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-3 border-b border-gb-bg2 bg-gb-bg0">
          <div>
            <h2 className="text-[15px] font-semibold text-gb-fg0">
              {GATE_TITLES[gateId] ?? "Gate Review"}
            </h2>
            <p className="text-[12px] text-gb-fg4 mt-0.5 font-data">
              {source.title}
            </p>
          </div>
          <button
            onClick={onClose}
            className="w-7 h-7 flex items-center justify-center rounded-md text-gb-fg4 hover:text-gb-fg1 hover:bg-gb-bg1 transition-colors text-[16px]"
            aria-label="Close panel"
          >
            ✕
          </button>
        </div>

        {/* Body */}
        <div className="flex-1 overflow-y-auto px-5 py-4 min-h-0">
          {/* Success flash overlay */}
          {submitSuccess && (
            <div className="flex flex-col items-center justify-center py-16 animate-fade-in">
              <div className={`w-12 h-12 rounded-full border-2 flex items-center justify-center mb-3 ${
                rerunTarget
                  ? "bg-gb-orange/20 border-gb-bright-orange"
                  : "bg-gb-green/20 border-gb-bright-green"
              }`}>
                <span className={`text-[20px] ${
                  rerunTarget ? "text-gb-bright-orange" : "text-gb-bright-green"
                }`}>{rerunTarget ? "↻" : "✓"}</span>
              </div>
              <p className={`text-[14px] font-semibold ${
                rerunTarget ? "text-gb-bright-orange" : "text-gb-bright-green"
              }`}>
                {rerunTarget
                  ? (RERUN_FLASH[rerunTarget]?.title ?? "Rerunning earlier stage")
                  : "Review submitted"}
              </p>
              <p className="text-gb-fg4 text-[12px] mt-1">
                {rerunTarget
                  ? (RERUN_FLASH[rerunTarget]?.sub ?? "Looping back with your feedback...")
                  : "Pipeline resuming..."}
              </p>
            </div>
          )}

          {!submitSuccess && loading && (
            <div className="flex items-center justify-center py-12">
              <p className="text-gb-gray text-sm">Loading review data...</p>
            </div>
          )}

          {!submitSuccess && error && (
            <div className="mb-4 px-4 py-2.5 rounded-lg bg-gb-tag-gate-bg border border-gb-orange text-gb-bright-orange text-[13px]">
              {error}
            </div>
          )}

          {!submitSuccess && !loading && payload && gateId === 0 && (
            <Gate0Review
              payload={payload}
              onSubmit={handleSubmit}
              submitting={submitting}
              recommendations={recommendations}
              brief={brief}
              onSaveBrief={handleSaveBrief}
              savingBrief={savingBrief}
            />
          )}

          {!submitSuccess && !loading && payload && gateId === 1 && (
            <Gate1Review
              payload={payload}
              onSubmit={handleSubmit}
              submitting={submitting}
              recommendations={recommendations}
              brief={brief}
              onSaveBrief={handleSaveBrief}
              savingBrief={savingBrief}
            />
          )}

          {!submitSuccess && !loading && payload && gateId === 2 && (
            <BundleReviewCanvas
              payload={payload}
              onSubmit={handleSubmit}
              submitting={submitting}
              recommendations={recommendations}
              brief={brief}
              onSaveBrief={handleSaveBrief}
              savingBrief={savingBrief}
            />
          )}

          {!submitSuccess && !loading && payload && gateId === "chunks" && (
            <ChunkReviewCanvas
              payload={payload}
              onSubmit={handleSubmit}
              submitting={submitting}
              recommendations={recommendations}
              brief={brief}
              onSaveBrief={handleSaveBrief}
              savingBrief={savingBrief}
            />
          )}
        </div>
      </div>
    </>
  );
}
