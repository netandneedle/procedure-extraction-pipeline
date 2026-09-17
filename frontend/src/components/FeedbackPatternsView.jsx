/**
 * FeedbackPatternsView — analyst surface over the feedback flywheel.
 *
 * Lists the learned feedback patterns most-useful-first (by salience), with
 * filters and per-row actions: promote a pattern to a permanent guardrail
 * (prompt | denylist), dismiss a noisy one, or edit its text/category.
 * Hit/miss/salience reflect the closed loop (whether surfacing the pattern
 * actually reduced re-corrections).
 */
import { useState, useEffect, useCallback, useRef } from "react";
import {
  listFeedbackPatterns,
  listCapturedCorrections,
  promotePattern,
  dismissPattern,
  editPattern,
} from "../api/feedbackPatterns";
import usePolling from "../hooks/usePolling";
import InfoDot from "./InfoDot";
import { AREA_TO_GATE_STATUS, GATE_STATUS_COLORS } from "../lib/gates";

// Gate area -> short noun for captured-correction chips. The accent color is
// derived from lib/gates.js (the single source of truth for gate palette) so
// the entity gate reads yellow here exactly as it does on every other
// surface — this previously hardcoded blue.
// Statuses whose corrections can never synthesize. Mirrors
// _TERMINAL_CAPTURE_STATUSES in api/routes/feedback_patterns.py: a hard-fail
// routes to END, skipping distribute AND synthesize_feedback, so the
// checkpoint is frozen with its corrections still in it.
const TERMINAL_CAPTURE_STATUSES = new Set(["failed"]);

const GATE_LABEL = {
  entities: "entity", chunks: "chunk", techniques: "technique", relationships: "relationship",
};
const GATE_TEXT = Object.fromEntries(
  Object.entries(AREA_TO_GATE_STATUS).map(([area, status]) => [
    area, GATE_STATUS_COLORS[status],
  ])
);

// Taxonomy mirrors FeedbackCategory in backend tool_models.py.
const CATEGORIES = [
  "defender_ioc", "brand_as_malware", "false_positive_entity", "mis_attribution",
  "over_chunked", "under_chunked", "missing_procedure", "wrong_predecessor",
  "parallel_capability_misordered", "artifact_loss",
  "missing_tactic", "wrong_technique", "thin_initial_access", "orphan_ioc",
  "wrong_relationship", "other",
];

const STATUSES = [
  "active", "promoted_to_prompt", "promoted_to_denylist",
  "dismissed", "archived", "pending",
];

// Pattern category -> gate area, so a synthesized pattern takes its accent
// from GATE_TEXT like the captured corrections above it. A second
// hand-kept palette here had already drifted: entity corrections rendered
// yellow on one row and the entity pattern beneath them blue. Mirrors
// _AREA_FOR_CATEGORY in the backend's feedback synthesis.
const CATEGORY_AREA = {
  defender_ioc: "entities", brand_as_malware: "entities",
  false_positive_entity: "entities", mis_attribution: "entities",
  over_chunked: "chunks", under_chunked: "chunks", missing_procedure: "chunks",
  wrong_predecessor: "chunks", parallel_capability_misordered: "chunks", artifact_loss: "chunks",
  missing_tactic: "techniques", wrong_technique: "techniques",
  thin_initial_access: "techniques", orphan_ioc: "techniques",
  wrong_relationship: "relationships",
};

const STATUS_TEXT = {
  active: "text-gb-bright-green",
  promoted_to_prompt: "text-gb-bright-blue",
  promoted_to_denylist: "text-gb-bright-purple",
  dismissed: "text-gb-fg4",
  archived: "text-gb-fg4",
  pending: "text-gb-bright-yellow",
};

const SALIENCE_FULL = 2.0; // salience value rendered as a full bar

// Split a textarea of comma/newline-separated terms into a clean list.
const parseTerms = (s) => s.split(/[\n,]/).map((x) => x.trim()).filter(Boolean);

// Evidence keys whose string values are likely literal IOC-ish entity values
// worth pre-filling into the denylist (vs. prose like rationale text).
// Deliberately NOT 'name'/'host'/'entity' — those catch ubiquitous legitimate
// strings (actor/tool/malware names like "PowerShell", "windows", or
// entity_type values), which would pre-fill a one-click block of a benign,
// high-traffic value. The analyst can still type such a term manually.
const VALUE_KEYS = /(^|_)(value|ioc|indicator|email|domain|address|url|hash)(_|$)/i;

function collectValueCandidates(obj, out, keyHint = false, depth = 0) {
  if (depth > 4 || obj == null) return;
  if (typeof obj === "string") {
    const s = obj.trim();
    if (keyHint && s && s.length <= 120) out.add(s);
    return;
  }
  if (Array.isArray(obj)) {
    obj.forEach((v) => collectValueCandidates(v, out, keyHint, depth + 1));
    return;
  }
  if (typeof obj === "object") {
    for (const [k, v] of Object.entries(obj)) {
      collectValueCandidates(v, out, keyHint || VALUE_KEYS.test(k), depth + 1);
    }
  }
}

// Pre-fill suggestions for a denylist promotion: existing terms win (re-promote),
// else literal values from evidence + technique IDs from applies_to.
function suggestDenylistTerms(p) {
  const existing = p.denylist_terms || {};
  if (
    (existing.values || []).length ||
    (existing.technique_ids || []).length ||
    (existing.entity_types || []).length
  ) {
    return {
      values: existing.values || [],
      technique_ids: existing.technique_ids || [],
      entity_types: existing.entity_types || [],
    };
  }
  const vals = new Set();
  collectValueCandidates(p.evidence || {}, vals);
  return {
    values: Array.from(vals),
    technique_ids: p.applies_to?.technique_ids || [],
    entity_types: p.applies_to?.entity_types || [],
  };
}

function SalienceBar({ value }) {
  if (value == null) return <span className="text-gb-fg4 font-data text-[11px]">—</span>;
  const pct = Math.max(0, Math.min(100, (value / SALIENCE_FULL) * 100));
  return (
    <div className="flex items-center gap-1.5">
      <div className="w-16 h-1.5 rounded-full bg-gb-bg2 overflow-hidden">
        <div className="h-full bg-gb-bright-green" style={{ width: `${pct}%` }} />
      </div>
      <span className="font-data text-[11px] text-gb-fg4">{value.toFixed(2)}</span>
    </div>
  );
}

export default function FeedbackPatternsView() {
  const [patterns, setPatterns] = useState([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState(null);
  // Corrections read out of a source's checkpoint, not yet generalized into
  // patterns. Mixes still-running sources with failed ones — see the panel.
  const [captured, setCaptured] = useState([]);
  const [capturedTotal, setCapturedTotal] = useState(0);
  // Collapsed by default: this list runs to dozens of full-sentence rationales
  // and expanded it buries the synthesized patterns underneath it.
  const [capturedOpen, setCapturedOpen] = useState(false);
  const [category, setCategory] = useState("");
  const [status, setStatus] = useState("");
  const [minSalience, setMinSalience] = useState(0);
  const [search, setSearch] = useState("");
  const [busyId, setBusyId] = useState(null);
  const [editId, setEditId] = useState(null);
  const [editText, setEditText] = useState("");
  const [editCategory, setEditCategory] = useState("");
  // Denylist confirm/edit form (one open at a time), pre-filled from the pattern.
  const [denylistId, setDenylistId] = useState(null);
  const [denylistValues, setDenylistValues] = useState("");
  const [denylistTids, setDenylistTids] = useState("");
  const [denylistTypes, setDenylistTypes] = useState("");

  // Monotonic fetch sequence: only the NEWEST in-flight patterns request may
  // write state. Without this, a slow silent poll built from an older filter
  // closure could resolve after a fresh filtered fetch and overwrite the
  // list with stale (previous-filter) results.
  const fetchSeq = useRef(0);

  // `silent` skips the loading/error chrome so background polls don't flash
  // the spinner or clobber a real error toast.
  const fetchPatterns = useCallback(async ({ silent = false } = {}) => {
    const seq = ++fetchSeq.current;
    if (!silent) setLoading(true);
    // Guarded: a silent poll must not clear a visible error banner either —
    // promote/dismiss/denylist failures share this err state, and the open
    // denylist form relies on the banner to explain a failed submit.
    if (!silent) setErr(null);
    try {
      const { patterns, total } = await listFeedbackPatterns({
        category: category || null,
        status: status || null,
        minSalience: minSalience > 0 ? minSalience : null,
        search: search || null,
        limit: 100,
      });
      if (seq !== fetchSeq.current) return; // superseded by a newer fetch
      setPatterns(patterns);
      setTotal(total);
    } catch (e) {
      if (!silent) setErr(e.message || String(e));
    } finally {
      if (!silent) setLoading(false);
    }
  }, [category, status, minSalience, search]);

  // Debounce all filter changes (incl. search keystrokes) uniformly.
  useEffect(() => {
    const t = setTimeout(() => fetchPatterns(), 300);
    return () => clearTimeout(t);
  }, [fetchPatterns]);

  // Poll BOTH captured corrections and synthesized patterns on a light
  // interval. Polling patterns too is the load-bearing part: when a run the
  // analyst is watching completes, its captured corrections vanish (they're
  // in-flight only) AND new patterns get synthesized — without this the list
  // would stay stale at its mount-time snapshot and the feedback would look
  // like it disappeared. Silent so it doesn't flicker the spinner.
  // usePolling reads the latest fetchPatterns through a ref, so filter
  // keystrokes (which change its identity) no longer restart the timer or
  // fire extra /captured requests.
  usePolling(async (isCancelled) => {
    try {
      const { sources, total } = await listCapturedCorrections();
      if (!isCancelled()) {
        setCaptured(sources);
        setCapturedTotal(total);
      }
    } catch {
      // Best-effort sidebar — don't surface its errors over the patterns list.
    }
    if (!isCancelled()) fetchPatterns({ silent: true });
  }, 8000);

  const applyUpdate = (updated) =>
    setPatterns((ps) => ps.map((p) => (p.id === updated.id ? updated : p)));

  async function withBusy(id, fn) {
    setBusyId(id);
    setErr(null);
    try {
      applyUpdate(await fn());
    } catch (e) {
      setErr(e.message || String(e));
    } finally {
      setBusyId(null);
    }
  }

  const doPromote = (id, action) => withBusy(id, () => promotePattern(id, action, "analyst"));
  const doDismiss = (id) => withBusy(id, () => dismissPattern(id));

  function startDenylist(p) {
    const sug = suggestDenylistTerms(p);
    setDenylistId(p.id);
    setDenylistValues(sug.values.join("\n"));
    setDenylistTids(sug.technique_ids.join("\n"));
    setDenylistTypes((sug.entity_types || []).join("\n"));
    setEditId(null);
  }
  async function confirmDenylist(id) {
    const terms = {
      values: parseTerms(denylistValues),
      technique_ids: parseTerms(denylistTids),
      entity_types: parseTerms(denylistTypes),
    };
    // Keep the form open on failure (withBusy surfaces err) so the analyst
    // doesn't lose what they typed; only close on success.
    setErr(null);
    setBusyId(id);
    try {
      applyUpdate(await promotePattern(id, "denylist", "analyst", terms));
      setDenylistId(null);
    } catch (e) {
      setErr(e.message || String(e));
    } finally {
      setBusyId(null);
    }
  }

  function startEdit(p) {
    setEditId(p.id);
    setEditText(p.pattern);
    setEditCategory(p.category);
  }
  async function saveEdit(id) {
    await withBusy(id, () => editPattern(id, { pattern: editText, category: editCategory }));
    setEditId(null);
  }

  const capturedStranded = captured.filter((s) =>
    TERMINAL_CAPTURE_STATUSES.has(s.status)
  );
  const capturedInFlight = captured.filter(
    (s) => !TERMINAL_CAPTURE_STATUSES.has(s.status)
  );
  const countOf = (group) =>
    group.reduce((n, s) => n + (s.corrections?.length || 0), 0);
  const capturedStrandedTotal = countOf(capturedStranded);
  const capturedInFlightTotal = countOf(capturedInFlight);

  return (
    <div className="flex flex-col min-h-0 flex-1">
      {/* Filter bar */}
      <div className="flex flex-wrap items-center gap-3 px-6 py-2.5 bg-gb-bg0 border-b border-gb-bg2 text-[12px]">
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search pattern text…"
          className="px-2.5 py-1 rounded bg-gb-bg1 border border-gb-bg2 text-gb-fg1 placeholder-gb-fg4 focus:outline-none focus:border-gb-bright-blue w-64"
        />
        <select
          value={category}
          onChange={(e) => setCategory(e.target.value)}
          className="px-2 py-1 rounded bg-gb-bg1 border border-gb-bg2 text-gb-fg1 focus:outline-none"
        >
          <option value="">All categories</option>
          {CATEGORIES.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
        <select
          value={status}
          onChange={(e) => setStatus(e.target.value)}
          className="px-2 py-1 rounded bg-gb-bg1 border border-gb-bg2 text-gb-fg1 focus:outline-none"
        >
          <option value="">All statuses</option>
          {STATUSES.map((s) => <option key={s} value={s}>{s}</option>)}
        </select>
        <label className="flex items-center gap-1.5 text-gb-gray">
          min salience
          <InfoDot term="salience" />
          <input
            type="range" min="0" max="2" step="0.05"
            value={minSalience}
            onChange={(e) => setMinSalience(parseFloat(e.target.value))}
            className="w-24"
          />
          <span className="font-data text-gb-fg4 w-8">{minSalience.toFixed(2)}</span>
        </label>
        <span className="ml-auto text-gb-gray">
          {loading ? "loading…" : `${patterns.length} of ${total}`}
        </span>
      </div>

      {err && (
        <div className="mx-6 mt-3 px-4 py-2 rounded-lg bg-gb-tag-gate-bg border border-gb-orange text-gb-bright-orange text-[12px]">
          {err}
        </div>
      )}

      {/* Captured corrections — raw gate decisions read out of a source's
          checkpoint, before the synthesizer generalizes them into patterns.

          TWO populations live here, and conflating them is what made this
          panel lie. A RUNNING source's corrections really do become patterns
          when it finishes. A FAILED source's never will: a hard-fail at
          validate_bundle routes straight to END, skipping distribute AND
          synthesize_feedback, so the corrections sit in a frozen checkpoint
          forever. The backend surfaces failed sources deliberately — see
          _TERMINAL_CAPTURE_STATUSES, "that visibility is the feature" — but
          the header used to promise both groups they were queued for
          processing.

          Collapsed by default. Eighty corrections at full length pushed the
          synthesized patterns — the actual subject of this tab — entirely
          off-screen. */}
      {capturedTotal > 0 && (
        <div className="mx-6 mt-3 rounded-lg bg-gb-bg0-s border border-dashed border-gb-bright-yellow/50">
          <button
            type="button"
            onClick={() => setCapturedOpen((v) => !v)}
            className="w-full flex items-center gap-2 px-4 py-2 text-left hover:bg-gb-bg1/40 rounded-t-lg"
            aria-expanded={capturedOpen}
          >
            <span className="font-data text-[10px] text-gb-fg4 w-3 shrink-0">
              {capturedOpen ? "▾" : "▸"}
            </span>
            <span className="font-data text-[11px] font-semibold text-gb-bright-yellow uppercase tracking-wide">
              Captured corrections ({capturedTotal})
            </span>
            <span className="text-[11px] text-gb-fg4 truncate">
              {capturedInFlight.length > 0 && (
                <>{capturedInFlightTotal} awaiting synthesis</>
              )}
              {capturedInFlight.length > 0 && capturedStranded.length > 0 && " · "}
              {capturedStranded.length > 0 && (
                <span className="text-gb-bright-orange">
                  {capturedStrandedTotal} from failed run
                  {capturedStranded.length === 1 ? "" : "s"} — never synthesized
                </span>
              )}
            </span>
          </button>

          {capturedOpen && (
            <div className="px-4 py-2 space-y-3 border-t border-gb-bg2 max-h-[45vh] overflow-auto">
              {[
                {
                  group: capturedInFlight,
                  note: "still running — these become patterns below when the run completes",
                  tone: "text-gb-fg4",
                },
                {
                  group: capturedStranded,
                  note:
                    "the run failed before the synthesizer ran, so these never " +
                    "became patterns and will not unless the source is re-run",
                  tone: "text-gb-bright-orange",
                },
              ].map(({ group, note, tone }, gi) =>
                group.length === 0 ? null : (
                  <div key={gi} className="space-y-2.5">
                    <p className={`text-[11px] ${tone}`}>{note}</p>
                    {group.map((src) => (
                      <div key={src.source_id}>
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-[12px] text-gb-fg1 font-medium truncate">{src.title}</span>
                          <span className="font-data text-[10px] text-gb-fg4 uppercase tracking-wide">{src.status}</span>
                        </div>
                        <div className="space-y-1 pl-2 border-l border-gb-bg2">
                          {src.corrections.map((c, i) => (
                            <div key={i} className="flex items-baseline gap-2 text-[12px]">
                              <span className={`font-data text-[10px] uppercase tracking-wide shrink-0 ${GATE_TEXT[c.gate] || "text-gb-fg4"}`}>
                                {GATE_LABEL[c.gate] || c.gate}
                              </span>
                              <span className="text-gb-fg2">
                                {c.summary}
                                {c.detail && <span className="text-gb-fg4"> — {c.detail}</span>}
                              </span>
                            </div>
                          ))}
                        </div>
                      </div>
                    ))}
                  </div>
                )
              )}
            </div>
          )}
        </div>
      )}

      {/* List */}
      <div className="flex-1 overflow-auto px-6 py-3 space-y-2">
        {!loading && patterns.length === 0 && (
          <p className="text-gb-gray text-sm py-8 text-center max-w-2xl mx-auto">
            No synthesized patterns yet. Patterns are the <em>learned rules</em> —
            the synthesizer writes them only when a run <strong>completes</strong>.
            {capturedInFlightTotal > 0
              ? " Your in-flight corrections are recorded above and will become patterns here once the run finishes."
              : capturedStrandedTotal > 0
              ? " The corrections above came from runs that failed before the synthesizer ran, so they never became patterns."
              : " Corrections you make at a gate are recorded immediately (and show above) once a run is in progress."}
          </p>
        )}

        {patterns.map((p) => (
          <div
            key={p.id}
            className="rounded-lg bg-gb-bg0-s border border-gb-bg2 px-4 py-3 text-[13px]"
          >
            <div className="flex items-start gap-3">
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1 flex-wrap">
                  <span className={`font-data text-[11px] font-semibold ${GATE_TEXT[CATEGORY_AREA[p.category]] || "text-gb-fg4"}`}>
                    {p.category}
                  </span>
                  <span className={`font-data text-[10px] uppercase tracking-wide ${STATUS_TEXT[p.status] || "text-gb-fg4"}`}>
                    {p.status}
                  </span>
                  <span className="font-data text-[10px] text-gb-fg4">
                    seen {p.occurrence_count}× · hit {p.hit_count} · miss {p.miss_count}
                  </span>
                  <SalienceBar value={p.salience} />
                </div>

                {editId === p.id ? (
                  <div className="space-y-1.5 mt-1">
                    <textarea
                      value={editText}
                      onChange={(e) => setEditText(e.target.value)}
                      rows={2}
                      className="w-full px-2 py-1 rounded bg-gb-bg1 border border-gb-bg2 text-gb-fg1 text-[12px] focus:outline-none focus:border-gb-bright-blue"
                    />
                    <select
                      value={editCategory}
                      onChange={(e) => setEditCategory(e.target.value)}
                      className="px-2 py-1 rounded bg-gb-bg1 border border-gb-bg2 text-gb-fg1 text-[12px]"
                    >
                      {CATEGORIES.map((c) => <option key={c} value={c}>{c}</option>)}
                    </select>
                  </div>
                ) : denylistId === p.id ? (
                  <div className="space-y-2 mt-1">
                    <p className="text-gb-fg1 leading-snug">{p.pattern}</p>
                    <div className="rounded-md bg-gb-bg1 border border-gb-bright-purple/40 px-2.5 py-2 space-y-2">
                      <p className="text-[11px] text-gb-bright-purple font-medium">
                        Promote to denylist — confirm what gets blocked deterministically
                      </p>
                      <div>
                        <label className="block text-[10px] uppercase tracking-wide text-gb-fg4 mb-0.5">
                          Entity values (one per line — case-insensitive exact match, auto-removed at Gate 0)
                        </label>
                        <textarea
                          value={denylistValues}
                          onChange={(e) => setDenylistValues(e.target.value)}
                          rows={2}
                          placeholder="e.g. info@cert.example"
                          className="w-full px-2 py-1 rounded bg-gb-bg0 border border-gb-bg2 text-gb-fg1 font-data text-[11px] focus:outline-none focus:border-gb-bright-purple"
                        />
                      </div>
                      <div>
                        <label className="block text-[10px] uppercase tracking-wide text-gb-fg4 mb-0.5">
                          Technique IDs (one per line — dropped from picks, e.g. T1204.003)
                        </label>
                        <textarea
                          value={denylistTids}
                          onChange={(e) => setDenylistTids(e.target.value)}
                          rows={2}
                          placeholder="e.g. T1204.003"
                          className="w-full px-2 py-1 rounded bg-gb-bg0 border border-gb-bg2 text-gb-fg1 font-data text-[11px] focus:outline-none focus:border-gb-bright-purple"
                        />
                      </div>
                      <div>
                        <label className="block text-[10px] uppercase tracking-wide text-gb-fg4 mb-0.5">
                          Limit to entity types (optional — one per line; blank = any type, e.g. organization)
                        </label>
                        <textarea
                          value={denylistTypes}
                          onChange={(e) => setDenylistTypes(e.target.value)}
                          rows={1}
                          placeholder="e.g. organization"
                          className="w-full px-2 py-1 rounded bg-gb-bg0 border border-gb-bg2 text-gb-fg1 font-data text-[11px] focus:outline-none focus:border-gb-bright-purple"
                        />
                      </div>
                      {!parseTerms(denylistValues).length && !parseTerms(denylistTids).length && (
                        <p className="text-[10px] text-gb-bright-yellow">
                          No terms — promoting now keeps the pattern advisory only (nothing enforced).
                        </p>
                      )}
                    </div>
                  </div>
                ) : (
                  <p className="text-gb-fg1 leading-snug">{p.pattern}</p>
                )}

                {(p.applies_to?.technique_ids?.length ||
                  p.applies_to?.tactics?.length ||
                  p.concepts?.length) ? (
                  <div className="flex flex-wrap gap-1.5 mt-1.5">
                    {(p.applies_to?.technique_ids || []).map((t) => (
                      <span key={t} className="font-data text-[10px] px-1.5 py-0.5 rounded bg-gb-bg1 text-gb-bright-aqua">{t}</span>
                    ))}
                    {(p.applies_to?.tactics || []).map((t) => (
                      <span key={t} className="font-data text-[10px] px-1.5 py-0.5 rounded bg-gb-bg1 text-gb-fg4">{t}</span>
                    ))}
                    {(p.concepts || []).map((c) => (
                      <span key={c} className="font-data text-[10px] px-1.5 py-0.5 rounded bg-gb-bg1 text-gb-gray">#{c}</span>
                    ))}
                  </div>
                ) : null}
              </div>

              {/* Actions */}
              <div className="flex flex-col gap-1 shrink-0">
                {editId === p.id ? (
                  <>
                    <button
                      onClick={() => saveEdit(p.id)}
                      disabled={busyId === p.id || !editText.trim()}
                      className="px-2 py-1 rounded text-[11px] font-medium bg-gb-green text-gb-bg0-h hover:bg-gb-bright-green disabled:opacity-40"
                    >
                      Save
                    </button>
                    <button
                      onClick={() => setEditId(null)}
                      className="px-2 py-1 rounded text-[11px] text-gb-fg4 hover:text-gb-fg1"
                    >
                      Cancel
                    </button>
                  </>
                ) : denylistId === p.id ? (
                  <>
                    <button
                      onClick={() => confirmDenylist(p.id)}
                      disabled={busyId === p.id}
                      className="px-2 py-1 rounded text-[11px] font-medium bg-gb-purple text-gb-bg0-h hover:bg-gb-bright-purple disabled:opacity-40"
                    >
                      Promote
                    </button>
                    <button
                      onClick={() => setDenylistId(null)}
                      className="px-2 py-1 rounded text-[11px] text-gb-fg4 hover:text-gb-fg1"
                    >
                      Cancel
                    </button>
                  </>
                ) : (
                  <>
                    <button
                      onClick={() => doPromote(p.id, "prompt")}
                      disabled={busyId === p.id}
                      title="Pin as a permanent rule — always injected for its category, never ranked out or aged out"
                      className="px-2 py-1 rounded text-[11px] font-medium text-gb-bright-blue hover:bg-gb-bg1 disabled:opacity-40"
                    >
                      ↑ prompt
                    </button>
                    <button
                      onClick={() => startDenylist(p)}
                      disabled={busyId === p.id}
                      title="Promote to a deterministic guardrail (blocks values / techniques)"
                      className="px-2 py-1 rounded text-[11px] font-medium text-gb-bright-purple hover:bg-gb-bg1 disabled:opacity-40"
                    >
                      ↑ denylist
                    </button>
                    <button
                      onClick={() => startEdit(p)}
                      disabled={busyId === p.id}
                      className="px-2 py-1 rounded text-[11px] text-gb-fg4 hover:text-gb-fg1 disabled:opacity-40"
                    >
                      edit
                    </button>
                    <button
                      onClick={() => doDismiss(p.id)}
                      disabled={busyId === p.id || p.status === "dismissed"}
                      className="px-2 py-1 rounded text-[11px] text-gb-bright-red hover:bg-gb-bg1 disabled:opacity-40"
                    >
                      ✕ dismiss
                    </button>
                  </>
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
