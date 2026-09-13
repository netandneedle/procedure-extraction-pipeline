/**
 * AddSourceModal — form for adding a new source to the queue.
 *
 * Workflow:
 *  1. User drops a file or picks one via the file picker.
 *  2. Frontend validates size + extension, uploads to /api/sources/upload.
 *  3. Backend returns { path, source_type } which we wire into the create payload.
 *  4. User fills title/reliability/gates/metadata, submits.
 *
 * source_type is auto-detected from the file extension and not user-editable.
 */
import { useRef, useState } from "react";
import { uploadSourceFile } from "../api/sources";
import {
  GATE_AI_REVIEWER_KEYS,
  GATE_ENABLE_KEYS,
  GATE_ENABLE_LABELS,
  GATE_HINT_TERMS,
  GATE_MODE_OPTIONS,
  UNATTENDED_GATE_MODES,
} from "../lib/gates";
import InfoDot from "./InfoDot";

// Per-gate enable/disable. Keys mirror backend GATE_KEYS in graph/state.py.
// Display order matches the pipeline's gate sequence. See lib/gates.js for
// the source of truth.
//
// Naming hangover: the 'procedures' key gates the post-draft technique
// review (Python gate_1), and the 'chunks' key gates the chunk-as-procedure
// review (Python gate_chunks). The 'procedures' name predates the chunk gate.
const DEFAULT_RELIABILITY = 50;

/** Number in 0-100, or the default when the field is blank or not a number. */
function reliabilityOrDefault(raw) {
  if (raw === "" || raw == null) return DEFAULT_RELIABILITY;
  const n = Number(raw);
  if (!Number.isFinite(n)) return DEFAULT_RELIABILITY;
  return Math.min(100, Math.max(0, n));
}

const ALL_GATES_ON = Object.fromEntries(GATE_ENABLE_KEYS.map((k) => [k, true]));
const ALL_GATES_REVIEW = Object.fromEntries(
  GATE_ENABLE_KEYS.map((k) => [k, "review"])
);

// Says what UNCHECKING did, not just that boxes are ticked. "All gates active"
// left the analyst to guess whether the alternative was auto-approve or skip;
// it is auto-approve, and the difference matters — the work still happens.
//
// Reads gate_modes as well as gates_enabled, because there are now two ways a
// stage can pass unreviewed and they are not the same thing: unchecked means
// nobody looks, "AI decides" means the AI looks and nobody checks it.
function summarizeGates(enabled, modes) {
  const label = (k) => GATE_ENABLE_LABELS[k];
  const off = GATE_ENABLE_KEYS.filter((k) => !enabled[k]);
  const unattended = GATE_ENABLE_KEYS.filter(
    (k) => enabled[k] && UNATTENDED_GATE_MODES.has(modes[k])
  );

  const parts = [];
  if (off.length === GATE_ENABLE_KEYS.length) {
    parts.push("Nothing stops for review — every stage auto-approves.");
  } else if (off.length > 0) {
    parts.push(`Auto-approved without review: ${off.map(label).join(", ")}.`);
  }
  if (unattended.length > 0) {
    parts.push(`AI decides unattended: ${unattended.map(label).join(", ")}.`);
  }
  if (parts.length === 0) {
    return `All ${GATE_ENABLE_KEYS.length} stages stop for review.`;
  }
  return parts.join(" ");
}

// "Does the source include sequential information?" Drives the pipeline's
// PREDECESSOR / orphan-backstop / PRECEDES-SRO behavior. Auto defers to the
// LLM classifier inside entity_extraction.
const SEQUENTIALITY_OPTIONS = [
  { value: "auto", label: "Auto-detect", help: "LLM classifies during entity extraction." },
  { value: "yes", label: "Yes", help: "Single intrusion narrative with chronological ordering." },
  { value: "no", label: "No", help: "Catalog or profile of procedures without intrinsic ordering." },
];

// `channel` is deliberately absent. The column still exists and still defaults
// to "manual" server-side (SourceCreate.channel), but nothing in the pipeline
// ever read it — it was a choice the analyst could not get wrong because it had
// no effect. Automated ingestion, when it exists, will set it directly.
const INITIAL_FORM = {
  title: "",
  source_reliability: DEFAULT_RELIABILITY,
  gates_enabled: { ...ALL_GATES_ON },
  gate_modes: { ...ALL_GATES_REVIEW },
  sequentiality: "auto",
  extract_figures: true,
  metadata: "",
};

const MAX_UPLOAD_BYTES = 50 * 1024 * 1024; // 50 MB, mirrors backend cap

// Extensions we accept — must match EXTENSION_TO_SOURCE_TYPE in backend/app/services/queue.py
const ACCEPTED_EXTENSIONS = [
  ".pdf", ".html", ".htm", ".md", ".markdown",
  ".txt", ".docx", ".png", ".jpg", ".jpeg",
];
const ACCEPT_ATTR = ACCEPTED_EXTENSIONS.join(",");

function humanBytes(n) {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

function hasAcceptedExtension(filename) {
  const lower = filename.toLowerCase();
  return ACCEPTED_EXTENSIONS.some((ext) => lower.endsWith(ext));
}

export default function AddSourceModal({ isOpen, onClose, onSubmit }) {
  const [form, setForm] = useState({ ...INITIAL_FORM });
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploaded, setUploaded] = useState(null); // { path, source_type, filename, size }
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);

  if (!isOpen) return null;

  function resetAndClose() {
    setForm({ ...INITIAL_FORM });
    setUploaded(null);
    setError(null);
    setDragActive(false);
    onClose();
  }

  function update(field, value) {
    setForm((prev) => ({ ...prev, [field]: value }));
    setError(null);
  }

  function updateGate(key, value) {
    setForm((prev) => ({
      ...prev,
      gates_enabled: { ...prev.gates_enabled, [key]: value },
    }));
    setError(null);
  }

  function updateGateMode(key, value) {
    setForm((prev) => ({
      ...prev,
      gate_modes: { ...prev.gate_modes, [key]: value },
    }));
    setError(null);
  }

  async function handleFile(file) {
    setError(null);
    if (!file) return;
    if (!hasAcceptedExtension(file.name)) {
      setError(`Unsupported file type. Allowed: ${ACCEPTED_EXTENSIONS.join(", ")}`);
      return;
    }
    if (file.size > MAX_UPLOAD_BYTES) {
      setError(`File is ${humanBytes(file.size)}. Max allowed is 50 MB.`);
      return;
    }

    setUploading(true);
    try {
      const res = await uploadSourceFile(file);
      setUploaded({ ...res, size: file.size });
      // Default the title to the filename if the user hasn't typed one.
      setForm((prev) => ({
        ...prev,
        title: prev.title.trim() ? prev.title : file.name.replace(/\.[^.]+$/, ""),
      }));
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : String(detail ?? "Upload failed."));
    } finally {
      setUploading(false);
    }
  }

  function onDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const file = e.dataTransfer?.files?.[0];
    handleFile(file);
  }

  function onDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(true);
  }

  function onDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
  }

  function onPickClick() {
    fileInputRef.current?.click();
  }

  function onPickChange(e) {
    const file = e.target.files?.[0];
    handleFile(file);
    // Allow re-picking the same file after an error.
    e.target.value = "";
  }

  function clearFile() {
    setUploaded(null);
    setError(null);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    if (!uploaded) {
      setError("Drop or select a file first.");
      return;
    }

    let metadata = {};
    if (form.metadata.trim()) {
      // Cap the input BEFORE JSON.parse: a deeply nested 100k+ char paste
      // blocks the main thread otherwise. Backend's
      // _METADATA_MAX_SERIALIZED_BYTES is 32 KB; surface a closer cap to
      // the analyst.
      if (form.metadata.length > 4096) {
        setError("Metadata is too large. Keep it under 4 KB of JSON.");
        return;
      }
      try {
        metadata = JSON.parse(form.metadata);
      } catch {
        setError("Metadata must be valid JSON.");
        return;
      }
      // Schema docs expect an object like {author, campaign, tlp}. Reject
      // arrays / non-object roots so the backend isn't trusted with an
      // unexpected shape.
      if (
        metadata === null ||
        typeof metadata !== "object" ||
        Array.isArray(metadata)
      ) {
        setError("Metadata must be a JSON object (e.g. {\"author\": \"...\"}).");
        return;
      }
    }

    setSubmitting(true);
    try {
      await onSubmit({
        title: form.title.trim() || uploaded.filename || "Untitled Source",
        source_type: uploaded.source_type,
        raw_content_path: uploaded.path,
        // "" would coerce to 0 — 30% of every procedure's confidence —
        // when the analyst clears the field to retype it and submits early.
        source_reliability: reliabilityOrDefault(form.source_reliability),
        gates_enabled: form.gates_enabled,
        gate_modes: form.gate_modes,
        sequentiality: form.sequentiality,
        extract_figures: form.extract_figures,
        metadata,
      });
      resetAndClose();
    } catch (err) {
      const detail = err?.response?.data?.detail;
      setError(typeof detail === "string" ? detail : String(detail ?? "Failed to create source."));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-gb-bg0-h/75"
      onClick={resetAndClose}
    >
      <form
        onClick={(e) => e.stopPropagation()}
        onSubmit={handleSubmit}
        className="bg-gb-bg0-s border border-gb-bg2 rounded-xl p-6 w-[480px] max-w-[90vw]"
      >
        <h2 className="text-base font-semibold text-gb-fg0 mb-4">Add Source</h2>

        {/* Drop zone / file picker */}
        <label className="block text-xs font-medium text-gb-fg4 mb-1">File</label>
        <div
          onDrop={onDrop}
          onDragOver={onDragOver}
          onDragEnter={onDragOver}
          onDragLeave={onDragLeave}
          onClick={!uploaded && !uploading ? onPickClick : undefined}
          className={`
            rounded-md border border-dashed px-3 py-5 mb-3 text-center text-[12px] transition-colors
            ${dragActive
              ? "border-gb-bright-blue bg-gb-bg0-h"
              : uploaded
                ? "border-gb-green bg-gb-bg0-h"
                : "border-gb-bg2 bg-gb-bg0-h cursor-pointer hover:border-gb-bright-yellow"
            }
          `}
        >
          {uploading ? (
            <p className="text-gb-fg4">Uploading…</p>
          ) : uploaded ? (
            <div className="flex items-center justify-between gap-2 text-left">
              <div className="min-w-0">
                <p className="font-data text-[12px] text-gb-fg1 truncate">{uploaded.filename}</p>
                <p className="font-data text-[10px] text-gb-fg4">
                  {humanBytes(uploaded.size)} · detected as <span className="text-gb-bright-blue">{uploaded.source_type}</span>
                </p>
              </div>
              <button
                type="button"
                onClick={clearFile}
                className="px-2 py-1 rounded text-[11px] border border-gb-bg2 text-gb-fg4 hover:border-gb-bright-red hover:text-gb-bright-red transition-colors"
              >
                Change
              </button>
            </div>
          ) : (
            <>
              <p className="text-gb-fg4">
                Drop a file here or <span className="text-gb-bright-yellow">click to browse</span>
              </p>
              <p className="font-data text-[10px] text-gb-gray mt-1">
                {ACCEPTED_EXTENSIONS.join(" ")} · max 50 MB
              </p>
            </>
          )}
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPT_ATTR}
            onChange={onPickChange}
            className="hidden"
          />
        </div>

        {/* Title */}
        <label className="block text-xs font-medium text-gb-fg4 mb-1">Title</label>
        <input
          type="text"
          value={form.title}
          onChange={(e) => update("title", e.target.value)}
          placeholder="e.g. APT29 SolarWinds Retrospective"
          className="w-full px-2.5 py-2 rounded-md border border-gb-bg2 bg-gb-bg0-h text-gb-fg1 text-[13px] mb-3 focus:outline-none focus:border-gb-bright-yellow focus:ring-1 focus:ring-gb-bright-yellow/25"
        />

        {/* Source reliability. The most consequential field in this form and
            the least obviously so — it is 30% of every extracted procedure's
            final confidence score, hence the hint. */}
        <label className="block text-xs font-medium text-gb-fg4 mb-1">
          Source Reliability (0-100)
          <InfoDot term="source-reliability" />
        </label>
        <input
          type="number"
          min={0}
          max={100}
          required
          value={form.source_reliability}
          onChange={(e) => update("source_reliability", e.target.value)}
          className="w-24 px-2.5 py-2 rounded-md border border-gb-bg2 bg-gb-bg0-h text-gb-fg1 text-[13px] mb-3 focus:outline-none focus:border-gb-bright-yellow"
        />

        {/* Gates — per-gate enable/disable */}
        <label className="block text-xs font-medium text-gb-fg4 mb-1">
          Gates
          <InfoDot term="gates" />
        </label>
        <div className="border border-gb-bg2 bg-gb-bg0-h rounded-md px-2.5 py-2 mb-3">
          <div className="flex flex-col gap-1.5">
            {GATE_ENABLE_KEYS.map((key) => {
              const enabled = form.gates_enabled[key];
              const hasReviewer = GATE_AI_REVIEWER_KEYS.has(key);
              return (
                <div key={key} className="flex items-center gap-2 text-[13px] text-gb-fg1">
                  {/* The ⓘ sits OUTSIDE the <label> — inside it, clicking the
                      hint would toggle the checkbox. Hence the separate
                      flex-1 spacer rather than stretching the label itself. */}
                  <label className="flex items-center gap-2 cursor-pointer min-w-0">
                    <input
                      type="checkbox"
                      checked={enabled}
                      onChange={(e) => updateGate(key, e.target.checked)}
                      className="accent-gb-bright-yellow"
                    />
                    <span className="truncate">{GATE_ENABLE_LABELS[key]}</span>
                  </label>
                  <InfoDot term={GATE_HINT_TERMS[key]} className="shrink-0" />
                  <div className="flex-1" />
                  {/* Mode only means something for an ENABLED gate — a gate
                      nobody reviews has no reviewer to configure. Gates with
                      no AI reviewer yet show nothing rather than a disabled
                      control implying one is coming. */}
                  {enabled && hasReviewer && (
                    <select
                      value={form.gate_modes[key]}
                      onChange={(e) => updateGateMode(key, e.target.value)}
                      className="shrink-0 px-1.5 py-0.5 rounded border border-gb-bg2 bg-gb-bg0-h text-gb-fg1 text-[11px] focus:outline-none focus:border-gb-bright-yellow"
                      title={
                        GATE_MODE_OPTIONS.find((o) => o.value === form.gate_modes[key])?.help
                      }
                    >
                      {GATE_MODE_OPTIONS.map((o) => (
                        <option key={o.value} value={o.value}>{o.label}</option>
                      ))}
                    </select>
                  )}
                </div>
              );
            })}
          </div>
          <p className="text-[11px] text-gb-fg4 mt-2">{summarizeGates(form.gates_enabled, form.gate_modes)}</p>
        </div>

        {/* Sequentiality */}
        <label className="block text-xs font-medium text-gb-fg4 mb-1">
          Does the source include sequential information?
        </label>
        <div className="border border-gb-bg2 bg-gb-bg0-h rounded-md px-2.5 py-2 mb-3">
          <div className="flex flex-col gap-1.5">
            {SEQUENTIALITY_OPTIONS.map((opt) => (
              <label
                key={opt.value}
                className="flex items-start gap-2 text-[13px] text-gb-fg1 cursor-pointer"
              >
                <input
                  type="radio"
                  name="sequentiality"
                  value={opt.value}
                  checked={form.sequentiality === opt.value}
                  onChange={(e) => update("sequentiality", e.target.value)}
                  className="mt-0.5 accent-gb-bright-yellow"
                />
                <span className="flex-1">
                  <span className="font-medium">{opt.label}</span>
                  <span className="text-[11px] text-gb-fg4 block leading-snug">{opt.help}</span>
                </span>
              </label>
            ))}
          </div>
        </div>

        {/* Figure extraction toggle */}
        <label className="flex items-start gap-2 text-[13px] text-gb-fg1 mb-3 cursor-pointer">
          <input
            type="checkbox"
            checked={form.extract_figures}
            onChange={(e) => update("extract_figures", e.target.checked)}
            className="mt-0.5 accent-gb-bright-yellow"
          />
          <span className="flex-1">
            <span className="font-medium">Extract figures with vision LLM</span>
            <span className="text-[11px] text-gb-fg4 block leading-snug">
              Run a vision pass over each figure (attack-chain diagrams, command-line screenshots) and inline the extracted text into the parsed body. Adds time + tokens; disable for all-prose sources.
            </span>
          </span>
        </label>

        {/* Metadata */}
        <label className="block text-xs font-medium text-gb-fg4 mb-1">
          Metadata (optional JSON)
          <InfoDot term="metadata" />
        </label>
        {/* The placeholder teaches which keys are worth typing: these three
            are among the six that reach the extraction prompt
            (entity_extraction._source_context). "tlp" is deliberately absent —
            nothing reads it. */}
        <textarea
          value={form.metadata}
          onChange={(e) => update("metadata", e.target.value)}
          placeholder='{"author": "Mandiant", "campaign": "APT29", "publication_date": "2026-03-14"}'
          rows={2}
          className="w-full px-2.5 py-2 rounded-md border border-gb-bg2 bg-gb-bg0-h text-gb-fg1 text-[13px] mb-3 resize-y focus:outline-none focus:border-gb-bright-yellow focus:ring-1 focus:ring-gb-bright-yellow/25 font-data"
        />

        {/* Error */}
        {error && (
          <p className="text-[12px] text-gb-bright-red mb-3">{error}</p>
        )}

        {/* Actions */}
        <div className="flex justify-end gap-2">
          <button
            type="button"
            onClick={resetAndClose}
            className="px-3.5 py-1.5 rounded-md text-[13px] border border-gb-bg2 text-gb-fg4 hover:bg-gb-bg1 hover:text-gb-fg1 transition-colors"
          >
            Cancel
          </button>
          <button
            type="submit"
            disabled={submitting || uploading || !uploaded}
            className="px-3.5 py-1.5 rounded-md text-[13px] font-medium bg-gb-green text-gb-bg0-h hover:bg-gb-bright-green transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {submitting ? "Adding..." : "Add to Queue"}
          </button>
        </div>
      </form>
    </div>
  );
}
