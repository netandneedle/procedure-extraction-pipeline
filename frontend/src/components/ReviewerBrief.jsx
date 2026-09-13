/**
 * ReviewerBrief — the AI reviewer's opening read of the report.
 *
 * Turn one of the reviewer's transcript, which is why it is editable here and
 * not merely displayed: a correction propagates to every downstream gate
 * instead of having to be re-argued at each one. It is the cheapest place in
 * the system to fix a misread.
 *
 * Collapsed by default. The analyst came to this panel to review entities;
 * the brief is context, not the task.
 */
import { useState, useCallback } from "react";

function listToText(items) {
  return (items ?? []).join("\n");
}

function textToList(text) {
  return text
    .split("\n")
    .map((s) => s.trim())
    .filter(Boolean);
}

export default function ReviewerBrief({ brief, onSave, saving }) {
  const [open, setOpen] = useState(false);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(null);
  const [error, setError] = useState(null);

  const read = brief?.payload?.initial_read;

  const startEdit = () => {
    setDraft({
      summary: read.summary ?? "",
      actors: listToText(read.actors),
      attack_chain: listToText(read.attack_chain),
      thin_areas: listToText(read.thin_areas),
      notes_for_later_gates: read.notes_for_later_gates ?? "",
    });
    setError(null);
    setEditing(true);
    setOpen(true);
  };

  const handleSave = useCallback(async () => {
    if (!draft?.summary.trim()) {
      setError("A summary is required.");
      return;
    }
    try {
      setError(null);
      await onSave({
        summary: draft.summary.trim(),
        actors: textToList(draft.actors),
        attack_chain: textToList(draft.attack_chain),
        thin_areas: textToList(draft.thin_areas),
        notes_for_later_gates: draft.notes_for_later_gates.trim(),
      });
      setEditing(false);
    } catch (err) {
      setError(err?.response?.data?.detail ?? err?.message ?? "Could not save.");
    }
  }, [draft, onSave]);

  // Every hook is above this line. The brief loads asynchronously, so this
  // component renders once with brief=null and again with it present; an
  // early return placed above useCallback changed the hook count between
  // those renders and React threw, blanking the gate.
  if (!read) return null;

  const analystEdited = Boolean(read.analyst_edited);

  const field = (key, label, rows, help) => (
    <div key={key}>
      <label className="block text-[11px] font-medium text-gb-fg4 mb-1">
        {label}
        {help && <span className="ml-1 font-normal text-gb-fg4/70">{help}</span>}
      </label>
      <textarea
        rows={rows}
        value={draft[key]}
        onChange={(e) => setDraft((p) => ({ ...p, [key]: e.target.value }))}
        className="w-full px-2 py-1.5 rounded border border-gb-bg2 bg-gb-bg0-h text-gb-fg1 text-[12px] focus:outline-none focus:border-gb-bright-purple"
      />
    </div>
  );

  return (
    <div className="border border-gb-purple/40 bg-gb-bg0-h rounded-md">
      <div className="flex items-center gap-2 px-3 py-2">
        <button
          onClick={() => setOpen((o) => !o)}
          className="flex items-center gap-2 text-[12px] font-semibold text-gb-bright-purple hover:text-gb-fg1 transition-colors"
        >
          <span>{open ? "▾" : "▸"}</span>
          <span>🤖 AI reviewer&rsquo;s read of this report</span>
        </button>
        {analystEdited && (
          <span
            className="text-[10px] font-data px-1.5 py-0.5 rounded bg-gb-blue/15 text-gb-bright-blue"
            title="You corrected this read. Every later gate uses your version."
          >
            yours
          </span>
        )}
        <div className="flex-1" />
        {!editing && (
          <button
            onClick={startEdit}
            className="text-[11px] text-gb-fg4 hover:text-gb-bright-purple transition-colors"
            title="Correct the read. It is the reviewer's first turn, so a fix here carries to every later gate."
          >
            ✎ correct
          </button>
        )}
      </div>

      {open && !editing && (
        <div className="px-3 pb-3 flex flex-col gap-2 text-[12px] text-gb-fg1">
          <p className="leading-relaxed">{read.summary}</p>
          {read.actors?.length > 0 && (
            <p className="font-data text-[11px]">
              <span className="text-gb-fg4">Actors: </span>
              {read.actors.join(", ")}
            </p>
          )}
          {read.attack_chain?.length > 0 && (
            <div>
              <span className="text-gb-fg4 text-[11px]">Chain:</span>
              <ol className="list-decimal list-inside text-[11px] font-data mt-0.5">
                {read.attack_chain.map((s, i) => <li key={i}>{s}</li>)}
              </ol>
            </div>
          )}
          {read.thin_areas?.length > 0 && (
            <div>
              {/* Where the report is vague is where extraction is most
                  likely to have invented something — worth the analyst's
                  attention before they start approving. */}
              <span className="text-gb-bright-orange text-[11px]">
                Thin in the source:
              </span>
              <ul className="list-disc list-inside text-[11px] mt-0.5 text-gb-fg2">
                {read.thin_areas.map((s, i) => <li key={i}>{s}</li>)}
              </ul>
            </div>
          )}
          {read.notes_for_later_gates && (
            <p className="text-[11px] text-gb-fg2 italic">
              {read.notes_for_later_gates}
            </p>
          )}
        </div>
      )}

      {open && editing && draft && (
        <div className="px-3 pb-3 flex flex-col gap-2">
          {field("summary", "Summary", 4)}
          {field("actors", "Actors", 2, "(one per line)")}
          {field("attack_chain", "Attack chain", 4, "(one step per line)")}
          {field("thin_areas", "Thin in the source", 3, "(one per line)")}
          {field("notes_for_later_gates", "Notes for later gates", 2)}
          {error && (
            <p className="text-[11px] text-gb-bright-red">{error}</p>
          )}
          <p className="text-[11px] text-gb-fg4">
            Saving re-runs the reviewer for this gate with your corrected read,
            and every later gate uses it too.
          </p>
          <div className="flex gap-2">
            <button
              onClick={handleSave}
              disabled={saving}
              className="px-2.5 py-1 rounded text-[11px] font-semibold font-data bg-gb-purple/15 text-gb-bright-purple border border-gb-purple hover:bg-gb-bg2 transition-colors disabled:opacity-50"
            >
              {saving ? "Re-reviewing…" : "Save & re-review"}
            </button>
            <button
              onClick={() => { setEditing(false); setError(null); }}
              disabled={saving}
              className="px-2.5 py-1 rounded text-[11px] font-data text-gb-fg4 hover:text-gb-fg1 transition-colors disabled:opacity-50"
            >
              Cancel
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
