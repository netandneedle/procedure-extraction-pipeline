/**
 * TechniqueEditor — Inline technique tag editor for Gate 1.
 *
 * Displays technique assignments as interactive tags. Each tag supports:
 *   - Click X to remove
 *   - Click pencil to expand inline edit form (technique, tactic, confidence, rationale)
 *   - "+" button opens an add form with searchable autocomplete
 *
 * Props:
 *   techniques: array of { technique_id, technique_name, tactic, confidence, rationale?, stix_id? }
 *   onChange: (updatedTechniques) => void
 *   catalogue: array from /api/techniques (full ATT&CK catalogue for search)
 */
import { useState, useMemo, useRef, useEffect, useCallback } from "react";

/** Confidence as colored bar. */
function ConfidenceBadge({ value }) {
  const pct = Math.round((value ?? 0) * 100);
  const color =
    pct >= 80
      ? "text-gb-bright-green"
      : pct >= 60
        ? "text-gb-bright-yellow"
        : pct >= 40
          ? "text-gb-bright-orange"
          : "text-gb-bright-red";
  return <span className={`font-data text-[10px] ${color}`}>{pct}%</span>;
}

/** Searchable technique autocomplete dropdown. */
function TechniqueSearch({ catalogue, onSelect, exclude, placeholder }) {
  const [query, setQuery] = useState("");
  const [open, setOpen] = useState(false);
  const ref = useRef(null);

  // Close on outside click
  useEffect(() => {
    function handleClick(e) {
      if (ref.current && !ref.current.contains(e.target)) setOpen(false);
    }
    document.addEventListener("mousedown", handleClick);
    return () => document.removeEventListener("mousedown", handleClick);
  }, []);

  const excludeSet = useMemo(
    () => new Set((exclude ?? []).map((t) => t.technique_id)),
    [exclude]
  );

  const results = useMemo(() => {
    if (!query || query.length < 2) return [];
    const q = query.toLowerCase();
    return catalogue
      .filter(
        (t) =>
          !excludeSet.has(t.technique_id) &&
          (t.technique_id.toLowerCase().includes(q) ||
            t.name.toLowerCase().includes(q) ||
            (t.description ?? "").toLowerCase().includes(q))
      )
      .slice(0, 15);
  }, [query, catalogue, excludeSet]);

  return (
    <div ref={ref} className="relative w-full">
      <input
        type="text"
        value={query}
        onChange={(e) => {
          setQuery(e.target.value);
          setOpen(true);
        }}
        onFocus={() => query.length >= 2 && setOpen(true)}
        placeholder={placeholder ?? "Search techniques (ID or name)..."}
        className="w-full bg-gb-bg1 text-gb-fg1 border border-gb-bg2 rounded px-2 py-1 text-[11px] font-data placeholder-gb-gray focus:border-gb-bright-blue focus:outline-none"
      />
      {open && results.length > 0 && (
        <div className="absolute z-50 top-full left-0 right-0 mt-0.5 bg-gb-bg0-s border border-gb-bg2 rounded shadow-lg max-h-48 overflow-y-auto">
          {results.map((t) => (
            <button
              key={t.technique_id}
              onClick={() => {
                onSelect(t);
                setQuery("");
                setOpen(false);
              }}
              className="w-full text-left px-2 py-1.5 hover:bg-gb-bg1 transition-colors flex items-start gap-2"
            >
              <span className="font-data text-[10px] text-gb-bright-blue whitespace-nowrap mt-px">
                {t.technique_id}
              </span>
              <span className="text-[11px] text-gb-fg1 leading-tight">
                {t.name}
                {t.tactics?.length > 0 && (
                  <span className="text-gb-fg4 ml-1">
                    ({t.tactics.join(", ")})
                  </span>
                )}
              </span>
            </button>
          ))}
        </div>
      )}
      {open && query.length >= 2 && results.length === 0 && (
        <div className="absolute z-50 top-full left-0 right-0 mt-0.5 bg-gb-bg0-s border border-gb-bg2 rounded shadow-lg px-3 py-2 text-[11px] text-gb-fg4">
          No matching techniques
        </div>
      )}
    </div>
  );
}

/** Inline edit form for a single technique. */
function TechniqueEditForm({ technique, catalogue, onSave, onCancel }) {
  const [tid, setTid] = useState(technique?.technique_id ?? "");
  const [name, setName] = useState(technique?.technique_name ?? "");
  const [tactic, setTactic] = useState(technique?.tactic ?? "");
  const [confidence, setConfidence] = useState(technique?.confidence ?? 0.7);
  const [rationale, setRationale] = useState(technique?.rationale ?? "");
  const [stixId, setStixId] = useState(technique?.stix_id ?? null);

  // Available tactics for the selected technique
  const availableTactics = useMemo(() => {
    const entry = catalogue.find((t) => t.technique_id === tid);
    return entry?.tactics ?? [];
  }, [tid, catalogue]);

  const handleTechniqueSelect = useCallback(
    (t) => {
      setTid(t.technique_id);
      setName(t.name);
      setStixId(t.stix_id);
      // Auto-set tactic if only one available
      if (t.tactics?.length === 1) {
        setTactic(t.tactics[0]);
      } else if (t.tactics && !t.tactics.includes(tactic)) {
        setTactic(t.tactics[0] ?? "");
      }
    },
    [tactic]
  );

  const handleSave = () => {
    if (!tid) return;
    // `|| 0.5` treated a deliberate 0 from the slider as "unset".
    const conf = parseFloat(confidence);
    onSave({
      technique_id: tid,
      technique_name: name,
      tactic,
      confidence: Number.isFinite(conf) ? conf : 0.5,
      rationale,
      stix_id: stixId,
    });
  };

  return (
    <div className="mt-1.5 p-2.5 bg-gb-bg0 border border-gb-bg2 rounded-lg space-y-2">
      {/* Technique selector */}
      <div>
        <label className="text-[10px] font-data font-semibold text-gb-fg4 uppercase tracking-wider">
          Technique
        </label>
        {tid ? (
          <div className="flex items-center gap-2 mt-0.5">
            <span className="font-data text-[11px] text-gb-bright-blue">
              {tid}
            </span>
            <span className="text-[11px] text-gb-fg1">{name}</span>
            <button
              onClick={() => {
                setTid("");
                setName("");
                setStixId(null);
              }}
              className="text-[10px] text-gb-fg4 hover:text-gb-bright-red ml-auto"
            >
              change
            </button>
          </div>
        ) : (
          <TechniqueSearch
            catalogue={catalogue}
            onSelect={handleTechniqueSelect}
            exclude={[]}
            placeholder="Search by ID or name..."
          />
        )}
      </div>

      {/* Tactic selector */}
      <div>
        <label className="text-[10px] font-data font-semibold text-gb-fg4 uppercase tracking-wider">
          Tactic
        </label>
        {availableTactics.length > 0 ? (
          <select
            value={tactic}
            onChange={(e) => setTactic(e.target.value)}
            className="mt-0.5 w-full bg-gb-bg1 text-gb-fg1 border border-gb-bg2 rounded px-2 py-1 text-[11px] font-data"
          >
            <option value="">Select tactic...</option>
            {availableTactics.map((t) => (
              <option key={t} value={t}>
                {t}
              </option>
            ))}
          </select>
        ) : (
          <input
            type="text"
            value={tactic}
            onChange={(e) => setTactic(e.target.value)}
            placeholder="e.g., execution"
            className="mt-0.5 w-full bg-gb-bg1 text-gb-fg1 border border-gb-bg2 rounded px-2 py-1 text-[11px] font-data placeholder-gb-gray"
          />
        )}
      </div>

      {/* Confidence slider */}
      <div>
        <label className="text-[10px] font-data font-semibold text-gb-fg4 uppercase tracking-wider">
          Confidence: {Math.round(confidence * 100)}%
        </label>
        <input
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={confidence}
          onChange={(e) => setConfidence(parseFloat(e.target.value))}
          className="mt-0.5 w-full h-1.5 rounded-full appearance-none bg-gb-bg2 accent-gb-bright-blue"
        />
      </div>

      {/* Rationale */}
      <div>
        <label className="text-[10px] font-data font-semibold text-gb-fg4 uppercase tracking-wider">
          Rationale
        </label>
        <input
          type="text"
          value={rationale}
          onChange={(e) => setRationale(e.target.value)}
          placeholder="Why this technique applies..."
          className="mt-0.5 w-full bg-gb-bg1 text-gb-fg1 border border-gb-bg2 rounded px-2 py-1 text-[11px] font-data placeholder-gb-gray"
        />
      </div>

      {/* Actions */}
      <div className="flex justify-end gap-2 pt-1">
        <button
          onClick={onCancel}
          className="px-2.5 py-1 rounded text-[11px] font-data text-gb-fg4 hover:text-gb-fg1 transition-colors"
        >
          Cancel
        </button>
        <button
          onClick={handleSave}
          disabled={!tid || !tactic}
          className="px-2.5 py-1 rounded text-[11px] font-semibold font-data bg-gb-bright-blue text-gb-bg0-h hover:bg-gb-blue transition-colors disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Save
        </button>
      </div>
    </div>
  );
}

/** Main TechniqueEditor component. */
export default function TechniqueEditor({ techniques, onChange, catalogue }) {
  const [editingIdx, setEditingIdx] = useState(null); // index of technique being edited
  const [adding, setAdding] = useState(false);

  const handleRemove = (idx) => {
    const updated = techniques.filter((_, i) => i !== idx);
    onChange(updated);
    // Fix stale index after array shift
    if (editingIdx === idx) {
      setEditingIdx(null);
    } else if (editingIdx !== null && editingIdx > idx) {
      setEditingIdx(editingIdx - 1);
    }
  };

  const handleEdit = (idx) => {
    setAdding(false);
    setEditingIdx(editingIdx === idx ? null : idx);
  };

  const handleEditSave = (idx, updatedTechnique) => {
    const updated = [...techniques];
    updated[idx] = updatedTechnique;
    onChange(updated);
    setEditingIdx(null);
  };

  const handleAddSave = (newTechnique) => {
    onChange([...techniques, newTechnique]);
    setAdding(false);
  };

  return (
    <div className="space-y-1.5">
      {/* Technique tags */}
      <div className="flex flex-wrap gap-1.5">
        {techniques.map((t, i) => (
          <div key={`${t.technique_id}-${i}`} className="flex flex-col">
            <div
              className={`inline-flex items-center gap-1 font-data text-[10px] px-1.5 py-0.5 rounded border transition-colors ${
                editingIdx === i
                  ? "bg-gb-bright-blue/20 text-gb-bright-blue border-gb-bright-blue"
                  : "bg-gb-tag-type-bg text-gb-bright-blue border-gb-blue"
              }`}
            >
              <span
                title={
                  t.technique_name
                    ? `${t.technique_id} — ${t.technique_name}` +
                      (t.confidence != null
                        ? ` (conf: ${Math.round(t.confidence * 100)}%)`
                        : "")
                    : t.technique_id
                }
              >
                {t.technique_id}
                {t.technique_name ? ` · ${t.technique_name}` : ""}
              </span>
              <ConfidenceBadge value={t.confidence} />
              {/* Edit button */}
              <button
                onClick={() => handleEdit(i)}
                className="text-[16px] leading-none text-gb-fg4 hover:text-gb-bright-yellow transition-colors ml-1 p-1"
                title="Edit technique"
              >
                ✎
              </button>
              {/* Remove button */}
              <button
                onClick={() => handleRemove(i)}
                className="text-[16px] leading-none text-gb-fg4 hover:text-gb-bright-red transition-colors p-1"
                title="Remove technique"
              >
                ✕
              </button>
            </div>

            {/* Inline edit form (expands below the tag) */}
            {editingIdx === i && (
              <TechniqueEditForm
                technique={t}
                catalogue={catalogue}
                onSave={(updated) => handleEditSave(i, updated)}
                onCancel={() => setEditingIdx(null)}
              />
            )}
          </div>
        ))}

        {/* Add button */}
        {!adding && (
          <button
            onClick={() => {
              setEditingIdx(null);
              setAdding(true);
            }}
            className="inline-flex items-center gap-0.5 font-data text-[10px] px-1.5 py-0.5 rounded border border-dashed border-gb-fg4 text-gb-fg4 hover:border-gb-bright-blue hover:text-gb-bright-blue transition-colors"
            title="Add technique"
          >
            + Add
          </button>
        )}
      </div>

      {/* Add form (appears below all tags) */}
      {adding && (
        <TechniqueEditForm
          technique={null}
          catalogue={catalogue}
          onSave={handleAddSave}
          onCancel={() => setAdding(false)}
        />
      )}
    </div>
  );
}
