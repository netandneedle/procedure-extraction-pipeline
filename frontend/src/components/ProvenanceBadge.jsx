/**
 * ProvenanceBadge — renders a chunk/procedure's source-fidelity category
 * as a compact icon + label badge.
 *
 * Categories (5):
 *   prose       — native PDF prose. High fidelity; analyst can grep the source.
 *   code        — fenced code block. Highest fidelity; verbatim characters.
 *   figure      — vision-LLM-transcribed figure. Medium fidelity; transcription
 *                 error possible.
 *   hybrid      — multi-chunk procedure with mixed provenance (forward-looking;
 *                 rare while procedures are 1:1 with chunks).
 *   paraphrased — no verbatim anchor. LLM rewrote past parsed_text.find()
 *                 reach. Content is still grounded in the source; the
 *                 analyst just can't deep-link to a specific passage.
 *
 * Color encodes a rough "trust this verbatim" hierarchy:
 *   green  → high (code/prose)
 *   yellow → medium-low (figure / paraphrased)
 *   orange → mixed (hybrid)
 */

const STYLES = {
  prose: {
    label: "Prose",
    icon: "¶",  // ¶ pilcrow
    badge: "bg-gb-tag-reliability-bg text-gb-bright-green border-gb-green",
    description:
      "Source is native PDF prose. High fidelity — find by grepping the source.",
  },
  code: {
    label: "Code",
    icon: "{}",
    badge: "bg-gb-tag-reliability-bg text-gb-bright-aqua border-gb-aqua",
    description:
      "Source is a fenced code block (verbatim characters, no LLM interpretation).",
  },
  figure: {
    label: "Figure",
    icon: "▣",  // ▣
    badge: "bg-gb-tag-gate-bg text-gb-bright-yellow border-gb-yellow",
    description:
      "Source was vision-LLM-transcribed from a figure. Medium fidelity — transcription error is possible.",
  },
  hybrid: {
    label: "Hybrid",
    icon: "⚂",  // ⚂ — composite die face
    badge: "bg-gb-tag-gate-bg text-gb-bright-orange border-gb-orange",
    description:
      "Procedure spans multiple chunks with different provenance (mixed prose / figure / code).",
  },
  paraphrased: {
    label: "Paraphrased",
    icon: "≈",  // ≈ — approximation (LLM rewrote, not verbatim)
    badge: "bg-gb-tag-gate-bg text-gb-bright-yellow border-gb-yellow",
    description:
      "No verbatim anchor — the LLM rewrote the source phrasing past where parsed_text.find() can locate it. Content is still grounded in the source; the analyst just can't programmatically link to a specific passage.",
  },
};

/** Resolve a provenance string to its style entry, falling back to paraphrased. */
export function getProvenanceStyle(provenance) {
  return STYLES[provenance] || STYLES.paraphrased;
}

export default function ProvenanceBadge({ provenance, compact = false, className = "" }) {
  if (!provenance) return null;
  const style = getProvenanceStyle(provenance);
  if (compact) {
    return (
      <span
        title={`${style.label}: ${style.description}`}
        className={`font-data text-[10px] font-medium px-1.5 py-0.5 rounded border cursor-help ${style.badge} ${className}`}
      >
        {style.icon} {style.label}
      </span>
    );
  }
  return (
    <span
      title={style.description}
      className={`font-data text-[10px] font-medium px-1.5 py-0.5 rounded border inline-flex items-center gap-1 cursor-help ${style.badge} ${className}`}
    >
      <span>{style.icon}</span>
      <span>{style.label}</span>
    </span>
  );
}
