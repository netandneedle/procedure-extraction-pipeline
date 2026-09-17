/**
 * SuggestionChip — one AI reviewer recommendation, inline on a review card.
 *
 * Shows the recommended action, its confidence, the reason, and the verbatim
 * source quote it rests on. The quote is shown rather than summarized on
 * purpose: an analyst who can see the evidence can judge the recommendation
 * in a second, and one who cannot is being asked to trust rather than review.
 *
 * When the grounding check could not find the quote in the report, that is
 * stated plainly. A recommendation resting on an unsupported quote is exactly
 * the failure this pipeline has been bitten by before, and the analyst should
 * see it in the same glance as the recommendation itself.
 */
import { CONFIDENCE_STYLES } from "../lib/reviewerSuggestions";

/** What this recommendation would actually do, in the analyst's words.
 *
 * A bare "edit" tells the analyst nothing — they would have to open the
 * draft to find out what changes. Naming the techniques it drops, or the
 * reason it rejects, makes the chip judgeable on its own.
 */
function actionLabel(rec) {
  if (rec.technique_id && !rec.action) return `promote ${rec.technique_id}`;
  const action = rec.action ?? "add";
  const drops = rec.remove_technique_ids ?? [];
  if (action === "edit" && drops.length) {
    return `drop ${drops.join(", ")}`;
  }
  if (action === "reject" && rec.reject_reason) {
    return `reject — ${rec.reject_reason.replace(/_/g, " ")}`;
  }
  if (action === "merge" && rec.merge_with?.length) {
    return `merge in ${rec.merge_with.join(", ")}`;
  }
  return action;
}

export default function SuggestionChip({ rec, applied, onApply, onDismiss }) {
  if (!rec) return null;
  const confidence = rec.confidence ?? "low";
  const unsupported = Boolean(rec.quote_unsupported);

  return (
    <div className="mt-2 rounded border border-gb-purple/40 bg-gb-purple/5 px-2 py-1.5">
      <div className="flex items-center gap-1.5 flex-wrap">
        <span className="text-[10px] font-data text-gb-bright-purple">🤖 AI</span>
        <span className="text-[11px] font-semibold font-data text-gb-fg1">
          {actionLabel(rec)}
        </span>
        <span
          className={`text-[9px] font-data px-1 py-0.5 rounded border ${CONFIDENCE_STYLES[confidence]}`}
          title={
            confidence === "high"
              ? "Stated plainly in the report. Included in bulk accept."
              : "Not included in bulk accept — open it and decide."
          }
        >
          {confidence}
        </span>
        {unsupported && (
          <span
            className="text-[9px] font-data px-1 py-0.5 rounded bg-gb-red/15 text-gb-bright-red border border-gb-red"
            title={
              "The quote below could not be matched against the report " +
              `(support ${rec.quote_source_support ?? "?"}). Confidence was ` +
              "forced to low. Treat the reasoning as unverified."
            }
          >
            ⚠ quote unverified
          </span>
        )}
        <div className="flex-1" />
        {/* No onApply means the caller already provides its own action for
            this recommendation — the chip is evidence only. Two buttons doing
            the same thing on one row reads as a bug. */}
        {!onApply ? null : applied ? (
          <button
            onClick={onDismiss}
            className="text-[10px] font-data text-gb-bright-aqua hover:text-gb-fg1 transition-colors"
            title="Undo — restore your own decision"
          >
            ✓ applied · undo
          </button>
        ) : (
          <button
            onClick={onApply}
            className="text-[10px] font-data text-gb-fg4 hover:text-gb-bright-purple transition-colors"
          >
            apply
          </button>
        )}
      </div>
      {rec.rationale && (
        <p className="text-[11px] text-gb-fg2 mt-1 leading-snug">{rec.rationale}</p>
      )}
      {rec.evidence_quote && (
        <p
          className={`text-[10px] font-data mt-1 pl-2 border-l-2 leading-snug ${
            unsupported
              ? "border-gb-red text-gb-fg4 line-through decoration-gb-red/60"
              : "border-gb-purple/50 text-gb-fg4"
          }`}
        >
          &ldquo;{rec.evidence_quote}&rdquo;
        </p>
      )}
    </div>
  );
}
