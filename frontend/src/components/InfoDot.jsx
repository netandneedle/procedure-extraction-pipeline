/**
 * InfoDot — a small ⓘ beside a label, defining one term from lib/glossary.js.
 *
 * WHY A VISIBLE GLYPH rather than a title= on the label itself. The app already
 * uses ~75 bare `title` attributes, and they share one flaw: nothing on screen
 * says help is there. Nobody hovers a word they have no reason to think is
 * hoverable. The ⓘ is the affordance; the tooltip is just its payload.
 *
 * KNOWN LIMITS of native tooltips, accepted deliberately: they do not appear on
 * touch devices, and they wait about a second before showing. The trade is that
 * this costs no layout, no portal, no focus management and no new dependency in
 * a form that is already tall. Should the delay start to grate, the glyph and
 * every call site stay exactly as they are — only this file changes.
 *
 * `tabIndex` + `aria-label` because a bare `title` is neither keyboard-reachable
 * nor reliably announced by a screen reader.
 */
import { hint } from "../lib/glossary";

export default function InfoDot({ term, className = "" }) {
  const text = hint(term);

  // Empty only when a key slipped past glossary.test.js's static scan in a
  // production build. Render nothing rather than an ⓘ that promises help and
  // then shows an empty tooltip.
  if (!text) return null;

  return (
    <span
      role="note"
      aria-label={text}
      title={text}
      tabIndex={0}
      className={
        "inline-block align-baseline ml-1 text-[10px] leading-none " +
        "text-gb-fg4 hover:text-gb-bright-yellow focus:text-gb-bright-yellow " +
        "cursor-help select-none transition-colors outline-none " +
        className
      }
    >
      ⓘ
    </span>
  );
}
