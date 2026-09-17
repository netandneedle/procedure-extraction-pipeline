/**
 * lib/glossary.js — one definition per term, for the whole UI.
 *
 * The pipeline's vocabulary is internal: "precedes", "denylist", "promote",
 * "source reliability". An analyst meeting these for the first time has no way
 * to learn what they do without reading the source. These are the definitions,
 * surfaced by <InfoDot term="..." />.
 *
 * ONE FILE, deliberately. "precedes" appears on the chunk canvas, the bundle
 * flow view and the bundle review canvas; it has to read identically in all
 * three. Same single-source-of-truth move as lib/gates.js and
 * lib/bundleGraphConstants.js, for the same reason: three hand-maintained
 * copies of a definition are three definitions that will drift.
 *
 * WRITING RULES — these entries are the deliverable, so they have a house style:
 *
 *   1. Say what it DOES, not what it is. "Weighs 30% of every procedure's final
 *      confidence score" beats "how much you trust this source" — the first
 *      tells the analyst what changes if they touch it.
 *   2. No internal vocabulary INSIDE a definition. A definition that explains
 *      "possible bucket" in terms of "the C+A+D review lane" has explained
 *      nothing. If a term needs another term, define it in plain words here.
 *   3. Under MAX_HINT_LENGTH. Native tooltips wrap badly past roughly this
 *      much text, and a definition that needs 300 characters is a sign the
 *      control itself is doing too much.
 *
 * Every value is checked against 1-3 by lib/__tests__/glossary.test.js, which
 * also scans the components for `term="..."` and fails on any key that is not
 * defined here.
 */

/** Native tooltips wrap badly past this. See writing rule 3. */
export const MAX_HINT_LENGTH = 240;

export const GLOSSARY = {
  // ── Add Source ──────────────────────────────────────────────────────
  "source-reliability":
    "How far you trust this publisher, 0-100. Weighs 30% of every extracted " +
    "procedure's final confidence score; the report's own detail supplies the " +
    "other 70%. Leave it at 50 if you have no view.",

  gates:
    "The four points where the pipeline can stop for a decision. Unchecking " +
    "one auto-approves that stage; setting it to AI decides lets the AI " +
    "answer it. Either way the work still happens, unreviewed.",

  "gate-entities":
    "Check the actors, malware, tools and indicators pulled out of the report, " +
    "before anything else is built on them.",

  "gate-chunks":
    "Check how the report was split into separate procedures, and the order " +
    "they run in.",

  "gate-procedures":
    "Check which ATT&CK techniques each procedure was mapped to.",

  "gate-bundle":
    "Check the relationships wired between objects in the finished STIX bundle " +
    "before it ships.",

  metadata:
    "Optional JSON kept with the source. Six keys also feed the extraction " +
    "prompt: author, threat_actor, campaign, malware_family, publication_date, " +
    "source_url. Anything else is stored but unused.",

  // ── Navigation ──────────────────────────────────────────────────────
  "tab-feedback":
    "Rules learned from your past corrections and reused on every future " +
    "source. Promote one to pin it into the prompts, or to block a term outright.",

  "tab-reviewer":
    "How often you took the AI reviewer's advice, per gate. Only fills in for " +
    "gates you set to AI Assisted Review.",

  // ── Gate 0 — entities ───────────────────────────────────────────────
  "entity-type":
    "What kind of thing this is — actor, malware, tool, indicator and so on. " +
    "Changing it changes how the entity is modeled in the bundle.",

  "entity-role":
    "What part this plays in the report: the victim, the attack's origin, the " +
    "report's publisher or author. Only organizations and locations have one.",

  "entity-rationale":
    "Your note on why you changed or removed this. It feeds the pattern " +
    "learner, so a sentence here is worth more than a blank.",

  denylist:
    "Matches a term you previously confirmed as never wanted. Removed by " +
    "default — flip it back to approve to override, for this source only.",

  // ── Gate 1 — procedures ─────────────────────────────────────────────
  precedes:
    "An arrow meaning this procedure runs before that one. Drag between the " +
    "handles on two nodes to add one; click an edge and press Delete to remove it.",

  "branch-point":
    "The chain splits here — more than one procedure follows this one.",

  "convergence-point":
    "Separate paths rejoin here — this procedure follows more than one other.",

  "source-excerpt":
    "The verbatim sentences from the report this procedure was drawn from. " +
    "Click it to highlight that passage in the source pane.",

  "flow-condition":
    "Marks a fork the report describes as a runtime check — \"if EDR is " +
    "present, abort; otherwise continue\". The procedures after it are split " +
    "into a true side and a false side.",

  // ── Gate 2 — techniques ─────────────────────────────────────────────
  "review-for-inclusion":
    "Techniques the AI thought plausible but could not tie confidently to this " +
    "procedure's objective. They stay out of the bundle unless you promote them.",

  "promote-technique":
    "Add this technique to the procedure so it ships in the bundle.",

  "technique-denylist":
    "Matches a technique you previously confirmed as never wanted. Held out " +
    "of the bundle but shown here; promoting it overrides that, for this " +
    "source only.",

  // ── Gate 3 — bundle ─────────────────────────────────────────────────
  "bundle-flow-tab":
    "Just the procedures and the order they run in — the kill chain at a " +
    "glance. Click one to open it in the Bundle tab.",

  "bundle-bundle-tab":
    "Every object in the bundle and the relationships between them, with one " +
    "procedure in focus at a time.",

  "bundle-layers":
    "Show or hide whole classes of object. Hidden ones drop out of the layout " +
    "entirely rather than sitting dimmed in place.",

  "relationship-type":
    "The verb on an edge — \"uses\", \"targets\", \"indicates\". It is what the " +
    "STIX relationship object records.",

  // ── Feedback ────────────────────────────────────────────────────────
  salience:
    "A usefulness score. Rules the analyst did not need to correct again " +
    "score higher; recent activity counts more than old, and often-seen " +
    "rules more than rare ones. Raising the minimum also hides rules not " +
    "yet scored.",
};

/**
 * Definition for a term. Throws on an unknown key in dev.
 *
 * Throwing is a development signal, not the real guard: the static scan in
 * glossary.test.js catches a bad key before it can reach anyone. Returning
 * undefined instead would render an ⓘ with an empty tooltip — help that
 * advertises itself and then says nothing, which is worse than no ⓘ at all.
 *
 * In production it returns "" and InfoDot renders nothing, so a key that
 * somehow slipped past the test costs a missing hint rather than a blank
 * tooltip or a white-screened review panel.
 */
export function hint(term) {
  const text = GLOSSARY[term];
  if (text === undefined) {
    if (import.meta.env?.DEV) {
      throw new Error(
        `glossary: unknown term "${term}". Add it to GLOSSARY in lib/glossary.js.`
      );
    }
    return "";
  }
  return text;
}
