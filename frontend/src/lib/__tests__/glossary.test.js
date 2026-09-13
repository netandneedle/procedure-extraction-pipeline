/**
 * Tests for the UI glossary.
 *
 * The load-bearing one is the static scan at the bottom. A component asking for
 * a term that does not exist is the failure this whole design is meant to avoid
 * — an ⓘ that advertises help and then shows nothing — and it is invisible at
 * runtime unless something looks for it. So something does, statically, over
 * the real source tree, before any of it renders.
 */
import { readdirSync, readFileSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

import { describe, expect, it } from "vitest";

import { GLOSSARY, MAX_HINT_LENGTH, hint } from "../glossary";

const SRC = fileURLToPath(new URL("../../", import.meta.url));

/** Every .jsx/.js file under src/, minus tests. */
function sourceFiles(dir = SRC, out = []) {
  for (const name of readdirSync(dir)) {
    const full = join(dir, name);
    if (statSync(full).isDirectory()) {
      if (name !== "__tests__" && name !== "node_modules") sourceFiles(full, out);
    } else if (/\.jsx?$/.test(name)) {
      out.push(full);
    }
  }
  return out;
}

/**
 * Strip comments so a `term="..."` written in a doc comment — as this file's
 * own header and InfoDot's do — is not mistaken for a real call site.
 *
 * Only block comments and whole-line `//` comments, never a trailing `//`:
 * that would eat "https://" inside a string literal. Erring toward keeping
 * text means the scan can over-report, which shows up as a visible failure,
 * rather than under-report, which would silently defeat the check.
 */
function stripComments(text) {
  return text
    .replace(/\/\*[\s\S]*?\*\//g, "")
    .replace(/^\s*\/\/.*$/gm, "");
}

/**
 * Every glossary term the source refers to, by either route.
 *
 * `<InfoDot term="x" />` is the usual one. `hint("x")` is the other: a few
 * places want the wording without the glyph — a compact card chip, a tab
 * button where an ⓘ would be a click target that does not switch tabs. Both
 * are real references, so both count, and catching `hint("typo")` here makes
 * this scan a static guard rather than leaving it to throw at render.
 *
 * `hintTerm: "x"` covers indirection: per-gate hints are declared on GATES in
 * lib/gates.js and passed through a derived map. Whatever the indirection, the
 * term must appear as a LITERAL somewhere, or this scan cannot vouch for it —
 * a term assembled at runtime (`term={`gate-${key}`}`) would be invisible
 * here, which is exactly why gates.js spells all four out.
 */
const TERM_REFS = [
  /\b(?:term|hintTerm)\s*[=:]\s*\{?\s*["']([^"']+)["']/g,
  /\bhint\(\s*["']([^"']+)["']/g,
];

function referencedTerms() {
  const found = new Map(); // term -> file it came from
  for (const file of sourceFiles()) {
    const text = stripComments(readFileSync(file, "utf8"));
    for (const re of TERM_REFS) {
      for (const m of text.matchAll(re)) {
        if (!found.has(m[1])) found.set(m[1], file.slice(SRC.length));
      }
    }
  }
  return found;
}

describe("GLOSSARY entries", () => {
  const entries = Object.entries(GLOSSARY);

  it("has entries", () => {
    expect(entries.length).toBeGreaterThan(0);
  });

  it.each(entries)("%s is non-empty and trimmed", (_term, text) => {
    expect(typeof text).toBe("string");
    expect(text.trim().length).toBeGreaterThan(0);
    expect(text).toBe(text.trim());
  });

  // Writing rule 3. A definition that needs more than this is a signal the
  // control is doing too much, not that the cap is too low.
  it.each(entries)("%s fits in a native tooltip", (_term, text) => {
    expect(text.length).toBeLessThanOrEqual(MAX_HINT_LENGTH);
  });

  // Writing rule 2 — a definition that explains one internal term using
  // another has explained nothing. These are the words this pass exists to
  // remove from the analyst's path; they must not reappear inside the fix.
  it.each(entries)("%s does not define jargon with jargon", (term, text) => {
    const banned = /\b(bucket|C\+A\+D|superstep|checkpoint|LangGraph|chunk_id|SRO|SDO|SCO)\b/i;
    // "denylist" and "precedes" are themselves defined terms, so they are
    // allowed to appear in their own entry and nowhere else.
    const selfReference = new RegExp(`\\b${term.replace(/-/g, "[- ]?")}\\b`, "i");
    const offending = text.match(banned);
    if (offending && !selfReference.test(offending[0])) {
      throw new Error(`"${term}" defines jargon with jargon: "${offending[0]}"`);
    }
  });

  it("has no duplicate definitions", () => {
    const texts = entries.map(([, t]) => t);
    expect(new Set(texts).size).toBe(texts.length);
  });
});

describe("hint()", () => {
  it("returns the definition for a known term", () => {
    expect(hint("precedes")).toBe(GLOSSARY.precedes);
  });

  it("throws on an unknown term rather than returning undefined", () => {
    // Silently returning undefined would render an ⓘ with an empty tooltip.
    expect(() => hint("no-such-term")).toThrow(/unknown term/);
  });
});

describe("every term a component asks for is defined", () => {
  // THE guard. Mutation-check: point any <InfoDot term="..."> at a bogus key
  // and this must fail.
  it("has no dangling term references", () => {
    const dangling = [...referencedTerms()]
      .filter(([term]) => !(term in GLOSSARY))
      .map(([term, file]) => `${term} (${file})`);
    expect(dangling).toEqual([]);
  });

  it("has no unused definitions", () => {
    const used = new Set(referencedTerms().keys());
    const unused = Object.keys(GLOSSARY).filter((t) => !used.has(t));
    expect(unused).toEqual([]);
  });
});
