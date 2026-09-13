/**
 * Tests for the AI reviewer's suggestion logic.
 *
 * `isBulkAcceptable` is the load-bearing rule here, and it is a product
 * decision more than a technical one. Assist mode's whole value rests on the
 * analyst actually reviewing: if one click could accept every recommendation,
 * the mode becomes autopilot wearing a human's badge, and the override signal
 * that decides whether autopilot is ever safe would never be generated.
 *
 * So the filter is deliberately narrow, and these tests pin it there.
 */

import { describe, expect, it } from "vitest";

import {
  applyRecommendation,
  edgeRecKey,
  indexByChunk,
  indexByDraft,
  indexByEntity,
  indexByPromotion,
  isBulkAcceptable,
  pendingEdgeRecs,
  pickRestoredDecision,
  rateLabel,
  restoreChunkSnapshot,
  verdictForRecommendation,
} from "../reviewerSuggestions";

const rec = (over = {}) => ({
  entity_id: "e1",
  action: "remove",
  confidence: "high",
  rationale: "because",
  evidence_quote: "the actor deployed it",
  ...over,
});

describe("isBulkAcceptable", () => {
  it("accepts a grounded high-confidence recommendation", () => {
    expect(isBulkAcceptable(rec())).toBe(true);
  });

  it("excludes medium and low", () => {
    expect(isBulkAcceptable(rec({ confidence: "medium" }))).toBe(false);
    expect(isBulkAcceptable(rec({ confidence: "low" }))).toBe(false);
  });

  it("excludes anything whose quote the report does not support", () => {
    // Belt-and-braces: the backend already forces such a recommendation to
    // "low", so this should be unreachable. It exists so a future path that
    // sets confidence without re-running the grounding check cannot slip an
    // unverified recommendation into a one-click bulk accept.
    expect(isBulkAcceptable(rec({ quote_unsupported: true }))).toBe(false);
  });

  it("excludes a missing recommendation rather than throwing", () => {
    expect(isBulkAcceptable(null)).toBe(false);
    expect(isBulkAcceptable(undefined)).toBe(false);
  });

  it("treats an unknown confidence value as not acceptable", () => {
    // Fails closed: a value we do not recognise must not be bulk-applied.
    expect(isBulkAcceptable(rec({ confidence: "certain" }))).toBe(false);
    expect(isBulkAcceptable(rec({ confidence: undefined }))).toBe(false);
  });
});

describe("indexByEntity", () => {
  it("keys recommendations by entity_id", () => {
    const out = indexByEntity({ entities: [rec(), rec({ entity_id: "e2" })] });
    expect(Object.keys(out).sort()).toEqual(["e1", "e2"]);
  });

  it("skips entries with no entity_id", () => {
    const out = indexByEntity({ entities: [rec({ entity_id: undefined })] });
    expect(out).toEqual({});
  });

  it("survives a null payload", () => {
    expect(indexByEntity(null)).toEqual({});
  });
});

describe("applyRecommendation", () => {
  const decision = {
    entity_id: "e1",
    action: "approve",
    edited_value: "original",
    edited_type: "malware",
    edited_role: "",
    rationale: "",
  };

  it("does not mutate the decision it is given", () => {
    const copy = { ...decision };
    applyRecommendation(decision, rec());
    expect(decision).toEqual(copy);
  });

  it("takes the recommended action", () => {
    expect(applyRecommendation(decision, rec()).action).toBe("remove");
  });

  it("leaves fields the recommendation did not specify", () => {
    // An "edit" naming a value but no type must not blank the type — the
    // extractor's classification stands unless the reviewer disputes it.
    const out = applyRecommendation(
      decision,
      rec({ action: "edit", edited_value: "corrected", edited_type: undefined }),
    );
    expect(out.edited_value).toBe("corrected");
    expect(out.edited_type).toBe("malware");
  });

  it("fills an EMPTY rationale with the reviewer's reason", () => {
    expect(applyRecommendation(decision, rec()).rationale).toBe("because");
  });

  it("never overwrites a rationale the analyst typed", () => {
    // Theirs is the more authoritative reason for the decision, and
    // silently replacing text someone wrote is the surprising behaviour —
    // the same data-loss class already fixed for technique lists.
    const typed = { ...decision, rationale: "my own reasoning" };
    expect(applyRecommendation(typed, rec()).rationale).toBe("my own reasoning");
  });

  it("returns the decision unchanged when there is no recommendation", () => {
    expect(applyRecommendation(decision, null)).toBe(decision);
  });
});

describe("verdictForRecommendation", () => {
  it("maps remove to discard", () => {
    expect(verdictForRecommendation({ action: "remove" })).toBe("discard");
  });

  it("splits reject by reason", () => {
    // Load-bearing: rechunk re-runs chunking for the WHOLE source while
    // remap only re-runs technique extraction. Getting this backwards
    // discards far more work than the reviewer asked for.
    expect(
      verdictForRecommendation({ action: "reject", reject_reason: "bad_chunk_boundary" })
    ).toBe("rechunk");
    expect(
      verdictForRecommendation({ action: "reject", reject_reason: "wrong_technique" })
    ).toBe("remap");
  });

  it("defaults a reason-less reject to the cheaper rerun", () => {
    expect(verdictForRecommendation({ action: "reject" })).toBe("remap");
  });

  it("treats approve and edit alike — both ship the draft", () => {
    expect(verdictForRecommendation({ action: "approve" })).toBe("approve");
    expect(verdictForRecommendation({ action: "edit" })).toBe("approve");
  });

  it("falls back to approve on junk rather than discarding", () => {
    expect(verdictForRecommendation(null)).toBe("approve");
    expect(verdictForRecommendation({ action: "nonsense" })).toBe("approve");
  });
});

describe("indexByDraft / indexByPromotion", () => {
  it("keys drafts by draft_id", () => {
    const out = indexByDraft({ drafts: [{ draft_id: "d1" }, { draft_id: "d2" }] });
    expect(Object.keys(out).sort()).toEqual(["d1", "d2"]);
  });

  it("keys promotions the same way the panel keys its promoted set", () => {
    const out = indexByPromotion({
      promotions: [{ chunk_id: "c1", technique_id: "T1219" }],
    });
    expect(out["c1|T1219"]).toBeTruthy();
  });

  it("skips incomplete records and survives null", () => {
    expect(indexByDraft({ drafts: [{}] })).toEqual({});
    expect(indexByPromotion({ promotions: [{ chunk_id: "c1" }] })).toEqual({});
    expect(indexByDraft(null)).toEqual({});
    expect(indexByPromotion(null)).toEqual({});
  });
});

describe("pickRestoredDecision", () => {
  const snapshot = {
    entity_id: "e1", action: "edit",
    edited_value: "CORRECTED BY HAND", rationale: "my own reasoning",
  };
  const fallback = { entity_id: "e1", action: "approve", edited_value: "orig" };

  it("restores the analyst's pre-apply decision when one was captured", () => {
    expect(pickRestoredDecision(snapshot, fallback)).toBe(snapshot);
  });

  it("falls back to a clean rebuild when nothing was ever applied", () => {
    expect(pickRestoredDecision(undefined, fallback)).toBe(fallback);
    expect(pickRestoredDecision(null, fallback)).toBe(fallback);
  });

  it("does not treat an empty-ish snapshot as absent", () => {
    // A decision object is always truthy, but guarding on truthiness rather
    // than undefined would be a latent trap for any future caller.
    const empty = {};
    expect(pickRestoredDecision(empty, fallback)).toBe(empty);
  });
});

describe("undo semantics (shared by both gates)", () => {
  // Gate 0 previously rebuilt the decision from the extractor's entity on
  // undo, destroying any value, type, role or rationale the analyst had set
  // by hand. Gate 1 had the same shape for technique lists. The rule both
  // now follow: undo restores the PRE-APPLY state, never the origin.
  it("a pre-apply snapshot round-trips the analyst's own edits", () => {
    const analystEdited = {
      entity_id: "e1",
      action: "edit",
      edited_value: "CORRECTED BY HAND",
      edited_type: "tool",
      rationale: "my own reasoning",
    };
    // Apply leaves the analyst's rationale alone (see above) but changes
    // action/value; the snapshot is what undo must give back.
    const afterApply = applyRecommendation(analystEdited, {
      action: "remove",
      rationale: "AI reason",
    });
    expect(afterApply.action).toBe("remove");
    expect(afterApply).not.toBe(analystEdited);

    // Restoring the snapshot returns every hand-set field.
    expect(analystEdited.edited_value).toBe("CORRECTED BY HAND");
    expect(analystEdited.rationale).toBe("my own reasoning");
    expect(analystEdited.action).toBe("edit");
  });
});

describe("chunk gate", () => {
  it("indexes chunk recommendations by chunk_id", () => {
    const out = indexByChunk({
      chunks: [
        { chunk_id: "chk-a", action: "drop" },
        { chunk_id: "chk-b", action: "approve" },
        { action: "drop" }, // no id — must not land under `undefined`
      ],
    });
    expect(Object.keys(out).sort()).toEqual(["chk-a", "chk-b"]);
    expect(out["chk-a"].action).toBe("drop");
  });

  it("returns an empty index when this gate had no AI review", () => {
    expect(indexByChunk(null)).toEqual({});
    expect(indexByChunk({})).toEqual({});
  });

  it("keys an edge recommendation on its action as well as its endpoints", () => {
    // "add A->B" and "remove A->B" are opposite advice about the same pair.
    // Keying on the pair alone would let one stand in for the other, so an
    // applied add would silently mark a remove as applied too.
    const add = { action: "add", from_chunk_id: "a", to_chunk_id: "b" };
    const remove = { action: "remove", from_chunk_id: "a", to_chunk_id: "b" };
    expect(edgeRecKey(add)).not.toBe(edgeRecKey(remove));
  });

  describe("pendingEdgeRecs", () => {
    // An edge the analyst's graph already satisfies is a description, not a
    // recommendation. Showing it invites them to "apply" a no-op and then
    // wonder why nothing changed.
    const payload = {
      edges: [
        { action: "add", from_chunk_id: "a", to_chunk_id: "b" },
        { action: "remove", from_chunk_id: "c", to_chunk_id: "d" },
      ],
    };

    it("hides an add whose edge already exists", () => {
      const out = pendingEdgeRecs(payload, new Set(["a->b"]));
      expect(out.map((r) => r.action)).toEqual([]);
    });

    it("hides a remove whose edge is already gone", () => {
      const out = pendingEdgeRecs(payload, new Set([]));
      expect(out.map((r) => r.action)).toEqual(["add"]);
    });

    it("shows both when neither is satisfied", () => {
      const out = pendingEdgeRecs(payload, new Set(["c->d"]));
      expect(out.map((r) => r.action).sort()).toEqual(["add", "remove"]);
    });

    it("is empty when there was no AI review", () => {
      expect(pendingEdgeRecs(null, new Set())).toEqual([]);
    });
  });

  it("never bulk-accepts a low-confidence or unverified chunk suggestion", () => {
    // Same rule as every other gate. Repeated here because the chunk gate is
    // the one where a wrong bulk accept is least recoverable: drop a chunk
    // and the procedure is gone from the bundle with nothing downstream able
    // to notice it was ever there.
    expect(isBulkAcceptable({ chunk_id: "c", action: "drop", confidence: "high" })).toBe(true);
    expect(isBulkAcceptable({ chunk_id: "c", action: "drop", confidence: "medium" })).toBe(false);
    expect(
      isBulkAcceptable({
        chunk_id: "c", action: "drop", confidence: "high", quote_unsupported: true,
      }),
    ).toBe(false);
  });
});

describe("chunk-gate undo restores the analyst's work", () => {
  // The rule the whole codebase now follows: undo gives back the PRE-APPLY
  // state, never a clean rebuild. Gate 0 shipped the other behaviour once
  // and Gate 1 shipped it for technique lists — in both cases declining one
  // AI suggestion silently destroyed unrelated hand edits.
  //
  // The chunk gate raises the stakes: a single chunk decision is spread
  // across the edits map, the dropped set AND the merge groups, so an undo
  // that forgets one channel leaves the canvas in a state the analyst never
  // chose and cannot see.
  const analystState = () => ({
    edits: { "chk-a": { text: "MY OWN WORDING" }, "chk-z": { text: "untouched" } },
    droppedIds: new Set(["chk-a", "chk-q"]),
    mergeGroups: { "chk-a": ["chk-b"], "chk-y": ["chk-x"] },
  });

  it("gives back every channel the analyst had set", () => {
    const before = analystState();
    const snapshot = {
      edits: before.edits["chk-a"],
      dropped: true,
      merge: before.mergeGroups["chk-a"],
    };
    // Pretend the AI's advice overwrote all three.
    const after = {
      edits: { ...before.edits, "chk-a": { text: "AI WORDING" } },
      droppedIds: new Set(["chk-q"]),
      mergeGroups: { "chk-y": ["chk-x"] },
    };

    const out = restoreChunkSnapshot("chk-a", snapshot, after);
    expect(out.edits["chk-a"]).toEqual({ text: "MY OWN WORDING" });
    expect(out.droppedIds.has("chk-a")).toBe(true);
    expect(out.mergeGroups["chk-a"]).toEqual(["chk-b"]);
  });

  it("clears only when the analyst had nothing there", () => {
    const snapshot = { edits: undefined, dropped: false, merge: undefined };
    const out = restoreChunkSnapshot("chk-a", snapshot, analystState());
    expect("chk-a" in out.edits).toBe(false);
    expect(out.droppedIds.has("chk-a")).toBe(false);
    expect("chk-a" in out.mergeGroups).toBe(false);
  });

  it("leaves every other chunk alone", () => {
    const out = restoreChunkSnapshot("chk-a", undefined, analystState());
    expect(out.edits["chk-z"]).toEqual({ text: "untouched" });
    expect(out.droppedIds.has("chk-q")).toBe(true);
    expect(out.mergeGroups["chk-y"]).toEqual(["chk-x"]);
  });

  it("does not mutate the state it was given", () => {
    const before = analystState();
    const out = restoreChunkSnapshot("chk-a", undefined, before);
    expect(before.edits["chk-a"]).toEqual({ text: "MY OWN WORDING" });
    expect(before.droppedIds.has("chk-a")).toBe(true);
    expect(out.edits).not.toBe(before.edits);
    expect(out.droppedIds).not.toBe(before.droppedIds);
    expect(out.mergeGroups).not.toBe(before.mergeGroups);
  });

  it("a missing snapshot means nothing was applied, so nothing is restored", () => {
    // Defensive: an apply that never recorded a snapshot must not leave the
    // recommendation permanently stuck in the applied state.
    const out = restoreChunkSnapshot("chk-a", undefined, analystState());
    expect("chk-a" in out.edits).toBe(false);
  });
});

describe("rateLabel — a rate never travels without its n", () => {
  // This is the readout's single easiest way to mislead. "100%" reads as a
  // settled fact; "100% (3/3)" reads as three data points. The number decides
  // whether a gate may run unattended, so the sample size is not decoration —
  // which is why both parts come from one function rather than two call sites
  // that could drift apart.
  it("returns the count alongside every percentage", () => {
    expect(rateLabel(3, 4)).toEqual({ pct: 75, n: "3/4" });
    expect(rateLabel(40, 42)).toEqual({ pct: 95, n: "40/42" });
  });

  it("refuses to produce a percentage when there is nothing to rate", () => {
    // A gate where the reviewer recommended nothing is an absence of evidence.
    // Rendering it as 0% would read as the reviewer being wrong about
    // everything — the opposite of what happened.
    expect(rateLabel(0, 0).pct).toBeNull();
    expect(rateLabel(0, 0).n).toBe("no data");
  });

  it("distinguishes no-data from genuine total disagreement", () => {
    expect(rateLabel(0, 5)).toEqual({ pct: 0, n: "0/5" });
  });

  it("never yields a percentage without a usable count", () => {
    for (const [a, t] of [[0, 0], [1, 1], [0, 3], [7, 9], [40, 42]]) {
      const { pct, n } = rateLabel(a, t);
      if (pct !== null) expect(n).toMatch(/^\d+\/\d+$/);
    }
  });
});
