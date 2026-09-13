/**
 * Tests for the chunk-review canvas's pure helpers.
 *
 * These three carry the canvas's trickiest rules, and each has a documented
 * failure mode behind it:
 *
 *   computeLiveEdges   — replay semantics. `addedKeys` must be derived as a
 *                        set difference AFTER replay, not accumulated during
 *                        it; the incremental version painted green "added"
 *                        styling on edges that already existed (H1 in the
 *                        frontend review).
 *   buildSourceSegments— overlapping <mark> spans are dropped rather than
 *                        nested, because nested marks make the click target
 *                        ambiguous.
 *   buildSubmitPayload — the on-wire diff. Only touched chunks appear;
 *                        opposing edge ops collapse; synthetic ids are
 *                        filtered because the backend cannot resolve them.
 *
 * Until now the frontend had no tests at all — `npm run build` was the only
 * check, and it cannot see any of this.
 */

import { describe, expect, it } from "vitest";

import {
  buildSourceSegments,
  buildSubmitPayload,
  computeLiveEdges,
} from "../ChunkReviewCanvas.jsx";

const chunk = (id, precedes = []) => ({ chunk_id: id, precedes_ids: precedes });

describe("computeLiveEdges", () => {
  const chunks = [chunk("a", ["b"]), chunk("b", ["c"]), chunk("c")];

  it("derives the original edge set from precedes_ids", () => {
    const { liveKeys, addedKeys } = computeLiveEdges(chunks, []);
    expect([...liveKeys].sort()).toEqual(["a->b", "b->c"]);
    expect([...addedKeys]).toEqual([]);
  });

  it("marks an analyst-added edge as added", () => {
    const { liveKeys, addedKeys } = computeLiveEdges(chunks, [
      { action: "add", from: "a", to: "c" },
    ]);
    expect(liveKeys.has("a->c")).toBe(true);
    expect([...addedKeys]).toEqual(["a->c"]);
  });

  it("removes an original edge", () => {
    const { liveKeys, addedKeys } = computeLiveEdges(chunks, [
      { action: "remove", from: "a", to: "b" },
    ]);
    expect(liveKeys.has("a->b")).toBe(false);
    expect([...addedKeys]).toEqual([]);
  });

  it("collapses add-then-remove of the same edge", () => {
    const { liveKeys, addedKeys } = computeLiveEdges(chunks, [
      { action: "add", from: "a", to: "c" },
      { action: "remove", from: "a", to: "c" },
    ]);
    expect(liveKeys.has("a->c")).toBe(false);
    expect([...addedKeys]).toEqual([]);
  });

  it("does not paint an ORIGINAL edge as added when it is toggled off and on", () => {
    // The H1 regression: accumulating `added` during replay marks a->b as
    // added on the re-add, even though it was there all along, so the
    // analyst sees a green "you added this" edge they never added.
    const { liveKeys, addedKeys } = computeLiveEdges(chunks, [
      { action: "remove", from: "a", to: "b" },
      { action: "add", from: "a", to: "b" },
    ]);
    expect(liveKeys.has("a->b")).toBe(true);
    expect([...addedKeys]).toEqual([]);
  });

  it("ignores edges pointing at chunks that are not present", () => {
    const { liveKeys } = computeLiveEdges([chunk("a", ["ghost"])], []);
    expect([...liveKeys]).toEqual([]);
  });

  it("ignores ops referencing chunks that are not present", () => {
    const { liveKeys } = computeLiveEdges(chunks, [
      { action: "add", from: "a", to: "ghost" },
    ]);
    expect([...liveKeys].sort()).toEqual(["a->b", "b->c"]);
  });

  it("tolerates a chunk with no precedes_ids field", () => {
    expect(() => computeLiveEdges([{ chunk_id: "a" }], [])).not.toThrow();
  });
});

describe("buildSourceSegments", () => {
  const text = "AAAA BBBB CCCC DDDD";

  it("returns nothing for empty text", () => {
    expect(buildSourceSegments("", null, [{ chunk_id: "a", source_span: [0, 4] }])).toEqual([]);
  });

  it("marks a chunk at its span", () => {
    const segs = buildSourceSegments(text, null, [{ chunk_id: "a", source_span: [0, 4] }]);
    const marked = segs.filter((s) => s.chunk_id);
    expect(marked).toHaveLength(1);
    expect(marked[0].chunk_id).toBe("a");
    expect(text.slice(0, 4)).toBe("AAAA");
  });

  it("keeps disjoint spans in document order", () => {
    const segs = buildSourceSegments(text, null, [
      { chunk_id: "second", source_span: [10, 14] },
      { chunk_id: "first", source_span: [0, 4] },
    ]);
    expect(segs.filter((s) => s.chunk_id).map((s) => s.chunk_id)).toEqual(["first", "second"]);
  });

  it("drops an overlapping span rather than nesting marks", () => {
    const segs = buildSourceSegments(text, null, [
      { chunk_id: "outer", source_span: [0, 14] },
      { chunk_id: "inner", source_span: [5, 9] },
    ]);
    expect(segs.filter((s) => s.chunk_id).map((s) => s.chunk_id)).toEqual(["outer"]);
  });

  it.each([
    ["negative start", [-1, 4]],
    ["end before start", [8, 4]],
    ["zero-length", [4, 4]],
    ["end past the document", [0, 9999]],
  ])("discards an invalid span: %s", (_label, span) => {
    const segs = buildSourceSegments(text, null, [{ chunk_id: "a", source_span: span }]);
    expect(segs.filter((s) => s.chunk_id)).toEqual([]);
  });

  it("covers the whole document across plain and marked segments", () => {
    // Lossless: concatenating every segment must reproduce the source byte
    // for byte, or the source pane silently drops text.
    const segs = buildSourceSegments(text, null, [{ chunk_id: "a", source_span: [5, 9] }]);
    expect(segs.map((s) => s.content).join("")).toBe(text);
  });
});

describe("buildSubmitPayload", () => {
  const base = {
    originalChunks: [chunk("a", ["b"]), chunk("b"), chunk("c")],
    edits: {},
    droppedIds: new Set(),
    addedChunks: [],
    edgeOps: [],
    operatorOverrides: {},
    conditionEdits: {},
  };

  it("emits a merge decision naming the absorbed chunks", () => {
    // One analyst gesture; the backend expands it into edit-survivor +
    // drop-absorbed + edge rewire.
    const out = buildSubmitPayload({ ...base, mergeGroups: { a: ["b"] } });
    expect(out.decisions).toEqual([
      { chunk_id: "a", action: "merge", merge_with: ["b"] },
    ]);
  });

  it("does not emit a separate drop for an absorbed chunk", () => {
    // The backend drops it as part of the merge; a second decision would be
    // redundant and could race the rewiring.
    const out = buildSubmitPayload({
      ...base, mergeGroups: { a: ["b"] }, droppedIds: new Set(["b"]),
    });
    expect(out.decisions).toEqual([
      { chunk_id: "a", action: "merge", merge_with: ["b"] },
    ]);
  });

  it("carries analyst-written merged text on the merge decision", () => {
    const out = buildSubmitPayload({
      ...base,
      mergeGroups: { a: ["b"] },
      edits: { a: { text: "combined wording" } },
    });
    expect(out.decisions).toEqual([{
      chunk_id: "a", action: "merge", merge_with: ["b"],
      edits: { text: "combined wording" },
    }]);
  });

  it("sends nothing when the analyst touched nothing", () => {
    // The backend safe-default approves every unmentioned chunk, so an
    // untouched review must not enumerate them.
    expect(buildSubmitPayload(base)).toEqual({});
  });

  it("emits a drop decision", () => {
    const out = buildSubmitPayload({ ...base, droppedIds: new Set(["a"]) });
    expect(out.decisions).toEqual([{ chunk_id: "a", action: "drop" }]);
  });

  it("emits an edit decision carrying only the edited fields", () => {
    const out = buildSubmitPayload({ ...base, edits: { b: { text: "new" } } });
    expect(out.decisions).toEqual([
      { chunk_id: "b", action: "edit", edits: { text: "new" } },
    ]);
  });

  it("prefers drop over edit for the same chunk", () => {
    const out = buildSubmitPayload({
      ...base, edits: { a: { text: "x" } }, droppedIds: new Set(["a"]),
    });
    expect(out.decisions).toEqual([{ chunk_id: "a", action: "drop" }]);
  });

  it("sends net edge additions only", () => {
    const out = buildSubmitPayload({
      ...base, edgeOps: [{ action: "add", from: "b", to: "c" }],
    });
    expect(out.edges).toEqual([{ action: "add", from: "b", to: "c" }]);
  });

  it("collapses an add then remove into no edge change", () => {
    const out = buildSubmitPayload({
      ...base,
      edgeOps: [
        { action: "add", from: "b", to: "c" },
        { action: "remove", from: "b", to: "c" },
      ],
    });
    expect(out.edges).toBeUndefined();
  });

  it("collapses removing then re-adding an ORIGINAL edge", () => {
    const out = buildSubmitPayload({
      ...base,
      edgeOps: [
        { action: "remove", from: "a", to: "b" },
        { action: "add", from: "a", to: "b" },
      ],
    });
    expect(out.edges).toBeUndefined();
  });

  it("emits a remove for an original edge the analyst deleted", () => {
    const out = buildSubmitPayload({
      ...base, edgeOps: [{ action: "remove", from: "a", to: "b" }],
    });
    expect(out.edges).toEqual([{ action: "remove", from: "a", to: "b" }]);
  });

  it("filters edges involving synthetic ids", () => {
    // The backend allocates real chunk_ids for added chunks, so it cannot
    // resolve an "add-1" reference. Documented v1 limitation.
    //
    // Note: two guards enforce this — an explicit isSyntheticId check and
    // the realIds membership test right after it. A synthetic id is never
    // in realIds, so either alone suffices and removing just one does not
    // fail this test. It pins the BEHAVIOUR, not a particular line;
    // removing both guards does fail it.
    const out = buildSubmitPayload({
      ...base, edgeOps: [{ action: "add", from: "a", to: "add-1" }],
    });
    expect(out.edges).toBeUndefined();
  });

  it("strips frontend-only fields from added chunks", () => {
    const out = buildSubmitPayload({
      ...base,
      addedChunks: [{ tmp_id: "add-1", _isAdded: true, text: "new chunk" }],
    });
    expect(out.added_chunks).toEqual([{ text: "new chunk" }]);
  });

  it("does not send a chunk that was added and then dropped", () => {
    const out = buildSubmitPayload({
      ...base,
      addedChunks: [{ tmp_id: "add-1", _isAdded: true, text: "new chunk" }],
      droppedIds: new Set(["add-1"]),
    });
    expect(out.added_chunks).toBeUndefined();
  });
});
