import { describe, expect, it } from "vitest";

import { ellipsize } from "../strings";

describe("ellipsize", () => {
  it("returns short strings untouched and truncates long ones with an ellipsis", () => {
    expect(ellipsize("short", 10)).toBe("short");
    expect(ellipsize("a long label here", 6)).toBe("a long…");
    expect(ellipsize("", 3)).toBe("");
    expect(ellipsize(null, 3)).toBe("");
  });
});
