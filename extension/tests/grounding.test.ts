import { describe, expect, it } from "vitest";
import { buildPrompt, extractAction, renderPageForModel, SYSTEM_PROMPT } from "~/lib/grounding";

describe("buildPrompt", () => {
  it("produces the exact ChatML string", () => {
    const prompt = buildPrompt("SYS", "USER");
    expect(prompt).toBe(
      "<|startoftext|><|im_start|>system\nSYS<|im_end|>\n<|im_start|>user\nUSER<|im_end|>\n<|im_start|>assistant\n"
    );
  });

  it("uses the real SYSTEM_PROMPT unmodified", () => {
    const prompt = buildPrompt(SYSTEM_PROMPT, "hi");
    expect(prompt.startsWith(`<|startoftext|><|im_start|>system\n${SYSTEM_PROMPT}<|im_end|>`)).toBe(true);
  });
});

describe("renderPageForModel", () => {
  it("strips metadata and serializes the rest", () => {
    const page = { url: "https://x", title: "t", elements: [], metadata: { extra: true } };
    const out = JSON.parse(renderPageForModel(page as never));
    expect(out).toEqual({ url: "https://x", title: "t", elements: [] });
  });
});

describe("extractAction", () => {
  it("parses a clean JSON action", () => {
    expect(extractAction('{"type":"click","index":3}')).toEqual({ type: "click", index: 3 });
  });

  it("parses the first JSON object out of noisy model text", () => {
    const text = 'Sure, here is the action:\n{"type":"click","index":3}\nDone.';
    expect(extractAction(text)).toEqual({ type: "click", index: 3 });
  });

  it("rejects garbage", () => {
    expect(extractAction("not json at all")).toBeNull();
    expect(extractAction("")).toBeNull();
  });

  it("rejects malformed JSON-looking text", () => {
    expect(extractAction('{"type":"click", index: 3}')).toBeNull();
  });
});
