import { Wllama } from "@wllama/wllama";
import wllamaWasmUrl from "@wllama/wllama/esm/wasm/wllama.wasm?url";
import type { OffscreenStatus } from "~/lib/messages";

// HTLM model on Hugging Face. wllama caches the download (Cache API/OPFS) after first load.
// q8 mirrors the README's Try-It snippet; swap MODEL_FILE for a smaller quant if published.
const MODEL_REPO = "espetro/htlm-lfm2.5-350m";
const MODEL_FILE = "lfm2.5-350m-mlx-q8.gguf";

let wllama: Wllama | null = null;
let state: OffscreenStatus = { status: "loading", pct: 0 };

function broadcast(message: unknown) {
  // No listener (e.g. side panel closed) rejects with "Receiving end does not exist" -- harmless.
  browser.runtime.sendMessage(message).catch(() => {});
}

async function handleComplete(prompt: string) {
  if (!wllama || state.status !== "ready") {
    broadcast({ type: "htlm-complete-error", message: "model not ready yet" });
    return;
  }
  try {
    let full = "";
    const chunks = await wllama.createCompletion({
      prompt, max_tokens: 128, temperature: 0, stop: ["<|im_end|>"], stream: true,
    });
    for await (const chunk of chunks) {
      const delta = chunk.choices?.[0]?.text;
      if (delta) { full += delta; broadcast({ type: "htlm-token", delta }); }
    }
    broadcast({ type: "htlm-complete-done", full });
  } catch (e) {
    broadcast({ type: "htlm-complete-error", message: e instanceof Error ? e.message : String(e) });
  }
}

browser.runtime.onMessage.addListener((message: any, _sender, sendResponse) => {
  if (message?.type === "htlm-get-status") {
    sendResponse(state);
    return;
  }
  if (message?.type === "htlm-complete") {
    handleComplete(message.prompt);
    return;
  }
});

(async () => {
  try {
    wllama = new Wllama({ default: wllamaWasmUrl });
    await wllama.loadModelFromHF(
      { repo: MODEL_REPO, file: MODEL_FILE },
      {
        n_ctx: 2048,
        progressCallback: ({ loaded, total }) => {
          const pct = total ? Math.round((loaded / total) * 100) : 0;
          state = { status: "loading", pct };
          broadcast({ type: "htlm-load-progress", pct });
        },
      }
    );
    state = { status: "ready" };
    broadcast({ type: "htlm-load-ready" });
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    state = { status: "error", message };
    broadcast({ type: "htlm-load-error", message });
  }
})();
