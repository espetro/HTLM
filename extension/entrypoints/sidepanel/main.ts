import { SYSTEM_PROMPT, buildPrompt, renderPageForModel, extractAction } from "~/lib/grounding";
import type { PageSchema } from "~/lib/grounding";
import type { OffscreenBroadcast, OffscreenStatus } from "~/lib/messages";

const $ = <T extends HTMLElement>(id: string) => document.getElementById(id) as T;
const statusEl = $<HTMLDivElement>("status");
const instructionEl = $<HTMLTextAreaElement>("instruction");
const groundBtn = $<HTMLButtonElement>("ground-btn");
const streamEl = $<HTMLPreElement>("stream");
const parsedEl = $<HTMLPreElement>("parsed");

function setStatus(msg: string, cls?: "ok" | "err") {
  statusEl.textContent = msg;
  statusEl.className = "status" + (cls ? " " + cls : "");
}

async function activeTabId(): Promise<number> {
  const [tab] = await browser.tabs.query({ active: true, currentWindow: true });
  if (!tab?.id) throw new Error("no active tab");
  if (/^(chrome|edge|about|chrome-extension):/.test(tab.url || "")) {
    throw new Error("can't ground a browser-internal page -- open a normal website first");
  }
  return tab.id;
}

async function extractPage(tabId: number): Promise<PageSchema> {
  // Inject the content script on demand (idempotent guard inside it), then ask it for the page.
  await browser.scripting.executeScript({ target: { tabId }, files: ["/content-scripts/content.js"] });
  return browser.tabs.sendMessage(tabId, { type: "htlm-extract" });
}

// Set once a ground request is in flight so the "-done"/"-error" broadcast from the
// offscreen document knows which tab to highlight the predicted element on.
let groundTabId: number | null = null;
let streamed = "";

groundBtn.addEventListener("click", async () => {
  const instruction = instructionEl.value.trim();
  if (!instruction) return;
  groundBtn.disabled = true;
  streamed = "";
  streamEl.textContent = "";
  parsedEl.textContent = "(reading page + grounding…)";
  parsedEl.classList.remove("bad");
  try {
    const tabId = await activeTabId();
    const page = await extractPage(tabId);
    if (!page?.elements?.length) throw new Error("no interactive elements found on the page");
    await browser.tabs.sendMessage(tabId, { type: "htlm-clear" });

    groundTabId = tabId;
    const prompt = buildPrompt(SYSTEM_PROMPT, `${instruction}\n\n${renderPageForModel(page)}`);
    await browser.runtime.sendMessage({ type: "htlm-complete", prompt });
  } catch (e) {
    parsedEl.textContent = "ERROR: " + (e instanceof Error ? e.message : String(e));
    parsedEl.classList.add("bad");
    setStatus("Error grounding this page.", "err");
    groundBtn.disabled = false;
  }
});

async function handleCompletionDone(full: string) {
  const tabId = groundTabId;
  try {
    if (tabId == null) throw new Error("no active grounding request");
    const action = extractAction(full);
    if (!action) {
      parsedEl.textContent = "Could not parse a JSON action from the output above.";
      parsedEl.classList.add("bad");
      return;
    }
    parsedEl.textContent = JSON.stringify(action, null, 2);
    if (typeof action.index === "number") {
      const res = await browser.tabs.sendMessage(tabId, { type: "htlm-highlight", index: action.index });
      if (!res?.ok) setStatus(`Predicted index ${action.index} not found on the live page.`, "err");
      else setStatus(`Highlighted element #${action.index} on the page.`, "ok");
    } else {
      setStatus(`Action: ${action.type} (no element to highlight).`, "ok");
    }
  } catch (e) {
    parsedEl.textContent = "ERROR: " + (e instanceof Error ? e.message : String(e));
    parsedEl.classList.add("bad");
    setStatus("Error grounding this page.", "err");
  } finally {
    groundBtn.disabled = false;
  }
}

function applyStatus(status: OffscreenStatus) {
  if (status.status === "ready") {
    setStatus("Model ready. Type an instruction and ground the page.", "ok");
    groundBtn.disabled = false;
  } else if (status.status === "error") {
    setStatus("Model load failed: " + status.message, "err");
  } else {
    setStatus(`Loading model (${status.pct}%)…`);
  }
}

browser.runtime.onMessage.addListener((message: OffscreenBroadcast) => {
  switch (message?.type) {
    case "htlm-load-progress":
      applyStatus({ status: "loading", pct: message.pct });
      return;
    case "htlm-load-ready":
      applyStatus({ status: "ready" });
      return;
    case "htlm-load-error":
      applyStatus({ status: "error", message: message.message });
      return;
    case "htlm-token":
      streamed += message.delta;
      streamEl.textContent = streamed;
      return;
    case "htlm-complete-done":
      handleCompletionDone(message.full);
      return;
    case "htlm-complete-error":
      parsedEl.textContent = "ERROR: " + message.message;
      parsedEl.classList.add("bad");
      setStatus("Error grounding this page.", "err");
      groundBtn.disabled = false;
      return;
  }
});

(async () => {
  setStatus("Connecting to model…");
  await browser.runtime.sendMessage({ type: "htlm-ensure-offscreen" });
  const status: OffscreenStatus = await browser.runtime.sendMessage({ type: "htlm-get-status" });
  applyStatus(status);
})();
