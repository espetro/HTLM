// Open the side panel when the toolbar icon is clicked. Opening it this way also grants
// activeTab for the current tab, which the panel needs to inject the content script.
export default defineBackground(() => {
  browser.runtime.onInstalled.addListener(() => {
    browser.sidePanel?.setPanelBehavior?.({ openPanelOnActionClick: true }).catch(() => {});
  });

  browser.runtime.onMessage.addListener((message: any, _sender, sendResponse) => {
    if (message?.type === "htlm-ensure-offscreen") {
      ensureOffscreenDocument()
        .then(() => sendResponse({ ok: true }))
        .catch((e) => sendResponse({ ok: false, error: String(e) }));
      return true; // keep the message channel open for the async response
    }
  });
});

// The offscreen document hosts wllama so the model stays loaded (worker + WASM memory)
// across side panel opens/closes instead of reloading on every panel open. Idempotent and
// race-safe: getContexts is the source of truth (survives service worker restarts), and
// `creating` collapses concurrent calls within a single service worker lifetime.
let creating: Promise<void> | null = null;

async function ensureOffscreenDocument() {
  const existing = await chrome.runtime.getContexts({
    contextTypes: [chrome.runtime.ContextType.OFFSCREEN_DOCUMENT],
  });
  if (existing.length > 0) return;
  if (!creating) {
    creating = chrome.offscreen
      .createDocument({
        url: chrome.runtime.getURL("offscreen.html"),
        reasons: [chrome.offscreen.Reason.WORKERS],
        justification: "Keeps the wllama WASM model resident across side panel opens/closes.",
      })
      .finally(() => {
        creating = null;
      });
  }
  await creating;
}
