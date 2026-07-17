// Content script: live tab DOM -> HTLM page schema, + overlay highlight of a predicted index.
// registration: "runtime" -- not auto-injected; the side panel injects it on demand via
// browser.scripting.executeScript, same as the hand-rolled version. The injection-guard keeps
// re-injection idempotent so the message listener and index->element map survive across repeated
// "ground" clicks on the same tab.
import { extractPage } from "~/lib/content-map";

export default defineContentScript({
  matches: ["<all_urls>"],
  registration: "runtime",
  main() {
    if ((window as unknown as { __htlmInjected?: boolean }).__htlmInjected) return;
    (window as unknown as { __htlmInjected?: boolean }).__htlmInjected = true;

    // Index -> live Element, rebuilt on every extract so highlight() can resolve what the model chose.
    let elMap: Element[] = [];

    let overlay: HTMLDivElement | null = null;
    function clearHighlight() {
      if (overlay) {
        overlay.remove();
        overlay = null;
      }
    }
    function highlight(index: number): boolean {
      clearHighlight();
      const el = elMap[index];
      if (!el) return false;
      el.scrollIntoView({ block: "center", behavior: "smooth" });
      const r = el.getBoundingClientRect();
      overlay = document.createElement("div");
      Object.assign(overlay.style, {
        position: "fixed", zIndex: "2147483647", pointerEvents: "none",
        left: `${r.left - 3}px`, top: `${r.top - 3}px`,
        width: `${r.width + 6}px`, height: `${r.height + 6}px`,
        border: "3px solid #16a34a", borderRadius: "6px",
        boxShadow: "0 0 0 3px rgba(22,163,74,.25), 0 0 12px rgba(22,163,74,.6)",
        transition: "all .15s ease",
      });
      document.documentElement.appendChild(overlay);
      return true;
    }

    browser.runtime.onMessage.addListener((msg, _sender, reply) => {
      if (msg.type === "htlm-extract") {
        const { page, elements } = extractPage();
        elMap = elements;
        reply(page);
      } else if (msg.type === "htlm-highlight") {
        reply({ ok: highlight(msg.index) });
      } else if (msg.type === "htlm-clear") {
        clearHighlight();
        reply({ ok: true });
      }
      return true;
    });
  },
});
