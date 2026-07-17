import { defineConfig } from "wxt";

// Must be set explicitly: when this key is omitted, Chrome's built-in MV3 default
// (script-src 'self'; object-src 'self') applies and lacks wasm-unsafe-eval, breaking
// WebAssembly.instantiate at runtime. wasm-unsafe-eval permits wasm instantiation, and a
// same-origin blob: worker runs under script-src 'self' without needing worker-src blob:
// declared (which Chrome MV3 rejects at manifest-validation time). Do not add worker-src
// blob: here.
export default defineConfig({
  manifest: {
    name: "HTLM grounder",
    description:
      "Ground the page you're on with the 350M HTLM model, running entirely in your browser (no server).",
    permissions: ["activeTab", "scripting", "sidePanel", "offscreen"],
    host_permissions: ["<all_urls>"],
    action: { default_title: "HTLM grounder" },
    content_security_policy: {
      extension_pages: "script-src 'self' 'wasm-unsafe-eval'; object-src 'self'",
    },
  },
  webExt: {
    // Exposes a TCP CDP endpoint so agent-browser can drive the built extension.
    chromiumArgs: ["--remote-debugging-port=9222"],
  },
});
