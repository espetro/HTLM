# HTLM grounder (browser extension)

A minimal MV3 extension, built with [wxt](https://wxt.dev), that grounds **the page you're
currently on** with the 350M HTLM model, running entirely in your browser via
[wllama](https://github.com/ngxson/wllama) WebAssembly — no server in the inference path.

It's the "drop it in and it works" proof for HTLM: unlike the offline Mind2Web accuracy number
(teacher-forced, static pages), here the model reads a real rendered DOM you chose and picks the
next action live. Because you're a human already on the page, there's no CORS fetch and no captcha
to bypass.

## How it works

1. **`entrypoints/content.ts`** (+ `lib/content-map.ts`) walks the active tab's DOM into HTLM's
   indexed page schema (`{index, role, label, tag, ...}`), mapping tags/ARIA roles to the same
   closed role enum the model was trained on (port of `data/pipeline/map_mind2web._map_role`,
   capped at 64 elements). It's injected on demand by the side panel, not declared in the manifest.
2. **`entrypoints/sidepanel/main.ts`** builds the exact LFM2.5 ChatML prompt (`lib/grounding.ts`,
   lifted from `runtime-bench/demo.html`) and runs `wllama.createCompletion` on the 350M GGUF in
   the panel.
3. The predicted `{type, index, ...}` action is shown, and the chosen element is outlined in the
   live page.

The model runs client-side; the model file downloads once from Hugging Face
(`espetro/htlm-lfm2.5-350m`) and wllama caches it.

## Develop

```bash
bun install
bun run dev      # wxt dev server, hot-reloading unpacked extension
bun test         # vitest: grounding.test.ts, content-map.test.ts
bun run build    # -> .output/chrome-mv3
```

`wxt.config.ts` also exposes a CDP endpoint (`--remote-debugging-port=9222`) so the built extension
can be driven headfully by agent-browser for a real-browser smoke test — the WASM/CSP/worker
wiring that's otherwise impossible to verify statically.

## Load it (unpacked)

1. `bun run build`.
2. `chrome://extensions` → enable **Developer mode**.
3. **Load unpacked** → select `.output/chrome-mv3`.
4. Open any normal website, click the **HTLM grounder** toolbar icon to open the side panel.
5. Wait for "Model ready" (first run downloads the GGUF), type an instruction, click
   **Ground current page**.

## Notes / limits

- First load downloads ~360 MB (q8). Change `MODEL_FILE` in `entrypoints/sidepanel/main.ts` to a
  smaller quant if one is published to the HF repo.
- Runs single-threaded when the page isn't cross-origin isolated (extension pages usually aren't),
  so inference is slower than the multi-threaded `runtime-bench` numbers — fine for a demo.
- One-shot grounding (predict the next action), not a multi-step agent. It highlights, it doesn't
  click.
- Can't ground browser-internal pages (`chrome://`, extension pages).
- Chrome/Edge only — `sidePanel` has no Firefox equivalent.
