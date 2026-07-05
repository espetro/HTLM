# Feasibility: on-device fine-tuned LLM for browser navigation

## Goal

A **<1GB, on-device, fine-tuned small model** specialized in DOM perception + action-grounding, used as a *tool* by an agent loop (planner, or a collapsed planner+executor) — not a full planner replacement. Two functions: compress a page's DOM into a compact structured representation, and ground a natural-language instruction into a concrete action (element + operation).

This assessment covers step 1 only. Step 2 (train + prove it out) is gated on this doc; steps 3-4 (SDK, reference extension) are gated on step 2's results.

## Foundation model candidates (bake-off, no pick yet)

All under 1GB fp16, open weights:

| Model | Params | License | Why it's a candidate |
|---|---|---|---|
| LiquidAI/LFM2.5-350M | 354M | LFM Open License v1.0 (free <$10M annual revenue, else contact Liquid) | Purpose-built for on-device tool-calling/data-extraction; a published fine-tune already hits 96-98% teacher-equivalence on tool calling; a community WebGPU+transformers.js browser demo already exists, de-risking browser integration |
| google/gemma-3-270m-it (+ FunctionGemma) | 268M | Gemma license (permissive, gated HF download) | Smallest of the three; Google ships a function-calling-tuned variant for mobile/edge; large fine-tuning ecosystem |
| Qwen2.5-0.5B-Instruct | 494M | Apache 2.0 (no revenue cap) | Used directly in academic web-agent papers (WebArena/Mind2Web element filtering + action grounding) — closest prior art; cleanest license for redistribution |

## Prior art

- **MindAct/Mind2Web**: two-stage "small model filters/ranks DOM elements → bigger model decides" pipeline.
- **ScribeAgent** (CMU): LoRA-fine-tuning open LLMs on production-scale trajectory data beats generic prompting on Mind2Web.
- **distil labs' LFM2.5 tool-calling fine-tune**: a 350M model matches/beats a 120B teacher on structured output when fine-tuned on curated synthetic data — the recipe we'd apply.

## Runtime feasibility (cross-browser)

- **WASM baseline**: `wllama` (llama.cpp → WASM, GGUF, CPU SIMD/threads, zero deps) works uniformly in Chrome, Firefox, Safari today. Safe default.
- **WebGPU**: shipped by default in both Chrome and Firefox, but Firefox rollout still uneven per-OS/platform in 2026 (Windows/macOS ARM64 first, Linux/Android later) — treat as progressive enhancement, not a requirement.
- `onnxruntime-web` supports wasm + webgpu execution providers from one export — useful as a single-artifact fallback path.
- **Extension shell**: Manifest V3 supported by both browsers. Avoid Chrome-only offscreen-document-only designs; build on `webextension-polyfill` (`browser.*`, not `chrome.*`-only) so the same shell loads in both. Mainly matters for step 3/4, but constrains which export/runtime gets validated in step 2.
- **Automation protocol for `runtime-bench` (open, deferred)**: CDP is Chrome-only/non-standard (`agent-browser` uses it). WebDriver BiDi is the W3C standard supported by both browsers and matches our "no Chrome-only APIs" rule, but neither `agent-browser` (CDP/Chrome-only) nor `camofox-browser` (Firefox via Camoufox/Juggler, not BiDi) speaks it today. Only affects the cross-browser latency harness in step 2 — trajectory recording for the data pipeline is unaffected and can keep using `agent-browser`/`camofox-browser`. Concrete pick deferred until `runtime-bench` is built.

## Verdict: go

Every individual piece (small tool-calling-capable base model, LoRA fine-tuning recipe, browser-WASM deployment, DOM-filtering architecture) has direct precedent and published results. Unproven part, and the actual point of step 2: whether a model this small holds up on *our* custom page-representation + action schema at acceptable accuracy/latency, in both browsers.

## Go/no-go bar for steps 3-4

- Action-grounding accuracy on held-out eval meaningfully close to a zero-shot large-LLM baseline on the same set (define "close" empirically once the baseline number exists).
- Per-step latency acceptable for interactive use (rough target: sub-200ms on CPU WASM) in **both** Chrome and Firefox.
- Quantized artifact stays under 1GB.
