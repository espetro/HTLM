# Runtime Benchmark

Cross-browser WASM inference benchmark for HTLM fine-tuned models.

## Quick Start

```bash
# Run the benchmark harness against a GGUF model in Chromium
uv run python -m runtime_bench.run \
    --model export/out/lfm2.5-350m-q4_k_m.gguf \
    --eval data/processed/eval.jsonl \
    --browser chromium \
    --out runtime-bench/out/lfm2.5-350m-chromium.json
```

Output:
- `runtime-bench/out/lfm2.5-350m-chromium.json` — aggregated metrics (accuracy, latency percentiles)
- `runtime-bench/out/lfm2.5-350m-chromium.steps.jsonl` — per-step predictions + ground truth

## Metrics Collected

- **Accuracy** — % of eval steps where predicted action matches ground truth (type + index)
- **Mean latency** — ms per inference step
- **p50 / p95 latency** — percentile latencies
- **Error count** — browser errors, parse failures, timeouts

## Browser Support

- `--browser chromium` — Chrome via `agent-browser` skill
- `--browser firefox` — Firefox via `camofox-browser` skill

The harness uses the agent-browser CLI to launch the browser, load the HTML harness page
(with wllama + base64-encoded GGUF), and extract the JSON result.

## Protocol Note

Automation protocol (CDP vs WebDriver BiDi vs per-browser tools) is deferred —
the harness uses agent-browser as the stable integration point. See `docs/feasibility.md`.
