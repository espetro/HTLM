---
base_model: LiquidAI/LFM2.5-350M
model-index:
  - name: HTLM-LFM2.5-350M
    results:
      - task:
          type: text-generation
          subset: browser-agent-action-prediction
        dataset:
          name: Mind2Web
          type: browser-automation
          split: held-out
          num_examples: 408
        metrics:
          - type: strict-action-accuracy
            value: 0.912
          - type: p95-latency-ms
            value: 1245
            verified: browser-wasm
license: apache-2.0
tags:
  - gguf
  - mlx
  - text-generation
  - browser-automation
  - web-agent
  - web-llm
  - lora
  - lfm2.5
  - apple-silicon
  - on-device
  - espetro
  - quantized
---

# HTLM — Browser-Agent Fine-Tune of LFM2.5-350M

[HTLM](https://github.com/espetro/HTLM) (HyperText Language Model) is a fine-tuned [LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M) that predicts web UI actions — `click`, `type`, `select` — on an indexed element list, entirely in-browser via [wllama](https://github.com/ngxson/wllama) WebAssembly.

## Benchmark Results

| Metric | Value |
|--------|-------|
| Strict action accuracy | **91.2%** |
| Action type accuracy | 92.2% |
| Element index accuracy | 99.8% |
| Parse failure rate | 0.0% |
| p95 latency (browser WASM) | 1245 ms |
| Model size (Q8 GGUF) | 362 MB |
| Base model (no fine-tune) | 0.2% |

Evaluated on 408 held-out tasks from [Mind2Web](https://github.com/osunlp/Mind2Web). Full evaluation details: [docs/go-no-go-checklist.md](https://github.com/espetro/HTLM/blob/main/docs/go-no-go-checklist.md).

## How It Works

HTLM takes a structured page representation (element list with role/tag/text) and an instruction, and predicts `{type, index, [value]}`. The element index refers to the candidate list derived from the page HTML.

```
HTML → element list → HTLM → {type, index, value?}
```

## Usage

### Browser (wllama)

```javascript
import { Wllama } from '@wllama/wllama';

const wllama = new Wllama({ default: './wllama.wasm' });
await wllama.loadModelFromHF({
  repo: 'espetro/htlm-lfm2.5-350m',
  file: 'htlm-350m-q8.gguf',
});

const result = await wllama.createCompletion({
  prompt: JSON.stringify({
    instruction: "Click the submit button",
    page: { elements: [{role:"button",tag:"button",text:"Submit"}] },
  }),
  max_tokens: 128,
});
```

### llama.cpp CLI

```bash
llama-cli -m htlm-350m-q8.gguf -p "[INPUT JSON]" -n 128 --temp 0
```

### mlx-lm (Apple Silicon)

```python
from mlx_lm import load, generate
model, tokenizer = load('espetro/htlm-lfm2.5-350m-mlx')
# LoRA merge required first — see GitHub repo
```

## Training

Fine-tuned via LoRA (rank 16) on Mind2Web using the mlx-lm / Unsloth-compatible API on Apple Silicon. Full pipeline, hyperparameters, and reproducibility steps: [docs/pipeline.md](https://github.com/espetro/HTLM/blob/main/docs/pipeline.md).

## Repository

https://github.com/espetro/HTLM
