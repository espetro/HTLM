# HTLM — Browser-Agent Fine-Tune

**Fine-tune a 350M browser-agent model on Apple Silicon. Run it in-browser. 91.2% accuracy.**

HTLM (HyperText Language Model) fine-tunes a 350M-parameter [LiquidAI LFM2.5](https://huggingface.co/LiquidAI/LFM2.5-350M) to predict web UI actions — click, type, select — on an indexed element list. Runs entirely in-browser via [wllama](https://github.com/ngxson/wllama) WebAssembly, no server required.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Model: LFM2.5-350M](https://img.shields.io/badge/Model-LFM2.5--350M-FFD700?style=flat-square)](https://huggingface.co/LiquidAI/LFM2.5-350M)

---

## Results

> **91.2%** strict action accuracy on Mind2Web held-out (n=408)  
> **1245 ms** p95 inference latency — browser WASM, no server  
> **362 MB** Q8 GGUF — downloads in under a minute  
> vs. near-random base model: **0.2%**

| Metric | Value |
|--------|-------|
| Strict action accuracy | **91.2%** |
| Action type accuracy | 92.2% |
| Element index accuracy | 99.8% |
| Parse failure rate | 0.0% |
| p95 latency (browser) | 1245 ms |
| Model size | 362 MB (Q8 GGUF) |
| Base model (no fine-tune) | 0.2% |

Fine-tuned with LoRA (rank 16) on [Mind2Web](https://github.com/osunlp/Mind2Web). Evaluation on 408 held-out tasks. See [docs/go-no-go-checklist.md](docs/go-no-go-checklist.md) for full evaluation details.

---

## Key Features

- **Browser-native inference** — runs entirely in-browser via wllama + WebAssembly (Chrome/Firefox)
- **Apple Silicon training** — canonical path: fine-tuned using the mlx-lm / Unsloth-compatible API on local MLX hardware
- **LoRA adapter** — swap adapters without re-downloading the base model
- **GGUF export** — export to Q4/Q8 GGUF for llama.cpp, ollama, or MLC-LLM
- **Action space** — predicts `click`, `type`, `select` on an indexed element list derived from page HTML

---

## How It Works

HTLM takes a page representation (structured HTML elements with role/tag/text attributes) and an instruction, and predicts the next action: `{type, index, [value]}` where `index` refers to an element in the page's candidate list.

```
HTML → element index list → HTLM → {type, index, value?}
```

See [docs/pipeline.md](docs/pipeline.md) for the full reproducible training pipeline.

---

## Try It

```javascript
import { Wllama } from '@wllama/wllama';

const wllama = new Wllama({ default: './wllama.wasm' });

  // Load from HuggingFace
await wllama.loadModelFromHF({
  repo: 'espetro/htlm-lfm2.5-350m',
  file: 'htlm-350m-q8.gguf',
});

const result = await wllama.createCompletion({
  prompt: JSON.stringify({
    instruction: "Click the submit button",
    page: { elements: [...] },
  }),
  max_tokens: 128,
});
```

Or use the model card: [espetro/htlm-lfm2.5-350m](https://huggingface.co/espetro/htlm-lfm2.5-350m)

---

## Train It

Fine-tuned via LoRA/QLoRA on Apple Silicon using the mlx-lm API (Unsloth-compatible). Training setup mirrors standard Unsloth fine-tuning patterns. **Verified: 91.2% strict accuracy on Mind2Web held-out (n=408).**

```python
from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig

model, tokenizer = FastLanguageModel.from_pretrained(
    "LiquidAI/LFM2.5-350M", max_seq_length=2048, dtype="bfloat16"
)
model = FastLanguageModel.get_peft_model(model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

trainer = SFTTrainer(
    model=model, train_dataset=train_records, tokenizer=tokenizer,
    args=SFTConfig(output_dir="./runs", per_device_train_batch_size=4, learning_rate=2e-4, num_train_epochs=1),
)
trainer.train()
model.save_pretrained("./adapter")
```

Full pipeline, dataset details, hyperparameters, and reproducibility steps:
**[docs/pipeline.md](docs/pipeline.md)**

---

## Links

| | |
|---|---|
| Model | [HuggingFace: espetro/htlm-lfm2.5-350m](https://huggingface.co/espetro/htlm-lfm2.5-350m) |
| Base model | [LiquidAI/LFM2.5-350M](https://huggingface.co/LiquidAI/LFM2.5-350M) |
| Runtime | [wllama](https://github.com/ngxson/wllama) |
| Training | [docs/pipeline.md](docs/pipeline.md) |
| Evaluation | [docs/go-no-go-checklist.md](docs/go-no-go-checklist.md) |
| Feasibility | [docs/feasibility.md](docs/feasibility.md) |

---

## Citation

```bibtex
@misc{htlm2026,
  title = {HTLM: Browser-Agent Fine-Tune of LFM2.5-350M},
  author = {espetro},
  year = {2026},
  url = {https://github.com/espetro/HTLM}
}
```
