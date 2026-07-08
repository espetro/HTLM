# HTLM — Browser-Agent Fine-Tune

**Fine-tune a 350M browser-agent model on Apple Silicon. Run it in-browser. 91.2% accuracy.**

HTLM (HyperText Language Model) fine-tunes a 350M-parameter [LiquidAI LFM2.5](https://huggingface.co/LiquidAI/LFM2.5-350M) to predict web UI actions — click, type, select — on an indexed element list. Runs entirely in-browser via [wllama](https://github.com/ngxson/wllama) WebAssembly, no server required.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Model: LFM2.5-350M](https://img.shields.io/badge/Model-LFM2.5--350M-FFD700?style=flat-square)](https://huggingface.co/LiquidAI/LFM2.5-350M)
[![HuggingFace: espetro/htlm-lfm2.5-350m](https://img.shields.io/badge/HuggingFace-espetro%2Fhtlm--lfm2.5--350m-yellow)](https://huggingface.co/espetro/htlm-lfm2.5-350m)

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

## How does HTLM compare?

HTLM's **91.2% strict action accuracy** measures whether a fine-tuned 350M model picks the right action type and element index from a pre-indexed list. This is **not the same metric** as the standard Mind2Web Step Success Rate (which additionally scores operation values like typed text). The closest published comparison points — all single-step action predictors fine-tuned on Mind2Web — are on Mind2Web's held-out test splits:

| Model | Params | Metric | Score |
|---|---|---|---|
| **HTLM (ours)** | **350M** | **Strict action accuracy (type + index match)** | **91.2%** |
| MindAct Flan-T5XL | 3B | Step Success Rate (element + operation correct) | 52.0% |
| GPT-4 (inspect_evals, test_task) | ~? | Step Success Rate | 41.7% |
| MindAct Flan-T5B | 220M | Step Success Rate | 41.0% |
| ScribeAgent-Large (zero-shot) | 32B | Step Success Rate (multi-stage QA) | 51.2% |

Among models fine-tuned on Mind2Web's training data, the previous best published single-step result was MindAct Flan-T5XL at 52.0% Step SR. HTLM achieves 91.2% on a simpler sub-task (no operation value scoring, pre-indexed elements) with a 350M model — demonstrating that a small, browser-runnable model can accurately predict the next action given the right action space simplification.

**What is not comparable**: WebVoyager (59% Task SR on custom live-site benchmark), SeeAct (51% Task SR on live websites), and other end-to-end agents evaluate multi-step task completion with full action diversity — a fundamentally different evaluation.

**Sources**: Mind2Web / MindAct (Deng et al., NeurIPS 2023) · SeeAct (Zheng et al., 2024) · ScribeAgent (Shen et al., 2024) · GPT-4 inspect_evals (ukgovernmentbeis/inspect_evals, 2024)

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
  file: 'lfm2.5-350m-mlx-q8.gguf',
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
