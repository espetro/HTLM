# Training

LoRA/QLoRA fine-tuning bake-off of 3 foundation model candidates for the browser-grounding task.

## Candidates

| Candidate | Params | Context | Architecture |
|---|---|---|---|
| LiquidAI/LFM2.5-350M | 354M | 32K | LFM (Llama-family) |
| google/gemma-3-270m-it (FunctionGemma) | 268M | 32K | Gemma 3 |
| Qwen/Qwen2.5-0.5B-Instruct | 494M | 32K | Qwen2.5 |

## Quick Start

```bash
# Install training deps
uv sync --extra train

# Bootstrap + map data (from issue #3)
uv run python -m data.pipeline.cli bootstrap --out data/raw/mind2web-axtree --axtree
uv run python -m data.pipeline.cli map-mind2web --src data/raw/mind2web-axtree --out data/processed/mind2web-htlm.jsonl
uv run python -m data.pipeline.cli split --in data/processed/mind2web-htlm.jsonl --train data/processed/train.jsonl --eval data/processed/eval.jsonl

# Train a single candidate
uv run python -m training.train --config training/configs/lfm2.5-350m.yaml

# Run full bake-off (all 3 candidates sequentially)
uv run python -m training.bakeoff
```

## Config Format

Each candidate has a YAML config under `training/configs/`:

```yaml
model_id: LiquidAI/LFM2.5-350M
formatter: llama           # llm-formatter key (llama|qwen2.5|gemma3)
quant: null               # null = full precision, "4bit" = QLoRA, "8bit" = LoRA
lora_rank: 16
lora_alpha: 32
lora_dropout: 0.05
lora_target_modules: [q_proj, k_proj, v_proj, o_proj]
learning_rate: 2.0e-4
num_epochs: 3
per_device_batch_size: 2
gradient_accumulation_steps: 8    # effective batch = 16
max_seq_length: 2048
train_data: data/processed/train.jsonl
eval_data: data/processed/eval.jsonl
output_dir: training/runs/lfm2.5-350m
fp16: true
gradient_checkpointing: true
```

## Bake-off Comparison Table

After running `bakeoff.py`, the script prints a table:

```
│ run_id             │ model_id            │ status   │ eval_loss │ elapsed_min │
─────────────────────...
│ lfm2.5-350m        │ LiquidAI/LFM2.5...  │ OK       │ 0.0412    │ 23.4       │
│ functiongemma       │ google/gemma-3...   │ OK       │ 0.0387    │ 18.1       │
│ qwen2.5-0.5b       │ Qwen/Qwen2.5...     │ OK       │ 0.0521    │ 31.2       │
```

The **go/no-go decision** (issue #6) uses these numbers + model size + per-token inference latency to pick the candidate.

## Adapter Merging (after bake-off)

After picking a winner, merge the LoRA adapter into the base model for GGUF export:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import llama_cpp

base = AutoModelForCausalLM.from_pretrained("LiquidAI/LFM2.5-350M", ...)
model = PeftModel.from_pretrained(base, "training/runs/lfm2.5-350m/final")
merged = model.merge_and_unload()
# Then export to GGUF via llama.cpp (see export/)
```
