# Training Pipeline

End-to-end pipeline for reproducing HTLM fine-tuning: data preparation → training → export.

---

## 1. Data Preparation

### Bootstrap Mind2Web

```bash
uv run python -m data.pipeline.cli bootstrap --out data/raw/mind2web
```

Downloads [Mind2Web](https://github.com/osunlp/Mind2Web) (train split, ~1009 tasks, 5.9GB raw). The base dataset (no AXT annotations) is used since element indexing is derived from the candidate list.

### Map to Training Records

```bash
uv run python -m data.pipeline.cli map-mind2web \
  --src data/raw/mind2web \
  --out data/processed/train.jsonl
```

Converts raw Mind2Web tasks into `TrainingRecord` format — one record per action step. Elements are filtered to interactive roles (`button`, `link`, `input`, `select`, `textarea`, etc.) and capped at 64 candidates. Produces ~5.85 records per task.

### Train / Eval Split

```bash
uv run python -m data.pipeline.cli split \
  --in data/processed/train.jsonl \
  --train data/processed/train.jsonl \
  --eval data/processed/eval.jsonl \
  --ratio 0.1
```

Task-grouped deterministic split (10% held-out evaluation, ~3625 train / ~408 eval records).

---

## 2. Training

### With mlx-lm / Unsloth API (recommended)

```bash
uv run python -m training.train_mlx \
  --config training/configs/lfm2.5-350m-mlx.yaml
```

Requires: `uv add mlx-tune`. See [training/train_mlx.py](training/train_mlx.py).

### With trl / peft (original run)

```bash
uv run python -m training.train \
  --config training/configs/lfm2.5-350m.yaml
```

Requires: transformers, peft, trl, bitsandbytes, accelerate.

### Hyperparameters

| Parameter | Value |
|---|---|
| Base model | LiquidAI/LFM2.5-350M |
| Fine-tune method | LoRA |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.0 |
| LoRA target modules | q_proj, k_proj, v_proj, o_proj |
| Learning rate | 2.0e-4 |
| Scheduler | cosine |
| Warmup ratio | 0.03 |
| Epochs | 1 |
| Effective batch size | 16 (4 × grad_accum 4) |
| Max seq length | 2048 |
| Gradient checkpointing | false |
| Precision | bf16 |

**Training time**: ~80 min on Apple Silicon MPS (1 epoch, 3625 records).

---

## 3. Evaluation

```bash
uv run python -m training.eval \
  --config training/configs/lfm2.5-350m.yaml \
  --adapter training/runs/lfm2.5-350m/final \
  --eval-data data/processed/eval.jsonl \
  --out training/runs/lfm2.5-350m/eval_metrics.json
```

Outputs: `strict_accuracy`, `type_accuracy`, `index_top1`, `parse_failure_rate`, `mean_latency_ms`, `p95_latency_ms`.

---

## 4. Export to GGUF

```bash
uv run python -m export.export \
  --adapter training/runs/lfm2.5-350m/final \
  --base LiquidAI/LFM2.5-350M \
  --out export/out/lfm2.5-350m-q8_0.gguf \
  --quantize q8_0
```

Options: `q4_k_m` (219 MB), `q4_k_s`, `q8_0` (362 MB), `f16`.

Requires: [llama.cpp](https://github.com/ggerganov/llama.cpp) built at `~/llama.cpp` (or `$LLAMA_CPP_DIR`).

---

## 5. Browser Runtime Benchmark

```bash
uv run python -m runtime-bench.bench \
  --model-path export/out/lfm2.5-350m-q8_0.gguf \
  --base LiquidAI/LFM2.5-350M \
  --eval-data data/processed/eval.jsonl \
  --out runtime-bench/out/lfm-full \
  --limit 408
```

Runs inference in Chrome via [wllama](https://github.com/ngxson/wllama) WebAssembly. Requires `runtime-bench/vendor/wllama.wasm` and a running `serve.py` (auto-spawned by bench.py).

---

## Results Summary

| Stage | Metric | Value |
|---|---|---|
| Training | Strict accuracy (MPS, fp16) | 71.6% |
| Browser bench | Strict accuracy (Q8, n=408) | 71.6% |
| Browser bench | p95 latency | 1186 ms |
| Export | Q8 GGUF size | 362 MB |
| Base model | Accuracy (no fine-tune) | 0.2% |
