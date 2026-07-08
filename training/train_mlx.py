"""LoRA fine-tuning via mlx-tune (Unsloth-compatible API on Apple MLX).

DRAFT — API signatures sourced from mlx-tune v0.6.0 source
(``mlx_tune/sft_trainer.py``, ``mlx_tune/__init__.py``). Needs `uv pip install
mlx-tune` + a smoke run on data/samples/travel.jsonl before it replaces the trl
path. That validation is deferred until the current trl/peft bake-off finishes
(running both on one GPU would contend for Metal).

Why a separate file, not an edit to train.py:
  - train.py (trl + peft on PyTorch/MPS) is the working, gate-passing path
    (LFM2.5-350M: 71.6% strict accuracy, 699ms p95). It stays as the reference
    and the active bake-off backend.
  - train_mlx.py is the MLX migration candidate. Once it smoke-passes, point
    bakeoff.py at it via a config flag (no other change needed — both emit
    output_dir/final + eval_metrics.json via the shared eval.py).

MLX-specific decisions (deviating from a naive Unsloth port):
  - load_in_4bit=False (bf16 base). On MLX `load_in_4bit` is NOT bitsandbytes —
    it means "load a pre-quantized mlx-community/*-4bit model". Our three models
    have no such variants, and a 4-bit base breaks GGUF export (mlx-lm#353). Our
    end goal is Q4 GGUF, and ≤500M fits in bf16 at ~2GB, so bf16 base is correct.
  - SFTConfig field is `grad_checkpoint` (NOT `gradient_checkpointing`).
  - Default target modules include MLP projections (gate/up/down) per project
    spec; q/k/v/o alone already passed the gate for LFM.
  - Our formatters are reused unchanged (they emit model-correct chat text via
    the tokenizer's apply_chat_template); mlx-tune's own template layer and
    train_on_responses_only are not needed for this go/no-go.

Usage:
    uv run python -m training.train_mlx --config training/configs/lfm2.5-350m.yaml

Per-model notes:
    LFM2.5-350M  (Llama-family): bf16, target q/k/v/o[+gate/up/down], bs=2/ga=8
    Gemma-3-270m (256k vocab):   bf16, target q/k/v/o+gate/up/down, bs=2/ga=8.
                                 Embeddings frozen by LoRA; vocab handled natively.
    Qwen2.5-0.5B:                bf16, target q/k/v/o[+gate/up/down], bs=2/ga=8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

# ponytail: mlx-lm 0.31.3 registers "NewlineTokenizer" by name (str); transformers
# 5.x AutoTokenizer.register requires a config class. Our tokenizers (LFM/Gemma/Qwen)
# don't use NewlineTokenizer, so no-op the string form. Drop if mlx-lm upstream fixes.
import transformers as _tf

_tf_register = _tf.AutoTokenizer.register
_tf.AutoTokenizer.register = lambda c, *a, **k: None if isinstance(c, str) else _tf_register(c, *a, **k)

# Reuse the framework-agnostic formatter layer (same contract as train.py +
# eval.py): get_formatter(model_id).format(record) -> messages, .apply_template.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.formatters import get_formatter  # noqa: E402


def train_mlx(config_path: str | Path) -> None:
    from mlx_tune import FastLanguageModel, SFTTrainer, SFTConfig  # noqa: F401

    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_id: str = config["model_id"]
    formatter_key: str = config.get("formatter", "llama")
    output_dir: str = config["output_dir"]
    max_seq_length: int = config.get("max_seq_length", 2048)

    print(f"[train_mlx] model={model_id} formatter={formatter_key} (mlx-tune)")
    print(f"[train_mlx] config={config_path}")

    # ── model + tokenizer (bf16 base; see module docstring on why not 4-bit) ──
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        load_in_4bit=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── LoRA: resolve formatter against the real model_id, then attach adapter ──
    # ponytail: get_formatter does substring match on model_id, so rebuild the
    # formatter bound to model_id (not the alias) for apply_template correctness.
    global_fmt = get_formatter(model_id)

    model = FastLanguageModel.get_peft_model(
        model,
        r=config.get("lora_rank", 16),
        lora_alpha=config.get("lora_alpha", 32),
        target_modules=config.get(
            "lora_target_modules",
            ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
    )

    # ── dataset (reuse shared formatter → {"text": ...}) ───────────────────────
    train_data_path = config.get("train_data")
    eval_data_path = config.get("eval_data")
    if not train_data_path or not Path(train_data_path).exists():
        raise SystemExit(f"train_data not found: {train_data_path}")

    train_ds = _load_dataset_bound(train_data_path, global_fmt, tokenizer, max_seq_length)
    eval_ds = None
    if eval_data_path and Path(eval_data_path).exists():
        eval_ds = _load_dataset_bound(eval_data_path, global_fmt, tokenizer, max_seq_length)

    # ── training config (mlx-tune SFTConfig field names from source) ───────────
    args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=config.get("num_epochs", 1),
        per_device_train_batch_size=config.get("per_device_batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        learning_rate=config.get("learning_rate", 2.0e-4),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        lr_scheduler_type=config.get("lr_scheduler", "cosine"),
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        bf16=True,                         # MLX honors bf16 compute
        grad_checkpoint=config.get("gradient_checkpointing", False),  # note: field is grad_checkpoint
        logging_steps=10,
        save_steps=10_000,                 # ponytail: save once at end via save_pretrained, not checkpoints
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        tokenizer=tokenizer,
        args=args,
    )

    print(f"[train_mlx] starting. train={len(train_ds)} eval={len(eval_ds) if eval_ds else 0}")
    trainer.train()

    # ── save adapter (mlx-tune nests relative adapter_path under output_dir) ────
    # Use an absolute final path so output is predictable for eval.py/bakeoff.py.
    final_dir = (Path(output_dir) / "final").resolve()
    model.save_pretrained(str(final_dir))
    print(f"[train_mlx] adapter saved -> {final_dir}")

    # GGUF export is intentionally NOT done here: (1) export_to_gguf signature
    # needs smoke confirmation, (2) it's Phase 4 (post go/no-go), (3) keeps the
    # train step focused. Run separately after validation:
    #     from mlx_tune import export_to_gguf
    #     export_to_gguf(model, "export/out/<run>-q4_k_m.gguf", tokenizer)


def _load_dataset_bound(path, formatter, tokenizer, max_seq_length):
    """Load JSONL + format into a list of {"text": ...} dicts.

    Returns a plain list (NOT a datasets.Dataset): mlx-tune's SFTTrainer accepts a
    list of dicts directly (sft_trainer.py:202 docstring + _prepare_data iterates
    self.train_dataset writing JSONL), which sidesteps the py3.14 dill/multiprocess
    pickle crash in datasets.Dataset.from_list. No datasets import needed.
    """
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    def format_record(ex: dict[str, Any]) -> dict[str, str]:
        msgs = formatter.format(ex)
        return {"text": formatter.apply_template(msgs, tokenizer)}

    return [format_record(r) for r in records]


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune with mlx-tune (MLX / Unsloth API).")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    args = p.parse_args()
    train_mlx(args.config)


if __name__ == "__main__":
    main()
