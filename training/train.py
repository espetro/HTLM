"""LoRA/QLoRA fine-tuning via trl.SFTTrainer.

Usage:
    uv run python -m training.train --config training/configs/lfm2.5-350m.yaml

Config format (YAML):
    model_id: LiquidAI/LFM2.5-350M          # HuggingFace model id or GGUF path
    formatter: llama                         # llm-formatter key (llama|qwen2.5|gemma3)
    quant: null                              # null = full precision, "4bit" = QLoRA, "8bit" = LoRA
    lora_rank: 16
    lora_alpha: 32
    lora_dropout: 0.05
    lora_target_modules: [q_proj, k_proj, v_proj, o_proj]
    learning_rate: 2.0e-4
    num_epochs: 3
    per_device_batch_size: 2
    gradient_accumulation_steps: 8          # effective batch = 16
    warmup_ratio: 0.1
    weight_decay: 0.01
    lr_scheduler: cosine
    max_seq_length: 2048
    train_data: data/processed/train.jsonl
    eval_data: data/processed/eval.jsonl
    output_dir: training/runs/lfm2.5-350m
    fp16: true
    gradient_checkpointing: true
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import yaml

# Ensure `training` is on the path for the formatters import
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.formatters import get_formatter  # noqa: E402


def _resolve_model_id(model_id: str) -> str:
    """Return the HF model id. GGUF paths are returned as-is (inference handles GGUF separately)."""
    return model_id


def _load_dataset(path: str | Path, formatter_name: str, tokenizer: Any, max_seq_length: int):
    """Load a JSONL dataset and format each record for training.

    Returns a datasets.Dataset ready for SFTTrainer.
    """
    from datasets import Dataset

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    def format_record(ex: dict[str, Any]) -> dict[str, str]:
        # Reconstruct the full formatted text using the formatter
        formatter = _formatter_from_name(formatter_name)
        messages = formatter.format(ex)
        text = formatter.apply_template(messages, tokenizer)
        return {"text": text}

    formatted = [format_record(r) for r in records]
    return Dataset.from_list(formatted)


def _formatter_from_name(name: str):
    """Dummy stand-in — replaced by get_formatter at runtime based on model id."""
    return get_formatter(name + "-dummy")


class MultiModelFormatter:
    """Deferred formatter that resolves the right one once the tokenizer is loaded."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._formatter = None

    def format_and_tokenize(self, examples: dict[str, Any], tokenizer: Any, max_seq_length: int):
        if self._formatter is None:
            self._formatter = get_formatter(self.model_id)
        texts = []
        for ex in examples["text"]:
            messages = self._formatter.format({"instruction": "", "page": {}, "action": {}})
            # For tokenization-only path, just decode the text field
            # The actual formatting happened in _load_dataset
            texts.append(ex if isinstance(ex, str) else ex.get("text", ""))
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
            return_tensors=None,
        )


def train(config_path: str | Path) -> None:
    config_path = Path(config_path)
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    model_id: str = config["model_id"]
    formatter_key: str = config.get("formatter", "llama")
    quant: str | None = config.get("quant", None)  # null | "4bit" | "8bit"
    output_dir: str = config["output_dir"]

    print(f"[train] model={model_id} formatter={formatter_key} quant={quant}")
    print(f"[train] config={config_path}")

    # ── tokenizer ─────────────────────────────────────────────────────────────
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        _resolve_model_id(model_id),
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── base model ────────────────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    load_kwargs: dict[str, Any] = {"trust_remote_code": True}
    if quant == "4bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        load_kwargs["device_map"] = "auto"
    elif quant == "8bit":
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        _resolve_model_id(model_id),
        **load_kwargs,
    )

    # ── LoRA config ───────────────────────────────────────────────────────────
    from peft import LoraConfig, get_peft_model, TaskType

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.get("lora_rank", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get("lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── dataset ───────────────────────────────────────────────────────────────
    train_data_path = config.get("train_data")
    eval_data_path = config.get("eval_data")

    if not train_data_path:
        raise SystemExit("train_data is required in config")
    if not Path(train_data_path).exists():
        raise SystemExit(f"train_data not found: {train_data_path}")

    train_ds = _load_dataset(train_data_path, formatter_key, tokenizer, config.get("max_seq_length", 2048))
    eval_ds = None
    if eval_data_path and Path(eval_data_path).exists():
        eval_ds = _load_dataset(eval_data_path, formatter_key, tokenizer, config.get("max_seq_length", 2048))

    # ── training arguments ───────────────────────────────────────────────────
    from transformers import TrainingArguments
    from trl import SFTTrainer

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config.get("num_epochs", 3),
        per_device_train_batch_size=config.get("per_device_batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 8),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        learning_rate=config.get("learning_rate", 2.0e-4),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        lr_scheduler_type=config.get("lr_scheduler", "cosine"),
        fp16=config.get("fp16", True),
        logging_steps=10,
        save_strategy="epoch",
        report_to=["tensorboard"],
        optim="paged_adamw_32bit" if quant else "adamw_torch",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        max_seq_length=config.get("max_seq_length", 2048),
        dataset_text_field="text",
    )

    print(f"[train] starting training. train={len(train_ds)} eval={len(eval_ds) if eval_ds else 0}")
    trainer.train()

    # ── save adapter ────────────────────────────────────────────────────────
    final_dir = Path(output_dir) / "final"
    trainer.save_model(str(final_dir))
    print(f"[train] adapter saved → {final_dir}")


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune a model with LoRA/QLoRA.")
    p.add_argument("--config", required=True, help="Path to YAML config file")
    args = p.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
