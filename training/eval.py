"""Strict action-accuracy eval for fine-tuned candidates.

The bake-off gate metric is strict_accuracy = (pred.type == gt.type) AND
(pred.index == gt.index), NOT the LM eval_loss that bakeoff.py reports. This
harness loads a base+LoRA adapter, generates greedily on the eval set, parses
the JSON action, and writes per-candidate metrics for the go/no-go gate.

Usage:
    uv run python -m training.eval \
      --config training/configs/lfm2.5-350m.yaml \
      --adapter training/runs/lfm2.5-350m/final \
      --eval-data data/processed/eval.jsonl \
      --out training/runs/lfm2.5-350m/eval_metrics.json \
      --limit 500

Metrics written to --out:
    strict_accuracy       type AND index match (the gate metric)
    type_accuracy         type match alone (debug signal)
    index_top1            index match alone (debug signal)
    parse_failure_rate    fraction of generations that fail JSON parse
    mean_latency_ms       avg generate() wall time on the eval device
    p95_latency_ms        95th-percentile generate() wall time
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml

# Ensure `training` is importable when run as `python -m training.eval`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.formatters import get_formatter, SYSTEM_PROMPT_GROUNDING  # noqa: E402
from training.train import _detect_device  # noqa: E402

_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _extract_action(text: str) -> dict[str, Any] | None:
    """Pull the first JSON object out of a generation; None on failure."""
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _build_prompt(record: dict[str, Any], formatter, tokenizer) -> str:
    """Build the generation prompt: formatter messages minus the assistant turn,
    with the generation prompt appended so the model completes the action."""
    messages = formatter.format(record)[:-1]  # drop ground-truth assistant message
    # Qwen/Gemma formatters emit [user] only (system added in apply_template);
    # Llama emits [system, user]. Normalize so a system turn is always present.
    if not any(m["role"] == "system" for m in messages):
        messages = [{"role": "system", "content": SYSTEM_PROMPT_GROUNDING}] + messages
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def evaluate(
    config_path: str | Path,
    adapter_path: str | Path,
    eval_data: str | Path,
    out_path: str | Path,
    limit: int | None = None,
) -> dict[str, Any]:
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_id: str = config["model_id"]
    device = _detect_device()
    print(f"[eval] model={model_id} adapter={adapter_path} device={device}")

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)
    if device != "cpu":
        base = base.to(device)
    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()
    formatter = get_formatter(model_id)

    records: list[dict[str, Any]] = []
    with open(eval_data, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if limit:
        records = records[:limit]
    print(f"[eval] {len(records)} records")

    strict = type_match = index_match = parse_fail = 0
    latencies: list[float] = []
    with torch.no_grad():
        for i, rec in enumerate(records):
            gt = rec.get("action", {})
            prompt = _build_prompt(rec, formatter, tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt")
            if device != "cpu":
                inputs = {k: v.to(device) for k, v in inputs.items()}
            t0 = time.perf_counter()
            out = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)
            new_tokens = out[0, inputs["input_ids"].shape[1]:]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            pred = _extract_action(text)
            if pred is None:
                parse_fail += 1
                continue
            if pred.get("type") == gt.get("type"):
                type_match += 1
            if pred.get("index") == gt.get("index"):
                index_match += 1
            if pred.get("type") == gt.get("type") and pred.get("index") == gt.get("index"):
                strict += 1
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(records)} strict={strict}/{i + 1}")

    n = len(records)
    latencies.sort()
    metrics = {
        "strict_accuracy": strict / n if n else 0.0,
        "type_accuracy": type_match / n if n else 0.0,
        "index_top1": index_match / n if n else 0.0,
        "parse_failure_rate": parse_fail / n if n else 0.0,
        "mean_latency_ms": (sum(latencies) / len(latencies)) if latencies else None,
        "p95_latency_ms": latencies[int(0.95 * (len(latencies) - 1))] if latencies else None,
        "n_records": n,
        "model_id": model_id,
        "device": device,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(
        f"[eval] strict={metrics['strict_accuracy']:.3f} type={metrics['type_accuracy']:.3f} "
        f"parse_fail={metrics['parse_failure_rate']:.3f} "
        f"mean={metrics['mean_latency_ms']:.0f}ms p95={metrics['p95_latency_ms']:.0f}ms"
    )
    print(f"[eval] written -> {out_path}")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Strict action-accuracy eval for a fine-tuned candidate.")
    p.add_argument("--config", required=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--eval-data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    evaluate(args.config, args.adapter, args.eval_data, args.out, args.limit)


if __name__ == "__main__":
    main()
