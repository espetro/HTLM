"""Bake-off orchestrator: run all 3 model configs and generate a comparison table.

Usage:
    uv run python -m training.bakeoff --configs training/configs/lfm2.5-350m.yaml training/configs/functiongemma.yaml training/configs/qwen2.5-0.5b.yaml

Each config is trained sequentially (GPU required). After each run, the script
scans the final eval_metrics.json and prints a summary row.

ponytail: sequential on a single GPU. Parallel execution (multi-GPU) requires
slurm or similar orchestration — not handled here.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
import json
from pathlib import Path
from typing import Any

CANDIDATE_CONFIGS = {
    "lfm2.5-350m": "training/configs/lfm2.5-350m.yaml",
    "functiongemma": "training/configs/functiongemma.yaml",
    "qwen2.5-0.5b": "training/configs/qwen2.5-0.5b.yaml",
}


def _run_candidate(config_path: str, eval_data: str | None, eval_limit: int | None) -> dict[str, Any]:
    """Train then eval one candidate. Returns a metrics row.

    pipeline: train.py → adapter at output_dir/final, then eval.py → output_dir/eval_metrics.json
    Gate metrics (strict_accuracy, p95 latency) come from eval.py, not from trainer eval_loss.
    """
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    run_id = output_dir.name
    adapter_dir = output_dir / "final"
    metrics_file = output_dir / "eval_metrics.json"
    eval_path = eval_data or config.get("eval_data")

    # ── train ────────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"[bakeoff] TRAIN {run_id}  model={config['model_id']} quant={config.get('quant')}")
    print(f"{'='*60}\n")
    t0 = time.time()
    train_res = subprocess.run(
        [sys.executable, "-m", "training.train", "--config", str(config_path)],
        capture_output=False,
    )
    train_s = time.time() - t0
    if train_res.returncode != 0:
        print(f"[bakeoff] training FAILED for {run_id} (exit {train_res.returncode})")
        return {"run_id": run_id, "model_id": config["model_id"], "status": "TRAIN_FAILED", "train_min": train_s / 60}

    # ── eval ─────────────────────────────────────────────────────────────────
    if not eval_path or not Path(eval_path).exists():
        print(f"[bakeoff] no eval_data → skipping eval for {run_id}")
        return {"run_id": run_id, "model_id": config["model_id"], "status": "TRAIN_OK_NO_EVAL", "train_min": train_s / 60}

    print(f"\n{'='*60}")
    print(f"[bakeoff] EVAL {run_id}  eval={eval_path}" + (f" limit={eval_limit}" if eval_limit else ""))
    print(f"{'='*60}\n")
    t0 = time.time()
    eval_cmd = [
        sys.executable, "-m", "training.eval",
        "--config", str(config_path),
        "--adapter", str(adapter_dir),
        "--eval-data", str(eval_path),
        "--out", str(metrics_file),
    ]
    if eval_limit:
        eval_cmd += ["--limit", str(eval_limit)]
    eval_res = subprocess.run(eval_cmd, capture_output=False)
    eval_s = time.time() - t0
    if eval_res.returncode != 0:
        print(f"[bakeoff] eval FAILED for {run_id} (exit {eval_res.returncode})")
        return {"run_id": run_id, "model_id": config["model_id"], "status": "EVAL_FAILED", "train_min": train_s / 60}

    metrics = json.loads(metrics_file.read_text()) if metrics_file.exists() else {}
    return {
        "run_id": run_id,
        "model_id": config["model_id"],
        "status": "OK",
        "strict_accuracy": metrics.get("strict_accuracy"),
        "type_accuracy": metrics.get("type_accuracy"),
        "index_top1": metrics.get("index_top1"),
        "parse_failure_rate": metrics.get("parse_failure_rate"),
        "mean_latency_ms": metrics.get("mean_latency_ms"),
        "p95_latency_ms": metrics.get("p95_latency_ms"),
        "n_records": metrics.get("n_records"),
        "train_min": train_s / 60,
        "eval_min": eval_s / 60,
    }


def _fmt(v: Any, spec: str = "") -> str:
    if v is None:
        return "N/A"
    try:
        return format(v, spec)
    except (TypeError, ValueError):
        return str(v)


def _print_table(rows: list[dict[str, Any]]) -> None:
    # Gate metrics aligned with docs/go-no-go-checklist.md columns.
    header = ["run_id", "strict_acc", "type_acc", "parse_fail", "mean_ms", "p95_ms", "n", "train_min"]
    widths = [16, 11, 10, 10, 9, 9, 6, 9]
    line = lambda cells: "│".join(f" {str(c):^{w}} " for c, w in zip(cells, widths))
    print("\n" + line(header))
    print("┼".join("─" * (w + 2) for w in widths))
    for r in rows:
        print(line([
            r.get("run_id", ""),
            _fmt(r.get("strict_accuracy"), ".3f"),
            _fmt(r.get("type_accuracy"), ".3f"),
            _fmt(r.get("parse_failure_rate"), ".3f"),
            _fmt(r.get("mean_latency_ms"), ".0f"),
            _fmt(r.get("p95_latency_ms"), ".0f"),
            _fmt(r.get("n_records")),
            _fmt(r.get("train_min"), ".1f"),
        ]))
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Bake-off orchestrator: train+eval each candidate, print gate-metric table.")
    p.add_argument(
        "--configs",
        nargs="+",
        help="Paths to YAML configs (default: all 3 CANDIDATE_CONFIGS)",
    )
    p.add_argument("--eval-data", help="Override eval JSONL path (default: each config's eval_data)")
    p.add_argument("--eval-limit", type=int, help="Cap eval records per candidate (speed; full eval if unset)")
    args = p.parse_args()

    configs = args.configs or [CANDIDATE_CONFIGS[k] for k in ["lfm2.5-350m", "functiongemma", "qwen2.5-0.5b"]]
    results = []

    for cfg in configs:
        res = _run_candidate(cfg, args.eval_data, args.eval_limit)
        results.append(res)
        _print_table(results)

    print("\n[Bake-off complete]")
    _print_table(results)
    print("Gate: strict_accuracy>=0.60 AND p95_latency_ms<2000. Best candidate → export GGUF Q4 (issue #5).")


if __name__ == "__main__":
    main()
