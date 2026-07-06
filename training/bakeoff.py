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


def _run_training(config_path: str) -> dict[str, Any]:
    """Run `train.py --config <config>` and return parsed metrics from the output dir."""
    import yaml

    with open(config_path) as f:
        config = yaml.safe_load(f)

    output_dir = Path(config["output_dir"])
    run_id = output_dir.name

    print(f"\n{'='*60}")
    print(f"[bakeoff] Starting training: {run_id}")
    print(f"[bakeoff] model={config['model_id']} quant={config.get('quant')}")
    print(f"{'='*60}\n")

    t0 = time.time()
    result = subprocess.run(
        [sys.executable, "-m", "training.train", "--config", str(config_path)],
        capture_output=False,  # show output live
    )
    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"[bakeoff] WARNING: training failed for {run_id} (exit {result.returncode})")
        return {"run_id": run_id, "status": "FAILED", "elapsed_s": elapsed}

    # Try to load eval metrics
    metrics_file = output_dir / "eval_metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            metrics = json.load(f)
    else:
        metrics = {}

    return {
        "run_id": run_id,
        "model_id": config["model_id"],
        "status": "OK",
        "elapsed_s": elapsed,
        "eval_loss": metrics.get("eval_loss"),
        "eval_steps_per_second": metrics.get("eval_steps_per_second"),
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    header = ["run_id", "model_id", "status", "eval_loss", "elapsed_min"]
    print("\n" + "│".join(f" {h:^20} " for h in header))
    print("─" * (22 * len(header)))
    for r in rows:
        vals = [
            r.get("run_id", ""),
            r.get("model_id", ""),
            r.get("status", ""),
            f"{r.get('eval_loss', 'N/A'):.4f}" if r.get("eval_loss") else "N/A",
            f"{r.get('elapsed_s', 0)/60:.1f}",
        ]
        print("│".join(f" {str(v):^20} " for v in vals))
    print()


def main() -> None:
    p = argparse.ArgumentParser(description="Bake-off orchestrator for 3 model candidates.")
    p.add_argument(
        "--configs",
        nargs="+",
        help="Paths to YAML config files (or use CANDIDATE_CONFIGS defaults)",
    )
    args = p.parse_args()

    configs = args.configs or [CANDIDATE_CONFIGS[k] for k in ["lfm2.5-350m", "functiongemma", "qwen2.5-0.5b"]]
    results = []

    for cfg in configs:
        res = _run_training(cfg)
        results.append(res)
        _print_table(results)

    print("\n[Bake-off complete]")
    _print_table(results)
    print("Next step: export best candidate to GGUF (issue #5), then run runtime-bench.")


if __name__ == "__main__":
    main()
