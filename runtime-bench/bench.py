"""Runtime benchmark harness for the HTLM browser-grounding model."""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Ensure repo root is on path so `training` and `data` packages resolve.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from data.pipeline.records import read_jsonl  # noqa: E402
from training.eval import _build_prompt, _extract_action  # noqa: E402
from training.formatters import get_formatter  # noqa: E402


@dataclass
class StepResult:
    step_index: int
    predicted_action: dict[str, Any]
    ground_truth_action: dict[str, Any]
    latency_ms: float | None
    correct: bool
    error: str | None


@dataclass
class BenchResult:
    run_id: str
    model_path: str
    browser: str
    total_steps: int
    correct_steps: int
    accuracy: float
    mean_latency_ms: float | None
    p50_latency_ms: float | None
    p95_latency_ms: float | None
    errors: int


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _wait_for_port(port: int, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(1.0)
            try:
                s.connect(("127.0.0.1", port))
                return
            except OSError:
                time.sleep(0.2)
    raise RuntimeError(f"Server on port {port} did not become ready in {timeout}s")


def _agent_browser(cmd: list[str]) -> tuple[str, str, int]:
    """Run an agent-browser command and return stdout, stderr, returncode."""
    binary = os.environ.get("AGENT_BROWSER", "/opt/homebrew/bin/agent-browser")
    r = subprocess.run([binary, *cmd], capture_output=True, text=True)
    return r.stdout, r.stderr, r.returncode


def _build_prompts_file(eval_path: str, model_id: str, limit: int | None, out_path: Path) -> int:
    try:
        from transformers import AutoTokenizer
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("transformers is required for prompt templating: " + str(e))

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    formatter = get_formatter(model_id)
    records = list(read_jsonl(eval_path))
    if limit:
        records = records[:limit]

    prompts: list[dict[str, Any]] = []
    for rec in records:
        prompt = _build_prompt(rec, formatter, tokenizer)
        action = rec.get("action", {})
        prompts.append({
            "prompt": prompt,
            "gt": {"type": action.get("type"), "index": action.get("index")},
        })

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(prompts, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[bench] wrote {len(prompts)} prompts -> {out_path}")
    return len(prompts)


def _action_matches(pred: dict[str, Any], gt: dict[str, Any]) -> bool:
    return pred.get("type") == gt.get("type") and pred.get("index") == gt.get("index")


def _run_server(
    port: int,
    *,
    harness_file: Path,
    models_dir: Path,
    wllama_dir: Path,
    prompts_file: Path,
    results_file: Path,
) -> subprocess.Popen:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "runtime-bench" / "serve.py"),
        "--port",
        str(port),
        "--harness-file",
        str(harness_file),
        "--models-dir",
        str(models_dir),
        "--wllama-dir",
        str(wllama_dir),
        "--prompts-file",
        str(prompts_file),
        "--results-file",
        str(results_file),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    _wait_for_port(port, timeout=30.0)
    return proc


def _run_browser_harness(port: int, timeout: float = 900.0) -> None:
    url = f"http://localhost:{port}/"
    stdout, stderr, rc = _agent_browser(["open", url])
    if rc != 0:
        raise RuntimeError(f"agent-browser open failed: rc={rc} stderr={stderr[:500]} stdout={stdout[:500]}")

    deadline = time.time() + timeout
    while time.time() < deadline:
        stdout, stderr, rc = _agent_browser(["eval", "({ready: window._ready, error: window._error || null})"])
        if rc != 0:
            raise RuntimeError(f"agent-browser eval failed: rc={rc} stderr={stderr[:500]}")
        try:
            state = json.loads(stdout.strip())
        except json.JSONDecodeError as e:
            raise RuntimeError(f"agent-browser returned non-JSON: {stdout[:200]} ({e})")
        if state.get("error"):
            raise RuntimeError(f"Harness error: {state['error']}")
        if state.get("ready") is True:
            return
        time.sleep(2.0)
    raise RuntimeError(f"Harness did not finish within {timeout}s")


def _score_results(raw_results: list[dict[str, Any]], prompts: list[dict[str, Any]]) -> tuple[dict[str, Any], list[StepResult]]:
    strict = type_match = index_match = parse_fail = 0
    latencies: list[float] = []
    step_results: list[StepResult] = []

    for i, raw in enumerate(raw_results):
        gt = prompts[i]["gt"]
        text = raw.get("prediction_text", "")
        latency = float(raw.get("total_ms", 0.0))
        latencies.append(latency)
        pred = _extract_action(text)
        if pred is None:
            parse_fail += 1
            pred_action: dict[str, Any] = {}
            error = "parse"
            correct = False
        else:
            pred_action = pred
            error = None
            t_ok = pred.get("type") == gt.get("type")
            i_ok = pred.get("index") == gt.get("index")
            if t_ok:
                type_match += 1
            if i_ok:
                index_match += 1
            correct = t_ok and i_ok
            if correct:
                strict += 1

        step_results.append(
            StepResult(
                step_index=i,
                predicted_action=pred_action,
                ground_truth_action=gt,
                latency_ms=latency,
                correct=correct,
                error=error,
            )
        )

    n = len(prompts)
    latencies.sort()
    mean = sum(latencies) / n if n else None
    p95 = latencies[int(0.95 * (n - 1))] if n else None

    metrics = {
        "strict_accuracy": strict / n if n else 0.0,
        "type_accuracy": type_match / n if n else 0.0,
        "index_top1": index_match / n if n else 0.0,
        "parse_failure_rate": parse_fail / n if n else 0.0,
        "mean_latency_ms": mean,
        "p95_latency_ms": p95,
        "n_records": n,
        "model_id": "unknown",
    }
    return metrics, step_results


def run_bench(
    model_path: str,
    base_model_id: str,
    eval_path: str,
    out_dir: str,
    run_id: str,
    browser: str = "chromium",
    limit: int | None = None,
    port: int | None = None,
) -> dict[str, Any]:
    model_path_obj = Path(model_path).resolve()
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    run_dir = Path(out_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    prompts_file = run_dir / "prompts.json"
    results_file = run_dir / "raw_results.json"
    metrics_file = run_dir / "eval_metrics.json"
    steps_file = run_dir / "steps.jsonl"

    _build_prompts_file(eval_path, base_model_id, limit, prompts_file)
    prompts = json.loads(prompts_file.read_text(encoding="utf-8"))

    harness_file = REPO_ROOT / "runtime-bench" / "harness.html"
    models_dir = model_path_obj.parent
    wllama_dir = REPO_ROOT / "runtime-bench" / "node_modules" / "@wllama" / "wllama" / "esm"
    if not (wllama_dir / "index.js").exists():
        raise FileNotFoundError(
            f"wllama ESM not found at {wllama_dir}. Run `bun add @wllama/wllama` in runtime-bench."
        )

    port = port or _find_free_port()
    server_proc = None
    try:
        print(f"[bench] starting server on port {port}")
        server_proc = _run_server(
            port,
            harness_file=harness_file,
            models_dir=models_dir,
            wllama_dir=wllama_dir,
            prompts_file=prompts_file,
            results_file=results_file,
        )

        print("[bench] opening browser harness")
        _run_browser_harness(port)

        if not results_file.exists():
            raise RuntimeError("Harness finished but raw_results.json was not written")
        raw_results = json.loads(results_file.read_text(encoding="utf-8"))
        print(f"[bench] collected {len(raw_results)} raw results")

        metrics, step_results = _score_results(raw_results, prompts)
        metrics["model_id"] = base_model_id
        metrics["run_id"] = run_id
        metrics["browser"] = browser

        ok_steps = sum(1 for s in step_results if s.correct)
        result = BenchResult(
            run_id=run_id,
            model_path=str(model_path_obj),
            browser=browser,
            total_steps=len(step_results),
            correct_steps=ok_steps,
            accuracy=metrics["strict_accuracy"],
            mean_latency_ms=metrics["mean_latency_ms"],
            p50_latency_ms=None,
            p95_latency_ms=metrics["p95_latency_ms"],
            errors=sum(1 for s in step_results if s.error),
        )

        metrics_file.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        steps_file.write_text(
            "".join(json.dumps(asdict(s), ensure_ascii=False) + "\n" for s in step_results),
            encoding="utf-8",
        )
        # Keep a <out>.json copy for the legacy schema contract.
        legacy_json = run_dir / f"{run_id}.json"
        legacy_json.write_text(json.dumps({**asdict(result), **metrics}, indent=2), encoding="utf-8")

        print(
            f"[bench] strict={metrics['strict_accuracy']:.3f} type={metrics['type_accuracy']:.3f} "
            f"parse_fail={metrics['parse_failure_rate']:.3f} "
            f"mean={metrics['mean_latency_ms']:.0f}ms p95={metrics['p95_latency_ms']:.0f}ms"
        )
        print(f"[bench] results -> {run_dir}")
        return metrics
    finally:
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                server_proc.kill()
        _agent_browser(["close"])


def main() -> None:
    p = argparse.ArgumentParser(description="HTLM runtime benchmark harness.")
    p.add_argument("--model-path", required=True, help="Path to the GGUF model file")
    p.add_argument("--base", required=True, help="Base model ID for tokenizer templating")
    p.add_argument("--eval-data", required=True, help="Path to eval.jsonl")
    p.add_argument("--out", required=True, help="Output directory for this run")
    p.add_argument("--browser", default="chromium", help="Browser label (informational)")
    p.add_argument("--run-id", default=None, help="Run ID (defaults to <model-stem>-<browser>")
    p.add_argument("--limit", type=int, default=None, help="Limit number of eval records")
    p.add_argument("--port", type=int, default=None, help="Server port (defaults to free port)")
    a = p.parse_args()
    run_id = a.run_id or f"{Path(a.model_path).stem}-{a.browser}"
    run_bench(a.model_path, a.base, a.eval_data, a.out, run_id, a.browser, a.limit, a.port)


if __name__ == "__main__":
    main()
