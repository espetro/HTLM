"""Runtime benchmark harness for the HTLM browser-grounding model."""
from __future__ import annotations
import argparse, base64, json, os, subprocess, sys, tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
    run_id: str; model_path: str; browser: str
    total_steps: int; correct_steps: int; accuracy: float
    mean_latency_ms: float | None; p50_latency_ms: float | None
    p95_latency_ms: float | None; errors: int

def _load_records(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: records.append(json.loads(line))
    return records

def _action_matches(pred, gt):
    return pred.get("type") == gt.get("type") and pred.get("index") == gt.get("index")

def _build_harness_html(model_path, instruction, page):
    model_b64 = ""
    if Path(model_path).exists():
        model_b64 = base64.b64encode(open(model_path, "rb").read()).decode()
    wllama_cdn = os.getenv("WLLAMA_CDN", "https://cdn.jsdelivr.net/npm/wllama@latest/dist/")
    sys_prompt = ("You are a browser grounding model. Output exactly one action as compact JSON. "
                  "Roles: button link input textarea select checkbox radio combobox searchbox menu menuitem tab switch. "
                  "Actions: click select type scroll wait done. Output only the JSON object.")
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
  body{{font-family:monospace;margin:2rem}}
  #result{{margin-top:1rem;padding:0.5rem;background:#eee;white-space:pre-wrap}}
  #log{{font-size:0.8rem;color:#666}}
</style></head><body>
<div id="log">Loading wllama...</div><div id="result"></div>
<script type="module">
const WLLAMA_CDN = {json.dumps(wllama_cdn)};
const INSTRUCTION = {json.dumps(instruction)};
const PAGE = {json.dumps(page, ensure_ascii=False)};
const MODEL_B64 = {json.dumps(model_b64)};
async function main() {{
  const log = document.getElementById('log');
  const resultDiv = document.getElementById('result');
  try {{
    const {{ default: wllama }} = await import(WLLAMA_CDN + 'wllama.cjs');
    log.textContent += ' | loading model...';
    const llama = await wllama({{ withCredentials: false }});
    await llama.loadModelFromUrl(MODEL_B64 ? 'data:application/octet-stream;base64,' + MODEL_B64 : null);
    log.textContent += ' | loaded | inferring...';
    const userMsg = INSTRUCTION + '\\n\\n' + JSON.stringify(PAGE);
    const fullPrompt = '<|begin_of_text|><|system|>\\n' + {json.dumps(sys_prompt)} + '<|end_of_turn|>\\n<|user|>\\n' + userMsg + '<|end_of_turn|>\\n<|assistant|>\\n';
    const t0 = performance.now();
    const output = await llama.createCompletion(fullPrompt, {{ nPredict: 128, temperature: 0.0, stop: ['<|end_of_turn|>'] }});
    const latencyMs = Math.round(performance.now() - t0);
    const text = output.content.trim();
    let action;
    try {{
      const m = text.match(/\\{{[\\s\\S]*\\}}/);
      action = m ? JSON.parse(m[0]) : {{ type: 'parse_error', raw: text.slice(0,200) }};
    }} catch(e) {{ action = {{ type: 'parse_error', raw: text.slice(0,200) }}; }}
    const result = {{ ...action, _latency_ms: latencyMs }};
    resultDiv.textContent = JSON.stringify(result);
    console.log('HTLM_RESULT:' + JSON.stringify(result));
  }} catch(e) {{
    resultDiv.textContent = JSON.stringify({{ error: e.message }});
  }}
}}
main();
</script></body></html>"""

def run_inference_in_browser(model_path, record, browser):
    harness_html = _build_harness_html(model_path, record.get("instruction",""), record.get("page",{}))
    hp = Path(tempfile.gettempdir()) / "htlm_bench.html"
    with open(hp, "w", encoding="utf-8") as f: f.write(harness_html)
    try:
        r = subprocess.run(
            [sys.executable, "-m", "agent_browser.cli",
             "--browser", browser, "--url", f"file://{hp}",
             "--wait-for", "#result", "--timeout", "30000"],
            capture_output=True, text=True, timeout=60)
        if r.returncode != 0: return None, None, f"browser: {r.stderr[:200]}"
        try:
            pred = json.loads(r.stdout.strip())
            latency = pred.get("_latency_ms")
            pred_action = {{k: v for k, v in pred.items() if not k.startswith("_")}}
            return pred_action, latency, None
        except json.JSONDecodeError: return None, None, f"parse: {r.stdout[:200]}"
    finally:
        hp.unlink(missing_ok=True)

def run_bench(model_path, eval_path, browser, out_path, run_id, limit=None):
    records = _load_records(eval_path)
    if limit: records = records[:limit]
    step_results, latencies = [], []
    for i, rec in enumerate(records):
        gt = rec.get("action", {{}})
        pred, latency, err = run_inference_in_browser(model_path, rec, browser)
        correct = _action_matches(pred or {{}}, gt) if pred else False
        step_results.append(StepResult(i, pred or {{}}, gt, latency, correct, err))
        if latency is not None: latencies.append(latency)
        print(f"  step {i}: correct={correct} latency={latency}ms")
    latencies.sort()
    ok = sum(1 for s in step_results if s.correct)
    result = BenchResult(
        run_id=run_id, model_path=model_path, browser=browser,
        total_steps=len(step_results), correct_steps=ok,
        accuracy=ok/len(step_results) if step_results else 0.0,
        mean_latency_ms=(sum(latencies)/len(latencies) if latencies else None),
        p50_latency_ms=(latencies[len(latencies)//2] if latencies else None),
        p95_latency_ms=(latencies[int(len(latencies)*0.95)] if latencies else None),
        errors=sum(1 for s in step_results if s.error))
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f: json.dump(asdict(result), f, indent=2)
    sp = out_path.with_suffix(".steps.jsonl")
    with open(sp, "w") as f:
        for s in step_results: f.write(json.dumps(asdict(s)) + "\n")
    print(f"[bench] acc={result.accuracy:.3f} mean={result.mean_latency_ms}ms p95={result.p95_latency_ms}ms errors={result.errors}")
    return result

def main():
    p = argparse.ArgumentParser(description="HTLM runtime benchmark harness.")
    p.add_argument("--model", required=True); p.add_argument("--eval", required=True)
    p.add_argument("--browser", default="chromium", choices=["chromium","firefox"])
    p.add_argument("--out", required=True); p.add_argument("--run-id", default=None)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()
    run_id = a.run_id or f"{Path(a.model).stem}-{a.browser}"
    run_bench(a.model, a.eval, a.browser, a.out, run_id, a.limit)

if __name__ == "__main__": main()
