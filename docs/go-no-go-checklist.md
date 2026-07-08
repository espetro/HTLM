# Go / No-Go Decision Checklist

**Step 2 gate.** Before starting SDK and extension work (step 3), the bake-off results must clear these thresholds. If no candidate clears the bar, return to step 1 (re-assess model size, data quality, or grounding schema).

---

## 1. Bake-off Results Table

Fill in after running `uv run python -m training.bakeoff`:

| Candidate | Params | eval_loss | Accuracy | Mean latency | p95 latency | Model size | Pass? |
|---|---|---|---|---|---|---|---|
| LFM2.5-350M (fine-tuned Q8_0, mlx-tune) | 354M | _ | 91.2% (n=408) | 824ms | 1245ms | 362MB | ✅ GO |
| LFM2.5-350M (fine-tuned Q4_K_M) | 354M | _ | 59.3% (n=408) | 832ms | 1218ms | 219MB | ❌ accuracy |
| LFM2.5-350M (base Q4_K_M control) | 354M | _ | 0.2% (n=408) | 1080ms | 2131ms | 219MB | ❌ accuracy+latency |
| FunctionGemma | 268M | _ | _% | _ms | _ms | ~134MB | ✅/❌ |
| Qwen2.5-0.5B | 494M | _ | _% | _ms | _ms | ~247MB | ✅/❌ |

Accuracy = % of eval steps where `predicted_action.type == ground_truth_action.type`
AND `predicted_action.index == ground_truth_action.index`.

---

## 2. Go Criteria

All three must be true to proceed to SDK/extension:

| Criterion | Threshold | Why it matters |
|---|---|---|
| **Accuracy** | ≥ 60% on held-out eval | Grounding quality must beat random (random ≈ 5-15% depending on element count) |
| **Latency (p95)** | < 2000ms per step in Chromium | Must feel responsive in-browser; 2s is the perceptible delay threshold |
| **Model size (Q4)** | < 1GB | Requirement from plan: on-device fine-tuned model must fit in <1GB |

If one candidate clears all three: **GO** — proceed to SDK.
If no candidate clears all three: **NO-GO** — return to step 1. Diagnose: is the schema too simple? Is training data insufficient? Should we try a larger model?
If multiple candidates clear all three: **GO with note** — pick based on latency × accuracy trade-off.

---

## 3. Per-Candidate Deep Dive (if borderline)

If accuracy is 50-65% or latency is 1500-2500ms, investigate:

- [ ] **Confusion matrix**: Is `type` often confused with `click`? Are indices consistently off by ±1 (layout ordering ambiguity)?
- [ ] **Error analysis**: Inspect `runtime-bench/out/<run>.steps.jsonl` for patterns — does the model fail on scroll-heavy tasks? Long pages?
- [ ] **Latency breakdown**: Is the bottleneck tokenization, model inference, or JSON parsing? (Profile separately.)
- [ ] **Distillation effect**: Did teacher distillation improve accuracy on eval? Compare distilled vs. non-distilled runs.

---

## 4. Decision

**Candidate selected:** LFM2.5-350M (fine-tuned, Q8_0)

**Decision:** ☑ GO — proceed to SDK/extension (step 3)
              ☐ NO-GO — re-assess step 1
              ☐ Conditional GO

**Rationale / blockers:**

The canonical mlx-tune run improves the LFM2.5-350M Q8_0 result to **91.2% strict** (MPS) and the browser runtime verifies it at **91.9% strict** on the same n=408 held-out set. This beats the previous trl-based path (71.6% strict) by ~19.6 pp. The Q8_0 artifact remains the selected candidate.

- **Fine-tuning signal (base vs fine-tuned Q8):** +91.0 pp (0.2% → 91.2%). Fine-tuning clearly transferred to the browser runtime.
- **Quantization signal (fine-tuned Q8 vs base Q4):** Q8_0 is canonical; the base Q4 control remains effectively random (0.2%).
- **Latency:** Q8_0 p95 = **1245ms** per step in Chromium (< 2000ms ✅). Mean = **824ms**. Native MPS p95 = **783ms**.
- **Size:** Q8_0 artifact = **362MB** (< 1GB ✅).
- **95% CI lower bound for Q8_0:** well above the 60% threshold.

Q8_0 clears all three go criteria with the new mlx-tune path. The improvement is sufficient to switch the canonical training pipeline from trl to mlx-tune; no additional quantization sweep is needed.

**Date:** 2026-07-08

---

## 5. SDK / Extension Work (step 3) — Unblock Checklist

Once GO is declared, these must be true before starting step 3:

- [ ] Fine-tuned adapter committed to `training/runs/<candidate>/final`
- [x] GGUF export at `export/out/<candidate>-q4_k_m.gguf`
- [x] Bench results committed to repo (`runtime-bench/out/`)
- [ ] Selected model ID and rationale added to `docs/prd.md`
