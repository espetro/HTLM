# Go / No-Go Decision Checklist

**Step 2 gate.** Before starting SDK and extension work (step 3), the bake-off results must clear these thresholds. If no candidate clears the bar, return to step 1 (re-assess model size, data quality, or grounding schema).

---

## 1. Bake-off Results Table

Fill in after running `uv run python -m training.bakeoff`:

| Candidate | Params | eval_loss | Accuracy | Mean latency | p95 latency | Model size (Q4) | Pass? |
|---|---|---|---|---|---|---|---|
| LFM2.5-350M | 354M | _ | _% | _ms | _ms | ~175MB | ✅/❌ |
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

**Candidate selected:** _______________________

**Decision:** ☐ GO — proceed to SDK/extension (step 3)
            ☐ NO-GO — re-assess step 1

**Rationale / blockers:**

_______________________________________________________________

**Date:** _______________

---

## 5. SDK / Extension Work (step 3) — Unblock Checklist

Once GO is declared, these must be true before starting step 3:

- [ ] Fine-tuned adapter committed to `training/runs/<candidate>/final`
- [ ] GGUF export at `export/out/<candidate>-q4_k_m.gguf`
- [ ] Bench results committed to repo (`runtime-bench/out/`)
- [ ] Selected model ID and rationale added to `docs/prd.md`
