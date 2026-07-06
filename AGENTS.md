# Agent context

## Project

GitHub Project: https://github.com/users/espetro/projects/21

## Issue tracker

GitHub Issues on this repo (`espetro/HTLM`).

## Roadmap

See `docs/prd.md` for product roadmap.

## Build Status (Step 2 — all code committed)

| Issue | Title | Status |
|---|---|---|
| #1 | Feasibility assessment | ✅ Done (committed 6b7c429) |
| #2 | Schema: page-representation + action | ✅ Done (committed ddb12d3) |
| #3 | Hybrid data pipeline | ✅ Done (committed ed079aa) |
| #4 | Fine-tune bake-off scripts | ✅ Done (committed 7e668a7) |
| #5 | Export + runtime-bench harness | ✅ Done (committed 9e2d822) |
| #6 | Go/no-go decision framework | ✅ Done (committed 19b710b) |

**All step 2 code is committed.** Remaining: run the actual bake-off (requires GPU rental + time), fill in `docs/go-no-go-checklist.md` with results.

## Key Files

- `data/schema/` — JSON schemas (page-representation, action, example)
- `data/pipeline/` — bootstrap, map, split, distill, CLI
- `training/` — LoRA/QLoRA train.py, bakeoff.py, per-candidate configs
- `export/export.py` — LoRA → GGUF export + quantization
- `runtime-bench/bench.py` — cross-browser WASM benchmark harness
- `docs/go-no-go-checklist.md` — decision framework (fill in after bake-off)
