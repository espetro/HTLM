# HTLM Data Pipeline

Converts web-agent datasets into the HTLM grounding format: `{instruction, page, action}`.

## Record Format

```json
{
  "instruction": "Search for flights from NYC to LA",
  "page": {
    "elements": [
      {"index": 0, "role": "combobox", "label": "From", "tag": "input"},
      {"index": 1, "role": "button", "label": "Search", "tag": "button"}
    ]
  },
  "action": {"type": "click", "index": 1}
}
```

Full schemas: `../schema/page-representation.json` and `../schema/action.json`.

## Pipeline Stages

```
bootstrap → map → (optional: distill) → split → formatted for training
```

### 1. Bootstrap

Download source datasets from HuggingFace into `data/raw/`.

```bash
# Standard Mind2Web (no bounding boxes)
uv run python -m data.pipeline.cli bootstrap --out data/raw/mind2web

# Mind2Web_AXT variant (has axtree_json with bounding boxes in CSS px)
uv run python -m data.pipeline.cli bootstrap --out data/raw/mind2web-axtree --axtree
```

Output: `data/raw/<name>/{train,test_task,test_website,test_domain}.jsonl` — raw HF rows, one per line.

Note: test splits (`test_task`, `test_website`, `test_domain`) for the standard Mind2Web are in password-protected zip archives. Set `MIND2WEB_ZIP_PASSWORD` env var and re-run to extract them. The AXTree variant downloads without a password.

### 2. Map

Convert raw Mind2Web rows into HTLM training records.

```bash
uv run python -m data.pipeline.cli map-mind2web \
    --src data/raw/mind2web-axtree \
    --out data/processed/mind2web-htlm.jsonl
```

Each input split produces records in the same output JSONL (the split label is not preserved — add it to `meta` if needed).

The mapper:
- Merges `neg_candidates` + `pos_candidates` into the flat `elements` list (neg first).
- Uses `is_original_target: true` to find the canonical element index.
- Maps `operation.op` → `click` / `type` / `select`; `operation.value` → `text` or `value`.
- Reads `axtree_json[*].bounding_box_rect` for bounds when available.
- Sets `meta.task_id = annotation_id` (enables task-grouped eval split).

Unsupported ops (HOVER, ENTER, etc.) are silently skipped.

### 3. Teacher Distillation (optional)

Augment records with a teacher LLM. Input records must have `{instruction, page}` and may have an existing `action` (ignored).

```bash
export TEACHER_API_KEY=sk-...           # required
export TEACHER_MODEL=gpt-4o-mini        # optional, default: gpt-4o-mini
export TEACHER_BASE_URL=...             # optional, for compatible endpoints
export TEACHER_JSON_MODE=auto           # auto|1|0 — auto detects openai.com

uv run python -m data.pipeline.cli distill \
    --in data/processed/eval_raw.jsonl \
    --out data/distilled/teacher-eval.jsonl \
    --model gpt-4o-mini
```

Output: records with `meta.teacher = <model>` provenance field.

### 4. Split

Deterministic train/eval split. Stable across runs and augmentation — same task always lands on the same side.

```bash
uv run python -m data.pipeline.cli split \
    --in data/processed/mind2web-htlm.jsonl \
    --train data/processed/train.jsonl \
    --eval data/processed/eval.jsonl \
    --ratio 0.1
```

Split key: `meta.task_id` when present, else `page.url ⊕ instruction`. Evaluates to a 256-bucket hash — records with hash < `ratio × 256` go to eval.

### 5. Format for Training

Use `data/pipeline/prompt.py`:

```python
from data.pipeline.prompt import training_messages, SYSTEM_PROMPT

record = {"instruction": "...", "page": {...}, "action": {...}}
messages = training_messages(record)  # → [{"role":"system",...}, {"role":"user",...}, {"role":"assistant",...}]
```

Then tokenize with the base model's chat template:

```python
tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
```

## Datasets

| Dataset | License | Notes |
|---|---|---|
| `osunlp/Mind2Web` | CC BY 4.0 | Static tasks, pre-pruned candidates, no bbox |
| `oottyy/Mind2Web_AXT` | CC BY 4.0 | Same, plus `axtree_json` with bbox in CSS px |
| `osunlp/Online-Mind2Web` | CC BY 4.0 | Live trajectories, no pre-pruned candidates |
| `web-arena-x/webarena` | Apache-2.0 | Configs + live env, bbox via accessibility tree |

Mind2Web is the bootstrap priority. Online-Mind2Web and WebArena need a live browser environment and are lower priority for initial fine-tuning data.

## Data Management

Large regenerable artifacts are gitignored:
- `data/raw/` — downloaded HF datasets
- `data/processed/` — mapped JSONL records
- `data/distilled/` — teacher outputs

Keep `data/schema/`, `data/pipeline/`, and `data/samples/` tracked.
