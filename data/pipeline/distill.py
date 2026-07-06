"""Teacher distillation scaffold.

Reads candidate records ({instruction, page}, action ignored if present),
asks a teacher LLM for a grounded action using the SAME prompt the student will
see, validates the reply against action.json, and writes a training record with
provenance in meta.

Gated on env: TEACHER_API_KEY (required), TEACHER_BASE_URL (default OpenAI),
TEACHER_MODEL (default gpt-4o-mini), TEACHER_JSON_MODE (default auto).

ponytail: a single teacher pass, no self-consistency / reranking. Add those if
grounding quality on the eval split is too low.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any

from data.pipeline.prompt import SYSTEM_PROMPT, input_messages
from data.pipeline.records import read_jsonl, validate_record, write_jsonl

_JSON_RE = re.compile(r"\{[\s\S]*\}")


def _extract_json(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
    m = _JSON_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _client():
    key = os.getenv("TEACHER_API_KEY")
    if not key:
        raise SystemExit("TEACHER_API_KEY not set; export it or skip distillation.")
    from openai import OpenAI  # optional dep

    return OpenAI(api_key=key, base_url=os.getenv("TEACHER_BASE_URL", "https://api.openai.com/v1"))


def _want_json_mode() -> bool:
    mode = os.getenv("TEACHER_JSON_MODE", "auto").lower()
    if mode == "1" or mode == "true":
        return True
    if mode == "0" or mode == "false":
        return False
    return "openai.com" in os.getenv("TEACHER_BASE_URL", "https://api.openai.com/v1")


def distill_one(client, model: str, record: dict[str, Any]) -> dict[str, Any] | None:
    msgs = input_messages(record, SYSTEM_PROMPT)
    kwargs: dict[str, Any] = {"model": model, "messages": msgs, "temperature": 0.0}
    if _want_json_mode():
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    content = resp.choices[0].message.content or ""
    action = _extract_json(content)
    if action is None:
        return None
    candidate = {"instruction": record["instruction"], "page": record["page"], "action": action}
    ok, _ = validate_record(candidate)
    return action if ok else None


def main() -> None:
    p = argparse.ArgumentParser(description="Teacher-distill actions for candidate (page,instruction) records.")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    model = args.model or os.getenv("TEACHER_MODEL", "gpt-4o-mini")
    client = _client()

    try:
        from tqdm import tqdm  # optional dep
        rows = list(read_jsonl(args.inp))
        if args.limit:
            rows = rows[: args.limit]
        it = tqdm(rows, desc="distill")
    except ImportError:
        it = iter(read_jsonl(args.inp))

    out, n_skip = [], 0
    for r in it:
        action = distill_one(client, model, r)
        if action is None:
            n_skip += 1
            continue
        out.append({
            "instruction": r["instruction"],
            "page": r["page"],
            "action": action,
            "meta": {"teacher": model, **(r.get("meta") or {})},
        })
    n = write_jsonl(args.out, out)
    print(f"wrote={n} skipped={n_skip} model={model}")


if __name__ == "__main__":
    main()
