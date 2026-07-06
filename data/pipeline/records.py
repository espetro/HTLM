"""Canonical training record IO + schema validation.

A training record is the unit the grounding model learns from:

    {"instruction": str, "page": <page-representation.json>, "action": <action.json>}

`meta` is optional provenance (source dataset, teacher model, task id) and is
ignored by the prompt builder.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

SCHEMA_DIR = Path(__file__).resolve().parent.parent / "schema"


@dataclass
class TrainingRecord:
    instruction: str
    page: dict[str, Any]
    action: dict[str, Any]
    meta: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "instruction": self.instruction,
            "page": self.page,
            "action": self.action,
        }
        if self.meta:
            obj["meta"] = self.meta
        return obj


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


def _load_schema(name: str) -> dict[str, Any]:
    with open(SCHEMA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def validate_record(record: dict[str, Any]) -> tuple[bool, list[str]]:
    """Validate page + action against the draft-07 schemas. Returns (ok, errors)."""
    import jsonschema  # optional dep; import lazily

    errors: list[str] = []
    schemas = {
        "page": _load_schema("page-representation.json"),
        "action": _load_schema("action.json"),
    }
    for field, schema in schemas.items():
        if field not in record:
            errors.append(f"missing field: {field}")
            continue
        for e in jsonschema.Draft7Validator(schema).iter_errors(record[field]):
            errors.append(f"{field}: {e.message}")
    return (not errors, errors)
