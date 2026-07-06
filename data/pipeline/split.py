"""Deterministic train/eval split.

Splits on the task unit (meta.task_id if present, else url+instruction) so all
steps of one task land on one side. Stable across runs and across later
distilled augmentation: the same task always maps to the same bucket, so new
synthetic data cannot leak eval tasks into train.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

from data.pipeline.records import read_jsonl, write_jsonl

_SALT = "htlm-split-v1"


def split_key(record: dict[str, Any]) -> str:
    meta = record.get("meta") or {}
    task_id = meta.get("task_id")
    if task_id:
        return f"task:{task_id}"
    page = record.get("page") or {}
    return f"{page.get('url', '')}\x1f{record.get('instruction', '')}"


def _bucket(record: dict[str, Any]) -> int:
    return hashlib.sha256(f"{_SALT}\x1f{split_key(record)}".encode()).digest()[0]


def is_eval(record: dict[str, Any], eval_ratio: float = 0.1) -> bool:
    return _bucket(record) < int(eval_ratio * 256)


def split_jsonl(in_path: str | Path, train_path: str | Path, eval_path: str | Path, eval_ratio: float = 0.1) -> tuple[int, int]:
    train, eval_ = [], []
    for r in read_jsonl(in_path):
        (eval_ if is_eval(r, eval_ratio) else train).append(r)
    n_train = write_jsonl(train_path, train)
    n_eval = write_jsonl(eval_path, eval_)
    return n_train, n_eval


def main() -> None:
    p = argparse.ArgumentParser(description="Deterministic train/eval split of a record JSONL.")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--train", required=True)
    p.add_argument("--eval", dest="evl", required=True)
    p.add_argument("--ratio", type=float, default=0.1)
    args = p.parse_args()
    n_train, n_eval = split_jsonl(args.inp, args.train, args.evl, args.ratio)
    print(f"train={n_train} eval={n_eval} (ratio={args.ratio})")


if __name__ == "__main__":
    main()
