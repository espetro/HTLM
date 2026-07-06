"""CLI entry point for the HTLM data pipeline.

Usage:
    # Bootstrap source datasets from HuggingFace
    uv run python -m data.pipeline.cli bootstrap --out data/raw/mind2web
    uv run python -m data.pipeline.cli bootstrap --out data/raw/mind2web-axtree --axtree

    # Map Mind2Web rows to HTLM records (one JSONL per split)
    uv run python -m data.pipeline.cli map-mind2web \
        --src data/raw/mind2web-axtree \
        --out data/processed/mind2web-htlm.jsonl

    # Deterministic train/eval split
    uv run python -m data.pipeline.cli split \
        --in data/processed/mind2web-htlm.jsonl \
        --train data/processed/train.jsonl \
        --eval data/processed/eval.jsonl

    # Teacher distillation (requires TEACHER_API_KEY)
    uv run python -m data.pipeline.cli distill \
        --in data/processed/eval_raw.jsonl \
        --out data/distilled/teacher-eval.jsonl \
        --model gpt-4o-mini
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data.pipeline import bootstrap, map_mind2web, split, distill
from data.pipeline.records import read_jsonl, write_jsonl, validate_record


def _cmd_bootstrap(args: argparse.Namespace) -> None:
    bootstrap.bootstrap(args.out, axtree=args.axtree)


def _cmd_map_mind2web(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_total, n_ok = 0, 0
    for split in ["train", "test_task", "test_website", "test_domain"]:
        src_file = Path(args.src) / f"{split}.jsonl"
        if not src_file.exists():
            continue
        split_records = []
        for row in read_jsonl(src_file):
            n_total += 1
            recs = map_mind2web.map_row(row)
            for rec in recs:
                obj = rec.to_dict()
                ok, _ = validate_record(obj)
                if ok:
                    n_ok += 1
                    split_records.append(obj)
        n_written = write_jsonl(out, split_records)
        print(f"  {split}: {len(split_records)} records written (ok={n_ok}/{n_total})")


def _cmd_split(args: argparse.Namespace) -> None:
    n_train, n_eval = split.split_jsonl(args.in_, args.train, args.evl, args.ratio)
    print(f"train={n_train} eval={n_eval}")


def _cmd_distill(args: argparse.Namespace) -> None:
    distill.main()  # its main() handles its own argparse


def main() -> None:
    p = argparse.ArgumentParser(description="HTLM data pipeline CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    bp = sub.add_parser("bootstrap", help="Download Mind2Web from HuggingFace")
    bp.add_argument("--out", required=True)
    bp.add_argument("--axtree", action="store_true")

    mp = sub.add_parser("map-mind2web", help="Map Mind2Web rows to HTLM records")
    mp.add_argument("--src", required=True, help="Output dir from bootstrap (data/raw/mind2web-axtree)")
    mp.add_argument("--out", required=True, help="Output JSONL path")

    sp = sub.add_parser("split", help="Split a JSONL into train/eval")
    sp.add_argument("--in", dest="in_", required=True)
    sp.add_argument("--train", required=True)
    sp.add_argument("--eval", dest="evl", required=True)
    sp.add_argument("--ratio", type=float, default=0.1)

    dp = sub.add_parser("distill", help="Teacher-distill actions for (page,instruction) records")
    dp.add_argument("--in", dest="in_", required=True)
    dp.add_argument("--out", required=True)
    dp.add_argument("--model", default=None)
    dp.add_argument("--limit", type=int, default=None)

    args = p.parse_args()
    if args.cmd == "bootstrap":
        _cmd_bootstrap(args)
    elif args.cmd == "map-mind2web":
        _cmd_map_mind2web(args)
    elif args.cmd == "split":
        _cmd_split(args)
    elif args.cmd == "distill":
        _cmd_distill(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
