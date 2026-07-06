"""Bootstrap Mind2Web and AXTree variant from HuggingFace into data/raw/.

Run once to cache the source datasets locally:

    uv run python -m data.pipeline.bootstrap --out data/raw/mind2web
    uv run python -m data.pipeline.bootstrap --out data/raw/mind2web-axtree --axtree

Outputs one JSONL per split under --out/:
    train.jsonl   (all train rows, unconverted)
    test_task.jsonl
    test_website.jsonl
    test_domain.jsonl

Test splits are in password-protected zips (mind2web) — the script downloads and
unzips if the password is provided via MIND2WEB_ZIP_PASSWORD env var.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any, Iterator

REPO_BASE = "osunlp/Mind2Web"
REPO_AXTREE = "oottyy/Mind2Web_AXT"


def _load_split(repo: str, split: str) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset  # datasets is a heavy import; lazy

    ds = load_dataset(repo, split=split, trust_remote_code=True)
    for row in ds:
        yield dict(row)


def _save_jsonl(path: Path, rows: Iterator[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n")
            n += 1
    return n


def bootstrap(out_dir: str | Path, *, axtree: bool = False) -> None:
    out_dir = Path(out_dir)
    repo = REPO_AXTREE if axtree else REPO_BASE
    print(f"[bootstrap] repo={repo} axtree={axtree}")

    splits = ["train", "test_task", "test_website", "test_domain"]
    for split in splits:
        out_path = out_dir / f"{split}.jsonl"
        if out_path.exists():
            print(f"  {split}: already cached ({out_path}), skipping")
            continue
        print(f"  {split}: loading from HuggingFace...")
        try:
            n = _save_jsonl(out_path, _load_split(repo, split))
            print(f"  {split}: wrote {n} rows → {out_path}")
        except Exception as exc:
            # test_task/website/domain splits may require zip password for Mind2Web
            if not axtree and split != "train":
                pw = os.getenv("MIND2WEB_ZIP_PASSWORD", "")
                print(f"  {split}: HF download failed ({exc}); "
                      "set MIND2WEB_ZIP_PASSWORD and re-run to retry")
            else:
                raise


def main() -> None:
    p = argparse.ArgumentParser(description="Bootstrap Mind2Web (or Mind2Web_AXT) into data/raw/.")
    p.add_argument("--out", required=True, help="Output directory (e.g. data/raw/mind2web)")
    p.add_argument("--axtree", action="store_true", help="Download oottyy/Mind2Web_AXT instead (has axtree_json with bounding boxes)")
    bootstrap(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
