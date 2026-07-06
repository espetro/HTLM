"""Export a fine-tuned LoRA adapter + base model to GGUF for wllama.

Usage:
    uv run python -m export.export \
        --adapter training/runs/lfm2.5-350m/final \
        --base LiquidAI/LFM2.5-350M \
        --out export/out/lfm2.5-350m-q4_k_m.gguf \
        --quantize q4_k_m

The standard approach:
  1. Merge LoRA adapter into base model (transformers).
  2. Convert to llama.cpp format (llama.cpp/convert_hf_to_gguf.py).
  3. Quantize (llama.cpp/quantize).

Requirements: llama.cpp installed at $LLAMA_CPP_DIR or in PATH.
Install: git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && mkdir build && cd build && cmake .. && make
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _find_llama_cpp() -> Path:
    """Locate llama.cpp build directory."""
    candidates = [
        Path(os.environ.get("LLAMA_CPP_DIR", "")) / "build",
        Path.home() / "llama.cpp" / "build",
        Path("/usr/local/bin"),  # if installed via make install
    ]
    for c in candidates:
        if c.exists() and (c / "quantize").exists():
            return c
    raise SystemExit(
        "llama.cpp not found. Set LLAMA_CPP_DIR or clone and build:\n"
        "  git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp/build\n"
        "  cmake .. -DLLAMA_BUILD_EXAMPLES=ON && make"
    )


def export_to_gguf(
    adapter_path: str | Path,
    base_model_id: str,
    output_path: str | Path,
    quantize: str = "q4_k_m",
) -> None:
    adapter_path = Path(adapter_path)
    output_path = Path(output_path)
    base_model_id = str(base_model_id)

    llama_build = _find_llama_cpp()
    convert_script = llama_build.parent / "convert_hf_to_gguf.py"
    quantize_bin = llama_build / "quantize"

    if not convert_script.exists():
        raise SystemExit(f"convert_hf_to_gguf.py not found at {convert_script}")
    if not quantize_bin.exists():
        raise SystemExit(f"quantize binary not found at {quantize_bin}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        merged_dir = Path(tmpdir) / "merged"
        print(f"[export] merging LoRA adapter into {base_model_id} ...")
        # ── merge LoRA ──────────────────────────────────────────────────────────
        merge_code = f"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained(
    "{base_model_id}",
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("{base_model_id}", trust_remote_code=True)
model = PeftModel.from_pretrained(base, "{adapter_path}")
merged = model.merge_and_unload()
merged.save_pretrained("{merged_dir}")
tokenizer.save_pretrained("{merged_dir}")
print("Merge complete.")
"""
        result = subprocess.run(
            [sys.executable, "-c", merge_code],
            capture_output=True, text=True, timeout=3600
        )
        if result.returncode != 0:
            raise SystemExit(f"Merge failed:\n{result.stderr}")
        print("[export] merge done.")

        # ── convert to GGUF ────────────────────────────────────────────────────
        fp32_path = output_path.with_suffix(".f32.gguf")
        print(f"[export] converting to FP32 GGUF → {fp32_path} ...")
        result = subprocess.run(
            [sys.executable, str(convert_script), str(merged_dir), "--outfile", str(fp32_path)],
            capture_output=True, text=True, timeout=3600
        )
        if result.returncode != 0:
            raise SystemExit(f"GGUF conversion failed:\n{result.stderr}\n{result.stdout}")
        print("[export] FP32 GGUF written.")

        # ── quantize ──────────────────────────────────────────────────────────
        print(f"[export] quantizing to {quantize} → {output_path} ...")
        result = subprocess.run(
            [str(quantize_bin), str(fp32_path), str(output_path), quantize],
            capture_output=True, text=True, timeout=3600
        )
        if result.returncode != 0:
            raise SystemExit(f"Quantization failed:\n{result.stderr}")
        print(f"[export] done → {output_path}")

        # Clean up FP32 intermediate
        fp32_path.unlink(missing_ok=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Export LoRA adapter + base model to GGUF.")
    p.add_argument("--adapter", required=True, help="Path to merged LoRA adapter output dir")
    p.add_argument("--base", required=True, help="HuggingFace model id or local path")
    p.add_argument("--out", required=True, help="Output GGUF path")
    p.add_argument(
        "--quantize",
        default="q4_k_m",
        choices=["q4_k_m", "q4_k_s", "q8_0", "f16"],
        help="Quantization type (default: q4_k_m; f16 = no quantization)",
    )
    args = p.parse_args()
    export_to_gguf(args.adapter, args.base, args.out, args.quantize)


if __name__ == "__main__":
    main()
