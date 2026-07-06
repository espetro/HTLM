# Export

Convert fine-tuned LoRA adapters + base models into deployable formats:
- **GGUF** — for `wllama` (WASM, runs in Chrome/Firefox)
- **ONNX** — for `onnxruntime-web` (fallback, no WASM needed)

## GGUF Export

```bash
# Install llama.cpp (one-time)
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && mkdir build && cd build && cmake .. -DLLAMA_BUILD_EXAMPLES=ON && make
export LLAMA_CPP_DIR=$PWD

# Export a fine-tuned model
uv run python -m export.export \
    --adapter training/runs/lfm2.5-350m/final \
    --base LiquidAI/LFM2.5-350M \
    --out export/out/lfm2.5-350m-q4_k_m.gguf \
    --quantize q4_k_m
```

The script:
1. Merges the LoRA adapter into the base model (FP16).
2. Converts to FP32 GGUF via `convert_hf_to_gguf.py`.
3. Quantizes to the target precision (default `q4_k_m`, ~700MB for 350M models).

## Quantization Types

| Type | Size vs FP16 | Quality | Recommended for |
|---|---|---|---|
| `f16` | 100% | baseline | testing only |
| `q4_k_m` | ~25% | near-lossless | **default** — <1GB target |
| `q4_k_s` | ~20% | good | smaller-than-q4_k_m |
| `q8_0` | ~50% | very high | when size budget allows |

## ONNX Export (optional fallback)

```bash
# Convert merged model to ONNX
optimum-cli export onnx \
    --model training/runs/lfm2.5-350m/final \
    --task text-generation \
    export/out/lfm2.5-350m.onnx
```
