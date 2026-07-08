#!/usr/bin/env bash
# Upload HTLM Q8 GGUF + model card to HuggingFace
# Usage: HF_TOKEN=hf_xxxx bash scripts/hf-upload.sh
# (Token needs "write" access for espetro or HackBarna namespace)
# Requires: `hf` CLI v1.19+ (Rust CLI, https://huggingface.co/docs/huggingface_hub/guides/cli)

set -euo pipefail

REPO="espetro/htlm-lfm2.5-350m"
GGUF="export/out/lfm2.5-350m-q8_0.gguf"
CARD="docs/model-card-template.md"
TOKEN="${HF_TOKEN:-}"

if [[ -z "$TOKEN" ]]; then
    echo "Error: HF_TOKEN not set. Get a write token at https://huggingface.co/settings/tokens"
    exit 1
fi

if ! command -v hf &>/dev/null; then
    echo "Error: 'hf' CLI not found. Install: https://huggingface.co/docs/huggingface_hub/guides/cli"
    exit 1
fi

echo "=== Logging in ==="
hf auth login --token "$TOKEN"

echo "=== Creating repo (if not exists) ==="
hf repos create "$REPO" --type model --exist-ok --token "$TOKEN"

echo "=== Uploading GGUF (362 MB) ==="
hf upload "$REPO" "$GGUF" --token "$TOKEN"

echo "=== Uploading model card as README.md ==="
hf upload "$REPO" "$CARD" "README.md" --token "$TOKEN"

echo "=== Done ==="
echo "Model: https://huggingface.co/$REPO"
