#!/usr/bin/env bash
# Upload HTLM Q8 GGUF + model card to HuggingFace
# Usage: HF_TOKEN=hf_xxxx bash scripts/hf-upload.sh
# (Token needs "write" access for espetro or HackBarna namespace)

set -euo pipefail

REPO="espetro/htlm-lfm2.5-350m"
GGUF="export/out/lfm2.5-350m-q8_0.gguf"
CARD="docs/model-card-template.md"
TOKEN="${HF_TOKEN:-}"

if [[ -z "$TOKEN" ]]; then
    echo "Error: HF_TOKEN not set. Get a token at https://huggingface.co/settings/tokens"
    echo "Token needs 'write' permission for espetro namespace."
    exit 1
fi

echo "=== Logging in ==="
 huggingface-cli login --token "$TOKEN"

echo "=== Creating repo ==="
 huggingface-cli repo create "$REPO" --type model --organization HackBarna --exist-ok

echo "=== Uploading GGUF (362 MB) ==="
 huggingface-cli upload "$REPO" \
    "$GGUF" \
    --token "$TOKEN" \
    --repo-type model

echo "=== Uploading model card ==="
 huggingface-cli upload "$REPO" \
    "$CARD" \
    --token "$TOKEN" \
    --repo-type model \
    --filename README.md

echo "=== Done ==="
echo "Model: https://huggingface.co/$REPO"
