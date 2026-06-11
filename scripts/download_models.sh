#!/usr/bin/env bash
# scripts/download_models.sh
# Download OPT backbones via huggingface-cli to .models/<tag>/
# Default: opt-1.3b, opt-2.7b, opt-6.7b (those required by the main-experiment
# hyperparameter table).
#
# Idempotent: each model is skipped if .models/<tag>/config.json already exists.
#
# Approx sizes (disk, fp16/bf16 pickle or safetensors, as HF hosts them):
#   opt-1.3b   2.5 GB
#   opt-2.7b   5.3 GB
#   opt-6.7b  13.0 GB
# Budget ~21 GB disk in total.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# Models to pull (space-separated list). Each must be a HF repo id whose local
# tag is the repo id without the org prefix.
MODELS="${MODELS:-facebook/opt-1.3b facebook/opt-2.7b facebook/opt-6.7b}"

mkdir -p .models

echo "[models] HF endpoint: $HF_ENDPOINT"

for repo in $MODELS; do
    tag="${repo##*/}"
    out=".models/$tag"
    if [[ -f "$out/config.json" ]]; then
        echo "[skip] $tag (already present at $out)"
        continue
    fi
    echo ""
    echo "[pull] $repo -> $out"
    huggingface-cli download "$repo" \
        --local-dir "$out" \
        --local-dir-use-symlinks False
done

echo ""
echo "[done] models under .models/:"
du -sh .models/*/ 2>/dev/null || true
