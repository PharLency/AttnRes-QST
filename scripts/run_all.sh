#!/usr/bin/env bash
# scripts/run_all.sh
#
# ONE-CLICK driver: run EVERYTHING (download data + download models + smoke
# + main experiments + ablations + summarize) for one, some, or all of
# {opt-1.3b, opt-2.7b, opt-6.7b}.
#
# Semantic per model:
#     1. (optional) smoke test: quick sanity on SST-2 with tiny samples.
#     2. main experiments    (scripts/run_parallel.sh MODEL_DIR=...)
#     3. ablation experiments (scripts/run_ablations.sh MODEL_DIR=...)
#     4. summarize            (python src/summarize_results.py)
#
# Steps are idempotent — each underlying launcher skips completed runs.
#
# Usage:
#     bash scripts/run_all.sh                                    # all 3 models
#     MODELS="opt-1.3b"          bash scripts/run_all.sh         # single model
#     MODELS="opt-1.3b opt-2.7b" bash scripts/run_all.sh         # subset
#     SKIP_SMOKE=1               bash scripts/run_all.sh         # no smoke
#     SKIP_ABLATIONS=1           bash scripts/run_all.sh         # main only
#     JOBS_PER_GPU=2 SEEDS=42    bash scripts/run_all.sh         # fast mode
#     DATA_ONLY=1 MODELS_ONLY=1  bash scripts/run_all.sh         # just setup

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODELS="${MODELS:-opt-1.3b opt-2.7b opt-6.7b}"
SEEDS="${SEEDS:-42 123 2024}"
TASKS="${TASKS:-cola sst2 mrpc rte qnli qqp stsb mnli}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"
SKIP_SMOKE="${SKIP_SMOKE:-0}"
SKIP_ABLATIONS="${SKIP_ABLATIONS:-0}"
SKIP_MAIN="${SKIP_MAIN:-0}"
DATA_ONLY="${DATA_ONLY:-0}"
MODELS_ONLY="${MODELS_ONLY:-0}"

section () {
    echo ""
    echo "############################################################"
    echo "# $1"
    echo "############################################################"
}

# ---------------------------------------------------------------------------
section "Step 0: data + models"
# ---------------------------------------------------------------------------

bash scripts/download_data.sh

# Build a space-separated list of HF repo ids from MODELS tags.
REPO_LIST=""
for tag in $MODELS; do
    REPO_LIST+=" facebook/$tag"
done
MODELS="$MODELS" HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" \
MODELS_="$REPO_LIST" bash -c '
    MODELS="$MODELS_" bash scripts/download_models.sh
'

if [[ "$DATA_ONLY" == "1" || "$MODELS_ONLY" == "1" ]]; then
    echo ""
    echo "[done] data + models only (DATA_ONLY/MODELS_ONLY set)."
    exit 0
fi

# ---------------------------------------------------------------------------
# Quick per-model smoke (tiny sample) to catch env breakage before the long
# main run. Runs just SST-2 + QST at 50 train samples / 1 epoch / bs=4.
# ---------------------------------------------------------------------------
smoke_one_model () {
    local tag="$1"
    local out="outputs/.smoke/${tag}"
    if [[ -f "$out/final_metrics.json" ]]; then
        echo "[smoke ${tag}] already done, skipping"
        return 0
    fi
    mkdir -p outputs/.smoke
    local gpu="$(echo ${GPU_IDS:-0} | awk '{print $1}')"
    echo "[smoke ${tag}] GPU $gpu  .../sst2 QST, 50 examples, 1 epoch"
    CUDA_VISIBLE_DEVICES=$gpu python src/train_glue.py \
        --task sst2 --seed 42 \
        --model_dir ".models/${tag}" \
        --max_train_samples 50 --max_eval_samples 16 \
        --epochs 1 --batch_size 4 --max_seqlen 128 \
        --lr 2e-4 --use_4bit --bf16 \
        --logging_steps 5 \
        --output_dir "$out" > "logs/smoke-${tag}.log" 2>&1
    if [[ ! -f "$out/final_metrics.json" ]]; then
        echo "[smoke ${tag}] FAILED, see logs/smoke-${tag}.log" >&2
        return 1
    fi
    echo "[smoke ${tag}] ok"
}

# ---------------------------------------------------------------------------
# Run main + ablations per model, sequentially.
# ---------------------------------------------------------------------------
for tag in $MODELS; do
    model_dir=".models/${tag}"
    if [[ ! -f "$model_dir/config.json" ]]; then
        echo ""
        echo "[warn] skipping $tag: $model_dir/config.json not found (download failed?)"
        continue
    fi

    if [[ "$SKIP_SMOKE" != "1" ]]; then
        section "Smoke test: $tag"
        smoke_one_model "$tag" || { echo "[abort] smoke failed for $tag"; exit 2; }
    fi

    if [[ "$SKIP_MAIN" != "1" ]]; then
        section "MAIN experiments: $tag (4 methods × $(echo $TASKS | wc -w) tasks × $(echo $SEEDS | wc -w) seeds)"
        SEEDS="$SEEDS" TASKS="$TASKS" JOBS_PER_GPU="$JOBS_PER_GPU" \
        MODEL_DIR="$model_dir" bash scripts/run_parallel.sh
    fi

    if [[ "$SKIP_ABLATIONS" != "1" ]]; then
        section "ABLATIONS: $tag"
        SEEDS="$SEEDS" TASKS="$TASKS" JOBS_PER_GPU="$JOBS_PER_GPU" \
        MODEL_DIR="$model_dir" bash scripts/run_ablations.sh
    fi
done

# ---------------------------------------------------------------------------
section "Summarize all results"
# ---------------------------------------------------------------------------
python src/summarize_results.py

echo ""
echo "############################################################"
echo "# [✓] run_all.sh complete"
echo "#"
echo "#   results/  <- main.csv + main.md"
echo "#   outputs/<model>/<method>/<task>-s<seed>/  <- per-run artefacts"
echo "#   logs/     <- per-run stdout/stderr"
echo "############################################################"
