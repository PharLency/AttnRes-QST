#!/usr/bin/env bash
# Tiny GPU smoke test for QST and AttnRes-QST.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL_DIR="${MODEL_DIR:-.models/opt-1.3b}"
MODEL_TAG="$(basename "$MODEL_DIR")"
TASKS="${TASKS:-sst2 mrpc rte}"
SEED="${SEED:-42}"
BS="${BS:-4}"
SEQLEN="${SEQLEN:-128}"
MAX_TRAIN="${MAX_TRAIN:-100}"
MAX_EVAL="${MAX_EVAL:-32}"
LR="${LR:-2e-4}"
GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    echo "ERROR: $MODEL_DIR/config.json not found. Run scripts/download_models.sh first." >&2
    exit 1
fi

mkdir -p outputs/.smoke logs

for task in $TASKS; do
    for method in qst attnres-block-bs2; do
        outdir="outputs/.smoke/${MODEL_TAG}/${method}/${task}-s${SEED}"
        log="logs/smoke-${MODEL_TAG}-${method}-${task}-s${SEED}.log"
        if [[ -f "$outdir/final_metrics.json" ]]; then
            echo "[skip] $outdir"
            continue
        fi

        cli="--task $task --seed $SEED --model_dir $MODEL_DIR"
        cli+=" --epochs 1 --batch_size $BS --max_seqlen $SEQLEN"
        cli+=" --max_train_samples $MAX_TRAIN --max_eval_samples $MAX_EVAL"
        cli+=" --lr $LR --use_4bit --bf16 --logging_steps 5"
        if [[ "$method" == "attnres-block-bs2" ]]; then
            cli+=" --use_attn_residual --attn_residual_type block --block_size 2"
        fi

        mkdir -p "$outdir"
        echo "[smoke] GPU $GPU  $method  $task  -> $outdir"
        CUDA_VISIBLE_DEVICES="$GPU" python src/train_glue.py $cli --output_dir "$outdir" > "$log" 2>&1
    done
done

echo "[done] smoke outputs are under outputs/.smoke/"

