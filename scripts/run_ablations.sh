#!/usr/bin/env bash
# scripts/run_ablations.sh
#
# Ablation experiments for one model, covering the same 8 GLUE tasks as the
# main run. Uses the same per-(model, task) batch-size / per-model epoch
# matrix from Table 8 so ablation numbers are directly comparable.
#
# Two ablations covered here (both across all 8 tasks):
#   P1 Uniform-Average residual (freeze_attn_query): proves gains come from
#      the LEARNABLE attention, not mere multi-layer averaging.
#   P3 Pre-attention Norm (RTE only, LayerNorm and none — RMSNorm is already
#      covered by the main run's attnres-block-bs2 method).
#
# The block-size=4 ablation is already part of the MAIN run, so it is NOT
# duplicated here.
#
# Output layout (same nested scheme as run_parallel.sh):
#     outputs/<model_tag>/attnres-uniform-avg/<task>-s<seed>/
#     outputs/<model_tag>/attnres-block-bs2-norm-{layernorm,none}/rte-s<seed>/
#
# Usage:
#     SEEDS="42 123 2024" MODEL_DIR=.models/opt-1.3b bash scripts/run_ablations.sh
#     JOBS_PER_GPU=2 MODEL_DIR=.models/opt-1.3b bash scripts/run_ablations.sh
#     GPU_IDS="0 1 2 3" MODEL_DIR=.models/opt-1.3b bash scripts/run_ablations.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

SEEDS="${SEEDS:-42 123 2024}"
TASKS="${TASKS:-cola sst2 mrpc rte qnli qqp stsb mnli}"
MODEL_DIR="${MODEL_DIR:-.models/opt-1.3b}"
MODEL_TAG="$(basename "$MODEL_DIR")"
LR="${LR:-2e-4}"
SEQLEN="${SEQLEN:-512}"
LOGGING_STEPS="${LOGGING_STEPS:-50}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"

epochs_for_model () {
    case "$1" in
        opt-1.3b) echo 20 ;;
        opt-2.7b) echo 15 ;;
        opt-6.7b) echo 10 ;;
        *)        echo "${EPOCHS:-20}" ;;
    esac
}
EPOCHS="${EPOCHS:-$(epochs_for_model "$MODEL_TAG")}"

bs_for_model_task () {
    local m=$1 t=$2
    case "$m" in
        opt-1.3b)
            case "$t" in mrpc|qnli|qqp) echo 8  ;; *) echo 32 ;; esac ;;
        opt-2.7b)
            case "$t" in mrpc|qnli|qqp) echo 8  ;; *) echo 16 ;; esac ;;
        opt-6.7b)
            case "$t" in mrpc|qnli|qqp) echo 4  ;; *) echo 8  ;; esac ;;
        *)
            echo "${BS:-8}" ;;
    esac
}

if [[ -z "${GPU_IDS:-}" ]]; then
    GPU_IDS="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"
fi
read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARRAY[@]}"
TOTAL_SLOTS=$((NUM_GPUS * JOBS_PER_GPU))

if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    echo "ERROR: $MODEL_DIR/config.json not found. Did you download the model?" >&2
    exit 1
fi

mkdir -p outputs logs

# --- Build job list ---------------------------------------------------------
declare -a JOBS

# P1: Uniform-Average residual, all tasks
for seed in $SEEDS; do
    for task in $TASKS; do
        BS=$(bs_for_model_task "$MODEL_TAG" "$task")
        COMMON="--task $task --seed $seed --model_dir $MODEL_DIR"
        COMMON+=" --epochs $EPOCHS --batch_size $BS --max_seqlen $SEQLEN"
        COMMON+=" --lr $LR --use_4bit --bf16 --logging_steps $LOGGING_STEPS"
        outdir="outputs/${MODEL_TAG}/attnres-uniform-avg/${task}-s${seed}"
        JOBS+=("${outdir}|${COMMON} --use_attn_residual --attn_residual_type full --attn_residual_norm none --freeze_attn_query")
    done
done

# P3: Pre-attention Norm on RTE (layernorm, none; rmsnorm is in main run)
for seed in $SEEDS; do
    task=rte
    BS=$(bs_for_model_task "$MODEL_TAG" "$task")
    for norm in layernorm none; do
        COMMON="--task $task --seed $seed --model_dir $MODEL_DIR"
        COMMON+=" --epochs $EPOCHS --batch_size $BS --max_seqlen $SEQLEN"
        COMMON+=" --lr $LR --use_4bit --bf16 --logging_steps $LOGGING_STEPS"
        outdir="outputs/${MODEL_TAG}/attnres-block-bs2-norm-${norm}/${task}-s${seed}"
        JOBS+=("${outdir}|${COMMON} --use_attn_residual --attn_residual_type block --block_size 2 --attn_residual_norm ${norm}")
    done
done

echo "=================================================================="
echo "Ablation launcher"
echo "  model:        $MODEL_DIR  (tag=$MODEL_TAG)"
echo "  tasks:        $TASKS"
echo "  seeds:        $SEEDS"
echo "  epochs:       $EPOCHS   lr: $LR   seq len: $SEQLEN"
echo "  GPUs to use:  $GPU_IDS  (N=$NUM_GPUS)"
echo "  jobs/GPU:     $JOBS_PER_GPU   total concurrent slots: $TOTAL_SLOTS"
echo "  job breakdown:"
echo "    P1 uniform-avg    $(echo $TASKS | wc -w) tasks × $(echo $SEEDS | wc -w) seeds"
echo "    P3 norm (RTE)     {layernorm, none} × $(echo $SEEDS | wc -w) seeds"
echo "  total jobs:   ${#JOBS[@]}"
echo "=================================================================="

# --- Scheduler --------------------------------------------------------------
declare -A PID_TO_GPU
declare -a FREE_GPUS
for ((slot=0; slot<JOBS_PER_GPU; slot++)); do
    FREE_GPUS+=("${GPU_ARRAY[@]}")
done

wait_for_slot () {
    wait -n || true
    for pid in "${!PID_TO_GPU[@]}"; do
        if ! kill -0 "$pid" 2>/dev/null; then
            FREE_GPUS+=("${PID_TO_GPU[$pid]}")
            unset 'PID_TO_GPU[$pid]'
        fi
    done
}

for i in "${!JOBS[@]}"; do
    item="${JOBS[$i]}"
    outdir="${item%%|*}"
    cli="${item#*|}"
    run_name="$(echo "$outdir" | sed 's|^outputs/||; s|/|__|g')"
    log="logs/${run_name}.log"

    if [[ -f "$outdir/final_metrics.json" ]]; then
        echo "[skip $((i+1))/${#JOBS[@]}] $outdir"
        continue
    fi

    while [[ ${#FREE_GPUS[@]} -eq 0 ]]; do
        wait_for_slot
    done
    gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")

    mkdir -p "$outdir"
    echo "[launch $((i+1))/${#JOBS[@]}] GPU $gpu  $outdir  (log: $log)"
    (
        CUDA_VISIBLE_DEVICES=$gpu python src/train_glue.py $cli \
            --output_dir "$outdir" > "$log" 2>&1
        rc=$?
        if [[ $rc -ne 0 ]]; then
            echo "[FAIL GPU $gpu] $outdir (exit $rc, see $log)" >&2
            exit $rc
        fi
    ) &
    PID_TO_GPU[$!]=$gpu
done

wait

echo ""
echo "=================================================================="
echo "[✓] ablations done for $MODEL_TAG."
echo "=================================================================="
