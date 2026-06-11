#!/usr/bin/env bash
# scripts/run_parallel.sh
#
# Main experiments for one model: trains QST baseline + all AttnRes variants
# on all 8 GLUE tasks, over multiple seeds, distributing jobs across visible
# GPUs in parallel.
#
# Follows the hyperparameter matrix from the QST paper (Table 8):
#
#     Model      | bs (RTE/STS-B/CoLA/SST-2/MNLI) | bs (MRPC/QNLI/QQP) | Epochs
#     OPT-1.3B   |              32                |         8          |   20
#     OPT-2.7B   |              16                |         8          |   15
#     OPT-6.7B   |               8                |         4          |   10
#     (common: LR=2e-4, r=16, peft_hidden_size=16, AdamW, warmup=0.06, linear)
#
# Output layout (nested, one leaf per run):
#     outputs/<model_tag>/<method>/<task>-s<seed>/
#
#     methods (this script launches exactly four per (task, seed)):
#         qst
#         attnres-block-bs2
#         attnres-block-bs4
#         attnres-full
#
# Usage:
#     # default: all 8 tasks × 3 seeds × 4 methods, OPT-1.3B
#     MODEL_DIR=.models/opt-1.3b bash scripts/run_parallel.sh
#
#     # fewer seeds
#     SEEDS="42" MODEL_DIR=.models/opt-1.3b bash scripts/run_parallel.sh
#
#     # specific tasks
#     TASKS="sst2 rte" MODEL_DIR=.models/opt-1.3b bash scripts/run_parallel.sh
#
#     # multi-job per GPU (H200 with small models has spare compute)
#     JOBS_PER_GPU=2 MODEL_DIR=.models/opt-1.3b bash scripts/run_parallel.sh
#
#     # restrict which GPUs to use
#     GPU_IDS="0 1 2 3" MODEL_DIR=.models/opt-1.3b bash scripts/run_parallel.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# --- Tunables ---------------------------------------------------------------
SEEDS="${SEEDS:-42 123 2024}"
TASKS="${TASKS:-cola sst2 mrpc rte qnli qqp stsb mnli}"
MODEL_DIR="${MODEL_DIR:-.models/opt-1.3b}"
MODEL_TAG="$(basename "$MODEL_DIR")"
LR="${LR:-2e-4}"
SEQLEN="${SEQLEN:-512}"
LOGGING_STEPS="${LOGGING_STEPS:-50}"
JOBS_PER_GPU="${JOBS_PER_GPU:-1}"

# Per-model epochs (Table 8). Override with EPOCHS env var.
epochs_for_model () {
    case "$1" in
        opt-1.3b) echo 20 ;;
        opt-2.7b) echo 15 ;;
        opt-6.7b) echo 10 ;;
        *)        echo "${EPOCHS:-20}" ;;
    esac
}
EPOCHS="${EPOCHS:-$(epochs_for_model "$MODEL_TAG")}"

# Per-(model, task) batch size (Table 8).
# Large-bs group: RTE, STS-B, CoLA, SST-2, MNLI.
# Small-bs group: MRPC, QNLI, QQP.
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

# GPU pool
if [[ -z "${GPU_IDS:-}" ]]; then
    GPU_IDS="$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')"
fi
read -ra GPU_ARRAY <<< "$GPU_IDS"
NUM_GPUS="${#GPU_ARRAY[@]}"
TOTAL_SLOTS=$((NUM_GPUS * JOBS_PER_GPU))

# Pre-flight: does the model exist?
if [[ ! -f "$MODEL_DIR/config.json" ]]; then
    echo "ERROR: $MODEL_DIR/config.json not found. Did you download the model?" >&2
    echo "  bash scripts/download_models.sh" >&2
    exit 1
fi

mkdir -p outputs logs results

# --- Build job list ---------------------------------------------------------
declare -a JOBS   # entries: "<run_outdir>|<cli args>"

for seed in $SEEDS; do
    for task in $TASKS; do
        BS=$(bs_for_model_task "$MODEL_TAG" "$task")
        COMMON="--task $task --seed $seed --model_dir $MODEL_DIR"
        COMMON+=" --epochs $EPOCHS --batch_size $BS --max_seqlen $SEQLEN"
        COMMON+=" --lr $LR --use_4bit --bf16 --logging_steps $LOGGING_STEPS"

        base="outputs/${MODEL_TAG}"
        JOBS+=("${base}/qst/${task}-s${seed}|${COMMON}")
        JOBS+=("${base}/attnres-block-bs2/${task}-s${seed}|${COMMON} --use_attn_residual --attn_residual_type block --block_size 2")
        JOBS+=("${base}/attnres-block-bs4/${task}-s${seed}|${COMMON} --use_attn_residual --attn_residual_type block --block_size 4")
        JOBS+=("${base}/attnres-full/${task}-s${seed}|${COMMON} --use_attn_residual --attn_residual_type full")
    done
done

# --- Header -----------------------------------------------------------------
echo "=================================================================="
echo "Main experiment launcher"
echo "  model:        $MODEL_DIR  (tag=$MODEL_TAG)"
echo "  tasks:        $TASKS"
echo "  seeds:        $SEEDS"
echo "  epochs:       $EPOCHS   lr: $LR   seq len: $SEQLEN"
echo "  per-task BS:  varies by (model, task) (Table 8)"
echo "  GPUs to use:  $GPU_IDS  (N=$NUM_GPUS)"
echo "  jobs/GPU:     $JOBS_PER_GPU   total concurrent slots: $TOTAL_SLOTS"
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
    # run_name for log = path slash -> dash
    run_name="$(echo "$outdir" | sed 's|^outputs/||; s|/|__|g')"
    log="logs/${run_name}.log"

    if [[ -f "$outdir/final_metrics.json" ]]; then
        echo "[skip $((i+1))/${#JOBS[@]}] $outdir (already has final_metrics.json)"
        continue
    fi

    while [[ ${#FREE_GPUS[@]} -eq 0 ]]; do
        wait_for_slot
    done
    gpu="${FREE_GPUS[0]}"
    FREE_GPUS=("${FREE_GPUS[@]:1}")

    mkdir -p "$outdir" "$(dirname "$log")"
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
echo "[✓] main runs complete for $MODEL_TAG. Summarize with:"
echo "    python src/summarize_results.py"
echo "=================================================================="
