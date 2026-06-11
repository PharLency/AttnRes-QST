#!/usr/bin/env bash
# Compact main run: a small task subset for quick reproduction.

set -euo pipefail

export MODEL_DIR="${MODEL_DIR:-.models/opt-1.3b}"
export SEEDS="${SEEDS:-42}"
export TASKS="${TASKS:-sst2 mrpc rte}"
export JOBS_PER_GPU="${JOBS_PER_GPU:-1}"

bash "$(cd "$(dirname "$0")" && pwd)/run_parallel.sh"

