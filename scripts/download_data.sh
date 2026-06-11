#!/usr/bin/env bash
# scripts/download_data.sh
# Download all 8 GLUE tasks via huggingface datasets, save to .datasets/<task>/
# in self-contained save_to_disk format.
#
# Idempotent: skips any task whose directory already has dataset_dict.json.
#
# Total size: ~155 MB on disk after save_to_disk.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Route HF hub through the China-friendly mirror by default; user can override.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "[data] HF endpoint: $HF_ENDPOINT"
echo "[data] target dir:  $ROOT/.datasets/"
mkdir -p .datasets

python - <<'EOF'
import os
from pathlib import Path
from datasets import load_dataset

TASKS = ["cola", "sst2", "mrpc", "rte", "qnli", "qqp", "stsb", "mnli"]

root = Path(os.environ.get("PWD", ".")) / ".datasets"
for task in TASKS:
    out = root / task
    if (out / "dataset_dict.json").exists():
        print(f"[skip] {task} (already present at {out})")
        continue
    print(f"[load] glue/{task} ...")
    ds = load_dataset("glue", task)
    ds.save_to_disk(str(out))
    print(f"[save] {task} -> {out}")
    for k, v in ds.items():
        print(f"         {k:30s} {len(v):>7,d} examples")
print("\n[done] all GLUE tasks saved under .datasets/")
EOF
