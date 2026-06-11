#!/usr/bin/env bash
# scripts/setup_gpu_env.sh
# Manual-style setup: each step prints what it's about to do, then runs one
# pip/conda command. You can also copy-paste any single block into a shell.
#
# Policy:
#   - PyTorch cu124 → official source (download.pytorch.org)
#   - everything else → USTC PyPI mirror
#
# Override with env vars if needed:
#   ENV_NAME    default attnres-qst
#   PY_VER      default 3.10
#   PIP_INDEX   default https://pypi.mirrors.ustc.edu.cn/simple
#   TORCH_INDEX default https://download.pytorch.org/whl/cu124

set -euo pipefail

ENV_NAME="${ENV_NAME:-attnres-qst}"
PY_VER="${PY_VER:-3.10}"
PIP_INDEX="${PIP_INDEX:-https://pypi.mirrors.ustc.edu.cn/simple}"
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"

PIP_HOST="$(echo "$PIP_INDEX"   | awk -F/ '{print $3}')"
TORCH_HOST="$(echo "$TORCH_INDEX" | awk -F/ '{print $3}')"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"


# ---------- Step 1: create / reuse conda env ----------
echo ""
echo "========== Step 1: conda env '$ENV_NAME' (python $PY_VER) =========="
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "(env already exists, skipping create)"
else
    conda create -y -n "$ENV_NAME" "python=$PY_VER"
fi

ENV_PREFIX="$(conda env list | awk -v env="$ENV_NAME" '$1==env {print $NF}')"
PY="$ENV_PREFIX/bin/python"
echo "python: $PY"


# ---------- Step 2: upgrade pip (via USTC) ----------
echo ""
echo "========== Step 2: upgrade pip (via $PIP_HOST) =========="
"$PY" -m pip install -U pip \
    --index-url "$PIP_INDEX" \
    --trusted-host "$PIP_HOST"


# ---------- Step 3: install torch cu124 (via official pytorch.org) ----------
echo ""
echo "========== Step 3: install torch cu124 (via $TORCH_HOST) =========="
"$PY" -m pip install torch torchvision torchaudio \
    --index-url "$TORCH_INDEX" \
    --trusted-host "$TORCH_HOST"


# ---------- Step 4: install project deps (via USTC) ----------
echo ""
echo "========== Step 4: install project deps (via $PIP_HOST) =========="
"$PY" -m pip install \
    "transformers>=4.46,<4.58" \
    "datasets>=4.0" \
    "evaluate>=0.4" \
    "peft>=0.10" \
    "accelerate>=0.33" \
    scikit-learn scipy matplotlib pandas tqdm tensorboard sentencepiece numpy \
    --index-url "$PIP_INDEX" \
    --trusted-host "$PIP_HOST"


# ---------- Step 5: install bitsandbytes (via USTC) ----------
echo ""
echo "========== Step 5: install bitsandbytes (via $PIP_HOST) =========="
"$PY" -m pip install "bitsandbytes>=0.43.3" \
    --index-url "$PIP_INDEX" \
    --trusted-host "$PIP_HOST"


# ---------- Step 6: CUDA + bnb sanity check ----------
echo ""
echo "========== Step 6: sanity check =========="
"$PY" - <<'EOF'
import sys
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    print("ERROR: torch can't see CUDA. Aborting.", file=sys.stderr)
    sys.exit(1)
print("cuda device:", torch.cuda.get_device_name(0))
print("cuda count: ", torch.cuda.device_count())
import bitsandbytes as bnb
print("bnb:", bnb.__version__)
EOF


# ---------- Step 7: AttnRes unit tests ----------
echo ""
echo "========== Step 7: AttnRes unit tests =========="
cd "$ROOT/src"
"$PY" test_attn_residual.py
cd "$ROOT"


echo ""
echo "[✓] env setup complete. Activate with:"
echo "    conda activate $ENV_NAME"
echo ""
echo "    Next: bash scripts/smoke_gpu.sh"
