# AttnRes-QST

AttnRes-QST adds attention-based cross-layer residual aggregation to the side network of Quantized Side Tuning (QST). The frozen OPT backbone and the original QST gate are preserved; only the side-state carry term is replaced with either full-history or block-window attention over previous side-network hidden states.

## Repository Layout

- `src/attn_residual.py`: full and block attention-residual modules.
- `src/qst_opt_attnres.py`: QST OPT subclasses that inject AttnRes into the side network.
- `src/train_glue.py`: GLUE training entry point for QST and AttnRes-QST.
- `src/summarize_results.py`: aggregates completed runs under `outputs/`.
- `src/count_trainable_params.py`: reproduces trainable-parameter accounting.
- `scripts/`: data/model download, smoke tests, and experiment launchers.
- `third_party/qst/`: minimal patched QST reference implementation required by the training code.

## Setup

For GPU experiments, create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

For CUDA servers, the helper script installs a CUDA-compatible PyTorch stack and runs the AttnRes unit tests:

```bash
bash scripts/setup_gpu_env.sh
```

## Data and Models

Download GLUE tasks into self-contained local directories:

```bash
bash scripts/download_data.sh
```

Download OPT backbones:

```bash
bash scripts/download_models.sh
```

By default, models are saved under `.models/` and datasets under `.datasets/`. These directories are intentionally ignored by Git.

## Quick Check

Run unit tests:

```bash
python src/test_attn_residual.py
```

Run a tiny GPU smoke test:

```bash
bash scripts/smoke_gpu.sh
```

## Experiments

Run a compact main experiment on SST-2, MRPC, and RTE with one seed:

```bash
bash scripts/run_main.sh
```

Run the full GLUE schedule for selected OPT backbones:

```bash
MODELS="opt-1.3b opt-2.7b opt-6.7b" bash scripts/run_all.sh
```

Aggregate results:

```bash
python src/summarize_results.py
```

## Third-Party Code

The QST code under `third_party/qst/` is based on the official QST repository and includes local compatibility patches documented in `third_party/qst/PATCHES.md`. Please keep upstream licenses and citations when redistributing.

## Acknowledgements

This project builds on two important open-source research efforts:

- Quantized Side Tuning (QST): https://github.com/youarespecialtome/qst
- Attention Residuals: https://github.com/MoonshotAI/Attention-Residuals

We thank the authors of both projects for releasing their code and ideas. AttnRes-QST reuses the patched QST implementation as a third-party dependency and adapts the Attention Residuals idea to QST's side-network hidden-state carry.
