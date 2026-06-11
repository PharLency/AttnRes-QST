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

## Main Results

Scores are validation results on GLUE, multiplied by 100 and reported as
mean (standard deviation) over seeds. `Avg.` is the unweighted average over the
eight displayed task metrics. Memory is reported only for QST-family rows and
denotes the mean logged peak GPU allocation under the task-specific batch
schedule; it should not be compared across backbone sizes as a controlled
systems benchmark.

| Backbone | Method | Param. (%) | Memory (GB) | RTE | MRPC | STS-B | CoLA | SST-2 | QNLI | QQP | MNLI | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OPT-1.3B | QLoRA | 4.356 | - | 82.55 (1.10) | 87.01 (0.85) | 90.38 (0.29) | 66.16 (1.61) | 94.82 (0.48) | 92.56 (0.11) | 90.71 (1.71) | 87.61 (0.24) | 86.48 |
| OPT-1.3B | LST | 2.508 | - | 81.24 (2.64) | 81.39 (1.27) | 85.49 (0.59) | 54.87 (0.89) | 94.53 (0.24) | 86.19 (0.42) | 89.75 (0.09) | 84.42 (0.15) | 82.24 |
| OPT-1.3B | LoRA | 2.360 | - | 82.70 (1.90) | 83.40 (0.90) | 89.30 (0.20) | 62.50 (1.70) | 93.70 (0.70) | 81.40 (9.30) | 86.90 (0.30) | 81.20 (0.10) | 82.60 |
| OPT-1.3B | Adapter | 0.480 | - | 82.20 (0.80) | 82.70 (1.40) | 89.70 (1.60) | 60.60 (3.00) | 93.80 (0.20) | 83.60 (0.10) | 86.30 (0.40) | 80.50 (0.10) | 82.40 |
| OPT-1.3B | QST | 0.451 | 11.1 | 75.45 (2.25) | 82.52 (1.02) | 87.52 (0.02) | 56.92 (0.72) | 94.69 (0.18) | 88.73 (0.25) | 89.64 (0.09) | 84.26 (0.27) | 82.47 |
| OPT-1.3B | Uniform | 0.451 | 11.4 | 75.33 (0.55) | 82.92 (0.62) | 87.89 (0.22) | 56.82 (1.02) | 94.53 (0.13) | 88.57 (0.17) | 89.87 (0.13) | 83.93 (0.22) | 82.48 |
| OPT-1.3B | Full | 0.452 | 12.9 | 75.93 (0.21) | 82.52 (0.37) | 87.94 (0.09) | 56.13 (1.37) | 94.46 (0.18) | 88.52 (0.32) | 89.90 (0.05) | 84.04 (0.06) | 82.43 |
| OPT-1.3B | Block-2 | 0.452 | 11.4 | 75.57 (1.46) | 82.92 (0.51) | 87.44 (0.22) | 57.26 (0.72) | 94.57 (0.40) | 88.69 (0.30) | 89.80 (0.16) | 84.14 (0.15) | 82.55 |
| OPT-1.3B | Block-4 | 0.452 | 11.7 | 75.81 (0.00) | 83.09 (1.30) | 87.77 (0.25) | 55.95 (0.49) | 94.65 (0.40) | 88.59 (0.29) | 89.84 (0.07) | 84.19 (0.16) | 82.49 |
| OPT-2.7B | QLoRA | 3.495 | - | 85.80 (1.37) | 88.24 (0.25) | 91.00 (0.10) | 65.95 (1.90) | 96.10 (0.11) | 87.55 (0.21) | 89.41 (4.12) | 88.50 (0.30) | 86.57 |
| OPT-2.7B | LST | 2.494 | - | 77.53 (0.55) | 79.66 (0.65) | 87.07 (0.56) | 59.70 (2.70) | 95.64 (1.17) | 88.94 (0.30) | 89.69 (0.07) | 85.48 (0.30) | 82.96 |
| OPT-2.7B | LoRA | 1.900 | - | 84.70 (1.40) | 84.60 (0.80) | 90.90 (0.10) | 64.50 (2.40) | 95.30 (0.60) | 83.00 (7.40) | 90.70 (0.10) | 82.60 (0.20) | 84.50 |
| OPT-2.7B | Adapter | 0.370 | - | 84.40 (0.70) | 83.70 (1.40) | 91.50 (1.90) | 63.40 (3.80) | 95.40 (0.30) | 83.60 (0.20) | 90.20 (0.30) | 81.10 (0.10) | 84.20 |
| OPT-2.7B | QST | 0.434 | 10.2 | 77.38 (0.21) | 81.86 (0.98) | 87.36 (0.17) | 59.18 (0.24) | 95.49 (0.54) | 89.23 (0.13) | 90.00 (0.15) | 85.43 (0.11) | 83.24 |
| OPT-2.7B | Uniform | 0.434 | 10.6 | 81.11 (0.75) | 81.86 (0.00) | 88.35 (0.35) | 60.72 (1.32) | 95.30 (0.20) | 89.35 (0.26) | 90.23 (0.06) | 85.67 (0.08) | 84.07 |
| OPT-2.7B | Full | 0.434 | 12.7 | 79.42 (0.36) | 82.03 (1.02) | 88.35 (0.19) | 59.73 (1.04) | 95.45 (0.37) | 89.33 (0.39) | 90.08 (0.04) | 85.62 (0.38) | 83.75 |
| OPT-2.7B | Block-2 | 0.434 | 10.5 | 79.90 (0.55) | 81.86 (0.65) | 87.96 (0.14) | 59.65 (0.51) | 95.60 (0.07) | 89.46 (0.23) | 89.94 (0.07) | 85.57 (0.01) | 83.74 |
| OPT-2.7B | Block-4 | 0.434 | 10.8 | 79.18 (0.75) | 82.19 (0.37) | 88.06 (0.09) | 60.37 (1.21) | 95.64 (0.11) | 89.44 (0.17) | 90.09 (0.07) | 85.75 (0.18) | 83.84 |
| OPT-6.7B | QLoRA | 2.254 | - | 86.76 (0.75) | 88.48 (1.07) | 91.52 (0.08) | 64.09 (3.00) | 96.21 (0.01) | 90.10 (0.58) | 88.67 (3.69) | 89.49 (0.22) | 86.91 |
| OPT-6.7B | QST | 0.426 | 8.6 | 79.66 (2.66) | 83.99 (0.99) | 87.29 (0.28) | 59.27 (0.27) | 96.18 (0.24) | 90.17 (0.22) | 90.39 (0.09) | 86.51 (0.16) | 84.18 |
| OPT-6.7B | Uniform | 0.426 | 8.9 | 80.51 (0.63) | 83.91 (0.28) | 88.66 (0.28) | 61.62 (1.12) | 95.87 (0.11) | 89.91 (0.17) | 90.39 (0.06) | 86.59 (0.12) | 84.68 |
| OPT-6.7B | Full | 0.427 | 10.6 | 80.51 (0.36) | 83.66 (0.99) | 88.93 (0.13) | 61.04 (0.15) | 96.14 (0.13) | 90.02 (0.18) | 90.21 (0.10) | 86.57 (0.12) | 84.63 |
| OPT-6.7B | Block-2 | 0.427 | 8.9 | 79.18 (1.10) | 84.56 (1.07) | 87.96 (0.06) | 59.04 (0.54) | 96.18 (0.33) | 90.19 (0.17) | 90.32 (0.11) | 86.69 (0.08) | 84.27 |
| OPT-6.7B | Block-4 | 0.427 | 9.1 | 80.63 (0.75) | 85.54 (0.85) | 88.32 (0.23) | 62.11 (1.50) | 96.14 (0.13) | 90.24 (0.43) | 90.27 (0.04) | 86.73 (0.12) | 85.00 |

## Third-Party Code

The QST code under `third_party/qst/` is based on the official QST repository and includes local compatibility patches documented in `third_party/qst/PATCHES.md`. Please keep upstream licenses and citations when redistributing.

## Acknowledgements

This project builds on two important open-source research efforts:

- Quantized Side Tuning (QST): https://github.com/youarespecialtome/qst
- Attention Residuals: https://github.com/MoonshotAI/Attention-Residuals

We thank the authors of both projects for releasing their code and ideas. AttnRes-QST reuses the patched QST implementation as a third-party dependency and adapts the Attention Residuals idea to QST's side-network hidden-state carry.
