"""Aggregate all outputs into a ranking table.

Walks the nested output tree:
    outputs/<model_tag>/<method>/<task>-s<seed>/final_metrics.json

Writes:
    results/main.csv   machine-readable (one row per completed run)
    results/main.md    human-readable markdown with a headline table
                       and per-(model, task) rankings

Run:
    python src/summarize_results.py
"""

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"
RESULTS_DIR = ROOT / "results"

# outputs/<model_tag>/<method>/<task>-s<seed>/
LEAF_RE = re.compile(r"^(?P<task>[a-z0-9]+)-s(?P<seed>\d+)$")

# Primary metric per task (matches train_glue.py::TASK_PRIMARY_METRIC).
TASK_PRIMARY = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "accuracy",
    "rte":  "accuracy",
    "qnli": "accuracy",
    "qqp":  "accuracy",
    "stsb": "pearson",
    "mnli": "accuracy",
}


def parse_nested_path(run_dir: Path):
    """Parse outputs/<model>/<method>/<task>-s<seed> from a run directory.

    Returns dict or None if the path does not match the expected 3-level layout.
    """
    try:
        rel = run_dir.relative_to(OUT_DIR)
    except ValueError:
        return None
    parts = rel.parts
    if len(parts) != 3:
        return None
    model, method, leaf = parts
    if model.startswith("."):       # e.g. outputs/.smoke/ (smoke test cache)
        return None
    m = LEAF_RE.match(leaf)
    if not m:
        return None
    return {
        "model":  model,
        "method": method,
        "task":   m["task"],
        "seed":   int(m["seed"]),
    }


def collect_one(run_dir: Path) -> Optional[Dict]:
    fm = run_dir / "final_metrics.json"
    if not fm.exists():
        return None
    try:
        final = json.loads(fm.read_text())
    except Exception as e:
        print(f"[warn] cannot parse {fm}: {e}")
        return None

    parsed = parse_nested_path(run_dir)
    if parsed is None:
        return None

    eval_metrics      = final.get("eval",            {}) or {}
    eval_last_metrics = final.get("eval_last_epoch", {}) or {}
    eval_mm_metrics   = final.get("eval_mismatched", {}) or {}
    train_metrics     = final.get("train",           {}) or {}
    best_metrics      = final.get("best",            {}) or {}
    params            = final.get("params",          {}) or {}
    setup             = final.get("setup",           {}) or {}

    task = parsed["task"]
    primary_key = "eval_" + TASK_PRIMARY.get(task, "accuracy")

    # final_metrics.json stores the best epoch according to the primary metric
    # used when the run was launched. If the reporting metric changes later
    # (e.g. MRPC/QQP F1 -> accuracy), recover the reporting best epoch from
    # log_history.json without rerunning the experiment.
    log_path = run_dir / "log_history.json"
    if log_path.exists():
        try:
            history = json.loads(log_path.read_text())
            best_entry = None
            best_val = -float("inf")
            for entry in history:
                val = entry.get(primary_key)
                if val is not None and val > best_val:
                    best_val = val
                    best_entry = entry
            if best_entry is not None:
                eval_metrics = dict(best_entry)
                best_metrics = {
                    "best_epoch": best_entry.get("epoch"),
                    f"best_{primary_key}": best_val,
                }
        except Exception as e:
            print(f"[warn] cannot parse {log_path}: {e}")

    # Prefer the primary metric; fall back to any eval_* that exists.
    def pick(md, key):
        return md.get(key) if md else None

    row = {
        **parsed,
        "run_dir":            str(run_dir.relative_to(ROOT)),
        # Primary metric (best-epoch + last-epoch)
        "primary_metric":     TASK_PRIMARY.get(task, "accuracy"),
        "best_primary":       pick(eval_metrics, primary_key),
        "last_primary":       pick(eval_last_metrics, primary_key),
        # All common metrics (best-epoch)
        "best_accuracy":      pick(eval_metrics, "eval_accuracy"),
        "best_f1":            pick(eval_metrics, "eval_f1"),
        "best_matthews":      pick(eval_metrics, "eval_matthews_correlation"),
        "best_pearson":       pick(eval_metrics, "eval_pearson"),
        "best_spearmanr":     pick(eval_metrics, "eval_spearmanr"),
        "best_eval_loss":     pick(eval_metrics, "eval_loss"),
        "best_epoch":         best_metrics.get("best_epoch"),
        # MNLI mismatched (eval once at end of training)
        "eval_mm_accuracy":   pick(eval_mm_metrics, "eval_mm_accuracy"),
        # Training
        "train_loss":         train_metrics.get("train_loss"),
        "train_runtime_s":    train_metrics.get("train_runtime"),
        # Params
        "trainable_params":   params.get("trainable"),
        "total_params":       params.get("total"),
        "attnres_params":     (params.get("breakdown") or {}).get("attn_residual"),
        "gpu_peak_memory_mb": final.get("gpu_peak_memory_mb"),
        # Setup
        "epochs":             setup.get("epochs"),
        "batch_size":         setup.get("batch_size"),
        "max_seqlen":         setup.get("max_seqlen"),
        "lr":                 setup.get("lr"),
        "use_4bit":           setup.get("use_4bit"),
        "bf16":               setup.get("bf16"),
    }
    return row


def format_val(x, prec=4):
    if x is None:
        return ""
    if isinstance(x, float):
        return f"{x:.{prec}f}"
    return str(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if not OUT_DIR.exists():
        print(f"No {OUT_DIR} directory yet. Run some experiments first.")
        return

    rows = []
    for run_dir in sorted(OUT_DIR.glob("*/*/*")):
        if not run_dir.is_dir():
            continue
        row = collect_one(run_dir)
        if row:
            rows.append(row)

    if not rows:
        print("No completed runs found under outputs/<model>/<method>/<task>-s<seed>/.")
        return

    RESULTS_DIR.mkdir(exist_ok=True, parents=True)

    # Sort primary: (model, task, method, seed) for readability.
    rows_sorted = sorted(rows, key=lambda r: (r["model"], r["task"], r["method"], r["seed"]))

    # ---- Write CSV ----
    headers = [
        "model", "method", "task", "seed",
        "best_primary", "last_primary", "primary_metric",
        "best_accuracy", "best_f1", "best_matthews", "best_pearson",
        "best_spearmanr", "best_eval_loss", "best_epoch",
        "eval_mm_accuracy",
        "train_loss", "train_runtime_s", "gpu_peak_memory_mb",
        "trainable_params", "total_params", "attnres_params",
        "epochs", "batch_size", "max_seqlen", "lr", "use_4bit", "bf16",
        "run_dir",
    ]
    csv_path = RESULTS_DIR / "main.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        w.writeheader()
        for r in rows_sorted:
            w.writerow(r)
    print(f"[write] {csv_path}  ({len(rows_sorted)} rows)")

    # ---- Write markdown (aggregated view: mean ± std across seeds) ----
    md_path = RESULTS_DIR / "main.md"
    lines = []
    lines.append("# Results summary")
    lines.append("")
    lines.append(f"Total runs: {len(rows_sorted)}")
    lines.append("")

    # Group by (model, method, task) and average primary across seeds.
    agg = defaultdict(list)
    for r in rows_sorted:
        if r["best_primary"] is None:
            continue
        key = (r["model"], r["method"], r["task"])
        agg[key].append(r["best_primary"])

    def mean_std(vals):
        if not vals:
            return (None, None, 0)
        if len(vals) == 1:
            return (vals[0], None, 1)
        n = len(vals)
        m = sum(vals) / n
        var = sum((x - m) ** 2 for x in vals) / (n - 1)
        return (m, var ** 0.5, n)

    # ---- Headline table: by (model, task) with all methods as columns ----
    # Collect the union of methods.
    all_methods = sorted({r["method"] for r in rows_sorted})
    # Put canonical order first, then anything else.
    method_order = [m for m in
                    ["qst", "attnres-uniform-avg", "attnres-block-bs2",
                     "attnres-block-bs4", "attnres-full"]
                    if m in all_methods]
    method_order += [m for m in all_methods if m not in method_order]

    models = sorted({r["model"] for r in rows_sorted})
    tasks_seen = sorted({r["task"] for r in rows_sorted})

    for model in models:
        lines.append(f"## {model}")
        lines.append("")
        # Header
        header = ["Task", "Primary"]
        for mth in method_order:
            header.append(mth)
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for task in tasks_seen:
            primary_name = TASK_PRIMARY.get(task, "accuracy")
            row_cells = [task, primary_name]
            for mth in method_order:
                key = (model, mth, task)
                vals = agg.get(key, [])
                m, s, n = mean_std(vals)
                if m is None:
                    cell = "—"
                elif s is None:
                    cell = f"{m*100:.2f}"
                else:
                    cell = f"{m*100:.2f} ± {s*100:.2f} (n={n})"
                row_cells.append(cell)
            lines.append("| " + " | ".join(row_cells) + " |")
        lines.append("")

    # ---- Per-(model, task) best-method ranking (for quick eyeballing) ----
    lines.append("## Ranking (best method per (model, task), by mean primary metric)")
    lines.append("")
    for model in models:
        lines.append(f"### {model}")
        lines.append("")
        for task in tasks_seen:
            scored = []
            for mth in method_order:
                vals = agg.get((model, mth, task), [])
                m, s, n = mean_std(vals)
                if m is not None:
                    scored.append((m, s, n, mth))
            if not scored:
                continue
            scored.sort(reverse=True)
            primary_name = TASK_PRIMARY.get(task, "accuracy")
            lines.append(f"- **{task}** ({primary_name})")
            for rank, (m, s, n, mth) in enumerate(scored):
                marker = " 🏆" if rank == 0 else ""
                sstr = f" ± {s*100:.2f}" if s is not None else ""
                lines.append(f"  - {mth}: {m*100:.2f}{sstr}  (n={n}){marker}")
        lines.append("")

    md_path.write_text("\n".join(lines) + "\n")
    print(f"[write] {md_path}")

    # ---- Compact stdout view ----
    print()
    print("Summary by (model, task, method) — mean primary metric:")
    for model in models:
        print(f"\n[{model}]")
        for task in tasks_seen:
            scored = []
            for mth in method_order:
                vals = agg.get((model, mth, task), [])
                m, s, n = mean_std(vals)
                if m is not None:
                    scored.append((m, s, n, mth))
            if not scored:
                continue
            primary_name = TASK_PRIMARY.get(task, "accuracy")
            scored.sort(reverse=True)
            top_m, top_s, top_n, top_mth = scored[0]
            sstr = f"±{top_s*100:.2f}" if top_s is not None else ""
            print(f"  {task:6s} ({primary_name:24s}): best={top_mth:28s} "
                  f"{top_m*100:.2f}{sstr} (n={top_n})")


if __name__ == "__main__":
    main()
