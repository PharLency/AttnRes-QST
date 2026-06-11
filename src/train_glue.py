"""Train QST on a GLUE task. Supports CPU smoke-test and GPU 4-bit mode.

Entry used during local Mac CPU debugging; on a GPU server add --use_4bit --bf16.

Usage (CPU smoke):
    python src/train_glue.py --task sst2 --max_train_samples 32 --max_eval_samples 16 \\
        --epochs 1 --batch_size 4 --max_seqlen 128

Usage (GPU, real run):
    python src/train_glue.py --task sst2 --use_4bit --bf16 --batch_size 16
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import load_from_disk

PROJECT_ROOT = Path(__file__).resolve().parent.parent
QST_DIR = PROJECT_ROOT / "third_party" / "qst"
sys.path.insert(0, str(QST_DIR))

from QSTConfig import QSTConfig  # noqa: E402
from modeling_opt_qst import QSTOPTForSequenceClassification  # noqa: E402
from qst_opt_attnres import (  # noqa: E402
    AttnResConfig,
    QSTOPTForSequenceClassificationAttnRes,
)


# --------------------------------------------------------------------------- #
#  Task metadata (all 8 GLUE tasks)
# --------------------------------------------------------------------------- #
TASK_TO_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "rte":  ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp":  ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
}

# num_labels == 1 triggers regression mode in the classification head.
TASK_NUM_LABELS = {
    "cola": 2, "sst2": 2, "mrpc": 2, "rte": 2,
    "qnli": 2, "qqp": 2, "mnli": 3, "stsb": 1,
}

# Primary metric used for best-epoch selection / headline number.
TASK_PRIMARY_METRIC = {
    "cola": "matthews_correlation",
    "sst2": "accuracy",
    "mrpc": "accuracy",
    "rte":  "accuracy",
    "qnli": "accuracy",
    "qqp":  "accuracy",
    "stsb": "pearson",
    "mnli": "accuracy",   # matched accuracy
}


def build_compute_metrics(task: str):
    """Return a compute_metrics callable appropriate for `task`.

    Handles:
      * classification (acc + optional F1 + optional Matthews correlation)
      * regression for STS-B (Pearson / Spearman on raw scalar predictions)
    """
    import numpy as _np
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef

    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        if isinstance(preds, tuple):
            preds = preds[0]

        if task == "stsb":
            # Regression: pooled_logits is shape [B, 1]; take the scalar.
            preds_reg = preds[:, 0] if preds.ndim > 1 else preds
            # pearsonr / spearmanr import lazily so scipy stays optional elsewhere.
            from scipy.stats import pearsonr, spearmanr
            p, _ = pearsonr(preds_reg, labels)
            s, _ = spearmanr(preds_reg, labels)
            return {"pearson": float(p), "spearmanr": float(s)}

        # Classification
        pred_labels = _np.argmax(preds, axis=-1)
        out = {"accuracy": float(accuracy_score(labels, pred_labels))}
        if task in ("mrpc", "qqp"):
            out["f1"] = float(f1_score(labels, pred_labels))
        if task == "cola":
            out["matthews_correlation"] = float(matthews_corrcoef(labels, pred_labels))
        return out

    return compute_metrics


def count_trainable_params(model):
    trainable = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return trainable, total


def param_breakdown(model):
    """Group trainable params by module name pattern.

    Useful for the report: "AttnRes adds only X parameters on top of QST".
    Returns OrderedDict for stable output ordering.
    """
    from collections import OrderedDict
    groups = OrderedDict([
        ("qst_layers",             0),
        ("downsample",             0),
        ("z_gate",                 0),
        ("final_layer_norm_qst",   0),
        ("upsample",               0),
        ("score_z",                0),
        ("score",                  0),
        ("attn_residual",          0),  # stays 0 for baseline QST
        ("other_trainable",        0),
        ("backbone_LEAK",          0),  # should always be 0; nonzero = bug
    ])
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if ".backbone." in name:
            groups["backbone_LEAK"] += p.numel()
        elif "attn_residual_modules" in name:
            groups["attn_residual"] += p.numel()
        elif ".qst_layers." in name:
            groups["qst_layers"] += p.numel()
        elif ".downsample." in name:
            groups["downsample"] += p.numel()
        elif name.endswith(".z") or ".z." in name:
            groups["z_gate"] += p.numel()
        elif "final_layer_norm_qst" in name:
            groups["final_layer_norm_qst"] += p.numel()
        elif "upsample" in name:
            groups["upsample"] += p.numel()
        elif name.endswith("score_z") or "score_z" in name:
            groups["score_z"] += p.numel()
        elif "score" in name:
            groups["score"] += p.numel()
        else:
            groups["other_trainable"] += p.numel()
    return groups


def extract_best(log_history, task):
    """Walk trainer.state.log_history for the best validation metric.

    Returns dict with the primary best value plus the full eval-entry metrics
    at the best epoch. Primary metric per task follows TASK_PRIMARY_METRIC.
    """
    primary = "eval_" + TASK_PRIMARY_METRIC[task]
    best = {}
    best_val = -float("inf")
    for entry in log_history:
        val = entry.get(primary)
        if val is None:
            continue
        if val > best_val:
            best_val = val
            best = {
                "best_epoch":              entry.get("epoch"),
                f"best_{primary}":         val,
                "best_eval_accuracy":      entry.get("eval_accuracy"),
                "best_eval_f1":            entry.get("eval_f1"),
                "best_eval_loss":          entry.get("eval_loss"),
                "best_eval_matthews":      entry.get("eval_matthews_correlation"),
                "best_eval_pearson":       entry.get("eval_pearson"),
                "best_eval_spearmanr":     entry.get("eval_spearmanr"),
            }
    return {k: v for k, v in best.items() if v is not None}


def find_best_epoch_eval_entry(log_history, task):
    """Return the full eval-log entry (or None) for the best-scoring epoch.

    Used to populate `final_metrics.eval` with the best-epoch metrics, which
    matches the original QST paper's load_best_model_at_end=True convention.
    """
    primary = "eval_" + TASK_PRIMARY_METRIC[task]
    best_entry = None
    best_val = -float("inf")
    for entry in log_history:
        val = entry.get(primary)
        if val is None:
            continue
        if val > best_val:
            best_val = val
            best_entry = dict(entry)
    return best_entry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=list(TASK_TO_KEYS), required=True)
    parser.add_argument("--model_dir", default=str(PROJECT_ROOT / ".models" / "opt-1.3b"))
    parser.add_argument("--data_dir", default=str(PROJECT_ROOT / ".datasets"))
    parser.add_argument("--output_dir", default=None,
                        help="Default: outputs/<task>-<timestamp>")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seqlen", type=int, default=128)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)

    # QST hyperparams
    parser.add_argument("--qst_r", type=int, default=16, help="hidden reduction ratio")
    parser.add_argument("--qst_alpha_r", type=int, default=16)
    parser.add_argument("--qst_dropout", type=float, default=0.1)

    # AttnRes hyperparams
    parser.add_argument("--use_attn_residual", action="store_true",
                        help="Enable AttnRes-QST (default: plain QST baseline).")
    parser.add_argument("--attn_residual_type", choices=["full", "block"], default="block")
    parser.add_argument("--block_size", type=int, default=2)
    parser.add_argument("--attn_residual_norm", choices=["rmsnorm", "layernorm", "none"],
                        default="rmsnorm")
    parser.add_argument("--freeze_attn_query", action="store_true",
                        help="Freeze AttnRes pseudo-query at 0 (alpha becomes "
                             "uniform, AttnRes degenerates to a plain mean). "
                             "Ablation: proves gains come from learnable attention.")

    # Runtime modes
    parser.add_argument("--use_4bit", action="store_true",
                        help="Load LLM in 4bit via bitsandbytes (GPU only).")
    parser.add_argument("--bf16", action="store_true",
                        help="Use bf16 for training (GPU only).")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--device", default=None,
                        help="cpu / cuda / mps. Default: auto.")
    parser.add_argument("--logging_steps", type=int, default=10)
    args = parser.parse_args()

    # Resolve model_dir / data_dir to absolute paths when they LOOK like local
    # paths (start with '.' or '/'). We do this unconditionally (not gated on
    # os.path.isdir) so that a wrong path produces a clear FileNotFoundError
    # instead of HF's confusing "Repo id must use alphanumeric chars" error.
    if args.model_dir.startswith((".", "/")):
        args.model_dir = os.path.abspath(args.model_dir)
        if not os.path.isdir(args.model_dir):
            raise FileNotFoundError(
                f"--model_dir points to a local path '{args.model_dir}' "
                f"that does not exist. Check that you've downloaded the model "
                f"there, or pass a HF repo id like 'facebook/opt-1.3b'."
            )
    if args.data_dir.startswith((".", "/")):
        args.data_dir = os.path.abspath(args.data_dir)
        if not os.path.isdir(args.data_dir):
            raise FileNotFoundError(
                f"--data_dir points to a local path '{args.data_dir}' "
                f"that does not exist. Check that you've run the self-containment "
                f"step that saves GLUE tasks into this directory."
            )

    set_seed(args.seed)

    # ---------- device ----------
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[setup] device={args.device}  use_4bit={args.use_4bit}  bf16={args.bf16}")

    # ---------- output dir ----------
    if args.output_dir is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        args.output_dir = str(PROJECT_ROOT / "outputs" / f"{args.task}-{ts}")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "run_args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # ---------- data ----------
    data_path = os.path.join(args.data_dir, args.task)
    print(f"[data] loading from {data_path}")
    raw = load_from_disk(data_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    s1_key, s2_key = TASK_TO_KEYS[args.task]

    def preprocess(ex):
        if s2_key is None:
            return tokenizer(ex[s1_key], truncation=True, max_length=args.max_seqlen)
        return tokenizer(ex[s1_key], ex[s2_key], truncation=True, max_length=args.max_seqlen)

    tokenized = raw.map(preprocess, batched=True, remove_columns=[c for c in raw["train"].column_names if c != "label"])

    train_ds = tokenized["train"]
    # MNLI has no plain "validation" split: it publishes validation_matched and
    # validation_mismatched. We use matched as the primary eval (best-epoch
    # selection) and evaluate on mismatched once at the end of training.
    val_split_primary = "validation_matched" if args.task == "mnli" else "validation"
    eval_ds = tokenized[val_split_primary]
    eval_ds_mm = tokenized.get("validation_mismatched") if args.task == "mnli" else None
    if args.max_train_samples is not None:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples is not None:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))
        if eval_ds_mm is not None:
            eval_ds_mm = eval_ds_mm.select(range(min(args.max_eval_samples, len(eval_ds_mm))))
    print(f"[data] train={len(train_ds)}  eval={len(eval_ds)}"
          + (f"  eval_mm={len(eval_ds_mm)}" if eval_ds_mm is not None else ""))

    # ---------- model ----------
    num_labels = TASK_NUM_LABELS[args.task]
    config = AutoConfig.from_pretrained(args.model_dir, num_labels=num_labels)
    if args.task == "stsb":
        config.problem_type = "regression"

    load_kwargs = dict(num_labels=num_labels)
    # Force the plain/eager attention implementation. The default SDPA path
    # can hit cuDNN Frontend crashes ("No execution plans support the graph")
    # on some CUDA/cuDNN/GPU combos, particularly with 4-bit quantization.
    # Eager is slower but numerically equivalent and universally supported.
    load_kwargs["attn_implementation"] = "eager"
    # STS-B is a regression task; num_labels=1 triggers MSE loss inside
    # QSTOPTForSequenceClassification only if problem_type is not already set.
    # Pin it explicitly so HF's heuristics can't flip it to a classification.
    if args.task == "stsb":
        load_kwargs["problem_type"] = "regression"
    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        compute_dtype = torch.bfloat16 if args.bf16 else (torch.float16 if args.fp16 else torch.float32)
        load_kwargs.update(dict(
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            ),
            torch_dtype=compute_dtype,
            device_map="auto",
        ))
    else:
        # CPU / smoke-test path: plain fp32
        load_kwargs.update(dict(dtype=torch.float32))

    print(f"[model] loading OPT from {args.model_dir}")
    llm = AutoModelForSequenceClassification.from_pretrained(args.model_dir, **load_kwargs)

    # QSTOPTForSequenceClassification reads llm.hf_device_map; set a sane default
    # when not loaded via device_map="auto".
    if not hasattr(llm, "hf_device_map") or llm.hf_device_map is None:
        llm.hf_device_map = {}

    qst_config = QSTConfig(
        add_layer_norm_before_adapter=False,
        add_layer_norm_after_adapter=True,
        r=args.qst_r,
        alpha_r=args.qst_alpha_r,
        dropout=args.qst_dropout,
        activation="swish",
        fan_in_fan_out=False,
        peft_hidden_size=16,
    )

    print(f"[model] wrapping with QSTOPTForSequenceClassification"
          f"{'AttnRes' if args.use_attn_residual else ''} (r={args.qst_r})")
    if args.use_attn_residual:
        attn_res_config = AttnResConfig(
            use_attn_residual=True,
            attn_residual_type=args.attn_residual_type,
            block_size=args.block_size,
            attn_residual_norm=args.attn_residual_norm,
            freeze_query=args.freeze_attn_query,
        )
        print(f"[model] AttnRes: type={args.attn_residual_type} "
              f"block_size={args.block_size} norm={args.attn_residual_norm} "
              f"freeze_query={args.freeze_attn_query}")
        model = QSTOPTForSequenceClassificationAttnRes(llm, config, qst_config, attn_res_config)
    else:
        model = QSTOPTForSequenceClassification(llm, config, qst_config)
    model.config.pad_token_id = tokenizer.pad_token_id
    # Disable KV cache: training doesn't need it, and new transformers'
    # OPTDecoderLayer no longer returns cache in its output tuple — the old
    # QST code (modeling_opt_qst.py:1214) indexes layer_outputs[1] under the
    # `if use_cache:` branch, which IndexErrors otherwise.
    model.config.use_cache = False
    model.model.config.use_cache = False
    model.model.decoder.config.use_cache = False

    # Freeze backbone (matches QST author's intent). QSTOPTDecoder already sets
    # requires_grad=False on embed/pos/proj/norm; decoder layers were copied as
    # references from `llm.layers`, whose params are still trainable unless we
    # freeze them here.
    for name, p in model.named_parameters():
        if ".backbone." in name:
            p.requires_grad = False

    trainable, total = count_trainable_params(model)
    print(f"[model] trainable={trainable:,}  total={total:,}  pct={100*trainable/total:.2f}%")
    pbreakdown = param_breakdown(model)
    print(f"[model] trainable breakdown:")
    for k, v in pbreakdown.items():
        if v > 0 or k == "backbone_LEAK":  # always show backbone_LEAK so it's visible if nonzero
            print(f"         {k:24s} {v:>12,d}")
    # Quick side-depth read for the record (matches QST: side_depth == backbone depth).
    side_depth = config.num_hidden_layers

    # ---------- training args ----------
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="no",            # smoke tests: don't fill disk
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="linear",
        logging_steps=args.logging_steps,
        logging_dir=os.path.join(args.output_dir, "tb"),
        bf16=args.bf16,
        fp16=args.fp16,
        report_to=[],
        seed=args.seed,
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=build_compute_metrics(args.task),
    )

    print("[train] starting…")
    t0 = time.time()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    train_result = trainer.train()
    print(f"[train] done in {time.time()-t0:.1f}s")

    print("[eval] final…")
    final_eval = trainer.evaluate()
    print(f"[eval] {final_eval}")

    # MNLI: also evaluate mismatched validation set. This is done ONCE after
    # training (we do not track mismatched across epochs). Both matched and
    # mismatched numbers are included in the final summary.
    final_eval_mm = None
    if eval_ds_mm is not None:
        print("[eval] mismatched…")
        final_eval_mm = trainer.evaluate(eval_dataset=eval_ds_mm, metric_key_prefix="eval_mm")
        print(f"[eval_mm] {final_eval_mm}")

    # ---------- dump every recorded metric we'll need for the report ----------
    best = extract_best(trainer.state.log_history, args.task)
    gpu_peak_mb = (
        torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else None
    )

    # final_metrics.eval is set to the best-epoch eval entry, matching original
    # QST's load_best_model_at_end=True convention. This makes our numbers
    # directly comparable to the QST paper's "Best Validation" column.
    # If for any reason no eval was logged during training, fall back to the
    # last-epoch eval from trainer.evaluate() above.
    best_eval_entry = find_best_epoch_eval_entry(trainer.state.log_history, args.task)
    eval_for_report = best_eval_entry if best_eval_entry is not None else final_eval

    summary = {
        "train":             train_result.metrics,
        "eval":              eval_for_report,             # best-epoch (report-canonical)
        "eval_last_epoch":   final_eval,                  # last-epoch (for transparency)
        "eval_mismatched":   final_eval_mm,               # only MNLI uses this
        "best":              best,                        # same info in normalized keys
        "params": {
            "trainable":     trainable,
            "total":         total,
            "pct_trainable": 100.0 * trainable / total,
            "breakdown":     dict(pbreakdown),
        },
        "setup": {
            "task":          args.task,
            "model_dir":     args.model_dir,
            "side_depth":    side_depth,
            "qst_r":         args.qst_r,
            "use_attn_residual":   args.use_attn_residual,
            "attn_residual_type":  args.attn_residual_type if args.use_attn_residual else None,
            "block_size":          args.block_size if args.use_attn_residual and args.attn_residual_type == "block" else None,
            "attn_residual_norm":  args.attn_residual_norm if args.use_attn_residual else None,
            "freeze_attn_query":   args.freeze_attn_query if args.use_attn_residual else None,
            "seed":          args.seed,
            "epochs":        args.epochs,
            "batch_size":    args.batch_size,
            "max_seqlen":    args.max_seqlen,
            "lr":            args.lr,
            "use_4bit":      args.use_4bit,
            "bf16":          args.bf16,
            "transformers":  transformers.__version__,
            "torch":         torch.__version__,
        },
        "gpu_peak_memory_mb": gpu_peak_mb,
    }

    with open(os.path.join(args.output_dir, "final_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Per-step training curve + per-epoch eval curve.
    # log_history has mixed entries: {loss, lr, epoch, step} for training steps
    # and {eval_loss, eval_accuracy, ..., epoch} for eval events.
    log_history = trainer.state.log_history
    with open(os.path.join(args.output_dir, "train_log.jsonl"), "w") as f:
        for entry in log_history:
            if "loss" in entry and "eval_loss" not in entry:
                f.write(json.dumps(entry, default=str) + "\n")
    with open(os.path.join(args.output_dir, "eval_log.jsonl"), "w") as f:
        for entry in log_history:
            if "eval_loss" in entry:
                f.write(json.dumps(entry, default=str) + "\n")
    with open(os.path.join(args.output_dir, "log_history.json"), "w") as f:
        json.dump(log_history, f, indent=2, default=str)

    print(f"[done] outputs → {args.output_dir}")
    if best:
        print(f"[best] {best}")


if __name__ == "__main__":
    main()
