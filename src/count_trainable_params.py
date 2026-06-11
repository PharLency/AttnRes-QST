"""Count trainable parameters for QST and AttnRes-QST models.

This script instantiates the model wrapper without loading data or running
training. It is intended for checking the parameter accounting used in the
paper tables.

Examples
--------
Use an existing training configuration:

    python src/count_trainable_params.py \
        --run_args outputs/opt-1.3b/attnres-block-bs2/cola-s42/run_args.json

Count several saved configurations:

    python src/count_trainable_params.py \
        --glob "outputs/opt-*/attnres-block-bs2/cola-s42/run_args.json"

Manual mode:

    python src/count_trainable_params.py \
        --model_dir .models/opt-1.3b --model_tag opt-1.3b \
        --task cola --method attnres-block-bs2 --use_4bit --bf16
"""

import argparse
import csv
import glob as globlib
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
QST_DIR = PROJECT_ROOT / "third_party" / "qst"
sys.path.insert(0, str(QST_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


TASK_NUM_LABELS = {
    "cola": 2,
    "sst2": 2,
    "mrpc": 2,
    "rte": 2,
    "qnli": 2,
    "qqp": 2,
    "mnli": 3,
    "stsb": 1,
}

NOMINAL_BACKBONE_PARAMS = {
    "opt-1.3b": 1.3e9,
    "facebook/opt-1.3b": 1.3e9,
    "opt-2.7b": 2.7e9,
    "facebook/opt-2.7b": 2.7e9,
    "opt-6.7b": 6.7e9,
    "facebook/opt-6.7b": 6.7e9,
}


def resolve_path(value):
    if value is None:
        return None
    if value.startswith((".", "/")):
        return os.path.abspath(value)
    return value


def infer_model_tag(model_dir, explicit=None):
    if explicit:
        return explicit
    name = Path(model_dir).name.lower().replace("facebook-", "")
    if name in NOMINAL_BACKBONE_PARAMS:
        return name
    if "opt-1.3b" in model_dir:
        return "opt-1.3b"
    if "opt-2.7b" in model_dir:
        return "opt-2.7b"
    if "opt-6.7b" in model_dir:
        return "opt-6.7b"
    return name


def method_to_attnres(method, args_dict):
    method = (method or "").lower()
    if method in ("qst", "baseline"):
        return dict(use_attn_residual=False)
    if method in ("attnres-uniform-avg", "uniform", "uniform-avg"):
        return dict(
            use_attn_residual=True,
            attn_residual_type="full",
            block_size=0,
            attn_residual_norm="none",
            freeze_attn_query=True,
        )
    if method == "attnres-full":
        return dict(
            use_attn_residual=True,
            attn_residual_type="full",
            block_size=0,
            attn_residual_norm=args_dict.get("attn_residual_norm", "rmsnorm"),
            freeze_attn_query=False,
        )
    if method.startswith("attnres-block"):
        block_size = args_dict.get("block_size")
        if block_size is None:
            block_size = 4 if "bs4" in method else 2
        return dict(
            use_attn_residual=True,
            attn_residual_type="block",
            block_size=int(block_size),
            attn_residual_norm=args_dict.get("attn_residual_norm", "rmsnorm"),
            freeze_attn_query=False,
        )
    return dict(
        use_attn_residual=bool(args_dict.get("use_attn_residual", False)),
        attn_residual_type=args_dict.get("attn_residual_type", "block"),
        block_size=int(args_dict.get("block_size", 2)),
        attn_residual_norm=args_dict.get("attn_residual_norm", "rmsnorm"),
        freeze_attn_query=bool(args_dict.get("freeze_attn_query", False)),
    )


def count_trainable_params(model):
    trainable = 0
    total = 0
    for p in model.parameters():
        n = p.numel()
        total += n
        if p.requires_grad:
            trainable += n
    return trainable, total


def param_breakdown(model):
    groups = OrderedDict(
        [
            ("qst_layers", 0),
            ("downsample", 0),
            ("z_gate", 0),
            ("final_layer_norm_qst", 0),
            ("upsample", 0),
            ("score_z", 0),
            ("score", 0),
            ("attn_residual", 0),
            ("other_trainable", 0),
            ("backbone_LEAK", 0),
        ]
    )
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


def build_model(spec):
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification

    from QSTConfig import QSTConfig
    from modeling_opt_qst import QSTOPTForSequenceClassification
    from qst_opt_attnres import (
        AttnResConfig,
        QSTOPTForSequenceClassificationAttnRes,
    )

    model_dir = resolve_path(spec["model_dir"])
    task = spec.get("task", "cola")
    num_labels = TASK_NUM_LABELS[task]

    config = AutoConfig.from_pretrained(model_dir, num_labels=num_labels)
    if task == "stsb":
        config.problem_type = "regression"

    load_kwargs = dict(num_labels=num_labels, attn_implementation="eager")
    if task == "stsb":
        load_kwargs["problem_type"] = "regression"
    if spec.get("use_4bit"):
        from transformers import BitsAndBytesConfig

        compute_dtype = (
            torch.bfloat16
            if spec.get("bf16")
            else (torch.float16 if spec.get("fp16") else torch.float32)
        )
        load_kwargs.update(
            dict(
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=compute_dtype,
                ),
                torch_dtype=compute_dtype,
                device_map="auto",
            )
        )
    else:
        load_kwargs["dtype"] = torch.float32

    llm = AutoModelForSequenceClassification.from_pretrained(model_dir, **load_kwargs)
    if not hasattr(llm, "hf_device_map") or llm.hf_device_map is None:
        llm.hf_device_map = {}

    qst_config = QSTConfig(
        add_layer_norm_before_adapter=False,
        add_layer_norm_after_adapter=True,
        r=int(spec.get("qst_r", 16)),
        alpha_r=int(spec.get("qst_alpha_r", 16)),
        dropout=float(spec.get("qst_dropout", 0.1)),
        activation="swish",
        fan_in_fan_out=False,
        peft_hidden_size=16,
    )

    if spec.get("use_attn_residual"):
        attn_res_config = AttnResConfig(
            use_attn_residual=True,
            attn_residual_type=spec.get("attn_residual_type", "block"),
            block_size=int(spec.get("block_size", 2) or 2),
            attn_residual_norm=spec.get("attn_residual_norm", "rmsnorm"),
            freeze_query=bool(spec.get("freeze_attn_query", False)),
        )
        model = QSTOPTForSequenceClassificationAttnRes(
            llm, config, qst_config, attn_res_config
        )
    else:
        model = QSTOPTForSequenceClassification(llm, config, qst_config)

    for name, p in model.named_parameters():
        if ".backbone." in name:
            p.requires_grad = False
    return model


def spec_from_run_args(path):
    data = json.loads(Path(path).read_text())
    run_dir = Path(path).parent
    method = run_dir.parent.name
    model_tag = run_dir.parent.parent.name
    data["method"] = method
    data["model_tag"] = model_tag
    data.update(method_to_attnres(method, data))
    return data


def spec_from_cli(args):
    spec = vars(args).copy()
    spec.update(method_to_attnres(args.method, spec))
    spec["model_tag"] = infer_model_tag(args.model_dir, args.model_tag)
    return spec


def summarize(spec):
    model = build_model(spec)
    trainable, total = count_trainable_params(model)
    breakdown = param_breakdown(model)
    model_tag = infer_model_tag(spec["model_dir"], spec.get("model_tag"))
    nominal = NOMINAL_BACKBONE_PARAMS.get(model_tag)
    nominal_pct = 100.0 * trainable / nominal if nominal else None
    row = OrderedDict(
        [
            ("model", model_tag),
            ("method", spec.get("method", "")),
            ("task", spec.get("task", "")),
            ("trainable", trainable),
            ("total", total),
            ("pct_internal_total", f"{100.0 * trainable / total:.3f}"),
            (
                "pct_nominal_backbone",
                f"{nominal_pct:.3f}" if nominal_pct is not None else "",
            ),
        ]
    )
    row.update((f"breakdown_{k}", v) for k, v in breakdown.items())
    return row


def print_row(row):
    print(
        f"{row['model']} {row['method']} {row['task']}: "
        f"trainable={row['trainable']:,} total={row['total']:,} "
        f"pct_internal={row['pct_internal_total']}%"
        + (
            f" pct_nominal={row['pct_nominal_backbone']}%"
            if row["pct_nominal_backbone"] != ""
            else ""
        )
    )
    for key, value in row.items():
        if key.startswith("breakdown_") and value:
            print(f"  {key.removeprefix('breakdown_'):24s} {value:>12,d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_args", nargs="*", default=[])
    parser.add_argument("--glob", nargs="*", default=[])
    parser.add_argument("--csv_out", default=None)
    parser.add_argument("--model_dir", default=str(PROJECT_ROOT / ".models" / "opt-1.3b"))
    parser.add_argument("--model_tag", default=None)
    parser.add_argument("--task", choices=sorted(TASK_NUM_LABELS), default="cola")
    parser.add_argument("--method", default="qst")
    parser.add_argument("--qst_r", type=int, default=16)
    parser.add_argument("--qst_alpha_r", type=int, default=16)
    parser.add_argument("--qst_dropout", type=float, default=0.1)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    paths = list(args.run_args)
    for pattern in args.glob:
        paths.extend(sorted(globlib.glob(pattern)))

    specs = [spec_from_run_args(p) for p in paths] if paths else [spec_from_cli(args)]
    rows = []
    for spec in specs:
        row = summarize(spec)
        rows.append(row)
        print_row(row)

    if args.csv_out:
        out = Path(args.csv_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[write] {out}")


if __name__ == "__main__":
    main()
