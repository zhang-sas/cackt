#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run hyperparameter sensitivity analysis for absa.run_joint using the current
seed-58 best configuration.

Hyperparameters analyzed:
  1) confidence_temperature
  2) pseudo_confidence_scale
  3) weight_pair_verifier

Usage:
  python run_sensitivity_analysis.py

Optional:
  python run_sensitivity_analysis.py --root out/Res/sensitivity_analysis_seed58
  python run_sensitivity_analysis.py --dry_run
  python run_sensitivity_analysis.py --keep_old
"""

import argparse
import datetime as _dt
import re
import shutil
import subprocess
import sys
from pathlib import Path


def build_common_args(args):
    """Build the shared full-model arguments from the current Restaurant best run."""
    return [
        "--data_dir", args.data_dir,
        "--train_file", args.train_file,
        "--predict_file", args.predict_file,
        "--bert_config_file", args.bert_config_file,
        "--vocab_file", args.vocab_file,
        "--init_checkpoint", args.init_checkpoint,

        "--num_train_epochs", str(args.num_train_epochs),
        "--train_batch_size", str(args.train_batch_size),
        "--predict_batch_size", str(args.predict_batch_size),
        "--learning_rate", str(args.learning_rate),
        "--max_seq_length", str(args.max_seq_length),
        "--save_proportion", str(args.save_proportion),

        # Candidate generation / CA-SCD settings from restaurant_FULL_seed58_p82_try2.
        "--candidate_threshold_mode", "dynamic",
        "--dynamic_threshold_scale", "0.32",
        "--candidate_min_keep", "1",
        # "--confidence_temperature", "1.0", # Moved to grid
        "--min_candidate_confidence", "0.59",

        # Sentiment-conditioned boundary and sentiment prior.
        "--use_sentiment_conditioned_boundary", "True",
        "--sentiment_prior_weight", "0.22",
        "--direct_sentiment_boundary_output", "False",
        "--drop_other_predictions", "True",

        # Aspect-category/sentiment confidence filter.
        "--use_ac_confidence_filter", "True",
        "--ac_min_confidence", "0.52",
        "--ac_other_margin", "0.08",

        # Training schedule.
        "--random_train", "0.9",
        "--expectation_start_step", "200",
        # "--pseudo_confidence_scale", "0.30", # Moved to grid

        # Loss weights.
        "--weight_span", "1e-7",
        "--weight_ac", "1",
        # "--weight_pair_verifier", "0.30", # Moved to grid

        # Decoding.
        "--n_best_size", "8",
        "--use_consistency_decoding", "True",
        "--consistency_pool_multiplier", "2",
        "--consistency_final_top_k", "5",
        "--pair_nms_overlap", "0.40",

        # Pair verifier.
        "--use_pair_verifier", "True",
        "--pair_verifier_alpha", "1.6",
        "--pair_verifier_min_keep_prob", "0.50",

        "--seed", str(args.seed),
    ]


# Define the hyperparameter search space
PARAM_GRID = {
    "confidence_temperature": ["0.5", "1.0", "1.5", "2.0"],
    "pseudo_confidence_scale": ["0.1", "0.3", "0.5", "0.7"],
    "weight_pair_verifier": ["0.0", "0.1", "0.2", "0.3"],
}

# Define the default baseline values to append when evaluating other parameters
DEFAULT_VALS = {
    "confidence_temperature": "1.0",
    "pseudo_confidence_scale": "0.30",
    "weight_pair_verifier": "0.30",
}


def make_experiments():
    experiments = []

    # Baseline configuration (all defaults)
    baseline_args = []
    for k, v in DEFAULT_VALS.items():
        baseline_args.extend([f"--{k}", v])
    experiments.append(("00_baseline", baseline_args))

    idx = 1
    # Generate single-variable control experiments
    for param_name, values in PARAM_GRID.items():
        for val in values:
            if val == DEFAULT_VALS[param_name]:
                continue  # Skip if it matches baseline to save compute

            exp_name = f"{idx:02d}_{param_name}_{val}"
            extra_args = [f"--{param_name}", val]

            # Fill in the rest of the parameters with their default values
            for default_param, default_val in DEFAULT_VALS.items():
                if default_param != param_name:
                    extra_args.extend([f"--{default_param}", default_val])

            experiments.append((exp_name, extra_args))
            idx += 1

    return experiments


def run_one(name, extra_args, common_args, root, log_dir, dry_run=False, keep_old=False):
    out_dir = root / name
    log_path = log_dir / f"{name}.log"
    command_path = root / f"{name}.cmd.txt"

    cmd = [
        sys.executable,
        "-m",
        "absa.run_joint",
        *common_args,
        "--output_dir",
        str(out_dir),
        *extra_args,
    ]

    print("\n" + "=" * 80)
    print(f"Running : {name}")
    print(f"Output  : {out_dir}")
    print(f"Log     : {log_path}")
    print("=" * 80)
    print(" ".join(cmd))

    if dry_run:
        return 0

    root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    command_path.write_text(" ".join(cmd) + "\n", encoding="utf-8")

    if out_dir.exists() and not keep_old:
        # Avoid loading stale checkpoints/results from a previous run.
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
            log_file.flush()

        return_code = process.wait()

    with (root / "status.txt").open("a", encoding="utf-8") as status_file:
        status_file.write(f"{name}\texit_code={return_code}\n")

    if return_code != 0:
        print(f"[WARN] {name} failed with exit code {return_code}; continuing.")

    return return_code


def parse_last_metrics(text):
    all_match = re.findall(
        r"P_all:\s*([0-9.]+),\s*R_all:\s*([0-9.]+),\s*F1_all:\s*([0-9.]+)",
        text,
    )
    ae_match = re.findall(
        r"P_ae:\s*([0-9.]+),\s*R_ae:\s*([0-9.]+),\s*F1_ae:\s*([0-9.]+)",
        text,
    )
    ac_match = re.findall(r"Acc_ac:\s*([0-9.]+)", text)

    if all_match:
        p_all, r_all, f1_all = all_match[-1]
    else:
        p_all = r_all = f1_all = "NA"

    if ae_match:
        p_ae, r_ae, f1_ae = ae_match[-1]
    else:
        p_ae = r_ae = f1_ae = "NA"

    acc_ac = ac_match[-1] if ac_match else "NA"
    return p_all, r_all, f1_all, p_ae, r_ae, f1_ae, acc_ac


def to_float_or_none(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def collect_summary(root):
    log_dir = root / "logs"
    rows = []

    for log_path in sorted(log_dir.glob("*.log")):
        text = log_path.read_text(encoding="utf-8", errors="ignore")
        rows.append((log_path.stem, *parse_last_metrics(text)))

    baseline_f1_all = None
    baseline_f1_ae = None
    baseline_acc_ac = None
    for row in rows:
        if row[0] == "00_baseline":
            baseline_f1_all = to_float_or_none(row[3])
            baseline_f1_ae = to_float_or_none(row[6])
            baseline_acc_ac = to_float_or_none(row[7])
            break

    summary_path = root / "summary.tsv"
    with summary_path.open("w", encoding="utf-8") as f:
        f.write(
            "exp\tP_all\tR_all\tF1_all\tP_ae\tR_ae\tF1_ae\tAcc_ac"
            "\tdelta_F1_all\tdelta_F1_ae\tdelta_Acc_ac\n"
        )
        for row in rows:
            f1_all = to_float_or_none(row[3])
            f1_ae = to_float_or_none(row[6])
            acc_ac = to_float_or_none(row[7])

            delta_f1_all = (
                f"{f1_all - baseline_f1_all:+.4f}"
                if f1_all is not None and baseline_f1_all is not None
                else "NA"
            )
            delta_f1_ae = (
                f"{f1_ae - baseline_f1_ae:+.4f}"
                if f1_ae is not None and baseline_f1_ae is not None
                else "NA"
            )
            delta_acc_ac = (
                f"{acc_ac - baseline_acc_ac:+.4f}"
                if acc_ac is not None and baseline_acc_ac is not None
                else "NA"
            )

            f.write("\t".join(row + (delta_f1_all, delta_f1_ae, delta_acc_ac)) + "\n")

    print("\n" + "=" * 80)
    print(f"Saved summary: {summary_path}")
    print("=" * 80)
    print(
        "exp\tP_all\tR_all\tF1_all\tP_ae\tR_ae\tF1_ae\tAcc_ac"
        "\tdelta_F1_all\tdelta_F1_ae\tdelta_Acc_ac"
    )
    for row in rows:
        pass  # Only used for loop, logic handles cleanly below

    print(summary_path.read_text(encoding="utf-8"))

    return summary_path


def parse_args():
    timestamp = _dt.datetime.now().strftime("%m%d_%H%M")
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--root",
        default=f"out/Res/sensitivity_analysis_seed58_{timestamp}",
        help="Root directory for all sensitivity analysis outputs.",
    )
    parser.add_argument("--data_dir", default="data/absa")
    parser.add_argument("--train_file", default="split_rest_total_train.txt")
    parser.add_argument("--predict_file", default="rest_total_test.txt")
    parser.add_argument("--bert_config_file", default="bert-large-uncased/bert_config.json")
    parser.add_argument("--vocab_file", default="bert-large-uncased/vocab.txt")
    parser.add_argument("--init_checkpoint", default="bert-large-uncased/pytorch_model.bin")

    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--predict_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", default="2e-5")
    parser.add_argument("--max_seq_length", type=int, default=85)
    parser.add_argument("--save_proportion", default="0.5")
    parser.add_argument("--seed", type=int, default=58)

    parser.add_argument(
        "--keep_old",
        action="store_true",
        help="Do not delete an existing output directory before running an experiment.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print commands; do not run experiments.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    root = Path(args.root)
    log_dir = root / "logs"
    root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    common_args = build_common_args(args)
    experiments = make_experiments()

    print(f"Results root: {root}")
    print(f"Logs dir    : {log_dir}")
    print("Experiments : " + ", ".join(name for name, _ in experiments))

    failed = []
    for name, extra_args in experiments:
        code = run_one(
            name=name,
            extra_args=extra_args,
            common_args=common_args,
            root=root,
            log_dir=log_dir,
            dry_run=args.dry_run,
            keep_old=args.keep_old,
        )
        if code != 0:
            failed.append(name)

    if not args.dry_run:
        collect_summary(root)

    print("\nAll requested sensitivity analyses finished.")
    if failed:
        print("Failed runs:", ", ".join(failed))
    print(f"Root: {root}")


if __name__ == "__main__":
    main()