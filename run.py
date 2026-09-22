"""Grid runner for the improved FCKT joint model with CA-SCD and AC post-processing.

Example:
    python run.py \
        --data_dir data/absa \
        --train_file split_laptop14_train.txt \
        --predict_file laptop14_test.txt \
        --random_train 0.9 \
        --candidate_threshold_mode dynamic \
        --dynamic_threshold_scale 0.0 \
        --min_candidate_confidence 0.50 \
        --use_sentiment_conditioned_boundary True \
        --sentiment_prior_weight 0.25 \
        --use_ac_confidence_filter True \
        --ac_min_confidence 0.40 \
        --ac_other_margin 0.00 \
        --use_consistency_decoding True \
        --consistency_pool_multiplier 2 \
        --consistency_final_top_k 0 \
        --n_best_size 8 \
        --learning_rate 2e-5 \
        --weight_span 1e-7 \
        --weight_ac 1 \
        --epochs 100 \
        --seed 58 \
        --tag laptop14_CA_SCD_ACF
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
from typing import Iterable, List


def str2bool(value):
    """Parse boolean CLI values and keep compatibility with absa.run_joint."""
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_csv_floats(value: str) -> List[float]:
    """Parse a comma-separated list of floats."""
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def build_command(args, random_train: float, logit_threshold: float,
                  learning_rate: float, weight_span: float, output_dir: str) -> List[str]:
    """Build one reproducible training command."""
    command = [
        sys.executable, "-m", "absa.run_joint",

        # Data/model paths.
        "--data_dir", args.data_dir,
        "--train_file", args.train_file,
        "--predict_file", args.predict_file,
        "--bert_config_file", args.bert_config_file,
        "--vocab_file", args.vocab_file,
        "--init_checkpoint", args.init_checkpoint,
        "--output_dir", output_dir,

        # Core training/evaluation settings.
        "--num_train_epochs", str(args.epochs),
        "--train_batch_size", str(args.train_batch_size),
        "--predict_batch_size", str(args.predict_batch_size),
        "--learning_rate", str(learning_rate),
        "--warmup_proportion", str(args.warmup_proportion),
        "--max_seq_length", str(args.max_seq_length),
        "--n_best_size", str(args.n_best_size),
        "--max_answer_length", str(args.max_answer_length),
        "--random_train", str(random_train),
        "--expectation_start_step", str(args.expectation_start_step),
        "--pseudo_confidence_scale", str(args.pseudo_confidence_scale),
        "--weight_span", str(weight_span),
        "--weight_ac", str(args.weight_ac),
        "--weight_pair_verifier", str(args.weight_pair_verifier),
        "--seed", str(args.seed),

        # Candidate generation and boundary calibration.
        "--logit_threshold", str(logit_threshold),
        "--candidate_threshold_mode", args.candidate_threshold_mode,
        "--dynamic_threshold_scale", str(args.dynamic_threshold_scale),
        "--candidate_min_keep", str(args.candidate_min_keep),
        "--confidence_temperature", str(args.confidence_temperature),
        "--min_candidate_confidence", str(args.min_candidate_confidence),

        # Sentiment-conditioned boundary transfer.
        "--use_sentiment_conditioned_boundary", str(args.use_sentiment_conditioned_boundary),
        "--sentiment_prior_weight", str(args.sentiment_prior_weight),
        "--direct_sentiment_boundary_output", str(args.direct_sentiment_boundary_output),
        "--drop_other_predictions", str(args.drop_other_predictions),

        # AC confidence post-processing.
        "--use_ac_confidence_filter", str(args.use_ac_confidence_filter),
        "--ac_min_confidence", str(args.ac_min_confidence),
        "--ac_other_margin", str(args.ac_other_margin),

        # Confidence-aware aspect-sentiment consistency decoding.
        "--use_consistency_decoding", str(args.use_consistency_decoding),
        "--consistency_pool_multiplier", str(args.consistency_pool_multiplier),
        "--consistency_final_top_k", str(args.consistency_final_top_k),
        "--pair_nms_overlap", str(args.pair_nms_overlap),
        "--use_pair_verifier", str(args.use_pair_verifier),
        "--pair_verifier_alpha", str(args.pair_verifier_alpha),
        "--pair_verifier_min_keep_prob", str(args.pair_verifier_min_keep_prob),

        # Pseudo-candidate branch control.
        "--train_pseudo_other", str(args.train_pseudo_other),
    ]

    if args.eval_file:
        command += ["--eval_file", args.eval_file]
    if args.disable_expectation:
        command += ["--disable_expectation"]
    if args.disable_pair_nms:
        command += ["--disable_pair_nms"]
    if args.do_train is not None:
        command += ["--do_train", str(args.do_train)]
    if args.do_predict is not None:
        command += ["--do_predict", str(args.do_predict)]
    if args.no_cuda:
        command += ["--no_cuda"]
    if args.debug:
        command += ["--debug"]
    return command


def run_command(command: Iterable[str]) -> int:
    """Run a subprocess and stream logs to the current terminal."""
    print("\n" + "=" * 80)
    print("Running:", " ".join(command))
    print("=" * 80)
    completed = subprocess.run(list(command), check=False)
    return completed.returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FCKT/CA-SCD grid experiments.")

    # Data/model paths.
    parser.add_argument("--data_dir", default="data/absa")
    parser.add_argument("--train_file", default="split_laptop14_train.txt")
    parser.add_argument("--predict_file", default="laptop14_test.txt")
    parser.add_argument("--eval_file", default=None, help="Optional dev file used by absa.run_joint for checkpoint selection.")
    parser.add_argument("--bert_config_file", default="bert-large-uncased/bert_config.json")
    parser.add_argument("--vocab_file", default="bert-large-uncased/vocab.txt")
    parser.add_argument("--init_checkpoint", default="bert-large-uncased/pytorch_model.bin")

    # Grid dimensions.
    parser.add_argument("--random_train", default="0.9", help="Comma-separated gold-branch probabilities.")
    parser.add_argument("--logit_threshold", default="7.5", help="Comma-separated fixed thresholds; ignored when candidate_threshold_mode is dynamic/none.")
    parser.add_argument("--learning_rate", default="2e-5", help="Comma-separated learning rates.")
    parser.add_argument("--weight_span", default="1e-7", help="Comma-separated contrastive-loss weights.")

    # Core training/evaluation settings.
    parser.add_argument("--epochs", default=100, type=float)
    parser.add_argument("--train_batch_size", default=16, type=int)
    parser.add_argument("--predict_batch_size", default=16, type=int)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--max_seq_length", default=85, type=int)
    parser.add_argument("--n_best_size", default=8, type=int)
    parser.add_argument("--max_answer_length", default=12, type=int)
    parser.add_argument("--weight_ac", default=1.0, type=float)
    parser.add_argument("--seed", default=58, type=int)
    parser.add_argument("--output_root", default="out/Res")
    parser.add_argument("--tag", default="CA_SCD_ACF")
    parser.add_argument("--do_train", default=None, type=str2bool, nargs="?", const=True)
    parser.add_argument("--do_predict", default=None, type=str2bool, nargs="?", const=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_cuda", action="store_true")

    # Candidate generation and boundary calibration.
    parser.add_argument("--candidate_threshold_mode", default="dynamic", choices=["fixed", "dynamic", "none"])
    parser.add_argument("--confidence_temperature", default=1.0, type=float)
    parser.add_argument("--min_candidate_confidence", default=0.50, type=float)
    parser.add_argument("--dynamic_threshold_scale", default=0.0, type=float)
    parser.add_argument("--candidate_min_keep", default=1, type=int)

    # Sentiment-conditioned boundary transfer.
    parser.add_argument("--use_sentiment_conditioned_boundary", default=True, type=str2bool, nargs="?", const=True)
    parser.add_argument("--sentiment_prior_weight", default=0.25, type=float)
    parser.add_argument("--direct_sentiment_boundary_output", default=False, type=str2bool, nargs="?", const=True)
    parser.add_argument("--drop_other_predictions", default=True, type=str2bool, nargs="?", const=True)

    # New AC confidence post-processing parameters.
    parser.add_argument("--use_ac_confidence_filter", default=True, type=str2bool, nargs="?", const=True)
    parser.add_argument("--ac_min_confidence", default=0.40, type=float)
    parser.add_argument("--ac_other_margin", default=0.00, type=float)

    # New CA-SCD / pair-level consistency parameters.
    parser.add_argument("--use_consistency_decoding", default=True, type=str2bool, nargs="?", const=True)
    parser.add_argument("--consistency_pool_multiplier", default=2, type=int)
    parser.add_argument("--consistency_final_top_k", default=0, type=int)
    parser.add_argument("--pair_nms_overlap", default=0.5, type=float)
    parser.add_argument("--disable_pair_nms", action="store_true")
    parser.add_argument("--use_pair_verifier", default=True, type=str2bool, nargs="?", const=True)
    parser.add_argument("--weight_pair_verifier", default=0.2, type=float)
    parser.add_argument("--pair_verifier_alpha", default=1.0, type=float)
    parser.add_argument("--pair_verifier_min_keep_prob", default=0.0, type=float)

    # Pseudo-candidate branch control.
    parser.add_argument("--expectation_start_step", default=200, type=int)
    parser.add_argument("--disable_expectation", action="store_true")
    parser.add_argument("--pseudo_confidence_scale", default=0.30, type=float)
    parser.add_argument("--train_pseudo_other", default=False, type=str2bool, nargs="?", const=True)

    args = parser.parse_args()

    random_trains = parse_csv_floats(args.random_train)
    thresholds = parse_csv_floats(args.logit_threshold)
    learning_rates = parse_csv_floats(args.learning_rate)
    span_weights = parse_csv_floats(args.weight_span)

    failures = []
    for random_train, threshold, lr, span_weight in itertools.product(
            random_trains, thresholds, learning_rates, span_weights):
        run_name = (
            f"{args.tag}-rt{random_train}-{args.candidate_threshold_mode}"
            f"-th{threshold}-dts{args.dynamic_threshold_scale}"
            f"-mcc{args.min_candidate_confidence}-acf{args.ac_min_confidence}"
            f"-tom{args.ac_other_margin}-top{args.consistency_final_top_k}"
            f"-lr{lr}-ws{span_weight}-wac{args.weight_ac}-seed{args.seed}"
        )
        output_dir = os.path.join(args.output_root, run_name)
        os.makedirs(output_dir, exist_ok=True)
        command = build_command(args, random_train, threshold, lr, span_weight, output_dir)
        return_code = run_command(command)
        if return_code != 0:
            failures.append((run_name, return_code))

    if failures:
        print("\nFailed runs:")
        for run_name, return_code in failures:
            print(f"  {run_name}: return code {return_code}")
        sys.exit(1)

    print("\nAll runs finished successfully.")


if __name__ == "__main__":
    main()
