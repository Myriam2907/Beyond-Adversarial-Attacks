import os
import json
import csv
import argparse
import shutil
import re
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
except ImportError as exc:
    raise ImportError(
        
    ) from exc


SIGNAL_ROOT = "./js_consistency_signal_mapillary"
OUT_ROOT = "./js_consistency_analysis_mapillary"

MODELS = ["mobilenet", "convnext", "efficientnet"]

CONDITIONS = [
    "clean",
    "fgsm",
    "rfgsm",
    "pgd",
    "random_patch",
    "gaussian",
    "salt_pepper",
    "light",
    "fog",
    "motion_blur",
]

DEFAULT_CALIBRATION_FRACTION = 0.50
DEFAULT_CLEAN_FPR = 0.05
DEFAULT_SEED = 123


def load_condition(model_key, condition):
    root = os.path.join(SIGNAL_ROOT, model_key, condition)

    required = {
        "js": os.path.join(root, "js_divergence.npy"),
        "labels": os.path.join(root, "labels.npy"),
        "pred_base": os.path.join(root, "pred_base.npy"),
        "filenames": os.path.join(root, "filenames.npy"),
    }

    missing = [p for p in required.values() if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"[{model_key}/{condition}] missing files:\n  "
            + "\n  ".join(missing)
        )

    js = np.load(required["js"]).astype(np.float32, copy=False)
    labels = np.load(required["labels"]).astype(np.int64, copy=False)
    pred_base = np.load(required["pred_base"]).astype(np.int64, copy=False)
    filenames = np.load(required["filenames"], allow_pickle=False).astype(str)

    n = len(js)
    if not (len(labels) == len(pred_base) == len(filenames) == n):
        raise RuntimeError(
            f"[{model_key}/{condition}] array length mismatch."
        )

    return {
        "js": js,
        "labels": labels,
        "pred_base": pred_base,
        "filenames": filenames,
    }


def extract_encoded_index(filename):
    base = os.path.basename(str(filename))
    match = re.match(r"^(\d+)(?:_|\.|$)", base)
    if match is None:
        return None
    return int(match.group(1))


def check_alignment(clean, other, model_key, condition):
    n_clean = len(clean["filenames"])
    n_other = len(other["filenames"])

    if n_clean != n_other:
        raise RuntimeError(
            f"[{model_key}/{condition}] image count differs "
            f"from clean: {n_clean} vs {n_other}."
        )

    if not np.array_equal(clean["labels"], other["labels"]):
        raise RuntimeError(
            f"[{model_key}/{condition}] labels are not aligned with clean."
        )

    if np.array_equal(clean["filenames"], other["filenames"]):
        return "exact_filenames"

    parsed = [extract_encoded_index(x) for x in other["filenames"]]

    if any(x is None for x in parsed):
        raise RuntimeError(
            f"[{model_key}/{condition}] filenames differ from clean "
            "and do not consistently encode global indices."
        )

    encoded = np.asarray(parsed, dtype=np.int64)
    expected = np.arange(n_other, dtype=np.int64)

    if not np.array_equal(encoded, expected):
        bad = np.flatnonzero(encoded != expected)
        first = int(bad[0]) if len(bad) else -1
        raise RuntimeError(
            f"[{model_key}/{condition}] generated filename indices "
            f"are not aligned. First mismatch at position {first}."
        )

    return "encoded_global_index"


def make_stratified_split(labels, calibration_fraction, seed):
    rng = np.random.default_rng(seed)

    calibration = []
    evaluation = []

    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls).copy()
        rng.shuffle(idx)
        n = len(idx)

        if n == 1:
            evaluation.extend(idx.tolist())
            continue

        n_cal = int(round(n * calibration_fraction))
        n_cal = max(1, min(n - 1, n_cal))

        calibration.extend(idx[:n_cal].tolist())
        evaluation.extend(idx[n_cal:].tolist())

    calibration = np.asarray(sorted(calibration), dtype=np.int64)
    evaluation = np.asarray(sorted(evaluation), dtype=np.int64)

    if len(calibration) == 0 or len(evaluation) == 0:
        raise RuntimeError("Calibration or evaluation split is empty.")

    if np.intersect1d(calibration, evaluation).size != 0:
        raise RuntimeError("Calibration/evaluation split overlaps.")

    return calibration, evaluation


def threshold_for_target_fpr(clean_cal_scores, target_fpr):
    percentile = 100.0 * (1.0 - target_fpr)
    return float(np.percentile(clean_cal_scores, percentile))


def rate_above(scores, threshold):
    return float(np.mean(scores > threshold))


def safe_auc(clean_scores, attack_scores):
    y_true = np.concatenate([
        np.zeros(len(clean_scores), dtype=np.int8),
        np.ones(len(attack_scores), dtype=np.int8),
    ])

    y_score = np.concatenate([
        clean_scores,
        attack_scores,
    ])

    return float(roc_auc_score(y_true, y_score))


def analyze_model(
    model_key,
    calibration_fraction,
    clean_fpr_target,
    seed,
):
    print("\n" + "=" * 88)
    print(f"[{model_key}] JS CONSISTENCY ANALYSIS")
    print("=" * 88)

    clean = load_condition(model_key, "clean")
    n = len(clean["js"])

    cal_idx, eval_idx = make_stratified_split(
        clean["labels"],
        calibration_fraction,
        seed,
    )

    print(f"  Clean images       : {n:,}")
    print(f"  Calibration split  : {len(cal_idx):,}")
    print(f"  Evaluation split   : {len(eval_idx):,}")

    clean_cal_scores = clean["js"][cal_idx]
    clean_eval_scores = clean["js"][eval_idx]

    threshold = threshold_for_target_fpr(
        clean_cal_scores,
        clean_fpr_target,
    )

    measured_clean_fpr = rate_above(
        clean_eval_scores,
        threshold,
    )

    print(f"  Threshold           : {threshold:.8f}")
    print(f"  Target clean FPR    : {100.0 * clean_fpr_target:.2f}%")
    print(f"  Held-out clean FPR  : {100.0 * measured_clean_fpr:.2f}%")

    model_out = os.path.join(OUT_ROOT, model_key)

   
    if os.path.exists(model_out):
        shutil.rmtree(model_out)

    Path(model_out).mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(model_out, "calibration_indices.npy"), cal_idx)
    np.save(os.path.join(model_out, "evaluation_indices.npy"), eval_idx)
    np.save(os.path.join(model_out, "clean_js_calibration.npy"), clean_cal_scores)
    np.save(os.path.join(model_out, "clean_js_evaluation.npy"), clean_eval_scores)

    rows = []

    clean_row = {
        "model": model_key,
        "condition": "clean",
        "num_total": int(n),
        "num_evaluated": int(len(eval_idx)),
        "threshold": float(threshold),
        "target_clean_fpr": float(clean_fpr_target),
        "detection_rate": float(measured_clean_fpr),
        "auc_vs_clean": None,
        "score_mean": float(np.mean(clean_eval_scores)),
        "score_median": float(np.median(clean_eval_scores)),
        "score_p95": float(np.percentile(clean_eval_scores, 95)),
        "detected_among_wrong": None,
        "detected_among_correct": None,
        "num_wrong_evaluated": None,
        "num_correct_evaluated": None,
        "alignment_mode": "clean_reference",
    }
    rows.append(clean_row)

    print(
        "\n  CONDITION           TPR/FPR      AUC      "
        "mean JS      median JS"
    )
    print("  " + "-" * 72)

    print(
        f"  {'clean':18s} "
        f"{100.0 * measured_clean_fpr:7.2f}%   "
        f"{'--':>6s}   "
        f"{np.mean(clean_eval_scores):10.6f}   "
        f"{np.median(clean_eval_scores):10.6f}"
    )

    for condition in CONDITIONS:
        if condition == "clean":
            continue

        other = load_condition(model_key, condition)

        alignment_mode = check_alignment(
            clean,
            other,
            model_key,
            condition,
        )

        attack_scores = other["js"][eval_idx]

        tpr = rate_above(
            attack_scores,
            threshold,
        )

        auc = safe_auc(
            clean_eval_scores,
            attack_scores,
        )

        wrong_mask = (
            other["pred_base"][eval_idx]
            != other["labels"][eval_idx]
        )
        correct_mask = ~wrong_mask

        det_wrong = (
            float(np.mean(attack_scores[wrong_mask] > threshold))
            if np.any(wrong_mask)
            else None
        )

        det_correct = (
            float(np.mean(attack_scores[correct_mask] > threshold))
            if np.any(correct_mask)
            else None
        )

        condition_out = os.path.join(
            model_out,
            condition,
        )
        Path(condition_out).mkdir(
            parents=True,
            exist_ok=True,
        )

        np.save(
            os.path.join(condition_out, "js_scores_eval.npy"),
            attack_scores,
        )

        suspicious_mask = attack_scores > threshold
        suspicious_indices = eval_idx[suspicious_mask]

        np.save(
            os.path.join(condition_out, "suspicious_indices.npy"),
            suspicious_indices,
        )

        row = {
            "model": model_key,
            "condition": condition,
            "num_total": int(n),
            "num_evaluated": int(len(eval_idx)),
            "threshold": float(threshold),
            "target_clean_fpr": float(clean_fpr_target),
            "detection_rate": float(tpr),
            "auc_vs_clean": float(auc),
            "score_mean": float(np.mean(attack_scores)),
            "score_median": float(np.median(attack_scores)),
            "score_p95": float(np.percentile(attack_scores, 95)),
            "detected_among_wrong": det_wrong,
            "detected_among_correct": det_correct,
            "num_wrong_evaluated": int(np.sum(wrong_mask)),
            "num_correct_evaluated": int(np.sum(correct_mask)),
            "alignment_mode": alignment_mode,
        }

        rows.append(row)

        print(
            f"  {condition:18s} "
            f"{100.0 * tpr:7.2f}%   "
            f"{auc:6.3f}   "
            f"{np.mean(attack_scores):10.6f}   "
            f"{np.median(attack_scores):10.6f}"
        )

    summary = {
        "model": model_key,
        "signal_root": SIGNAL_ROOT,
        "output_root": model_out,
        "num_total": int(n),
        "calibration_fraction": float(calibration_fraction),
        "num_calibration": int(len(cal_idx)),
        "num_evaluation": int(len(eval_idx)),
        "seed": int(seed),
        "threshold": float(threshold),
        "target_clean_fpr": float(clean_fpr_target),
        "heldout_clean_fpr": float(measured_clean_fpr),
        "signal_definition": (
            "Jensen-Shannon divergence between the 401-class "
            "probability distributions of the original image and "
            "the deterministic 224->208->224 resized image."
        ),
        "score_direction": "higher = more suspicious",
        "rows": rows,
    }

    with open(
        os.path.join(model_out, "js_analysis_results.json"),
        "w",
    ) as f:
        json.dump(summary, f, indent=2)

    return summary


def save_combined_csv(all_results):
    path = os.path.join(
        OUT_ROOT,
        "js_combined_summary.csv",
    )

    fields = [
        "model",
        "condition",
        "num_total",
        "num_evaluated",
        "threshold",
        "target_clean_fpr",
        "detection_rate",
        "auc_vs_clean",
        "score_mean",
        "score_median",
        "score_p95",
        "detected_among_wrong",
        "detected_among_correct",
        "num_wrong_evaluated",
        "num_correct_evaluated",
        "alignment_mode",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for result in all_results:
            for row in result["rows"]:
                writer.writerow({
                    key: row.get(key)
                    for key in fields
                })

    return path


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "all",
            "mobilenet",
            "convnext",
            "efficientnet",
        ],
        default="all",
    )

    parser.add_argument(
        "--calibration_fraction",
        type=float,
        default=DEFAULT_CALIBRATION_FRACTION,
    )

    parser.add_argument(
        "--clean_fpr",
        type=float,
        default=DEFAULT_CLEAN_FPR,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(SIGNAL_ROOT):
        raise FileNotFoundError(
            f"JS signal root not found: {SIGNAL_ROOT}"
        )

    if not (0.0 < args.calibration_fraction < 1.0):
        raise ValueError(
            "--calibration_fraction must be between 0 and 1."
        )

    if not (0.0 < args.clean_fpr < 1.0):
        raise ValueError(
            "--clean_fpr must be between 0 and 1."
        )

    selected_models = (
        MODELS
        if args.model == "all"
        else [args.model]
    )

    print("=" * 88)
    print("MAPILLARY JS CONSISTENCY SIGNAL ANALYSIS")
    print("=" * 88)
    print(f"Models               : {selected_models}")
    print(f"Input JS signals     : {SIGNAL_ROOT}")
    print(f"NEW output only      : {OUT_ROOT}")
    print(f"Calibration fraction : {args.calibration_fraction}")
    print(f"Target clean FPR     : {100.0 * args.clean_fpr:.2f}%")
    print(f"Seed                 : {args.seed}")
    print("Previous results     : NOT MODIFIED")

    Path(OUT_ROOT).mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    for model_key in selected_models:
        result = analyze_model(
            model_key=model_key,
            calibration_fraction=args.calibration_fraction,
            clean_fpr_target=args.clean_fpr,
            seed=args.seed,
        )
        all_results.append(result)

    combined_json = os.path.join(
        OUT_ROOT,
        "js_combined_results.json",
    )

    with open(combined_json, "w") as f:
        json.dump(all_results, f, indent=2)

    combined_csv = save_combined_csv(all_results)

    print("\n" + "=" * 88)
    print("JS CONSISTENCY ANALYSIS COMPLETE")
    print("=" * 88)

    for result in all_results:
        print(f"\n{result['model']}:")

        for row in result["rows"]:
            condition = row["condition"]
            rate = 100.0 * row["detection_rate"]

            if condition == "clean":
                print(
                    f"  {condition:14s} "
                    f"FPR = {rate:6.2f}%"
                )
            else:
                print(
                    f"  {condition:14s} "
                    f"TPR = {rate:6.2f}% "
                    f"| AUC = {row['auc_vs_clean']:.3f}"
                )

    print(f"\nCombined JSON -> {combined_json}")
    print(f"Combined CSV  -> {combined_csv}")
    print(
        "\nNo previous detector/attack/"
        "Mahalanobis/DDPM results were modified."
    )


if __name__ == "__main__":
    main()