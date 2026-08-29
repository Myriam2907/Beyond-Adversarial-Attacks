import os
import json
import csv
import argparse
import re
from pathlib import Path

import numpy as np


OUT_ROOT = "./combined_detector_js_calibrated_mapillary"
JS_SIGNAL_ROOT = "./js_consistency_signal_mapillary"
JS_ANALYSIS_ROOT = "./js_consistency_analysis_mapillary"

MODELS = {
    "mobilenet": {"eval_root": "./eval_mobilenet"},
    "convnext": {"eval_root": "./eval_convnext"},
    "efficientnet": {"eval_root": "./eval_efficientnet"},
}

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

DEFAULT_TARGET_COMBINED_FPR = 0.06


def require(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return path


def pct(x):
    return None if x is None else 100.0 * float(x)


def extract_encoded_index(filename):
    base = os.path.basename(str(filename))
    m = re.match(r"^(\d+)(?:_|\.|$)", base)
    return None if m is None else int(m.group(1))


def check_alignment(old_labels, js_labels, js_filenames, model_key, condition):
    if len(old_labels) != len(js_labels):
        raise RuntimeError(
            f"[{model_key}/{condition}] old/js length mismatch."
        )

    if not np.array_equal(old_labels, js_labels):
        raise RuntimeError(
            f"[{model_key}/{condition}] old detector labels and JS labels differ."
        )

    parsed = [extract_encoded_index(x) for x in js_filenames]

    if all(x is not None for x in parsed):
        encoded = np.asarray(parsed, dtype=np.int64)
        expected = np.arange(len(encoded), dtype=np.int64)

        if not np.array_equal(encoded, expected):
            bad = np.flatnonzero(encoded != expected)
            first = int(bad[0]) if len(bad) else -1
            raise RuntimeError(
                f"[{model_key}/{condition}] JS filenames are not "
                f"global-index aligned. First mismatch: {first}."
            )


def load_eval_indices(model_key):
    return np.load(
        require(
            os.path.join(
                JS_ANALYSIS_ROOT,
                model_key,
                "evaluation_indices.npy",
            )
        )
    ).astype(np.int64, copy=False)


def load_condition(model_key, cfg, condition):
    old_dir = os.path.join(cfg["eval_root"], condition)
    js_dir = os.path.join(JS_SIGNAL_ROOT, model_key, condition)

    old_mask = np.load(
        require(os.path.join(old_dir, "suspicious_mask.npy"))
    ).astype(bool, copy=False)

    labels = np.load(
        require(os.path.join(old_dir, "label.npy"))
    ).astype(np.int64, copy=False)

    preds = np.load(
        require(os.path.join(old_dir, "pred.npy"))
    ).astype(np.int64, copy=False)

    js_scores = np.load(
        require(os.path.join(js_dir, "js_divergence.npy"))
    ).astype(np.float32, copy=False)

    js_labels = np.load(
        require(os.path.join(js_dir, "labels.npy"))
    ).astype(np.int64, copy=False)

    js_filenames = np.load(
        require(os.path.join(js_dir, "filenames.npy")),
        allow_pickle=False,
    ).astype(str)

    n = len(labels)

    if not (
        len(old_mask)
        == len(preds)
        == len(js_scores)
        == len(js_labels)
        == len(js_filenames)
        == n
    ):
        raise RuntimeError(
            f"[{model_key}/{condition}] array lengths differ."
        )

    check_alignment(
        labels,
        js_labels,
        js_filenames,
        model_key,
        condition,
    )

    return {
        "old_mask": old_mask,
        "labels": labels,
        "preds": preds,
        "js_scores": js_scores,
    }


def compute_rates(mask, labels, preds):
    mask = np.asarray(mask, dtype=bool)
    wrong = preds != labels
    correct = ~wrong

    return {
        "rate": float(mask.mean()),
        "ecdr": float(mask[wrong].mean()) if np.any(wrong) else None,
        "det_correct": float(mask[correct].mean()) if np.any(correct) else None,
        "n_wrong": int(wrong.sum()),
        "n_correct": int(correct.sum()),
    }


def choose_js_threshold_for_combined_fpr(
    old_clean_mask,
    clean_js_scores,
    target_combined_fpr,
):
    old_clean_mask = np.asarray(old_clean_mask, dtype=bool)
    clean_js_scores = np.asarray(clean_js_scores, dtype=np.float64)

    old_fpr = float(old_clean_mask.mean())

    if old_fpr > target_combined_fpr:
        return {
            "threshold": float("inf"),
            "old_clean_fpr": old_fpr,
            "js_clean_fpr": 0.0,
            "combined_clean_fpr": old_fpr,
            "status": "old_detector_already_above_target; JS disabled",
        }

    unflagged_scores = clean_js_scores[~old_clean_mask]
    n_total = len(old_clean_mask)
    old_flagged = int(old_clean_mask.sum())
    max_total_flags = int(np.floor(target_combined_fpr * n_total))
    additional_budget = max_total_flags - old_flagged

    if additional_budget <= 0:
        threshold = float("inf")
    elif additional_budget >= len(unflagged_scores):
        threshold = float("-inf")
    else:
        unique_desc = np.unique(unflagged_scores)[::-1]

        threshold = float("inf")
        for candidate in unique_desc:
            js_mask = clean_js_scores > candidate
            combined = old_clean_mask | js_mask
            if combined.mean() <= target_combined_fpr:
                threshold = float(candidate)
                break

        if threshold == float("inf"):
            threshold = float(np.max(unflagged_scores))

    js_mask = clean_js_scores > threshold
    combined_mask = old_clean_mask | js_mask

    return {
        "threshold": float(threshold),
        "old_clean_fpr": float(old_clean_mask.mean()),
        "js_clean_fpr": float(js_mask.mean()),
        "combined_clean_fpr": float(combined_mask.mean()),
        "status": "ok",
    }


def analyze_condition(
    model_key,
    condition,
    data,
    eval_idx,
    js_threshold,
):
    labels_e = data["labels"][eval_idx]
    preds_e = data["preds"][eval_idx]
    old_e = data["old_mask"][eval_idx]
    js_scores_e = data["js_scores"][eval_idx]

    js_e = js_scores_e > js_threshold
    or_e = old_e | js_e
    and_e = old_e & js_e

    old_only_e = old_e & ~js_e
    js_only_e = js_e & ~old_e
    both_e = old_e & js_e
    neither_e = ~old_e & ~js_e

    metrics = {
        "old": compute_rates(old_e, labels_e, preds_e),
        "js": compute_rates(js_e, labels_e, preds_e),
        "or": compute_rates(or_e, labels_e, preds_e),
        "and": compute_rates(and_e, labels_e, preds_e),
    }

    overlap = {
        "old_only": int(old_only_e.sum()),
        "js_only": int(js_only_e.sum()),
        "both": int(both_e.sum()),
        "neither": int(neither_e.sum()),
    }

    
    js_full = data["js_scores"] > js_threshold
    or_full = data["old_mask"] | js_full

    out_dir = os.path.join(OUT_ROOT, model_key, condition)
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    np.save(
        os.path.join(out_dir, "calibrated_js_mask.npy"),
        js_full,
    )
    np.save(
        os.path.join(out_dir, "calibrated_combined_or_mask.npy"),
        or_full,
    )
    np.save(
        os.path.join(out_dir, "old_mask.npy"),
        data["old_mask"],
    )

    result = {
        "model": model_key,
        "condition": condition,
        "num_total": int(len(data["labels"])),
        "num_heldout": int(len(eval_idx)),
        "js_threshold": float(js_threshold),
        "heldout_metrics": metrics,
        "heldout_overlap": overlap,
    }

    with open(
        os.path.join(out_dir, "results.json"),
        "w",
    ) as f:
        json.dump(result, f, indent=2)

    return result


def analyze_model(model_key, target_combined_fpr):
    cfg = MODELS[model_key]
    eval_idx = load_eval_indices(model_key)

    clean = load_condition(
        model_key,
        cfg,
        "clean",
    )

    clean_old_e = clean["old_mask"][eval_idx]
    clean_js_e = clean["js_scores"][eval_idx]

    calibration = choose_js_threshold_for_combined_fpr(
        old_clean_mask=clean_old_e,
        clean_js_scores=clean_js_e,
        target_combined_fpr=target_combined_fpr,
    )

    js_threshold = calibration["threshold"]

    print("\n" + "=" * 112)
    print(f"[{model_key}] JOINT CLEAN-FPR CALIBRATION")
    print("=" * 112)
    print(
        f"Target combined clean FPR : "
        f"{100.0 * target_combined_fpr:.2f}%"
    )
    print(
        f"Old clean FPR             : "
        f"{100.0 * calibration['old_clean_fpr']:.2f}%"
    )
    print(
        f"Calibrated JS clean FPR   : "
        f"{100.0 * calibration['js_clean_fpr']:.2f}%"
    )
    print(
        f"Final OR clean FPR        : "
        f"{100.0 * calibration['combined_clean_fpr']:.2f}%"
    )
    print(f"JS threshold              : {js_threshold}")
    print(f"Status                    : {calibration['status']}")

    print(
        "\nCONDITION        "
        "OLD TPR/FPR    JS TPR/FPR     OR TPR/FPR    "
        "OLD ECDR     JS ECDR      OR ECDR"
    )
    print("-" * 112)

    results = {}

    for condition in CONDITIONS:
        data = load_condition(
            model_key,
            cfg,
            condition,
        )

        result = analyze_condition(
            model_key=model_key,
            condition=condition,
            data=data,
            eval_idx=eval_idx,
            js_threshold=js_threshold,
        )

        results[condition] = result
        m = result["heldout_metrics"]

        def p(v):
            return float("nan") if v is None else 100.0 * v

        print(
            f"{condition:15s} "
            f"{p(m['old']['rate']):10.2f}% "
            f"{p(m['js']['rate']):12.2f}% "
            f"{p(m['or']['rate']):12.2f}% "
            f"{p(m['old']['ecdr']):10.2f}% "
            f"{p(m['js']['ecdr']):10.2f}% "
            f"{p(m['or']['ecdr']):10.2f}%"
        )

    model_out = os.path.join(OUT_ROOT, model_key)
    Path(model_out).mkdir(parents=True, exist_ok=True)

    model_summary = {
        "model": model_key,
        "target_combined_clean_fpr": float(target_combined_fpr),
        "calibration": calibration,
        "evaluation_indices_source": os.path.join(
            JS_ANALYSIS_ROOT,
            model_key,
            "evaluation_indices.npy",
        ),
        "results": results,
    }

    with open(
        os.path.join(
            model_out,
            "calibrated_combination_ALL_results.json",
        ),
        "w",
    ) as f:
        json.dump(model_summary, f, indent=2)

    return model_summary


def save_csv(all_results):
    path = os.path.join(
        OUT_ROOT,
        "calibrated_combined_summary.csv",
    )

    fields = [
        "model",
        "condition",
        "target_combined_clean_fpr_percent",
        "js_threshold",
        "old_rate_percent",
        "js_rate_percent",
        "combined_or_rate_percent",
        "old_ecdr_percent",
        "js_ecdr_percent",
        "combined_or_ecdr_percent",
        "old_det_correct_percent",
        "js_det_correct_percent",
        "combined_or_det_correct_percent",
        "old_only_count",
        "js_only_count",
        "both_count",
        "neither_count",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for model_key, model_summary in all_results.items():
            target = 100.0 * model_summary[
                "target_combined_clean_fpr"
            ]
            threshold = model_summary["calibration"]["threshold"]

            for condition, result in model_summary["results"].items():
                m = result["heldout_metrics"]
                o = result["heldout_overlap"]

                writer.writerow({
                    "model": model_key,
                    "condition": condition,
                    "target_combined_clean_fpr_percent": target,
                    "js_threshold": threshold,
                    "old_rate_percent": pct(m["old"]["rate"]),
                    "js_rate_percent": pct(m["js"]["rate"]),
                    "combined_or_rate_percent": pct(m["or"]["rate"]),
                    "old_ecdr_percent": pct(m["old"]["ecdr"]),
                    "js_ecdr_percent": pct(m["js"]["ecdr"]),
                    "combined_or_ecdr_percent": pct(m["or"]["ecdr"]),
                    "old_det_correct_percent": pct(m["old"]["det_correct"]),
                    "js_det_correct_percent": pct(m["js"]["det_correct"]),
                    "combined_or_det_correct_percent": pct(
                        m["or"]["det_correct"]
                    ),
                    "old_only_count": o["old_only"],
                    "js_only_count": o["js_only"],
                    "both_count": o["both"],
                    "neither_count": o["neither"],
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
        "--target_combined_fpr",
        type=float,
        default=DEFAULT_TARGET_COMBINED_FPR,
        help=(
            "Desired FINAL clean FPR for OLD OR JS. "
            "Default = 0.06 (6%%)."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not (0.0 < args.target_combined_fpr < 1.0):
        raise ValueError(
            "--target_combined_fpr must be between 0 and 1."
        )

    selected_models = (
        list(MODELS.keys())
        if args.model == "all"
        else [args.model]
    )

    print("=" * 112)
    print("MAPILLARY JOINT CALIBRATION: OLD DETECTOR + JS")
    print("=" * 112)
    print(f"Models                    : {selected_models}")
    print(
        f"Target combined clean FPR : "
        f"{100.0 * args.target_combined_fpr:.2f}%"
    )
    print(f"NEW output only           : {OUT_ROOT}")
    print("Previous results          : NOT MODIFIED")

    Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)

    all_results = {}

    for model_key in selected_models:
        all_results[model_key] = analyze_model(
            model_key,
            args.target_combined_fpr,
        )

    combined_json = os.path.join(
        OUT_ROOT,
        "calibrated_combined_results.json",
    )

    with open(combined_json, "w") as f:
        json.dump(all_results, f, indent=2)

    csv_path = save_csv(all_results)

    print("\n" + "=" * 112)
    print("JOINT CALIBRATION COMPLETE")
    print("=" * 112)
    print(f"Combined JSON -> {combined_json}")
    print(f"Combined CSV  -> {csv_path}")
    print(
        "\nNo previous detector, threshold, JS, Mahalanobis, "
        "attack, or DDPM results were modified."
    )


if __name__ == "__main__":
    main()