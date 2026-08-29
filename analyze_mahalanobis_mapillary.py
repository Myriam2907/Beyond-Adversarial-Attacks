import os
import json
import shutil
import csv
import argparse
import re
from pathlib import Path

import numpy as np

try:
    from sklearn.decomposition import PCA
    from sklearn.metrics import roc_auc_score
except ImportError as exc:
    raise ImportError(
        
    ) from exc


FEATURE_ROOT = "./feature_signal_mapillary"
OUT_ROOT = "./mahalanobis_signal_mapillary"

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

DEFAULT_PCA_DIM = 128
DEFAULT_REFERENCE_FRACTION = 0.50
DEFAULT_CLEAN_FPR = 0.05
DEFAULT_RIDGE = 1e-3
DEFAULT_SEED = 123


def condition_dir(model_key, condition):
    return os.path.join(FEATURE_ROOT, model_key, condition)


def load_condition(model_key, condition):
    root = condition_dir(model_key, condition)

    paths = {
        "features": os.path.join(root, "features.npy"),
        "labels": os.path.join(root, "labels.npy"),
        "predictions": os.path.join(root, "predictions.npy"),
        "filenames": os.path.join(root, "filenames.npy"),
    }

    missing = [p for p in paths.values() if not os.path.isfile(p)]
    if missing:
        raise FileNotFoundError(
            f"[{model_key}/{condition}] missing files:\n  "
            + "\n  ".join(missing)
        )

    features = np.load(paths["features"], mmap_mode="r")
    labels = np.load(paths["labels"], mmap_mode="r")
    preds = np.load(paths["predictions"], mmap_mode="r")
    filenames = np.load(paths["filenames"], allow_pickle=False)

    n = features.shape[0]
    if not (len(labels) == len(preds) == len(filenames) == n):
        raise RuntimeError(
            f"[{model_key}/{condition}] array length mismatch."
        )

    return {
        "features": features,
        "labels": np.asarray(labels, dtype=np.int64),
        "predictions": np.asarray(preds, dtype=np.int64),
        "filenames": np.asarray(filenames, dtype=str),
    }


def _extract_encoded_index(filename):
    
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
            f"[{model_key}/{condition}] image count differs from clean: "
            f"{n_clean} vs {n_other}."
        )

    if not np.array_equal(clean["labels"], other["labels"]):
        raise RuntimeError(
            f"[{model_key}/{condition}] labels are not aligned with clean."
        )

    if np.array_equal(clean["filenames"], other["filenames"]):
        return "exact_filenames"

    
    parsed = [_extract_encoded_index(x) for x in other["filenames"]]
    if any(x is None for x in parsed):
        raise RuntimeError(
            f"[{model_key}/{condition}] filenames differ from clean and "
            "the generated filenames do not consistently encode the original "
            "global image index. Refusing to assume alignment."
        )

    encoded = np.asarray(parsed, dtype=np.int64)
    expected = np.arange(n_other, dtype=np.int64)

    if not np.array_equal(encoded, expected):
        bad = np.flatnonzero(encoded != expected)
        first = int(bad[0]) if len(bad) else -1
        raise RuntimeError(
            f"[{model_key}/{condition}] generated filename indices are not "
            f"position-aligned. First mismatch at position {first}: "
            f"encoded={encoded[first] if first >= 0 else 'n/a'}, "
            f"expected={first}."
        )

    return "encoded_global_index"


def make_class_stratified_split(labels, reference_fraction, seed):
    rng = np.random.default_rng(seed)

    ref_indices = []
    cal_indices = []

    for cls in np.unique(labels):
        idx = np.flatnonzero(labels == cls).copy()
        rng.shuffle(idx)
        n = len(idx)

        if n == 1:
            cal_indices.extend(idx.tolist())
            continue

        n_ref = int(round(n * reference_fraction))
        n_ref = max(1, min(n - 1, n_ref))

        ref_indices.extend(idx[:n_ref].tolist())
        cal_indices.extend(idx[n_ref:].tolist())

    ref_indices = np.asarray(sorted(ref_indices), dtype=np.int64)
    cal_indices = np.asarray(sorted(cal_indices), dtype=np.int64)

    if len(ref_indices) == 0 or len(cal_indices) == 0:
        raise RuntimeError("Reference or calibration split is empty.")

    if np.intersect1d(ref_indices, cal_indices).size != 0:
        raise RuntimeError("Reference/calibration split overlaps.")

    return ref_indices, cal_indices


def fit_feature_model(
    clean_features,
    clean_labels,
    ref_idx,
    pca_dim,
    ridge,
    seed,
):
    x_ref = np.asarray(clean_features[ref_idx], dtype=np.float32)
    y_ref = clean_labels[ref_idx]

    original_dim = x_ref.shape[1]
    actual_pca_dim = min(
        int(pca_dim),
        original_dim,
        len(x_ref) - 1,
    )

    if actual_pca_dim < 2:
        raise RuntimeError("PCA dimension became < 2.")

    print(
        f"  Fitting PCA: {original_dim} -> {actual_pca_dim}"
    )

    pca = PCA(
        n_components=actual_pca_dim,
        svd_solver="randomized",
        random_state=seed,
        whiten=False,
    )

    z_ref = pca.fit_transform(x_ref).astype(np.float64, copy=False)

    global_mean = z_ref.mean(axis=0)

    class_means = {}
    for cls in np.unique(y_ref):
        mask = (y_ref == cls)
        class_means[int(cls)] = z_ref[mask].mean(axis=0)

    residuals = np.empty_like(z_ref, dtype=np.float64)
    for i, cls in enumerate(y_ref):
        residuals[i] = z_ref[i] - class_means[int(cls)]

    covariance = (
        residuals.T @ residuals
    ) / max(1, len(residuals) - 1)

    avg_var = float(np.trace(covariance) / covariance.shape[0])
    ridge_value = float(ridge) * max(avg_var, 1e-12)

    covariance = covariance + ridge_value * np.eye(
        covariance.shape[0],
        dtype=np.float64,
    )

    eigvals, eigvecs = np.linalg.eigh(covariance)

    eig_floor = max(np.max(eigvals) * 1e-10, 1e-12)
    eigvals = np.clip(eigvals, eig_floor, None)

    inv_sqrt = (
        eigvecs
        @ np.diag(1.0 / np.sqrt(eigvals))
        @ eigvecs.T
    )

    explained = float(np.sum(pca.explained_variance_ratio_))

    return {
        "pca": pca,
        "class_means": class_means,
        "global_mean": global_mean,
        "inv_sqrt": inv_sqrt,
        "pca_dim": actual_pca_dim,
        "original_dim": original_dim,
        "ridge_value": ridge_value,
        "explained_variance_ratio_sum": explained,
        "num_reference_classes": len(class_means),
    }


def score_features(features, predictions, fitted, batch_size=4096):
    n = features.shape[0]
    scores = np.empty(n, dtype=np.float32)

    pca = fitted["pca"]
    class_means = fitted["class_means"]
    global_mean = fitted["global_mean"]
    inv_sqrt = fitted["inv_sqrt"]

    fallback_count = 0

    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)

        x = np.asarray(
            features[start:end],
            dtype=np.float32,
        )

        z = pca.transform(x).astype(np.float64, copy=False)
        pred_batch = predictions[start:end]

        means = np.empty_like(z, dtype=np.float64)

        for i, cls in enumerate(pred_batch):
            mean = class_means.get(int(cls))
            if mean is None:
                mean = global_mean
                fallback_count += 1
            means[i] = mean

        residual = z - means
        whitened = residual @ inv_sqrt

        scores[start:end] = np.sqrt(
            np.sum(whitened ** 2, axis=1)
        ).astype(np.float32)

    return scores, fallback_count


def threshold_for_target_fpr(clean_scores, target_fpr):
    percentile = 100.0 * (1.0 - target_fpr)
    return float(np.percentile(clean_scores, percentile))


def detection_rate(scores, threshold):
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
    pca_dim,
    reference_fraction,
    clean_fpr,
    ridge,
    seed,
):
    print("\n" + "=" * 88)
    print(f"[{model_key}] MAHALANOBIS ANALYSIS")
    print("=" * 88)

    clean = load_condition(model_key, "clean")

    n = len(clean["labels"])

    print(f"  Clean images      : {n:,}")
    print(f"  Feature dimension : {clean['features'].shape[1]}")

    ref_idx, cal_idx = make_class_stratified_split(
        clean["labels"],
        reference_fraction,
        seed,
    )

    print(f"  Reference split   : {len(ref_idx):,}")
    print(f"  Calibration split : {len(cal_idx):,}")

    fitted = fit_feature_model(
        clean_features=clean["features"],
        clean_labels=clean["labels"],
        ref_idx=ref_idx,
        pca_dim=pca_dim,
        ridge=ridge,
        seed=seed,
    )

    print(
        f"  PCA variance kept : "
        f"{100.0 * fitted['explained_variance_ratio_sum']:.2f}%"
    )
    print(
        f"  Reference classes : "
        f"{fitted['num_reference_classes']}"
    )
    print(
        f"  Ridge value       : "
        f"{fitted['ridge_value']:.6e}"
    )

    clean_scores_all, clean_fallback = score_features(
        clean["features"],
        clean["predictions"],
        fitted,
    )

    clean_scores_cal = clean_scores_all[cal_idx]

    threshold = threshold_for_target_fpr(
        clean_scores_cal,
        clean_fpr,
    )

    measured_clean_fpr = detection_rate(
        clean_scores_cal,
        threshold,
    )

    print(f"\n  Threshold          : {threshold:.6f}")
    print(f"  Target clean FPR   : {100.0 * clean_fpr:.2f}%")
    print(
        f"  Measured clean FPR : "
        f"{100.0 * measured_clean_fpr:.2f}%"
    )

    model_out = os.path.join(OUT_ROOT, model_key)

    
    if os.path.exists(model_out):
        shutil.rmtree(model_out)

    Path(model_out).mkdir(parents=True, exist_ok=True)

    np.save(
        os.path.join(model_out, "reference_indices.npy"),
        ref_idx,
    )
    np.save(
        os.path.join(model_out, "calibration_indices.npy"),
        cal_idx,
    )
    np.save(
        os.path.join(model_out, "clean_scores_all.npy"),
        clean_scores_all,
    )
    np.save(
        os.path.join(model_out, "clean_scores_calibration.npy"),
        clean_scores_cal,
    )

    rows = []

    clean_row = {
        "model": model_key,
        "condition": "clean",
        "num_total": int(n),
        "num_evaluated": int(len(cal_idx)),
        "threshold": threshold,
        "target_clean_fpr": float(clean_fpr),
        "detection_rate": measured_clean_fpr,
        "auc_vs_clean": None,
        "score_mean": float(np.mean(clean_scores_cal)),
        "score_median": float(np.median(clean_scores_cal)),
        "score_p95": float(np.percentile(clean_scores_cal, 95)),
        "detected_among_wrong": None,
        "detected_among_correct": None,
        "num_wrong_evaluated": None,
        "num_correct_evaluated": None,
        "fallback_predicted_class_count_all": int(clean_fallback),
        "alignment_mode": "clean_reference",
    }

    rows.append(clean_row)

    print(
        "\n  CONDITION           TPR/FPR      AUC      "
        "mean score   median"
    )
    print("  " + "-" * 70)

    print(
        f"  {'clean':18s} "
        f"{100.0 * measured_clean_fpr:7.2f}%   "
        f"{'--':>6s}   "
        f"{np.mean(clean_scores_cal):10.3f}   "
        f"{np.median(clean_scores_cal):10.3f}"
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

        scores_all, fallback_count = score_features(
            other["features"],
            other["predictions"],
            fitted,
        )

        scores_eval = scores_all[cal_idx]

        tpr = detection_rate(scores_eval, threshold)
        auc = safe_auc(clean_scores_cal, scores_eval)

        wrong_mask_eval = (
            other["predictions"][cal_idx]
            != other["labels"][cal_idx]
        )
        correct_mask_eval = ~wrong_mask_eval

        det_wrong = (
            float(
                np.mean(
                    scores_eval[wrong_mask_eval] > threshold
                )
            )
            if np.any(wrong_mask_eval)
            else None
        )

        det_correct = (
            float(
                np.mean(
                    scores_eval[correct_mask_eval] > threshold
                )
            )
            if np.any(correct_mask_eval)
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
            os.path.join(
                condition_out,
                "mahalanobis_scores_all.npy",
            ),
            scores_all,
        )

        np.save(
            os.path.join(
                condition_out,
                "mahalanobis_scores_eval.npy",
            ),
            scores_eval,
        )

        suspicious_mask_eval = (
            scores_eval > threshold
        )

        suspicious_indices = (
            cal_idx[suspicious_mask_eval]
        )

        np.save(
            os.path.join(
                condition_out,
                "suspicious_indices.npy",
            ),
            suspicious_indices,
        )

        row = {
            "model": model_key,
            "condition": condition,
            "num_total": int(n),
            "num_evaluated": int(len(cal_idx)),
            "threshold": threshold,
            "target_clean_fpr": float(clean_fpr),
            "detection_rate": float(tpr),
            "auc_vs_clean": float(auc),
            "score_mean": float(np.mean(scores_eval)),
            "score_median": float(np.median(scores_eval)),
            "score_p95": float(np.percentile(scores_eval, 95)),
            "detected_among_wrong": det_wrong,
            "detected_among_correct": det_correct,
            "num_wrong_evaluated": int(np.sum(wrong_mask_eval)),
            "num_correct_evaluated": int(np.sum(correct_mask_eval)),
            "fallback_predicted_class_count_all": int(fallback_count),
            "alignment_mode": alignment_mode,
        }

        rows.append(row)

        print(
            f"  {condition:18s} "
            f"{100.0 * tpr:7.2f}%   "
            f"{auc:6.3f}   "
            f"{np.mean(scores_eval):10.3f}   "
            f"{np.median(scores_eval):10.3f}"
        )

    model_summary = {
        "model": model_key,
        "feature_root": FEATURE_ROOT,
        "output_root": model_out,
        "num_clean_images": int(n),
        "reference_fraction": float(reference_fraction),
        "num_reference": int(len(ref_idx)),
        "num_calibration": int(len(cal_idx)),
        "seed": int(seed),
        "pca_original_dim": int(fitted["original_dim"]),
        "pca_dim": int(fitted["pca_dim"]),
        "pca_explained_variance_ratio_sum": float(
            fitted["explained_variance_ratio_sum"]
        ),
        "num_reference_classes": int(
            fitted["num_reference_classes"]
        ),
        "ridge_parameter": float(ridge),
        "ridge_value": float(fitted["ridge_value"]),
        "threshold": float(threshold),
        "target_clean_fpr": float(clean_fpr),
        "measured_clean_fpr": float(measured_clean_fpr),
        "score_definition": (
            "Mahalanobis distance in clean-reference PCA space "
            "to the mean of the sample's predicted class, using "
            "a shared within-class covariance estimated from clean "
            "reference samples."
        ),
        "rows": rows,
    }

    with open(
        os.path.join(model_out, "mahalanobis_results.json"),
        "w",
    ) as f:
        json.dump(model_summary, f, indent=2)

    return model_summary


def save_combined_csv(all_results):
    path = os.path.join(
        OUT_ROOT,
        "mahalanobis_combined_summary.csv",
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
        "fallback_predicted_class_count_all",
        "alignment_mode",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
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
        "--pca_dim",
        type=int,
        default=DEFAULT_PCA_DIM,
    )

    parser.add_argument(
        "--reference_fraction",
        type=float,
        default=DEFAULT_REFERENCE_FRACTION,
    )

    parser.add_argument(
        "--clean_fpr",
        type=float,
        default=DEFAULT_CLEAN_FPR,
    )

    parser.add_argument(
        "--ridge",
        type=float,
        default=DEFAULT_RIDGE,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(FEATURE_ROOT):
        raise FileNotFoundError(
            f"Feature root not found: {FEATURE_ROOT}"
        )

    if not (0.0 < args.reference_fraction < 1.0):
        raise ValueError(
            "--reference_fraction must be between 0 and 1."
        )

    if not (0.0 < args.clean_fpr < 1.0):
        raise ValueError(
            "--clean_fpr must be between 0 and 1."
        )

    if args.pca_dim < 2:
        raise ValueError(
            "--pca_dim must be >= 2."
        )

    if args.ridge <= 0:
        raise ValueError(
            "--ridge must be > 0."
        )

    selected_models = (
        MODELS
        if args.model == "all"
        else [args.model]
    )

    print("=" * 88)
    print("MAPILLARY MAHALANOBIS FEATURE-SIGNAL ANALYSIS")
    print("=" * 88)
    print(f"Models             : {selected_models}")
    print(f"Input features     : {FEATURE_ROOT}")
    print(f"NEW output only    : {OUT_ROOT}")
    print(f"PCA dimension      : {args.pca_dim}")
    print(
        f"Reference fraction : "
        f"{args.reference_fraction}"
    )
    print(
        f"Target clean FPR   : "
        f"{100.0 * args.clean_fpr:.2f}%"
    )
    print(f"Ridge              : {args.ridge}")
    print(f"Seed               : {args.seed}")
    print("Previous results   : NOT MODIFIED")

    Path(OUT_ROOT).mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results = []

    for model_key in selected_models:
        result = analyze_model(
            model_key=model_key,
            pca_dim=args.pca_dim,
            reference_fraction=args.reference_fraction,
            clean_fpr=args.clean_fpr,
            ridge=args.ridge,
            seed=args.seed,
        )
        all_results.append(result)

    combined_json = os.path.join(
        OUT_ROOT,
        "mahalanobis_combined_results.json",
    )

    with open(combined_json, "w") as f:
        json.dump(all_results, f, indent=2)

    combined_csv = save_combined_csv(all_results)

    print("\n" + "=" * 88)
    print("MAHALANOBIS ANALYSIS COMPLETE")
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
                auc = row["auc_vs_clean"]
                print(
                    f"  {condition:14s} "
                    f"TPR = {rate:6.2f}% "
                    f"| AUC = {auc:.3f}"
                )

    print(f"\nCombined JSON -> {combined_json}")
    print(f"Combined CSV  -> {combined_csv}")
    print(
        "\nNo previous detector/attack/DDPM "
        "results were modified."
    )


if __name__ == "__main__":
    main()