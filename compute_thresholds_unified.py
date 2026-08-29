import os
import json
import shutil
import argparse
from pathlib import Path

import numpy as np


MODELS = {
    "mobilenet": {
        "clean_dir": "./eval_mobilenet/clean",
        "out_dir": "./thresholds_mobilenet",
    },
    "convnext": {
        "clean_dir": "./eval_convnext/clean",
        "out_dir": "./thresholds_convnext",
    },
    "efficientnet": {
        "clean_dir": "./eval_efficientnet/clean",
        "out_dir": "./thresholds_efficientnet",
    },
}


REQUIRED_FILES = [
    "energy.npy",
    "confidence.npy",
    "2pass_conf_drop.npy",
    "2pass_logit_l2.npy",
    "3pass_max_conf_drop_critical.npy",
    "3pass_max_logit_l2_critical.npy",
    "critical_pred_mask.npy",
]


def load_arr(clean_dir, name):
    path = os.path.join(clean_dir, name)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing required file: {path}\n"
            "Generate clean detector features first with:\n"
            "  python eval_attacked_unified_v2.py --model <model> --attacks clean"
        )

    return np.load(path, allow_pickle=False)


def finite_values(x):
   
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return x[np.isfinite(x)]


def percentile_or_none(x, percentile):
   
    x = finite_values(x)

    if x.size == 0:
        return None

    return float(np.percentile(x, percentile))


def validate_clean_arrays(arrays, clean_dir):
    
    lengths = {
        name: int(np.asarray(arr).reshape(-1).shape[0])
        for name, arr in arrays.items()
    }

    unique_lengths = set(lengths.values())

    if len(unique_lengths) != 1:
        details = "\n".join(
            f"  {name}: {length}"
            for name, length in lengths.items()
        )

        raise RuntimeError(
            f"Clean-array alignment mismatch in {clean_dir}:\n{details}"
        )

    n = next(iter(unique_lengths))

    if n == 0:
        raise RuntimeError(
            f"Clean arrays are empty in {clean_dir}"
        )

    return n


def select_valid_3pass(values, critical_mask, signal_name):
    t confidence drop can legitimately be negative.
   
    if values.shape[0] != mask.shape[0]:
        raise RuntimeError(
            f"3-pass alignment mismatch for {signal_name}: "
            f"{values.shape[0]} values vs {mask.shape[0]} mask entries."
        )

    selected = values[mask]
    selected = selected[np.isfinite(selected)]

    return selected


def validate_percentile(p_high):
    if not (50.0 < p_high < 100.0):
        raise ValueError(
            "--pct must be > 50 and < 100. "
            "Example: --pct 95 gives p95 high thresholds and p5 confidence."
        )


def clean_output_dir(out_dir, keep_old):
    if os.path.exists(out_dir) and not keep_old:
        print(f"  deleting old threshold results: {out_dir}")
        shutil.rmtree(out_dir)

    Path(out_dir).mkdir(
        parents=True,
        exist_ok=True
    )


def compute_for_model(model_key, p_high, keep_old=False):
    cfg = MODELS[model_key]
    clean_dir = cfg["clean_dir"]
    out_dir = cfg["out_dir"]

    if not os.path.isdir(clean_dir):
        raise FileNotFoundError(
            f"Clean evaluation directory missing: {clean_dir}\n"
            f"Generate it first with:\n"
            f"  python eval_attacked_unified_v2.py "
            f"--model {model_key} --attacks clean"
        )

    for filename in REQUIRED_FILES:
        path = os.path.join(clean_dir, filename)

        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing required clean feature: {path}\n"
                f"Regenerate clean detector features with:\n"
                f"  python eval_attacked_unified_v2.py "
                f"--model {model_key} --attacks clean"
            )

    clean_output_dir(
        out_dir,
        keep_old=keep_old
    )

    p_low = 100.0 - p_high

    arrays = {
        "energy": load_arr(clean_dir, "energy.npy"),
        "confidence": load_arr(clean_dir, "confidence.npy"),
        "conf_drop_2pass": load_arr(
            clean_dir,
            "2pass_conf_drop.npy"
        ),
        "logit_l2_2pass": load_arr(
            clean_dir,
            "2pass_logit_l2.npy"
        ),
        "conf_drop_3pass_raw": load_arr(
            clean_dir,
            "3pass_max_conf_drop_critical.npy"
        ),
        "logit_l2_3pass_raw": load_arr(
            clean_dir,
            "3pass_max_logit_l2_critical.npy"
        ),
        "critical_pred_mask": load_arr(
            clean_dir,
            "critical_pred_mask.npy"
        ),
    }

    n_clean = validate_clean_arrays(
        arrays,
        clean_dir
    )

    critical_mask = (
        np.asarray(
            arrays["critical_pred_mask"]
        )
        .reshape(-1)
        .astype(bool)
    )

    cd3_valid = select_valid_3pass(
        arrays["conf_drop_3pass_raw"],
        critical_mask,
        "3-pass confidence drop"
    )

    l2_3_valid = select_valid_3pass(
        arrays["logit_l2_3pass_raw"],
        critical_mask,
        "3-pass logit L2"
    )

    thresholds = {
        "calibration_source": "clean_only",
        "percentile_high": float(p_high),
        "percentile_low": float(p_low),

        
        "energy_threshold": percentile_or_none(
            arrays["energy"],
            p_high
        ),

        
        "confidence_min_threshold": percentile_or_none(
            arrays["confidence"],
            p_low
        ),

        
        "conf_drop_2pass_threshold": percentile_or_none(
            arrays["conf_drop_2pass"],
            p_high
        ),

        "logit_l2_2pass_threshold": percentile_or_none(
            arrays["logit_l2_2pass"],
            p_high
        ),

       
        "conf_drop_3pass_threshold": percentile_or_none(
            cd3_valid,
            p_high
        ),

        "logit_l2_3pass_threshold": percentile_or_none(
            l2_3_valid,
            p_high
        ),
    }

    threshold_keys = [
        "energy_threshold",
        "confidence_min_threshold",
        "conf_drop_2pass_threshold",
        "logit_l2_2pass_threshold",
        "conf_drop_3pass_threshold",
        "logit_l2_3pass_threshold",
    ]

    missing = [
        key
        for key in threshold_keys
        if thresholds[key] is None
    ]

    calibration_stats = {
        "model": model_key,
        "clean_dir": clean_dir,
        "num_clean_samples": int(n_clean),
        "num_clean_predicted_critical": int(
            critical_mask.sum()
        ),
        "fraction_clean_predicted_critical": float(
            critical_mask.mean()
        ),

        "clean_signal_means": {
            "energy": float(
                finite_values(
                    arrays["energy"]
                ).mean()
            ),
            "confidence": float(
                finite_values(
                    arrays["confidence"]
                ).mean()
            ),
            "conf_drop_2pass": float(
                finite_values(
                    arrays["conf_drop_2pass"]
                ).mean()
            ),
            "logit_l2_2pass": float(
                finite_values(
                    arrays["logit_l2_2pass"]
                ).mean()
            ),
            "conf_drop_3pass_predicted_critical": (
                float(cd3_valid.mean())
                if cd3_valid.size
                else None
            ),
            "logit_l2_3pass_predicted_critical": (
                float(l2_3_valid.mean())
                if l2_3_valid.size
                else None
            ),
        },

        "threshold_rules": {
            "energy": "suspicious_if_value_gt_threshold",
            "confidence": "suspicious_if_value_lt_threshold",
            "conf_drop_2pass": "suspicious_if_value_gt_threshold",
            "logit_l2_2pass": "suspicious_if_value_gt_threshold",
            "conf_drop_3pass": (
                "predicted-critical only; "
                "suspicious_if_value_gt_threshold"
            ),
            "logit_l2_3pass": (
                "predicted-critical only; "
                "suspicious_if_value_gt_threshold"
            ),
        },
    }

    if missing:
        warning = (
            "Could not compute threshold(s): "
            + ", ".join(missing)
            + ". This usually means there were no valid predicted-critical "
              "clean samples for the 3-pass calibration."
        )
        thresholds["_warning"] = warning
        calibration_stats["_warning"] = warning

    threshold_path = os.path.join(
        out_dir,
        f"{model_key}_anomaly_thresholds.json"
    )

    stats_path = os.path.join(
        out_dir,
        f"{model_key}_threshold_calibration_stats.json"
    )

    with open(
        threshold_path,
        "w"
    ) as f:
        json.dump(
            thresholds,
            f,
            indent=2
        )

    with open(
        stats_path,
        "w"
    ) as f:
        json.dump(
            calibration_stats,
            f,
            indent=2
        )

    print(
        f"\n[{model_key}]"
    )

    print(
        f"  clean samples                : {n_clean}"
    )

    print(
        f"  predicted-critical clean     : "
        f"{int(critical_mask.sum())} "
        f"({100.0 * critical_mask.mean():.3f}%)"
    )

    print(
        f"  percentile calibration       : "
        f"p{p_high:g} / p{p_low:g}"
    )

    print(
        f"  energy threshold             : "
        f"{thresholds['energy_threshold']}"
    )

    print(
        f"  confidence minimum           : "
        f"{thresholds['confidence_min_threshold']}"
    )

    print(
        f"  2-pass confidence-drop thr   : "
        f"{thresholds['conf_drop_2pass_threshold']}"
    )

    print(
        f"  2-pass logit-L2 threshold    : "
        f"{thresholds['logit_l2_2pass_threshold']}"
    )

    print(
        f"  3-pass confidence-drop thr   : "
        f"{thresholds['conf_drop_3pass_threshold']}"
    )

    print(
        f"  3-pass logit-L2 threshold    : "
        f"{thresholds['logit_l2_3pass_threshold']}"
    )

    if missing:
        print(
            f"  WARNING: {thresholds['_warning']}"
        )

    print(
        f"  thresholds saved -> {threshold_path}"
    )

    print(
        f"  calibration stats -> {stats_path}"
    )

    return {
        "thresholds": thresholds,
        "calibration_stats": calibration_stats,
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=list(MODELS.keys()) + ["all"],
        required=True
    )

    parser.add_argument(
        "--pct",
        type=float,
        default=95.0,
        help=(
            "High clean percentile for energy/change signals. "
            "Confidence uses the complementary low percentile. "
            "Default: 95 -> p95 high / p5 low."
        )
    )

    parser.add_argument(
        "--keep_old",
        action="store_true",
        help=(
            "Keep existing files in the model threshold directory. "
            "By default the old threshold directory is deleted first."
        )
    )

    args = parser.parse_args()

    validate_percentile(
        args.pct
    )

    p_low = 100.0 - args.pct

    print(
        "=" * 78
    )

    print(
        "CLEAN-ONLY DETECTOR THRESHOLD CALIBRATION V2"
    )

    print(
        "=" * 78
    )

    print(
        "Calibration data : CLEAN ONLY"
    )

    print(
        f"Percentiles      : p{args.pct:g} high / p{p_low:g} low"
    )

    print(
        "Attacks used     : NONE"
    )

    keys = (
        list(MODELS.keys())
        if args.model == "all"
        else [args.model]
    )

    results = {}

    for model_key in keys:
        results[model_key] = compute_for_model(
            model_key=model_key,
            p_high=args.pct,
            keep_old=args.keep_old
        )

    print(
        "\n==================== FINAL THRESHOLD SUMMARY ===================="
    )

    for model_key in keys:
        t = results[
            model_key
        ]["thresholds"]

        print(
            f"\n{model_key}:"
        )

        print(
            f"  energy > {t['energy_threshold']}"
        )

        print(
            f"  confidence < {t['confidence_min_threshold']}"
        )

        print(
            f"  2pass_conf_drop > {t['conf_drop_2pass_threshold']}"
        )

        print(
            f"  2pass_logit_l2 > {t['logit_l2_2pass_threshold']}"
        )

        print(
            f"  3pass_conf_drop > {t['conf_drop_3pass_threshold']} "
            f"(predicted-critical only)"
        )

        print(
            f"  3pass_logit_l2 > {t['logit_l2_3pass_threshold']} "
            f"(predicted-critical only)"
        )


if __name__ == "__main__":
    main()
