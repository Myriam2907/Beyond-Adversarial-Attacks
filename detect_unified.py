import os
import csv
import json
import argparse
from pathlib import Path

import numpy as np




CRITICAL_IDS = [241, 242, 243, 265]

DEFAULT_ATTACKS = [
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

MODELS = {
    "mobilenet": {
        "eval_root": "./eval_mobilenet",
        "thresholds": (
            "./thresholds_mobilenet/"
            "mobilenet_anomaly_thresholds.json"
        ),
    },
    "convnext": {
        "eval_root": "./eval_convnext",
        "thresholds": (
            "./thresholds_convnext/"
            "convnext_anomaly_thresholds.json"
        ),
    },
    "efficientnet": {
        "eval_root": "./eval_efficientnet",
        "thresholds": (
            "./thresholds_efficientnet/"
            "efficientnet_anomaly_thresholds.json"
        ),
    },
}




REQUIRED_ARRAYS = [
    "energy.npy",
    "confidence.npy",
    "pred.npy",
    "label.npy",
    "2pass_conf_drop.npy",
    "2pass_logit_l2.npy",
    "2pass_changed.npy",
    "3pass_max_conf_drop_critical.npy",
    "3pass_max_logit_l2_critical.npy",
    "3pass_changed_critical.npy",
    "critical_pred_mask.npy",
]

REQUIRED_THRESHOLD_KEYS = [
    "energy_threshold",
    "confidence_min_threshold",
    "conf_drop_2pass_threshold",
    "logit_l2_2pass_threshold",
    "conf_drop_3pass_threshold",
    "logit_l2_3pass_threshold",
]




def load_array(eval_dir, filename, required=True):
    path = os.path.join(eval_dir, filename)

    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(
                f"Missing required file: {path}\n"
                "Regenerate detector features with:\n"
                "  python eval_attacked_unified_v2.py "
                "--model <model>"
            )
        return None

    return np.load(path, allow_pickle=True)


def load_thresholds(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Threshold file missing: {path}\n"
            "Run clean-only threshold calibration first with:\n"
            "  python compute_thresholds_unified_v2.py "
            "--model <model>"
        )

    with open(path, "r") as f:
        thresholds = json.load(f)

    missing = [
        key
        for key in REQUIRED_THRESHOLD_KEYS
        if key not in thresholds
    ]

    if missing:
        raise KeyError(
            f"Threshold file {path} is missing keys: {missing}"
        )

    
    mandatory_non_null = [
        "energy_threshold",
        "confidence_min_threshold",
        "conf_drop_2pass_threshold",
        "logit_l2_2pass_threshold",
    ]

    bad = [
        key
        for key in mandatory_non_null
        if thresholds.get(key) is None
    ]

    if bad:
        raise ValueError(
            f"Threshold file {path} has null mandatory threshold(s): {bad}"
        )

    return thresholds


def validate_array_alignment(arrays, eval_dir):
    
    lengths = {}

    for name, arr in arrays.items():
        arr = np.asarray(arr)

        if arr.ndim == 0:
            raise RuntimeError(
                f"{eval_dir}/{name}: expected per-image array, got scalar."
            )

        lengths[name] = int(arr.shape[0])

    unique_lengths = set(lengths.values())

    if len(unique_lengths) != 1:
        details = "\n".join(
            f"  {name}: {length}"
            for name, length in lengths.items()
        )

        raise RuntimeError(
            f"Array alignment mismatch in {eval_dir}:\n{details}"
        )

    n = next(iter(unique_lengths))

    if n == 0:
        raise RuntimeError(
            f"No samples found in {eval_dir}"
        )

    return n


def validate_weak_k(weak_k):
   
    if not (1 <= weak_k <= 5):
        raise ValueError(
            "--weak_k must be between 1 and 5 inclusive."
        )


def remove_old_detection_outputs(eval_dir):
    
    files = [
        "anomaly_detection_results.json",
        "suspicious_indices.npy",
        "suspicious_mask.npy",
        "suspicious_files.txt",
    ]

    for filename in files:
        path = os.path.join(eval_dir, filename)

        if os.path.exists(path):
            os.remove(path)



def detect_one(
    eval_dir,
    thresholds,
    attack_name,
    weak_k
):
    
    arrays = {
        "energy": load_array(eval_dir, "energy.npy"),
        "confidence": load_array(eval_dir, "confidence.npy"),
        "pred": load_array(eval_dir, "pred.npy"),
        "label": load_array(eval_dir, "label.npy"),

        "conf_drop_2": load_array(
            eval_dir,
            "2pass_conf_drop.npy"
        ),

        "logit_l2_2": load_array(
            eval_dir,
            "2pass_logit_l2.npy"
        ),

        "changed_2": load_array(
            eval_dir,
            "2pass_changed.npy"
        ),

        "conf_drop_3": load_array(
            eval_dir,
            "3pass_max_conf_drop_critical.npy"
        ),

        "logit_l2_3": load_array(
            eval_dir,
            "3pass_max_logit_l2_critical.npy"
        ),

        "changed_3": load_array(
            eval_dir,
            "3pass_changed_critical.npy"
        ),

        "critical_pred_mask": load_array(
            eval_dir,
            "critical_pred_mask.npy"
        ),
    }

    n = validate_array_alignment(
        arrays,
        eval_dir
    )

    
    energy = np.asarray(
        arrays["energy"],
        dtype=np.float64
    ).reshape(-1)

    confidence = np.asarray(
        arrays["confidence"],
        dtype=np.float64
    ).reshape(-1)

    pred = np.asarray(
        arrays["pred"]
    ).reshape(-1)

    label = np.asarray(
        arrays["label"]
    ).reshape(-1)

    conf_drop_2 = np.asarray(
        arrays["conf_drop_2"],
        dtype=np.float64
    ).reshape(-1)

    logit_l2_2 = np.asarray(
        arrays["logit_l2_2"],
        dtype=np.float64
    ).reshape(-1)

    changed_2 = np.asarray(
        arrays["changed_2"]
    ).reshape(-1)

    conf_drop_3 = np.asarray(
        arrays["conf_drop_3"],
        dtype=np.float64
    ).reshape(-1)

    logit_l2_3 = np.asarray(
        arrays["logit_l2_3"],
        dtype=np.float64
    ).reshape(-1)

    changed_3 = np.asarray(
        arrays["changed_3"]
    ).reshape(-1)

    critical_pred_mask = np.asarray(
        arrays["critical_pred_mask"]
    ).reshape(-1).astype(bool)

    finite_required = {
        "energy": energy,
        "confidence": confidence,
        "conf_drop_2": conf_drop_2,
        "logit_l2_2": logit_l2_2,
    }

    for name, arr in finite_required.items():
        if not np.all(np.isfinite(arr)):
            count = int((~np.isfinite(arr)).sum())

            raise RuntimeError(
                f"{eval_dir}: {name} contains "
                f"{count} non-finite values."
            )

    
    for name, arr in [
        ("conf_drop_3", conf_drop_3),
        ("logit_l2_3", logit_l2_3),
    ]:
        selected = arr[critical_pred_mask]

        if selected.size and not np.all(np.isfinite(selected)):
            count = int(
                (~np.isfinite(selected)).sum()
            )

            raise RuntimeError(
                f"{eval_dir}: valid {name} contains "
                f"{count} non-finite values."
            )

    
    
    valid_3 = critical_pred_mask

    
    flag_energy = (
        energy >
        float(
            thresholds["energy_threshold"]
        )
    )

    
    flag_confidence = (
        confidence <
        float(
            thresholds["confidence_min_threshold"]
        )
    )

    
    flag_conf_drop_2 = (
        conf_drop_2 >
        float(
            thresholds["conf_drop_2pass_threshold"]
        )
    )

    
    flag_logit_l2_2 = (
        logit_l2_2 >
        float(
            thresholds["logit_l2_2pass_threshold"]
        )
    )

    
    flag_changed_2 = (
        changed_2 == 1
    )

    
    flag_conf_drop_3 = np.zeros(
        n,
        dtype=bool
    )

    flag_logit_l2_3 = np.zeros(
        n,
        dtype=bool
    )

    flag_changed_3 = np.zeros(
        n,
        dtype=bool
    )

    conf3_threshold = thresholds.get(
        "conf_drop_3pass_threshold"
    )

    l23_threshold = thresholds.get(
        "logit_l2_3pass_threshold"
    )

    if conf3_threshold is not None:
        flag_conf_drop_3[
            valid_3
        ] = (
            conf_drop_3[
                valid_3
            ] >
            float(
                conf3_threshold
            )
        )

    if l23_threshold is not None:
        flag_logit_l2_3[
            valid_3
        ] = (
            logit_l2_3[
                valid_3
            ] >
            float(
                l23_threshold
            )
        )

    flag_changed_3[
        valid_3
    ] = (
        changed_3[
            valid_3
        ] == 1
    )

    

    strong_signals = (
        flag_changed_2
        |
        flag_changed_3
        |
        flag_logit_l2_2
    )

    weak_count = (
        flag_energy.astype(np.int16)
        +
        flag_confidence.astype(np.int16)
        +
        flag_conf_drop_2.astype(np.int16)
        +
        flag_conf_drop_3.astype(np.int16)
        +
        flag_logit_l2_3.astype(np.int16)
    )

    suspicious = (
        strong_signals
        |
        (
            weak_count >= weak_k
        )
    )

    
    correct = (
        pred == label
    )

    wrong = ~correct

    n_suspicious = int(
        suspicious.sum()
    )

    suspicious_rate = (
        100.0 *
        n_suspicious /
        n
    )

    accuracy = (
        100.0 *
        correct.mean()
    )

    wrong_total = int(
        wrong.sum()
    )

    wrong_flagged = int(
        (
            suspicious &
            wrong
        ).sum()
    )

    if wrong_total > 0:
        detection_rate_wrong = (
            100.0 *
            wrong_flagged /
            wrong_total
        )
    else:
        detection_rate_wrong = None

    correct_total = int(
        correct.sum()
    )

    correct_flagged = int(
        (
            suspicious &
            correct
        ).sum()
    )

    if correct_total > 0:
        detection_rate_correct = (
            100.0 *
            correct_flagged /
            correct_total
        )
    else:
        detection_rate_correct = None

    is_clean = (
        attack_name == "clean"
    )

    
    flag_arrays = {
        "energy": flag_energy,
        "confidence": flag_confidence,
        "conf_drop_2pass": flag_conf_drop_2,
        "logit_l2_2pass": flag_logit_l2_2,
        "changed_2pass": flag_changed_2,
        "conf_drop_3pass_predicted_critical": flag_conf_drop_3,
        "logit_l2_3pass_predicted_critical": flag_logit_l2_3,
        "changed_3pass_predicted_critical": flag_changed_3,
        "any_strong_signal": strong_signals,
        f"weak_count_ge_{weak_k}": (
            weak_count >= weak_k
        ),
    }

    flags = {}

    for name, flag in flag_arrays.items():
        count = int(
            flag.sum()
        )

        flags[name] = {
            "count": count,
            "rate_percent_of_all_samples": float(
                100.0 *
                count /
                n
            ),
        }

    
    n_valid_3 = int(
        valid_3.sum()
    )

    if n_valid_3 > 0:
        third_pass_rates = {
            "conf_drop_3pass_flag_rate_percent_of_valid_3pass": float(
                100.0 *
                flag_conf_drop_3[
                    valid_3
                ].mean()
            ),

            "logit_l2_3pass_flag_rate_percent_of_valid_3pass": float(
                100.0 *
                flag_logit_l2_3[
                    valid_3
                ].mean()
            ),

            "changed_3pass_flag_rate_percent_of_valid_3pass": float(
                100.0 *
                flag_changed_3[
                    valid_3
                ].mean()
            ),
        }
    else:
        third_pass_rates = {
            "conf_drop_3pass_flag_rate_percent_of_valid_3pass": None,
            "logit_l2_3pass_flag_rate_percent_of_valid_3pass": None,
            "changed_3pass_flag_rate_percent_of_valid_3pass": None,
        }

    
    weak_vote_histogram = {
        str(k): int(
            (
                weak_count == k
            ).sum()
        )
        for k in range(6)
    }

    results = {
        "attack_name": attack_name,
        "n_samples": int(n),

        "accuracy_percent": float(
            accuracy
        ),

        "n_suspicious": int(
            n_suspicious
        ),

        "suspicious_rate_percent": float(
            suspicious_rate
        ),

        "fpr_clean_percent": (
            float(
                suspicious_rate
            )
            if is_clean
            else None
        ),

        "tpr_attack_percent": (
            None
            if is_clean
            else float(
                suspicious_rate
            )
        ),

        "wrong_total": int(
            wrong_total
        ),

        "wrong_flagged": int(
            wrong_flagged
        ),

        "detection_rate_of_wrong_percent": (
            None
            if detection_rate_wrong is None
            else float(
                detection_rate_wrong
            )
        ),

        "correct_total": int(
            correct_total
        ),

        "correct_flagged": int(
            correct_flagged
        ),

        "detection_rate_of_correct_percent": (
            None
            if detection_rate_correct is None
            else float(
                detection_rate_correct
            )
        ),

        "weak_k": int(
            weak_k
        ),

        "num_weak_signals": 5,

        "fusion_rule": (
            "any_strong_signal OR weak_count>=weak_k"
        ),

        "strong_signal_names": [
            "changed_2pass",
            "changed_3pass_predicted_critical",
            "logit_l2_2pass",
        ],

        "weak_signal_names": [
            "energy",
            "confidence",
            "conf_drop_2pass",
            "conf_drop_3pass_predicted_critical",
            "logit_l2_3pass_predicted_critical",
        ],

        "valid_3pass_samples": int(
            n_valid_3
        ),

        "valid_3pass_fraction_percent": float(
            100.0 *
            n_valid_3 /
            n
        ),

        "flags": flags,

        "third_pass_rates": third_pass_rates,

        "weak_vote_histogram": weak_vote_histogram,
    }

    
    remove_old_detection_outputs(
        eval_dir
    )

    out_json = os.path.join(
        eval_dir,
        "anomaly_detection_results.json"
    )

    with open(
        out_json,
        "w"
    ) as f:
        json.dump(
            results,
            f,
            indent=2
        )

    suspicious_indices = np.flatnonzero(
        suspicious
    ).astype(
        np.int64
    )

    np.save(
        os.path.join(
            eval_dir,
            "suspicious_indices.npy"
        ),
        suspicious_indices
    )

    np.save(
        os.path.join(
            eval_dir,
            "suspicious_mask.npy"
        ),
        suspicious.astype(
            np.uint8
        )
    )

    
    filenames_path = os.path.join(
        eval_dir,
        "filenames.npy"
    )

    if os.path.exists(
        filenames_path
    ):
        filenames = np.load(
            filenames_path,
            allow_pickle=True
        )

        if len(filenames) != n:
            raise RuntimeError(
                f"Filename alignment mismatch in {eval_dir}: "
                f"{len(filenames)} filenames vs {n} samples."
            )

        with open(
            os.path.join(
                eval_dir,
                "suspicious_files.txt"
            ),
            "w"
        ) as f:
            for path in filenames[
                suspicious_indices
            ]:
                f.write(
                    str(path) +
                    "\n"
                )

    return results




def detect_model(
    model_key,
    attacks,
    weak_k
):
    cfg = MODELS[
        model_key
    ]

    thresholds = load_thresholds(
        cfg["thresholds"]
    )

    if not os.path.isdir(
        cfg["eval_root"]
    ):
        raise FileNotFoundError(
            f"Evaluation root missing: {cfg['eval_root']}\n"
            "Generate detector features first with:\n"
            f"  python eval_attacked_unified_v2.py "
            f"--model {model_key}"
        )

    print(
        f"\n[{model_key}]"
    )

    print(
        f"  thresholds : {cfg['thresholds']}"
    )

    print(
        f"  weak_k     : {weak_k}"
    )

    print(
        "  fusion     : "
        "any strong signal OR >= weak_k weak signals"
    )

    results = {}

    for attack in attacks:
        eval_dir = os.path.join(
            cfg["eval_root"],
            attack
        )

        if not os.path.isdir(
            eval_dir
        ):
            print(
                f"  SKIP missing: {eval_dir}"
            )
            continue

        result = detect_one(
            eval_dir=eval_dir,
            thresholds=thresholds,
            attack_name=attack,
            weak_k=weak_k
        )

        results[
            attack
        ] = result

        if attack == "clean":
            tag = "FPR"
            rate = result[
                "fpr_clean_percent"
            ]
        else:
            tag = "TPR"
            rate = result[
                "tpr_attack_percent"
            ]

        det_wrong = result[
            "detection_rate_of_wrong_percent"
        ]

        det_wrong_text = (
            "N/A"
            if det_wrong is None
            else f"{det_wrong:.2f}%"
        )

        print(
            f"  {attack:13s} "
            f"acc={result['accuracy_percent']:6.2f}%  "
            f"{tag}={rate:6.2f}%  "
            f"det_of_wrong={det_wrong_text:>7s}  "
            f"valid3={result['valid_3pass_samples']}"
        )

    
    model_summary_path = os.path.join(
        cfg["eval_root"],
        "detection_ALL_results.json"
    )

    with open(
        model_summary_path,
        "w"
    ) as f:
        json.dump(
            {
                "model": model_key,
                "threshold_file": cfg["thresholds"],
                "weak_k": weak_k,
                "attacks_requested": attacks,
                "results": results,
            },
            f,
            indent=2
        )

    return results




def save_combined_summaries(
    all_results,
    model_keys,
    attacks,
    weak_k
):
    combined_json = {
        "weak_k": weak_k,
        "models": {},
    }

    for model_key in model_keys:
        combined_json[
            "models"
        ][
            model_key
        ] = all_results[
            model_key
        ]

    json_path = (
        "./detection_combined_results.json"
    )

    with open(
        json_path,
        "w"
    ) as f:
        json.dump(
            combined_json,
            f,
            indent=2
        )

    csv_path = (
        "./detection_combined_summary.csv"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:
        writer = csv.writer(
            f
        )

        writer.writerow([
            "model",
            "condition",
            "metric_type",
            "detection_rate_percent",
            "accuracy_percent",
            "detection_rate_of_wrong_percent",
            "valid_3pass_samples",
            "weak_k",
        ])

        for model_key in model_keys:
            for attack in attacks:
                result = all_results[
                    model_key
                ].get(
                    attack
                )

                if result is None:
                    continue

                if attack == "clean":
                    metric_type = "FPR"
                    rate = result[
                        "fpr_clean_percent"
                    ]
                else:
                    metric_type = "TPR"
                    rate = result[
                        "tpr_attack_percent"
                    ]

                writer.writerow([
                    model_key,
                    attack,
                    metric_type,
                    rate,
                    result[
                        "accuracy_percent"
                    ],
                    result[
                        "detection_rate_of_wrong_percent"
                    ],
                    result[
                        "valid_3pass_samples"
                    ],
                    weak_k,
                ])

    return json_path, csv_path




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=list(
            MODELS.keys()
        ) + ["all"],
        required=True
    )

    parser.add_argument(
        "--attacks",
        nargs="+",
        choices=DEFAULT_ATTACKS,
        default=DEFAULT_ATTACKS,
        help=(
            "Conditions to evaluate. "
            "Default: clean + all adversarial/environmental conditions."
        )
    )

    parser.add_argument(
        "--weak_k",
        type=int,
        default=3,
        help=(
            "Number of weak signals required when no strong signal fires. "
            "Valid range: 1..5. Default: 3."
        )
    )

    args = parser.parse_args()

    validate_weak_k(
        args.weak_k
    )

    model_keys = (
        list(
            MODELS.keys()
        )
        if args.model == "all"
        else [
            args.model
        ]
    )

    print(
        "=" * 88
    )

    print(
        "UNIFIED ATTACK/CORRUPTION DETECTION V2"
    )

    print(
        "=" * 88
    )

    print(
        f"Models     : {model_keys}"
    )

    print(
        f"Conditions : {args.attacks}"
    )

    print(
        f"weak_k     : {args.weak_k}"
    )

    print(
        "Thresholds : CLEAN-derived and frozen"
    )

    print(
        "3-pass gate: predicted-critical mask "
        "from eval_attacked_unified_v2.py"
    )

    all_results = {}

    for model_key in model_keys:
        all_results[
            model_key
        ] = detect_model(
            model_key=model_key,
            attacks=args.attacks,
            weak_k=args.weak_k
        )

   
    print(
        "\n"
        + "=" * 88
    )

    print(
        "DETECTION RATE (clean = FPR, all other conditions = TPR)"
    )

    print(
        "=" * 88
    )

    header = (
        "model        "
        + "  ".join(
            f"{attack[:10]:>10s}"
            for attack in args.attacks
        )
    )

    print(
        header
    )

    for model_key in model_keys:
        row = f"{model_key:13s}"

        for attack in args.attacks:
            result = all_results[
                model_key
            ].get(
                attack
            )

            if result is None:
                row += (
                    f"  {'--':>10s}"
                )
                continue

            if attack == "clean":
                rate = result[
                    "fpr_clean_percent"
                ]
            else:
                rate = result[
                    "tpr_attack_percent"
                ]

            row += (
                f"  {rate:10.2f}"
            )

        print(
            row
        )

    print(
        "\nclean column: FALSE POSITIVE RATE "
        "(lower is better)"
    )

    print(
        "all other columns: TRUE POSITIVE / DETECTION RATE "
        "(higher is better)"
    )

    json_path, csv_path = save_combined_summaries(
        all_results=all_results,
        model_keys=model_keys,
        attacks=args.attacks,
        weak_k=args.weak_k
    )

    print(
        f"\nCombined JSON -> {json_path}"
    )

    print(
        f"Combined CSV  -> {csv_path}"
    )


if __name__ == "__main__":
    main()
