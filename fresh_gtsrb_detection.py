
import os
import json
import csv
import argparse
from pathlib import Path

import numpy as np



ROOT = "./gtsrb_repeat"

SIGNAL_ROOT = os.path.join(
    ROOT,
    "signals",
)

OUTPUT_ROOT = os.path.join(
    ROOT,
    "fresh_detection",
)


MODELS = [
    "mobilenet",
    "convnext",
    "efficientnet",
]


CONDITIONS = [
    "clean",
    "fgsm",
    "rfgsm",
    "pgd",
    "patch",
    "random_patch",
    "gaussian_noise",
    "salt_pepper",
    "light",
    "fog",
    "motion_blur",
]



WEAK_K = 3


LOW_PERCENTILE = 5.0
HIGH_PERCENTILE = 95.0


JS_TARGET_FPR = 0.05



REQUIRED_SIGNALS = [
    "sample_id",
    "label",
    "pred",

    "confidence",
    "energy",

    "conf_drop_2",
    "logit_l2_2",
    "changed_2",

    "critical_pred_mask",

    "conf_drop_3",
    "logit_l2_3",
    "changed_3",

    "js_divergence",
]



def load_npy(
    model,
    condition,
    name,
):
    path = os.path.join(
        SIGNAL_ROOT,
        model,
        condition,
        f"{name}.npy",
    )

    if not os.path.isfile(
        path
    ):
        raise FileNotFoundError(
            f"\nMissing signal file:\n"
            f"{path}\n"
        )

    return np.load(
        path,
        allow_pickle=False,
    )


def save_npy(
    directory,
    name,
    array,
):
    np.save(
        os.path.join(
            directory,
            f"{name}.npy",
        ),
        array,
    )


def percentile(
    values,
    q,
):
    values = np.asarray(
        values
    )

    if values.size == 0:
        raise RuntimeError(
            f"Cannot calculate percentile {q}: "
            "empty array."
        )

    return float(
        np.percentile(
            values,
            q,
        )
    )


def safe_percentage(
    numerator,
    denominator,
):
    if denominator == 0:
        return float(
            "nan"
        )

    return (
        float(
            numerator
        )
        /
        float(
            denominator
        )
        *
        100.0
    )



def load_condition(
    model,
    condition,
):
    data = {}

    for signal in REQUIRED_SIGNALS:

        data[
            signal
        ] = load_npy(
            model,
            condition,
            signal,
        )

   
    data["sample_id"] = (
        data["sample_id"]
        .astype(
            np.int64
        )
    )

    data["label"] = (
        data["label"]
        .astype(
            np.int64
        )
    )

    data["pred"] = (
        data["pred"]
        .astype(
            np.int64
        )
    )

    data["confidence"] = (
        data["confidence"]
        .astype(
            np.float64
        )
    )

    data["energy"] = (
        data["energy"]
        .astype(
            np.float64
        )
    )

    data["conf_drop_2"] = (
        data["conf_drop_2"]
        .astype(
            np.float64
        )
    )

    data["logit_l2_2"] = (
        data["logit_l2_2"]
        .astype(
            np.float64
        )
    )

    data["changed_2"] = (
        data["changed_2"]
        .astype(
            bool
        )
    )

    data["critical_pred_mask"] = (
        data["critical_pred_mask"]
        .astype(
            bool
        )
    )

    data["conf_drop_3"] = (
        data["conf_drop_3"]
        .astype(
            np.float64
        )
    )

    data["logit_l2_3"] = (
        data["logit_l2_3"]
        .astype(
            np.float64
        )
    )

    data["changed_3"] = (
        data["changed_3"]
        .astype(
            np.int8
        )
    )

    data["js_divergence"] = (
        data["js_divergence"]
        .astype(
            np.float64
        )
    )

    
    lengths = {
        key: len(
            value
        )
        for key, value
        in data.items()
    }

    if len(
        set(
            lengths.values()
        )
    ) != 1:

        raise RuntimeError(
            f"{model}/{condition}: "
            f"array length mismatch:\n"
            f"{lengths}"
        )

  
    check_signals = [
        "confidence",
        "energy",
        "conf_drop_2",
        "logit_l2_2",
        "js_divergence",
    ]

    for name in check_signals:

        if not np.all(
            np.isfinite(
                data[name]
            )
        ):

            raise RuntimeError(
                f"{model}/{condition}: "
                f"{name} contains NaN/Inf."
            )

    return data



def check_alignment(
    clean,
    other,
    model,
    condition,
):
    if not np.array_equal(
        clean["sample_id"],
        other["sample_id"],
    ):

        raise RuntimeError(
            f"\nSample ID mismatch:\n"
            f"{model}/{condition}"
        )

    if not np.array_equal(
        clean["label"],
        other["label"],
    ):

        raise RuntimeError(
            f"\nGround-truth label mismatch:\n"
            f"{model}/{condition}"
        )



def calibrate_ours(
    clean,
):
    """
    Independent fresh calibration.

    ALL thresholds come from clean data.

    Directions:

    confidence:
        low = suspicious

    energy:
        high / less negative = suspicious

    confidence drop:
        high = suspicious

    logit L2:
        high = suspicious
    """

    critical = clean[
        "critical_pred_mask"
    ]

    if critical.sum() == 0:

        raise RuntimeError(
            "No predicted-critical clean samples."
        )

    thresholds = {
        "confidence_low":
            percentile(
                clean[
                    "confidence"
                ],
                LOW_PERCENTILE,
            ),

        "energy_high":
            percentile(
                clean[
                    "energy"
                ],
                HIGH_PERCENTILE,
            ),

        "conf_drop_2_high":
            percentile(
                clean[
                    "conf_drop_2"
                ],
                HIGH_PERCENTILE,
            ),

        "logit_l2_2_high":
            percentile(
                clean[
                    "logit_l2_2"
                ],
                HIGH_PERCENTILE,
            ),

        "conf_drop_3_high":
            percentile(
                clean[
                    "conf_drop_3"
                ][critical],
                HIGH_PERCENTILE,
            ),

        "logit_l2_3_high":
            percentile(
                clean[
                    "logit_l2_3"
                ][critical],
                HIGH_PERCENTILE,
            ),
    }

    return thresholds


def apply_ours(
    data,
    thresholds,
):
   
    s_low_conf = (
        data[
            "confidence"
        ]
        <
        thresholds[
            "confidence_low"
        ]
    )

    
    s_energy = (
        data[
            "energy"
        ]
        >
        thresholds[
            "energy_high"
        ]
    )

    
    s_drop2 = (
        data[
            "conf_drop_2"
        ]
        >
        thresholds[
            "conf_drop_2_high"
        ]
    )

    
    s_l2_2 = (
        data[
            "logit_l2_2"
        ]
        >
        thresholds[
            "logit_l2_2_high"
        ]
    )

    
    critical = data[
        "critical_pred_mask"
    ]

    s_drop3 = (
        critical
        &
        (
            data[
                "conf_drop_3"
            ]
            >
            thresholds[
                "conf_drop_3_high"
            ]
        )
    )

   
    s_l2_3 = (
        critical
        &
        (
            data[
                "logit_l2_3"
            ]
            >
            thresholds[
                "logit_l2_3_high"
            ]
        )
    )

   
    weak_votes = (
        s_low_conf.astype(
            np.int16
        )
        +
        s_energy.astype(
            np.int16
        )
        +
        s_drop2.astype(
            np.int16
        )
        +
        s_l2_2.astype(
            np.int16
        )
        +
        s_drop3.astype(
            np.int16
        )
        +
        s_l2_3.astype(
            np.int16
        )
    )

    weak_detector = (
        weak_votes
        >=
        WEAK_K
    )

  
    strong_changed_2 = (
        data[
            "changed_2"
        ]
    )

    strong_changed_3 = (
        critical
        &
        (
            data[
                "changed_3"
            ]
            > 0
        )
    )

    strong_detector = (
        strong_changed_2
        |
        strong_changed_3
    )

    
    ours_mask = (
        weak_detector
        |
        strong_detector
    )

    components = {
        "low_confidence":
            s_low_conf,

        "high_energy":
            s_energy,

        "high_conf_drop_2":
            s_drop2,

        "high_logit_l2_2":
            s_l2_2,

        "high_conf_drop_3":
            s_drop3,

        "high_logit_l2_3":
            s_l2_3,

        "changed_2":
            strong_changed_2,

        "changed_3":
            strong_changed_3,

        "weak_votes":
            weak_votes,

        "weak_detector":
            weak_detector,

        "strong_detector":
            strong_detector,
    }

    return (
        ours_mask.astype(
            bool
        ),
        components,
    )



def calibrate_js(
    clean,
):
    """
    JS alone is independently calibrated.

    JS_TARGET_FPR = 0.05 means the clean JS threshold is
    approximately the 95th percentile.

    No Ours mask is involved here.
    """

    percentile_level = (
        100.0
        *
        (
            1.0
            -
            JS_TARGET_FPR
        )
    )

    threshold = percentile(
        clean[
            "js_divergence"
        ],
        percentile_level,
    )

    return threshold



def apply_js(
    data,
    threshold,
):
    return (
        data[
            "js_divergence"
        ]
        >
        threshold
    )



def evaluate_mask(
    data,
    mask,
    condition,
):
    labels = data[
        "label"
    ]

    pred = data[
        "pred"
    ]

    mask = np.asarray(
        mask,
        dtype=bool,
    )

    correct = (
        pred
        ==
        labels
    )

    wrong = (
        ~correct
    )

    n = len(
        labels
    )

    num_correct = int(
        correct.sum()
    )

    num_wrong = int(
        wrong.sum()
    )

    num_detected = int(
        mask.sum()
    )

    detected_wrong = int(
        (
            mask
            &
            wrong
        ).sum()
    )

    accuracy = (
        100.0
        *
        num_correct
        /
        n
    )

    detection_rate = (
        100.0
        *
        num_detected
        /
        n
    )

    ecdr = safe_percentage(
        detected_wrong,
        num_wrong,
    )

    result = {
        "n":
            int(
                n
            ),

        "accuracy_percent":
            float(
                accuracy
            ),

        "num_correct":
            num_correct,

        "num_wrong":
            num_wrong,

        "num_detected":
            num_detected,

        "num_detected_wrong":
            detected_wrong,

        "ecdr_percent":
            float(
                ecdr
            ),
    }

    if condition == "clean":

        result[
            "fpr_percent"
        ] = float(
            detection_rate
        )

        result[
            "tpr_percent"
        ] = None

    else:

        result[
            "fpr_percent"
        ] = None

        result[
            "tpr_percent"
        ] = float(
            detection_rate
        )

    return result



def save_components(
    output_dir,
    components,
):
    component_dir = os.path.join(
        output_dir,
        "ours_components",
    )

    Path(
        component_dir
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    for name, values in components.items():

        save_npy(
            component_dir,
            name,
            values,
        )



def evaluate_model(
    model,
):
    print(
        "\n"
        + "#" * 130
    )

    print(
        f"MODEL: {model.upper()}"
    )

    print(
        "#" * 130
    )

  
    clean = load_condition(
        model,
        "clean",
    )

    
    ours_thresholds = calibrate_ours(
        clean
    )

   
    js_threshold = calibrate_js(
        clean
    )

    
    model_output = os.path.join(
        OUTPUT_ROOT,
        model,
    )

    Path(
        model_output
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    thresholds = {
        "model":
            model,

        "source":
            "fresh clean-only calibration",

        "ours": {
            **ours_thresholds,

            "weak_k":
                int(
                    WEAK_K
                ),

            "low_percentile":
                float(
                    LOW_PERCENTILE
                ),

            "high_percentile":
                float(
                    HIGH_PERCENTILE
                ),
        },

        "js": {
            "threshold":
                float(
                    js_threshold
                ),

            "target_clean_fpr":
                float(
                    JS_TARGET_FPR
                ),

            "percentile":
                float(
                    100.0
                    *
                    (
                        1
                        -
                        JS_TARGET_FPR
                    )
                ),
        },
    }

    with open(
        os.path.join(
            model_output,
            "thresholds.json",
        ),
        "w",
    ) as f:

        json.dump(
            thresholds,
            f,
            indent=2,
        )

   
    print(
        "\nOURS CLEAN THRESHOLDS"
    )

    print(
        "-" * 80
    )

    for name, value in (
        ours_thresholds.items()
    ):

        print(
            f"{name:<30}: "
            f"{value:.10f}"
        )

    print(
        f"{'weak_k':<30}: "
        f"{WEAK_K}"
    )

    print(
        "\nJS CLEAN THRESHOLD"
    )

    print(
        "-" * 80
    )

    print(
        f"{'JS threshold':<30}: "
        f"{js_threshold:.12f}"
    )

   
    rows = []

    for condition in CONDITIONS:

        print(
            "\n"
            + "=" * 110
        )

        print(
            f"{model.upper()} | "
            f"{condition}"
        )

        print(
            "=" * 110
        )

        data = load_condition(
            model,
            condition,
        )

        check_alignment(
            clean,
            data,
            model,
            condition,
        )

        
        ours_mask, components = (
            apply_ours(
                data,
                ours_thresholds,
            )
        )

      
        js_mask = apply_js(
            data,
            js_threshold,
        )

        
        ours_js_mask = (
            ours_mask
            |
            js_mask
        )

        
        ours_metrics = evaluate_mask(
            data,
            ours_mask,
            condition,
        )

        js_metrics = evaluate_mask(
            data,
            js_mask,
            condition,
        )

        combined_metrics = evaluate_mask(
            data,
            ours_js_mask,
            condition,
        )

        
        condition_output = os.path.join(
            model_output,
            condition,
        )

        Path(
            condition_output
        ).mkdir(
            parents=True,
            exist_ok=True,
        )

        
        save_npy(
            condition_output,
            "sample_id",
            data[
                "sample_id"
            ],
        )

        save_npy(
            condition_output,
            "label",
            data[
                "label"
            ],
        )

        save_npy(
            condition_output,
            "pred",
            data[
                "pred"
            ],
        )

       
        save_npy(
            condition_output,
            "ours_mask",
            ours_mask,
        )

        save_npy(
            condition_output,
            "js_mask",
            js_mask,
        )

        save_npy(
            condition_output,
            "ours_js_mask",
            ours_js_mask,
        )

       
        js_added_mask = (
            js_mask
            &
            ~ours_mask
        )

        save_npy(
            condition_output,
            "js_added_over_ours_mask",
            js_added_mask,
        )

       
        save_components(
            condition_output,
            components,
        )

       
        condition_metrics = {
            "model":
                model,

            "condition":
                condition,

            "ours":
                ours_metrics,

            "js":
                js_metrics,

            "ours_js":
                combined_metrics,

            "js_added_over_ours":
                int(
                    js_added_mask.sum()
                ),
        }

        with open(
            os.path.join(
                condition_output,
                "metrics.json",
            ),
            "w",
        ) as f:

            json.dump(
                condition_metrics,
                f,
                indent=2,
            )

      
        acc = ours_metrics[
            "accuracy_percent"
        ]

        print(
            f"Accuracy      : "
            f"{acc:.3f}%"
        )

        if condition == "clean":

            print(
                f"Ours FPR      : "
                f"{ours_metrics['fpr_percent']:.3f}%"
            )

            print(
                f"JS FPR        : "
                f"{js_metrics['fpr_percent']:.3f}%"
            )

            print(
                f"Ours+JS FPR   : "
                f"{combined_metrics['fpr_percent']:.3f}%"
            )

        else:

            print(
                f"Ours TPR      : "
                f"{ours_metrics['tpr_percent']:.3f}%"
            )

            print(
                f"JS TPR        : "
                f"{js_metrics['tpr_percent']:.3f}%"
            )

            print(
                f"Ours+JS TPR   : "
                f"{combined_metrics['tpr_percent']:.3f}%"
            )

            print(
                f"Ours ECDR     : "
                f"{ours_metrics['ecdr_percent']:.3f}%"
            )

            print(
                f"JS ECDR       : "
                f"{js_metrics['ecdr_percent']:.3f}%"
            )

            print(
                f"Ours+JS ECDR  : "
                f"{combined_metrics['ecdr_percent']:.3f}%"
            )

       
        row = {
            "model":
                model,

            "condition":
                condition,

            "accuracy_percent":
                float(
                    acc
                ),

            "ours_fpr_percent":
                (
                    ours_metrics[
                        "fpr_percent"
                    ]
                    if condition
                    ==
                    "clean"
                    else ""
                ),

            "js_fpr_percent":
                (
                    js_metrics[
                        "fpr_percent"
                    ]
                    if condition
                    ==
                    "clean"
                    else ""
                ),

            "ours_js_fpr_percent":
                (
                    combined_metrics[
                        "fpr_percent"
                    ]
                    if condition
                    ==
                    "clean"
                    else ""
                ),

            "ours_tpr_percent":
                (
                    ""
                    if condition
                    ==
                    "clean"
                    else ours_metrics[
                        "tpr_percent"
                    ]
                ),

            "js_tpr_percent":
                (
                    ""
                    if condition
                    ==
                    "clean"
                    else js_metrics[
                        "tpr_percent"
                    ]
                ),

            "ours_js_tpr_percent":
                (
                    ""
                    if condition
                    ==
                    "clean"
                    else combined_metrics[
                        "tpr_percent"
                    ]
                ),

            "ours_ecdr_percent":
                (
                    ""
                    if condition
                    ==
                    "clean"
                    else ours_metrics[
                        "ecdr_percent"
                    ]
                ),

            "js_ecdr_percent":
                (
                    ""
                    if condition
                    ==
                    "clean"
                    else js_metrics[
                        "ecdr_percent"
                    ]
                ),

            "ours_js_ecdr_percent":
                (
                    ""
                    if condition
                    ==
                    "clean"
                    else combined_metrics[
                        "ecdr_percent"
                    ]
                ),

            "js_added_over_ours":
                int(
                    js_added_mask.sum()
                ),
        }

        rows.append(
            row
        )

    return rows



def validate_all_inputs(
    selected_models,
):
    print(
        "\nChecking extracted signals..."
    )

    missing = []

    for model in selected_models:

        for condition in CONDITIONS:

            for signal in REQUIRED_SIGNALS:

                path = os.path.join(
                    SIGNAL_ROOT,
                    model,
                    condition,
                    f"{signal}.npy",
                )

                if not os.path.isfile(
                    path
                ):

                    missing.append(
                        path
                    )

    if missing:

        print(
            "\nMissing files:"
        )

        for path in missing[
            :40
        ]:

            print(
                f"  {path}"
            )

        if len(
            missing
        ) > 40:

            print(
                f"... and "
                f"{len(missing) - 40} more."
            )

        raise RuntimeError(
            "Signal extraction is incomplete."
        )

    print(
        "✅ All required signals exist."
    )



def write_csv(
    filename,
    rows,
):
    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with open(
        filename,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )


def create_paper_rows(
    rows,
):
    """
    One clean row plus three detector rows for each attack.
    """

    output = []

    for row in rows:

        model = row[
            "model"
        ]

        condition = row[
            "condition"
        ]

        accuracy = row[
            "accuracy_percent"
        ]

        if condition == "clean":

            output.append(
                {
                    "Model":
                        model,

                    "Condition":
                        "clean",

                    "Accuracy":
                        accuracy,

                    "Method":
                        "Ours",

                    "FPR":
                        row[
                            "ours_fpr_percent"
                        ],

                    "TPR":
                        "",

                    "ECDR":
                        "",
                }
            )

            output.append(
                {
                    "Model":
                        model,

                    "Condition":
                        "clean",

                    "Accuracy":
                        accuracy,

                    "Method":
                        "JS",

                    "FPR":
                        row[
                            "js_fpr_percent"
                        ],

                    "TPR":
                        "",

                    "ECDR":
                        "",
                }
            )

            output.append(
                {
                    "Model":
                        model,

                    "Condition":
                        "clean",

                    "Accuracy":
                        accuracy,

                    "Method":
                        "Ours+JS",

                    "FPR":
                        row[
                            "ours_js_fpr_percent"
                        ],

                    "TPR":
                        "",

                    "ECDR":
                        "",
                }
            )

        else:

            output.append(
                {
                    "Model":
                        model,

                    "Condition":
                        condition,

                    "Accuracy":
                        accuracy,

                    "Method":
                        "Ours",

                    "FPR":
                        "",

                    "TPR":
                        row[
                            "ours_tpr_percent"
                        ],

                    "ECDR":
                        row[
                            "ours_ecdr_percent"
                        ],
                }
            )

            output.append(
                {
                    "Model":
                        model,

                    "Condition":
                        condition,

                    "Accuracy":
                        accuracy,

                    "Method":
                        "JS",

                    "FPR":
                        "",

                    "TPR":
                        row[
                            "js_tpr_percent"
                        ],

                    "ECDR":
                        row[
                            "js_ecdr_percent"
                        ],
                }
            )

            output.append(
                {
                    "Model":
                        model,

                    "Condition":
                        condition,

                    "Accuracy":
                        accuracy,

                    "Method":
                        "Ours+JS",

                    "FPR":
                        "",

                    "TPR":
                        row[
                            "ours_js_tpr_percent"
                        ],

                    "ECDR":
                        row[
                            "ours_js_ecdr_percent"
                        ],
                }
            )

    return output



def print_summary(
    rows,
):
    print(
        "\n"
        + "=" * 170
    )

    print(
        "FRESH GTSRB DETECTION SUMMARY"
    )

    print(
        "=" * 170
    )

    print(
        f"{'Model':<14}"
        f"{'Condition':<18}"
        f"{'Acc':>9}"
        f"{'Ours':>10}"
        f"{'JS':>10}"
        f"{'Ours+JS':>12}"
        f"{'Ours ECDR':>13}"
        f"{'JS ECDR':>12}"
        f"{'O+JS ECDR':>14}"
    )

    print(
        "-" * 170
    )

    for row in rows:

        if row[
            "condition"
        ] == "clean":

            print(
                f"{row['model']:<14}"
                f"{row['condition']:<18}"
                f"{row['accuracy_percent']:>8.2f}%"
                f"{row['ours_fpr_percent']:>9.2f}%"
                f"{row['js_fpr_percent']:>9.2f}%"
                f"{row['ours_js_fpr_percent']:>11.2f}%"
                f"{'-':>13}"
                f"{'-':>12}"
                f"{'-':>14}"
            )

        else:

            print(
                f"{row['model']:<14}"
                f"{row['condition']:<18}"
                f"{row['accuracy_percent']:>8.2f}%"
                f"{row['ours_tpr_percent']:>9.2f}%"
                f"{row['js_tpr_percent']:>9.2f}%"
                f"{row['ours_js_tpr_percent']:>11.2f}%"
                f"{row['ours_ecdr_percent']:>12.2f}%"
                f"{row['js_ecdr_percent']:>11.2f}%"
                f"{row['ours_js_ecdr_percent']:>13.2f}%"
            )



def main():

    parser = argparse.ArgumentParser(
        description=(
            "Fresh independent evaluation of "
            "OURS, JS and OURS+JS."
        )
    )

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

    args = parser.parse_args()

    if args.model == "all":

        selected_models = list(
            MODELS
        )

    else:

        selected_models = [
            args.model
        ]

    Path(
        OUTPUT_ROOT
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "=" * 130
    )

    print(
        "FRESH GTSRB — OURS / JS / OURS+JS"
    )

    print(
        "=" * 130
    )

    print(
        f"Models              : "
        f"{selected_models}"
    )

    print(
        f"Conditions          : "
        f"{CONDITIONS}"
    )

    print(
        f"Signals             : "
        f"{SIGNAL_ROOT}"
    )

    print(
        f"Output              : "
        f"{OUTPUT_ROOT}"
    )

    print(
        f"Ours low percentile : "
        f"{LOW_PERCENTILE}"
    )

    print(
        f"Ours high percentile: "
        f"{HIGH_PERCENTILE}"
    )

    print(
        f"Ours weak K         : "
        f"{WEAK_K}"
    )

    print(
        f"JS clean FPR target : "
        f"{JS_TARGET_FPR * 100:.2f}%"
    )

  
    validate_all_inputs(
        selected_models
    )

    all_rows = []

    for model in selected_models:

        model_rows = evaluate_model(
            model
        )

        all_rows.extend(
            model_rows
        )

    
    full_csv = os.path.join(
        OUTPUT_ROOT,
        "full_results.csv",
    )

    write_csv(
        full_csv,
        all_rows,
    )

    
    with open(
        os.path.join(
            OUTPUT_ROOT,
            "full_results.json",
        ),
        "w",
    ) as f:

        json.dump(
            all_rows,
            f,
            indent=2,
        )

   
    paper_rows = create_paper_rows(
        all_rows
    )

    paper_csv = os.path.join(
        OUTPUT_ROOT,
        "paper_results.csv",
    )

    write_csv(
        paper_csv,
        paper_rows,
    )

    
    print_summary(
        all_rows
    )

    print(
        "\n"
        + "=" * 130
    )

    print(
        "DONE"
    )

    print(
        "=" * 130
    )

    print(
        "\nFull results:"
    )

    print(
        f"  {full_csv}"
    )

    print(
        "\nPaper-format results:"
    )

    print(
        f"  {paper_csv}"
    )

    print(
        "\nMasks:"
    )

    print(
        "  ./gtsrb_repeat/fresh_detection/"
        "<model>/<condition>/ours_mask.npy"
    )

    print(
        "  ./gtsrb_repeat/fresh_detection/"
        "<model>/<condition>/js_mask.npy"
    )

    print(
        "  ./gtsrb_repeat/fresh_detection/"
        "<model>/<condition>/ours_js_mask.npy"
    )


if __name__ == "__main__":
    main()