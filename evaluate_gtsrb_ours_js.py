

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
    "ours_js_final",
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


DEFAULT_WEAK_K = 3


DEFAULT_HIGH_PERCENTILE = 95.0
DEFAULT_LOW_PERCENTILE = 5.0


DEFAULT_COMBINED_CLEAN_FPR = 0.06




def load_array(
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

    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"\nMissing signal file:\n"
            f"{path}\n\n"
            "Make sure signal extraction completed successfully."
        )

    return np.load(
        path,
        allow_pickle=False,
    )


def save_array(
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
    x,
    q,
):
    x = np.asarray(
        x
    )

    if x.size == 0:
        raise RuntimeError(
            f"Cannot calculate percentile {q}: empty array."
        )

    return float(
        np.percentile(
            x,
            q,
        )
    )


def safe_rate(
    numerator,
    denominator,
):
    if denominator == 0:
        return float("nan")

    return float(
        numerator
        / denominator
    )




def load_signals(
    model,
    condition,
):
    """
    Load all signals needed by detector evaluation.
    """

    data = {
        "sample_id": load_array(
            model,
            condition,
            "sample_id",
        ).astype(
            np.int64
        ),

        "label": load_array(
            model,
            condition,
            "label",
        ).astype(
            np.int64
        ),

        "pred": load_array(
            model,
            condition,
            "pred",
        ).astype(
            np.int64
        ),

        "confidence": load_array(
            model,
            condition,
            "confidence",
        ).astype(
            np.float64
        ),

        "energy": load_array(
            model,
            condition,
            "energy",
        ).astype(
            np.float64
        ),

        "conf_drop_2": load_array(
            model,
            condition,
            "conf_drop_2",
        ).astype(
            np.float64
        ),

        "logit_l2_2": load_array(
            model,
            condition,
            "logit_l2_2",
        ).astype(
            np.float64
        ),

        "changed_2": load_array(
            model,
            condition,
            "changed_2",
        ).astype(
            bool
        ),

        "critical_pred_mask": load_array(
            model,
            condition,
            "critical_pred_mask",
        ).astype(
            bool
        ),

        "conf_drop_3": load_array(
            model,
            condition,
            "conf_drop_3",
        ).astype(
            np.float64
        ),

        "logit_l2_3": load_array(
            model,
            condition,
            "logit_l2_3",
        ).astype(
            np.float64
        ),

        "changed_3": load_array(
            model,
            condition,
            "changed_3",
        ).astype(
            np.int8
        ),

        "js": load_array(
            model,
            condition,
            "js_divergence",
        ).astype(
            np.float64
        ),
    }

   

    lengths = {
        key: len(
            value
        )
        for key, value
        in data.items()
    }

    unique_lengths = set(
        lengths.values()
    )

    if len(
        unique_lengths
    ) != 1:
        raise RuntimeError(
            f"{model}/{condition}: "
            f"signal length mismatch:\n"
            f"{lengths}"
        )

  

    numeric_check = [
        "confidence",
        "energy",
        "conf_drop_2",
        "logit_l2_2",
        "js",
    ]

    for key in numeric_check:

        if not np.all(
            np.isfinite(
                data[key]
            )
        ):
            raise RuntimeError(
                f"{model}/{condition}: "
                f"{key} contains NaN or Inf."
            )

    
    valid_critical = data[
        "critical_pred_mask"
    ]

    for key in [
        "conf_drop_3",
        "logit_l2_3",
    ]:

        values = data[
            key
        ][
            valid_critical
        ]

        if (
            values.size > 0
            and not np.all(
                np.isfinite(
                    values
                )
            )
        ):
            raise RuntimeError(
                f"{model}/{condition}: "
                f"valid {key} contains NaN/Inf."
            )

    return data




def validate_alignment(
    clean,
    attack,
    model,
    condition,
):
    """
    All generated conditions should correspond to exactly the
    same original test samples in the same order.
    """

    if not np.array_equal(
        clean["sample_id"],
        attack["sample_id"],
    ):
        raise RuntimeError(
            f"\nSample-ID mismatch:\n"
            f"{model}/{condition}\n\n"
            "Do not evaluate detection until alignment is fixed."
        )

    if not np.array_equal(
        clean["label"],
        attack["label"],
    ):
        raise RuntimeError(
            f"\nGround-truth label mismatch:\n"
            f"{model}/{condition}\n\n"
            "Do not evaluate detection until alignment is fixed."
        )



def calibrate_ours(
    clean,
    high_percentile=95.0,
    low_percentile=5.0,
):
    """
    Threshold directions:

    confidence:
        LOW confidence is suspicious -> p5

    energy:
        energy = -logsumexp(logits)
        Larger / less-negative values are suspicious -> p95

    confidence drops:
        Larger instability -> p95

    logit L2:
        Larger instability -> p95

    Critical third-pass thresholds are computed only on clean
    samples for which the BASE classifier predicted a critical
    class.
    """

    critical = clean[
        "critical_pred_mask"
    ]

    if int(
        critical.sum()
    ) == 0:
        raise RuntimeError(
            "No clean predicted-critical samples available "
            "for third-pass threshold calibration."
        )

    thresholds = {
        "confidence_low": percentile(
            clean[
                "confidence"
            ],
            low_percentile,
        ),

        "energy_high": percentile(
            clean[
                "energy"
            ],
            high_percentile,
        ),

        "conf_drop_2_high": percentile(
            clean[
                "conf_drop_2"
            ],
            high_percentile,
        ),

        "logit_l2_2_high": percentile(
            clean[
                "logit_l2_2"
            ],
            high_percentile,
        ),

        "conf_drop_3_high": percentile(
            clean[
                "conf_drop_3"
            ][
                critical
            ],
            high_percentile,
        ),

        "logit_l2_3_high": percentile(
            clean[
                "logit_l2_3"
            ][
                critical
            ],
            high_percentile,
        ),
    }

    return thresholds




def apply_ours(
    data,
    thresholds,
    weak_k,
):
    """
    Ours = strong OR >= weak_k weak anomaly signals.

    STRONG:
        - prediction changed in second pass
        - critical prediction changed in third-pass test

    WEAK:
        - low confidence
        - high energy
        - high 2-pass confidence drop
        - high 2-pass logit L2
        - high critical 3-pass confidence drop
        - high critical 3-pass logit L2

    Third-pass weak signals count only when base prediction is
    one of the critical classes.
    """

   

    low_confidence = (
        data[
            "confidence"
        ]
        <
        thresholds[
            "confidence_low"
        ]
    )

    high_energy = (
        data[
            "energy"
        ]
        >
        thresholds[
            "energy_high"
        ]
    )

    high_drop_2 = (
        data[
            "conf_drop_2"
        ]
        >
        thresholds[
            "conf_drop_2_high"
        ]
    )

    high_l2_2 = (
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

    high_drop_3 = (
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

    high_l2_3 = (
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
        low_confidence.astype(
            np.int16
        )
        +
        high_energy.astype(
            np.int16
        )
        +
        high_drop_2.astype(
            np.int16
        )
        +
        high_l2_2.astype(
            np.int16
        )
        +
        high_drop_3.astype(
            np.int16
        )
        +
        high_l2_3.astype(
            np.int16
        )
    )

   

    changed_2 = data[
        "changed_2"
    ]

    changed_3 = (
        critical
        &
        (
            data[
                "changed_3"
            ]
            > 0
        )
    )

    strong_signal = (
        changed_2
        |
        changed_3
    )

  

    mask = (
        strong_signal
        |
        (
            weak_votes
            >= weak_k
        )
    )

    components = {
        "low_confidence": (
            low_confidence
        ),

        "high_energy": (
            high_energy
        ),

        "high_conf_drop_2": (
            high_drop_2
        ),

        "high_logit_l2_2": (
            high_l2_2
        ),

        "high_conf_drop_3": (
            high_drop_3
        ),

        "high_logit_l2_3": (
            high_l2_3
        ),

        "changed_2": (
            changed_2
        ),

        "changed_3": (
            changed_3
        ),

        "strong_signal": (
            strong_signal
        ),

        "weak_votes": (
            weak_votes
        ),
    }

    return (
        mask.astype(
            bool
        ),
        components,
    )



def calibrate_js_for_union(
    clean_js,
    ours_clean_mask,
    target_combined_fpr,
):
    """
    Calibrate JS using CLEAN data only.

    Key idea:
    Ours+JS = Ours OR JS.

    Therefore only JS flags on clean samples NOT ALREADY rejected
    by Ours can increase combined clean FPR.

    We choose a JS threshold so that the combined clean FPR
    does not exceed the requested target whenever possible.

    If Ours already exceeds the target, JS receives zero new
    clean-FP budget. In that case the JS threshold is placed at
    the maximum JS value among residual clean samples, so using
    "js > threshold" introduces no additional clean false positives.

    This does NOT tune using any attacks.
    """

    clean_js = np.asarray(
        clean_js,
        dtype=np.float64,
    )

    ours_clean_mask = np.asarray(
        ours_clean_mask,
        dtype=bool,
    )

    n = len(
        clean_js
    )

    ours_count = int(
        ours_clean_mask.sum()
    )

    ours_fpr = (
        ours_count
        / n
    )

    
    requested_total = int(
        np.floor(
            target_combined_fpr
            * n
            + 1e-12
        )
    )

   
    allowed_total = max(
        requested_total,
        ours_count,
    )

    additional_budget = (
        allowed_total
        - ours_count
    )

    residual_mask = (
        ~ours_clean_mask
    )

    residual_js = clean_js[
        residual_mask
    ]

    if residual_js.size == 0:

        threshold = float(
            np.inf
        )

        return {
            "threshold": (
                threshold
            ),

            "ours_clean_fpr": (
                ours_fpr
            ),

            "target_combined_fpr": float(
                target_combined_fpr
            ),

            "additional_clean_fp_budget": 0,

            "residual_clean_samples": 0,
        }



    if additional_budget <= 0:

    
        threshold = float(
            np.max(
                residual_js
            )
        )

    

    elif additional_budget >= len(
        residual_js
    ):

        threshold = float(
            -np.inf
        )

    

    else:

        
        descending = np.sort(
            residual_js
        )[
            ::-1
        ]

        
        threshold = float(
            descending[
                additional_budget
            ]
        )

    return {
        "threshold": (
            threshold
        ),

        "ours_clean_fpr": float(
            ours_fpr
        ),

        "target_combined_fpr": float(
            target_combined_fpr
        ),

        "additional_clean_fp_budget": int(
            additional_budget
        ),

        "residual_clean_samples": int(
            residual_js.size
        ),
    }




def compute_metrics(
    data,
    detector_mask,
    condition,
):
    detector_mask = np.asarray(
        detector_mask,
        dtype=bool,
    )

    labels = data[
        "label"
    ]

    preds = data[
        "pred"
    ]

    correct = (
        preds
        == labels
    )

    wrong = (
        ~correct
    )

    n = len(
        labels
    )

    detected = int(
        detector_mask.sum()
    )

    wrong_count = int(
        wrong.sum()
    )

    detected_wrong = int(
        (
            detector_mask
            &
            wrong
        ).sum()
    )

    accuracy = (
        correct.mean()
    )

    overall_detection_rate = (
        detector_mask.mean()
    )

    ecdr = safe_rate(
        detected_wrong,
        wrong_count,
    )

    metrics = {
        "condition": (
            condition
        ),

        "n": int(
            n
        ),

        "accuracy": float(
            accuracy
        ),

        "accuracy_percent": float(
            accuracy
            * 100.0
        ),

        "num_wrong": (
            wrong_count
        ),

        "num_detected": (
            detected
        ),

        "detection_rate": float(
            overall_detection_rate
        ),

        "detection_rate_percent": float(
            overall_detection_rate
            * 100.0
        ),

        "num_detected_wrong": (
            detected_wrong
        ),

        "ecdr": float(
            ecdr
        ),

        "ecdr_percent": float(
            ecdr
            * 100.0
        ) if np.isfinite(
            ecdr
        ) else float(
            "nan"
        ),
    }

    if condition == "clean":

        metrics[
            "fpr"
        ] = float(
            overall_detection_rate
        )

        metrics[
            "fpr_percent"
        ] = float(
            overall_detection_rate
            * 100.0
        )

        metrics[
            "tpr"
        ] = None

        metrics[
            "tpr_percent"
        ] = None

    else:

        metrics[
            "fpr"
        ] = None

        metrics[
            "fpr_percent"
        ] = None

        metrics[
            "tpr"
        ] = float(
            overall_detection_rate
        )

        metrics[
            "tpr_percent"
        ] = float(
            overall_detection_rate
            * 100.0
        )

    return (
        metrics,
        correct,
        wrong,
    )




def component_statistics(
    components,
):
    stats = {}

    for name, array in components.items():

        if name == "weak_votes":

            stats[
                "weak_votes_mean"
            ] = float(
                np.mean(
                    array
                )
            )

            stats[
                "weak_votes_max"
            ] = int(
                np.max(
                    array
                )
            )

        else:

            stats[
                f"{name}_percent"
            ] = float(
                np.mean(
                    array
                )
                * 100.0
            )

    return stats




def evaluate_model(
    model,
    weak_k,
    high_percentile,
    low_percentile,
    combined_clean_fpr,
):
    print(
        "\n"
        + "#" * 120
    )

    print(
        f"MODEL: {model.upper()}"
    )

    print(
        "#" * 120
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

    

    clean = load_signals(
        model,
        "clean",
    )

    n = len(
        clean[
            "label"
        ]
    )

    print(
        f"Clean samples: {n}"
    )

   

    ours_thresholds = calibrate_ours(
        clean,
        high_percentile=(
            high_percentile
        ),
        low_percentile=(
            low_percentile
        ),
    )

    ours_clean_mask, clean_components = (
        apply_ours(
            clean,
            ours_thresholds,
            weak_k,
        )
    )

    ours_clean_fpr = float(
        ours_clean_mask.mean()
    )

    

    js_calibration = (
        calibrate_js_for_union(
            clean_js=clean[
                "js"
            ],
            ours_clean_mask=(
                ours_clean_mask
            ),
            target_combined_fpr=(
                combined_clean_fpr
            ),
        )
    )

    js_threshold = float(
        js_calibration[
            "threshold"
        ]
    )


    js_clean_mask = (
        clean[
            "js"
        ]
        >
        js_threshold
    )

    combined_clean_mask = (
        ours_clean_mask
        |
        js_clean_mask
    )

    js_clean_fpr = float(
        js_clean_mask.mean()
    )

    combined_clean_fpr_actual = float(
        combined_clean_mask.mean()
    )

    

    print(
        "\nOURS CLEAN THRESHOLDS"
    )

    print(
        "-" * 70
    )

    print(
        f"Confidence low       : "
        f"{ours_thresholds['confidence_low']:.10f}"
    )

    print(
        f"Energy high          : "
        f"{ours_thresholds['energy_high']:.10f}"
    )

    print(
        f"2-pass conf-drop high: "
        f"{ours_thresholds['conf_drop_2_high']:.10f}"
    )

    print(
        f"2-pass logit-L2 high : "
        f"{ours_thresholds['logit_l2_2_high']:.10f}"
    )

    print(
        f"3-pass conf-drop high: "
        f"{ours_thresholds['conf_drop_3_high']:.10f}"
    )

    print(
        f"3-pass logit-L2 high : "
        f"{ours_thresholds['logit_l2_3_high']:.10f}"
    )

    print(
        f"Weak-k               : "
        f"{weak_k}"
    )

    print(
        "\nJS CALIBRATION"
    )

    print(
        "-" * 70
    )

    print(
        f"JS threshold          : "
        f"{js_threshold:.12f}"
    )

    print(
        f"Ours clean FPR        : "
        f"{ours_clean_fpr * 100:.3f}%"
    )

    print(
        f"JS-only clean FPR     : "
        f"{js_clean_fpr * 100:.3f}%"
    )

    print(
        f"Ours+JS clean FPR     : "
        f"{combined_clean_fpr_actual * 100:.3f}%"
    )

    print(
        f"Requested max FPR     : "
        f"{combined_clean_fpr * 100:.3f}%"
    )



    thresholds_output = {
        "model": (
            model
        ),

        "calibration_data": (
            "clean only"
        ),

        "high_percentile": float(
            high_percentile
        ),

        "low_percentile": float(
            low_percentile
        ),

        "weak_k": int(
            weak_k
        ),

        "ours": (
            ours_thresholds
        ),

        "js": {
            **js_calibration,

            "threshold": float(
                js_threshold
            ),

            "actual_js_only_clean_fpr": float(
                js_clean_fpr
            ),

            "actual_ours_js_clean_fpr": float(
                combined_clean_fpr_actual
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
            thresholds_output,
            f,
            indent=2,
        )

    

    rows = []

    for condition in CONDITIONS:

        print(
            "\n"
            + "=" * 100
        )

        print(
            f"{model.upper()} | "
            f"{condition}"
        )

        print(
            "=" * 100
        )

        data = load_signals(
            model,
            condition,
        )

        validate_alignment(
            clean,
            data,
            model,
            condition,
        )

      

        ours_mask, components = (
            apply_ours(
                data,
                ours_thresholds,
                weak_k,
            )
        )

   

        js_mask = (
            data[
                "js"
            ]
            >
            js_threshold
        )

        

        ours_js_mask = (
            ours_mask
            |
            js_mask
        )

        

        (
            ours_metrics,
            correct,
            wrong,
        ) = compute_metrics(
            data,
            ours_mask,
            condition,
        )

        (
            js_metrics,
            _,
            _,
        ) = compute_metrics(
            data,
            js_mask,
            condition,
        )

        (
            combined_metrics,
            _,
            _,
        ) = compute_metrics(
            data,
            ours_js_mask,
            condition,
        )

       

        new_by_js = (
            js_mask
            &
            ~ours_mask
        )

        wrong_new_by_js = (
            new_by_js
            &
            wrong
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

        save_array(
            condition_output,
            "sample_id",
            data[
                "sample_id"
            ],
        )

        save_array(
            condition_output,
            "label",
            data[
                "label"
            ],
        )

        save_array(
            condition_output,
            "pred",
            data[
                "pred"
            ],
        )

        save_array(
            condition_output,
            "correct",
            correct.astype(
                bool
            ),
        )

        save_array(
            condition_output,
            "wrong",
            wrong.astype(
                bool
            ),
        )

        save_array(
            condition_output,
            "ours_mask",
            ours_mask.astype(
                bool
            ),
        )

        save_array(
            condition_output,
            "js_mask",
            js_mask.astype(
                bool
            ),
        )

        save_array(
            condition_output,
            "ours_js_mask",
            ours_js_mask.astype(
                bool
            ),
        )

        save_array(
            condition_output,
            "js_new_flags",
            new_by_js.astype(
                bool
            ),
        )

        save_array(
            condition_output,
            "weak_votes",
            components[
                "weak_votes"
            ],
        )

        
        for component_name, values in (
            components.items()
        ):

            if component_name == "weak_votes":
                continue

            save_array(
                condition_output,
                component_name,
                values,
            )

      

        result = {
            "model": (
                model
            ),

            "condition": (
                condition
            ),

            "accuracy_percent": float(
                ours_metrics[
                    "accuracy_percent"
                ]
            ),

            "ours": (
                ours_metrics
            ),

            "js": (
                js_metrics
            ),

            "ours_js": (
                combined_metrics
            ),

            "js_added_samples": int(
                new_by_js.sum()
            ),

            "js_added_wrong_samples": int(
                wrong_new_by_js.sum()
            ),

            "components": (
                component_statistics(
                    components
                )
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
                result,
                f,
                indent=2,
            )

        

        print(
            f"Accuracy       : "
            f"{ours_metrics['accuracy_percent']:.3f}%"
        )

        if condition == "clean":

            print(
                f"Ours FPR       : "
                f"{ours_metrics['fpr_percent']:.3f}%"
            )

            print(
                f"JS FPR         : "
                f"{js_metrics['fpr_percent']:.3f}%"
            )

            print(
                f"Ours+JS FPR    : "
                f"{combined_metrics['fpr_percent']:.3f}%"
            )

        else:

            print(
                f"Ours TPR       : "
                f"{ours_metrics['tpr_percent']:.3f}%"
            )

            print(
                f"JS TPR         : "
                f"{js_metrics['tpr_percent']:.3f}%"
            )

            print(
                f"Ours+JS TPR    : "
                f"{combined_metrics['tpr_percent']:.3f}%"
            )

            print(
                f"Ours ECDR      : "
                f"{ours_metrics['ecdr_percent']:.3f}%"
            )

            print(
                f"JS ECDR        : "
                f"{js_metrics['ecdr_percent']:.3f}%"
            )

            print(
                f"Ours+JS ECDR   : "
                f"{combined_metrics['ecdr_percent']:.3f}%"
            )

            print(
                f"New JS flags   : "
                f"{int(new_by_js.sum())}"
            )

            print(
                f"New wrong flags: "
                f"{int(wrong_new_by_js.sum())}"
            )

     

        row = {
            "model": (
                model
            ),

            "condition": (
                condition
            ),

            "n": int(
                len(
                    data[
                        "label"
                    ]
                )
            ),

            "accuracy_percent": float(
                ours_metrics[
                    "accuracy_percent"
                ]
            ),

            "ours_fpr_percent": (
                ours_metrics[
                    "fpr_percent"
                ]
                if condition
                == "clean"
                else ""
            ),

            "js_fpr_percent": (
                js_metrics[
                    "fpr_percent"
                ]
                if condition
                == "clean"
                else ""
            ),

            "ours_js_fpr_percent": (
                combined_metrics[
                    "fpr_percent"
                ]
                if condition
                == "clean"
                else ""
            ),

            "ours_tpr_percent": (
                ""
                if condition
                == "clean"
                else ours_metrics[
                    "tpr_percent"
                ]
            ),

            "js_tpr_percent": (
                ""
                if condition
                == "clean"
                else js_metrics[
                    "tpr_percent"
                ]
            ),

            "ours_js_tpr_percent": (
                ""
                if condition
                == "clean"
                else combined_metrics[
                    "tpr_percent"
                ]
            ),

            "ours_ecdr_percent": float(
                ours_metrics[
                    "ecdr_percent"
                ]
            ),

            "js_ecdr_percent": float(
                js_metrics[
                    "ecdr_percent"
                ]
            ),

            "ours_js_ecdr_percent": float(
                combined_metrics[
                    "ecdr_percent"
                ]
            ),

            "js_added_samples": int(
                new_by_js.sum()
            ),

            "js_added_wrong_samples": int(
                wrong_new_by_js.sum()
            ),
        }

        rows.append(
            row
        )



    calibration_summary = {
        "model": (
            model
        ),

        "num_clean_samples": int(
            n
        ),

        "ours_clean_fpr_percent": float(
            ours_clean_fpr
            * 100.0
        ),

        "js_only_clean_fpr_percent": float(
            js_clean_fpr
            * 100.0
        ),

        "ours_js_clean_fpr_percent": float(
            combined_clean_fpr_actual
            * 100.0
        ),

        "js_threshold": float(
            js_threshold
        ),

        "ours_thresholds": (
            ours_thresholds
        ),

        "clean_components": (
            component_statistics(
                clean_components
            )
        ),
    }

    with open(
        os.path.join(
            model_output,
            "clean_calibration.json",
        ),
        "w",
    ) as f:

        json.dump(
            calibration_summary,
            f,
            indent=2,
        )

    return rows




def save_csv(
    path,
    rows,
):
    if not rows:
        return

    fields = list(
        rows[0].keys()
    )

    with open(
        path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )



def create_compact_rows(
    rows,
):
    compact = []

    for row in rows:

        condition = row[
            "condition"
        ]

        if condition == "clean":

            compact.append(
                {
                    "Model": (
                        row[
                            "model"
                        ]
                    ),

                    "Attack": (
                        "clean"
                    ),

                    "Accuracy": (
                        row[
                            "accuracy_percent"
                        ]
                    ),

                    "Method": (
                        "Ours"
                    ),

                    "FPR": (
                        row[
                            "ours_fpr_percent"
                        ]
                    ),

                    "TPR": "",
                    "ECDR": (
                        row[
                            "ours_ecdr_percent"
                        ]
                    ),
                }
            )

            compact.append(
                {
                    "Model": (
                        row[
                            "model"
                        ]
                    ),

                    "Attack": (
                        "clean"
                    ),

                    "Accuracy": (
                        row[
                            "accuracy_percent"
                        ]
                    ),

                    "Method": (
                        "JS"
                    ),

                    "FPR": (
                        row[
                            "js_fpr_percent"
                        ]
                    ),

                    "TPR": "",
                    "ECDR": (
                        row[
                            "js_ecdr_percent"
                        ]
                    ),
                }
            )

            compact.append(
                {
                    "Model": (
                        row[
                            "model"
                        ]
                    ),

                    "Attack": (
                        "clean"
                    ),

                    "Accuracy": (
                        row[
                            "accuracy_percent"
                        ]
                    ),

                    "Method": (
                        "Ours+JS"
                    ),

                    "FPR": (
                        row[
                            "ours_js_fpr_percent"
                        ]
                    ),

                    "TPR": "",
                    "ECDR": (
                        row[
                            "ours_js_ecdr_percent"
                        ]
                    ),
                }
            )

        else:

            for method, tpr_key, ecdr_key in [
                (
                    "Ours",
                    "ours_tpr_percent",
                    "ours_ecdr_percent",
                ),
                (
                    "JS",
                    "js_tpr_percent",
                    "js_ecdr_percent",
                ),
                (
                    "Ours+JS",
                    "ours_js_tpr_percent",
                    "ours_js_ecdr_percent",
                ),
            ]:

                compact.append(
                    {
                        "Model": (
                            row[
                                "model"
                            ]
                        ),

                        "Attack": (
                            condition
                        ),

                        "Accuracy": (
                            row[
                                "accuracy_percent"
                            ]
                        ),

                        "Method": (
                            method
                        ),

                        "FPR": "",

                        "TPR": (
                            row[
                                tpr_key
                            ]
                        ),

                        "ECDR": (
                            row[
                                ecdr_key
                            ]
                        ),
                    }
                )

    return compact




def print_final_table(
    rows,
):
    print(
        "\n"
        + "=" * 170
    )

    print(
        "FINAL GTSRB DETECTION SUMMARY"
    )

    print(
        "=" * 170
    )

    header = (
        f"{'Model':<14}"
        f"{'Condition':<18}"
        f"{'Acc':>9}"
        f"{'Ours':>10}"
        f"{'JS':>10}"
        f"{'Ours+JS':>12}"
        f"{'Ours ECDR':>13}"
        f"{'JS ECDR':>11}"
        f"{'O+JS ECDR':>13}"
    )

    print(
        header
    )

    print(
        "-" * 170
    )

    for row in rows:

        model = row[
            "model"
        ]

        condition = row[
            "condition"
        ]

        acc = row[
            "accuracy_percent"
        ]

        if condition == "clean":

            ours = row[
                "ours_fpr_percent"
            ]

            js = row[
                "js_fpr_percent"
            ]

            combined = row[
                "ours_js_fpr_percent"
            ]

            print(
                f"{model:<14}"
                f"{condition:<18}"
                f"{acc:>8.2f}%"
                f"{ours:>9.2f}%"
                f"{js:>9.2f}%"
                f"{combined:>11.2f}%"
                f"{'-':>13}"
                f"{'-':>11}"
                f"{'-':>13}"
            )

        else:

            print(
                f"{model:<14}"
                f"{condition:<18}"
                f"{acc:>8.2f}%"
                f"{row['ours_tpr_percent']:>9.2f}%"
                f"{row['js_tpr_percent']:>9.2f}%"
                f"{row['ours_js_tpr_percent']:>11.2f}%"
                f"{row['ours_ecdr_percent']:>12.2f}%"
                f"{row['js_ecdr_percent']:>10.2f}%"
                f"{row['ours_js_ecdr_percent']:>12.2f}%"
            )




def validate_inputs(
    selected_models,
):
    required_files = [
        "sample_id.npy",
        "label.npy",
        "pred.npy",
        "confidence.npy",
        "energy.npy",
        "conf_drop_2.npy",
        "logit_l2_2.npy",
        "changed_2.npy",
        "critical_pred_mask.npy",
        "conf_drop_3.npy",
        "logit_l2_3.npy",
        "changed_3.npy",
        "js_divergence.npy",
    ]

    missing = []

    for model in selected_models:

        for condition in CONDITIONS:

            directory = os.path.join(
                SIGNAL_ROOT,
                model,
                condition,
            )

            for filename in required_files:

                path = os.path.join(
                    directory,
                    filename,
                )

                if not os.path.isfile(
                    path
                ):
                    missing.append(
                        path
                    )

    if missing:

        preview = "\n".join(
            f"  {path}"
            for path in missing[
                :30
            ]
        )

        more = ""

        if len(
            missing
        ) > 30:

            more = (
                f"\n... plus "
                f"{len(missing) - 30} more"
            )

        raise FileNotFoundError(
            "\nDetection cannot start because "
            "signal extraction is incomplete.\n\n"
            f"Missing files:\n{preview}"
            f"{more}\n\n"
            "Wait for extract_gtsrb_all_signals.py "
            "to finish first."
        )




def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate GTSRB Ours, JS, and Ours+JS "
            "from previously extracted signals."
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

    parser.add_argument(
        "--weak-k",
        type=int,
        default=(
            DEFAULT_WEAK_K
        ),
    )

    parser.add_argument(
        "--high-percentile",
        type=float,
        default=(
            DEFAULT_HIGH_PERCENTILE
        ),
    )

    parser.add_argument(
        "--low-percentile",
        type=float,
        default=(
            DEFAULT_LOW_PERCENTILE
        ),
    )

    parser.add_argument(
        "--combined-clean-fpr",
        type=float,
        default=(
            DEFAULT_COMBINED_CLEAN_FPR
        ),
        help=(
            "Desired maximum Ours+JS clean FPR. "
            "Default = 0.06 (6%%). "
            "If Ours already exceeds it, JS is not "
            "allowed to introduce additional clean FPs."
        ),
    )

    args = parser.parse_args()

    

    if args.weak_k < 1:
        raise ValueError(
            "--weak-k must be >= 1"
        )

    if not (
        0.0
        <
        args.high_percentile
        <
        100.0
    ):
        raise ValueError(
            "--high-percentile must be between 0 and 100."
        )

    if not (
        0.0
        <
        args.low_percentile
        <
        100.0
    ):
        raise ValueError(
            "--low-percentile must be between 0 and 100."
        )

    if not (
        0.0
        <=
        args.combined_clean_fpr
        <=
        1.0
    ):
        raise ValueError(
            "--combined-clean-fpr must be between 0 and 1."
        )

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
        "=" * 120
    )

    print(
        "GTSRB — OURS / JS / OURS+JS DETECTION"
    )

    print(
        "=" * 120
    )

    print(
        f"Models                : "
        f"{selected_models}"
    )

    print(
        f"Conditions            : "
        f"{CONDITIONS}"
    )

    print(
        f"Signal root           : "
        f"{SIGNAL_ROOT}"
    )

    print(
        f"Output root           : "
        f"{OUTPUT_ROOT}"
    )

    print(
        f"High percentile       : "
        f"{args.high_percentile}"
    )

    print(
        f"Low percentile        : "
        f"{args.low_percentile}"
    )

    print(
        f"Weak-k                : "
        f"{args.weak_k}"
    )

    print(
        f"Combined-clean target : "
        f"{args.combined_clean_fpr * 100:.2f}%"
    )

  

    validate_inputs(
        selected_models
    )

    print(
        "\n✅ All required signal files are present."
    )

   

    all_rows = []

    for model in selected_models:

        model_rows = evaluate_model(
            model=model,
            weak_k=args.weak_k,
            high_percentile=(
                args.high_percentile
            ),
            low_percentile=(
                args.low_percentile
            ),
            combined_clean_fpr=(
                args.combined_clean_fpr
            ),
        )

        all_rows.extend(
            model_rows
        )

  

    full_csv = os.path.join(
        OUTPUT_ROOT,
        "detection_results.csv",
    )

    save_csv(
        full_csv,
        all_rows,
    )



    with open(
        os.path.join(
            OUTPUT_ROOT,
            "detection_results.json",
        ),
        "w",
    ) as f:

        json.dump(
            all_rows,
            f,
            indent=2,
        )

    

    compact_rows = create_compact_rows(
        all_rows
    )

    save_csv(
        os.path.join(
            OUTPUT_ROOT,
            "paper_table_compact.csv",
        ),
        compact_rows,
    )


    print_final_table(
        all_rows
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "DONE"
    )

    print(
        "=" * 120
    )

    print(
        "\nResults saved to:"
    )

    print(
        f"  {full_csv}"
    )

    print(
        "\nPaper-friendly table:"
    )

    print(
        "  "
        + os.path.join(
            OUTPUT_ROOT,
            "paper_table_compact.csv",
        )
    )

    print(
        "\nFinal detector masks:"
    )

    print(
        "  "
        + os.path.join(
            OUTPUT_ROOT,
            "<model>",
            "<condition>",
            "ours_js_mask.npy",
        )
    )

    print(
        "\nThese Ours+JS masks are the masks "
        "to use later for selective DDPM."
    )


if __name__ == "__main__":
    main()