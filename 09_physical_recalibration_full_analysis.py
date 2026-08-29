

import os
import re
import json
import csv
import shutil
import argparse
from pathlib import Path

import numpy as np
import torch




ROOT = (
    "/home/Traffic_Signs_2"
)

PIPELINE_ROOT = os.path.join(
    ROOT,
    "physical_pipeline"
)

MODEL_ROOT = os.path.join(
    ROOT,
    "physical_models"
)

ATTACKED_ROOT = os.path.join(
    ROOT,
    "attacked"
)

OUT_ROOT = os.path.join(
    ROOT,
    "physical_recalibrated_analysis"
)

DDPM_ALL_ROOT = os.path.join(
    ROOT,
    "physical_ddpm_input_all70"
)



MAX_CALIBRATION_FPR = 10.0



PERCENTILE_GRID = [
    80.0,
    85.0,
    90.0,
    92.5,
    95.0,
    97.5,
    99.0
]


WEAK_K_GRID = [
    1,
    2,
    3,
    4,
    5
]



MODELS = {

    "mobilenet": {

        "checkpoint": os.path.join(
            MODEL_ROOT,
            "mobilenetv3_comma_clean_best.pth"
        ),

        "eval_root": os.path.join(
            PIPELINE_ROOT,
            "eval_mobilenet"
        ),

        "js_root": os.path.join(
            PIPELINE_ROOT,
            "js_signal",
            "mobilenet"
        )
    },


    "convnext": {

        "checkpoint": os.path.join(
            MODEL_ROOT,
            "convnext_tiny_comma_clean_best.pth"
        ),

        "eval_root": os.path.join(
            PIPELINE_ROOT,
            "eval_convnext"
        ),

        "js_root": os.path.join(
            PIPELINE_ROOT,
            "js_signal",
            "convnext"
        )
    },


    "efficientnet": {

        "checkpoint": os.path.join(
            MODEL_ROOT,
            "efficientnet_v2_s_comma_clean_best.pth"
        ),

        "eval_root": os.path.join(
            PIPELINE_ROOT,
            "eval_efficientnet"
        ),

        "js_root": os.path.join(
            PIPELINE_ROOT,
            "js_signal",
            "efficientnet"
        )
    }
}


MODEL_KEYS = list(
    MODELS.keys()
)




def safe_mkdir(path):

    Path(path).mkdir(
        parents=True,
        exist_ok=True
    )


def load_npy(
    folder,
    filename
):

    path = os.path.join(
        folder,
        filename
    )

    if not os.path.exists(path):

        raise FileNotFoundError(
            path
        )

    return np.load(
        path,
        allow_pickle=True
    )


def load_checkpoint(path):

    if not os.path.exists(path):

        raise FileNotFoundError(
            path
        )

    return torch.load(
        path,
        map_location="cpu",
        weights_only=False
    )




def numeric_suffix(path):

    name = Path(
        str(path)
    ).stem

    matches = re.findall(
        r"(\d+)",
        name
    )

    if not matches:

        raise RuntimeError(
            f"No numeric suffix found in filename: {path}"
        )

    return int(
        matches[-1]
    )


def sample_key_from_path(
    path,
    class_name=None
):

    p = Path(
        str(path)
    )

    if class_name is None:

        class_name = p.parent.name

    return (
        str(class_name),
        numeric_suffix(
            p
        )
    )


def checkpoint_sample_to_key(
    item,
    idx_to_class
):

    """
    Supports likely checkpoint formats:

        path
        (path, label)
        [path, label]
        {"path": ..., "label": ...}

    """

    path = None
    label = None


    if isinstance(
        item,
        str
    ):

        path = item


    elif isinstance(
        item,
        (tuple, list)
    ):

        if len(item) >= 1:

            path = item[0]

        if len(item) >= 2:

            try:

                label = int(
                    item[1]
                )

            except Exception:

                label = None


    elif isinstance(
        item,
        dict
    ):

        for key in [
            "path",
            "filepath",
            "filename",
            "file"
        ]:

            if key in item:

                path = item[
                    key
                ]

                break


        for key in [
            "label",
            "target",
            "class_id"
        ]:

            if key in item:

                try:

                    label = int(
                        item[
                            key
                        ]
                    )

                except Exception:

                    pass

                break


    if path is None:

        raise RuntimeError(
            f"Unsupported checkpoint sample entry: {item}"
        )


    if (
        label is not None
        and label in idx_to_class
    ):

        class_name = idx_to_class[
            label
        ]

    else:

        class_name = Path(
            str(path)
        ).parent.name


    return sample_key_from_path(
        path,
        class_name
    )



def load_signal_condition(
    cfg,
    condition
):

    folder = os.path.join(
        cfg[
            "eval_root"
        ],
        condition
    )


    data = {

        "filenames":
            load_npy(
                folder,
                "filenames.npy"
            ),

        "label":
            load_npy(
                folder,
                "label.npy"
            ).astype(
                np.int64
            ),

        "pred":
            load_npy(
                folder,
                "pred.npy"
            ).astype(
                np.int64
            ),

        "confidence":
            load_npy(
                folder,
                "confidence.npy"
            ).astype(
                np.float64
            ),

        "energy":
            load_npy(
                folder,
                "energy.npy"
            ).astype(
                np.float64
            ),

        "conf_drop_2":
            load_npy(
                folder,
                "2pass_conf_drop.npy"
            ).astype(
                np.float64
            ),

        "logit_l2_2":
            load_npy(
                folder,
                "2pass_logit_l2.npy"
            ).astype(
                np.float64
            ),

        "changed_2":
            load_npy(
                folder,
                "2pass_changed.npy"
            ).astype(
                np.int64
            ),

        "conf_drop_3":
            load_npy(
                folder,
                "3pass_max_conf_drop_critical.npy"
            ).astype(
                np.float64
            ),

        "logit_l2_3":
            load_npy(
                folder,
                "3pass_max_logit_l2_critical.npy"
            ).astype(
                np.float64
            ),

        "changed_3":
            load_npy(
                folder,
                "3pass_changed_critical.npy"
            ).astype(
                np.int64
            ),

        "critical":
            load_npy(
                folder,
                "critical_pred_mask.npy"
            ).astype(
                bool
            )
    }


    n = len(
        data[
            "label"
        ]
    )


    for key, value in data.items():

        if len(value) != n:

            raise RuntimeError(
                f"{condition}: length mismatch for {key}"
            )


    return data


def load_js_condition(
    cfg,
    condition
):

    folder = os.path.join(
        cfg[
            "js_root"
        ],
        condition
    )


    return {

        "filenames":
            load_npy(
                folder,
                "filenames.npy"
            ),

        "js":
            load_npy(
                folder,
                "js_divergence.npy"
            ).astype(
                np.float64
            ),

        "labels":
            load_npy(
                folder,
                "labels.npy"
            ).astype(
                np.int64
            ),

        "pred":
            load_npy(
                folder,
                "pred_base.npy"
            ).astype(
                np.int64
            )
    }




def build_split(
    model_key,
    cfg,
    clean
):

    checkpoint = load_checkpoint(
        cfg[
            "checkpoint"
        ]
    )


    if "class_to_idx" not in checkpoint:

        raise RuntimeError(
            f"{model_key}: class_to_idx missing from checkpoint"
        )


    class_to_idx = checkpoint[
        "class_to_idx"
    ]


    idx_to_class = {

        int(v): k

        for k, v in class_to_idx.items()
    }


    train_samples = checkpoint.get(
        "train_samples"
    )


    val_samples = checkpoint.get(
        "val_samples"
    )


    if train_samples is None:

        raise RuntimeError(
            f"{model_key}: train_samples missing from checkpoint"
        )


    if val_samples is None:

        raise RuntimeError(
            f"{model_key}: val_samples missing from checkpoint"
        )


    train_keys = {

        checkpoint_sample_to_key(
            x,
            idx_to_class
        )

        for x in train_samples
    }


    val_keys = {

        checkpoint_sample_to_key(
            x,
            idx_to_class
        )

        for x in val_samples
    }


    clean_keys = [

        sample_key_from_path(
            x
        )

        for x in clean[
            "filenames"
        ]
    ]


    calibration_mask = np.asarray(
        [
            k in train_keys
            for k in clean_keys
        ],
        dtype=bool
    )


    test_mask = np.asarray(
        [
            k in val_keys
            for k in clean_keys
        ],
        dtype=bool
    )


    overlap = (
        calibration_mask
        &
        test_mask
    ).sum()


    if overlap != 0:

        raise RuntimeError(
            f"{model_key}: calibration/test overlap = {overlap}"
        )


    print(
        f"\n[{model_key}] split recovered"
    )

    print(
        f"  calibration : "
        f"{calibration_mask.sum()}"
    )

    print(
        f"  held-out    : "
        f"{test_mask.sum()}"
    )


    if int(
        calibration_mask.sum()
    ) != 50:

        raise RuntimeError(
            f"{model_key}: expected 50 calibration samples, "
            f"got {calibration_mask.sum()}"
        )


    if int(
        test_mask.sum()
    ) != 20:

        raise RuntimeError(
            f"{model_key}: expected 20 held-out samples, "
            f"got {test_mask.sum()}"
        )


    if (
        calibration_mask
        |
        test_mask
    ).sum() != 70:

        raise RuntimeError(
            f"{model_key}: 50/20 split does not cover all 70 samples"
        )


    return (
        calibration_mask,
        test_mask,
        checkpoint
    )




def percentile(
    x,
    q
):

    x = np.asarray(
        x,
        dtype=np.float64
    )


    x = x[
        np.isfinite(
            x
        )
    ]


    if len(x) == 0:

        return None


    return float(
        np.percentile(
            x,
            q
        )
    )


def compute_old_thresholds(
    clean,
    calibration_mask,
    q
):

    critical_calibration = (
        calibration_mask
        &
        clean[
            "critical"
        ]
        &
        (
            clean[
                "changed_3"
            ]
            !=
            -1
        )
    )


    return {

        "q":
            float(q),

        "energy":
            percentile(
                clean[
                    "energy"
                ][
                    calibration_mask
                ],
                q
            ),

        "confidence":
            percentile(
                clean[
                    "confidence"
                ][
                    calibration_mask
                ],
                100.0 - q
            ),

        "conf_drop_2":
            percentile(
                clean[
                    "conf_drop_2"
                ][
                    calibration_mask
                ],
                q
            ),

        "logit_l2_2":
            percentile(
                clean[
                    "logit_l2_2"
                ][
                    calibration_mask
                ],
                q
            ),

        "conf_drop_3":
            percentile(
                clean[
                    "conf_drop_3"
                ][
                    critical_calibration
                ],
                q
            ),

        "logit_l2_3":
            percentile(
                clean[
                    "logit_l2_3"
                ][
                    critical_calibration
                ],
                q
            ),

        "n_critical_calibration":
            int(
                critical_calibration.sum()
            )
    }




def old_detector_mask(
    data,
    thresholds,
    weak_k
):

    n = len(
        data[
            "label"
        ]
    )


    flag_energy = (
        data[
            "energy"
        ]
        >
        thresholds[
            "energy"
        ]
    )


    flag_confidence = (
        data[
            "confidence"
        ]
        <
        thresholds[
            "confidence"
        ]
    )


    flag_conf_drop_2 = (
        data[
            "conf_drop_2"
        ]
        >
        thresholds[
            "conf_drop_2"
        ]
    )


    flag_logit_l2_2 = (
        data[
            "logit_l2_2"
        ]
        >
        thresholds[
            "logit_l2_2"
        ]
    )


    flag_changed_2 = (
        data[
            "changed_2"
        ]
        ==
        1
    )


    valid_3 = (
        data[
            "critical"
        ]
        &
        (
            data[
                "changed_3"
            ]
            !=
            -1
        )
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


    if thresholds[
        "conf_drop_3"
    ] is not None:

        flag_conf_drop_3[
            valid_3
        ] = (

            data[
                "conf_drop_3"
            ][
                valid_3
            ]

            >

            thresholds[
                "conf_drop_3"
            ]
        )


    if thresholds[
        "logit_l2_3"
    ] is not None:

        flag_logit_l2_3[
            valid_3
        ] = (

            data[
                "logit_l2_3"
            ][
                valid_3
            ]

            >

            thresholds[
                "logit_l2_3"
            ]
        )


    flag_changed_3[
        valid_3
    ] = (
        data[
            "changed_3"
        ][
            valid_3
        ]
        ==
        1
    )


    

    strong = (

        flag_changed_2

        |

        flag_changed_3

        |

        flag_logit_l2_2
    )


    weak_count = (

        flag_energy.astype(
            np.int64
        )

        +

        flag_confidence.astype(
            np.int64
        )

        +

        flag_conf_drop_2.astype(
            np.int64
        )

        +

        flag_conf_drop_3.astype(
            np.int64
        )

        +

        flag_logit_l2_3.astype(
            np.int64
        )
    )


    suspicious = (

        strong

        |

        (
            weak_count
            >=
            int(
                weak_k
            )
        )
    )


    return suspicious




def detection_metrics(
    mask,
    data,
    subset_mask
):

    mask = mask[
        subset_mask
    ]


    label = data[
        "label"
    ][
        subset_mask
    ]


    pred = data[
        "pred"
    ][
        subset_mask
    ]


    wrong = (
        pred
        !=
        label
    )


    n = len(
        mask
    )


    flagged = int(
        mask.sum()
    )


    wrong_total = int(
        wrong.sum()
    )


    wrong_flagged = int(
        (
            mask
            &
            wrong
        ).sum()
    )


    return {

        "n":
            int(n),

        "flagged":
            flagged,

        "rate_percent":
            float(
                100.0
                *
                flagged
                /
                n
            ),

        "wrong_total":
            wrong_total,

        "wrong_flagged":
            wrong_flagged,

        "wrong_detection_percent":
            (
                float(
                    100.0
                    *
                    wrong_flagged
                    /
                    wrong_total
                )

                if wrong_total > 0

                else None
            ),

        "classifier_accuracy_percent":
            float(
                100.0
                *
                (
                    pred
                    ==
                    label
                ).mean()
            )
    }




def js_threshold_from_calibration(
    js_clean,
    calibration_mask,
    q
):

    return float(
        np.percentile(
            js_clean[
                calibration_mask
            ],
            q
        )
    )



def search_best_configuration(
    clean,
    qr,
    clean_js,
    qr_js,
    calibration_mask
):

    candidates = []


    for q_old in PERCENTILE_GRID:

        thresholds = compute_old_thresholds(
            clean,
            calibration_mask,
            q_old
        )


        for weak_k in WEAK_K_GRID:

            old_clean = old_detector_mask(
                clean,
                thresholds,
                weak_k
            )


            old_qr = old_detector_mask(
                qr,
                thresholds,
                weak_k
            )


            for q_js in PERCENTILE_GRID:

                js_threshold = (
                    js_threshold_from_calibration(
                        clean_js,
                        calibration_mask,
                        q_js
                    )
                )


                js_clean_mask = (
                    clean_js
                    >
                    js_threshold
                )


                js_qr_mask = (
                    qr_js
                    >
                    js_threshold
                )


                methods = {

                    "OLD":
                        (
                            old_clean,
                            old_qr
                        ),

                    "JS":
                        (
                            js_clean_mask,
                            js_qr_mask
                        ),

                    "OR":
                        (
                            old_clean
                            |
                            js_clean_mask,

                            old_qr
                            |
                            js_qr_mask
                        ),

                    "AND":
                        (
                            old_clean
                            &
                            js_clean_mask,

                            old_qr
                            &
                            js_qr_mask
                        )
                }


                for method_name, (
                    clean_mask,
                    qr_mask
                ) in methods.items():


                    clean_metrics = (
                        detection_metrics(
                            clean_mask,
                            clean,
                            calibration_mask
                        )
                    )


                    qr_metrics = (
                        detection_metrics(
                            qr_mask,
                            qr,
                            calibration_mask
                        )
                    )


                    clean_fpr = clean_metrics[
                        "rate_percent"
                    ]


                    qr_tpr = qr_metrics[
                        "rate_percent"
                    ]


                    eligible = (
                        clean_fpr
                        <=
                        MAX_CALIBRATION_FPR
                        +
                        1e-9
                    )


                    candidates.append({

                        "eligible":
                            bool(
                                eligible
                            ),

                        "q_old":
                            float(
                                q_old
                            ),

                        "q_js":
                            float(
                                q_js
                            ),

                        "weak_k":
                            int(
                                weak_k
                            ),

                        "method":
                            method_name,

                        "clean_fpr":
                            float(
                                clean_fpr
                            ),

                        "qr_tpr":
                            float(
                                qr_tpr
                            ),

                        "qr_wrong_total":
                            int(
                                qr_metrics[
                                    "wrong_total"
                                ]
                            ),

                        "qr_wrong_flagged":
                            int(
                                qr_metrics[
                                    "wrong_flagged"
                                ]
                            ),

                        "old_thresholds":
                            thresholds,

                        "js_threshold":
                            float(
                                js_threshold
                            )
                    })


    eligible = [
        x
        for x in candidates
        if x[
            "eligible"
        ]
    ]


    if not eligible:

        raise RuntimeError(
            "No configuration satisfies calibration FPR constraint."
        )


    

    eligible.sort(
        key=lambda x: (
            -x[
                "qr_tpr"
            ],
            -x[
                "qr_wrong_flagged"
            ],
            x[
                "clean_fpr"
            ],
            -x[
                "q_old"
            ],
            -x[
                "q_js"
            ],
            -x[
                "weak_k"
            ]
        )
    )


    return (
        eligible[0],
        candidates
    )




def apply_config(
    best,
    clean,
    qr,
    clean_js,
    qr_js
):

    old_clean = old_detector_mask(
        clean,
        best[
            "old_thresholds"
        ],
        best[
            "weak_k"
        ]
    )


    old_qr = old_detector_mask(
        qr,
        best[
            "old_thresholds"
        ],
        best[
            "weak_k"
        ]
    )


    js_clean = (
        clean_js
        >
        best[
            "js_threshold"
        ]
    )


    js_qr = (
        qr_js
        >
        best[
            "js_threshold"
        ]
    )


    masks = {

        "OLD": {
            "clean":
                old_clean,
            "qr":
                old_qr
        },

        "JS": {
            "clean":
                js_clean,
            "qr":
                js_qr
        },

        "OR": {
            "clean":
                (
                    old_clean
                    |
                    js_clean
                ),
            "qr":
                (
                    old_qr
                    |
                    js_qr
                )
        },

        "AND": {
            "clean":
                (
                    old_clean
                    &
                    js_clean
                ),
            "qr":
                (
                    old_qr
                    &
                    js_qr
                )
        }
    }


    return masks




def analyze_model(
    model_key
):

    cfg = MODELS[
        model_key
    ]


    print("\n")
    print("#" * 100)
    print(
        f"MODEL: {model_key.upper()}"
    )
    print("#" * 100)


    clean = load_signal_condition(
        cfg,
        "clean"
    )


    qr = load_signal_condition(
        cfg,
        "qr"
    )


    clean_js_data = load_js_condition(
        cfg,
        "clean"
    )


    qr_js_data = load_js_condition(
        cfg,
        "qr"
    )


    clean_js = clean_js_data[
        "js"
    ]


    qr_js = qr_js_data[
        "js"
    ]


   

    if len(clean_js) != 70:

        raise RuntimeError(
            f"{model_key}: expected 70 clean JS samples"
        )


    if len(qr_js) != 70:

        raise RuntimeError(
            f"{model_key}: expected 70 QR JS samples"
        )


    if not np.array_equal(
        clean[
            "label"
        ],
        clean_js_data[
            "labels"
        ]
    ):

        raise RuntimeError(
            f"{model_key}: clean JS label alignment mismatch"
        )


    if not np.array_equal(
        qr[
            "label"
        ],
        qr_js_data[
            "labels"
        ]
    ):

        raise RuntimeError(
            f"{model_key}: QR JS label alignment mismatch"
        )


    

    (
        calibration_mask,
        test_mask,
        checkpoint
    ) = build_split(
        model_key,
        cfg,
        clean
    )


    

    best, search_results = (
        search_best_configuration(
            clean,
            qr,
            clean_js,
            qr_js,
            calibration_mask
        )
    )


    print(
        "\nBEST CALIBRATION CONFIGURATION"
    )

    print(
        f"  method        : "
        f"{best['method']}"
    )

    print(
        f"  OLD q         : "
        f"{best['q_old']}"
    )

    print(
        f"  JS q          : "
        f"{best['q_js']}"
    )

    print(
        f"  weak_k        : "
        f"{best['weak_k']}"
    )

    print(
        f"  calib FPR     : "
        f"{best['clean_fpr']:.2f}%"
    )

    print(
        f"  calib QR TPR  : "
        f"{best['qr_tpr']:.2f}%"
    )

    print(
        f"  calib wrong   : "
        f"{best['qr_wrong_flagged']}/"
        f"{best['qr_wrong_total']}"
    )




    masks = apply_config(
        best,
        clean,
        qr,
        clean_js,
        qr_js
    )


    

    heldout_results = {}


    print(
        "\nHELD-OUT 20-SAMPLE TEST"
    )

    print(
        f"{'Method':10s}"
        f"{'Clean FPR':>14s}"
        f"{'QR TPR':>14s}"
        f"{'QR wrong':>16s}"
    )

    print(
        "-" * 60
    )


    for method_name in [
        "OLD",
        "JS",
        "OR",
        "AND"
    ]:


        clean_metrics = detection_metrics(
            masks[
                method_name
            ][
                "clean"
            ],
            clean,
            test_mask
        )


        qr_metrics = detection_metrics(
            masks[
                method_name
            ][
                "qr"
            ],
            qr,
            test_mask
        )


        heldout_results[
            method_name
        ] = {

            "clean":
                clean_metrics,

            "qr":
                qr_metrics
        }


        print(
            f"{method_name:10s}"
            f"{clean_metrics['rate_percent']:13.2f}%"
            f"{qr_metrics['rate_percent']:13.2f}%"
            f"{qr_metrics['wrong_flagged']:7d}/"
            f"{qr_metrics['wrong_total']:<7d}"
        )


    chosen_method = best[
        "method"
    ]


    chosen_test = heldout_results[
        chosen_method
    ]


    model_out = os.path.join(
        OUT_ROOT,
        model_key
    )


    safe_mkdir(
        model_out
    )


    
    np.save(
        os.path.join(
            model_out,
            "calibration_mask.npy"
        ),
        calibration_mask
    )


    np.save(
        os.path.join(
            model_out,
            "heldout_mask.npy"
        ),
        test_mask
    )


    for method_name in masks:

        np.save(
            os.path.join(
                model_out,
                f"{method_name.lower()}_clean_mask.npy"
            ),
            masks[
                method_name
            ][
                "clean"
            ]
        )


        np.save(
            os.path.join(
                model_out,
                f"{method_name.lower()}_qr_mask.npy"
            ),
            masks[
                method_name
            ][
                "qr"
            ]
        )


    csv_path = os.path.join(
        model_out,
        "calibration_search.csv"
    )


    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        fieldnames = [
            "eligible",
            "q_old",
            "q_js",
            "weak_k",
            "method",
            "clean_fpr",
            "qr_tpr",
            "qr_wrong_total",
            "qr_wrong_flagged"
        ]


        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )


        writer.writeheader()


        for row in search_results:

            writer.writerow({

                key: row[
                    key
                ]

                for key in fieldnames
            })


    result = {

        "model":
            model_key,

        "split": {

            "calibration_n":
                int(
                    calibration_mask.sum()
                ),

            "heldout_n":
                int(
                    test_mask.sum()
                ),

            "calibration_source":
                "checkpoint train_samples",

            "heldout_source":
                "checkpoint val_samples"
        },


        "selection_rule": {

            "max_calibration_clean_fpr_percent":
                MAX_CALIBRATION_FPR,

            "primary_objective":
                "maximize calibration QR TPR",

            "tie_breaks": [
                "more calibration QR errors caught",
                "lower calibration clean FPR",
                "more conservative percentile"
            ]
        },


        "best_configuration":
            best,


        "heldout_results":
            heldout_results,


        "chosen_method_heldout":
            {

                "method":
                    chosen_method,

                "clean":
                    chosen_test[
                        "clean"
                    ],

                "qr":
                    chosen_test[
                        "qr"
                    ]
            }
    }


    with open(
        os.path.join(
            model_out,
            "recalibration_results.json"
        ),
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )


    return {

        "result":
            result,

        "masks":
            masks,

        "clean":
            clean,

        "qr":
            qr,

        "calibration_mask":
            calibration_mask,

        "test_mask":
            test_mask
    }



def ensemble_analysis(
    all_models
):

    print("\n\n")
    print("=" * 100)
    print(
        "3-MODEL ENSEMBLE — HELD-OUT 20"
    )
    print("=" * 100)


   

    first = MODEL_KEYS[0]


    reference_test = all_models[
        first
    ][
        "test_mask"
    ]


    for model_key in MODEL_KEYS[1:]:

        if not np.array_equal(
            reference_test,
            all_models[
                model_key
            ][
                "test_mask"
            ]
        ):

            raise RuntimeError(
                "Held-out masks differ across models."
            )



    clean_model_masks = []

    qr_model_masks = []


    for model_key in MODEL_KEYS:

        chosen_method = (
            all_models[
                model_key
            ][
                "result"
            ][
                "best_configuration"
            ][
                "method"
            ]
        )


        clean_model_masks.append(

            all_models[
                model_key
            ][
                "masks"
            ][
                chosen_method
            ][
                "clean"
            ]

        )


        qr_model_masks.append(

            all_models[
                model_key
            ][
                "masks"
            ][
                chosen_method
            ][
                "qr"
            ]

        )


    clean_votes = np.stack(
        clean_model_masks,
        axis=0
    ).sum(
        axis=0
    )


    qr_votes = np.stack(
        qr_model_masks,
        axis=0
    ).sum(
        axis=0
    )


    ensemble_masks = {

        "ANY": {
            "clean":
                clean_votes >= 1,
            "qr":
                qr_votes >= 1
        },

        "TWO_OF_THREE": {
            "clean":
                clean_votes >= 2,
            "qr":
                qr_votes >= 2
        },

        "ALL_THREE": {
            "clean":
                clean_votes >= 3,
            "qr":
                qr_votes >= 3
        }
    }


    test = reference_test


    harmful_qr = np.zeros(
        70,
        dtype=bool
    )


    for model_key in MODEL_KEYS:

        data = all_models[
            model_key
        ][
            "qr"
        ]


        harmful_qr |= (
            data[
                "pred"
            ]
            !=
            data[
                "label"
            ]
        )


    results = {}


    print(
        f"{'Ensemble':18s}"
        f"{'Clean FPR':>14s}"
        f"{'QR TPR':>14s}"
        f"{'Harmful caught':>18s}"
    )

    print(
        "-" * 70
    )


    for name, masks in ensemble_masks.items():

        clean_mask = masks[
            "clean"
        ][
            test
        ]


        qr_mask = masks[
            "qr"
        ][
            test
        ]


        harmful_test = harmful_qr[
            test
        ]


        harmful_total = int(
            harmful_test.sum()
        )


        harmful_caught = int(
            (
                qr_mask
                &
                harmful_test
            ).sum()
        )


        clean_fpr = float(
            100.0
            *
            clean_mask.mean()
        )


        qr_tpr = float(
            100.0
            *
            qr_mask.mean()
        )


        print(
            f"{name:18s}"
            f"{clean_fpr:13.2f}%"
            f"{qr_tpr:13.2f}%"
            f"{harmful_caught:9d}/"
            f"{harmful_total:<8d}"
        )


        results[
            name
        ] = {

            "clean_fpr_percent":
                clean_fpr,

            "qr_tpr_percent":
                qr_tpr,

            "harmful_qr_total":
                harmful_total,

            "harmful_qr_caught":
                harmful_caught,

            "harmful_detection_percent":
                (
                    float(
                        100.0
                        *
                        harmful_caught
                        /
                        harmful_total
                    )

                    if harmful_total > 0

                    else None
                )
        }


        np.save(
            os.path.join(
                OUT_ROOT,
                f"ensemble_{name.lower()}_clean_mask.npy"
            ),
            masks[
                "clean"
            ]
        )


        np.save(
            os.path.join(
                OUT_ROOT,
                f"ensemble_{name.lower()}_qr_mask.npy"
            ),
            masks[
                "qr"
            ]
        )


    with open(
        os.path.join(
            OUT_ROOT,
            "ensemble_results.json"
        ),
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )


    return results



def create_ddpm_all70_ablation(
    all_models
):

    print("\n\n")
    print("=" * 100)
    print(
        "CREATE DDPM ALL-70 QR ABLATION INPUT"
    )
    print("=" * 100)


    for model_key in MODEL_KEYS:

        model_out = os.path.join(
            DDPM_ALL_ROOT,
            model_key
        )


        if os.path.isdir(
            model_out
        ):

            shutil.rmtree(
                model_out
            )


        safe_mkdir(
            model_out
        )


        filenames = all_models[
            model_key
        ][
            "qr"
        ][
            "filenames"
        ]


        for idx, src in enumerate(
            filenames
        ):

            src = str(
                src
            )


            if not os.path.exists(
                src
            ):

                raise FileNotFoundError(
                    src
                )


            class_name = Path(
                src
            ).parent.name


            class_out = os.path.join(
                model_out,
                class_name
            )


            safe_mkdir(
                class_out
            )


            dst_name = (
                f"{idx:03d}_"
                f"{Path(src).name}"
            )


            shutil.copy2(
                src,
                os.path.join(
                    class_out,
                    dst_name
                )
            )


        print(
            f"{model_key:15s}: "
            f"{len(filenames)} images -> "
            f"{model_out}"
        )




def print_final_summary(
    all_models,
    ensemble_results
):

    print("\n\n")
    print("=" * 110)
    print(
        "FINAL RECALIBRATED HELD-OUT PHYSICAL TEST SUMMARY"
    )
    print("=" * 110)


    print(
        f"{'Model':16s}"
        f"{'Selected':12s}"
        f"{'Calib FPR':>12s}"
        f"{'Calib TPR':>12s}"
        f"{'Test FPR':>12s}"
        f"{'Test TPR':>12s}"
        f"{'Test wrong':>14s}"
    )


    print(
        "-" * 110
    )


    summary = {}


    for model_key in MODEL_KEYS:

        result = all_models[
            model_key
        ][
            "result"
        ]


        best = result[
            "best_configuration"
        ]


        chosen = result[
            "chosen_method_heldout"
        ]


        clean_test = chosen[
            "clean"
        ]


        qr_test = chosen[
            "qr"
        ]


        print(

            f"{model_key:16s}"

            f"{best['method']:12s}"

            f"{best['clean_fpr']:11.2f}%"

            f"{best['qr_tpr']:11.2f}%"

            f"{clean_test['rate_percent']:11.2f}%"

            f"{qr_test['rate_percent']:11.2f}%"

            f"{qr_test['wrong_flagged']:6d}/"
            f"{qr_test['wrong_total']:<7d}"
        )


        summary[
            model_key
        ] = {

            "selected_method":
                best[
                    "method"
                ],

            "q_old":
                best[
                    "q_old"
                ],

            "q_js":
                best[
                    "q_js"
                ],

            "weak_k":
                best[
                    "weak_k"
                ],

            "calibration_clean_fpr_percent":
                best[
                    "clean_fpr"
                ],

            "calibration_qr_tpr_percent":
                best[
                    "qr_tpr"
                ],

            "heldout_clean_fpr_percent":
                clean_test[
                    "rate_percent"
                ],

            "heldout_qr_tpr_percent":
                qr_test[
                    "rate_percent"
                ],

            "heldout_wrong_total":
                qr_test[
                    "wrong_total"
                ],

            "heldout_wrong_flagged":
                qr_test[
                    "wrong_flagged"
                ]
        }


    with open(
        os.path.join(
            OUT_ROOT,
            "FINAL_RECALIBRATED_SUMMARY.json"
        ),
        "w"
    ) as f:

        json.dump(
            {
                "per_model":
                    summary,

                "ensemble":
                    ensemble_results
            },
            f,
            indent=2
        )



def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--skip_ddpm_all70_extract",
        action="store_true",
        help=(
            "Do not create the separate all-70 QR DDPM ablation input."
        )
    )


    args = parser.parse_args()


    safe_mkdir(
        OUT_ROOT
    )


    print("=" * 110)
    print(
        "PHYSICAL DEFENSE RECALIBRATION + HELD-OUT TEST + ENSEMBLE"
    )
    print("=" * 110)


    print(
        f"Calibration FPR constraint : "
        f"<= {MAX_CALIBRATION_FPR:.2f}%"
    )


    print(
        f"Threshold grid             : "
        f"{PERCENTILE_GRID}"
    )


    print(
        f"weak_k grid                : "
        f"{WEAK_K_GRID}"
    )


    print(
        "\nIMPORTANT:"
    )

    print(
        "  50 checkpoint training samples = detector calibration"
    )

    print(
        "  20 checkpoint validation samples = untouched held-out physical test"
    )


    all_models = {}


    for model_key in MODEL_KEYS:

        all_models[
            model_key
        ] = analyze_model(
            model_key
        )


    

    ensemble_results = ensemble_analysis(
        all_models
    )


    

    if not args.skip_ddpm_all70_extract:

        create_ddpm_all70_ablation(
            all_models
        )


    

    print_final_summary(
        all_models,
        ensemble_results
    )


    print("\n")
    print("=" * 110)
    print(
        "NEW ANALYSIS COMPLETE"
    )
    print("=" * 110)


    print(
        f"\nResults:"
    )

    print(
        OUT_ROOT
    )


    if not args.skip_ddpm_all70_extract:

        print(
            "\nDDPM all-70 ablation inputs:"
        )

        print(
            DDPM_ALL_ROOT
        )


if __name__ == "__main__":

    main()

