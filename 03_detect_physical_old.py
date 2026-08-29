

import os
import json
import argparse
from pathlib import Path

import numpy as np




ROOT = (
    "/home/Traffic_Signs_2/"
    "physical_pipeline"
)

WEAK_K = 3


MODELS = {

    "mobilenet": {
        "eval_root": os.path.join(
            ROOT,
            "eval_mobilenet"
        ),
        "threshold_file": os.path.join(
            ROOT,
            "thresholds",
            "mobilenet",
            "mobilenet_physical_thresholds.json"
        ),
        "out_root": os.path.join(
            ROOT,
            "old_detector",
            "mobilenet"
        )
    },

    "convnext": {
        "eval_root": os.path.join(
            ROOT,
            "eval_convnext"
        ),
        "threshold_file": os.path.join(
            ROOT,
            "thresholds",
            "convnext",
            "convnext_physical_thresholds.json"
        ),
        "out_root": os.path.join(
            ROOT,
            "old_detector",
            "convnext"
        )
    },

    "efficientnet": {
        "eval_root": os.path.join(
            ROOT,
            "eval_efficientnet"
        ),
        "threshold_file": os.path.join(
            ROOT,
            "thresholds",
            "efficientnet",
            "efficientnet_physical_thresholds.json"
        ),
        "out_root": os.path.join(
            ROOT,
            "old_detector",
            "efficientnet"
        )
    }
}


CONDITIONS = [
    "clean",
    "qr"
]




def load_json(path):

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r") as f:
        return json.load(f)


def load_array(folder, filename):

    path = os.path.join(
        folder,
        filename
    )

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    return np.load(
        path,
        allow_pickle=True
    )



def detect_condition(
    model_key,
    condition,
    cfg,
    thresholds
):

    eval_dir = os.path.join(
        cfg["eval_root"],
        condition
    )

    out_dir = os.path.join(
        cfg["out_root"],
        condition
    )

    Path(out_dir).mkdir(
        parents=True,
        exist_ok=True
    )


    print("\n" + "=" * 90)

    print(
        f"{model_key.upper()} — {condition.upper()}"
    )

    print("=" * 90)

    print(
        f"Signals : {eval_dir}"
    )




    label = load_array(
        eval_dir,
        "label.npy"
    )

    pred = load_array(
        eval_dir,
        "pred.npy"
    )

    confidence = load_array(
        eval_dir,
        "confidence.npy"
    )

    energy = load_array(
        eval_dir,
        "energy.npy"
    )

    conf_drop_2 = load_array(
        eval_dir,
        "2pass_conf_drop.npy"
    )

    logit_l2_2 = load_array(
        eval_dir,
        "2pass_logit_l2.npy"
    )

    changed_2 = load_array(
        eval_dir,
        "2pass_changed.npy"
    )

    conf_drop_3 = load_array(
        eval_dir,
        "3pass_max_conf_drop_critical.npy"
    )

    logit_l2_3 = load_array(
        eval_dir,
        "3pass_max_logit_l2_critical.npy"
    )

    changed_3 = load_array(
        eval_dir,
        "3pass_changed_critical.npy"
    )

    critical_pred_mask = load_array(
        eval_dir,
        "critical_pred_mask.npy"
    ).astype(bool)

    filenames = load_array(
        eval_dir,
        "filenames.npy"
    )


    n = len(label)



    arrays = [
        pred,
        confidence,
        energy,
        conf_drop_2,
        logit_l2_2,
        changed_2,
        conf_drop_3,
        logit_l2_3,
        changed_3,
        critical_pred_mask,
        filenames
    ]


    for arr in arrays:

        if len(arr) != n:

            raise RuntimeError(
                "Length mismatch in detector inputs."
            )


   

   
    flag_energy = (
        energy
        >
        thresholds[
            "energy_threshold"
        ]
    )


    
    flag_confidence = (
        confidence
        <
        thresholds[
            "confidence_min_threshold"
        ]
    )


    
    flag_conf_drop_2 = (
        conf_drop_2
        >
        thresholds[
            "conf_drop_2pass_threshold"
        ]
    )


  
    flag_logit_l2_2 = (
        logit_l2_2
        >
        thresholds[
            "logit_l2_2pass_threshold"
        ]
    )


  
    flag_changed_2 = (
        changed_2 == 1
    )


   

    valid_3 = (
        critical_pred_mask
        &
        (changed_3 != -1)
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


    if valid_3.any():

        flag_conf_drop_3[
            valid_3
        ] = (
            conf_drop_3[
                valid_3
            ]
            >
            thresholds[
                "conf_drop_3pass_threshold"
            ]
        )


        flag_logit_l2_3[
            valid_3
        ] = (
            logit_l2_3[
                valid_3
            ]
            >
            thresholds[
                "logit_l2_3pass_threshold"
            ]
        )


        flag_changed_3[
            valid_3
        ] = (
            changed_3[
                valid_3
            ]
            == 1
        )


    strong_signals = (

        flag_changed_2

        |

        flag_changed_3

        |

        flag_logit_l2_2

    )


    -

    weak_count = (

        flag_energy.astype(int)

        +

        flag_confidence.astype(int)

        +

        flag_conf_drop_2.astype(int)

        +

        flag_conf_drop_3.astype(int)

        +

        flag_logit_l2_3.astype(int)

    )



    suspicious = (

        strong_signals

        |

        (
            weak_count
            >=
            WEAK_K
        )

    )


    flag_count = (

        flag_energy.astype(int)

        +

        flag_confidence.astype(int)

        +

        flag_conf_drop_2.astype(int)

        +

        flag_logit_l2_2.astype(int)

        +

        flag_changed_2.astype(int)

        +

        flag_conf_drop_3.astype(int)

        +

        flag_logit_l2_3.astype(int)

        +

        flag_changed_3.astype(int)

    )




    correct = (
        pred == label
    )

    wrong = (
        ~correct
    )


    n_suspicious = int(
        suspicious.sum()
    )

    suspicious_rate = (
        n_suspicious
        /
        n
        *
        100.0
    )


    accuracy = (
        correct.mean()
        *
        100.0
    )


    wrong_total = int(
        wrong.sum()
    )


    wrong_flagged = int(
        (
            suspicious
            &
            wrong
        ).sum()
    )


    wrong_not_flagged = int(
        (
            (~suspicious)
            &
            wrong
        ).sum()
    )


    correct_flagged = int(
        (
            suspicious
            &
            correct
        ).sum()
    )


    correct_not_flagged = int(
        (
            (~suspicious)
            &
            correct
        ).sum()
    )


    detection_rate_wrong = (

        wrong_flagged
        /
        wrong_total
        *
        100.0

        if wrong_total > 0

        else 0.0
    )


    

    is_clean = (
        condition == "clean"
    )


    fpr_clean = (
        suspicious_rate
        if is_clean
        else None
    )


    tpr_attack = (
        None
        if is_clean
        else suspicious_rate
    )



    weak_trigger = (
        weak_count
        >=
        WEAK_K
    )


    strong_only = int(
        (
            strong_signals
            &
            (~weak_trigger)
        ).sum()
    )


    weak_only = int(
        (
            (~strong_signals)
            &
            weak_trigger
        ).sum()
    )


    both = int(
        (
            strong_signals
            &
            weak_trigger
        ).sum()
    )




    results = {

        "model":
            model_key,

        "condition":
            condition,

        "n_samples":
            int(n),

        "accuracy_percent":
            float(accuracy),

        "weak_k":
            int(WEAK_K),

        "detection_strategy":
            (
                "tiered: "
                "strong_signal OR weak_count>=3"
            ),



        "n_suspicious":
            n_suspicious,

        "suspicious_rate_percent":
            float(
                suspicious_rate
            ),

        "fpr_clean_percent":
            (
                float(fpr_clean)
                if fpr_clean is not None
                else None
            ),

        "tpr_attack_percent":
            (
                float(tpr_attack)
                if tpr_attack is not None
                else None
            ),


     

        "wrong_total":
            wrong_total,

        "wrong_flagged":
            wrong_flagged,

        "wrong_not_flagged":
            wrong_not_flagged,

        "detection_rate_of_wrong_percent":
            float(
                detection_rate_wrong
            ),

        "correct_flagged":
            correct_flagged,

        "correct_not_flagged":
            correct_not_flagged,




        "predicted_critical_samples":
            int(
                critical_pred_mask.sum()
            ),

        "valid_3pass_samples":
            int(
                valid_3.sum()
            ),



        "flags": {

            "energy":
                int(
                    flag_energy.sum()
                ),

            "confidence":
                int(
                    flag_confidence.sum()
                ),

            "conf_drop_2pass":
                int(
                    flag_conf_drop_2.sum()
                ),

            "logit_l2_2pass":
                int(
                    flag_logit_l2_2.sum()
                ),

            "changed_2pass":
                int(
                    flag_changed_2.sum()
                ),

            "conf_drop_3pass":
                int(
                    flag_conf_drop_3.sum()
                ),

            "logit_l2_3pass":
                int(
                    flag_logit_l2_3.sum()
                ),

            "changed_3pass":
                int(
                    flag_changed_3.sum()
                )
        },


        

        "tiered_breakdown": {

            "strong_only":
                strong_only,

            "weak_only":
                weak_only,

            "both":
                both
        },



        "flag_statistics": {

            "average_flags_per_sample":
                float(
                    flag_count.mean()
                ),

            "samples_with_0_flags":
                int(
                    (
                        flag_count == 0
                    ).sum()
                ),

            "samples_with_1_flag":
                int(
                    (
                        flag_count == 1
                    ).sum()
                ),

            "samples_with_2_flags":
                int(
                    (
                        flag_count == 2
                    ).sum()
                ),

            "samples_with_3_flags":
                int(
                    (
                        flag_count == 3
                    ).sum()
                ),

            "samples_with_4plus_flags":
                int(
                    (
                        flag_count >= 4
                    ).sum()
                )
        }
    }



    np.save(
        os.path.join(
            out_dir,
            "suspicious_mask.npy"
        ),
        suspicious
    )


    np.save(
        os.path.join(
            out_dir,
            "strong_signal_mask.npy"
        ),
        strong_signals
    )


    np.save(
        os.path.join(
            out_dir,
            "weak_count.npy"
        ),
        weak_count
    )


    np.save(
        os.path.join(
            out_dir,
            "flag_energy.npy"
        ),
        flag_energy
    )


    np.save(
        os.path.join(
            out_dir,
            "flag_confidence.npy"
        ),
        flag_confidence
    )


    np.save(
        os.path.join(
            out_dir,
            "flag_conf_drop_2pass.npy"
        ),
        flag_conf_drop_2
    )


    np.save(
        os.path.join(
            out_dir,
            "flag_logit_l2_2pass.npy"
        ),
        flag_logit_l2_2
    )


    np.save(
        os.path.join(
            out_dir,
            "flag_changed_2pass.npy"
        ),
        flag_changed_2
    )


    np.save(
        os.path.join(
            out_dir,
            "flag_conf_drop_3pass.npy"
        ),
        flag_conf_drop_3
    )


    np.save(
        os.path.join(
            out_dir,
            "flag_logit_l2_3pass.npy"
        ),
        flag_logit_l2_3
    )


    np.save(
        os.path.join(
            out_dir,
            "flag_changed_3pass.npy"
        ),
        flag_changed_3
    )



    np.save(
        os.path.join(
            out_dir,
            "filenames.npy"
        ),
        filenames,
        allow_pickle=True
    )


    np.save(
        os.path.join(
            out_dir,
            "label.npy"
        ),
        label
    )


    np.save(
        os.path.join(
            out_dir,
            "pred.npy"
        ),
        pred
    )


    

    result_path = os.path.join(
        out_dir,
        "detector_results.json"
    )


    with open(
        result_path,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )


   

    print(
        f"\nSamples            : {n}"
    )

    print(
        f"Classifier accuracy: "
        f"{accuracy:.2f}%"
    )


    print(
        f"\nSuspicious         : "
        f"{n_suspicious}/{n} "
        f"({suspicious_rate:.2f}%)"
    )


    if is_clean:

        print(
            f"CLEAN FPR          : "
            f"{fpr_clean:.2f}%"
        )

    else:

        print(
            f"QR TPR             : "
            f"{tpr_attack:.2f}%"
        )


    print(
        f"\nWrong predictions  : "
        f"{wrong_total}"
    )

    print(
        f"Wrong caught       : "
        f"{wrong_flagged}"
    )

    print(
        f"Wrong missed       : "
        f"{wrong_not_flagged}"
    )

    print(
        f"Detection of wrong : "
        f"{detection_rate_wrong:.2f}%"
    )


    print(
        "\nIndividual signal flags:"
    )


    for name, count in results[
        "flags"
    ].items():

        print(
            f"  {name:22s}: "
            f"{count:3d}/{n} "
            f"({100.0*count/n:6.2f}%)"
        )


    print(
        "\nTiered contribution:"
    )

    print(
        f"  Strong only : "
        f"{strong_only}"
    )

    print(
        f"  Weak only   : "
        f"{weak_only}"
    )

    print(
        f"  Both        : "
        f"{both}"
    )


    print(
        f"\nSaved -> {out_dir}"
    )


    return results




def run_model(
    model_key
):

    cfg = MODELS[
        model_key
    ]


    thresholds = load_json(
        cfg[
            "threshold_file"
        ]
    )


    print("\n\n")
    print("#" * 100)

    print(
        f"MODEL: {model_key.upper()}"
    )

    print("#" * 100)


    print(
        f"Thresholds: "
        f"{cfg['threshold_file']}"
    )


    model_results = {}


    for condition in CONDITIONS:

        result = detect_condition(
            model_key,
            condition,
            cfg,
            thresholds
        )

        model_results[
            condition
        ] = result


    return model_results




def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--model",
        choices=[
            "all",
            "mobilenet",
            "convnext",
            "efficientnet"
        ],
        default="all"
    )


    args = parser.parse_args()


    selected_models = (

        list(
            MODELS.keys()
        )

        if args.model == "all"

        else [
            args.model
        ]
    )


    print("=" * 100)

    print(
        "PHYSICAL OLD MULTI-SIGNAL DETECTOR — STEP 3"
    )

    print("=" * 100)


    print(
        f"weak_k = {WEAK_K}"
    )


    all_results = {}


    for model_key in selected_models:

        all_results[
            model_key
        ] = run_model(
            model_key
        )



    print("\n\n")
    print("=" * 100)

    print(
        "FINAL OLD DETECTOR COMPARISON"
    )

    print("=" * 100)


    print(
        f"{'Model':18s} "
        f"{'Clean FPR':>14s} "
        f"{'QR TPR':>14s} "
        f"{'QR Wrong Caught':>18s}"
    )

    print("-" * 100)


    summary = {}


    for model_key in selected_models:

        clean_result = (
            all_results[
                model_key
            ][
                "clean"
            ]
        )

        qr_result = (
            all_results[
                model_key
            ][
                "qr"
            ]
        )


        print(
            f"{model_key:18s} "
            f"{clean_result['fpr_clean_percent']:13.2f}% "
            f"{qr_result['tpr_attack_percent']:13.2f}% "
            f"{qr_result['wrong_flagged']:8d}/"
            f"{qr_result['wrong_total']:<8d}"
        )


        summary[
            model_key
        ] = {

            "clean_fpr_percent":
                clean_result[
                    "fpr_clean_percent"
                ],

            "qr_tpr_percent":
                qr_result[
                    "tpr_attack_percent"
                ],

            "qr_classifier_accuracy":
                qr_result[
                    "accuracy_percent"
                ],

            "qr_wrong_total":
                qr_result[
                    "wrong_total"
                ],

            "qr_wrong_flagged":
                qr_result[
                    "wrong_flagged"
                ],

            "qr_wrong_detection_percent":
                qr_result[
                    "detection_rate_of_wrong_percent"
                ]
        }


    summary_path = os.path.join(
        ROOT,
        "old_detector",
        "old_detector_summary.json"
    )


    Path(
        os.path.dirname(
            summary_path
        )
    ).mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        summary_path,
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )


    print(
        f"\nSummary saved -> "
        f"{summary_path}"
    )


    print("\n" + "=" * 100)

    print(
        "STEP 3 COMPLETE"
    )

    print("=" * 100)


if __name__ == "__main__":

    main()
