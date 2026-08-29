

import os
import json
import argparse
from pathlib import Path

import numpy as np




ROOT = (
    "/home/Traffic_Signs_2/"
    "physical_pipeline"
)

PERCENTILE_HIGH = 95.0
PERCENTILE_LOW = 5.0


MODELS = {

    "mobilenet": {
        "clean_dir": os.path.join(
            ROOT,
            "eval_mobilenet",
            "clean"
        )
    },

    "convnext": {
        "clean_dir": os.path.join(
            ROOT,
            "eval_convnext",
            "clean"
        )
    },

    "efficientnet": {
        "clean_dir": os.path.join(
            ROOT,
            "eval_efficientnet",
            "clean"
        )
    }
}


OUT_ROOT = os.path.join(
    ROOT,
    "thresholds"
)



def load_array(
    folder,
    filename
):

    path = os.path.join(
        folder,
        filename
    )

    if not os.path.exists(
        path
    ):

        raise FileNotFoundError(
            path
        )

    return np.load(
        path,
        allow_pickle=True
    )


def finite_values(
    x
):

    x = np.asarray(
        x
    ).reshape(-1)

    return x[
        np.isfinite(x)
    ]


def percentile_or_none(
    x,
    percentile
):

    x = finite_values(
        x
    )

    if len(x) == 0:

        return None

    return float(
        np.percentile(
            x,
            percentile
        )
    )




def calibrate(
    model_key
):

    clean_dir = (
        MODELS[
            model_key
        ][
            "clean_dir"
        ]
    )


    out_dir = os.path.join(
        OUT_ROOT,
        model_key
    )


    Path(
        out_dir
    ).mkdir(
        parents=True,
        exist_ok=True
    )


    print("\n" + "=" * 80)

    print(
        f"MODEL: {model_key.upper()}"
    )

    print("=" * 80)

    print(
        "Clean signal folder:"
    )

    print(
        clean_dir
    )


  

    confidence = load_array(
        clean_dir,
        "confidence.npy"
    )

    energy = load_array(
        clean_dir,
        "energy.npy"
    )

    conf_drop_2 = load_array(
        clean_dir,
        "2pass_conf_drop.npy"
    )

    logit_l2_2 = load_array(
        clean_dir,
        "2pass_logit_l2.npy"
    )

    conf_drop_3 = load_array(
        clean_dir,
        "3pass_max_conf_drop_critical.npy"
    )

    logit_l2_3 = load_array(
        clean_dir,
        "3pass_max_logit_l2_critical.npy"
    )

    changed_3 = load_array(
        clean_dir,
        "3pass_changed_critical.npy"
    )

    critical_pred_mask = load_array(
        clean_dir,
        "critical_pred_mask.npy"
    ).astype(
        bool
    )




    n = len(
        confidence
    )


    arrays = [

        energy,
        conf_drop_2,
        logit_l2_2,
        conf_drop_3,
        logit_l2_3,
        changed_3,
        critical_pred_mask
    ]


    for arr in arrays:

        if len(arr) != n:

            raise RuntimeError(
                "Signal length mismatch."
            )


    

    valid_3 = (

        critical_pred_mask

        &

        (
            changed_3
            != -1
        )

    )


    cd3_valid = (
        conf_drop_3[
            valid_3
        ]
    )


    l23_valid = (
        logit_l2_3[
            valid_3
        ]
    )


 

    thresholds = {

        "calibration_source":
            "physical_clean_only",

        "model":
            model_key,

        "num_clean_samples":
            int(n),

        "percentile_high":
            PERCENTILE_HIGH,

        "percentile_low":
            PERCENTILE_LOW,


      

        "energy_threshold":
            percentile_or_none(
                energy,
                PERCENTILE_HIGH
            ),



        "confidence_min_threshold":
            percentile_or_none(
                confidence,
                PERCENTILE_LOW
            ),


    

        "conf_drop_2pass_threshold":
            percentile_or_none(
                conf_drop_2,
                PERCENTILE_HIGH
            ),

        "logit_l2_2pass_threshold":
            percentile_or_none(
                logit_l2_2,
                PERCENTILE_HIGH
            ),


  

        "conf_drop_3pass_threshold":
            percentile_or_none(
                cd3_valid,
                PERCENTILE_HIGH
            ),

        "logit_l2_3pass_threshold":
            percentile_or_none(
                l23_valid,
                PERCENTILE_HIGH
            )
    }


   

    stats = {

        "model":
            model_key,

        "clean_dir":
            clean_dir,

        "num_clean_samples":
            int(n),

        "num_clean_predicted_critical":
            int(
                critical_pred_mask.sum()
            ),

        "fraction_clean_predicted_critical":
            float(
                critical_pred_mask.mean()
            ),


        "clean_signal_means": {

            "energy":
                float(
                    np.mean(
                        finite_values(
                            energy
                        )
                    )
                ),

            "confidence":
                float(
                    np.mean(
                        finite_values(
                            confidence
                        )
                    )
                ),

            "conf_drop_2pass":
                float(
                    np.mean(
                        finite_values(
                            conf_drop_2
                        )
                    )
                ),

            "logit_l2_2pass":
                float(
                    np.mean(
                        finite_values(
                            logit_l2_2
                        )
                    )
                ),

            "conf_drop_3pass_critical":
                (
                    float(
                        np.mean(
                            cd3_valid
                        )
                    )
                    if len(cd3_valid) > 0
                    else None
                ),

            "logit_l2_3pass_critical":
                (
                    float(
                        np.mean(
                            l23_valid
                        )
                    )
                    if len(l23_valid) > 0
                    else None
                )
        },


        "threshold_rules": {

            "energy":
                "suspicious_if_value_gt_threshold",

            "confidence":
                "suspicious_if_value_lt_threshold",

            "conf_drop_2pass":
                "suspicious_if_value_gt_threshold",

            "logit_l2_2pass":
                "suspicious_if_value_gt_threshold",

            "conf_drop_3pass":
                "predicted-critical only; suspicious_if_value_gt_threshold",

            "logit_l2_3pass":
                "predicted-critical only; suspicious_if_value_gt_threshold"
        }
    }




    threshold_path = os.path.join(
        out_dir,
        f"{model_key}_physical_thresholds.json"
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
            stats,
            f,
            indent=2
        )


    

    print(
        f"Clean samples              : {n}"
    )

    print(
        f"Predicted critical         : "
        f"{critical_pred_mask.sum()}"
    )

    print(
        f"Energy threshold           : "
        f"{thresholds['energy_threshold']:.6f}"
    )

    print(
        f"Confidence minimum         : "
        f"{thresholds['confidence_min_threshold']:.6f}"
    )

    print(
        f"2-pass conf-drop threshold : "
        f"{thresholds['conf_drop_2pass_threshold']:.6f}"
    )

    print(
        f"2-pass logit-L2 threshold  : "
        f"{thresholds['logit_l2_2pass_threshold']:.6f}"
    )

    print(
        f"3-pass conf-drop threshold : "
        f"{thresholds['conf_drop_3pass_threshold']:.6f}"
    )

    print(
        f"3-pass logit-L2 threshold  : "
        f"{thresholds['logit_l2_3pass_threshold']:.6f}"
    )

    print(
        "\nSaved:"
    )

    print(
        threshold_path
    )

    print(
        stats_path
    )



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


    selected = (

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
        "PHYSICAL CLEAN-ONLY THRESHOLD CALIBRATION — STEP 2"
    )

    print("=" * 100)

    print(
        f"High percentile : p{PERCENTILE_HIGH:g}"
    )

    print(
        f"Low percentile  : p{PERCENTILE_LOW:g}"
    )


    for model_key in selected:

        calibrate(
            model_key
        )


    print("\n" + "=" * 100)

    print(
        "STEP 2 COMPLETE"
    )

    print("=" * 100)


if __name__ == "__main__":

    main()
