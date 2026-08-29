

import os
import json
import argparse
from pathlib import Path

import numpy as np


ROOT = (
    "/home/Traffic_Signs_2/"
    "physical_pipeline"
)

JS_ROOT = os.path.join(
    ROOT,
    "js_signal"
)

OUT_ROOT = os.path.join(
    ROOT,
    "js_analysis"
)

PERCENTILE = 95.0


MODELS = [
    "mobilenet",
    "convnext",
    "efficientnet"
]


def load(folder, name):

    path = os.path.join(
        folder,
        name
    )

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    return np.load(
        path,
        allow_pickle=True
    )


def analyze_model(model_key):

    print("\n" + "=" * 90)
    print(model_key.upper())
    print("=" * 90)


    clean_dir = os.path.join(
        JS_ROOT,
        model_key,
        "clean"
    )

    qr_dir = os.path.join(
        JS_ROOT,
        model_key,
        "qr"
    )

    out_dir = os.path.join(
        OUT_ROOT,
        model_key
    )

    Path(out_dir).mkdir(
        parents=True,
        exist_ok=True
    )




    clean_js = load(
        clean_dir,
        "js_divergence.npy"
    )

    clean_labels = load(
        clean_dir,
        "labels.npy"
    )

    clean_pred = load(
        clean_dir,
        "pred_base.npy"
    )

    clean_files = load(
        clean_dir,
        "filenames.npy"
    )



    qr_js = load(
        qr_dir,
        "js_divergence.npy"
    )

    qr_labels = load(
        qr_dir,
        "labels.npy"
    )

    qr_pred = load(
        qr_dir,
        "pred_base.npy"
    )

    qr_files = load(
        qr_dir,
        "filenames.npy"
    )




    threshold = float(
        np.percentile(
            clean_js,
            PERCENTILE
        )
    )


    

    clean_mask = (
        clean_js
        >
        threshold
    )

    qr_mask = (
        qr_js
        >
        threshold
    )



    clean_fpr = (
        clean_mask.mean()
        *
        100.0
    )

    qr_tpr = (
        qr_mask.mean()
        *
        100.0
    )



    clean_wrong = (
        clean_pred
        !=
        clean_labels
    )

    qr_wrong = (
        qr_pred
        !=
        qr_labels
    )


    clean_wrong_total = int(
        clean_wrong.sum()
    )

    qr_wrong_total = int(
        qr_wrong.sum()
    )


    clean_wrong_flagged = int(
        (
            clean_mask
            &
            clean_wrong
        ).sum()
    )


    qr_wrong_flagged = int(
        (
            qr_mask
            &
            qr_wrong
        ).sum()
    )


    qr_wrong_detection = (

        qr_wrong_flagged
        /
        qr_wrong_total
        *
        100.0

        if qr_wrong_total > 0
        else 0.0
    )


    qr_correct = (
        qr_pred
        ==
        qr_labels
    )

    qr_correct_flagged = int(
        (
            qr_mask
            &
            qr_correct
        ).sum()
    )




    np.save(
        os.path.join(
            out_dir,
            "clean_js_mask.npy"
        ),
        clean_mask
    )


    np.save(
        os.path.join(
            out_dir,
            "qr_js_mask.npy"
        ),
        qr_mask
    )


    np.save(
        os.path.join(
            out_dir,
            "clean_filenames.npy"
        ),
        clean_files,
        allow_pickle=True
    )


    np.save(
        os.path.join(
            out_dir,
            "qr_filenames.npy"
        ),
        qr_files,
        allow_pickle=True
    )


    result = {

        "model":
            model_key,

        "percentile":
            PERCENTILE,

        "js_threshold":
            threshold,

        "clean_n":
            int(
                len(clean_js)
            ),

        "qr_n":
            int(
                len(qr_js)
            ),

        "clean_fpr_percent":
            float(
                clean_fpr
            ),

        "qr_tpr_percent":
            float(
                qr_tpr
            ),

        "clean_wrong_total":
            clean_wrong_total,

        "clean_wrong_flagged":
            clean_wrong_flagged,

        "qr_wrong_total":
            qr_wrong_total,

        "qr_wrong_flagged":
            qr_wrong_flagged,

        "qr_wrong_detection_percent":
            float(
                qr_wrong_detection
            ),

        "qr_correct_flagged":
            qr_correct_flagged,

        "clean_js_mean":
            float(
                clean_js.mean()
            ),

        "clean_js_median":
            float(
                np.median(
                    clean_js
                )
            ),

        "clean_js_p95":
            float(
                np.percentile(
                    clean_js,
                    95
                )
            ),

        "qr_js_mean":
            float(
                qr_js.mean()
            ),

        "qr_js_median":
            float(
                np.median(
                    qr_js
                )
            ),

        "qr_js_p95":
            float(
                np.percentile(
                    qr_js,
                    95
                )
            )
    }


    with open(
        os.path.join(
            out_dir,
            "js_results.json"
        ),
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=2
        )




    print(
        f"JS threshold       : "
        f"{threshold:.8f}"
    )

    print(
        f"Clean FPR          : "
        f"{clean_fpr:.2f}% "
        f"({clean_mask.sum()}/{len(clean_mask)})"
    )

    print(
        f"QR TPR             : "
        f"{qr_tpr:.2f}% "
        f"({qr_mask.sum()}/{len(qr_mask)})"
    )

    print(
        f"QR wrong total     : "
        f"{qr_wrong_total}"
    )

    print(
        f"QR wrong caught    : "
        f"{qr_wrong_flagged}/{qr_wrong_total}"
    )

    print(
        f"Wrong detection    : "
        f"{qr_wrong_detection:.2f}%"
    )

    print(
        f"QR correct flagged : "
        f"{qr_correct_flagged}"
    )

    print(
        f"Clean JS mean      : "
        f"{clean_js.mean():.8f}"
    )

    print(
        f"QR JS mean         : "
        f"{qr_js.mean():.8f}"
    )


    return result


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "all",
            *MODELS
        ],
        default="all"
    )

    args = parser.parse_args()


    selected = (
        MODELS
        if args.model == "all"
        else [args.model]
    )


    Path(
        OUT_ROOT
    ).mkdir(
        parents=True,
        exist_ok=True
    )


    print("=" * 100)
    print(
        "PHYSICAL JS CLEAN-THRESHOLD ANALYSIS — STEP 5"
    )
    print("=" * 100)

    print(
        f"Clean percentile: p{PERCENTILE:g}"
    )


    all_results = {}


    for model_key in selected:

        all_results[
            model_key
        ] = analyze_model(
            model_key
        )


    print("\n")
    print("=" * 100)
    print(
        "JS DETECTOR SUMMARY"
    )
    print("=" * 100)


    print(
        f"{'Model':18s}"
        f"{'Clean FPR':>14s}"
        f"{'QR TPR':>14s}"
        f"{'QR Wrong':>16s}"
    )

    print("-" * 100)


    for model_key in selected:

        r = all_results[
            model_key
        ]

        print(
            f"{model_key:18s}"
            f"{r['clean_fpr_percent']:13.2f}%"
            f"{r['qr_tpr_percent']:13.2f}%"
            f"{r['qr_wrong_flagged']:7d}/"
            f"{r['qr_wrong_total']:<7d}"
        )


    with open(
        os.path.join(
            OUT_ROOT,
            "js_summary.json"
        ),
        "w"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2
        )


    print("\n" + "=" * 100)
    print(
        "STEP 5 COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":

    main()
