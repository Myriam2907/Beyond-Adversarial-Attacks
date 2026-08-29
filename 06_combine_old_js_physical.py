

import os
import json
import argparse
from pathlib import Path

import numpy as np


ROOT = (
    "/home/Traffic_Signs_2/"
    "physical_pipeline"
)

MODELS = [
    "mobilenet",
    "convnext",
    "efficientnet"
]


OLD_ROOT = os.path.join(
    ROOT,
    "old_detector"
)

JS_ROOT = os.path.join(
    ROOT,
    "js_analysis"
)

EVAL_ROOTS = {
    "mobilenet": os.path.join(
        ROOT,
        "eval_mobilenet"
    ),
    "convnext": os.path.join(
        ROOT,
        "eval_convnext"
    ),
    "efficientnet": os.path.join(
        ROOT,
        "eval_efficientnet"
    )
}

OUT_ROOT = os.path.join(
    ROOT,
    "combined_detector"
)


def load(path):

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    return np.load(
        path,
        allow_pickle=True
    )


def metrics(
    mask,
    label,
    pred
):

    correct = (
        pred == label
    )

    wrong = (
        ~correct
    )

    n = len(mask)

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

    correct_flagged = int(
        (
            mask
            &
            correct
        ).sum()
    )

    wrong_detection = (

        100.0
        *
        wrong_flagged
        /
        wrong_total

        if wrong_total > 0

        else 0.0
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
            float(
                wrong_detection
            ),

        "correct_flagged":
            correct_flagged
    }


def analyze_condition(
    model_key,
    condition
):

    

    old_dir = os.path.join(
        OLD_ROOT,
        model_key,
        condition
    )

    old_mask = load(
        os.path.join(
            old_dir,
            "suspicious_mask.npy"
        )
    ).astype(bool)


  

    js_dir = os.path.join(
        JS_ROOT,
        model_key
    )

    js_name = (
        "clean_js_mask.npy"
        if condition == "clean"
        else "qr_js_mask.npy"
    )

    js_mask = load(
        os.path.join(
            js_dir,
            js_name
        )
    ).astype(bool)


    

    eval_dir = os.path.join(
        EVAL_ROOTS[
            model_key
        ],
        condition
    )

    label = load(
        os.path.join(
            eval_dir,
            "label.npy"
        )
    )

    pred = load(
        os.path.join(
            eval_dir,
            "pred.npy"
        )
    )

    filenames = load(
        os.path.join(
            eval_dir,
            "filenames.npy"
        )
    )



    n = len(label)

    if not (
        len(old_mask)
        ==
        len(js_mask)
        ==
        len(pred)
        ==
        len(filenames)
        ==
        n
    ):

        raise RuntimeError(
            f"Length mismatch: "
            f"{model_key}/{condition}"
        )



    or_mask = (
        old_mask
        |
        js_mask
    )

    and_mask = (
        old_mask
        &
        js_mask
    )


    old_only = (
        old_mask
        &
        (~js_mask)
    )

    js_only = (
        js_mask
        &
        (~old_mask)
    )

    both = (
        old_mask
        &
        js_mask
    )

    neither = (
        (~old_mask)
        &
        (~js_mask)
    )


    

    results = {

        "old":
            metrics(
                old_mask,
                label,
                pred
            ),

        "js":
            metrics(
                js_mask,
                label,
                pred
            ),

        "or":
            metrics(
                or_mask,
                label,
                pred
            ),

        "and":
            metrics(
                and_mask,
                label,
                pred
            ),

        "overlap": {

            "old_only":
                int(
                    old_only.sum()
                ),

            "js_only":
                int(
                    js_only.sum()
                ),

            "both":
                int(
                    both.sum()
                ),

            "neither":
                int(
                    neither.sum()
                )
        }
    }



    out_dir = os.path.join(
        OUT_ROOT,
        model_key,
        condition
    )

    Path(
        out_dir
    ).mkdir(
        parents=True,
        exist_ok=True
    )


    np.save(
        os.path.join(
            out_dir,
            "old_mask.npy"
        ),
        old_mask
    )

    np.save(
        os.path.join(
            out_dir,
            "js_mask.npy"
        ),
        js_mask
    )

    np.save(
        os.path.join(
            out_dir,
            "combined_or_mask.npy"
        ),
        or_mask
    )

    np.save(
        os.path.join(
            out_dir,
            "combined_and_mask.npy"
        ),
        and_mask
    )

    np.save(
        os.path.join(
            out_dir,
            "old_only_mask.npy"
        ),
        old_only
    )

    np.save(
        os.path.join(
            out_dir,
            "js_only_mask.npy"
        ),
        js_only
    )

    np.save(
        os.path.join(
            out_dir,
            "both_mask.npy"
        ),
        both
    )

    np.save(
        os.path.join(
            out_dir,
            "filenames.npy"
        ),
        filenames,
        allow_pickle=True
    )


    with open(
        os.path.join(
            out_dir,
            "comparison_results.json"
        ),
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=2
        )



    print("\n" + "=" * 90)

    print(
        f"{model_key.upper()} — "
        f"{condition.upper()}"
    )

    print("=" * 90)


    print(
        f"{'Detector':12s}"
        f"{'Flagged':>12s}"
        f"{'Rate':>12s}"
        f"{'Wrong caught':>18s}"
        f"{'Wrong det.':>14s}"
    )

    print("-" * 90)


    for name in [
        "old",
        "js",
        "or",
        "and"
    ]:

        r = results[
            name
        ]

        print(
            f"{name.upper():12s}"
            f"{r['flagged']:6d}/{r['n']:<5d}"
            f"{r['rate_percent']:11.2f}%"
            f"{r['wrong_flagged']:8d}/{r['wrong_total']:<8d}"
            f"{r['wrong_detection_percent']:13.2f}%"
        )


    print(
        "\nOverlap:"
    )

    print(
        f"  OLD only : "
        f"{results['overlap']['old_only']}"
    )

    print(
        f"  JS only  : "
        f"{results['overlap']['js_only']}"
    )

    print(
        f"  BOTH     : "
        f"{results['overlap']['both']}"
    )

    print(
        f"  NEITHER  : "
        f"{results['overlap']['neither']}"
    )


    return results


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
        else [
            args.model
        ]
    )


    print("=" * 100)

    print(
        "PHYSICAL OLD + JS COMBINATION — STEP 6"
    )

    print("=" * 100)


    all_results = {}


    for model_key in selected:

        all_results[
            model_key
        ] = {}

        for condition in [
            "clean",
            "qr"
        ]:

            all_results[
                model_key
            ][
                condition
            ] = analyze_condition(
                model_key,
                condition
            )



    print("\n\n")
    print("=" * 100)

    print(
        "FINAL PHYSICAL COMBINED DETECTOR SUMMARY"
    )

    print("=" * 100)


    print(
        f"{'Model':16s}"
        f"{'Method':10s}"
        f"{'Clean FPR':>14s}"
        f"{'QR TPR':>14s}"
        f"{'QR Wrong':>16s}"
    )

    print("-" * 100)


    for model_key in selected:

        for method in [
            "old",
            "js",
            "or",
            "and"
        ]:

            clean = (
                all_results[
                    model_key
                ][
                    "clean"
                ][
                    method
                ]
            )

            qr = (
                all_results[
                    model_key
                ][
                    "qr"
                ][
                    method
                ]
            )


            print(
                f"{model_key:16s}"
                f"{method.upper():10s}"
                f"{clean['rate_percent']:13.2f}%"
                f"{qr['rate_percent']:13.2f}%"
                f"{qr['wrong_flagged']:7d}/"
                f"{qr['wrong_total']:<7d}"
            )


    Path(
        OUT_ROOT
    ).mkdir(
        parents=True,
        exist_ok=True
    )


    with open(
        os.path.join(
            OUT_ROOT,
            "combined_summary.json"
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
        "STEP 6 COMPLETE"
    )

    print("=" * 100)


if __name__ == "__main__":
    main()
