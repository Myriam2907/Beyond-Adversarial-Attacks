

import os
import re
import csv
import json
import shutil
import argparse
from pathlib import Path

import numpy as np




ROOT = "./gtsrb_repeat"

ATTACK_ROOT = os.path.join(
    ROOT,
    "attacks",
)

DETECTION_ROOT = os.path.join(
    ROOT,
    "fresh_detection",
)

OUTPUT_ROOT = os.path.join(
    ROOT,
    "suspicious_ours_js",
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


GRADIENT_ATTACKS = {
    "fgsm",
    "rfgsm",
    "pgd",
}


SHARED_FOLDER_MAP = {
    "clean": "clean_png",
    "patch": "patch_png",
    "random_patch": "random_patch_png",
    "gaussian_noise": "gaussian_noise_png",
    "salt_pepper": "salt_pepper_png",
    "light": "light_png",
    "fog": "fog_png",
    "motion_blur": "motion_blur_png",
}


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".ppm",
    ".webp",
}



FILENAME_PATTERN = re.compile(
    r"^(?P<sample_id>\d+)_y(?P<label>\d+)"
)



def get_source_directory(
    model,
    condition,
):
    """
    Return the exact image source corresponding to this model/condition.
    """

    if condition in GRADIENT_ATTACKS:

        return os.path.join(
            ATTACK_ROOT,
            model,
            f"{condition}_png",
        )

    if condition not in SHARED_FOLDER_MAP:

        raise ValueError(
            f"Unknown condition: {condition}"
        )

    return os.path.join(
        ATTACK_ROOT,
        SHARED_FOLDER_MAP[
            condition
        ],
    )


def get_detection_directory(
    model,
    condition,
):
    return os.path.join(
        DETECTION_ROOT,
        model,
        condition,
    )



def load_detection_arrays(
    model,
    condition,
):
    directory = get_detection_directory(
        model,
        condition,
    )

    required = {
        "sample_id": "sample_id.npy",
        "label": "label.npy",
        "pred": "pred.npy",
        "ours": "ours_mask.npy",
        "js": "js_mask.npy",
        "ours_js": "ours_js_mask.npy",
    }

    arrays = {}

    for key, filename in required.items():

        path = os.path.join(
            directory,
            filename,
        )

        if not os.path.isfile(
            path
        ):
            raise FileNotFoundError(
                f"\nMissing detector output:\n"
                f"{path}\n\n"
                "Make sure fresh_gtsrb_detection.py "
                "completed successfully."
            )

        arrays[
            key
        ] = np.load(
            path,
            allow_pickle=False,
        )

    
    arrays["sample_id"] = (
        arrays["sample_id"]
        .astype(
            np.int64
        )
    )

    arrays["label"] = (
        arrays["label"]
        .astype(
            np.int64
        )
    )

    arrays["pred"] = (
        arrays["pred"]
        .astype(
            np.int64
        )
    )

    arrays["ours"] = (
        arrays["ours"]
        .astype(
            bool
        )
    )

    arrays["js"] = (
        arrays["js"]
        .astype(
            bool
        )
    )

    arrays["ours_js"] = (
        arrays["ours_js"]
        .astype(
            bool
        )
    )

    lengths = {
        key: len(
            value
        )
        for key, value
        in arrays.items()
    }

    if len(
        set(
            lengths.values()
        )
    ) != 1:

        raise RuntimeError(
            f"{model}/{condition}: "
            f"detector array lengths do not match:\n"
            f"{lengths}"
        )

   
    expected_union = (
        arrays[
            "ours"
        ]
        |
        arrays[
            "js"
        ]
    )

    if not np.array_equal(
        expected_union,
        arrays[
            "ours_js"
        ],
    ):
        raise RuntimeError(
            f"\n{model}/{condition}: "
            "ours_js_mask.npy is NOT equal to "
            "ours_mask OR js_mask.\n"
            "Stopping to avoid extracting incorrect samples."
        )

    return arrays



def index_source_images(
    source_directory,
):
    """
    Build mapping:

        sample_id -> {
            path,
            filename,
            label
        }

    Ground-truth class is also parsed from filename.
    """

    if not os.path.isdir(
        source_directory
    ):

        raise FileNotFoundError(
            f"Source image directory missing:\n"
            f"{source_directory}"
        )

    index = {}

    for current_root, _, files in os.walk(
        source_directory
    ):

        for filename in files:

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension not in IMAGE_EXTENSIONS:
                continue

            match = FILENAME_PATTERN.match(
                filename
            )

            if match is None:

                raise RuntimeError(
                    "\nCould not parse source filename:\n"
                    f"{filename}\n\n"
                    "Expected something like:\n"
                    "00000_y16.png"
                )

            sample_id = int(
                match.group(
                    "sample_id"
                )
            )

            label = int(
                match.group(
                    "label"
                )
            )

            full_path = os.path.join(
                current_root,
                filename,
            )

            if sample_id in index:

                raise RuntimeError(
                    f"Duplicate sample ID {sample_id} "
                    f"in {source_directory}"
                )

            index[
                sample_id
            ] = {
                "path": full_path,
                "filename": filename,
                "label": label,
            }

    if len(
        index
    ) == 0:

        raise RuntimeError(
            f"No images found under:\n"
            f"{source_directory}"
        )

    return index



def prepare_output_directory(
    model,
    condition,
    overwrite,
):
    condition_output = os.path.join(
        OUTPUT_ROOT,
        model,
        condition,
    )

    image_output = os.path.join(
        condition_output,
        "images",
    )

    if os.path.isdir(
        condition_output
    ):

        if overwrite:

            shutil.rmtree(
                condition_output
            )

        else:

            raise RuntimeError(
                f"\nOutput already exists:\n"
                f"{condition_output}\n\n"
                "Use --overwrite if you intentionally "
                "want to replace it."
            )

    Path(
        image_output
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        condition_output,
        image_output,
    )



def extract_condition(
    model,
    condition,
    overwrite=False,
):
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

    source_directory = get_source_directory(
        model,
        condition,
    )

    print(
        f"Source       : "
        f"{source_directory}"
    )

   
    arrays = load_detection_arrays(
        model,
        condition,
    )

    n = len(
        arrays[
            "sample_id"
        ]
    )

   
    source_index = index_source_images(
        source_directory
    )

    print(
        f"Source images: "
        f"{len(source_index)}"
    )

    
    if len(
        source_index
    ) != n:

        raise RuntimeError(
            f"\n{model}/{condition}: "
            "number of detector samples does not match "
            "source image count.\n"
            f"Detector samples: {n}\n"
            f"Source images   : {len(source_index)}"
        )

    for sample_id, expected_label in zip(
        arrays[
            "sample_id"
        ],
        arrays[
            "label"
        ],
    ):

        sample_id = int(
            sample_id
        )

        expected_label = int(
            expected_label
        )

        if sample_id not in source_index:

            raise RuntimeError(
                f"{model}/{condition}: "
                f"source image for sample ID "
                f"{sample_id} not found."
            )

        filename_label = source_index[
            sample_id
        ][
            "label"
        ]

        if filename_label != expected_label:

            raise RuntimeError(
                f"\nLabel mismatch for "
                f"{model}/{condition} "
                f"sample {sample_id}:\n"
                f"detector label = {expected_label}\n"
                f"filename label = {filename_label}"
            )

    
    final_mask = arrays[
        "ours_js"
    ]

    suspicious_indices = np.flatnonzero(
        final_mask
    )

    num_suspicious = len(
        suspicious_indices
    )

    print(
        f"Detected      : "
        f"{num_suspicious}/{n} "
        f"({100.0 * num_suspicious / n:.2f}%)"
    )

   
    (
        condition_output,
        image_output,
    ) = prepare_output_directory(
        model,
        condition,
        overwrite,
    )

    metadata_rows = []

    copied = 0

    
    for detector_index in suspicious_indices:

        detector_index = int(
            detector_index
        )

        sample_id = int(
            arrays[
                "sample_id"
            ][
                detector_index
            ]
        )

        label = int(
            arrays[
                "label"
            ][
                detector_index
            ]
        )

        pred = int(
            arrays[
                "pred"
            ][
                detector_index
            ]
        )

        source_info = source_index[
            sample_id
        ]

        source_path = source_info[
            "path"
        ]

        filename = source_info[
            "filename"
        ]

        destination_path = os.path.join(
            image_output,
            filename,
        )

        shutil.copy2(
            source_path,
            destination_path,
        )

        classifier_correct = (
            pred
            ==
            label
        )

        ours_flag = bool(
            arrays[
                "ours"
            ][
                detector_index
            ]
        )

        js_flag = bool(
            arrays[
                "js"
            ][
                detector_index
            ]
        )

        combined_flag = bool(
            arrays[
                "ours_js"
            ][
                detector_index
            ]
        )

        metadata_rows.append(
            {
                "detector_index":
                    detector_index,

                "sample_id":
                    sample_id,

                "filename":
                    filename,

                "label":
                    label,

                "prediction":
                    pred,

                "classifier_correct":
                    int(
                        classifier_correct
                    ),

                "classifier_wrong":
                    int(
                        not classifier_correct
                    ),

                "ours_flag":
                    int(
                        ours_flag
                    ),

                "js_flag":
                    int(
                        js_flag
                    ),

                "ours_js_flag":
                    int(
                        combined_flag
                    ),

                "source_path":
                    source_path,

                "extracted_path":
                    destination_path,
            }
        )

        copied += 1

   
    if copied != num_suspicious:

        raise RuntimeError(
            f"{model}/{condition}: "
            f"expected {num_suspicious} copies "
            f"but copied {copied}."
        )

    
    metadata_path = os.path.join(
        condition_output,
        "metadata.csv",
    )

    fieldnames = [
        "detector_index",
        "sample_id",
        "filename",
        "label",
        "prediction",
        "classifier_correct",
        "classifier_wrong",
        "ours_flag",
        "js_flag",
        "ours_js_flag",
        "source_path",
        "extracted_path",
    ]

    with open(
        metadata_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            metadata_rows
        )

    
    predictions = arrays[
        "pred"
    ]

    labels = arrays[
        "label"
    ]

    wrong = (
        predictions
        != labels
    )

    detected_wrong = (
        final_mask
        &
        wrong
    )

    detected_correct = (
        final_mask
        &
        ~wrong
    )

    ours_only = (
        arrays[
            "ours"
        ]
        &
        ~arrays[
            "js"
        ]
    )

    js_only = (
        arrays[
            "js"
        ]
        &
        ~arrays[
            "ours"
        ]
    )

    both = (
        arrays[
            "ours"
        ]
        &
        arrays[
            "js"
        ]
    )

    summary = {
        "model":
            model,

        "condition":
            condition,

        "source_directory":
            source_directory,

        "total_samples":
            int(
                n
            ),

        "classifier_correct":
            int(
                (~wrong).sum()
            ),

        "classifier_wrong":
            int(
                wrong.sum()
            ),

        "ours_detected":
            int(
                arrays[
                    "ours"
                ].sum()
            ),

        "js_detected":
            int(
                arrays[
                    "js"
                ].sum()
            ),

        "ours_js_detected":
            int(
                final_mask.sum()
            ),

        "ours_only":
            int(
                ours_only.sum()
            ),

        "js_only":
            int(
                js_only.sum()
            ),

        "both_ours_and_js":
            int(
                both.sum()
            ),

        "detected_wrong":
            int(
                detected_wrong.sum()
            ),

        "detected_correct":
            int(
                detected_correct.sum()
            ),

        "extracted_images":
            int(
                copied
            ),

        "extraction_rate_percent":
            float(
                100.0
                *
                copied
                /
                n
            ),
    }

    with open(
        os.path.join(
            condition_output,
            "summary.json",
        ),
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    print(
        f"Wrong detected: "
        f"{summary['detected_wrong']}"
    )

    print(
        f"Correct flagged: "
        f"{summary['detected_correct']}"
    )

    print(
        f"Ours only     : "
        f"{summary['ours_only']}"
    )

    print(
        f"JS only       : "
        f"{summary['js_only']}"
    )

    print(
        f"Both          : "
        f"{summary['both_ours_and_js']}"
    )

    print(
        f"Saved images  : "
        f"{image_output}"
    )

    print(
        f"Metadata      : "
        f"{metadata_path}"
    )

    return summary



def validate_inputs(
    selected_models,
    selected_conditions,
):
    print(
        "\nChecking detector masks and image folders..."
    )

    missing = []

    required_detection_files = [
        "sample_id.npy",
        "label.npy",
        "pred.npy",
        "ours_mask.npy",
        "js_mask.npy",
        "ours_js_mask.npy",
    ]

    for model in selected_models:

        for condition in selected_conditions:

            source = get_source_directory(
                model,
                condition,
            )

            if not os.path.isdir(
                source
            ):

                missing.append(
                    source
                )

            detection_dir = (
                get_detection_directory(
                    model,
                    condition,
                )
            )

            for filename in required_detection_files:

                path = os.path.join(
                    detection_dir,
                    filename,
                )

                if not os.path.isfile(
                    path
                ):

                    missing.append(
                        path
                    )

    if missing:

        print(
            "\nMissing required inputs:"
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
                f"{len(missing) - 40} more"
            )

        raise RuntimeError(
            "Cannot extract suspicious samples."
        )

    print(
        "✅ All required inputs exist."
    )



def save_global_summary(
    summaries,
):
    csv_path = os.path.join(
        OUTPUT_ROOT,
        "extraction_summary.csv",
    )

    fields = [
        "model",
        "condition",
        "total_samples",
        "classifier_correct",
        "classifier_wrong",
        "ours_detected",
        "js_detected",
        "ours_js_detected",
        "ours_only",
        "js_only",
        "both_ours_and_js",
        "detected_wrong",
        "detected_correct",
        "extracted_images",
        "extraction_rate_percent",
        "source_directory",
    ]

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for summary in summaries:

            writer.writerow(
                {
                    key:
                        summary[
                            key
                        ]

                    for key in fields
                }
            )

    json_path = os.path.join(
        OUTPUT_ROOT,
        "extraction_summary.json",
    )

    with open(
        json_path,
        "w",
    ) as f:

        json.dump(
            summaries,
            f,
            indent=2,
        )

    return (
        csv_path,
        json_path,
    )



def print_final_summary(
    summaries,
):
    print(
        "\n"
        + "=" * 130
    )

    print(
        "FINAL OURS+JS SUSPICIOUS IMAGE EXTRACTION SUMMARY"
    )

    print(
        "=" * 130
    )

    print(
        f"{'Model':<14}"
        f"{'Condition':<20}"
        f"{'Total':>9}"
        f"{'Detected':>11}"
        f"{'Rate':>10}"
        f"{'WrongDet':>11}"
        f"{'CorrectDet':>12}"
        f"{'JS-only':>10}"
    )

    print(
        "-" * 130
    )

    for s in summaries:

        print(
            f"{s['model']:<14}"
            f"{s['condition']:<20}"
            f"{s['total_samples']:>9d}"
            f"{s['ours_js_detected']:>11d}"
            f"{s['extraction_rate_percent']:>9.2f}%"
            f"{s['detected_wrong']:>11d}"
            f"{s['detected_correct']:>12d}"
            f"{s['js_only']:>10d}"
        )



def main():

    parser = argparse.ArgumentParser(
        description=(
            "Extract all GTSRB images flagged "
            "by the final OURS+JS detector."
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
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=CONDITIONS,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Delete and recreate existing extraction "
            "directories."
        ),
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

    selected_conditions = list(
        args.conditions
    )

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
        "GTSRB — EXTRACT FINAL OURS+JS SUSPICIOUS IMAGES"
    )

    print(
        "=" * 120
    )

    print(
        f"Models      : "
        f"{selected_models}"
    )

    print(
        f"Conditions  : "
        f"{selected_conditions}"
    )

    print(
        f"Detection   : "
        f"{DETECTION_ROOT}"
    )

    print(
        f"Output      : "
        f"{OUTPUT_ROOT}"
    )

   
    validate_inputs(
        selected_models,
        selected_conditions,
    )

   
    summaries = []

    for model in selected_models:

        for condition in selected_conditions:

            summary = extract_condition(
                model=model,
                condition=condition,
                overwrite=args.overwrite,
            )

            summaries.append(
                summary
            )

  
    (
        csv_path,
        json_path,
    ) = save_global_summary(
        summaries
    )

    print_final_summary(
        summaries
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
        "\nSuspicious images:"
    )

    print(
        f"  {OUTPUT_ROOT}/"
    )

    print(
        "\nSummary:"
    )

    print(
        f"  {csv_path}"
    )

    print(
        f"  {json_path}"
    )

    print(
        "\nNEXT STEP:"
    )

    print(
        "Run DDPM reconstruction on the extracted "
        "images for each model/condition."
    )


if __name__ == "__main__":
    main()