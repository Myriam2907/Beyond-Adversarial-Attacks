

import os
import shutil
import json
from pathlib import Path

import numpy as np




ROOT = (
    "/home/Traffic_Signs_2/"
    "physical_pipeline"
)

OUT_ROOT = (
    "/home/Traffic_Signs_2/"
    "physical_ddpm_input"
)

MODELS = [
    "mobilenet",
    "convnext",
    "efficientnet"
]




def extract_model(model_key):

    combined_dir = os.path.join(
        ROOT,
        "combined_detector",
        model_key,
        "qr"
    )

    mask_path = os.path.join(
        combined_dir,
        "combined_or_mask.npy"
    )

    filenames_path = os.path.join(
        combined_dir,
        "filenames.npy"
    )


    if not os.path.exists(mask_path):
        raise FileNotFoundError(mask_path)

    if not os.path.exists(filenames_path):
        raise FileNotFoundError(filenames_path)


    mask = np.load(
        mask_path
    ).astype(bool)


    filenames = np.load(
        filenames_path,
        allow_pickle=True
    )


    if len(mask) != len(filenames):

        raise RuntimeError(
            f"{model_key}: "
            f"mask/files mismatch "
            f"{len(mask)} != {len(filenames)}"
        )


    model_out = os.path.join(
        OUT_ROOT,
        model_key
    )


    if os.path.exists(model_out):

        shutil.rmtree(
            model_out
        )


    Path(model_out).mkdir(
        parents=True,
        exist_ok=True
    )


    selected_indices = np.where(
        mask
    )[0]


    rows = []


    for idx in selected_indices:

        src = str(
            filenames[idx]
        )


        if not os.path.exists(src):

            raise FileNotFoundError(
                src
            )


        class_name = os.path.basename(
            os.path.dirname(
                src
            )
        )


        class_out = os.path.join(
            model_out,
            class_name
        )


        Path(class_out).mkdir(
            parents=True,
            exist_ok=True
        )


        filename = os.path.basename(
            src
        )


       
        dst_name = (
            f"{idx:03d}_"
            f"{filename}"
        )


        dst = os.path.join(
            class_out,
            dst_name
        )


        shutil.copy2(
            src,
            dst
        )


        rows.append({

            "dataset_index":
                int(idx),

            "class_name":
                class_name,

            "source":
                src,

            "destination":
                dst
        })



    np.save(
        os.path.join(
            model_out,
            "selected_indices.npy"
        ),
        selected_indices
    )


    with open(
        os.path.join(
            model_out,
            "manifest.json"
        ),
        "w"
    ) as f:

        json.dump(
            {
                "model":
                    model_key,

                "selection_rule":
                    "OLD OR JS",

                "total_qr_images":
                    int(len(mask)),

                "num_selected":
                    int(mask.sum()),

                "selected_indices":
                    [
                        int(x)
                        for x in selected_indices
                    ],

                "files":
                    rows
            },
            f,
            indent=2
        )


    print("\n" + "=" * 80)

    print(
        model_key.upper()
    )

    print("=" * 80)

    print(
        f"Total QR images : "
        f"{len(mask)}"
    )

    print(
        f"Flagged OR      : "
        f"{mask.sum()}"
    )

    print(
        f"Output          : "
        f"{model_out}"
    )




def main():

    Path(
        OUT_ROOT
    ).mkdir(
        parents=True,
        exist_ok=True
    )


    print("=" * 100)

    print(
        "PHYSICAL DDPM INPUT EXTRACTION — STEP 7"
    )

    print("=" * 100)

    print(
        "Selection = OLD OR JS"
    )


    for model_key in MODELS:

        extract_model(
            model_key
        )


    print("\n" + "=" * 100)

    print(
        "STEP 7 COMPLETE"
    )

    print("=" * 100)


if __name__ == "__main__":

    main()
