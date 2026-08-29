
import os
import sys
import json
import argparse
import importlib.util
from pathlib import Path

import torch
import torch.nn as nn

from PIL import Image, ImageOps
from torchvision import models, transforms




ROOT = (
    "/home/Traffic_Signs_2"
)

OLD_DDPM_SCRIPT = os.path.join(
    ROOT,
    "reconstruct_ddpm_mapillary_combined_js_final.py"
)

DDPM_DIR = os.path.join(
    ROOT,
    "ddpm_mapillary_64x64"
)

INPUT_ROOT = os.path.join(
    ROOT,
    "physical_ddpm_input_all70"
)

OUTPUT_ROOT = os.path.join(
    ROOT,
    "physical_ddpm_recon_all70"
)

MODEL_DIR = os.path.join(
    ROOT,
    "physical_models"
)


MODEL_CONFIG = {

    "mobilenet": {
        "checkpoint": os.path.join(
            MODEL_DIR,
            "mobilenetv3_comma_clean_best.pth"
        )
    },

    "convnext": {
        "checkpoint": os.path.join(
            MODEL_DIR,
            "convnext_tiny_comma_clean_best.pth"
        )
    },

    "efficientnet": {
        "checkpoint": os.path.join(
            MODEL_DIR,
            "efficientnet_v2_s_comma_clean_best.pth"
        )
    }
}


def load_old_module():

    if not os.path.exists(
        OLD_DDPM_SCRIPT
    ):
        raise FileNotFoundError(
            "\nMissing original DDPM script:\n"
            f"{OLD_DDPM_SCRIPT}\n\n"
            "Do NOT continue with a different DDPM implementation."
        )


    spec = importlib.util.spec_from_file_location(
        "mapillary_ddpm_original",
        OLD_DDPM_SCRIPT
    )

    module = importlib.util.module_from_spec(
        spec
    )

    spec.loader.exec_module(
        module
    )

    return module


R = load_old_module()



def pad_to_square(
    image
):

    w, h = image.size

    side = max(
        w,
        h
    )

    left = (
        side - w
    ) // 2

    right = (
        side - w - left
    )

    top = (
        side - h
    ) // 2

    bottom = (
        side - h - top
    )

    return ImageOps.expand(
        image,
        border=(
            left,
            top,
            right,
            bottom
        ),
        fill=0
    )


def physical_load_batch(
    records
):

    tensors = []

    for record in records:

        with Image.open(
            record["abs_path"]
        ) as image:

            image = image.convert(
                "RGB"
            )

            image = pad_to_square(
                image
            )

            
            image = image.resize(
                (
                    224,
                    224
                )
            )

            x = transforms.functional.to_tensor(
                image
            )

            tensors.append(
                x
            )


    return torch.stack(
        tensors,
        dim=0
    ).to(
        R.DEVICE,
        non_blocking=True
    )


R.load_batch = physical_load_batch



def build_model(
    model_key,
    num_classes
):

    if model_key == "mobilenet":

        model = models.mobilenet_v3_large(
            weights=None
        )

        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features,
            num_classes
        )


    elif model_key == "convnext":

        model = models.convnext_tiny(
            weights=None
        )

        model.classifier[2] = nn.Linear(
            model.classifier[2].in_features,
            num_classes
        )


    elif model_key == "efficientnet":

        model = models.efficientnet_v2_s(
            weights=None
        )

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )


    else:

        raise ValueError(
            model_key
        )


    return model


def load_physical_classifier(
    model_key
):

    checkpoint_path = (
        MODEL_CONFIG[
            model_key
        ][
            "checkpoint"
        ]
    )


    print(
        f"\nLoading {model_key} classifier..."
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )


    checkpoint = torch.load(
        checkpoint_path,
        map_location=R.DEVICE,
        weights_only=False
    )


    class_to_idx = checkpoint[
        "class_to_idx"
    ]


    classes = checkpoint[
        "classes"
    ]


    num_classes = len(
        classes
    )


    model = build_model(
        model_key,
        num_classes
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True
    )


    model = model.to(
        R.DEVICE
    )

    model.eval()


    print(
        f"Classes   : {num_classes}"
    )

    print(
        f"Device    : {R.DEVICE}"
    )


    return (
        model,
        class_to_idx
    )




def run_model(
    model_key,
    reconstructor,
    t_candidates,
    batch_size
):

    input_dir = os.path.join(
        INPUT_ROOT,
        model_key
    )


    output_dir = os.path.join(
        OUTPUT_ROOT,
        model_key
    )


    if not os.path.isdir(
        input_dir
    ):

        raise FileNotFoundError(
            input_dir
        )


    classifier, class_to_idx = (
        load_physical_classifier(
            model_key
        )
    )


    print("\n" + "#" * 90)

    print(
        f"PHYSICAL DDPM — {model_key.upper()}"
    )

    print("#" * 90)


    stats = R.reconstruct_attack(

        model_key=model_key,

        attack_name="physical_qr",

        in_dir=input_dir,

        out_dir=output_dir,

        reconstructor=reconstructor,

        classifier=classifier,

        class_to_idx=class_to_idx,

        batch_size=batch_size,

        t_candidates=t_candidates,

        save_images=True
    )


    del classifier


    if R.DEVICE.type == "cuda":
        torch.cuda.empty_cache()


    return stats



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


    parser.add_argument(
        "--steps",
        type=int,
        default=100
    )


    parser.add_argument(
        "--batch",
        type=int,
        default=32
    )


    parser.add_argument(
        "--seed",
        type=int,
        default=123
    )


    parser.add_argument(
        "--t_candidates",
        default="skip,20,40,80,120,160"
    )


    args = parser.parse_args()


    models_to_run = (

        list(
            MODEL_CONFIG.keys()
        )

        if args.model == "all"

        else [
            args.model
        ]
    )


    t_candidates = R.parse_t_candidates(
        args.t_candidates
    )


    Path(
        OUTPUT_ROOT
    ).mkdir(
        parents=True,
        exist_ok=True
    )



    print("=" * 100)

    print(
        "PHYSICAL QR DDPM RECONSTRUCTION + RECLASSIFICATION — STEP 8"
    )

    print("=" * 100)


    print(
        f"Device       : {R.DEVICE}"
    )

    print(
        f"Models       : {models_to_run}"
    )

    print(
        f"DDPM dir     : {DDPM_DIR}"
    )

    print(
        "DDPM size    : 64"
    )

    print(
        "Final size   : 224"
    )

    print(
        f"Steps        : {args.steps}"
    )

    print(
        f"Batch        : {args.batch}"
    )

    print(
        "Candidates   : "
        + str([
            (
                "skip"
                if t is None
                else t
            )
            for t in t_candidates
        ])
    )

    print(
        f"Seed         : {args.seed}"
    )

    print(
        "Selection    : highest classifier confidence"
    )

    print(
        "Ground truth : NOT used for candidate selection"
    )




    reconstructor = (
        R.DDPMPartialReconstructor(

            ddpm_dir=DDPM_DIR,

            device=R.DEVICE,

            inference_steps=args.steps,

            seed=args.seed,

            use_amp=True,

            use_ema=True
        )
    )


    reconstructor.apply_ema_once()


    all_results = {}


    try:

        for model_key in models_to_run:

            all_results[
                model_key
            ] = run_model(
                model_key,
                reconstructor,
                t_candidates,
                args.batch
            )


    finally:

        reconstructor.restore_raw_weights()



    summary_path = os.path.join(
        OUTPUT_ROOT,
        "physical_ddpm_summary.json"
    )


    with open(
        summary_path,
        "w"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2
        )


    

    print("\n\n")

    print("=" * 100)

    print(
        "FINAL PHYSICAL DDPM SUMMARY"
    )

    print("=" * 100)


    print(
        f"{'Model':16s}"
        f"{'N':>7s}"
        f"{'Before':>12s}"
        f"{'After':>12s}"
        f"{'Improve':>12s}"
        f"{'Recovered':>14s}"
        f"{'Damaged':>12s}"
    )

    print("-" * 100)


    for model_key in models_to_run:

        r = all_results[
            model_key
        ]


        recovery = (
            "N/A"
            if r[
                "recovery_rate_of_wrong_percent"
            ] is None

            else (
                f"{r['wrong_before_recovered']}/"
                f"{r['wrong_before_total']}"
            )
        )


        damage = (

            f"{r['correct_before_damaged']}/"
            f"{r['correct_before_total']}"

        )


        print(

            f"{model_key:16s}"

            f"{r['n_images']:7d}"

            f"{r['accuracy_before_percent']:11.2f}%"

            f"{r['accuracy_after_percent']:11.2f}%"

            f"{r['accuracy_improvement_points']:+11.2f}"

            f"{recovery:>14s}"

            f"{damage:>12s}"
        )


        print(
            f"  chosen t: "
            f"{r['chosen_t_counts']}"
        )


    print(
        f"\nSummary saved -> "
        f"{summary_path}"
    )


    print("\n" + "=" * 100)

    print(
        "STEP 8 COMPLETE"
    )

    print("=" * 100)


if __name__ == "__main__":

    main()
