#!/usr/bin/env python3

import os
import json
import argparse
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageOps

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from tqdm import tqdm


# ============================================================
# CONFIG
# ============================================================

CLEAN_DIR = (
    "/home/cpsslab/Desktop/myriam/Traffic_Signs_2/"
    "Clean Dataset/Myriam"
)

QR_DIR = (
    "/home/cpsslab/Desktop/myriam/Traffic_Signs_2/"
    "attacked"
)

MODEL_DIR = (
    "/home/cpsslab/Desktop/myriam/Traffic_Signs_2/"
    "physical_models"
)

OUT_ROOT = (
    "/home/cpsslab/Desktop/myriam/Traffic_Signs_2/"
    "physical_pipeline"
)

IMG_SIZE = 224

BATCH_SIZE = 16

NUM_WORKERS = 4


MEAN = (
    0.485,
    0.456,
    0.406
)

STD = (
    0.229,
    0.224,
    0.225
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# MODEL CONFIG
# ============================================================

MODELS = {

    "mobilenet": {

        "checkpoint": os.path.join(
            MODEL_DIR,
            "mobilenetv3_comma_clean_best.pth"
        ),

        "output": os.path.join(
            OUT_ROOT,
            "eval_mobilenet"
        )
    },

    "convnext": {

        "checkpoint": os.path.join(
            MODEL_DIR,
            "convnext_tiny_comma_clean_best.pth"
        ),

        "output": os.path.join(
            OUT_ROOT,
            "eval_convnext"
        )
    },

    "efficientnet": {

        "checkpoint": os.path.join(
            MODEL_DIR,
            "efficientnet_v2_s_comma_clean_best.pth"
        ),

        "output": os.path.join(
            OUT_ROOT,
            "eval_efficientnet"
        )
    }
}


# ============================================================
# CONDITIONS
# ============================================================

CONDITIONS = {

    "clean": CLEAN_DIR,

    "qr": QR_DIR
}


# ============================================================
# PAD TO SQUARE
#
# IMPORTANT:
# this matches preprocessing used when training the physical
# classifiers.
# ============================================================

class PadToSquare:

    def __call__(self, img):

        w, h = img.size

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
            img,
            border=(
                left,
                top,
                right,
                bottom
            ),
            fill=0
        )


# ============================================================
# DATASET
#
# We do NOT use torchvision ImageFolder directly because we
# want to force the exact class mapping stored in checkpoint.
# ============================================================

class FixedClassDataset(Dataset):

    def __init__(
        self,
        root,
        class_to_idx
    ):

        self.root = root

        self.class_to_idx = (
            class_to_idx
        )

        self.samples = []

        valid_extensions = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        )

        for class_name, class_id in sorted(
            class_to_idx.items(),
            key=lambda x: x[1]
        ):

            class_dir = os.path.join(
                root,
                class_name
            )

            if not os.path.isdir(
                class_dir
            ):

                raise FileNotFoundError(
                    f"Missing class folder:\n"
                    f"{class_dir}"
                )

            filenames = sorted([
                f
                for f in os.listdir(
                    class_dir
                )
                if f.lower().endswith(
                    valid_extensions
                )
            ])

            for filename in filenames:

                path = os.path.join(
                    class_dir,
                    filename
                )

                self.samples.append(
                    (
                        path,
                        int(class_id)
                    )
                )


        print(
            f"Dataset: {root}"
        )

        print(
            f"Samples: {len(self.samples)}"
        )


    def __len__(self):

        return len(
            self.samples
        )


    def __getitem__(
        self,
        index
    ):

        path, label = (
            self.samples[index]
        )

        image = Image.open(
            path
        ).convert(
            "RGB"
        )

        # --------------------------------------------
        # Same geometric preprocessing as training.
        # Keep image in [0,1].
        # Do NOT normalize here.
        # --------------------------------------------

        image = PadToSquare()(
            image
        )

        image = image.resize(
            (
                IMG_SIZE,
                IMG_SIZE
            )
        )

        x01 = transforms.functional.to_tensor(
            image
        )

        return (
            x01,
            label,
            path,
            index
        )


# ============================================================
# NORMALIZATION
# ============================================================

def normalize(
    x
):

    mean = torch.tensor(
        MEAN,
        device=x.device,
        dtype=x.dtype
    ).view(
        1,
        3,
        1,
        1
    )

    std = torch.tensor(
        STD,
        device=x.device,
        dtype=x.dtype
    ).view(
        1,
        3,
        1,
        1
    )

    return (
        x - mean
    ) / std


# ============================================================
# EXACT OLD WEAK AUGMENTATION 1
#
# Mapillary pipeline:
#
# slight contrast/brightness change
# +
# small 3x3 smoothing
# ============================================================

def weak_augment_1(
    x
):

    y = x.clone()


    # --------------------------------------------
    # brightness / contrast
    # --------------------------------------------

    y = (
        (y - 0.5)
        * 1.03
        + 0.5
        + 0.015
    )

    y = y.clamp(
        0,
        1
    )


    # --------------------------------------------
    # mild smoothing
    # --------------------------------------------

    blur = F.avg_pool2d(
        y,
        kernel_size=3,
        stride=1,
        padding=1
    )


    y = (
        0.85 * y
        +
        0.15 * blur
    )


    return y.clamp(
        0,
        1
    )


# ============================================================
# EXACT OLD WEAK AUGMENTATION 2
#
# Used for third-pass critical-class analysis.
# ============================================================

def weak_augment_2(
    x
):

    y = x.clone()


    # --------------------------------------------
    # opposite mild brightness / contrast shift
    # --------------------------------------------

    y = (
        (y - 0.5)
        * 0.97
        + 0.5
        - 0.010
    )

    y = y.clamp(
        0,
        1
    )


    # --------------------------------------------
    # slightly different smoothing
    # --------------------------------------------

    blur = F.avg_pool2d(
        y,
        kernel_size=5,
        stride=1,
        padding=2
    )


    y = (
        0.90 * y
        +
        0.10 * blur
    )


    return y.clamp(
        0,
        1
    )


# ============================================================
# FORWARD SIGNALS
# ============================================================

@torch.no_grad()
def forward_signals(
    model,
    x01
):

    logits = model(
        normalize(
            x01
        )
    )


    probs = F.softmax(
        logits,
        dim=1
    )


    confidence, pred = (
        probs.max(
            dim=1
        )
    )


    # --------------------------------------------
    # same energy definition used previously
    #
    # higher / less negative = more suspicious
    # --------------------------------------------

    energy = (
        -torch.logsumexp(
            logits,
            dim=1
        )
    )


    return (
        logits,
        pred,
        confidence,
        energy
    )


# ============================================================
# BUILD MODEL
# ============================================================

def build_model(
    model_key,
    num_classes
):

    if model_key == "mobilenet":

        model = (
            models.mobilenet_v3_large(
                weights=None
            )
        )

        in_features = (
            model.classifier[3]
            .in_features
        )

        model.classifier[3] = (
            nn.Linear(
                in_features,
                num_classes
            )
        )


    elif model_key == "convnext":

        model = (
            models.convnext_tiny(
                weights=None
            )
        )

        in_features = (
            model.classifier[2]
            .in_features
        )

        model.classifier[2] = (
            nn.Linear(
                in_features,
                num_classes
            )
        )


    elif model_key == "efficientnet":

        model = (
            models.efficientnet_v2_s(
                weights=None
            )
        )

        in_features = (
            model.classifier[1]
            .in_features
        )

        model.classifier[1] = (
            nn.Linear(
                in_features,
                num_classes
            )
        )


    else:

        raise ValueError(
            model_key
        )


    return model


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_model(
    model_key
):

    cfg = MODELS[
        model_key
    ]


    checkpoint_path = (
        cfg["checkpoint"]
    )


    if not os.path.exists(
        checkpoint_path
    ):

        raise FileNotFoundError(
            checkpoint_path
        )


    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )


    classes = checkpoint[
        "classes"
    ]


    class_to_idx = checkpoint[
        "class_to_idx"
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
        ]
    )


    model = model.to(
        DEVICE
    )


    model.eval()


    return (
        model,
        classes,
        class_to_idx
    )


# ============================================================
# EVALUATE ONE FOLDER
# ============================================================

@torch.no_grad()
def eval_folder(
    model,
    img_dir,
    out_dir,
    classes,
    class_to_idx
):

    Path(
        out_dir
    ).mkdir(
        parents=True,
        exist_ok=True
    )


    # ========================================================
    # CRITICAL CLASSES
    #
    # Physical dataset:
    # STOP  = 6
    # YIELD = 9
    #
    # Determine IDs from names instead of hard-coding.
    # ========================================================

    if "stop" not in class_to_idx:

        raise KeyError(
            "stop class missing"
        )


    if "yield" not in class_to_idx:

        raise KeyError(
            "yield class missing"
        )


    STOP_ID = int(
        class_to_idx[
            "stop"
        ]
    )

    YIELD_ID = int(
        class_to_idx[
            "yield"
        ]
    )


    CRITICAL_IDS = [
        STOP_ID,
        YIELD_ID
    ]


    # ========================================================
    # DATA
    # ========================================================

    dataset = FixedClassDataset(
        img_dir,
        class_to_idx
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type
            ==
            "cuda"
        )
    )


    # ========================================================
    # STORAGE
    # ========================================================

    all_indices = []

    all_paths = []

    all_labels = []


    all_pred = []

    all_confidence = []

    all_energy = []


    all_pred_aug1 = []

    all_confidence_aug1 = []

    all_energy_aug1 = []


    all_conf_drop_2pass = []

    all_logit_l2_2pass = []

    all_changed_2pass = []


    all_conf_drop_3pass = []

    all_logit_l2_3pass = []

    all_changed_3pass = []


    all_critical_pred_mask = []


    correct = 0

    total = 0

    critical_total = 0


    # ========================================================
    # LOOP
    # ========================================================

    for (
        x01,
        y,
        paths,
        indices
    ) in tqdm(
        loader,
        desc=os.path.basename(
            out_dir
        )
    ):


        x01 = x01.to(
            DEVICE,
            non_blocking=True
        )


        y = y.to(
            DEVICE,
            non_blocking=True
        )


        # ====================================================
        # PASS 1 — ORIGINAL
        # ====================================================

        (
            logits,
            pred,
            conf,
            energy
        ) = forward_signals(
            model,
            x01
        )


        # ====================================================
        # PASS 2 — WEAK AUGMENT 1
        # ====================================================

        x_aug1 = (
            weak_augment_1(
                x01
            )
        )


        (
            logits_aug1,
            pred_aug1,
            conf_aug1,
            energy_aug1
        ) = forward_signals(
            model,
            x_aug1
        )


        # --------------------------------------------
        # 2-pass confidence drop
        # --------------------------------------------

        conf_drop_2 = (
            conf
            -
            conf_aug1
        )


        # --------------------------------------------
        # 2-pass logit L2
        # --------------------------------------------

        logit_l2_2 = (
            torch.norm(
                logits
                -
                logits_aug1,
                p=2,
                dim=1
            )
        )


        # --------------------------------------------
        # prediction changed?
        # --------------------------------------------

        changed_2 = (
            pred
            !=
            pred_aug1
        ).long()


        # ====================================================
        # PASS 3 — ONLY FOR BASE-PREDICTED STOP / YIELD
        #
        # This matches the final clean-threshold calibration
        # logic: third-pass thresholds are calibrated on
        # base predictions considered critical.
        # ====================================================

        critical_pred_mask = (
            torch.zeros_like(
                pred,
                dtype=torch.bool
            )
        )


        for cid in CRITICAL_IDS:

            critical_pred_mask |= (
                pred == cid
            )


        # --------------------------------------------
        # Default -1 = third pass not applicable
        # --------------------------------------------

        conf_drop_3 = (
            torch.full_like(
                conf,
                -1.0
            )
        )


        logit_l2_3 = (
            torch.full_like(
                conf,
                -1.0
            )
        )


        changed_3 = (
            torch.full_like(
                pred,
                -1
            )
        )


        if critical_pred_mask.any():

            critical_total += int(
                critical_pred_mask
                .sum()
                .item()
            )


            x_crit = x01[
                critical_pred_mask
            ]


            logits_crit = logits[
                critical_pred_mask
            ]


            pred_crit = pred[
                critical_pred_mask
            ]


            conf_crit = conf[
                critical_pred_mask
            ]


            # ================================================
            # Third inference pass
            # ================================================

            x_aug2 = (
                weak_augment_2(
                    x_crit
                )
            )


            (
                logits_aug2,
                pred_aug2,
                conf_aug2,
                _
            ) = forward_signals(
                model,
                x_aug2
            )


            # --------------------------------------------
            # Confidence drop from augmentation #1
            # --------------------------------------------

            cd1 = (
                conf_crit
                -
                conf_aug1[
                    critical_pred_mask
                ]
            )


            # --------------------------------------------
            # Confidence drop from augmentation #2
            # --------------------------------------------

            cd2 = (
                conf_crit
                -
                conf_aug2
            )


            # --------------------------------------------
            # maximum 3-pass confidence instability
            # --------------------------------------------

            conf_drop_max = (
                torch.maximum(
                    cd1,
                    cd2
                )
            )


            # --------------------------------------------
            # logit L2 pass1-pass2
            # --------------------------------------------

            l21 = torch.norm(

                logits_crit
                -
                logits_aug1[
                    critical_pred_mask
                ],

                p=2,
                dim=1
            )


            # --------------------------------------------
            # logit L2 pass1-pass3
            # --------------------------------------------

            l22 = torch.norm(

                logits_crit
                -
                logits_aug2,

                p=2,
                dim=1
            )


            # --------------------------------------------
            # maximum logit deviation
            # --------------------------------------------

            l2max = torch.maximum(
                l21,
                l22
            )


            # --------------------------------------------
            # Did either transformed prediction change?
            # --------------------------------------------

            changed = (

                (
                    pred_crit
                    !=
                    pred_aug1[
                        critical_pred_mask
                    ]
                )

                |

                (
                    pred_crit
                    !=
                    pred_aug2
                )

            ).long()


            conf_drop_3[
                critical_pred_mask
            ] = conf_drop_max


            logit_l2_3[
                critical_pred_mask
            ] = l2max


            changed_3[
                critical_pred_mask
            ] = changed


        # ====================================================
        # CLASSIFIER ACCURACY
        # ====================================================

        correct += (
            pred == y
        ).sum().item()


        total += (
            y.numel()
        )


        # ====================================================
        # SAVE BATCH
        # ====================================================

        all_indices.append(

            np.asarray(
                [
                    int(v)
                    for v
                    in indices
                ]
            )
        )


        all_paths.extend(
            list(
                paths
            )
        )


        all_labels.append(
            y.cpu().numpy()
        )


        all_pred.append(
            pred.cpu().numpy()
        )


        all_confidence.append(
            conf.cpu().numpy()
        )


        all_energy.append(
            energy.cpu().numpy()
        )


        all_pred_aug1.append(
            pred_aug1.cpu().numpy()
        )


        all_confidence_aug1.append(
            conf_aug1.cpu().numpy()
        )


        all_energy_aug1.append(
            energy_aug1.cpu().numpy()
        )


        all_conf_drop_2pass.append(
            conf_drop_2.cpu().numpy()
        )


        all_logit_l2_2pass.append(
            logit_l2_2.cpu().numpy()
        )


        all_changed_2pass.append(
            changed_2.cpu().numpy()
        )


        all_conf_drop_3pass.append(
            conf_drop_3.cpu().numpy()
        )


        all_logit_l2_3pass.append(
            logit_l2_3.cpu().numpy()
        )


        all_changed_3pass.append(
            changed_3.cpu().numpy()
        )


        all_critical_pred_mask.append(
            critical_pred_mask
            .cpu()
            .numpy()
            .astype(
                np.uint8
            )
        )


    # ========================================================
    # CONCATENATE
    # ========================================================

    dataset_index = np.concatenate(
        all_indices
    )


    filenames = np.asarray(
        all_paths,
        dtype=object
    )


    label = np.concatenate(
        all_labels
    )


    pred = np.concatenate(
        all_pred
    )


    confidence = np.concatenate(
        all_confidence
    )


    energy = np.concatenate(
        all_energy
    )


    pred_aug1 = np.concatenate(
        all_pred_aug1
    )


    confidence_aug1 = np.concatenate(
        all_confidence_aug1
    )


    energy_aug1 = np.concatenate(
        all_energy_aug1
    )


    conf_drop_2 = np.concatenate(
        all_conf_drop_2pass
    )


    logit_l2_2 = np.concatenate(
        all_logit_l2_2pass
    )


    changed_2 = np.concatenate(
        all_changed_2pass
    )


    conf_drop_3 = np.concatenate(
        all_conf_drop_3pass
    )


    logit_l2_3 = np.concatenate(
        all_logit_l2_3pass
    )


    changed_3 = np.concatenate(
        all_changed_3pass
    )


    critical_pred_mask = (
        np.concatenate(
            all_critical_pred_mask
        )
        .astype(
            bool
        )
    )


    # ========================================================
    # SANITY CHECK
    # ========================================================

    n = len(
        label
    )


    arrays_to_check = {

        "dataset_index":
            dataset_index,

        "filenames":
            filenames,

        "pred":
            pred,

        "confidence":
            confidence,

        "energy":
            energy,

        "2pass_conf_drop":
            conf_drop_2,

        "2pass_logit_l2":
            logit_l2_2,

        "2pass_changed":
            changed_2,

        "3pass_conf_drop":
            conf_drop_3,

        "3pass_logit_l2":
            logit_l2_3,

        "3pass_changed":
            changed_3,

        "critical_pred_mask":
            critical_pred_mask
    }


    for name, array in (
        arrays_to_check.items()
    ):

        if len(array) != n:

            raise RuntimeError(
                f"{name} length "
                f"{len(array)} != {n}"
            )


    # ========================================================
    # SAVE .NPY
    # ========================================================

    np.save(
        os.path.join(
            out_dir,
            "dataset_index.npy"
        ),
        dataset_index
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


    np.save(
        os.path.join(
            out_dir,
            "confidence.npy"
        ),
        confidence
    )


    np.save(
        os.path.join(
            out_dir,
            "energy.npy"
        ),
        energy
    )


    # --------------------------------------------------------
    # Save first transformed pass too.
    # Useful for debugging and consistency.
    # --------------------------------------------------------

    np.save(
        os.path.join(
            out_dir,
            "pred_aug.npy"
        ),
        pred_aug1
    )


    np.save(
        os.path.join(
            out_dir,
            "confidence_aug.npy"
        ),
        confidence_aug1
    )


    np.save(
        os.path.join(
            out_dir,
            "energy_aug.npy"
        ),
        energy_aug1
    )


    # --------------------------------------------------------
    # 2-pass
    # --------------------------------------------------------

    np.save(
        os.path.join(
            out_dir,
            "2pass_conf_drop.npy"
        ),
        conf_drop_2
    )


    np.save(
        os.path.join(
            out_dir,
            "2pass_logit_l2.npy"
        ),
        logit_l2_2
    )


    np.save(
        os.path.join(
            out_dir,
            "2pass_changed.npy"
        ),
        changed_2
    )


    # --------------------------------------------------------
    # 3-pass
    # --------------------------------------------------------

    np.save(
        os.path.join(
            out_dir,
            "3pass_max_conf_drop_critical.npy"
        ),
        conf_drop_3
    )


    np.save(
        os.path.join(
            out_dir,
            "3pass_max_logit_l2_critical.npy"
        ),
        logit_l2_3
    )


    np.save(
        os.path.join(
            out_dir,
            "3pass_changed_critical.npy"
        ),
        changed_3
    )


    np.save(
        os.path.join(
            out_dir,
            "critical_pred_mask.npy"
        ),
        critical_pred_mask
    )


    # ========================================================
    # STATISTICS
    # ========================================================

    accuracy = (
        (
            pred == label
        ).mean()
        * 100.0
    )


    valid_3 = (
        critical_pred_mask
        &
        (
            changed_3
            != -1
        )
    )


    stats = {

        "image_directory":
            img_dir,

        "output_directory":
            out_dir,

        "num_samples":
            int(n),

        "accuracy_percent":
            float(
                accuracy
            ),

        "num_classes":
            int(
                len(classes)
            ),

        "classes":
            classes,

        "critical_class_ids":
            CRITICAL_IDS,

        "critical_class_names": [
            "stop",
            "yield"
        ],

        "base_predicted_critical_samples":
            int(
                critical_pred_mask
                .sum()
            ),

        "base_predicted_critical_percent":
            float(
                critical_pred_mask
                .mean()
                * 100.0
            ),

        # --------------------------------------------
        # base inference
        # --------------------------------------------

        "confidence_mean":
            float(
                confidence.mean()
            ),

        "confidence_p5":
            float(
                np.percentile(
                    confidence,
                    5
                )
            ),

        "energy_mean":
            float(
                energy.mean()
            ),

        "energy_p95":
            float(
                np.percentile(
                    energy,
                    95
                )
            ),

        # --------------------------------------------
        # 2-pass
        # --------------------------------------------

        "2pass_conf_drop_mean":
            float(
                conf_drop_2.mean()
            ),

        "2pass_conf_drop_p95":
            float(
                np.percentile(
                    conf_drop_2,
                    95
                )
            ),

        "2pass_logit_l2_mean":
            float(
                logit_l2_2.mean()
            ),

        "2pass_logit_l2_p95":
            float(
                np.percentile(
                    logit_l2_2,
                    95
                )
            ),

        "2pass_changed_count":
            int(
                (
                    changed_2 == 1
                ).sum()
            ),

        "2pass_changed_percent":
            float(
                (
                    changed_2 == 1
                ).mean()
                * 100.0
            )
    }


    # ========================================================
    # 3-PASS STATS
    # ========================================================

    if valid_3.any():

        valid_cd3 = (
            conf_drop_3[
                valid_3
            ]
        )


        valid_l23 = (
            logit_l2_3[
                valid_3
            ]
        )


        valid_changed3 = (
            changed_3[
                valid_3
            ]
        )


        stats.update({

            "3pass_valid_samples":
                int(
                    valid_3.sum()
                ),

            "3pass_conf_drop_mean":
                float(
                    valid_cd3.mean()
                ),

            "3pass_conf_drop_p95":
                float(
                    np.percentile(
                        valid_cd3,
                        95
                    )
                ),

            "3pass_logit_l2_mean":
                float(
                    valid_l23.mean()
                ),

            "3pass_logit_l2_p95":
                float(
                    np.percentile(
                        valid_l23,
                        95
                    )
                ),

            "3pass_changed_count":
                int(
                    (
                        valid_changed3
                        == 1
                    ).sum()
                ),

            "3pass_changed_percent":
                float(
                    (
                        valid_changed3
                        == 1
                    ).mean()
                    * 100.0
                )
        })


    else:

        stats.update({

            "3pass_valid_samples":
                0,

            "3pass_conf_drop_mean":
                None,

            "3pass_conf_drop_p95":
                None,

            "3pass_logit_l2_mean":
                None,

            "3pass_logit_l2_p95":
                None,

            "3pass_changed_count":
                0,

            "3pass_changed_percent":
                None
        })


    # ========================================================
    # SAVE STATS
    # ========================================================

    with open(
        os.path.join(
            out_dir,
            "eval_stats.json"
        ),
        "w"
    ) as f:

        json.dump(
            stats,
            f,
            indent=2
        )


    # ========================================================
    # SAVE CLASS MAP
    # ========================================================

    with open(
        os.path.join(
            out_dir,
            "class_to_idx.json"
        ),
        "w"
    ) as f:

        json.dump(
            class_to_idx,
            f,
            indent=2
        )


    # ========================================================
    # PRINT
    # ========================================================

    print("\n" + "=" * 80)

    print(
        f"EVALUATION COMPLETE: {out_dir}"
    )

    print("=" * 80)


    print(
        f"Images                  : "
        f"{n}"
    )


    print(
        f"Accuracy                : "
        f"{accuracy:.2f}%"
    )


    print(
        f"Mean confidence         : "
        f"{confidence.mean():.6f}"
    )


    print(
        f"Confidence p5           : "
        f"{np.percentile(confidence,5):.6f}"
    )


    print(
        f"Mean energy             : "
        f"{energy.mean():.6f}"
    )


    print(
        f"Energy p95              : "
        f"{np.percentile(energy,95):.6f}"
    )


    print(
        f"2-pass conf drop mean   : "
        f"{conf_drop_2.mean():.6f}"
    )


    print(
        f"2-pass conf drop p95    : "
        f"{np.percentile(conf_drop_2,95):.6f}"
    )


    print(
        f"2-pass logit L2 mean    : "
        f"{logit_l2_2.mean():.6f}"
    )


    print(
        f"2-pass logit L2 p95     : "
        f"{np.percentile(logit_l2_2,95):.6f}"
    )


    print(
        f"2-pass changed          : "
        f"{(changed_2 == 1).sum()}/{n} "
        f"({(changed_2 == 1).mean()*100:.2f}%)"
    )


    print(
        f"Predicted STOP/YIELD    : "
        f"{critical_pred_mask.sum()}/{n}"
    )


    if valid_3.any():

        print(
            f"3-pass conf drop mean   : "
            f"{conf_drop_3[valid_3].mean():.6f}"
        )


        print(
            f"3-pass logit L2 mean    : "
            f"{logit_l2_3[valid_3].mean():.6f}"
        )


        print(
            f"3-pass changed          : "
            f"{(changed_3[valid_3] == 1).sum()}/"
            f"{valid_3.sum()}"
        )


    print(
        "\nSaved signals ->"
    )

    print(
        out_dir
    )


# ============================================================
# RUN MODEL
# ============================================================

def run_model(
    model_key,
    selected_conditions
):

    print("\n\n")
    print("#" * 90)
    print(
        f"MODEL: {model_key.upper()}"
    )
    print("#" * 90)


    (
        model,
        classes,
        class_to_idx
    ) = load_model(
        model_key
    )


    print(
        f"Checkpoint : "
        f"{MODELS[model_key]['checkpoint']}"
    )


    print(
        f"Classes    : "
        f"{len(classes)}"
    )


    print(
        f"STOP ID    : "
        f"{class_to_idx['stop']}"
    )


    print(
        f"YIELD ID   : "
        f"{class_to_idx['yield']}"
    )


    for condition in (
        selected_conditions
    ):

        img_dir = CONDITIONS[
            condition
        ]


        out_dir = os.path.join(
            MODELS[
                model_key
            ][
                "output"
            ],
            condition
        )


        print("\n" + "-" * 90)

        print(
            f"Condition : {condition}"
        )

        print(
            f"Images    : {img_dir}"
        )

        print(
            f"Output    : {out_dir}"
        )

        print(
            "-" * 90
        )


        eval_folder(
            model=model,
            img_dir=img_dir,
            out_dir=out_dir,
            classes=classes,
            class_to_idx=class_to_idx
        )


# ============================================================
# CLI
# ============================================================

def parse_args():

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
        "--conditions",
        nargs="+",
        choices=[
            "clean",
            "qr"
        ],
        default=[
            "clean",
            "qr"
        ]
    )


    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()


    Path(
        OUT_ROOT
    ).mkdir(
        parents=True,
        exist_ok=True
    )


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
        "PHYSICAL TRAFFIC-SIGN SIGNAL EXTRACTION — STEP 1"
    )
    print("=" * 100)


    print(
        f"Device      : {DEVICE}"
    )


    print(
        f"Models      : {selected_models}"
    )


    print(
        f"Conditions  : {args.conditions}"
    )


    print(
        f"Image size  : {IMG_SIZE}"
    )


    print(
        f"Clean dir   : {CLEAN_DIR}"
    )


    print(
        f"QR dir      : {QR_DIR}"
    )


    print(
        f"Output root : {OUT_ROOT}"
    )


    for model_key in (
        selected_models
    ):

        run_model(
            model_key,
            args.conditions
        )


    print("\n")
    print("=" * 100)
    print(
        "STEP 1 COMPLETE"
    )
    print("=" * 100)


if __name__ == "__main__":

    main()
