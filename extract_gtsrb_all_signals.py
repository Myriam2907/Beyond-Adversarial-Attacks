

import os
import re
import json
import time
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode
from tqdm import tqdm




ROOT = "./gtsrb_repeat"

ATTACK_ROOT = os.path.join(ROOT, "attacks")
MODEL_ROOT = os.path.join(ROOT, "models")
OUT_ROOT = os.path.join(ROOT, "signals")

NUM_CLASSES = 43

IMG_SIZE = 224
JS_RESIZE_SIZE = 208

NUM_WORKERS = 8

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


DEFAULT_CRITICAL_IDS = [
    13,
    14,
]


#

GRADIENT_ATTACKS = [
    "fgsm",
    "rfgsm",
    "pgd",
]

SHARED_CONDITIONS = [
    "clean",
    "patch",
    "random_patch",
    "gaussian_noise",
    "salt_pepper",
    "light",
    "fog",
    "motion_blur",
]

ALL_CONDITIONS = [
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




IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".ppm",
    ".webp",
}




def build_mobilenet():
    model = models.mobilenet_v3_large(
        weights=None
    )

    in_features = (
        model.classifier[-1].in_features
    )

    model.classifier[-1] = nn.Linear(
        in_features,
        NUM_CLASSES,
    )

    return model


def build_convnext():
    model = models.convnext_tiny(
        weights=None
    )

    in_features = (
        model.classifier[-1].in_features
    )

    model.classifier[-1] = nn.Linear(
        in_features,
        NUM_CLASSES,
    )

    return model


def build_efficientnet():
    model = models.efficientnet_v2_s(
        weights=None
    )

    in_features = (
        model.classifier[-1].in_features
    )

    model.classifier[-1] = nn.Linear(
        in_features,
        NUM_CLASSES,
    )

    return model


MODEL_CONFIGS = {
    "mobilenet": {
        "builder": build_mobilenet,
        "checkpoint": os.path.join(
            MODEL_ROOT,
            "mobilenet",
            "mobilenet_gtsrb_best.pth",
        ),
        "batch_size": 64,
    },

    "convnext": {
        "builder": build_convnext,
        "checkpoint": os.path.join(
            MODEL_ROOT,
            "convnext",
            "convnext_gtsrb_best.pth",
        ),
        "batch_size": 48,
    },

    "efficientnet": {
        "builder": build_efficientnet,
        "checkpoint": os.path.join(
            MODEL_ROOT,
            "efficientnet",
            "efficientnet_gtsrb_best.pth",
        ),
        "batch_size": 48,
    },
}




class GTSRBGeneratedDataset(Dataset):
   

    LABEL_PATTERN = re.compile(
        r"^(?P<sample_id>\d+)_y(?P<label>\d+)"
    )

    def __init__(
        self,
        root,
        transform=None,
    ):
        self.root = os.path.abspath(
            root
        )

        self.transform = transform

        if not os.path.isdir(
            self.root
        ):
            raise FileNotFoundError(
                f"Dataset folder does not exist:\n"
                f"{self.root}"
            )

        samples = []

        for current_root, _, files in os.walk(
            self.root
        ):
            for filename in files:

                ext = os.path.splitext(
                    filename
                )[1].lower()

                if ext not in IMAGE_EXTENSIONS:
                    continue

                match = self.LABEL_PATTERN.match(
                    filename
                )

                if match is None:
                    raise RuntimeError(
                        "\nCould not parse generated GTSRB filename:\n"
                        f"{os.path.join(current_root, filename)}\n\n"
                        "Expected format similar to:\n"
                        "00000_y16.png\n"
                    )

                sample_id = int(
                    match.group("sample_id")
                )

                label = int(
                    match.group("label")
                )

                if not (
                    0 <= label < NUM_CLASSES
                ):
                    raise RuntimeError(
                        f"Invalid GTSRB label {label} in:\n"
                        f"{filename}"
                    )

                full_path = os.path.join(
                    current_root,
                    filename,
                )

                rel_path = os.path.relpath(
                    full_path,
                    self.root,
                )

                samples.append(
                    (
                        sample_id,
                        label,
                        full_path,
                        rel_path,
                    )
                )

        if len(samples) == 0:
            raise RuntimeError(
                f"No generated GTSRB images found in:\n"
                f"{self.root}"
            )

        # IMPORTANT:
        # Sort by original sample ID.
        #
        # Do NOT sort by label.
        # This ensures clean/attack sample alignment.
        samples.sort(
            key=lambda item: (
                item[0],
                item[3],
            )
        )

        # Detect duplicate sample IDs.
        sample_ids = [
            item[0]
            for item in samples
        ]

        if len(sample_ids) != len(
            set(sample_ids)
        ):
            duplicates = []

            seen = set()

            for sid in sample_ids:
                if sid in seen:
                    duplicates.append(
                        sid
                    )
                seen.add(
                    sid
                )

            raise RuntimeError(
                "Duplicate generated sample IDs found.\n"
                f"First duplicates: {duplicates[:10]}"
            )

        self.samples = samples

        labels = np.asarray(
            [
                x[1]
                for x in samples
            ],
            dtype=np.int64,
        )

        print(
            f"Loaded {len(self.samples)} images"
        )

        print(
            f"Label range: "
            f"{labels.min()}..{labels.max()}"
        )

        print(
            f"Sample-ID range: "
            f"{sample_ids[0]}..{sample_ids[-1]}"
        )

    def __len__(
        self,
    ):
        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):
        (
            sample_id,
            label,
            full_path,
            rel_path,
        ) = self.samples[
            index
        ]

        image = Image.open(
            full_path
        ).convert(
            "RGB"
        )

        if self.transform is not None:
            image = self.transform(
                image
            )

        return (
            image,
            int(label),
            int(sample_id),
            rel_path,
        )




IMAGE_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(
            (
                IMG_SIZE,
                IMG_SIZE,
            ),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),

        transforms.ToTensor(),
    ]
)




def normalize_batch(
    x,
):
    """
    x:
        float tensor in [0,1]
        shape Bx3x224x224
    """

    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=x.dtype,
        device=x.device,
    ).view(
        1,
        3,
        1,
        1,
    )

    std = torch.tensor(
        IMAGENET_STD,
        dtype=x.dtype,
        device=x.device,
    ).view(
        1,
        3,
        1,
        1,
    )

    return (
        x - mean
    ) / std




def weak_augment_1(
    x,
):
    """
    First weak detector transformation.

    Slight contrast/brightness change followed by mild blur.
    """

    y = (
        (x - 0.5)
        * 1.03
        + 0.5
        + 0.015
    )

    y = y.clamp(
        0.0,
        1.0,
    )

    blurred = F.avg_pool2d(
        y,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    y = (
        0.85 * y
        + 0.15 * blurred
    )

    return y.clamp(
        0.0,
        1.0,
    )


def weak_augment_2(
    x,
):
    """
    Second weak detector transformation.

    Used only for samples whose base prediction is one of
    the configured critical GTSRB classes.
    """

    y = (
        (x - 0.5)
        * 0.97
        + 0.5
        - 0.010
    )

    y = y.clamp(
        0.0,
        1.0,
    )

    blurred = F.avg_pool2d(
        y,
        kernel_size=5,
        stride=1,
        padding=2,
    )

    y = (
        0.90 * y
        + 0.10 * blurred
    )

    return y.clamp(
        0.0,
        1.0,
    )




def js_resize_transform(
    x,
):
    """
    Mild deterministic transformation:

        224 x 224
            ->
        208 x 208
            ->
        224 x 224
    """

    small = F.interpolate(
        x,
        size=(
            JS_RESIZE_SIZE,
            JS_RESIZE_SIZE,
        ),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    restored = F.interpolate(
        small,
        size=(
            IMG_SIZE,
            IMG_SIZE,
        ),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    return restored.clamp(
        0.0,
        1.0,
    )




def unwrap_state_dict(
    checkpoint,
):
    """
    Support common PyTorch checkpoint structures.
    """

    if not isinstance(
        checkpoint,
        dict,
    ):
        return checkpoint

    candidate_keys = [
        "state_dict",
        "model_state_dict",
        "model",
        "net",
    ]

    state = checkpoint

    for key in candidate_keys:
        if (
            key in checkpoint
            and isinstance(
                checkpoint[key],
                dict,
            )
        ):
            state = checkpoint[
                key
            ]
            break

    cleaned = {}

    for key, value in state.items():

        new_key = key

        if new_key.startswith(
            "module."
        ):
            new_key = new_key[
                len("module.") :
            ]

        if new_key.startswith(
            "model."
        ):
            new_key = new_key[
                len("model.") :
            ]

        cleaned[
            new_key
        ] = value

    return cleaned


def load_model(
    model_key,
):
    config = MODEL_CONFIGS[
        model_key
    ]

    checkpoint_path = config[
        "checkpoint"
    ]

    if not os.path.isfile(
        checkpoint_path
    ):
        raise FileNotFoundError(
            f"Missing checkpoint:\n"
            f"{checkpoint_path}"
        )

    model = config[
        "builder"
    ]()

    model = model.to(
        DEVICE
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
    )

    state_dict = unwrap_state_dict(
        checkpoint
    )

    try:
        model.load_state_dict(
            state_dict,
            strict=True,
        )

    except RuntimeError:
        print(
            "\n"
            + "=" * 100
        )

        print(
            f"FAILED TO LOAD CHECKPOINT FOR {model_key}"
        )

        print(
            f"Checkpoint: {checkpoint_path}"
        )

        print(
            "=" * 100
        )

        raise

    model.eval()

    print(
        f"Loaded {model_key}: "
        f"{checkpoint_path}"
    )

    return model




def get_condition_directory(
    model_key,
    condition,
):
    """
    Gradient attacks are model-specific.

    Environmental/noise/patch sets are shared.
    """

    if condition in GRADIENT_ATTACKS:

        return os.path.join(
            ATTACK_ROOT,
            model_key,
            f"{condition}_png",
        )

    shared_map = {
        "clean": "clean_png",
        "patch": "patch_png",
        "random_patch": "random_patch_png",
        "gaussian_noise": "gaussian_noise_png",
        "salt_pepper": "salt_pepper_png",
        "light": "light_png",
        "fog": "fog_png",
        "motion_blur": "motion_blur_png",
    }

    if condition not in shared_map:
        raise ValueError(
            f"Unknown condition: {condition}"
        )

    return os.path.join(
        ATTACK_ROOT,
        shared_map[
            condition
        ],
    )




@torch.inference_mode()
def classifier_forward(
    model,
    images,
):
    logits = model(
        normalize_batch(
            images
        )
    )

    logits = logits.float()

    probabilities = F.softmax(
        logits,
        dim=1,
    )

    confidence, prediction = (
        probabilities.max(
            dim=1
        )
    )


    energy = -torch.logsumexp(
        logits,
        dim=1,
    )

    return (
        logits,
        probabilities,
        prediction,
        confidence,
        energy,
    )




def jensen_shannon_divergence(
    p,
    q,
    eps=1e-12,
):
    """
    JS(P,Q) =
        0.5 KL(P || M)
        +
        0.5 KL(Q || M)

    where:

        M = 0.5 * (P + Q)
    """

    p = torch.clamp(
        p.float(),
        min=eps,
    )

    q = torch.clamp(
        q.float(),
        min=eps,
    )

    midpoint = (
        p + q
    ) * 0.5

    kl_p = torch.sum(
        p
        * (
            torch.log(p)
            - torch.log(
                midpoint
            )
        ),
        dim=1,
    )

    kl_q = torch.sum(
        q
        * (
            torch.log(q)
            - torch.log(
                midpoint
            )
        ),
        dim=1,
    )

    js = 0.5 * (
        kl_p
        + kl_q
    )

    return js




@torch.inference_mode()
def warmup_model(
    model,
):
    if DEVICE.type != "cuda":
        return

    dummy = torch.rand(
        8,
        3,
        IMG_SIZE,
        IMG_SIZE,
        device=DEVICE,
    )

    for _ in range(
        5
    ):
        _ = model(
            normalize_batch(
                dummy
            )
        )

    torch.cuda.synchronize()




def save_array(
    out_dir,
    name,
    array,
):
    np.save(
        os.path.join(
            out_dir,
            f"{name}.npy",
        ),
        array,
    )




@torch.inference_mode()
def extract_condition_signals(
    model,
    model_key,
    condition,
    critical_ids,
):
    input_dir = get_condition_directory(
        model_key,
        condition,
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        f"{model_key.upper()} | "
        f"{condition}"
    )

    print(
        "=" * 110
    )

    print(
        f"Input : {input_dir}"
    )

    dataset = GTSRBGeneratedDataset(
        input_dir,
        transform=IMAGE_TRANSFORM,
    )

    loader = DataLoader(
        dataset,
        batch_size=MODEL_CONFIGS[
            model_key
        ][
            "batch_size"
        ],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type
            == "cuda"
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    output_dir = os.path.join(
        OUT_ROOT,
        model_key,
        condition,
    )

    Path(
        output_dir
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

   

    buffers = {
        # identification
        "sample_id": [],
        "label": [],
        "pred": [],
        "correct": [],

        # base signals
        "confidence": [],
        "energy": [],

        # 2-pass
        "conf_drop_2": [],
        "logit_l2_2": [],
        "changed_2": [],

        # 3-pass critical
        "critical_pred_mask": [],
        "conf_drop_3": [],
        "logit_l2_3": [],
        "changed_3": [],

        # JS
        "js_divergence": [],
        "js_pred": [],
        "js_confidence": [],
        "js_changed": [],

      
        "entropy_base": [],
        "entropy_js": [],
    }

    filenames = []

    critical_ids_tensor = torch.tensor(
        critical_ids,
        dtype=torch.long,
        device=DEVICE,
    )

    start = time.perf_counter()

    
    for (
        images,
        labels,
        sample_ids,
        rel_paths,
    ) in tqdm(
        loader,
        desc=(
            f"{model_key}/{condition}"
        ),
    ):
        images = images.to(
            DEVICE,
            non_blocking=True,
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True,
        )

        sample_ids = sample_ids.to(
            DEVICE,
            non_blocking=True,
        )

        (
            logits,
            probs,
            pred,
            confidence,
            energy,
        ) = classifier_forward(
            model,
            images,
        )

        correct = (
            pred == labels
        )

       
        aug1 = weak_augment_1(
            images
        )

        (
            logits_aug1,
            probs_aug1,
            pred_aug1,
            confidence_aug1,
            _,
        ) = classifier_forward(
            model,
            aug1,
        )

        conf_drop_2 = (
            confidence
            - confidence_aug1
        )

        logit_l2_2 = torch.linalg.vector_norm(
            logits
            - logits_aug1,
            ord=2,
            dim=1,
        )

        changed_2 = (
            pred
            != pred_aug1
        )

        

        critical_mask = (
            pred.unsqueeze(1)
            == critical_ids_tensor.unsqueeze(0)
        ).any(
            dim=1
        )

        batch_size = images.size(
            0
        )

        
        conf_drop_3 = torch.full(
            (
                batch_size,
            ),
            -1.0,
            dtype=torch.float32,
            device=DEVICE,
        )

        logit_l2_3 = torch.full(
            (
                batch_size,
            ),
            -1.0,
            dtype=torch.float32,
            device=DEVICE,
        )

        changed_3 = torch.full(
            (
                batch_size,
            ),
            -1,
            dtype=torch.int8,
            device=DEVICE,
        )

       
        if bool(
            critical_mask.any()
        ):
            critical_images = images[
                critical_mask
            ]

            aug2 = weak_augment_2(
                critical_images
            )

            (
                logits_aug2,
                probs_aug2,
                pred_aug2,
                confidence_aug2,
                _,
            ) = classifier_forward(
                model,
                aug2,
            )

            base_logits_critical = logits[
                critical_mask
            ]

            base_pred_critical = pred[
                critical_mask
            ]

            base_conf_critical = confidence[
                critical_mask
            ]

            aug1_logits_critical = (
                logits_aug1[
                    critical_mask
                ]
            )

            aug1_pred_critical = (
                pred_aug1[
                    critical_mask
                ]
            )

            aug1_conf_critical = (
                confidence_aug1[
                    critical_mask
                ]
            )

            
            drop_aug1 = (
                base_conf_critical
                - aug1_conf_critical
            )

            drop_aug2 = (
                base_conf_critical
                - confidence_aug2
            )

        
            max_drop = torch.maximum(
                drop_aug1,
                drop_aug2,
            )

            l2_aug1 = (
                torch.linalg.vector_norm(
                    base_logits_critical
                    - aug1_logits_critical,
                    ord=2,
                    dim=1,
                )
            )

            l2_aug2 = (
                torch.linalg.vector_norm(
                    base_logits_critical
                    - logits_aug2,
                    ord=2,
                    dim=1,
                )
            )

            max_l2 = torch.maximum(
                l2_aug1,
                l2_aug2,
            )

           
            pred_changed = (
                (
                    base_pred_critical
                    != aug1_pred_critical
                )
                |
                (
                    base_pred_critical
                    != pred_aug2
                )
            )

            conf_drop_3[
                critical_mask
            ] = max_drop

            logit_l2_3[
                critical_mask
            ] = max_l2

            changed_3[
                critical_mask
            ] = pred_changed.to(
                torch.int8
            )

       
        js_images = js_resize_transform(
            images
        )

        (
            logits_js,
            probs_js,
            pred_js,
            confidence_js,
            _,
        ) = classifier_forward(
            model,
            js_images,
        )

        js_score = (
            jensen_shannon_divergence(
                probs,
                probs_js,
            )
        )

        js_changed = (
            pred
            != pred_js
        )

       
        eps = 1e-12

        entropy_base = -torch.sum(
            probs
            * torch.log(
                probs.clamp(
                    min=eps
                )
            ),
            dim=1,
        )

        entropy_js = -torch.sum(
            probs_js
            * torch.log(
                probs_js.clamp(
                    min=eps
                )
            ),
            dim=1,
        )

        
        buffers[
            "sample_id"
        ].append(
            sample_ids.cpu()
            .numpy()
            .astype(
                np.int64
            )
        )

        buffers[
            "label"
        ].append(
            labels.cpu()
            .numpy()
            .astype(
                np.int64
            )
        )

        buffers[
            "pred"
        ].append(
            pred.cpu()
            .numpy()
            .astype(
                np.int64
            )
        )

        buffers[
            "correct"
        ].append(
            correct.cpu()
            .numpy()
            .astype(
                np.bool_
            )
        )

        buffers[
            "confidence"
        ].append(
            confidence.cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        buffers[
            "energy"
        ].append(
            energy.cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        buffers[
            "conf_drop_2"
        ].append(
            conf_drop_2.cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        buffers[
            "logit_l2_2"
        ].append(
            logit_l2_2.cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        buffers[
            "changed_2"
        ].append(
            changed_2.cpu()
            .numpy()
            .astype(
                np.bool_
            )
        )

        buffers[
            "critical_pred_mask"
        ].append(
            critical_mask.cpu()
            .numpy()
            .astype(
                np.bool_
            )
        )

        buffers[
            "conf_drop_3"
        ].append(
            conf_drop_3.cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        buffers[
            "logit_l2_3"
        ].append(
            logit_l2_3.cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        buffers[
            "changed_3"
        ].append(
            changed_3.cpu()
            .numpy()
            .astype(
                np.int8
            )
        )

        buffers[
            "js_divergence"
        ].append(
            js_score.cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        buffers[
            "js_pred"
        ].append(
            pred_js.cpu()
            .numpy()
            .astype(
                np.int64
            )
        )

        buffers[
            "js_confidence"
        ].append(
            confidence_js.cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        buffers[
            "js_changed"
        ].append(
            js_changed.cpu()
            .numpy()
            .astype(
                np.bool_
            )
        )

        buffers[
            "entropy_base"
        ].append(
            entropy_base.cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        buffers[
            "entropy_js"
        ].append(
            entropy_js.cpu()
            .numpy()
            .astype(
                np.float32
            )
        )

        filenames.extend(
            rel_paths
        )

    
    arrays = {}

    for name, pieces in buffers.items():
        arrays[
            name
        ] = np.concatenate(
            pieces,
            axis=0,
        )

    filenames = np.asarray(
        filenames,
        dtype=str,
    )

    n = len(
        dataset
    )

  
    for name, array in arrays.items():
        if len(
            array
        ) != n:
            raise RuntimeError(
                f"{model_key}/{condition}: "
                f"{name} length = "
                f"{len(array)}, expected {n}"
            )

    if len(
        filenames
    ) != n:
        raise RuntimeError(
            f"{model_key}/{condition}: "
            f"filename count mismatch"
        )

    
    ids = arrays[
        "sample_id"
    ]

    if len(
        np.unique(ids)
    ) != n:
        raise RuntimeError(
            f"{model_key}/{condition}: "
            "sample IDs are not unique."
        )

   
    labels = arrays[
        "label"
    ]

    preds = arrays[
        "pred"
    ]

    correct_mask = (
        preds
        == labels
    )

    wrong_mask = (
        ~correct_mask
    )

    accuracy = (
        correct_mask.mean()
        * 100.0
    )

   
    for name, array in arrays.items():

        save_array(
            output_dir,
            name,
            array,
        )

    save_array(
        output_dir,
        "filenames",
        filenames,
    )

    
    save_array(
        output_dir,
        "2pass_conf_drop",
        arrays[
            "conf_drop_2"
        ],
    )

    save_array(
        output_dir,
        "2pass_logit_l2",
        arrays[
            "logit_l2_2"
        ],
    )

    save_array(
        output_dir,
        "2pass_changed",
        arrays[
            "changed_2"
        ],
    )

    save_array(
        output_dir,
        "3pass_max_conf_drop_critical",
        arrays[
            "conf_drop_3"
        ],
    )

    save_array(
        output_dir,
        "3pass_max_logit_l2_critical",
        arrays[
            "logit_l2_3"
        ],
    )

    save_array(
        output_dir,
        "3pass_changed_critical",
        arrays[
            "changed_3"
        ],
    )

   
    elapsed = (
        time.perf_counter()
        - start
    )

    critical_mask_np = arrays[
        "critical_pred_mask"
    ]

    stats = {
        "model": model_key,
        "condition": condition,

        "input_directory": input_dir,

        "num_samples": int(
            n
        ),

        "accuracy_percent": float(
            accuracy
        ),

        "num_correct": int(
            correct_mask.sum()
        ),

        "num_wrong": int(
            wrong_mask.sum()
        ),

        "mean_confidence": float(
            arrays[
                "confidence"
            ].mean()
        ),

        "mean_energy": float(
            arrays[
                "energy"
            ].mean()
        ),

        "mean_conf_drop_2": float(
            arrays[
                "conf_drop_2"
            ].mean()
        ),

        "mean_logit_l2_2": float(
            arrays[
                "logit_l2_2"
            ].mean()
        ),

        "changed_2_percent": float(
            arrays[
                "changed_2"
            ].mean()
            * 100.0
        ),

        "num_predicted_critical": int(
            critical_mask_np.sum()
        ),

        "mean_js": float(
            arrays[
                "js_divergence"
            ].mean()
        ),

        "median_js": float(
            np.median(
                arrays[
                    "js_divergence"
                ]
            )
        ),

        "p95_js": float(
            np.percentile(
                arrays[
                    "js_divergence"
                ],
                95,
            )
        ),

        "p99_js": float(
            np.percentile(
                arrays[
                    "js_divergence"
                ],
                99,
            )
        ),

        "js_prediction_changed_percent": float(
            arrays[
                "js_changed"
            ].mean()
            * 100.0
        ),

        "elapsed_seconds": float(
            elapsed
        ),

        "average_ms_per_image": float(
            elapsed
            * 1000.0
            / n
        ),
    }

    if bool(
        critical_mask_np.any()
    ):
        stats[
            "critical_mean_conf_drop_3"
        ] = float(
            arrays[
                "conf_drop_3"
            ][
                critical_mask_np
            ].mean()
        )

        stats[
            "critical_mean_logit_l2_3"
        ] = float(
            arrays[
                "logit_l2_3"
            ][
                critical_mask_np
            ].mean()
        )

        stats[
            "critical_changed_3_percent"
        ] = float(
            (
                arrays[
                    "changed_3"
                ][
                    critical_mask_np
                ]
                > 0
            ).mean()
            * 100.0
        )

    else:
        stats[
            "critical_mean_conf_drop_3"
        ] = None

        stats[
            "critical_mean_logit_l2_3"
        ] = None

        stats[
            "critical_changed_3_percent"
        ] = None

    with open(
        os.path.join(
            output_dir,
            "stats.json",
        ),
        "w",
    ) as f:

        json.dump(
            stats,
            f,
            indent=2,
        )

   
    print()

    print(
        f"N                      : {n}"
    )

    print(
        f"Accuracy               : "
        f"{accuracy:.3f}%"
    )

    print(
        f"Correct                : "
        f"{correct_mask.sum()}"
    )

    print(
        f"Wrong                  : "
        f"{wrong_mask.sum()}"
    )

    print(
        f"Mean confidence        : "
        f"{stats['mean_confidence']:.6f}"
    )

    print(
        f"Mean energy            : "
        f"{stats['mean_energy']:.6f}"
    )

    print(
        f"Mean conf drop (2)     : "
        f"{stats['mean_conf_drop_2']:.6f}"
    )

    print(
        f"Mean logit L2 (2)      : "
        f"{stats['mean_logit_l2_2']:.6f}"
    )

    print(
        f"Pred changed (2)       : "
        f"{stats['changed_2_percent']:.3f}%"
    )

    print(
        f"Predicted critical     : "
        f"{stats['num_predicted_critical']}"
    )

    print(
        f"Mean JS                : "
        f"{stats['mean_js']:.8f}"
    )

    print(
        f"Median JS              : "
        f"{stats['median_js']:.8f}"
    )

    print(
        f"P95 JS                 : "
        f"{stats['p95_js']:.8f}"
    )

    print(
        f"JS pred changed        : "
        f"{stats['js_prediction_changed_percent']:.3f}%"
    )

    print(
        f"Runtime                : "
        f"{elapsed:.2f} sec"
    )

    print(
        f"Saved                  : "
        f"{output_dir}"
    )

    return stats



def run_model(
    model_key,
    conditions,
    critical_ids,
):
    print(
        "\n"
        + "#" * 110
    )

    print(
        f"MODEL: "
        f"{model_key.upper()}"
    )

    print(
        "#" * 110
    )

    model = load_model(
        model_key
    )

    warmup_model(
        model
    )

    model_results = {}

    
    reference_ids = None
    reference_labels = None

    for condition in conditions:

        stats = extract_condition_signals(
            model=model,
            model_key=model_key,
            condition=condition,
            critical_ids=critical_ids,
        )

        model_results[
            condition
        ] = stats

        condition_dir = os.path.join(
            OUT_ROOT,
            model_key,
            condition,
        )

        ids = np.load(
            os.path.join(
                condition_dir,
                "sample_id.npy",
            )
        )

        labels = np.load(
            os.path.join(
                condition_dir,
                "label.npy",
            )
        )

       
        if reference_ids is None:
            reference_ids = ids
            reference_labels = labels

        else:
            if not np.array_equal(
                reference_ids,
                ids,
            ):
                raise RuntimeError(
                    "\n"
                    f"Sample-ID alignment mismatch for "
                    f"{model_key}/{condition}.\n"
                    "Do NOT continue to detector calibration."
                )

            if not np.array_equal(
                reference_labels,
                labels,
            ):
                raise RuntimeError(
                    "\n"
                    f"Label alignment mismatch for "
                    f"{model_key}/{condition}.\n"
                    "Do NOT continue to detector calibration."
                )

   
    summary_path = os.path.join(
        OUT_ROOT,
        model_key,
        "signal_extraction_summary.json",
    )

    with open(
        summary_path,
        "w",
    ) as f:

        json.dump(
            model_results,
            f,
            indent=2,
        )

    config = {
        "model": model_key,

        "checkpoint": MODEL_CONFIGS[
            model_key
        ][
            "checkpoint"
        ],

        "num_classes": NUM_CLASSES,

        "image_size": IMG_SIZE,

        "js_transform": {
            "original": IMG_SIZE,
            "downsample": JS_RESIZE_SIZE,
            "restore": IMG_SIZE,
            "mode": "bilinear",
            "antialias": True,
        },

        "critical_ids": [
            int(x)
            for x in critical_ids
        ],

        "conditions": list(
            conditions
        ),

        "mean": list(
            IMAGENET_MEAN
        ),

        "std": list(
            IMAGENET_STD
        ),
    }

    config_path = os.path.join(
        OUT_ROOT,
        model_key,
        "signal_extraction_config.json",
    )

    with open(
        config_path,
        "w",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )

    del model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return model_results



def print_final_summary(
    results,
    model_keys,
    conditions,
):
    print(
        "\n"
        + "=" * 140
    )

    print(
        "FINAL GTSRB SIGNAL EXTRACTION SUMMARY"
    )

    print(
        "=" * 140
    )

    header = (
        f"{'Model':<14}"
        f"{'Condition':<20}"
        f"{'N':>8}"
        f"{'Accuracy':>12}"
        f"{'MeanConf':>14}"
        f"{'MeanEnergy':>15}"
        f"{'MeanJS':>16}"
    )

    print(
        header
    )

    print(
        "-" * 140
    )

    for model_key in model_keys:

        for condition in conditions:

            stat = results[
                model_key
            ][
                condition
            ]

            print(
                f"{model_key:<14}"
                f"{condition:<20}"
                f"{stat['num_samples']:>8d}"
                f"{stat['accuracy_percent']:>11.3f}%"
                f"{stat['mean_confidence']:>14.6f}"
                f"{stat['mean_energy']:>15.6f}"
                f"{stat['mean_js']:>16.8f}"
            )



def validate_inputs(
    model_keys,
    conditions,
):
    print(
        "\nChecking required checkpoints and attack folders..."
    )

    errors = []

    for model_key in model_keys:

        checkpoint = MODEL_CONFIGS[
            model_key
        ][
            "checkpoint"
        ]

        if not os.path.isfile(
            checkpoint
        ):
            errors.append(
                f"Missing checkpoint: {checkpoint}"
            )

        for condition in conditions:

            directory = get_condition_directory(
                model_key,
                condition,
            )

            if not os.path.isdir(
                directory
            ):
                errors.append(
                    f"Missing folder: "
                    f"{model_key}/{condition} -> "
                    f"{directory}"
                )

    if errors:

        print(
            "\nINPUT VALIDATION FAILED"
        )

        for error in errors:
            print(
                f"  - {error}"
            )

        raise RuntimeError(
            "Required files/folders are missing."
        )

    print(
        "✅ All requested checkpoints and attack folders exist."
    )



def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract old detector + JS signals "
            "for all repeated GTSRB experiments."
        )
    )

    parser.add_argument(
        "--model",
        choices=[
            "mobilenet",
            "convnext",
            "efficientnet",
            "all",
        ],
        default="all",
    )

    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=ALL_CONDITIONS,
        default=ALL_CONDITIONS,
    )

    parser.add_argument(
        "--critical_ids",
        nargs="+",
        type=int,
        default=DEFAULT_CRITICAL_IDS,
    )

    args = parser.parse_args()

    
    if args.model == "all":

        model_keys = [
            "mobilenet",
            "convnext",
            "efficientnet",
        ]

    else:

        model_keys = [
            args.model
        ]

   
    for class_id in args.critical_ids:

        if not (
            0
            <= class_id
            < NUM_CLASSES
        ):
            raise ValueError(
                f"Invalid critical GTSRB class: "
                f"{class_id}"
            )

    Path(
        OUT_ROOT
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    
    print(
        "=" * 110
    )

    print(
        "GTSRB — ALL DETECTOR SIGNAL EXTRACTION"
    )

    print(
        "=" * 110
    )

    print(
        f"Device        : {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            "GPU           : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Models        : {model_keys}"
    )

    print(
        f"Conditions    : {args.conditions}"
    )

    print(
        f"Critical IDs  : {args.critical_ids}"
    )

    print(
        f"Output        : {OUT_ROOT}"
    )

   
    validate_inputs(
        model_keys,
        args.conditions,
    )

   
    all_results = {}

    for model_key in model_keys:

        all_results[
            model_key
        ] = run_model(
            model_key=model_key,
            conditions=args.conditions,
            critical_ids=args.critical_ids,
        )

   
    print_final_summary(
        all_results,
        model_keys,
        args.conditions,
    )

   
    global_summary_path = os.path.join(
        OUT_ROOT,
        "all_models_signal_summary.json",
    )

    with open(
        global_summary_path,
        "w",
    ) as f:

        json.dump(
            all_results,
            f,
            indent=2,
        )

    print(
        "\n"
        + "=" * 110
    )

    print(
        "DONE"
    )

    print(
        "=" * 110
    )

    print(
        "\nSignals saved under:"
    )

    print(
        f"    {OUT_ROOT}"
    )

    print(
        "\nNext step:"
    )

    print(
        "  1. Use CLEAN signals only to calibrate "
        "the original detector thresholds."
    )

    print(
        "  2. Freeze those thresholds."
    )

    print(
        "  3. Calibrate the JS threshold using CLEAN data only."
    )

    print(
        "  4. Compute Ours / JS / Ours+JS."
    )

    print(
        "  5. Report clean FPR, attack TPR, and ECDR."
    )

    print(
        "  6. Only then extract final suspicious samples for DDPM."
    )


if __name__ == "__main__":
    main()