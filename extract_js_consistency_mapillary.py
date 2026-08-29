import os
import json
import shutil
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.transforms import InterpolationMode
from tqdm import tqdm




CLEAN_DIR = "/mnt/dataset/mapillary_cropped/val"

IMG_SIZE = 224
PERTURB_SIZE = 208

NUM_WORKERS = 12

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ADV_ATTACKS = [
    "fgsm",
    "rfgsm",
    "pgd",
    "random_patch",
]

ENV_ATTACKS = [
    "gaussian",
    "salt_pepper",
    "light",
    "fog",
    "motion_blur",
]

ALL_CONDITIONS = [
    "clean",
    *ADV_ATTACKS,
    *ENV_ATTACKS,
]

OUT_ROOT = "./js_consistency_signal_mapillary"




def make_mobilenet(num_classes):
    m = models.mobilenet_v3_large(
        weights=None
    )
    m.classifier[-1] = nn.Linear(
        m.classifier[-1].in_features,
        num_classes
    )
    return m


def make_convnext(num_classes):
    m = models.convnext_tiny(
        weights=None
    )
    m.classifier[-1] = nn.Linear(
        m.classifier[-1].in_features,
        num_classes
    )
    return m


def make_efficientnet(num_classes):
    m = models.efficientnet_v2_s(
        weights=None
    )
    m.classifier[-1] = nn.Linear(
        m.classifier[-1].in_features,
        num_classes
    )
    return m


MODELS = {
    "mobilenet": {
        "build": make_mobilenet,
        "ckpt": "./mapillary_baseline_results/mobilenetv3_mapillary_best.pth",
        "class_to_idx": "./mapillary_baseline_results/class_to_idx.json",
        "adv_root": "./attacks_mobilenet_eps",
        "env_root": "./env_mobilenet",
        "batch_size": 128,
    },
    "convnext": {
        "build": make_convnext,
        "ckpt": "./convnext_mapillary_results/convnext_mapillary_best.pth",
        "class_to_idx": "./convnext_mapillary_results/class_to_idx.json",
        "adv_root": "./attacks_convnext_eps",
        "env_root": "./env_convnext",
        "batch_size": 96,
    },
    "efficientnet": {
        "build": make_efficientnet,
        "ckpt": "./efficientnetv2_mapillary_results/efficientnetv2_mapillary_best.pth",
        "class_to_idx": "./efficientnetv2_mapillary_results/class_to_idx.json",
        "adv_root": "./attacks_efficientnet_eps",
        "env_root": "./env_efficientnet",
        "batch_size": 96,
    },
}




class FixedClassImageFolderWithPaths(datasets.ImageFolder):
   

    def __init__(
        self,
        root,
        class_to_idx,
        transform=None,
    ):
        self.fixed_class_to_idx = class_to_idx

        super().__init__(
            root=root,
            transform=transform,
            allow_empty=True,
        )

    def find_classes(self, directory):
        classes = list(
            self.fixed_class_to_idx.keys()
        )
        return (
            classes,
            self.fixed_class_to_idx,
        )

    def __getitem__(self, index):
        image, label = super().__getitem__(
            index
        )
        path = self.samples[index][0]
        return image, label, path




def load_class_map(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"class_to_idx missing: {path}"
        )

    with open(path, "r") as f:
        mapping = json.load(f)

    return {
        str(k): int(v)
        for k, v in mapping.items()
    }


def load_model(cfg, num_classes):
    if not os.path.isfile(cfg["ckpt"]):
        raise FileNotFoundError(
            f"checkpoint missing: {cfg['ckpt']}"
        )

    model = cfg["build"](
        num_classes
    ).to(DEVICE)

    state = torch.load(
        cfg["ckpt"],
        map_location=DEVICE,
    )

    model.load_state_dict(
        state
    )

    model.eval()

    return model


def get_image_dir(cfg, condition):
    if condition == "clean":
        return CLEAN_DIR

    if condition in ADV_ATTACKS:
        return os.path.join(
            cfg["adv_root"],
            f"{condition}_png",
        )

    if condition in ENV_ATTACKS:
        return os.path.join(
            cfg["env_root"],
            f"{condition}_png",
        )

    raise ValueError(
        f"Unknown condition: {condition}"
    )


def js_divergence_from_logits(
    logits_a,
    logits_b,
    eps=1e-12,
):
   

    p = torch.softmax(
        logits_a.float(),
        dim=1,
    )

    q = torch.softmax(
        logits_b.float(),
        dim=1,
    )

    p = torch.clamp(
        p,
        min=eps,
    )

    q = torch.clamp(
        q,
        min=eps,
    )

    m = 0.5 * (
        p + q
    )

    kl_pm = torch.sum(
        p * (
            torch.log(p)
            - torch.log(m)
        ),
        dim=1,
    )

    kl_qm = torch.sum(
        q * (
            torch.log(q)
            - torch.log(m)
        ),
        dim=1,
    )

    js = 0.5 * (
        kl_pm + kl_qm
    )

    return (
        js,
        p,
        q,
    )


def normalize_batch(x):
    mean = torch.tensor(
        MEAN,
        device=x.device,
        dtype=x.dtype,
    ).view(
        1, 3, 1, 1
    )

    std = torch.tensor(
        STD,
        device=x.device,
        dtype=x.dtype,
    ).view(
        1, 3, 1, 1
    )

    return (
        x - mean
    ) / std


def mild_resize_transform(x):
    

    x_small = torch.nn.functional.interpolate(
        x,
        size=(PERTURB_SIZE, PERTURB_SIZE),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    x_back = torch.nn.functional.interpolate(
        x_small,
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )

    return torch.clamp(
        x_back,
        0.0,
        1.0,
    )



@torch.inference_mode()
def extract_condition(
    model,
    model_key,
    condition,
    image_dir,
    class_to_idx,
    batch_size,
):
    if not os.path.isdir(image_dir):
        print(
            f"[{model_key}/{condition}] SKIP - "
            f"missing directory: {image_dir}"
        )
        return None

    out_dir = os.path.join(
        OUT_ROOT,
        model_key,
        condition,
    )

   
    if os.path.exists(out_dir):
        shutil.rmtree(
            out_dir
        )

    Path(out_dir).mkdir(
        parents=True,
        exist_ok=True,
    )

 
    tfms = transforms.Compose([
        transforms.Resize(
            (IMG_SIZE, IMG_SIZE),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
    ])

    ds = FixedClassImageFolderWithPaths(
        image_dir,
        class_to_idx,
        transform=tfms,
    )

    if len(ds) == 0:
        print(
            f"[{model_key}/{condition}] "
            f"SKIP - no images"
        )
        return None

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type == "cuda"
        ),
        persistent_workers=(
            NUM_WORKERS > 0
        ),
    )

    js_all = []
    labels_all = []
    pred_base_all = []
    pred_trans_all = []
    conf_base_all = []
    conf_trans_all = []
    changed_all = []
    entropy_base_all = []
    entropy_trans_all = []
    filenames_all = []

    for images, labels, paths in tqdm(
        loader,
        desc=f"{model_key}/{condition}",
    ):
        images = images.to(
            DEVICE,
            non_blocking=True,
        )

       
        x_base = normalize_batch(
            images
        )

        
        transformed = mild_resize_transform(
            images
        )

        x_trans = normalize_batch(
            transformed
        )

        logits_base = model(
            x_base
        )

        logits_trans = model(
            x_trans
        )

        js, p, q = (
            js_divergence_from_logits(
                logits_base,
                logits_trans,
            )
        )

        conf_base, pred_base = (
            p.max(
                dim=1
            )
        )

        conf_trans, pred_trans = (
            q.max(
                dim=1
            )
        )

        changed = (
            pred_base != pred_trans
        )

        entropy_base = -torch.sum(
            p * torch.log(
                torch.clamp(
                    p,
                    min=1e-12,
                )
            ),
            dim=1,
        )

        entropy_trans = -torch.sum(
            q * torch.log(
                torch.clamp(
                    q,
                    min=1e-12,
                )
            ),
            dim=1,
        )

        js_all.append(
            js.cpu().numpy().astype(
                np.float32
            )
        )

        labels_all.append(
            labels.numpy().astype(
                np.int64
            )
        )

        pred_base_all.append(
            pred_base.cpu().numpy().astype(
                np.int64
            )
        )

        pred_trans_all.append(
            pred_trans.cpu().numpy().astype(
                np.int64
            )
        )

        conf_base_all.append(
            conf_base.cpu().numpy().astype(
                np.float32
            )
        )

        conf_trans_all.append(
            conf_trans.cpu().numpy().astype(
                np.float32
            )
        )

        changed_all.append(
            changed.cpu().numpy().astype(
                np.bool_
            )
        )

        entropy_base_all.append(
            entropy_base.cpu().numpy().astype(
                np.float32
            )
        )

        entropy_trans_all.append(
            entropy_trans.cpu().numpy().astype(
                np.float32
            )
        )

        filenames_all.extend(
            list(paths)
        )

    js_np = np.concatenate(
        js_all
    )

    labels_np = np.concatenate(
        labels_all
    )

    pred_base_np = np.concatenate(
        pred_base_all
    )

    pred_trans_np = np.concatenate(
        pred_trans_all
    )

    conf_base_np = np.concatenate(
        conf_base_all
    )

    conf_trans_np = np.concatenate(
        conf_trans_all
    )

    changed_np = np.concatenate(
        changed_all
    )

    entropy_base_np = np.concatenate(
        entropy_base_all
    )

    entropy_trans_np = np.concatenate(
        entropy_trans_all
    )

    
    root_abs = os.path.abspath(
        image_dir
    )

    rel_paths = []

    for p in filenames_all:
        p_abs = os.path.abspath(
            p
        )

        rel_paths.append(
            os.path.relpath(
                p_abs,
                root_abs,
            )
        )

    filenames_np = np.asarray(
        rel_paths,
        dtype=str,
    )

    n = len(ds)

    if not (
        len(js_np)
        == len(labels_np)
        == len(pred_base_np)
        == len(pred_trans_np)
        == len(conf_base_np)
        == len(conf_trans_np)
        == len(changed_np)
        == len(entropy_base_np)
        == len(entropy_trans_np)
        == len(filenames_np)
        == n
    ):
        raise RuntimeError(
            f"[{model_key}/{condition}] "
            "saved arrays are misaligned."
        )

    
    np.save(
        os.path.join(
            out_dir,
            "js_divergence.npy",
        ),
        js_np,
    )

    np.save(
        os.path.join(
            out_dir,
            "labels.npy",
        ),
        labels_np,
    )

    np.save(
        os.path.join(
            out_dir,
            "pred_base.npy",
        ),
        pred_base_np,
    )

    np.save(
        os.path.join(
            out_dir,
            "pred_transformed.npy",
        ),
        pred_trans_np,
    )

    np.save(
        os.path.join(
            out_dir,
            "confidence_base.npy",
        ),
        conf_base_np,
    )

    np.save(
        os.path.join(
            out_dir,
            "confidence_transformed.npy",
        ),
        conf_trans_np,
    )

    np.save(
        os.path.join(
            out_dir,
            "prediction_changed.npy",
        ),
        changed_np,
    )

    np.save(
        os.path.join(
            out_dir,
            "entropy_base.npy",
        ),
        entropy_base_np,
    )

    np.save(
        os.path.join(
            out_dir,
            "entropy_transformed.npy",
        ),
        entropy_trans_np,
    )

    np.save(
        os.path.join(
            out_dir,
            "filenames.npy",
        ),
        filenames_np,
    )

    base_acc = float(
        np.mean(
            pred_base_np
            == labels_np
        )
    )

    transformed_acc = float(
        np.mean(
            pred_trans_np
            == labels_np
        )
    )

    changed_rate = float(
        np.mean(
            changed_np
        )
    )

    metadata = {
        "model": model_key,
        "condition": condition,
        "source_dir": image_dir,
        "num_images": int(n),
        "num_classes": int(
            len(class_to_idx)
        ),
        "base_image_size": IMG_SIZE,
        "transformation": (
            f"bilinear resize "
            f"{IMG_SIZE}->{PERTURB_SIZE}->{IMG_SIZE}"
        ),
        "signal": (
            "Jensen-Shannon divergence between the full "
            "classifier probability vector on the original "
            "image and on the mildly resized image."
        ),
        "score_direction": (
            "higher = less prediction-distribution consistency"
        ),
        "base_accuracy": base_acc,
        "transformed_accuracy": transformed_acc,
        "prediction_changed_rate": changed_rate,
        "js_mean": float(
            np.mean(
                js_np
            )
        ),
        "js_median": float(
            np.median(
                js_np
            )
        ),
        "js_p95": float(
            np.percentile(
                js_np,
                95,
            )
        ),
        "output_dir": out_dir,
    }

    with open(
        os.path.join(
            out_dir,
            "metadata.json",
        ),
        "w",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print(
        f"\n[{model_key}/{condition}] DONE"
    )
    print(
        f"  images          : {n:,}"
    )
    print(
        f"  base accuracy   : "
        f"{100.0 * base_acc:.2f}%"
    )
    print(
        f"  transformed acc : "
        f"{100.0 * transformed_acc:.2f}%"
    )
    print(
        f"  pred changed    : "
        f"{100.0 * changed_rate:.2f}%"
    )
    print(
        f"  JS mean/median  : "
        f"{np.mean(js_np):.6f} / "
        f"{np.median(js_np):.6f}"
    )
    print(
        f"  JS p95          : "
        f"{np.percentile(js_np, 95):.6f}"
    )
    print(
        f"  output          : {out_dir}"
    )

    return metadata




def parse_args():
    parser = argparse.ArgumentParser()

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
        choices=ALL_CONDITIONS,
        default=ALL_CONDITIONS,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    selected_models = (
        list(MODELS.keys())
        if args.model == "all"
        else [args.model]
    )

    print("=" * 88)
    print(
        "MAPILLARY JS PROBABILITY-CONSISTENCY SIGNAL EXTRACTION"
    )
    print("=" * 88)
    print(
        f"Device           : {DEVICE}"
    )
    print(
        f"Models           : {selected_models}"
    )
    print(
        f"Conditions       : {args.conditions}"
    )
    print(
        f"Base size        : {IMG_SIZE}"
    )
    print(
        f"Perturbation     : "
        f"{IMG_SIZE}->{PERTURB_SIZE}->{IMG_SIZE}"
    )
    print(
        f"NEW output root  : {OUT_ROOT}"
    )
    print(
        "Previous results : NOT MODIFIED"
    )

    Path(
        OUT_ROOT
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    combined = []

    for model_key in selected_models:
        cfg = MODELS[
            model_key
        ]

        class_to_idx = (
            load_class_map(
                cfg["class_to_idx"]
            )
        )

        num_classes = len(
            class_to_idx
        )

        model = load_model(
            cfg,
            num_classes,
        )

        print(
            "\n" + "=" * 88
        )
        print(
            f"[{model_key}] MODEL READY"
        )
        print(
            "=" * 88
        )
        print(
            f"checkpoint : {cfg['ckpt']}"
        )
        print(
            f"classes    : {num_classes}"
        )

        for condition in args.conditions:
            image_dir = get_image_dir(
                cfg,
                condition,
            )

            result = extract_condition(
                model=model,
                model_key=model_key,
                condition=condition,
                image_dir=image_dir,
                class_to_idx=class_to_idx,
                batch_size=cfg[
                    "batch_size"
                ],
            )

            if result is not None:
                combined.append(
                    result
                )

        del model

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    summary_path = os.path.join(
        OUT_ROOT,
        "js_extraction_summary.json",
    )

    with open(
        summary_path,
        "w",
    ) as f:
        json.dump(
            combined,
            f,
            indent=2,
        )

    print(
        "\n" + "=" * 88
    )
    print(
        "JS SIGNAL EXTRACTION COMPLETE"
    )
    print(
        "=" * 88
    )
    print(
        f"Summary -> {summary_path}"
    )
    print(
        "\nNo previous detector/attack/"
        "Mahalanobis/DDPM results were modified."
    )


if __name__ == "__main__":
    main()