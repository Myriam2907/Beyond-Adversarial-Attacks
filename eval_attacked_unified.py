import os
import json
import time
import shutil
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm


CLEAN_DIR = "/mnt/dataset/mapillary_cropped/val"

IMG_SIZE = 224
NUM_WORKERS = 12

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CRITICAL_IDS = [241, 242, 243, 265]

ADV_ATTACKS = ["fgsm", "rfgsm", "pgd", "random_patch"]
ENV_ATTACKS = ["gaussian", "salt_pepper", "light", "fog", "motion_blur"]
ALL_ATTACKS = ["clean"] + ADV_ATTACKS + ENV_ATTACKS


def make_mobilenet(num_classes):
    m = models.mobilenet_v3_large(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


def make_convnext(num_classes):
    m = models.convnext_tiny(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


def make_efficientnet(num_classes):
    m = models.efficientnet_v2_s(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


MODELS = {
    "mobilenet": {
        "build": make_mobilenet,
        "ckpt": "./mapillary_baseline_results/mobilenetv3_mapillary_best.pth",
        "class_to_idx": "./mapillary_baseline_results/class_to_idx.json",
        "adv_root": "./attacks_mobilenet_eps",
        "env_root": "./env_mobilenet",
        "out_root": "./eval_mobilenet",
        "batch_size": 64,
    },
    "convnext": {
        "build": make_convnext,
        "ckpt": "./convnext_mapillary_results/convnext_mapillary_best.pth",
        "class_to_idx": "./convnext_mapillary_results/class_to_idx.json",
        "adv_root": "./attacks_convnext_eps",
        "env_root": "./env_convnext",
        "out_root": "./eval_convnext",
        "batch_size": 64,
    },
    "efficientnet": {
        "build": make_efficientnet,
        "ckpt": "./efficientnetv2_mapillary_results/efficientnetv2_mapillary_best.pth",
        "class_to_idx": "./efficientnetv2_mapillary_results/class_to_idx.json",
        "adv_root": "./attacks_efficientnet_eps",
        "env_root": "./env_efficientnet",
        "out_root": "./eval_efficientnet",
        "batch_size": 64,
    },
}


class FixedClassImageFolderWithPaths(datasets.ImageFolder):
    def __init__(self, root, class_to_idx, transform=None):
        self.fixed_class_to_idx = class_to_idx
        super().__init__(root=root, transform=transform, allow_empty=True)

    def find_classes(self, directory):
        classes = list(self.fixed_class_to_idx.keys())
        return classes, self.fixed_class_to_idx

    def __getitem__(self, index):
        img, label = super().__getitem__(index)
        path = self.samples[index][0]
        return img, label, path, index


def mean_std(device, dtype=torch.float32):
    mean = torch.tensor(MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def normalize(x):
    mean, std = mean_std(x.device, x.dtype)
    return (x - mean) / std


def weak_augment_1(x):
    y = ((x - 0.5) * 1.03 + 0.5 + 0.015).clamp(0, 1)
    blur = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)
    return (0.85 * y + 0.15 * blur).clamp(0, 1)


def weak_augment_2(x):
    y = ((x - 0.5) * 0.97 + 0.5 - 0.010).clamp(0, 1)
    blur = F.avg_pool2d(y, kernel_size=5, stride=1, padding=2)
    return (0.90 * y + 0.10 * blur).clamp(0, 1)


@torch.no_grad()
def forward_signals(model, x):
    logits = model(normalize(x))
    probs = F.softmax(logits, dim=1)
    confidence, pred = probs.max(dim=1)
    energy = -torch.logsumexp(logits, dim=1)
    return logits, pred, confidence, energy


def validate_critical_ids(class_to_idx, model_key):
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}

    print(f"\n[{model_key}] CRITICAL_IDS -> class names:")

    missing = []
    for cid in CRITICAL_IDS:
        name = idx_to_class.get(cid)
        if name is None:
            missing.append(cid)
            name = "<<NOT FOUND>>"
        print(f"    {cid}: {name}")

    if missing:
        raise ValueError(
            f"{model_key}: CRITICAL_IDS not found in class map: {missing}"
        )

    return idx_to_class


def get_image_dir(cfg, attack):
    if attack == "clean":
        return CLEAN_DIR
    if attack in ADV_ATTACKS:
        return os.path.join(cfg["adv_root"], f"{attack}_png")
    if attack in ENV_ATTACKS:
        return os.path.join(cfg["env_root"], f"{attack}_png")
    raise ValueError(f"Unknown attack/corruption: {attack}")


def load_model(cfg, num_classes):
    if not os.path.exists(cfg["ckpt"]):
        raise FileNotFoundError(f"checkpoint missing: {cfg['ckpt']}")

    model = cfg["build"](num_classes).to(DEVICE)
    state = torch.load(cfg["ckpt"], map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def gpu_warmup(model):
    if DEVICE.type != "cuda":
        return

    warm = torch.randn(8, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)

    with torch.no_grad():
        for _ in range(10):
            _ = model(warm)

    torch.cuda.synchronize()


@torch.no_grad()
def eval_folder(model, img_dir, out_dir, class_to_idx, batch_size):
    if not os.path.isdir(img_dir):
        print(f"  SKIP - missing directory: {img_dir}")
        return None

    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    ds = FixedClassImageFolderWithPaths(
        img_dir,
        class_to_idx,
        transform=tfms
    )

    if len(ds) == 0:
        print(f"  SKIP - 0 images: {img_dir}")
        return None

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda")
    )

    buf = {
        k: []
        for k in [
            "label",
            "pred",
            "confidence",
            "energy",
            "cd2",
            "l2_2",
            "ch2",
            "cd3",
            "l2_3",
            "ch3",
            "critical_pred_mask",
        ]
    }

    filenames = []
    total_ms = 0.0
    crit_set = set(CRITICAL_IDS)

    for x, y, paths, idx in tqdm(loader, desc=f"  {os.path.basename(out_dir)}"):
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        filenames.extend([str(p) for p in paths])

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()

        logits, pred, conf, energy = forward_signals(model, x)

        x1 = weak_augment_1(x)
        logits1, pred1, conf1, _ = forward_signals(model, x1)

        cd2 = conf - conf1
        l2_2 = torch.norm(logits - logits1, p=2, dim=1)
        ch2 = (pred != pred1).long()

        cd3 = torch.full_like(conf, -1.0)
        l2_3 = torch.full_like(conf, -1.0)
        ch3 = torch.full_like(pred, -1)

        critical_pred_mask = torch.zeros_like(pred, dtype=torch.bool)
        for cid in crit_set:
            critical_pred_mask |= (pred == cid)

        if critical_pred_mask.any():
            xc = x[critical_pred_mask]
            x2 = weak_augment_2(xc)

            logits2, pred2, conf2, _ = forward_signals(model, x2)

            conf_base_c = conf[critical_pred_mask]
            conf1_c = conf1[critical_pred_mask]

            cd1c = conf_base_c - conf1_c
            cd2c = conf_base_c - conf2

            cd3[critical_pred_mask] = torch.maximum(cd1c, cd2c)

            l21 = torch.norm(
                logits[critical_pred_mask] - logits1[critical_pred_mask],
                p=2,
                dim=1
            )
            l22 = torch.norm(
                logits[critical_pred_mask] - logits2,
                p=2,
                dim=1
            )

            l2_3[critical_pred_mask] = torch.maximum(l21, l22)

            ch3[critical_pred_mask] = (
                (pred[critical_pred_mask] != pred1[critical_pred_mask]) |
                (pred[critical_pred_mask] != pred2)
            ).long()

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        total_ms += (time.perf_counter() - t0) * 1000.0

        buf["label"].append(y.cpu().numpy())
        buf["pred"].append(pred.cpu().numpy())
        buf["confidence"].append(conf.cpu().numpy())
        buf["energy"].append(energy.cpu().numpy())
        buf["cd2"].append(cd2.cpu().numpy())
        buf["l2_2"].append(l2_2.cpu().numpy())
        buf["ch2"].append(ch2.cpu().numpy())
        buf["cd3"].append(cd3.cpu().numpy())
        buf["l2_3"].append(l2_3.cpu().numpy())
        buf["ch3"].append(ch3.cpu().numpy())
        buf["critical_pred_mask"].append(
            critical_pred_mask.long().cpu().numpy()
        )

    cat = {k: np.concatenate(v) for k, v in buf.items()}
    filenames_arr = np.asarray(filenames, dtype=object)

    n = len(ds)

    for key, arr in cat.items():
        if len(arr) != n:
            raise RuntimeError(
                f"Alignment error for {key}: {len(arr)} values "
                f"but dataset has {n} samples."
            )

    if len(filenames_arr) != n:
        raise RuntimeError(
            f"Alignment error for filenames: {len(filenames_arr)} paths "
            f"but dataset has {n} samples."
        )

    np.save(os.path.join(out_dir, "label.npy"), cat["label"])
    np.save(os.path.join(out_dir, "pred.npy"), cat["pred"])
    np.save(os.path.join(out_dir, "filenames.npy"), filenames_arr)
    np.save(os.path.join(out_dir, "confidence.npy"), cat["confidence"])
    np.save(os.path.join(out_dir, "energy.npy"), cat["energy"])
    np.save(os.path.join(out_dir, "2pass_conf_drop.npy"), cat["cd2"])
    np.save(os.path.join(out_dir, "2pass_logit_l2.npy"), cat["l2_2"])
    np.save(os.path.join(out_dir, "2pass_changed.npy"), cat["ch2"])
    np.save(
        os.path.join(out_dir, "3pass_max_conf_drop_critical.npy"),
        cat["cd3"]
    )
    np.save(
        os.path.join(out_dir, "3pass_max_logit_l2_critical.npy"),
        cat["l2_3"]
    )
    np.save(
        os.path.join(out_dir, "3pass_changed_critical.npy"),
        cat["ch3"]
    )
    np.save(
        os.path.join(out_dir, "critical_pred_mask.npy"),
        cat["critical_pred_mask"]
    )

    acc = float((cat["pred"] == cat["label"]).mean() * 100.0)
    avg_ms = float(total_ms / len(ds))

    critical_count = int(cat["critical_pred_mask"].sum())

    stats = {
        "accuracy_percent": acc,
        "avg_detection_ms_per_img": avg_ms,
        "num_samples": int(len(ds)),
        "mean_confidence": float(cat["confidence"].mean()),
        "mean_energy": float(cat["energy"].mean()),
        "mean_2pass_conf_drop": float(cat["cd2"].mean()),
        "mean_2pass_logit_l2": float(cat["l2_2"].mean()),
        "frac_2pass_changed": float(cat["ch2"].mean()),
        "num_predicted_critical": critical_count,
        "frac_predicted_critical": float(critical_count / len(ds)),
    }

    cmask_np = cat["critical_pred_mask"].astype(bool)

    if cmask_np.any():
        stats["mean_3pass_max_conf_drop_predicted_critical"] = float(
            cat["cd3"][cmask_np].mean()
        )
        stats["mean_3pass_max_logit_l2_predicted_critical"] = float(
            cat["l2_3"][cmask_np].mean()
        )
        stats["frac_3pass_changed_predicted_critical"] = float(
            cat["ch3"][cmask_np].mean()
        )
    else:
        stats["mean_3pass_max_conf_drop_predicted_critical"] = None
        stats["mean_3pass_max_logit_l2_predicted_critical"] = None
        stats["frac_3pass_changed_predicted_critical"] = None

    with open(os.path.join(out_dir, "eval_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(
        f"    acc={acc:.2f}%  "
        f"conf={stats['mean_confidence']:.3f}  "
        f"energy={stats['mean_energy']:.3f}  "
        f"cd2={stats['mean_2pass_conf_drop']:.4f}  "
        f"ch2={stats['frac_2pass_changed']:.4f}  "
        f"critical_pred={critical_count}/{len(ds)}  "
        f"{avg_ms:.3f} ms/img"
    )

    return stats


def eval_model(model_key, attacks, delete_old_results=True):
    cfg = MODELS[model_key]

    if not os.path.exists(cfg["ckpt"]):
        raise FileNotFoundError(f"checkpoint missing: {cfg['ckpt']}")
    if not os.path.exists(cfg["class_to_idx"]):
        raise FileNotFoundError(
            f"class_to_idx missing: {cfg['class_to_idx']}"
        )

    with open(cfg["class_to_idx"], "r") as f:
        class_to_idx = json.load(f)

    idx_to_class = validate_critical_ids(class_to_idx, model_key)
    num_classes = len(class_to_idx)

    if delete_old_results and os.path.exists(cfg["out_root"]):
        print(
            f"[{model_key}] deleting old evaluation results: "
            f"{cfg['out_root']}"
        )
        shutil.rmtree(cfg["out_root"])

    Path(cfg["out_root"]).mkdir(parents=True, exist_ok=True)

    model = load_model(cfg, num_classes)

    print(
        f"[{model_key}] loaded {cfg['ckpt']}  classes={num_classes}"
    )

    gpu_warmup(model)

    results = {}

    for attack in attacks:
        img_dir = get_image_dir(cfg, attack)
        out_dir = os.path.join(cfg["out_root"], attack)

        print(f"\n[{model_key}] {attack} <- {img_dir}")

        results[attack] = eval_folder(
            model=model,
            img_dir=img_dir,
            out_dir=out_dir,
            class_to_idx=class_to_idx,
            batch_size=cfg["batch_size"]
        )

    run_config = {
        "model": model_key,
        "checkpoint": cfg["ckpt"],
        "class_to_idx": cfg["class_to_idx"],
        "clean_dir": CLEAN_DIR,
        "adv_root": cfg["adv_root"],
        "env_root": cfg["env_root"],
        "out_root": cfg["out_root"],
        "image_size": IMG_SIZE,
        "critical_ids": CRITICAL_IDS,
        "critical_names": {
            str(cid): idx_to_class[cid]
            for cid in CRITICAL_IDS
        },
        "critical_pass_gating": "base_prediction",
        "attacks_requested": attacks,
        "weak_augment_1": {
            "contrast": 1.03,
            "brightness": 0.015,
            "avg_pool_kernel": 3,
            "mix_original": 0.85,
            "mix_blur": 0.15,
        },
        "weak_augment_2": {
            "contrast": 0.97,
            "brightness": -0.010,
            "avg_pool_kernel": 5,
            "mix_original": 0.90,
            "mix_blur": 0.10,
        },
    }

    with open(
        os.path.join(cfg["out_root"], "evaluation_run_config.json"),
        "w"
    ) as f:
        json.dump(run_config, f, indent=2)

    del model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return results


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        choices=list(MODELS.keys()) + ["all"],
        required=True
    )

    ap.add_argument(
        "--attacks",
        nargs="+",
        choices=ALL_ATTACKS,
        default=ALL_ATTACKS,
        help=(
            "Attacks/corruptions to evaluate. "
            "Default: clean + all adversarial + all environmental."
        )
    )

    ap.add_argument(
        "--keep_old",
        action="store_true",
        help=(
            "Do not delete the entire model evaluation directory first. "
            "Requested attack subfolders are still overwritten."
        )
    )

    args = ap.parse_args()

    print("=" * 78)
    print("UNIFIED DETECTOR FEATURE EVALUATION V2")
    print("=" * 78)
    print("Device:", DEVICE)
    print("Attacks:", args.attacks)

    keys = (
        list(MODELS.keys())
        if args.model == "all"
        else [args.model]
    )

    all_results = {}

    for model_key in keys:
        all_results[model_key] = eval_model(
            model_key=model_key,
            attacks=args.attacks,
            delete_old_results=not args.keep_old
        )

    print(
        "\n==================== SUMMARY (accuracy %) ===================="
    )

    header = (
        "model        "
        + "  ".join(
            f"{a[:10]:>10s}"
            for a in args.attacks
        )
    )

    print(header)

    for model_key in keys:
        row = f"{model_key:13s}"

        for attack in args.attacks:
            stats = all_results[model_key].get(attack)

            if stats is None:
                row += f"  {'--':>10s}"
            else:
                row += f"  {stats['accuracy_percent']:10.2f}"

        print(row)


if __name__ == "__main__":
    main()
