

import os
import json
import csv
import shutil
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
from tqdm import tqdm



VAL_DIR = "/mnt/dataset/mapillary_cropped/val"

IMG_SIZE = 224
NUM_WORKERS = 12
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ATTACKS_TO_RUN = ["gaussian", "salt_pepper", "light", "fog", "motion_blur"]


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
        "out_root": "./env_mobilenet",
        "batch_size": 64,
    },
    "convnext": {
        "build": make_convnext,
        "ckpt": "./convnext_mapillary_results/convnext_mapillary_best.pth",
        "class_to_idx": "./convnext_mapillary_results/class_to_idx.json",
        "out_root": "./env_convnext",
        "batch_size": 128,
    },
    "efficientnet": {
        "build": make_efficientnet,
        "ckpt": "./efficientnetv2_mapillary_results/efficientnetv2_mapillary_best.pth",
        "class_to_idx": "./efficientnetv2_mapillary_results/class_to_idx.json",
        "out_root": "./env_efficientnet",
        "batch_size": 128,
    },
}



FROZEN_PARAMS = {
    "gaussian": {"sigma": 0.30},
    "salt_pepper": {"prob": 0.20},
    "light": {"brightness": -0.45, "contrast": 0.55, "gamma": 3.0},
    "fog": {"fog_strength": 0.95, "blur_kernel": 111},
    "motion_blur": {"kernel_size": 61},
}

CANDIDATES = {
    "gaussian": [
        {"sigma": s}
        for s in (0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
    ],

    "salt_pepper": [
        {"prob": p}
        for p in (0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45)
    ],

    
    "light": [
        {"brightness": -0.20, "contrast": 0.70, "gamma": 1.5},
        {"brightness": -0.30, "contrast": 0.60, "gamma": 2.0},
        {"brightness": -0.40, "contrast": 0.55, "gamma": 2.5},
        {"brightness": -0.45, "contrast": 0.55, "gamma": 3.0},
        {"brightness": -0.50, "contrast": 0.50, "gamma": 3.5},
        {"brightness": -0.55, "contrast": 0.45, "gamma": 4.0},
        {"brightness": 0.25, "contrast": 1.80, "gamma": 1.0},
        {"brightness": 0.30, "contrast": 2.20, "gamma": 1.0},
        {"brightness": 0.35, "contrast": 2.60, "gamma": 1.0},
    ],

    "fog": [
        {"fog_strength": 0.65, "blur_kernel": 81},
        {"fog_strength": 0.75, "blur_kernel": 91},
        {"fog_strength": 0.85, "blur_kernel": 111},
        {"fog_strength": 0.95, "blur_kernel": 111},
        {"fog_strength": 1.00, "blur_kernel": 131},
    ],

    "motion_blur": [
        {"kernel_size": k}
        for k in (21, 31, 41, 51, 61, 71, 81, 91)
    ],
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



def mean_std(device):
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device).view(1, 3, 1, 1)
    return mean, std


def to_norm(x01):
    mean, std = mean_std(x01.device)
    return (x01 - mean) / std


def x01_to_uint8_nhwc(x01):
    x = (x01.clamp(0, 1) * 255.0).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def save_png(arr_hwc, out_path):
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_hwc).save(out_path, format="PNG", optimize=True)


@torch.no_grad()
def predict(model, x01):
    return model(to_norm(x01)).argmax(dim=1)



def gaussian_noise(x01, sigma, indices, seed_base=510000):
    
    noise = torch.empty_like(x01)

    for i, idx in enumerate(indices):
        g = torch.Generator(device="cpu")
        g.manual_seed(seed_base + int(idx))

        n = torch.randn(
            x01[i].shape,
            generator=g,
            dtype=x01.dtype,
            device="cpu",
        ) * sigma

        noise[i] = n.to(x01.device)

    return (x01 + noise).clamp(0, 1)


def salt_pepper_noise(x01, prob, indices, seed_base=520000):
    
    adv = x01.clone()
    _, _, H, W = adv.shape

    for i, idx in enumerate(indices):
        g = torch.Generator(device="cpu")
        g.manual_seed(seed_base + int(idx))

        rand = torch.rand((1, H, W), generator=g)

        salt = (rand < (prob / 2.0)).to(adv.device)
        pepper = (
            (rand >= (prob / 2.0)) &
            (rand < prob)
        ).to(adv.device)

        adv[i] = torch.where(
            salt,
            torch.ones_like(adv[i]),
            adv[i],
        )
        adv[i] = torch.where(
            pepper,
            torch.zeros_like(adv[i]),
            adv[i],
        )

    return adv.clamp(0, 1)


def light_change(x01, brightness, contrast, gamma=1.0):
    
    x = x01.clamp(1e-6, 1.0)

    if gamma != 1.0:
        x = x.pow(gamma)

    x = (x - 0.5) * contrast + 0.5 + brightness

    return x.clamp(0, 1)


def fog_effect(
    x01,
    fog_strength,
    blur_kernel,
    indices,
    seed_base=530000,
):
    
    B, _, H, W = x01.shape
    veil = torch.ones_like(x01)

    haze_items = []

    for idx in indices:
        g = torch.Generator(device="cpu")
        g.manual_seed(seed_base + int(idx))

        haze_i = torch.rand(
            (1, H, W),
            generator=g,
            dtype=x01.dtype,
            device="cpu",
        )
        haze_items.append(haze_i)

    haze = torch.stack(haze_items, dim=0).to(x01.device)

    haze = F.avg_pool2d(
        haze,
        kernel_size=blur_kernel,
        stride=1,
        padding=blur_kernel // 2,
    )

    if haze.shape[-2:] != (H, W):
        haze = F.interpolate(
            haze,
            size=(H, W),
            mode="bilinear",
            align_corners=False,
        )

    haze = haze.clamp(0, 1)

    alpha = (
        fog_strength + 0.30 * (haze - 0.5)
    ).clamp(0, 1)

    adv = x01 * (1.0 - alpha) + veil * alpha

    return adv.clamp(0, 1)


def motion_blur(x01, kernel_size):
    
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError(
            f"motion_blur kernel_size must be a positive odd integer, got {kernel_size}"
        )

    kernel = torch.zeros(
        (kernel_size, kernel_size),
        device=x01.device,
        dtype=x01.dtype,
    )
    kernel[kernel_size // 2, :] = 1.0 / kernel_size

    kernel = kernel.view(
        1, 1, kernel_size, kernel_size
    ).repeat(3, 1, 1, 1)

    adv = F.conv2d(
        x01,
        kernel,
        padding=kernel_size // 2,
        groups=3,
    )

    return adv.clamp(0, 1)


def apply_attack(x01, name, params, indices):
    if name == "gaussian":
        return gaussian_noise(
            x01,
            params["sigma"],
            indices,
        )

    if name == "salt_pepper":
        return salt_pepper_noise(
            x01,
            params["prob"],
            indices,
        )

    if name == "light":
        return light_change(
            x01,
            params["brightness"],
            params["contrast"],
            params.get("gamma", 1.0),
        )

    if name == "fog":
        return fog_effect(
            x01,
            params["fog_strength"],
            params["blur_kernel"],
            indices,
        )

    if name == "motion_blur":
        return motion_blur(
            x01,
            params["kernel_size"],
        )

    raise ValueError(f"Unknown environmental corruption: {name}")


def make_loader(cfg):
    with open(cfg["class_to_idx"]) as f:
        class_to_idx = json.load(f)

    tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    ds = FixedClassImageFolderWithPaths(
        VAL_DIR,
        class_to_idx,
        transform=tfms,
    )

    if len(ds) == 0:
        raise RuntimeError(
            "0 images loaded. Check VAL_DIR and class_to_idx match."
        )

    loader = DataLoader(
        ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
    )

    return ds, loader, class_to_idx


def load_model(cfg, num_classes):
    if not os.path.exists(cfg["ckpt"]):
        raise FileNotFoundError(
            f"checkpoint missing: {cfg['ckpt']}"
        )

    model = cfg["build"](num_classes).to(DEVICE)
    model.load_state_dict(
        torch.load(
            cfg["ckpt"],
            map_location=DEVICE,
        )
    )
    model.eval()

    return model



@torch.no_grad()
def eval_attack_acc(model, loader, name, params):
    """
    Evaluate one corruption candidate on the ENTIRE validation loader.
    """
    correct = 0
    total = 0

    for x01, y, paths, indices in tqdm(
        loader,
        desc=f"    tune/{name}",
        leave=False,
    ):
        x01 = x01.to(DEVICE)
        y = y.to(DEVICE)
        idx = [int(v) for v in indices]

        adv = apply_attack(
            x01,
            name,
            params,
            idx,
        )

        pred = predict(model, adv)

        correct += (pred == y).sum().item()
        total += y.numel()

    if total == 0:
        raise RuntimeError(
            f"No samples evaluated while tuning {name}."
        )

    return correct / total


def choose_candidate(results, target_low, target_high):
    
    mid = (target_low + target_high) / 2.0

    in_band = [
        (params, acc)
        for params, acc in results
        if target_low <= acc <= target_high
    ]

    if in_band:
        pick = min(
            in_band,
            key=lambda item: abs(item[1] - mid),
        )
        return pick, True

    pick = min(
        results,
        key=lambda item: abs(item[1] - mid),
    )
    return pick, False


def tune_reference(ref_key, target_low, target_high):
    
    if not (0.0 <= target_low <= 1.0):
        raise ValueError("target_low must be between 0 and 1")

    if not (0.0 <= target_high <= 1.0):
        raise ValueError("target_high must be between 0 and 1")

    if target_low >= target_high:
        raise ValueError(
            "target_low must be smaller than target_high"
        )

    cfg = MODELS[ref_key]

    ds, loader, class_to_idx = make_loader(cfg)
    model = load_model(
        cfg,
        len(class_to_idx),
    )

    frozen = {}

    print(
        f"\n=== TUNING ON FULL VALIDATION SET ===\n"
        f"reference model : {ref_key}\n"
        f"images          : {len(ds)}\n"
        f"target accuracy : {target_low*100:.1f}% - {target_high*100:.1f}%"
    )

    tuning_report = {}

    for name in ATTACKS_TO_RUN:
        print(f"\n  [{name}]")

        results = []

        for params in CANDIDATES[name]:
            acc = eval_attack_acc(
                model,
                loader,
                name,
                params,
            )

            results.append(
                (params, acc)
            )

            print(
                f"    params={params} "
                f"-> attacked_acc={acc*100:.2f}%"
            )

        (picked_params, picked_acc), reached = choose_candidate(
            results,
            target_low,
            target_high,
        )

        frozen[name] = picked_params

        tuning_report[name] = {
            "target_low": target_low,
            "target_high": target_high,
            "target_reached": reached,
            "chosen_params": picked_params,
            "chosen_accuracy_percent": 100.0 * picked_acc,
            "all_candidates": [
                {
                    "params": params,
                    "accuracy_percent": 100.0 * acc,
                }
                for params, acc in results
            ],
        }

        if reached:
            print(
                f"  -> FROZEN {name}: {picked_params} "
                f"({picked_acc*100:.2f}%) [IN TARGET BAND]"
            )
        else:
            print(
                f"  -> WARNING: no {name} candidate reached "
                f"{target_low*100:.1f}-{target_high*100:.1f}% accuracy."
            )
            print(
                f"     Closest candidate: {picked_params} "
                f"({picked_acc*100:.2f}%)."
            )
            print(
                "     Consider expanding this candidate grid before using "
                "the result in the final experiment."
            )

    report_path = f"./env_tuning_report_{ref_key}.json"

    with open(report_path, "w") as f:
        json.dump(
            {
                "reference_model": ref_key,
                "num_images": len(ds),
                "target_low": target_low,
                "target_high": target_high,
                "by_attack": tuning_report,
            },
            f,
            indent=2,
        )

    frozen_path = f"./env_frozen_params_{ref_key}.json"

    with open(frozen_path, "w") as f:
        json.dump(
            frozen,
            f,
            indent=2,
        )

    print(f"\nTuning report saved to: {report_path}")
    print(f"Frozen parameters saved to: {frozen_path}")

    del model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return frozen



def generate_for_model(model_key, frozen_params):
    cfg = MODELS[model_key]
    out_root = cfg["out_root"]

    
    if os.path.exists(out_root):
        shutil.rmtree(out_root)

    Path(out_root).mkdir(
        parents=True,
        exist_ok=True,
    )

    ds, loader, class_to_idx = make_loader(cfg)
    idx_to_class = {
        int(v): k
        for k, v in class_to_idx.items()
    }

    model = load_model(
        cfg,
        len(class_to_idx),
    )

    print(
        f"\n[{model_key}] "
        f"images={len(ds)} "
        f"classes={len(class_to_idx)} "
        f"ckpt={cfg['ckpt']}"
    )

    metadata_path = os.path.join(
        out_root,
        "metadata.csv",
    )

    stats = {}

    with open(
        metadata_path,
        "w",
        newline="",
    ) as f:
        writer = csv.writer(f)

        writer.writerow([
            "dataset_index",
            "true_label_idx",
            "true_label_name",
            "attack",
            "params_json",
            "out_path",
        ])

        for name in ATTACKS_TO_RUN:
            params = frozen_params[name]

            attack_dir = os.path.join(
                out_root,
                f"{name}_png",
            )

            Path(attack_dir).mkdir(
                parents=True,
                exist_ok=True,
            )

            correct_clean = 0
            correct_adv = 0
            max_u8 = 0
            total = 0

            for x01, y, paths, indices in tqdm(
                loader,
                desc=f"[{model_key}] {name}",
            ):
                x01 = x01.to(DEVICE)
                y = y.to(DEVICE)

                idx = [
                    int(v)
                    for v in indices
                ]

                with torch.no_grad():
                    clean_pred = predict(
                        model,
                        x01,
                    )

                    correct_clean += (
                        clean_pred == y
                    ).sum().item()

                adv = apply_attack(
                    x01,
                    name,
                    params,
                    idx,
                )

              
                adv_u8 = (
                    adv.clamp(0, 1) * 255
                ).round()

                x_u8 = (
                    x01.clamp(0, 1) * 255
                ).round()

                current_max = int(
                    (
                        adv_u8 - x_u8
                    ).abs().max().item()
                )

                max_u8 = max(
                    max_u8,
                    current_max,
                )

                with torch.no_grad():
                    adv_pred = predict(
                        model,
                        adv,
                    )

                    correct_adv += (
                        adv_pred == y
                    ).sum().item()

                arr = x01_to_uint8_nhwc(
                    adv
                )

                total += y.numel()

                y_cpu = (
                    y.detach()
                    .cpu()
                    .numpy()
                )

                for i, dsidx in enumerate(idx):
                    label_idx = int(
                        y_cpu[i]
                    )

                    label_name = idx_to_class[
                        label_idx
                    ]

                    save_path = os.path.join(
                        attack_dir,
                        label_name,
                        f"{dsidx:06d}_y{label_idx}.png",
                    )

                    save_png(
                        arr[i],
                        save_path,
                    )

                    writer.writerow([
                        dsidx,
                        label_idx,
                        label_name,
                        name,
                        json.dumps(params),
                        save_path,
                    ])

            clean_acc = (
                100.0 *
                correct_clean /
                total
            )

            adv_acc = (
                100.0 *
                correct_adv /
                total
            )

            stats[name] = {
                "params": params,
                "clean_acc_percent": clean_acc,
                "attacked_acc_percent": adv_acc,
                "accuracy_drop": clean_acc - adv_acc,
                "max_pixel_diff_after_save": max_u8,
                "num_samples": total,
            }

            changed = (
                "OK"
                if max_u8 >= 1
                else "WARNING: no change after uint8 conversion"
            )

            print(
                f"[{model_key}] {name}: "
                f"clean={clean_acc:.2f}% "
                f"adv={adv_acc:.2f}% "
                f"drop={clean_acc-adv_acc:.2f} "
                f"max_u8={max_u8} "
                f"[{changed}]"
            )

    stats_path = os.path.join(
        out_root,
        "final_env_stats.json",
    )

    with open(
        stats_path,
        "w",
    ) as f:
        json.dump(
            stats,
            f,
            indent=2,
        )

   
    params_path = os.path.join(
        out_root,
        "frozen_env_params.json",
    )

    with open(
        params_path,
        "w",
    ) as f:
        json.dump(
            frozen_params,
            f,
            indent=2,
        )

    print(
        f"[{model_key}] stats -> {stats_path}"
    )

    del model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return stats


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        choices=list(MODELS.keys()) + ["all"],
        required=True,
    )

    ap.add_argument(
        "--tune_ref",
        choices=list(MODELS.keys()),
        default=None,
        help=(
            "Tune corruption strengths on the FULL validation set of this "
            "reference model, then freeze them for all selected models."
        ),
    )

    ap.add_argument(
        "--no_tune",
        action="store_true",
        help=(
            "Skip tuning and use the hardcoded FROZEN_PARAMS."
        ),
    )

    ap.add_argument(
        "--target_low",
        type=float,
        default=0.60,
    )

    ap.add_argument(
        "--target_high",
        type=float,
        default=0.75,
    )

    args = ap.parse_args()

    if args.no_tune and args.tune_ref is not None:
        raise ValueError(
            "Use either --no_tune OR --tune_ref, not both."
        )

    print("Device:", DEVICE)

    if args.no_tune:
        frozen = FROZEN_PARAMS

        print(
            "\nUsing hardcoded FROZEN_PARAMS "
            "(tuning skipped):"
        )

    elif args.tune_ref is not None:
        frozen = tune_reference(
            args.tune_ref,
            args.target_low,
            args.target_high,
        )

        print(
            "\nFrozen strengths after full-set tuning:"
        )

    else:
        raise ValueError(
            "Choose one mode: use --tune_ref MODEL to tune first, "
            "or --no_tune to use the hardcoded parameters."
        )

    print(
        json.dumps(
            frozen,
            indent=2,
        )
    )

    keys = (
        list(MODELS.keys())
        if args.model == "all"
        else [args.model]
    )

    all_stats = {}

    for key in keys:
        all_stats[key] = generate_for_model(
            key,
            frozen,
        )

    print(
        "\n==================== SUMMARY ===================="
    )

    for key in keys:
        print(f"\n{key}:")

        for attack, s in all_stats[key].items():
            print(
                f"  {attack:12s} "
                f"clean={s['clean_acc_percent']:.2f}% "
                f"adv={s['attacked_acc_percent']:.2f}% "
                f"drop={s['accuracy_drop']:.2f} "
                f"max_u8={s['max_pixel_diff_after_save']}"
            )


if __name__ == "__main__":
    main()