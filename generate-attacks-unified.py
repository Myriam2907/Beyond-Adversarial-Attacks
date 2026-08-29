import os
import csv
import json
import shutil
import argparse
from pathlib import Path

import numpy as np
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
ATTACKS_TO_RUN = ["fgsm", "rfgsm", "pgd", "random_patch"]


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
        "out_root": "./attacks_mobilenet_eps",
        "batch_size": 64,
    },
    "convnext": {
        "build": make_convnext,
        "ckpt": "./convnext_mapillary_results/convnext_mapillary_best.pth",
        "class_to_idx": "./convnext_mapillary_results/class_to_idx.json",
        "out_root": "./attacks_convnext_eps",
        "batch_size": 128,
    },
    "efficientnet": {
        "build": make_efficientnet,
        "ckpt": "./efficientnetv2_mapillary_results/efficientnetv2_mapillary_best.pth",
        "class_to_idx": "./efficientnetv2_mapillary_results/class_to_idx.json",
        "out_root": "./attacks_efficientnet_eps",
        "batch_size": 128,
    },
}


def parse_frac(s):
    s = str(s).strip()
    if "/" in s:
        a, b = s.split("/", 1)
        return float(a) / float(b)
    return float(s)


def mean_std(device, dtype=torch.float32):
    mean = torch.tensor(MEAN, device=device, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=device, dtype=dtype).view(1, 3, 1, 1)
    return mean, std


def to_norm(x01):
    mean, std = mean_std(x01.device, x01.dtype)
    return (x01 - mean) / std


def quantize_to_png_tensor(x01):
  
    return (x01.clamp(0, 1) * 255.0).round() / 255.0


def tensor_to_uint8_nhwc(x01):
    x = (x01.clamp(0, 1) * 255.0).round().to(torch.uint8)
    return x.permute(0, 2, 3, 1).contiguous().cpu().numpy()


def save_png(arr_hwc, out_path):
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr_hwc).save(out_path, format="PNG", optimize=True)


@torch.no_grad()
def predict(model, x01):
    return model(to_norm(x01)).argmax(dim=1)

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


def make_loader(cfg):
    with open(cfg["class_to_idx"], "r") as f:
        class_to_idx = json.load(f)

    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    ds = FixedClassImageFolderWithPaths(VAL_DIR, class_to_idx, transform=tfm)
    if len(ds) == 0:
        raise RuntimeError("0 images loaded. Check VAL_DIR and class_to_idx.")

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
        raise FileNotFoundError(f"Missing checkpoint: {cfg['ckpt']}")

    model = cfg["build"](num_classes).to(DEVICE)
    state = torch.load(cfg["ckpt"], map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model


def deterministic_uniform_like(x_sample, low, high, seed):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    noise = torch.empty(
        x_sample.shape,
        dtype=x_sample.dtype,
        device="cpu",
    ).uniform_(low, high, generator=g)
    return noise.to(x_sample.device)



def fgsm_attack(model, x01, y, eps):
    x = x01.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(to_norm(x)), y)
    loss.backward()
    adv = x + eps * x.grad.detach().sign()
    return adv.clamp(0, 1).detach()


def rfgsm_attack(model, x01, y, eps, alpha, indices, seed_base=222000):

    noise = torch.empty_like(x01)
    for i, idx in enumerate(indices):
        noise[i] = deterministic_uniform_like(
            x01[i], -eps, eps, seed_base + int(idx)
        )

    x = (x01 + noise).clamp(0, 1)
    x = x.detach().clone().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(to_norm(x)), y)
    loss.backward()

    x = x + alpha * x.grad.detach().sign()
    x = torch.maximum(torch.minimum(x, x01 + eps), x01 - eps)
    return x.clamp(0, 1).detach()


def pgd_attack(model, x01, y, eps, alpha, steps, indices, seed_base=333000):
    
    x = torch.empty_like(x01)

    for i, idx in enumerate(indices):
        noise = deterministic_uniform_like(
            x01[i], -eps, eps, seed_base + int(idx)
        )
        x[i] = (x01[i] + noise).clamp(0, 1)

    for _ in range(steps):
        x = x.detach().clone().requires_grad_(True)
        model.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(to_norm(x)), y)
        loss.backward()

        x = x + alpha * x.grad.detach().sign()
        x = torch.maximum(torch.minimum(x, x01 + eps), x01 - eps)
        x = x.clamp(0, 1).detach()

    return x


def random_patch_attack(x01, patch_ratio, indices, seed_base=444000):
    
    if not (0.0 < patch_ratio <= 1.0):
        raise ValueError(f"patch_ratio must be in (0,1], got {patch_ratio}")

    adv = x01.clone()
    _, C, H, W = adv.shape
    ph = max(1, int(round(H * patch_ratio)))
    pw = max(1, int(round(W * patch_ratio)))

    for i, idx in enumerate(indices):
        g = torch.Generator(device="cpu")
        g.manual_seed(seed_base + int(idx))

        cy, cx = H // 2, W // 2
        jy = int(torch.randint(-ph // 2, ph // 2 + 1, (1,), generator=g).item())
        jx = int(torch.randint(-pw // 2, pw // 2 + 1, (1,), generator=g).item())

        y0 = max(0, min(cy - ph // 2 + jy, H - ph))
        x0 = max(0, min(cx - pw // 2 + jx, W - pw))

        patch = torch.rand(
            (C, ph, pw),
            generator=g,
            dtype=x01.dtype,
            device="cpu",
        ).to(adv.device)

        adv[i, :, y0:y0 + ph, x0:x0 + pw] = patch

    return adv.clamp(0, 1)


def run_attack(model, x01, y, name, params, indices):
    if name == "fgsm":
        return fgsm_attack(model, x01, y, params["eps"])
    if name == "rfgsm":
        return rfgsm_attack(
            model, x01, y, params["eps"], params["alpha"], indices
        )
    if name == "pgd":
        return pgd_attack(
            model,
            x01,
            y,
            params["eps"],
            params["alpha"],
            params["steps"],
            indices,
        )
    if name == "random_patch":
        return random_patch_attack(x01, params["patch_ratio"], indices)
    raise ValueError(name)



def batch_linf_uint8(clean01, attacked01):
    clean_u8 = (clean01.clamp(0, 1) * 255.0).round()
    attack_u8 = (attacked01.clamp(0, 1) * 255.0).round()
    return (attack_u8 - clean_u8).abs().flatten(1).max(dim=1).values


def generate_for_model(model_key, eps, rfgsm_alpha, pgd_alpha, pgd_steps, patch_ratio):
    cfg = MODELS[model_key]
    out_root = cfg["out_root"]

    if os.path.exists(out_root):
        print(f"\n[{model_key}] deleting old results: {out_root}")
        shutil.rmtree(out_root)
    Path(out_root).mkdir(parents=True, exist_ok=True)

    ds, loader, class_to_idx = make_loader(cfg)
    idx_to_class = {int(v): k for k, v in class_to_idx.items()}
    model = load_model(cfg, len(class_to_idx))

    print(f"\n[{model_key}] images={len(ds)} classes={len(class_to_idx)}")
    print(
        f"[{model_key}] eps={eps*255:.4f}/255 "
        f"rfgsm_alpha={rfgsm_alpha*255:.4f}/255 "
        f"pgd_alpha={pgd_alpha*255:.4f}/255 "
        f"pgd_steps={pgd_steps} patch_ratio={patch_ratio}"
    )

    params_by_attack = {
        "fgsm": {"eps": eps},
        "rfgsm": {"eps": eps, "alpha": rfgsm_alpha},
        "pgd": {"eps": eps, "alpha": pgd_alpha, "steps": pgd_steps},
        "random_patch": {"patch_ratio": patch_ratio},
    }

    metadata_path = os.path.join(out_root, "metadata.csv")
    stats = {}

    with open(metadata_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset_index",
            "true_label_idx",
            "true_label_name",
            "attack",
            "params_json",
            "clean_correct",
            "attacked_correct_after_png",
            "linf_u8",
            "out_path",
        ])

        for attack_name in ATTACKS_TO_RUN:
            params = params_by_attack[attack_name]
            attack_dir = os.path.join(out_root, f"{attack_name}_png")
            Path(attack_dir).mkdir(parents=True, exist_ok=True)

            total = 0
            clean_correct_total = 0
            attacked_correct_total = 0
            originally_correct = 0
            successful_attacks = 0
            changed_images = 0
            linf_values = []

            for x01, y, paths, indices in tqdm(
                loader, desc=f"[{model_key}] {attack_name}"
            ):
                x01 = x01.to(DEVICE)
                y = y.to(DEVICE)
                idx_list = [int(v) for v in indices]

                with torch.no_grad():
                    clean_pred = predict(model, x01)

                clean_correct_mask = clean_pred == y
                clean_correct_total += clean_correct_mask.sum().item()

                adv_float = run_attack(
                    model, x01, y, attack_name, params, idx_list
                )

                
                adv_png = quantize_to_png_tensor(adv_float)
                clean_png = quantize_to_png_tensor(x01)

                with torch.no_grad():
                    attacked_pred = predict(model, adv_png)

                attacked_correct_mask = attacked_pred == y
                attacked_correct_total += attacked_correct_mask.sum().item()

                originally_correct += clean_correct_mask.sum().item()
                success_mask = clean_correct_mask & (attacked_pred != y)
                successful_attacks += success_mask.sum().item()

                linf_batch = batch_linf_uint8(clean_png, adv_png)
                changed_images += (linf_batch > 0).sum().item()
                linf_values.extend(
                    linf_batch.detach().cpu().numpy().astype(float).tolist()
                )

                adv_arr = tensor_to_uint8_nhwc(adv_png)
                y_cpu = y.detach().cpu().numpy()
                clean_ok_cpu = clean_correct_mask.detach().cpu().numpy()
                attack_ok_cpu = attacked_correct_mask.detach().cpu().numpy()
                linf_cpu = linf_batch.detach().cpu().numpy()
                total += y.numel()

                for i, dsidx in enumerate(idx_list):
                    li = int(y_cpu[i])
                    lname = idx_to_class[li]
                    save_path = os.path.join(
                        attack_dir, lname, f"{dsidx:06d}_y{li}.png"
                    )
                    save_png(adv_arr[i], save_path)
                    writer.writerow([
                        dsidx,
                        li,
                        lname,
                        attack_name,
                        json.dumps(params),
                        int(clean_ok_cpu[i]),
                        int(attack_ok_cpu[i]),
                        float(linf_cpu[i]),
                        save_path,
                    ])

            clean_acc = 100.0 * clean_correct_total / total
            attacked_acc = 100.0 * attacked_correct_total / total
            drop = clean_acc - attacked_acc
            asr = (
                100.0 * successful_attacks / originally_correct
                if originally_correct > 0 else float("nan")
            )

            linf_arr = np.asarray(linf_values, dtype=np.float64)
            linf_min = float(linf_arr.min()) if linf_arr.size else 0.0
            linf_mean = float(linf_arr.mean()) if linf_arr.size else 0.0
            linf_max = float(linf_arr.max()) if linf_arr.size else 0.0
            changed_pct = 100.0 * changed_images / total

            stats[attack_name] = {
                "params": params,
                "num_samples": total,
                "clean_accuracy_percent": clean_acc,
                "attacked_accuracy_after_png_percent": attacked_acc,
                "accuracy_drop_points": drop,
                "originally_correct_samples": originally_correct,
                "successful_attacks_on_originally_correct": successful_attacks,
                "attack_success_rate_percent": asr,
                "changed_images": changed_images,
                "changed_images_percent": changed_pct,
                "linf_u8_min": linf_min,
                "linf_u8_mean": linf_mean,
                "linf_u8_max": linf_max,
            }

            print(f"\n[{model_key}] {attack_name}")
            print(f"  clean accuracy          : {clean_acc:.2f}%")
            print(f"  attacked accuracy (PNG) : {attacked_acc:.2f}%")
            print(f"  accuracy drop           : {drop:.2f} points")
            print(
                f"  attack success rate     : {asr:.2f}% "
                f"({successful_attacks}/{originally_correct})"
            )
            print(
                f"  changed images          : {changed_images}/{total} "
                f"({changed_pct:.2f}%)"
            )
            print(
                f"  L_inf uint8 min/mean/max: "
                f"{linf_min:.2f}/{linf_mean:.2f}/{linf_max:.2f}"
            )

            if attack_name in {"fgsm", "rfgsm", "pgd"}:
                expected_bound = int(np.ceil(eps * 255.0 + 1e-12))
                if linf_max <= expected_bound:
                    print(
                        f"  epsilon-bound check     : OK "
                        f"(max_u8 <= {expected_bound})"
                    )
                else:
                    print(
                        f"  WARNING: max_u8={linf_max:.0f} exceeds "
                        f"expected bound {expected_bound}"
                    )

    stats_path = os.path.join(out_root, "final_attack_stats_v3.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    config_path = os.path.join(out_root, "attack_config_v3.json")
    with open(config_path, "w") as f:
        json.dump({
            "model": model_key,
            "checkpoint": cfg["ckpt"],
            "validation_dir": VAL_DIR,
            "image_size": IMG_SIZE,
            "eps": eps,
            "eps_in_255_units": eps * 255.0,
            "rfgsm_alpha": rfgsm_alpha,
            "rfgsm_alpha_in_255_units": rfgsm_alpha * 255.0,
            "pgd_alpha": pgd_alpha,
            "pgd_alpha_in_255_units": pgd_alpha * 255.0,
            "pgd_steps": pgd_steps,
            "patch_ratio": patch_ratio,
            "attacks": ATTACKS_TO_RUN,
            "evaluation_is_after_png_quantization": True,
        }, f, indent=2)

    print(f"\n[{model_key}] stats -> {stats_path}")
    print(f"[{model_key}] config -> {config_path}")

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
    ap.add_argument("--eps", default="1/255")
    ap.add_argument("--rfgsm_alpha", default="0.25/255")
    ap.add_argument("--pgd_alpha", default="0.25/255")
    ap.add_argument("--pgd_steps", type=int, default=10)
    ap.add_argument("--patch_ratio", type=float, default=0.30)

    args = ap.parse_args()

    eps = parse_frac(args.eps)
    rfgsm_alpha = parse_frac(args.rfgsm_alpha)
    pgd_alpha = parse_frac(args.pgd_alpha)

    if eps <= 0:
        raise ValueError("eps must be > 0")
    if rfgsm_alpha <= 0:
        raise ValueError("rfgsm_alpha must be > 0")
    if pgd_alpha <= 0:
        raise ValueError("pgd_alpha must be > 0")
    if args.pgd_steps < 1:
        raise ValueError("pgd_steps must be >= 1")
    if not (0.0 < args.patch_ratio <= 1.0):
        raise ValueError("patch_ratio must be in (0,1]")

    print("=" * 78)
    print("UNIFIED ADVERSARIAL EVALUATION V3")
    print("=" * 78)
    print(f"Device       : {DEVICE}")
    print(f"eps          : {eps*255:.4f}/255")
    print(f"R-FGSM alpha : {rfgsm_alpha*255:.4f}/255")
    print(f"PGD alpha    : {pgd_alpha*255:.4f}/255")
    print(f"PGD steps    : {args.pgd_steps}")
    print(f"patch ratio  : {args.patch_ratio}")
    print(
        "\nOld attack-result folders for selected models will be deleted "
        "before regeneration."
    )

    if eps * 255.0 < 1.0:
        print(
            "WARNING: eps < 1/255; many perturbations may disappear "
            "after PNG quantization."
        )

    keys = list(MODELS.keys()) if args.model == "all" else [args.model]
    all_stats = {}

    for key in keys:
        all_stats[key] = generate_for_model(
            model_key=key,
            eps=eps,
            rfgsm_alpha=rfgsm_alpha,
            pgd_alpha=pgd_alpha,
            pgd_steps=args.pgd_steps,
            patch_ratio=args.patch_ratio,
        )

    print("\n" + "=" * 78)
    print("FINAL SUMMARY")
    print("=" * 78)

    for key in keys:
        print(f"\n{key}:")
        for attack_name in ATTACKS_TO_RUN:
            s = all_stats[key][attack_name]
            print(
                f"  {attack_name:13s} "
                f"clean={s['clean_accuracy_percent']:.2f}%  "
                f"adv_png={s['attacked_accuracy_after_png_percent']:.2f}%  "
                f"drop={s['accuracy_drop_points']:.2f}  "
                f"ASR={s['attack_success_rate_percent']:.2f}%  "
                f"changed={s['changed_images_percent']:.2f}%  "
                f"Linf_u8(mean/max)={s['linf_u8_mean']:.2f}/{s['linf_u8_max']:.0f}"
            )


if __name__ == "__main__":
    main()
