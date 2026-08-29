import os
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm



DATA_DIR = "./data"
OUT_ROOT = "./gtsrb_repeat/attacks"

IMG_SIZE = 224
NUM_CLASSES = 43
BATCH_SIZE = 64
NUM_WORKERS = 0 
SEED = 123

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


MODEL_PATHS = {
    "mobilenet": "./gtsrb_repeat/models/mobilenet/mobilenet_gtsrb_best.pth",
    "convnext": "./gtsrb_repeat/models/convnext/convnext_gtsrb_best.pth",
    "efficientnet": "./gtsrb_repeat/models/efficientnet/efficientnet_gtsrb_best.pth",
}


MODEL_NAMES = ["mobilenet", "convnext", "efficientnet"]


FGSM_EPS = 1.0 / 255.0
RFGSM_EPS = 1.0 / 255.0
RFGSM_ALPHA = 0.25 / 255.0
PGD_EPS = 1.0 / 255.0
PGD_ALPHA = 0.25 / 255.0
PGD_STEPS = 3


PATCH_FRAC = 0.30
PATCH_COLOR = "white"  
RANDOM_PATCH_COLOR_CHOICES = [(255, 255, 255), (0, 0, 0)]

LIGHT_BRIGHTNESS = 1.25
LIGHT_CONTRAST = 1.15
GLARE_STRENGTH = 0.55

FOG_ALPHA = 0.95
FOG_BLUR_KERNEL = 111  
FOG_WHITE_STRENGTH = 0.22

MOTION_BLUR_KERNEL = 4

GAUSSIAN_SIGMA = 0.30
SALT_PEPPER_PROB = 0.25

OVERWRITE = False



def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def ensure_dirs():
    shared_dirs = [
        "clean_png",
        "fog_png",
        "light_png",
        "motion_blur_png",
        "patch_png",
        "random_patch_png",
        "gaussian_noise_png",
        "salt_pepper_png",
    ]

    Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)
    for d in shared_dirs:
        Path(os.path.join(OUT_ROOT, d)).mkdir(parents=True, exist_ok=True)

    for model_name in MODEL_NAMES:
        for d in ["fgsm_png", "rfgsm_png", "pgd_png"]:
            Path(os.path.join(OUT_ROOT, model_name, d)).mkdir(parents=True, exist_ok=True)


def save_pil(img: Image.Image, out_path: str):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG", optimize=True)



def build_model(model_name: str):
    model_name = model_name.lower()

    if model_name == "mobilenet":
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)

    elif model_name == "convnext":
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)

    elif model_name == "efficientnet":
        model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model


def load_model(model_name: str, ckpt_path: str):
    print(f"\nLoading {model_name}: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model = build_model(model_name).to(DEVICE)
    state = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()
    return model



def tensor01_to_pil(x01: torch.Tensor) -> Image.Image:
    x = (x01.clamp(0, 1) * 255.0).round().to(torch.uint8)
    x = x.permute(1, 2, 0).contiguous().cpu().numpy()
    return Image.fromarray(x)


def batch01_to_uint8(x01_batch: torch.Tensor) -> torch.Tensor:
    x = (x01_batch.clamp(0, 1) * 255.0).round().to(torch.uint8)
    x = x.permute(0, 2, 3, 1).contiguous().cpu()
    return x


def normalize_batch(x01: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(MEAN, device=x01.device).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=x01.device).view(1, 3, 1, 1)
    return (x01 - mean) / std



def patch_attack_pil(img: Image.Image) -> Image.Image:
    img = img.copy()
    w, h = img.size
    pw = max(1, int(round(w * PATCH_FRAC)))
    ph = max(1, int(round(h * PATCH_FRAC)))

    left = (w - pw) // 2
    top = (h - ph) // 2

    patch_color = (255, 255, 255) if PATCH_COLOR == "white" else (0, 0, 0)
    patch = Image.new("RGB", (pw, ph), patch_color)
    img.paste(patch, (left, top))
    return img


def random_patch_attack_pil(img: Image.Image, idx: int) -> Image.Image:
    img = img.copy()
    w, h = img.size
    pw = max(1, int(round(w * PATCH_FRAC)))
    ph = max(1, int(round(h * PATCH_FRAC)))

    rng = np.random.default_rng(SEED + idx)
    left = int(rng.integers(0, max(1, w - pw + 1)))
    top = int(rng.integers(0, max(1, h - ph + 1)))
    color = RANDOM_PATCH_COLOR_CHOICES[int(rng.integers(0, len(RANDOM_PATCH_COLOR_CHOICES)))]

    patch = Image.new("RGB", (pw, ph), color)
    img.paste(patch, (left, top))
    return img


def light_attack_pil(img: Image.Image) -> Image.Image:
    img2 = TF.adjust_brightness(img, LIGHT_BRIGHTNESS)
    img2 = TF.adjust_contrast(img2, LIGHT_CONTRAST)

    x = TF.to_tensor(img2)
    _, H, W = x.shape

    cx = int(0.75 * W)
    cy = int(0.25 * H)
    sigma = 0.18 * min(H, W)

    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    yy = yy.float()
    xx = xx.float()
    blob = torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma ** 2))
    blob = blob.unsqueeze(0).repeat(3, 1, 1)

    x = (x + GLARE_STRENGTH * blob).clamp(0, 1)
    return tensor01_to_pil(x)


def fog_attack_pil(img: Image.Image) -> Image.Image:
    x = TF.to_tensor(img)
    # brighten toward white
    x_bright = (FOG_ALPHA * x + (1.0 - FOG_ALPHA) * torch.ones_like(x)).clamp(0, 1)
    # soften contrast using a heavy blur to simulate haze
    k = FOG_BLUR_KERNEL if FOG_BLUR_KERNEL % 2 == 1 else FOG_BLUR_KERNEL + 1
    x_blur = TF.gaussian_blur(x_bright, kernel_size=[k, k], sigma=[8.0, 8.0])
    x_fog = ((1.0 - FOG_WHITE_STRENGTH) * x_bright + FOG_WHITE_STRENGTH * x_blur).clamp(0, 1)
    return tensor01_to_pil(x_fog)


def _motion_blur_kernel_2d(kernel_size: int, channels: int = 3) -> torch.Tensor:
    k = max(2, int(kernel_size))
    kernel = torch.zeros((k, k), dtype=torch.float32)
    kernel[k // 2, :] = 1.0
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, k, k).repeat(channels, 1, 1, 1)
    return kernel


def motion_blur_attack_pil(img: Image.Image) -> Image.Image:
    x = TF.to_tensor(img).unsqueeze(0)
    k = _motion_blur_kernel_2d(MOTION_BLUR_KERNEL, channels=3)
    pad = MOTION_BLUR_KERNEL // 2
    x_blur = F.conv2d(x, k, padding=pad, groups=3)
    x_blur = x_blur[:, :, : x.shape[2], : x.shape[3]]  # keep exact size if even kernel
    return tensor01_to_pil(x_blur.squeeze(0).clamp(0, 1))


def gaussian_noise_attack_pil(img: Image.Image, idx: int) -> Image.Image:
    x = TF.to_tensor(img)
    gen = torch.Generator().manual_seed(SEED + idx)
    noise = torch.randn(x.shape, generator=gen) * GAUSSIAN_SIGMA
    x_noisy = (x + noise).clamp(0, 1)
    return tensor01_to_pil(x_noisy)


def salt_pepper_attack_pil(img: Image.Image, idx: int) -> Image.Image:
    x = TF.to_tensor(img)
    rng = np.random.default_rng(SEED + idx)
    mask = rng.random((x.shape[1], x.shape[2]))

    salt = mask < (SALT_PEPPER_PROB / 2.0)
    pepper = (mask >= (SALT_PEPPER_PROB / 2.0)) & (mask < SALT_PEPPER_PROB)

    x[:, salt] = 1.0
    x[:, pepper] = 0.0
    return tensor01_to_pil(x)


def fgsm_attack_batch(model, x01_batch: torch.Tensor, y_batch: torch.Tensor, eps: float) -> torch.Tensor:
    x = x01_batch.detach().clone().requires_grad_(True)
    logits = model(normalize_batch(x))
    loss = F.cross_entropy(logits, y_batch)
    model.zero_grad(set_to_none=True)
    loss.backward()

    grad_sign = x.grad.detach().sign()
    adv = (x + eps * grad_sign).clamp(0, 1).detach()
    return adv


def rfgsm_attack_batch(model, x01_batch: torch.Tensor, y_batch: torch.Tensor, eps: float, alpha: float) -> torch.Tensor:
   
    rand = torch.empty_like(x01_batch).uniform_(-alpha, alpha)
    x_start = (x01_batch + rand).clamp(0, 1).detach().clone().requires_grad_(True)

    logits = model(normalize_batch(x_start))
    loss = F.cross_entropy(logits, y_batch)
    model.zero_grad(set_to_none=True)
    loss.backward()

    grad_sign = x_start.grad.detach().sign()
    adv = x_start + (eps - alpha) * grad_sign
    adv = torch.max(torch.min(adv, x01_batch + eps), x01_batch - eps)
    adv = adv.clamp(0, 1).detach()
    return adv


def pgd_attack_batch(model, x01_batch: torch.Tensor, y_batch: torch.Tensor, eps: float, alpha: float, steps: int) -> torch.Tensor:
    x_orig = x01_batch.detach()
    x_adv = x_orig.clone().detach()

    
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0, 1)

    for _ in range(steps):
        x_adv.requires_grad_(True)
        logits = model(normalize_batch(x_adv))
        loss = F.cross_entropy(logits, y_batch)
        model.zero_grad(set_to_none=True)
        loss.backward()

        grad_sign = x_adv.grad.detach().sign()
        x_adv = x_adv.detach() + alpha * grad_sign
        x_adv = torch.max(torch.min(x_adv, x_orig + eps), x_orig - eps)
        x_adv = x_adv.clamp(0, 1)

    return x_adv.detach()



def generate_shared_attacks(test_ds):
    print("\n" + "=" * 100)
    print("STEP 1/2: Generating shared image sets (clean + environmental / non-gradient attacks)")
    print("=" * 100)

    meta_path = os.path.join(OUT_ROOT, "shared_metadata.csv")
    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "true_label", "attack", "out_path"])

    attack_builders = {
        "clean": lambda img, idx: img,
        "fog": fog_attack_pil,
        "light": light_attack_pil,
        "motion_blur": motion_blur_attack_pil,
        "patch": patch_attack_pil,
        "random_patch": random_patch_attack_pil,
        "gaussian_noise": gaussian_noise_attack_pil,
        "salt_pepper": salt_pepper_attack_pil,
    }

    for idx in tqdm(range(len(test_ds)), desc="Shared sets"):
        img_pil, y = test_ds[idx]
        img_pil = img_pil.convert("RGB")
        img_resized = TF.resize(img_pil, [IMG_SIZE, IMG_SIZE])

        records = []

        for attack_name in [
            "clean",
            "fog",
            "light",
            "motion_blur",
            "patch",
            "random_patch",
            "gaussian_noise",
            "salt_pepper",
        ]:
            if attack_name == "clean":
                out_dir = "clean_png"
                img_out = img_resized
            elif attack_name == "fog":
                out_dir = "fog_png"
                img_out = attack_builders[attack_name](img_resized)
            elif attack_name == "light":
                out_dir = "light_png"
                img_out = attack_builders[attack_name](img_resized)
            elif attack_name == "motion_blur":
                out_dir = "motion_blur_png"
                img_out = attack_builders[attack_name](img_resized)
            elif attack_name == "patch":
                out_dir = "patch_png"
                img_out = attack_builders[attack_name](img_resized)
            elif attack_name == "random_patch":
                out_dir = "random_patch_png"
                img_out = attack_builders[attack_name](img_resized, idx)
            elif attack_name == "gaussian_noise":
                out_dir = "gaussian_noise_png"
                img_out = attack_builders[attack_name](img_resized, idx)
            elif attack_name == "salt_pepper":
                out_dir = "salt_pepper_png"
                img_out = attack_builders[attack_name](img_resized, idx)
            else:
                raise ValueError(attack_name)

            out_path = os.path.join(OUT_ROOT, out_dir, f"{idx:05d}_y{y}.png")
            if OVERWRITE or (not os.path.exists(out_path)):
                save_pil(img_out, out_path)
            records.append([idx, y, attack_name, out_path])

        with open(meta_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(records)

    print(f"✅ Shared metadata saved to: {meta_path}")



def generate_gradient_attacks_for_model(model_name: str, ckpt_path: str, test_ds):
    print("\n" + "=" * 100)
    print(f"STEP 2/2: Generating model-specific gradient attacks for {model_name}")
    print("=" * 100)

    model = load_model(model_name, ckpt_path)

    meta_path = os.path.join(OUT_ROOT, model_name, "gradient_metadata.csv")
    with open(meta_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "true_label", "attack", "out_path"])

    n = len(test_ds)
    n_batches = math.ceil(n / BATCH_SIZE)

    for b in tqdm(range(n_batches), desc=f"{model_name} gradient batches"):
        start = b * BATCH_SIZE
        end = min(n, start + BATCH_SIZE)

        imgs01 = []
        labels = []
        indices = []

        for idx in range(start, end):
            img_pil, y = test_ds[idx]
            img_pil = img_pil.convert("RGB")
            img_resized = TF.resize(img_pil, [IMG_SIZE, IMG_SIZE])
            x01 = TF.to_tensor(img_resized)
            imgs01.append(x01)
            labels.append(y)
            indices.append(idx)

        x01_batch = torch.stack(imgs01, dim=0).to(DEVICE)
        y_batch = torch.tensor(labels, dtype=torch.long, device=DEVICE)

        fgsm_batch = fgsm_attack_batch(model, x01_batch, y_batch, eps=FGSM_EPS)
        rfgsm_batch = rfgsm_attack_batch(model, x01_batch, y_batch, eps=RFGSM_EPS, alpha=RFGSM_ALPHA)
        pgd_batch = pgd_attack_batch(model, x01_batch, y_batch, eps=PGD_EPS, alpha=PGD_ALPHA, steps=PGD_STEPS)

        fgsm_uint8 = batch01_to_uint8(fgsm_batch)
        rfgsm_uint8 = batch01_to_uint8(rfgsm_batch)
        pgd_uint8 = batch01_to_uint8(pgd_batch)

        records = []
        for i, idx in enumerate(indices):
            y = labels[i]

            fgsm_path = os.path.join(OUT_ROOT, model_name, "fgsm_png", f"{idx:05d}_y{y}.png")
            rfgsm_path = os.path.join(OUT_ROOT, model_name, "rfgsm_png", f"{idx:05d}_y{y}.png")
            pgd_path = os.path.join(OUT_ROOT, model_name, "pgd_png", f"{idx:05d}_y{y}.png")

            if OVERWRITE or (not os.path.exists(fgsm_path)):
                Image.fromarray(fgsm_uint8[i].numpy()).save(fgsm_path, format="PNG", optimize=True)
            if OVERWRITE or (not os.path.exists(rfgsm_path)):
                Image.fromarray(rfgsm_uint8[i].numpy()).save(rfgsm_path, format="PNG", optimize=True)
            if OVERWRITE or (not os.path.exists(pgd_path)):
                Image.fromarray(pgd_uint8[i].numpy()).save(pgd_path, format="PNG", optimize=True)

            records.append([idx, y, "fgsm", fgsm_path])
            records.append([idx, y, "rfgsm", rfgsm_path])
            records.append([idx, y, "pgd", pgd_path])

        with open(meta_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(records)

    print(f"✅ Gradient metadata saved to: {meta_path}")



def write_summary(test_ds):
    summary = {
        "data_dir": DATA_DIR,
        "out_root": OUT_ROOT,
        "n_test_images": len(test_ds),
        "image_size": IMG_SIZE,
        "device": str(DEVICE),
        "shared_sets": [
            "clean_png",
            "fog_png",
            "light_png",
            "motion_blur_png",
            "patch_png",
            "random_patch_png",
            "gaussian_noise_png",
            "salt_pepper_png",
        ],
        "model_specific_sets": {
            m: ["fgsm_png", "rfgsm_png", "pgd_png"] for m in MODEL_NAMES
        },
        "model_paths": MODEL_PATHS,
        "attack_params": {
            "fgsm_eps": FGSM_EPS,
            "rfgsm_eps": RFGSM_EPS,
            "rfgsm_alpha": RFGSM_ALPHA,
            "pgd_eps": PGD_EPS,
            "pgd_alpha": PGD_ALPHA,
            "pgd_steps": PGD_STEPS,
            "patch_frac": PATCH_FRAC,
            "patch_color": PATCH_COLOR,
            "light_brightness": LIGHT_BRIGHTNESS,
            "light_contrast": LIGHT_CONTRAST,
            "glare_strength": GLARE_STRENGTH,
            "fog_alpha": FOG_ALPHA,
            "fog_blur_kernel": FOG_BLUR_KERNEL,
            "fog_white_strength": FOG_WHITE_STRENGTH,
            "motion_blur_kernel": MOTION_BLUR_KERNEL,
            "gaussian_sigma": GAUSSIAN_SIGMA,
            "salt_pepper_prob": SALT_PEPPER_PROB,
        },
    }

    out_json = os.path.join(OUT_ROOT, "attack_generation_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary saved to: {out_json}")



def main():
    set_seed(SEED)
    ensure_dirs()

    test_ds = datasets.GTSRB(root=DATA_DIR, split="test", download=True, transform=None)

    print("\n" + "=" * 100)
    print("GENERATING NEW GTSRB ATTACK DATASETS (ALL OLD FILES REMAIN UNTOUCHED)")
    print("=" * 100)
    print(f"Device: {DEVICE}")
    print(f"Test images: {len(test_ds):,}")
    print(f"Output root: {os.path.abspath(OUT_ROOT)}")

    generate_shared_attacks(test_ds)

    for model_name in MODEL_NAMES:
        generate_gradient_attacks_for_model(model_name, MODEL_PATHS[model_name], test_ds)

    write_summary(test_ds)

    print("\n" + "=" * 100)
    print("DONE")
    print("=" * 100)
    print("Generated shared sets:")
    for d in [
        "clean_png",
        "fog_png",
        "light_png",
        "motion_blur_png",
        "patch_png",
        "random_patch_png",
        "gaussian_noise_png",
        "salt_pepper_png",
    ]:
        print(f"  {os.path.join(OUT_ROOT, d)}")

    print("\nGenerated model-specific sets:")
    for model_name in MODEL_NAMES:
        print(f"  {model_name}:")
        for d in ["fgsm_png", "rfgsm_png", "pgd_png"]:
            print(f"    {os.path.join(OUT_ROOT, model_name, d)}")


if __name__ == "__main__":
    main()