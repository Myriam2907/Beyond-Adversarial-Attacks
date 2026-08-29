
import os
import csv
import json
import time
import math
import random
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.transforms import InterpolationMode




CLEAN_DIR = "/mnt/dataset/mapillary_cropped/val"

IMG_SIZE = 224
PERTURB_SIZE = 208
NUM_WORKERS = 8

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

CRITICAL_IDS = [241, 242, 243, 265]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUT_ROOT = "./runtime_benchmark_new_js_subset_v2"


OLD_DDPM_REFERENCE_MS = {
    "mobilenet": 48.40,
    "efficientnet": 48.61,
    "convnext": 50.47,
}

MODELS = {
    "mobilenet": {
        "build": lambda n: _make_mobilenet(n),
        "ckpt": "./mapillary_baseline_results/mobilenetv3_mapillary_best.pth",
        "class_to_idx": "./mapillary_baseline_results/class_to_idx.json",
        "batch_size": 64,
        "ddpm_input_root": "./ddpm_input_combined_js_mobilenet",
    },
    "convnext": {
        "build": lambda n: _make_convnext(n),
        "ckpt": "./convnext_mapillary_results/convnext_mapillary_best.pth",
        "class_to_idx": "./convnext_mapillary_results/class_to_idx.json",
        "batch_size": 64,
        "ddpm_input_root": "./ddpm_input_combined_js_convnext",
    },
    "efficientnet": {
        "build": lambda n: _make_efficientnet(n),
        "ckpt": "./efficientnetv2_mapillary_results/efficientnetv2_mapillary_best.pth",
        "class_to_idx": "./efficientnetv2_mapillary_results/class_to_idx.json",
        "batch_size": 64,
        "ddpm_input_root": "./ddpm_input_combined_js_efficientnet",
    },
}




def _make_mobilenet(num_classes):
    m = models.mobilenet_v3_large(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


def _make_convnext(num_classes):
    m = models.convnext_tiny(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


def _make_efficientnet(num_classes):
    m = models.efficientnet_v2_s(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
    return m


class FixedClassImageFolder(datasets.ImageFolder):
    def __init__(self, root, class_to_idx, transform=None):
        self.fixed_class_to_idx = class_to_idx
        super().__init__(
            root=root,
            transform=transform,
            allow_empty=True,
        )

    def find_classes(self, directory):
        classes = list(self.fixed_class_to_idx.keys())
        return classes, self.fixed_class_to_idx




def normalize(x):
    mean = torch.tensor(
        MEAN, device=x.device, dtype=x.dtype
    ).view(1, 3, 1, 1)

    std = torch.tensor(
        STD, device=x.device, dtype=x.dtype
    ).view(1, 3, 1, 1)

    return (x - mean) / std


def weak_augment_1(x):
    y = ((x - 0.5) * 1.03 + 0.5 + 0.015).clamp(0, 1)
    blur = F.avg_pool2d(
        y, kernel_size=3, stride=1, padding=1
    )
    return (0.85 * y + 0.15 * blur).clamp(0, 1)


def weak_augment_2(x):
    y = ((x - 0.5) * 0.97 + 0.5 - 0.010).clamp(0, 1)
    blur = F.avg_pool2d(
        y, kernel_size=5, stride=1, padding=2
    )
    return (0.90 * y + 0.10 * blur).clamp(0, 1)


def mild_resize_transform(x):
    x_small = F.interpolate(
        x,
        size=(PERTURB_SIZE, PERTURB_SIZE),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    x_back = F.interpolate(
        x_small,
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return x_back.clamp(0, 1)


def js_divergence_from_logits(logits_a, logits_b, eps=1e-12):
    p = torch.softmax(logits_a.float(), dim=1).clamp_min(eps)
    q = torch.softmax(logits_b.float(), dim=1).clamp_min(eps)
    m = 0.5 * (p + q)

    kl_pm = torch.sum(
        p * (torch.log(p) - torch.log(m)),
        dim=1,
    )
    kl_qm = torch.sum(
        q * (torch.log(q) - torch.log(m)),
        dim=1,
    )
    return 0.5 * (kl_pm + kl_qm)


@torch.inference_mode()
def base_forward(model, x):
    logits = model(normalize(x))
    probs = F.softmax(logits, dim=1)
    conf, pred = probs.max(dim=1)
    energy = -torch.logsumexp(logits, dim=1)
    return logits, pred, conf, energy




def load_class_map(path):
    with open(path, "r") as f:
        return {str(k): int(v) for k, v in json.load(f).items()}


def load_model(cfg, num_classes):
    model = cfg["build"](num_classes).to(DEVICE)
    state = torch.load(cfg["ckpt"], map_location=DEVICE)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if any(str(k).startswith("module.") for k in state.keys()):
        state = {
            str(k).replace("module.", "", 1): v
            for k, v in state.items()
        }

    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def warmup(model, batch_size):
    if DEVICE.type != "cuda":
        return

    x = torch.rand(
        min(batch_size, 32),
        3,
        IMG_SIZE,
        IMG_SIZE,
        device=DEVICE,
    )

    with torch.inference_mode():
        for _ in range(12):
            _ = model(normalize(x))

    sync()


def deterministic_subset_indices(n_total, n_requested, seed):
    n = min(int(n_requested), int(n_total))
    rng = np.random.default_rng(seed)
    idx = rng.choice(
        n_total,
        size=n,
        replace=False,
    )
    idx.sort()
    return idx.tolist()


def make_loader(class_to_idx, n, seed, batch_size):
    tfm = transforms.Compose([
        transforms.Resize(
            (IMG_SIZE, IMG_SIZE),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        transforms.ToTensor(),
    ])

    ds = FixedClassImageFolder(
        CLEAN_DIR,
        class_to_idx,
        transform=tfm,
    )

    indices = deterministic_subset_indices(
        len(ds), n, seed
    )

    subset = Subset(ds, indices)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    return loader, len(subset)


@torch.inference_mode()
def benchmark_model(model_key, cfg, n, seed):
    class_to_idx = load_class_map(cfg["class_to_idx"])
    model = load_model(cfg, len(class_to_idx))

    loader, n_actual = make_loader(
        class_to_idx,
        n=n,
        seed=seed,
        batch_size=cfg["batch_size"],
    )

    warmup(model, cfg["batch_size"])

    cls_total_ms = 0.0
    old_extra_total_ms = 0.0
    js_extra_total_ms = 0.0
    recls_total_ms = 0.0
    n_seen = 0
    n_critical = 0

    for x, _ in loader:
        x = x.to(DEVICE, non_blocking=True)
        b = x.shape[0]
        n_seen += b

       
        sync()
        t0 = time.perf_counter()

        logits, pred, conf, energy = base_forward(model, x)

        sync()
        cls_total_ms += (time.perf_counter() - t0) * 1000.0

        
        sync()
        t0 = time.perf_counter()

        x1 = weak_augment_1(x)
        logits1, pred1, conf1, _ = base_forward(model, x1)

        _cd2 = conf - conf1
        _l2_2 = torch.norm(logits - logits1, p=2, dim=1)
        _ch2 = pred != pred1

        critical_mask = torch.zeros_like(
            pred,
            dtype=torch.bool,
        )

        for cid in CRITICAL_IDS:
            critical_mask |= (pred == cid)

        n_critical += int(critical_mask.sum().item())

        if critical_mask.any():
            xc = x[critical_mask]
            x2 = weak_augment_2(xc)
            logits2, pred2, conf2, _ = base_forward(model, x2)

            conf_base_c = conf[critical_mask]
            conf1_c = conf1[critical_mask]

            _cd3 = torch.maximum(
                conf_base_c - conf1_c,
                conf_base_c - conf2,
            )

            _l2_3 = torch.maximum(
                torch.norm(
                    logits[critical_mask]
                    - logits1[critical_mask],
                    p=2,
                    dim=1,
                ),
                torch.norm(
                    logits[critical_mask]
                    - logits2,
                    p=2,
                    dim=1,
                ),
            )

            _ch3 = (
                (pred[critical_mask] != pred1[critical_mask])
                | (pred[critical_mask] != pred2)
            )

        sync()
        old_extra_total_ms += (
            time.perf_counter() - t0
        ) * 1000.0

     
        sync()
        t0 = time.perf_counter()

        x_js = mild_resize_transform(x)
        logits_js = model(normalize(x_js))
        _js = js_divergence_from_logits(
            logits,
            logits_js,
        )

        sync()
        js_extra_total_ms += (
            time.perf_counter() - t0
        ) * 1000.0

        
        sync()
        t0 = time.perf_counter()

        _ = model(normalize(x))

        sync()
        recls_total_ms += (
            time.perf_counter() - t0
        ) * 1000.0

    if n_seen == 0:
        raise RuntimeError("No images were benchmarked.")

    cls_ms = cls_total_ms / n_seen
    old_extra_ms = old_extra_total_ms / n_seen
    js_extra_ms = js_extra_total_ms / n_seen
    recls_ms = recls_total_ms / n_seen

    result = {
        "model": model_key,
        "device": str(DEVICE),
        "n_images": int(n_seen),
        "batch_size": int(cfg["batch_size"]),
        "classification_ms_per_image": float(cls_ms),
        "old_detector_extra_ms_per_image": float(old_extra_ms),
        "js_extra_ms_per_image": float(js_extra_ms),
        "detector_extra_total_ms_per_image": float(
            old_extra_ms + js_extra_ms
        ),
        "classification_plus_detector_ms_per_image": float(
            cls_ms + old_extra_ms + js_extra_ms
        ),
        "reclassification_ms_per_image": float(recls_ms),
        "critical_prediction_fraction_percent": float(
            100.0 * n_critical / n_seen
        ),
    }

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return result



def import_reconstruction_module():
    candidates = [
        "./reconstruct_ddpm_mapillary_combined_js_final.py",
        "/mnt/data/reconstruct_ddpm_mapillary_combined_js_final.py",
    ]

    script_path = None
    for p in candidates:
        if os.path.exists(p):
            script_path = os.path.abspath(p)
            break

    if script_path is None:
        raise FileNotFoundError(
            "Could not find reconstruct_ddpm_mapillary_combined_js_final.py"
        )

    spec = importlib.util.spec_from_file_location(
        "ddpm_current_runtime_module",
        script_path,
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def choose_ddpm_attack_root(input_root):
    preferred = [
        "fgsm",
        "rfgsm",
        "pgd",
        "random_patch",
        "gaussian",
        "salt_pepper",
        "fog",
        "motion_blur",
        "light",
    ]

    for attack in preferred:
        p = os.path.join(input_root, attack)
        if os.path.isdir(p):
            return attack, p

    return None, None


@torch.inference_mode()
def benchmark_ddpm_subset(model_key, cfg, ddpm_n, batch_size, steps):
    
    R = import_reconstruction_module()

    attack, in_dir = choose_ddpm_attack_root(
        cfg["ddpm_input_root"]
    )

    if in_dir is None:
        return {
            "ddpm_measured": False,
            "reason": "No combined-JS DDPM input folder found.",
        }

    
    classifier, class_to_idx = R.load_classifier(
        model_key,
        R.MODELS[model_key],
    )

    records = R.list_images_recursive(
        in_dir,
        class_to_idx,
    )

    if not records:
        del classifier
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

        return {
            "ddpm_measured": False,
            "reason": f"No images found in {in_dir}",
        }

    
    records = records[: min(ddpm_n, len(records))]

    reconstructor = R.DDPMPartialReconstructor(
        ddpm_dir=R.DDPM_DIR,
        device=R.DEVICE,
        inference_steps=steps,
        seed=123,
        use_amp=True,
        use_ema=True,
    )

    t_candidates = [
        None,
        20,
        40,
        80,
        120,
        160,
    ]

    
    warm_records = records[: min(batch_size, len(records))]
    xw = R.load_batch(warm_records)
    _ = R.classifier_outputs(classifier, xw)
    sync()

    total_ms = 0.0
    total_images = 0

    for start in range(0, len(records), batch_size):
        batch_records = records[
            start:start + batch_size
        ]
        x = R.load_batch(batch_records)

        sync()
        t0 = time.perf_counter()

        best_conf = None

        for requested_t in t_candidates:
            if requested_t is None:
                candidate_x = x
            else:
                candidate_x, _, _ = reconstructor.reconstruct_with_t(
                    x,
                    int(requested_t),
                    deterministic_key=(
                        f"runtime-benchmark|{model_key}|"
                        f"{attack}|{start}|{requested_t}"
                    ),
                )

            candidate_x = R.quantize_png_equivalent(
                candidate_x
            )

            _, candidate_conf = R.classifier_outputs(
                classifier,
                candidate_x,
            )

            if best_conf is None:
                best_conf = candidate_conf
            else:
                best_conf = torch.maximum(
                    best_conf,
                    candidate_conf,
                )

        sync()

        total_ms += (
            time.perf_counter() - t0
        ) * 1000.0

        total_images += x.shape[0]

    measured = total_ms / total_images

    del classifier
    del reconstructor
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "ddpm_measured": True,
        "attack_used": attack,
        "n_images": int(total_images),
        "batch_size": int(batch_size),
        "steps": int(steps),
        "t_candidates": ["skip", 20, 40, 80, 120, 160],
        "ddpm_plus_candidate_selection_ms_per_image": float(measured),
    }




def make_final_rows(results, ddpm_results, use_reference_ddpm):
    rows = []

    for r in results:
        model = r["model"]

        ddpm_measured = None
        if model in ddpm_results:
            d = ddpm_results[model]
            if d.get("ddpm_measured"):
                ddpm_measured = d[
                    "ddpm_plus_candidate_selection_ms_per_image"
                ]

        if ddpm_measured is not None:
            ddpm_ms = ddpm_measured
            ddpm_source = "measured_subset"
        elif use_reference_ddpm:
            ddpm_ms = OLD_DDPM_REFERENCE_MS[model]
            ddpm_source = "old_reference"
        else:
            ddpm_ms = None
            ddpm_source = "not_available"

        total_flagged = None

        if ddpm_ms is not None:
            total_flagged = (
                r["classification_ms_per_image"]
                + r["old_detector_extra_ms_per_image"]
                + r["js_extra_ms_per_image"]
                + ddpm_ms
                + r["reclassification_ms_per_image"]
            )

        row = dict(r)
        row.update({
            "ddpm_ms_per_flagged_image": ddpm_ms,
            "ddpm_source": ddpm_source,
            "total_flagged_pipeline_ms_per_image": total_flagged,
        })

        rows.append(row)

    return rows


def save_outputs(rows, ddpm_details, args):
    Path(OUT_ROOT).mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = os.path.join(
        OUT_ROOT,
        "runtime_subset_results.json",
    )

    csv_path = os.path.join(
        OUT_ROOT,
        "runtime_subset_results.csv",
    )

    tex_path = os.path.join(
        OUT_ROOT,
        "runtime_subset_table.tex",
    )

    payload = {
        "configuration": {
            "device": str(DEVICE),
            "subset_n_requested": int(args.n),
            "seed": int(args.seed),
            "measure_ddpm": bool(args.measure_ddpm),
            "ddpm_n_requested": int(args.ddpm_n),
            "ddpm_steps": int(args.ddpm_steps),
            "old_ddpm_reference_used_when_not_measured": bool(
                args.use_old_ddpm_reference
            ),
            "output_root": OUT_ROOT,
            "note": (
                "New output directory only; old experiment results are untouched."
            ),
        },
        "results": rows,
        "ddpm_benchmark_details": ddpm_details,
    }

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    fields = [
        "model",
        "n_images",
        "classification_ms_per_image",
        "old_detector_extra_ms_per_image",
        "js_extra_ms_per_image",
        "detector_extra_total_ms_per_image",
        "classification_plus_detector_ms_per_image",
        "ddpm_ms_per_flagged_image",
        "ddpm_source",
        "reclassification_ms_per_image",
        "total_flagged_pipeline_ms_per_image",
        "critical_prediction_fraction_percent",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        for r in rows:
            writer.writerow({
                k: r.get(k)
                for k in fields
            })

    with open(tex_path, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{lrrrrrrr}\n")
        f.write("\\toprule\n")
        f.write(
            "Architecture & Cls. & Old Det. & JS & Det. Total & "
            "DDPM & Recls. & Total \\\\\n"
        )
        f.write("\\midrule\n")

        pretty = {
            "mobilenet": "MobileNetV3",
            "efficientnet": "EfficientNetV2-S",
            "convnext": "ConvNeXt-Tiny",
        }

        def v(x):
            return "--" if x is None else f"{x:.3f}"

        for r in rows:
            f.write(
                f"{pretty[r['model']]} & "
                f"{v(r['classification_ms_per_image'])} & "
                f"{v(r['old_detector_extra_ms_per_image'])} & "
                f"{v(r['js_extra_ms_per_image'])} & "
                f"{v(r['detector_extra_total_ms_per_image'])} & "
                f"{v(r['ddpm_ms_per_flagged_image'])} & "
                f"{v(r['reclassification_ms_per_image'])} & "
                f"{v(r['total_flagged_pipeline_ms_per_image'])} \\\\\n"
            )

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write(
            "\\caption{Subset runtime benchmark of the updated MTSD pipeline "
            "(ms/image). Old Det. and JS are incremental costs after reusing "
            "the base classifier output.}\n"
        )
        f.write("\\label{tab:runtime_new_js_subset}\n")
        f.write("\\end{table}\n")

    return json_path, csv_path, tex_path




def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--model",
        choices=[
            "all",
            "mobilenet",
            "convnext",
            "efficientnet",
        ],
        default="all",
    )

    p.add_argument(
        "--n",
        type=int,
        default=2000,
        help="Number of clean images per model for classifier/detector timing.",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=123,
    )

    p.add_argument(
        "--measure_ddpm",
        action="store_true",
        help="Also benchmark current DDPM on a tiny flagged subset.",
    )

    p.add_argument(
        "--ddpm_n",
        type=int,
        default=64,
        help="Flagged images per model for optional DDPM timing.",
    )

    p.add_argument(
        "--ddpm_batch",
        type=int,
        default=16,
    )

    p.add_argument(
        "--ddpm_steps",
        type=int,
        default=100,
    )

    p.add_argument(
        "--no_old_ddpm_reference",
        dest="use_old_ddpm_reference",
        action="store_false",
        help="Do not use old DDPM averages as provisional reference.",
    )

    p.set_defaults(
        use_old_ddpm_reference=True
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.n <= 0:
        raise ValueError("--n must be > 0")

    if args.ddpm_n <= 0:
        raise ValueError("--ddpm_n must be > 0")

    models_to_run = (
        list(MODELS.keys())
        if args.model == "all"
        else [args.model]
    )

    print("=" * 102)
    print("FAST SUBSET RUNTIME BENCHMARK — UPDATED MTSD OLD+JS PIPELINE")
    print("=" * 102)
    print(f"Device              : {DEVICE}")
    print(f"Models              : {models_to_run}")
    print(f"Timing subset/model : {args.n}")
    print(f"Measure DDPM        : {args.measure_ddpm}")
    print(f"NEW output only     : {OUT_ROOT}")
    print("Old results         : NOT MODIFIED")

    results = []

    for model_key in models_to_run:
        print(f"\n[{model_key}] benchmarking classifier + old detector + JS ...")

        r = benchmark_model(
            model_key,
            MODELS[model_key],
            n=args.n,
            seed=args.seed,
        )

        results.append(r)

        print(
            f"  Cls      : {r['classification_ms_per_image']:.4f} ms/img\n"
            f"  Old Det  : {r['old_detector_extra_ms_per_image']:.4f} ms/img\n"
            f"  JS       : {r['js_extra_ms_per_image']:.4f} ms/img\n"
            f"  DetTotal : {r['detector_extra_total_ms_per_image']:.4f} ms/img\n"
            f"  Cls+Det  : {r['classification_plus_detector_ms_per_image']:.4f} ms/img\n"
            f"  Recls    : {r['reclassification_ms_per_image']:.4f} ms/img"
        )

    ddpm_details = {}

    if args.measure_ddpm:
        print("\n" + "=" * 102)
        print("OPTIONAL SMALL DDPM BENCHMARK")
        print("=" * 102)
        print(
            "For reliable timing, do NOT run this at the same time as another "
            "GPU-heavy DDPM job."
        )

        for model_key in models_to_run:
            print(
                f"\n[{model_key}] DDPM benchmark on <= {args.ddpm_n} flagged images ..."
            )

            try:
                d = benchmark_ddpm_subset(
                    model_key,
                    MODELS[model_key],
                    ddpm_n=args.ddpm_n,
                    batch_size=args.ddpm_batch,
                    steps=args.ddpm_steps,
                )
            except Exception as exc:
                d = {
                    "ddpm_measured": False,
                    "reason": str(exc),
                }

            ddpm_details[model_key] = d

            if d.get("ddpm_measured"):
                print(
                    "  DDPM+selection: "
                    f"{d['ddpm_plus_candidate_selection_ms_per_image']:.3f} ms/img "
                    f"(n={d['n_images']})"
                )
            else:
                print(
                    f"  DDPM timing unavailable: {d.get('reason')}"
                )

    rows = make_final_rows(
        results,
        ddpm_details,
        use_reference_ddpm=args.use_old_ddpm_reference,
    )

    json_path, csv_path, tex_path = save_outputs(
        rows,
        ddpm_details,
        args,
    )

    print("\n" + "=" * 102)
    print("FINAL SUBSET RUNTIME RESULTS")
    print("=" * 102)

    print(
        f"{'Model':<14}"
        f"{'Cls':>10}"
        f"{'OldDet':>10}"
        f"{'JS':>10}"
        f"{'DetTotal':>11}"
        f"{'DDPM':>10}"
        f"{'Recls':>10}"
        f"{'Total':>11}"
    )
    print("-" * 102)

    for r in rows:
        def fnum(x):
            return "N/A" if x is None else f"{x:.3f}"

        print(
            f"{r['model']:<14}"
            f"{fnum(r['classification_ms_per_image']):>10}"
            f"{fnum(r['old_detector_extra_ms_per_image']):>10}"
            f"{fnum(r['js_extra_ms_per_image']):>10}"
            f"{fnum(r['detector_extra_total_ms_per_image']):>11}"
            f"{fnum(r['ddpm_ms_per_flagged_image']):>10}"
            f"{fnum(r['reclassification_ms_per_image']):>10}"
            f"{fnum(r['total_flagged_pipeline_ms_per_image']):>11}"
        )

        if r["ddpm_source"] == "old_reference":
            print(
                f"  NOTE [{r['model']}]: DDPM={r['ddpm_ms_per_flagged_image']:.2f} ms "
                "is the OLD reference average, not a new subset measurement."
            )

    print("\nSaved NEW files only:")
    print(f"  {json_path}")
    print(f"  {csv_path}")
    print(f"  {tex_path}")
    print("\nNo old detector/DDPM/result directory was deleted or overwritten.")


if __name__ == "__main__":
    main()