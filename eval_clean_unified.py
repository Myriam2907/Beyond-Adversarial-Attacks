

import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm



VAL_DIR = "/mnt/dataset/mapillary_cropped/val"

IMG_SIZE = 224
NUM_WORKERS = 12
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
        "out_dir": "./mapillary_baseline_results",
        "prefix": "mobilenet",
        "batch_size": 128,
    },
    "convnext": {
        "build": make_convnext,
        "ckpt": "./convnext_mapillary_results/convnext_mapillary_best.pth",
        "class_to_idx": "./convnext_mapillary_results/class_to_idx.json",
        "out_dir": "./convnext_mapillary_results",
        "prefix": "convnext",
        "batch_size": 128,
    },
    "efficientnet": {
        "build": make_efficientnet,
        "ckpt": "./efficientnetv2_mapillary_results/efficientnetv2_mapillary_best.pth",
        "class_to_idx": "./efficientnetv2_mapillary_results/class_to_idx.json",
        "out_dir": "./efficientnetv2_mapillary_results",
        "prefix": "efficientnet",
        "batch_size": 128,
    },
}



class FixedClassImageFolder(datasets.ImageFolder):
    def __init__(self, root, class_to_idx, transform=None):
        self.fixed_class_to_idx = class_to_idx
        super().__init__(root=root, transform=transform, allow_empty=True)

    def find_classes(self, directory):
        classes = list(self.fixed_class_to_idx.keys())
        return classes, self.fixed_class_to_idx


def energy_from_logits(logits):
    return -torch.logsumexp(logits, dim=1)


@torch.no_grad()
def eval_model(model_key):
    cfg = MODELS[model_key]

    if not os.path.exists(cfg["ckpt"]):
        raise FileNotFoundError(f"checkpoint missing: {cfg['ckpt']}")
    if not os.path.exists(cfg["class_to_idx"]):
        raise FileNotFoundError(f"class_to_idx missing: {cfg['class_to_idx']}")

    Path(cfg["out_dir"]).mkdir(parents=True, exist_ok=True)

    with open(cfg["class_to_idx"]) as f:
        class_to_idx = json.load(f)
    num_classes = len(class_to_idx)

    tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    ds = FixedClassImageFolder(VAL_DIR, class_to_idx, transform=tfms)
    if len(ds) == 0:
        raise RuntimeError("0 images loaded. Check VAL_DIR and class_to_idx match.")
    loader = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=(DEVICE.type == "cuda"))

    model = cfg["build"](num_classes).to(DEVICE)
    model.load_state_dict(torch.load(cfg["ckpt"], map_location=DEVICE))
    model.eval()

    print(f"\n[{model_key}] images={len(ds)} classes={num_classes} ckpt={cfg['ckpt']}")

    softmax = nn.Softmax(dim=1)
    logits_list, conf_list, energy_list = [], [], []
    pred_list, label_list, time_ms_list = [], [], []

   
    if DEVICE.type == "cuda":
        warm = torch.randn(8, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
        for _ in range(10):
            _ = model(warm)
        torch.cuda.synchronize()

    for images, labels in tqdm(loader, desc=f"[{model_key}] clean eval"):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        logits = model(images)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        per_img_ms = ((t1 - t0) * 1000.0) / images.size(0)

        probs = softmax(logits)
        conf, pred = probs.max(dim=1)
        energy = energy_from_logits(logits)

        logits_list.append(logits.cpu().numpy())
        conf_list.append(conf.cpu().numpy())
        energy_list.append(energy.cpu().numpy())
        pred_list.append(pred.cpu().numpy())
        label_list.append(labels.cpu().numpy())
        time_ms_list.append(np.full(images.size(0), per_img_ms, dtype=np.float32))

    logits_all = np.concatenate(logits_list)
    conf_all = np.concatenate(conf_list)
    energy_all = np.concatenate(energy_list)
    pred_all = np.concatenate(pred_list)
    label_all = np.concatenate(label_list)
    time_ms_all = np.concatenate(time_ms_list)

    acc = 100.0 * (pred_all == label_all).mean()
    avg_ms = float(time_ms_all.mean())

    p = cfg["prefix"]
    od = cfg["out_dir"]
    np.save(os.path.join(od, f"{p}_val_logits.npy"), logits_all)
    np.save(os.path.join(od, f"{p}_val_confidences.npy"), conf_all)
    np.save(os.path.join(od, f"{p}_val_energy.npy"), energy_all)
    np.save(os.path.join(od, f"{p}_val_predictions.npy"), pred_all)
    np.save(os.path.join(od, f"{p}_val_labels.npy"), label_all)
    np.save(os.path.join(od, f"{p}_val_inference_time_ms.npy"), time_ms_all)

    stats = {
        "model": model_key,
        "dataset": "Mapillary cropped val",
        "num_samples": int(len(ds)),
        "num_classes": int(num_classes),
        "accuracy_percent": float(acc),
        "avg_inference_time_ms_per_image": avg_ms,
        "mean_confidence": float(conf_all.mean()),
        "mean_energy": float(energy_all.mean()),
        "model_path": cfg["ckpt"],
    }
    with open(os.path.join(od, f"{p}_baseline_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(f"[{model_key}] accuracy={acc:.3f}%  avg={avg_ms:.4f} ms/img  "
          f"mean_conf={conf_all.mean():.4f}  mean_energy={energy_all.mean():.4f}")
    print(f"[{model_key}] saved arrays + {p}_baseline_stats.json -> {od}")

    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS.keys()) + ["all"], required=True)
    args = ap.parse_args()

    print("Device:", DEVICE)
    keys = list(MODELS.keys()) if args.model == "all" else [args.model]

    all_stats = {}
    for k in keys:
        all_stats[k] = eval_model(k)

    print("\n==================== SUMMARY ====================")
    for k in keys:
        s = all_stats[k]
        print(f"  {k:13s} acc={s['accuracy_percent']:.3f}%  "
              f"{s['avg_inference_time_ms_per_image']:.4f} ms/img  "
              f"conf={s['mean_confidence']:.4f}  energy={s['mean_energy']:.4f}")


if __name__ == "__main__":
    main()