import os
import re
import csv
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm


ATTACKED_ROOT = "./attacked_pngs"  
OUT_ROOT = "./gtsrb_repeat/ours_js" 

MODEL_PATHS = {
    "mobilenet": "./gtsrb_repeat/models/mobilenet/mobilenet_gtsrb_best.pth",
    "convnext": "./gtsrb_repeat/models/convnext/convnext_gtsrb_best.pth",
    "efficientnet": "./gtsrb_repeat/models/efficientnet/efficientnet_gtsrb_best.pth",
}

ATTACK_DIRS = {
    "clean": os.path.join(ATTACKED_ROOT, "clean_png"),
    "fgsm": os.path.join(ATTACKED_ROOT, "fgsm_png"),
    "patch": os.path.join(ATTACKED_ROOT, "patch_png"),
    "light": os.path.join(ATTACKED_ROOT, "light_png"),
}

IMG_SIZE = 224
NUM_CLASSES = 43
BATCH_SIZE = 128
NUM_WORKERS = 4
SEED = 123

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

STOP_ID = 14
YIELD_ID = 13


PERCENTILE = 95
WEAK_K = 2

# JS settings
JS_RESIZE = 208
JS_ONLY_TARGET_FPR = 0.05
COMBINED_TARGET_FPR = 0.06

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"



def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



class PNGFolderWithLabel(Dataset):
    def __init__(self, folder, transform=None):
        self.folder = folder
        self.transform = transform
        self.paths = sorted([
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith(".png")
        ])
        if len(self.paths) == 0:
            raise RuntimeError(f"No PNG files found in: {folder}")

        self.label_re = re.compile(r"_y(\d+)\.png$", re.IGNORECASE)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        m = self.label_re.search(os.path.basename(p))
        if not m:
            raise RuntimeError(
                f"Filename must end with _y<label>.png; got {os.path.basename(p)}"
            )
        y = int(m.group(1))
        img = Image.open(p).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y, p



def build_model(model_name):
    model_name = model_name.lower()

    if model_name == "mobilenet":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)

    elif model_name == "convnext":
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)

    elif model_name == "efficientnet":
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model


def load_model(model_name, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = build_model(model_name).to(DEVICE)
    state = torch.load(checkpoint_path, map_location=DEVICE)

    
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    model.load_state_dict(state)
    model.eval()
    return model



def energy_from_logits(logits):
    return -torch.logsumexp(logits, dim=1)


def perturb1(x):
   
    x2 = TF.gaussian_blur(x, kernel_size=[3, 3], sigma=[0.1, 0.6])
    x2 = x2 + 0.03
    return x2.clamp(-5, 5)


def perturb2(x):
    
    x3 = TF.gaussian_blur(x, kernel_size=[5, 5], sigma=[0.2, 0.9])
    x3 = x3 - 0.02
    return x3.clamp(-5, 5)



def js_resize_perturb(x):
    """224 -> 208 -> 224 resize consistency transformation."""
    small = F.interpolate(
        x,
        size=(JS_RESIZE, JS_RESIZE),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    restored = F.interpolate(
        small,
        size=(IMG_SIZE, IMG_SIZE),
        mode="bilinear",
        align_corners=False,
        antialias=True,
    )
    return restored


def js_divergence(p, q, eps=1e-8):
    """Per-image Jensen-Shannon divergence between two class distributions."""
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)), dim=1)
    kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)), dim=1)
    return 0.5 * (kl_pm + kl_qm)



@torch.no_grad()
def evaluate_folder(model, model_name, attack_name, png_dir, out_dir):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    ds = PNGFolderWithLabel(png_dir, transform=tfm)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )

    softmax = nn.Softmax(dim=1)

    logits_list = []
    confidence_list = []
    energy_list = []
    pred_list = []
    label_list = []
    filenames_list = []
    time_ms_list = []

    change2_list = []
    confdrop2_list = []
    logitdiff2_list = []

    change3_list = []
    maxconfdrop3_list = []
    maxlogitdiff3_list = []

    js_list = []

   
    if DEVICE.type == "cuda":
        warm = torch.randn(8, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
        for _ in range(10):
            _ = model(warm)
        torch.cuda.synchronize()

    desc = f"{model_name}/{attack_name}"
    for images, labels, paths in tqdm(loader, desc=desc):
        filenames_list.extend(list(paths))
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits1 = model(images)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        per_img_ms = ((t1 - t0) * 1000.0) / images.size(0)

        probs1 = softmax(logits1)
        conf1, pred1 = probs1.max(dim=1)
        energy1 = energy_from_logits(logits1)

        
        images2 = perturb1(images)
        logits2 = model(images2)
        probs2 = softmax(logits2)
        conf2, pred2 = probs2.max(dim=1)

        changed2 = (pred2 != pred1).int()
        conf_drop2 = (conf1 - conf2).clamp(min=0.0)
        l2_diff2 = torch.norm(logits1 - logits2, dim=1)

        
        is_critical = (labels == STOP_ID) | (labels == YIELD_ID)

        changed3 = torch.full_like(changed2, fill_value=-1)
        max_conf_drop3 = torch.full_like(conf_drop2, fill_value=-1.0)
        max_l2_diff3 = torch.full_like(l2_diff2, fill_value=-1.0)

        if is_critical.any():
            crit_idx = is_critical.nonzero(as_tuple=True)[0]

            images3 = perturb2(images[crit_idx])
            logits3 = model(images3)
            probs3 = softmax(logits3)
            conf3, pred3 = probs3.max(dim=1)

            changed_any = (
                (pred2[crit_idx] != pred1[crit_idx]) |
                (pred3 != pred1[crit_idx])
            ).int()
            changed3[crit_idx] = changed_any

            drop13 = (conf1[crit_idx] - conf3).clamp(min=0.0)
            max_conf_drop3[crit_idx] = torch.maximum(
                conf_drop2[crit_idx], drop13
            )

            l2_diff13 = torch.norm(logits1[crit_idx] - logits3, dim=1)
            max_l2_diff3[crit_idx] = torch.maximum(
                l2_diff2[crit_idx], l2_diff13
            )

        # ----------------------------------------------------
        # NEW JS PASS -- 224 -> 208 -> 224
        # ----------------------------------------------------
        images_js = js_resize_perturb(images)
        logits_js = model(images_js)
        probs_js = softmax(logits_js)
        js_score = js_divergence(probs1, probs_js)

       
        logits_list.append(logits1.cpu().numpy())
        confidence_list.append(conf1.cpu().numpy())
        energy_list.append(energy1.cpu().numpy())
        pred_list.append(pred1.cpu().numpy())
        label_list.append(labels.cpu().numpy())
        time_ms_list.append(
            np.full(images.size(0), per_img_ms, dtype=np.float32)
        )

        change2_list.append(changed2.cpu().numpy())
        confdrop2_list.append(conf_drop2.cpu().numpy())
        logitdiff2_list.append(l2_diff2.cpu().numpy())

        change3_list.append(changed3.cpu().numpy())
        maxconfdrop3_list.append(max_conf_drop3.cpu().numpy())
        maxlogitdiff3_list.append(max_l2_diff3.cpu().numpy())

        js_list.append(js_score.cpu().numpy())

    
    arrays = {
        "logits": np.concatenate(logits_list, axis=0),
        "confidence": np.concatenate(confidence_list, axis=0),
        "energy": np.concatenate(energy_list, axis=0),
        "pred": np.concatenate(pred_list, axis=0),
        "label": np.concatenate(label_list, axis=0),
        "time_ms": np.concatenate(time_ms_list, axis=0),
        "2pass_changed": np.concatenate(change2_list, axis=0),
        "2pass_conf_drop": np.concatenate(confdrop2_list, axis=0),
        "2pass_logit_l2": np.concatenate(logitdiff2_list, axis=0),
        "3pass_changed_true_stop_yield": np.concatenate(change3_list, axis=0),
        "3pass_max_conf_drop_true_stop_yield": np.concatenate(maxconfdrop3_list, axis=0),
        "3pass_max_logit_l2_true_stop_yield": np.concatenate(maxlogitdiff3_list, axis=0),
        "js": np.concatenate(js_list, axis=0),
    }

    n = len(arrays["label"])
    if len(filenames_list) != n:
        raise RuntimeError(
            f"Filename count mismatch for {model_name}/{attack_name}: "
            f"{len(filenames_list)} vs {n}"
        )

    
    for name, arr in arrays.items():
        np.save(os.path.join(out_dir, f"{name}.npy"), arr)
    np.save(
        os.path.join(out_dir, "filenames.npy"),
        np.asarray(filenames_list, dtype=object),
    )

    accuracy = 100.0 * float((arrays["pred"] == arrays["label"]).mean())
    stats = {
        "model": model_name,
        "attack": attack_name,
        "n_samples": int(n),
        "accuracy_percent": accuracy,
        "avg_base_inference_ms_per_image": float(arrays["time_ms"].mean()),
        "js_mean": float(arrays["js"].mean()),
        "js_std": float(arrays["js"].std()),
    }

    with open(os.path.join(out_dir, "evaluation_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print(
        f"  {model_name:12s} {attack_name:6s} | "
        f"acc={accuracy:7.3f}% | n={n:,} | "
        f"JS mean={stats['js_mean']:.8f}"
    )

    return stats



def load_eval(eval_dir):
    names = [
        "energy", "confidence", "pred", "label",
        "2pass_conf_drop", "2pass_logit_l2", "2pass_changed",
        "3pass_max_conf_drop_true_stop_yield",
        "3pass_max_logit_l2_true_stop_yield",
        "3pass_changed_true_stop_yield",
        "js",
    ]
    return {
        name: np.load(os.path.join(eval_dir, f"{name}.npy"))
        for name in names
    }


def compute_ours_thresholds(clean_dir, model_name):
    x = load_eval(clean_dir)
    label = x["label"]
    critical = (label == STOP_ID) | (label == YIELD_ID)

    if not critical.any():
        raise RuntimeError("No STOP/YIELD samples found in clean evaluation.")

    thresholds = {
        "model": model_name,
        "percentile": PERCENTILE,
        "weak_k": WEAK_K,
        "clean_samples": int(len(label)),

        "energy_threshold": float(np.percentile(x["energy"], PERCENTILE)),
        "confidence_min_threshold": float(
            np.percentile(x["confidence"], 100 - PERCENTILE)
        ),
        "conf_drop_2pass_threshold": float(
            np.percentile(x["2pass_conf_drop"], PERCENTILE)
        ),
        "logit_l2_2pass_threshold": float(
            np.percentile(x["2pass_logit_l2"], PERCENTILE)
        ),
        "conf_drop_3pass_threshold": float(
            np.percentile(
                x["3pass_max_conf_drop_true_stop_yield"][critical],
                PERCENTILE,
            )
        ),
        "logit_l2_3pass_threshold": float(
            np.percentile(
                x["3pass_max_logit_l2_true_stop_yield"][critical],
                PERCENTILE,
            )
        ),
    }

    return thresholds



def ours_flags(x, thresholds):
    label = x["label"]
    n = len(label)

    flag_energy = x["energy"] > thresholds["energy_threshold"]
    flag_confidence = x["confidence"] < thresholds["confidence_min_threshold"]
    flag_conf_drop_2pass = (
        x["2pass_conf_drop"] > thresholds["conf_drop_2pass_threshold"]
    )
    flag_logit_2pass = (
        x["2pass_logit_l2"] > thresholds["logit_l2_2pass_threshold"]
    )
    flag_changed_2pass = (x["2pass_changed"] == 1)

    critical = (label == STOP_ID) | (label == YIELD_ID)

    flag_conf_drop_3pass = np.zeros(n, dtype=bool)
    flag_logit_3pass = np.zeros(n, dtype=bool)
    flag_changed_3pass = np.zeros(n, dtype=bool)

    if critical.any():
        flag_conf_drop_3pass[critical] = (
            x["3pass_max_conf_drop_true_stop_yield"][critical]
            > thresholds["conf_drop_3pass_threshold"]
        )
        flag_logit_3pass[critical] = (
            x["3pass_max_logit_l2_true_stop_yield"][critical]
            > thresholds["logit_l2_3pass_threshold"]
        )
        flag_changed_3pass[critical] = (
            x["3pass_changed_true_stop_yield"][critical] == 1
        )

   
    strong = flag_changed_2pass | flag_changed_3pass | flag_logit_2pass

    weak_count = (
        flag_energy.astype(np.int16)
        + flag_confidence.astype(np.int16)
        + flag_conf_drop_2pass.astype(np.int16)
        + flag_conf_drop_3pass.astype(np.int16)
        + flag_logit_3pass.astype(np.int16)
    )

    ours = strong | (weak_count >= WEAK_K)
    return ours



def calibrate_js(clean_x, ours_clean):
    js = clean_x["js"]
    n = len(js)

    
    js_only_threshold = float(
        np.percentile(js, 100.0 * (1.0 - JS_ONLY_TARGET_FPR))
    )
    js_only = js > js_only_threshold


    max_combined = int(np.floor(COMBINED_TARGET_FPR * n))
    ours_count = int(ours_clean.sum())
    budget = max_combined - ours_count

    if budget <= 0:
        combined_js_threshold = float("inf")
    else:
        candidate_scores = js[~ours_clean]
        if len(candidate_scores) == 0:
            combined_js_threshold = float("inf")
        elif budget >= len(candidate_scores):
            combined_js_threshold = float("-inf")
        else:
            
            unique_scores, counts = np.unique(candidate_scores, return_counts=True)
            order = np.argsort(unique_scores)[::-1]
            unique_scores = unique_scores[order]
            counts = counts[order]

            cumulative = 0
            chosen_threshold = float("inf")
            for score, count in zip(unique_scores, counts):
                if cumulative + int(count) > budget:
                    break
                cumulative += int(count)
                
                chosen_threshold = float(np.nextafter(score, -np.inf))

            combined_js_threshold = chosen_threshold

    js_combined = js > combined_js_threshold
    combined = ours_clean | js_combined

    return {
        "js_only_threshold": js_only_threshold,
        "js_only_target_clean_fpr": JS_ONLY_TARGET_FPR,
        "js_only_actual_clean_fpr": float(js_only.mean()),
        "ours_clean_fpr": float(ours_clean.mean()),
        "combined_js_threshold": combined_js_threshold,
        "combined_target_clean_fpr": COMBINED_TARGET_FPR,
        "combined_actual_clean_fpr": float(combined.mean()),
        "clean_samples": int(n),
    }



def method_metrics(flag, pred, label):
    wrong = pred != label
    wrong_total = int(wrong.sum())
    caught_wrong = int((flag & wrong).sum())

    return {
        "flagged": int(flag.sum()),
        "flag_rate_percent": float(flag.mean() * 100.0),
        "wrong_total": wrong_total,
        "wrong_caught": caught_wrong,
        "wrong_detection_percent": float(
            caught_wrong / wrong_total * 100.0 if wrong_total else 0.0
        ),
    }


def evaluate_methods_for_model(model_name, model_root, thresholds, js_cal):
    results = {}

    js_only_t = js_cal["js_only_threshold"]
    js_combined_t = js_cal["combined_js_threshold"]

    for attack_name in ATTACK_DIRS:
        eval_dir = os.path.join(model_root, attack_name)
        x = load_eval(eval_dir)

        ours = ours_flags(x, thresholds)
        js_only = x["js"] > js_only_t
        js_for_combined = x["js"] > js_combined_t
        ours_plus_js = ours | js_for_combined

        pred = x["pred"]
        label = x["label"]
        accuracy = float((pred == label).mean() * 100.0)

        results[attack_name] = {
            "accuracy_percent": accuracy,
            "ours": method_metrics(ours, pred, label),
            "js": method_metrics(js_only, pred, label),
            "ours_plus_js": method_metrics(ours_plus_js, pred, label),
        }

    return results



def save_summary(all_results):
    Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)

    json_path = os.path.join(OUT_ROOT, "final_ours_js_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    csv_path = os.path.join(OUT_ROOT, "final_ours_js_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "model", "attack", "accuracy_percent",
            "ours_flag_percent", "js_flag_percent", "ours_plus_js_flag_percent",
            "wrong_total",
            "ours_wrong_caught", "js_wrong_caught", "ours_plus_js_wrong_caught",
            "ours_wrong_detection_percent", "js_wrong_detection_percent",
            "ours_plus_js_wrong_detection_percent",
        ])

        for model_name, model_result in all_results.items():
            for attack_name, r in model_result["results"].items():
                writer.writerow([
                    model_name,
                    attack_name,
                    r["accuracy_percent"],
                    r["ours"]["flag_rate_percent"],
                    r["js"]["flag_rate_percent"],
                    r["ours_plus_js"]["flag_rate_percent"],
                    r["ours"]["wrong_total"],
                    r["ours"]["wrong_caught"],
                    r["js"]["wrong_caught"],
                    r["ours_plus_js"]["wrong_caught"],
                    r["ours"]["wrong_detection_percent"],
                    r["js"]["wrong_detection_percent"],
                    r["ours_plus_js"]["wrong_detection_percent"],
                ])

    print(f"\nSaved JSON: {os.path.abspath(json_path)}")
    print(f"Saved CSV : {os.path.abspath(csv_path)}")



def main():
    set_seed()
    Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("GTSRB FRESH THREE-MODEL PIPELINE: OURS vs JS vs OURS+JS")
    print("=" * 110)
    print(f"Device: {DEVICE}")
    print(f"Existing attacked PNG root (READ ONLY): {os.path.abspath(ATTACKED_ROOT)}")
    print(f"New result root: {os.path.abspath(OUT_ROOT)}")

    
    for attack_name, p in ATTACK_DIRS.items():
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Missing {attack_name} image folder: {p}")

    for model_name, p in MODEL_PATHS.items():
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing {model_name} checkpoint: {p}")

    evaluation_stats = {}

    
    print("\nSTEP 1/4: Computing OURS signals + JS signal for all models/sets")
    print("-" * 110)

    for model_name, checkpoint_path in MODEL_PATHS.items():
        print(f"\nLoading {model_name}: {checkpoint_path}")
        model = load_model(model_name, checkpoint_path)
        evaluation_stats[model_name] = {}

        for attack_name, png_dir in ATTACK_DIRS.items():
            out_dir = os.path.join(OUT_ROOT, "eval", model_name, attack_name)
            stats = evaluate_folder(
                model=model,
                model_name=model_name,
                attack_name=attack_name,
                png_dir=png_dir,
                out_dir=out_dir,
            )
            evaluation_stats[model_name][attack_name] = stats

        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

  
    print("\nSTEP 2/4: Computing fresh OURS thresholds from each model's CLEAN signals")
    print("-" * 110)

    thresholds_by_model = {}
    for model_name in MODEL_PATHS:
        model_root = os.path.join(OUT_ROOT, "eval", model_name)
        clean_dir = os.path.join(model_root, "clean")
        thresholds = compute_ours_thresholds(clean_dir, model_name)
        thresholds_by_model[model_name] = thresholds

        threshold_path = os.path.join(OUT_ROOT, f"{model_name}_ours_thresholds.json")
        with open(threshold_path, "w") as f:
            json.dump(thresholds, f, indent=2)
        print(f"{model_name:12s} -> {threshold_path}")

    
    print("\nSTEP 3/4: Calibrating JS-alone and OURS+JS on CLEAN")
    print("-" * 110)

    js_by_model = {}
    for model_name in MODEL_PATHS:
        model_root = os.path.join(OUT_ROOT, "eval", model_name)
        clean_x = load_eval(os.path.join(model_root, "clean"))
        ours_clean = ours_flags(clean_x, thresholds_by_model[model_name])
        js_cal = calibrate_js(clean_x, ours_clean)
        js_by_model[model_name] = js_cal

        js_path = os.path.join(OUT_ROOT, f"{model_name}_js_thresholds.json")
        with open(js_path, "w") as f:
            json.dump(js_cal, f, indent=2)

        print(
            f"{model_name:12s} | "
            f"OURS clean FPR={js_cal['ours_clean_fpr']*100:6.2f}% | "
            f"JS clean FPR={js_cal['js_only_actual_clean_fpr']*100:6.2f}% | "
            f"OURS+JS clean FPR={js_cal['combined_actual_clean_fpr']*100:6.2f}%"
        )

    
    print("\nSTEP 4/4: Final OURS vs JS vs OURS+JS comparison")
    print("=" * 110)

    all_results = {}
    for model_name in MODEL_PATHS:
        model_root = os.path.join(OUT_ROOT, "eval", model_name)
        results = evaluate_methods_for_model(
            model_name,
            model_root,
            thresholds_by_model[model_name],
            js_by_model[model_name],
        )
        all_results[model_name] = {
            "checkpoint": MODEL_PATHS[model_name],
            "ours_thresholds": thresholds_by_model[model_name],
            "js_calibration": js_by_model[model_name],
            "results": results,
        }

    print(
        f"{'Model':<14}{'Attack':<10}{'Acc':>9}"
        f"{'OURS':>11}{'JS':>11}{'OURS+JS':>11}"
        f"{'OURS wrong':>15}{'JS wrong':>13}{'+JS wrong':>13}"
    )
    print("-" * 110)

    for model_name, model_result in all_results.items():
        for attack_name, r in model_result["results"].items():
            wt = r["ours"]["wrong_total"]
            print(
                f"{model_name:<14}{attack_name:<10}"
                f"{r['accuracy_percent']:>8.2f}%"
                f"{r['ours']['flag_rate_percent']:>10.2f}%"
                f"{r['js']['flag_rate_percent']:>10.2f}%"
                f"{r['ours_plus_js']['flag_rate_percent']:>10.2f}%"
                f"{r['ours']['wrong_caught']:>7}/{wt:<7}"
                f"{r['js']['wrong_caught']:>6}/{wt:<6}"
                f"{r['ours_plus_js']['wrong_caught']:>6}/{wt:<6}"
            )
        print("-" * 110)

    save_summary(all_results)

    with open(os.path.join(OUT_ROOT, "evaluation_stats.json"), "w") as f:
        json.dump(evaluation_stats, f, indent=2)

    print("\nDONE.")
    print("Old attack PNGs were read only; no old result directory was modified.")
    print(f"All fresh outputs are under: {os.path.abspath(OUT_ROOT)}")


if __name__ == "__main__":
    main()