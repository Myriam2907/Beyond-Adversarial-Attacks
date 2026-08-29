
import os
import re
import time
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm



NUM_CLASSES = 43
IMG_SIZE = 224
BATCH_SIZE = 128
NUM_WORKERS = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


YIELD_ID = 13
STOP_ID = 14


MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]



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
            raise RuntimeError(f"Filename must end with _y<label>.png, got: {os.path.basename(p)}")
        y = int(m.group(1))

        img = Image.open(p).convert("RGB")
        if self.transform:
            img = self.transform(img)

        
        return img, y, p



def build_model():
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
    return model



def energy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    
    return -torch.logsumexp(logits, dim=1)


def js_resize_perturb(x: torch.Tensor) -> torch.Tensor:
    
    x_small = F.interpolate(
        x, size=(208, 208), mode="bilinear", align_corners=False, antialias=True
    )
    x_restore = F.interpolate(
        x_small, size=(224, 224), mode="bilinear", align_corners=False, antialias=True
    )
    return x_restore


def js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    m = 0.5 * (p + q)
    kl_pm = torch.sum(p * (torch.log(p) - torch.log(m)), dim=1)
    kl_qm = torch.sum(q * (torch.log(q) - torch.log(m)), dim=1)
    return 0.5 * (kl_pm + kl_qm)



def perturb1(x: torch.Tensor) -> torch.Tensor:
    x2 = TF.gaussian_blur(x, kernel_size=[3, 3], sigma=[0.1, 0.6])
    x2 = x2 + 0.03
    return x2.clamp(-5, 5)

def perturb2(x: torch.Tensor) -> torch.Tensor:
    x3 = TF.gaussian_blur(x, kernel_size=[5, 5], sigma=[0.2, 0.9])
    x3 = x3 - 0.02
    return x3.clamp(-5, 5)


@torch.no_grad()
def eval_folder(model, loader, out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    
    final_linear = model.classifier[-1]
    captured = {"emb": None}

    def hook_fn(module, inp, out):
        captured["emb"] = inp[0].detach()

    hook = final_linear.register_forward_hook(hook_fn)
    softmax = nn.Softmax(dim=1)

    
    logits_list, conf_list, energy_list, emb_list = [], [], [], []
    js_list = []
    pred_list, label_list, time_ms_list = [], [], []

    
    paths_list = []

    
    change2_list, confdrop2_list, logitdiff2_list = [], [], []

    
    change3_list, maxconfdrop3_list, maxlogitdiff3_list = [], [], []

    if DEVICE.type == "cuda":
        warm = torch.randn(8, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
        for _ in range(10):
            _ = model(warm)
        torch.cuda.synchronize()

    
    for images, labels, paths in tqdm(loader, desc=f"Eval {os.path.basename(out_dir)}"):
       
        paths_list.extend(list(paths))

        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits1 = model(images)
        emb1 = captured["emb"]
        if emb1 is None:
            hook.remove()
            raise RuntimeError("Embedding hook failed.")

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        per_img_ms = ((t1 - t0) * 1000.0) / images.size(0)

        probs1 = softmax(logits1)
        conf1, pred1 = probs1.max(dim=1)
        en1 = energy_from_logits(logits1)

       
        images_js = js_resize_perturb(images)
        logits_js = model(images_js)
        probs_js = softmax(logits_js)
        js_score = js_divergence(probs1, probs_js)

       
        images2 = perturb1(images)
        logits2 = model(images2)
        probs2 = softmax(logits2)
        conf2, pred2 = probs2.max(dim=1)

        changed2 = (pred2 != pred1).int()
        conf_drop2 = (conf1 - conf2).clamp(min=0.0)
        l2_diff2 = torch.norm((logits1 - logits2), dim=1)

       
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

            
            crit_changed_any = ((pred2[crit_idx] != pred1[crit_idx]) |
                                (pred3 != pred1[crit_idx])).int()
            changed3[crit_idx] = crit_changed_any

           
            drop13 = (conf1[crit_idx] - conf3).clamp(min=0.0)
            max_conf_drop3[crit_idx] = torch.max(conf_drop2[crit_idx], drop13)

            l2_diff13 = torch.norm((logits1[crit_idx] - logits3), dim=1)
            max_l2_diff3[crit_idx] = torch.max(l2_diff2[crit_idx], l2_diff13)

        
        logits_list.append(logits1.cpu().numpy())
        conf_list.append(conf1.cpu().numpy())
        energy_list.append(en1.cpu().numpy())
        js_list.append(js_score.cpu().numpy())
        emb_list.append(emb1.cpu().numpy())
        pred_list.append(pred1.cpu().numpy())
        label_list.append(labels.cpu().numpy())
        time_ms_list.append(np.full(images.size(0), per_img_ms, dtype=np.float32))

        change2_list.append(changed2.cpu().numpy())
        confdrop2_list.append(conf_drop2.cpu().numpy())
        logitdiff2_list.append(l2_diff2.cpu().numpy())

        change3_list.append(changed3.cpu().numpy())
        maxconfdrop3_list.append(max_conf_drop3.cpu().numpy())
        maxlogitdiff3_list.append(max_l2_diff3.cpu().numpy())

    hook.remove()

    
    logits_all = np.concatenate(logits_list, axis=0)
    conf_all = np.concatenate(conf_list, axis=0)
    energy_all = np.concatenate(energy_list, axis=0)
    js_all = np.concatenate(js_list, axis=0)
    emb_all = np.concatenate(emb_list, axis=0)
    pred_all = np.concatenate(pred_list, axis=0)
    label_all = np.concatenate(label_list, axis=0)
    time_ms_all = np.concatenate(time_ms_list, axis=0)

    change2_all = np.concatenate(change2_list, axis=0)
    confdrop2_all = np.concatenate(confdrop2_list, axis=0)
    logitdiff2_all = np.concatenate(logitdiff2_list, axis=0)

    change3_all = np.concatenate(change3_list, axis=0)
    maxconfdrop3_all = np.concatenate(maxconfdrop3_list, axis=0)
    maxlogitdiff3_all = np.concatenate(maxlogitdiff3_list, axis=0)

   
    if len(paths_list) != int(label_all.shape[0]):
        raise RuntimeError(
            f"paths_list length mismatch: {len(paths_list)} vs n_samples {int(label_all.shape[0])}. "
            "This should never happen with shuffle=False."
        )

    
    np.save(os.path.join(out_dir, "logits.npy"), logits_all)
    np.save(os.path.join(out_dir, "confidence.npy"), conf_all)
    np.save(os.path.join(out_dir, "energy.npy"), energy_all)
    np.save(os.path.join(out_dir, "js.npy"), js_all)
    np.save(os.path.join(out_dir, "embeddings.npy"), emb_all)
    np.save(os.path.join(out_dir, "pred.npy"), pred_all)
    np.save(os.path.join(out_dir, "label.npy"), label_all)
    np.save(os.path.join(out_dir, "time_ms.npy"), time_ms_all)

    np.save(os.path.join(out_dir, "2pass_changed.npy"), change2_all)
    np.save(os.path.join(out_dir, "2pass_conf_drop.npy"), confdrop2_all)
    np.save(os.path.join(out_dir, "2pass_logit_l2.npy"), logitdiff2_all)

    np.save(os.path.join(out_dir, "3pass_changed_true_stop_yield.npy"), change3_all)
    np.save(os.path.join(out_dir, "3pass_max_conf_drop_true_stop_yield.npy"), maxconfdrop3_all)
    np.save(os.path.join(out_dir, "3pass_max_logit_l2_true_stop_yield.npy"), maxlogitdiff3_all)

   
    np.save(os.path.join(out_dir, "filenames.npy"), np.array(paths_list, dtype=object))

    
    n = int(label_all.shape[0])
    acc = 100.0 * float((pred_all == label_all).mean())
    avg_ms = float(time_ms_all.mean())
    change2_rate = 100.0 * float((change2_all == 1).mean())

   
    crit_mask = (label_all == STOP_ID) | (label_all == YIELD_ID)
    if crit_mask.any():
        crit_change3 = change3_all[crit_mask]  # 0/1
        change3_rate = 100.0 * float((crit_change3 == 1).mean())
    else:
        change3_rate = 0.0

    stats = {
        "accuracy_percent": float(f"{acc:.3f}"),
        "avg_inference_time_ms_per_image": float(f"{avg_ms:.3f}"),
        "2pass_label_change_rate_percent": float(f"{change2_rate:.3f}"),
        "3pass_label_change_rate_percent": float(f"{change3_rate:.3f}"),
        "n_samples": n,
        "js_mean": float(js_all.mean()),
        "js_std": float(js_all.std()),
    }

    with open(os.path.join(out_dir, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\n=== SUMMARY ===")
    print(json.dumps(stats, indent=2))
    print(f"✅ Saved filenames.npy ({n} paths) to: {os.path.join(out_dir, 'filenames.npy')}")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--png_dir", required=True,
                        help="Folder containing PNGs (filenames end with _y<label>.png)")
    parser.add_argument("--out_dir", required=True,
                        help="Output folder to save npy + stats.json")
    parser.add_argument("--model_path", required=True,
                        help="Path to trained model weights (.pth)")
    args = parser.parse_args()

    print("Device:", DEVICE)
    os.makedirs(args.out_dir, exist_ok=True)

    tfm = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    ds = PNGFolderWithLabel(args.png_dir, transform=tfm)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,  
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0 and DEVICE.type == "cuda"),
    )

    model = build_model().to(DEVICE)
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()

    eval_folder(model, loader, args.out_dir)


if __name__ == "__main__":
    main()