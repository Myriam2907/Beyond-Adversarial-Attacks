import os
import json
import time
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from tqdm import tqdm



DATA_DIR = "./data"
OUT_ROOT = "./gtsrb_repeat/models"

IMG_SIZE = 224
NUM_CLASSES = 43


EPOCHS = 20
BATCH_SIZE = 64
NUM_WORKERS = 4
LR_BACKBONE = 1e-4
LR_HEAD = 5e-4
WEIGHT_DECAY = 1e-4
VAL_FRAC = 0.10
PATIENCE = 5
MIN_DELTA = 1e-4
SEED = 123


MODEL_NAMES = [
    "mobilenet",
    "convnext",
    "efficientnet",
]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = DEVICE.type == "cuda"
USE_AMP = DEVICE.type == "cuda"


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]



def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def build_transforms():
    """
    Train augmentation is intentionally mild because traffic-sign classes
    can be sensitive to aggressive geometric transforms.
    """
    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    eval_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    return train_tfms, eval_tfms


def stratified_train_val_indices(dataset, val_frac=0.10, seed=123):
    """
    Make a class-stratified train/validation split from the official GTSRB
    training split so every class remains represented in both partitions.
    """
    labels = []
    for _, y in dataset:
        labels.append(int(y))

    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    train_indices = []
    val_indices = []

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)

        n_val = max(1, int(round(len(cls_idx) * val_frac)))
        
        n_val = min(n_val, max(1, len(cls_idx) - 1))

        val_indices.extend(cls_idx[:n_val].tolist())
        train_indices.extend(cls_idx[n_val:].tolist())

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return train_indices, val_indices


def build_dataloaders():
    train_tfms, eval_tfms = build_transforms()

    
    split_source = datasets.GTSRB(
        root=DATA_DIR,
        split="train",
        download=True,
        transform=None,
    )

    train_idx, val_idx = stratified_train_val_indices(
        split_source,
        val_frac=VAL_FRAC,
        seed=SEED,
    )

    
    train_full = datasets.GTSRB(
        root=DATA_DIR,
        split="train",
        download=True,
        transform=train_tfms,
    )

    val_full = datasets.GTSRB(
        root=DATA_DIR,
        split="train",
        download=True,
        transform=eval_tfms,
    )

    test_ds = datasets.GTSRB(
        root=DATA_DIR,
        split="test",
        download=True,
        transform=eval_tfms,
    )

    train_ds = Subset(train_full, train_idx)
    val_ds = Subset(val_full, val_idx)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )

    split_info = {
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "test_samples": len(test_ds),
        "val_fraction": VAL_FRAC,
        "seed": SEED,
    }

    return train_loader, val_loader, test_loader, split_info



def build_model(model_name: str):
    """
    Build one of the three ImageNet-pretrained classifiers and replace the
    final classifier with a 43-class GTSRB head.
    """
    model_name = model_name.lower()

    if model_name == "mobilenet":
        weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
        head_params = list(model.classifier.parameters())

    elif model_name == "convnext":
        weights = models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        model = models.convnext_tiny(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
        head_params = list(model.classifier.parameters())

    elif model_name == "efficientnet":
        weights = models.EfficientNet_V2_S_Weights.IMAGENET1K_V1
        model = models.efficientnet_v2_s(weights=weights)
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
        head_params = list(model.classifier.parameters())

    else:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            "Choose mobilenet, convnext, or efficientnet."
        )

    return model, head_params


def make_optimizer(model, head_params):
    """
    Use a smaller LR for pretrained backbone parameters and a larger LR for
    the newly initialized GTSRB classifier head.
    """
    head_param_ids = {id(p) for p in head_params}
    backbone_params = [
        p for p in model.parameters()
        if id(p) not in head_param_ids
    ]

    optimizer = torch.optim.AdamW(
        [
            {
                "params": backbone_params,
                "lr": LR_BACKBONE,
            },
            {
                "params": head_params,
                "lr": LR_HEAD,
            },
        ],
        weight_decay=WEIGHT_DECAY,
    )

    return optimizer



def run_epoch(model, loader, criterion, optimizer=None, scaler=None):
    is_train = optimizer is not None

    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    pbar = tqdm(loader, leave=False)

    for images, labels in pbar:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            if USE_AMP:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(images)
                    loss = criterion(logits, labels)
            else:
                logits = model(images)
                loss = criterion(logits, labels)

            if is_train:
                if scaler is not None and USE_AMP:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()

        preds = logits.argmax(dim=1)
        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * total_correct / max(1, total_samples):.2f}%",
        )

    avg_loss = total_loss / max(1, total_samples)
    accuracy = 100.0 * total_correct / max(1, total_samples)

    return avg_loss, accuracy


@torch.no_grad()
def evaluate_test(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_preds = []
    all_labels = []
    all_conf = []

    if DEVICE.type == "cuda":
        # Warm-up before timing.
        warm = torch.randn(8, 3, IMG_SIZE, IMG_SIZE, device=DEVICE)
        for _ in range(10):
            _ = model(warm)
        torch.cuda.synchronize()

    total_inference_seconds = 0.0

    for images, labels in tqdm(loader, desc="Test", leave=False):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        logits = model(images)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        total_inference_seconds += t1 - t0

        loss = criterion(logits, labels)
        probs = torch.softmax(logits, dim=1)
        conf, preds = probs.max(dim=1)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (preds == labels).sum().item()
        total_samples += batch_size

        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        all_conf.append(conf.cpu().numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    confidence = np.concatenate(all_conf)

    avg_loss = total_loss / max(1, total_samples)
    accuracy = 100.0 * total_correct / max(1, total_samples)
    avg_ms_per_image = (
        total_inference_seconds * 1000.0 / max(1, total_samples)
    )

    return {
        "test_loss": float(avg_loss),
        "test_accuracy_percent": float(accuracy),
        "avg_inference_time_ms_per_image": float(avg_ms_per_image),
        "n_test_samples": int(total_samples),
        "mean_confidence": float(confidence.mean()),
        "median_confidence": float(np.median(confidence)),
        "predictions": preds,
        "labels": labels,
        "confidence": confidence,
    }



def train_one_model(model_name, train_loader, val_loader, test_loader, split_info):
    model_name = model_name.lower()
    model_dir = Path(OUT_ROOT) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = model_dir / f"{model_name}_gtsrb_best.pth"
    history_path = model_dir / "training_history.json"
    result_path = model_dir / "final_test_results.json"

    print("\n" + "=" * 80)
    print(f"TRAINING MODEL: {model_name.upper()}")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"Output directory: {model_dir.resolve()}")
    print(f"Checkpoint: {checkpoint_path.resolve()}")

    model, head_params = build_model(model_name)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, head_params)

    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
    )

    
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=USE_AMP,
    )

    best_val_acc = -1.0
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    start_training = time.time()

    for epoch in range(1, EPOCHS + 1):
        epoch_start = time.time()

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print("-" * 80)

        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
        )

        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            scaler=None,
        )

        scheduler.step()

        lr_backbone = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]
        epoch_seconds = time.time() - epoch_start

        epoch_info = {
            "epoch": epoch,
            "train_loss": float(train_loss),
            "train_accuracy_percent": float(train_acc),
            "val_loss": float(val_loss),
            "val_accuracy_percent": float(val_acc),
            "lr_backbone": float(lr_backbone),
            "lr_head": float(lr_head),
            "epoch_seconds": float(epoch_seconds),
        }
        history.append(epoch_info)

        print(
            f"Train: loss={train_loss:.5f}, acc={train_acc:.3f}% | "
            f"Val: loss={val_loss:.5f}, acc={val_acc:.3f}%"
        )

        
        improved = (
            val_acc > best_val_acc + MIN_DELTA
            or (
                abs(val_acc - best_val_acc) <= MIN_DELTA
                and val_loss < best_val_loss
            )
        )

        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            epochs_without_improvement = 0

            
            torch.save(model.state_dict(), checkpoint_path)

            file_size_mb = checkpoint_path.stat().st_size / (1024 ** 2)
            print(
                f"Saved NEW best checkpoint: {checkpoint_path} "
                f"({file_size_mb:.2f} MB)"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"No validation improvement "
                f"({epochs_without_improvement}/{PATIENCE})"
            )


        with open(history_path, "w") as f:
            json.dump(
                {
                    "model": model_name,
                    "config": training_config_dict(),
                    "split": split_info,
                    "best_val_accuracy_percent": float(best_val_acc),
                    "best_val_loss": float(best_val_loss),
                    "epochs": history,
                },
                f,
                indent=2,
            )

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping after epoch {epoch}.")
            break

    total_training_seconds = time.time() - start_training

    if not checkpoint_path.exists():
        raise RuntimeError(
            f"Training ended but checkpoint was not created: {checkpoint_path}"
        )

  
    print("\nVerifying saved checkpoint...")
    saved_state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(saved_state, dict) or len(saved_state) == 0:
        raise RuntimeError(
            f"Invalid checkpoint produced for {model_name}: {checkpoint_path}"
        )

    checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 ** 2)
    print(
        f"Checkpoint verification OK: {len(saved_state)} tensors, "
        f"{checkpoint_size_mb:.2f} MB"
    )

    
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=DEVICE)
    )
    model.eval()

    print("\nEvaluating BEST checkpoint on official GTSRB test split...")
    test_results = evaluate_test(model, test_loader, criterion)

    
    np.save(model_dir / "test_predictions.npy", test_results.pop("predictions"))
    np.save(model_dir / "test_labels.npy", test_results.pop("labels"))
    np.save(model_dir / "test_confidence.npy", test_results.pop("confidence"))

    final_results = {
        "model": model_name,
        "checkpoint": str(checkpoint_path),
        "checkpoint_size_mb": float(checkpoint_size_mb),
        "best_val_accuracy_percent": float(best_val_acc),
        "best_val_loss": float(best_val_loss),
        "total_training_seconds": float(total_training_seconds),
        "split": split_info,
        "config": training_config_dict(),
        **test_results,
    }

    with open(result_path, "w") as f:
        json.dump(final_results, f, indent=2)

    print("\nFINAL RESULT")
    print(f"Model: {model_name}")
    print(f"Best validation accuracy: {best_val_acc:.3f}%")
    print(
        f"Test accuracy: "
        f"{final_results['test_accuracy_percent']:.3f}%"
    )
    print(
        f"Inference: "
        f"{final_results['avg_inference_time_ms_per_image']:.4f} ms/image"
    )
    print(f"Checkpoint: {checkpoint_path.resolve()}")
    print(f"Results: {result_path.resolve()}")

   
    del model
    del optimizer
    del scheduler
    del scaler
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return final_results



def training_config_dict():
    return {
        "img_size": IMG_SIZE,
        "num_classes": NUM_CLASSES,
        "epochs_max": EPOCHS,
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "lr_backbone": LR_BACKBONE,
        "lr_head": LR_HEAD,
        "weight_decay": WEIGHT_DECAY,
        "val_fraction": VAL_FRAC,
        "early_stopping_patience": PATIENCE,
        "min_delta": MIN_DELTA,
        "seed": SEED,
        "imagenet_mean": MEAN,
        "imagenet_std": STD,
        "amp": USE_AMP,
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "pretraining": "ImageNet",
    }



def main():
    set_seed(SEED)

    Path(OUT_ROOT).mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("GTSRB THREE-MODEL TRAINING")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"New output root: {Path(OUT_ROOT).resolve()}")
    print("Old files/results are NOT written by this script.")
    print(f"Models: {MODEL_NAMES}")

    print("\nPreparing GTSRB train/validation/test loaders...")
    train_loader, val_loader, test_loader, split_info = build_dataloaders()

    print(json.dumps(split_info, indent=2))

    all_results = {}

    for model_name in MODEL_NAMES:
        result = train_one_model(
            model_name=model_name,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            split_info=split_info,
        )
        all_results[model_name] = result

    summary_path = Path(OUT_ROOT).parent / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 100)
    print("ALL THREE MODELS FINISHED")
    print("=" * 100)
    print(
        f"{'Model':<16} "
        f"{'Best Val Acc':>14} "
        f"{'Test Acc':>12} "
        f"{'Checkpoint MB':>15} "
        f"{'ms/image':>12}"
    )
    print("-" * 100)

    for model_name in MODEL_NAMES:
        r = all_results[model_name]
        print(
            f"{model_name:<16} "
            f"{r['best_val_accuracy_percent']:>13.3f}% "
            f"{r['test_accuracy_percent']:>11.3f}% "
            f"{r['checkpoint_size_mb']:>15.2f} "
            f"{r['avg_inference_time_ms_per_image']:>12.4f}"
        )

    print("-" * 100)
    print(f"Summary saved to: {summary_path.resolve()}")
    print("\nFresh checkpoints:")
    for model_name in MODEL_NAMES:
        p = Path(OUT_ROOT) / model_name / f"{model_name}_gtsrb_best.pth"
        print(f"  {model_name:12s}: {p.resolve()}")

    print("\nNEXT STEP after these models finish:")
    print("Use these three NEW checkpoints to recompute each model's clean OURS signals/thresholds,")
    print("then evaluate the EXISTING attacked_pngs and compute OURS, JS, and OURS+JS.")


if __name__ == "__main__":
    main()