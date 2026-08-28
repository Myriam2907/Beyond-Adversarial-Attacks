import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from tqdm import tqdm



TRAIN_DIR = "/home/Downloads/dataset/mapillary_cropped/train"
VAL_DIR   = "/home/Downloads/dataset/mapillary_cropped/val"

OUT_DIR = "./mapillary_baseline_results"

IMG_SIZE = 224
BATCH_SIZE = 128
EPOCHS = 8
LR = 3e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class FixedClassImageFolder(datasets.ImageFolder):

    def __init__(self, root, class_to_idx, transform=None):

        self.fixed_class_to_idx = class_to_idx

        super().__init__(
            root=root,
            transform=transform,
            allow_empty=True
        )

    def find_classes(self, directory):

        classes = list(self.fixed_class_to_idx.keys())

        return classes, self.fixed_class_to_idx



def get_dataloaders():

    train_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),

        transforms.RandomRotation(12),

        transforms.ColorJitter(
            brightness=0.3,
            contrast=0.3,
            saturation=0.3
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    val_tfms = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),

        transforms.ToTensor(),

        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        ),
    ])

    
    train_ds = datasets.ImageFolder(
        TRAIN_DIR,
        transform=train_tfms
    )

    val_ds = FixedClassImageFolder(
        VAL_DIR,
        class_to_idx=train_ds.class_to_idx,
        transform=val_tfms
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(
            NUM_WORKERS > 0 and DEVICE.type == "cuda"
        ),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(
            NUM_WORKERS > 0 and DEVICE.type == "cuda"
        ),
    )

    return train_loader, val_loader, train_ds, val_ds



def build_model(num_classes):

    model = models.mobilenet_v3_large(
        weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
    )

    in_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        in_features,
        num_classes
    )

    return model



def train_one_epoch(model, loader, optimizer):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Train", leave=False):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)

        loss = F.cross_entropy(
            logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item() * images.size(0)

        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()

        total += images.size(0)

    return total_loss / total, correct / total



@torch.no_grad()
def eval_accuracy(model, loader):

    model.eval()

    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Eval", leave=False):

        images = images.to(
            DEVICE,
            non_blocking=True
        )

        labels = labels.to(
            DEVICE,
            non_blocking=True
        )

        logits = model(images)

        preds = logits.argmax(dim=1)

        correct += (preds == labels).sum().item()

        total += images.size(0)

    return correct / total



def main():

    print(f"Using device: {DEVICE}")

    os.makedirs(
        OUT_DIR,
        exist_ok=True
    )

    train_loader, val_loader, train_ds, val_ds = get_dataloaders()

    num_classes = len(train_ds.classes)

    print(f"Train samples: {len(train_ds):,}")
    print(f"Val samples  : {len(val_ds):,}")
    print(f"Classes      : {num_classes}")

    print("\nChecking class mapping consistency...")

    print(
        "Same mapping:",
        train_ds.class_to_idx == val_ds.class_to_idx
    )

    with open(
        os.path.join(OUT_DIR, "class_to_idx.json"),
        "w"
    ) as f:

        json.dump(
            train_ds.class_to_idx,
            f,
            indent=2
        )

    with open(
        os.path.join(OUT_DIR, "classes.json"),
        "w"
    ) as f:

        json.dump(
            train_ds.classes,
            f,
            indent=2
        )

    model = build_model(
        num_classes
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_acc = -1.0

    best_path = os.path.join(
        OUT_DIR,
        "mobilenetv3_mapillary_best.pth"
    )

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):

        tr_loss, tr_acc = train_one_epoch(
            model,
            train_loader,
            optimizer
        )

        val_acc = eval_accuracy(
            model,
            val_loader
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"loss {tr_loss:.4f} | "
            f"train acc {tr_acc*100:.2f}% | "
            f"val acc {val_acc*100:.2f}%"
        )

        if val_acc > best_acc:

            best_acc = val_acc

            torch.save(
                model.state_dict(),
                best_path
            )

            print(f"Saved best model: {best_path}")

    print(
        f"\nBest clean val accuracy: "
        f"{best_acc*100:.3f}%"
    )

    print(f"Saved model: {best_path}")


if __name__ == "__main__":
    main()