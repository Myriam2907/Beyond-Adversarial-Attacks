import os
import random
import copy
from pathlib import Path

import torch
import torch.nn as nn

from PIL import Image, ImageOps

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = "/home/cpsslab/Desktop/myriam/Traffic_Signs_2/Clean Dataset/Myriam"

OUTPUT_DIR = "/home/cpsslab/Desktop/myriam/Traffic_Signs_2/physical_models"

SEED = 42

TRAIN_PER_CLASS = 5
VAL_PER_CLASS = 2

BATCH_SIZE = 8

HEAD_EPOCHS = 20
FINETUNE_EPOCHS = 10

HEAD_LR = 1e-3
FINETUNE_LR = 1e-5

IMAGE_SIZE = 224

NUM_WORKERS = 4


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 80)
print("PHYSICAL TRAFFIC SIGN CLASSIFIER")
print("=" * 80)

print("Device:", DEVICE)


# ============================================================
# CLASSES
# ============================================================

classes = sorted([
    d.name
    for d in Path(DATA_ROOT).iterdir()
    if d.is_dir()
])

class_to_idx = {
    class_name: idx
    for idx, class_name in enumerate(classes)
}

num_classes = len(classes)

print("\nClasses:")

for name, idx in class_to_idx.items():
    print(f"{idx:2d}: {name}")

print("\nNumber of classes:", num_classes)


# ============================================================
# SPLIT 5 TRAIN / 2 VALIDATION PER CLASS
# ============================================================

train_samples = []
val_samples = []

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)

for class_name in classes:

    folder = Path(DATA_ROOT) / class_name

    files = sorted([
        p for p in folder.iterdir()
        if p.suffix.lower() in valid_extensions
    ])

    if len(files) < TRAIN_PER_CLASS + VAL_PER_CLASS:
        raise ValueError(
            f"{class_name} has only {len(files)} images. "
            f"Need at least {TRAIN_PER_CLASS + VAL_PER_CLASS}."
        )

    rng = random.Random(
        SEED + class_to_idx[class_name]
    )

    rng.shuffle(files)

    train_files = files[:TRAIN_PER_CLASS]

    val_files = files[
        TRAIN_PER_CLASS:
        TRAIN_PER_CLASS + VAL_PER_CLASS
    ]

    print(
        f"{class_name:15s}: "
        f"{len(train_files)} train | "
        f"{len(val_files)} val"
    )

    for p in train_files:

        train_samples.append(
            (
                str(p),
                class_to_idx[class_name]
            )
        )

    for p in val_files:

        val_samples.append(
            (
                str(p),
                class_to_idx[class_name]
            )
        )


print("\nTotal train images:", len(train_samples))
print("Total val images:", len(val_samples))


# ============================================================
# PAD IMAGE TO SQUARE
# ============================================================

class PadToSquare:

    def __call__(self, img):

        w, h = img.size

        max_side = max(w, h)

        pad_left = (max_side - w) // 2
        pad_right = max_side - w - pad_left

        pad_top = (max_side - h) // 2
        pad_bottom = max_side - h - pad_top

        return ImageOps.expand(
            img,
            border=(
                pad_left,
                pad_top,
                pad_right,
                pad_bottom
            ),
            fill=0
        )


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    PadToSquare(),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomRotation(
        degrees=8
    ),

    transforms.RandomPerspective(
        distortion_scale=0.12,
        p=0.30
    ),

    transforms.ColorJitter(
        brightness=0.20,
        contrast=0.20,
        saturation=0.15
    ),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.04, 0.04),
        scale=(0.95, 1.05)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([

    PadToSquare(),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET
# ============================================================

class TrafficSignDataset(Dataset):

    def __init__(
        self,
        samples,
        transform=None
    ):

        self.samples = samples
        self.transform = transform


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, index):

        path, label = self.samples[index]

        image = Image.open(path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label, path


train_dataset = TrafficSignDataset(
    train_samples,
    transform=train_transform
)

val_dataset = TrafficSignDataset(
    val_samples,
    transform=val_transform
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# MODEL
# ============================================================

weights = models.MobileNet_V3_Large_Weights.DEFAULT

model = models.mobilenet_v3_large(
    weights=weights
)

in_features = model.classifier[3].in_features

model.classifier[3] = nn.Linear(
    in_features,
    num_classes
)

model = model.to(DEVICE)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader):

    model.eval()

    total = 0
    correct = 0
    running_loss = 0.0

    per_class_correct = {
        i: 0 for i in range(num_classes)
    }

    per_class_total = {
        i: 0 for i in range(num_classes)
    }

    with torch.no_grad():

        for images, labels, paths in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            for label, pred in zip(
                labels,
                predictions
            ):

                lid = label.item()

                per_class_total[lid] += 1

                if label.item() == pred.item():

                    per_class_correct[lid] += 1


    loss = (
        running_loss / total
        if total > 0 else 0
    )

    accuracy = (
        100.0 * correct / total
        if total > 0 else 0
    )

    return (
        loss,
        accuracy,
        per_class_correct,
        per_class_total
    )


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_stage(
    model,
    optimizer,
    epochs,
    stage_name
):

    print("\n" + "=" * 80)
    print(stage_name)
    print("=" * 80)

    best_accuracy = -1.0

    best_weights = copy.deepcopy(
        model.state_dict()
    )

    for epoch in range(epochs):

        model.train()

        total = 0
        correct = 0
        running_loss = 0.0

        for images, labels, paths in train_loader:

            images = images.to(
                DEVICE,
                non_blocking=True
            )

            labels = labels.to(
                DEVICE,
                non_blocking=True
            )

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = outputs.argmax(dim=1)

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

        train_loss = running_loss / total

        train_acc = (
            100.0 * correct / total
        )

        val_loss, val_acc, _, _ = evaluate(
            model,
            val_loader
        )

        print(
            f"Epoch {epoch+1:02d}/{epochs:02d} | "
            f"Train loss {train_loss:.4f} | "
            f"Train acc {train_acc:6.2f}% | "
            f"Val loss {val_loss:.4f} | "
            f"Val acc {val_acc:6.2f}%"
        )

        if val_acc > best_accuracy:

            best_accuracy = val_acc

            best_weights = copy.deepcopy(
                model.state_dict()
            )

            print(
                f"  -> New best val accuracy: "
                f"{best_accuracy:.2f}%"
            )

    model.load_state_dict(
        best_weights
    )

    return model, best_accuracy


# ============================================================
# STAGE 1
# TRAIN CLASSIFIER HEAD ONLY
# ============================================================

for param in model.features.parameters():
    param.requires_grad = False

for param in model.classifier.parameters():
    param.requires_grad = True


optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=HEAD_LR,
    weight_decay=1e-4
)


model, stage1_best = train_stage(
    model,
    optimizer,
    HEAD_EPOCHS,
    "STAGE 1 - CLASSIFIER HEAD"
)


# ============================================================
# STAGE 2
# FINE-TUNE LAST BLOCKS
# ============================================================

for param in model.features.parameters():
    param.requires_grad = False

for block in model.features[-3:]:

    for param in block.parameters():

        param.requires_grad = True


for param in model.classifier.parameters():
    param.requires_grad = True


optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=FINETUNE_LR,
    weight_decay=1e-4
)


model, stage2_best = train_stage(
    model,
    optimizer,
    FINETUNE_EPOCHS,
    "STAGE 2 - FINE TUNING"
)


# ============================================================
# FINAL RESULTS
# ============================================================

(
    final_loss,
    final_acc,
    class_correct,
    class_total
) = evaluate(
    model,
    val_loader
)


print("\n" + "=" * 80)
print("FINAL VALIDATION RESULTS")
print("=" * 80)

print(
    f"\nValidation accuracy: "
    f"{final_acc:.2f}%"
)

print("\nPer-class accuracy:")

for idx, name in enumerate(classes):

    total = class_total[idx]

    correct = class_correct[idx]

    accuracy = (
        100.0 * correct / total
        if total > 0 else 0
    )

    print(
        f"{name:15s}: "
        f"{correct}/{total} "
        f"({accuracy:.2f}%)"
    )


# ============================================================
# SAVE MODEL + SPLIT
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "mobilenetv3_comma_clean_best.pth"
)

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "classes":
            classes,

        "class_to_idx":
            class_to_idx,

        "image_size":
            IMAGE_SIZE,

        "validation_accuracy":
            final_acc,

        "train_samples":
            train_samples,

        "val_samples":
            val_samples
    },

    MODEL_PATH
)


print("\nSaved model:")
print(MODEL_PATH)

print("\nDone.")
