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
IMAGE_SIZE = 224
NUM_WORKERS = 4

HEAD_EPOCHS = 20
FINETUNE_EPOCHS = 10

HEAD_LR = 1e-3
FINETUNE_LR = 1e-5


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
    print(f"{idx}: {name}")


# ============================================================
# SAME SPLIT AS BEFORE
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


print("\nTotal train:", len(train_samples))
print("Total val:", len(val_samples))


# ============================================================
# PAD TO SQUARE
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

    transforms.RandomRotation(8),

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

    def __init__(self, samples, transform=None):

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


criterion = nn.CrossEntropyLoss()


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model):

    model.eval()

    total = 0
    correct = 0
    running_loss = 0.0

    class_correct = {
        i: 0 for i in range(num_classes)
    }

    class_total = {
        i: 0 for i in range(num_classes)
    }

    with torch.no_grad():

        for images, labels, paths in val_loader:

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

                class_total[lid] += 1

                if label.item() == pred.item():
                    class_correct[lid] += 1


    loss = running_loss / total

    acc = 100.0 * correct / total

    return (
        loss,
        acc,
        class_correct,
        class_total
    )


# ============================================================
# TRAIN STAGE
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

    best_acc = -1.0
    best_loss = float("inf")

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
            model
        )

        print(
            f"Epoch {epoch+1:02d}/{epochs:02d} | "
            f"Train loss {train_loss:.4f} | "
            f"Train acc {train_acc:6.2f}% | "
            f"Val loss {val_loss:.4f} | "
            f"Val acc {val_acc:6.2f}%"
        )

        # Best accuracy;
        # if same accuracy, keep lower validation loss
        if (
            val_acc > best_acc
            or
            (
                val_acc == best_acc
                and val_loss < best_loss
            )
        ):

            best_acc = val_acc
            best_loss = val_loss

            best_weights = copy.deepcopy(
                model.state_dict()
            )

            print(
                f"  -> New best: "
                f"{best_acc:.2f}% "
                f"(loss {best_loss:.4f})"
            )


    model.load_state_dict(
        best_weights
    )

    return model


# ============================================================
# MODEL BUILDER
# ============================================================

def build_model(model_name):

    if model_name == "convnext":

        weights = (
            models.ConvNeXt_Tiny_Weights.DEFAULT
        )

        model = models.convnext_tiny(
            weights=weights
        )

        in_features = (
            model.classifier[2].in_features
        )

        model.classifier[2] = nn.Linear(
            in_features,
            num_classes
        )

    elif model_name == "efficientnet":

        weights = (
            models.EfficientNet_V2_S_Weights.DEFAULT
        )

        model = models.efficientnet_v2_s(
            weights=weights
        )

        in_features = (
            model.classifier[1].in_features
        )

        model.classifier[1] = nn.Linear(
            in_features,
            num_classes
        )

    else:

        raise ValueError(
            f"Unknown model: {model_name}"
        )

    return model.to(DEVICE)


# ============================================================
# FREEZE / UNFREEZE
# ============================================================

def freeze_backbone(model, model_name):

    if model_name == "convnext":

        for p in model.features.parameters():
            p.requires_grad = False

        for p in model.classifier.parameters():
            p.requires_grad = True


    elif model_name == "efficientnet":

        for p in model.features.parameters():
            p.requires_grad = False

        for p in model.classifier.parameters():
            p.requires_grad = True


def unfreeze_last_blocks(
    model,
    model_name
):

    # first freeze everything
    for p in model.parameters():
        p.requires_grad = False

    if model_name == "convnext":

        # fine-tune final ConvNeXt stage
        for block in model.features[-2:]:

            for p in block.parameters():
                p.requires_grad = True

        for p in model.classifier.parameters():
            p.requires_grad = True


    elif model_name == "efficientnet":

        # fine-tune final EfficientNet blocks
        for block in model.features[-2:]:

            for p in block.parameters():
                p.requires_grad = True

        for p in model.classifier.parameters():
            p.requires_grad = True


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def run_model(model_name):

    print("\n\n")
    print("#" * 90)
    print(f"TRAINING MODEL: {model_name.upper()}")
    print("#" * 90)

    model = build_model(
        model_name
    )

    # --------------------------------------------------------
    # STAGE 1
    # --------------------------------------------------------

    freeze_backbone(
        model,
        model_name
    )

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=HEAD_LR,
        weight_decay=1e-4
    )

    model = train_stage(
        model,
        optimizer,
        HEAD_EPOCHS,
        f"{model_name.upper()} - STAGE 1"
    )


    # --------------------------------------------------------
    # STAGE 2
    # --------------------------------------------------------

    unfreeze_last_blocks(
        model,
        model_name
    )

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=FINETUNE_LR,
        weight_decay=1e-4
    )

    model = train_stage(
        model,
        optimizer,
        FINETUNE_EPOCHS,
        f"{model_name.upper()} - STAGE 2"
    )


    # --------------------------------------------------------
    # FINAL RESULTS
    # --------------------------------------------------------

    (
        val_loss,
        val_acc,
        class_correct,
        class_total
    ) = evaluate(
        model
    )


    print("\n" + "=" * 80)

    print(
        f"{model_name.upper()} FINAL RESULTS"
    )

    print("=" * 80)

    print(
        f"Validation accuracy: "
        f"{val_acc:.2f}%"
    )

    print(
        f"Validation loss: "
        f"{val_loss:.4f}"
    )

    print("\nPer-class:")

    for idx, name in enumerate(classes):

        total = class_total[idx]

        correct = class_correct[idx]

        acc = (
            100.0 * correct / total
            if total > 0
            else 0
        )

        print(
            f"{name:15s}: "
            f"{correct}/{total} "
            f"({acc:.2f}%)"
        )


    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    if model_name == "convnext":

        filename = (
            "convnext_tiny_comma_clean_best.pth"
        )

    else:

        filename = (
            "efficientnet_v2_s_comma_clean_best.pth"
        )

    save_path = os.path.join(
        OUTPUT_DIR,
        filename
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
                val_acc,

            "validation_loss":
                val_loss,

            "train_samples":
                train_samples,

            "val_samples":
                val_samples,

            "model_name":
                model_name
        },

        save_path
    )

    print("\nSaved:")
    print(save_path)


# ============================================================
# RUN BOTH
# ============================================================

run_model("convnext")

run_model("efficientnet")

print("\n\nALL DONE.")
