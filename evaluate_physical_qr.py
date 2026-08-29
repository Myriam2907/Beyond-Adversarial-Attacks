import os
import re
import csv
from pathlib import Path

import torch
import torch.nn as nn

from PIL import Image, ImageOps
from torchvision import transforms, models




CLEAN_ROOT = "/home/Traffic_Signs_2/Clean Dataset"
ATTACK_ROOT = "/home/Traffic_Signs_2/attacked"
MODEL_DIR = "/home/Traffic_Signs_2/physical_models"
OUTPUT_DIR = "/home/Traffic_Signs_2/physical_attack_results_all70"

os.makedirs(OUTPUT_DIR, exist_ok=True)




DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 100)
print("PHYSICAL QR ATTACK EVALUATION - ALL 70 PAIRS")
print("=" * 100)
print("Device:", DEVICE)




IMAGE_SIZE = 224

class PadToSquare:
    def __call__(self, img):
        w, h = img.size
        side = max(w, h)

        left = (side - w) // 2
        right = side - w - left
        top = (side - h) // 2
        bottom = side - h - top

        return ImageOps.expand(
            img,
            border=(left, top, right, bottom),
            fill=0
        )


transform = transforms.Compose([
    PadToSquare(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])




MODEL_PATHS = {
    "MobileNetV3-Large":
        os.path.join(MODEL_DIR, "mobilenetv3_comma_clean_best.pth"),

    "ConvNeXt-Tiny":
        os.path.join(MODEL_DIR, "convnext_tiny_comma_clean_best.pth"),

    "EfficientNetV2-S":
        os.path.join(MODEL_DIR, "efficientnet_v2_s_comma_clean_best.pth")
}



def build_model(model_name, num_classes):

    if model_name == "MobileNetV3-Large":
        model = models.mobilenet_v3_large(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

    elif model_name == "ConvNeXt-Tiny":
        model = models.convnext_tiny(weights=None)
        in_features = model.classifier[2].in_features
        model.classifier[2] = nn.Linear(in_features, num_classes)

    elif model_name == "EfficientNetV2-S":
        model = models.efficientnet_v2_s(weights=None)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {model_name}")

    return model




def get_image_index(path):
    filename = os.path.basename(path)

    match = re.search(
        r"_(\d+)\.(jpg|jpeg|png|bmp)$",
        filename,
        flags=re.IGNORECASE
    )

    if match is None:
        raise ValueError(f"Cannot extract image index from: {filename}")

    return int(match.group(1))


def find_attacked_image(class_name, clean_path):
    index = get_image_index(clean_path)
    attack_class_dir = os.path.join(ATTACK_ROOT, class_name)

    if not os.path.isdir(attack_class_dir):
        raise FileNotFoundError(f"Missing folder: {attack_class_dir}")

    candidates = []

    for filename in os.listdir(attack_class_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        full_path = os.path.join(attack_class_dir, filename)

        try:
            idx = get_image_index(full_path)
        except ValueError:
            continue

        if idx == index:
            candidates.append(full_path)

    if len(candidates) == 0:
        raise FileNotFoundError(
            f"No attacked counterpart found for class={class_name}, index={index}"
        )

    if len(candidates) > 1:
        raise RuntimeError(
            f"Multiple attacked counterparts found for class={class_name}, index={index}: {candidates}"
        )

    return candidates[0]


def predict_image(model, image_path, classes):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(image)
        probs = torch.softmax(logits, dim=1)
        confidence, pred = torch.max(probs, dim=1)

    pred_id = pred.item()
    return pred_id, classes[pred_id], confidence.item()


def collect_all_clean_samples(classes, class_to_idx):
    samples = []

    for class_name in classes:
        class_dir = os.path.join(CLEAN_ROOT, class_name)

        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Missing clean class folder: {class_dir}")

        files = sorted([
            os.path.join(class_dir, f)
            for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".bmp"))
        ])

        for path in files:
            samples.append((path, class_to_idx[class_name]))

    return samples




def evaluate_model(model_name, checkpoint_path):

    print("\n\n" + "#" * 100)
    print(model_name)
    print("#" * 100)

    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )

    classes = checkpoint["classes"]
    class_to_idx = checkpoint["class_to_idx"]
    num_classes = len(classes)

    model = build_model(model_name, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(DEVICE)
    model.eval()

    all_samples = collect_all_clean_samples(classes, class_to_idx)

    print(f"\nTotal clean-attacked pairs: {len(all_samples)}")

    total = 0
    clean_correct = 0
    attacked_correct = 0
    eligible_attacks = 0
    successful_attacks = 0

    per_class = {
        class_name: {
            "total": 0,
            "clean_correct": 0,
            "attacked_correct": 0,
            "eligible": 0,
            "successful": 0
        }
        for class_name in classes
    }

    rows = []

    for clean_path, true_id in all_samples:

        true_class = classes[true_id]
        attacked_path = find_attacked_image(true_class, clean_path)

        clean_pred_id, clean_pred_class, clean_conf = predict_image(model, clean_path, classes)
        attack_pred_id, attack_pred_class, attack_conf = predict_image(model, attacked_path, classes)

        is_clean_correct = (clean_pred_id == true_id)
        is_attack_correct = (attack_pred_id == true_id)

        attack_eligible = is_clean_correct
        attack_success = is_clean_correct and (not is_attack_correct)

        total += 1
        per_class[true_class]["total"] += 1

        if is_clean_correct:
            clean_correct += 1
            per_class[true_class]["clean_correct"] += 1

        if is_attack_correct:
            attacked_correct += 1
            per_class[true_class]["attacked_correct"] += 1

        if attack_eligible:
            eligible_attacks += 1
            per_class[true_class]["eligible"] += 1

        if attack_success:
            successful_attacks += 1
            per_class[true_class]["successful"] += 1

        rows.append({
            "model": model_name,
            "true_class": true_class,
            "sample_index": get_image_index(clean_path),
            "clean_path": clean_path,
            "attacked_path": attacked_path,
            "clean_prediction": clean_pred_class,
            "clean_confidence": clean_conf,
            "clean_correct": is_clean_correct,
            "attacked_prediction": attack_pred_class,
            "attacked_confidence": attack_conf,
            "attacked_correct": is_attack_correct,
            "attack_success": attack_success
        })

    clean_accuracy = 100.0 * clean_correct / total
    attacked_accuracy = 100.0 * attacked_correct / total
    accuracy_drop = clean_accuracy - attacked_accuracy
    asr = 100.0 * successful_attacks / eligible_attacks if eligible_attacks > 0 else 0.0

    print("\n" + "=" * 100)
    print(f"{model_name} — ALL 70 PHYSICAL RESULTS")
    print("=" * 100)
    print(f"Clean accuracy     : {clean_accuracy:.2f}% ({clean_correct}/{total})")
    print(f"Attacked accuracy  : {attacked_accuracy:.2f}% ({attacked_correct}/{total})")
    print(f"Accuracy drop      : {accuracy_drop:.2f} percentage points")
    print(f"Eligible attacks   : {eligible_attacks}")
    print(f"Successful attacks : {successful_attacks}")
    print(f"Attack Success Rate: {asr:.2f}%")

    print("\nPer-class results:")
    print("-" * 100)
    print(f"{'Class':15s} {'Clean':>10s} {'QR':>10s} {'ASR':>10s}")
    print("-" * 100)

    for class_name in classes:
        stats = per_class[class_name]
        n = stats["total"]

        clean_acc = 100.0 * stats["clean_correct"] / n if n > 0 else 0.0
        qr_acc = 100.0 * stats["attacked_correct"] / n if n > 0 else 0.0
        class_asr = 100.0 * stats["successful"] / stats["eligible"] if stats["eligible"] > 0 else 0.0

        print(f"{class_name:15s} {clean_acc:9.2f}% {qr_acc:9.2f}% {class_asr:9.2f}%")

    safe_model_name = model_name.lower().replace("-", "_").replace(" ", "_")
    csv_path = os.path.join(OUTPUT_DIR, f"{safe_model_name}_all70_results.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nSaved CSV:")
    print(csv_path)

    return {
        "model": model_name,
        "total": total,
        "clean_accuracy": clean_accuracy,
        "attacked_accuracy": attacked_accuracy,
        "accuracy_drop": accuracy_drop,
        "eligible_attacks": eligible_attacks,
        "successful_attacks": successful_attacks,
        "asr": asr
    }




summaries = []

for model_name, checkpoint_path in MODEL_PATHS.items():
    if not os.path.exists(checkpoint_path):
        print(f"[ERROR] Missing checkpoint: {checkpoint_path}")
        continue

    summary = evaluate_model(model_name, checkpoint_path)
    summaries.append(summary)

print("\n\n" + "=" * 100)
print("FINAL ALL-70 COMPARISON")
print("=" * 100)
print(f"{'Model':22s} {'Clean Acc':>12s} {'QR Acc':>12s} {'Drop':>12s} {'ASR':>12s}")
print("-" * 100)

for s in summaries:
    print(
        f"{s['model']:22s} "
        f"{s['clean_accuracy']:11.2f}% "
        f"{s['attacked_accuracy']:11.2f}% "
        f"{s['accuracy_drop']:11.2f} "
        f"{s['asr']:11.2f}%"
    )

if len(summaries) > 0:
    summary_path = os.path.join(OUTPUT_DIR, "physical_qr_summary_all70.csv")

    with open(summary_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summaries[0].keys())
        writer.writeheader()
        writer.writerows(summaries)

    print("\nSummary saved:")
    print(summary_path)

print("\nDone.")