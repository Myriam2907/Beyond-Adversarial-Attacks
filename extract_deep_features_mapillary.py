import os
import json
import shutil
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm




CLEAN_DIR = "/mnt/dataset/mapillary_cropped/val"

IMG_SIZE = 224
NUM_WORKERS = 12

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ADV_ATTACKS = ["fgsm", "rfgsm", "pgd", "random_patch"]
ENV_ATTACKS = ["gaussian", "salt_pepper", "light", "fog", "motion_blur"]
ALL_ATTACKS = ["clean"] + ADV_ATTACKS + ENV_ATTACKS


FEATURE_ROOT = "./feature_signal_mapillary"




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
        "adv_root": "./attacks_mobilenet_eps",
        "env_root": "./env_mobilenet",
        "batch_size": 64,
    },
    "convnext": {
        "build": make_convnext,
        "ckpt": "./convnext_mapillary_results/convnext_mapillary_best.pth",
        "class_to_idx": "./convnext_mapillary_results/class_to_idx.json",
        "adv_root": "./attacks_convnext_eps",
        "env_root": "./env_convnext",
        "batch_size": 64,
    },
    "efficientnet": {
        "build": make_efficientnet,
        "ckpt": "./efficientnetv2_mapillary_results/efficientnetv2_mapillary_best.pth",
        "class_to_idx": "./efficientnetv2_mapillary_results/class_to_idx.json",
        "adv_root": "./attacks_efficientnet_eps",
        "env_root": "./env_efficientnet",
        "batch_size": 64,
    },
}




class FixedClassImageFolderWithPaths(datasets.ImageFolder):
    

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

    def __getitem__(self, index):
        image, label = super().__getitem__(index)
        path = self.samples[index][0]
        return image, label, path



def get_image_dir(cfg, condition):
    if condition == "clean":
        return CLEAN_DIR

    if condition in ADV_ATTACKS:
        return os.path.join(
            cfg["adv_root"],
            f"{condition}_png"
        )

    if condition in ENV_ATTACKS:
        return os.path.join(
            cfg["env_root"],
            f"{condition}_png"
        )

    raise ValueError(
        f"Unknown condition: {condition}"
    )


def load_class_map(path):
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"class_to_idx missing: {path}"
        )

    with open(path, "r") as f:
        class_to_idx = json.load(f)

   
    class_to_idx = {
        str(k): int(v)
        for k, v in class_to_idx.items()
    }

    return class_to_idx


def load_model(cfg, num_classes):
    if not os.path.isfile(cfg["ckpt"]):
        raise FileNotFoundError(
            f"checkpoint missing: {cfg['ckpt']}"
        )

    model = cfg["build"](num_classes).to(DEVICE)

    state = torch.load(
        cfg["ckpt"],
        map_location=DEVICE
    )

    
    model.load_state_dict(state)
    model.eval()

    return model


def find_final_linear(model):
    

    last_name = None
    last_module = None

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            last_name = name
            last_module = module

    if last_module is None:
        raise RuntimeError(
            "Could not find a final nn.Linear layer."
        )

    return last_name, last_module




class PenultimateFeatureCapture:
   

    def __init__(self, linear_layer):
        self.features = None

        def hook(module, inputs):
            if len(inputs) != 1:
                raise RuntimeError(
                    "Unexpected final Linear input."
                )

            x = inputs[0]

            
            if x.ndim > 2:
                x = torch.flatten(x, 1)

            self.features = x.detach()

        self.handle = linear_layer.register_forward_pre_hook(
            hook
        )

    def clear(self):
        self.features = None

    def close(self):
        self.handle.remove()




@torch.inference_mode()
def extract_condition(
    model,
    final_linear,
    model_key,
    condition,
    image_dir,
    class_to_idx,
    batch_size,
    save_dtype,
):
    if not os.path.isdir(image_dir):
        print(
            f"[{model_key}/{condition}] SKIP - "
            f"missing directory: {image_dir}"
        )
        return None

    out_dir = os.path.join(
        FEATURE_ROOT,
        model_key,
        condition
    )

    
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    Path(out_dir).mkdir(
        parents=True,
        exist_ok=True
    )

    tfms = transforms.Compose([
        transforms.Resize(
            (IMG_SIZE, IMG_SIZE)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            MEAN,
            STD
        ),
    ])

    ds = FixedClassImageFolderWithPaths(
        image_dir,
        class_to_idx,
        transform=tfms
    )

    if len(ds) == 0:
        print(
            f"[{model_key}/{condition}] "
            f"SKIP - no images"
        )
        return None

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE.type == "cuda"),
        persistent_workers=(NUM_WORKERS > 0),
    )

    capture = PenultimateFeatureCapture(
        final_linear
    )

    all_features = []
    all_labels = []
    all_preds = []
    all_conf = []
    all_paths = []

    try:
        for images, labels, paths in tqdm(
            loader,
            desc=f"{model_key}/{condition}"
        ):
            images = images.to(
                DEVICE,
                non_blocking=True
            )

            capture.clear()

            logits = model(images)

            if capture.features is None:
                raise RuntimeError(
                    "Feature hook did not capture "
                    "the final-linear input."
                )

            features = capture.features

            probs = torch.softmax(
                logits,
                dim=1
            )

            confidence, pred = probs.max(
                dim=1
            )

            if save_dtype == "float16":
                feat_np = (
                    features
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float16)
                )
            else:
                feat_np = (
                    features
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )

            all_features.append(feat_np)

            all_labels.append(
                labels.numpy().astype(
                    np.int64
                )
            )

            all_preds.append(
                pred.cpu().numpy().astype(
                    np.int64
                )
            )

            all_conf.append(
                confidence
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            all_paths.extend(
                list(paths)
            )

    finally:
        capture.close()

    features_np = np.concatenate(
        all_features,
        axis=0
    )

    labels_np = np.concatenate(
        all_labels,
        axis=0
    )

    preds_np = np.concatenate(
        all_preds,
        axis=0
    )

    conf_np = np.concatenate(
        all_conf,
        axis=0
    )

    paths_np = np.asarray(
        all_paths,
        dtype=str
    )

    n = len(ds)

    if not (
        features_np.shape[0]
        == labels_np.shape[0]
        == preds_np.shape[0]
        == conf_np.shape[0]
        == paths_np.shape[0]
        == n
    ):
        raise RuntimeError(
            f"{model_key}/{condition}: "
            "saved arrays are misaligned."
        )

    
    relative_paths = []

    root_abs = os.path.abspath(
        image_dir
    )

    for p in all_paths:
        p_abs = os.path.abspath(p)

        try:
            rel = os.path.relpath(
                p_abs,
                root_abs
            )
        except Exception:
            rel = p_abs

        relative_paths.append(rel)

    relative_paths_np = np.asarray(
        relative_paths,
        dtype=str
    )

    np.save(
        os.path.join(
            out_dir,
            "features.npy"
        ),
        features_np
    )

    np.save(
        os.path.join(
            out_dir,
            "labels.npy"
        ),
        labels_np
    )

    np.save(
        os.path.join(
            out_dir,
            "predictions.npy"
        ),
        preds_np
    )

    np.save(
        os.path.join(
            out_dir,
            "confidence.npy"
        ),
        conf_np
    )

    np.save(
        os.path.join(
            out_dir,
            "filenames.npy"
        ),
        relative_paths_np
    )

    acc = float(
        (preds_np == labels_np).mean()
    )

    metadata = {
        "model": model_key,
        "condition": condition,
        "source_dir": image_dir,
        "num_images": int(n),
        "feature_shape": list(
            features_np.shape
        ),
        "feature_dim": int(
            features_np.shape[1]
        ),
        "saved_feature_dtype": str(
            features_np.dtype
        ),
        "image_size": IMG_SIZE,
        "normalization_mean": list(MEAN),
        "normalization_std": list(STD),
        "accuracy": acc,
        "feature_definition": (
            "Input to the final nn.Linear "
            "classification layer "
            "(penultimate classifier representation)."
        ),
        "output_dir": out_dir,
    }

    with open(
        os.path.join(
            out_dir,
            "metadata.json"
        ),
        "w"
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2
        )

    size_mb = (
        features_np.nbytes
        / (1024 ** 2)
    )

    print(
        f"\n[{model_key}/{condition}] DONE"
    )
    print(
        f"  images       : {n:,}"
    )
    print(
        f"  feature shape: "
        f"{features_np.shape}"
    )
    print(
        f"  feature dtype: "
        f"{features_np.dtype}"
    )
    print(
        f"  feature file : "
        f"{size_mb:.1f} MB"
    )
    print(
        f"  accuracy     : "
        f"{100.0 * acc:.2f}%"
    )
    print(
        f"  output       : {out_dir}"
    )

    return metadata




def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=[
            "all",
            "mobilenet",
            "convnext",
            "efficientnet",
        ],
        default="all",
    )

    parser.add_argument(
        "--attacks",
        nargs="+",
        default=ALL_ATTACKS,
        choices=ALL_ATTACKS,
        help=(
            "Conditions to extract. "
            "Default: clean + all 9 attacks/corruptions."
        ),
    )

    parser.add_argument(
        "--save_dtype",
        choices=[
            "float16",
            "float32",
        ],
        default="float16",
        help=(
            "Storage dtype for features.npy. "
            "float16 is recommended to save disk space. "
            "Convert to float32 when computing Mahalanobis."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    selected_models = (
        list(MODELS.keys())
        if args.model == "all"
        else [args.model]
    )

    print("=" * 88)
    print(
        "MAPILLARY PENULTIMATE FEATURE EXTRACTION"
    )
    print("=" * 88)
    print(
        f"Device       : {DEVICE}"
    )
    print(
        f"Models       : {selected_models}"
    )
    print(
        f"Conditions   : {args.attacks}"
    )
    print(
        f"Save dtype   : {args.save_dtype}"
    )
    print(
        f"Output root  : {FEATURE_ROOT}"
    )
    print(
        "Old pipeline : NOT MODIFIED"
    )

    Path(FEATURE_ROOT).mkdir(
        parents=True,
        exist_ok=True
    )

    combined = []

    for model_key in selected_models:
        cfg = MODELS[model_key]

        class_to_idx = load_class_map(
            cfg["class_to_idx"]
        )

        num_classes = len(
            class_to_idx
        )

        model = load_model(
            cfg,
            num_classes
        )

        final_linear_name, final_linear = (
            find_final_linear(model)
        )

        print("\n" + "=" * 88)
        print(
            f"[{model_key}] MODEL READY"
        )
        print("=" * 88)
        print(
            f"checkpoint    : {cfg['ckpt']}"
        )
        print(
            f"classes       : {num_classes}"
        )
        print(
            f"feature layer : input to "
            f"{final_linear_name}"
        )
        print(
            f"feature dim   : "
            f"{final_linear.in_features}"
        )

        for condition in args.attacks:
            image_dir = get_image_dir(
                cfg,
                condition
            )

            result = extract_condition(
                model=model,
                final_linear=final_linear,
                model_key=model_key,
                condition=condition,
                image_dir=image_dir,
                class_to_idx=class_to_idx,
                batch_size=cfg["batch_size"],
                save_dtype=args.save_dtype,
            )

            if result is not None:
                combined.append(result)

        del model

        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    summary_path = os.path.join(
        FEATURE_ROOT,
        "feature_extraction_summary.json"
    )

    with open(
        summary_path,
        "w"
    ) as f:
        json.dump(
            combined,
            f,
            indent=2
        )

    print("\n" + "=" * 88)
    print("FEATURE EXTRACTION COMPLETE")
    print("=" * 88)
    print(
        f"Summary -> {summary_path}"
    )
    print(
        "\nNo previous attack/detector/DDPM "
        "results were changed."
    )


if __name__ == "__main__":
    main()