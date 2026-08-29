import os
import csv
import json
import time
import math
import shutil
import argparse
import hashlib
import inspect
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from tqdm import tqdm

from diffusers import DDPMScheduler, UNet2DModel




DDPM_SIZE = 64
FINAL_SIZE = 224

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_ATTACKS = [
    "fgsm",
    "rfgsm",
    "pgd",
    "random_patch",
    "gaussian",
    "salt_pepper",
    "light",
    "fog",
    "motion_blur",
]

DDPM_DIR = "./ddpm_mapillary_64x64"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)




def make_mobilenet(num_classes):
    model = models.mobilenet_v3_large(weights=None)
    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        num_classes
    )
    return model


def make_convnext(num_classes):
    model = models.convnext_tiny(weights=None)
    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        num_classes
    )
    return model


def make_efficientnet(num_classes):
    model = models.efficientnet_v2_s(weights=None)
    model.classifier[-1] = nn.Linear(
        model.classifier[-1].in_features,
        num_classes
    )
    return model


MODELS = {
    "mobilenet": {
        "build": make_mobilenet,
        "ckpt": "./mapillary_baseline_results/mobilenetv3_mapillary_best.pth",
        "class_to_idx": "./mapillary_baseline_results/class_to_idx.json",
        "in_root": "./ddpm_input_combined_js_mobilenet",
        "out_root": "./ddpm_recon_combined_js_mobilenet",
    },
    "convnext": {
        "build": make_convnext,
        "ckpt": "./convnext_mapillary_results/convnext_mapillary_best.pth",
        "class_to_idx": "./convnext_mapillary_results/class_to_idx.json",
        "in_root": "./ddpm_input_combined_js_convnext",
        "out_root": "./ddpm_recon_combined_js_convnext",
    },
    "efficientnet": {
        "build": make_efficientnet,
        "ckpt": "./efficientnetv2_mapillary_results/efficientnetv2_mapillary_best.pth",
        "class_to_idx": "./efficientnetv2_mapillary_results/class_to_idx.json",
        "in_root": "./ddpm_input_combined_js_efficientnet",
        "out_root": "./ddpm_recon_combined_js_efficientnet",
    },
}


def safe_mkdir(path):
    Path(path).mkdir(
        parents=True,
        exist_ok=True
    )


def clean_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

    safe_mkdir(path)


def parse_t_candidates(text):
    """
    Example:
        "skip,20,40,80,120,160"
            -> [None, 20, 40, 80, 120, 160]
    """
    out = []

    for token in (text or "").split(","):
        token = token.strip().lower()

        if not token:
            continue

        if token == "skip":
            out.append(None)
        else:
            value = int(token)

            if value < 0:
                raise ValueError(
                    f"Negative timestep candidate is invalid: {value}"
                )

            out.append(value)

    if not out:
        raise ValueError(
            "No valid --t_candidates were provided."
        )

    dedup = []

    for value in out:
        if value not in dedup:
            dedup.append(value)

    return dedup


def robust_torch_load(path, device):
    try:
        return torch.load(
            path,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        return torch.load(
            path,
            map_location=device
        )


def load_state_dict_strict(model, ckpt_path, device):
    
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Classifier checkpoint missing: {ckpt_path}"
        )

    ckpt = robust_torch_load(
        ckpt_path,
        device
    )

    if (
        isinstance(ckpt, dict)
        and "state_dict" in ckpt
        and isinstance(ckpt["state_dict"], dict)
    ):
        state = ckpt["state_dict"]

    elif (
        isinstance(ckpt, dict)
        and any(
            isinstance(v, torch.Tensor)
            for v in ckpt.values()
        )
    ):
        state = ckpt

    else:
        raise RuntimeError(
            f"Unsupported classifier checkpoint format: {ckpt_path}"
        )

    if any(
        key.startswith("module.")
        for key in state.keys()
    ):
        state = {
            key.replace("module.", "", 1): value
            for key, value in state.items()
        }

    model.load_state_dict(
        state,
        strict=True
    )


def stable_int_hash(text):
    
    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        "little"
    ) & 0x7FFFFFFF




def resolve_ddpm_layout(ddpm_dir):
    
    ddpm_dir = os.path.abspath(
        ddpm_dir
    )

    if not os.path.isdir(ddpm_dir):
        raise FileNotFoundError(
            f"DDPM directory not found: {ddpm_dir}"
        )

    nested_unet = os.path.join(
        ddpm_dir,
        "unet"
    )

    nested_scheduler = os.path.join(
        ddpm_dir,
        "scheduler"
    )


    if (
        os.path.isdir(nested_unet)
        and os.path.exists(
            os.path.join(
                nested_unet,
                "config.json"
            )
        )
    ):
        unet_dir = nested_unet

        if os.path.isdir(nested_scheduler):
            scheduler_dir = nested_scheduler
        else:
            scheduler_dir = ddpm_dir

        return unet_dir, scheduler_dir

   
    root_config = os.path.join(
        ddpm_dir,
        "config.json"
    )

    if os.path.exists(root_config):
        return ddpm_dir, ddpm_dir

    raise FileNotFoundError(
        "Could not identify DDPM checkpoint layout.\n"
        f"Checked:\n"
        f"  {ddpm_dir}/config.json\n"
        f"  {ddpm_dir}/unet/config.json\n"
        "Expected a local Diffusers UNet/DDPM checkpoint."
    )




class DDPMPartialReconstructor:
  
  

    def __init__(
        self,
        ddpm_dir,
        device,
        inference_steps=100,
        seed=123,
        use_amp=True,
        use_ema=True,
    ):
        self.ddpm_dir = os.path.abspath(
            ddpm_dir
        )

        self.device = device
        self.inference_steps = int(
            inference_steps
        )
        self.base_seed = int(
            seed
        )

        self.use_amp = (
            bool(use_amp)
            and self.device.type == "cuda"
        )

        self.use_ema = bool(
            use_ema
        )

        if self.inference_steps <= 0:
            raise ValueError(
                "--steps must be > 0."
            )

        (
            self.unet_dir,
            self.scheduler_dir
        ) = resolve_ddpm_layout(
            self.ddpm_dir
        )

        print(
            "\nLoading local DDPM..."
        )

        print(
            f"  UNet dir      : {self.unet_dir}"
        )

        print(
            f"  Scheduler dir : {self.scheduler_dir}"
        )

        self.model = UNet2DModel.from_pretrained(
            self.unet_dir,
            local_files_only=True
        ).to(
            self.device
        ).eval()

        try:
            self.scheduler = DDPMScheduler.from_pretrained(
                self.scheduler_dir,
                local_files_only=True
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not load DDPMScheduler from the resolved local "
                f"directory:\n  {self.scheduler_dir}\n"
                "The trained scheduler configuration must be reused rather "
                "than inventing a new schedule."
            ) from exc

        # Validate model sample size when exposed in config.
        configured_size = getattr(
            self.model.config,
            "sample_size",
            None
        )

        if configured_size is not None:
            if isinstance(
                configured_size,
                (list, tuple)
            ):
                sizes = {
                    int(x)
                    for x in configured_size
                }

                if sizes != {DDPM_SIZE}:
                    raise ValueError(
                        "DDPM checkpoint sample_size does not match "
                        f"expected {DDPM_SIZE}: {configured_size}"
                    )
            else:
                if int(configured_size) != DDPM_SIZE:
                    raise ValueError(
                        "DDPM checkpoint sample_size does not match "
                        f"expected {DDPM_SIZE}: {configured_size}"
                    )

        self.ema_shadow = None

        ema_candidates = [
            os.path.join(
                self.ddpm_dir,
                "ema_shadow.pt"
            ),
            os.path.join(
                self.unet_dir,
                "ema_shadow.pt"
            ),
        ]

        if self.use_ema:
            for ema_path in ema_candidates:
                if os.path.exists(ema_path):
                    self.ema_shadow = robust_torch_load(
                        ema_path,
                        "cpu"
                    )

                    print(
                        f"  EMA file      : {ema_path}"
                    )
                    break

            if self.ema_shadow is None:
                print(
                    "  EMA            : requested but not found; "
                    "using saved UNet weights"
                )

        self._ema_applied = False
        self._ema_backup = None

        
        step_sig = inspect.signature(
            self.scheduler.step
        )

        self.scheduler_step_has_generator = (
            "generator"
            in step_sig.parameters
        )

        n_params = sum(
            p.numel()
            for p in self.model.parameters()
        )

        print(
            f"  Device         : {self.device}"
        )

        print(
            f"  Inference steps: {self.inference_steps}"
        )

        print(
            f"  AMP            : {self.use_amp}"
        )

        print(
            f"  Base seed      : {self.base_seed}"
        )

        print(
            f"  Parameters     : {n_params / 1e6:.2f}M"
        )

    def apply_ema_once(self):
       
        if (
            not self.use_ema
            or self.ema_shadow is None
            or self._ema_applied
        ):
            return

        backup = {}

        for name, param in self.model.named_parameters():
            if name not in self.ema_shadow:
                continue

            backup[
                name
            ] = param.detach().cpu().clone()

            param.data.copy_(
                self.ema_shadow[
                    name
                ].to(
                    device=param.device,
                    dtype=param.dtype
                )
            )

        self._ema_backup = backup
        self._ema_applied = True

        print(
            f"  EMA            : applied ({len(backup)} parameters)"
        )

    def restore_raw_weights(self):
        if (
            not self._ema_applied
            or self._ema_backup is None
        ):
            return

        for name, param in self.model.named_parameters():
            if name in self._ema_backup:
                param.data.copy_(
                    self._ema_backup[
                        name
                    ].to(
                        device=param.device,
                        dtype=param.dtype
                    )
                )

        self._ema_backup = None
        self._ema_applied = False

    def get_grid(self):
        self.scheduler.set_timesteps(
            self.inference_steps,
            device=self.device
        )

        return self.scheduler.timesteps

    def snap_t(self, requested_t, grid):
        requested_t = int(
            requested_t
        )

        max_train_t = int(
            self.scheduler.config.num_train_timesteps
        ) - 1

        if requested_t > max_train_t:
            raise ValueError(
                f"Requested t={requested_t} exceeds scheduler's "
                f"maximum training timestep {max_train_t}."
            )

        idx = torch.argmin(
            (
                grid -
                requested_t
            ).abs()
        )

        idx = int(
            idx.item()
        )

        snapped = int(
            grid[
                idx
            ].item()
        )

        return snapped, idx

    def make_generator(
        self,
        deterministic_key
    ):
        seed = (
            self.base_seed
            +
            stable_int_hash(
                deterministic_key
            )
        ) % 2147483647

        generator = torch.Generator(
            device=self.device
        )

        generator.manual_seed(
            seed
        )

        return generator, seed

    def add_noise(
        self,
        x,
        timestep,
        generator
    ):
        
        noise = torch.randn(
            x.shape,
            device=x.device,
            dtype=x.dtype,
            generator=generator
        )

        t_batch = torch.full(
            (x.shape[0],),
            int(timestep),
            device=x.device,
            dtype=torch.long
        )

        return self.scheduler.add_noise(
            x,
            noise,
            t_batch
        )

    @torch.inference_mode()
    def denoise_from(
        self,
        x,
        grid,
        start_idx,
        generator
    ):
        for timestep_tensor in grid[
            start_idx:
        ]:
            timestep = int(
                timestep_tensor.item()
            )

            t_batch = torch.full(
                (x.shape[0],),
                timestep,
                device=x.device,
                dtype=torch.long
            )

            if self.use_amp:
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):
                    predicted_noise = self.model(
                        x,
                        t_batch
                    ).sample
            else:
                predicted_noise = self.model(
                    x,
                    t_batch
                ).sample

            step_kwargs = {}

            if self.scheduler_step_has_generator:
                step_kwargs[
                    "generator"
                ] = generator

            x = self.scheduler.step(
                predicted_noise,
                timestep,
                x,
                **step_kwargs
            ).prev_sample

        return x

    @torch.inference_mode()
    def reconstruct_with_t(
        self,
        x_224_01,
        requested_t,
        deterministic_key
    ):
        x_64 = F.interpolate(
            x_224_01,
            size=(
                DDPM_SIZE,
                DDPM_SIZE
            ),
            mode="bicubic",
            align_corners=False,
            antialias=True
        ).clamp(
            0,
            1
        )

        x = (
            x_64 *
            2.0 -
            1.0
        )

        grid = self.get_grid()

        snapped_t, start_idx = self.snap_t(
            requested_t,
            grid
        )

        generator, used_seed = self.make_generator(
            deterministic_key
        )

        x_noisy = self.add_noise(
            x,
            snapped_t,
            generator
        )

        x_denoised = self.denoise_from(
            x_noisy,
            grid,
            start_idx,
            generator
        )

        x_rec_64 = (
            (
                x_denoised +
                1.0
            ) /
            2.0
        ).clamp(
            0,
            1
        )

        x_rec_224 = F.interpolate(
            x_rec_64,
            size=(
                FINAL_SIZE,
                FINAL_SIZE
            ),
            mode="bicubic",
            align_corners=False,
            antialias=True
        ).clamp(
            0,
            1
        )

        return (
            x_rec_224,
            snapped_t,
            used_seed
        )




def load_classifier(
    model_key,
    cfg
):
    if not os.path.exists(
        cfg["class_to_idx"]
    ):
        raise FileNotFoundError(
            f"class_to_idx missing: {cfg['class_to_idx']}"
        )

    with open(
        cfg["class_to_idx"],
        "r"
    ) as f:
        class_to_idx = json.load(
            f
        )

    num_classes = len(
        class_to_idx
    )

    model = cfg["build"](
        num_classes
    ).to(
        DEVICE
    )

    load_state_dict_strict(
        model,
        cfg["ckpt"],
        DEVICE
    )

    model.eval()

    print(
        f"\n[{model_key}] classifier loaded"
    )

    print(
        f"  checkpoint : {cfg['ckpt']}"
    )

    print(
        f"  classes    : {num_classes}"
    )

    return model, class_to_idx


def normalize_for_classifier(
    x
):
    mean = torch.tensor(
        IMAGENET_MEAN,
        device=x.device,
        dtype=x.dtype
    ).view(
        1,
        3,
        1,
        1
    )

    std = torch.tensor(
        IMAGENET_STD,
        device=x.device,
        dtype=x.dtype
    ).view(
        1,
        3,
        1,
        1
    )

    return (
        x -
        mean
    ) / std


@torch.inference_mode()
def classifier_outputs(
    classifier,
    x
):
    logits = classifier(
        normalize_for_classifier(
            x
        )
    )

    probs = torch.softmax(
        logits,
        dim=1
    )

    confidence, pred = probs.max(
        dim=1
    )

    return pred, confidence




VALID_IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".webp",
)


def list_images_recursive(
    attack_dir,
    class_to_idx
):
   
    records = []

    attack_root = os.path.abspath(
        attack_dir
    )

    for root, dirs, files in os.walk(
        attack_root
    ):
        dirs.sort()
        files.sort()

        for filename in files:
            if not filename.lower().endswith(
                VALID_IMAGE_EXTENSIONS
            ):
                continue

            abs_path = os.path.join(
                root,
                filename
            )

            rel_path = os.path.relpath(
                abs_path,
                attack_root
            )

            parts = Path(
                rel_path
            ).parts

            if len(parts) < 2:
                raise RuntimeError(
                    "Expected class-subfolder layout but found image "
                    f"directly under attack directory:\n  {abs_path}\n"
                    "The extractor must preserve class folders."
                )

            class_name = parts[
                0
            ]

            if class_name not in class_to_idx:
                raise RuntimeError(
                    f"Unknown class folder '{class_name}' in:\n"
                    f"  {abs_path}\n"
                    "This does not match the classifier's class_to_idx.json."
                )

            label = int(
                class_to_idx[
                    class_name
                ]
            )

            records.append({
                "abs_path": abs_path,
                "rel_path": rel_path,
                "class_name": class_name,
                "label": label,
            })

    records.sort(
        key=lambda record: record[
            "rel_path"
        ]
    )

    return records


#

INPUT_TRANSFORM = transforms.Compose([
    transforms.Resize(
        (
            FINAL_SIZE,
            FINAL_SIZE
        )
    ),
    transforms.ToTensor(),
])


def load_batch(
    records
):
    tensors = []

    for record in records:
        with Image.open(
            record["abs_path"]
        ) as image:
            image = image.convert(
                "RGB"
            )

            tensors.append(
                INPUT_TRANSFORM(
                    image
                )
            )

    return torch.stack(
        tensors,
        dim=0
    ).to(
        DEVICE,
        non_blocking=True
    )


def save_tensor_png(
    tensor_01,
    destination
):
    safe_mkdir(
        os.path.dirname(
            destination
        )
    )

    
    array = (
        tensor_01
        .detach()
        .cpu()
        .clamp(
            0,
            1
        )
        .permute(
            1,
            2,
            0
        )
        .numpy()
        *
        255.0
    )

    array = np.rint(
        array
    ).clip(
        0,
        255
    ).astype(
        np.uint8
    )

    Image.fromarray(
        array,
        mode="RGB"
    ).save(
        destination,
        format="PNG"
    )


def quantize_png_equivalent(x_01):
   
    return (
        torch.round(
            x_01.clamp(0, 1) * 255.0
        ) / 255.0
    ).clamp(0, 1)




def reconstruct_attack(
    model_key,
    attack_name,
    in_dir,
    out_dir,
    reconstructor,
    classifier,
    class_to_idx,
    batch_size,
    t_candidates,
    save_images=True,
):
    records = list_images_recursive(
        in_dir,
        class_to_idx
    )

    n_images = len(
        records
    )

    print(
        "\n"
        + "=" * 78
    )

    print(
        f"[{model_key}] RECONSTRUCT: {attack_name}"
    )

    print(
        "=" * 78
    )

    print(
        f"Input      : {in_dir}"
    )

    print(
        f"Output     : {out_dir}"
    )

    print(
        f"Images     : {n_images}"
    )

    print(
        f"Batch      : {batch_size}"
    )

    print(
        "Candidates : "
        + str([
            (
                "skip"
                if t is None
                else t
            )
            for t in t_candidates
        ])
    )

    
    clean_dir(
        out_dir
    )

    if n_images == 0:
        stats = {
            "model": model_key,
            "attack": attack_name,
            "n_images": 0,
            "accuracy_before_percent": None,
            "accuracy_after_percent": None,
            "accuracy_improvement_points": None,
        }

        with open(
            os.path.join(
                out_dir,
                "reconstruction_results.json"
            ),
            "w"
        ) as f:
            json.dump(
                stats,
                f,
                indent=2
            )

        return stats

    labels_all = np.asarray(
        [
            record["label"]
            for record in records
        ],
        dtype=np.int64
    )

    pred_before_all = []
    pred_after_all = []
    conf_before_all = []
    conf_after_all = []
    selected_t_all = []
    selected_seed_all = []

    reconstruction_time_ms_total = 0.0
    candidate_eval_time_ms_total = 0.0

    batch_time_per_image_ms = []

    selection_rows = []

    num_batches = math.ceil(
        n_images /
        batch_size
    )

    for batch_id, start in enumerate(
        tqdm(
            range(
                0,
                n_images,
                batch_size
            ),
            total=num_batches,
            desc=f"{model_key}/{attack_name}"
        )
    ):
        batch_records = records[
            start:
            start + batch_size
        ]

        x = load_batch(
            batch_records
        )

        labels = torch.tensor(
            [
                record["label"]
                for record in batch_records
            ],
            device=DEVICE,
            dtype=torch.long
        )

        pred_before, conf_before = classifier_outputs(
            classifier,
            x
        )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        total_start = time.perf_counter()

        best_x = None
        best_conf = None
        best_pred = None
        best_t = None
        best_seed = None

      
        for candidate_index, requested_t in enumerate(
            t_candidates
        ):
            candidate_start = time.perf_counter()

            if requested_t is None:
                candidate_x = x
                snapped_t = -1
                used_seed = -1

            else:
                deterministic_key = (
                    f"{model_key}|{attack_name}|"
                    f"batch={batch_id}|"
                    f"candidate={requested_t}"
                )

                (
                    candidate_x,
                    snapped_t,
                    used_seed
                ) = reconstructor.reconstruct_with_t(
                    x,
                    int(
                        requested_t
                    ),
                    deterministic_key
                )

         
            candidate_x = quantize_png_equivalent(
                candidate_x
            )

            candidate_pred, candidate_conf = classifier_outputs(
                classifier,
                candidate_x
            )

            if DEVICE.type == "cuda":
                torch.cuda.synchronize()

            candidate_eval_time_ms_total += (
                time.perf_counter()
                -
                candidate_start
            ) * 1000.0

            if best_conf is None:
                best_x = candidate_x
                best_conf = candidate_conf
                best_pred = candidate_pred

                best_t = torch.full(
                    (
                        x.shape[
                            0
                        ],
                    ),
                    int(
                        snapped_t
                    ),
                    device=DEVICE,
                    dtype=torch.long
                )

                best_seed = torch.full(
                    (
                        x.shape[
                            0
                        ],
                    ),
                    int(
                        used_seed
                    ),
                    device=DEVICE,
                    dtype=torch.long
                )

            else:
                better = (
                    candidate_conf >
                    best_conf
                )

                best_conf = torch.where(
                    better,
                    candidate_conf,
                    best_conf
                )

                best_pred = torch.where(
                    better,
                    candidate_pred,
                    best_pred
                )

                best_x = torch.where(
                    better.view(
                        -1,
                        1,
                        1,
                        1
                    ),
                    candidate_x,
                    best_x
                )

                candidate_t_tensor = torch.full(
                    (
                        x.shape[
                            0
                        ],
                    ),
                    int(
                        snapped_t
                    ),
                    device=DEVICE,
                    dtype=torch.long
                )

                best_t = torch.where(
                    better,
                    candidate_t_tensor,
                    best_t
                )

                candidate_seed_tensor = torch.full(
                    (
                        x.shape[
                            0
                        ],
                    ),
                    int(
                        used_seed
                    ),
                    device=DEVICE,
                    dtype=torch.long
                )

                best_seed = torch.where(
                    better,
                    candidate_seed_tensor,
                    best_seed
                )

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        batch_elapsed_ms = (
            time.perf_counter()
            -
            total_start
        ) * 1000.0

        reconstruction_time_ms_total += (
            batch_elapsed_ms
        )

        batch_time_per_image_ms.extend(
            [
                batch_elapsed_ms /
                x.shape[
                    0
                ]
            ]
            *
            x.shape[
                0
            ]
        )

        pred_before_np = (
            pred_before
            .detach()
            .cpu()
            .numpy()
        )

        conf_before_np = (
            conf_before
            .detach()
            .cpu()
            .numpy()
        )

        pred_after_np = (
            best_pred
            .detach()
            .cpu()
            .numpy()
        )

        conf_after_np = (
            best_conf
            .detach()
            .cpu()
            .numpy()
        )

        best_t_np = (
            best_t
            .detach()
            .cpu()
            .numpy()
        )

        best_seed_np = (
            best_seed
            .detach()
            .cpu()
            .numpy()
        )

        pred_before_all.extend(
            pred_before_np.tolist()
        )

        conf_before_all.extend(
            conf_before_np.tolist()
        )

        pred_after_all.extend(
            pred_after_np.tolist()
        )

        conf_after_all.extend(
            conf_after_np.tolist()
        )

        selected_t_all.extend(
            best_t_np.tolist()
        )

        selected_seed_all.extend(
            best_seed_np.tolist()
        )

        
        if save_images:
            for i, record in enumerate(
                batch_records
            ):
                destination = os.path.join(
                    out_dir,
                    record[
                        "rel_path"
                    ]
                )

                save_tensor_png(
                    best_x[
                        i
                    ],
                    destination
                )

        for i, record in enumerate(
            batch_records
        ):
            selection_rows.append({
                "relative_path": record[
                    "rel_path"
                ],
                "class_name": record[
                    "class_name"
                ],
                "true_label": int(
                    labels[
                        i
                    ].item()
                ),
                "pred_before": int(
                    pred_before_np[
                        i
                    ]
                ),
                "confidence_before": float(
                    conf_before_np[
                        i
                    ]
                ),
                "pred_after": int(
                    pred_after_np[
                        i
                    ]
                ),
                "confidence_after": float(
                    conf_after_np[
                        i
                    ]
                ),
                "selected_t": int(
                    best_t_np[
                        i
                    ]
                ),
                "selected_seed": int(
                    best_seed_np[
                        i
                    ]
                ),
            })

    pred_before_all = np.asarray(
        pred_before_all,
        dtype=np.int64
    )

    pred_after_all = np.asarray(
        pred_after_all,
        dtype=np.int64
    )

    conf_before_all = np.asarray(
        conf_before_all,
        dtype=np.float32
    )

    conf_after_all = np.asarray(
        conf_after_all,
        dtype=np.float32
    )

    selected_t_all = np.asarray(
        selected_t_all,
        dtype=np.int64
    )

    selected_seed_all = np.asarray(
        selected_seed_all,
        dtype=np.int64
    )

    before_correct = (
        pred_before_all ==
        labels_all
    )

    after_correct = (
        pred_after_all ==
        labels_all
    )

    accuracy_before = float(
        100.0 *
        before_correct.mean()
    )

    accuracy_after = float(
        100.0 *
        after_correct.mean()
    )

    accuracy_improvement = float(
        accuracy_after -
        accuracy_before
    )

    wrong_before = ~before_correct

    wrong_before_total = int(
        wrong_before.sum()
    )

    recovered_wrong = (
        wrong_before
        &
        after_correct
    )

    n_recovered_wrong = int(
        recovered_wrong.sum()
    )

    if wrong_before_total > 0:
        recovery_rate_of_wrong = float(
            100.0 *
            n_recovered_wrong /
            wrong_before_total
        )
    else:
        recovery_rate_of_wrong = None

    correct_before_total = int(
        before_correct.sum()
    )

    damaged_correct = (
        before_correct
        &
        (~after_correct)
    )

    n_damaged_correct = int(
        damaged_correct.sum()
    )

    if correct_before_total > 0:
        degradation_rate_of_correct = float(
            100.0 *
            n_damaged_correct /
            correct_before_total
        )
    else:
        degradation_rate_of_correct = None

    prediction_changed = (
        pred_before_all !=
        pred_after_all
    )

 
    chosen_t_counts = {}

    for value in selected_t_all:
        key = (
            "skip"
            if int(
                value
            ) == -1
            else str(
                int(
                    value
                )
            )
        )

        chosen_t_counts[
            key
        ] = (
            chosen_t_counts.get(
                key,
                0
            ) +
            1
        )

    avg_ms = float(
        np.mean(
            batch_time_per_image_ms
        )
    )

    min_ms = float(
        np.min(
            batch_time_per_image_ms
        )
    )

    max_ms = float(
        np.max(
            batch_time_per_image_ms
        )
    )

    stats = {
        "model": model_key,
        "attack": attack_name,
        "n_images": int(
            n_images
        ),
        "evaluation_scope": (
            "images present in ddpm_input for this condition "
            "(normally detector-flagged images)"
        ),
        "accuracy_before_percent": accuracy_before,
        "accuracy_after_percent": accuracy_after,
        "accuracy_improvement_points": accuracy_improvement,

        "wrong_before_total": wrong_before_total,
        "wrong_before_recovered": n_recovered_wrong,
        "recovery_rate_of_wrong_percent": recovery_rate_of_wrong,

        "correct_before_total": correct_before_total,
        "correct_before_damaged": n_damaged_correct,
        "degradation_rate_of_correct_percent": degradation_rate_of_correct,

        "prediction_changed_count": int(
            prediction_changed.sum()
        ),
        "prediction_changed_percent": float(
            100.0 *
            prediction_changed.mean()
        ),

        "mean_confidence_before": float(
            conf_before_all.mean()
        ),
        "mean_confidence_after": float(
            conf_after_all.mean()
        ),

        "chosen_t_counts": chosen_t_counts,

        "timing": {
            "min_ms_per_image": min_ms,
            "avg_ms_per_image": avg_ms,
            "max_ms_per_image": max_ms,
            "total_candidate_selection_ms": float(
                reconstruction_time_ms_total
            ),
            "throughput_images_per_second": float(
                1000.0 /
                avg_ms
            ) if avg_ms > 0 else 0.0,
        },

        "ddpm": {
            "ddpm_size": DDPM_SIZE,
            "final_size": FINAL_SIZE,
            "inference_steps": int(
                reconstructor.inference_steps
            ),
            "base_seed": int(
                reconstructor.base_seed
            ),
            "t_candidates_requested": [
                (
                    "skip"
                    if t is None
                    else int(
                        t
                    )
                )
                for t in t_candidates
            ],
            "candidate_selection": (
                "highest_classifier_confidence_after_png_equivalent_quantization"
            ),
            "ground_truth_used_for_candidate_selection": False,
            "unet_dir": reconstructor.unet_dir,
            "scheduler_dir": reconstructor.scheduler_dir,
            "amp": bool(
                reconstructor.use_amp
            ),
            "ema_applied": bool(
                reconstructor._ema_applied
            ),
        },

        "input_dir": in_dir,
        "output_dir": out_dir,
        "saved_images": bool(
            save_images
        ),
    }

    np.save(
        os.path.join(
            out_dir,
            "true_labels.npy"
        ),
        labels_all
    )

    np.save(
        os.path.join(
            out_dir,
            "pred_before.npy"
        ),
        pred_before_all
    )

    np.save(
        os.path.join(
            out_dir,
            "pred_after.npy"
        ),
        pred_after_all
    )

    np.save(
        os.path.join(
            out_dir,
            "confidence_before.npy"
        ),
        conf_before_all
    )

    np.save(
        os.path.join(
            out_dir,
            "confidence_after.npy"
        ),
        conf_after_all
    )

    np.save(
        os.path.join(
            out_dir,
            "selected_t.npy"
        ),
        selected_t_all
    )

   
    csv_path = os.path.join(
        out_dir,
        "reconstruction_per_image.csv"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:
        fieldnames = [
            "relative_path",
            "class_name",
            "true_label",
            "pred_before",
            "confidence_before",
            "pred_after",
            "confidence_after",
            "selected_t",
            "selected_seed",
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        writer.writerows(
            selection_rows
        )

    results_path = os.path.join(
        out_dir,
        "reconstruction_results.json"
    )

    with open(
        results_path,
        "w"
    ) as f:
        json.dump(
            stats,
            f,
            indent=2
        )

    print(
        f"\n  Accuracy: "
        f"{accuracy_before:.2f}% -> {accuracy_after:.2f}% "
        f"({accuracy_improvement:+.2f} points)"
    )

    print(
        "  Wrong-before recovery: "
        + (
            "N/A"
            if recovery_rate_of_wrong is None
            else f"{recovery_rate_of_wrong:.2f}% "
                 f"({n_recovered_wrong}/{wrong_before_total})"
        )
    )

    print(
        "  Correct-before damaged: "
        + (
            "N/A"
            if degradation_rate_of_correct is None
            else f"{degradation_rate_of_correct:.2f}% "
                 f"({n_damaged_correct}/{correct_before_total})"
        )
    )

    print(
        f"  Timing: min/avg/max = "
        f"{min_ms:.2f}/{avg_ms:.2f}/{max_ms:.2f} ms/img"
    )

    print(
        f"  Chosen t: {chosen_t_counts}"
    )

    return stats




def reconstruct_model(
    model_key,
    cfg,
    attacks,
    reconstructor,
    batch_size,
    t_candidates,
    save_images
):
    if not os.path.isdir(
        cfg["in_root"]
    ):
        raise FileNotFoundError(
            f"DDPM input root missing: {cfg['in_root']}\n"
            "Run flagged extraction first with:\n"
            "  python extract_for_ddpm_unified_v2.py "
            "--model all --only_flagged"
        )

    classifier, class_to_idx = load_classifier(
        model_key,
        cfg
    )

    safe_mkdir(
        cfg["out_root"]
    )

    all_stats = {}

    for attack_name in attacks:
        in_dir = os.path.join(
            cfg["in_root"],
            attack_name
        )

        if not os.path.isdir(
            in_dir
        ):
            print(
                f"\n[{model_key}] SKIP missing: {in_dir}"
            )

            continue

        out_dir = os.path.join(
            cfg["out_root"],
            attack_name
        )

        stats = reconstruct_attack(
            model_key=model_key,
            attack_name=attack_name,
            in_dir=in_dir,
            out_dir=out_dir,
            reconstructor=reconstructor,
            classifier=classifier,
            class_to_idx=class_to_idx,
            batch_size=batch_size,
            t_candidates=t_candidates,
            save_images=save_images,
        )

        all_stats[
            attack_name
        ] = stats

    model_summary_path = os.path.join(
        cfg["out_root"],
        "ddpm_reconstruction_ALL_results.json"
    )

    with open(
        model_summary_path,
        "w"
    ) as f:
        json.dump(
            {
                "model": model_key,
                "classifier_checkpoint": cfg[
                    "ckpt"
                ],
                "class_to_idx": cfg[
                    "class_to_idx"
                ],
                "input_root": cfg[
                    "in_root"
                ],
                "output_root": cfg[
                    "out_root"
                ],
                "results": all_stats,
            },
            f,
            indent=2
        )

    del classifier

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    return all_stats



def save_combined_summary(
    all_results,
    model_keys,
    attacks
):
    json_path = (
        "./ddpm_reconstruction_combined_results.json"
    )

    with open(
        json_path,
        "w"
    ) as f:
        json.dump(
            {
                "models": all_results
            },
            f,
            indent=2
        )

    csv_path = (
        "./ddpm_reconstruction_combined_summary.csv"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:
        writer = csv.writer(
            f
        )

        writer.writerow([
            "model",
            "attack",
            "n_images",
            "accuracy_before_percent",
            "accuracy_after_percent",
            "accuracy_improvement_points",
            "recovery_rate_of_wrong_percent",
            "degradation_rate_of_correct_percent",
            "avg_ms_per_image",
        ])

        for model_key in model_keys:
            for attack_name in attacks:
                stats = all_results.get(
                    model_key,
                    {}
                ).get(
                    attack_name
                )

                if stats is None:
                    continue

                writer.writerow([
                    model_key,
                    attack_name,
                    stats.get(
                        "n_images"
                    ),
                    stats.get(
                        "accuracy_before_percent"
                    ),
                    stats.get(
                        "accuracy_after_percent"
                    ),
                    stats.get(
                        "accuracy_improvement_points"
                    ),
                    stats.get(
                        "recovery_rate_of_wrong_percent"
                    ),
                    stats.get(
                        "degradation_rate_of_correct_percent"
                    ),
                    stats.get(
                        "timing",
                        {}
                    ).get(
                        "avg_ms_per_image"
                    ),
                ])

    return json_path, csv_path



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        choices=list(
            MODELS.keys()
        ) + ["all"],
        required=True
    )

    parser.add_argument(
        "--attacks",
        nargs="+",
        choices=DEFAULT_ATTACKS,
        default=DEFAULT_ATTACKS
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help=(
            "Number of scheduler inference steps. "
            "Default preserves the prior reconstruction setup: 100."
        )
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=32
    )

    parser.add_argument(
        "--t_candidates",
        type=str,
        default="skip,20,40,80,120,160",
        help=(
            "Comma-separated requested forward-noise timesteps. "
            "'skip' keeps the original input as a candidate. "
            "Selection is by classifier confidence, never by true label."
        )
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123
    )

    parser.add_argument(
        "--no_amp",
        action="store_true"
    )

    parser.add_argument(
        "--no_ema",
        action="store_true"
    )

    parser.add_argument(
        "--no_save_images",
        action="store_true"
    )

    args = parser.parse_args()

    if args.batch <= 0:
        raise ValueError(
            "--batch must be > 0."
        )

    if not os.path.isdir(DDPM_DIR):
        raise FileNotFoundError(
            f"Confirmed DDPM directory not found: {DDPM_DIR}"
        )

    expected_ddpm_files = [
        os.path.join(DDPM_DIR, "model_index.json"),
        os.path.join(DDPM_DIR, "unet", "config.json"),
        os.path.join(DDPM_DIR, "unet", "diffusion_pytorch_model.safetensors"),
        os.path.join(DDPM_DIR, "scheduler", "scheduler_config.json"),
    ]

    missing_ddpm_files = [
        path for path in expected_ddpm_files
        if not os.path.exists(path)
    ]

    if missing_ddpm_files:
        raise FileNotFoundError(
            "The confirmed Mapillary DDPM checkpoint is incomplete. "
            "Missing:\n  " + "\n  ".join(missing_ddpm_files)
        )

    t_candidates = parse_t_candidates(
        args.t_candidates
    )

    model_keys = (
        list(
            MODELS.keys()
        )
        if args.model == "all"
        else [
            args.model
        ]
    )

    print(
        "=" * 88
    )

    print(
        "MAPILLARY DDPM RECONSTRUCTION + RECLASSIFICATION FINAL"
    )

    print(
        "=" * 88
    )

    print(
        f"Device       : {DEVICE}"
    )

    print(
        f"Models       : {model_keys}"
    )

    print(
        f"Attacks      : {args.attacks}"
    )

    print(
        f"DDPM dir     : {DDPM_DIR} (confirmed final Mapillary checkpoint)"
    )

    print(
        f"DDPM size    : {DDPM_SIZE}"
    )

    print(
        f"Final size   : {FINAL_SIZE}"
    )

    print(
        f"Steps        : {args.steps}"
    )

    print(
        f"Batch        : {args.batch}"
    )

    print(
        "Candidates   : "
        + str([
            (
                "skip"
                if t is None
                else t
            )
            for t in t_candidates
        ])
    )

    print(
        f"Seed         : {args.seed}"
    )

    print(
        "Evaluation   : PNG-equivalent reconstruction tensors"
    )

    print(
        "Run mode     : all requested models/attacks in one command"
    )

    reconstructor = DDPMPartialReconstructor(
        ddpm_dir=DDPM_DIR,
        device=DEVICE,
        inference_steps=args.steps,
        seed=args.seed,
        use_amp=(
            not args.no_amp
        ),
        use_ema=(
            not args.no_ema
        ),
    )

    reconstructor.apply_ema_once()

    all_results = {}

    try:
        for model_key in model_keys:
            all_results[
                model_key
            ] = reconstruct_model(
                model_key=model_key,
                cfg=MODELS[
                    model_key
                ],
                attacks=args.attacks,
                reconstructor=reconstructor,
                batch_size=args.batch,
                t_candidates=t_candidates,
                save_images=(
                    not args.no_save_images
                ),
            )

    finally:
        reconstructor.restore_raw_weights()

   
    print(
        "\n"
        + "=" * 88
    )

    print(
        "DDPM RECONSTRUCTION SUMMARY "
        "(accuracy on extracted/flagged subset)"
    )

    print(
        "=" * 88
    )

    for model_key in model_keys:
        print(
            f"\n{model_key}:"
        )

        for attack_name in args.attacks:
            stats = all_results.get(
                model_key,
                {}
            ).get(
                attack_name
            )

            if stats is None:
                print(
                    f"  {attack_name:13s} --"
                )
                continue

            n = stats[
                "n_images"
            ]

            before = stats.get(
                "accuracy_before_percent"
            )

            after = stats.get(
                "accuracy_after_percent"
            )

            improvement = stats.get(
                "accuracy_improvement_points"
            )

            if before is None:
                print(
                    f"  {attack_name:13s} n={n:6d}"
                )
            else:
                print(
                    f"  {attack_name:13s} "
                    f"n={n:6d}  "
                    f"{before:6.2f}% -> {after:6.2f}%  "
                    f"({improvement:+6.2f} pts)"
                )

    json_path, csv_path = save_combined_summary(
        all_results=all_results,
        model_keys=model_keys,
        attacks=args.attacks
    )

    print(
        f"\nCombined JSON -> {json_path}"
    )

    print(
        f"Combined CSV  -> {csv_path}"
    )


if __name__ == "__main__":
    main()