

import os
import re
import csv
import json
import time
import random
import shutil
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms, models
from tqdm import tqdm

from diffusers import (
    DDPMScheduler,
    UNet2DModel,
)



ROOT = "./gtsrb_repeat"


INPUT_ROOT = os.path.join(
    ROOT,
    "suspicious_ours_js",
)


OUTPUT_ROOT = os.path.join(
    ROOT,
    "ddpm_reconstruction_ours_js",
)


MODEL_PATHS = {

    "mobilenet": os.path.join(
        ROOT,
        "models",
        "mobilenet",
        "mobilenet_gtsrb_best.pth",
    ),

    "convnext": os.path.join(
        ROOT,
        "models",
        "convnext",
        "convnext_gtsrb_best.pth",
    ),

    "efficientnet": os.path.join(
        ROOT,
        "models",
        "efficientnet",
        "efficientnet_gtsrb_best.pth",
    ),
}


MODELS = [
    "mobilenet",
    "convnext",
    "efficientnet",
]


CONDITIONS = [
    "clean",
    "fgsm",
    "rfgsm",
    "pgd",
    "patch",
    "random_patch",
    "gaussian_noise",
    "salt_pepper",
    "light",
    "fog",
    "motion_blur",
]


IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".ppm",
    ".webp",
}


FILENAME_PATTERN = re.compile(
    r"^(?P<sample_id>\d+)_y(?P<label>\d+)"
)


DDPM_SIZE = 64

FINAL_SIZE = 224

NUM_CLASSES = 43


IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406,
]


IMAGENET_STD = [
    0.229,
    0.224,
    0.225,
]


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)




def set_seed(
    seed,
):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )



def parse_t_candidates(
    text,
):

    candidates = []

    for token in (
        text or ""
    ).split(","):

        token = (
            token
            .strip()
            .lower()
        )

        if not token:

            continue

        if token == "skip":

            candidates.append(
                None
            )

        else:

            candidates.append(
                int(
                    token
                )
            )

    if not candidates:

        raise ValueError(
            "At least one reconstruction candidate is required."
        )

    return candidates



def parse_filename(
    path,
):

    basename = os.path.basename(
        path
    )

    match = FILENAME_PATTERN.match(
        basename
    )

    if match is None:

        raise RuntimeError(
            "\nCould not parse image filename:\n"
            f"{basename}\n\n"
            "Expected format such as:\n"
            "00000_y16.png"
        )

    sample_id = int(
        match.group(
            "sample_id"
        )
    )

    label = int(
        match.group(
            "label"
        )
    )

    return (
        sample_id,
        label,
    )


def list_images(
    folder,
):

    if not os.path.isdir(
        folder
    ):

        return []

    images = []

    for filename in os.listdir(
        folder
    ):

        extension = os.path.splitext(
            filename
        )[1].lower()

        if extension in IMAGE_EXTENSIONS:

            images.append(
                os.path.join(
                    folder,
                    filename,
                )
            )

    images.sort()

    return images



def get_input_image_directory(
    model,
    condition,
):

    return os.path.join(
        INPUT_ROOT,
        model,
        condition,
        "images",
    )


def get_input_metadata_path(
    model,
    condition,
):

    return os.path.join(
        INPUT_ROOT,
        model,
        condition,
        "metadata.csv",
    )


def get_output_condition_directory(
    model,
    condition,
):

    return os.path.join(
        OUTPUT_ROOT,
        model,
        condition,
    )


def prepare_output_directory(
    model,
    condition,
    overwrite,
):

    condition_output = (
        get_output_condition_directory(
            model,
            condition,
        )
    )

    if os.path.isdir(
        condition_output
    ):

        if overwrite:

            shutil.rmtree(
                condition_output
            )

        else:

            raise RuntimeError(
                "\nOutput already exists:\n"
                f"{condition_output}\n\n"
                "Use --overwrite if you intentionally "
                "want to replace these results."
            )

    image_output = os.path.join(
        condition_output,
        "images",
    )

    Path(
        image_output
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        condition_output,
        image_output,
    )



def build_classifier(
    model_key,
):

    if model_key == "mobilenet":

        model = (
            models.mobilenet_v3_large(
                weights=None
            )
        )

        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            NUM_CLASSES,
        )

    elif model_key == "convnext":

        model = (
            models.convnext_tiny(
                weights=None
            )
        )

        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            NUM_CLASSES,
        )

    elif model_key == "efficientnet":

        model = (
            models.efficientnet_v2_s(
                weights=None
            )
        )

        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features,
            NUM_CLASSES,
        )

    else:

        raise ValueError(
            f"Unknown classifier: {model_key}"
        )

    return model



def extract_classifier_state_dict(
    checkpoint,
):

    
    if (
        isinstance(
            checkpoint,
            dict,
        )
        and
        isinstance(
            checkpoint.get(
                "model_state_dict"
            ),
            dict,
        )
    ):

        state = checkpoint[
            "model_state_dict"
        ]

   
    elif (
        isinstance(
            checkpoint,
            dict,
        )
        and
        isinstance(
            checkpoint.get(
                "state_dict"
            ),
            dict,
        )
    ):

        state = checkpoint[
            "state_dict"
        ]

    
    elif (
        isinstance(
            checkpoint,
            dict,
        )
        and
        any(
            isinstance(
                value,
                torch.Tensor,
            )
            for value
            in checkpoint.values()
        )
    ):

        state = checkpoint

    else:

        raise RuntimeError(
            "Unsupported classifier checkpoint format."
        )

   
    if any(
        key.startswith(
            "module."
        )
        for key
        in state.keys()
    ):

        state = {

            key.replace(
                "module.",
                "",
                1,
            ):
            value

            for key, value
            in state.items()
        }

    return state


def load_classifier(
    model_key,
):

    checkpoint_path = MODEL_PATHS[
        model_key
    ]

    if not os.path.isfile(
        checkpoint_path
    ):

        raise FileNotFoundError(
            "\nClassifier checkpoint missing:\n"
            f"{checkpoint_path}"
        )

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"LOADING CLASSIFIER: {model_key.upper()}"
    )

    print(
        "=" * 80
    )

    print(
        f"Checkpoint: {checkpoint_path}"
    )

    model = build_classifier(
        model_key
    )

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
            weights_only=False,
        )

    except TypeError:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=DEVICE,
        )

    state = extract_classifier_state_dict(
        checkpoint
    )

    model.load_state_dict(
        state,
        strict=True,
    )

    model = (
        model
        .to(
            DEVICE
        )
        .eval()
    )

    print(
        "Classifier loaded successfully."
    )

    return model



def normalize_for_classifier(
    x,
):

    mean = torch.tensor(
        IMAGENET_MEAN,
        dtype=x.dtype,
        device=x.device,
    ).view(
        1,
        3,
        1,
        1,
    )

    std = torch.tensor(
        IMAGENET_STD,
        dtype=x.dtype,
        device=x.device,
    ).view(
        1,
        3,
        1,
        1,
    )

    return (
        x
        -
        mean
    ) / std


@torch.no_grad()
def classifier_outputs(
    classifier,
    x,
):

    logits = classifier(
        normalize_for_classifier(
            x
        )
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )

    (
        confidence,
        prediction,
    ) = probabilities.max(
        dim=1
    )

    return (
        prediction,
        confidence,
    )



def find_ddpm_layout(
    ddpm_dir,
):

   
    direct_unet_config = os.path.join(
        ddpm_dir,
        "config.json",
    )

    nested_unet_config = os.path.join(
        ddpm_dir,
        "unet",
        "config.json",
    )

    if os.path.isfile(
        direct_unet_config
    ):

        unet_dir = ddpm_dir

    elif os.path.isfile(
        nested_unet_config
    ):

        unet_dir = os.path.join(
            ddpm_dir,
            "unet",
        )

    else:

        raise FileNotFoundError(
            "\nCould not find DDPM UNet configuration.\n\n"
            "Checked:\n"
            f"  {direct_unet_config}\n"
            f"  {nested_unet_config}"
        )

   
    direct_scheduler_config = os.path.join(
        ddpm_dir,
        "scheduler_config.json",
    )

    nested_scheduler_config = os.path.join(
        ddpm_dir,
        "scheduler",
        "scheduler_config.json",
    )

    if os.path.isfile(
        direct_scheduler_config
    ):

        scheduler_dir = ddpm_dir

    elif os.path.isfile(
        nested_scheduler_config
    ):

        scheduler_dir = os.path.join(
            ddpm_dir,
            "scheduler",
        )

    else:

        raise FileNotFoundError(
            "\nCould not find DDPM scheduler configuration.\n\n"
            "Checked:\n"
            f"  {direct_scheduler_config}\n"
            f"  {nested_scheduler_config}"
        )

    return (
        unet_dir,
        scheduler_dir,
    )



def load_ema_shadow(
    ema_path,
):

   

    try:

        checkpoint = torch.load(
            ema_path,
            map_location="cpu",
            weights_only=False,
        )

    except TypeError:

        checkpoint = torch.load(
            ema_path,
            map_location="cpu",
        )

    if not isinstance(
        checkpoint,
        dict,
    ):

        raise RuntimeError(
            f"Invalid EMA checkpoint:\n{ema_path}"
        )

   
    if (
        "shadow" in checkpoint
        and
        isinstance(
            checkpoint[
                "shadow"
            ],
            dict,
        )
    ):

        print(
            "EMA format     : new {decay, shadow} format"
        )

        if "decay" in checkpoint:

            print(
                f"EMA decay      : {checkpoint['decay']}"
            )

        shadow = checkpoint[
            "shadow"
        ]

   
    else:

        print(
            "EMA format     : direct/raw state dictionary"
        )

        shadow = checkpoint

    return shadow



class DDPMPartialReconstructor:

    def __init__(
        self,
        ddpm_dir,
        steps=100,
        seed=123,
        use_amp=True,
        use_ema=True,
    ):

        self.device = DEVICE

        self.steps = int(
            steps
        )

        self.seed = int(
            seed
        )

        self.use_amp = (
            bool(
                use_amp
            )
            and
            DEVICE.type == "cuda"
        )

        self.use_ema = bool(
            use_ema
        )

       
        (
            unet_dir,
            scheduler_dir,
        ) = find_ddpm_layout(
            ddpm_dir
        )

        
        print(
            "\n"
            + "=" * 90
        )

        print(
            "LOADING DDPM"
        )

        print(
            "=" * 90
        )

        print(
            f"DDPM directory : {ddpm_dir}"
        )

        print(
            f"UNet directory : {unet_dir}"
        )

        print(
            f"Scheduler      : {scheduler_dir}"
        )

        self.model = (
            UNet2DModel
            .from_pretrained(
                unet_dir,
                local_files_only=True,
            )
            .to(
                DEVICE
            )
            .eval()
        )

        
        self.scheduler = (
            DDPMScheduler
            .from_pretrained(
                scheduler_dir,
                local_files_only=True,
            )
        )

        
        self.generator = torch.Generator(
            device=DEVICE
        )

        self.generator.manual_seed(
            self.seed
        )

        
        self.ema_shadow = None

        if self.use_ema:

            possible_ema_paths = [

                os.path.join(
                    ddpm_dir,
                    "ema_shadow.pt",
                ),

                os.path.join(
                    unet_dir,
                    "ema_shadow.pt",
                ),
            ]

            selected_ema_path = None

            for ema_path in possible_ema_paths:

                if os.path.isfile(
                    ema_path
                ):

                    selected_ema_path = (
                        ema_path
                    )

                    break

            if selected_ema_path is not None:

                print(
                    f"EMA checkpoint : {selected_ema_path}"
                )

                self.ema_shadow = (
                    load_ema_shadow(
                        selected_ema_path
                    )
                )

            else:

                print(
                    "EMA checkpoint : NOT FOUND"
                )

        
        if self.ema_shadow is not None:

            applied = 0

            missing_from_ema = []

            with torch.no_grad():

                for (
                    name,
                    parameter,
                ) in self.model.named_parameters():

                    if name in self.ema_shadow:

                        ema_tensor = (
                            self.ema_shadow[
                                name
                            ]
                        )

                        parameter.copy_(
                            ema_tensor.to(
                                device=parameter.device,
                                dtype=parameter.dtype,
                            )
                        )

                        applied += 1

                    else:

                        missing_from_ema.append(
                            name
                        )

            print(
                f"EMA parameters applied : {applied}"
            )

            print(
                f"EMA parameters missing : {len(missing_from_ema)}"
            )

            if applied == 0:

                raise RuntimeError(
                    "\nEMA file was found but ZERO parameters "
                    "were applied.\n"
                    "This indicates an incompatible EMA checkpoint."
                )

        else:

            print(
                "EMA             : disabled/not available; "
                "using raw DDPM weights."
            )

        
        parameter_count = sum(
            parameter.numel()
            for parameter
            in self.model.parameters()
        )

        print(
            f"Device          : {DEVICE}"
        )

        print(
            f"Parameters      : "
            f"{parameter_count / 1e6:.2f} M"
        )

        print(
            f"DDPM resolution : "
            f"{DDPM_SIZE}x{DDPM_SIZE}"
        )

        print(
            f"Inference steps : "
            f"{self.steps}"
        )

        print(
            f"AMP             : "
            f"{'ON' if self.use_amp else 'OFF'}"
        )

        print(
            f"Seed            : "
            f"{self.seed}"
        )

        print(
            "=" * 90
        )


   
    def get_grid(
        self,
    ):

        self.scheduler.set_timesteps(
            self.steps,
            device=DEVICE,
        )

        return (
            self.scheduler.timesteps
        )


   
    @staticmethod
    def snap_timestep(
        requested_t,
        grid,
    ):

        requested_t = int(
            requested_t
        )

        index = torch.argmin(
            (
                grid
                -
                requested_t
            ).abs()
        )

        index = int(
            index.item()
        )

        snapped_t = int(
            grid[
                index
            ].item()
        )

        return (
            snapped_t,
            index,
        )


   
    @torch.no_grad()
    def reconstruct(
        self,
        x_224,
        requested_t,
    ):

       
        x_64 = F.interpolate(
            x_224,
            size=(
                DDPM_SIZE,
                DDPM_SIZE,
            ),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )

        
        x = (
            x_64
            * 2.0
            -
            1.0
        )

      
        grid = self.get_grid()

        (
            snapped_t,
            start_index,
        ) = self.snap_timestep(
            requested_t,
            grid,
        )

        
        noise = torch.randn(
            x.shape,
            generator=self.generator,
            device=x.device,
            dtype=x.dtype,
        )

        timestep_batch = torch.full(
            (
                x.shape[0],
            ),
            snapped_t,
            dtype=torch.long,
            device=x.device,
        )

        noisy_x = (
            self.scheduler.add_noise(
                x,
                noise,
                timestep_batch,
            )
        )

       
        x_reverse = noisy_x

        for timestep in grid[
            start_index:
        ]:

            timestep_int = int(
                timestep.item()
            )

            timestep_batch = torch.full(
                (
                    x_reverse.shape[0],
                ),
                timestep_int,
                dtype=torch.long,
                device=x_reverse.device,
            )

            if self.use_amp:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                ):

                    predicted_noise = (
                        self.model(
                            x_reverse,
                            timestep_batch,
                        ).sample
                    )

            else:

                predicted_noise = (
                    self.model(
                        x_reverse,
                        timestep_batch,
                    ).sample
                )

            x_reverse = (
                self.scheduler.step(
                    predicted_noise,
                    timestep_int,
                    x_reverse,
                ).prev_sample
            )

       
        reconstructed_64 = (
            (
                x_reverse
                + 1.0
            )
            / 2.0
        ).clamp(
            0.0,
            1.0,
        )

       
        reconstructed_224 = (
            F.interpolate(
                reconstructed_64,
                size=(
                    FINAL_SIZE,
                    FINAL_SIZE,
                ),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            .clamp(
                0.0,
                1.0,
            )
        )

        return (
            reconstructed_224,
            snapped_t,
        )



def load_extraction_metadata(
    model,
    condition,
):

    metadata_path = (
        get_input_metadata_path(
            model,
            condition,
        )
    )

    if not os.path.isfile(
        metadata_path
    ):

        print(
            "WARNING: extraction metadata does not exist:\n"
            f"{metadata_path}"
        )

        return {}

    metadata = {}

    with open(
        metadata_path,
        "r",
        newline="",
    ) as f:

        reader = csv.DictReader(
            f
        )

        for row in reader:

            filename = row[
                "filename"
            ]

            if filename in metadata:

                raise RuntimeError(
                    f"Duplicate filename in metadata: "
                    f"{filename}"
                )

            metadata[
                filename
            ] = row

    return metadata



def validate_metadata(
    image_paths,
    metadata,
    model,
    condition,
):

    if not metadata:

        return

    if len(
        image_paths
    ) != len(
        metadata
    ):

        raise RuntimeError(
            f"\n{model}/{condition}: "
            "image count does not match metadata count.\n"
            f"Images   : {len(image_paths)}\n"
            f"Metadata : {len(metadata)}"
        )

    for path in image_paths:

        (
            sample_id,
            label,
        ) = parse_filename(
            path
        )

        filename = os.path.basename(
            path
        )

        if filename not in metadata:

            raise RuntimeError(
                f"{model}/{condition}: "
                f"{filename} missing from metadata.csv"
            )

        row = metadata[
            filename
        ]

        metadata_sample_id = int(
            row[
                "sample_id"
            ]
        )

        metadata_label = int(
            row[
                "label"
            ]
        )

        if metadata_sample_id != sample_id:

            raise RuntimeError(
                f"Sample-ID mismatch for "
                f"{filename}"
            )

        if metadata_label != label:

            raise RuntimeError(
                f"Ground-truth label mismatch "
                f"for {filename}"
            )



def save_png(
    tensor,
    path,
):

    array = (
        tensor
        .detach()
        .cpu()
        .permute(
            1,
            2,
            0,
        )
        .clamp(
            0.0,
            1.0,
        )
        .numpy()
    )

    array = np.rint(
        array
        * 255.0
    ).astype(
        np.uint8
    )

    Image.fromarray(
        array
    ).save(
        path,
        format="PNG",
    )



def reconstruct_condition(
    model_key,
    condition,
    classifier,
    reconstructor,
    candidates,
    batch_size,
    overwrite,
    save_images=True,
):

    
    input_dir = (
        get_input_image_directory(
            model_key,
            condition,
        )
    )

    if not os.path.isdir(
        input_dir
    ):

        raise FileNotFoundError(
            "\nInput directory not found:\n"
            f"{input_dir}"
        )

    image_paths = list_images(
        input_dir
    )

   
    metadata = (
        load_extraction_metadata(
            model_key,
            condition,
        )
    )

    validate_metadata(
        image_paths,
        metadata,
        model_key,
        condition,
    )

    (
        output_dir,
        output_image_dir,
    ) = prepare_output_directory(
        model_key,
        condition,
        overwrite,
    )

    num_images = len(
        image_paths
    )

    print(
        "\n"
        + "=" * 110
    )

    print(
        f"{model_key.upper()} | "
        f"{condition}"
    )

    print(
        "=" * 110
    )

    print(
        f"Input       : {input_dir}"
    )

    print(
        f"Output      : {output_dir}"
    )

    print(
        f"Suspicious  : {num_images:,}"
    )

    print(
        f"Batch       : {batch_size}"
    )

    print(
        "Candidates  : "
        + str(
            [
                (
                    "skip"
                    if t is None
                    else t
                )
                for t in candidates
            ]
        )
    )

   
    if num_images == 0:

        summary = {

            "model":
                model_key,

            "condition":
                condition,

            "n_images":
                0,
        }

        with open(
            os.path.join(
                output_dir,
                "summary.json",
            ),
            "w",
        ) as f:

            json.dump(
                summary,
                f,
                indent=2,
            )

        return summary

   
    transform = transforms.Compose(
        [
            transforms.Resize(
                (
                    FINAL_SIZE,
                    FINAL_SIZE,
                )
            ),

            transforms.ToTensor(),
        ]
    )

  
    rows = []

    chosen_t_counts = {}

    timing_per_image = []

    correct_before = 0

    correct_after = 0

    recovered_wrong = 0

    damaged_correct = 0

    stayed_wrong = 0

    stayed_correct = 0

    total_candidate_selection_ms = (
        0.0
    )

   
    for start in tqdm(
        range(
            0,
            num_images,
            batch_size,
        ),
        desc=(
            f"{model_key}/"
            f"{condition}"
        ),
    ):

        batch_paths = image_paths[
            start:
            start + batch_size
        ]

        images = []

        labels = []

        sample_ids = []

        filenames = []

       
        for path in batch_paths:

            image = (
                Image.open(
                    path
                )
                .convert(
                    "RGB"
                )
            )

            images.append(
                transform(
                    image
                )
            )

            (
                sample_id,
                label,
            ) = parse_filename(
                path
            )

            sample_ids.append(
                sample_id
            )

            labels.append(
                label
            )

            filenames.append(
                os.path.basename(
                    path
                )
            )

        x = torch.stack(
            images
        ).to(
            DEVICE,
            non_blocking=True,
        )

        y = torch.tensor(
            labels,
            dtype=torch.long,
            device=DEVICE,
        )

       
        (
            prediction_before,
            confidence_before,
        ) = classifier_outputs(
            classifier,
            x,
        )

       
        if DEVICE.type == "cuda":

            torch.cuda.synchronize()

        time_start = (
            time.perf_counter()
        )

        
        best_x = None

        best_confidence = None

        best_timestep = None

        for requested_t in candidates:

           
            if requested_t is None:

                candidate_x = x

                snapped_timestep = -1

           
            else:

                (
                    candidate_x,
                    snapped_timestep,
                ) = reconstructor.reconstruct(
                    x,
                    requested_t,
                )

            
            (
                _,
                candidate_confidence,
            ) = classifier_outputs(
                classifier,
                candidate_x,
            )

            candidate_t = torch.full(
                (
                    x.shape[0],
                ),
                int(
                    snapped_timestep
                ),
                dtype=torch.long,
                device=DEVICE,
            )

            
            if best_confidence is None:

                best_confidence = (
                    candidate_confidence.clone()
                )

                best_x = (
                    candidate_x.clone()
                )

                best_timestep = (
                    candidate_t.clone()
                )

            
            else:

                better = (
                    candidate_confidence
                    >
                    best_confidence
                )

                best_confidence = (
                    torch.where(
                        better,
                        candidate_confidence,
                        best_confidence,
                    )
                )

                best_x = (
                    torch.where(
                        better.view(
                            -1,
                            1,
                            1,
                            1,
                        ),
                        candidate_x,
                        best_x,
                    )
                )

                best_timestep = (
                    torch.where(
                        better,
                        candidate_t,
                        best_timestep,
                    )
                )

    

        if DEVICE.type == "cuda":

            torch.cuda.synchronize()

        time_end = (
            time.perf_counter()
        )

        batch_time_ms = (
            time_end
            -
            time_start
        ) * 1000.0

        total_candidate_selection_ms += (
            batch_time_ms
        )

        time_per_image_ms = (
            batch_time_ms
            /
            x.shape[0]
        )

        timing_per_image.extend(
            [
                time_per_image_ms
            ]
            *
            x.shape[0]
        )

       
        (
            prediction_after,
            confidence_after,
        ) = classifier_outputs(
            classifier,
            best_x,
        )

        
        before_correct_mask = (
            prediction_before
            ==
            y
        )

        after_correct_mask = (
            prediction_after
            ==
            y
        )

        correct_before += int(
            before_correct_mask
            .sum()
            .item()
        )

        correct_after += int(
            after_correct_mask
            .sum()
            .item()
        )

       
        recovered_mask = (
            ~before_correct_mask
            &
            after_correct_mask
        )

        damaged_mask = (
            before_correct_mask
            &
            ~after_correct_mask
        )

        stayed_wrong_mask = (
            ~before_correct_mask
            &
            ~after_correct_mask
        )

        stayed_correct_mask = (
            before_correct_mask
            &
            after_correct_mask
        )

        recovered_wrong += int(
            recovered_mask
            .sum()
            .item()
        )

        damaged_correct += int(
            damaged_mask
            .sum()
            .item()
        )

        stayed_wrong += int(
            stayed_wrong_mask
            .sum()
            .item()
        )

        stayed_correct += int(
            stayed_correct_mask
            .sum()
            .item()
        )

        
        prediction_before_np = (
            prediction_before
            .detach()
            .cpu()
            .numpy()
        )

        prediction_after_np = (
            prediction_after
            .detach()
            .cpu()
            .numpy()
        )

        confidence_before_np = (
            confidence_before
            .detach()
            .cpu()
            .numpy()
        )

        confidence_after_np = (
            confidence_after
            .detach()
            .cpu()
            .numpy()
        )

        best_timestep_np = (
            best_timestep
            .detach()
            .cpu()
            .numpy()
        )

        best_x_cpu = (
            best_x
            .detach()
            .cpu()
        )

        
        for i in range(
            x.shape[0]
        ):

            timestep_value = int(
                best_timestep_np[
                    i
                ]
            )

            timestep_key = (
                "skip"
                if timestep_value == -1
                else str(
                    timestep_value
                )
            )

            chosen_t_counts[
                timestep_key
            ] = (
                chosen_t_counts.get(
                    timestep_key,
                    0,
                )
                +
                1
            )

            was_correct = (
                int(
                    prediction_before_np[
                        i
                    ]
                )
                ==
                labels[
                    i
                ]
            )

            is_correct = (
                int(
                    prediction_after_np[
                        i
                    ]
                )
                ==
                labels[
                    i
                ]
            )

            if (
                not was_correct
                and
                is_correct
            ):

                outcome = (
                    "recovered"
                )

            elif (
                was_correct
                and
                not is_correct
            ):

                outcome = (
                    "damaged"
                )

            elif was_correct:

                outcome = (
                    "stayed_correct"
                )

            else:

                outcome = (
                    "stayed_wrong"
                )

            output_path = os.path.join(
                output_image_dir,
                filenames[
                    i
                ],
            )

            if save_images:

                save_png(
                    best_x_cpu[
                        i
                    ],
                    output_path,
                )

            extraction_row = (
                metadata.get(
                    filenames[
                        i
                    ],
                    {},
                )
            )

            rows.append(
                {

                    "model":
                        model_key,

                    "condition":
                        condition,

                    "sample_id":
                        sample_ids[
                            i
                        ],

                    "filename":
                        filenames[
                            i
                        ],

                    "label":
                        labels[
                            i
                        ],

                    "prediction_before":
                        int(
                            prediction_before_np[
                                i
                            ]
                        ),

                    "confidence_before":
                        float(
                            confidence_before_np[
                                i
                            ]
                        ),

                    "correct_before":
                        int(
                            was_correct
                        ),

                    "chosen_t":
                        timestep_key,

                    "prediction_after":
                        int(
                            prediction_after_np[
                                i
                            ]
                        ),

                    "confidence_after":
                        float(
                            confidence_after_np[
                                i
                            ]
                        ),

                    "correct_after":
                        int(
                            is_correct
                        ),

                    "outcome":
                        outcome,

                    "ours_flag":
                        extraction_row.get(
                            "ours_flag",
                            "",
                        ),

                    "js_flag":
                        extraction_row.get(
                            "js_flag",
                            "",
                        ),

                    "ours_js_flag":
                        extraction_row.get(
                            "ours_js_flag",
                            "",
                        ),

                    "input_path":
                        batch_paths[
                            i
                        ],

                    "output_path":
                        (
                            output_path
                            if save_images
                            else ""
                        ),
                }
            )

   
    wrong_before = (
        num_images
        -
        correct_before
    )

    wrong_after = (
        num_images
        -
        correct_after
    )

    accuracy_before = (
        100.0
        *
        correct_before
        /
        num_images
    )

    accuracy_after = (
        100.0
        *
        correct_after
        /
        num_images
    )

    accuracy_improvement = (
        accuracy_after
        -
        accuracy_before
    )

    recovery_rate = (

        100.0
        *
        recovered_wrong
        /
        wrong_before

        if wrong_before > 0

        else 0.0
    )

    degradation_rate = (

        100.0
        *
        damaged_correct
        /
        correct_before

        if correct_before > 0

        else 0.0
    )

  
    minimum_ms = float(
        np.min(
            timing_per_image
        )
    )

    average_ms = float(
        np.mean(
            timing_per_image
        )
    )

    maximum_ms = float(
        np.max(
            timing_per_image
        )
    )

    
    metadata_output_path = (
        os.path.join(
            output_dir,
            "reconstruction_metadata.csv",
        )
    )

    fieldnames = [

        "model",

        "condition",

        "sample_id",

        "filename",

        "label",

        "prediction_before",

        "confidence_before",

        "correct_before",

        "chosen_t",

        "prediction_after",

        "confidence_after",

        "correct_after",

        "outcome",

        "ours_flag",

        "js_flag",

        "ours_js_flag",

        "input_path",

        "output_path",
    ]

    with open(
        metadata_output_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

   
    summary = {

        "model":
            model_key,

        "condition":
            condition,

        "n_images":
            int(
                num_images
            ),

        "correct_before":
            int(
                correct_before
            ),

        "wrong_before":
            int(
                wrong_before
            ),

        "correct_after":
            int(
                correct_after
            ),

        "wrong_after":
            int(
                wrong_after
            ),

        "accuracy_before_percent":
            float(
                accuracy_before
            ),

        "accuracy_after_percent":
            float(
                accuracy_after
            ),

        "accuracy_improvement_points":
            float(
                accuracy_improvement
            ),

        "recovered_wrong":
            int(
                recovered_wrong
            ),

        "damaged_correct":
            int(
                damaged_correct
            ),

        "stayed_wrong":
            int(
                stayed_wrong
            ),

        "stayed_correct":
            int(
                stayed_correct
            ),

        "recovery_rate_of_wrong_percent":
            float(
                recovery_rate
            ),

        "degradation_rate_of_correct_percent":
            float(
                degradation_rate
            ),

        "chosen_t_counts":
            chosen_t_counts,

        "timing": {

            "total_candidate_selection_ms":
                float(
                    total_candidate_selection_ms
                ),

            "min_ms_per_image":
                float(
                    minimum_ms
                ),

            "avg_ms_per_image":
                float(
                    average_ms
                ),

            "max_ms_per_image":
                float(
                    maximum_ms
                ),
        },

        "t_candidates_requested":
            [

                (
                    "skip"
                    if t is None
                    else int(
                        t
                    )
                )

                for t in candidates
            ],

        "input_directory":
            input_dir,

        "output_directory":
            output_dir,

        "saved_images":
            bool(
                save_images
            ),
    }

    summary_path = os.path.join(
        output_dir,
        "summary.json",
    )

    with open(
        summary_path,
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

   
    print(
        "\n"
        + "-" * 80
    )

    print(
        f"{model_key.upper()} | "
        f"{condition} RESULT"
    )

    print(
        "-" * 80
    )

    print(
        f"Images             : "
        f"{num_images:,}"
    )

    print(
        f"Accuracy before    : "
        f"{accuracy_before:.2f}%"
    )

    print(
        f"Accuracy after     : "
        f"{accuracy_after:.2f}%"
    )

    print(
        f"Accuracy change    : "
        f"{accuracy_improvement:+.2f} points"
    )

    print(
        f"Recovered wrong    : "
        f"{recovered_wrong}/"
        f"{wrong_before}"
    )

    print(
        f"Recovery rate      : "
        f"{recovery_rate:.2f}%"
    )

    print(
        f"Damaged correct    : "
        f"{damaged_correct}/"
        f"{correct_before}"
    )

    print(
        f"Degradation rate   : "
        f"{degradation_rate:.2f}%"
    )

    print(
        f"Chosen timesteps   : "
        f"{chosen_t_counts}"
    )

    print(
        "Timing min/avg/max : "
        f"{minimum_ms:.2f} / "
        f"{average_ms:.2f} / "
        f"{maximum_ms:.2f} ms/image"
    )

    print(
        f"Metadata           : "
        f"{metadata_output_path}"
    )

    print(
        f"Summary            : "
        f"{summary_path}"
    )

    return summary



def validate_inputs(
    selected_models,
    selected_conditions,
    ddpm_dir,
):

    print(
        "\nChecking required inputs..."
    )

    missing = []

    if not os.path.isdir(
        ddpm_dir
    ):

        missing.append(
            ddpm_dir
        )

    
    possible_weight_paths = [

        os.path.join(
            ddpm_dir,
            "diffusion_pytorch_model.safetensors",
        ),

        os.path.join(
            ddpm_dir,
            "diffusion_pytorch_model.bin",
        ),

        os.path.join(
            ddpm_dir,
            "unet",
            "diffusion_pytorch_model.safetensors",
        ),

        os.path.join(
            ddpm_dir,
            "unet",
            "diffusion_pytorch_model.bin",
        ),
    ]

    if not any(
        os.path.isfile(
            path
        )
        for path
        in possible_weight_paths
    ):

        missing.append(
            "DDPM model weights"
        )

  
    for model in selected_models:

        checkpoint = MODEL_PATHS[
            model
        ]

        if not os.path.isfile(
            checkpoint
        ):

            missing.append(
                checkpoint
            )

        for condition in selected_conditions:

            directory = (
                get_input_image_directory(
                    model,
                    condition,
                )
            )

            if not os.path.isdir(
                directory
            ):

                missing.append(
                    directory
                )

    if missing:

        print(
            "\nMissing required inputs:"
        )

        for path in missing[
            :100
        ]:

            print(
                f"  {path}"
            )

        raise RuntimeError(
            "\nRequired reconstruction inputs "
            "are missing."
        )

    print(
        "All required reconstruction inputs exist."
    )



def save_global_summary(
    summaries,
):

    Path(
        OUTPUT_ROOT
    ).mkdir(
        parents=True,
        exist_ok=True,
    )

    json_path = os.path.join(
        OUTPUT_ROOT,
        "reconstruction_summary.json",
    )

    csv_path = os.path.join(
        OUTPUT_ROOT,
        "reconstruction_summary.csv",
    )

   
    with open(
        json_path,
        "w",
    ) as f:

        json.dump(
            summaries,
            f,
            indent=2,
        )

 
    fields = [

        "model",

        "condition",

        "n_images",

        "accuracy_before_percent",

        "accuracy_after_percent",

        "accuracy_improvement_points",

        "correct_before",

        "wrong_before",

        "correct_after",

        "wrong_after",

        "recovered_wrong",

        "damaged_correct",

        "recovery_rate_of_wrong_percent",

        "degradation_rate_of_correct_percent",

        "avg_ms_per_image",
    ]

    with open(
        csv_path,
        "w",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()

        for summary in summaries:

            writer.writerow(
                {

                    "model":
                        summary.get(
                            "model"
                        ),

                    "condition":
                        summary.get(
                            "condition"
                        ),

                    "n_images":
                        summary.get(
                            "n_images"
                        ),

                    "accuracy_before_percent":
                        summary.get(
                            "accuracy_before_percent"
                        ),

                    "accuracy_after_percent":
                        summary.get(
                            "accuracy_after_percent"
                        ),

                    "accuracy_improvement_points":
                        summary.get(
                            "accuracy_improvement_points"
                        ),

                    "correct_before":
                        summary.get(
                            "correct_before"
                        ),

                    "wrong_before":
                        summary.get(
                            "wrong_before"
                        ),

                    "correct_after":
                        summary.get(
                            "correct_after"
                        ),

                    "wrong_after":
                        summary.get(
                            "wrong_after"
                        ),

                    "recovered_wrong":
                        summary.get(
                            "recovered_wrong"
                        ),

                    "damaged_correct":
                        summary.get(
                            "damaged_correct"
                        ),

                    "recovery_rate_of_wrong_percent":
                        summary.get(
                            "recovery_rate_of_wrong_percent"
                        ),

                    "degradation_rate_of_correct_percent":
                        summary.get(
                            "degradation_rate_of_correct_percent"
                        ),

                    "avg_ms_per_image":
                        summary.get(
                            "timing",
                            {},
                        ).get(
                            "avg_ms_per_image"
                        ),
                }
            )

    return (
        csv_path,
        json_path,
    )



def print_final_summary(
    summaries,
):

    print(
        "\n"
        + "=" * 155
    )

    print(
        "FINAL GTSRB OURS+JS -> DDPM RECONSTRUCTION SUMMARY"
    )

    print(
        "=" * 155
    )

    print(
        f"{'Model':<14}"
        f"{'Condition':<20}"
        f"{'N':>8}"
        f"{'Before':>11}"
        f"{'After':>11}"
        f"{'Delta':>11}"
        f"{'Recovered':>12}"
        f"{'Damaged':>11}"
        f"{'Avg ms/img':>14}"
    )

    print(
        "-" * 155
    )

    for summary in summaries:

        if (
            "accuracy_before_percent"
            not in summary
        ):

            continue

        print(
            f"{summary['model']:<14}"
            f"{summary['condition']:<20}"
            f"{summary['n_images']:>8d}"
            f"{summary['accuracy_before_percent']:>10.2f}%"
            f"{summary['accuracy_after_percent']:>10.2f}%"
            f"{summary['accuracy_improvement_points']:>+10.2f}"
            f"{summary['recovered_wrong']:>12d}"
            f"{summary['damaged_correct']:>11d}"
            f"{summary['timing']['avg_ms_per_image']:>14.2f}"
        )

    print(
        "=" * 155
    )



def main():

    parser = argparse.ArgumentParser(
        description=(
            "Run DDPM reconstruction on GTSRB "
            "samples flagged by the final OURS+JS detector."
        )
    )

  
    parser.add_argument(
        "--ddpm_dir",
        required=True,
        help=(
            "Directory containing the existing "
            "clean-trained GTSRB DDPM."
        ),
    )

    
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
        "--conditions",
        nargs="+",
        choices=CONDITIONS,
        default=CONDITIONS,
    )

  
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help=(
            "Number of DDPM inference steps."
        ),
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=32,
        help=(
            "Reconstruction batch size."
        ),
    )

    parser.add_argument(
        "--t_candidates",
        type=str,
        default=(
            "skip,20,40,80,120,160"
        ),
        help=(
            "Comma-separated partial diffusion "
            "strength candidates."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=123,
    )

   
    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    parser.add_argument(
        "--no_amp",
        action="store_true",
    )

    parser.add_argument(
        "--no_ema",
        action="store_true",
    )

    parser.add_argument(
        "--no_save_images",
        action="store_true",
    )

    args = parser.parse_args()

   
    if args.batch <= 0:

        raise ValueError(
            "--batch must be > 0."
        )

    if args.steps <= 0:

        raise ValueError(
            "--steps must be > 0."
        )

   
    set_seed(
        args.seed
    )

  
    if args.model == "all":

        selected_models = list(
            MODELS
        )

    else:

        selected_models = [
            args.model
        ]

    selected_conditions = list(
        args.conditions
    )

    candidates = parse_t_candidates(
        args.t_candidates
    )

    print(
        "=" * 120
    )

    print(
        "GTSRB — FINAL OURS+JS SUSPICIOUS SAMPLES "
        "-> DDPM RECONSTRUCTION"
    )

    print(
        "=" * 120
    )

    print(
        f"Device       : {DEVICE}"
    )

    if DEVICE.type == "cuda":

        print(
            f"GPU          : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"Models       : "
        f"{selected_models}"
    )

    print(
        f"Conditions   : "
        f"{selected_conditions}"
    )

    print(
        f"Input root   : "
        f"{INPUT_ROOT}"
    )

    print(
        f"Output root  : "
        f"{OUTPUT_ROOT}"
    )

    print(
        f"DDPM dir     : "
        f"{args.ddpm_dir}"
    )

    print(
        f"Steps        : "
        f"{args.steps}"
    )

    print(
        f"Batch        : "
        f"{args.batch}"
    )

    print(
        "Candidates   : "
        + str(
            [
                (
                    "skip"
                    if t is None
                    else t
                )
                for t in candidates
            ]
        )
    )

    print(
        f"Seed         : "
        f"{args.seed}"
    )

    print(
        f"EMA          : "
        f"{'OFF' if args.no_ema else 'ON'}"
    )

    print(
        f"AMP          : "
        f"{'OFF' if args.no_amp else 'ON'}"
    )

    print(
        "Selection    : maximum classifier confidence"
    )

    print(
        "Label usage  : evaluation ONLY; "
        "never candidate selection"
    )

    print(
        "=" * 120
    )

    
    validate_inputs(
        selected_models,
        selected_conditions,
        args.ddpm_dir,
    )

    
    reconstructor = (
        DDPMPartialReconstructor(
            ddpm_dir=args.ddpm_dir,
            steps=args.steps,
            seed=args.seed,
            use_amp=(
                not args.no_amp
            ),
            use_ema=(
                not args.no_ema
            ),
        )
    )

    summaries = []

    for model_key in selected_models:

        classifier = (
            load_classifier(
                model_key
            )
        )

        for condition in selected_conditions:

            summary = (
                reconstruct_condition(
                    model_key=model_key,
                    condition=condition,
                    classifier=classifier,
                    reconstructor=reconstructor,
                    candidates=candidates,
                    batch_size=args.batch,
                    overwrite=args.overwrite,
                    save_images=(
                        not args.no_save_images
                    ),
                )
            )

            summaries.append(
                summary
            )

        del classifier

        if DEVICE.type == "cuda":

            torch.cuda.empty_cache()

   
    (
        csv_path,
        json_path,
    ) = save_global_summary(
        summaries
    )

   
    print_final_summary(
        summaries
    )

    print(
        "\n"
        + "=" * 120
    )

    print(
        "DONE"
    )

    print(
        "=" * 120
    )

    print(
        f"CSV summary:\n"
        f"  {csv_path}"
    )

    print(
        f"\nJSON summary:\n"
        f"  {json_path}"
    )

    print(
        "\nReconstructed images:\n"
        f"  {OUTPUT_ROOT}/"
        "<model>/<condition>/images/"
    )


if __name__ == "__main__":

    main()