

import os
import json
import argparse
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image, ImageOps

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.transforms import InterpolationMode

from tqdm import tqdm



ROOT = (
    "/home/Traffic_Signs_2"
)

CLEAN_DIR = os.path.join(
    ROOT,
    "Clean Dataset",
    
)

QR_DIR = os.path.join(
    ROOT,
    "attacked"
)

MODEL_DIR = os.path.join(
    ROOT,
    "physical_models"
)

OUT_ROOT = os.path.join(
    ROOT,
    "physical_pipeline",
    "js_signal"
)


IMG_SIZE = 224
PERTURB_SIZE = 208

BATCH_SIZE = 16
NUM_WORKERS = 4

MEAN = (
    0.485,
    0.456,
    0.406
)

STD = (
    0.229,
    0.224,
    0.225
)


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)




MODELS = {

    "mobilenet": {
        "checkpoint": os.path.join(
            MODEL_DIR,
            "mobilenetv3_comma_clean_best.pth"
        )
    },

    "convnext": {
        "checkpoint": os.path.join(
            MODEL_DIR,
            "convnext_tiny_comma_clean_best.pth"
        )
    },

    "efficientnet": {
        "checkpoint": os.path.join(
            MODEL_DIR,
            "efficientnet_v2_s_comma_clean_best.pth"
        )
    }
}


CONDITIONS = {

    "clean": CLEAN_DIR,

    "qr": QR_DIR
}



class PadToSquare:

    def __call__(self, img):

        w, h = img.size

        side = max(
            w,
            h
        )

        left = (
            side - w
        ) // 2

        right = (
            side - w - left
        )

        top = (
            side - h
        ) // 2

        bottom = (
            side - h - top
        )

        return ImageOps.expand(
            img,
            border=(
                left,
                top,
                right,
                bottom
            ),
            fill=0
        )




class FixedClassDataset(Dataset):

    def __init__(
        self,
        root,
        class_to_idx
    ):

        self.samples = []

        valid_ext = (
            ".jpg",
            ".jpeg",
            ".png",
            ".bmp",
            ".webp"
        )

        for class_name, class_id in sorted(
            class_to_idx.items(),
            key=lambda x: x[1]
        ):

            folder = os.path.join(
                root,
                class_name
            )

            if not os.path.isdir(
                folder
            ):

                raise FileNotFoundError(
                    folder
                )


            filenames = sorted([
                f
                for f in os.listdir(
                    folder
                )
                if f.lower().endswith(
                    valid_ext
                )
            ])


            for filename in filenames:

                path = os.path.join(
                    folder,
                    filename
                )

                self.samples.append(
                    (
                        path,
                        int(class_id)
                    )
                )


    def __len__(self):

        return len(
            self.samples
        )


    def __getitem__(
        self,
        idx
    ):

        path, label = (
            self.samples[idx]
        )


        image = Image.open(
            path
        ).convert(
            "RGB"
        )


        image = PadToSquare()(
            image
        )


        image = image.resize(
            (
                IMG_SIZE,
                IMG_SIZE
            ),
           
        )


        x = transforms.functional.to_tensor(
            image
        )


        return (
            x,
            label,
            path
        )



def normalize(
    x
):

    mean = torch.tensor(
        MEAN,
        dtype=x.dtype,
        device=x.device
    ).view(
        1,
        3,
        1,
        1
    )


    std = torch.tensor(
        STD,
        dtype=x.dtype,
        device=x.device
    ).view(
        1,
        3,
        1,
        1
    )


    return (
        x - mean
    ) / std




def build_model(
    model_key,
    num_classes
):

    if model_key == "mobilenet":

        model = (
            models.mobilenet_v3_large(
                weights=None
            )
        )

        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features,
            num_classes
        )


    elif model_key == "convnext":

        model = (
            models.convnext_tiny(
                weights=None
            )
        )

        model.classifier[2] = nn.Linear(
            model.classifier[2].in_features,
            num_classes
        )


    elif model_key == "efficientnet":

        model = (
            models.efficientnet_v2_s(
                weights=None
            )
        )

        model.classifier[1] = nn.Linear(
            model.classifier[1].in_features,
            num_classes
        )


    else:

        raise ValueError(
            model_key
        )


    return model



def load_model(
    model_key
):

    checkpoint_path = (
        MODELS[
            model_key
        ][
            "checkpoint"
        ]
    )


    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE,
        weights_only=False
    )


    classes = checkpoint[
        "classes"
    ]


    class_to_idx = checkpoint[
        "class_to_idx"
    ]


    model = build_model(
        model_key,
        len(classes)
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )


    model = model.to(
        DEVICE
    )


    model.eval()


    return (
        model,
        classes,
        class_to_idx
    )




def js_divergence(
    p,
    q,
    eps=1e-12
):

    p = p.clamp_min(
        eps
    )

    q = q.clamp_min(
        eps
    )


    m = (
        0.5
        *
        (
            p + q
        )
    )


    kl_pm = (
        p
        *
        (
            torch.log(p)
            -
            torch.log(m)
        )
    ).sum(
        dim=1
    )


    kl_qm = (
        q
        *
        (
            torch.log(q)
            -
            torch.log(m)
        )
    ).sum(
        dim=1
    )


    return (
        0.5
        *
        (
            kl_pm
            +
            kl_qm
        )
    )




def entropy(
    p,
    eps=1e-12
):

    p = p.clamp_min(
        eps
    )


    return -(
        p
        *
        torch.log(
            p
        )
    ).sum(
        dim=1
    )




def resize_perturbation(
    x
):

    x_small = F.interpolate(
        x,
        size=(
            PERTURB_SIZE,
            PERTURB_SIZE
        ),
        mode="bilinear",
        align_corners=False
    )


    x_back = F.interpolate(
        x_small,
        size=(
            IMG_SIZE,
            IMG_SIZE
        ),
        mode="bilinear",
        align_corners=False
    )


    return x_back




@torch.no_grad()
def evaluate_condition(
    model_key,
    model,
    classes,
    class_to_idx,
    condition,
    image_dir
):

    out_dir = os.path.join(
        OUT_ROOT,
        model_key,
        condition
    )


    Path(
        out_dir
    ).mkdir(
        parents=True,
        exist_ok=True
    )


    dataset = FixedClassDataset(
        image_dir,
        class_to_idx
    )


    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(
            DEVICE.type
            ==
            "cuda"
        )
    )


    all_js = []

    all_labels = []

    all_pred_base = []

    all_pred_trans = []

    all_conf_base = []

    all_conf_trans = []

    all_changed = []

    all_entropy_base = []

    all_entropy_trans = []

    all_filenames = []


    for (
        x,
        y,
        paths
    ) in tqdm(
        loader,
        desc=(
            f"{model_key}/{condition}"
        )
    ):


        x = x.to(
            DEVICE,
            non_blocking=True
        )


        y = y.to(
            DEVICE,
            non_blocking=True
        )


        logits_base = model(
            normalize(
                x
            )
        )


        probs_base = F.softmax(
            logits_base,
            dim=1
        )


        conf_base, pred_base = (
            probs_base.max(
                dim=1
            )
        )



        x_trans = (
            resize_perturbation(
                x
            )
        )


        logits_trans = model(
            normalize(
                x_trans
            )
        )


        probs_trans = F.softmax(
            logits_trans,
            dim=1
        )


        conf_trans, pred_trans = (
            probs_trans.max(
                dim=1
            )
        )



        js = js_divergence(
            probs_base,
            probs_trans
        )


        ent_base = entropy(
            probs_base
        )


        ent_trans = entropy(
            probs_trans
        )


        changed = (
            pred_base
            !=
            pred_trans
        )




        all_js.append(
            js.cpu().numpy()
        )


        all_labels.append(
            y.cpu().numpy()
        )


        all_pred_base.append(
            pred_base.cpu().numpy()
        )


        all_pred_trans.append(
            pred_trans.cpu().numpy()
        )


        all_conf_base.append(
            conf_base.cpu().numpy()
        )


        all_conf_trans.append(
            conf_trans.cpu().numpy()
        )


        all_changed.append(
            changed.cpu().numpy()
        )


        all_entropy_base.append(
            ent_base.cpu().numpy()
        )


        all_entropy_trans.append(
            ent_trans.cpu().numpy()
        )


        all_filenames.extend(
            list(
                paths
            )
        )



    js_np = np.concatenate(
        all_js
    )


    labels_np = np.concatenate(
        all_labels
    )


    pred_base_np = np.concatenate(
        all_pred_base
    )


    pred_trans_np = np.concatenate(
        all_pred_trans
    )


    conf_base_np = np.concatenate(
        all_conf_base
    )


    conf_trans_np = np.concatenate(
        all_conf_trans
    )


    changed_np = np.concatenate(
        all_changed
    ).astype(bool)


    entropy_base_np = np.concatenate(
        all_entropy_base
    )


    entropy_trans_np = np.concatenate(
        all_entropy_trans
    )


    filenames_np = np.asarray(
        all_filenames,
        dtype=object
    )


    n = len(
        labels_np
    )


 

    np.save(
        os.path.join(
            out_dir,
            "js_divergence.npy"
        ),
        js_np
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
            "pred_base.npy"
        ),
        pred_base_np
    )


    np.save(
        os.path.join(
            out_dir,
            "pred_transformed.npy"
        ),
        pred_trans_np
    )


    np.save(
        os.path.join(
            out_dir,
            "confidence_base.npy"
        ),
        conf_base_np
    )


    np.save(
        os.path.join(
            out_dir,
            "confidence_transformed.npy"
        ),
        conf_trans_np
    )


    np.save(
        os.path.join(
            out_dir,
            "prediction_changed.npy"
        ),
        changed_np
    )


    np.save(
        os.path.join(
            out_dir,
            "entropy_base.npy"
        ),
        entropy_base_np
    )


    np.save(
        os.path.join(
            out_dir,
            "entropy_transformed.npy"
        ),
        entropy_trans_np
    )


    np.save(
        os.path.join(
            out_dir,
            "filenames.npy"
        ),
        filenames_np,
        allow_pickle=True
    )



    base_acc = (
        pred_base_np
        ==
        labels_np
    ).mean()


    transformed_acc = (
        pred_trans_np
        ==
        labels_np
    ).mean()


    changed_rate = (
        changed_np.mean()
    )


    metadata = {

        "model":
            model_key,

        "condition":
            condition,

        "source_dir":
            image_dir,

        "num_images":
            int(n),

        "num_classes":
            int(
                len(classes)
            ),

        "base_image_size":
            IMG_SIZE,

        "perturb_size":
            PERTURB_SIZE,

        "transformation":
            (
                "bilinear resize "
                "224->208->224"
            ),

        "signal":
            (
                "Jensen-Shannon divergence "
                "between full softmax "
                "probability distributions"
            ),

        "base_accuracy_percent":
            float(
                base_acc
                *
                100.0
            ),

        "transformed_accuracy_percent":
            float(
                transformed_acc
                *
                100.0
            ),

        "prediction_changed_percent":
            float(
                changed_rate
                *
                100.0
            ),

        "js_mean":
            float(
                js_np.mean()
            ),

        "js_median":
            float(
                np.median(
                    js_np
                )
            ),

        "js_p95":
            float(
                np.percentile(
                    js_np,
                    95
                )
            ),

        "js_max":
            float(
                js_np.max()
            )
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



    print(
        f"\n[{model_key}/{condition}]"
    )


    print(
        f"Images             : {n}"
    )


    print(
        f"Base accuracy      : "
        f"{base_acc*100:.2f}%"
    )


    print(
        f"Transformed acc    : "
        f"{transformed_acc*100:.2f}%"
    )


    print(
        f"Prediction changed : "
        f"{changed_rate*100:.2f}%"
    )


    print(
        f"JS mean            : "
        f"{js_np.mean():.8f}"
    )


    print(
        f"JS median          : "
        f"{np.median(js_np):.8f}"
    )


    print(
        f"JS p95             : "
        f"{np.percentile(js_np,95):.8f}"
    )


    print(
        f"JS max             : "
        f"{js_np.max():.8f}"
    )


    print(
        f"Saved -> {out_dir}"
    )



def run_model(
    model_key
):

    (
        model,
        classes,
        class_to_idx
    ) = load_model(
        model_key
    )


    print("\n" + "#" * 90)

    print(
        f"MODEL: {model_key.upper()}"
    )

    print("#" * 90)


    for (
        condition,
        image_dir
    ) in CONDITIONS.items():

        evaluate_condition(
            model_key,
            model,
            classes,
            class_to_idx,
            condition,
            image_dir
        )




def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--model",
        choices=[
            "all",
            "mobilenet",
            "convnext",
            "efficientnet"
        ],
        default="all"
    )


    args = parser.parse_args()


    selected = (

        list(
            MODELS.keys()
        )

        if args.model == "all"

        else [
            args.model
        ]
    )


    Path(
        OUT_ROOT
    ).mkdir(
        parents=True,
        exist_ok=True
    )


    print("=" * 100)

    print(
        "PHYSICAL JS CONSISTENCY EXTRACTION — STEP 4"
    )

    print("=" * 100)


    print(
        f"Device       : {DEVICE}"
    )


    print(
        f"Resize       : "
        f"{IMG_SIZE}->{PERTURB_SIZE}->{IMG_SIZE}"
    )


    print(
        f"Models       : {selected}"
    )


    for model_key in selected:

        run_model(
            model_key
        )


    print("\n" + "=" * 100)

    print(
        "STEP 4 COMPLETE"
    )

    print("=" * 100)


if __name__ == "__main__":

    main()
