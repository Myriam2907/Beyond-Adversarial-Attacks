
import os
import json
import math
import random
import argparse
from pathlib import Path
from contextlib import nullcontext

import numpy as np
from PIL import Image, ImageFile

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from diffusers import UNet2DModel, DDPMScheduler

ImageFile.LOAD_TRUNCATED_IMAGES = True



def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_image_file(path: Path):
    return path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".webp"}


class ImageOnlyRecursiveDataset(Dataset):
    """
    Recursively scans a directory and loads all image files.
    Useful when GTSRB training images are stored in class subfolders.
    """
    def __init__(self, root_dir, image_size=64):
        self.root_dir = Path(root_dir)
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.root_dir}")

        self.image_paths = sorted(
            [p for p in self.root_dir.rglob("*") if p.is_file() and is_image_file(p)]
        )

        if len(self.image_paths) == 0:
            raise RuntimeError(f"No image files found under: {self.root_dir}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),                         # [0,1]
            transforms.Normalize([0.5, 0.5, 0.5],         # -> [-1,1]
                                 [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img


class EMAHelper:
    """
    Simple EMA for model weights.
    Saves a shadow copy of floating tensors in state_dict().
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = decay
        self.shadow = {}

        for k, v in model.state_dict().items():
            if torch.is_floating_point(v):
                self.shadow[k] = v.detach().cpu().clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        current_state = model.state_dict()
        for k, v in current_state.items():
            if k in self.shadow and torch.is_floating_point(v):
                self.shadow[k].mul_(self.decay).add_(v.detach().cpu(), alpha=(1.0 - self.decay))

    @torch.no_grad()
    def copy_to_model(self, model: nn.Module):
        model_state = model.state_dict()
        for k in model_state.keys():
            if k in self.shadow:
                model_state[k].copy_(self.shadow[k].to(model_state[k].device))

    def state_dict(self):
        return {
            "decay": self.decay,
            "shadow": self.shadow
        }



def build_unet(image_size: int):
    """
    64x64 RGB DDPM UNet.
    """
    model = UNet2DModel(
        sample_size=image_size,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    return model



def train(args):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = args.amp and device.type == "cuda"

    print("=" * 100)
    print("TRAINING GTSRB DDPM")
    print("=" * 100)
    print(f"Data dir        : {args.data_dir}")
    print(f"Save dir        : {args.save_dir}")
    print(f"Device          : {device}")
    print(f"AMP enabled     : {use_amp}")
    print(f"Image size      : {args.image_size}")
    print(f"Epochs          : {args.epochs}")
    print(f"Batch size      : {args.batch_size}")
    print(f"Learning rate   : {args.lr}")
    print(f"EMA decay       : {args.ema_decay}")
    print("=" * 100)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset = ImageOnlyRecursiveDataset(
        root_dir=args.data_dir,
        image_size=args.image_size
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    print(f"Found {len(dataset)} training images.")

    model = build_unet(args.image_size).to(device)

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type="epsilon",
        clip_sample=True,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    ema = EMAHelper(model, decay=args.ema_decay)

    history = {
        "config": vars(args),
        "epoch_losses": [],
        "num_images": len(dataset),
    }

    global_step = 0
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}/{args.epochs}", leave=True)

        for clean_images in pbar:
            clean_images = clean_images.to(device, non_blocking=True)

            noise = torch.randn_like(clean_images)
            bsz = clean_images.size(0)

            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=device,
                dtype=torch.long,
            )

            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            optimizer.zero_grad(set_to_none=True)

            amp_ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_amp else nullcontext()
            with amp_ctx:
                noise_pred = model(noisy_images, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)

            scaler.scale(loss).backward()

            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            scaler.step(optimizer)
            scaler.update()

            ema.update(model)

            running_loss += loss.item()
            num_batches += 1
            global_step += 1

            pbar.set_postfix(loss=f"{loss.item():.6f}")

        avg_loss = running_loss / max(num_batches, 1)
        history["epoch_losses"].append({
            "epoch": epoch,
            "avg_loss": avg_loss,
            "global_step": global_step,
        })

        print(f"[Epoch {epoch:03d}] avg loss = {avg_loss:.6f}")

        
        latest_dir = save_dir / "latest"
        latest_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(latest_dir, safe_serialization=True)
        noise_scheduler.save_pretrained(latest_dir)

        torch.save(
            ema.state_dict(),
            latest_dir / "ema_shadow.pt"
        )

        with open(latest_dir / "training_history.json", "w") as f:
            json.dump(history, f, indent=2)

        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_dir = save_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(best_dir, safe_serialization=True)
            noise_scheduler.save_pretrained(best_dir)

            torch.save(
                ema.state_dict(),
                best_dir / "ema_shadow.pt"
            )

            with open(best_dir / "training_history.json", "w") as f:
                json.dump(history, f, indent=2)

            print(f"Saved new best checkpoint to: {best_dir}")

    
    final_dir = save_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(final_dir, safe_serialization=True)
    noise_scheduler.save_pretrained(final_dir)
    torch.save(ema.state_dict(), final_dir / "ema_shadow.pt")

    with open(final_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    
    model.save_pretrained(save_dir, safe_serialization=True)
    noise_scheduler.save_pretrained(save_dir)
    torch.save(ema.state_dict(), save_dir / "ema_shadow.pt")

    with open(save_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print("=" * 100)
    print("TRAINING FINISHED")
    print("=" * 100)
    print(f"Best loss       : {best_loss:.6f}")
    print(f"Main checkpoint : {save_dir}")
    print(f"Files created:")
    print(f"  {save_dir / 'config.json'}")
    print(f"  {save_dir / 'diffusion_pytorch_model.safetensors'}")
    print(f"  {save_dir / 'scheduler_config.json'}")
    print(f"  {save_dir / 'ema_shadow.pt'}")
    print(f"  {save_dir / 'training_history.json'}")
    print("=" * 100)



def parse_args():
    parser = argparse.ArgumentParser(description="Train a 64x64 DDPM on clean GTSRB images.")

    parser.add_argument("--data_dir", type=str, required=True,
                        help="Root directory containing clean GTSRB training images (recursive scan).")
    parser.add_argument("--save_dir", type=str, default="./ddpm_gtsrb_64x64",
                        help="Output directory for the trained DDPM.")
    parser.add_argument("--image_size", type=int, default=64,
                        help="Training image size.")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-6,
                        help="Weight decay.")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="Dataloader workers.")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed.")
    parser.add_argument("--grad_clip", type=float, default=1.0,
                        help="Gradient clipping norm. Set 0 to disable.")
    parser.add_argument("--num_train_timesteps", type=int, default=1000,
                        help="Number of diffusion timesteps.")
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_cap_v2",
                        choices=["linear", "scaled_linear", "squaredcos_cap_v2", "sigmoid"],
                        help="Noise schedule.")
    parser.add_argument("--ema_decay", type=float, default=0.9999,
                        help="EMA decay.")
    parser.add_argument("--amp", action="store_true",
                        help="Use mixed precision training on CUDA.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args)