import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from diffusers import DDPMPipeline
from diffusers import DDPMScheduler
from diffusers import UNet2DModel




TRAIN_DIR = "/home/Downloads/dataset/mapillary_cropped/train"

OUT_DIR = "./ddpm_mapillary_64x64"

IMAGE_SIZE = 64

BATCH_SIZE = 128

EPOCHS = 30

LR = 1e-4

NUM_WORKERS = 4

SAVE_EVERY = 5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)




transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=transform
)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=(NUM_WORKERS > 0),
)

print(f"Train images: {len(dataset):,}")




model = UNet2DModel(
    sample_size=IMAGE_SIZE,
    in_channels=3,
    out_channels=3,

    layers_per_block=2,

    block_out_channels=(
        128,
        128,
        256,
        256,
        512,
        512,
    ),

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
).to(DEVICE)




noise_scheduler = DDPMScheduler(
    num_train_timesteps=1000
)



optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)



Path(OUT_DIR).mkdir(
    parents=True,
    exist_ok=True
)

print("\nStarting DDPM training...\n")

for epoch in range(1, EPOCHS + 1):

    model.train()

    epoch_loss = 0.0

    pbar = tqdm(loader)

    for clean_images, _ in pbar:

        clean_images = clean_images.to(DEVICE)

        noise = torch.randn_like(clean_images)

        bsz = clean_images.shape[0]

        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=DEVICE
        ).long()

        noisy_images = noise_scheduler.add_noise(
            clean_images,
            noise,
            timesteps
        )

        noise_pred = model(
            noisy_images,
            timesteps
        ).sample

        loss = torch.nn.functional.mse_loss(
            noise_pred,
            noise
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

        pbar.set_description(
            f"Epoch {epoch}/{EPOCHS} | loss {loss.item():.6f}"
        )

    avg_loss = epoch_loss / len(loader)

    print(
        f"\nEpoch {epoch}/{EPOCHS} "
        f"| avg loss {avg_loss:.6f}"
    )


    if epoch % SAVE_EVERY == 0 or epoch == EPOCHS:

        save_dir = f"{OUT_DIR}_epoch{epoch}"

        pipeline = DDPMPipeline(
            unet=model,
            scheduler=noise_scheduler
        )

        pipeline.save_pretrained(save_dir)

        print(f"\nSaved checkpoint: {save_dir}")



pipeline = DDPMPipeline(
    unet=model,
    scheduler=noise_scheduler
)

pipeline.save_pretrained(OUT_DIR)

print(f"\nFinal DDPM saved to: {OUT_DIR}")

print("\nDONE.")