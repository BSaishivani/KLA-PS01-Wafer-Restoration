"""
KLA PS01 — Standalone Training Script
AI-Based Restoration of Degraded Images for Semiconductor Inspection

Input:
    128x128 grayscale noisy low-resolution .npy images

Target:
    256x256 grayscale clean ground-truth .npy images

Model:
    KLAResUNet

Loss:
    Weighted combination of:
        - L1 reconstruction loss
        - SSIM loss
        - Edge loss

This script reproduces the training procedure used for the
submitted restoration model.
"""

import os
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split

from model import KLAResUNet


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATASET
# ============================================================

class KLADataset(Dataset):

    def __init__(self, input_dir, target_dir):

        self.input_dir = input_dir
        self.target_dir = target_dir

        input_files = {
            f for f in os.listdir(input_dir)
            if f.endswith(".npy")
            and not f.startswith("._")
        }

        target_files = {
            f for f in os.listdir(target_dir)
            if f.endswith(".npy")
            and not f.startswith("._")
        }

        self.files = sorted(
            input_files.intersection(target_files)
        )

        if len(self.files) == 0:
            raise RuntimeError(
                "No matching .npy input/target pairs found."
            )

        print("Paired samples:", len(self.files))


    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):

        filename = self.files[index]

        x = np.load(
            os.path.join(
                self.input_dir,
                filename
            )
        ).astype(np.float32)

        y = np.load(
            os.path.join(
                self.target_dir,
                filename
            )
        ).astype(np.float32)

        # Input is allowed to contain values outside [0, 1]
        # because of the degradation/noise process.

        x = torch.from_numpy(x).unsqueeze(0)
        y = torch.from_numpy(y).unsqueeze(0)

        return x, y


# ============================================================
# SSIM
# ============================================================

def gaussian_window(
    window_size=11,
    sigma=1.5,
    device="cpu"
):

    coords = torch.arange(
        window_size,
        dtype=torch.float32,
        device=device
    ) - window_size // 2

    g = torch.exp(
        -(coords ** 2) /
        (2 * sigma ** 2)
    )

    g = g / g.sum()

    window = g[:, None] * g[None, :]

    return window.unsqueeze(0).unsqueeze(0)


def ssim(
    img1,
    img2,
    window_size=11,
    sigma=1.5
):

    window = gaussian_window(
        window_size,
        sigma,
        img1.device
    )

    mu1 = F.conv2d(
        img1,
        window,
        padding=window_size // 2
    )

    mu2 = F.conv2d(
        img2,
        window,
        padding=window_size // 2
    )

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu12 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(
            img1 * img1,
            window,
            padding=window_size // 2
        )
        - mu1_sq
    )

    sigma2_sq = (
        F.conv2d(
            img2 * img2,
            window,
            padding=window_size // 2
        )
        - mu2_sq
    )

    sigma12 = (
        F.conv2d(
            img1 * img2,
            window,
            padding=window_size // 2
        )
        - mu12
    )

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    result = (
        (2 * mu12 + C1)
        * (2 * sigma12 + C2)
    ) / (
        (mu1_sq + mu2_sq + C1)
        * (sigma1_sq + sigma2_sq + C2)
    )

    return result.mean()


# ============================================================
# EDGE LOSS
# ============================================================

def edge_loss(pred, target):

    sobel_x = torch.tensor(
        [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ],
        dtype=torch.float32,
        device=pred.device
    ).view(1, 1, 3, 3)

    sobel_y = torch.tensor(
        [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ],
        dtype=torch.float32,
        device=pred.device
    ).view(1, 1, 3, 3)

    pred_x = F.conv2d(
        pred,
        sobel_x,
        padding=1
    )

    pred_y = F.conv2d(
        pred,
        sobel_y,
        padding=1
    )

    target_x = F.conv2d(
        target,
        sobel_x,
        padding=1
    )

    target_y = F.conv2d(
        target,
        sobel_y,
        padding=1
    )

    pred_edges = torch.sqrt(
        pred_x ** 2 +
        pred_y ** 2 +
        1e-8
    )

    target_edges = torch.sqrt(
        target_x ** 2 +
        target_y ** 2 +
        1e-8
    )

    return F.l1_loss(
        pred_edges,
        target_edges
    )


# ============================================================
# RESTORATION LOSS
# ============================================================

def restoration_loss(
    pred,
    target
):

    l1 = F.l1_loss(
        pred,
        target
    )

    ssim_loss = 1.0 - ssim(
        pred,
        target
    )

    edge = edge_loss(
        pred,
        target
    )

    total = (
        0.60 * l1
        + 0.25 * ssim_loss
        + 0.15 * edge
    )

    return total


# ============================================================
# PSNR
# ============================================================

def calculate_psnr(
    pred,
    target
):

    mse = F.mse_loss(
        pred,
        target
    ).item()

    if mse <= 1e-12:
        return 100.0

    return 10.0 * np.log10(
        1.0 / mse
    )


# ============================================================
# TRAINING
# ============================================================

def train(args):

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 70)
    print("KLA PS01 — TRAINING")
    print("=" * 70)

    print("Device:", device)

    if torch.cuda.is_available():
        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )


    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = KLADataset(
        args.input_dir,
        args.target_dir
    )

    validation_size = int(
        0.10 * len(dataset)
    )

    training_size = (
        len(dataset)
        - validation_size
    )

    train_dataset, val_dataset = random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(SEED)
    )

    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )


    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available()
    )


    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = KLAResUNet().to(device)

    print(
        "Parameters:",
        sum(
            p.numel()
            for p in model.parameters()
        )
    )


    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )


    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=torch.cuda.is_available()
    )


    # --------------------------------------------------------
    # Checkpoint directory
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(args.checkpoint),
        exist_ok=True
    )

    best_loss = float("inf")


    # --------------------------------------------------------
    # Epoch loop
    # --------------------------------------------------------

    for epoch in range(args.epochs):

        model.train()

        train_loss = 0.0


        for x, y in train_loader:

            x = x.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            with torch.amp.autocast(
                "cuda",
                enabled=torch.cuda.is_available()
            ):

                pred = model(x)

                loss = restoration_loss(
                    pred,
                    y
                )


            scaler.scale(
                loss
            ).backward()

            scaler.unscale_(
                optimizer
            )

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            scaler.step(
                optimizer
            )

            scaler.update()

            train_loss += (
                loss.item()
                * x.size(0)
            )


        train_loss /= len(
            train_loader.dataset
        )


        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_loss = 0.0
        val_psnr = 0.0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(
                    device,
                    non_blocking=True
                )

                y = y.to(
                    device,
                    non_blocking=True
                )

                pred = model(x)

                loss = restoration_loss(
                    pred,
                    y
                )

                val_loss += (
                    loss.item()
                    * x.size(0)
                )

                val_psnr += (
                    calculate_psnr(
                        pred,
                        y
                    )
                    * x.size(0)
                )


        val_loss /= len(
            val_loader.dataset
        )

        val_psnr /= len(
            val_loader.dataset
        )


        scheduler.step()


        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_loss:

            best_loss = val_loss

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch + 1,

                    "val_loss":
                        val_loss,

                    "val_psnr":
                        val_psnr
                },
                args.checkpoint
            )

            marker = " ⭐ BEST"

        else:

            marker = ""


        print(
            f"Epoch [{epoch + 1:02d}/{args.epochs}] "
            f"| Train Loss: {train_loss:.5f} "
            f"| Val Loss: {val_loss:.5f} "
            f"| PSNR: {val_psnr:.2f} dB "
            f"| LR: {scheduler.get_last_lr()[0]:.2e}"
            f"{marker}"
        )


    print("=" * 70)
    print("TRAINING COMPLETE")
    print("Best validation loss:", best_loss)
    print("Best checkpoint:", args.checkpoint)
    print("=" * 70)


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train KLA PS01 restoration model"
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing degraded NoisyLR .npy files"
    )

    parser.add_argument(
        "--target_dir",
        required=True,
        help="Directory containing clean GT .npy files"
    )

    parser.add_argument(
        "--checkpoint",
        default="kla_resunet_best.pt",
        help="Output checkpoint path"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=8
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2
    )

    args = parser.parse_args()

    train(args)
