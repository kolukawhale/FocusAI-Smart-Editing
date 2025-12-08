import os
from typing import Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.dataset import FullImageDataset
from models.enhancer import EnhancementCNN
from train.losses import VGGPerceptualLoss, total_loss


def set_seed(seed: int = 42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(
    data_root: str,
    resolution: int = 256,
    batch_size: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = FullImageDataset(data_root, split="train", resolution=resolution)
    val_ds = FullImageDataset(data_root, split="val", resolution=resolution)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


def train_model(
    data_root: str,
    out_dir: str,
    device: torch.device,
    epochs: int = 8,
    batch_size: int = 4,
    lr: float = 1e-3,
    resolution: int = 256,
):
    os.makedirs(out_dir, exist_ok=True)
    set_seed(42)

    train_loader, val_loader = build_loaders(
        data_root=data_root,
        resolution=resolution,
        batch_size=batch_size,
    )

    model = EnhancementCNN().to(device)
    vgg_loss = VGGPerceptualLoss(style_weight=0.0).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_score = float("-inf")
    best_path = os.path.join(out_dir, "best.pt")

    for ep in range(1, epochs + 1):
        print(f"\n=== Epoch {ep}/{epochs} ===")
        model.train()
        running_loss = 0.0

        for x, y in tqdm(train_loader, desc="Train", ncols=80):
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            pred = model(x)

            loss, loss_dict = total_loss(pred, y, vgg_loss)
            loss.backward()
            opt.step()

            running_loss += loss.item()

        running_loss /= len(train_loader)
        scheduler.step()

        # ----- validation -----
        model.eval()
        val_ssim = 0.0
        val_percept = 0.0
        val_color = 0.0
        val_batches = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc="Val  ", ncols=80):
                x = x.to(device)
                y = y.to(device)

                pred = model(x)

                _, loss_dict = total_loss(pred, y, vgg_loss)
                val_ssim += loss_dict["ssim"]
                val_percept += loss_dict["percept"]
                val_color += loss_dict["color"]
                val_batches += 1

        avg_ssim = val_ssim / val_batches
        avg_percept = val_percept / val_batches
        avg_color = val_color / val_batches

        val_score = 0.4 * avg_ssim - 0.3 * avg_percept - 0.3 * avg_color

        print(
            f"Train loss: {running_loss:.4f} | "
            f"SSIM: {avg_ssim:.3f} | Percept: {avg_percept:.3f} | "
            f"Color: {avg_color:.3f} | Score: {val_score:.4f}"
        )

        if val_score > best_score:
            best_score = val_score
            torch.save(model.state_dict(), best_path)
            print(f"→ Saved best model to {best_path}")

    print(f"\nDone. Best validation score: {best_score:.4f}")
