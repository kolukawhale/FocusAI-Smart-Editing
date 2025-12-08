import os
import argparse
import torch

from data.dataset import create_splits
from train.trainer import train_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Folder that contains FocusAI_data/raw and FocusAI_data/c",
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Where to save checkpoints (default: <data-root>/FocusAI_data/runs)",
    )
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--resolution", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()

    data_root = args.data_root
    out_dir = args.out_dir or os.path.join(data_root, "FocusAI_data", "runs")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # create splits.json if it doesn't exist
    splits_path = os.path.join(data_root, "splits.json")
    if not os.path.exists(splits_path):
        print("No splits.json found, creating one...")
        create_splits(data_root)

    train_model(
        data_root=data_root,
        out_dir=out_dir,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        resolution=args.resolution,
    )


if __name__ == "__main__":
    main()
