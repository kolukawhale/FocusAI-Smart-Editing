import os
import json
import random
from typing import Optional, Dict, List

from PIL import Image
from torchvision import transforms as T
from torch.utils.data import Dataset
import torch


VALID_EXT = (".png", ".jpg", ".jpeg", ".tif", ".bmp", ".webp")


def find_file_with_same_name(folder: str, base: str) -> str:
    """
    Finds a matching file in 'folder' that has the same basename (ignoring extension).
    Example: base='cat' → matches 'cat.jpg' or 'cat.png'
    """
    for f in os.listdir(folder):
        if not f.lower().endswith(VALID_EXT):
            continue
        if os.path.splitext(f)[0] == base:
            return os.path.join(folder, f)
    raise FileNotFoundError(f"No match for '{base}' in {folder}")


class FullImageDataset(Dataset):
    """
    Loads matching raw/ and c/ images even if their extensions differ.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        resolution: int = 512,
        jitter: Optional[T.ColorJitter] = None,
    ):
        self.root = root
        self.input_dir = os.path.join(root, "raw")
        self.target_dir = os.path.join(root, "c")
        self.resolution = resolution
        self.jitter = jitter

        # --- read splits.json ---
        splits_path = os.path.join(root, "splits.json")
        if not os.path.exists(splits_path):
            raise FileNotFoundError(f"{splits_path} not found. Run create_splits(root).")

        with open(splits_path, "r") as f:
            splits = json.load(f)
        self.files: List[str] = splits[split]   # basenames only, no ext

        self.resize = T.Resize((resolution, resolution), interpolation=T.InterpolationMode.LANCZOS)
        self.to_tensor = T.ToTensor()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        base = self.files[idx]

        x_path = find_file_with_same_name(self.input_dir, base)
        y_path = find_file_with_same_name(self.target_dir, base)

        x = Image.open(x_path).convert("RGB")
        y = Image.open(y_path).convert("RGB")

        x = self.resize(x)
        y = self.resize(y)

        if self.jitter is not None:
            x = self.jitter(x)

        return self.to_tensor(x).float(), self.to_tensor(y).float()


def create_splits(
    root: str,
    train_size: int = 50,
    val_size: int = 10,
    test_size: int = 5,
    seed: int = 42,
) -> Dict[str, list]:

    raw_dir = os.path.join(root, "raw")
    target_dir = os.path.join(root, "c")

    raw_basenames = {
        os.path.splitext(f)[0]
        for f in os.listdir(raw_dir)
        if f.lower().endswith(VALID_EXT)
    }

    target_basenames = {
        os.path.splitext(f)[0]
        for f in os.listdir(target_dir)
        if f.lower().endswith(VALID_EXT)
    }

    common = list(raw_basenames & target_basenames)
    if not common:
        raise RuntimeError("No matching filenames found between raw/ and c/.")

    random.Random(seed).shuffle(common)

    train = common[:train_size]
    val   = common[train_size : train_size + val_size]
    test  = common[train_size + val_size : train_size + val_size + test_size]

    splits = {"train": train, "val": val, "test": test}

    path = os.path.join(root, "splits.json")
    with open(path, "w") as f:
        json.dump(splits, f, indent=2)

    print(f"Saved splits to {path}")
    print(f"Train={len(train)}, Val={len(val)}, Test={len(test)}")

    return splits
