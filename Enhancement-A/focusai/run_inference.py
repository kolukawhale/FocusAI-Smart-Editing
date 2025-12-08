import argparse
import os
import json
import random
from PIL import Image
import torch
import numpy as np

from models.enhancer import EnhancementCNN

def load_model(model_path, device):
    model = EnhancementCNN().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model


def enhance(model, img_path, device):
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img).astype("float32") / 255.0

    t = (
        torch.tensor(arr)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device)
    )

    with torch.no_grad():
        out = model(t)[0].permute(1, 2, 0).cpu().numpy()

    out = (out * 255).clip(0, 255).astype("uint8")
    return Image.fromarray(out)


def main():
    parser = argparse.ArgumentParser(description="Run inference on a random test image")
    parser.add_argument("--data-root", required=True, help="FocusAI_data root folder")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # ----------------------------
    # Load splits.json
    # ----------------------------
    splits_path = os.path.join(args.data_root, "splits.json")
    if not os.path.exists(splits_path):
        raise FileNotFoundError(f"Could not find {splits_path}")

    with open(splits_path, "r") as f:
        splits = json.load(f)

    test_files = splits.get("test", [])
    if len(test_files) == 0:
        raise ValueError("No test images found in splits.json")

    # pick random test image
    fname = random.choice(test_files)
    print("Chosen random test file:", fname)

    raw_path = os.path.join(args.data_root, "raw", f"{fname}.png",)

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"Raw input not found: {raw_path}")

    # ----------------------------
    # Load trained model
    # ----------------------------
    model_path = os.path.join(args.data_root,  "FocusAI_data", "runs","best.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = load_model(model_path, device)

    # ----------------------------
    # Enhance Image
    # ----------------------------
    enhanced = enhance(model, raw_path, device)

    # ----------------------------
    # Save output
    # ----------------------------
    out_dir = os.path.join(args.data_root, "runs", "inference")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"enhanced_{fname}.png")
    enhanced.save(out_path)

    print("Saved enhanced output to:", out_path)
    print("Done.")


if __name__ == "__main__":
    main()
