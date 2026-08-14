"""
inference.py

Standalone evaluation script for KLA AI-Based Restoration of Degraded Images challenge.

Usage:
    python inference.py --input_dir /path/to/test/NoisyLR --output_dir /path/to/output --weights model_weights.pth

This script:
    1. Loads the trained U-Net restoration model.
    2. Reads all degraded (.npy) images from input_dir.
    3. Runs inference (denoise + super-resolve) on each image.
    4. Saves restored images as .npy files to output_dir, preserving filenames.
    5. Reports total and average inference time.

No manual edits should be required to run this on a fresh machine, as long as:
    - requirements.txt packages are installed
    - model_weights.pth is present in the same directory (or path passed via --weights)
"""

import argparse
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model definition (must match the architecture used during training)
# ---------------------------------------------------------------------------
class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU()
        )

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU()
        )
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
        )

        self.out = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return torch.sigmoid(self.out(d1))


# ---------------------------------------------------------------------------
# Preprocessing helpers (must mirror training-time preprocessing exactly)
# ---------------------------------------------------------------------------
def preprocess(arr: np.ndarray, target_size: int = 256) -> torch.Tensor:
    """
    Convert a raw degraded .npy array into a model-ready tensor:
      - cast to float32
      - normalize to [0, 1] based on observed training data range
      - upsample (bicubic) to target_size x target_size
      - clamp to [0, 1] to handle out-of-range speckle values
    """
    x = arr.astype(np.float32)

    # Training and test data are already stored in a [0,1]-like range.
    # Speckle noise can occasionally push values slightly outside [0,1]
    # (verified: real test data observed max ~1.54), which is handled by
    # the clamp below rather than any rescaling — rescaling (e.g. /255)
    # would incorrectly crush already-correct [0,1] values toward zero.

    tensor = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W]

    tensor = F.interpolate(
        tensor, size=(target_size, target_size), mode="bicubic", align_corners=False
    )
    tensor = torch.clamp(tensor, 0.0, 1.0)
    return tensor


def postprocess(tensor: torch.Tensor) -> np.ndarray:
    """Convert model output tensor back to a numpy array in [0,1] float32."""
    arr = tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
    return arr.astype(np.float32)


# ---------------------------------------------------------------------------
# Main inference routine
# ---------------------------------------------------------------------------
def run_inference(input_dir: str, output_dir: str, weights_path: str, target_size: int = 256):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)

    model = UNet().to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {input_dir}")

    print(f"Found {len(files)} images to restore.")

    total_time = 0.0
    with torch.no_grad():
        for fname in files:
            in_path = os.path.join(input_dir, fname)
            out_path = os.path.join(output_dir, fname)

            arr = np.load(in_path)
            x = preprocess(arr, target_size=target_size).to(device)

            if device == "cuda":
                torch.cuda.synchronize()
            start = time.time()

            pred = model(x)

            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.time() - start
            total_time += elapsed

            restored = postprocess(pred)
            np.save(out_path, restored)

    avg_time_ms = (total_time / len(files)) * 1000
    print(f"Done. Restored images saved to: {output_dir}")
    print(f"Average inference time per image: {avg_time_ms:.2f} ms")
    print(f"Total inference time: {total_time:.2f} s for {len(files)} images")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run trained restoration model on a directory of degraded .npy images."
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Path to directory containing degraded input .npy images."
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Path to directory where restored .npy images will be saved."
    )
    parser.add_argument(
        "--weights", type=str, default="model_weights.pth",
        help="Path to trained model weights (.pth). Default: model_weights.pth in current directory."
    )
    parser.add_argument(
        "--target_size", type=int, default=256,
        help="Output resolution the model reconstructs to (default: 256)."
    )

    args = parser.parse_args()
    run_inference(args.input_dir, args.output_dir, args.weights, args.target_size)
