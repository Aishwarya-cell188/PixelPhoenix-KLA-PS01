# PixelPhoenix-KLA-PS01
### AI-Based Restoration of Degraded Semiconductor Inspection Images

This repository contains our solution for the KLA hackathon problem statement **"AI-Based Restoration of Degraded Images."** The model restores noisy, low-resolution grayscale semiconductor inspection images into clean, high-resolution outputs.

---

## Problem Overview

Input images suffer from a combination of:
- **Speckle noise**
- **Gaussian noise**
- **Reduced spatial resolution** (128→256 or 256→512)

The model learns to reverse these degradations, producing a restored image that closely matches the original ground truth.

---

## Approach

- **Architecture:** A lightweight U-Net (single-channel grayscale input/output) with skip connections to preserve fine structural detail while removing noise.
- **Input handling:** Degraded images are bicubic-upsampled to the target resolution before being passed to the model, so the network focuses on joint denoising + detail reconstruction rather than upsampling itself.
- **Loss function:** A combined L1 + SSIM loss (`0.7 * L1 + 0.3 * (1 - SSIM)`), balancing pixel-level accuracy with structural/textural fidelity.
- **Optimizer:** Adam, with a `ReduceLROnPlateau` learning rate scheduler for stable convergence.

---

## Results (on held-out validation split)

| Metric | Score |
|---|---|
| PSNR | 27.94 |
| SSIM | 0.774 |
| LPIPS | 0.283 |
| Inference time | ~5.9 ms/image (NVIDIA T4) |
| Model size | 1.78 MB (465,953 parameters) |

---

## Repository Structure

```
PixelPhoenix-KLA-PS01/
├── inference.py            # Standalone evaluation script (KLA runs this)
├── model_weights.pth       # Trained model weights
├── KLA_training.ipynb      # Training notebook (reproduces training from scratch)
├── requirements.txt        # Python dependencies
├── restored_outputs/       # Model outputs on the provided test set
└── README.md
```

---

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Aishwarya-cell188/PixelPhoenix-KLA-PS01.git
   cd PixelPhoenix-KLA-PS01
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Running Inference

The evaluation script takes an input directory of degraded `.npy` images and writes restored `.npy` outputs to a specified output directory.

```bash
python inference.py --input_dir <path_to_degraded_images> --output_dir <path_to_save_restored_images> --weights model_weights.pth
```

**Example:**
```bash
python inference.py --input_dir ./test_data/NoisyLR --output_dir ./restored_outputs --weights model_weights.pth
```

The script automatically uses GPU if available, and falls back to CPU otherwise. It prints total and average inference time upon completion.

---

## Reproducing Training

Open `KLA_training.ipynb` in Google Colab (GPU runtime recommended — T4 or better) and run all cells in order. The notebook covers:
- Dataset loading and inspection
- Model definition (U-Net)
- Training loop with validation tracking
- Metric computation (PSNR / SSIM / LPIPS)
- Checkpoint saving

---

## Team

**Team Name:** PixelPhoenix
**Problem Statement:** AI-Based Restoration of Degraded Images (PS01)

---

## Notes

- Validation was performed on an in-distribution held-out split (90/10) from the provided training data. Out-of-distribution evaluation was not performed due to the absence of source-labeled OOD data in the training set.
- The model was trained and benchmarked on an NVIDIA T4 GPU (Google Colab). Inference is expected to run at least as fast on KLA's benchmarking hardware.

