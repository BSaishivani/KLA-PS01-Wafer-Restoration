
# KLA PS01 — AI-Based Restoration of Degraded Images

## 1. Project Overview

This project addresses the AI-Based Restoration of Degraded Images
problem for semiconductor inspection.

The goal is to reconstruct a clean, high-resolution image from a
degraded low-resolution noisy inspection image.

### Input

- 128 × 128 pixels
- Single-channel grayscale
- Noisy and low-resolution

### Output

- 256 × 256 pixels
- Single-channel grayscale
- Restored image

---


## 2. Proposed Solution

We use a lightweight residual attention U-Net architecture called
**KLAResUNet**.

The model combines:

- Convolutional feature extraction
- Residual blocks
- Channel attention
- Encoder-decoder architecture
- Skip connections
- Pixel Shuffle based 2× super-resolution
- Sigmoid output reconstruction

The model contains **1,821,101 parameters**, providing a balance
between restoration quality and inference speed.

### Pipeline

    Degraded 128x128 Image
              |
              v
         Normalization
              |
              v
          KLAResUNet
              |
              v
       Encoder / Bottleneck
              |
              v
           Decoder
              |
              v
        Pixel Shuffle x2
              |
              v
       Restored 256x256 Image

---


## 3. Dataset

The training dataset contains paired NumPy arrays.

Each sample contains:

- `GT` — clean high-resolution ground-truth image
- `NoisyLR` — degraded low-resolution image

Dataset statistics:

- Total paired samples: 3,200
- Training samples: 2,880
- Validation samples: 320
- Test images: 400

Ground truth:

    256 x 256

Degraded input:

    128 x 128

The degraded images may contain values outside the [0, 1] range.
This behaviour is expected because of the simulated degradation.

---


## 4. Model Architecture

### KLAResUNet

Total parameters:

    1,821,101

Trainable parameters:

    1,821,101

Input:

    1 x 128 x 128

Output:

    1 x 256 x 256

Major components:

1. Input convolution
2. Residual attention encoder
3. Downsampling blocks
4. Residual attention bottleneck
5. Decoder with skip connections
6. Pixel Shuffle x2 super-resolution
7. Reconstruction convolution

---


## 5. Training Configuration

Framework:

    PyTorch

Hardware:

    NVIDIA Tesla T4

Training configuration:

- Epochs: 30
- Batch size: 8
- Optimizer: AdamW
- Initial learning rate: 0.0001
- Weight decay: 0.0001
- Scheduler: CosineAnnealingLR
- Mixed precision: Enabled

The restoration loss combines:

- L1 reconstruction loss
- SSIM-related structural loss
- Edge loss

Loss weights:

- L1: 0.60
- SSIM: 0.25
- Edge: 0.15

Best checkpoint:

    weights/kla_resunet_best.pt

---


## 6. Validation Results

The final model was evaluated on 320 validation images.

| Metric | AI Model |
|---|---:|
| PSNR | 25.0746 dB |
| SSIM | 0.720521 |
| LPIPS | 0.354090 |
| MAE | 0.045348 |

### Bicubic Baseline

| Metric | Bicubic | AI Model |
|---|---:|---:|
| PSNR | 22.7117 dB | 25.0746 dB |
| SSIM | 0.528509 | 0.720521 |
| MAE | 0.056934 | 0.045348 |

Improvement:

- PSNR: +2.3629 dB
- SSIM: +0.192012
- MAE reduction: approximately 20.35%

The reported PSNR, SSIM and LPIPS values are validation metrics.
The available development test set does not contain ground-truth
images, so these metrics are not claimed as official test scores.

---


## 7. Test Inference

The available test set contains 400 degraded images.

All 400 images were successfully processed.

### Standalone inference benchmark

Hardware:

    NVIDIA Tesla T4

Average inference time:

    18.35 ms/image

Median inference time:

    15.85 ms/image

Total time for 400 images:

    7.34 seconds

The standalone benchmark is the preferred inference measurement
because it executes the actual repository inference script.

---


## 8. Repository Structure

    KLA_PS01/
    |
    +-- README.md
    +-- model.py
    +-- inference.py
    +-- requirements.txt
    |
    +-- weights/
    |   +-- kla_resunet_best.pt
    |
    +-- outputs/
    |   +-- test_restored/
    |
    +-- results/
    |   +-- validation_results.csv
    |   +-- inference_results.csv
    |   +-- final_results_summary.json
    |
    +-- figures/
    |   +-- training_validation_loss.png
    |   +-- validation_psnr.png
    |   +-- ai_vs_bicubic.png
    |
    +-- references/

---


## 9. Installation

Clone the repository:

    git clone <YOUR_GITHUB_REPOSITORY_URL>
    cd KLA_PS01

Create a virtual environment:

    python -m venv venv

Activate it.

Linux/macOS:

    source venv/bin/activate

Windows:

    venv\Scripts\activate

Install dependencies:

    pip install -r requirements.txt

---


## 10. Running Inference

The primary evaluation script is:

    inference.py

It accepts:

    --input_dir
    --output_dir

Example:

    python inference.py --input_dir ./test/NoisyLR --output_dir ./outputs/restored

The script automatically loads:

    weights/kla_resunet_best.pt

No manual modification of the source code is required.

---


## 11. Input and Output Format

Input files must be NumPy `.npy` files containing grayscale images
with shape:

    128 x 128

Example:

    000000.npy

The output is saved using the same filename.

Output format:

    NumPy .npy
    Shape: 256 x 256
    Dtype: float32
    Range: [0, 1]

Example:

    input/000000.npy
            |
            v
    output/000000.npy

---


## 12. Standalone Inference Verification

The standalone inference script was tested independently on all
400 available test images.

Result:

    400 / 400 successfully processed

Verified output:

    Shape: (256, 256)
    Dtype: float32
    Range: [0, 1]

The script completed without requiring manual source-code edits.

---


## 13. Reproducibility

The repository contains:

- `model.py` — model architecture
- `inference.py` — standalone evaluation script
- `requirements.txt` — Python dependencies
- `weights/kla_resunet_best.pt` — trained model
- `results/` — evaluation results
- `figures/` — result visualizations

A reviewer can install the dependencies and run the inference
script directly on a directory of `.npy` test images.

---


## 14. Limitations

PSNR, SSIM and LPIPS are reported on the validation split because
the available development test images do not have corresponding
ground-truth images.

The model also has difficult restoration cases under severe
degradation. These cases are retained for honest robustness
analysis.

---


## 15. Team

**Team Name:** <TEAM_NAME>

**Institution:** B. V. Raju Institute of Technology

**Problem Statement:** AI-Based Restoration of Degraded Images for
Semiconductor Inspection

---

## 16. License

Add the license required by the hackathon or institution before
public submission.
