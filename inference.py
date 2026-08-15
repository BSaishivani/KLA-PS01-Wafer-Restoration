
import argparse
import os
import time

import numpy as np
import torch

from model import KLAResUNet


# ============================================================
# CONFIGURATION
# ============================================================

# These are the exact normalization statistics used
# during training.
MEAN = 0.433536
STD = 0.284787


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# LOAD MODEL
# ============================================================

def load_model(weights_path, device):

    model = KLAResUNet().to(device)

    checkpoint = torch.load(
        weights_path,
        map_location=device,
        weights_only=False
    )

    # The submitted checkpoint contains:
    # checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and \
       "model_state_dict" in checkpoint:

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        # Also support a raw state_dict
        state_dict = checkpoint

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    return model


# ============================================================
# LOAD INPUT IMAGE
# ============================================================

def load_input(path, device):

    image = np.load(path).astype(
        np.float32
    )

    # Remove dimensions of size 1
    image = np.squeeze(image)

    if image.ndim != 2:
        raise ValueError(
            f"Expected a 2D grayscale image, "
            f"but {path} has shape {image.shape}"
        )

    # Expected challenge input resolution
    if image.shape != (128, 128):

        raise ValueError(
            f"Expected input shape (128, 128), "
            f"but {path} has shape {image.shape}"
        )

    # Convert:
    # (128,128)
    # ->
    # (1,1,128,128)

    tensor = torch.from_numpy(
        image
    ).unsqueeze(0).unsqueeze(0)

    tensor = tensor.to(
        device
    )

    # Same normalization used during training
    tensor = (
        tensor - MEAN
    ) / STD

    return tensor


# ============================================================
# RESTORE ONE IMAGE
# ============================================================

def restore_image(
    model,
    input_path,
    output_path,
    device
):

    tensor = load_input(
        input_path,
        device
    )

    with torch.no_grad():

        restored = model(
            tensor
        )

    # Convert:
    # (1,1,256,256)
    # ->
    # (256,256)

    restored = (
        restored
        .squeeze()
        .cpu()
        .numpy()
    )

    # Model output is already constrained
    # to [0,1], but clip for safety.

    restored = np.clip(
        restored,
        0.0,
        1.0
    ).astype(
        np.float32
    )

    np.save(
        output_path,
        restored
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "KLA PS01 AI image restoration "
            "inference"
        )
    )

    parser.add_argument(
        "--input_dir",
        required=True,
        help=(
            "Directory containing degraded "
            ".npy images"
        )
    )

    parser.add_argument(
        "--output_dir",
        required=True,
        help=(
            "Directory where restored "
            ".npy images will be saved"
        )
    )

    parser.add_argument(
        "--weights",
        default=None,
        help=(
            "Path to trained model weights. "
            "Defaults to weights/"
            "kla_resunet_best.pt relative "
            "to this script."
        )
    )

    args = parser.parse_args()

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    script_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    if args.weights is None:

        weights_path = os.path.join(
            script_dir,
            "weights",
            "kla_resunet_best.pt"
        )

    else:

        weights_path = args.weights

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not os.path.isdir(
        args.input_dir
    ):

        raise FileNotFoundError(
            f"Input directory not found: "
            f"{args.input_dir}"
        )

    if not os.path.isfile(
        weights_path
    ):

        raise FileNotFoundError(
            f"Model weights not found: "
            f"{weights_path}"
        )

    os.makedirs(
        args.output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print("=" * 70)
    print("KLA PS01 — INFERENCE")
    print("=" * 70)

    print(
        "Device:",
        device
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print(
        "Input directory:",
        args.input_dir
    )

    print(
        "Output directory:",
        args.output_dir
    )

    print(
        "Weights:",
        weights_path
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = load_model(
        weights_path,
        device
    )

    print(
        "Model loaded successfully."
    )

    # --------------------------------------------------------
    # Find input files
    # --------------------------------------------------------

    input_files = sorted(
        [
            filename
            for filename in os.listdir(
                args.input_dir
            )
            if filename.endswith(".npy")
            and not filename.startswith("._")
        ]
    )

    if len(input_files) == 0:

        raise RuntimeError(
            "No .npy images found in "
            "the input directory."
        )

    print(
        f"Images found: {len(input_files)}"
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    times = []

    print("\nStarting inference...\n")

    for index, filename in enumerate(
        input_files
    ):

        input_path = os.path.join(
            args.input_dir,
            filename
        )

        output_path = os.path.join(
            args.output_dir,
            filename
        )

        start = time.perf_counter()

        restore_image(
            model,
            input_path,
            output_path,
            device
        )

        # Synchronize GPU before measuring time
        if device.type == "cuda":

            torch.cuda.synchronize()

        elapsed = (
            time.perf_counter()
            - start
        )

        times.append(elapsed)

        if (
            index == 0
            or (index + 1) % 50 == 0
            or index == len(input_files) - 1
        ):

            print(
                f"Processed "
                f"{index + 1}/"
                f"{len(input_files)} "
                f"| {filename} "
                f"| {elapsed:.4f} sec"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    times = np.asarray(
        times
    )

    print("\n" + "=" * 70)
    print("INFERENCE COMPLETE")
    print("=" * 70)

    print(
        f"Images processed : "
        f"{len(input_files)}"
    )

    print(
        f"Average time     : "
        f"{times.mean():.6f} sec/image"
    )

    print(
        f"Median time      : "
        f"{np.median(times):.6f} sec/image"
    )

    print(
        f"Total time       : "
        f"{times.sum():.4f} sec"
    )

    print(
        "Output directory:",
        args.output_dir
    )

    print("\nDone.")


if __name__ == "__main__":

    main()
