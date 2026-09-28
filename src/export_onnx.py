"""
src/export_onnx.py
==================
Export the Phase 1 3D UNet to ONNX for low-latency production inference.

Usage:
    python -m src.export_onnx [--checkpoint artifacts/best_unet.pt]
                             [--output artifacts/best_unet.onnx]
                             [--opset 17]

If the checkpoint does not exist, a randomly-initialized UNet3D state_dict
is saved to the checkpoint path so the pipeline is runnable end-to-end
(weights are identical across both engines, so the equivalence check is
still valid). When the real trained checkpoint is available, it is used
unchanged.
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import onnx
import torch

from src.model import UNet3D

# Canonical Phase 1 input shape: (batch, channels, depth, height, width)
DEFAULT_INPUT_SHAPE = (1, 4, 128, 128, 128)

# Dynamic axes: batch + spatial dims are dynamic; channels stay fixed at 4.
DYNAMIC_AXES = {
    "input": {0: "batch", 2: "depth", 3: "height", 4: "width"},
    "output": {0: "batch", 2: "depth", 3: "height", 4: "width"},
}


def _placeholder_marker_path(path: str) -> str:
    return path + ".PLACEHOLDER"


def _ensure_checkpoint(checkpoint_path: str, seed: int = 42) -> bool:
    """Create a dummy checkpoint if the real one is missing.

    The dummy weights are randomly initialized but identical across both
    PyTorch and ONNX Runtime, so numerical equivalence checks remain valid.

    Returns ``True`` when the checkpoint is a placeholder (either created
    now or previously without a trained replacement), and writes a
    ``<path>.PLACEHOLDER`` marker so the app can warn users that the
    model output is not clinically meaningful.
    """
    marker = _placeholder_marker_path(checkpoint_path)
    if os.path.exists(checkpoint_path):
        # A pre-existing checkpoint without a marker is assumed trained;
        # a marker left behind means it was never replaced by real weights.
        return os.path.exists(marker)

    os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
    torch.manual_seed(seed)
    model = UNet3D(in_channels=4, out_channels=4)
    torch.save(model.state_dict(), checkpoint_path)
    with open(marker, "w", encoding="utf-8") as fh:
        fh.write(
            "Placeholder checkpoint created by src.export_onnx._ensure_checkpoint\n"
            f"seed={seed} — NOT trained. Volumes/predictions from this model are\n"
            "not clinically meaningful. Replace with a trained checkpoint,\n"
            "re-run `python -m src.export_onnx`, then delete this marker file.\n"
        )
    print(f"[export][WARNING] Checkpoint not found; saved UNTRAINED dummy "
          f"state_dict -> {checkpoint_path} (marker: {marker})")
    return True


def export_to_onnx(
    checkpoint_path: str = "artifacts/best_unet.pt",
    output_path: str = "artifacts/best_unet.onnx",
    opset_version: int = 17,
    input_shape: tuple[int, ...] = DEFAULT_INPUT_SHAPE,
    device: Optional[torch.device] = None,
) -> str:
    """Load the Phase 1 UNet3D and export it to ONNX with dynamic axes.

    Args:
        checkpoint_path: Path to the saved state_dict checkpoint.
        output_path: Destination path for the exported ONNX model.
        opset_version: ONNX opset version to target.
        input_shape: Static shape used for tracing (batch, 4, D, H, W).
        device: Torch device for the export forward pass.

    Returns:
        The path to the exported ONNX model.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    is_placeholder = _ensure_checkpoint(checkpoint_path)

    model = UNet3D(in_channels=4, out_channels=4)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device).eval()

    dummy_input = torch.randn(*input_shape, device=device)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=DYNAMIC_AXES,
        opset_version=opset_version,
        do_constant_folding=True,
        export_params=True,
    )

    # Validate the exported graph.
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)

    # Mirror the placeholder marker onto the ONNX artifact so runtime
    # consumers (app.py) can warn about untrained weights.
    onnx_marker = _placeholder_marker_path(output_path)
    if is_placeholder:
        with open(onnx_marker, "w", encoding="utf-8") as fh:
            fh.write(
                "Placeholder ONNX export from an UNTRAINED checkpoint.\n"
                "Predictions are not clinically meaningful. Replace the\n"
                "checkpoint with trained weights and re-run export.\n"
            )
        print(f"[export][WARNING] Placeholder weights exported; marker -> "
              f"{onnx_marker}")
    elif os.path.exists(onnx_marker):
        os.remove(onnx_marker)

    print(f"[export] ONNX model saved -> {output_path}")
    print(f"[export] Opset: {opset_version} | Device: {device}")
    print(f"[export] Dynamic axes: {DYNAMIC_AXES}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the Phase 1 3D UNet to ONNX with dynamic axes."
    )
    parser.add_argument(
        "--checkpoint",
        default="artifacts/best_unet.pt",
        help="Path to the PyTorch state_dict checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/best_unet.onnx",
        help="Destination path for the exported ONNX model.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17).",
    )
    args = parser.parse_args()

    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
    )


if __name__ == "__main__":
    main()