"""
src/verify_onnx.py
==================
Utility script to verify numerical equivalence between the PyTorch
UNet3D and the exported ONNX model under a strict tolerance.

Usage:
    python -m src.verify_onnx [--checkpoint artifacts/best_unet.pt]
                              [--onnx artifacts/best_unet.onnx]
                              [--atol 1e-3]
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from src.inference import ONNXInferenceEngine, verify_onnx_equivalence
from src.model import load_model

# Canonical Phase 1 input shape: (batch, channels, depth, height, width)
DEFAULT_INPUT_SHAPE = (1, 4, 128, 128, 128)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify PyTorch <-> ONNX Runtime numerical equivalence."
    )
    parser.add_argument(
        "--checkpoint",
        default="artifacts/best_unet.pt",
        help="Path to the PyTorch state_dict checkpoint.",
    )
    parser.add_argument(
        "--onnx",
        default="artifacts/best_unet.onnx",
        help="Path to the exported ONNX model.",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for the equivalence check (default: 1e-3).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the deterministic sample input.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[verify] Loading PyTorch model from {args.checkpoint}")
    pytorch_model = load_model(args.checkpoint, device=device)

    print(f"[verify] Loading ONNX Runtime engine from {args.onnx}")
    onnx_engine = ONNXInferenceEngine(args.onnx)
    print(f"[verify] Active providers: {onnx_engine.active_providers}")

    # Deterministic sample input mimicking a preprocessed scan stack.
    sample_input = np.random.RandomState(args.seed).randn(*DEFAULT_INPUT_SHAPE).astype(
        np.float32
    )

    result = verify_onnx_equivalence(
        pytorch_model=pytorch_model,
        onnx_engine=onnx_engine,
        sample_input=sample_input,
        atol=args.atol,
        device=device,
    )

    print(f"[verify] Input shape        : {DEFAULT_INPUT_SHAPE}")
    print(f"[verify] Max abs diff       : {result.max_abs_diff:.3e}")
    print(f"[verify] Mean abs diff      : {result.mean_abs_diff:.3e}")
    print(f"[verify] Tolerance (atol)   : {result.atol:.3e}")
    print(f"[verify] PASSED             : {result.passed}")

    if not result.passed:
        raise SystemExit(
            f"Equivalence check FAILED: max_abs_diff={result.max_abs_diff:.3e} "
            f"> atol={args.atol:.3e}"
        )


if __name__ == "__main__":
    main()