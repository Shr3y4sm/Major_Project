"""
src/inference.py
================
Inference utilities for the Phase 1 3D UNet.

Provides:
  * ``ONNXInferenceEngine`` — low-latency ONNX Runtime wrapper with
    CUDA/CPU provider fallback for executing forward passes on
    preprocessed scan arrays.
  * ``pytorch_predict`` — reference PyTorch forward pass for parity checks.
  * ``verify_onnx_equivalence`` — strict PyTorch <-> ONNX Runtime
    numerical equivalence check (default atol=1e-3).
  * Phase 1 evaluation metrics (``dice_score``, ``evaluate_patient``).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import onnxruntime as ort
import torch
from scipy.spatial.distance import directed_hausdorff

from src.model import UNet3D, load_model

__all__ = [
    "ONNXInferenceEngine",
    "pytorch_predict",
    "verify_onnx_equivalence",
    "dice_score",
    "hausdorff_distance",
    "evaluate_patient",
]

# Canonical Phase 1 input shape: (batch, channels, depth, height, width)
INPUT_SHAPE = (1, 4, 128, 128, 128)


@dataclass(frozen=True)
class EquivalenceResult:
    """Result of a PyTorch <-> ONNX Runtime equivalence check."""

    max_abs_diff: float
    mean_abs_diff: float
    passed: bool
    atol: float


class ONNXInferenceEngine:
    """ONNX Runtime inference engine for the Phase 1 3D UNet.

    Loads an exported ``.onnx`` model and executes forward passes on
    preprocessed scan arrays. Prefers the CUDA execution provider and
    automatically falls back to CPU when CUDA is unavailable.

    Args:
        onnx_path: Path to the exported ONNX model.
        providers: Optional explicit provider list. If ``None``, the
            engine attempts ``["CUDAExecutionProvider", "CPUExecutionProvider"]``
            and drops unavailable providers automatically.
        sess_options: Optional ``onnxruntime.SessionOptions``.
    """

    def __init__(
        self,
        onnx_path: str,
        providers: Optional[Sequence[str]] = None,
        sess_options: Optional[ort.SessionOptions] = None,
    ) -> None:
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"ONNX model not found at '{onnx_path}'. "
                "Run `python -m src.export_onnx` first."
            )

        if providers is None:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers,
        )

        # Resolve the actual providers in use (after ORT drops unavailable ones).
        self._active_providers: list[str] = self._session.get_providers()

        # Cache input/output metadata for fast dispatch.
        self._input_name: str = self._session.get_inputs()[0].name
        self._output_name: str = self._session.get_outputs()[0].name
        self._input_shape: tuple[int, ...] = tuple(
            self._session.get_inputs()[0].shape
        )

    @property
    def active_providers(self) -> list[str]:
        """Execution providers actually in use by the session."""
        return list(self._active_providers)

    @property
    def input_name(self) -> str:
        return self._input_name

    @property
    def output_name(self) -> str:
        return self._output_name

    def predict(self, scan: np.ndarray) -> np.ndarray:
        """Run a forward pass and return raw logits.

        Args:
            scan: Preprocessed scan array. Accepts either ``(4, 128, 128, 128)``
                (channels-first, no batch dim) or ``(1, 4, 128, 128, 128)``.

        Returns:
            Raw logits of shape ``(1, 4, 128, 128, 128)`` as float32.
        """
        if scan.ndim == 4:
            scan = scan[np.newaxis, ...]
        if scan.ndim != 5:
            raise ValueError(
                f"Expected a 4D (C,D,H,W) or 5D (B,C,D,H,W) scan array, "
                f"got shape {scan.shape}."
            )
        if scan.shape[1] != 4:
            raise ValueError(
                f"Expected 4 input channels (t1, t1ce, t2, flair), "
                f"got {scan.shape[1]}."
            )

        scan = np.ascontiguousarray(scan, dtype=np.float32)
        logits = self._session.run(
            [self._output_name], {self._input_name: scan}
        )[0]
        return np.asarray(logits, dtype=np.float32)

    def predict_class_map(self, scan: np.ndarray) -> np.ndarray:
        """Run a forward pass and return the argmax segmentation map.

        Args:
            scan: Preprocessed scan array (4D or 5D, see :meth:`predict`).

        Returns:
            Integer class map of shape ``(128, 128, 128)`` (batch dim removed).
        """
        logits = self.predict(scan)
        return np.argmax(logits, axis=1)[0].astype(np.int16)


def pytorch_predict(
    model: UNet3D,
    scan: np.ndarray,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Reference PyTorch forward pass returning raw logits.

    Args:
        model: The Phase 1 UNet3D in evaluation mode.
        scan: Preprocessed scan array (4D or 5D).
        device: Torch device (defaults to CUDA if available).

    Returns:
        Raw logits of shape ``(1, 4, 128, 128, 128)`` as float32.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if scan.ndim == 4:
        scan = scan[np.newaxis, ...]

    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(np.ascontiguousarray(scan, dtype=np.float32)).to(device)
        logits = model(x).cpu().numpy()
    return np.asarray(logits, dtype=np.float32)


def verify_onnx_equivalence(
    pytorch_model: UNet3D,
    onnx_engine: ONNXInferenceEngine,
    sample_input: np.ndarray,
    atol: float = 1e-3,
    device: Optional[torch.device] = None,
) -> EquivalenceResult:
    """Verify numerical equivalence between PyTorch and ONNX Runtime.

    Feeds an identical preprocessed scan array to both engines and
    compares the raw logits element-wise.

    Args:
        pytorch_model: Phase 1 UNet3D in evaluation mode.
        onnx_engine: Loaded :class:`ONNXInferenceEngine`.
        sample_input: Preprocessed scan array (4D or 5D).
        atol: Absolute tolerance for the equivalence check.
        device: Torch device for the PyTorch forward pass.

    Returns:
        An :class:`EquivalenceResult` with the max/mean absolute
        difference and whether the check passed.
    """
    if sample_input.ndim == 4:
        sample_input = sample_input[np.newaxis, ...]

    torch_logits = pytorch_predict(pytorch_model, sample_input, device=device)
    onnx_logits = onnx_engine.predict(sample_input)

    if torch_logits.shape != onnx_logits.shape:
        raise ValueError(
            f"Shape mismatch: PyTorch {torch_logits.shape} vs "
            f"ONNX {onnx_logits.shape}."
        )

    diff = np.abs(torch_logits - onnx_logits)
    max_abs_diff = float(diff.max())
    mean_abs_diff = float(diff.mean())
    passed = max_abs_diff <= atol

    return EquivalenceResult(
        max_abs_diff=max_abs_diff,
        mean_abs_diff=mean_abs_diff,
        passed=passed,
        atol=atol,
    )


def dice_score(pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
    """Dice score for a single class (Phase 1 implementation)."""
    p = (pred == class_id).astype(np.float32)
    t = (target == class_id).astype(np.float32)
    union = p.sum() + t.sum()
    return 1.0 if union == 0 else float(2.0 * (p * t).sum() / union)


def hausdorff_distance(pred: np.ndarray, target: np.ndarray, class_id: int) -> float:
    """Directed Hausdorff distance for a single class (Phase 1 impl)."""
    p = np.argwhere(pred == class_id)
    t = np.argwhere(target == class_id)
    if len(p) == 0 or len(t) == 0:
        return float("nan")
    if len(p) > 2000:
        p = p[np.random.choice(len(p), 2000, replace=False)]
    if len(t) > 2000:
        t = t[np.random.choice(len(t), 2000, replace=False)]
    return float(max(directed_hausdorff(p, t)[0], directed_hausdorff(t, p)[0]))


def evaluate_patient(pred: np.ndarray, target: np.ndarray) -> dict[str, dict[str, float]]:
    """Compute Phase 1 evaluation metrics for a patient (WT/TC/ET)."""
    pred_wt = np.isin(pred, [1, 2, 3]).astype(np.uint8)
    target_wt = np.isin(target, [1, 2, 3]).astype(np.uint8)
    pred_core = np.isin(pred, [1, 3]).astype(np.uint8)
    target_core = np.isin(target, [1, 3]).astype(np.uint8)
    return {
        "Whole Tumor": {
            "dice": dice_score(pred_wt, target_wt, 1),
            "hausdorff": hausdorff_distance(pred_wt, target_wt, 1),
        },
        "Tumor Core": {
            "dice": dice_score(pred_core, target_core, 1),
            "hausdorff": hausdorff_distance(pred_core, target_core, 1),
        },
        "Enhancing Tumor": {
            "dice": dice_score(pred, target, 3),
            "hausdorff": hausdorff_distance(pred, target, 3),
        },
    }