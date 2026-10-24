"""Inference adapter for the bundled public MONAI BraTS segmentation model.

The project keeps its own 3D U-Net implementation for training experiments.
This module provides a separately labelled, pretrained inference option for
the demonstration study when no locally trained checkpoint is available.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from scipy.ndimage import zoom

from src.dicom_loader import MODALITY_ORDER, PatientScans

MODEL_PATH = os.environ.get(
    "MONAI_BRATS_MODEL_PATH",
    "artifacts/monai/brats_mri_segmentation/models/model.pt",
)


def is_available(model_path: str = MODEL_PATH) -> bool:
    """Return whether the public pretrained model is installed locally."""
    return os.path.isfile(model_path)


@lru_cache(maxsize=1)
def _load_model(model_path: str):
    """Load the public BraTS 2018 SegResNet checkpoint once."""
    import torch
    from monai.networks.nets import SegResNet

    model = SegResNet(
        blocks_down=(1, 2, 2, 4),
        blocks_up=(1, 1, 1),
        init_filters=16,
        in_channels=4,
        out_channels=3,
        dropout_prob=0.2,
    )
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def _normalise_nonzero(volume: np.ndarray) -> np.ndarray:
    """Match the model bundle's channel-wise nonzero z-score normalisation."""
    out = np.asarray(volume, dtype=np.float32).copy()
    foreground = out != 0
    if not foreground.any():
        return out
    values = out[foreground]
    std = float(values.std())
    if std > 1e-6:
        out[foreground] = (values - float(values.mean())) / std
    else:
        out[foreground] = 0.0
    return out


def _pad_to_multiple_of_eight(stack: np.ndarray) -> tuple[np.ndarray, tuple[slice, ...]]:
    """Pad spatial axes so SegResNet's downsampling path remains exact."""
    spatial = stack.shape[1:]
    pads = [(0, 0)]
    restore: list[slice] = [slice(None)]
    for size in spatial:
        padded = int(np.ceil(size / 8.0) * 8)
        before = (padded - size) // 2
        after = padded - size - before
        pads.append((before, after))
        restore.append(slice(before, before + size))
    return np.pad(stack, pads, mode="constant"), tuple(restore)


def predict_brats_regions_with_explanations(
    scans: PatientScans,
    model_path: str = MODEL_PATH,
    inference_shape: tuple[int, int, int] = (128, 128, 128),
    detailed_modality_support: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Predict labels plus confidence and occlusion-based modality support.

    The model emits overlapping binary maps for tumour core, whole tumour,
    and enhancing tumour. They are converted to native BraTS labels in a
    deterministic priority order: ET (4), core (1), then edema (2).
    """
    if not is_available(model_path):
        raise FileNotFoundError(
            f"Pretrained BraTS model not found at '{model_path}'."
        )
    shapes = {tuple(scans.volumes[m].shape) for m in MODALITY_ORDER}
    if len(shapes) != 1:
        raise ValueError("Pretrained inference requires matching modality shapes.")

    import torch

    stack = np.stack(
        [_normalise_nonzero(scans.volumes[m]) for m in MODALITY_ORDER], axis=0
    )
    original_shape = tuple(int(s) for s in stack.shape[1:])
    if inference_shape != original_shape:
        factors = (1.0,) + tuple(
            target / source for target, source in zip(inference_shape, original_shape)
        )
        stack = zoom(stack, factors, order=1).astype(np.float32)
    padded, restore = _pad_to_multiple_of_eight(stack)
    model = _load_model(model_path)
    with torch.inference_mode():
        logits = model(torch.from_numpy(padded[None]).float())
        probabilities = torch.sigmoid(logits)[0].cpu().numpy()

        # Optional single-modality occlusion. It requires four additional 3D
        # model passes, so it is intentionally off during the interactive
        # review workflow.
        support: dict[str, float] = {}
        if detailed_modality_support:
            for index, modality in enumerate(MODALITY_ORDER):
                ablated = padded.copy()
                ablated[index] = 0.0
                ablated_logits = model(torch.from_numpy(ablated[None]).float())
                ablated_probabilities = torch.sigmoid(ablated_logits)[0].cpu().numpy()
                support[modality] = float(np.mean(np.abs(probabilities - ablated_probabilities)))

    probabilities = probabilities[(slice(None),) + restore[1:]]
    tumor_core, whole_tumor, enhancing = probabilities > 0.5
    labels = np.zeros(whole_tumor.shape, dtype=np.int16)
    labels[whole_tumor] = 2
    labels[tumor_core] = 1
    labels[enhancing] = 4

    # Approximate confidence for the final mutually-exclusive rendered label.
    # The bundle predicts overlapping TC/WT/ET maps, hence this confidence is
    # region probability support rather than a calibrated clinical probability.
    core_p, whole_p, enhancing_p = probabilities
    confidence = 1.0 - whole_p
    confidence[labels == 2] = whole_p[labels == 2] * (1.0 - core_p[labels == 2])
    confidence[labels == 1] = core_p[labels == 1] * (1.0 - enhancing_p[labels == 1])
    confidence[labels == 4] = enhancing_p[labels == 4]
    confidence = np.clip(confidence, 0.0, 1.0).astype(np.float32)
    if tuple(labels.shape) != original_shape:
        factors = tuple(target / source for target, source in zip(original_shape, labels.shape))
        labels = zoom(labels.astype(np.float32), factors, order=0).astype(np.int16)
        confidence = zoom(confidence, factors, order=1).astype(np.float32)
    if support:
        total = sum(support.values()) or 1.0
        support = {key: value / total for key, value in support.items()}
    return labels, confidence, support


def predict_brats_regions(
    scans: PatientScans,
    model_path: str = MODEL_PATH,
    inference_shape: tuple[int, int, int] = (128, 128, 128),
) -> np.ndarray:
    """Predict BraTS labels ``0, 1, 2, 4`` for aligned four-modal MRI."""
    return predict_brats_regions_with_explanations(
        scans, model_path=model_path, inference_shape=inference_shape
    )[0]
