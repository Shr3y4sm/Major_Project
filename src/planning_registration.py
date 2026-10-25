"""Guardrailed preparation of planning MRI for a common surgical grid.

This module deliberately performs *affine resampling only*. It makes a
planning volume comparable with a reference MRI while preserving the original
geometry and recording provenance. It never claims clinical registration: a
rigid/deformable optimisation and surgeon-reviewed QA are separate steps.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Optional

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to

__all__ = [
    "PlanningRegistrationResult",
    "extract_b0",
    "prepare_planning_volume",
]


@dataclass(frozen=True)
class PlanningRegistrationResult:
    """Provenance for a planning volume put onto a reference MRI grid."""

    output_path: str
    report_path: str
    reference_shape: tuple[int, int, int]
    moving_shape: tuple[int, ...]
    selected_volume: Optional[int]
    status: str
    review_required: bool


def extract_b0(dwi: np.ndarray, bvals_path: Optional[str] = None
               ) -> tuple[np.ndarray, Optional[int]]:
    """Select the first b≈0 frame from a DWI series, preserving 3D inputs."""
    data = np.asarray(dwi)
    if data.ndim == 3:
        return data, None
    if data.ndim != 4:
        raise ValueError(f"Planning volume must be 3D or 4D, got {data.shape}")
    index = 0
    if bvals_path:
        values = np.loadtxt(bvals_path, ndmin=1)
        if values.size != data.shape[3]:
            raise ValueError(
                f"b-values count ({values.size}) does not match DWI frames "
                f"({data.shape[3]})")
        candidates = np.flatnonzero(values <= 100)
        if candidates.size:
            index = int(candidates[0])
    return data[..., index], index


def prepare_planning_volume(reference_path: str, moving_path: str,
                            output_path: str, report_path: str,
                            bvals_path: Optional[str] = None
                            ) -> PlanningRegistrationResult:
    """Resample a structural/planning volume to a reference MRI grid.

    The output is suitable for a visual registration QA step only. Its status
    remains ``affine-resampled-review-required`` because no optimisation or
    clinical acceptance threshold has been applied.
    """
    reference = nib.load(reference_path)
    moving = nib.load(moving_path)
    if len(reference.shape) != 3:
        raise ValueError(f"Reference must be 3D, got {reference.shape}")
    selected, selected_index = extract_b0(
        moving.get_fdata(dtype=np.float32), bvals_path)
    selected_img = nib.Nifti1Image(selected.astype(np.float32), moving.affine)
    aligned = resample_from_to(
        selected_img, (reference.shape, reference.affine), order=1, cval=0.0)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    nib.save(aligned, output_path)

    result = PlanningRegistrationResult(
        output_path=output_path,
        report_path=report_path,
        reference_shape=tuple(int(x) for x in reference.shape),
        moving_shape=tuple(int(x) for x in moving.shape),
        selected_volume=selected_index,
        status="affine-resampled-review-required",
        review_required=True,
    )
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(asdict(result), handle, indent=2)
    return result
