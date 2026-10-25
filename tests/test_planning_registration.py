"""Tests for safe planning-volume preparation."""

from __future__ import annotations

import json

import nibabel as nib
import numpy as np

from src.planning_registration import extract_b0, prepare_planning_volume


def test_extract_b0_uses_first_low_b_value() -> None:
    dwi = np.stack([np.full((3, 4, 5), value) for value in (9, 3, 7)], axis=-1)
    selected, index = extract_b0(dwi)
    assert index == 0
    assert np.all(selected == 9)


def test_prepare_planning_volume_resamples_b0_and_marks_review(tmp_path) -> None:
    affine = np.eye(4)
    ref = tmp_path / "reference.nii.gz"
    moving = tmp_path / "dwi.nii.gz"
    bvals = tmp_path / "dwi.bval"
    output = tmp_path / "aligned_b0.nii.gz"
    report = tmp_path / "registration.json"
    nib.save(nib.Nifti1Image(np.ones((5, 6, 7), dtype=np.float32), affine), ref)
    dwi = np.zeros((3, 4, 5, 2), dtype=np.float32)
    dwi[..., 1] = 4
    nib.save(nib.Nifti1Image(dwi, affine), moving)
    bvals.write_text("1000 0\n", encoding="utf-8")

    result = prepare_planning_volume(
        str(ref), str(moving), str(output), str(report), str(bvals))

    assert result.status == "affine-resampled-review-required"
    assert result.review_required is True
    assert result.selected_volume == 1
    assert nib.load(output).shape == (5, 6, 7)
    assert json.loads(report.read_text(encoding="utf-8"))["selected_volume"] == 1
