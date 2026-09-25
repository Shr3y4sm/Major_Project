"""Tests for the unified DICOM+NIfTI layer + exports (no model needed)."""

from __future__ import annotations

import os
import tempfile

import nibabel as nib
import numpy as np

from src.io_unified import discover_nifti_files, infer_nifti_modality
from src.mesh_export import export_label_meshes, write_stl_binary
from src.postprocess import cleanup_prediction, rano_diameters, tumor_flag


def _write(path, shape=(16, 16, 16)):
    img = nib.Nifti1Image(np.zeros(shape, np.float32), np.eye(4))
    nib.save(img, path)


def test_hospital_tokens():
    assert infer_nifti_modality("pt_T1-GAD.nii.gz") == "t1ce"
    assert infer_nifti_modality("scan_post_gad.nii") == "t1ce"
    assert infer_nifti_modality("ax_T2W.nii.gz") == "t2"
    assert infer_nifti_modality("brain_MPRAGE.nii.gz") == "t1"
    assert infer_nifti_modality("case_seg.nii.gz") is None


def test_folder_discovery_and_seg_skip():
    with tempfile.TemporaryDirectory() as d:
        _write(os.path.join(d, "s_T1.nii.gz"))
        _write(os.path.join(d, "s_T1-GAD.nii.gz"))
        _write(os.path.join(d, "s_T2W.nii"))
        _write(os.path.join(d, "s_FLAIR.nii.gz"))
        _write(os.path.join(d, "s_seg.nii.gz"))
        m = discover_nifti_files(d)
        assert set(m) == {"t1", "t1ce", "t2", "flair"}


def test_nested_dotnii_dirs():
    """BraTS portal style: dirs named `*.nii` holding the real file."""
    import shutil
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(os.path.join(d, "case_t1.nii"))
        _write(os.path.join(d, "case_t1.nii", "inner_t1.nii"))
        os.makedirs(os.path.join(d, "case_t1ce.nii"))
        _write(os.path.join(d, "case_t1ce.nii", "inner_t1ce.nii"))
        _write(os.path.join(d, "case_t2.nii.gz"))
        _write(os.path.join(d, "case_flair.nii.gz"))
        m = discover_nifti_files(d)
        assert set(m) == {"t1", "t1ce", "t2", "flair"}


def test_real_brats_folder_layout():
    m = discover_nifti_files(
        "sample_brats_patient/BraTS2021_00495")
    assert set(m) == {"t1", "t1ce", "t2", "flair"}


def test_stl_and_cleanup_and_rano_and_flag():
    seg = np.zeros((20, 20, 20), np.int16)
    seg[5:12, 5:12, 5:12] = 2
    seg[7:10, 7:10, 7:10] = 3
    seg[0, 0, 0] = 1  # single-voxel noise -> cleanup removes
    clean = cleanup_prediction(seg, min_voxels=5)
    assert clean[0, 0, 0] == 0 and clean[8, 8, 8] == 3
    with tempfile.TemporaryDirectory() as d:
        paths = export_label_meshes(clean, d, spacing=(1.0, 1.0, 1.0))
        assert set(paths) == {2, 3}
        for p in paths.values():
            assert os.path.getsize(p) > 84  # binary STL header+tris
            with open(p, "rb") as f:
                assert f.read(80)  # header present
    rano = rano_diameters(np.isin(clean, [1, 2, 3]), (1.0, 1.0, 1.0))
    assert rano["dmax_mm"] > 0
    assert "No measurable" in tumor_flag({"Total Whole Tumor": 0.01})
    assert "Tiny finding" in tumor_flag({"Total Whole Tumor": 0.2})
    assert tumor_flag({"Total Whole Tumor": 5.0}) == ""


if __name__ == "__main__":
    test_hospital_tokens()
    test_folder_discovery_and_seg_skip()
    test_nested_dotnii_dirs()
    test_real_brats_folder_layout()
    test_stl_and_cleanup_and_rano_and_flag()
    print("unified tests PASSED")
