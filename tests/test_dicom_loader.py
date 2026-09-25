"""Smoke tests for the multi-modal ingestion engine (src/dicom_loader.py).

Covers:
  * NIfTI dict/list ingestion
  * prepare_for_inference output shape + metadata
  * Synthetic DICOM series ingestion
  * Negative cases: missing modality, shape mismatch, orientation mismatch
"""

from __future__ import annotations

import os
import shutil
import tempfile

import nibabel as nib
import numpy as np

from src.dicom_loader import (
    MissingModalityError,
    OrientationMismatchError,
    ShapeMismatchError,
    load_patient_scans,
    prepare_for_inference,
)

MODALITIES = ["t1", "t1ce", "t2", "flair"]

# DICOM series descriptions per modality (inference tokens).
DICOM_SERIES_DESC = {"t1": "AX T1", "t1ce": "AX T1C", "t2": "AX T2", "flair": "AX FLAIR"}


def _make_volume(shape: tuple[int, int, int], seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    vol = rng.rand(*shape).astype(np.float32) * 100.0
    # Add a bright "brain" region so the T2 mask is non-trivial.
    vol[
        shape[0] // 4 : 3 * shape[0] // 4,
        shape[1] // 4 : 3 * shape[1] // 4,
        shape[2] // 4 : 3 * shape[2] // 4,
    ] += 200.0
    return vol


def _make_affine(shape: tuple[int, int, int], spacing: tuple[float, float, float]) -> np.ndarray:
    """Build a RAS affine with the given voxel spacing (mm)."""
    affine = np.eye(4)
    affine[0, 0] = spacing[2]  # dx
    affine[1, 1] = spacing[1]  # dy
    affine[2, 2] = spacing[0]  # dz
    return affine


def _write_nifti(path: str, data: np.ndarray, affine: np.ndarray) -> None:
    img = nib.Nifti1Image(data, affine)
    nib.save(img, path)


def _create_synthetic_dicom_series(
    series_dir: str,
    modality: str,
    n_slices: int = 5,
    shape: tuple[int, int] = (8, 8),
    spacing_mm: float = 5.0,
) -> str:
    """Create a minimal but valid multi-slice MR DICOM series with pydicom."""
    import pydicom
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import (
        ExplicitVRLittleEndian,
        generate_uid,
        SecondaryCaptureImageStorage,
    )

    os.makedirs(series_dir, exist_ok=True)
    series_uid = generate_uid()

    rng = np.random.RandomState(hash(modality) % 2**32)
    for i in range(n_slices):
        file_meta = FileMetaDataset()
        file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = Dataset()
        ds.file_meta = file_meta
        ds.PatientName = "Test^Patient"
        ds.PatientID = "TEST001"
        ds.Modality = "MR"
        ds.SeriesInstanceUID = series_uid
        ds.SeriesDescription = DICOM_SERIES_DESC[modality]
        ds.InstanceNumber = i + 1
        ds.Rows, ds.Columns = shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = spacing_mm
        ds.SpacingBetweenSlices = spacing_mm
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.ImagePositionPatient = [0.0, 0.0, i * spacing_mm]
        ds.PatientPosition = "HFS"
        ds.RescaleIntercept = 0
        ds.RescaleSlope = 1
        ds.WindowCenter = 128
        ds.WindowWidth = 256
        # Required by dicom2nifti's generic converter (set_tr_te).
        ds.RepetitionTime = 2000.0
        ds.EchoTime = 30.0

        pix = rng.randint(0, 300, size=(shape[0], shape[1]), dtype=np.uint16)
        ds.PixelData = pix.tobytes()

        pydicom.dcmwrite(os.path.join(series_dir, f"slice_{i:03d}.dcm"), ds)
    return series_uid


def _create_synthetic_dicom_patient(
    tmpdir: str,
    n_slices: int = 5,
) -> tuple[str, dict[str, str]]:
    """Create one DICOM series per modality under a root dir."""
    root = os.path.join(tmpdir, "dicom_patient")
    os.makedirs(root, exist_ok=True)
    uids: dict[str, str] = {}
    for mod in MODALITIES:
        series_dir = os.path.join(root, mod)
        uid = _create_synthetic_dicom_series(series_dir, mod, n_slices=n_slices)
        uids[mod] = uid
    return root, uids


def _create_synthetic_patient(tmpdir: str, shape: tuple[int, int, int] = (64, 64, 64)) -> dict[str, str]:
    """Create 4 synthetic NIfTI files and return modality -> path mapping."""
    spacing = (5.0, 1.0, 1.0)  # (dz, dy, dx) mm
    affine = _make_affine(shape, spacing)
    paths: dict[str, str] = {}
    for i, mod in enumerate(MODALITIES):
        p = os.path.join(tmpdir, f"patient_{mod}.nii.gz")
        _write_nifti(p, _make_volume(shape, seed=i), affine)
        paths[mod] = p
    return paths


def test_nifti_dict_ingestion() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = _create_synthetic_patient(tmp)
        scans = load_patient_scans(paths)

        assert set(scans.volumes.keys()) == set(MODALITIES)
        assert scans.source == "nifti-dict"
        for mod in MODALITIES:
            assert scans.volumes[mod].dtype == np.float32
            assert scans.volumes[mod].shape == (64, 64, 64)
            assert scans.metadata[mod].voxel_spacing_mm == (1.0, 1.0, 5.0)
            assert scans.metadata[mod].original_shape == (64, 64, 64)
    print("PASS: test_nifti_dict_ingestion")


def test_nifti_list_ingestion() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = _create_synthetic_patient(tmp)
        path_list = [paths[m] for m in MODALITIES]
        scans = load_patient_scans(path_list)

        assert set(scans.volumes.keys()) == set(MODALITIES)
        assert scans.source == "nifti-dict"
    print("PASS: test_nifti_list_ingestion")


def test_prepare_for_inference() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = _create_synthetic_patient(tmp)
        scans = load_patient_scans(paths)
        tensor, metadata = prepare_for_inference(scans, target_size=128)

        assert tensor.shape == (1, 4, 128, 128, 128)
        assert tensor.dtype == np.float32
        assert np.isfinite(tensor).all()

        for mod in MODALITIES:
            assert metadata[mod].original_shape == (64, 64, 64)
            assert metadata[mod].voxel_spacing_mm == (1.0, 1.0, 5.0)
            assert metadata[mod].crop_slices is not None
            assert metadata[mod].cropped_shape[0] > 0
    print("PASS: test_prepare_for_inference")


def test_missing_modality() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = _create_synthetic_patient(tmp)
        del paths["flair"]
        try:
            load_patient_scans(paths)
            raise AssertionError("Expected MissingModalityError")
        except MissingModalityError as e:
            assert "flair" in str(e)
    print("PASS: test_missing_modality")


def test_shape_mismatch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = _create_synthetic_patient(tmp)
        # Overwrite t2 with a different shape.
        _write_nifti(
            paths["t2"],
            _make_volume((32, 32, 32), seed=99),
            _make_affine((32, 32, 32), (5.0, 1.0, 1.0)),
        )
        try:
            load_patient_scans(paths)
            raise AssertionError("Expected ShapeMismatchError")
        except ShapeMismatchError as e:
            assert "t2" in str(e)
    print("PASS: test_shape_mismatch")


def test_orientation_mismatch() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        paths = _create_synthetic_patient(tmp)
        # Flip the t1 affine orientation (LAS instead of RAS).
        flipped = _make_affine((64, 64, 64), (5.0, 1.0, 1.0))
        flipped[0, 0] = -flipped[0, 0]
        _write_nifti(paths["t1"], _make_volume((64, 64, 64), seed=0), flipped)
        try:
            load_patient_scans(paths)
            raise AssertionError("Expected OrientationMismatchError")
        except OrientationMismatchError as e:
            assert "t1" in str(e)
    print("PASS: test_orientation_mismatch")


def test_dicom_directory_ingestion() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, _uids = _create_synthetic_dicom_patient(tmp)
        scans = load_patient_scans(root)

        assert set(scans.volumes.keys()) == set(MODALITIES)
        assert scans.source.startswith("dicom:")
        for mod in MODALITIES:
            assert scans.volumes[mod].dtype == np.float32
            assert len(scans.volumes[mod].shape) == 3
            # 5 slices x 8 x 8 voxels (reorientation may transpose axes).
            assert int(np.prod(scans.volumes[mod].shape)) == 5 * 8 * 8
            assert int(np.prod(scans.metadata[mod].original_shape)) == 5 * 8 * 8
    print("PASS: test_dicom_directory_ingestion")


def test_dicom_missing_series() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root, _uids = _create_synthetic_dicom_patient(tmp)
        # Remove the FLAIR subdir so only 3 series exist.
        shutil.rmtree(os.path.join(root, "flair"))
        try:
            load_patient_scans(root)
            raise AssertionError("Expected MissingModalityError")
        except MissingModalityError as e:
            assert "flair" in str(e)
    print("PASS: test_dicom_missing_series")


def test_synth_dicom_patient_roundtrip() -> None:
    """End-to-end over the Track-A synthetic DICOM patient (if present)."""
    import shutil
    from src.io_unified import load_unified
    from src.dicom_loader import prepare_for_inference

    root = "/tmp/synth_dicom_patient"
    if not os.path.isdir(root):
        print("SKIP: test_synth_dicom_patient_roundtrip (no /tmp patient)")
        return
    scans, warnings = load_unified(root)
    assert set(scans.volumes.keys()) == set(MODALITIES)
    tensor, _ = prepare_for_inference(scans)
    assert tensor.shape == (1, 4, 128, 128, 128)
    # ZIP twin must behave identically.
    zpath = "/tmp/synth_dicom_patient.zip"
    if os.path.exists(zpath):
        scans2, _ = load_unified(zpath)
        assert set(scans2.volumes.keys()) == set(MODALITIES)
    print("PASS: test_synth_dicom_patient_roundtrip")


def test_dicom_grouping_sorts_by_instance() -> None:
    """Same-basename slices across modalities must not interleave series."""
    from src.dicom_loader import _discover_dicom_files, _group_dicom_series

    root = "/tmp/synth_dicom_patient"
    if not os.path.isdir(root):
        print("SKIP: test_dicom_grouping_sorts_by_instance (no /tmp patient)")
        return
    files = _discover_dicom_files(root)
    groups = _group_dicom_series(files)
    assert len(groups) == 4
    for uid, flist in groups.items():
        assert len(flist) == 16, f"{uid}: {len(flist)}"
    print("PASS: test_dicom_grouping_sorts_by_instance")


def test_invalid_input_type() -> None:
    try:
        load_patient_scans(12345)  # type: ignore[arg-type]
        raise AssertionError("Expected TypeError")
    except TypeError:
        pass
    print("PASS: test_invalid_input_type")


if __name__ == "__main__":
    test_nifti_dict_ingestion()
    test_nifti_list_ingestion()
    test_prepare_for_inference()
    test_missing_modality()
    test_shape_mismatch()
    test_orientation_mismatch()
    test_dicom_directory_ingestion()
    test_dicom_missing_series()
    test_synth_dicom_patient_roundtrip()
    test_dicom_grouping_sorts_by_instance()
    test_invalid_input_type()
    print("\nAll ingestion engine tests PASSED.")