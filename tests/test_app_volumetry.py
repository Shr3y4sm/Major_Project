"""Regression tests for volumetric reporting + enhancing-overlay fixes.

Covers three bugs fixed together:
  1. ``_compute_volumes`` mixed original-grid voxel counts with 128^3
     resampled spacing (systematic ~1.5x over-report).
  2. Native BraTS label 4 (enhancing) was never counted/rendered: sidecar
     labels {0,1,2,4} vs code assuming {0,1,2,3}.
  3. Metrics came from the (placeholder) model while overlays came from
     the ground-truth sidecar, so 0 cm^3 was reported next to a visible
     tumour. Now both use one source of truth + a model-health warning.
"""

from __future__ import annotations

import os

import nibabel as nib
import numpy as np

import app as app_mod
from src.dicom_loader import SpatialMetadata

BRATS2020_PATHS = {
    "t1": "sample_brats_patient/BraTS2020/BraTS20_Training_001_t1.nii",
    "t1ce": "sample_brats_patient/BraTS2020/BraTS20_Training_001_t1ce.nii",
    "t2": "sample_brats_patient/BraTS2020/BraTS20_Training_001_t2.nii",
    "flair": "sample_brats_patient/BraTS2020/BraTS20_Training_001_flair.nii",
}
BRATS2020_SEG = "sample_brats_patient/BraTS2020/BraTS20_Training_001_seg.nii"


def _have_sample() -> bool:
    return all(os.path.exists(p) for p in BRATS2020_PATHS.values()) and \
        os.path.exists(BRATS2020_SEG)


def test_compute_volumes_uses_original_spacing_exact():
    """Volumes must equal count * orig_spacing / 1000 (no 128^3 factor)."""
    spacing = (1.0, 1.0, 1.0)
    meta = {"t2": SpatialMetadata(
        affine=np.eye(4), voxel_spacing_mm=spacing,
        original_shape=(20, 20, 20), cropped_shape=(13, 17, 13),
        crop_slices=(slice(0, 20),) * 3)}
    seg = np.zeros((20, 20, 20), np.int16)
    seg[2:7, 2:7, 2:7] = 1        # 125 voxels necrotic
    seg[8:14, 8:14, 8:14] = 2     # 216 voxels edema
    seg[15:19, 15:19, 15:19] = 4  # 64 voxels native enhancing
    vols = app_mod._compute_volumes(seg, meta)
    assert abs(vols["Necrotic Core"] - 125 * 1e-3) < 1e-9, vols
    assert abs(vols["Peritumoral Edema"] - 216 * 1e-3) < 1e-9, vols
    # Label 4 counted as enhancing.
    assert abs(vols["Enhancing Tumor"] - 64 * 1e-3) < 1e-9, vols
    assert abs(vols["Total Whole Tumor"]
               - (125 + 216 + 64) * 1e-3) < 1e-9, vols
    print("PASS: test_compute_volumes_uses_original_spacing_exact")


def test_compute_volumes_nonzero_per_class_known_case():
    """Known tumour case: every class must report > 0 cm^3 (regression)."""
    if not _have_sample():
        print("SKIP: test_compute_volumes_nonzero_per_class_known_case "
              "(missing sample files)")
        return
    seg = nib.load(BRATS2020_SEG).get_fdata().astype(np.int16)
    seg[seg == 4] = 3  # same remap as _load_reference_segmentation
    meta = {"t2": SpatialMetadata(
        affine=np.eye(4), voxel_spacing_mm=(1.0, 1.0, 1.0),
        original_shape=seg.shape, cropped_shape=seg.shape,
        crop_slices=(slice(None),) * 3)}
    vols = app_mod._compute_volumes(seg, meta)
    for cls in ("Necrotic Core", "Peritumoral Edema", "Enhancing Tumor"):
        assert vols[cls] > 0, f"{cls} is 0 cm^3: {vols}"
    # BraTS 2020 Training 001 at 1mm iso: counts == mm^3.
    counts = {int(l): int((seg == l).sum()) for l in np.unique(seg)}
    assert abs(vols["Necrotic Core"] - counts[1] / 1000.0) < 1e-6
    assert abs(vols["Enhancing Tumor"] - counts[3] / 1000.0) < 1e-6
    print("PASS: test_compute_volumes_nonzero_per_class_known_case")


def test_reference_segmentation_remaps_label_4():
    """Sidecar load must convert native label 4 -> 3 and return the path."""
    if not _have_sample():
        print("SKIP: test_reference_segmentation_remaps_label_4 (missing samples)")
        return
    shape = nib.load(BRATS2020_SEG).shape
    seg, path = app_mod._load_reference_segmentation(BRATS2020_PATHS, shape)
    assert seg is not None and path is not None
    labels = set(int(l) for l in np.unique(seg))
    assert 4 not in labels, f"label 4 survived remap: {labels}"
    assert 3 in labels, f"enhancing (4->3) missing: {labels}"
    assert path.endswith("_seg.nii")
    print("PASS: test_reference_segmentation_remaps_label_4")


def test_reference_segmentation_missing_returns_none_tuple():
    # Use a shape that matches no bundled sample sidecar — otherwise the
    # documented shape-based fallback in _discover_segmentation_path kicks in.
    seg, path = app_mod._load_reference_segmentation(
        {"t1": "/nonexistent/t1.nii", "t1ce": "/nonexistent/t1ce.nii",
         "t2": "/nonexistent/t2.nii", "flair": "/nonexistent/flair.nii"},
        (11, 12, 13))
    assert seg is None and path is None
    print("PASS: test_reference_segmentation_missing_returns_none_tuple")


def test_3d_plot_renders_enhancing_from_remapped_sidecar():
    """After 4->3 remap the 3D figure must contain an Enhancing trace."""
    if not _have_sample():
        print("SKIP: test_3d_plot_renders_enhancing (missing samples)")
        return
    shape = nib.load(BRATS2020_SEG).shape
    seg, _ = app_mod._load_reference_segmentation(BRATS2020_PATHS, shape)
    t2 = nib.load(BRATS2020_PATHS["t2"]).get_fdata().astype(np.float32)
    fig = app_mod._build_3d_plot(seg, t2, None, show_classes=[1, 2, 3])
    names = {tr.name for tr in fig.data}
    assert "Enhancing Tumor" in names, f"no enhancing trace: {sorted(names)}"
    assert "Necrotic Core" in names, f"no necrotic trace: {sorted(names)}"
    print("PASS: test_3d_plot_renders_enhancing_from_remapped_sidecar")


def test_model_health_warning_detects_placeholder_output():
    """Single-class (non-background) prediction must raise a warning."""
    all_edema = np.full((8, 8, 8), 2, dtype=np.int16)
    warn = app_mod._model_health_warning(all_edema)
    assert "placeholder" in warn.lower() or "single tumour class" in warn, warn

    no_bg = np.zeros((8, 8, 8), np.int16)
    no_bg[:4] = 2
    no_bg[4:] = 3
    warn2 = app_mod._model_health_warning(no_bg)
    assert "no background" in warn2, warn2

    healthy = np.zeros((8, 8, 8), np.int16)
    healthy[2:5, 2:5, 2:5] = 1
    healthy[5:7, 5:7, 5:7] = 2
    saved = app_mod.ONNX_PATH
    try:
        # Marker-free path isolates the prediction-health check.
        app_mod.ONNX_PATH = "/nonexistent/model.onnx"
        assert app_mod._model_health_warning(healthy) == ""
    finally:
        app_mod.ONNX_PATH = saved
    # Real repo artifacts carry a marker -> even healthy output is flagged.
    assert "placeholder marker" in app_mod._model_health_warning(healthy)
    print("PASS: test_model_health_warning_detects_placeholder_output")


def test_metrics_markdown_includes_source_and_warning():
    vols = {"Necrotic Core": 1.0, "Peritumoral Edema": 2.0,
            "Enhancing Tumor": 3.0, "Total Whole Tumor": 6.0}
    md = app_mod._build_metrics_markdown(
        vols, source="ground-truth sidecar (x_seg.nii)",
        model_warning="🚨 Model health: test")
    assert "ground-truth sidecar" in md, md
    assert "Model health" in md, md
    md2 = app_mod._build_metrics_markdown(vols)
    assert "Segmentation source" not in md2
    print("PASS: test_metrics_markdown_includes_source_and_warning")


def test_mesh_export_includes_native_label_4():
    """Raw sidecars with label 4 must still produce an enhancing mesh."""
    from src.mesh_export import meshes_from_segmentation
    seg = np.zeros((24, 24, 24), np.int16)
    seg[4:12, 4:12, 4:12] = 4
    meshes = meshes_from_segmentation(seg)
    assert 3 in meshes, f"enhancing mesh missing for label 4: {list(meshes)}"
    print("PASS: test_mesh_export_includes_native_label_4")


def test_sidecar_discovery_never_crosses_patients():
    """Upload-cache paths: match by id first, never another patient's seg."""
    mods = ("t1", "t1ce", "t2", "flair")
    if not os.path.isdir("sample_brats_patient"):
        print("SKIP: test_sidecar_discovery_never_crosses_patients")
        return
    # Upload-style paths (cache dir, no local seg) with a known id.
    paths = {m: f"/tmp/gradio/cache/BraTS2021_00495_{m}.nii" for m in mods}
    found = app_mod._discover_segmentation_path(paths, (240, 240, 155))
    assert found is not None and "BraTS2021_00495" in os.path.basename(found), \
        f"wrong sidecar: {found}"
    # Known id with NO matching sidecar -> must not fall back to another
    # patient's tumour just because the shape matches.
    paths2 = {m: f"/tmp/gradio/cache/BraTS2099_00001_{m}.nii" for m in mods}
    assert app_mod._discover_segmentation_path(paths2, (240, 240, 155)) is None
    # No id in the name -> documented best-effort shape fallback still works.
    paths3 = {m: f"/tmp/gradio/cache/case_{m}.nii" for m in mods}
    assert app_mod._discover_segmentation_path(paths3, (240, 240, 155)) is not None
    print("PASS: test_sidecar_discovery_never_crosses_patients")


def test_ensure_checkpoint_marker_semantics():
    """Placeholder checkpoint creation must flag itself via .PLACEHOLDER."""
    import tempfile
    from src.export_onnx import _ensure_checkpoint, _placeholder_marker_path
    with tempfile.TemporaryDirectory() as d:
        ckpt = os.path.join(d, "model.pt")
        assert _ensure_checkpoint(ckpt) is True
        assert os.path.exists(ckpt)
        marker = _placeholder_marker_path(ckpt)
        assert os.path.exists(marker), "marker missing after dummy creation"
        # Existing dummy + marker -> still reported as placeholder.
        assert _ensure_checkpoint(ckpt) is True
        # Deleting the marker simulates replacing with trained weights.
        os.remove(marker)
        assert _ensure_checkpoint(ckpt) is False
    print("PASS: test_ensure_checkpoint_marker_semantics")


if __name__ == "__main__":
    test_compute_volumes_uses_original_spacing_exact()
    test_compute_volumes_nonzero_per_class_known_case()
    test_reference_segmentation_remaps_label_4()
    test_reference_segmentation_missing_returns_none_tuple()
    test_3d_plot_renders_enhancing_from_remapped_sidecar()
    test_model_health_warning_detects_placeholder_output()
    test_metrics_markdown_includes_source_and_warning()
    test_mesh_export_includes_native_label_4()
    test_ensure_checkpoint_marker_semantics()
    print("app volumetry tests PASSED")