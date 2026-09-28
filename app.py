"""
app.py
======
Interactive clinical web application for brain tumour segmentation.

Built on Gradio + Plotly + ONNX Runtime. Supports two input modes:
  * Mode A: 4 individual NIfTI file uploads (T1, T1ce, T2, FLAIR).
  * Mode B: A single ZIP archive containing DICOM series folders.

Pipeline:
  load_patient_scans() -> prepare_for_inference() -> ONNXInferenceEngine
  -> volumetric metrics (cm^3) + 3D surface rendering + MPR 2D viewer.

Run:
    python app.py
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

import gradio as gr
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import gaussian_filter, zoom
from skimage import measure

from src.anonymize import anonymize_dicom_zip
from src.bratsify import BratsifyConfig, BratsifyReport, save_qa_png, write_provenance
from src.dicom_loader import (
    IngestionError,
    PatientScans,
    SpatialMetadata,
    prepare_for_inference,
)
from src.inference import ONNXInferenceEngine
from src.io_unified import build_ingestion_report, load_unified
from src.mesh_export import export_label_meshes, export_nifti_seg, try_export_glb
from src.partial_analysis import (
    PROTOCOL_CHECKLIST_MD,
    asymmetry_heatmap,
    blend_heat_overlay,
    ellipsoid_estimate,
)
from src.postprocess import cleanup_prediction, rano_diameters, tumor_flag
from src.report import write_html_report

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ONNX_CANDIDATES = (os.environ.get("ONNX_PATH", ""), "artifacts/best_unet.onnx",
                   "artifacts/unet3d.onnx")
ONNX_CANDIDATES = tuple(p for p in ONNX_CANDIDATES if p)
ONNX_PATH = next((p for p in ONNX_CANDIDATES if os.path.exists(p)),
                 ONNX_CANDIDATES[0] if ONNX_CANDIDATES else "artifacts/best_unet.onnx")
TARGET_SIZE = 128
NORMAL_WT_THRESHOLD_CM3 = 0.3  # below this -> normal/tiny banner
APP_VERSION = "1.0.0"

CLASS_NAMES = {
    0: "Background",
    1: "Necrotic Core",
    2: "Peritumoral Edema",
    3: "Enhancing Tumor",
    # Native BraTS label 4 (enhancing). Sidecars are remapped to 3 at load,
    # but the key is kept so raw sidecars still render if passed through.
    4: "Enhancing Tumor",
}

# 3D surface colors (RGBA) per class.
CLASS_COLORS = {
    1: "#FF004D",  # Necrotic core
    2: "#00FF4D",  # Edema shell
    3: "#00D0FF",  # Enhancing core
    4: "#00D0FF",  # Native BraTS enhancing label
}

# MPR overlay colors (RGBA) per class.
MPR_COLORS = {
    1: (220, 20, 60, 200),    # Crimson
    2: (255, 215, 0, 200),    # Yellow
    3: (0, 255, 255, 200),    # Cyan
    4: (0, 208, 255, 200),    # Native BraTS enhancing label
}

# ---------------------------------------------------------------------------
# Lazy-loaded ONNX engine
# ---------------------------------------------------------------------------

_engine: Optional[ONNXInferenceEngine] = None


def _purge_stale_workdirs() -> None:
    """Delete brats_out_* / brats_preview_* temp dirs older than retention.

    Per-run workdirs are intentionally KEPT (Gradio serves result files from
    them after the function returns), but without cleanup they accumulate PHI
    on shared machines. Retention is controlled by ARTIFACT_RETENTION_HOURS
    (default 24h; "0" = delete at end of run via _maybe_cleanup_workdir).
    """
    try:
        hours = float(os.environ.get("ARTIFACT_RETENTION_HOURS", "24") or 24)
    except ValueError:
        hours = 24.0
    if hours <= 0:
        return  # end-of-run cleanup handles this case
    import shutil
    import tempfile
    import time
    cutoff = time.time() - hours * 3600
    tmp = tempfile.gettempdir()
    for prefix in ("brats_out_", "brats_preview_", "anon_dicom_",
                   "anon_src_", "anon_out_", "brats_dicom_",
                   "brats_convert_", "unified_nifti_", "brats_spacing_"):
        try:
            names = os.listdir(tmp)
        except OSError:
            return
        for name in names:
            if not name.startswith(prefix):
                continue
            path = os.path.join(tmp, name)
            try:
                if os.path.isdir(path) and os.path.getmtime(path) < cutoff:
                    shutil.rmtree(path, ignore_errors=True)
            except OSError:
                continue


def _maybe_cleanup_workdir(workdir: str) -> None:
    """Remove the per-run workdir when ARTIFACT_RETENTION_HOURS=0."""
    if os.environ.get("ARTIFACT_RETENTION_HOURS", "24").strip() == "0":
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)


def _get_engine() -> ONNXInferenceEngine:
    """Load the ONNX engine once and cache it."""
    global _engine
    if _engine is None:
        if not os.path.exists(ONNX_PATH):
            raise FileNotFoundError(
                f"ONNX model not found at '{ONNX_PATH}'. "
                "Run `python -m src.export_onnx` first."
            )
        _engine = ONNXInferenceEngine(ONNX_PATH)
    return _engine


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def _resample_prediction_to_original(
    pred_128: np.ndarray,
    metadata: dict[str, SpatialMetadata],
) -> np.ndarray:
    """Resample the 128^3 prediction back to the cropped-original space.

    The model predicts on the resampled (128,128,128) grid. To align
    overlays with the real FLAIR/T1ce scan and compute exact physical
    volumes, we inverse-zoom the prediction to the cropped shape, then
    place it back into the full original volume using the crop slices.

    Args:
        pred_128: (128, 128, 128) integer class map from the model.
        metadata: Per-modality SpatialMetadata (uses 't2' crop info).

    Returns:
        (D, H, W) integer class map in the original volume space.
    """
    t2_meta = metadata["t2"]
    cropped_shape = t2_meta.cropped_shape
    original_shape = t2_meta.original_shape
    crop_slices = t2_meta.crop_slices

    # Inverse-zoom from 128^3 back to the cropped shape.
    factors = [c / 128.0 for c in cropped_shape]
    pred_cropped = zoom(
        pred_128.astype(np.float32), factors, order=0
    ).astype(np.int16)

    # Place into the full original volume.
    pred_orig = np.zeros(original_shape, dtype=np.int16)
    pred_orig[crop_slices] = pred_cropped
    return pred_orig


def _compute_volumes(
    pred_orig: np.ndarray,
    metadata: dict[str, SpatialMetadata],
) -> dict[str, float]:
    """Compute exact physical volumes (cm^3) for each tumour class.

    The segmentation lives in ORIGINAL volume space, so every voxel
    occupies ``prod(voxel_spacing_mm)`` mm^3. The 128^3 resampled grid
    must NOT be used for this conversion — doing so scales all volumes
    by ``prod(cropped_shape) / prod(128)`` (observed ~1.5x over-report).

    Native BraTS label 4 (enhancing) is counted together with label 3 so
    raw sidecars and remapped predictions report identical totals.

    Args:
        pred_orig: (D, H, W) class map in original space.
        metadata: Per-modality SpatialMetadata.

    Returns:
        Mapping of class name -> volume in cm^3, plus 'Total Whole Tumor'.
    """
    t2_meta = metadata["t2"]
    orig_spacing = np.asarray(t2_meta.voxel_spacing_mm, dtype=np.float64)  # (dx,dy,dz)

    # Original-space voxel volume (mm^3 -> cm^3).
    voxel_volume_mm3 = float(np.prod(orig_spacing))
    voxel_volume_cm3 = voxel_volume_mm3 / 1000.0

    volumes: dict[str, float] = {}
    total = 0.0
    for cls_id in (1, 2, 3):
        if cls_id == 3:
            count = int(np.sum(np.isin(pred_orig, (3, 4))))
        else:
            count = int(np.sum(pred_orig == cls_id))
        vol = count * voxel_volume_cm3
        volumes[CLASS_NAMES[cls_id]] = vol
        total += vol
    volumes["Total Whole Tumor"] = total
    return volumes


def _build_metrics_markdown(volumes: dict[str, float],
                            flag: str = "",
                            source: str = "",
                            model_warning: str = "") -> str:
    """Render the volumetric metrics as a clinical Markdown summary."""
    lines = [
        "### 📊 Quantitative Volumetric Analysis",
        "",
        "| Structure | Volume (cm³) |",
        "|-----------|-------------:|",
    ]
    for name in ("Necrotic Core", "Peritumoral Edema", "Enhancing Tumor"):
        lines.append(f"| {name} | {volumes[name]:.2f} |")
    lines.append(f"| **Total Whole Tumor** | **{volumes['Total Whole Tumor']:.2f}** |")
    lines.append("")
    lines.append(
        "> Volumes computed from original voxel spacing (mm³) mapped to "
        "physical space. Whole Tumor = Necrotic + Edema + Enhancing."
    )
    if source:
        lines += ["", f"> 📂 Segmentation source: **{source}**"]
    if model_warning:
        lines += ["", "> " + model_warning]
    if flag:
        lines += ["", "> ⚠️ " + flag]
    lines += ["",
              "> 🧪 Research visualization only — not for diagnosis. "
              "Requires radiologist verification."]
    return "\n".join(lines)


def _discover_segmentation_path(
    paths: dict[str, str],
    volume_shape: Optional[tuple[int, ...]] = None,
) -> Optional[str]:
    """Find a BraTS ``*_seg.nii`` sidecar near the uploaded modalities."""
    patient_ids = {
        match.group(0)
        for path in paths.values()
        for match in [re.search(r"BraTS\d{4}_\d{5}", path, re.IGNORECASE)]
        if match
    }
    visited: set[Path] = set()
    for raw_path in paths.values():
        directory = Path(raw_path).resolve().parent
        for _ in range(5):
            if directory in visited or not directory.exists():
                break
            visited.add(directory)
            for candidate in directory.glob("*_seg.nii*"):
                if candidate.is_file() and (
                    not patient_ids
                    or any(pid.lower() in candidate.name.lower() for pid in patient_ids)
                ):
                    return str(candidate)
            directory = directory.parent

    # Gradio may copy uploads away from their original BraTS folder. For the
    # bundled sample, recover the sidecar by its shared volume shape — but
    # never cross patients: when upload names carry a BraTS id, only a sidecar
    # with the same id is acceptable (all bundled sidecars share the
    # 240x240x155 shape, so a shape-only match could pick the wrong tumour).
    sample_root = Path.cwd() / "sample_brats_patient"
    if volume_shape is not None and sample_root.is_dir():
        candidates = [c for c in sample_root.rglob("*_seg.nii*") if c.is_file()]
        if patient_ids:
            candidates = [
                c for c in candidates
                if any(pid.lower() in c.name.lower() for pid in patient_ids)
            ]
            if not candidates:
                return None  # id known, no matching sidecar -> report no sidecar
        for candidate in candidates:
            if nib.load(str(candidate)).shape == volume_shape:
                return str(candidate)
    return None


def _load_reference_segmentation(
    paths: dict[str, str],
    shape: tuple[int, ...],
) -> tuple[Optional[np.ndarray], Optional[str]]:
    """Load a same-space BraTS segmentation sidecar when one is available.

    Returns ``(segmentation, path)`` or ``(None, None)``. Native BraTS
    label 4 (enhancing tumour) is remapped to the contiguous label 3 used
    by the training notebook, so downstream volume counting, 3D/MPR
    overlays and mesh export all agree on one label space.
    """
    segmentation_path = _discover_segmentation_path(paths, shape)
    if segmentation_path is None:
        return None, None
    segmentation = nib.load(segmentation_path).get_fdata().astype(np.int16)
    if segmentation.shape != shape:
        return None, None
    segmentation[segmentation == 4] = 3
    return segmentation, segmentation_path


def _model_health_warning(pred_128: np.ndarray) -> str:
    """Return a user-facing warning when the ONNX model looks unusable.

    The repo ships a seed-42 placeholder checkpoint when no trained
    weights exist (``src.export_onnx._ensure_checkpoint``); such a model
    argmaxes to a single class for every voxel, which would otherwise
    silently report 0 cm^3 for most tumour classes. Also honours the
    ``*.PLACEHOLDER`` marker written next to placeholder exports.
    """
    reasons: list[str] = []
    classes = np.unique(pred_128)
    if classes.size == 0:
        reasons.append("model prediction is empty")
    elif classes.size == 1 and int(classes[0]) != 0:
        reasons.append(
            f"model prediction is the single tumour class {int(classes[0])} "
            "across the entire volume (not even background) — the ONNX "
            "weights are the untrained placeholder (seed-42 init), not a "
            "trained model"
        )
    elif 0 not in classes:
        reasons.append(
            "model prediction contains no background class — every voxel "
            "was labelled as tumour, which indicates placeholder/broken "
            "weights"
        )
    marker = Path(str(ONNX_PATH) + ".PLACEHOLDER")
    if marker.is_file():
        reasons.append(
            f"placeholder marker present at '{marker.name}' — retrain and "
            "re-run `python -m src.export_onnx` for real predictions"
        )
    if not reasons:
        return ""
    return (
        "🚨 Model health: " + "; ".join(reasons) +
        ". Treat displayed metrics accordingly."
    )


# ---------------------------------------------------------------------------
# 3D surface rendering
# ---------------------------------------------------------------------------


def _build_3d_plot(
    pred_orig: np.ndarray,
    t2_volume: np.ndarray,
    reference_seg: Optional[np.ndarray] = None,
    show_classes: list[int] | None = None,
) -> go.Figure:
    """Build an interactive 3D surface rendering of tumour classes.

    Uses marching_cubes on each class mask (plus a translucent brain
    surface from the T2 scan for anatomical context).

    Args:
        pred_orig: (D, H, W) class map in original space.
        t2_volume: Original T2 volume (float32) for the brain surface.

    Returns:
        A Plotly figure with Mesh3d traces.
    """
    fig = go.Figure()

    # Match notebook cell 7: smooth the thresholded brain mask before meshing.
    t2_norm = (t2_volume - t2_volume.min()) / (t2_volume.max() - t2_volume.min() + 1e-8)
    brain_mask = gaussian_filter((t2_norm > 0.15).astype(np.float32), sigma=1)
    if brain_mask.any():
        try:
            verts, faces, _, _ = measure.marching_cubes(
                brain_mask.astype(np.float32), level=0.5, step_size=2
            )
            fig.add_trace(
                go.Mesh3d(
                    x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                    color="#FFFFFF",
                    opacity=0.10,
                    name="Brain Surface",
                    showlegend=True,
                    lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2, roughness=0.5),
                    lightposition=dict(x=100, y=200, z=150),
                )
            )
        except Exception:
            pass  # No brain surface if marching cubes fails.

    # Use native BraTS label 4 for enhancing when a ground-truth sidecar exists.
    segmentation = reference_seg if reference_seg is not None else pred_orig
    class_ids = (1, 2, 4) if reference_seg is not None else (1, 2, 3)
    if show_classes is not None:
        wanted = set(show_classes)
        class_ids = tuple(c for c in class_ids if c in wanted or (c == 4 and 3 in wanted))

    # Tumour class surfaces. Only run marching_cubes when the class mask
    # contains at least one voxel — evaluates the boolean mask explicitly
    # with `.any()` (never `if mask:` on a NumPy array).
    for cls_id in class_ids:
        mask = segmentation == cls_id
        if not mask.any():
            continue
        try:
            verts, faces, _, _ = measure.marching_cubes(
                mask.astype(np.float32), level=0.5, step_size=1
            )
            fig.add_trace(
                go.Mesh3d(
                    x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                    color=CLASS_COLORS[cls_id],
                    opacity=0.35 if cls_id == 2 else 1.0,
                    name=CLASS_NAMES[cls_id],
                    showlegend=True,
                    lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2, roughness=0.5),
                    lightposition=dict(x=100, y=200, z=150),
                )
            )
        except Exception:
            continue  # Skip classes that fail to mesh.

    fig.update_layout(
        title="Clinical 3D Reconstruction",
        paper_bgcolor="#050505",
        plot_bgcolor="#050505",
        font=dict(color="white"),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=True,
        legend=dict(bgcolor="rgba(0,0,0,0.4)", bordercolor="white", borderwidth=1),
        scene=dict(
            xaxis=dict(showgrid=False, showticklabels=False, showbackground=False, visible=False),
            yaxis=dict(showgrid=False, showticklabels=False, showbackground=False, visible=False),
            zaxis=dict(showgrid=False, showticklabels=False, showbackground=False, visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=1.5, y=1.5, z=0.8)),
        ),
    )
    return fig


# ---------------------------------------------------------------------------
# MPR 2D viewer
# ---------------------------------------------------------------------------


def _overlay_segmentation(
    scan_slice: np.ndarray,
    pred_slice: np.ndarray,
    opacity: float = 0.8,
    show_classes: list[int] | None = None,
) -> np.ndarray:
    """Blend a segmentation overlay onto a grayscale scan slice (RGB)."""
    if show_classes is None:
        show_classes = [1, 2, 3]
    alpha_scale = float(np.clip(opacity, 0.0, 1.0)) / (200 / 255)
    # Normalize scan slice to [0, 1].
    s_min, s_max = scan_slice.min(), scan_slice.max()
    if s_max - s_min < 1e-8:
        gray = np.zeros_like(scan_slice)
    else:
        gray = (scan_slice - s_min) / (s_max - s_min)

    rgb = np.stack([gray, gray, gray], axis=-1)

    for cls_id, (r, g, b, a) in MPR_COLORS.items():
        mapped = 3 if cls_id == 4 else cls_id
        if mapped not in show_classes:
            continue
        mask = pred_slice == cls_id
        if mask.any():
            a_eff = min(255, int(a * alpha_scale))
            rgb[mask, 0] = rgb[mask, 0] * (1 - a_eff / 255) + (r / 255) * (a_eff / 255)
            rgb[mask, 1] = rgb[mask, 1] * (1 - a_eff / 255) + (g / 255) * (a_eff / 255)
            rgb[mask, 2] = rgb[mask, 2] * (1 - a_eff / 255) + (b / 255) * (a_eff / 255)

    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def _build_mpr_plot(
    scan_volume: np.ndarray,
    pred_orig: np.ndarray,
    axial_idx: int,
    sagittal_idx: int,
    coronal_idx: int,
    opacity: float = 0.8,
    show_classes: list[int] | None = None,
) -> go.Figure:
    """Build a 3-panel MPR viewer with segmentation overlays.

    Args:
        scan_volume: Original FLAIR (or T1ce) volume for the background.
        pred_orig: (D, H, W) class map in original space.
        axial_idx / sagittal_idx / coronal_idx: Slice indices.

    Returns:
        A Plotly figure with three heatmap subplots.
    """
    D, H, W = pred_orig.shape

    axial_idx = int(np.clip(axial_idx, 0, D - 1))
    sagittal_idx = int(np.clip(sagittal_idx, 0, W - 1))
    coronal_idx = int(np.clip(coronal_idx, 0, H - 1))

    axial = _overlay_segmentation(scan_volume[axial_idx], pred_orig[axial_idx],
                                  opacity, show_classes)
    sagittal = _overlay_segmentation(scan_volume[:, :, sagittal_idx], pred_orig[:, :, sagittal_idx],
                                     opacity, show_classes)
    coronal = _overlay_segmentation(scan_volume[:, coronal_idx, :], pred_orig[:, coronal_idx, :],
                                    opacity, show_classes)

    fig = go.Figure()
    fig.add_trace(go.Image(z=axial, name="Axial", xaxis="x", yaxis="y", hoverinfo="skip"))
    fig.add_trace(go.Image(z=sagittal, name="Sagittal", xaxis="x2", yaxis="y2", hoverinfo="skip"))
    fig.add_trace(go.Image(z=coronal, name="Coronal", xaxis="x3", yaxis="y3", hoverinfo="skip"))

    fig.update_layout(
        title="Multi-Planar Reconstruction (MPR) — Segmentation Overlay",
        paper_bgcolor="#050505",
        plot_bgcolor="#050505",
        font=dict(color="white"),
        margin=dict(l=0, r=0, b=0, t=40),
        xaxis=dict(domain=[0.0, 0.32], showticklabels=False, showgrid=False),
        yaxis=dict(domain=[0.0, 1.0], showticklabels=False, showgrid=False),
        xaxis2=dict(domain=[0.34, 0.66], showticklabels=False, showgrid=False),
        yaxis2=dict(domain=[0.0, 1.0], showticklabels=False, showgrid=False),
        xaxis3=dict(domain=[0.68, 1.0], showticklabels=False, showgrid=False),
        yaxis3=dict(domain=[0.0, 1.0], showticklabels=False, showgrid=False),
        annotations=[
            dict(text="Axial", x=0.16, y=1.05, xref="paper", yref="paper", showarrow=False, font=dict(color="white")),
            dict(text="Sagittal", x=0.50, y=1.05, xref="paper", yref="paper", showarrow=False, font=dict(color="white")),
            dict(text="Coronal", x=0.84, y=1.05, xref="paper", yref="paper", showarrow=False, font=dict(color="white")),
        ],
    )
    return fig


def _snapshot_mpr_pngs(scan_volume: np.ndarray, pred_orig: np.ndarray,
                     out_dir: str) -> list[str]:
    """Save axial/sagittal/coronal overlay PNGs for the HTML report."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    os.makedirs(out_dir, exist_ok=True)
    D, H, W = pred_orig.shape
    views = [
        ("axial", _overlay_segmentation(scan_volume[D // 2],
                                        pred_orig[D // 2])),
        ("sagittal", _overlay_segmentation(scan_volume[:, :, W // 2],
                                           pred_orig[:, :, W // 2])),
        ("coronal", _overlay_segmentation(scan_volume[:, H // 2, :],
                                          pred_orig[:, H // 2, :])),
    ]
    paths = []
    for name, rgb in views:
        p = os.path.join(out_dir, f"mpr_{name}.png")
        plt.imsave(p, rgb)
        paths.append(p)
    return paths


def _build_heat_mpr_plot(scan_volume: np.ndarray, heat: np.ndarray,
                       axial_idx: int, sagittal_idx: int, coronal_idx: int,
                       opacity: float = 0.55,
                       percentile: float = 98.0) -> go.Figure:
    """MPR with NON-SPECIFIC asymmetry heat (partial-data Tier 2).

    Orange = asymmetric T2 signal, NOT a tumor segmentation. The title and
    the caller-side banner must always carry that disclaimer.
    """
    D, H, W = scan_volume.shape
    axial_idx = int(np.clip(axial_idx, 0, D - 1))
    sagittal_idx = int(np.clip(sagittal_idx, 0, W - 1))
    coronal_idx = int(np.clip(coronal_idx, 0, H - 1))
    axial = blend_heat_overlay(scan_volume[axial_idx], heat[axial_idx],
                               opacity, percentile)
    sagittal = blend_heat_overlay(scan_volume[:, :, sagittal_idx],
                                  heat[:, :, sagittal_idx], opacity,
                                  percentile)
    coronal = blend_heat_overlay(scan_volume[:, coronal_idx, :],
                                 heat[:, coronal_idx, :], opacity,
                                 percentile)
    fig = go.Figure()
    fig.add_trace(go.Image(z=axial, name="Axial", xaxis="x", yaxis="y",
                           hoverinfo="skip"))
    fig.add_trace(go.Image(z=sagittal, name="Sagittal", xaxis="x2",
                           yaxis="y2", hoverinfo="skip"))
    fig.add_trace(go.Image(z=coronal, name="Coronal", xaxis="x3",
                           yaxis="y3", hoverinfo="skip"))
    fig.update_layout(
        title="MPR — T2 Asymmetry Heat (NON-SPECIFIC, not tumor segmentation)",
        paper_bgcolor="#050505",
        plot_bgcolor="#050505",
        font=dict(color="white"),
        margin=dict(l=0, r=0, b=0, t=40),
        xaxis=dict(domain=[0.0, 0.32], showticklabels=False, showgrid=False),
        yaxis=dict(domain=[0.0, 1.0], showticklabels=False, showgrid=False),
        xaxis2=dict(domain=[0.34, 0.66], showticklabels=False, showgrid=False),
        yaxis2=dict(domain=[0.0, 1.0], showticklabels=False, showgrid=False),
        xaxis3=dict(domain=[0.68, 1.0], showticklabels=False, showgrid=False),
        yaxis3=dict(domain=[0.0, 1.0], showticklabels=False, showgrid=False),
        annotations=[
            dict(text="Axial", x=0.16, y=1.05, xref="paper", yref="paper",
                 showarrow=False, font=dict(color="white")),
            dict(text="Sagittal", x=0.50, y=1.05, xref="paper", yref="paper",
                 showarrow=False, font=dict(color="white")),
            dict(text="Coronal", x=0.84, y=1.05, xref="paper", yref="paper",
                 showarrow=False, font=dict(color="white")),
        ],
    )
    return fig


def _partial_estimate_markdown(est: dict, heat_max: float,
                               found: list[str]) -> str:
    """Render the ellipsoid-estimate panel with honesty guardrails."""
    lines = [
        "### 📐 Geometric Estimate (partial data — NOT AI segmentation)",
        "",
        f"Source series: {found}. Heat max: {heat_max:.3f}.",
        "",
    ]
    if not est.get("found"):
        lines += [
            f"No region above threshold (p{est.get('threshold')}, "
            f"n={est.get('n_voxels')}). Likely normal scan — "
            "radiologist review required.",
        ]
    else:
        lines += [
            "| Axis | Diameter (mm) |",
            "|------|--------------:|",
            f"| D1 | {est['d1_mm']:.1f} |",
            f"| D2 | {est['d2_mm']:.1f} |",
            f"| D3 | {est['d3_mm']:.1f} |",
            f"| **Ellipsoid volume (π/6·D1·D2·D3)** | **{est['volume_cm3']:.2f} cm³** |",
            "",
            f"Heat bbox (z,y,x): {est['bbox']}, peak slice {est['slice']}.",
            "",
            "> " + str(est.get("note", "")),
        ]
    lines += ["",
              "> ⚠️ Enhancing tumor needs T1ce; edema needs FLAIR. "
              "This estimate cannot replace a 4-modality segmentation."]
    return "\n".join(lines)


def _build_anatomy_mpr_plot(scan_volume: np.ndarray,
                            axial_idx: int, sagittal_idx: int,
                            coronal_idx: int) -> go.Figure:
    """Grayscale MPR with NO segmentation overlay (preview / partial data)."""
    empty = np.zeros_like(scan_volume, dtype=np.int16)
    fig = _build_mpr_plot(scan_volume, empty, axial_idx, sagittal_idx,
                          coronal_idx, opacity=0.0, show_classes=[])
    # Re-title so users know this is anatomy-only, not a failed overlay.
    try:
        fig.layout.title.text = ("Anatomy Preview (no segmentation — "
                                 "upload all 4 modalities for tumor overlay)")
    except Exception:
        pass
    return fig


def _build_brain_only_3d_plot(volume: np.ndarray) -> go.Figure:
    """Translucent brain surface only (no tumor meshes)."""
    empty = np.zeros_like(volume, dtype=np.int16)
    fig = _build_3d_plot(empty, volume, None, show_classes=[])
    try:
        fig.layout.title.text = ("Brain Surface Preview (no tumor mesh — "
                                 "needs T1/T1ce/T2/FLAIR segmentation)")
    except Exception:
        pass
    return fig


def _clamp_idx(idx: int, n: int) -> int:
    return int(np.clip(int(idx), 0, max(0, n - 1)))


def load_available_dicom_volumes(
    input_source: str,
) -> tuple[dict[str, np.ndarray], list[str], list[str]]:
    """Convert whatever diagnostic DICOM series exist (partial OK).

    Returns ``(volumes, found_modalities, notes)`` where notes describe
    skipped scouts and any series that failed conversion. Never raises
    MissingModalityError — used for the anatomy-preview fallback.
    """
    import tempfile
    import zipfile as _zf

    from src.dicom_loader import (
        _convert_dicom_series_to_nifti,
        _discover_dicom_files,
        _group_dicom_series,
        _infer_modality_from_dicom,
        _read_dicom_series_metadata,
        _validate_slice_continuity,
    )

    work_dir = None
    notes: list[str] = []
    try:
        if _zf.is_zipfile(input_source):
            work_dir = tempfile.mkdtemp(prefix="brats_preview_")
            with _zf.ZipFile(input_source, "r") as zf:
                zf.extractall(work_dir)
            search_root = work_dir
        else:
            search_root = input_source
        files = _discover_dicom_files(search_root)
        if not files:
            return {}, [], [f"No DICOM files found under '{input_source}'."]
        groups = _group_dicom_series(files)
        volumes: dict[str, np.ndarray] = {}
        for uid, series_files in groups.items():
            meta = _read_dicom_series_metadata(series_files[0])
            mod = _infer_modality_from_dicom(meta)
            desc = str(meta.get("series_desc") or uid)
            if mod == "__scout__":
                notes.append(f"Skipped scout {desc!r} "
                             f"({len(series_files)} slices).")
                continue
            if mod is None:
                notes.append(f"Skipped unrecognized series {desc!r}.")
                continue
            if mod in volumes:
                notes.append(f"Skipped duplicate series for '{mod}' "
                             f"({desc!r}).")
                continue
            try:
                _validate_slice_continuity(series_files)
                data, _ = _convert_dicom_series_to_nifti(
                    series_files, work_dir or search_root)
                volumes[mod] = data
            except Exception as exc:
                notes.append(f"Could not convert {desc!r}: {exc}")
        return volumes, sorted(volumes.keys()), notes
    finally:
        if work_dir is not None:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)


def _run_pipeline(input_source: str | dict,
                  axial_idx: int, sagittal_idx: int, coronal_idx: int,
                  bg_modality: str, overlay_opacity: float,
                  show_classes: list[int], cleanup_min_vox: int):
    """Shared pipeline for NIfTI + DICOM. Returns all UI outputs."""
    import tempfile
    workdir = tempfile.mkdtemp(prefix="brats_out_")
    _purge_stale_workdirs()
    use_bratsify = os.environ.get("BRATSIFY", "1").lower() not in ("0", "false", "no")
    b_report: Optional[BratsifyReport] = None
    qa_png_path = os.path.join(workdir, "bratsify_qa.png")
    if use_bratsify:
        b_cfg = BratsifyConfig(qa_path=qa_png_path, qa_modality="t2")
        scans, warnings, b_report = load_unified(
            input_source, harmonize=False, ras_canonical=True,
            reference="t1ce", bratsify=True, bratsify_cfg=b_cfg
        )
    else:
        scans, warnings = load_unified(
            input_source, harmonize=True, ras_canonical=True, reference="t1ce"
        )
    tensor, metadata = prepare_for_inference(scans, target_size=TARGET_SIZE)
    engine = _get_engine()
    pred_128 = engine.predict_class_map(tensor)
    pred_orig = _resample_prediction_to_original(pred_128, metadata)
    if cleanup_min_vox > 0:
        pred_orig = cleanup_prediction(pred_orig, min_voxels=cleanup_min_vox)
    model_warning = _model_health_warning(pred_128)

    # Single source of truth: metrics, overlays, snapshots and exports must
    # all describe the SAME segmentation, otherwise the reported volumes
    # disagree with what the user sees. Prefer the ground-truth sidecar
    # (label-4 already remapped to 3) when one exists, else the prediction.
    scan_volume = scans.volumes.get(bg_modality, scans.volumes["flair"])
    ref_seg, ref_path = None, None
    if isinstance(input_source, dict):
        ref_seg, ref_path = _load_reference_segmentation(
            input_source, scans.volumes["t2"].shape)
    if ref_seg is not None:
        seg_analysis = ref_seg
        seg_source = f"ground-truth sidecar ({os.path.basename(ref_path)})"
    else:
        seg_analysis = pred_orig
        seg_source = f"model prediction ({ONNX_PATH})"
    volumes = _compute_volumes(seg_analysis, metadata)
    flag = tumor_flag(volumes, threshold_cm3=NORMAL_WT_THRESHOLD_CM3)
    metrics_md = _build_metrics_markdown(volumes, flag, source=seg_source,
                                         model_warning=model_warning)

    plot_3d = _build_3d_plot(seg_analysis, scans.volumes["t2"],
                             None, show_classes=show_classes)
    plot_mpr = _build_mpr_plot(scan_volume, seg_analysis, axial_idx,
                               sagittal_idx, coronal_idx,
                               opacity=overlay_opacity,
                               show_classes=show_classes)
    # Exports
    t2_meta = metadata["t2"]
    spacing = tuple(float(s) for s in t2_meta.voxel_spacing_mm)
    stl_paths = export_label_meshes(seg_analysis, os.path.join(workdir, "stl"),
                                    spacing=spacing)
    glb_path = try_export_glb(seg_analysis, os.path.join(workdir, "tumor.glb"),
                              spacing=spacing)
    ref_key = "flair" if "flair" in scans.metadata else "t1ce"
    nifti_path = os.path.join(workdir, "segmentation.nii.gz")
    try:
        from src.io_unified import discover_nifti_files  # noqa
        export_nifti_seg(seg_analysis, _reference_path_for_export(
            input_source, ref_key), nifti_path)
    except Exception:
        nib.save(nib.Nifti1Image(seg_analysis.astype(np.int16), np.eye(4)),
                 nifti_path)
    snaps = _snapshot_mpr_pngs(scan_volume, seg_analysis, workdir)
    rano = rano_diameters(np.isin(seg_analysis, [1, 2, 3]), spacing)
    report_text = build_ingestion_report(warnings) + f"\nModel: {ONNX_PATH}"
    report_text += f"\nSegmentation source: {seg_source}"
    if model_warning:
        report_text += f"\n{model_warning}"
    if b_report:
        report_text += f"\n\n{b_report.to_markdown()}"
    report = write_html_report(os.path.join(workdir, "report.html"), volumes,
                               rano, snaps,
                               report_text,
                               scans.source)
    stl_list = [stl_paths[k] for k in sorted(stl_paths)]
    extra_downloads: list[str] = []
    if b_report:
        if os.path.isfile(qa_png_path) and os.path.getsize(qa_png_path) > 0:
            extra_downloads.append(qa_png_path)
        prov_json_path = os.path.join(workdir, "bratsify_provenance.json")
        write_provenance(b_report, prov_json_path)
        extra_downloads.append(prov_json_path)

    if b_report:
        warn_md = b_report.to_markdown()
    else:
        warn_md = ("### ℹ️ Ingestion\n\n" + build_ingestion_report(warnings)
                   if warnings else "")
    if model_warning:
        warn_md += ("\n\n" if warn_md else "") + "> " + model_warning
    if flag:
        warn_md += ("\n\n" if warn_md else "") + "> ⚠️ " + flag
    downloads = stl_list + ([glb_path] if glb_path else []) + [nifti_path] + extra_downloads
    _maybe_cleanup_workdir(workdir)
    return plot_3d, plot_mpr, metrics_md, warn_md, report, downloads


def _reference_path_for_export(input_source, ref_key: str) -> str:
    if isinstance(input_source, dict):
        return input_source[ref_key]
    raise ValueError("no reference path for DICOM input")


def run_analysis(
    mode: str,
    t1_file: Optional[gr.File],
    t1ce_file: Optional[gr.File],
    t2_file: Optional[gr.File],
    flair_file: Optional[gr.File],
    nifti_folder: Optional[gr.File],
    zip_file: Optional[gr.File],
    axial_idx: int,
    sagittal_idx: int,
    coronal_idx: int,
    bg_modality: str,
    overlay_opacity: float,
    show_classes: list[int],
    cleanup_min_vox: int,
    anonymize: bool,
    show_heat: bool,
    heat_percentile: float,
) -> tuple[go.Figure, go.Figure, str, str, Optional[str], list[str] | None]:
    """Run the full analysis pipeline and return visual + text outputs.

    Returns:
        (plot_3d, plot_mpr, metrics_md, warn_md, report_html, downloads)
    """
    error_md = ""
    try:
        # --- Build input source (BOTH DICOM + NIfTI compatible) ---
        if mode == "NIfTI Files":
            if nifti_folder is not None:
                input_source = nifti_folder.name  # folder or ZIP bundle
            else:
                if not all([t1_file, t1ce_file, t2_file, flair_file]):
                    raise ValueError(
                        "Upload all 4 NIfTI files (T1, T1ce, T2, FLAIR) "
                        "or a single NIfTI folder/ZIP."
                    )
                input_source = {
                    "t1": t1_file.name,
                    "t1ce": t1ce_file.name,
                    "t2": t2_file.name,
                    "flair": flair_file.name,
                }
        else:  # DICOM ZIP / folder
            if zip_file is None:
                raise ValueError(
                    "Please upload a ZIP archive (or folder export) "
                    "containing DICOM series."
                )
            input_source = zip_file.name
            if anonymize:
                try:
                    input_source = anonymize_dicom_zip(input_source)
                except Exception as exc:
                    raise ValueError(
                        f"DICOM anonymization failed: {exc}") from exc

        plot_3d, plot_mpr, metrics_md, warn_md, report, downloads = \
            _run_pipeline(input_source, axial_idx, sagittal_idx, coronal_idx,
                          bg_modality.lower(), overlay_opacity,
                          [int(c) for c in show_classes], cleanup_min_vox)
        return plot_3d, plot_mpr, metrics_md, warn_md, report, downloads

    except IngestionError as exc:
        # Partial-DICOM fallback: instead of blank viewers, render the
        # anatomy that IS present (grayscale MPR + brain surface) and tell
        # the user exactly which modalities are missing. This is what makes
        # a T1+T2-only ZIP like DICOM.zip show something useful.
        if mode != "NIfTI Files" and "input_source" in dir():
            try:
                volumes, found, notes = load_available_dicom_volumes(
                    input_source)
            except Exception:
                volumes, found, notes = {}, [], []
            if volumes:
                pick = ("t2" if "t2" in volumes else
                        "flair" if "flair" in volumes else
                        "t1ce" if "t1ce" in volumes else found[0])
                vol = volumes[pick]
                D, H, W = vol.shape
                heat = None
                heat_note = ""
                if show_heat and pick in ("t2", "flair"):
                    try:
                        heat = asymmetry_heatmap(vol)
                    except Exception as exc:
                        heat_note = f"Asymmetry heat failed: {exc}."
                        heat = None
                spacing = None
                try:
                    import pydicom  # noqa: F401
                    # Spacing for the estimate: derive from converted volume
                    # via the loader metadata path when possible.
                    from src.dicom_loader import (
                        _discover_dicom_files, _group_dicom_series,
                        _read_dicom_series_metadata)
                    import tempfile as _tf
                    import zipfile as _zz
                    _root = input_source
                    _tmp = None
                    if _zz.is_zipfile(input_source):
                        _tmp = _tf.mkdtemp(prefix="brats_spacing_")
                        with _zz.ZipFile(input_source) as _zf2:
                            _zf2.extractall(_tmp)
                        _root = _tmp
                    try:
                        _files = _discover_dicom_files(_root)
                        _groups = _group_dicom_series(_files)
                        for _uid, _fl in _groups.items():
                            _m = _read_dicom_series_metadata(_fl[0])
                            from src.dicom_loader import \
                                _infer_modality_from_dicom as _inf
                            if _inf(_m) == pick:
                                import nibabel as _nib
                                from src.dicom_loader import \
                                    _convert_dicom_series_to_nifti as _cv
                                _, _aff = _cv(_fl, _tmp or _root)
                                spacing = tuple(
                                    float(abs(_aff[i, i])) for i in range(3))
                                break
                    finally:
                        if _tmp is not None:
                            import shutil
                            shutil.rmtree(_tmp, ignore_errors=True)
                except Exception:
                    spacing = None
                if spacing is None:
                    spacing = (5.0, 1.0, 1.0)
                    heat_note += (" Voxel spacing unknown — estimate uses "
                                  "fallback (5,1,1)mm.")
                est = None
                if heat is not None:
                    est = ellipsoid_estimate(heat, spacing,
                                             percentile=heat_percentile)
                plot_3d = _build_brain_only_3d_plot(vol)
                if heat is not None:
                    plot_mpr = _build_heat_mpr_plot(
                        vol, heat, _clamp_idx(axial_idx, D),
                        _clamp_idx(sagittal_idx, W),
                        _clamp_idx(coronal_idx, H),
                        opacity=overlay_opacity, percentile=heat_percentile)
                else:
                    plot_mpr = _build_anatomy_mpr_plot(
                        vol, _clamp_idx(axial_idx, D),
                        _clamp_idx(sagittal_idx, W),
                        _clamp_idx(coronal_idx, H))
                missing = [m for m in ("t1", "t1ce", "t2", "flair")
                           if m not in volumes]
                metrics_md = (_partial_estimate_markdown(
                    est, float(heat.max()) if heat is not None else 0.0,
                    found) if est is not None else "")
                warn_md = ("### 🩻 Partial-DICOM Analysis (T2 signal, honest)\n\n"
                           f"Found series: {found}. "
                           f"Missing for tumor segmentation: {missing}.\n\n"
                           + (("\n".join(f"- {n}" for n in notes) + "\n\n")
                              if notes else "")
                           + ("Orange overlay = NON-SPECIFIC asymmetry "
                              "(tumor/edema/bleed/artifact/normal) — NOT a "
                              "tumor segmentation.\n\n" if heat is not None
                              else "")
                           + (heat_note + "\n\n" if heat_note else "")
                           + "Upload all 4 modalities (T1, T1ce, T2, FLAIR) "
                           "for the full segmentation + volumetry.\n\n"
                           + "*(Note: the sample `DICOM.zip` is a normal brain scan from Zenodo 16956 — "
                           "neither tumor nor volume is expected.)*\n\n"
                           + PROTOCOL_CHECKLIST_MD)
                error_md = (f"### ⚠️ Ingestion Error\n\n{exc}\n\n"
                            "Showing partial-data analysis above "
                            "(anatomy + asymmetry heat + geometric estimate).")
                return plot_3d, plot_mpr, metrics_md, \
                    warn_md + "\n\n" + error_md, None, None
        error_md = f"### ⚠️ Ingestion Error\n\n{exc}"
    except FileNotFoundError as exc:
        error_md = f"### ⚠️ Model / File Error\n\n{exc}"
    except Exception as exc:  # noqa: BLE001 - surface all runtime errors in UI
        error_md = f"### ⚠️ Processing Error\n\n{exc}"

    # Return empty placeholders on error.
    empty_fig = go.Figure()
    empty_fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="white"), title="No data to display",
    )
    return empty_fig, empty_fig, "", error_md, None, None


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Brain Tumour Segmentation") as demo:
        gr.Markdown(
            f"""
            # 🧠 Brain Tumour Segmentation — Clinical Workflow
            **Multi-modal MRI (T1, T1ce, T2, FLAIR) → 3D UNet → Volumetric Analysis**

            **A (Radiology):** MPR viewer with overlay opacity + background
            choice + volumetry. **B (Surgery):** in-situ 3D + STL/GLB export
            + printable report. Accepts **NIfTI x4 / folder / ZIP** and
            **DICOM folder / ZIP**.

            > ⚠️ **Research visualization only (v{APP_VERSION}) — NOT a medical
            > device.** Outputs require radiologist verification; not for
            > diagnosis, treatment, or surgical navigation.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(
                    choices=["NIfTI Files", "DICOM ZIP"],
                    value="NIfTI Files",
                    label="Input Mode (DICOM + NIfTI compatible)",
                )

                with gr.Group(visible=True) as nifti_group:
                    t1_file = gr.File(label="T1 (NIfTI)", file_types=[".nii", ".nii.gz"])
                    t1ce_file = gr.File(label="T1ce (NIfTI)", file_types=[".nii", ".nii.gz"])
                    t2_file = gr.File(label="T2 (NIfTI)", file_types=[".nii", ".nii.gz"])
                    flair_file = gr.File(label="FLAIR (NIfTI)", file_types=[".nii", ".nii.gz"])
                    nifti_folder = gr.File(
                        label="…or NIfTI folder/ZIP bundle (optional override)",
                        file_types=[".zip"])

                with gr.Group(visible=False) as zip_group:
                    zip_file = gr.File(label="DICOM ZIP / folder export (.zip)",
                                       file_types=[".zip"])
                    anonymize = gr.Checkbox(
                        value=True,
                        label="Anonymize DICOM on upload (strip PHI, keep geometry)")

                run_btn = gr.Button("▶ Run Analysis", variant="primary")

                with gr.Accordion("A — MPR Slice Controls", open=True):
                    axial_slider = gr.Slider(0, 240, value=78, step=1, label="Axial Slice")
                    sagittal_slider = gr.Slider(0, 240, value=120, step=1, label="Sagittal Slice")
                    coronal_slider = gr.Slider(0, 240, value=120, step=1, label="Coronal Slice")
                    bg_modality = gr.Radio(["FLAIR", "T1CE"], value="FLAIR",
                                           label="MPR background")
                    overlay_opacity = gr.Slider(0.0, 1.0, value=0.8, step=0.05,
                                                label="Overlay opacity")

                with gr.Accordion("A+B — Classes + Cleanup", open=True):
                    show_classes = gr.CheckboxGroup(
                        [(CLASS_NAMES[1], 1), (CLASS_NAMES[2], 2),
                         (CLASS_NAMES[3], 3)],
                        value=[1, 2, 3], label="Show classes (3D + MPR)")
                    cleanup_min_vox = gr.Slider(0, 500, value=50, step=10,
                                                label="Remove blobs smaller than (voxels)")

                with gr.Accordion("🩻 Partial-DICOM Aids (T1+T2-only scans)",
                                   open=True):
                    show_heat = gr.Checkbox(
                        value=True,
                        label="Show T2 asymmetry heat (NON-SPECIFIC — not tumor)")
                    heat_percentile = gr.Slider(90.0, 99.9, value=98.0,
                                                step=0.1,
                                                label="Heat threshold percentile")
                    gr.Markdown(PROTOCOL_CHECKLIST_MD)

            with gr.Column(scale=3):
                error_md = gr.Markdown("")
                metrics_md = gr.Markdown("")

                with gr.Tabs():
                    with gr.Tab("3D Surface (B — surgical)"):
                        plot_3d = gr.Plot(label="3D Tumour Surface")
                    with gr.Tab("MPR 2D Viewer (A — radiology)"):
                        plot_mpr = gr.Plot(label="Multi-Planar Reconstruction")

                with gr.Accordion("B — Exports (STL/GLB + NIfTI-seg + report)", open=False):
                    report_html = gr.File(label="Surgical / radiology report (HTML)")
                    downloads = gr.Files(label="Meshes + segmentation downloads")

        # Toggle input groups based on mode.
        def _toggle_mode(selected: str) -> tuple[dict, dict]:
            if selected == "NIfTI Files":
                return gr.update(visible=True), gr.update(visible=False)
            return gr.update(visible=False), gr.update(visible=True)

        mode.change(
            _toggle_mode,
            inputs=[mode],
            outputs=[nifti_group, zip_group],
        )

        run_btn.click(
            run_analysis,
            inputs=[
                mode,
                t1_file, t1ce_file, t2_file, flair_file, nifti_folder, zip_file,
                axial_slider, sagittal_slider, coronal_slider,
                bg_modality, overlay_opacity, show_classes, cleanup_min_vox,
                anonymize, show_heat, heat_percentile,
            ],
            outputs=[plot_3d, plot_mpr, metrics_md, error_md,
                     report_html, downloads],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    host = os.environ.get("APP_HOST", "127.0.0.1")
    port = int(os.environ.get("APP_PORT", "7860") or 7860)
    app = build_app()
    app.launch(
        server_name=host,
        server_port=port,
        theme=gr.themes.Soft(),
    )
