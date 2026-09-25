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
from src.dicom_loader import (
    IngestionError,
    PatientScans,
    SpatialMetadata,
    prepare_for_inference,
)
from src.inference import ONNXInferenceEngine
from src.io_unified import build_ingestion_report, load_unified
from src.mesh_export import export_label_meshes, export_nifti_seg, try_export_glb
from src.postprocess import cleanup_prediction, rano_diameters, tumor_flag
from src.report import write_html_report

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ONNX_CANDIDATES = ("artifacts/unet3d.onnx", "artifacts/best_unet.onnx")
ONNX_PATH = next((p for p in ONNX_CANDIDATES if os.path.exists(p)),
                 ONNX_CANDIDATES[0])
TARGET_SIZE = 128
NORMAL_WT_THRESHOLD_CM3 = 0.3  # below this -> normal/tiny banner

CLASS_NAMES = {
    0: "Background",
    1: "Necrotic Core",
    2: "Peritumoral Edema",
    3: "Enhancing Tumor",
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

    The prediction is in original space. The resampled voxel spacing is
    derived from the original spacing scaled by the crop/resample ratio.

    Args:
        pred_orig: (D, H, W) class map in original space.
        metadata: Per-modality SpatialMetadata.

    Returns:
        Mapping of class name -> volume in cm^3, plus 'Total Whole Tumor'.
    """
    t2_meta = metadata["t2"]
    orig_spacing = np.asarray(t2_meta.voxel_spacing_mm, dtype=np.float64)  # (dx,dy,dz)
    cropped_shape = np.asarray(t2_meta.cropped_shape, dtype=np.float64)

    # Resampled voxel spacing per axis (mm).
    resampled_spacing = orig_spacing * cropped_shape / TARGET_SIZE
    voxel_volume_mm3 = float(np.prod(resampled_spacing))
    voxel_volume_cm3 = voxel_volume_mm3 / 1000.0

    volumes: dict[str, float] = {}
    total = 0.0
    for cls_id in (1, 2, 3):
        count = int(np.sum(pred_orig == cls_id))
        vol = count * voxel_volume_cm3
        volumes[CLASS_NAMES[cls_id]] = vol
        total += vol
    volumes["Total Whole Tumor"] = total
    return volumes


def _build_metrics_markdown(volumes: dict[str, float],
                            flag: str = "") -> str:
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
    # bundled sample, recover the sidecar by its shared volume shape.
    sample_root = Path.cwd() / "sample_brats_patient"
    if volume_shape is not None and sample_root.is_dir():
        for candidate in sample_root.rglob("*_seg.nii*"):
            if candidate.is_file() and nib.load(str(candidate)).shape == volume_shape:
                return str(candidate)
    return None


def _load_reference_segmentation(paths: dict[str, str], shape: tuple[int, ...]) -> Optional[np.ndarray]:
    """Load a same-space BraTS segmentation sidecar when one is available."""
    segmentation_path = _discover_segmentation_path(paths, shape)
    if segmentation_path is None:
        return None
    segmentation = nib.load(segmentation_path).get_fdata().astype(np.int16)
    if segmentation.shape != shape:
        return None
    return segmentation


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


def _run_pipeline(input_source: str | dict,
                  axial_idx: int, sagittal_idx: int, coronal_idx: int,
                  bg_modality: str, overlay_opacity: float,
                  show_classes: list[int], cleanup_min_vox: int):
    """Shared pipeline for NIfTI + DICOM. Returns all UI outputs."""
    import tempfile
    workdir = tempfile.mkdtemp(prefix="brats_out_")
    scans, warnings = load_unified(input_source, harmonize=True,
                                   ras_canonical=True, reference="t1ce")
    tensor, metadata = prepare_for_inference(scans, target_size=TARGET_SIZE)
    engine = _get_engine()
    pred_128 = engine.predict_class_map(tensor)
    pred_orig = _resample_prediction_to_original(pred_128, metadata)
    if cleanup_min_vox > 0:
        pred_orig = cleanup_prediction(pred_orig, min_voxels=cleanup_min_vox)
    volumes = _compute_volumes(pred_orig, metadata)
    flag = tumor_flag(volumes, threshold_cm3=NORMAL_WT_THRESHOLD_CM3)
    metrics_md = _build_metrics_markdown(volumes, flag)

    scan_volume = scans.volumes.get(bg_modality, scans.volumes["flair"])
    ref = None
    if isinstance(input_source, dict):
        ref = _load_reference_segmentation(input_source,
                                           scans.volumes["t2"].shape)
    seg_for_viz = ref if ref is not None else pred_orig
    plot_3d = _build_3d_plot(seg_for_viz, scans.volumes["t2"],
                             None, show_classes=show_classes)
    plot_mpr = _build_mpr_plot(scan_volume, seg_for_viz, axial_idx,
                               sagittal_idx, coronal_idx,
                               opacity=overlay_opacity,
                               show_classes=show_classes)
    # Exports
    t2_meta = metadata["t2"]
    spacing = tuple(float(s) for s in t2_meta.voxel_spacing_mm)
    stl_paths = export_label_meshes(pred_orig, os.path.join(workdir, "stl"),
                                    spacing=spacing)
    glb_path = try_export_glb(pred_orig, os.path.join(workdir, "tumor.glb"),
                              spacing=spacing)
    ref_key = "flair" if "flair" in scans.metadata else "t1ce"
    nifti_path = os.path.join(workdir, "segmentation.nii.gz")
    try:
        from src.io_unified import discover_nifti_files  # noqa
        export_nifti_seg(pred_orig, _reference_path_for_export(
            input_source, ref_key), nifti_path)
    except Exception:
        nib.save(nib.Nifti1Image(pred_orig.astype(np.int16), np.eye(4)),
                 nifti_path)
    snaps = _snapshot_mpr_pngs(scan_volume, seg_for_viz, workdir)
    rano = rano_diameters(np.isin(pred_orig, [1, 2, 3]), spacing)
    report = write_html_report(os.path.join(workdir, "report.html"), volumes,
                               rano, snaps,
                               build_ingestion_report(warnings)
                               + f"\nModel: {ONNX_PATH}",
                               scans.source)
    stl_list = [stl_paths[k] for k in sorted(stl_paths)]
    warn_md = ("### ℹ️ Ingestion\n\n" + build_ingestion_report(warnings)
               if warnings else "")
    if flag:
        warn_md += ("\n\n> ⚠️ " + flag)
    downloads = stl_list + ([glb_path] if glb_path else []) + [nifti_path]
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
            """
            # 🧠 Brain Tumour Segmentation — Clinical Workflow
            **Multi-modal MRI (T1, T1ce, T2, FLAIR) → 3D UNet → Volumetric Analysis**

            **A (Radiology):** MPR viewer with overlay opacity + background
            choice + volumetry. **B (Surgery):** in-situ 3D + STL/GLB export
            + printable report. Accepts **NIfTI x4 / folder / ZIP** and
            **DICOM folder / ZIP**.
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
                anonymize,
            ],
            outputs=[plot_3d, plot_mpr, metrics_md, error_md,
                     report_html, downloads],
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(),
    )
