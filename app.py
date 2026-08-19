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
from typing import Optional

import gradio as gr
import numpy as np
import plotly.graph_objects as go
from scipy.ndimage import zoom
from skimage import measure

from src.dicom_loader import (
    IngestionError,
    PatientScans,
    SpatialMetadata,
    load_patient_scans,
    prepare_for_inference,
)
from src.inference import ONNXInferenceEngine

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ONNX_PATH = "artifacts/unet3d.onnx"
TARGET_SIZE = 128

CLASS_NAMES = {
    0: "Background",
    1: "Necrotic Core",
    2: "Peritumoral Edema",
    3: "Enhancing Tumor",
}

# 3D surface colors (RGBA) per class.
CLASS_COLORS = {
    1: "rgba(220, 20, 60, 0.85)",    # Crimson
    2: "rgba(255, 215, 0, 0.85)",    # Yellow
    3: "rgba(0, 255, 255, 0.85)",    # Cyan
}

# MPR overlay colors (RGBA) per class.
MPR_COLORS = {
    1: (220, 20, 60, 200),    # Crimson
    2: (255, 215, 0, 200),    # Yellow
    3: (0, 255, 255, 200),    # Cyan
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


def _build_metrics_markdown(volumes: dict[str, float]) -> str:
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
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 3D surface rendering
# ---------------------------------------------------------------------------


def _build_3d_plot(
    pred_orig: np.ndarray,
    t2_volume: np.ndarray,
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

    # Translucent brain surface from the T2 brain mask.
    t2_norm = (t2_volume - t2_volume.min()) / (t2_volume.max() - t2_volume.min() + 1e-8)
    brain_mask = t2_norm > 0.15
    try:
        verts, faces, _, _ = measure.marching_cubes(
            brain_mask.astype(np.float32), level=0.5, step_size=2
        )
        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
                i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                color="rgba(200, 200, 200, 0.12)",
                opacity=0.12,
                name="Brain Surface",
                showlegend=True,
                lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2, roughness=0.5),
                lightposition=dict(x=100, y=200, z=150),
            )
        )
    except Exception:
        pass  # No brain surface if marching cubes fails.

    # Tumour class surfaces.
    for cls_id in (1, 2, 3):
        mask = (pred_orig == cls_id).astype(np.float32)
        if mask.sum() == 0:
            continue
        try:
            verts, faces, _, _ = measure.marching_cubes(mask, level=0.5, step_size=1)
            fig.add_trace(
                go.Mesh3d(
                    x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                    color=CLASS_COLORS[cls_id],
                    opacity=0.85,
                    name=CLASS_NAMES[cls_id],
                    showlegend=True,
                    lighting=dict(ambient=0.4, diffuse=0.8, specular=0.2, roughness=0.5),
                    lightposition=dict(x=100, y=200, z=150),
                )
            )
        except Exception:
            continue  # Skip classes that fail to mesh.

    fig.update_layout(
        title="3D Tumour Surface Reconstruction",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
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
) -> np.ndarray:
    """Blend a segmentation overlay onto a grayscale scan slice (RGB)."""
    # Normalize scan slice to [0, 1].
    s_min, s_max = scan_slice.min(), scan_slice.max()
    if s_max - s_min < 1e-8:
        gray = np.zeros_like(scan_slice)
    else:
        gray = (scan_slice - s_min) / (s_max - s_min)

    rgb = np.stack([gray, gray, gray], axis=-1)

    for cls_id, (r, g, b, a) in MPR_COLORS.items():
        mask = pred_slice == cls_id
        if mask.any():
            rgb[mask, 0] = rgb[mask, 0] * (1 - a / 255) + (r / 255) * (a / 255)
            rgb[mask, 1] = rgb[mask, 1] * (1 - a / 255) + (g / 255) * (a / 255)
            rgb[mask, 2] = rgb[mask, 2] * (1 - a / 255) + (b / 255) * (a / 255)

    return rgb


def _build_mpr_plot(
    scan_volume: np.ndarray,
    pred_orig: np.ndarray,
    axial_idx: int,
    sagittal_idx: int,
    coronal_idx: int,
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

    axial = _overlay_segmentation(scan_volume[axial_idx], pred_orig[axial_idx])
    sagittal = _overlay_segmentation(scan_volume[:, :, sagittal_idx], pred_orig[:, :, sagittal_idx])
    coronal = _overlay_segmentation(scan_volume[:, coronal_idx, :], pred_orig[:, coronal_idx, :])

    fig = go.Figure()
    fig.add_trace(go.Image(z=axial, name="Axial", xaxis="x1", yaxis="y1", hoverinfo="skip"))
    fig.add_trace(go.Image(z=sagittal, name="Sagittal", xaxis="x2", yaxis="y2", hoverinfo="skip"))
    fig.add_trace(go.Image(z=coronal, name="Coronal", xaxis="x3", yaxis="y3", hoverinfo="skip"))

    fig.update_layout(
        title="Multi-Planar Reconstruction (MPR) — Segmentation Overlay",
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="white"),
        margin=dict(l=0, r=0, b=0, t=40),
        xaxis1=dict(domain=[0.0, 0.32], showticklabels=False, showgrid=False),
        yaxis1=dict(domain=[0.0, 1.0], showticklabels=False, showgrid=False),
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


# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------


def run_analysis(
    mode: str,
    t1_file: Optional[gr.File],
    t1ce_file: Optional[gr.File],
    t2_file: Optional[gr.File],
    flair_file: Optional[gr.File],
    zip_file: Optional[gr.File],
    axial_idx: int,
    sagittal_idx: int,
    coronal_idx: int,
) -> tuple[go.Figure, go.Figure, str, str]:
    """Run the full analysis pipeline and return visual + text outputs.

    Returns:
        (plot_3d, plot_mpr, metrics_md, error_md)
    """
    error_md = ""
    try:
        # --- Build input source ---
        if mode == "NIfTI Files":
            if not all([t1_file, t1ce_file, t2_file, flair_file]):
                raise ValueError(
                    "Please upload all 4 NIfTI files (T1, T1ce, T2, FLAIR)."
                )
            input_source = {
                "t1": t1_file.name,
                "t1ce": t1ce_file.name,
                "t2": t2_file.name,
                "flair": flair_file.name,
            }
        else:  # DICOM ZIP
            if zip_file is None:
                raise ValueError("Please upload a ZIP archive containing DICOM series.")
            input_source = zip_file.name

        # --- Ingest + preprocess ---
        scans: PatientScans = load_patient_scans(input_source)
        tensor, metadata = prepare_for_inference(scans, target_size=TARGET_SIZE)

        # --- Inference ---
        engine = _get_engine()
        pred_128 = engine.predict_class_map(tensor)  # (128,128,128) int16

        # --- Resample prediction to original space ---
        pred_orig = _resample_prediction_to_original(pred_128, metadata)

        # --- Volumetric metrics ---
        volumes = _compute_volumes(pred_orig, metadata)
        metrics_md = _build_metrics_markdown(volumes)

        # --- 3D surface rendering ---
        t2_volume = scans.volumes["t2"]
        plot_3d = _build_3d_plot(pred_orig, t2_volume)

        # --- MPR viewer (FLAIR preferred, fallback T1ce) ---
        scan_volume = scans.volumes.get("flair") or scans.volumes["t1ce"]
        plot_mpr = _build_mpr_plot(
            scan_volume, pred_orig, axial_idx, sagittal_idx, coronal_idx
        )

        return plot_3d, plot_mpr, metrics_md, error_md

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
    return empty_fig, empty_fig, "", error_md


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Brain Tumour Segmentation") as demo:
        gr.Markdown(
            """
            # 🧠 Brain Tumour Segmentation — Clinical Workflow
            **Multi-modal MRI (T1, T1ce, T2, FLAIR) → 3D UNet → Volumetric Analysis**

            Upload either **4 NIfTI files** or a **ZIP of DICOM series** to run
            automated tumour segmentation with quantitative volume reporting.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(
                    choices=["NIfTI Files", "DICOM ZIP"],
                    value="NIfTI Files",
                    label="Input Mode",
                )

                with gr.Group(visible=True) as nifti_group:
                    t1_file = gr.File(label="T1 (NIfTI)", file_types=[".nii", ".nii.gz"])
                    t1ce_file = gr.File(label="T1ce (NIfTI)", file_types=[".nii", ".nii.gz"])
                    t2_file = gr.File(label="T2 (NIfTI)", file_types=[".nii", ".nii.gz"])
                    flair_file = gr.File(label="FLAIR (NIfTI)", file_types=[".nii", ".nii.gz"])

                with gr.Group(visible=False) as zip_group:
                    zip_file = gr.File(label="DICOM ZIP Archive", file_types=[".zip"])

                run_btn = gr.Button("▶ Run Analysis", variant="primary")

                with gr.Accordion("MPR Slice Controls", open=True):
                    axial_slider = gr.Slider(0, 127, value=64, step=1, label="Axial Slice")
                    sagittal_slider = gr.Slider(0, 127, value=64, step=1, label="Sagittal Slice")
                    coronal_slider = gr.Slider(0, 127, value=64, step=1, label="Coronal Slice")

            with gr.Column(scale=3):
                error_md = gr.Markdown("")
                metrics_md = gr.Markdown("")

                with gr.Tabs():
                    with gr.Tab("3D Surface"):
                        plot_3d = gr.Plot(label="3D Tumour Surface")
                    with gr.Tab("MPR 2D Viewer"):
                        plot_mpr = gr.Plot(label="Multi-Planar Reconstruction")

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
                t1_file, t1ce_file, t2_file, flair_file, zip_file,
                axial_slider, sagittal_slider, coronal_slider,
            ],
            outputs=[plot_3d, plot_mpr, metrics_md, error_md],
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
