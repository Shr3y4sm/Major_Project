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
from src.postprocess import (cleanup_prediction, rano_classification,
                             rano_diameters, tumor_flag)
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
        "### Volumetric analysis",
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
        lines += ["", f"> Segmentation source: **{source}**"]
    if model_warning:
        lines += ["", "> " + model_warning]
    if flag:
        lines += ["", "> " + flag]
    lines += ["",
              "> Research visualization only — not for diagnosis. "
              "Requires radiologist verification."]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Longitudinal prior comparison (RANO-style delta)
# ---------------------------------------------------------------------------

def _spd(wt_mask: np.ndarray, spacing) -> float:
    r = rano_diameters(wt_mask, spacing)
    return float(r["dmax_mm"]) * float(r["dperp_mm"])


def _synthesize_demo_prior(current_seg: np.ndarray, scenario: str):
    """Synthetic same-shape prior for demo only (no registration).

    Regression -> prior LARGER (dilated shell) => current reads as shrinkage.
    Progression -> prior SMALLER (eroded) => current reads as growth.
    """
    if scenario not in ("Regression (tumour shrank)",
                        "Progression (tumour grew)"):
        return None
    from scipy.ndimage import binary_dilation, binary_erosion
    prior = current_seg.astype(np.int16).copy()
    wt = np.isin(current_seg, [1, 2, 3, 4])
    if scenario.startswith("Regression"):
        prior[binary_dilation(wt, iterations=3) & ~wt] = 2
    else:
        prior[~binary_erosion(wt, iterations=3)] = 0
    prior[prior == 4] = 3
    return prior


def _load_prior_segmentation(prior_path, demo_scenario, current_seg):
    if demo_scenario and demo_scenario != "Off":
        return _synthesize_demo_prior(current_seg, demo_scenario)
    if prior_path:
        try:
            prior = nib.load(prior_path).get_fdata().astype(np.int16)
            prior[prior == 4] = 3
            return prior
        except Exception:
            return None
    return None


def _compare_to_prior(current_seg: np.ndarray, prior_seg, metadata,
                      spacing, current_volumes, synthetic: bool = False) -> dict:
    if prior_seg is None:
        return {"ok": False, "note": "no prior supplied"}
    if prior_seg.shape != current_seg.shape:
        return {"ok": False, "note":
                (f"prior shape {tuple(prior_seg.shape)} does not match "
                 f"current {tuple(current_seg.shape)} — volumetric comparison "
                 "requires a same-space prior (no resampling applied)")}
    prior = prior_seg.astype(np.int16).copy()
    prior[prior == 4] = 3
    prior_volumes = _compute_volumes(prior, metadata)
    rows = []
    for name in ("Necrotic Core", "Peritumoral Edema", "Enhancing Tumor",
                 "Total Whole Tumor"):
        p = float(prior_volumes[name])
        c = float(current_volumes[name])
        d = c - p
        rows.append((name, p, c, d, (d / p * 100.0) if p > 1e-9 else None))
    call, call_pct = rano_classification(
        _spd(np.isin(current_seg, [1, 2, 3, 4]), spacing),
        _spd(np.isin(prior, [1, 2, 3, 4]), spacing))
    return {"ok": True, "rows": rows, "call": call, "call_pct": call_pct,
            "synthetic": synthetic}


def _comparison_markdown(cmp: dict) -> str:
    if not cmp.get("ok"):
        return "" if cmp.get("note") == "no prior supplied" else \
            f"\n\n### Comparison to prior study\n\n> ⚠️ {cmp['note']}\n"
    lines = ["", "### Comparison to prior study", "",
             "| Structure | Prior (cm³) | Current (cm³) | Δ (cm³) | Δ (%) |",
             "|-----------|------------:|--------------:|--------:|------:|"]
    for name, p, c, d, pct in cmp["rows"]:
        lines.append(f"| {name} | {p:.2f} | {c:.2f} | {d:+.2f} | "
                     f"{'—' if pct is None else f'{pct:+.1f}'} |")
    cp = cmp.get("call_pct")
    lines += ["", f"**RANO-style response:** {cmp['call']}"
              + (f" ({cp:+.1f}% SPD vs prior)" if cp is not None else ""), "",
              "> Based on whole-tumor SPD: Progressive ≥ +25%, Partial "
              "response ≤ −50%, otherwise Stable. Research use — "
              "requires radiologist review."]
    if cmp.get("synthetic"):
        lines += ["", "> *Synthetic demo prior (demo only — not a real prior "
                  "study).*"]
    return "\n".join(lines)


def _comparison_html(cmp: dict) -> str:
    if not cmp.get("ok"):
        return "" if cmp.get("note") == "no prior supplied" else \
            f"<p><b>Comparison to prior:</b> {cmp['note']}</p>"
    rows = "".join(
        f"<tr><td>{n}</td><td>{p:.2f}</td><td>{c:.2f}</td><td>{d:+.2f}</td>"
        f"<td>{'—' if pct is None else f'{pct:+.1f}'}</td></tr>"
        for n, p, c, d, pct in cmp["rows"])
    cp = cmp.get("call_pct")
    return (f"<table border='1' cellpadding='6' cellspacing='0'>"
            f"<tr><th>Structure</th><th>Prior cm3</th><th>Current cm3</th>"
            f"<th>Delta cm3</th><th>Delta %</th></tr>{rows}</table>"
            f"<p><b>RANO-style response:</b> {cmp['call']}"
            f"{f' ({cp:+.1f}% SPD)' if cp is not None else ''}</p>"
            f"<p><small>Progressive &ge; +25% SPD, Partial response "
            f"&le; &minus;50% SPD, otherwise Stable.</small></p>"
            f"{'<br><i>Synthetic demo prior.</i>' if cmp.get('synthetic') else ''}")


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
        "Model health: " + "; ".join(reasons) +
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
        "### Geometric estimate (partial data — not AI segmentation)",
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
              "> Enhancing tumor needs T1ce; edema needs FLAIR. "
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
                  show_classes: list[int], cleanup_min_vox: int,
                  prior_path: Optional[str] = None,
                  demo_prior: str = "Off"):
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
    prior_seg = _load_prior_segmentation(prior_path, demo_prior, seg_analysis)
    cmp = _compare_to_prior(
        seg_analysis, prior_seg, metadata, spacing, volumes,
        synthetic=(prior_path is None
                   and demo_prior not in (None, "", "Off")))
    metrics_md += _comparison_markdown(cmp)
    report_text = build_ingestion_report(warnings) + f"\nModel: {ONNX_PATH}"
    report_text += f"\nSegmentation source: {seg_source}"
    if model_warning:
        report_text += f"\n{model_warning}"
    if b_report:
        report_text += f"\n\n{b_report.to_markdown()}"
    report = write_html_report(os.path.join(workdir, "report.html"), volumes,
                               rano, snaps,
                               report_text,
                               scans.source,
                               comparison=_comparison_html(cmp))
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
        warn_md = ("### Ingestion\n\n" + build_ingestion_report(warnings)
                   if warnings else "")
    if model_warning:
        warn_md += ("\n\n" if warn_md else "") + "> " + model_warning
    if flag:
        warn_md += ("\n\n" if warn_md else "") + "> " + flag
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
    prior_seg_file: Optional[gr.File] = None,
    demo_prior: str = "Off",
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
                          [int(c) for c in show_classes], cleanup_min_vox,
                          prior_path=(prior_seg_file.name
                                      if prior_seg_file is not None else None),
                          demo_prior=demo_prior)
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
                warn_md = ("### Partial-scan analysis (T2 signal, non-diagnostic)\n\n"
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
                error_md = (f"### Ingestion note\n\n{exc}\n\n"
                            "Showing partial-data analysis above "
                            "(anatomy + asymmetry heat + geometric estimate).")
                return plot_3d, plot_mpr, metrics_md, \
                    warn_md + "\n\n" + error_md, None, None
        error_md = f"### Ingestion note\n\n{exc}"
    except FileNotFoundError as exc:
        error_md = f"### Model / file note\n\n{exc}"
    except Exception as exc:  # noqa: BLE001 - surface all runtime errors in UI
        error_md = f"### Processing note\n\n{exc}"

    # Return empty placeholders on error.
    empty_fig = go.Figure()
    empty_fig.update_layout(
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        font=dict(color="white"), title="No data to display",
    )
    return empty_fig, empty_fig, "", error_md, None, None


# Apple-minimal UI theme (UI-only: component variable names are unchanged so
# all .change()/.click() bindings keep working.)
_APPLE_FONT_STACK = (
    "-apple-system, BlinkMacSystemFont, 'SF Pro Display', 'SF Pro Text', "
    "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
)

_APPLE_THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=_APPLE_FONT_STACK,
    font_mono="ui-monospace, 'SF Mono', Menlo, Consolas, monospace",
).set(
    body_background_fill="#F5F5F7",
    body_background_fill_dark="#000000",
    body_text_color="#1D1D1F",
    body_text_color_dark="#F5F5F7",
    body_text_color_subdued="#86868B",
    background_fill_primary="#FFFFFF",
    background_fill_primary_dark="#1D1D1F",
    background_fill_secondary="#F5F5F7",
    background_fill_secondary_dark="#2C2C2E",
    border_color_primary="#E5E5EA",
    border_color_primary_dark="#38383A",
    color_accent_soft="#E8F1FD",
    color_accent_soft_dark="#0A2A4A",
    button_primary_background_fill="#0071E3",
    button_primary_background_fill_hover="#0077ED",
    button_primary_background_fill_dark="#0A84FF",
    button_primary_background_fill_hover_dark="#409CFF",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_primary_border_color="transparent",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_hover="#F5F5F7",
    button_secondary_background_fill_dark="#1D1D1F",
    button_secondary_background_fill_hover_dark="#2C2C2E",
    button_secondary_border_color="#D2D2D7",
    button_secondary_border_color_dark="#48484A",
    button_secondary_text_color="#1D1D1F",
    button_secondary_text_color_dark="#F5F5F7",
    block_radius="16px",
    block_title_text_size="15px",
    block_label_text_size="12px",
    block_label_text_weight="600",
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#1D1D1F",
    input_border_color="#D2D2D7",
    input_border_color_dark="#48484A",
    slider_color="#0071E3",
    slider_color_dark="#0A84FF",
    checkbox_background_color_selected="#0071E3",
    checkbox_background_color_selected_dark="#0A84FF",
)

_APPLE_CSS = """

/* ============================================================================
   NEUROVISTA — design system (Apple-minimal + Awwwards craft)
   Palette: bg #F7F7F9 · surface #FFFFFF · ink #1C1C1F · muted #6E6E73
            accent #0A84FF (blue) · secondary #00C7A6 (teal)
   Dark:  forced <html data-theme="dark"> · default to system pref
   ============================================================================ */
:root {
  --ntt-bg: #F7F7F9;
  --ntt-bg-soft: #EDEFF2;
  --ntt-surface: #FFFFFF;
  --ntt-surface-soft: #F5F5F7;
  --ntt-ink: #1C1C1F;
  --ntt-ink-2: #3F3F46;
  --ntt-ink-3: #6E6E73;
  --ntt-ink-4: #98989D;
  --ntt-border: #E2E2E6;
  --ntt-border-strong: #D2D2D7;
  --ntt-accent: #0A84FF;
  --ntt-accent-ink: #0A6FD4;
  --ntt-accent-2: #00C7A6;
  --ntt-shadow-1: 0 1px 2px rgba(28,28,31,.05), 0 10px 28px rgba(28,28,31,.06);
  --ntt-shadow-2: 0 14px 40px rgba(28,28,31,.10);
  --ntt-radius: 18px;
  --ntt-radius-sm: 12px;
  --ntt-pill: 999px;
  --ntt-font: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Display",
    "SF Pro Text", "Inter", "Helvetica Neue", Arial, sans-serif;
  --ntt-mono: "SF Mono", "IBM Plex Mono", ui-monospace, "Consolas", monospace;
}
html[data-theme="dark"] {
  --ntt-bg: #05060B; --ntt-bg-soft: #0A0C14; --ntt-surface: #11131C;
  --ntt-surface-soft: #171923; --ntt-ink: #F2F2F5; --ntt-ink-2: #C9C9D0;
  --ntt-ink-3: #A1A1A6; --ntt-ink-4: #6E6E73; --ntt-border: #2A2D38;
  --ntt-border-strong: #38383A; --ntt-accent: #3B8CFF; --ntt-accent-ink: #5B93FF;
  --ntt-accent-2: #00D4AA; --ntt-shadow-1: 0 1px 2px rgba(0,0,0,.30),
    0 10px 28px rgba(0,0,0,.34); --ntt-shadow-2: 0 14px 40px rgba(0,0,0,.42);
}
html, body { height: 100%; }
body { margin: 0; background: var(--ntt-bg); color: var(--ntt-ink);
  font-family: var(--ntt-font); font-size: 15px; line-height: 1.6;
  -webkit-font-smoothing: antialiased; text-rendering: optimizeLegibility; }
::selection { background: rgba(10,132,255,.26); color: var(--ntt-ink); }
::-webkit-scrollbar { width: 11px; height: 11px; }
::-webkit-scrollbar-track { background: var(--ntt-bg-soft); }
::-webkit-scrollbar-thumb { background: var(--ntt-border-strong);
  border-radius: 999px; border: 2px solid var(--ntt-bg); }
::-webkit-scrollbar-thumb:hover { background: var(--ntt-ink-3); }

@media (prefers-color-scheme: dark) {
  html:not([data-theme]) {
    --ntt-bg: #05060B; --ntt-bg-soft: #0A0C14; --ntt-surface: #11131C;
    --ntt-surface-soft: #171923; --ntt-ink: #F2F2F5; --ntt-ink-2: #C9C9D0;
    --ntt-ink-3: #A1A1A6; --ntt-ink-4: #6E6E73; --ntt-border: #2A2D38;
    --ntt-border-strong: #38383A; --ntt-accent: #3B8CFF; --ntt-accent-ink: #5B93FF;
    --ntt-accent-2: #00D4AA; --ntt-shadow-1: 0 1px 2px rgba(0,0,0,.30), 0 10px 28px rgba(0,0,0,.34);
    --ntt-shadow-2: 0 14px 40px rgba(0,0,0,.42);
  }
}

/* ============================================================================
   COMPONENT STYLES — nav, theme toggle, hero
   ============================================================================ */
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }

/* ---- sticky nav ---- */
.ntt-nav { position: sticky; top: 0; z-index: 60;
  background: rgba(247,247,249,.82); border-bottom: 1px solid var(--ntt-border);
  -webkit-backdrop-filter: saturate(180%) blur(18px); backdrop-filter: saturate(180%) blur(18px); }
html[data-theme="dark"] .ntt-nav { background: rgba(5,6,11,.74); }
@media (prefers-color-scheme: dark) { html:not([data-theme]) .ntt-nav { background: rgba(5,6,11,.74); } }
.ntt-row { display: flex; align-items: center; gap: 18px; padding: 11px 24px; }
.ntt-logo { display: flex; align-items: center; gap: 11px; margin-right: auto; }
.ntt-logo-mark { width: 34px; height: 34px; border-radius: 10px; display: grid; place-items: center;
  font-weight: 800; font-size: 17px; color: #fff;
  background: linear-gradient(135deg, var(--ntt-accent), var(--ntt-accent-2));
  box-shadow: 0 4px 12px rgba(10,132,255,.30); }
.ntt-logo-name { font-weight: 700; font-size: 15px; letter-spacing: -.01em; line-height: 1.15; color: var(--ntt-ink); }
.ntt-logo-tag { font-size: 11.5px; color: var(--ntt-ink-3); line-height: 1.2; }
.ntt-nav-links { display: flex; gap: 22px; }
.ntt-nav-links a { color: var(--ntt-ink-2); text-decoration: none; font-size: 13.5px; font-weight: 500; }
.ntt-nav-links a:hover { color: var(--ntt-accent); }

/* ---- theme toggle (compact pill) ---- */
.ntt-theme { display: flex; align-items: center; gap: 9px; }
.ntt-theme-label { font-size: 12px; font-weight: 500; color: var(--ntt-ink-3); user-select: none; cursor: pointer; }
.ntt-theme-input { position: absolute; opacity: 0; width: 1px; height: 1px; margin: 0; }
.ntt-theme-visual { position: relative; display: inline-block; width: 48px; height: 27px;
  border-radius: 999px; cursor: pointer; background: var(--ntt-bg-soft);
  border: 1px solid var(--ntt-border-strong); transition: background .25s, border-color .25s; }
.ntt-theme-visual::after { content: ""; position: absolute; top: 2px; left: 2px; width: 21px; height: 21px;
  border-radius: 50%; background: #fff; box-shadow: 0 1px 3px rgba(0,0,0,.25);
  transition: transform .25s cubic-bezier(.4,0,.2,1); }
.ntt-theme-input:checked + .ntt-theme-visual { background: var(--ntt-accent); border-color: var(--ntt-accent); }
.ntt-theme-input:checked + .ntt-theme-visual::after { transform: translateX(21px); }
.ntt-theme-visual svg { position: absolute; top: 50%; transform: translateY(-50%);
  width: 12px; height: 12px; pointer-events: none; transition: opacity .25s; }
.ntt-sun { right: 7px; color: #F5A623; opacity: 1; }
.ntt-moon { left: 7px; color: #E7ECF5; opacity: 0; }
.ntt-theme-input:checked + .ntt-theme-visual .ntt-sun { opacity: 0; }
.ntt-theme-input:checked + .ntt-theme-visual .ntt-moon { opacity: 1; }
.ntt-theme-visual:focus-within { outline: 2px solid var(--ntt-accent); outline-offset: 2px; }

/* ---- hero ---- */
.ntt-hero { position: relative; overflow: hidden; padding: 54px 24px 40px; }
.ntt-eyebrow { margin: 0 0 14px; font-size: 12.5px; font-weight: 600; letter-spacing: .12em;
  text-transform: uppercase; color: var(--ntt-accent); }
.ntt-hero h1 { margin: 0 0 18px; font-size: clamp(34px, 5vw, 56px); line-height: 1.05;
  font-weight: 800; letter-spacing: -.03em; color: var(--ntt-ink); }
.ntt-grad { background: linear-gradient(120deg, var(--ntt-accent), var(--ntt-accent-2));
  -webkit-background-clip: text; background-clip: text; color: transparent; }
.ntt-sub { margin: 0 0 24px; max-width: 640px; font-size: 17px; line-height: 1.55; color: var(--ntt-ink-3); }
.ntt-hero-cta { display: flex; gap: 12px; margin-bottom: 34px; }
.ntt-secondary { display: inline-block; padding: 11px 22px; border-radius: 999px; font-size: 14px;
  font-weight: 600; color: var(--ntt-ink); background: var(--ntt-surface);
  border: 1px solid var(--ntt-border-strong); box-shadow: var(--ntt-shadow-1); }
.ntt-stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 14px; max-width: 840px; }
.ntt-stat { display: flex; flex-direction: column; gap: 3px; padding: 16px 18px;
  background: var(--ntt-surface); border: 1px solid var(--ntt-border);
  border-radius: var(--ntt-radius-sm); box-shadow: var(--ntt-shadow-1); }
.ntt-num { font-size: 22px; font-weight: 750; letter-spacing: -.02em; color: var(--ntt-ink); }
.ntt-raw { font-size: 11.5px; line-height: 1.35; color: var(--ntt-ink-3); }
.ntt-aura { position: absolute; top: -140px; right: -90px; width: 400px; height: 400px;
  border-radius: 50%; z-index: -1; pointer-events: none;
  background: radial-gradient(circle, rgba(10,132,255,.16), transparent 68%); }
@media (max-width: 760px) { .ntt-stats { grid-template-columns: repeat(2, 1fr); } }
/* ==== CONTINUED ==== */

/* ---- section labels & hints ---- */
.ntt-section { margin: 6px 0 12px; font-size: 12px; font-weight: 700; letter-spacing: .12em;
  text-transform: uppercase; color: var(--ntt-ink-3); }
.ntt-hint { margin: 0 0 8px; font-size: 12.5px; color: var(--ntt-ink-3); }

/* ---- layout columns (source / results cards) ---- */
.ntt-side, .ntt-results { background: var(--ntt-surface); border: 1px solid var(--ntt-border);
  border-radius: var(--ntt-radius); padding: 20px 22px; box-shadow: var(--ntt-shadow-1); }
.ntt-col { min-width: 0; }

/* ---- upload dropzones / groups ---- */
.ntt-uploads { border: 1px dashed var(--ntt-border-strong); border-radius: var(--ntt-radius-sm);
  padding: 14px 16px; background: var(--ntt-surface-soft); margin-bottom: 8px; }
.ntt-uploads .gr-file, .ntt-uploads .gr-form { border: none !important; background: transparent !important; }

/* ---- primary button ---- */
.ntt-primary, #ntt-run { font-weight: 650 !important; font-size: 15px !important;
  border: none !important; border-radius: 12px !important;
  background: linear-gradient(180deg, var(--ntt-accent), var(--ntt-accent-ink)) !important;
  box-shadow: 0 6px 18px rgba(10,132,255,.28) !important; }
.ntt-primary:hover, #ntt-run:hover { filter: brightness(1.05); }

/* ---- accordions ---- */
.ntt-acc { border: 1px solid var(--ntt-border) !important; border-radius: var(--ntt-radius-sm) !important;
  background: var(--ntt-surface-soft) !important; }

/* ---- results panel ---- */
.ntt-results .gr-markdown, #ntt-metrics { color: var(--ntt-ink-2); }
#ntt-metrics table { width: 100%; border-collapse: collapse; }
#ntt-metrics th, #ntt-metrics td { text-align: left; padding: 9px 12px;
  border-bottom: 1px solid var(--ntt-border); font-size: 13.5px; }
#ntt-metrics th { color: var(--ntt-ink-3); font-weight: 600; }
.ntt-results .gr-plot, .ntt-results .gr-plot > div { border-radius: var(--ntt-radius-sm); overflow: hidden; }

/* ---- alert / error strip ---- */
.ntt-alert { border-radius: var(--ntt-radius-sm); }
.ntt-alert:not(:empty) { border: 1px solid rgba(255,69,58,.35); background: rgba(255,69,58,.08);
  color: #C7362B; padding: 10px 14px; border-radius: var(--ntt-radius-sm); }
html[data-theme="dark"] .ntt-alert:not(:empty) { color: #FF8E84; border-color: rgba(255,69,58,.4); }

/* ---- mode radio as segmented control ---- */
#ntt-mode { background: var(--ntt-surface-soft); border: 1px solid var(--ntt-border);
  border-radius: 12px; padding: 6px 10px; }
#ntt-mode label { font-weight: 550; }

/* ---- tabs ---- */
.ntt-results .gr-tabs button, .ntt-results .tab-nav button { font-weight: 600; }

/* ---- footer ---- */
.ntt-foot { max-width: 1180px; margin: 8px auto 0; padding: 26px 24px 44px;
  border-top: 1px solid var(--ntt-border); }
.ntt-foot-row { display: flex; align-items: center; justify-content: space-between;
  gap: 16px; flex-wrap: wrap; margin-bottom: 14px; }
.ntt-foot-mini { font-size: 13px; line-height: 1.5; color: var(--ntt-ink-2); max-width: 820px; }
.ntt-foot-mini b { color: var(--ntt-ink); }
.ntt-foot-link { font-size: 13px; font-weight: 600; color: var(--ntt-accent); text-decoration: none;
  border: 1px solid var(--ntt-border-strong); border-radius: 999px; padding: 6px 14px; }
.ntt-foot-link:hover { border-color: var(--ntt-accent); }
.ntt-meta { display: flex; flex-wrap: wrap; gap: 8px 22px; font-size: 12.5px;
  color: var(--ntt-ink-3); margin-bottom: 16px; }
.ntt-disclaimer-wrap { font-size: 12px; line-height: 1.55; color: var(--ntt-ink-3);
  border-top: 1px dashed var(--ntt-border); padding-top: 12px; }

@media (max-width: 760px) {
  .ntt-row { flex-wrap: wrap; gap: 10px; padding: 10px 16px; }
  .ntt-hero { padding: 36px 16px 28px; }
  .ntt-side, .ntt-results { padding: 16px; }
}
"""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    with gr.Blocks(title="NeuroVista — Brain Tumour Segmentation") as demo:
        # ---------------- top nav ----------------
        gr.HTML(
            f"""
            <header class="ntt-nav" id="top">
              <div class="ntt-row">
                <div class="ntt-logo">
                  <div class="ntt-logo-mark">N</div>
                  <div class="ntt-logo-text">
                    <div class="ntt-logo-name">NeuroVista</div>
                    <div class="ntt-logo-tag">Brain tumour segmentation</div>
                  </div>
                </div>
                <nav class="ntt-nav-links">
                  <a href="#input">Input</a>
                  <a href="#output">Output</a>
                </nav>
                <div class="ntt-theme">
                  <label class="ntt-theme-label" for="ntt-theme-toggle">Theme</label>
                  <input type="checkbox" id="ntt-theme-toggle" class="ntt-theme-input">
                  <label class="ntt-theme-visual" for="ntt-theme-toggle">
                    <svg class="ntt-sun" viewBox="0 0 24 24" fill="none">
                      <circle cx="12" cy="12" r="4" stroke="currentColor" stroke-width="2.2"/>
                      <path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M4.8 4.8l2.1 2.1M17.1 17.1l2.1 2.1M4.8 19.2l2.1-2.1M17.1 6.9l2.1-2.1"
                            stroke="currentColor" stroke-width="2.2" stroke-linecap="round"/>
                    </svg>
                    <svg class="ntt-moon" viewBox="0 0 24 24" fill="none">
                      <path d="M21.6 12.7A9.2 9.2 0 1 1 11.3 2.4a7 7 0 0 0 10.3 10.3z"
                            stroke="currentColor" stroke-width="2.2" stroke-linejoin="round"/>
                    </svg>
                  </label>
                </div>
              </div>
            </header>
            """
        )
        # ---------------- hero ----------------
        gr.HTML(
            f"""
            <section class="ntt-hero">
              <p class="ntt-eyebrow">NeuroVista &middot; clinical-grade segmentation</p>
              <h1>Raw MRI <span class="ntt-grad">in.</span><br>Surgical-grade report <span class="ntt-grad">out.</span></h1>
              <p class="ntt-sub">Four MRI sequences &rarr; 3D tumour volumes, surface renders and surgical exports, in under a minute, on a standard laptop.</p>
              <div class="ntt-hero-cta">
                <span class="ntt-secondary">Upload scans</span>
              </div>
              <div class="ntt-stats">
                <div class="ntt-stat"><span class="ntt-num">1 mm</span><span class="ntt-raw">isometric</span></div>
                <div class="ntt-stat"><span class="ntt-num">3 classes</span><span class="ntt-raw">necrotic &middot; oedema &middot; enhancing</span></div>
                <div class="ntt-stat"><span class="ntt-num">128&#179;</span><span class="ntt-raw">volume grid</span></div>
                <div class="ntt-stat"><span class="ntt-num">&lt; 60 s</span><span class="ntt-raw">typical run</span></div>
              </div>
              <div class="ntt-aura"></div>
            </section>
            """
        )
        # ---------------- controls ----------------
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["ntt-side"]):
                gr.Markdown('<p class="ntt-section">1 &middot; Source</p>')
                mode = gr.Radio(
                    choices=["NIfTI Files", "DICOM ZIP"],
                    value="NIfTI Files",
                    label="Source",
                    elem_id="ntt-mode",
                )
                with gr.Group(visible=True, elem_classes=["ntt-uploads"]) as nifti_group:
                    gr.Markdown('<p class="ntt-hint"><b>4 required scans</b> &mdash; T1, T1ce, T2, FLAIR.</p>')
                    t1_file = gr.File(label="T1", file_types=[".nii", ".nii.gz"])
                    t1ce_file = gr.File(label="T1ce", file_types=[".nii", ".nii.gz"])
                    t2_file = gr.File(label="T2", file_types=[".nii", ".nii.gz"])
                    flair_file = gr.File(label="FLAIR", file_types=[".nii", ".nii.gz"])
                    nifti_folder = gr.File(label="Folder or ZIP bundle (optional)", file_types=[".zip"])
                    prior_seg_file = gr.File(label="Prior segmentation (optional)", file_types=[".nii", ".nii.gz"])
                    demo_prior = gr.Radio(["Off", "Regression (tumour shrank)", "Progression (tumour grew)"],
                                          value="Off", label="Synthesize demo prior")
                with gr.Group(visible=False, elem_classes=["ntt-uploads"]) as zip_group:
                    gr.Markdown('<p class="ntt-hint"><b>One file</b> &mdash; zip of DICOM series; anonymize PHI on upload.</p>')
                    zip_file = gr.File(label="DICOM ZIP", file_types=[".zip"])
                    anonymize = gr.Checkbox(value=True, label="Anonymize on upload (strip PHI, keep geometry)")
                gr.Markdown('<p class="ntt-section">2 &middot; Configure</p>')
                run_btn = gr.Button("Run analysis", elem_classes=["ntt-primary"], elem_id="ntt-run")
                with gr.Row():
                    with gr.Column(scale=1, elem_classes=["ntt-col"]):
                        with gr.Accordion("Slice controls", open=False, elem_classes=["ntt-acc"]):
                            axial_slider = gr.Slider(0, 240, value=78, step=1, label="Axial slice")
                            sagittal_slider = gr.Slider(0, 240, value=120, step=1, label="Sagittal slice")
                            coronal_slider = gr.Slider(0, 240, value=120, step=1, label="Coronal slice")
                            bg_modality = gr.Radio(["FLAIR", "T1CE"], value="FLAIR", label="Background")
                            overlay_opacity = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Overlay opacity")
                    with gr.Column(scale=1, elem_classes=["ntt-col"]):
                        with gr.Accordion("Classes and cleanup", open=False, elem_classes=["ntt-acc"]):
                            show_classes = gr.CheckboxGroup(
                                [(CLASS_NAMES[1], 1), (CLASS_NAMES[2], 2), (CLASS_NAMES[3], 3)],
                                value=[1, 2, 3], label="Visible classes")
                            cleanup_min_vox = gr.Slider(0, 500, value=50, step=10,
                                                         label="Remove blobs smaller than (voxels)")
                    with gr.Column(scale=1, elem_classes=["ntt-col"]):
                        with gr.Accordion("Partial-scan aids", open=False, elem_classes=["ntt-acc"]):
                            show_heat = gr.Checkbox(value=True, label="Show asymmetry heat (non-specific)")
                            heat_percentile = gr.Slider(90.0, 99.9, value=98.0, step=0.1,
                                                        label="Heat threshold percentile")
                            gr.Markdown(PROTOCOL_CHECKLIST_MD)
            with gr.Column(scale=3, elem_classes=["ntt-results"]):
                gr.Markdown('<p class="ntt-section">3 &middot; Output</p>')
                error_md = gr.Markdown("", elem_classes=["ntt-alert"])
                metrics_md = gr.Markdown("", elem_id="ntt-metrics")
                with gr.Tabs():
                    with gr.Tab("3D Surface"):
                        plot_3d = gr.Plot(label="3D tumour surface")
                    with gr.Tab("MPR Viewer"):
                        plot_mpr = gr.Plot(label="Multi-planar reconstruction")
                with gr.Accordion("Exports", open=False):
                    report_html = gr.File(label="Report (HTML)")
                    downloads = gr.Files(label="Meshes and segmentation")
        # ---------------- mode toggle ----------------
        def _toggle_mode(selected: str) -> tuple[dict, dict]:
            if selected == "NIfTI Files":
                return gr.update(visible=True), gr.update(visible=False)
            return gr.update(visible=False), gr.update(visible=True)
        mode.change(
            _toggle_mode,
            inputs=[mode],
            outputs=[nifti_group, zip_group],
        )
        # ---------------- run ----------------
        run_btn.click(
            run_analysis,
            inputs=[
                mode,
                t1_file, t1ce_file, t2_file, flair_file, nifti_folder, zip_file,
                axial_slider, sagittal_slider, coronal_slider,
                bg_modality, overlay_opacity, show_classes, cleanup_min_vox,
                anonymize, show_heat, heat_percentile,
                prior_seg_file, demo_prior,
            ],
            outputs=[plot_3d, plot_mpr, metrics_md, error_md,
                     report_html, downloads],
        )
        # ---------------- footer ----------------
        gr.HTML(
            value=(
                "<footer class=\"ntt-foot\">\n"
                "  <div class=\"ntt-foot-row\">\n"
                "    <div class=\"ntt-foot-mini\"><b>NeuroVista</b> &mdash; clinical-grade brain tumour segmentation. <b>Research preview;</b> outputs require radiologist verification and are not a medical device.</div>\n"
                "    <a class=\"ntt-foot-link\" href=\"#top\">Back to top</a>\n"
                "  </div>\n"
                "  <div class=\"ntt-meta\">\n"
                "    <span>Research preview &middot; v@@APP_VERSION@@</span>\n"
                "    <span>PHI stripped on DICOM upload</span>\n"
                "    <span>BraTS-style 4-modal inference</span>\n"
                "    <span>Open-source pipeline</span>\n"
                "  </div>\n"
                "  <div class=\"ntt-disclaimer-wrap\"><b>Research visualization only</b> &mdash; not a medical device. Results require radiologist verification; not for diagnosis, treatment, or surgical navigation.</div>\n"
                "</footer>"
            ).replace("@@APP_VERSION@@", APP_VERSION),
            head=(
                "<script>\n"
                "  (function () {\n"
                "    var stored = null;\n"
                "    try { stored = localStorage.getItem('ntt-theme'); } catch (e) {}\n"
                "    var prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;\n"
                "    var theme = stored || (prefersDark ? 'dark' : 'light');\n"
                "    document.documentElement.dataset.theme = theme;\n"
                "    window.addEventListener('load', function () {\n"
                "      var toggle = document.getElementById('ntt-theme-toggle');\n"
                "      if (toggle && toggle.checked !== (theme === 'dark')) {\n"
                "        toggle.checked = (theme === 'dark');\n"
                "      }\n"
                "    });\n"
                "    document.addEventListener('change', function (e) {\n"
                "      var t = e.target;\n"
                "      if (t && t.id === 'ntt-theme-toggle') {\n"
                "        var next = t.checked ? 'dark' : 'light';\n"
                "        document.documentElement.dataset.theme = next;\n"
                "        try { localStorage.setItem('ntt-theme', next); } catch (e) {}\n"
                "      }\n"
                "    });\n"
                "  })();\n"
                "</script>"
            ),
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
        theme=_APPLE_THEME,
        css=_APPLE_CSS,
        server_name=host,
        server_port=port,
    )
