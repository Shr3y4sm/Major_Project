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
from src.bratsify import BratsifyConfig, BratsifyReport, write_provenance
from src.dicom_loader import (
    IngestionError,
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
from src.pretrained_inference import (
    is_available as pretrained_brats_available,
    predict_brats_regions_with_explanations,
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


def _use_pretrained_brats_model() -> bool:
    """Select the installed public model while the local U-Net is a placeholder.

    Set ``USE_PRETRAINED_BRATS=0`` to force the project's exported ONNX U-Net.
    A trained local ONNX model takes precedence automatically once its
    ``.PLACEHOLDER`` marker has been removed.
    """
    if os.environ.get("USE_PRETRAINED_BRATS", "1").lower() in ("0", "false", "no"):
        return False
    return pretrained_brats_available() and Path(str(ONNX_PATH) + ".PLACEHOLDER").is_file()


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
    confidence_slice: np.ndarray | None = None,
) -> np.ndarray:
    """Blend segmentation and optional lower-support cues onto an MRI slice."""
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

    # Amber is reserved for tumour voxels with lower model support. It is a
    # review cue, not an uncertainty-calibrated diagnosis.
    if confidence_slice is not None:
        low_support = (pred_slice > 0) & (confidence_slice < 0.70)
        if low_support.any():
            amber = np.array([245, 158, 11], dtype=float) / 255
            alpha = 0.72
            rgb[low_support] = ((1 - alpha) * rgb[low_support]
                                + alpha * amber)

    return np.clip(rgb * 255, 0, 255).astype(np.uint8)


def _build_mpr_plot(
    scan_volume: np.ndarray,
    pred_orig: np.ndarray,
    axial_idx: int,
    sagittal_idx: int,
    coronal_idx: int,
    opacity: float = 0.8,
    show_classes: list[int] | None = None,
    confidence_map: np.ndarray | None = None,
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
                                  opacity, show_classes,
                                  None if confidence_map is None else confidence_map[axial_idx])
    sagittal = _overlay_segmentation(scan_volume[:, :, sagittal_idx], pred_orig[:, :, sagittal_idx],
                                     opacity, show_classes,
                                     None if confidence_map is None else confidence_map[:, :, sagittal_idx])
    coronal = _overlay_segmentation(scan_volume[:, coronal_idx, :], pred_orig[:, coronal_idx, :],
                                    opacity, show_classes,
                                    None if confidence_map is None else confidence_map[:, coronal_idx, :])

    fig = go.Figure()
    fig.add_trace(go.Image(z=axial, name="Axial", xaxis="x", yaxis="y", hoverinfo="skip"))
    fig.add_trace(go.Image(z=sagittal, name="Sagittal", xaxis="x2", yaxis="y2", hoverinfo="skip"))
    fig.add_trace(go.Image(z=coronal, name="Coronal", xaxis="x3", yaxis="y3", hoverinfo="skip"))

    fig.update_layout(
        title=("Multi-Planar Reconstruction — tumour overlay; amber = lower model support"
               if confidence_map is not None
               else "Multi-Planar Reconstruction (MPR) — Segmentation Overlay"),
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
                     out_dir: str,
                     confidence_map: np.ndarray | None = None) -> list[str]:
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
                                        pred_orig[D // 2],
                                        confidence_slice=(None if confidence_map is None
                                                          else confidence_map[D // 2]))),
        ("sagittal", _overlay_segmentation(scan_volume[:, :, W // 2],
                                           pred_orig[:, :, W // 2],
                                           confidence_slice=(None if confidence_map is None
                                                             else confidence_map[:, :, W // 2]))),
        ("coronal", _overlay_segmentation(scan_volume[:, H // 2, :],
                                          pred_orig[:, H // 2, :],
                                          confidence_slice=(None if confidence_map is None
                                                            else confidence_map[:, H // 2, :]))),
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
                  demo_prior: str = "Off",
                  use_bratsify_override: Optional[bool] = None):
    """Shared pipeline for NIfTI + DICOM. Returns all UI outputs."""
    import tempfile
    workdir = tempfile.mkdtemp(prefix="brats_out_")
    _purge_stale_workdirs()
    use_bratsify = (os.environ.get("BRATSIFY", "1").lower()
                    not in ("0", "false", "no"))
    if use_bratsify_override is not None:
        use_bratsify = use_bratsify_override
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
    confidence_map = None
    modality_support: dict[str, float] | None = None
    if _use_pretrained_brats_model():
        # The public model consumes the aligned full-resolution modalities and
        # returns a BraTS label map in that same space. It is used only while
        # the project's own ONNX U-Net is the explicit placeholder artifact.
        pred_orig, confidence_map, modality_support = \
            predict_brats_regions_with_explanations(scans)
        model_warning = (
            "Inference engine: public pretrained MONAI BraTS model "
            "(BraTS 2018; 3-region segmentation)."
        )
        model_name = "pretrained MONAI BraTS SegResNet"
    else:
        engine = _get_engine()
        pred_128 = engine.predict_class_map(tensor)
        pred_orig = _resample_prediction_to_original(pred_128, metadata)
        model_warning = _model_health_warning(pred_128)
        model_name = ONNX_PATH
    if cleanup_min_vox > 0:
        pred_orig = cleanup_prediction(pred_orig, min_voxels=cleanup_min_vox)

    # The model output is the single source of truth for every displayed and
    # exported result. Ground-truth sidecars remain evaluation-only inputs and
    # must never replace a live prediction.
    scan_volume = scans.volumes.get(bg_modality, scans.volumes["flair"])
    seg_analysis = pred_orig
    seg_source = f"model prediction ({model_name})"
    volumes = _compute_volumes(seg_analysis, metadata)
    flag = tumor_flag(volumes, threshold_cm3=NORMAL_WT_THRESHOLD_CM3)
    metrics_md = _build_metrics_markdown(volumes, flag, source=seg_source,
                                         model_warning=model_warning)
    if confidence_map is not None:
        tumor_confidence = confidence_map[seg_analysis > 0]
        mean_confidence = (float(tumor_confidence.mean())
                           if tumor_confidence.size else 0.0)
        lower_support_fraction = (float(np.mean(tumor_confidence < 0.70))
                                  if tumor_confidence.size else 0.0)
        metrics_md += (
            "\n\n---\n\n### Explainability review\n\n"
            f"**Mean model support within segmented tumour:** {mean_confidence:.0%}. "
            "This is region-probability support, not calibrated diagnostic certainty.\n\n"
            "Review low-support or anatomically implausible regions against all source sequences before accepting the segmentation."
        )
        if modality_support:
            ranking = sorted(modality_support.items(), key=lambda item: item[1], reverse=True)
            ranked_text = " · ".join(
                f"{name.upper()} {score:.0%}" for name, score in ranking
            )
            metrics_md += (
                "\n\n**Modality sensitivity (single-sequence occlusion):** "
                f"{ranked_text}. Higher values mean withholding that sequence changed the model output more."
            )

    plot_3d = _build_3d_plot(seg_analysis, scans.volumes["t2"],
                             None, show_classes=show_classes)
    plot_mpr = _build_mpr_plot(scan_volume, seg_analysis, axial_idx,
                               sagittal_idx, coronal_idx,
                               opacity=overlay_opacity,
                               show_classes=show_classes,
                               confidence_map=confidence_map)
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
    snaps = _snapshot_mpr_pngs(scan_volume, seg_analysis, workdir,
                               confidence_map=confidence_map)
    rano = rano_diameters(np.isin(seg_analysis, [1, 2, 3]), spacing)
    prior_seg = _load_prior_segmentation(prior_path, demo_prior, seg_analysis)
    cmp = _compare_to_prior(
        seg_analysis, prior_seg, metadata, spacing, volumes,
        synthetic=(prior_path is None
                   and demo_prior not in (None, "", "Off")))
    metrics_md += _comparison_markdown(cmp)
    report_text = build_ingestion_report(warnings) + f"\nModel: {model_name}"
    report_text += f"\nSegmentation source: {seg_source}"
    if model_warning:
        report_text += f"\n{model_warning}"
    if b_report:
        report_text += f"\n\n{b_report.to_markdown()}"
    report = write_html_report(os.path.join(workdir, "report.html"), volumes,
                               rano, snaps,
                               report_text,
                               scans.source,
                               comparison=_comparison_html(cmp),
                               explainability=(
                                   "Mean model support within segmented tumour: "
                                   f"{mean_confidence:.0%}. "
                                   f"{lower_support_fraction:.1%} of segmented tumour voxels "
                                   "were below the 70% review threshold. Amber in the MPR "
                                   "snapshots marks those lower-support regions. This is a "
                                   "model-review cue, not calibrated diagnostic certainty."
                               ) if confidence_map is not None else "")
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
) -> tuple[go.Figure, go.Figure, str, str, str, Optional[str], list[str] | None]:
    """Run the full analysis pipeline and return visual + text outputs.

    Returns:
        (plot_3d, plot_mpr, metrics_md, status_md, error_md, report_html,
        downloads)
    """
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
        return plot_3d, plot_mpr, metrics_md, warn_md, "", report, downloads

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
                return plot_3d, plot_mpr, metrics_md, warn_md, error_md, None, None
        error_md = f"### Ingestion note\n\n{exc}"
    except FileNotFoundError as exc:
        error_md = f"### Model / file note\n\n{exc}"
    except Exception as exc:  # noqa: BLE001 - surface all runtime errors in UI
        error_md = f"### Processing note\n\n{exc}"

    # Keep the last successful study visible. A validation error should guide
    # the next action, not erase a result that a user may still be reviewing.
    return (gr.skip(), gr.skip(), gr.skip(), "", error_md,
            gr.skip(), gr.skip())


def run_local_brats_demo(
    axial_idx: int = 78,
    sagittal_idx: int = 120,
    coronal_idx: int = 120,
    bg_modality: str = "flair",
    overlay_opacity: float = 0.8,
    show_classes: Optional[list[int]] = None,
    cleanup_min_vox: int = 50,
) -> tuple[go.Figure, go.Figure, str, str, Optional[str], list[str] | None]:
    """Run the bundled four-modal BraTS study without a manual upload.

    This is a convenience path for the presentation demo. It uses exactly the
    same ingestion and analysis pipeline as uploaded NIfTI studies.
    """
    source = {
        "t1": "BRATS/BraTS20_Training_002_t1.nii",
        "t1ce": "BRATS/BraTS20_Training_002_t1ce.nii",
        "t2": "BRATS/BraTS20_Training_002_t2.nii",
        "flair": "BRATS/BraTS20_Training_002_flair.nii",
    }
    missing = [path for path in source.values() if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            "Local BraTS demo files are unavailable: " + ", ".join(missing)
        )
    return _run_pipeline(
        source,
        axial_idx,
        sagittal_idx,
        coronal_idx,
        bg_modality.lower(),
        overlay_opacity,
        show_classes or [1, 2, 3],
        cleanup_min_vox,
        use_bratsify_override=False,
    )


def _message_update(value: str):
    """Hide empty Markdown outputs so blank alert containers never render."""
    return gr.update(value=value, visible=bool(value and value.strip()))


def _analysis_outputs_for_ui(result):
    """Map pipeline output into visible-on-demand status and error messages."""
    (plot_3d, plot_mpr, metrics_md, status_md, error_md,
     report, downloads) = result
    return (plot_3d, plot_mpr, metrics_md, _message_update(status_md),
            _message_update(error_md), report, downloads)


def run_analysis_ui(*args):
    """UI adapter that prevents empty status and error panels from rendering."""
    return _analysis_outputs_for_ui(run_analysis(*args))


def run_local_brats_demo_ui(*args):
    """Adapt the local demo for the UI's separate status and error strips."""
    try:
        plot_3d, plot_mpr, metrics_md, status_md, report, downloads = \
            run_local_brats_demo(*args)
        return _analysis_outputs_for_ui(
            (plot_3d, plot_mpr, metrics_md, status_md, "", report, downloads)
        )
    except Exception as exc:  # noqa: BLE001 - keep the UI actionable
        empty_fig = go.Figure()
        empty_fig.update_layout(
            paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
            font=dict(color="white"), title="Demo unavailable",
        )
        return _analysis_outputs_for_ui(
            (empty_fig, empty_fig, "", "",
             f"### Local demo note\n\n{exc}", None, None)
        )


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
    body_background_fill="#F7FAFC",
    body_background_fill_dark="#F7FAFC",
    body_text_color="#102A43",
    body_text_color_dark="#102A43",
    body_text_color_subdued="#52667A",
    background_fill_primary="#FFFFFF",
    background_fill_primary_dark="#FFFFFF",
    background_fill_secondary="#F1F5F9",
    background_fill_secondary_dark="#F1F5F9",
    border_color_primary="#D9E2EC",
    border_color_primary_dark="#D9E2EC",
    color_accent_soft="#E0F2FE",
    color_accent_soft_dark="#E0F2FE",
    button_primary_background_fill="#1D4ED8",
    button_primary_background_fill_hover="#1E40AF",
    button_primary_background_fill_dark="#1D4ED8",
    button_primary_background_fill_hover_dark="#1E40AF",
    button_primary_text_color="#FFFFFF",
    button_primary_text_color_dark="#FFFFFF",
    button_primary_border_color="transparent",
    button_secondary_background_fill="#FFFFFF",
    button_secondary_background_fill_hover="#F1F5F9",
    button_secondary_background_fill_dark="#FFFFFF",
    button_secondary_background_fill_hover_dark="#F1F5F9",
    button_secondary_border_color="#CBD5E1",
    button_secondary_border_color_dark="#CBD5E1",
    button_secondary_text_color="#102A43",
    button_secondary_text_color_dark="#102A43",
    block_radius="16px",
    block_title_text_size="15px",
    block_label_text_size="12px",
    block_label_text_weight="600",
    input_background_fill="#FFFFFF",
    input_background_fill_dark="#FFFFFF",
    input_border_color="#CBD5E1",
    input_border_color_dark="#CBD5E1",
    slider_color="#1D4ED8",
    slider_color_dark="#1D4ED8",
    checkbox_background_color_selected="#1D4ED8",
    checkbox_background_color_selected_dark="#1D4ED8",
)

_APPLE_CSS = """

/* ============================================================================
   NEUROVISTA — accessible clinical workspace, light-only palette
   ============================================================================ */
:root {
  --ntt-bg: #F7FAFC;
  --ntt-bg-soft: #F1F5F9;
  --ntt-surface: #FFFFFF;
  --ntt-surface-soft: #F8FAFC;
  --ntt-ink: #102A43;
  --ntt-ink-2: #243B53;
  --ntt-ink-3: #52667A;
  --ntt-ink-4: #7B8FA1;
  --ntt-border: #D9E2EC;
  --ntt-border-strong: #BCCCDC;
  --ntt-accent: #1D4ED8;
  --ntt-accent-ink: #1E40AF;
  --ntt-accent-2: #0F766E;
  --ntt-shadow-1: 0 1px 2px rgba(16,42,67,.05), 0 8px 24px rgba(16,42,67,.07);
  --ntt-shadow-2: 0 16px 40px rgba(16,42,67,.12);
  --ntt-radius: 18px;
  --ntt-radius-sm: 12px;
  --ntt-pill: 999px;
  --ntt-font: system-ui, -apple-system, BlinkMacSystemFont, "SF Pro Display",
    "SF Pro Text", "Inter", "Helvetica Neue", Arial, sans-serif;
  --ntt-mono: "SF Mono", "IBM Plex Mono", ui-monospace, "Consolas", monospace;
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


/* ============================================================================
   COMPONENT STYLES — workflow workspace
   ============================================================================ */
.gradio-container { max-width: 1180px !important; margin: 0 auto !important; }
footer { display: none !important; }
.gradio-container, #input label, #input .label-wrap, #input .label-wrap span {
  color: var(--ntt-ink) !important;
}
.gradio-container .label-wrap, .gradio-container .label-wrap *,
.gradio-container .block-label, .gradio-container .block-label * {
  color: var(--ntt-ink) !important;
  opacity: 1 !important;
}
.ntt-source-mode, .ntt-source-mode *, .ntt-source-mode label,
.ntt-source-mode span {
  color: var(--ntt-ink) !important;
  opacity: 1 !important;
}
#ntt-mode, #ntt-mode *, #ntt-mode label, #ntt-mode span {
  color: #102A43 !important;
  opacity: 1 !important;
}
/* Gradio's dark-system file uploader rules otherwise leak into a light page. */
.ntt-file, .ntt-file > *, .ntt-file .wrap, .ntt-file .file-preview,
.ntt-file .file-preview-holder, .ntt-file .file-preview > div {
  background: #FFFFFF !important; color: var(--ntt-ink) !important;
  border-color: var(--ntt-border-strong) !important;
}
.ntt-file label, .ntt-file span, .ntt-file button, .ntt-file div {
  color: var(--ntt-ink) !important;
}
.ntt-file svg { color: var(--ntt-accent) !important; stroke: currentColor !important; }

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
.ntt-workspace-head { margin: 30px 24px 26px; padding: 26px 28px; position: relative;
  overflow: hidden; border: 1px solid var(--ntt-border); border-radius: var(--ntt-radius);
  background: linear-gradient(125deg, var(--ntt-surface), var(--ntt-surface-soft));
  box-shadow: var(--ntt-shadow-1); }
.ntt-workspace-head::after { content: ""; position: absolute; width: 340px; height: 340px;
  right: -145px; top: -250px; border-radius: 50%; background: rgba(10,132,255,.12); }
.ntt-workspace-kicker { margin: 0 0 7px; color: var(--ntt-accent); text-transform: uppercase;
  letter-spacing: .11em; font-size: 11px; font-weight: 750; }
.ntt-workspace-title { margin: 0; font-size: clamp(25px, 3vw, 36px); line-height: 1.12;
  letter-spacing: -.025em; color: var(--ntt-ink); }
.ntt-workspace-copy { margin: 9px 0 21px; max-width: 640px; color: var(--ntt-ink-3); font-size: 14px; }
.ntt-flow { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 10px; position: relative; z-index: 1; }
.ntt-flow-step { min-width: 0; padding: 12px 14px; border-radius: 10px; background: color-mix(in srgb, var(--ntt-surface) 80%, transparent);
  border: 1px solid var(--ntt-border); }
.ntt-flow-num { display: block; margin-bottom: 4px; color: var(--ntt-accent); font-size: 11px; font-weight: 750; letter-spacing: .08em; }
.ntt-flow-label { display: block; color: var(--ntt-ink); font-size: 13px; font-weight: 700; }
.ntt-flow-detail { display: block; margin-top: 2px; color: var(--ntt-ink-3); font-size: 12px; line-height: 1.35; }
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
#ntt-demo { margin-top: 10px; border-radius: 12px !important; font-weight: 650 !important;
  border: 1px solid var(--ntt-border-strong) !important; background: var(--ntt-surface) !important;
  color: var(--ntt-ink) !important; }
#ntt-demo:hover { border-color: var(--ntt-accent) !important; color: var(--ntt-accent) !important; }
.ntt-ready { margin: 0 0 16px; padding: 12px 14px; border-radius: var(--ntt-radius-sm);
  border: 1px solid rgba(0,199,166,.32); background: rgba(0,199,166,.08); color: var(--ntt-ink-2);
  font-size: 12.5px; line-height: 1.45; }
.ntt-ready b { color: var(--ntt-ink); }
.ntt-ready-dot { display: inline-block; width: 8px; height: 8px; margin-right: 7px;
  border-radius: 50%; background: var(--ntt-accent-2); box-shadow: 0 0 0 4px rgba(0,199,166,.12); }
.ntt-step { margin: 0 0 14px; color: var(--ntt-ink-3); font-size: 13px; }
.ntt-empty { padding: 28px 8px 20px; color: var(--ntt-ink-3); font-size: 14px; }
.ntt-empty b { color: var(--ntt-ink-2); }

/* ---- accordions ---- */
.ntt-acc { border: 1px solid var(--ntt-border) !important; border-radius: var(--ntt-radius-sm) !important;
  background: var(--ntt-surface-soft) !important; }

/* ---- results panel ---- */
.ntt-results .gr-markdown, #ntt-metrics, #ntt-metrics * { color: var(--ntt-ink-2) !important; }
#ntt-metrics table { width: 100%; border-collapse: collapse; }
#ntt-metrics th, #ntt-metrics td { text-align: left; padding: 9px 12px;
  border-bottom: 1px solid var(--ntt-border); font-size: 13.5px; background: #FFFFFF; }
#ntt-metrics th { color: var(--ntt-ink) !important; font-weight: 700; background: #F1F5F9; }
.ntt-results .gr-plot, .ntt-results .gr-plot > div { border-radius: var(--ntt-radius-sm); overflow: hidden; }

/* ---- run status and error strips ---- */
.ntt-status { border: 1px solid #93C5FD; background: #EFF6FF; color: #163A63;
  padding: 12px 14px; border-radius: var(--ntt-radius-sm); }
.ntt-alert { border: 1px solid #FCA5A5; background: #FEF2F2; color: #7F1D1D;
  padding: 12px 14px; border-radius: var(--ntt-radius-sm); }
.ntt-status p, .ntt-alert p { margin: 0; }

/* ---- mode radio as segmented control ---- */
#ntt-mode { background: var(--ntt-surface-soft); border: 1px solid var(--ntt-border);
  border-radius: 12px; padding: 6px 10px; }
#ntt-mode label { font-weight: 550; }

/* ---- tabs ---- */
.ntt-results .gr-tabs button, .ntt-results .tab-nav button { font-weight: 600; }

@media (max-width: 760px) {
  .ntt-hero { padding: 36px 16px 28px; }
  .ntt-workspace-head { margin: 18px 16px 20px; padding: 20px; }
  .ntt-flow { grid-template-columns: 1fr; }
  .ntt-side, .ntt-results { padding: 16px; }
}
"""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def build_app() -> gr.Blocks:
    engine_label = (
        "Pretrained BraTS model ready for the local four-modal demo"
        if _use_pretrained_brats_model()
        else "Project ONNX 3D U-Net selected"
    )
    with gr.Blocks(title="NeuroVista — Brain Tumour Segmentation") as demo:
        # ---------------- workflow header ----------------
        gr.HTML(
            """
            <section class="ntt-workspace-head">
              <p class="ntt-workspace-kicker">NeuroVista &middot; MRI review workspace</p>
              <h1 class="ntt-workspace-title">Segment, inspect, and export in one workspace.</h1>
              <p class="ntt-workspace-copy">Use all four MRI modalities for model segmentation, then review region volumes, overlays, and a 3D surface before exporting the result.</p>
              <div class="ntt-flow" aria-label="Analysis workflow">
                <div class="ntt-flow-step"><span class="ntt-flow-num">01 · INPUT</span><span class="ntt-flow-label">Choose a study</span><span class="ntt-flow-detail">Four NIfTI scans or one DICOM ZIP.</span></div>
                <div class="ntt-flow-step"><span class="ntt-flow-num">02 · ANALYSE</span><span class="ntt-flow-label">Run segmentation</span><span class="ntt-flow-detail">Use the local demo when presenting.</span></div>
                <div class="ntt-flow-step"><span class="ntt-flow-num">03 · REVIEW</span><span class="ntt-flow-label">Inspect and export</span><span class="ntt-flow-detail">Volumetry, MPR, surface, report, meshes.</span></div>
              </div>
            </section>
            """
        )
        # ---------------- controls ----------------
        with gr.Row(elem_id="input"):
            with gr.Column(scale=1, elem_classes=["ntt-side"]):
                gr.Markdown('<p class="ntt-section">1 &middot; Source</p>')
                gr.HTML(
                    f'<div class="ntt-ready"><span class="ntt-ready-dot"></span>'
                    f'<b>Analysis ready</b><br>{engine_label}</div>'
                )
                mode = gr.Radio(
                    choices=["NIfTI Files", "DICOM ZIP"],
                    value="NIfTI Files",
                    show_label=False,
                    elem_id="ntt-mode",
                    elem_classes=["ntt-source-mode"],
                )
                with gr.Group(visible=True, elem_classes=["ntt-uploads"]) as nifti_group:
                    gr.Markdown('<p class="ntt-hint"><b>4 required scans</b> &mdash; T1, T1ce, T2, FLAIR.</p>')
                    t1_file = gr.File(label="T1", file_types=[".nii", ".nii.gz"], elem_classes=["ntt-file"])
                    t1ce_file = gr.File(label="T1ce", file_types=[".nii", ".nii.gz"], elem_classes=["ntt-file"])
                    t2_file = gr.File(label="T2", file_types=[".nii", ".nii.gz"], elem_classes=["ntt-file"])
                    flair_file = gr.File(label="FLAIR", file_types=[".nii", ".nii.gz"], elem_classes=["ntt-file"])
                    nifti_folder = gr.File(label="Folder or ZIP bundle (optional)", file_types=[".zip"], elem_classes=["ntt-file"])
                    prior_seg_file = gr.File(label="Prior segmentation (optional)", file_types=[".nii", ".nii.gz"], elem_classes=["ntt-file"])
                    demo_prior = gr.Radio(["Off", "Regression (tumour shrank)", "Progression (tumour grew)"],
                                          value="Off", label="Synthesize demo prior")
                with gr.Group(visible=False, elem_classes=["ntt-uploads"]) as zip_group:
                    gr.Markdown('<p class="ntt-hint"><b>One file</b> &mdash; zip of DICOM series; anonymize PHI on upload.</p>')
                    zip_file = gr.File(label="DICOM ZIP", file_types=[".zip"], elem_classes=["ntt-file"])
                    anonymize = gr.Checkbox(value=True, label="Anonymize on upload (strip PHI, keep geometry)")
                gr.Markdown('<p class="ntt-section">2 &middot; Configure</p>')
                gr.Markdown('<p class="ntt-step">Upload a complete study, then start analysis. Advanced settings are optional.</p>')
                run_btn = gr.Button("Run analysis", elem_classes=["ntt-primary"], elem_id="ntt-run")
                demo_btn = gr.Button("Run local BraTS demo", elem_id="ntt-demo")
                with gr.Accordion("Advanced view and cleanup", open=False, elem_classes=["ntt-acc"]):
                    axial_slider = gr.Slider(0, 240, value=78, step=1, label="Axial slice")
                    sagittal_slider = gr.Slider(0, 240, value=120, step=1, label="Sagittal slice")
                    coronal_slider = gr.Slider(0, 240, value=120, step=1, label="Coronal slice")
                    bg_modality = gr.Radio(["FLAIR", "T1CE"], value="FLAIR", label="Background")
                    overlay_opacity = gr.Slider(0.0, 1.0, value=0.8, step=0.05, label="Overlay opacity")
                    show_classes = gr.CheckboxGroup(
                        [(CLASS_NAMES[1], 1), (CLASS_NAMES[2], 2), (CLASS_NAMES[3], 3)],
                        value=[1, 2, 3], label="Visible classes")
                    cleanup_min_vox = gr.Slider(0, 500, value=50, step=10,
                                                 label="Remove blobs smaller than (voxels)")
                with gr.Accordion("Partial-scan aids", open=False, elem_classes=["ntt-acc"]):
                    show_heat = gr.Checkbox(value=True, label="Show asymmetry heat (non-specific)")
                    heat_percentile = gr.Slider(90.0, 99.9, value=98.0, step=0.1,
                                                label="Heat threshold percentile")
                    gr.Markdown(PROTOCOL_CHECKLIST_MD)
            with gr.Column(scale=3, elem_classes=["ntt-results"], elem_id="output"):
                gr.Markdown('<p class="ntt-section">3 &middot; Output</p>')
                status_md = gr.Markdown("", visible=False, elem_classes=["ntt-status"])
                error_md = gr.Markdown("", visible=False, elem_classes=["ntt-alert"])
                metrics_md = gr.Markdown(
                    '<div class="ntt-empty"><b>Results will appear here.</b><br>'
                    'Run the local demo or upload all four MRI modalities to begin.</div>',
                    elem_id="ntt-metrics",
                )
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
            run_analysis_ui,
            inputs=[
                mode,
                t1_file, t1ce_file, t2_file, flair_file, nifti_folder, zip_file,
                axial_slider, sagittal_slider, coronal_slider,
                bg_modality, overlay_opacity, show_classes, cleanup_min_vox,
                anonymize, show_heat, heat_percentile,
                prior_seg_file, demo_prior,
            ],
            outputs=[plot_3d, plot_mpr, metrics_md, status_md, error_md,
                     report_html, downloads],
        )
        demo_btn.click(
            run_local_brats_demo_ui,
            inputs=[
                axial_slider, sagittal_slider, coronal_slider,
                bg_modality, overlay_opacity, show_classes, cleanup_min_vox,
            ],
            outputs=[plot_3d, plot_mpr, metrics_md, status_md, error_md,
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
        theme=_APPLE_THEME,
        css=_APPLE_CSS,
        server_name=host,
        server_port=port,
    )
