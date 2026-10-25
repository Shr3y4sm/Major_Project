"""Research-only multimodal pre-operative planning primitives.

This module maps an existing tumour segmentation against *already derived*
critical-anatomy masks. It deliberately does not create an operative route,
infer vessels from routine MRI, or claim a navigation-ready registration.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from nibabel.processing import resample_from_to
from scipy.ndimage import distance_transform_edt
from skimage import measure


LAYER_COLORS = {"Vessels": "#ef4444", "White-matter tracts": "#38bdf8",
                "Functional cortex": "#fbbf24"}


@dataclass(frozen=True)
class LayerReview:
    name: str
    supplied: bool
    status: str
    voxel_count: int = 0
    volume_cm3: float = 0.0
    minimum_clearance_mm: float | None = None
    overlaps_tumour: bool = False


@dataclass(frozen=True)
class PreoperativePlan:
    tumour_volume_cm3: float
    layers: tuple[LayerReview, ...]
    review_required: bool = True

    def markdown(self) -> str:
        rows = []
        for layer in self.layers:
            clearance = (f"{layer.minimum_clearance_mm:.1f} mm"
                         if layer.minimum_clearance_mm is not None else "—")
            rows.append(
                f"| {layer.name} | {layer.status} | {clearance} | "
                f"{'Yes' if layer.overlaps_tumour else 'No'} |")
        return (
            "### Pre-operative mapping review\n\n"
            f"Draft tumour volume: **{self.tumour_volume_cm3:.2f} cm³**.\n\n"
            "| Critical layer | Mapping status | Minimum tumour clearance | Overlap |\n"
            "| --- | --- | ---: | --- |\n" + "\n".join(rows) +
            "\n\nClearance is a geometric review measurement, not a safety margin or a "
            "recommended surgical route. Every supplied layer requires registration QA and "
            "surgeon review against source images."
        )


def _mask_on_tumour_grid(path: str, tumour_img: nib.Nifti1Image) -> tuple[np.ndarray, str]:
    image = nib.load(path)
    if len(image.shape) != 3:
        raise ValueError(f"{Path(path).name}: expected a 3D mask, got {image.shape}")
    same_grid = image.shape == tumour_img.shape and np.allclose(image.affine, tumour_img.affine)
    if not same_grid:
        image = resample_from_to(image, (tumour_img.shape, tumour_img.affine), order=0)
        status = "resampled — registration QA required"
    else:
        status = "grid matched — registration QA required"
    return image.get_fdata() > 0, status


def build_preoperative_plan(tumour_segmentation: str,
                             layer_paths: dict[str, str | None],
                             report_path: str | None = None
                             ) -> tuple[PreoperativePlan, dict[str, np.ndarray], go.Figure]:
    """Map supplied critical-anatomy masks onto the tumour grid.

    Input masks must have been produced and clinically reviewed upstream. Any
    geometric resampling is recorded as still requiring QA.
    """
    tumour_img = nib.load(tumour_segmentation)
    tumour = tumour_img.get_fdata() > 0
    if tumour.ndim != 3:
        raise ValueError("Tumour segmentation must be 3D")
    spacing = tuple(float(v) for v in nib.affines.voxel_sizes(tumour_img.affine)[:3])
    tumour_volume = float(tumour.sum() * np.prod(spacing) / 1000)
    layers: dict[str, np.ndarray] = {}
    reviews: list[LayerReview] = []
    # Distance from every voxel to the nearest tumour voxel, in physical mm.
    tumour_distance = distance_transform_edt(~tumour, sampling=spacing)
    for name in ("Vessels", "White-matter tracts", "Functional cortex"):
        path = layer_paths.get(name)
        if not path:
            reviews.append(LayerReview(name, False, "not supplied"))
            continue
        mask, status = _mask_on_tumour_grid(path, tumour_img)
        layers[name] = mask
        overlap = bool(np.any(mask & tumour))
        clearance = float(tumour_distance[mask].min()) if mask.any() else None
        reviews.append(LayerReview(
            name, True, status, int(mask.sum()),
            float(mask.sum() * np.prod(spacing) / 1000), clearance, overlap))
    plan = PreoperativePlan(tumour_volume, tuple(reviews))
    if report_path:
        payload = asdict(plan)
        Path(report_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return plan, layers, _planning_figure(tumour, layers, spacing)


def _planning_figure(tumour: np.ndarray, layers: dict[str, np.ndarray], spacing: tuple[float, ...]) -> go.Figure:
    fig = go.Figure()
    for name, mask, color, opacity in [("Tumour", tumour, "#a855f7", .72),
                                       *((n, m, LAYER_COLORS[n], .42) for n, m in layers.items())]:
        if not mask.any():
            continue
        try:
            verts, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), .5, spacing=spacing)
            fig.add_trace(go.Mesh3d(x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
                                    i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                                    name=name, color=color, opacity=opacity))
        except ValueError:
            continue
    fig.update_layout(title="Critical-anatomy review — draft", paper_bgcolor="#202023",
                      plot_bgcolor="#202023", font=dict(color="#f5f5f7"),
                      margin=dict(l=0, r=0, b=0, t=44), legend=dict(bgcolor="rgba(0,0,0,0)"),
                      scene=dict(bgcolor="#202023", aspectmode="data",
                                 xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)))
    return fig
