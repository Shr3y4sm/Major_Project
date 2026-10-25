"""Atlas-normalized educational anatomy projection.

This is a learning visualization. Registration output is always marked as
review-required and is not a navigation transform or clinical measurement.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import SimpleITK as sitk
import nibabel as nib
import numpy as np
import plotly.graph_objects as go
from nibabel.processing import resample_from_to
from scipy.ndimage import distance_transform_edt
from skimage import measure

ATLAS_GROUPS = {
    "Motor and frontal cortex": ([2001, 2002, 2101, 2102, 2301, 2302, 2501, 2502], "#3b82f6"),
    "Language-associated cortex": ([2311, 2312, 2401, 2402, 2701, 2702, 6211, 6212, 6221, 6222], "#f59e0b"),
    "Parietal and sensory cortex": ([6001, 6002, 6101, 6102, 6201, 6202, 6301, 6302], "#10b981"),
    "Temporal cortex": ([8101, 8102, 8121, 8122, 8201, 8202, 8211, 8212, 8301, 8302], "#ef6b73"),
    "Deep nuclei": ([7001, 7002, 7011, 7012, 7021, 7022, 7101, 7102], "#8b5cf6"),
    "Cerebellum": ([9001, 9002, 9011, 9012, 9021, 9022, 9031, 9032, 9041, 9042], "#94a3b8"),
}


def _mesh(fig: go.Figure, mask: np.ndarray, name: str, color: str, opacity: float,
          step: int = 2) -> bool:
    if not mask.any():
        return False
    try:
        verts, faces, _, _ = measure.marching_cubes(mask.astype(np.float32), .5, step_size=step)
    except ValueError:
        return False
    fig.add_trace(go.Mesh3d(x=verts[:, 2], y=verts[:, 1], z=verts[:, 0],
                            i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                            name=name, color=color, opacity=opacity,
                            flatshading=False, hovertemplate=f"{name}<extra></extra>",
                            lighting=dict(ambient=.55, diffuse=.72, specular=.18, roughness=.6),
                            lightposition=dict(x=120, y=160, z=180)))
    return True


def _centroid(mask: np.ndarray) -> tuple[float, float, float]:
    points = np.argwhere(mask)
    centre = points.mean(axis=0)
    return float(centre[2]), float(centre[1]), float(centre[0])


def _anatomy_summary(projected_tumour: np.ndarray,
                     group_masks: list[tuple[str, np.ndarray]],
                     spacing: tuple[float, float, float]) -> str:
    """Describe lesion proximity in template space without clinical claims."""
    if not projected_tumour.any():
        return (
            "### Anatomy explorer\n\n"
            "> No tumour voxels were available after atlas projection. Anatomy layers remain a "
            "reference-only learning view."
        )
    distance = distance_transform_edt(~projected_tumour, sampling=spacing)
    rows: list[str] = []
    for name, mask in group_masks:
        if np.any(mask & projected_tumour):
            relation = "Overlaps mapped lesion"
        else:
            relation = f"{float(distance[mask].min()):.1f} mm in template space"
        rows.append(f"| {name} | {relation} |")
    return (
        "### Atlas anatomy context\n\n"
        "| Reference structure | Lesion relationship |\n| --- | --- |\n" + "\n".join(rows) +
        "\n\n> These are reference-atlas relationships after affine registration, not patient-specific "
        "clearance distances, navigation coordinates, or functional-risk estimates."
    )


def atlas_learning_figure(t1_path: str, tumour_path: str, template_path: str,
                          atlas_path: str) -> tuple[go.Figure, str, str]:
    """Register tumour to a template only for an educational atlas projection."""
    fixed = sitk.ReadImage(template_path, sitk.sitkFloat32)
    moving = sitk.ReadImage(t1_path, sitk.sitkFloat32)
    initial = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY)
    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(32)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(.01, seed=7)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(1.5, .01, 40)
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetInitialTransform(initial, inPlace=False)
    # Small synthetic/test volumes cannot safely sustain a multi-resolution
    # pyramid. Real clinical and MNI volumes use the more stable three levels.
    if min(fixed.GetSize()) >= 32:
        registration.SetShrinkFactorsPerLevel([4, 2, 1])
        registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    else:
        registration.SetShrinkFactorsPerLevel([1])
        registration.SetSmoothingSigmasPerLevel([0])
    transform = registration.Execute(fixed, moving)
    tumour = sitk.ReadImage(tumour_path, sitk.sitkUInt8)
    projected = sitk.Resample(tumour, fixed, transform, sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as handle:
        projected_path = handle.name
    try:
        sitk.WriteImage(projected, projected_path)
        projected_data = nib.load(projected_path).get_fdata() > 0
    finally:
        Path(projected_path).unlink(missing_ok=True)
    template = nib.load(template_path)
    atlas = nib.load(atlas_path)
    atlas_data = resample_from_to(atlas, (template.shape, template.affine), order=0).get_fdata()
    brain = template.get_fdata() > 20
    fig = go.Figure()
    _mesh(fig, brain, "Reference brain envelope", "#d6dbe4", .18, step=2)
    group_masks: list[tuple[str, np.ndarray]] = []
    for name, (codes, color) in ATLAS_GROUPS.items():
        mask = np.isin(atlas_data, codes)
        if _mesh(fig, mask, name, color, .58, step=1):
            # Start in a clean orientation view. The user can deliberately
            # reveal cortex systems or deep structures with the explorer.
            fig.data[-1].visible = False
            group_masks.append((name, mask))
    tumour_present = _mesh(fig, projected_data, "Tumour projection — educational", "#d946ef", .94, step=1)
    labels_present = False
    if group_masks:
        positions = [_centroid(mask) for _, mask in group_masks]
        fig.add_trace(go.Scatter3d(
            x=[p[0] for p in positions], y=[p[1] for p in positions], z=[p[2] for p in positions],
            mode="markers+text", text=[name for name, _ in group_masks], textposition="top center",
            marker=dict(size=3, color="#ffffff", opacity=.86), textfont=dict(color="#ffffff", size=11),
            name="Anatomy labels", hoverinfo="skip", showlegend=False, visible=False))
        labels_present = True

    # One readable anatomy layer at a time is considerably more useful than a
    # permanently overlaid mass of semi-transparent regions.
    group_count = len(group_masks)
    def visibility(include_groups: set[str], labels: bool) -> list[bool]:
        values = [True]
        values.extend(name in include_groups for name, _ in group_masks)
        if tumour_present:
            values.append(True)
        if labels_present:
            values.append(labels)
        return values
    def tumour_focus_visibility() -> list[bool]:
        values = [False] + [False] * group_count
        if tumour_present:
            values.append(True)
        if labels_present:
            values.append(False)
        return values
    all_groups = {name for name, _ in group_masks}
    cortex = {name for name in all_groups if name not in {"Deep nuclei", "Cerebellum"}}
    deep = {name for name in all_groups if name in {"Deep nuclei", "Cerebellum"}}
    fig.update_layout(
        title="Atlas-normalized anatomy explorer — educational, QA required",
        paper_bgcolor="#202023", plot_bgcolor="#202023", font=dict(color="#f5f5f7"),
        margin=dict(l=0, r=0, b=0, t=86), legend=dict(bgcolor="rgba(0,0,0,0)"),
        updatemenus=[dict(type="buttons", direction="right", x=.01, y=1.18, xanchor="left",
                          yanchor="top", bgcolor="#34343a", bordercolor="#5c5c63", font=dict(color="#f5f5f7"),
                          buttons=[
                              dict(label="Orientation", method="update", args=[{"visible": visibility(set(), False)}]),
                              dict(label="Cortical systems", method="update", args=[{"visible": visibility(cortex, True)}]),
                              dict(label="Deep structures", method="update", args=[{"visible": visibility(deep, True)}]),
                              dict(label="Full anatomy", method="update", args=[{"visible": visibility(all_groups, True)}]),
                              dict(label="Tumour focus", method="update", args=[{"visible": tumour_focus_visibility()}]),
                          ])],
        scene=dict(bgcolor="#202023", aspectmode="data", xaxis=dict(visible=False),
                   yaxis=dict(visible=False), zaxis=dict(visible=False),
                   camera=dict(eye=dict(x=1.7, y=-1.7, z=.72))))
    spacing = tuple(float(value) for value in nib.affines.voxel_sizes(template.affine)[:3])
    summary = _anatomy_summary(projected_data, group_masks, spacing)
    return (
        fig,
        "Educational atlas projection only — affine registration requires visual QA; not for navigation or clearance measurements.",
        summary,
    )
