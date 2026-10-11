"""Partial-data analysis: asymmetry heat + ellipsoid estimate (Tier 2).

For scans WITHOUT the full T1/T1ce/T2/FLAIR set (e.g. T1+T2 only), the
4-channel tumor UNet cannot run honestly — enhancing tumor is defined by
contrast (T1ce) and edema by FLAIR. This module squeezes SAFE signal
instead of fabricating a segmentation:

* ``asymmetry_heatmap`` — mirror left/right hemispheres, report the
  normalized absolute difference as a NON-SPECIFIC hyperintensity map.
  Never called a tumor segmentation.
* ``ellipsoid_estimate`` — geometric (pi/6 * D1*D2*D3) volume from the
  bounding box of a thresholded heat region. Reported as an ESTIMATE.
* ``blend_heat_overlay`` — orange heat blend for the MPR viewer.

All functions are numpy/scipy only.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter

__all__ = [
    "asymmetry_heatmap",
    "ellipsoid_estimate",
    "blend_heat_overlay",
    "PROTOCOL_CHECKLIST_MD",
]

PROTOCOL_CHECKLIST_MD = """### Request from PACS (tumour protocol)
- AX T1 (pre-contrast, 3D or 5mm axial)
- AX T1C / T1-GAD post-contrast (same geometry as T1)
- AX T2 / T2W (axial)
- AX FLAIR (axial, same orientation if possible)
- Export as DICOM (not screenshots), one series per modality
- Avoid localizer/scout-only exports (`Survey_MST` is skipped automatically)
"""


def _robust_normalize(vol: np.ndarray) -> np.ndarray:
    v = vol.astype(np.float64)
    lo, hi = np.percentile(v, [2, 99.5])
    if hi - lo < 1e-9:
        return np.zeros_like(v)
    return np.clip((v - lo) / (hi - lo), 0.0, 1.0)


def asymmetry_heatmap(volume: np.ndarray,
                      smooth_mm: float = 1.0) -> np.ndarray:
    """Mirror-difference heatmap in [0, 1].

    Assumes the LAST axis (W) is left-right. Returns smoothed |V - flip(V)|.
    High values = asymmetric signal (non-specific: tumor, edema, bleed,
    artifact, or normal asymmetry — NOT a tumor mask).
    """
    norm = _robust_normalize(volume)
    flipped = norm[:, :, ::-1]
    heat = np.abs(norm - flipped).astype(np.float64)
    if smooth_mm > 0:
        heat = gaussian_filter(heat, sigma=smooth_mm)
    mx = float(heat.max())
    if mx > 1e-12:
        heat = heat / mx
    return heat.astype(np.float32)


def ellipsoid_estimate(heat: np.ndarray,
                       spacing_mm: tuple[float, float, float],
                       percentile: float = 98.0,
                       min_voxels: int = 100,
                       max_bbox_frac: float = 0.9) -> dict:
    """Bounding-box ellipsoid estimate from a heat threshold.

    Returns dict with diameters mm, volume cm3, bbox, slice, n_voxels and
    a flag when nothing exceeds threshold (normal scan).
    Guards: if the hot-voxel bbox spans >= ``max_bbox_frac`` of ANY axis
    (global asymmetry — normal anatomy, acquisition gradient, motion),
    returns ``found=False`` with an explanatory note instead of an absurd
    whole-head "volume".
    THIS IS A GEOMETRIC ESTIMATE, not an AI segmentation.
    """
    thr = float(np.percentile(heat, percentile))
    mask = heat > thr  # strict: a flat zero-heat scan yields empty mask
    n = int(mask.sum())
    if n < min_voxels:
        return {"found": False, "threshold": round(thr, 3), "n_voxels": n,
                "note": "No region above threshold — normal or "
                        "below-detection scan."}
    zz, yy, xx = np.argwhere(mask).T
    D, H, W = heat.shape
    fr = [float(zz.max() - zz.min() + 1) / D,
          float(yy.max() - yy.min() + 1) / H,
          float(xx.max() - xx.min() + 1) / W]
    if max(fr) >= max_bbox_frac:
        return {"found": False, "threshold": round(thr, 3), "n_voxels": n,
                "note": "Heat is diffuse/global (bbox "
                        f"{[round(x,2) for x in fr]} of volume) — normal "
                        "anatomical asymmetry or acquisition gradient, "
                        "not a focal lesion. No estimate reported."}
    zz, yy, xx = np.argwhere(mask).T
    dz = (zz.max() - zz.min() + 1) * float(spacing_mm[0])
    dy = (yy.max() - yy.min() + 1) * float(spacing_mm[1])
    dx = (xx.max() - xx.min() + 1) * float(spacing_mm[2])
    vol_cm3 = float(np.pi / 6.0 * dz * dy * dx / 1000.0)
    return {"found": True, "threshold": round(thr, 3), "n_voxels": n,
            "d1_mm": round(dz, 1), "d2_mm": round(dy, 1),
            "d3_mm": round(dx, 1),
            "volume_cm3": round(vol_cm3, 2),
            "bbox": ((int(zz.min()), int(zz.max())),
                     (int(yy.min()), int(yy.max())),
                     (int(xx.min()), int(xx.max()))),
            "slice": int((zz.min() + zz.max()) // 2),
            "note": "GEOMETRIC ESTIMATE from T2 asymmetry — not an AI "
                    "tumor segmentation. Confirm on full 4-modality scan."}


def blend_heat_overlay(base_slice: np.ndarray, heat_slice: np.ndarray,
                       opacity: float = 0.55,
                       percentile: float = 98.0) -> np.ndarray:
    """Grayscale base + orange heat where heat >= percentile. Returns RGB."""
    s = base_slice.astype(np.float64)
    lo, hi = np.percentile(s, [2, 99.5])
    gray = np.zeros_like(s) if hi - lo < 1e-9 else (s - lo) / (hi - lo)
    rgb = np.stack([gray, gray, gray], axis=-1)
    thr = float(np.percentile(heat_slice, percentile))
    mask = heat_slice > thr
    if mask.any():
        a = float(np.clip(opacity, 0.0, 1.0))
        # orange (1.0, 0.55, 0.0)
        rgb[mask, 0] = rgb[mask, 0] * (1 - a) + 1.0 * a
        rgb[mask, 1] = rgb[mask, 1] * (1 - a) + 0.55 * a
        rgb[mask, 2] = rgb[mask, 2] * (1 - a) + 0.0 * a
    return np.clip(rgb * 255, 0, 255).astype(np.uint8)
