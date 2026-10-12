"""Post-processing: cleanup + normal-scan flag + RANO diameters."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import label as cc_label

__all__ = ["cleanup_prediction", "rano_diameters", "tumor_flag",
           "rano_classification"]


def cleanup_prediction(pred: np.ndarray, min_voxels: int = 50) -> np.ndarray:
    """Remove tiny connected components per tumor class (keeps WT stable)."""
    out = pred.copy()
    for cls in (1, 2, 3):
        mask = (pred == cls)
        if not mask.any():
            continue
        lab, n = cc_label(mask)
        sizes = np.bincount(lab.ravel())
        for comp in range(1, n + 1):
            if sizes[comp] < min_voxels:
                out[lab == comp] = 0
    return out


def rano_diameters(wt_mask: np.ndarray,
                   spacing_mm: tuple[float, float, float]) -> dict:
    """Max in-plane diameters (mm) on axial slices: (Dmax, Dperp)."""
    best = (0.0, 0.0, -1)
    D, H, W = wt_mask.shape
    for z in range(D):
        sl = wt_mask[z]
        if not sl.any():
            continue
        ys, xs = np.argwhere(sl).T
        if len(xs) < 2:
            continue
        # Brute force on bbox-downsampled points for speed.
        pts = np.stack([xs, ys], axis=1)
        if len(pts) > 400:
            idx = np.random.RandomState(0).choice(len(pts), 400, replace=False)
            pts = pts[idx]
        dmax, pair = 0.0, (pts[0], pts[0])
        for i in range(0, len(pts), 4):
            diff = pts.astype(float) - pts[i].astype(float)
            dists = np.sqrt((diff ** 2).sum(axis=1))
            j = int(dists.argmax())
            if dists[j] > dmax:
                dmax, pair = float(dists[j]), (pts[i], pts[j])
        # Perpendicular extent approx: width orthogonal to main axis.
        v = (pair[1] - pair[0]).astype(float)
        vn = float(np.hypot(v[0], v[1])) + 1e-9
        perp = np.abs((pts.astype(float) - pair[0].astype(float))
                      @ np.array([-v[1], v[0]]) / vn)
        dperp = float(perp.max())
        sx = float(np.mean(spacing_mm[1:]))
        cand = (dmax * sx, dperp * sx)
        if cand[0] * cand[1] > best[0] * best[1]:
            best = (cand[0], cand[1], z)
    return {"dmax_mm": round(float(best[0]), 1),
            "dperp_mm": round(float(best[1]), 1),
            "slice": int(best[2])}


def tumor_flag(volumes: dict[str, float], threshold_cm3: float = 0.3) -> str:
    """Normal-scan banner text based on whole-tumor volume."""
    wt = float(volumes.get("Total Whole Tumor", 0.0))
    if wt < 0.05:
        return ("No measurable tumor segmented (WT < 0.05 cm3). "
                "Likely normal scan or below model resolution — "
                "radiologist review required.")
    if wt < threshold_cm3:
        return (f"Tiny finding (WT {wt:.2f} cm3 < {threshold_cm3} cm3): "
                "possible noise / small focus — confirm on MPR.")
    return ""


def rano_classification(cur_spd: float,
                        prior_spd: float) -> tuple[str, float | None]:
    """RANO-style response call from change in whole-tumor SPD.

    SPD (sum of perpendicular diameters) is approximated as
    ``dmax_mm * dperp_mm`` from ``rano_diameters``. Thresholds (research
    use — requires radiologist review):
      * Progressive      >= +25% increase vs prior
      * Partial response <= -50% decrease vs prior
      * Stable           otherwise
    Returns ``(call, pct_change)``; ``pct_change`` is None when the prior
    has no measurable lesion (prior_spd <= 0) but the current does.
    """
    if prior_spd <= 0:
        if cur_spd > 0:
            return "New / no prior measurable lesion", None
        return "Stable", 0.0
    pct = (cur_spd - prior_spd) / prior_spd * 100.0
    if pct >= 25.0:
        return "Progressive", pct
    if pct <= -50.0:
        return "Partial response", pct
    return "Stable", pct
