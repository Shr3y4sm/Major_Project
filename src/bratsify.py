"""
src/bratsify.py
===============
PHASE A — "BraTS-ifier".

Converts *raw hospital* multi-modal MRI (DICOM/NIfTI, any spacing and
orientation, with skull) into **BraTS-grade** volumes, so the Phase 1 UNet
receives inputs that look like the data it was trained on.

The measured problem
--------------------
The Phase 1 model was trained on BraTS 2021: 240x240x155, 1 mm isotropic,
skull-stripped, rigidly co-registered, N4 bias-corrected. Hospital exports
violate every one of those assumptions:

* ``harmonize_shapes`` resampled by **shape ratio**, ignoring the affine, so
  a sub-FOV series combined with a 6 mm header error produced a T1/FLAIR
  NCC of **-0.04** (i.e. anti-correlated). Making the resample affine-aware
  raised the same pair to **NCC 0.63**.
* Plain intensity NCC is a poor alignment metric for T1<->FLAIR: it landed
  **41 voxels** off, whereas **gradient-magnitude NCC** landed **1 voxel** off.
* Phase correlation is unusable here (near-symmetric anatomy): it reported a
  30-voxel shift where the truth was 3.
* A naive multiplicative "N4-lite" *harmed* the data (in-mask intensity CV
  0.003 -> 0.269), so real N4 (SimpleITK) is used instead.

Pipeline (each stage independently guarded)
-------------------------------------------
1. ``head_mask``          percentile -> smooth -> largest CC -> fill holes.
2. ``estimate_com_shift`` centre-of-mass offset, used as the **search seed**
   and as a sanity guard (alignment is skipped past 15% of the reference FOV).
3. affine-aware resample **straight to the final BraTS grid** (a single
   interpolation, using each modality's own affine; skipped entirely when a
   modality already sits on that grid).
4. ``estimate_shift``     residual translation refinement (grad-NCC pyramid),
   applied only when it measurably improves the NCC.
5. ``n4_bias_correct``    real SimpleITK N4, guarded (see below).
6. brain-only z-score, provenance JSON, checkerboard QA PNG.

Guarantees
----------
* **Never breaks the demo**: any failing stage is caught, warned about, and
  the original scans are returned unchanged.
* **Deterministic**: no randomness anywhere.
* **Auditable**: every affine/shift/field statistic is recorded in
  ``BratsifyReport.provenance`` and can be written with
  :func:`write_provenance`.

N4 guard calibration (measured on ``BRATS/BraTS20_Training_002_t1.nii``)
-----------------------------------------------------------------------
A *healthy* BraTS T1 yields an in-mask bias field with p1/p99 = [0.666,
0.954] and a peak deviation of 0.35. A guard of "revert if the field leaves
+/-25%" would therefore reject perfectly good data, so the guard is instead
expressed on the **bulk** of the field (p1/p99 within [0.40, 2.50]) plus a
runaway check on the mean intensity scale (0.25x - 4x).

Runtime (measured, 240x240x155 float32, this machine)
-----------------------------------------------------
``n4_shrink=4`` -> **2.8 s**; ``n4_shrink=2`` -> **18.7 s**. Hence the
default shrink factor of 4.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_fill_holes,
    gaussian_filter,
    label as nd_label,
    shift as nd_shift,
    zoom,
)

from src.dicom_loader import (
    MODALITY_ORDER,
    PatientScans,
    SpatialMetadata,
)

__all__ = [
    "BratsifyError",
    "BratsifyConfig",
    "BratsifyReport",
    "head_mask",
    "gradient_magnitude",
    "ncc",
    "estimate_com_shift",
    "estimate_shift",
    "apply_shift",
    "resample_to_grid",
    "resample_to_spacing",
    "target_grid_from_reference",
    "to_brats_canvas",
    "n4_bias_correct",
    "simpleitk_available",
    "bratsify_scans",
    "save_qa_png",
    "write_provenance",
    "export_brats_style",
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class BratsifyError(Exception):
    """Raised for unrecoverable BraTS-ification failures."""


#: Target number of voxels per NCC evaluation inside the shift search. The
#: search subsamples (with an identical stride for both images) until each
#: candidate costs about this much; measured to be the difference between a
#: 7.6 s and a ~1.5 s refinement per modality pair.
_EVAL_BUDGET = 200_000

#: How many coarse-level candidates are re-scored with the final, full
#: resolution objective. Cheap (a few strided evaluations) and it removes the
#: failure mode where the heavily downsampled coarse metric prefers a wrong
#: basin that the +-2 refinements can never escape.
_COARSE_TOPK = 8


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BratsifyConfig:
    """Tunable knobs for the BraTS-ifier.

    Attributes:
        enabled: Master switch; ``False`` makes :func:`bratsify_scans` a no-op.
        n4: Run real N4 bias-field correction (needs SimpleITK).
        n4_shrink: Downsample factor passed to ``sitk.Shrink``. Measured:
            4 -> 2.8 s, 2 -> 18.7 s on a 240x240x155 volume.
        n4_iterations: Per-fitting-level iteration counts; the **length**
            selects the number of B-spline fitting levels.
        n4_control_points: B-spline mesh size (isotropic).
        n4_convergence: Convergence threshold.
        n4_field_lo / n4_field_hi: Revert N4 when the in-mask p1/p99 of the
            recovered field leaves this range (see module docstring).
        n4_min_mean_scale / n4_max_mean_scale: Revert N4 when correcting
            changes the in-mask mean intensity by more than this factor.
        n4_min_spread: If the field's in-mask p1..p99 spread is below this,
            the correction is a no-op and is skipped (no intensity surgery).
        n4_min_mask_frac: Minimum fraction of the volume that must be masked
            for N4 to be attempted.
        target_spacing: Desired output voxel spacing in mm.
        brats_canvas: Preferred output shape (BraTS 2021 is 240x240x155).
            Only ever grown, never shrunk below the brain extent, so no
            anatomy is ever cut away.
        pad_value: Fill value for out-of-FOV regions.
        register: Enable the COM pre-align + grad-NCC residual refinement.
        max_shift_frac: Skip alignment for a modality whose COM offset
            exceeds this fraction of the reference field of view.
        min_overlap: Minimum fraction of the moving head mask that must
            overlap the reference mask for a shift to be accepted.
        min_ncc_gain: Minimum gradient-NCC improvement required before a
            residual shift is applied. Without this, the search can find a
            spurious 0.01 gain and nudge already well-aligned data; with it,
            the BraTS-style co-registered case is left untouched.
        search_window_px: Half-width of the coarse search window (voxels).
        zscore_brain: Apply brain-only z-score normalisation at the end.
            Left ``False`` by default because ``prepare_for_inference``
            already z-scores (doing it twice would be a no-op on float
            data, but it would destroy absolute intensities for exports).
        head_pct: Intensity percentile used to build the head mask.
    """

    enabled: bool = True
    n4: bool = True
    n4_shrink: int = 4
    n4_iterations: tuple[int, ...] = (50, 50, 50, 50)
    n4_control_points: int = 4
    n4_convergence: float = 0.001
    n4_field_lo: float = 0.40
    n4_field_hi: float = 2.50
    n4_min_mean_scale: float = 0.25
    n4_max_mean_scale: float = 4.00
    n4_min_spread: float = 0.05
    n4_min_mask_frac: float = 0.02
    target_spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    brats_canvas: Optional[tuple[int, int, int]] = (240, 240, 155)
    pad_value: float = 0.0
    register: bool = True
    max_shift_frac: float = 0.15
    min_overlap: float = 0.50
    min_ncc_gain: float = 0.02
    search_window_px: int = 24
    zscore_brain: bool = False
    head_pct: float = 20.0
    qa_path: Optional[str] = None
    qa_modality: str = "t2"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class BratsifyReport:
    """Audit trail for a BraTS-ification run.

    Attributes:
        applied: ``True`` when the volumes were actually rewritten.
        reference: Modality used to define the output grid.
        steps: Human-readable list of completed stages.
        warnings: Non-fatal problems (also surfaced in the UI).
        n4: Per-modality N4 statistics.
        registration: Per-modality alignment statistics.
        final_shape / final_spacing / final_affine: Output geometry.
        seconds: Wall-clock duration of the whole run.
        provenance: Full machine-readable JSON payload.
    """

    applied: bool = False
    reference: str = "t1ce"
    steps: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    n4: dict[str, dict[str, Any]] = field(default_factory=dict)
    registration: dict[str, dict[str, Any]] = field(default_factory=dict)
    final_shape: Optional[tuple[int, int, int]] = None
    final_spacing: Optional[tuple[float, float, float]] = None
    final_affine: Optional[np.ndarray] = None
    seconds: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_markdown(self) -> str:
        """Render a short Markdown summary for the Gradio UI."""
        if not self.applied:
            head = "**Phase A (BraTS-ifier):** not applied."
            body = self.warnings or ["disabled or no-op"]
            return head + "\n\n" + "\n".join(f"- {w}" for w in body)
        lines = [
            "**Phase A (BraTS-ifier): applied**",
            f"- Reference grid: `{self.reference}` -> "
            f"{tuple(self.final_shape or ())} @ "
            f"{tuple(round(float(s), 3) for s in (self.final_spacing or ()))} mm",
            f"- Runtime: {self.seconds:.1f}s",
        ]
        for mod in MODALITY_ORDER:
            reg = self.registration.get(mod, {})
            if not reg:
                continue
            if reg.get("skipped"):
                lines.append(f"- `{mod}`: alignment skipped ({reg.get('reason')})")
            elif mod == self.reference:
                lines.append(f"- `{mod}`: reference")
            else:
                lines.append(
                    f"- `{mod}`: shift {reg.get('shift_mm', 0.0):.1f} mm, "
                    f"grad-NCC {reg.get('ncc_after', 0.0):.3f}, "
                    f"overlap {100 * reg.get('overlap', 0.0):.0f}%"
                )
        n4_msgs = []
        for mod, info in self.n4.items():
            if not info.get("available"):
                n4_msgs.append(f"`{mod}`: SimpleITK missing")
            elif info.get("reverted"):
                n4_msgs.append(f"`{mod}`: reverted ({info.get('reason')})")
            elif info.get("applied"):
                n4_msgs.append(
                    f"`{mod}`: field p1/p99 "
                    f"[{info['field_p1']:.2f}, {info['field_p99']:.2f}]"
                )
            elif info.get("skipped"):
                n4_msgs.append(f"`{mod}`: skipped ({info.get('reason')})")
        if n4_msgs:
            lines.append("- N4 bias field: " + "; ".join(n4_msgs))
        if self.warnings:
            lines.append("- Warnings:")
            lines += [f"    - {w}" for w in self.warnings]
        return "\n".join(lines)

    def to_json(self) -> str:
        """Serialise the provenance payload to pretty JSON."""

        def _default(o: Any):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, np.integer):
                return int(o)
            if isinstance(o, np.floating):
                return float(o)
            if isinstance(o, tuple):
                return list(o)
            return str(o)

        return json.dumps(self.provenance, indent=2, default=_default)


# ---------------------------------------------------------------------------
# Masks and image metrics
# ---------------------------------------------------------------------------


def head_mask(volume: np.ndarray, pct: float = 20.0, sigma: float = 1.0,
              close_iters: int = 2, downsample: int = 2) -> np.ndarray:
    """Robust head/brain mask for a raw MRI volume.

    Percentile threshold -> Gaussian smoothing -> largest connected
    component -> morphological closing -> hole filling. This mirrors the
    Phase 1 ``brain_mask_from_t2`` spirit but is self-contained per
    modality (hospital series often lack a matching T2) and robust to the
    arbitrary intensity scaling of DICOM exports.

    The morphology runs on a ``downsample``-reduced copy and the mask is
    nearest-neighbour upsampled afterwards: at full resolution the identical
    result cost **1.53 s** per call on a 240x240x155 volume, and this
    routine is called a dozen times per study. A blocky boundary is
    irrelevant for its four uses (centre of mass, bounding box, NCC support,
    N4 fitting mask), and downsampling is skipped for small inputs.

    Args:
        volume: Raw intensity volume.
        pct: Percentile of the non-zero intensities used as the threshold.
        sigma: Gaussian smoothing sigma before thresholding.
        close_iters: Binary-closing iterations (bridges thin gaps).
        downsample: Morphology reduction factor (1 disables the shortcut).

    Returns:
        A boolean array with the same shape as ``volume``.
    """
    vol = np.asarray(volume, dtype=np.float32)
    if not np.isfinite(vol).any():
        return np.zeros(vol.shape, dtype=bool)
    ds = int(max(1, downsample))
    if ds > 1 and min(vol.shape) >= 64:
        work = zoom(vol, 1.0 / ds, order=1)
        sigma_w = max(0.5, float(sigma) / ds)
        iters_w = max(1, int(round(close_iters / ds)))
    else:
        ds = 1
        work = vol
        sigma_w = float(sigma)
        iters_w = int(close_iters)

    pos = work[np.isfinite(work) & (work > 0)]
    if pos.size < 32:
        out = (np.isfinite(vol) & (vol != 0)) if np.any(vol != 0) \
            else np.ones(vol.shape, dtype=bool)
        return out
    thr = float(np.percentile(pos, pct))
    raw = (work > thr) & np.isfinite(work)
    smooth = gaussian_filter(raw.astype(np.float32), sigma=sigma_w) > 0.5
    if smooth.any():
        lab, n_lab = nd_label(smooth)
        if n_lab > 1:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            smooth = lab == int(np.argmax(sizes))
    else:
        smooth = raw
    smooth = binary_closing(smooth, iterations=max(0, iters_w))
    smooth = binary_fill_holes(smooth)
    if not smooth.any():  # pathological input -> fall back to the raw cut
        smooth = binary_fill_holes(raw)
    if ds == 1:
        return smooth.astype(bool)
    up = zoom(smooth.astype(np.float32), ds, order=0) > 0.5
    out = np.zeros(vol.shape, dtype=bool)
    sl = tuple(slice(0, min(int(a), int(b)))
               for a, b in zip(vol.shape, up.shape))
    out[sl] = up[sl]
    return out


def gradient_magnitude(volume: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Edge-strength map used for cross-modality registration.

    Measured: plain intensity NCC mis-registered a T1<->FLAIR pair by 41
    voxels, while gradient-magnitude NCC was accurate to 1 voxel.

    NOTE: ``np.gradient`` in NumPy 2.x rejects ``ax=`` (TypeError); the
    keyword must be ``axis=``.
    """
    vol = np.asarray(volume, dtype=np.float32)
    grads = [np.gradient(vol, axis=k) for k in range(3)]
    mag = np.sqrt(sum(g * g for g in grads))
    return gaussian_filter(mag, sigma=sigma)


def ncc(a: np.ndarray, b: np.ndarray, mask: Optional[np.ndarray] = None,
        stride: int = 1) -> float:
    """Normalised cross-correlation over ``mask`` (``stride`` can subsample).

    Returns ``-1.0`` when the comparison is degenerate (too few voxels or
    a constant signal), which the search treats as "no evidence".
    """
    if stride > 1:
        sl = (slice(None, None, stride),) * a.ndim
        a, b = a[sl], b[sl]
        if mask is not None:
            mask = mask[sl]
    if mask is None:
        x, y = a.ravel(), b.ravel()
    else:
        sel = np.asarray(mask, dtype=bool)
        if sel.shape != a.shape:
            return -1.0
        x, y = a[sel], b[sel]
    if x.size < 32:
        return -1.0
    x = x.astype(np.float64) - float(x.mean())
    y = y.astype(np.float64) - float(y.mean())
    denom = float(np.sqrt((x * x).sum() * (y * y).sum()))
    if denom < 1e-12:
        return -1.0
    return float((x * y).sum() / denom)


def _bbox_center(mask: np.ndarray) -> np.ndarray:
    """Centre of the mask bounding box, in voxel coordinates."""
    idx = np.argwhere(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return np.asarray(mask.shape, dtype=np.float64) / 2.0
    return (idx.min(axis=0) + idx.max(axis=0) + 1).astype(np.float64) / 2.0


def _com(mask: np.ndarray) -> np.ndarray:
    """Centre of mass of a mask in voxel coordinates."""
    idx = np.argwhere(np.asarray(mask, dtype=bool))
    if idx.size == 0:
        return np.asarray(mask.shape, dtype=np.float64) / 2.0
    return idx.mean(axis=0)


# ---------------------------------------------------------------------------
# Rigid (translation) alignment
# ---------------------------------------------------------------------------


def estimate_com_shift(moving_mask: np.ndarray,
                       fixed_mask: np.ndarray) -> np.ndarray:
    """Voxel shift that brings ``moving_mask``'s centre of mass onto the fixed.

    The returned vector is ``com(moving) - com(fixed)``, i.e. it can be
    passed straight to :func:`apply_shift` to pre-align the moving volume.
    """
    return _com(moving_mask) - _com(fixed_mask)


def _overlap_views(fixed: np.ndarray, moving: np.ndarray, shift: np.ndarray,
                   fixed_mask: np.ndarray, moving_mask: np.ndarray,
                   stride: int = 1):
    """Slice ``fixed``/``moving`` down to the region that still overlaps.

    ``shift`` is (dz, dy, dx) applied to ``moving`` in integer voxels. The
    overlap is obtained by plain slicing, which is far cheaper than calling
    ``scipy.ndimage.shift`` for every candidate offset.

    Returns ``(fv, mv, fmv, mmv)`` — the overlapping sub-arrays plus their
    masks — or ``None`` when the intersection is too small to be useful.
    """
    shape = np.asarray(fixed.shape)
    off = np.asarray(shift, dtype=int)
    f_start = np.maximum(0, off)
    f_end = np.minimum(shape, shape + off)
    m_start = np.maximum(0, -off)
    m_end = np.minimum(shape, shape - off)
    if np.any(f_end - f_start < 4) or np.any(m_end - m_start < 4):
        return None
    step = max(1, int(stride))
    fs = tuple(slice(int(a), int(b), step) for a, b in zip(f_start, f_end))
    ms = tuple(slice(int(a), int(b), step) for a, b in zip(m_start, m_end))
    return (fixed[fs], moving[ms],
            np.asarray(fixed_mask, bool)[fs], np.asarray(moving_mask, bool)[ms])


def estimate_shift(moving: np.ndarray, fixed: np.ndarray,
                   moving_mask: Optional[np.ndarray] = None,
                   fixed_mask: Optional[np.ndarray] = None,
                   window: int = 24, min_overlap: float = 0.50,
                   levels: tuple[int, ...] = (8, 4, 2, 1),
                   stride_at_full: int = 2,
                   fixed_grad: Optional[np.ndarray] = None,
                   initial: Optional[Sequence[int]] = None) -> dict[str, Any]:
    """Estimate the (dz, dy, dx) translation aligning ``moving`` onto ``fixed``.

    Multi-resolution search over **gradient-magnitude** images: a wide sweep
    at the coarsest level (``window`` full-res voxels, matching the measured
    +-24 px prototype) followed by a local +-2 refinement at each finer
    level. Candidates are scored by masked NCC over the *intersection* of the
    two head masks, so padding and out-of-FOV regions cannot create a
    spurious maximum. Plain phase correlation is deliberately **not** used:
    on near-symmetric anatomy it reported a 30-voxel shift where the truth
    was 3.

    Args:
        moving / fixed: Volumes already resampled onto the same grid.
        moving_mask / fixed_mask: Head masks (derived internally if omitted).
        window: Half-width of the coarse search window, in full-res voxels.
        min_overlap: Minimum overlap fraction of the moving mask required for
            a candidate to be scored at all.
        levels: Downsample factors from coarse to fine.
        stride_at_full: Sub-sampling stride used at the finest level (perf).
        fixed_grad: Optional precomputed edge map of ``fixed``; the reference
            edge map is identical for every modality in a study, so reusing
            it saves a full-resolution gradient pass (~0.8 s) per pair.
        initial: Optional starting translation (full-res voxels), used as the
            search seed. The centre-of-mass offset between the two head masks
            is normally passed here so the coarse sweep starts in the right
            region instead of having to search the full +-``window``.

    Returns:
        ``{"shift", "ncc_before", "ncc_after", "overlap", "ok", "reason"}``
        where ``shift`` is the integer (dz, dy, dx) to feed
        :func:`apply_shift`.
    """
    fm_full = head_mask(fixed) if fixed_mask is None else np.asarray(fixed_mask, bool)
    mm_full = head_mask(moving) if moving_mask is None else np.asarray(moving_mask, bool)
    out: dict[str, Any] = {
        "shift": np.zeros(3, dtype=np.int32),
        "ncc_before": -1.0,
        "ncc_after": -1.0,
        "overlap": 0.0,
        "ok": False,
        "reason": "",
    }
    if fm_full.sum() < 64 or mm_full.sum() < 64:
        out["reason"] = "empty head mask"
        return out

    def _score(fg: np.ndarray, mg: np.ndarray, f_mask: np.ndarray,
               m_mask: np.ndarray, shift: np.ndarray,
               stride: int) -> tuple[float, float]:
        views = _overlap_views(fg, mg, shift, f_mask, m_mask, stride)
        if views is None:
            return -1.0, 0.0
        fv, mv, fmv, mmv = views
        sel = fmv & mmv
        if float(sel.sum()) < 64:
            return -1.0, 0.0
        frac = float(sel.sum()) / max(1.0, float(mmv.sum()))
        if frac < min_overlap:
            return -1.0, frac
        return ncc(fv, mv, sel), frac

    lvls = tuple(int(max(1, l)) for l in levels)
    # Cap the coarsest level so the anatomy still spans enough voxels to be
    # discriminative at that scale. Diagnostics on a 96^3 phantom (brain
    # extent 38 voxels) showed level 8 collapsing the brain to ~5 voxels,
    # which let the coarse sweep lock onto a local maximum of 0.61 at full
    # resolution while the true shift scored 1.00 -- and the +-2 refinements
    # of the finer levels could not escape that basin. Requiring the smallest
    # brain extent to stay >= 8 voxels at the coarsest level fixes it and is
    # a no-op for real head volumes (150 voxels of brain still allow level 8).
    extents: list[int] = []
    for msk in (fm_full, mm_full):
        idx_m = np.argwhere(msk)
        if idx_m.size:
            extents.append(int(np.min(idx_m.max(axis=0) - idx_m.min(axis=0) + 1)))
    if extents:
        allowance = max(1, min(extents) // 8)
        capped = tuple(l for l in lvls if l <= allowance)
        lvls = capped if capped else (2, 1)
    coarse = lvls[0]
    zero = np.zeros(3, dtype=np.int32)
    # Edge maps are computed ONCE at full resolution and downsampled for the
    # coarse levels. This keeps the metric consistent across levels and
    # removes six redundant gradient+morphology passes (~1 s per pair).
    fg_full = gradient_magnitude(fixed) if fixed_grad is None \
        else np.asarray(fixed_grad, dtype=np.float32)
    mg_full = gradient_magnitude(moving)
    out["ncc_before"], out["overlap"] = _score(fg_full, mg_full, fm_full,
                                              mm_full, zero, 1)

    # Per-level sub-sampling keeps every NCC evaluation near
    # ``_EVAL_BUDGET`` voxels; without it the +-2 sweeps at the 120^3 levels
    # dominated the runtime (measured 7.6 s per pair).
    def _stride_for(n_voxels: int) -> int:
        if n_voxels <= _EVAL_BUDGET:
            return 1
        return int(max(1, np.ceil((n_voxels / float(_EVAL_BUDGET)) ** (1.0 / 3.0))))

    current = np.zeros(3, dtype=np.int64) if initial is None else \
        np.rint(np.asarray(initial, dtype=np.float64)).astype(np.int64)
    for lvl in lvls:
        if lvl == 1:
            fg, mg = fg_full, mg_full
            f_mask, m_mask = fm_full, mm_full
        else:
            fg = zoom(fg_full, 1.0 / lvl, order=1)
            mg = zoom(mg_full, 1.0 / lvl, order=1)
            f_mask = zoom(fm_full.astype(np.float32), 1.0 / lvl, order=0) > 0.5
            m_mask = zoom(mm_full.astype(np.float32), 1.0 / lvl, order=0) > 0.5
        stride = max(_stride_for(int(fg.size)),
                     stride_at_full if lvl == 1 else 1)
        base = current if lvl == 1 else np.round(current / lvl).astype(np.int64)
        half = max(0, int(np.ceil((window if lvl == coarse else 2) / lvl)))
        # Bound the coarse sweep: a small ``lvl`` (forced by a small brain)
        # combined with a large window would otherwise explode into tens of
        # thousands of candidates.
        if lvl == coarse:
            half = min(half, 8)
        best = base.copy()
        best_val, best_frac = _score(fg, mg, f_mask, m_mask, best, stride)
        scored: list[tuple[float, float, np.ndarray]] = [(best_val, best_frac, best)]
        for dz in range(-half, half + 1):
            for dy in range(-half, half + 1):
                for dx in range(-half, half + 1):
                    if dz == 0 and dy == 0 and dx == 0:
                        continue
                    cand = np.array([dz, dy, dx], dtype=np.int64) + base
                    val, frac = _score(fg, mg, f_mask, m_mask, cand, stride)
                    scored.append((val, frac, cand))
                    if val > best_val + 1e-9:
                        best_val, best_frac, best = val, frac, cand
        if lvl == coarse and len(scored) > 1:
            # The coarse level merely NOMINATES: re-score its best candidates
            # with the final full-resolution objective before committing, so a
            # downsampled misranking cannot trap the pyramid. NOTE the
            # candidates are in LEVEL coordinates, so they must be scaled by
            # ``lvl`` before being compared against the full-resolution images.
            scored.sort(key=lambda t: t[0], reverse=True)
            fs = _stride_for(int(fg_full.size))
            rb, rv, rf = best, best_val, best_frac
            for _v, _f, cand in scored[:_COARSE_TOPK]:
                v2, f2 = _score(fg_full, mg_full, fm_full, mm_full,
                                cand * lvl, fs)
                if v2 > rv + 1e-9:
                    rv, rf, rb = v2, f2, cand
            best, best_val, best_frac = rb, rv, rf
        current = (best if lvl == 1 else best * lvl).astype(np.int64)
        out["ncc_after"] = best_val
        out["overlap"] = best_frac
        if lvl == 1:
            out["shift"] = current.astype(np.int32)

    out["ok"] = bool(
        out["ncc_after"] > 0.0
        and out["overlap"] >= min_overlap
        and np.isfinite(out["ncc_after"])
    )
    if not out["ok"] and not out["reason"]:
        out["reason"] = (
            f"no candidate beat the overlap/NCC guard "
            f"(overlap {out['overlap']:.2f}, ncc {out['ncc_after']:.3f})"
        )
    return out


def apply_shift(volume: np.ndarray, shift: Sequence[int] | np.ndarray,
                order: int = 1, cval: float = 0.0) -> np.ndarray:
    """Sub-voxel-capable translation; ``shift`` moves volume *content*."""
    sh = tuple(float(s) for s in np.asarray(shift).ravel())
    if all(abs(s) < 1e-9 for s in sh):
        return np.asarray(volume, dtype=np.float32)
    out = nd_shift(np.asarray(volume, dtype=np.float32), sh, order=order,
                   mode="constant", cval=float(cval))
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Grid / resampling helpers
# ---------------------------------------------------------------------------


def resample_to_grid(volume: np.ndarray, affine: np.ndarray, ref_shape,
                     ref_affine: np.ndarray, order: int = 1,
                     cval: float = 0.0) -> np.ndarray:
    """Affine-aware resample of ``volume`` onto an explicit target grid.

    This replaces the shape-ratio ``zoom`` used by ``harmonize_shapes``,
    which ignores the affine entirely (measured: NCC -0.04 for a sub-FOV
    series vs. 0.63 with a true affine-aware resample).

    When the source already lives on the target grid the data is returned
    untouched: interpolation would be both wasted work (measured 2.3 s per
    volume) and an unnecessary blurring of the input.

    Raises:
        BratsifyError: if the moving affine is singular.
    """
    aff = np.asarray(affine, dtype=np.float64)
    if not np.isfinite(aff).all() or abs(float(np.linalg.det(aff[:3, :3]))) < 1e-9:
        raise BratsifyError("non-invertible affine; cannot resample")
    vol = np.asarray(volume, dtype=np.float32)
    same_shape = tuple(int(s) for s in vol.shape) == tuple(int(s) for s in ref_shape)
    if same_shape and np.allclose(aff, np.asarray(ref_affine, np.float64),
                                  atol=1e-4, rtol=0.0):
        return vol
    from nibabel.processing import resample_from_to
    import nibabel as nib

    moving = nib.Nifti1Image(vol, aff)
    target = (tuple(int(s) for s in ref_shape), np.asarray(ref_affine, np.float64))
    out = resample_from_to(moving, target, order=int(order),
                           mode="constant", cval=float(cval))
    return np.asarray(out.dataobj, dtype=np.float32)


def resample_to_spacing(volume: np.ndarray, affine: np.ndarray,
                        spacing: Sequence[float],
                        order: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Resample to a target voxel spacing; returns ``(volume, affine)``.

    Exact for axis-aligned affines (verified: a 60-voxel 2 mm axis becomes
    120 voxels at 1 mm).
    """
    aff = np.asarray(affine, dtype=np.float64).copy()
    vol = np.asarray(volume, dtype=np.float32)
    current = np.array([float(np.linalg.norm(aff[:3, i])) for i in range(3)])
    target = np.asarray([float(s) for s in spacing], dtype=np.float64)
    factors = current / np.maximum(target, 1e-9)
    if np.allclose(factors, 1.0, atol=1e-6):
        return vol, aff
    out = zoom(vol, tuple(factors), order=int(order)).astype(np.float32)
    # More voxels per unit of space => SMALLER voxels: divide, don't multiply.
    # (Verified: 2 mm x 120 voxels -> zoom 2 -> 1 mm x 240 voxels, and the
    # physical extent 240 mm is preserved.)
    for i in range(3):
        aff[:3, i] = aff[:3, i] / float(factors[i])
    return out, aff


def target_grid_from_reference(ref_shape, ref_affine, spacing=(1.0, 1.0, 1.0)
                               ) -> tuple[tuple[int, int, int], np.ndarray]:
    """Grid that holds the reference volume resampled to ``spacing``.

    The origin is chosen so that the *world-space bounding box centre* of
    the reference volume stays fixed, which keeps the head centred in the
    output regardless of the input spacing.
    """
    shape = np.asarray(ref_shape, dtype=np.float64)
    aff = np.asarray(ref_affine, dtype=np.float64).copy()
    cur = np.array([float(np.linalg.norm(aff[:3, i])) for i in range(3)])
    tgt = np.asarray([float(s) for s in spacing], dtype=np.float64)
    new_shape = tuple(int(max(1, round(s * c / t))) for s, c, t in
                      zip(shape, cur, tgt))
    # Preserve the world centre: c_old -> c_new.
    centre_old = aff[:3, :3] @ ((shape - 1.0) / 2.0) + aff[:3, 3]
    unit = aff[:3, :3] / cur  # unit direction vectors (mm per voxel removed)
    new_aff = np.eye(4, dtype=np.float64)
    new_aff[:3, :3] = unit * tgt
    new_shape_f = np.asarray(new_shape, dtype=np.float64)
    new_aff[:3, 3] = centre_old - (unit * tgt) @ ((new_shape_f - 1.0) / 2.0)
    return new_shape, new_aff


def to_brats_canvas(volume: np.ndarray, affine: np.ndarray,
                    canvas: Optional[Sequence[int]] = (240, 240, 155),
                    mask: Optional[np.ndarray] = None,
                    cval: float = 0.0
                    ) -> tuple[np.ndarray, np.ndarray, bool, Optional[np.ndarray]]:
    """Centre a volume inside the BraTS canvas, *never* cutting anatomy.

    The canvas is only ever **grown** to at least the brain bounding box, so
    a scan whose brain exceeds 155 mm in z is padded rather than cropped.
    This is a deliberate safety choice: ``prepare_for_inference`` re-crops to
    the brain bbox anyway, so a smaller canvas would only risk losing the
    tumour and gain nothing.

    Returns:
        ``(padded_volume, new_affine, grow_only, padded_mask)`` where
        ``grow_only`` is ``True`` when the requested canvas had to be
        enlarged, and ``padded_mask`` is the identically padded mask when one
        was supplied (``None`` otherwise).
    """
    vol = np.asarray(volume, dtype=np.float32)
    aff = np.asarray(affine, dtype=np.float64).copy()
    if canvas is None:
        return vol, aff, False, (None if mask is None else np.asarray(mask, bool))
    want = np.asarray([int(c) for c in canvas], dtype=np.int64)
    brain = head_mask(vol) if mask is None else np.asarray(mask, bool)
    idx = np.argwhere(brain)
    if idx.size == 0:
        extent = np.asarray(vol.shape, dtype=np.int64)
    else:
        extent = (idx.max(axis=0) - idx.min(axis=0) + 3).astype(np.int64)
    target = np.maximum(want, extent)
    target = np.maximum(target, np.asarray(vol.shape))
    grow_only = bool(np.any(target > want))
    if np.array_equal(target, np.asarray(vol.shape)):
        return vol, aff, grow_only, (None if mask is None else brain)
    centre = _bbox_center(brain)
    off = np.round(target / 2.0 - centre).astype(np.int64)
    out = np.full(tuple(int(t) for t in target), float(cval), dtype=np.float32)
    src_shape = np.asarray(vol.shape, dtype=np.int64)
    s_start = np.maximum(0, -off)
    s_end = np.minimum(src_shape, target - off)
    if np.any(s_end <= s_start):
        return vol, aff, grow_only, (None if mask is None else brain)
    d_start = np.maximum(0, off)
    d_end = np.minimum(target, src_shape + off)
    sl_src = tuple(slice(int(a), int(b)) for a, b in zip(s_start, s_end))
    sl_dst = tuple(slice(int(a), int(b)) for a, b in zip(d_start, d_end))
    out[sl_dst] = vol[sl_src]
    # In index space, destination voxel `k + off` corresponds to source voxel `k`.
    # Therefore, world(k_dst) = world(k_src) requires:
    # A_dst @ (k + off) = A_src @ k
    # A_dst @ k + A_dst @ off = A_src @ k  (with A_dst[:3, :3] == A_src[:3, :3])
    # A_src @ k + t_dst + R @ off = A_src @ k + t_src
    # => t_dst = t_src - R @ off
    aff[:3, 3] = aff[:3, 3] - aff[:3, :3] @ off.astype(np.float64)
    out_mask: Optional[np.ndarray] = None
    if mask is not None:
        out_mask = np.zeros(tuple(int(t) for t in target), dtype=bool)
        out_mask[sl_dst] = brain[sl_src]
    return out, aff, grow_only, out_mask


# ---------------------------------------------------------------------------
# N4 bias-field correction
# ---------------------------------------------------------------------------


def simpleitk_available() -> bool:
    """``True`` when SimpleITK can be imported (N4 is then possible)."""
    try:
        import SimpleITK  # noqa: F401
        return True
    except Exception:
        return False


def n4_bias_correct(volume: np.ndarray, mask: np.ndarray,
                    spacing: Sequence[float] = (1.0, 1.0, 1.0),
                    cfg: Optional[BratsifyConfig] = None
                    ) -> tuple[np.ndarray, dict[str, Any]]:
    """Guardrailed SimpleITK N4 correction of one modality.

    N4 needs strictly positive, reasonably scaled input, so the volume is
    multiplied by ``1000 / median(in-mask intensity)`` before fitting and the
    *field* (not the corrected image) is divided back out afterwards, which
    makes the rescale an exact no-op on the final result
    (``corrected = vol / exp(log_bias)``).

    Rejection rules, calibrated against a known-good BraTS T1 whose field has
    in-mask p1/p99 = [0.666, 0.954] (so a naive "+-25%" rule would have
    rejected healthy data):

    * mask covers < ``n4_min_mask_frac`` of the volume -> skip;
    * field's in-mask p1/p99 spread < ``n4_min_spread`` -> no-op, skip;
    * p1/p99 outside ``[n4_field_lo, n4_field_hi]`` -> revert;
    * in-mask mean intensity scale outside
      ``[n4_min_mean_scale, n4_max_mean_scale]`` -> revert;
    * non-finite field -> revert.

    A naive multiplicative field estimate was measured to *harm* the data
    (in-mask intensity CV 0.003 -> 0.269), hence these guards.

    Returns:
        ``(volume_or_original, info)``; ``info`` always contains ``applied``/
        ``available`` and, when relevant, ``reverted``/``skipped``/``reason``
        plus the measured field statistics and runtime.
    """
    cfg = cfg or BratsifyConfig()
    vol = np.asarray(volume, dtype=np.float32)
    m = np.asarray(mask, bool)
    info: dict[str, Any] = {"applied": False, "available": simpleitk_available(),
                            "reason": ""}
    if not cfg.n4:
        info["reason"] = "disabled"
        return vol, info
    if not info["available"]:
        info["reason"] = "SimpleITK not installed"
        return vol, info
    if m.sum() < 64 or float(m.mean()) < cfg.n4_min_mask_frac:
        info["reason"] = f"mask too small ({float(m.mean()):.3f} of volume)"
        return vol, info

    import SimpleITK as sitk

    pos = vol[m]
    median = float(np.median(pos)) if pos.size else 0.0
    scale = 1000.0 / median if median > 1e-6 else 1.0
    work = np.clip(vol * scale, 0.0, None).astype(np.float32)

    t0 = time.time()
    try:
        img = sitk.GetImageFromArray(work)
        # ITK expects (z, y, x) array order -> spacing must be reversed.
        img.SetSpacing(tuple(float(s) for s in np.asarray(spacing, float)[::-1]))
        mask_img = sitk.GetImageFromArray(m.astype(np.uint8))
        mask_img.CopyInformation(img)
        filt = sitk.N4BiasFieldCorrectionImageFilter()
        filt.SetMaximumNumberOfIterations(list(cfg.n4_iterations))
        filt.SetNumberOfControlPoints([int(cfg.n4_control_points)] * 3)
        filt.SetConvergenceThreshold(float(cfg.n4_convergence))
        shrink = int(max(1, cfg.n4_shrink))
        if shrink > 1:
            fit_img = sitk.Shrink(img, [shrink] * 3)
            fit_mask = sitk.Shrink(mask_img, [shrink] * 3)
        else:
            fit_img, fit_mask = img, mask_img
        filt.Execute(fit_img, fit_mask)
        log_field = filt.GetLogBiasFieldAsImage(img)
    except Exception as exc:  # pragma: no cover - ITK-internal failures
        info["reason"] = f"N4 failed: {exc}"
        info["seconds"] = time.time() - t0
        return vol, info
    info["seconds"] = time.time() - t0

    field = np.exp(np.asarray(sitk.GetArrayFromImage(log_field), dtype=np.float32))
    if field.shape != vol.shape:
        field = np.exp(np.asarray(
            sitk.GetArrayFromImage(sitk.Resample(log_field, img)),
            dtype=np.float32))
    if not np.isfinite(field).all() or float(np.min(field)) <= 1e-6:
        info["reverted"] = True
        info["reason"] = "non-positive or non-finite bias field"
        return vol, info

    inner = field[m]
    p1 = float(np.percentile(inner, 1.0))
    p99 = float(np.percentile(inner, 99.0))
    info["field_p1"], info["field_p99"] = p1, p99
    info["field_min"], info["field_max"] = float(field.min()), float(field.max())
    if (p99 - p1) < cfg.n4_min_spread:
        info["skipped"] = True
        info["reason"] = f"bias field negligible (p99-p1={p99 - p1:.3f})"
        return vol, info
    if p1 < cfg.n4_field_lo or p99 > cfg.n4_field_hi:
        info["reverted"] = True
        info["reason"] = (
            f"implausible bias field (in-mask p1={p1:.2f}, p99={p99:.2f} "
            f"outside [{cfg.n4_field_lo}, {cfg.n4_field_hi}])"
        )
        return vol, info

    corrected = vol / np.maximum(field, 1e-6)
    mean_before = float(vol[m].mean())
    mean_after = float(corrected[m].mean())
    mean_ratio = (mean_before / mean_after) if abs(mean_after) > 1e-12 else 0.0
    info["mean_scale"] = mean_ratio
    if not np.isfinite(mean_ratio) or mean_ratio < cfg.n4_min_mean_scale \
            or mean_ratio > cfg.n4_max_mean_scale:
        info["reverted"] = True
        info["reason"] = f"runaway intensity scale (x{mean_ratio:.2f})"
        return vol, info

    info["applied"] = True
    info["mean_before"], info["mean_after"] = mean_before, mean_after
    return corrected.astype(np.float32), info


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _fov_extents(affine: np.ndarray, shape) -> np.ndarray:
    """Per-world-axis physical extent (mm) of a volume."""
    return np.abs(np.asarray(affine, np.float64)[:3, :3]) @ np.asarray(shape, float)


def _empty_report(reference: str, cfg: BratsifyConfig) -> BratsifyReport:
    """Report pre-populated with the configuration block of the provenance."""
    rep = BratsifyReport(reference=reference)
    rep.provenance = {
        "phase": "A",
        "stage": "bratsify",
        "config": {
            "n4": bool(cfg.n4),
            "n4_shrink": int(cfg.n4_shrink),
            "n4_iterations": list(cfg.n4_iterations),
            "n4_control_points": int(cfg.n4_control_points),
            "target_spacing": list(cfg.target_spacing),
            "brats_canvas": list(cfg.brats_canvas) if cfg.brats_canvas else None,
            "register": bool(cfg.register),
            "max_shift_frac": float(cfg.max_shift_frac),
            "min_overlap": float(cfg.min_overlap),
            "min_ncc_gain": float(cfg.min_ncc_gain),
        },
        "reference": reference,
        "modalities": {},
        "warnings": [],
        "steps": [],
        "applied": False,
    }
    return rep


def bratsify_scans(scans: PatientScans, cfg: Optional[BratsifyConfig] = None,
                   reference: str = "t1ce"
                   ) -> tuple[PatientScans, BratsifyReport]:
    """Convert raw hospital scans into BraTS-grade volumes.

    Runs the full Phase A chain (see the module docstring): head masks ->
    guarded COM pre-alignment -> single affine-aware resample onto the BraTS
    grid -> residual grad-NCC refinement -> guarded N4 -> optional brain-only
    z-score. Every stage is wrapped so that **any** failure returns the
    original ``scans`` unchanged together with an explanatory report.

    Args:
        scans: Validated multi-modal :class:`PatientScans`.
        cfg: Optional :class:`BratsifyConfig`; defaults are the measured ones.
        reference: Modality whose geometry defines the output grid.

    Returns:
        ``(scans_or_new_scans, report)``.
    """
    cfg = cfg or BratsifyConfig()
    ref_key = reference if reference in scans.volumes else MODALITY_ORDER[0]
    report = _empty_report(ref_key, cfg)
    t0 = time.time()
    if not cfg.enabled:
        report.warnings.append("BraTS-ifier disabled by config.")
        return scans, report

    try:
        vols: dict[str, np.ndarray] = {}
        affs: dict[str, np.ndarray] = {}
        for mod in MODALITY_ORDER:
            mod_vol = np.asarray(scans.volumes[mod], dtype=np.float32)
            mod_aff = np.asarray(scans.metadata[mod].affine, dtype=np.float64)
            if mod_aff.shape != (4, 4) or not np.isfinite(mod_aff).all():
                raise BratsifyError(f"'{mod}' has an invalid affine")
            if any(int(s) <= 0 for s in mod_vol.shape):
                raise BratsifyError(f"'{mod}' has an empty volume")
            vols[mod] = mod_vol
            affs[mod] = mod_aff
    except Exception as exc:
        report.warnings.append(f"BraTS-ifier skipped: {exc}")
        return scans, report

    native_masks = {m: head_mask(vols[m], cfg.head_pct) for m in MODALITY_ORDER}
    report.steps.append("head masks (per modality)")

    # --- 1. output grid: reference geometry at the target spacing ---------
    tgt_shape, tgt_aff = target_grid_from_reference(
        vols[ref_key].shape, affs[ref_key], cfg.target_spacing)
    try:
        ref_iso = resample_to_grid(vols[ref_key], affs[ref_key], tgt_shape, tgt_aff)
    except Exception as exc:
        report.warnings.append(
            f"BraTS-ifier skipped: canonical resample failed ({exc}).")
        return scans, report
    report.steps.append(
        f"affine-aware resample of '{ref_key}' to "
        f"{tuple(int(s) for s in tgt_shape)} "
        f"@ {tuple(float(s) for s in cfg.target_spacing)} mm")

    # --- 2. grow-only BraTS canvas (pure padding, no interpolation) -------
    grew = False
    ref_brain = head_mask(ref_iso, cfg.head_pct)
    if cfg.brats_canvas:
        ref_iso, tgt_aff, grew, ref_brain = to_brats_canvas(
            ref_iso, tgt_aff, cfg.brats_canvas, mask=ref_brain,
            cval=cfg.pad_value)
        tgt_shape = tuple(int(s) for s in ref_iso.shape)
        report.steps.append(
            f"BraTS canvas {tuple(int(s) for s in tgt_shape)}"
            + (" (grown to fit the brain)" if grew else ""))
    if grew:
        report.warnings.append(
            "Requested BraTS canvas was enlarged to keep the whole brain in "
            "the field of view (no anatomy cropped).")

    outputs: dict[str, np.ndarray] = {ref_key: ref_iso}
    finals: dict[str, np.ndarray] = {ref_key: ref_brain}
    raw_resampled_for_qa: dict[str, np.ndarray] = {}
    reg_info: dict[str, dict[str, Any]] = {
        ref_key: {"reference": True, "skipped": False, "shift_mm": 0.0,
                  "shift_px": [0, 0, 0], "overlap": 1.0, "ncc_after": 1.0}}
    ref_com_world = (affs[ref_key][:3, :3] @ _com(native_masks[ref_key])
                     + affs[ref_key][:3, 3])
    ref_extents = _fov_extents(affs[ref_key], vols[ref_key].shape)
    out_spacing = tuple(float(np.linalg.norm(tgt_aff[:3, i])) for i in range(3))
    # Edge map of the reference: identical for every pairwise refinement.
    ref_grad = gradient_magnitude(ref_iso)

    # --- 3. align + resample every other modality onto the grid ----------
    for mod in MODALITY_ORDER:
        if mod == ref_key:
            continue
        info: dict[str, Any] = {"reference": False, "skipped": False,
                                "shift_mm": 0.0, "shift_px": [0, 0, 0],
                                "overlap": 0.0, "ncc_before": -1.0,
                                "ncc_after": -1.0, "reason": ""}
        mod_com_world = (affs[mod][:3, :3] @ _com(native_masks[mod])
                         + affs[mod][:3, 3])
        world_shift = mod_com_world - ref_com_world
        frac = float(np.max(np.abs(world_shift)
                            / np.maximum(ref_extents, 1e-6)))
        info["com_shift_mm"] = float(np.linalg.norm(world_shift))
        info["com_shift_frac"] = frac
        use_com = bool(cfg.register) and frac <= cfg.max_shift_frac
        if cfg.register and not use_com:
            info["skipped"] = True
            info["reason"] = (
                f"COM offset {100 * frac:.1f}% of the reference FOV exceeds "
                f"the {100 * cfg.max_shift_frac:.0f}% guard")
            report.warnings.append(
                f"'{mod}': alignment skipped ({info['reason']}); using the "
                "affine-only placement.")
        # Resample with the RAW affine only. Folding the centre-of-mass
        # offset into the affine would force an interpolation even for
        # already co-registered studies (measured 2.1 s per modality) and
        # would nudge good data, because the COM of a head mask depends on
        # the modality's contrast. The COM offset is used as the *search
        # seed* below instead, so a genuine header error is still found.
        try:
            resampled = resample_to_grid(vols[mod], affs[mod], tgt_shape,
                                         tgt_aff, cval=cfg.pad_value)
        except Exception as exc:
            report.warnings.append(
                f"'{mod}': resample failed ({exc}); substituted the "
                "reference volume.")
            resampled = ref_iso
            info["skipped"] = True
            info["reason"] = f"resample failed: {exc}"
        raw_resampled_for_qa[mod] = resampled.copy()
        mod_mask = head_mask(resampled, cfg.head_pct)
        if use_com and not info["skipped"]:
            seed = -estimate_com_shift(mod_mask, finals[ref_key])
            est = estimate_shift(resampled, ref_iso, moving_mask=mod_mask,
                                 fixed_mask=finals[ref_key],
                                 window=cfg.search_window_px,
                                 min_overlap=cfg.min_overlap,
                                 fixed_grad=ref_grad, initial=seed)
            info["ncc_before"] = float(est["ncc_before"])
            info["overlap"] = float(est["overlap"])
            shift = np.asarray(est["shift"], dtype=int)
            gained = float(est["ncc_after"]) - float(est["ncc_before"])
            info["ncc_gain"] = gained
            if est["ok"] and int(np.abs(shift).sum()) > 0 \
                    and gained >= cfg.min_ncc_gain:
                resampled = apply_shift(resampled, shift, cval=cfg.pad_value)
                # Shift the mask identically (nearest neighbour) instead of
                # recomputing it: head_mask costs ~0.5 s per call and the
                # mask has already been validated on the un-shifted volume.
                mod_mask = nd_shift(mod_mask.astype(np.float32),
                                    tuple(float(s) for s in shift),
                                    order=0, mode="constant", cval=0.0) > 0.5
                info["shift_px"] = [int(s) for s in shift]
                info["ncc_after"] = float(est["ncc_after"])
                info["shift_mm"] = float(np.linalg.norm(
                    shift.astype(float) * np.asarray(out_spacing)))
            elif not est["ok"]:
                info["skipped"] = True
                info["reason"] = f"residual refinement rejected: {est['reason']}"
                report.warnings.append(f"'{mod}': {info['reason']}")
            else:
                # Already aligned (or the candidate shift was not a real
                # improvement) -> leave the data untouched.
                info["shift_px"] = [0, 0, 0]
                info["ncc_after"] = float(est["ncc_before"])
                info["reason"] = (
                    f"no shift applied (best gain {gained:+.3f} < "
                    f"{cfg.min_ncc_gain}); input already aligned")
        outputs[mod] = np.asarray(resampled, dtype=np.float32)
        finals[mod] = mod_mask
        reg_info[mod] = info
    report.steps.append("rigid residual refinement (gradient-magnitude NCC)")
    report.registration = reg_info

    # --- 4. guarded N4 bias-field correction ------------------------------
    if cfg.n4:
        for mod in MODALITY_ORDER:
            corrected, n4_info = n4_bias_correct(outputs[mod], finals[mod],
                                                 out_spacing, cfg)
            outputs[mod] = corrected
            report.n4[mod] = n4_info
            if n4_info.get("applied"):
                report.steps.append(f"N4 applied to '{mod}'")
            elif n4_info.get("reverted"):
                report.warnings.append(
                    f"'{mod}': N4 reverted ({n4_info.get('reason')}).")
        if not simpleitk_available():
            report.warnings.append(
                "SimpleITK not installed: N4 bias correction skipped "
                "(pip install simpleitk).")

    # --- 5. optional brain-only z-score -----------------------------------
    if cfg.zscore_brain:
        for mod in MODALITY_ORDER:
            brain = finals[mod]
            if brain.sum() >= 64:
                sel = outputs[mod][brain]
                mu, sd = float(sel.mean()), float(sel.std())
                if sd > 1e-6:
                    outputs[mod] = ((outputs[mod] - mu) / sd).astype(np.float32)
        report.steps.append("brain-only z-score")

    # --- 6. package -------------------------------------------------------
    new_meta: dict[str, SpatialMetadata] = {}
    for mod in MODALITY_ORDER:
        new_meta[mod] = SpatialMetadata(
            affine=tgt_aff.copy(),
            voxel_spacing_mm=tuple(float(s) for s in out_spacing),
            original_shape=tuple(int(s) for s in outputs[mod].shape),
            cropped_shape=tuple(int(s) for s in outputs[mod].shape),
            crop_slices=(slice(None), slice(None), slice(None)),
        )
    out_scans = PatientScans(volumes=outputs, metadata=new_meta,
                             source=scans.source + f"|bratsify->{ref_key}")

    report.applied = True
    report.final_shape = tuple(int(s) for s in outputs[ref_key].shape)
    report.final_spacing = out_spacing
    report.final_affine = tgt_aff.copy()
    report.seconds = time.time() - t0
    report.provenance["applied"] = True
    report.provenance["steps"] = list(report.steps)
    report.provenance["warnings"] = list(report.warnings)
    report.provenance["final"] = {
        "shape": [int(s) for s in report.final_shape],
        "spacing_mm": [float(s) for s in out_spacing],
        "affine": tgt_aff.tolist(),
        "canvas_grown": bool(grew),
    }
    report.provenance["modalities"] = {
        mod: {
            "registration": reg_info.get(mod, {}),
            "n4": report.n4.get(mod, {}),
            "intensity": {
                "min": float(np.min(outputs[mod])),
                "max": float(np.max(outputs[mod])),
                "mean_brain": float(outputs[mod][finals[mod]].mean())
                if finals[mod].any() else 0.0,
            },
        }
        for mod in MODALITY_ORDER
    }
    report.provenance["seconds"] = float(report.seconds)

    # --- 7. optional visual QA output -------------------------------------
    if cfg.qa_path:
        qa_mod = cfg.qa_modality if cfg.qa_modality in raw_resampled_for_qa else (
            next((m for m in ("t2", "flair", "t1") if m in raw_resampled_for_qa), None)
        )
        if qa_mod is not None:
            save_qa_png(
                cfg.qa_path,
                outputs[ref_key],
                raw_resampled_for_qa[qa_mod],
                outputs[qa_mod],
                modality=qa_mod,
            )

    return out_scans, report


# ---------------------------------------------------------------------------
# QA / audit artefacts
# ---------------------------------------------------------------------------


def _norm_slice(sl: np.ndarray) -> np.ndarray:
    """Per-slice min-max normalisation for display (mirrors the app)."""
    lo, hi = float(np.min(sl)), float(np.max(sl))
    if hi - lo < 1e-8:
        return np.zeros_like(sl, dtype=np.float32)
    return ((sl - lo) / (hi - lo)).astype(np.float32)


def _checkerboard(a: np.ndarray, b: np.ndarray, block: int = 16) -> np.ndarray:
    """RGB checkerboard mixing two same-shape 2D slices (A/B interleave)."""
    a2, b2 = _norm_slice(a), _norm_slice(b)
    h, w = a2.shape
    zz, yy = np.mgrid[0:h, 0:w]
    take_a = (((zz // max(1, block)) + (yy // max(1, block))) % 2) == 0
    gray = np.where(take_a, a2, b2)
    rgb = np.stack([gray, gray, gray], axis=-1)
    # Tint B (the moving/after image) so the two sources are distinguishable.
    rgb[~take_a] = rgb[~take_a] * np.array([0.85, 0.95, 1.0], dtype=np.float32)
    return np.clip(rgb * 255.0, 0, 255).astype(np.uint8)


def save_qa_png(path: str, reference: np.ndarray, before: np.ndarray,
                after: np.ndarray, modality: str = "t1", block: int = 16
                ) -> Optional[str]:
    """Write a 4-panel checkerboard QA PNG (reference / before / after / diff).

    Checkerboarding is the standard visual test for registration: if the
    gyral edges line up across tile boundaries, the volumes are aligned.

    Returns:
        The written path, or ``None`` when matplotlib is unavailable — QA
        must never break the pipeline.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    ref = np.asarray(reference, dtype=np.float32)
    bfr = np.asarray(before, dtype=np.float32)
    aft = np.asarray(after, dtype=np.float32)
    if ref.ndim == 3:
        z = ref.shape[0] // 2
        ref, bfr, aft = ref[z], bfr[z], aft[z]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2))
    axes[0].imshow(_norm_slice(ref), cmap="gray")
    axes[0].set_title("reference")
    axes[1].imshow(_checkerboard(ref, bfr, block))
    axes[1].set_title("checkerboard BEFORE")
    axes[2].imshow(_checkerboard(ref, aft, block))
    axes[2].set_title("checkerboard AFTER")
    axes[3].imshow(np.abs(_norm_slice(ref) - _norm_slice(aft)), cmap="magma")
    axes[3].set_title("|reference - after|")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"Phase A QA — {modality} (central slice, {block}px tiles)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def write_provenance(report: BratsifyReport, path: str) -> str:
    """Persist the full transform chain as JSON for audit/download."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report.to_json())
    return path


def export_brats_style(scans: PatientScans, out_dir: str,
                       report: Optional[BratsifyReport] = None,
                       prefix: str = "bratsify") -> dict[str, str]:
    """Write BraTS-named NIfTIs (``<prefix>_t1.nii.gz`` ...) on the shared grid.

    Args:
        scans: BraTS-ified scans (or any :class:`PatientScans`).
        out_dir: Destination directory.
        report: When given, its provenance JSON is written alongside.
        prefix: Filename prefix.

    Returns:
        Mapping of modality -> written path (plus ``"provenance"``).
    """
    import nibabel as nib

    os.makedirs(out_dir, exist_ok=True)
    paths: dict[str, str] = {}
    for mod in MODALITY_ORDER:
        meta = scans.metadata[mod]
        img = nib.Nifti1Image(np.asarray(scans.volumes[mod], dtype=np.float32),
                              np.asarray(meta.affine, dtype=np.float64))
        img.set_qform(np.asarray(meta.affine, dtype=np.float64), code=1)
        img.set_sform(np.asarray(meta.affine, dtype=np.float64), code=1)
        p = os.path.join(out_dir, f"{prefix}_{mod}.nii.gz")
        nib.save(img, p)
        paths[mod] = p
    if report is not None:
        paths["provenance"] = write_provenance(
            report, os.path.join(out_dir, f"{prefix}_provenance.json"))
    return paths
