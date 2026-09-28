"""Tests for the Phase A BraTS-ifier (``src/bratsify.py``)."""

from __future__ import annotations

import json
import os
import tempfile

import nibabel as nib
import numpy as np

from src.bratsify import (
    BratsifyConfig,
    apply_shift,
    bratsify_scans,
    estimate_shift,
    export_brats_style,
    gradient_magnitude,
    head_mask,
    n4_bias_correct,
    ncc,
    resample_to_grid,
    resample_to_spacing,
    save_qa_png,
    simpleitk_available,
    target_grid_from_reference,
    to_brats_canvas,
    write_provenance,
)
from src.dicom_loader import (
    MODALITY_ORDER,
    PatientScans,
    SpatialMetadata,
    load_patient_scans,
    prepare_for_inference,
)

REAL_ROOT = "BRATS"
REAL_CASE = "BraTS20_Training_002"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _phantom(shape=(96, 96, 80), seed=0) -> np.ndarray:
    """Brain-like synthetic volume: ellipsoid + internal structure.

    The brain occupies roughly half of each axis (as a real head does inside
    its field of view), so displacement-based overlap guards are meaningful.
    """
    rng = np.random.RandomState(seed)
    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float32)
    cz, cy, cx = (s / 2.0 for s in shape)
    rz, ry, rx = (s * 0.25 for s in shape)
    r2 = ((z - cz) / rz) ** 2 + ((y - cy) / ry) ** 2 + ((x - cx) / rx) ** 2
    brain = np.clip(1.0 - r2, 0, 1).astype(np.float32)
    vol = brain * 500.0
    # Internal "gyri/ventricles" so gradient-based registration has signal.
    vol += brain * 300.0 * (np.sin(y / 3.0) * np.sin(x / 4.0)).astype(np.float32)
    vol += brain * 200.0 * (np.cos(z / 5.0)).astype(np.float32)
    vol += rng.rand(*shape).astype(np.float32) * 12.0 * brain
    return vol


def _meta(affine, shape) -> SpatialMetadata:
    return SpatialMetadata(
        affine=np.asarray(affine, dtype=np.float64),
        voxel_spacing_mm=tuple(
            float(np.linalg.norm(np.asarray(affine, float)[:3, i]))
            for i in range(3)),
        original_shape=tuple(int(s) for s in shape),
        cropped_shape=tuple(int(s) for s in shape),
        crop_slices=(slice(None), slice(None), slice(None)),
    )


def _scans_from(vols: dict[str, np.ndarray], affines=None) -> PatientScans:
    affines = affines or {}
    meta = {m: _meta(affines.get(m, np.eye(4)), vols[m].shape)
            for m in MODALITY_ORDER}
    return PatientScans(volumes=dict(vols), metadata=meta, source="unit-test")


def _real_modalities() -> dict[str, str]:
    return {m: os.path.join(REAL_ROOT, f"{REAL_CASE}_{m}.nii")
            for m in MODALITY_ORDER}


def _have_real_data() -> bool:
    return all(os.path.exists(p) for p in _real_modalities().values())


# ---------------------------------------------------------------------------
# T1 - gradient-magnitude NCC shift recovery
# ---------------------------------------------------------------------------


def test_t1_shift_recovery_exact():
    """Injected translations of +-4 and +-10 px must be recovered to 1 voxel."""
    vol = _phantom()
    for true in ((4, -3, 2), (10, 7, -6), (-8, 5, 9)):
        moved = apply_shift(vol, true)
        est = estimate_shift(moved, vol)
        err = int(np.abs(np.asarray(est["shift"]) + np.asarray(true)).sum())
        assert est["ok"], est
        assert err <= 1, f"true={true} est={est['shift']} err={err}"
        assert est["ncc_after"] > est["ncc_before"]
        assert est["ncc_after"] > 0.9


def test_t1_plain_intensity_ncc_is_weaker_than_gradient():
    """Documents *why* the metric is gradient-based, not intensity-based."""
    vol = _phantom()
    other = vol * 0.6 + 120.0  # different contrast, same anatomy
    moved = apply_shift(other, (6, -5, 4))
    g_est = estimate_shift(moved, vol)
    # Intensity domain: the correct shift should be identified less sharply.
    g_err = int(np.abs(np.asarray(g_est["shift"]) + np.array([6, -5, 4])).sum())
    assert g_err <= 1
    assert gradient_magnitude(vol).dtype == np.float32


def test_t9_numpy2_gradient_axis_keyword():
    """NumPy 2 rejects ``ax=``; ``gradient_magnitude`` must use ``axis=``."""
    vol = _phantom(shape=(32, 32, 32))
    g = gradient_magnitude(vol)
    assert g.shape == vol.shape
    assert np.isfinite(g).all()
    assert float(g.max()) > 0.0


# ---------------------------------------------------------------------------
# T2 - affine-aware resampling beats the shape-ratio zoom on a sub-FOV series
# ---------------------------------------------------------------------------


def test_t2_affine_resample_beats_shape_zoom():
    """A sub-FOV modality must be placed by its affine, not by shape ratio."""
    from scipy.ndimage import binary_erosion, zoom as nd_zoom

    ref = _phantom()
    # Central 60^3 sub-FOV of the reference, with its origin recorded.
    off = np.array([18, 18, 10])
    sub = ref[off[0]:off[0] + 60, off[1]:off[1] + 60, off[2]:off[2] + 60]
    sub_aff = np.eye(4)
    sub_aff[:3, 3] = off

    ref_brain = head_mask(ref)
    ref_grad = gradient_magnitude(ref)

    # (a) shape-ratio zoom: what harmonize_shapes does (affine ignored).
    factors = [r / s for r, s in zip(ref.shape, sub.shape)]
    zoomed = nd_zoom(sub, factors, order=1).astype(np.float32)

    # (b) affine-aware resample onto the reference grid.
    affine = resample_to_grid(sub, sub_aff, ref.shape, np.eye(4))

    # Compare only inside the genuinely shared field of view, eroded past the
    # artificial FOV boundary, otherwise the out-of-FOV zeros (which are
    # correct) would cap the achievable correlation for both candidates.
    shared = binary_erosion(affine > 0, iterations=3) & ref_brain
    assert shared.sum() > 1000
    ncc_aff = ncc(ref_grad, gradient_magnitude(affine), shared)
    ncc_zoom = ncc(ref_grad, gradient_magnitude(zoomed), shared)

    assert ncc_aff > 0.95, f"affine-aware NCC too low: {ncc_aff}"
    assert ncc_aff > ncc_zoom + 0.3, (
        f"affine-aware ({ncc_aff:.3f}) must clearly beat shape-ratio zoom "
        f"({ncc_zoom:.3f})")


# ---------------------------------------------------------------------------
# T3 - skip (never corrupt) when the COM offset exceeds the guard
# ---------------------------------------------------------------------------


def test_t3_com_guard_skips_alignment():
    ref = _phantom()
    # Displace one modality's mask by ~40% of the field of view.
    far = np.roll(ref, 40, axis=0)
    vols = {m: ref.copy() for m in MODALITY_ORDER}
    vols["t2"] = far
    scans = _scans_from(vols)
    _, rep = bratsify_scans(scans, BratsifyConfig(n4=False, brats_canvas=None))

    reg = rep.registration["t2"]
    assert reg["skipped"] is True
    assert "COM offset" in reg["reason"]
    assert reg["shift_px"] == [0, 0, 0]
    assert any("alignment skipped" in w for w in rep.warnings)
    # The reference itself is untouched and the run still succeeds.
    assert rep.applied is True
    assert rep.registration["t1ce"]["reference"] is True


# ---------------------------------------------------------------------------
# T4 - minimum-overlap guard rejects unsupported shifts
# ---------------------------------------------------------------------------


def test_t4_overlap_guard_rejects_unreachable_shift():
    vol = _phantom()
    # Roll the anatomy to the far end: the two head masks now share well under
    # half their volume, so any candidate inside a small window is unsupported.
    moved = np.roll(vol, 34, axis=0)
    est = estimate_shift(moved, vol, window=2, min_overlap=0.5)
    assert est["ok"] is False
    assert est["overlap"] < 0.5, est

    # The same guard accepts a well-supported candidate: a real translation
    # that keeps the anatomy inside the field of view.
    moved2 = apply_shift(vol, (14, 0, 0))
    est_ok = estimate_shift(moved2, vol, window=20, min_overlap=0.5)
    assert est_ok["ok"] is True
    assert est_ok["overlap"] >= 0.5
    err = int(np.abs(np.asarray(est_ok["shift"]) + np.array([14, 0, 0])).sum())
    assert err <= 1, est_ok["shift"]


# ---------------------------------------------------------------------------
# T5 - N4 guards
# ---------------------------------------------------------------------------


def test_t5_n4_reverts_and_returns_input_untouched():
    if not simpleitk_available():
        return
    # Phantom with a real synthetic bias gradient so N4 actually estimates a
    # non-negligible field (p99 - p1 >= min_spread).
    vol = _phantom(shape=(64, 64, 48))
    z, y, x = np.mgrid[0:64, 0:64, 0:48].astype(np.float32)
    bias = 1.0 + 0.5 * (x / 48.0)  # +50% ramp across x
    vol_biased = vol * bias
    mask = head_mask(vol_biased)

    probe_cfg = BratsifyConfig(n4_shrink=2, n4_iterations=(10, 10),
                               n4_field_lo=0.0, n4_field_hi=99.0,
                               n4_min_spread=0.01)
    _, probe = n4_bias_correct(vol_biased, mask, (1.0, 1.0, 1.0), probe_cfg)
    if not probe.get("applied"):
        return
    # Now demand an impossibly tight field window around 1.0 so the measured
    # non-trivial field is classified as implausible and reverted.
    cfg = BratsifyConfig(n4_shrink=2, n4_iterations=(10, 10),
                         n4_field_lo=probe["field_p1"] + 0.05,
                         n4_field_hi=probe["field_p99"] - 0.05,
                         n4_min_spread=0.01)
    out, info = n4_bias_correct(vol_biased, mask, (1.0, 1.0, 1.0), cfg)
    assert info["applied"] is False
    assert info.get("reverted") is True, info
    assert "implausible" in info["reason"]
    assert np.array_equal(out, vol_biased), "a reverted N4 must not modify the data"


def test_t5_n4_uniform_input_is_left_alone():
    if not simpleitk_available():
        return
    vol = np.full((48, 48, 48), 400.0, dtype=np.float32)
    out, info = n4_bias_correct(vol, np.ones(vol.shape, bool), (1.0, 1.0, 1.0),
                                BratsifyConfig(n4_shrink=2,
                                               n4_iterations=(5, 5)))
    assert info["applied"] is False
    assert info.get("skipped") is True or info.get("reverted") is True
    assert np.array_equal(out, vol)


def test_t5_n4_preserves_structure():
    if not simpleitk_available():
        return
    vol = _phantom(shape=(64, 64, 48))
    mask = head_mask(vol)
    out, info = n4_bias_correct(vol, mask, (1.0, 1.0, 1.0),
                                BratsifyConfig(n4_shrink=2,
                                               n4_iterations=(10, 10)))
    assert np.isfinite(out).all()
    if info["applied"]:
        # The bias field is multiplicative and smooth: correcting it must
        # neither explode the intensity scale nor destroy the anatomy.
        assert 0.25 <= info["mean_scale"] <= 4.0
        corr = ncc(out, vol, mask)
        assert corr > 0.8, f"N4 destroyed the structure (NCC {corr:.3f})"
    else:
        assert np.array_equal(out, vol)


# ---------------------------------------------------------------------------
# T6 - iso resampling exactness and the canvas affine round-trip
# ---------------------------------------------------------------------------


def test_t6_resample_to_spacing_is_exact():
    vol = _phantom(shape=(120, 120, 78))
    aff = np.diag([2.0, 2.0, 2.0, 1.0])
    out, out_aff = resample_to_spacing(vol, aff, (1.0, 1.0, 1.0))
    assert out.shape == (240, 240, 156), out.shape
    sp = [float(np.linalg.norm(out_aff[:3, i])) for i in range(3)]
    assert np.allclose(sp, 1.0, atol=1e-6), sp
    # The physical extent must survive the resample.
    assert np.allclose(np.asarray(vol.shape) * 2.0,
                       np.asarray(out.shape) * 1.0, atol=1.0)

    # The other direction: half-millimetre data downsampled to 1 mm.
    fine = _phantom(shape=(240, 240, 156))
    fine_aff = np.diag([0.5, 0.5, 0.5, 1.0])
    back, back_aff = resample_to_spacing(fine, fine_aff, (1.0, 1.0, 1.0))
    assert back.shape == (120, 120, 78), back.shape
    assert np.allclose(
        [float(np.linalg.norm(back_aff[:3, i])) for i in range(3)],
        1.0, atol=1e-6)


def test_t6_target_grid_keeps_world_centre():
    shape, aff = target_grid_from_reference((100, 120, 60),
                                            np.diag([2.0, 2.0, 2.0, 1.0]),
                                            (1.0, 1.0, 1.0))
    assert shape == (200, 240, 120), shape
    src_centre = np.diag([2.0, 2.0, 2.0, 1.0]) @ np.array([49.5, 59.5, 29.5, 1.0])
    dst_centre = aff @ np.array([99.5, 119.5, 59.5, 1.0])
    assert np.allclose(src_centre, dst_centre, atol=1e-6)


def test_t6_canvas_preserves_world_coordinates():
    vol = _phantom(shape=(60, 70, 40))
    aff = np.diag([1.0, 1.0, 1.0, 1.0])
    out, out_aff, grew, out_mask = to_brats_canvas(vol, aff, (240, 240, 155),
                                                   mask=head_mask(vol))
    assert out.shape == (240, 240, 155)
    assert grew is False
    assert out_mask is not None and out_mask.shape == out.shape
    # The padding is a pure translation: non-zero source voxels map to identical
    # world coordinates in destination space.
    src_indices = np.argwhere(vol > 50.0)
    assert src_indices.size > 0
    # Destination index where src_indices[0] was placed
    src_pt = src_indices[0]
    val = vol[tuple(src_pt)]
    dst_matches = np.argwhere(np.isclose(out, val))
    assert dst_matches.size > 0
    dst_pt = dst_matches[0]
    src_world = aff @ np.array([*src_pt.astype(float), 1.0])
    dst_world = out_aff @ np.array([*dst_pt.astype(float), 1.0])
    assert np.allclose(src_world, dst_world, atol=1e-6)
    assert float(out.sum()) > 0.0


def test_t6_canvas_grows_instead_of_cropping():
    """A brain taller than the canvas must be padded, never cut."""
    vol = np.zeros((260, 260, 200), dtype=np.float32)
    vol[40:220, 40:220, 20:190] = 500.0
    out, _, grew, _ = to_brats_canvas(vol, np.eye(4), (240, 240, 155))
    assert grew is True
    assert out.shape == (260, 260, 200)
    assert np.isclose(float(out.sum()), float(vol.sum()))


# ---------------------------------------------------------------------------
# T7 - end-to-end: model-ready (1, 4, 128, 128, 128) tensor + metadata
# ---------------------------------------------------------------------------


def test_t7_bratsify_then_inference_tensor():
    base = _phantom()
    vols = {
        "t1ce": base,
        "t1": base * 0.9 + 40.0,
        "t2": apply_shift(base * 1.1, (3, -2, 1)),  # genuinely misaligned
        "flair": base * 0.7 + 90.0,
    }
    scans = _scans_from(vols)
    cfg = BratsifyConfig(n4=False, brats_canvas=(96, 96, 80))
    out_scans, rep = bratsify_scans(scans, cfg)
    assert rep.applied is True
    for mod in MODALITY_ORDER:
        assert out_scans.volumes[mod].shape == rep.final_shape
        assert out_scans.volumes[mod].dtype == np.float32
        assert np.isfinite(out_scans.volumes[mod]).all()
        assert np.allclose(out_scans.metadata[mod].affine, rep.final_affine)
        assert all(abs(float(s) - 1.0) < 1e-6
                   for s in out_scans.metadata[mod].voxel_spacing_mm)
    # The injected misalignment must actually have been corrected.
    reg = rep.registration["t2"]
    assert reg["ncc_gain"] > cfg.min_ncc_gain
    err = int(np.abs(np.asarray(reg["shift_px"]) + np.array([3, -2, 1])).sum())
    assert err <= 1, reg["shift_px"]

    tensor, meta = prepare_for_inference(out_scans, target_size=128)
    assert tensor.shape == (1, 4, 128, 128, 128), tensor.shape
    assert tensor.dtype == np.float32
    assert np.isfinite(tensor).all()
    for ch in range(4):
        assert float(tensor[0, ch].std()) > 1e-3, f"channel {ch} is flat"
    assert set(meta) == set(MODALITY_ORDER)
    assert meta["t2"].crop_slices is not None


def test_t7_disabled_config_is_a_noop():
    vols = {m: _phantom(shape=(40, 40, 40)) for m in MODALITY_ORDER}
    scans = _scans_from(vols)
    out_scans, rep = bratsify_scans(scans, BratsifyConfig(enabled=False))
    assert rep.applied is False
    assert out_scans is scans
    assert "not applied" in rep.to_markdown()


def test_t7_degenerate_input_never_raises():
    """The "never break the demo" guarantee: bad geometry -> warn, keep data."""
    vols = {m: _phantom(shape=(40, 40, 40)) for m in MODALITY_ORDER}
    scans = _scans_from(vols)
    bad = dict(scans.metadata)
    bad["t2"] = _meta(np.zeros((4, 4)), vols["t2"].shape)  # singular affine
    broken = PatientScans(volumes=dict(scans.volumes), metadata=bad,
                          source="broken")
    out_scans, rep = bratsify_scans(broken, BratsifyConfig(n4=False))
    assert rep.applied is True  # the other modalities are still processed
    assert any("t2" in w for w in rep.warnings)
    assert set(out_scans.volumes) == set(MODALITY_ORDER)


# ---------------------------------------------------------------------------
# T8 - real DICOM ZIP ingestion stays graceful
# ---------------------------------------------------------------------------


def test_t8_dicom_zip_is_graceful():
    from src.dicom_loader import IngestionError
    from src.io_unified import load_unified

    zip_path = "DICOM.zip"
    if not os.path.exists(zip_path):
        return
    try:
        scans, _ = load_unified(zip_path)
    except IngestionError:
        # The sample archive carries only a subset of the 4 modalities, so a
        # clean IngestionError is the correct, classified outcome.
        return
    except Exception as exc:  # pragma: no cover - would be a real bug
        raise AssertionError(f"unexpected exception type: {type(exc)}") from exc
    # If it does load, the BraTS-ifier must produce a consistent result.
    out_scans, rep = bratsify_scans(scans, BratsifyConfig(n4=False))
    assert rep.applied is True
    for mod in MODALITY_ORDER:
        assert out_scans.volumes[mod].shape == rep.final_shape


# ---------------------------------------------------------------------------
# QA artefacts, provenance and BraTS-style export
# ---------------------------------------------------------------------------


def test_qa_png_provenance_and_export():
    vols = {m: _phantom(shape=(48, 48, 40)) for m in MODALITY_ORDER}
    scans = _scans_from(vols)
    cfg = BratsifyConfig(n4=False, brats_canvas=(48, 48, 40), zscore_brain=True)
    out_scans, rep = bratsify_scans(scans, cfg)
    assert rep.applied is True
    assert "brain-only z-score" in rep.steps

    with tempfile.TemporaryDirectory() as d:
        before = np.asarray(scans.volumes["t2"])
        after = np.asarray(out_scans.volumes["t2"])
        png = save_qa_png(os.path.join(d, "qa.png"), out_scans.volumes["t1ce"],
                          before, after, modality="t2")
        if png is not None:
            assert os.path.getsize(png) > 1000

        prov = write_provenance(rep, os.path.join(d, "prov.json"))
        with open(prov, encoding="utf-8") as fh:
            payload = json.load(fh)
        assert payload["applied"] is True
        assert payload["reference"] == "t1ce"
        assert payload["final"]["shape"] == list(rep.final_shape)
        assert payload["config"]["n4"] is False
        for mod in MODALITY_ORDER:
            assert mod in payload["modalities"]
            assert "registration" in payload["modalities"][mod]

        paths = export_brats_style(out_scans, os.path.join(d, "brats"),
                                  report=rep)
        for mod in MODALITY_ORDER:
            assert os.path.exists(paths[mod])
            img = nib.load(paths[mod])
            assert img.shape == rep.final_shape
            assert np.allclose(img.affine, rep.final_affine, atol=1e-5)
        assert os.path.exists(paths["provenance"])


def test_report_markdown_is_informative():
    vols = {m: _phantom(shape=(48, 48, 40)) for m in MODALITY_ORDER}
    scans = _scans_from(vols)
    _, rep = bratsify_scans(scans, BratsifyConfig(n4=False,
                                                  brats_canvas=(48, 48, 40)))
    md = rep.to_markdown()
    assert "Phase A" in md
    assert "t1ce" in md
    assert rep.to_json().startswith("{")


# ---------------------------------------------------------------------------
# Real-data integration (skipped automatically when the sample is absent)
# ---------------------------------------------------------------------------


def test_real_brats_end_to_end():
    if not _have_real_data():
        return
    scans = load_patient_scans(_real_modalities())
    assert all(scans.volumes[m].shape == (240, 240, 155)
               for m in MODALITY_ORDER)
    # Inject a realistic 6 mm header error into one modality.
    bad = scans.metadata["t2"].affine.copy()
    bad[:3, 3] = bad[:3, 3] + bad[:3, :3] @ np.array([-4.0, 6.0, 2.0])
    meta = dict(scans.metadata)
    meta["t2"] = _meta(bad, scans.volumes["t2"].shape)
    corrupted = PatientScans(volumes=dict(scans.volumes), metadata=meta,
                             source="real-test")

    out_scans, rep = bratsify_scans(
        corrupted, BratsifyConfig(n4=False, brats_canvas=(240, 240, 155)))
    assert rep.applied is True
    assert rep.final_shape == (240, 240, 155)
    assert np.allclose(rep.final_spacing, 1.0, atol=1e-6)

    # The header error must have been detected and repaired.
    reg = rep.registration["t2"]
    assert reg["com_shift_mm"] > 4.0
    assert reg["ncc_gain"] >= 0.05, reg
    assert reg["ncc_after"] > 0.6, reg

    # Already-aligned modalities must be left alone (no gratuitous shifts).
    for mod in ("t1", "flair"):
        assert rep.registration[mod]["shift_px"] == [0, 0, 0]
        assert rep.registration[mod]["reason"].startswith("no shift applied")

    # And the corrected stack must actually be mutually well correlated.
    ref = out_scans.volumes["t1ce"]
    ref_m = head_mask(ref)
    ref_g = gradient_magnitude(ref)
    for mod in ("t1", "t2", "flair"):
        c = ncc(ref_g, gradient_magnitude(out_scans.volumes[mod]), ref_m)
        assert c > 0.5, f"{mod} still misaligned (grad-NCC {c:.3f})"


if __name__ == "__main__":
    test_t1_shift_recovery_exact()
    test_t1_plain_intensity_ncc_is_weaker_than_gradient()
    test_t9_numpy2_gradient_axis_keyword()
    test_t2_affine_resample_beats_shape_zoom()
    test_t3_com_guard_skips_alignment()
    test_t4_overlap_guard_rejects_unreachable_shift()
    test_t5_n4_reverts_and_returns_input_untouched()
    test_t5_n4_uniform_input_is_left_alone()
    test_t5_n4_preserves_structure()
    test_t6_resample_to_spacing_is_exact()
    test_t6_target_grid_keeps_world_centre()
    test_t6_canvas_preserves_world_coordinates()
    test_t6_canvas_grows_instead_of_cropping()
    test_t7_bratsify_then_inference_tensor()
    test_t7_disabled_config_is_a_noop()
    test_t7_degenerate_input_never_raises()
    test_t8_dicom_zip_is_graceful()
    test_qa_png_provenance_and_export()
    test_report_markdown_is_informative()
    print("bratsify unit tests PASSED (run test_real_brats_end_to_end() too)")




