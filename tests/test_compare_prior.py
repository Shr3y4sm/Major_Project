"""Tests for longitudinal prior comparison (RANO-style delta).

Feature #1: given a same-space prior segmentation (or a synthesized demo
prior), report per-class volume deltas and a RANO-style response call from
the change in whole-tumor SPD (Progressive >= +25%, Partial response
<= -50%, otherwise Stable). No registration is performed — a mismatched
prior shape is reported as a note, never silently resampled.
"""

from __future__ import annotations

import numpy as np

import app as app_mod
from src.dicom_loader import SpatialMetadata
from src.postprocess import rano_classification


def _meta(spacing=(1.0, 1.0, 1.0), shape=(20, 20, 20)) -> dict:
    return {"t2": SpatialMetadata(
        affine=np.eye(4), voxel_spacing_mm=spacing,
        original_shape=shape, cropped_shape=shape,
        crop_slices=(slice(0, shape[0]),) * 3)}


# ---------------------------------------------------------------------------
# rano_classification thresholds
# ---------------------------------------------------------------------------

def test_rano_classification_thresholds():
    # +30% increase -> Progressive
    assert rano_classification(130.0, 100.0) == ("Progressive", 30.0)
    # exactly +25% boundary -> Progressive (>= 25)
    assert rano_classification(125.0, 100.0)[0] == "Progressive"
    # -60% decrease -> Partial response
    assert rano_classification(40.0, 100.0) == ("Partial response", -60.0)
    # exactly -50% boundary -> Partial response (<= -50)
    assert rano_classification(50.0, 100.0)[0] == "Partial response"
    # -20% -> Stable
    assert rano_classification(80.0, 100.0) == ("Stable", -20.0)
    # +10% -> Stable
    assert rano_classification(110.0, 100.0)[0] == "Stable"
    print("PASS: test_rano_classification_thresholds")


def test_rano_classification_zero_prior_guard():
    # prior has no measurable lesion, current does -> "New"
    call, pct = rano_classification(50.0, 0.0)
    assert call == "New / no prior measurable lesion" and pct is None
    # both empty -> Stable, 0%
    assert rano_classification(0.0, 0.0) == ("Stable", 0.0)
    print("PASS: test_rano_classification_zero_prior_guard")


# ---------------------------------------------------------------------------
# _compare_to_prior
# ---------------------------------------------------------------------------

def test_compare_prior_shape_mismatch_is_a_note():
    cur = np.zeros((20, 20, 20), np.int16)
    cur[5:10, 5:10, 5:10] = 2
    prior = np.zeros((19, 20, 20), np.int16)  # mismatched D
    meta = _meta()
    vols = app_mod._compute_volumes(cur, meta)
    cmp = app_mod._compare_to_prior(cur, prior, meta, (1, 1, 1), vols)
    assert cmp["ok"] is False
    assert "does not match" in cmp["note"]
    # markdown must contain the warning, not a table
    md = app_mod._comparison_markdown(cmp)
    assert "does not match" in md and "|" not in md
    print("PASS: test_compare_prior_shape_mismatch_is_a_note")


def test_compare_prior_delta_math_exact():
    # Spacing 1 mm => 1 voxel = 1e-3 cm^3.
    cur = np.zeros((20, 20, 20), np.int16)
    cur[2:7, 2:7, 2:7] = 1        # 125 necrotic (current)
    cur[8:14, 8:14, 8:14] = 2     # 216 edema (current)
    prior = np.zeros((20, 20, 20), np.int16)
    prior[2:4, 2:4, 2:4] = 1      # 8 necrotic (prior) -> +117
    prior[8:18, 8:18, 8:18] = 2   # 1000 edema (prior) -> -784
    meta = _meta()
    cur_vols = app_mod._compute_volumes(cur, meta)
    cmp = app_mod._compare_to_prior(cur, prior, meta, (1, 1, 1), cur_vols)
    assert cmp["ok"] is True
    rows = {r[0]: r for r in cmp["rows"]}
    p, c, d, pct = rows["Necrotic Core"][1:5]
    assert abs(p - 8 * 1e-3) < 1e-9
    assert abs(c - 125 * 1e-3) < 1e-9
    assert abs(d - 117 * 1e-3) < 1e-9
    assert abs(pct - (117.0 / 8.0 * 100.0)) < 1e-6
    # Edema shrank: prior 1.0, current 0.216, delta -0.784
    pe = rows["Peritumoral Edema"]
    assert abs(pe[1] - 1000 * 1e-3) < 1e-9
    assert abs(pe[2] - 216 * 1e-3) < 1e-9
    assert abs(pe[3] - (-784 * 1e-3)) < 1e-9
    print("PASS: test_compare_prior_delta_math_exact")

def test_compare_prior_markdown_and_html_contain_call():
    cur = np.zeros((30, 30, 30), np.int16)
    cur[10:25, 10:25, 10:25] = 2  # big current tumour
    prior = np.zeros((30, 30, 30), np.int16)
    prior[13:22, 13:22, 13:22] = 2  # smaller prior -> current grew
    meta = _meta(shape=(30, 30, 30))
    vols = app_mod._compute_volumes(cur, meta)
    cmp = app_mod._compare_to_prior(cur, prior, meta, (1, 1, 1), vols,
                                    synthetic=True)
    assert cmp["ok"] is True
    md = app_mod._comparison_markdown(cmp)
    assert "Comparison to prior study" in md
    assert "RANO-style response" in md
    assert "Synthetic demo prior" in md  # synthetic disclaimer
    html = app_mod._comparison_html(cmp)
    assert "RANO-style response" in html
    assert "Synthetic demo prior" in html
    print("PASS: test_compare_prior_markdown_and_html_contain_call")


def test_compare_prior_no_prior_returns_empty():
    cur = np.zeros((20, 20, 20), np.int16)
    cur[5:10, 5:10, 5:10] = 2
    meta = _meta()
    vols = app_mod._compute_volumes(cur, meta)
    cmp = app_mod._compare_to_prior(cur, None, meta, (1, 1, 1), vols)
    assert cmp["ok"] is False and cmp["note"] == "no prior supplied"
    # Nothing rendered in markdown or HTML when no prior.
    assert app_mod._comparison_markdown(cmp) == ""
    assert app_mod._comparison_html(cmp) == ""
    print("PASS: test_compare_prior_no_prior_returns_empty")


# ---------------------------------------------------------------------------
# Synthetic demo prior
# ---------------------------------------------------------------------------

def test_synthesize_demo_prior_direction_and_shape():
    cur = np.zeros((30, 30, 30), np.int16)
    cur[10:22, 10:22, 10:22] = 2

    reg = app_mod._synthesize_demo_prior(cur, "Regression (tumour shrank)")
    pro = app_mod._synthesize_demo_prior(cur, "Progression (tumour grew)")
    off = app_mod._synthesize_demo_prior(cur, "Off")

    assert reg.shape == cur.shape and pro.shape == cur.shape
    assert off is None

    def wt(seg):
        return int(np.isin(seg, [1, 2, 3, 4]).sum())

    # Regression: prior is LARGER than current -> current reads as shrinkage.
    assert wt(reg) > wt(cur)
    # Progression: prior is SMALLER than current -> current reads as growth.
    assert wt(pro) < wt(cur)
    # Label 4 never left behind.
    assert not np.any(reg == 4) and not np.any(pro == 4)
    print("PASS: test_synthesize_demo_prior_direction_and_shape")


def test_demo_prior_end_to_end_regression_reads_shrink():
    """Regression demo: current smaller than prior => WT decreases."""
    cur = np.zeros((30, 30, 30), np.int16)
    cur[12:20, 12:20, 12:20] = 2
    prior = app_mod._synthesize_demo_prior(cur, "Regression (tumour shrank)")
    meta = _meta(shape=(30, 30, 30))
    vols = app_mod._compute_volumes(cur, meta)
    cmp = app_mod._compare_to_prior(cur, prior, meta, (1, 1, 1), vols,
                                    synthetic=True)
    assert cmp["ok"] is True
    wt_row = [r for r in cmp["rows"] if r[0] == "Total Whole Tumor"][0]
    # current WT < prior WT
    assert wt_row[2] < wt_row[1]
    print("PASS: test_demo_prior_end_to_end_regression_reads_shrink")


# ---------------------------------------------------------------------------
# write_html_report backward compatibility
# ---------------------------------------------------------------------------

def test_write_html_report_with_and_without_comparison():
    import tempfile
    from src.report import write_html_report
    vols = {"Necrotic Core": 1.0, "Peritumoral Edema": 2.0,
            "Enhancing Tumor": 3.0, "Total Whole Tumor": 6.0}
    rano = {"dmax_mm": 10, "dperp_mm": 5, "slice": 7}

    d = tempfile.mkdtemp(prefix="cmp_report_")
    # Without comparison kwarg (regression guard) — must still work.
    p0 = write_html_report(f"{d}/r0.html", vols, rano, [], "notes", "test")
    with open(p0) as f:
        html0 = f.read()
    assert "Comparison to prior" not in html0

    # With comparison — section present.
    p1 = write_html_report(f"{d}/r1.html", vols, rano, [], "notes", "test",
                           comparison="<p>delta</p>")
    with open(p1) as f:
        html1 = f.read()
    assert "Comparison to prior study" in html1 and "delta" in html1
    print("PASS: test_write_html_report_with_and_without_comparison")


if __name__ == "__main__":
    test_rano_classification_thresholds()
    test_rano_classification_zero_prior_guard()
    test_compare_prior_shape_mismatch_is_a_note()
    test_compare_prior_delta_math_exact()
    test_compare_prior_markdown_and_html_contain_call()
    test_compare_prior_no_prior_returns_empty()
    test_synthesize_demo_prior_direction_and_shape()
    test_demo_prior_end_to_end_regression_reads_shrink()
    test_write_html_report_with_and_without_comparison()
    print("\ncompare-prior tests PASSED")

