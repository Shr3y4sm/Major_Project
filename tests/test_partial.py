"""Tests for partial-data Tier 2 (asymmetry heat + ellipsoid estimate)."""

from __future__ import annotations

import numpy as np

from src.partial_analysis import (
    asymmetry_heatmap,
    blend_heat_overlay,
    ellipsoid_estimate,
)


def _symmetric_blob(shape=(20, 24, 26)):
    rng = np.random.RandomState(0)
    base = rng.rand(*shape).astype(np.float32) * 10 + 50
    # symmetric anatomy: mirror-average so heat ~ 0
    base = (base + base[:, :, ::-1]) / 2
    return base


def test_heat_zero_on_symmetric():
    vol = _symmetric_blob()
    heat = asymmetry_heatmap(vol)
    assert heat.shape == vol.shape
    assert float(heat.max()) < 0.05


def test_heat_fires_on_asymmetric_lesion():
    vol = _symmetric_blob()
    vol[8:13, 10:15, 16:21] += 80  # right-side only "lesion"
    heat = asymmetry_heatmap(vol)
    assert float(heat.max()) > 0.5
    est = ellipsoid_estimate(heat, (1.0, 1.0, 1.0), percentile=98.0)
    assert est["found"] is True
    assert est["volume_cm3"] > 0
    assert est["d1_mm"] > 0 and est["d2_mm"] > 0 and est["d3_mm"] > 0


def test_estimate_empty_when_normal():
    vol = np.zeros((20, 24, 26), dtype=np.float32)
    heat = np.zeros_like(vol)
    est = ellipsoid_estimate(heat, (5.0, 1.0, 1.0), percentile=98.0,
                             min_voxels=10)
    assert est["found"] is False


def test_estimate_rejected_when_diffuse():
    rng = np.random.RandomState(1)
    heat = rng.rand(20, 24, 26).astype(np.float32) * 0.2 + 0.8
    heat[8:13, 10:15, 11:16] = 5.0  # focal hot block inside global warmth
    # low percentile -> huge mask spanning the volume -> must be rejected
    est = ellipsoid_estimate(heat, (1.0, 1.0, 1.0), percentile=10.0)
    assert est["found"] is False
    assert "diffuse" in est["note"]


def test_blend_overlay_rgb():
    vol = _symmetric_blob()
    heat = asymmetry_heatmap(vol)
    rgb = blend_heat_overlay(vol[10], heat[10])
    assert rgb.shape == (vol.shape[1], vol.shape[2], 3)
    assert rgb.dtype == np.uint8


if __name__ == "__main__":
    test_heat_zero_on_symmetric()
    test_heat_fires_on_asymmetric_lesion()
    test_estimate_empty_when_normal()
    test_estimate_rejected_when_diffuse()
    test_blend_overlay_rgb()
    print("partial-data tests PASSED")
