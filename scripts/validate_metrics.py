#!/usr/bin/env python3
"""
BraTS Segmentation Metrics Validation Script
Computes Dice, HD95 (95th percentile Hausdorff), Sensitivity, Specificity
for ET, TC, WT sub-regions following BraTS 2021 convention.

Usage:
    python scripts/validate_metrics.py --pred_dir ./predictions --gt_dir ./ground_truth --spacing 1.0 1.0 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.spatial.distance import cdist


def compute_dice(pred: np.ndarray, gt: np.ndarray, label) -> float:
    """Dice coefficient for a single class or list of classes."""
    if isinstance(label, list):
        pred_mask = np.isin(pred, label)
        gt_mask = np.isin(gt, label)
    else:
        pred_mask = pred == label
        gt_mask = gt == label
    intersection = int(np.sum(pred_mask & gt_mask))
    denom = int(np.sum(pred_mask) + np.sum(gt_mask))
    return 2.0 * intersection / (denom + 1e-8)


def compute_hausdorff95(
    pred: np.ndarray,
    gt: np.ndarray,
    label,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> float:
    """95th percentile Hausdorff distance in mm."""
    if isinstance(label, list):
        pred_mask = np.isin(pred, label)
        gt_mask = np.isin(gt, label)
    else:
        pred_mask = pred == label
        gt_mask = gt == label

    pred_coords = np.argwhere(pred_mask) * np.array(spacing)
    gt_coords = np.argwhere(gt_mask) * np.array(spacing)

    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return float("inf")

    dists = cdist(pred_coords, gt_coords)
    d1 = np.min(dists, axis=1)
    d2 = np.min(dists, axis=0)
    all_dists = np.concatenate([d1, d2])
    return float(np.percentile(all_dists, 95))


def compute_sensitivity_specificity(
    pred: np.ndarray, gt: np.ndarray, label
) -> tuple[float, float]:
    """Sensitivity and specificity for a single class or list of classes."""
    if isinstance(label, list):
        pred_mask = np.isin(pred, label)
        gt_mask = np.isin(gt, label)
    else:
        pred_mask = pred == label
        gt_mask = gt == label

    tp = int(np.sum(pred_mask & gt_mask))
    fp = int(np.sum(pred_mask & ~gt_mask))
    fn = int(np.sum(~pred_mask & gt_mask))
    tn = int(np.sum(~pred_mask & ~gt_mask))

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    return sensitivity, specificity


# BraTS label mapping (project convention):
#   0 = Background
#   1 = Necrotic / Non-enhancing Core
#   2 = Peritumoral Edema
#   3 = Enhancing Tumor
# Sub-regions:
#   ET = {3}          (project label 3 == BraTS label 4)
#   TC = {1, 3}       (necrotic + enhancing)
#   WT = {1, 2, 3}    (whole tumor)
REGIONS = {
    "ET": 3,
    "TC": [1, 3],
    "WT": [1, 2, 3],
}


def evaluate_case(
    pred_path: str,
    gt_path: str,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict:
    """Compute all metrics for a single prediction/GT pair."""
    pred = nib.load(pred_path).get_fdata().astype(np.uint8)
    gt = nib.load(gt_path).get_fdata().astype(np.uint8)

    results: dict = {}
    for region_name, label in REGIONS.items():
        results[region_name] = {
            "dice": compute_dice(pred, gt, label),
            "hd95": compute_hausdorff95(pred, gt, label, spacing),
        }
        sens, spec = compute_sensitivity_specificity(pred, gt, label)
        results[region_name]["sensitivity"] = sens
        results[region_name]["specificity"] = spec
    return results


def evaluate_dataset(
    pred_dir: str,
    gt_dir: str,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> dict:
    """Evaluate all cases in pred_dir against matching GT in gt_dir."""
    pred_files = sorted(Path(pred_dir).glob("*.nii*"))
    gt_files = sorted(Path(gt_dir).glob("*.nii*"))
    all_results: dict = {}

    for pred_file in pred_files:
        case_id = pred_file.stem.replace(".nii", "").replace(".gz", "")
        gt_file = next((f for f in gt_files if case_id in f.stem), None)
        if gt_file is None:
            print(f"Warning: No GT found for {case_id}")
            continue

        case_results = evaluate_case(str(pred_file), str(gt_file), spacing)
        all_results[case_id] = case_results
        print(
            f"{case_id}: Dice WT={case_results['WT']['dice']:.4f}, "
            f"TC={case_results['TC']['dice']:.4f}, "
            f"ET={case_results['ET']['dice']:.4f}"
        )

    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS (Mean ± SD)")
    print("=" * 60)
    for region in ["WT", "TC", "ET"]:
        dices = [r[region]["dice"] for r in all_results.values()]
        hds = [r[region]["hd95"] for r in all_results.values()
               if r[region]["hd95"] != float("inf")]
        sens = [r[region]["sensitivity"] for r in all_results.values()]
        specs = [r[region]["specificity"] for r in all_results.values()]

        print(f"\n{region}:")
        print(f"  Dice:     {np.mean(dices):.4f} ± {np.std(dices):.4f}")
        if hds:
            print(f"  HD95:     {np.mean(hds):.2f} ± {np.std(hds):.2f} mm")
        else:
            print(f"  HD95:     N/A")
        print(f"  Sens:     {np.mean(sens):.4f} ± {np.std(sens):.4f}")
        print(f"  Spec:     {np.mean(specs):.4f} ± {np.std(specs):.4f}")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate BraTS segmentation metrics"
    )
    parser.add_argument(
        "--pred_dir",
        required=True,
        help="Directory containing prediction NIfTI files",
    )
    parser.add_argument(
        "--gt_dir",
        required=True,
        help="Directory containing ground truth NIfTI files",
    )
    parser.add_argument(
        "--spacing",
        nargs=3,
        type=float,
        default=[1.0, 1.0, 1.0],
        help="Voxel spacing (x y z) in mm (default: 1.0 1.0 1.0)",
    )
    args = parser.parse_args()

    evaluate_dataset(args.pred_dir, args.gt_dir, tuple(args.spacing))