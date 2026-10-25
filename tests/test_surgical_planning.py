import json

import nibabel as nib
import numpy as np

from src.surgical_planning import build_preoperative_plan


def test_plan_maps_layer_and_reports_geometric_clearance(tmp_path):
    affine = np.eye(4)
    tumour = np.zeros((12, 12, 12), dtype=np.uint8)
    tumour[4:6, 4:6, 4:6] = 1
    vessel = np.zeros_like(tumour)
    vessel[8, 4:6, 4:6] = 1
    seg_path, vessel_path, report = (tmp_path / "seg.nii.gz", tmp_path / "vessel.nii.gz", tmp_path / "plan.json")
    nib.save(nib.Nifti1Image(tumour, affine), seg_path)
    nib.save(nib.Nifti1Image(vessel, affine), vessel_path)
    plan, masks, figure = build_preoperative_plan(str(seg_path), {"Vessels": str(vessel_path)}, str(report))
    vessel_review = next(item for item in plan.layers if item.name == "Vessels")
    assert vessel_review.supplied is True
    assert vessel_review.minimum_clearance_mm == 3.0
    assert "White-matter tracts" not in masks
    assert len(figure.data) == 2
    assert json.loads(report.read_text())["review_required"] is True
