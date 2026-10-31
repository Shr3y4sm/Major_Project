# BraTS2021_00000 demo case

One coherent pre-operative glioblastoma case for the segmentation demo:

- `BraTS2021_00000_t1.nii.gz` — native T1-weighted
- `BraTS2021_00000_t1ce.nii.gz` — post-contrast T1-weighted (T1Gd)
- `BraTS2021_00000_t2.nii.gz` — T2-weighted
- `BraTS2021_00000_flair.nii.gz` — T2-FLAIR
- `BraTS2021_00000_seg.nii.gz` — reference segmentation (never shown as model prediction)

Source: BraTS 2021 training data via Hugging Face MedSeg-7D (MaybeRichard/MedSeg-7D), derived from the public BraTS 2021 challenge release (CC BY 4.0). All images are skull-stripped, co-registered, and resampled to 1 mm³ isotropic resolution.

Reference segmentation labels: 0 = background, 1 = necrotic core, 2 = edema, 4 = enhancing tumor (native BraTS convention; remapped to 0-3 internally).