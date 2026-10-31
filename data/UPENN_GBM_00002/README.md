# UPENN-GBM-00002_11 demo case

One coherent pre-operative glioblastoma case for the segmentation demo:

- `T1.nii.gz`
- `T1ce.nii.gz`
- `T2.nii.gz`
- `FLAIR.nii.gz`
- `tumor_segmentation_reference.nii.gz` — reference only; never shown as a model prediction.

Source: the public UPENN-GBM NIfTI release, derived from the TCIA UPENN-GBM
collection (CC BY 4.0). The images are skull-stripped and co-registered.

This release does **not** include patient-specific vessel masks, tract masks,
or functional-cortex maps. Do not substitute data from another patient for
those surgical planning layers.
