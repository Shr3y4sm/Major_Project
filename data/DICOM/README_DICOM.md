# DICOM Brain Tumor Dataset

This directory contains tools and documentation for obtaining DICOM-format brain tumor data for the DICOM mode demo of the BraTS segmentation pipeline.

## Directory Structure

```
data/DICOM/
├── real_patient/               # Downloaded real DICOM patient (run download_dicom.py)
│   ├── t1/                     # T1-weighted series
│   ├── t1ce/                   # Post-contrast T1 series
│   ├── t2/                     # T2-weighted series
│   └── flair/                  # FLAIR series
├── synth_patient/              # Synthetic DICOM patient (generated from BraTS NIfTI)
│   ├── t1/
│   ├── t1ce/
│   ├── t2/
│   └── flair/
├── download_dicom.py           # Download real DICOM from TCIA
├── generate_dicom.py           # Generate synthetic DICOM from NIfTI
└── README_DICOM.md             # This file
```

## DICOM Modality Tokens

The DICOM loader (`src/dicom_loader.py`) infers modality from `SeriesDescription` using these tokens:

| Modality | SeriesDescription Tokens |
|----------|-------------------------|
| t1ce     | `t1ce`, `t1c`, `t1 post`, `t1 contrast`, `t1-gad`, `t1gad`, `postgad`, `post-gad`, `post gad`, `gad` |
| t1       | `t1`, `t1w`, `t1 weighted`, `t1w_ffe`, `t1w ffe`, `mprage`, `t1 ffe`, `t1ffe` |
| t2       | `t2`, `t2w`, `t2 weighted`, `t2w_tse`, `t2w tse`, `t2 tse`, `t2tse` |
| flair    | `flair`, `t2 flair`, `t2flair` |

**Important**: `t1ce` tokens must be evaluated before `t1` tokens because matching is substring-based.

## Option 1: Download Real DICOM from TCIA (Recommended for Demo)

Run the download script to fetch actual clinical DICOM from The Cancer Imaging Archive:

```bash
pip install requests
python data/DICOM/download_dicom.py
```

This queries TCIA's `TCGA-GBM` collection (glioblastoma patients), selects one series per modality (T1, T1ce, T2, FLAIR), and organizes them into `data/DICOM/real_patient/`.

### TCIA API Endpoints

| Endpoint | Purpose |
|----------|---------|
| `GET /query/getSeries?CollectionName=TCGA-GBM` | List all series in the collection |
| `GET /query/getImage?SeriesInstanceUID=<UID>` | Download a series as ZIP |

### Alternative Collections

Change `COLLECTION` in `download_dicom.py` to:
- `TCGA-LGG` — Lower Grade Glioma
- `UPENN-GBM` — University of Pennsylvania GBM
- `BRAIN-TUMOR-PROGRESSION` — Brain tumor progression
- `DICOM-GLIOMA-SEG` — Glioma segmentation

## Option 2: Generate Synthetic DICOM from BraTS NIfTI

```bash
pip install pydicom scipy nibabel
python data/DICOM/generate_dicom.py
```

Converts existing BraTS NIfTI files (`data/BraTS2021_00000/`) into synthetic DICOM Part-10 files in `data/DICOM/synth_patient/`.

## Using the DICOM Loader

```python
from src.dicom_loader import load_patient_scans, prepare_for_inference

# Load from downloaded DICOM directory
scans = load_patient_scans("data/DICOM/real_patient")

# Prepare for inference
tensor, metadata = prepare_for_inference(scans)
# tensor shape: (1, 4, 128, 128, 128) float32
```

## DICOM Modality Tokens

The DICOM loader (`src/dicom_loader.py`) infers modality from `SeriesDescription`:

| Modality | SeriesDescription Tokens |
|----------|-------------------------|
| t1ce | t1ce, t1c, t1 post, t1 contrast, t1-gad, t1gad, postgad, post-gad, post gad, gad |
| t1 | t1, t1w, t1 weighted, mprage, t1 ffe, t1ffe, etc. |
| t2 | t2, t2w, t2 weighted, t2 tse, etc. |
| flair | flair, t2 flair, t2flair |

**Important**: t1ce tokens are matched before t1 tokens (substring-based matching).

## Notes

- Scout/localizer series are automatically skipped (tokens: survey, scout, localizer, loc, topogram, plan, calibration, mpr, mst, pilot)
- All 4 modalities must be present for successful ingestion
- DICOM files are converted to temporary NIfTI via `dicom2nifti` internally
- TCGA-GBM contains glioblastoma (aggressive brain tumor) — ideal for the demo
