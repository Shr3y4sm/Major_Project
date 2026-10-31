#!/usr/bin/env python3
"""Download actual DICOM brain tumor data from TCIA (The Cancer Imaging Archive).

Downloads real clinical DICOM series for a glioma/glioblastoma patient,
organized by modality into subdirectories compatible with the DICOM loader.

Usage:
    python data/DICOM/download_dicom.py

Collections:
    TCGA-GBM   - Glioblastoma Multiforme (aggressive brain tumor)
    TCGA-LGG   - Lower Grade Glioma
    UPENN-GBM  - University of Pennsylvania GBM

Output:
    data/DICOM/real_patient/
        ├── t1/        (SeriesDescription contains "T1" but not "T1C"/"T1CE")
        ├── t1ce/      (SeriesDescription contains "T1C" or "T1CE")
        ├── t2/        (SeriesDescription contains "T2")
        └── flair/     (SeriesDescription contains "FLAIR")
"""

from __future__ import annotations

import os
import re
import sys
import zipfile
import io

try:
    import requests
except ImportError:
    print("Missing dependency: requests")
    print("Install with: pip install requests")
    sys.exit(1)

# TCIA API base URL
TCIA_BASE = "https://services.cancerimagingarchive.net/services/v4/TCIA/query"

# Collection to use (TCGA-GBM has glioblastoma - aggressive brain tumor)
COLLECTION = "TCGA-GBM"

# Modality token mapping for SeriesDescription classification
MODALITY_TOKENS = {
    "t1ce": ["t1ce", "t1c", "t1 post", "t1 contrast", "t1-gad", "t1gad",
             "postgad", "post-gad", "post gad", "gad"],
    "t1": ["t1", "t1w", "t1 weighted", "t1w_ffe", "t1w ffe", "mprage",
           "t1 ffe", "t1ffe"],
    "t2": ["t2", "t2w", "t2 weighted", "t2w_tse", "t2w tse", "t2 tse", "t2tse"],
    "flair": ["flair", "t2 flair", "t2flair"],
}

SCOUT_TOKENS = ("survey", "scout", "localizer", "loc", "topogram",
                "plan", "calibration", "mpr", "mst", "pilot")

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "real_patient",
)


def classify_modality(series_desc: str) -> str | None:
    """Classify a DICOM series by its SeriesDescription."""
    desc_lower = series_desc.lower()

    # Skip scouts/localizers
    if any(tok in desc_lower for tok in SCOUT_TOKENS):
        return None

    # Check t1ce first (before t1, due to substring matching)
    for mod in ("t1ce", "t1", "t2", "flair"):
        for tok in MODALITY_TOKENS[mod]:
            if tok.lower() in desc_lower:
                return mod
    return None


def get_series(collection: str = COLLECTION) -> list[dict]:
    """Query TCIA for all series in a collection."""
    url = f"{TCIA_BASE}/getSeries"
    params = {"CollectionName": collection, "format": "json"}
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def download_series(series_instance_uid: str) -> bytes:
    """Download a DICOM series ZIP from TCIA."""
    url = f"{TCIA_BASE}/getImage"
    params = {"SeriesInstanceUID": series_instance_uid}
    resp = requests.get(url, params=params, timeout=300)
    resp.raise_for_status()
    return resp.content


def extract_dicom_from_zip(zip_bytes: bytes, target_dir: str) -> int:
    """Extract DICOM files from a ZIP archive into target_dir."""
    os.makedirs(target_dir, exist_ok=True)
    count = 0
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for name in zf.namelist():
            if name.lower().endswith((".dcm", ".dicom", ".ima")):
                zf.extract(name, target_dir)
                count += 1
    return count


def main() -> None:
    print(f"Querying TCIA collection: {COLLECTION}")
    series_list = get_series(COLLECTION)
    print(f"Found {len(series_list)} series")

    # Filter to MR modality only, exclude scouts
    mr_series = []
    for s in series_list:
        mod = s.get("Modality", "")
        desc = s.get("SeriesDescription", "")
        if mod != "MR":
            continue
        cls = classify_modality(desc)
        if cls is None:
            continue
        s["_classified_modality"] = cls
        mr_series.append(s)

    print(f"MR series (non-scout): {len(mr_series)}")

    # Group by classified modality
    by_modality: dict[str, list[dict]] = {}
    for s in mr_series:
        cls = s["_classified_modality"]
        by_modality.setdefault(cls, []).append(s)

    for mod, series in by_modality.items():
        print(f"  {mod}: {len(series)} series")

    # Select one series per modality (prefer the first valid one)
    selected: dict[str, dict] = {}
    for mod in ("t1", "t1ce", "t2", "flair"):
        if mod in by_modality and by_modality[mod]:
            selected[mod] = by_modality[mod][0]

    if len(selected) < 4:
        missing = set(("t1", "t1ce", "t2", "flair")) - set(selected)
        print(f"\nWARNING: Missing modalities: {missing}")
        print("Try a different collection or increase the query limit.")
        if not selected:
            print("No series downloaded.")
            return

    print(f"\nSelected {len(selected)} modalities for download")

    # Download and organize
    os.makedirs(OUT_DIR, exist_ok=True)
    for mod, s in selected.items():
        uid = s["SeriesInstanceUID"]
        desc = s["SeriesDescription"]
        print(f"\nDownloading {mod} (SeriesInstanceUID={uid})")
        print(f"  SeriesDescription: {desc}")

        try:
            zip_bytes = download_series(uid)
            target_dir = os.path.join(OUT_DIR, mod)
            n = extract_dicom_from_zip(zip_bytes, target_dir)
            print(f"  Extracted {n} DICOM files -> {target_dir}")
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print(f"\nDICOM patient data -> {OUT_DIR}")
    print("You can now run: load_patient_scans('data/DICOM/real_patient')")


if __name__ == "__main__":
    main()
