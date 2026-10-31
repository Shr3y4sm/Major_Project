#!/usr/bin/env python3
"""Generate synthetic DICOM brain tumor dataset from existing BraTS NIfTI files.

Creates a DICOM directory structure with 4 modalities (t1, t1ce, t2, flair)
suitable for testing the DICOM loader pipeline.
"""
from __future__ import annotations

import os
import shutil
import sys

import nibabel as nib
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import MRImageStorage, ExplicitVRLittleEndian, generate_uid
    from scipy.ndimage import zoom
except ImportError as exc:
    print(f"Missing dependency: {exc}")
    print("Install with: pip install pydicom scipy nibabel")
    sys.exit(1)

DESC = {"t1": "AX T1", "t1ce": "AX T1C", "t2": "AX T2", "flair": "AX FLAIR"}

# Source BraTS patient NIfTI files
SRC_PATIENT = "/home/shreyas/Downloads/Major_Project/data/BraTS2021_00000"
FILES = {
    "t1": os.path.join(SRC_PATIENT, "BraTS2021_00000_t1.nii.gz"),
    "t1ce": os.path.join(SRC_PATIENT, "BraTS2021_00000_t1ce.nii.gz"),
    "t2": os.path.join(SRC_PATIENT, "BraTS2021_00000_t2.nii.gz"),
    "flair": os.path.join(SRC_PATIENT, "BraTS2021_00000_flair.nii.gz"),
}

OUT_DIR = "/home/shreyas/Downloads/Major_Project/data/DICOM/synth_patient"
ROWS = 64
COLS = 64
STEP = 5  # Take every Nth axial slice


def main() -> None:
    if os.path.isdir(OUT_DIR):
        shutil.rmtree(OUT_DIR)

    study_uid = generate_uid()
    frame_uid = generate_uid()
    z_spacing = float(STEP) * 1.0
    n_written: dict[str, int] = {}

    for mod, rel in FILES.items():
        if not os.path.exists(rel):
            print(f"WARNING: Source file not found: {rel}, skipping {mod}")
            continue

        img = nib.load(rel)
        vol = np.asarray(img.dataobj, dtype=np.float32)
        # Take middle slices for a reasonable volume
        idx = list(range(vol.shape[2] // 4, 3 * vol.shape[2] // 4, STEP))
        small = zoom(
            vol[:, :, idx],
            (ROWS / vol.shape[0], COLS / vol.shape[1], 1),
            order=1,
        )
        small = small.transpose(2, 0, 1)  # (slices, rows, cols)
        vmax = float(small.max()) or 1.0
        pixels = (small / vmax * 1000).astype(np.uint16)

        series_uid = generate_uid()
        sdir = os.path.join(OUT_DIR, mod)
        os.makedirs(sdir, exist_ok=True)

        for i in range(pixels.shape[0]):
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = MRImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.ImplementationClassUID = generate_uid()

            ds = FileDataset(
                f"slice_{i:03d}.dcm", {}, file_meta=meta, preamble=b"\x00" * 128
            )
            ds.SOPClassUID = MRImageStorage
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.StudyInstanceUID = study_uid
            ds.SeriesInstanceUID = series_uid
            ds.FrameOfReferenceUID = frame_uid
            ds.Modality = "MR"
            ds.SeriesDescription = DESC[mod]
            ds.ProtocolName = DESC[mod]
            ds.PatientName = "SYNTH^PATIENT"
            ds.PatientID = "SYNTH001"
            ds.InstanceNumber = i + 1
            ds.Rows, ds.Columns = ROWS, COLS
            ds.SamplesPerPixel = 1
            ds.PhotometricInterpretation = "MONOCHROME2"
            ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 16, 15
            ds.PixelRepresentation = 0
            ds.PixelSpacing = [1.0, 1.0]
            ds.SliceThickness = z_spacing
            ds.SpacingBetweenSlices = z_spacing
            ds.SliceLocation = float(i) * z_spacing
            ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
            ds.ImagePositionPatient = [0.0, 0.0, float(i) * z_spacing]
            ds.ImageType = ["ORIGINAL", "PRIMARY", "AXIAL"]
            ds.RepetitionTime, ds.EchoTime = 2000.0, 30.0
            ds.RescaleIntercept, ds.RescaleSlope = 0, 1
            ds.PixelData = pixels[i].tobytes()
            ds.is_little_endian, ds.is_implicit_VR = True, False
            ds.save_as(os.path.join(sdir, f"slice_{i:03d}.dcm"))

        n_written[mod] = pixels.shape[0]
        print(f"  {mod}: {pixels.shape[0]} slices -> {sdir}")

    print(f"\nSynthetic DICOM patient -> {OUT_DIR}: {n_written}")


if __name__ == "__main__":
    main()
