"""Build 1 DICOM-standard test patient from BraTS NIfTI (test-only, synthetic headers).

Reads the 4 modality volumes for one sample patient, downsamples to a small
but geometrically valid axial stack, and writes real DICOM Part-10 files
(one SeriesInstanceUID per modality) with hospital-style SeriesDescriptions.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/make_synth_dicom_patient.py \
        --src sample_brats_patient/BraTS2021_00495 \
        --out /tmp/synth_dicom_patient --rows 64 --cols 64 --step 5
"""
from __future__ import annotations

import argparse
import os
import shutil

import nibabel as nib
import numpy as np


DESC = {"t1": "AX T1", "t1ce": "AX T1C", "t2": "AX T2", "flair": "AX FLAIR"}
FILES = {"t1": "BraTS2021_00495_t1.nii/00000209_brain_t1.nii",
         "t1ce": "BraTS2021_00495_t1ce.nii/00000209_brain_t1ce.nii",
         "t2": "BraTS2021_00495_t2.nii/00000209_brain_t2.nii",
         "flair": "BraTS2021_00495_flair.nii/00000209_brain_flair.nii"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="sample_brats_patient/BraTS2021_00495")
    ap.add_argument("--out", default="/tmp/synth_dicom_patient")
    ap.add_argument("--rows", type=int, default=64)
    ap.add_argument("--cols", type=int, default=64)
    ap.add_argument("--step", type=int, default=5,
                    help="Take every Nth axial slice; z-spacing becomes step*1mm.")
    args = ap.parse_args()

    import pydicom
    from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
    from pydicom.uid import MRImageStorage, ExplicitVRLittleEndian, generate_uid
    from scipy.ndimage import zoom

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    study_uid = generate_uid()
    frame_uid = generate_uid()
    z_spacing = float(args.step) * 1.0
    n_written = {}
    for mod, rel in FILES.items():
        img = nib.load(os.path.join(args.src, rel))
        vol = np.asarray(img.dataobj, dtype=np.float32)  # (240,240,155)
        idx = list(range(vol.shape[2] // 4, 3 * vol.shape[2] // 4, args.step))
        small = zoom(vol[:, :, idx], (args.rows / vol.shape[0],
                                      args.cols / vol.shape[1], 1), order=1)
        small = small.transpose(2, 0, 1)  # (slices, rows, cols)
        vmax = float(small.max()) or 1.0
        pixels = (small / vmax * 1000).astype(np.uint16)
        series_uid = generate_uid()
        sdir = os.path.join(args.out, mod)
        os.makedirs(sdir, exist_ok=True)
        for i in range(pixels.shape[0]):
            meta = FileMetaDataset()
            meta.MediaStorageSOPClassUID = MRImageStorage
            meta.MediaStorageSOPInstanceUID = generate_uid()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.ImplementationClassUID = generate_uid()
            ds = FileDataset(f"slice_{i:03d}.dcm", {}, file_meta=meta,
                             preamble=b"\x00" * 128)
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
            ds.Rows, ds.Columns = args.rows, args.cols
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
    print(f"Wrote synthetic DICOM patient -> {args.out}: {n_written}")


if __name__ == "__main__":
    main()
