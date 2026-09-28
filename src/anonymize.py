"""PHI anonymization for DICOM (upload) and NIfTI (filenames)."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

__all__ = ["anonymize_dicom_tree", "anonymize_dicom_zip", "clean_display_name"]

# DICOM tags to REMOVE (PHI). Geometry + series description kept.
_REMOVE_KEYWORDS = [
    "PatientName", "PatientID", "PatientBirthDate", "PatientSex",
    "PatientAge", "PatientAddress", "PatientTelephoneNumbers",
    "InstitutionName", "InstitutionAddress", "ReferringPhysicianName",
    "PerformingPhysicianName", "OperatorsName", "PhysiciansOfRecord",
    "StudyID", "AccessionNumber",
]


def _hash_id(value: str, salt: str = "brats-mvp") -> str:
    return "ANON-" + hashlib.sha256(
        f"{salt}:{value}".encode()).hexdigest()[:12].upper()


def anonymize_dicom_tree(src_dir: str, dst_dir: str | None = None) -> str:
    """Copy DICOM tree to dst (or tmp) with PHI tags removed. Returns dst."""
    import pydicom

    if dst_dir is None:
        dst_dir = tempfile.mkdtemp(prefix="anon_dicom_")
    os.makedirs(dst_dir, exist_ok=True)
    for root, _, files in os.walk(src_dir):
        rel = os.path.relpath(root, src_dir)
        out_root = os.path.join(dst_dir, rel)
        os.makedirs(out_root, exist_ok=True)
        for f in files:
            src = os.path.join(root, f)
            dst = os.path.join(out_root, f)
            try:
                ds = pydicom.dcmread(src, force=True)
            except Exception:
                shutil.copy2(src, dst)  # non-DICOM: copy as-is
                continue
            for kw in _REMOVE_KEYWORDS:
                if kw in ds:
                    del ds[kw]
            # Drop group-0000 command elements: illegal in stored files and
            # present in some raw dumps (this real Zenodo ZIP has them).
            # Iterate over a snapshot since we delete during iteration.
            for tag in list(ds.keys()):
                try:
                    if int(tag.group) == 0:
                        del ds[tag]
                except Exception:
                    pass
            try:
                pid = str(ds.get("PatientID", "unknown"))
            except Exception:
                pid = "unknown"
            ds.PatientID = _hash_id(pid)
            ds.PatientName = "ANONYMIZED"
            ds.save_as(dst)
    return dst_dir


def anonymize_dicom_zip(zip_path: str) -> str:
    """Anonymize a DICOM ZIP -> new anonymized ZIP path (in tmp)."""
    tmp = tempfile.mkdtemp(prefix="anon_src_")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp)
    anon_dir = tempfile.mkdtemp(prefix="anon_out_")
    anonymize_dicom_tree(tmp, anon_dir)
    # Unique per-call name: the old fixed "anonymized_upload.zip" raced when
    # two users uploaded concurrently.
    fd, out_zip = tempfile.mkstemp(prefix="anonymized_upload_",
                                   suffix=".zip")
    os.close(fd)
    os.remove(out_zip)
    shutil.make_archive(out_zip.replace(".zip", ""), "zip", anon_dir)
    shutil.rmtree(tmp, ignore_errors=True)
    shutil.rmtree(anon_dir, ignore_errors=True)
    return out_zip


def clean_display_name(path: str) -> str:
    """Strip PHI-ish folder prefixes for UI display; keep filename."""
    return Path(path).name
