"""
src/preprocessing.py
====================
Typed refactor of the Phase 1 preprocessing pipeline (BraTS 2021).

The logic is extracted verbatim from `braintumour-phase-1.ipynb` (Phase 1):
T2-based brain masking, bounding-box cropping, linear resampling to a
target size, and per-modality z-score normalization. No logic changes.
"""

from __future__ import annotations

import os
import re
import tarfile
from typing import Optional

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, zoom

LOCAL_ROOT = os.environ.get("BRATS_LOCAL_ROOT", "./BraTS2021_Training_Data")
TARGET_SIZE = 128

LABELS = {
    0: "Background",
    1: "Necrotic Core",
    2: "Edema",
    3: "Enhancing Tumor",  # Remapped from original BraTS label 4
}
MODALITIES = ["t1", "t1ce", "t2", "flair"]

__all__ = [
    "LOCAL_ROOT",
    "TARGET_SIZE",
    "LABELS",
    "MODALITIES",
    "is_kaggle",
    "find_kaggle_tar",
    "get_data_source",
    "get_tar_path",
    "get_local_root",
    "get_patient_ids",
    "load_patient_volume",
    "zscore_normalize",
    "brain_mask_from_t2",
    "crop_to_bbox",
    "resample_volume",
    "preprocess_patient",
]


def is_kaggle() -> bool:
    return os.path.exists("/kaggle/input")


def find_kaggle_tar() -> Optional[str]:
    root = "/kaggle/input"
    if not os.path.isdir(root):
        return None
    queue = [(root, 0)]
    while queue:
        cur, depth = queue.pop(0)
        if depth > 4:
            continue
        try:
            entries = sorted(os.listdir(cur))
        except OSError:
            continue
        for name in entries:
            if name.endswith(".tar") and "training" in name.lower():
                return os.path.join(cur, name)
            full = os.path.join(cur, name)
            if os.path.isdir(full) and depth < 4:
                queue.append((full, depth + 1))
    return None


def _discover_kaggle_patient_root(max_depth: int = 4) -> Optional[str]:
    root = "/kaggle/input"
    if not os.path.isdir(root):
        return None
    queue = [(root, 0)]
    while queue:
        cur, depth = queue.pop(0)
        if depth > max_depth:
            continue
        try:
            entries = sorted(os.listdir(cur))
        except OSError:
            continue
        patient_dirs = [d for d in entries if re.match(r"^BraTS2021_\d{5}$", d)]
        if patient_dirs:
            return cur
        for name in entries:
            full = os.path.join(cur, name)
            if os.path.isdir(full) and depth < max_depth:
                queue.append((full, depth + 1))
    return None


def get_data_source() -> tuple[Optional[str], Optional[str]]:
    if is_kaggle():
        root = _discover_kaggle_patient_root()
        if root:
            return None, root
        tar = find_kaggle_tar()
        if tar:
            return tar, None
        return None, None
    return None, get_local_root()


def get_tar_path() -> Optional[str]:
    tar, _ = get_data_source()
    return tar


def get_local_root() -> str:
    return LOCAL_ROOT


def _match_patient_dir(name: str) -> Optional[str]:
    m = re.search(r"BraTS2021_\d{5}", name)
    return m.group(0) if m else None


def get_patient_ids(
    tar_path: Optional[str] = None,
    local_root: Optional[str] = None,
) -> list[str]:
    if tar_path is not None and os.path.exists(tar_path):
        ids: list[str] = []
        with tarfile.open(tar_path, "r") as tar:
            seen: set[str] = set()
            for member in tar.getmembers():
                pid = _match_patient_dir(member.name)
                if pid and pid not in seen:
                    seen.add(pid)
                    ids.append(pid)
        return sorted(ids)

    if local_root is None:
        local_root = get_local_root()
    if os.path.isdir(local_root):
        return sorted(
            d for d in os.listdir(local_root)
            if re.match(r"^BraTS2021_\d{5}$", d)
        )

    raise FileNotFoundError("No dataset found.")


def _extract_patient_files(
    patient_id: str,
    tar_path: Optional[str] = None,
    local_root: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> dict[str, str]:
    if tar_path is not None and os.path.exists(tar_path):
        if work_dir is None:
            work_dir = os.path.join(os.getcwd(), ".brats_cache")
        os.makedirs(work_dir, exist_ok=True)
        patient_dir = os.path.join(work_dir, patient_id)
        if not os.path.exists(patient_dir):
            with tarfile.open(tar_path, "r") as tar:
                members = [
                    m for m in tar.getmembers()
                    if _match_patient_dir(m.name) == patient_id
                ]
                tar.extractall(path=work_dir, members=members)
        search_root = work_dir
    else:
        if local_root is None:
            local_root = get_local_root()
        search_root = local_root

    files: dict[str, str] = {}
    for root, _, filenames in os.walk(search_root):
        if patient_id in root or any(patient_id in f for f in filenames):
            for f in filenames:
                if patient_id in f and f.endswith(".nii.gz"):
                    for mod in MODALITIES:
                        if f"_{mod}.nii.gz" in f:
                            files[mod] = os.path.join(root, f)
                    if "_seg.nii.gz" in f:
                        files["seg"] = os.path.join(root, f)

    if len(files) < 5:
        case_path = os.path.join(search_root, patient_id, patient_id)
        for mod in MODALITIES:
            files[mod] = f"{case_path}_{mod}.nii.gz"
        files["seg"] = f"{case_path}_seg.nii.gz"

    return files


def load_patient_volume(
    patient_id: str,
    tar_path: Optional[str] = None,
    local_root: Optional[str] = None,
    work_dir: Optional[str] = None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    files = _extract_patient_files(patient_id, tar_path, local_root, work_dir)
    modalities: dict[str, np.ndarray] = {}
    for mod in MODALITIES:
        modalities[mod] = nib.load(files[mod]).get_fdata().astype(np.float32)
    seg = nib.load(files["seg"]).get_fdata().astype(np.int16)
    # Remap BraTS label 4 to contiguous label index 3
    seg[seg == 4] = 3
    return modalities, seg


def zscore_normalize(volume: np.ndarray) -> np.ndarray:
    mean, std = volume.mean(), volume.std()
    return (volume - mean) / std if std >= 1e-6 else np.zeros_like(volume)


def brain_mask_from_t2(
    t2: np.ndarray,
    threshold: float = 0.15,
    sigma: float = 1.0,
) -> np.ndarray:
    raw = (t2 - t2.min()) / (t2.max() - t2.min() + 1e-8)
    mask = gaussian_filter((raw > threshold).astype(np.float32), sigma=sigma)
    return (mask > 0.5).astype(np.float32)


def crop_to_bbox(
    volume: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, tuple[slice, slice, slice]]:
    idx = np.argwhere(mask > 0)
    if len(idx) == 0:
        return volume, (slice(None), slice(None), slice(None))
    zmin, ymin, xmin = idx.min(axis=0)
    zmax, ymax, xmax = idx.max(axis=0) + 1
    slc = (slice(zmin, zmax), slice(ymin, ymax), slice(xmin, xmax))
    return volume[slc], slc


def resample_volume(volume: np.ndarray, target_size: int = TARGET_SIZE) -> np.ndarray:
    factors = [target_size / s for s in volume.shape]
    return zoom(volume, factors, order=1)


def preprocess_patient(
    patient_id: str,
    tar_path: Optional[str] = None,
    local_root: Optional[str] = None,
    work_dir: Optional[str] = None,
    target_size: int = TARGET_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    modalities, seg = load_patient_volume(patient_id, tar_path, local_root, work_dir)
    mask = brain_mask_from_t2(modalities["t2"])
    cropped = {m: crop_to_bbox(modalities[m], mask)[0] for m in MODALITIES}
    seg_cropped, _ = crop_to_bbox(seg, mask)

    resampled = {m: resample_volume(cropped[m], target_size) for m in MODALITIES}
    seg_resampled = np.round(
        resample_volume(seg_cropped.astype(np.float32), target_size)
    ).astype(np.int16)
    stack = np.stack([zscore_normalize(resampled[m]) for m in MODALITIES], axis=0)
    return stack, seg_resampled