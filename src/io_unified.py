"""Unified hospital ingestion for BOTH DICOM and NIfTI (part 1/2)."""

from __future__ import annotations

import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
from scipy.ndimage import zoom

from src.dicom_loader import (
    MODALITY_ORDER,
    PatientScans,
    SpatialMetadata,
    load_patient_scans,
)

__all__ = [
    "EXPANDED_NIFTI_TOKENS",
    "infer_nifti_modality",
    "discover_nifti_files",
    "load_unified",
    "harmonize_shapes",
    "canonicalize_to_ras",
    "build_ingestion_report",
]

EXPANDED_NIFTI_TOKENS: dict[str, tuple[str, ...]] = {
    "t1ce": ("t1ce", "t1c", "t1-gad", "t1gad", "t1_gad", "t1post",
             "t1-post", "t1 post", "postgad", "post-gad", "gad",
             "t1 contrast", "t1contrast"),
    "t1": ("t1", "t1w", "t1_weighted", "t1-weighted", "mprage", "bravo"),
    "t2": ("t2", "t2w", "t2_weighted", "t2-weighted", "tse-t2", "frfse"),
    "flair": ("flair", "t2flair", "t2-flair", "t2 flair", "flair-t2"),
}

_SEG_HINT = re.compile(r"(seg|label|mask|parc)", re.IGNORECASE)


def _tok_re(tok: str) -> str:
    # Suffix boundary: allow end-of-string, dot, underscore, dash, space.
    # (Bare `($|[^a-z0-9])` on the right would reject `t2w.nii` because `.`
    # IS non-alnum — instead we must explicitly forbid only alnum suffix,
    # and also forbid `t1` matching inside `t1ce`/`t1c`.
    neg = r"(?!ce\b|c\b|(?<=[a-z0-9]))"
    if tok == "t1":
        return rf"(^|[^a-z0-9])t1(?!ce|c(?![a-z])|[a-z0-9])"
    return rf"(^|[^a-z0-9]){re.escape(tok)}(?![a-z0-9])"


def infer_nifti_modality(path: str) -> Optional[str]:
    """Infer modality from a NIfTI filename with expanded hospital tokens."""
    name = Path(path).name.lower()
    if "seg" in name and _SEG_HINT.search(name):
        return None
    for mod in ("t1ce", "t1", "t2", "flair"):  # order matters
        for tok in EXPANDED_NIFTI_TOKENS[mod]:
            if re.search(_tok_re(tok), name):
                return mod
    return None


def discover_nifti_files(root: str) -> dict[str, str]:
    """Recursively find one NIfTI per modality under ``root``."""
    from src.dicom_loader import MissingModalityError, IngestionError

    root_p = Path(root)
    # Handle files nested inside directories literally named `*.nii`
    # (BraTS portal / PACS exports): index those inner files first.
    nested: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        for dn in list(dirnames):
            if dn.lower().endswith((".nii", ".nii.gz")):
                inner = os.path.join(dirpath, dn)
                try:
                    inner_files = sorted(os.listdir(inner))
                except OSError:
                    continue
                for f in inner_files:
                    fp = os.path.join(inner, f)
                    if os.path.isfile(fp) and f.lower().endswith((".nii", ".nii.gz")):
                        nested.append(fp)
    cands = sorted(set(nested) | {
        str(p) for p in root_p.rglob("*.nii*")
        if p.is_file() and ".nii" in p.suffixes})
    mapping: dict[str, str] = {}
    ambiguous: list[str] = []
    for c in sorted(cands):
        mod = infer_nifti_modality(c)
        if mod is None:
            continue
        if mod in mapping:
            ambiguous.append(f"'{mod}' twice: '{mapping[mod]}' vs '{c}'")
            continue
        mapping[mod] = c
    missing = [m for m in MODALITY_ORDER if m not in mapping]
    if ambiguous:
        raise IngestionError(
            "Ambiguous NIfTI folder: " + "; ".join(ambiguous) + ". "
            "Rename files so each of t1/t1ce/t2/flair appears once."
        )
    if missing:
        found = sorted(mapping.keys()) or ["(none inferred)"]
        raise MissingModalityError(
            f"Missing NIfTI modality/modalities: {missing}. Found: {found} "
            f"under '{root}'. Tokens t1/t1ce/t2/flair "
            "(t1-gad/post/gad accepted for t1ce)."
        )
    return mapping

def _extract_zip_to_tmp(zip_path: str) -> str:
    tmp = tempfile.mkdtemp(prefix="unified_nifti_")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp)
    return tmp


def _looks_like_nifti_bundle(search_root: str) -> bool:
    found_dcm = False
    for root, dirs, files in os.walk(search_root):
        # Ignore our own conversion scratch dirs.
        dirs[:] = [d for d in dirs if not d.startswith("convert_")]
        for f in files:
            fl = f.lower()
            if fl.endswith(".nii") or fl.endswith(".nii.gz"):
                # Ignore NIfTIs produced by a previous conversion pass.
                if "convert_" in root:
                    continue
                # Ignore symlinked conversion inputs named *.nii that are
                # actually DICOM (dicom2nifti scratch uses symlinks).
                fp = os.path.join(root, f)
                try:
                    if os.path.islink(fp):
                        continue
                except OSError:
                    pass
                return True
            if fl.endswith((".dcm", ".dicom", ".ima")):
                found_dcm = True
        for dn in dirs:
            if dn.lower().endswith((".nii", ".nii.gz")):
                inner = os.path.join(root, dn)
                try:
                    for f in os.listdir(inner):
                        if f.lower().endswith((".nii", ".nii.gz")):
                            return True
                except OSError:
                    continue
    # A tree containing ONLY DICOM is never a NIfTI bundle, even if a
    # parent folder happens to be named `*.nii`.
    return False


def harmonize_shapes(scans: PatientScans, reference: str = "t1ce"):
    """Resample all modalities to reference shape (trilinear)."""
    warnings: list[str] = []
    ref_shape = scans.volumes[reference].shape
    if all(scans.volumes[m].shape == ref_shape for m in MODALITY_ORDER):
        return scans, warnings
    new_volumes: dict[str, np.ndarray] = {}
    new_meta: dict[str, SpatialMetadata] = {}
    for mod in MODALITY_ORDER:
        vol = scans.volumes[mod]
        if vol.shape != ref_shape:
            factors = [r / s for r, s in zip(ref_shape, vol.shape)]
            vol = zoom(vol, factors, order=1).astype(np.float32)
            warnings.append(
                f"Resampled '{mod}' {scans.volumes[mod].shape} -> "
                f"{ref_shape} (trilinear) to match '{reference}'."
            )
        new_volumes[mod] = vol
        orig = scans.metadata[mod]
        new_meta[mod] = SpatialMetadata(
            affine=orig.affine,
            voxel_spacing_mm=orig.voxel_spacing_mm,
            original_shape=tuple(int(s) for s in vol.shape),
            cropped_shape=tuple(int(s) for s in vol.shape),
            crop_slices=(slice(None), slice(None), slice(None)),
        )
    out = PatientScans(volumes=new_volumes, metadata=new_meta,
                       source=scans.source + f"|harmonized->{reference}")
    return out, warnings


def canonicalize_to_ras(scans: PatientScans):
    """Flip axes so every modality shares reference orientation signs."""
    warnings: list[str] = []
    ref_signs = np.sign(scans.metadata[MODALITY_ORDER[0]].affine[:3, :3])
    new_volumes: dict[str, np.ndarray] = {}
    new_meta: dict[str, SpatialMetadata] = {}
    for mod in MODALITY_ORDER:
        vol = scans.volumes[mod]
        aff = scans.metadata[mod].affine.copy()
        signs = np.sign(aff[:3, :3])
        flip_axes: list[int] = []
        for ax in range(3):
            ref_col = int(np.argmax(np.abs(ref_signs[:, ax])))
            cur_col = int(np.argmax(np.abs(signs[:, ax])))
            if cur_col == ref_col and signs[cur_col, ax] != ref_signs[ref_col, ax]:
                flip_axes.append(ax)
        if flip_axes:
            for ax in flip_axes:
                vol = np.flip(vol, axis=ax)
                n = vol.shape[ax]
                aff[:3, 3] = aff[:3, 3] + aff[:3, ax] * (n - 1)
                aff[:3, ax] = -aff[:3, ax]
            warnings.append(f"RAS-canonicalized '{mod}' (axes {flip_axes}).")
        new_volumes[mod] = np.ascontiguousarray(vol, dtype=np.float32)
        new_meta[mod] = SpatialMetadata(
            affine=aff,
            voxel_spacing_mm=tuple(
                float(np.linalg.norm(aff[:3, i])) for i in range(3)),
            original_shape=tuple(int(s) for s in vol.shape),
            cropped_shape=tuple(int(s) for s in vol.shape),
            crop_slices=(slice(None), slice(None), slice(None)),
        )
    out = PatientScans(volumes=new_volumes, metadata=new_meta,
                       source=scans.source + "|ras")
    return out, warnings


def build_ingestion_report(warnings: Sequence[str]) -> str:
    if not warnings:
        return "Ingestion: all 4 modalities validated, no resampling needed."
    return "Ingestion warnings:\n" + "\n".join(f"- {w}" for w in warnings)


def load_unified(input_source, harmonize: bool = True,
                 ras_canonical: bool = True, reference: str = "t1ce"):
    """One entry point for BOTH NIfTI and DICOM (dir / ZIP / dict / list)."""
    import shutil as _sh
    from src.dicom_loader import IngestionError

    tmp_dirs: list[str] = []
    try:
        if isinstance(input_source, str):
            if os.path.isdir(input_source):
                if _looks_like_nifti_bundle(input_source):
                    scans = load_patient_scans(
                        discover_nifti_files(input_source))
                else:
                    scans = load_patient_scans(input_source)
            elif zipfile.is_zipfile(input_source):
                tmp = _extract_zip_to_tmp(input_source)
                tmp_dirs.append(tmp)
                if _looks_like_nifti_bundle(tmp):
                    scans = load_patient_scans(discover_nifti_files(tmp))
                else:
                    scans = load_patient_scans(tmp)
            else:
                raise IngestionError(
                    f"Input '{input_source}' is not a dir or ZIP.")
        elif isinstance(input_source, dict):
            bad = [f"{k}={v}" for k, v in input_source.items()
                   if not os.path.exists(v)]
            if bad:
                raise IngestionError(f"NIfTI files not found: {bad}")
            scans = load_patient_scans(input_source)
        elif isinstance(input_source, (list, tuple)):
            if len(input_source) == 4 and all(isinstance(p, str)
                                              for p in input_source):
                mapping: dict[str, str] = {}
                for p in input_source:
                    mod = infer_nifti_modality(p)
                    if mod is not None and mod not in mapping:
                        mapping[mod] = p
                scans = (load_patient_scans(mapping) if len(mapping) == 4
                         else load_patient_scans(input_source))
            else:
                scans = load_patient_scans(input_source)
        else:
            raise TypeError(
                f"Unsupported type {type(input_source).__name__}. "
                "Expected dict / list / str.")
        warnings: list[str] = []
        if ras_canonical:
            try:
                scans, w = canonicalize_to_ras(scans)
                warnings.extend(w)
            except Exception as exc:
                warnings.append(f"RAS canonicalization skipped: {exc}")
        if harmonize:
            try:
                scans, w = harmonize_shapes(scans, reference=reference)
                warnings.extend(w)
            except Exception as exc:
                warnings.append(f"Shape harmonization skipped: {exc}")
        return scans, warnings
    finally:
        for d in tmp_dirs:
            _sh.rmtree(d, ignore_errors=True)

