"""
src/dicom_loader.py
===================
Multi-modal data ingestion engine for the Phase 1 3D UNet.

Supports two input sources:
  * NIfTI files (``.nii`` / ``.nii.gz``) — a dict or list of 4 paths
    representing T1, T1ce, T2, and FLAIR modalities.
  * DICOM series — a root directory or ZIP containing multi-slice DICOM
    series for all 4 modalities, converted to temporary 3D volumes via
    ``dicom2nifti``.

The engine validates modality completeness, spatial orientation, slice
continuity, and uniform matrix dimensions, then integrates with the
Phase 1 preprocessing pipeline (T2 brain-mask crop, resampling to
(128, 128, 128), per-channel z-score normalization).

The output is a ``(1, 4, 128, 128, 128)`` float32 tensor ready for
``ONNXInferenceEngine.predict``, plus per-modality spatial metadata
(affine, voxel spacing, crop slices) required for exact physical
volume calculations (cm^3).
"""

from __future__ import annotations

import os
import re
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Union

import nibabel as nib
import numpy as np

from src.preprocessing import (
    MODALITIES,
    TARGET_SIZE,
    brain_mask_from_t2,
    crop_to_bbox,
    resample_volume,
    zscore_normalize,
)

__all__ = [
    "IngestionError",
    "MissingModalityError",
    "OrientationMismatchError",
    "SliceContinuityError",
    "ShapeMismatchError",
    "DICOMCorruptError",
    "SpatialMetadata",
    "PatientScans",
    "PlanningSeries",
    "inspect_surgical_planning_series",
    "load_patient_scans",
    "prepare_for_inference",
]

# Canonical modality order used for stacking (matches Phase 1).
MODALITY_ORDER: tuple[str, ...] = tuple(MODALITIES)

# DICOM series-description tokens used to infer modality.
# NOTE: `t1ce` MUST be evaluated before `t1` (and longer tokens before
# shorter ones), because matching is substring-based for DICOM metadata
# and "AX T1C" would otherwise be misclassified as `t1`.
_MODALITY_TOKENS: dict[str, tuple[str, ...]] = {
    "t1ce": ("t1ce", "t1c", "t1 post", "t1 contrast", "t1-gad", "t1gad",
             "postgad", "post-gad", "post gad", "gad"),
    "t1": ("t1", "t1w", "t1 weighted", "t1w_ffe", "t1w ffe", "mprage",
           "t1 ffe", "t1ffe"),
    "t2": ("t2", "t2w", "t2 weighted", "t2w_tse", "t2w tse", "t2 tse",
           "t2tse"),
    "flair": ("flair", "t2 flair", "t2flair"),
}

# Series descriptions that are scouts/localizers/surveys, not diagnostic
# modalities. These are SKIPPED (not errors) during DICOM ingestion.
_SCOUT_TOKENS = ("survey", "scout", "localizer", "loc", "topogram",
                 "plan", "calibration", "mpr", "mst", "pilot")

# DICOM file extensions (case-insensitive).
_DICOM_EXTS = {".dcm", ".dicom", ".ima"}

# These labels only identify likely planning inputs from DICOM metadata. They
# do not establish image quality, spatial registration, or clinical validity.
_PLANNING_SERIES_TOKENS: dict[str, tuple[str, ...]] = {
    "vascular": ("tof", "mra", "angiography", "angiogram", "cta", "ct angio"),
    "tractography": ("dti", "diffusion tensor", "tractography"),
    "functional": ("fmri", "bold", "functional", "language task", "motor task"),
    "bone": ("ct head", "head ct", "bone", "cranial ct"),
}


# ---------------------------------------------------------------------------
# Custom exceptions
# ---------------------------------------------------------------------------


class IngestionError(Exception):
    """Base class for all ingestion-related errors."""


class MissingModalityError(IngestionError):
    """Raised when one or more required modalities are absent."""


class OrientationMismatchError(IngestionError):
    """Raised when modalities do not share the same spatial orientation."""


class SliceContinuityError(IngestionError):
    """Raised when a DICOM series has non-continuous or non-uniform slices."""


class ShapeMismatchError(IngestionError):
    """Raised when modalities do not share uniform matrix dimensions."""


class DICOMCorruptError(IngestionError):
    """Raised when a DICOM file cannot be read or is corrupt."""


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpatialMetadata:
    """Spatial metadata for a single modality volume.

    Attributes:
        affine: 4x4 RAS affine matrix of the original (uncropped) volume.
        voxel_spacing_mm: (dx, dy, dz) voxel spacing in millimetres,
            derived from the affine columns.
        original_shape: (D, H, W) shape of the original volume.
        cropped_shape: (D, H, W) shape after the T2 brain-mask crop.
        crop_slices: (slice, slice, slice) applied to the original volume.
    """

    affine: np.ndarray
    voxel_spacing_mm: tuple[float, float, float]
    original_shape: tuple[int, int, int]
    cropped_shape: tuple[int, int, int]
    crop_slices: tuple[slice, slice, slice]


@dataclass(frozen=True)
class PatientScans:
    """Validated multi-modal scan data for a single patient.

    Attributes:
        volumes: Mapping of modality name -> float32 volume array.
        metadata: Mapping of modality name -> :class:`SpatialMetadata`.
        source: Provenance string describing the input source.
    """

    volumes: dict[str, np.ndarray]
    metadata: dict[str, SpatialMetadata]
    source: str


@dataclass(frozen=True)
class PlanningSeries:
    """A candidate pre-operative planning series discovered in DICOM metadata.

    Discovery is intentionally metadata-only. A candidate must still be
    converted, registered to the tumour MRI, and reviewed before it can be
    shown as a navigable critical-structure layer.
    """

    role: str
    series_uid: str
    description: str
    protocol: str
    slice_count: int


# ---------------------------------------------------------------------------
# NIfTI ingestion
# ---------------------------------------------------------------------------


def _infer_modality_from_path(path: str) -> Optional[str]:
    """Infer the modality from a NIfTI filename.

    Returns the modality key (e.g. ``"t1ce"``) or ``None`` if it cannot
    be determined. Matching is case-insensitive and token-based.
    """
    name = Path(path).name.lower()
    for mod, tokens in _MODALITY_TOKENS.items():
        for tok in tokens:
            if re.search(rf"(^|[^a-z0-9]){re.escape(tok)}([^a-z0-9]|$)", name):
                return mod
    return None


def _load_nifti_volume(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load a NIfTI file and return (data, affine)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"NIfTI file not found: {path}")
    img = nib.load(path)
    data = img.get_fdata().astype(np.float32)
    affine = np.asarray(img.affine, dtype=np.float64)
    return data, affine


def _load_nifti_dict(paths: dict[str, str]) -> PatientScans:
    """Load volumes from an explicit modality -> path mapping."""
    volumes: dict[str, np.ndarray] = {}
    metadata: dict[str, SpatialMetadata] = {}

    for mod in MODALITY_ORDER:
        if mod not in paths:
            raise MissingModalityError(
                f"Missing modality '{mod}' in NIfTI path mapping. "
                f"Provided keys: {sorted(paths.keys())}"
            )
        data, affine = _load_nifti_volume(paths[mod])
        volumes[mod] = data
        metadata[mod] = _build_metadata(data, affine)

    return PatientScans(volumes=volumes, metadata=metadata, source="nifti-dict")


def _load_nifti_list(paths: Sequence[str]) -> PatientScans:
    """Load volumes from a list of 4 NIfTI paths, inferring modality."""
    if len(paths) != len(MODALITY_ORDER):
        raise IngestionError(
            f"Expected {len(MODALITY_ORDER)} NIfTI paths (one per modality), "
            f"got {len(paths)}."
        )

    mapping: dict[str, str] = {}
    for p in paths:
        mod = _infer_modality_from_path(p)
        if mod is None:
            raise IngestionError(
                f"Cannot infer modality from filename: {p}. "
                "Rename the file to include t1/t1ce/t2/flair, or pass a "
                "dict mapping modality -> path."
            )
        if mod in mapping:
            raise IngestionError(
                f"Duplicate modality '{mod}' inferred from multiple paths: "
                f"{mapping[mod]} and {p}."
            )
        mapping[mod] = p

    return _load_nifti_dict(mapping)


# ---------------------------------------------------------------------------
# DICOM ingestion
# ---------------------------------------------------------------------------


def _is_dicom_file(path: str) -> bool:
    """Heuristic check: extension-based, or magic bytes for extensionless."""
    ext = Path(path).suffix.lower()
    if ext in _DICOM_EXTS:
        return True
    if ext:
        return False
    # Extensionless: check for the DICOM preamble magic (128 zero bytes + 'DICM').
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def _discover_dicom_files(root: str) -> list[str]:
    """Recursively find all DICOM files under a directory."""
    files: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        # Skip our own conversion scratch dirs (symlinked copies) — they
        # would otherwise double every series on a re-scan of the tree.
        dirnames[:] = [d for d in dirnames if not d.startswith("convert_")]
        for f in filenames:
            full = os.path.join(dirpath, f)
            if _is_dicom_file(full):
                files.append(full)
    return sorted(files)


def _extract_zip(zip_path: str) -> str:
    """Extract a ZIP archive to a temp directory and return its path."""
    tmp = tempfile.mkdtemp(prefix="brats_dicom_")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(tmp)
    return tmp


def _read_dicom_series_metadata(dcm_path: str) -> dict[str, object]:
    """Read key DICOM tags from a single file for series grouping.

    Raises :class:`DICOMCorruptError` if the file cannot be parsed.
    """
    try:
        import pydicom

        # force=True tolerates real-world DICOM files that lack a standard
        # 128-byte preamble + 'DICM' meta header (some PACS exports / raw dumps).
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=True, force=True)
    except Exception as exc:  # pydicom raises various exceptions
        raise DICOMCorruptError(f"Failed to read DICOM file '{dcm_path}': {exc}") from exc

    def _get(tag: str) -> object:
        try:
            return ds.get(tag)
        except Exception:
            return None

    return {
        "series_uid": str(_get("SeriesInstanceUID") or ""),
        "series_desc": str(_get("SeriesDescription") or ""),
        "protocol": str(_get("ProtocolName") or ""),
        "image_type": str(_get("ImageType") or ""),
        "position": _get("ImagePositionPatient"),
        "spacing": _get("PixelSpacing"),
        "slice_thickness": _get("SliceThickness"),
    }


def _infer_modality_from_dicom(meta: dict[str, object]) -> Optional[str]:
    """Infer modality from DICOM series description/protocol/image type.

    Returns ``'__scout__'`` for localizer/survey series (to be skipped),
    a modality key for diagnostic series, or ``None`` when unrecognized.
    """
    haystack = " ".join(
        str(meta.get(k) or "") for k in ("series_desc", "protocol", "image_type")
    ).lower()
    if any(tok in haystack for tok in _SCOUT_TOKENS):
        return "__scout__"
    for mod, tokens in _MODALITY_TOKENS.items():
        for tok in tokens:
            if tok in haystack:
                return mod
    return None


def _infer_planning_role(meta: dict[str, object]) -> Optional[str]:
    """Return the strongest surgical-planning role suggested by metadata."""
    haystack = " ".join(
        str(meta.get(k) or "") for k in ("series_desc", "protocol", "image_type")
    ).lower()
    for role, tokens in _PLANNING_SERIES_TOKENS.items():
        if any(token in haystack for token in tokens):
            return role
    return None


def inspect_surgical_planning_series(root: str) -> list[PlanningSeries]:
    """Discover candidate planning series without loading their pixel data.

    ``root`` may be a DICOM directory or ZIP. Returned entries are *not*
    registered layers and should be presented as requiring technical and
    surgical review. Failures are raised as :class:`IngestionError` so callers
    can retain the tumour workflow while showing a precise planning-data note.
    """
    work_dir: Optional[str] = None
    try:
        if zipfile.is_zipfile(root):
            work_dir = _extract_zip(root)
            search_root = work_dir
        else:
            search_root = root
        files = _discover_dicom_files(search_root)
        if not files:
            raise IngestionError(f"No DICOM files found under '{root}'.")
        candidates: list[PlanningSeries] = []
        for uid, series_files in _group_dicom_series(files).items():
            meta = _read_dicom_series_metadata(series_files[0])
            role = _infer_planning_role(meta)
            if role is None:
                continue
            candidates.append(PlanningSeries(
                role=role,
                series_uid=uid,
                description=str(meta.get("series_desc") or "Unnamed series"),
                protocol=str(meta.get("protocol") or ""),
                slice_count=len(series_files),
            ))
        return sorted(candidates, key=lambda item: (item.role, item.description))
    finally:
        if work_dir is not None:
            import shutil
            shutil.rmtree(work_dir, ignore_errors=True)


def _sort_series_by_position(files: Sequence[str]) -> list[str]:
    """Sort a series by InstanceNumber, falling back to IPP z, then name.

    Conversion scratch dirs (``convert_*``) hold symlinks back into the
    source tree — they must be excluded by the caller, but as a guard we
    also de-duplicate on (SeriesUID-independent) SOPInstanceUID / path so
    a re-scanned tree never yields doubled slices.
    """
    seen: set[str] = set()
    uniq: list[str] = []
    try:
        import pydicom
        for f in files:
            try:
                if os.path.islink(f):
                    continue  # conversion scratch symlink, not source data
                ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
                sop = str(ds.get("SOPInstanceUID") or f)
            except Exception:
                sop = f
            if sop in seen:
                continue
            seen.add(sop)
            uniq.append(f)
    except Exception:
        uniq = [f for f in files if not os.path.islink(f)]
    def key(f: str) -> tuple:
        try:
            import pydicom
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            inst = ds.get("InstanceNumber")
            if inst is not None:
                return (0, int(inst), f)
            pos = ds.get("ImagePositionPatient")
            if pos is not None and len(pos) >= 3:
                return (1, float(pos[2]), f)
        except Exception:
            pass
        return (2, 0.0, f)
    return sorted(uniq, key=key)


def _group_dicom_series(files: Sequence[str]) -> dict[str, list[str]]:
    """Group DICOM files by SeriesInstanceUID.

    Returns a mapping of series UID -> sorted list of file paths.
    Sorting is by InstanceNumber (PACS-acquisition order), NOT by
    lexicographic filename — filenames repeat across modalities
    (e.g. `t1/slice_000.dcm` vs `t2/slice_000.dcm`) and naive sorting
    interleaves series, breaking slice-continuity validation.
    """
    groups: dict[str, list[str]] = {}
    for f in files:
        meta = _read_dicom_series_metadata(f)
        uid = str(meta["series_uid"])
        if not uid:
            raise DICOMCorruptError(
                f"DICOM file '{f}' has no SeriesInstanceUID; cannot group series."
            )
        groups.setdefault(uid, []).append(f)
    for uid in groups:
        groups[uid] = _sort_series_by_position(groups[uid])
    return groups


def _validate_slice_continuity(files: Sequence[str]) -> None:
    """Validate that a DICOM series has continuous, uniformly spaced slices.

    Checks that slice positions are monotonic with near-constant spacing
    ALONG THE SERIES NORMAL (projection of ImagePositionPatient onto the
    slice normal from ImageOrientationPatient), not raw z. Multi-orientation
    localizers (e.g. 3-plane surveys) project to a constant and are
    reported as non-stackable rather than "non-monotonic".
    """
    origins: list[np.ndarray] = []
    normal: np.ndarray | None = None
    for f in files:
        try:
            import pydicom
            ds = pydicom.dcmread(f, stop_before_pixels=True, force=True)
            pos = ds.get("ImagePositionPatient")
            ori = ds.get("ImageOrientationPatient")
            if pos is None or len(pos) < 3:
                raise SliceContinuityError(
                    f"DICOM file '{f}' is missing ImagePositionPatient; "
                    "cannot verify slice continuity."
                )
            origins.append(np.asarray([float(pos[0]), float(pos[1]),
                                       float(pos[2])]))
            if normal is None and ori is not None and len(ori) >= 6:
                r = np.asarray([float(ori[0]), float(ori[1]), float(ori[2])])
                c = np.asarray([float(ori[3]), float(ori[4]), float(ori[5])])
                normal = np.cross(r, c)
                n = float(np.linalg.norm(normal))
                normal = normal / n if n > 1e-12 else None
        except SliceContinuityError:
            raise
        except Exception as exc:
            raise SliceContinuityError(
                f"Failed to read slice position of '{f}': {exc}") from exc

    if len(origins) < 2:
        return  # Single-slice series: nothing to compare.

    if normal is None:
        axis = int(np.argmax(np.ptp(np.stack(origins), axis=0)))
        positions = np.asarray([o[axis] for o in origins], dtype=np.float64)
    else:
        positions = np.asarray([float(o @ normal) for o in origins],
                               dtype=np.float64)

    if float(np.ptp(positions)) < 1e-6:
        raise SliceContinuityError(
            "DICOM series is a multi-orientation localizer/survey "
            "(no stackable slice axis); it cannot be reconstructed as a "
            "3D volume. Exclude scout series and retry."
        )

    diffs = np.diff(positions)
    if np.any(diffs <= 0):
        raise SliceContinuityError(
            "DICOM series has non-monotonic slice positions; "
            "slices are out of order or duplicated."
        )

    median_spacing = float(np.median(diffs))
    if median_spacing <= 0:
        raise SliceContinuityError("DICOM series has non-positive slice spacing.")

    # Allow up to 1% deviation from the median slice spacing.
    tolerance = 0.01 * median_spacing
    if np.any(np.abs(diffs - median_spacing) > tolerance):
        raise SliceContinuityError(
            "DICOM series has non-uniform slice spacing; "
            "slices may be missing or duplicated."
        )


def _convert_dicom_series_to_nifti(
    series_files: Sequence[str],
    work_dir: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a DICOM series to a 3D volume using dicom2nifti.

    Returns (data, affine). Raises :class:`DICOMCorruptError` on failure.
    """
    try:
        import dicom2nifti.settings as d2n_settings
        from dicom2nifti.convert_dicom import dicom_series_to_nifti

        # Tolerate DICOM files lacking the standard 128-byte preamble +
        # 'DICM' marker (common in some PACS exports / raw dumps). This
        # mirrors the force=True used in _read_dicom_series_metadata.
        d2n_settings.pydicom_read_force = True

        # Unique scratch dir OUTSIDE the source tree (temp dir), so a
        # re-scan of the source tree never picks up our symlinks/outputs.
        # (Scratch inside search_root caused series doubling on re-runs.)
        scratch = tempfile.mkdtemp(prefix="brats_convert_")
        series_dir = os.path.join(scratch, "dicom")
        os.makedirs(series_dir, exist_ok=True)
        for f in series_files:
            dst = os.path.join(series_dir, os.path.basename(f))
            if not os.path.exists(dst):
                os.symlink(f, dst)

        # dicom_series_to_nifti returns a dict with the NIfTI image under
        # 'NII' and the file path under 'NII_FILE'. It RAISES on failure
        # (unlike convert_directory, which silently swallows conversion
        # errors and produces no output). An explicit output_file is
        # required (None triggers a NoneType path error).
        out_nii = os.path.join(scratch, "converted.nii.gz")
        result = dicom_series_to_nifti(series_dir, output_file=out_nii, reorient_nifti=True)
        img = result["NII"]
        data = img.get_fdata().astype(np.float32)
        shape = img.shape
        if len(shape) != 3:
            raise IngestionError(
                f"Expected a 3D volume from DICOM conversion, got shape {shape}."
            )
        affine = np.asarray(img.affine, dtype=np.float64)
        import shutil
        shutil.rmtree(scratch, ignore_errors=True)
        return data, affine
    except Exception as exc:
        import shutil
        try:
            shutil.rmtree(scratch, ignore_errors=True)
        except Exception:
            pass
        raise DICOMCorruptError(
            f"Failed to convert DICOM series to NIfTI: {exc}"
        ) from exc


def _load_dicom_root(root: str) -> PatientScans:
    """Load all 4 modalities from a directory (or ZIP) of DICOM series."""
    work_dir: Optional[str] = None
    try:
        if zipfile.is_zipfile(root):
            work_dir = _extract_zip(root)
            search_root = work_dir
        else:
            search_root = root

        files = _discover_dicom_files(search_root)
        if not files:
            raise IngestionError(
                f"No DICOM files found under '{root}'. "
                "Expected .dcm/.dicom/.ima files or extensionless DICOM."
            )

        groups = _group_dicom_series(files)

        # Map each series to a modality (scouts skipped, not errors).
        # Continuity is validated per DIAGNOSTIC series only: scouts are
        # multi-orientation localizers and never stackable.
        series_modality: dict[str, str] = {}
        skipped_scouts: list[str] = []
        skipped_planning: list[str] = []
        for uid, series_files in groups.items():
            meta = _read_dicom_series_metadata(series_files[0])
            mod = _infer_modality_from_dicom(meta)
            if mod == "__scout__":
                skipped_scouts.append(
                    f"{meta.get('series_desc')!r} ({len(series_files)} slices)")
                continue
            if mod is None:
                planning_role = _infer_planning_role(meta)
                if planning_role is not None:
                    skipped_planning.append(
                        f"{planning_role}: {meta.get('series_desc')!r} "
                        f"({len(series_files)} slices)")
                    continue
                raise IngestionError(
                    f"Cannot infer modality for DICOM series '{uid}' "
                    f"(description: {meta.get('series_desc')!r}). "
                    "Rename the series description to include t1/t1ce/t2/flair."
                )
            if mod in series_modality:
                raise IngestionError(
                    f"Multiple DICOM series map to modality '{mod}' "
                    f"(series {series_modality[mod]} and {uid})."
                )
            series_modality[mod] = uid

        # Validate completeness.
        missing = [m for m in MODALITY_ORDER if m not in series_modality]
        if missing:
            scout_note = (f" Skipped scout/localizer series: "
                          f"{'; '.join(skipped_scouts)}." if skipped_scouts else "")
            planning_note = (f" Detected planning candidate series: "
                             f"{'; '.join(skipped_planning)}."
                             if skipped_planning else "")
            raise MissingModalityError(
                f"Missing DICOM series for modality/modalities: {missing}. "
                f"Found series: {sorted(series_modality.keys())}."
                + scout_note + planning_note
            )

        # Convert each series and validate.
        volumes: dict[str, np.ndarray] = {}
        metadata: dict[str, SpatialMetadata] = {}
        for mod in MODALITY_ORDER:
            uid = series_modality[mod]
            series_files = groups[uid]
            _validate_slice_continuity(series_files)
            data, affine = _convert_dicom_series_to_nifti(series_files, work_dir or search_root)
            volumes[mod] = data
            metadata[mod] = _build_metadata(data, affine)

        return PatientScans(
            volumes=volumes,
            metadata=metadata,
            source=f"dicom:{root}",
        )
    finally:
        # Clean up the temp extraction dir.
        if work_dir is not None:
            import shutil

            shutil.rmtree(work_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _build_metadata(
    data: np.ndarray,
    affine: np.ndarray,
    crop_slices: Optional[tuple[slice, slice, slice]] = None,
) -> SpatialMetadata:
    """Build :class:`SpatialMetadata` from a volume and its affine."""
    if crop_slices is None:
        crop_slices = (slice(None), slice(None), slice(None))

    # Voxel spacing = L2 norm of each affine column (in mm): (dx, dy, dz).
    spacing = tuple(
        float(np.linalg.norm(affine[:3, i])) for i in range(3)
    )

    cropped_shape = tuple(
        data.shape[i] if sl == slice(None) else sl.stop - sl.start
        for i, sl in enumerate(crop_slices)
    )

    return SpatialMetadata(
        affine=affine.copy(),
        voxel_spacing_mm=spacing,
        original_shape=tuple(int(s) for s in data.shape),
        cropped_shape=tuple(int(s) for s in cropped_shape),
        crop_slices=crop_slices,
    )


def _validate_orientation(metadata: dict[str, SpatialMetadata]) -> None:
    """Validate that all modalities share the same spatial orientation."""
    ref = metadata[MODALITY_ORDER[0]].affine[:3, :3]
    ref_signs = np.sign(ref)
    for mod in MODALITY_ORDER[1:]:
        aff = metadata[mod].affine[:3, :3]
        if not np.array_equal(np.sign(aff), ref_signs):
            raise OrientationMismatchError(
                f"Modality '{mod}' has a different spatial orientation than "
                f"'{MODALITY_ORDER[0]}'. All modalities must share the same "
                "RAS/LAS orientation for consistent cropping."
            )


def _validate_uniform_shape(volumes: dict[str, np.ndarray]) -> None:
    """Validate that all modalities share identical matrix dimensions."""
    ref_shape = volumes[MODALITY_ORDER[0]].shape
    for mod in MODALITY_ORDER[1:]:
        if volumes[mod].shape != ref_shape:
            raise ShapeMismatchError(
                f"Modality '{mod}' has shape {volumes[mod].shape}, but "
                f"'{MODALITY_ORDER[0]}' has shape {ref_shape}. "
                "All modalities must share uniform matrix dimensions."
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_patient_scans(
    input_source: Union[dict[str, str], Sequence[str], str],
) -> PatientScans:
    """Load and validate multi-modal scan data for a single patient.

    Args:
        input_source: One of:
            * ``dict[str, str]`` — mapping of modality -> NIfTI path.
            * ``list[str]`` / ``tuple[str, ...]`` — 4 NIfTI paths; modality
              inferred from filenames.
            * ``str`` — a directory or ZIP containing DICOM series for all
              4 modalities.

    Returns:
        A validated :class:`PatientScans` with volumes and per-modality
        spatial metadata.

    Raises:
        MissingModalityError: If not all 4 modalities are present.
        OrientationMismatchError: If modalities differ in orientation.
        SliceContinuityError: If a DICOM series has non-continuous slices.
        ShapeMismatchError: If modalities differ in matrix dimensions.
        DICOMCorruptError: If a DICOM file/series cannot be read.
        IngestionError: For other ingestion failures.
    """
    if isinstance(input_source, dict):
        scans = _load_nifti_dict(input_source)
    elif isinstance(input_source, (list, tuple)):
        scans = _load_nifti_list(input_source)
    elif isinstance(input_source, str):
        if os.path.isdir(input_source) or zipfile.is_zipfile(input_source):
            scans = _load_dicom_root(input_source)
        else:
            raise IngestionError(
                f"Input source '{input_source}' is neither a directory, "
                "a ZIP archive, nor a NIfTI path mapping."
            )
    else:
        raise TypeError(
            f"Unsupported input_source type: {type(input_source).__name__}. "
            "Expected dict[str, str], list[str], or str."
        )

    # Cross-modality validation.
    _validate_orientation(scans.metadata)
    _validate_uniform_shape(scans.volumes)

    return scans


def prepare_for_inference(
    scans: PatientScans,
    target_size: int = TARGET_SIZE,
) -> tuple[np.ndarray, dict[str, SpatialMetadata]]:
    """Apply Phase 1 preprocessing and return a batch-ready tensor.

    Steps (reusing Phase 1 functions unchanged):
        1. T2 brain-mask bounding-box crop.
        2. Resample each cropped modality to ``(target_size,)``.
        3. Per-channel z-score normalization.
        4. Stack -> ``(4, D, H, W)`` -> add batch dim -> ``(1, 4, D, H, W)``.

    Args:
        scans: Validated :class:`PatientScans`.
        target_size: Target spatial size (Phase 1 default: 128).

    Returns:
        A tuple ``(tensor, metadata)`` where:
            * ``tensor`` is a float32 array of shape
              ``(1, 4, target_size, target_size, target_size)`` ready for
              ``ONNXInferenceEngine.predict``.
            * ``metadata`` maps each modality to its :class:`SpatialMetadata`
              (with ``crop_slices`` populated) for physical volume
              calculations.
    """
    t2 = scans.volumes["t2"]
    mask = brain_mask_from_t2(t2)

    cropped: dict[str, np.ndarray] = {}
    crop_slices: dict[str, tuple[slice, slice, slice]] = {}
    for mod in MODALITY_ORDER:
        vol, slc = crop_to_bbox(scans.volumes[mod], mask)
        cropped[mod] = vol
        crop_slices[mod] = slc

    resampled = {
        mod: resample_volume(cropped[mod], target_size) for mod in MODALITY_ORDER
    }
    normalized = {
        mod: zscore_normalize(resampled[mod]) for mod in MODALITY_ORDER
    }

    stack = np.stack([normalized[mod] for mod in MODALITY_ORDER], axis=0)
    tensor = stack[np.newaxis, ...].astype(np.float32)

    # Rebuild metadata with crop slices populated.
    metadata: dict[str, SpatialMetadata] = {}
    for mod in MODALITY_ORDER:
        orig = scans.metadata[mod]
        metadata[mod] = SpatialMetadata(
            affine=orig.affine,
            voxel_spacing_mm=orig.voxel_spacing_mm,
            original_shape=orig.original_shape,
            cropped_shape=tuple(int(s) for s in cropped[mod].shape),
            crop_slices=crop_slices[mod],
        )

    return tensor, metadata
