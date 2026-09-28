"""Mesh + segmentation export: STL (no new deps), GLB optional, NIfTI-seg."""

from __future__ import annotations

import os
import struct
from typing import Optional

import nibabel as nib
import numpy as np
from skimage import measure

__all__ = [
    "meshes_from_segmentation",
    "write_stl_binary",
    "export_label_meshes",
    "export_nifti_seg",
    "try_export_glb",
]

LABEL_INFO = {
    1: ("necrotic", "#FF004D", (255, 0, 77)),
    2: ("edema", "#00FF4D", (0, 255, 77)),
    3: ("enhancing", "#00D0FF", (0, 208, 255)),
}


def meshes_from_segmentation(seg: np.ndarray, step_size: int = 1,
                             spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)):
    """Marching-cubes meshes per label: {label: (verts_mm, faces)}.

    Native BraTS label 4 (enhancing) is merged into label 3 so raw
    sidecars export an enhancing mesh instead of silently dropping it.
    """
    out: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for label in (1, 2, 3):
        mask = np.isin(seg, (label, 4)) if label == 3 else (seg == label)
        if not mask.any():
            continue
        try:
            verts, faces, _, _ = measure.marching_cubes(
                mask.astype(np.float32), level=0.5, step_size=step_size)
        except Exception:
            continue
        verts = verts * np.asarray(spacing, dtype=np.float64)
        out[label] = (verts.astype(np.float32),
                      faces.astype(np.int32))
    return out


def write_stl_binary(path: str, verts: np.ndarray, faces: np.ndarray,
                     label: str = "tumor") -> str:
    """Write binary STL with zero new dependencies."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tris = verts[faces]  # (N,3,3)
    with open(path, "wb") as f:
        header = f"BRATS {label} {len(tris)} tris".encode()[:80]
        f.write(header + b" " * (80 - len(header)))
        f.write(struct.pack("<I", int(len(tris))))
        for tri in tris:
            v0, v1, v2 = tri
            n = np.cross(v1 - v0, v2 - v0)
            ln = float(np.linalg.norm(n))
            n = n / ln if ln > 1e-12 else np.array([0, 0, 1])
            f.write(struct.pack("<12f", n[0], n[1], n[2],
                                v0[0], v0[1], v0[2],
                                v1[0], v1[1], v1[2],
                                v2[0], v2[1], v2[2]))
            f.write(struct.pack("<H", 0))
    return path


def export_label_meshes(seg: np.ndarray, out_dir: str,
                        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
                        step_size: int = 1) -> dict[int, str]:
    """Export one STL per non-empty label. Returns {label: path}."""
    os.makedirs(out_dir, exist_ok=True)
    meshes = meshes_from_segmentation(seg, step_size=step_size, spacing=spacing)
    paths: dict[int, str] = {}
    for label, (verts, faces) in meshes.items():
        name = LABEL_INFO[label][0]
        paths[label] = write_stl_binary(
            os.path.join(out_dir, f"tumor_{name}.stl"), verts, faces, name)
    return paths


def export_nifti_seg(seg: np.ndarray, reference_nii_path: str,
                     out_path: str) -> str:
    """Save integer label map reusing reference affine/header."""
    ref = nib.load(reference_nii_path)
    img = nib.Nifti1Image(seg.astype(np.int16), ref.affine, ref.header)
    img.set_data_dtype(np.int16)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    nib.save(img, out_path)
    return out_path


def try_export_glb(seg: np.ndarray, out_path: str,
                   spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
                   ) -> Optional[str]:
    """GLB export if `trimesh` installed; else None (STL remains)."""
    try:
        import trimesh
    except Exception:
        return None
    meshes = meshes_from_segmentation(seg, spacing=spacing)
    if not meshes:
        return None
    scene = trimesh.Scene()
    for label, (verts, faces) in meshes.items():
        name, _, rgb = LABEL_INFO[label]
        m = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
        m.visual.vertex_colors = np.tile(
            np.array([*rgb, 255], dtype=np.uint8), (len(verts), 1))
        scene.add_geometry(m, geom_name=f"tumor_{name}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    scene.export(out_path, file_type="glb")
    return out_path
