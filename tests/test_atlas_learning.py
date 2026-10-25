from pathlib import Path

import nibabel as nib
import numpy as np

from src.atlas_learning import atlas_learning_figure


def _write(path: Path, data: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def test_atlas_projection_contains_labelled_anatomy_and_tumour(tmp_path):
    shape = (20, 20, 20)
    z, y, x = np.indices(shape)
    t1 = (z * 2 + y * 3 + x).astype(np.float32)
    t1[:3] = 0
    t1[17:] = 0
    tumour = np.zeros(shape, dtype=np.uint8)
    tumour[8:12, 8:12, 8:12] = 1
    atlas = np.zeros(shape, dtype=np.int16)
    atlas[3:8, 3:8, 3:8] = 2001
    atlas[12:17, 12:17, 12:17] = 7001
    paths = {name: tmp_path / f"{name}.nii.gz" for name in ("t1", "tumour", "atlas")}
    _write(paths["t1"], t1)
    _write(paths["tumour"], tumour)
    _write(paths["atlas"], atlas)

    figure, note, summary = atlas_learning_figure(
        str(paths["t1"]), str(paths["tumour"]), str(paths["t1"]), str(paths["atlas"]))

    names = {trace.name for trace in figure.data}
    assert "Motor and frontal cortex" in names
    assert "Deep nuclei" in names
    assert "Tumour projection — educational" in names
    assert "not for navigation" in note
    assert "Atlas anatomy context" in summary
