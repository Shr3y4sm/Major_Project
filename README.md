# Brain Tumour Segmentation

This project implements a multimodal 3D brain tumour segmentation workflow for BraTS-style MRI data. It reconstructs a 3D U-Net from the Phase 1 notebook architecture, preprocesses four MRI modalities, runs inference with ONNX Runtime, and presents the result in an interactive Gradio application.

The application supports both individual NIfTI uploads and a ZIP archive containing DICOM series. Results include tumour-class volume estimates in cm3, an interactive 3D surface view, and a three-plane 2D multiplanar reconstruction (MPR) viewer.

## Implemented Features

- Four-channel MRI ingestion in the canonical order `T1`, `T1ce`, `T2`, `FLAIR`.
- NIfTI input through either an explicit modality-to-file mapping or filenames from which modalities can be inferred.
- Recursive DICOM discovery from a directory or ZIP archive.
- DICOM series grouping by `SeriesInstanceUID` and modality inference from series description, protocol, or image type.
- Validation for missing modalities, inconsistent volume shapes, incompatible orientations, corrupt DICOM data, and non-uniform slice spacing.
- T2-based brain masking and bounding-box cropping.
- Resampling of each cropped modality to `128 x 128 x 128` voxels.
- Per-modality z-score normalization and batch preparation as a float32 tensor with shape `(1, 4, 128, 128, 128)`.
- 3D U-Net inference with ONNX Runtime, preferring CUDA and falling back to CPU when CUDA is unavailable.
- Resampling of the predicted class map back into the original scan space.
- Physical tumour volume calculation using voxel spacing from the source affine.
- Interactive Plotly 3D meshes and MPR overlays.
- Optional discovery of a nearby BraTS segmentation sidecar for reference visualization.
- PyTorch-to-ONNX numerical equivalence verification.
- Ingestion and preprocessing smoke tests, including synthetic NIfTI and DICOM data.

## Tumour Labels

The inference model produces four classes:

| Class | Name | Meaning |
| ---: | --- | --- |
| 0 | Background | Non-tumour voxels |
| 1 | Necrotic Core | Necrotic or non-enhancing tumour core |
| 2 | Peritumoral Edema | Peritumoral oedema |
| 3 | Enhancing Tumor | Enhancing tumour; BraTS label 4 is remapped to 3 during notebook preprocessing |

The application reports volumes for Necrotic Core, Peritumoral Edema, Enhancing Tumor, and their sum as Total Whole Tumor.

## Processing Pipeline

```text
NIfTI files or DICOM ZIP
        |
        v
Validated PatientScans
        |
        v
T2 brain mask -> bounding-box crop -> 128^3 resampling -> channel z-score normalization
        |
        v
(1, 4, 128, 128, 128) input tensor
        |
        v
ONNX Runtime 3D U-Net
        |
        v
Class map -> original-space reconstruction -> volumes and visualizations
```

## Repository Structure

```text
.
|-- app.py                         Gradio application and visualisation workflow
|-- requirements.txt               Runtime and development dependencies
|-- artifacts/
|   |-- best_unet.pt               PyTorch state_dict checkpoint
|   |-- best_unet.onnx             ONNX model
|   |-- best_unet.onnx.data        External ONNX weights
|   |-- unet3d.onnx                Additional ONNX export
|   `-- unet3d.onnx.data            External ONNX weights
|-- src/
|   |-- dicom_loader.py             NIfTI/DICOM ingestion and validation
|   |-- preprocessing.py            Phase 1 dataset and preprocessing helpers
|   |-- model.py                    3D U-Net architecture and checkpoint loading
|   |-- inference.py                ONNX/PyTorch inference and evaluation metrics
|   |-- export_onnx.py              PyTorch checkpoint to ONNX export utility
|   `-- verify_onnx.py              PyTorch/ONNX equivalence command
|-- tests/
|   `-- test_dicom_loader.py         Ingestion and preprocessing smoke tests
|-- sample_brats_patient/            Small sample NIfTI/DICOM-style patient inputs
|-- braintumour-phase-1.ipynb        Phase 1 model and preprocessing notebook
`-- previous/braintumour-phase-0.ipynb
```

## Requirements

- Python 3.10 or newer. The development environment used for this repository is Python 3.12.3.
- A CPU is sufficient for inference, although a compatible CUDA installation can be used through ONNX Runtime GPU.
- Enough memory for 3D inference on a `128^3` volume. GPU memory requirements depend on the execution provider and model build.

## Setup

Create and activate a virtual environment, then install the dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The test module uses `pytest`. Install it separately because it is not currently listed in `requirements.txt`:

```bash
python -m pip install pytest
```

The requirements file includes both `onnxruntime` and the Linux-specific `onnxruntime-gpu` package. ONNX Runtime still falls back to `CPUExecutionProvider` when CUDA is not available. If package resolution for a particular machine does not support both providers, install the appropriate CPU or GPU ONNX Runtime package for that environment and keep the provider configuration in `src/inference.py` unchanged.

## Run the Application

From the repository root, ensure the ONNX model and its external data file are present in `artifacts/`, then run:

```bash
source .venv/bin/activate
python app.py
```

Open `http://localhost:7860` in a browser.

The application binds to `0.0.0.0:7860`. To use another port, change the `server_port` value in the `app.launch` call in `app.py`.

### NIfTI Input

Choose `NIfTI Files` and upload one file for each modality:

1. T1
2. T1ce
3. T2
4. FLAIR

All four volumes must have matching matrix dimensions and compatible spatial orientations. Filenames should contain the modality tokens `t1`, `t1ce`, `t2`, and `flair` when files are passed to the loader as a list. The Gradio application passes an explicit mapping, so the upload controls determine the modality.

### DICOM ZIP Input

Choose `DICOM ZIP` and upload an archive containing all four multi-slice series. The loader searches recursively for `.dcm`, `.dicom`, `.ima`, and recognised extensionless DICOM files. Series metadata must contain modality tokens that identify `t1`, `t1ce`, `t2`, and `flair`; `t1ce` is checked before `t1` to avoid ambiguous matching.

The loader validates slice ordering and spacing before converting each series to NIfTI with `dicom2nifti`.

## Model Export

Export a PyTorch checkpoint to ONNX with:

```bash
python -m src.export_onnx \
  --checkpoint artifacts/best_unet.pt \
  --output artifacts/best_unet.onnx \
  --opset 17
```

The export uses a sample input shape of `(1, 4, 128, 128, 128)`, keeps the channel count fixed at four, and marks batch and spatial dimensions as dynamic. The script validates the resulting ONNX graph with `onnx.checker`.

If the checkpoint path does not exist, the export utility creates a randomly initialized checkpoint so that the technical pipeline can run end to end. Those dummy weights are not a trained clinical model and must be replaced with the intended trained checkpoint for meaningful predictions.

## Verify PyTorch and ONNX Outputs

Run the deterministic equivalence check with:

```bash
python -m src.verify_onnx \
  --checkpoint artifacts/best_unet.pt \
  --onnx artifacts/best_unet.onnx \
  --atol 1e-3
```

The command reports the active ONNX Runtime providers, maximum and mean absolute logit differences, and whether the maximum difference is within the requested tolerance.

## Run Tests

```bash
python -m pytest -q
```

The current tests cover NIfTI dictionary/list ingestion, preprocessing output shape and metadata, synthetic DICOM ingestion, missing modalities, shape mismatch, orientation mismatch, and invalid input types.

## Dataset and Notebook Workflow

`src/preprocessing.py` retains the Phase 1 dataset helpers for local BraTS folders, Kaggle-mounted data, and training archives. Set `BRATS_LOCAL_ROOT` when using a local BraTS dataset root:

```bash
export BRATS_LOCAL_ROOT=/path/to/BraTS2021_Training_Data
```

The Phase 1 preprocessing functions load the four modalities and segmentation, remap BraTS label 4 to class 3, apply the T2 brain crop, resample the volumes, and normalize each modality. The notebooks document the earlier training and experimentation workflow; the current application uses the exported model artifacts for inference.

## Important Limitations

- This repository provides an inference and visualization workflow, not a complete training command-line pipeline.
- Predictions and reported volumes are research outputs and are not a substitute for clinical diagnosis or treatment decisions.
- Volume accuracy depends on correct modality identity, registration, orientation, voxel spacing, and acquisition quality.
- The application uses nearest-neighbour interpolation when mapping the predicted class map back from the `128^3` model grid to the cropped volume space.
- DICOM modality identification relies on descriptive metadata tokens and may require series names that follow the supported naming conventions.
- ONNX external-data files are required beside their corresponding `.onnx` files; moving only the `.onnx` file will make model loading fail.

## License

No license file is currently included in the repository.