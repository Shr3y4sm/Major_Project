# Demo Script — Live Demonstration for Judges

**Project:** Major_Project — BraTS Brain Tumor Segmentation MVP

**Estimated demo time:** 15–20 minutes

**Prerequisites:** Python 3.9+, all dependencies installed, model file available

---

## Prerequisites & Setup

### 1. Install dependencies

```bash
cd /home/shreyas/Downloads/Major_Project
pip install -r requirements.txt
```

Required packages: `numpy`, `nibabel`, `scipy`, `scikit-image`, `onnxruntime`, `torch`, `pydicom`, `dicom2nifti`, `gradio`, `plotly`, `pyvista`, `SimpleITK` (optional, for N4 in bratsify), `trimesh` (optional, for GLB export)

### 2. Export model to ONNX (if not already done)

```bash
python -m src.export_onnx \
    --checkpoint path/to/model.pth \
    --output model.onnx
```

This produces `model.onnx` (opset 13, dynamic batch axis).

### 3. Verify ONNX↔PyTorch equivalence

```bash
python -m src.verify_onnx \
    --checkpoint path/to/model.pth \
    --onnx model.onnx
```

**Expected output:** Max abs difference ≈ 6.71e-08 (well within float32 tolerance)

### 4. Run the test suite

```bash
pytest tests/ -v
```

**Expected output:** 62/62 tests passing

### 5. Launch the web UI

```bash
python app.py
```


---

## Demo Scenario 1: Full 4-Modality NIfTI → Complete Pipeline

**Goal:** Demonstrate the complete end-to-end pipeline with all 4 BraTS modalities.

### Step 1: Prepare test data
Have ready a directory or ZIP containing 4 NIfTI files:
- `patient_T1.nii.gz`
- `patient_T1ce.nii.gz`
- `patient_T2.nii.gz`
- `patient_FLAIR.nii.gz`

Or a dict mapping: `{"t1": "...", "t1ce": "...", "t2": "...", "flair": "..."}`

### Step 2: Upload in the UI
1. Open `http://localhost:7860`
2. Select **"Full 4-Modality"** mode (ModeToggle)
3. Upload the 4 NIfTI files (or the directory containing them)
4. Click **Process**

### Step 3: Watch processing status
The ProcessingStatus panel shows each stage:
- ✅ Ingestion (DICOM/NIfTI loading + validation)
- ✅ BraTS-ifier (registration, N4, harmonization, centering)
- ✅ Preprocessing (T2 brain mask, crop, resample, z-score)
- ✅ ONNX Inference (segmentation)
- ✅ Post-processing (cleanup, volumetry, RANO)
- ✅ Export (STL, NIfTI-seg, HTML report)

### Step 4: Review MPR visualization
The MPRViewer (Plotly) shows:
- Axial, coronal, sagittal slices
- Segmentation overlay (colors: red=necrotic, green=edema, cyan=enhancing)
- Interactive slice navigation

### Step 5: Review 3D tumor view
The Tumor3DView (PyVista) shows:
- 3D surface meshes per tumor label
- Interactive rotation/zoom
- Color-coded by label

### Step 6: Review volumetry table
The VolumesTable shows per-label volumes in cm³:
- Necrotic Core: X.XX cm³
- Edema: X.XX cm³
- Enhancing Tumor: X.XX cm³
- Whole Tumor: X.XX cm³ (sum)

### Step 7: Download outputs
From the DownloadPanel, download:
- `.stl` files per tumor label (3D printing / surgical planning)
- `.nii.gz` segmentation map (nibabel-compatible)
- `.html` surgical/radiology report (printable one-pager)

### Step 8: Review HTML report
Open the downloaded HTML report. Verify it contains:
- Volumetry table
- RANO-style diameters (dmax_mm × dperp_mm)
- MPR snapshot images (embedded)
- Ingestion/provenance notes
- Research-only disclaimer

**Judges' quick-check:** Verify the disclaimer text reads "Research visualization only — not for diagnosis or treatment decisions."

---

## Demo Scenario 2: Partial Scan (DICOM ZIP, Missing Modalities)

**Goal:** Demonstrate honest partial-scan handling — no zero-filling, no fake segmentation.

### Step 1: Prepare partial DICOM data
Have a DICOM ZIP or directory with only 2 modalities (e.g., T1 + T2, missing T1ce and FLAIR).

### Step 2: Upload in the UI
1. Select **"Partial Scan"** mode
2. Upload the DICOM ZIP
3. Click **Process**

### Step 3: Observe fallback behavior
The system should:
- Detect that T1ce and FLAIR are missing
- **Not** run the 4-channel U-Net (would be dishonest)
- Run `partial_analysis.py` instead:
  - Asymmetry heatmap on T2 (mirror L/R hemispheres)
  - Ellipsoid volume estimate (geometric, π/6 × D1×D2×D3)
  - Orange heat overlay on MPR

### Step 4: Review MPR with heat overlay
The MPRViewer shows:
- T2 grayscale base
- Orange heat overlay where asymmetry exceeds 98th percentile
- Explicit label: "NON-SPECIFIC — not a tumor segmentation"

### Step 5: Review PACS protocol checklist
The UI should display the protocol checklist (from `PROTOCOL_CHECKLIST_MD`):
- Request AX T1, T1C post-contrast, T2, FLAIR from PACS
- Export as DICOM (not screenshots)
- One series per modality
- Avoid scout-only exports

### Step 6: Verify no fake segmentation
Confirm that:
- No STL meshes are generated for tumor labels (no AI segmentation exists)
- No volumetry table shows tumor volumes
- The ellipsoid estimate is labeled "GEOMETRIC ESTIMATE — not an AI tumor segmentation"

**Judges' key question:** "What if only T1 and T2 are available?"
**Answer:** We don't zero-fill. We run partial analysis — asymmetry heatmap + ellipsoid estimate — clearly labeled as non-specific, and request the missing modalities from PACS.

---

## Demo Scenario 3: ONNX↔PyTorch Equivalence Verification

**Goal:** Prove the deployed ONNX model matches the PyTorch original.

### Step 1: Run verification
```bash
python -m src.verify_onnx \
    --checkpoint path/to/model.pth \
    --onnx model.onnx
```

### Step 2: Review output
**Expected:** `Max abs difference: 6.71e-08`

### Step 3: Explain to judges
- Identical random input tensor `(1, 4, 128, 128, 128)` run through both models
- Max absolute difference of ~6.7e-08 confirms numerical equivalence within float32 precision
- ONNX enables CPU-only deployment without PyTorch

**Judges' key question:** "How do you know the ONNX model is the same as PyTorch?"
**Answer:** We verified with identical inputs — max abs diff is 6.71e-08, well within float32 tolerance.

---

## Demo Scenario 4: Test Suite Execution

**Goal:** Show that the codebase is well-tested.

### Step 1: Run tests
```bash
pytest tests/ -v
```

### Step 2: Review results
**Expected:** 62/62 tests passing across:
- `tests/test_bratsify.py` — Bratsify guards, N4, registration, harmonization, centering, placeholder detection
- `tests/test_inference.py` — ONNX engine, equivalence, metrics, volumetry

### Step 3: Run validation script
```bash
python scripts/validate_metrics.py
```

**Expected output:** Confirms N4 guard, registration improvement (−0.04 → +0.63 NCC, 41 → 1 voxel), placeholder detection

---

## Demo Scenario 5: DICOM PHI Anonymization

**Goal:** Demonstrate PHI-safe handling of DICOM uploads.

### Step 1: Upload a DICOM ZIP with PHI
The ZIP contains DICOM files with PatientName, PatientID, etc.

### Step 2: Verify anonymization
Check that before processing:
- PHI tags are removed (PatientName, PatientID, BirthDate, etc.)
- PatientID is hashed to "ANON-XXXXXXXXXXXX"
- Group-0000 command elements are dropped

**Key code:** `src/anonymize.py` (`_REMOVE_KEYWORDS`, `_hash_id`, `anonymize_dicom_tree`)

**Judges' key question:** "Is patient data protected?"
**Answer:** Yes — all PHI DICOM tags are removed and PatientID is hashed before any processing occurs.

---

## Judges' Quick-Check Commands

| Command | Purpose | Expected |
|---------|---------|----------|
| `pytest tests/ -v` | Run all tests | 62/62 passed |
| `python scripts/validate_metrics.py` | Numerical validation | Guards pass, NCC improved |
| `python -m src.verify_onnx --checkpoint ... --onnx ...` | ONNX equivalence | Max abs diff ≈ 6.71e-08 |
| `python app.py` | Launch web UI | Gradio at localhost:7860 |

---

## Common Judge Questions & Answers (Demo Context)

| Question | Answer |
|----------|--------|
| "What happens if a hospital uploads DICOM with missing modalities?" | Partial-analysis fallback — asymmetry heatmap + ellipsoid estimate, clearly labeled non-specific. No zero-filling. |
| "How do you know the ONNX model is correct?" | Verified against PyTorch with identical inputs — max abs diff 6.71e-08. |
| "Is this FDA-cleared for clinical use?" | No. Research visualization only. Requires radiologist verification. |
| "What if the BraTS-ifier N4 step produces bad intensities?" | Guarded — only applied if p1/p99 stays in [0.40, 2.50], otherwise skipped with warning. |
| "Can this run on a CPU-only hospital workstation?" | Yes — ONNX Runtime CPU, no GPU required. |
| "How are tumor volumes calculated?" | Per-label voxel count × voxel spacing³, converted to cm³, using SpatialMetadata from ingestion. |
| "What are the model limitations?" | Trained on BraTS 2021 (adult glioma). Degrades on post-op, hemorrhage, pediatric, low-quality, metastases, meningioma. |

---

## Files to Have Ready for Demo

| File | Purpose |
|------|---------|
| `app.py` | Gradio web UI |
| `model.onnx` | Exported ONNX model |
| Test NIfTI files (4 modalities) | Full pipeline demo |
| Test DICOM ZIP (2 modalities) | Partial scan demo |
| `tests/test_*.py` | Test suite (62 tests) |
| `scripts/validate_metrics.py` | Numerical validation |
The Gradio app will launch at `http://localhost:7860` (or the port shown in terminal).