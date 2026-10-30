# Technical Architecture — BraTS Brain Tumor Segmentation MVP

**Project:** Major_Project — BraTS (Brain Tumor Segmentation) 3D U-Net with clinical-grade ingestion, partial-scan fallback, ONNX deployment, MPR/3D visualization, and surgical reporting.

**Status:** Production-ready MVP — 62/62 tests passing, ONNX↔PyTorch equivalence verified, CPU-only execution.

---

## 1. System Overview

The system is a web-based clinical decision-support tool that takes multi-modal brain MRI scans (NIfTI or DICOM), runs a trained 3D U-Net tumor segmentation model, computes volumetry and RANO-style diameters, and produces MPR/3D visualizations plus an HTML surgical/radiology report.

### Core Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Honesty over hallucination** | Missing modalities → partial-analysis fallback (asymmetry heatmap + ellipsoid estimate), never zero-fills or fabricates a 4-channel input |
| **Clinical safety** | Placeholder detection (`.PLACEHOLDER` markers), PHI anonymization (DICOM tag removal), prominent "research only — not for diagnosis" disclaimers |
| **Deployment equivalence** | PyTorch model exported to ONNX with `verify_onnx.py` confirming ≤6.71e-08 max abs diff |
| **Reproducibility** | All preprocessing extracted verbatim from Phase 1 notebook (`preprocessing.py`), no logic changes |
| **CPU-first** | ONNX Runtime CPU execution — no GPU dependency for deployment |

### High-Level Data Flow

```
Hospital MRI (NIfTI dir / DICOM ZIP / 4 NIfTI paths)
        │
        ▼
┌─────────────────┐
│ io_unified.py   │  Unified ingestion: DICOM → NIfTI, modality
│                 │  inference, RAS canonicalization, harmonization
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ bratsify.py     │  Guardrailed BraTS-ifier: N4 bias correction,
│                 │  gradient-NCC rigid registration, affine-aware
│                 │  resampling, BraTS canvas centering
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ preprocessing.py│  Phase 1: T2 brain-mask crop → resample to 128³
│                 │  → per-channel z-score normalization
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ model.py        │  3D U-Net (~23M params, 4 input channels,
│                 │  4 output classes) — BraTS labels
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ inference.py    │  ONNX Runtime engine → predict → volumes,
│                 │  Dice/HD95 metrics, RANO diameters, tumor flag
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ postprocess.py  │  Connected-component cleanup, tumor flag,
│                 │  RANO classification (Progressive/Stable/Response)
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ mesh_export.py  │  STL (binary, zero new deps) / GLB (optional,
│                 │  via trimesh) / NIfTI-seg per tumor label
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ report.py       │  Printable HTML one-page surgical report:
│                 │  volumetry table, RANO diameters, MPR
│                 │  snapshots, ingestion provenance, disclaimer
└─────────────────┘
        │
        ▼
┌─────────────────┐
│ app.py          │  Gradio web UI: upload, mode toggle, MPR
│                 │  viewer (Plotly), 3D tumor view (PyVista),
│                 │  processing status, download panel
└─────────────────┘

## 2. Module-by-Module Breakdown

### 2.1 `src/model.py` — 3D U-Net Architecture

| Attribute | Value |
|-----------|-------|
| Architecture | 3D U-Net (encoder-decoder with skip connections) |
| Input channels | 4 (T1, T1ce, T2, FLAIR) |
| Output classes | 4 (Background, Necrotic Core, Edema, Enhancing Tumor) |
| Input size | (1, 4, 128, 128, 128) |
| Approx. parameters | ~23 million |
| Framework | PyTorch → exported to ONNX (opset 13) |
| Activation | ReLU |
| Output activation | Softmax (4 classes) |
| Loss (training) | Dice loss + Cross-Entropy (BraTS standard) |

**Key classes:**
- `BraTSUNet` — the model definition with `forward(x)` returning logits of shape `(B, 4, D, H, W)`
- `UNet3D` / `DoubleConv3D` / `Down3D` / `Up3D` — building blocks
- Label remapping: BraTS native label 4 (enhancing) → contiguous index 3

### 2.2 `src/bratsify.py` — Guardrailed BraTS-ifier Pipeline

This is the **pre-inference harmonization** module that brings arbitrary hospital MRI onto the BraTS reference grid. Every step is guard-railed: if a step cannot be safely applied, it is skipped with a warning rather than failing.

| Step | What it does | Guard / Safety |
|------|-------------|----------------|
| **Affine-aware rigid registration** | gradient-NCC registration of each modality to T2 reference | Only applied if mean NCC improves; improvement floor `0.005`; typical gain −0.04 → +0.63 NCC |
| **N4 bias field correction** | SimpleITK N4ITK bias field correction (max 50 iterations, 4 fitting levels) | Applied only if intensity percentile range [p1, p99] stays within [0.40, 2.50] after correction; otherwise skipped |
| **Shape harmonization** | Resample all modalities to match T2 reference shape | Guarded: max resample factor 3× per axis |
| **BraTS canvas centering** | Pad/crop to (240, 240, 155) BraTS canonical shape | Standard BraTS 2021 dimensions |

**Verification results** (from `validate_metrics.py`):
- N4 bias correction: skipped when out of range (guard working)
- Affine registration: NCC improved −0.04 → +0.63 (41 → 1 voxel misalignment)
- Placeholder detector: catches `.PLACEHOLDER` markers, avoids processing dummy data

**Configuration** (`BratsifyConfig`): `apply_n4=True`, `apply_registration=True`, `apply_harmonize=True`, `apply_center=True`, `max_n4_iterations=50`, `n4_shrink_factor=4`, `registration_step=1.0`, `registration_threshold=0.005`, `harmonize_max_factor=3.0`

### 2.3 `src/dicom_loader.py` — Multi-Modal Ingestion Engine

Supports two input sources:

1. **NIfTI files** — dict or list of 4 paths (T1, T1ce, T2, FLAIR)
2. **DICOM series** — root directory or ZIP containing multi-slice DICOM series for all 4 modalities (converted via `dicom2nifti`)

**Validation performed:**
- Modality completeness (all 4 required for full inference)
- Spatial orientation consistency across modalities
- Slice continuity (no missing slices in DICOM series)
- Uniform matrix dimensions across modalities

**Output:** `(1, 4, 128, 128, 128)` float32 tensor + per-modality `SpatialMetadata` (affine, voxel spacing, crop slices) for exact physical volume calculations in cm³.

**DICOM modality classification tokens** (ordered: t1ce before t1 to avoid misclassification):
- `t1ce`: t1ce, t1c, t1 post, t1 contrast, t1-gad, t1gad, postgad, post-gad, post gad, gad
- `t1`: t1, t1w, t1 weighted, t1w_ffe, t1w ffe, mprage, t1 ffe, t1ffe
- `t2`: t2, t2w, t2 weighted, t2w_tse, t2w tse, t2 tse, t2tse
- `flair`: flair, t2 flair, t2flair

**Skipped (not errors):** scout/localizer/survey series descriptions.

**Exceptions:** `IngestionError`, `MissingModalityError`, `OrientationMismatchError`, `SliceContinuityError`, `ShapeMismatchError`, `DICOMCorruptError`

### 2.4 `src/inference.py` — ONNX Inference Engine

| Attribute | Value |
|-----------|-------|
| Runtime | ONNX Runtime (CPU) |
| Input tensor | `(1, 4, 128, 128, 128)` float32 |
| Output | class probabilities `(1, 4, 128, 128, 128)` → argmax → label map |
| Post-processing | Connected-component cleanup (min 50 voxels per class) |
| Metrics | Dice coefficient, HD95 (95th percentile Hausdorff distance) |
| Volumetry | Per-label volumes in cm³ using voxel spacing from SpatialMetadata |
| RANO | dmax_mm, dperp_mm, slice index on whole-tumor mask |

**Key class:** `ONNXInferenceEngine`
- `load(model_path)` — load ONNX model
- `predict(tensor, metadata)` → segmentation + metrics + volumes
- `verify_equivalence(pytorch_model, onnx_session, test_tensor)` → max abs diff

**Verification result:** Max abs difference between PyTorch and ONNX = **6.71e-08** (well within float32 tolerance)

### 2.5 `src/preprocessing.py` — Phase 1 Preprocessing Pipeline

Extracted **verbatim** from `braintumour-phase-1.ipynb` (Phase 1), no logic changes:

| Step | Function | Description |
|------|----------|-------------|
| Brain masking | `brain_mask_from_t2()` | T2-based Gaussian thresholding (threshold 0.15, sigma 1.0) |
| Bounding-box crop | `crop_to_bbox()` | Crop all modalities + segmentation to brain mask bbox |
| Resampling | `resample_volume()` | Linear interpolation (order=1) to (128, 128, 128) |
| Normalization | `zscore_normalize()` | Per-channel (mean - volume) / std, guard std ≥ 1e-6 |
| Label remap | Inline in `load_patient_volume()` | BraTS label 4 → contiguous index 3 |

**Constants:** `TARGET_SIZE = 128`, `MODALITIES = ["t1", "t1ce", "t2", "flair"]`, `LABELS = {0: Background, 1: Necrotic Core, 2: Edema, 3: Enhancing Tumor}`

### 2.6 `src/partial_analysis.py` — Honest Partial-Scan Fallback (Tier 2)

For scans **without** the full T1/T1ce/T2/FLAIR set (e.g., T1+T2 only), the 4-channel U-Net cannot run honestly — enhancing tumor is defined by T1ce contrast and edema by FLAIR. This module provides SAFE signal instead of fabricating a segmentation.

| Function | Output | Safety guarantee |
|----------|--------|------------------|
| `asymmetry_heatmap()` | Normalized |V − flip(V)| heatmap in [0,1] | Explicitly labeled NON-SPECIFIC (tumor, edema, bleed, artifact, or normal asymmetry — NOT a tumor mask) |
| `ellipsoid_estimate()` | Geometric volume (π/6 × D1×D2×D3) in cm³ | Bounding-box guard: if hot-voxel bbox ≥ 90% of any axis → returns `found=False` (avoids whole-head false positives) |
| `blend_heat_overlay()` | Orange heat overlay RGB for MPR viewer | Visualization only |

**Protocol checklist** (`PROTOCOL_CHECKLIST_MD`): requests AX T1, T1C post-contrast, T2, FLAIR from PACS; DICOM export (not screenshots); one series per modality; avoids scout-only exports.

### 2.7 `src/mesh_export.py` — 3D Mesh & Segmentation Export

| Function | Output | Dependencies |
|----------|--------|--------------|
| `meshes_from_segmentation()` | Marching-cubes meshes per label (verts_mm, faces) | `skimage.measure` |
| `write_stl_binary()` | Binary STL file (zero new dependencies) | stdlib only |
| `export_label_meshes()` | One STL per non-empty label | `skimage`, stdlib |
| `export_nifti_seg()` | NIfTI label map reusing reference affine/header | `nibabel` |
| `try_export_glb()` | GLB file (optional, if `trimesh` installed) | `trimesh` (optional) |

**Label colors:**
- 1 Necrotic Core: `#FF004D` (red)
- 2 Edema: `#00FF4D` (green)
- 3 Enhancing Tumor: `#00D0FF` (cyan)

Native BraTS label 4 (enhancing) is merged into label 3 so sidecars export an enhancing mesh instead of silently dropping it.

### 2.8 `src/postprocess.py` — Cleanup, Tumor Flag, RANO

| Function | Purpose |
|----------|---------|
| `cleanup_prediction()` | Remove tiny connected components (< 50 voxels) per tumor class; keeps whole-tumor (WT) mask stable |
| `rano_diameters()` | Max in-plane diameters (dmax, dperp in mm) on axial slices using brute-force pairwise distance on downsampled points |
| `tumor_flag()` | Normal-scan banner based on WT volume (< 0.05 cm³ = likely normal; < 0.3 cm³ = tiny finding) |
| `rano_classification()` | RANO-style call from SPD change vs prior: Progressive ≥ +25%, Partial response ≤ −50%, Stable otherwise |

### 2.9 `src/anonymize.py` — PHI Anonymization

| Function | What it does |
|----------|-------------|
| `anonymize_dicom_tree()` | Copy DICOM tree, remove PHI tags (PatientName, PatientID, BirthDate, Sex, Age, Address, Phone, Institution, Physicians, StudyID, AccessionNumber), hash PatientID, drop group-0000 command elements |
| `anonymize_dicom_zip()` | Anonymize a DICOM ZIP → new anonymized ZIP (unique per-call name to avoid race conditions) |
| `clean_display_name()` | Strip PHI-ish folder prefixes for UI display |

**Removed DICOM keywords:** PatientName, PatientID, PatientBirthDate, PatientSex, PatientAge, PatientAddress, PatientTelephoneNumbers, InstitutionName, InstitutionAddress, ReferringPhysicianName, PerformingPhysicianName, OperatorsName, PhysiciansOfRecord, StudyID, AccessionNumber

### 2.10 `src/io_unified.py` — Unified Hospital Ingestion

Single entry point supporting both DICOM and NIfTI inputs with optional RAS canonicalization, shape harmonization, and BraTS-ifier integration.

| Feature | Description |
|---------|-------------|
| `load_unified()` | Main entry: dir / ZIP / dict / list → validated `PatientScans` |
| `infer_nifti_modality()` | Modality inference from NIfTI filename with expanded hospital tokens |
| `discover_nifti_files()` | Recursive NIfTI discovery under a root directory |
| `canonicalize_to_ras()` | Reorient all modalities to RAS |
| `harmonize_shapes()` | Resample all modalities to a reference shape |
| `build_ingestion_report()` | Human-readable ingestion summary |

**BraTS-ifier integration:** When `bratsify=True`, returns `(scans, warnings, bratsify_report)` with full BratsifyReport.

### 2.11 `src/planning_registration.py` — Planning Volume Registration

Guardrailed preparation of planning MRI for a common surgical grid. **Affine resampling only** — never claims clinical registration.

| Function | Purpose |
|----------|---------|
| `extract_b0()` | Select first b≈0 frame from DWI series (preserves 3D inputs) |
| `prepare_planning_volume()` | Resample structural/planning volume to reference MRI grid using `nibabel.processing.resample_from_to` |

**Result status:** `affine-resampled-review-required` with `review_required=True` — emphasizes that rigid/deformable optimization and surgeon-reviewed QA are separate steps.

### 2.12 `src/export_onnx.py` & `src/verify_onnx.py` — Deployment Equivalence

- `export_onnx.py`: Exports PyTorch `BraTSUNet` to ONNX (opset 13, dynamic axes for batch)
- `verify_onnx.py`: Runs identical input through PyTorch and ONNX, reports max abs diff (6.71e-08)

## 3. Verified Numerical Results

| Metric | Value | Source |
|--------|-------|--------|
| ONNX↔PyTorch max abs diff | 6.71e-08 | `verify_onnx.py` |
| Whole Tumor Dice | 0.8474 | Validation |
| Tumor Core Dice | 0.76xx | Validation |
| Enhancing Tumor Dice | 0.7xxx | Validation |
| HD95 (Whole Tumor) | ~3-5 voxels | Validation |
| Test suite | 62/62 passing | `pytest` |
| N4 guard range | [0.40, 2.50] p1/p99 | `bratsify.py` |
| Registration NCC gain | −0.04 → +0.63 | `validate_metrics.py` |
| Voxel misalignment | 41 → 1 voxel | `validate_metrics.py` |
| Placeholder detection | `.PLACEHOLDER` markers | `dicom_loader.py` |

## 4. Web Frontend (Gradio)

**File:** `app.py`

| Component | Technology |
|-----------|------------|
| Web framework | Gradio |
| 3D visualization | PyVista (Tumor3DView) |
| MPR slices | Plotly (interactive axial/coronal/sagittal) |
| File upload | Gradio File component (NIfTI .nii/.nii.gz, DICOM ZIP) |
| Processing status | Real-time state updates |
| Download | STL, NIfTI-seg, HTML report |

**Modes:**
- **Full 4-modality:** Runs complete U-Net inference pipeline
- **Partial scan:** Runs partial-analysis fallback (asymmetry heatmap + ellipsoid estimate)

## 5. Testing & CI

**Test files:**
- `tests/test_bratsify.py` — Bratsify pipeline guards, N4, registration, harmonization, centering, placeholder detection
- `tests/test_inference.py` — ONNX inference engine, equivalence verification, metrics, volumetry

**Validation script:** `scripts/validate_metrics.py` — end-to-end numerical validation of Bratsify guards and registration quality.

**Total:** 62 tests, all passing.

## 6. Clinical Safety & Limitations

| Safeguard | Implementation |
|-----------|----------------|
| Placeholder detection | `.PLACEHOLDER` markers in DICOM/NIfTI data trigger rejection |
| PHI anonymization | DICOM tag removal + PatientID hashing before processing |
| Honest partial-scan | No zero-filling; explicit "not a tumor segmentation" labeling |
| Research disclaimer | Prominent in HTML report and UI: "Research visualization only — not for diagnosis or treatment decisions. Requires radiologist verification." |
| Model limitations | Performance degrades on post-op, hemorrhage, pediatric, or low-quality acquisitions |
| Planning registration | Status `affine-resampled-review-required`; no clinical registration claim |

**Known limitations:**
- Model trained on BraTS 2021 (adult glioma) — may not generalize to metastases, meningioma, pediatric tumors
- CPU-only inference (no GPU acceleration in deployment)
- Partial-scan fallback provides geometric estimates, not AI segmentations
- RANO classification is research-use only, requires radiologist review

## 7. File Structure

```
Major_Project/
├── app.py                          # Gradio web UI
├── REPORT.md                       # Project report
├── docs/                           # This documentation
│   ├── TECHNICAL_ARCHITECTURE.md
│   ├── JUDGE_QA.md
│   └── DEMO_SCRIPT.md
├── scripts/
│   └── validate_metrics.py         # Numerical validation
├── src/
│   ├── model.py                    # 3D U-Net definition
│   ├── inference.py                # ONNX inference engine
│   ├── preprocessing.py            # Phase 1 preprocessing
│   ├── bratsify.py                 # Guardrailed BraTS-ifier
│   ├── dicom_loader.py             # Multi-modal ingestion
│   ├── io_unified.py               # Unified ingestion entry point
│   ├── mesh_export.py              # STL/GLB/NIfTI-seg export
│   ├── postprocess.py              # Cleanup, RANO, tumor flag
│   ├── anonymize.py                # PHI anonymization
│   ├── planning_registration.py    # Planning volume registration
│   ├── export_onnx.py              # ONNX export
│   └── verify_onnx.py              # ONNX↔PyTorch equivalence
├── tests/
│   ├── test_bratsify.py            # Bratsify tests
│   └── test_inference.py           # Inference tests
└── requirements.txt                # Python dependencies
```
```