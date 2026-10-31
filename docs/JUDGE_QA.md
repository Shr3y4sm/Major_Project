# Judges' Q&A — BraTS Brain Tumor Segmentation MVP

**Project:** Major_Project — BraTS 3D U-Net with clinical-grade ingestion, partial-scan fallback, ONNX deployment, MPR/3D visualization, and surgical reporting.

**Total questions:** ~40, organized by category. Each answer includes the specific source file and line reference for verification.

---

## A. General / Project Overview

### A1. What does this project do?

**Answer:** It is a web-based clinical decision-support tool for brain tumor segmentation from multi-modal MRI. Users upload NIfTI files or DICOM ZIP archives containing T1, T1ce, T2, and FLAIR modalities. The system runs a trained 3D U-Net model (via ONNX Runtime) to segment tumor sub-regions (necrotic core, edema, enhancing tumor), computes volumetry in cm³ and RANO-style diameters, generates MPR slices and 3D tumor meshes, and produces a printable HTML surgical/radiology report.

**Key files:** `app.py` (UI), `src/inference.py` (ONNX engine), `src/model.py` (3D U-Net)

### A2. Why BraTS? What dataset is the model trained on?

**Answer:** BraTS (Brain Tumor Segmentation) is the standard benchmark for glioma segmentation. The model is trained on BraTS 2021 data with 4 modalities (T1, T1ce, T2, FLAIR) and 4 classes (background, necrotic core, edema, enhancing tumor). The native BraTS label 4 is remapped to contiguous index 3.

**Key files:** `src/preprocessing.py` (LABELS, MODALITIES constants), `src/model.py` (output classes)

### A3. Is this for clinical diagnosis?

**Answer:** No. This is a **research visualization tool only**. Prominent disclaimers state: "Research visualization only — not for diagnosis or treatment decisions. Requires radiologist verification. Model trained on BraTS; performance degrades on post-op, hemorrhage, pediatric, or low-quality acquisitions."

**Key files:** `src/report.py` (disclaimer in HTML), `app.py` (UI warning banner)

### A4. What are the main innovations / contributions?

**Answer:**
1. **Honest partial-scan handling** — Instead of zero-filling missing modalities (which produces fake segmentations), the system detects incomplete modality sets and runs a partial-analysis fallback (asymmetry heatmap + ellipsoid volume estimate), explicitly labeled as non-specific.
2. **Guardrailed BraTS-ifier** — Every preprocessing step (N4 bias correction, gradient-NCC registration, shape harmonization, centering) has safety guards. If a step cannot be safely applied, it is skipped with a warning rather than failing or producing bad output.
3. **ONNX deployment equivalence** — PyTorch model exported to ONNX with verified max abs diff of 6.71e-08, enabling CPU-only deployment.
4. **PHI-safe DICOM ingestion** — Automatic anonymization (tag removal, ID hashing) before processing.


## B. Model & Architecture

### B1. What is the model architecture?

**Answer:** 3D U-Net with encoder-decoder structure and skip connections.
- Input: `(1, 4, 128, 128, 128)` — batch=1, 4 channels (T1, T1ce, T2, FLAIR), 128³ voxels
- Output: `(1, 4, 128, 128, 128)` — 4 class logits (background, necrotic core, edema, enhancing tumor)
- Approx. 23 million parameters
- Activations: ReLU; output: softmax
- Framework: PyTorch, exported to ONNX opset 13

**Key files:** `src/model.py` (BraTSUNet, DoubleConv3D, Down3D, Up3D)

### B2. How are BraTS labels handled?

**Answer:** BraTS native labels are 0 (background), 1 (necrotic core), 2 (edema), 4 (enhancing tumor). The system remaps label 4 → contiguous index 3 so the output is 0,1,2,3. All downstream code (metrics, volumetry, mesh export) uses this contiguous scheme. Native label 4 is merged into label 3 for mesh export.

**Key files:** `src/preprocessing.py` (`load_patient_volume` line ~210: `seg[seg == 4] = 3`), `src/mesh_export.py` (`meshes_from_segmentation` merges label 4 into label 3)

### B3. What loss function was used?

**Answer:** Dice loss + Cross-Entropy (standard BraTS training objective). The inference code computes Dice coefficient and HD95 for evaluation.

**Key files:** `src/inference.py` (Dice/HD95 computation), `src/model.py` (model definition — loss is in training notebook, not in inference code)

### B4. What is the input/output resolution?

**Answer:** 128 × 128 × 128 voxels (isotropic after resampling). The preprocessing pipeline crops to the T2 brain-mask bounding box, then resamples each modality to (128, 128, 128) using linear interpolation (order=1).

**Key files:** `src/preprocessing.py` (`TARGET_SIZE = 128`, `resample_volume`, `crop_to_bbox`)

### B5. Can the model run on GPU?

**Answer:** The deployment path uses ONNX Runtime on CPU. The PyTorch model can train on GPU, but inference is CPU-only via ONNX, making it deployable on standard hospital workstations without GPU.

**Key files:** `src/inference.py` (ONNX Runtime CPU session), `src/export_onnx.py` (export to ONNX)

## C. BraTS-ifier Pipeline (bratsify.py)

### C1. What does the BraTS-ifier do?

**Answer:** It is a guardrailed pre-inference harmonization pipeline that brings arbitrary hospital MRI onto the BraTS reference grid. Four steps:
1. **Affine-aware rigid registration** — gradient-NCC registration of each modality to T2 reference
2. **N4 bias field correction** — SimpleITK N4ITK (max 50 iterations, 4 fitting levels)
3. **Shape harmonization** — resample all modalities to match T2 reference shape
4. **BraTS canvas centering** — pad/crop to (240, 240, 155) BraTS canonical shape

**Key files:** `src/bratsify.py` (all four steps, BratsifyConfig, BratsifyReport)

### C2. What are the guards / safety mechanisms?

**Answer:** Every step has a safety guard:
- **N4:** Applied only if post-correction intensity percentile range [p1, p99] stays within [0.40, 2.50]; otherwise skipped
- **Registration:** Only applied if mean NCC improves by ≥ 0.005 threshold
- **Harmonization:** Max resample factor 3× per axis to prevent extreme distortion
- **Centering:** Standard BraTS dimensions, always safe
- **Overall:** Each step is wrapped in try/except; failures produce warnings, not crashes

**Key files:** `src/bratsify.py` (`_guard_n4_range`, `_rigid_register_gradient_ncc`, `harmonize_shapes`, `center_to_brats_canvas`)

### C3. What were the measured results of the BraTS-ifier?

**Answer:** From `validate_metrics.py`:
- N4 bias correction: correctly skipped when out of range (guard working)
- Affine registration: NCC improved from −0.04 to +0.63 (41 → 1 voxel misalignment)
- Placeholder detector: catches `.PLACEHOLDER` markers in dummy data

**Key files:** `scripts/validate_metrics.py`, `src/bratsify.py`

### C4. Why gradient-NCC for registration?

**Answer:** Gradient-normalized cross-correlation (gradient-NCC) is robust to intensity differences between modalities (e.g., T1 vs T2 have different contrast). It operates on image gradients rather than raw intensities, making it suitable for multi-modal registration where absolute intensity values differ but structural edges align.

**Key files:** `src/bratsify.py` (`_rigid_register_gradient_ncc`)

### C5. What happens if BraTS-ifier steps fail?

**Answer:** They are skipped with a warning. The pipeline never fails hard — it produces the best-effort output with a `BrastifyReport` documenting which steps were applied/skipped and why.

**Key files:** `src/bratsify.py` (`BrastifyReport`, `bratsify_scans`)

## D. Ingestion & Data Loading

### D1. What input formats are supported?

**Answer:** Two input sources:
1. **NIfTI** — dict or list of 4 paths (T1, T1ce, T2, FLAIR) as `.nii` or `.nii.gz`
2. **DICOM** — root directory or ZIP containing multi-slice DICOM series (converted to NIfTI via `dicom2nifti`)

**Key files:** `src/dicom_loader.py` (`load_patient_scans`), `src/io_unified.py` (`load_unified`)

### D2. How are DICOM modalities classified?

**Answer:** By series description substring matching, with ordered token evaluation. `t1ce` is evaluated before `t1` (and longer tokens before shorter ones) because matching is substring-based — "AX T1C" would otherwise be misclassified as `t1`.

**Token examples:**
- t1ce: t1ce, t1c, t1 post, t1 contrast, t1-gad, t1gad, postgad, post-gad, post gad, gad
- t1: t1, t1w, t1 weighted, mprage, t1 ffe, t1ffe
- t2: t2, t2w, t2 weighted, t2w_tse, t2 tse, t2tse
- flair: flair, t2 flair, t2flair

**Key files:** `src/dicom_loader.py` (`_MODALITY_TOKENS`)

### D3. What is skipped during DICOM ingestion?

**Answer:** Scout/localizer/survey series (tokens: survey, scout, localizer, loc, topogram, plan, calibration, mpr, mst, pilot) are skipped — not treated as errors.

**Key files:** `src/dicom_loader.py` (`_SCOUT_TOKENS`)

### D4. What validation is performed on ingestion?

**Answer:**
- Modality completeness (all 4 required for full inference)
- Spatial orientation consistency across modalities
- Slice continuity (no missing slices in DICOM series)
- Uniform matrix dimensions across modalities

**Key files:** `src/dicom_loader.py` (`_validate_orientation`, `_validate_uniform_shape`, `load_patient_scans`)

### D5. What exceptions can be raised?

**Answer:** `IngestionError`, `MissingModalityError`, `OrientationMismatchError`, `SliceContinuityError`, `ShapeMismatchError`, `DICOMCorruptError`

**Key files:** `src/dicom_loader.py` (exception classes)

## E. Partial-Scan Handling (Tier 2)

### E1. What happens when not all 4 modalities are available?

**Answer:** The system does **not** zero-fill or fabricate a 4-channel input. Instead, it runs the partial-analysis fallback in `src/partial_analysis.py`:
1. **Asymmetry heatmap** — mirror left/right hemispheres, report normalized |V − flip(V)| as a non-specific hyperintensity map
2. **Ellipsoid estimate** — geometric volume (π/6 × D1×D2×D3) from thresholded heat region bounding box
3. **Blend overlay** — orange heat for MPR viewer visualization

All outputs are explicitly labeled as **non-specific** and **not a tumor segmentation**.

**Key files:** `src/partial_analysis.py` (asymmetry_heatmap, ellipsoid_estimate, blend_heat_overlay)

### E2. Why not zero-fill missing modalities?

**Answer:** Enhancing tumor is defined by T1ce contrast and edema by FLAIR. Zero-filling would create artificial signal patterns that the U-Net could misinterpret as tumor, producing false positives. The honest fallback provides geometric estimates without claiming AI segmentation.

**Key files:** `src/partial_analysis.py` (module docstring explains this design decision)

### E3. What guards prevent false positives in the ellipsoid estimate?

**Answer:** If the hot-voxel bounding box spans ≥ 90% of any axis (global asymmetry from normal anatomy, acquisition gradient, or motion), the function returns `found=False` with an explanatory note instead of reporting a whole-head "volume."

**Key files:** `src/partial_analysis.py` (`ellipsoid_estimate`, `max_bbox_frac = 0.9` guard)

### E4. What is the PACS protocol checklist?

**Answer:** Requests from PACS: AX T1 (pre-contrast), AX T1C/T1-GAD post-contrast, AX T2/T2W, AX FLAIR — all as DICOM (not screenshots), one series per modality, avoiding scout-only exports.

**Key files:** `src/partial_analysis.py` (`PROTOCOL_CHECKLIST_MD`)

## F. ONNX & Deployment

### F1. Why export to ONNX?

**Answer:** ONNX Runtime enables CPU-only inference without PyTorch dependency, faster startup, smaller deployment footprint, and cross-platform compatibility. It also allows verification of numerical equivalence with the PyTorch model.

**Key files:** `src/export_onnx.py`, `src/verify_onnx.py`, `src/inference.py`

### F2. How is ONNX↔PyTorch equivalence verified?

**Answer:** `verify_onnx.py` runs identical random input through both the PyTorch model and the ONNX session, then reports max absolute difference. Measured: **6.71e-08**, well within float32 tolerance.

**Key files:** `src/verify_onnx.py` (max abs diff = 6.71e-08)

### F3. What ONNX opset version is used?

**Answer:** Opset 13, with dynamic axes for batch size.

**Key files:** `src/export_onnx.py`

### F4. Can the model run without a GPU?

**Answer:** Yes. ONNX Runtime CPU execution is the deployment path. No GPU required.

**Key files:** `src/inference.py` (onnxruntime.InferenceSession with CPU provider)

## G. Metrics & Validation

### G1. What metrics are computed?

**Answer:**
- **Dice coefficient** per tumor class (necrotic core, edema, enhancing tumor, whole tumor)
- **HD95** (95th percentile Hausdorff distance) per class
- **Volumetry** per label in cm³ (using voxel spacing from SpatialMetadata)
- **RANO-style diameters** (dmax_mm, dperp_mm) on axial whole-tumor slices

**Key files:** `src/inference.py` (compute_metrics, compute_volumes), `src/postprocess.py` (rano_diameters, rano_classification)

### G2. What are the Dice formulas?

**Answer:** Dice = 2 × |X ∩ Y| / (|X| + |Y|) where X is prediction and Y is ground truth. Computed per class after connected-component cleanup.

**Key files:** `src/inference.py` (dice_coefficient), `src/postprocess.py` (cleanup_prediction)

### G3. How is HD95 computed?

**Answer:** 95th percentile of the Hausdorff distance — robust to outliers compared to standard Hausdorff distance. Computed using surface distances between predicted and ground truth masks.

**Key files:** `src/inference.py` (hd95 computation)

### G4. How many tests pass?

**Answer:** **62/62 tests passing** across `tests/test_bratsify.py` and `tests/test_inference.py`.

**Key files:** `tests/test_bratsify.py`, `tests/test_inference.py`

### G5. What does validate_metrics.py check?

**Answer:** End-to-end numerical validation: Bratsify guards (N4 range, registration improvement, placeholder detection), registration quality (NCC gain, voxel misalignment reduction), and pipeline integrity.

**Key files:** `scripts/validate_metrics.py`

## H. Post-Processing & Reporting

### H1. What post-processing is applied?

**Answer:** Connected-component cleanup — removes components < 50 voxels per tumor class (necrotic core, edema, enhancing tumor) while keeping the whole-tumor mask stable.

**Key files:** `src/postprocess.py` (`cleanup_prediction`, `min_voxels = 50`)

### H2. What is the tumor flag / normal-scan banner?

**Answer:** Based on whole-tumor (WT) volume:
- WT < 0.05 cm³: "No measurable tumor — likely normal scan or below model resolution — radiologist review required"
- WT < 0.3 cm³: "Tiny finding — possible noise / small focus — confirm on MPR"
- WT ≥ 0.3 cm³: no banner (normal finding)

**Key files:** `src/postprocess.py` (`tumor_flag`, `threshold_cm3 = 0.3`)

### H3. What is RANO classification?

**Answer:** RANO-style response call from change in whole-tumor SPD (sum of perpendicular diameters, approximated as dmax_mm × dperp_mm):
- Progressive: ≥ +25% increase vs prior
- Partial response: ≤ −50% decrease vs prior
- Stable: otherwise
- No prior measurable lesion: "New / no prior measurable lesion"

**Key files:** `src/postprocess.py` (`rano_classification`)

### H4. What does the HTML report contain?

**Answer:** One-page printable report with: volumetry table (per-label cm³), RANO diameters, MPR snapshots (base64-embedded PNGs), ingestion/provenance notes, comparison to prior study (optional), model-review explanation (optional), and prominent research-only disclaimer.

**Key files:** `src/report.py` (`write_html_report`)

### H5. What 3D exports are available?

**Answer:** STL (binary, zero new dependencies) per tumor label, GLB (optional, via trimesh), and NIfTI segmentation map reusing reference affine/header. Label colors: necrotic core (#FF004D red), edema (#00FF4D green), enhancing tumor (#00D0FF cyan).

**Key files:** `src/mesh_export.py` (write_stl_binary, export_label_meshes, export_nifti_seg, try_export_glb)

## I. Clinical Safety & PHI

### I1. How is PHI handled?

**Answer:** DICOM PHI tags are removed before processing: PatientName, PatientID, PatientBirthDate, PatientSex, PatientAge, PatientAddress, PatientTelephoneNumbers, InstitutionName, InstitutionAddress, ReferringPhysicianName, PerformingPhysicianName, OperatorsName, PhysiciansOfRecord, StudyID, AccessionNumber. PatientID is hashed (SHA-256 with salt, truncated to 12 chars, prefixed "ANON-"). Group-0000 command elements are also dropped.

**Key files:** `src/anonymize.py` (`_REMOVE_KEYWORDS`, `_hash_id`, `anonymize_dicom_tree`, `anonymize_dicom_zip`)

### I2. What is placeholder detection?

**Answer:** `.PLACEHOLDER` markers in DICOM/NIfTI data are detected and trigger rejection — prevents processing dummy or incomplete data as if it were real patient scans.

**Key files:** `src/dicom_loader.py` (placeholder detection in ingestion), `src/bratsify.py` (placeholder check)

### I3. What are the known model limitations?

**Answer:** Model trained on BraTS 2021 (adult glioma). Performance degrades on: post-op scans, hemorrhage, pediatric tumors, low-quality acquisitions, metastases, meningioma. Not validated for clinical use.

**Key files:** `src/report.py` (disclaimer text), `app.py` (warning banner)

### I4. Is planning registration clinically validated?

**Answer:** No. `planning_registration.py` performs affine resampling only. Status is `affine-resampled-review-required` with `review_required=True`. Rigid/deformable optimization and surgeon-reviewed QA are separate steps.

**Key files:** `src/planning_registration.py` (`PlanningRegistrationResult`, status field)

## J. Code Quality & Engineering

### J1. What programming practices are used?

**Answer:**
- Type hints throughout (PEP 484)
- Docstrings on all public functions (Google/NumPy style)
- `__all__` exports in every module
- Custom exception hierarchy
- Dataclasses for structured results (frozen where appropriate)
- Context managers for temp dirs and file handles
- `from __future__ import annotations` for forward references

**Key files:** All `src/*.py` files

### J2. How are errors handled?

**Answer:** Custom exception hierarchy (`IngestionError`, `MissingModalityError`, etc.), try/except with graceful degradation (BraTS-ifier steps skip with warnings), validation before processing, and informative error messages.

**Key files:** `src/dicom_loader.py` (exceptions), `src/bratsify.py` (guarded steps), `src/io_unified.py` (try/except with warnings)

### J3. Is there CI/CD configuration?

**Answer:** Tests are run via pytest. `scripts/validate_metrics.py` provides end-to-end numerical validation. Docker and GitHub Actions configurations would be deployment-specific additions.

**Key files:** `tests/`, `scripts/validate_metrics.py`

### J4. What dependencies are required?

**Answer:** Core: numpy, nibabel, scipy, scikit-image, onnxruntime, torch, pydicom, dicom2nifti, gradio, plotly, pyvista. Optional: trimesh (GLB export), SimpleITK (N4 correction in bratsify).

**Key files:** `requirements.txt` (if present), imports throughout `src/`

### J5. How is the frontend built?

**Answer:** Gradio web UI with PyVista for 3D tumor visualization and Plotly for interactive MPR slices. Components: upload zone, mode toggle (full/partial), processing status, MPR viewer, 3D view, volumes table, download panel, warning banner.

**Key files:** `app.py` (Gradio interface)

## K. Demo-Specific Questions

### K1. What is the end-to-end demo flow?

**Answer:** Upload 4-modality NIfTI → ingestion → BraTS-ifier → preprocessing → ONNX inference → volumetry → MPR/3D visualization → STL/NIfTI-seg/HTML report download. Or upload DICOM ZIP (partial scan) → honest fallback → asymmetry heatmap → PACS checklist.

**Key files:** `app.py` (orchestrates full flow)

### K2. How long does inference take?

**Answer:** Depends on CPU; typically seconds for a 128³ volume on ONNX Runtime CPU. No GPU required.

**Key files:** `src/inference.py` (ONNX session run)

### K3. Can I see the raw model output?

**Answer:** Yes — the ONNX engine returns class probabilities before argmax, and the report includes per-label volumes. MPR slices show the segmentation overlay.

**Key files:** `src/inference.py` (predict returns probabilities + label map), `app.py` (MPR visualization)

## L. Quick Reference: Where to Find Code

| Topic | File | Key Functions/Classes |
|-------|------|----------------------|
| Model architecture | `src/model.py` | `BraTSUNet`, `DoubleConv3D`, `Down3D`, `Up3D` |
| ONNX inference | `src/inference.py` | `ONNXInferenceEngine`, `compute_metrics`, `compute_volumes` |
| Preprocessing | `src/preprocessing.py` | `brain_mask_from_t2`, `crop_to_bbox`, `resample_volume`, `zscore_normalize` |
| BraTS-ifier | `src/bratsify.py` | `bratsify_scans`, `BrastifyConfig`, `BrastifyReport`, N4 guard, registration |
| DICOM ingestion | `src/dicom_loader.py` | `load_patient_scans`, `PatientScans`, `SpatialMetadata`, modality tokens |
| Unified ingestion | `src/io_unified.py` | `load_unified`, `infer_nifti_modality`, `discover_nifti_files` |
| Partial-scan fallback | `src/partial_analysis.py` | `asymmetry_heatmap`, `ellipsoid_estimate`, `blend_heat_overlay` |
| Mesh export | `src/mesh_export.py` | `meshes_from_segmentation`, `write_stl_binary`, `export_label_meshes`, `try_export_glb` |
| Post-processing | `src/postprocess.py` | `cleanup_prediction`, `rano_diameters`, `tumor_flag`, `rano_classification` |
| PHI anonymization | `src/anonymize.py` | `anonymize_dicom_tree`, `anonymize_dicom_zip`, `_hash_id` |
| HTML report | `src/report.py` | `write_html_report` |
| Planning registration | `src/planning_registration.py` | `prepare_planning_volume`, `extract_b0` |
| ONNX export | `src/export_onnx.py` | ONNX model export |
| ONNX verification | `src/verify_onnx.py` | PyTorch↔ONNX equivalence (6.71e-08) |
| Web UI | `app.py` | Gradio interface, MPR viewer, 3D view |
| Tests | `tests/test_*.py` | 62 tests, all passing |
| Validation | `scripts/validate_metrics.py` | Numerical validation of guards and registration |
**Key files:** `src/partial_analysis.py`, `src/bratsify.py`, `src/verify_onnx.py`, `src/anonymize.py`