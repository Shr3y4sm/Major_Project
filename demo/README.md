# Evaluator Demo Guide (5 minutes)

## 1. Start the app — one command

```bash
bash demo/demo_run.sh
# -> open http://127.0.0.1:7860
```

## 2. Full tumour-segmentation flow (3 min)

1. Tab **NIfTI Files** -> upload the 4 files from
   `sample_brats_patient/BraTS2020/`
   (`BraTS20_Training_001_t1`, `_t1ce`, `_t2`, `_flair`).
2. Press **Run Analysis** and narrate the pipeline:
   *ingest -> BraTS-ifier (1 mm resample -> align -> N4 fix -> canvas)
   -> 3D U-Net -> volumes + 3D + report.*
3. Show the **MPR tab** (overlay + slice sliders), the **3D tab**
   (rotate the tumour in-situ), the **metrics panel** (cm^3 + RANO),
   and the **Exports** (STL mesh, `segmentation.nii.gz`, HTML report,
   QA PNG, provenance JSON).

## 3. Hospital-input credibility (1 min)

Switch to **DICOM ZIP** -> upload `DICOM.zip` -> Run.
Say first: *"real Philips scan, but only T1+T2 and healthy —
watch what a responsible tool does."*
The app shows anatomy-only views plus a banner naming the missing
modalities (`t1ce`, `flair`) and a PACS request checklist.

## 4. Offline fallback (no network / upload fails)

```bash
PYTHONPATH=. .venv/bin/python scripts/make_synth_dicom_patient.py \
  --src sample_brats_patient/BraTS2021_00495 \
  --out /tmp/synth_dicom_patient --rows 64 --cols 64 --step 5
cd /tmp && zip -qr synth_dicom_patient.zip synth_dicom_patient
# upload the ZIP in DICOM mode instead.
```

## 5. One-line close

> "Raw hospital MRI in, BraTS-grade auto-segmentation and surgical 3D
> out — minutes instead of manual contouring hours, with honesty
> guardrails when the scan is incomplete."
