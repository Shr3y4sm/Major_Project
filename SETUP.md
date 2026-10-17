# Setup & Run — Commands

This is the standalone command reference for installing dependencies, running
the app, and running the test suite. The development environment used here is
**Python 3.12.3**.

> Work in the repo root, e.g. `/media/shreyas/E/Major_Project`. This is the
> active working copy (has the `.venv`, `artifacts/` weights, and sample data).

---

## 1. Activate the virtualenv

```bash
cd /media/shreyas/E/Major_Project
source .venv/bin/activate
```

## 2. Install / verify requirements

**Fresh install** (loose constraints — resolves latest compatible versions):

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Reproduce the exact tested environment** (recommended — pinned):

```bash
python -m pip install -r requirements.lock.txt
```

**Verify only** (make no changes; confirm everything is already installed):

```bash
python -m pip check
python -c "import gradio, torch, nibabel, pydicom, SimpleITK, plotly, onnxruntime; print('all deps OK')"
```

## 3. Run the app

**Quick start** (defaults: `BRATSIFY=1`, port 7860):

```bash
cd /media/shreyas/E/Major_Project
PYTHONPATH=. .venv/bin/python app.py
# -> http://127.0.0.1:7860
```

**One-command evaluator demo** (sets env vars, prints a banner):

```bash
bash demo/demo_run.sh
```

**Environment overrides:**

```bash
BRATSIFY=0 APP_HOST=127.0.0.1 APP_PORT=7861 bash demo/demo_run.sh
```

- `BRATSIFY=1` (default) → BraTS standardiser pipeline (N4 + registration + canvas)
- `BRATSIFY=0` → legacy path
- `APP_HOST` / `APP_PORT` → bind address and port

## 4. Run the tests

```bash
cd /media/shreyas/E/Major_Project
PYTHONPATH=. .venv/bin/python -m pytest tests/ -q
```

Run a single suite:

```bash
PYTHONPATH=. .venv/bin/python tests/test_compare_prior.py
```

Suites: `test_unified`, `test_partial`, `test_dicom_loader`, `test_bratsify`,
`test_compare_prior`, `test_app_volumetry`.

---

## Environment notes

- **Model weights** live in `artifacts/` (git-ignored). The app runs best with
  the trained `best_unet.pt` → export ONNX via
  `python -m src.export_onnx`, then verify with `python -m src.verify_onnx`.
- **Sample data** lives in `sample_brats_patient/` and `BRATS/` (real local
  directories, git-ignored for PHI safety), plus `DICOM.zip`.
- **Known test note:** `test_model_health_warning_detects_placeholder_output`
  uses a synthetic marker so it passes whether placeholder or trained weights
  are installed.
- **Docker (optional):** `docker build -t brats-seg .` then
  `docker run --rm -p 7860:7860 -v /path/artifacts:/app/artifacts:ro brats-seg`.
