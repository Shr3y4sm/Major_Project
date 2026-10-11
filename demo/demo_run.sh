#!/usr/bin/env bash
# demo_run.sh — one-command evaluator demo launcher.
# Major Project: Hospital MRI -> BraTS-grade Auto-Segmentation + 3D Visualisation.
#
# Usage:
#   bash demo/demo_run.sh
#
# Env overrides:
#   BRATSIFY=0|1   (default 1 — standardiser pipeline on)
#   APP_HOST       (default 127.0.0.1)
#   APP_PORT       (default 7860)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT"
export BRATSIFY="${BRATSIFY:-1}"
export APP_HOST="${APP_HOST:-127.0.0.1}"
export APP_PORT="${APP_PORT:-7860}"

if [ ! -x .venv/bin/python ]; then
  echo "ERROR: .venv not found. Create it first:"
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt"
  exit 1
fi

echo "==============================================================="
echo " Brain Tumour Segmentation + BraTS-ifier — evaluator demo"
echo " Demo case  : sample_brats_patient/BraTS2020 (BraTS20_Training_001)"
echo " App URL    : http://${APP_HOST}:${APP_PORT}"
echo " BraTS-ifier: ${BRATSIFY} (0 = legacy path, 1 = standardiser)"
echo "==============================================================="
exec .venv/bin/python app.py
