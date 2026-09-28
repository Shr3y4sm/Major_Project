"""UI smoke test: real HTTP round-trip through the Gradio app.

Uploads the four BraTS2021_00495 NIfTIs (id-bearing filenames) to the
running server, invokes the actual Run-button handler (/run_analysis),
and asserts the rendered outputs:
  * metrics name the id-matched ground-truth sidecar,
  * all three tumour classes report EXACT volumes (1mm iso counts/1000),
  * placeholder model-health warning is visible (not silent),
  * report + downloads are produced, figures are non-empty.
"""
import os
import re
import sys

from gradio_client import Client, handle_file

SERVER = "http://127.0.0.1:7860"
UP = "/tmp/ui_uploads"

# Ground truth for BraTS2021_00495 (labels {1,2,4} @ 1mm iso).
EXPECTED = {"Necrotic Core": 20.808, "Peritumoral Edema": 46.447,
            "Enhancing Tumor": 32.047}

files = {m: os.path.join(UP, f"BraTS2021_00495_{m}.nii") for m in
         ("t1", "t1ce", "t2", "flair")}
for p in files.values():
    assert os.path.isfile(p), p

print("connecting to", SERVER)
client = Client(SERVER, verbose=False, httpx_kwargs={"timeout": 600.0})

result = client.predict(
    "NIfTI Files",
    handle_file(files["t1"]),
    handle_file(files["t1ce"]),
    handle_file(files["t2"]),
    handle_file(files["flair"]),
    None,   # nifti_folder (zip override)
    None,   # zip_file (DICOM)
    78, 120, 120,          # axial, sagittal, coronal
    "FLAIR",               # bg_modality
    0.8,                   # overlay_opacity
    [1, 2, 3],             # show_classes
    50,                    # cleanup_min_vox
    True,                  # anonymize
    True,                  # show_heat
    98.0,                  # heat_percentile
    api_name="/run_analysis",
)

plot_3d, plot_mpr, metrics_md, warn_md, report, downloads = result

print("=== METRICS ===")
print(metrics_md)
print("=== WARN (head) ===")
print((warn_md or "")[:400])

# 1. Source: id-matched sidecar for the uploaded patient.
assert "ground-truth sidecar" in metrics_md, metrics_md
assert "BraTS2021_00495_seg.nii" in metrics_md, \
    "sidecar must be id-matched to the uploaded patient"

# 2. Exact per-class volumes.
parsed = {m.group(1).strip(): float(m.group(2)) for m in
          re.finditer(r"\| ([A-Za-z ]+) \| ([\d.]+) \|", metrics_md)}
print("parsed:", parsed)
for cls, exp in EXPECTED.items():
    assert abs(parsed[cls] - exp) < 0.02, (cls, parsed[cls], exp)

# 3. Placeholder warning visible in BOTH panels.
assert "Model health" in metrics_md, "warning missing from metrics"
assert "Model health" in (warn_md or ""), "warning missing from warn panel"

# 4. Artifacts produced.
assert report and os.path.isfile(str(report).split(" ")[0].strip("'\"")) or \
    report is not None, "HTML report missing"
assert downloads, "no downloads returned"
print("downloads:", len(downloads) if isinstance(downloads, list) else downloads)

# 5. Figures: Gradio 6 returns {"type": "plotly", "plot": "<json string or spec path>"}.
import json as _json

def _fig_traces(fig):
    """Return the set of trace names from a Gradio Plot payload, or None."""
    if not isinstance(fig, dict) or not fig.get("plot"):
        return None
    payload = fig["plot"]
    if isinstance(payload, str) and os.path.isfile(payload):
        with open(payload, encoding="utf-8") as fh:
            spec = _json.load(fh)
    elif isinstance(payload, str):
        try:
            spec = _json.loads(payload)
        except _json.JSONDecodeError:
            return None
    elif isinstance(payload, dict):
        spec = payload
    else:
        return None
    if isinstance(spec, dict) and "data" in spec:
        data = spec["data"]
    elif isinstance(spec, list):
        data = spec
    else:
        return None
    return {t.get("name") for t in data if isinstance(t, dict)}

traces_3d = _fig_traces(plot_3d)
assert traces_3d, f"3D figure empty: {plot_3d!r}"
assert "Enhancing Tumor" in traces_3d, f"no enhancing trace: {traces_3d}"
assert "Necrotic Core" in traces_3d, f"no necrotic trace: {traces_3d}"
traces_mpr = _fig_traces(plot_mpr)
assert traces_mpr is not None, f"MPR figure empty: {plot_mpr!r}"

print("UI SMOKE TEST PASSED")