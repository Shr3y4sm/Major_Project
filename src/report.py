"""One-page HTML surgical report (volumes + RANO + snapshots)."""

from __future__ import annotations

import base64
import os
from datetime import datetime, timezone

__all__ = ["write_html_report"]


def _img_b64(path: str) -> str:
    if not path or not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def write_html_report(out_path: str, volumes: dict[str, float],
                      rano: dict, snapshots: list[str],
                      ingestion_notes: str, source: str) -> str:
    """Write a printable HTML report. Returns path."""
    rows = "".join(
        f"<tr><td>{k}</td><td>{v:.2f} cm3</td></tr>"
        for k, v in volumes.items())
    imgs = "".join(
        f"<img src='data:image/png;base64,{_img_b64(p)}' "
        f"style='max-width:32%;margin:2px;border:1px solid #444'/>"
        for p in snapshots if _img_b64(p))
    html = f"""<html><head><meta charset='utf-8'>
<title>Brain Tumour Segmentation Report</title></head>
<body style='font-family:sans-serif;max-width:900px;margin:auto'>
<h2>Brain Tumour Segmentation — Surgical / Radiology Summary</h2>
<p><b>Date (UTC):</b> {datetime.now(timezone.utc):%Y-%m-%d %H:%M} |
<b>Source:</b> {source}</p>
<h3>Volumetry</h3>
<table border='1' cellpadding='6' cellspacing='0'>
<tr><th>Structure</th><th>Volume</th></tr>{rows}</table>
<h3>RANO-style diameters (axial, whole tumor)</h3>
<p>Dmax {rano.get('dmax_mm')} mm x Dperp {rano.get('dperp_mm')} mm
(slice {rano.get('slice')})</p>
<h3>Snapshots (MPR)</h3><div>{imgs or '<i>no snapshots</i>'}</div>
<h3>Ingestion / provenance</h3><pre>{ingestion_notes}</pre>
<hr><p><i>Research visualization only — not for diagnosis or treatment
decisions. Requires radiologist verification. Model trained on BraTS;
performance degrades on post-op, hemorrhage, pediatric, or low-quality
acquisitions.</i></p></body></html>"""
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(html)
    return out_path
