"""Custom FastAPI presentation layer for NeuroVista.

The analysis backend remains in :mod:`app`; this module deliberately owns only
HTTP, uploads, result serialization, and the client interface.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from src.atlas_learning import atlas_learning_figure
from src.learning_simulation import generic_neuroanatomy_figure
from src.surgical_planning import build_preoperative_plan


PROJECT_ROOT = Path(__file__).resolve().parent
ATLAS_TEMPLATE = PROJECT_ROOT / "data" / "atlas" / "MNI152_T1_2mm_brain.nii.gz"
ATLAS_LABELS = PROJECT_ROOT / "data" / "atlas" / "AAL_atlas.nii.gz"


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NeuroVista — MRI review</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root{--bg:#f5f5f7;--card:#fff;--ink:#1d1d1f;--muted:#6e6e73;--line:#d2d2d7;--blue:#0071e3;--soft:#eef6ff;--radius:22px}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}header{background:rgba(255,255,255,.82);backdrop-filter:blur(18px);border-bottom:1px solid #e7e7e9}.bar{max-width:1280px;margin:auto;padding:16px 24px;display:flex;justify-content:space-between;align-items:center}.brand{font-weight:700;letter-spacing:-.03em}.brand i{display:inline-block;width:9px;height:9px;background:var(--blue);border-radius:50%;margin-right:8px}.tag{font-size:12px;color:var(--muted)}main{max-width:1280px;margin:auto;padding:46px 24px 72px}.intro{max-width:690px;margin:0 0 36px}.eyebrow{color:var(--blue);font-weight:700;font-size:11px;letter-spacing:.1em;text-transform:uppercase}.intro h1{font-size:clamp(34px,5vw,58px);line-height:1.02;letter-spacing:-.055em;margin:9px 0 14px}.intro p{color:var(--muted);font-size:17px;margin:0}.workflow{display:flex;gap:20px;margin-top:28px}.workflow div{border-left:1px solid var(--line);padding-left:13px;font-size:13px;color:var(--muted)}.workflow b{display:block;color:var(--ink);font-size:12px}.grid{display:grid;grid-template-columns:340px 1fr;gap:20px;align-items:start}.panel{background:var(--card);border:1px solid #e1e1e5;border-radius:var(--radius);box-shadow:0 10px 30px rgba(0,0,0,.035)}.source{padding:20px;position:sticky;top:16px}.section-title{font-size:11px;letter-spacing:.1em;color:var(--muted);font-weight:700;text-transform:uppercase;margin:0 0 15px}.ready{background:#f0faf6;border:1px solid #c7eadb;border-radius:13px;padding:10px 12px;font-size:12px;color:#356151;margin-bottom:14px}.ready b{color:var(--ink)}.mode{display:grid;grid-template-columns:1fr 1fr;background:#eeeeF0;padding:4px;border-radius:12px;margin-bottom:14px}.mode label{padding:7px;text-align:center;font-size:13px;border-radius:8px}.mode input{accent-color:var(--blue)}fieldset{border:1px solid var(--line);border-radius:14px;padding:12px;margin:10px 0}legend{font-size:12px;font-weight:700;padding:0 5px}label{display:block;font-size:12px;font-weight:600;margin:9px 0 5px}input[type=file]{display:block;width:100%;font:12px inherit;color:var(--muted);padding:8px;border:1px dashed #b7c5d8;border-radius:10px;background:#fbfdff}details{border:1px solid #c9dcf3;background:var(--soft);border-radius:14px;padding:11px 12px;margin-top:12px}summary{cursor:pointer;font-weight:700;font-size:13px}details p{font-size:12px;color:var(--muted);margin:8px 0}input[type=checkbox]{accent-color:var(--blue)}.check{display:flex;gap:8px;font-weight:500;color:var(--ink);line-height:1.3}.check input{margin-top:2px}.primary{width:100%;border:0;border-radius:999px;background:var(--blue);color:white;font-weight:700;padding:13px;margin-top:16px;cursor:pointer}.primary:hover{background:#0077ed}.note{font-size:11px;color:var(--muted);margin:10px 0 0}.review{padding:28px;min-height:700px}.empty{display:grid;place-items:center;min-height:580px;text-align:center;color:var(--muted)}.empty strong{display:block;color:var(--ink);font-size:20px;margin-bottom:6px}.hidden{display:none!important}.status{padding:12px 14px;border-radius:12px;background:#eff6ff;color:#174a7e;margin-bottom:16px}.error{background:#fff1f1;color:#9b1c1c}.metrics{line-height:1.65}.tabs{display:flex;gap:20px;border-bottom:1px solid var(--line);margin-top:22px}.tab{padding:11px 1px;color:var(--muted);border:0;background:transparent;font:600 13px inherit;cursor:pointer}.tab.active{color:var(--blue);border-bottom:2px solid var(--blue)}.view{padding-top:18px}.plot{width:100%;min-height:510px}.surgical{background:#202023;border-radius:18px;padding:24px;color:#f5f5f7}.surgical h2{margin:5px 0;font-size:28px;letter-spacing:-.03em}.surgical p{color:#c6c6cc}.surgical .metrics{color:#e9e9ed}.downloads a{display:inline-block;margin:7px 8px 0 0;padding:8px 12px;border:1px solid var(--line);border-radius:999px;color:var(--blue);text-decoration:none;font-size:12px}@media(max-width:850px){main{padding:28px 16px}.grid{grid-template-columns:1fr}.source{position:static}.workflow{gap:10px}.intro h1{font-size:38px}}
</style></head><body><header><div class="bar"><div class="brand"><i></i>NeuroVista</div><div class="tag">MRI review workspace</div></div></header><main><section class="intro"><div class="eyebrow">Clinical review</div><h1>Clarity before the next decision.</h1><p>Review a complete brain MRI study, inspect the draft segmentation, and prepare a concise export for the care team.</p><div class="workflow"><div><b>01 · Import</b>Complete MRI study</div><div><b>02 · Analyse</b>Draft segmentation</div><div><b>03 · Review</b>Verify and export</div></div></section><div class="grid"><form class="panel source" id="study"><p class="section-title">Study input</p><div class="ready"><b>Analysis ready</b><br>Pretrained BraTS model available locally.</div><div class="mode"><label><input type="radio" name="mode" value="nifti" checked> NIfTI</label><label><input type="radio" name="mode" value="dicom"> DICOM ZIP</label></div><div id="nifti"><fieldset><legend>Required MRI sequences</legend><label>T1<input required type="file" name="t1" accept=".nii,.nii.gz"></label><label>T1ce<input required type="file" name="t1ce" accept=".nii,.nii.gz"></label><label>T2<input required type="file" name="t2" accept=".nii,.nii.gz"></label><label>FLAIR<input required type="file" name="flair" accept=".nii,.nii.gz"></label></fieldset></div><div id="dicom" class="hidden"><fieldset><legend>DICOM study</legend><label>DICOM ZIP<input type="file" name="dicom_zip" accept=".zip"></label><label class="check"><input type="checkbox" name="anonymize" checked>Strip patient identifiers before local processing</label></fieldset></div><details><summary>Add surgical planning data</summary><p>Optional review candidates only. No trajectory or distance is generated.</p><label>TOF-MRA / CTA<input type="file" name="vascular" accept=".nii,.nii.gz"></label><details><summary>Diffusion MRI</summary><label>DWI<input type="file" name="dwi" accept=".nii,.nii.gz"></label><label>b-values<input type="file" name="bval" accept=".bval,.txt"></label><label>b-vectors<input type="file" name="bvec" accept=".bvec,.txt"></label><label class="check"><input type="checkbox" name="same_patient">Same patient and pre-operative session</label></details></details><button class="primary">Run analysis</button><p class="note">Research review only — clinician verification required.</p></form><section class="panel review"><div id="message" class="hidden"></div><div id="empty" class="empty"><div><strong>Ready when your study is.</strong>Upload a complete MRI study to begin review.</div></div><div id="results" class="hidden"><div id="metrics" class="metrics"></div><nav class="tabs"><button class="tab active" data-view="surface">3D surface</button><button class="tab" data-view="mpr">MPR viewer</button><button class="tab" data-view="planning">Surgical review</button></nav><div class="view" id="surface"><div id="surface-plot" class="plot"></div></div><div class="view hidden" id="mpr"><div id="mpr-plot" class="plot"></div></div><div class="view hidden" id="planning"><div class="surgical"><div class="eyebrow">Pre-operative review</div><h2>Plan around what matters.</h2><div id="planning-summary" class="metrics"></div><div id="planning-plot" class="plot"></div></div></div><div class="downloads" id="downloads"></div></div></section></div></main><script>
const q=s=>document.querySelector(s),qa=s=>document.querySelectorAll(s);qa('input[name=mode]').forEach(x=>x.onchange=()=>{q('#nifti').classList.toggle('hidden',x.value!=='nifti'||!x.checked);q('#dicom').classList.toggle('hidden',x.value!=='dicom'||!x.checked)});qa('.tab').forEach(b=>b.onclick=()=>{qa('.tab').forEach(x=>x.classList.remove('active'));qa('.view').forEach(x=>x.classList.add('hidden'));b.classList.add('active');q('#'+b.dataset.view).classList.remove('hidden')});function text(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/\n/g,'<br>')}q('#study').onsubmit=async e=>{e.preventDefault();const f=new FormData(e.target),msg=q('#message');msg.className='status';msg.textContent='Analysing the MRI study…';q('#empty').classList.add('hidden');try{const r=await fetch('/api/analyse',{method:'POST',body:f}),d=await r.json();if(!r.ok)throw Error(d.detail||'Analysis failed');msg.classList.add('hidden');q('#results').classList.remove('hidden');q('#metrics').innerHTML=text(d.metrics);q('#planning-summary').innerHTML=text(d.planning);Plotly.react('surface-plot',d.surface.data,d.surface.layout,{responsive:true});Plotly.react('mpr-plot',d.mpr.data,d.mpr.layout,{responsive:true});Plotly.react('planning-plot',d.surgical.data,d.surgical.layout,{responsive:true});q('#downloads').innerHTML=d.downloads.map(x=>`<a href="/api/download/${d.token}/${x.name}">${x.name}</a>`).join('')}catch(err){msg.className='status error';msg.textContent=err.message}};
</script></body></html>"""

SURGICAL_INPUTS = """<details><summary>Add surgical mapping layers</summary>
<p>Upload same-patient, pre-operative, clinician-derived masks. The app reports
geometry for review; it does not generate a surgical route.</p>
<label>TOF-MRA / CTA source image<input type="file" name="vascular" accept=".nii,.nii.gz"></label>
<label>Registered vessel mask<input type="file" name="vessel_mask" accept=".nii,.nii.gz"></label>
<label>Registered white-matter tract mask<input type="file" name="tract_mask" accept=".nii,.nii.gz"></label>
<label>Registered functional cortex map<input type="file" name="functional_mask" accept=".nii,.nii.gz"></label>
<details><summary>Diffusion source data</summary><label>DWI<input type="file" name="dwi" accept=".nii,.nii.gz"></label>
<label>b-values<input type="file" name="bval" accept=".bval,.txt"></label>
<label>b-vectors<input type="file" name="bvec" accept=".bvec,.txt"></label></details>
<label class="check"><input type="checkbox" name="same_patient">I confirm planning data is from the same patient and pre-operative session</label></details>"""
PAGE = re.sub(r'<details><summary>Add surgical planning data</summary>.*?</details></details>',
              SURGICAL_INPUTS, PAGE, count=1)
PAGE = PAGE.replace(
    '<button class="tab" data-view="planning">Surgical review</button>',
    '<button class="tab" data-view="planning">Surgical review</button>'
    '<button class="tab" data-view="learning">Learning simulation</button>')
PAGE = PAGE.replace(
    '<div class="downloads" id="downloads"></div>',
    '<div class="view hidden" id="learning"><div class="surgical">'
    '<div class="eyebrow">Educational anatomy projection</div><h2>See the lesion in anatomical context.</h2>'
    '<p>The patient tumour is affine-registered to a labelled MNI anatomy atlas for '
    'learning and discussion. It is not a navigation transform, a patient-specific '
    'vessel map, or a clearance measurement.</p>'
    '<div id="learning-summary" class="metrics"></div>'
    '<div id="learning-plot" class="plot"></div></div></div>'
    '<div class="downloads" id="downloads"></div>')
PAGE = PAGE.replace(
    "Plotly.react('planning-plot',d.surgical.data,d.surgical.layout,{responsive:true});",
    "Plotly.react('planning-plot',d.surgical.data,d.surgical.layout,{responsive:true});"
    "q('#learning-summary').innerHTML=report(d.learning_summary);"
    "Plotly.react('learning-plot',d.learning.data,d.learning.layout,{responsive:true});")
PAGE = PAGE.replace("</style>", """
.metrics h3{margin:20px 0 10px;font-size:18px;letter-spacing:-.02em;color:#1d1d1f}.metrics h3:first-child{margin-top:0}
.atlas-note{margin:22px 0 8px!important;padding:10px 12px;border-left:3px solid #a78bfa;background:#30283a;border-radius:0 9px 9px 0;color:#ede9fe!important;font-size:12px}
.metrics table{width:100%;border-collapse:separate;border-spacing:0;border:1px solid #e1e1e5;border-radius:13px;overflow:hidden;margin:12px 0}
.metrics th{background:#f7f7f8;text-align:left;font-size:12px;color:#6e6e73;padding:10px 12px}.metrics td{padding:11px 12px;border-top:1px solid #ececef}.metrics th:last-child,.metrics td:last-child{text-align:right;font-variant-numeric:tabular-nums}
.metrics blockquote{margin:12px 0;padding:10px 13px;border-left:3px solid #0071e3;background:#f3f8ff;border-radius:0 10px 10px 0;color:#38485d;font-size:13px}.metrics hr{border:0;border-top:1px solid #e1e1e5;margin:22px 0}.metrics p{margin:10px 0;color:#4a4a4f}
</style>""")
PAGE = PAGE.replace(
    "function text(s){return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/\\n/g,'<br>')}",
    """function esc(s){return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}
function inline(s){return esc(s).replace(/\\*\\*(.*?)\\*\\*/g,'<strong>$1</strong>')}
function report(s){const l=String(s||'').split('\\n');let h='',i=0;while(i<l.length){const x=l[i];if(!x.trim()){i++;continue}if(x.startsWith('### ')){h+=`<h3>${inline(x.slice(4))}</h3>`;i++;continue}if(x==='---'){h+='<hr>';i++;continue}if(x.startsWith('> ')){let b=[];while(i<l.length&&l[i].startsWith('> ')){b.push(l[i].slice(2));i++}h+=`<blockquote>${inline(b.join(' '))}</blockquote>`;continue}if(x.startsWith('|')){const rows=[];while(i<l.length&&l[i].startsWith('|')){if(!/^\\|[- :|]+\\|$/.test(l[i]))rows.push(l[i].split('|').slice(1,-1).map(v=>v.trim()));i++}if(rows.length){h+='<table><thead><tr>'+rows[0].map(v=>`<th>${inline(v)}</th>`).join('')+'</tr></thead><tbody>'+rows.slice(1).map(r=>'<tr>'+r.map(v=>`<td>${inline(v)}</td>`).join('')+'</tr>').join('')+'</tbody></table>'}continue}let p=[];while(i<l.length&&l[i].trim()&&!l[i].startsWith('### ')&&!l[i].startsWith('> ')&&!l[i].startsWith('|')&&l[i]!=='---'){p.push(l[i]);i++}h+=`<p>${inline(p.join(' '))}</p>`}return h}""")
PAGE = PAGE.replace("q('#metrics').innerHTML=text(d.metrics);q('#planning-summary').innerHTML=text(d.planning);",
                    "q('#metrics').innerHTML=report(d.metrics);q('#planning-summary').innerHTML=report(d.planning);")


def create_app(backend: Any) -> FastAPI:
    """Create the standalone custom client using the existing analysis code."""
    api = FastAPI(title="NeuroVista")
    exports: dict[str, dict[str, Path]] = {}

    @api.get("/", response_class=HTMLResponse)
    def index() -> str:
        return PAGE

    def save(upload: UploadFile | None, directory: Path) -> str | None:
        if upload is None or not upload.filename:
            return None
        target = directory / Path(upload.filename).name
        with target.open("wb") as handle:
            shutil.copyfileobj(upload.file, handle)
        return str(target)

    @api.post("/api/analyse")
    def analyse(
        mode: str = Form("nifti"),
        t1: UploadFile | None = File(None), t1ce: UploadFile | None = File(None),
        t2: UploadFile | None = File(None), flair: UploadFile | None = File(None),
        dicom_zip: UploadFile | None = File(None), vascular: UploadFile | None = File(None),
        dwi: UploadFile | None = File(None), bval: UploadFile | None = File(None),
        bvec: UploadFile | None = File(None), vessel_mask: UploadFile | None = File(None),
        tract_mask: UploadFile | None = File(None), functional_mask: UploadFile | None = File(None),
        anonymize: bool = Form(False),
        same_patient: bool = Form(False),
    ) -> JSONResponse:
        upload_dir = Path(tempfile.mkdtemp(prefix="neurovista_upload_"))
        try:
            if mode == "nifti":
                files = {key: save(value, upload_dir) for key, value in {
                    "t1": t1, "t1ce": t1ce, "t2": t2, "flair": flair}.items()}
                if not all(files.values()):
                    raise ValueError("Upload T1, T1ce, T2, and FLAIR to run analysis.")
                source: str | dict[str, str] = files  # type: ignore[assignment]
            else:
                source = save(dicom_zip, upload_dir)
                if not source:
                    raise ValueError("Upload a DICOM ZIP to run analysis.")
                if anonymize:
                    source = backend.anonymize_dicom_zip(source)
            result = backend._run_pipeline(
                source, 78, 120, 120, "flair", .8, [1, 2, 3], 50,
                planning_vascular_path=save(vascular, upload_dir),
                planning_dwi_path=save(dwi, upload_dir),
                planning_bval_path=save(bval, upload_dir),
                planning_bvec_path=save(bvec, upload_dir),
                planning_same_patient_confirmed=same_patient,
            )
            surface, mpr, metrics, warnings, _report, downloads, surgical, planning = result
            layer_paths = {
                "Vessels": save(vessel_mask, upload_dir),
                "White-matter tracts": save(tract_mask, upload_dir),
                "Functional cortex": save(functional_mask, upload_dir),
            }
            if any(layer_paths.values()) and not same_patient:
                raise ValueError("Confirm that planning data is from the same patient and pre-operative session.")
            segmentation = next((path for path in downloads if Path(path).name == "segmentation.nii.gz"), None)
            learning = generic_neuroanatomy_figure(show_simulated_tumour=False)
            learning_summary = (
                "### Anatomy explorer\n\n"
                "> Patient-to-atlas projection was unavailable for this study. The displayed anatomy "
                "reference is generic and is not patient-specific."
            )
            if segmentation:
                plan_report = upload_dir / "preoperative_mapping_review.json"
                plan, _layers, surgical = build_preoperative_plan(segmentation, layer_paths, str(plan_report))
                planning = plan.markdown()
                downloads = [*downloads, str(plan_report)]
                if isinstance(source, dict) and ATLAS_TEMPLATE.is_file() and ATLAS_LABELS.is_file():
                    try:
                        learning, note, learning_summary = atlas_learning_figure(
                            source["t1"], segmentation, str(ATLAS_TEMPLATE), str(ATLAS_LABELS))
                        learning_summary += "\n\n> The purple lesion is the model segmentation " \
                            "projected into a labelled MNI reference atlas. " + note
                    except Exception as atlas_error:
                        learning_summary = (
                            "### Anatomy explorer\n\n"
                            "> Atlas projection could not be created for this study (" + str(atlas_error) + "). "
                            "The generic reference shown is not patient-specific."
                        )
            token = uuid.uuid4().hex
            exports[token] = {Path(path).name: Path(path) for path in downloads if Path(path).is_file()}
            return JSONResponse({
                "surface": json.loads(surface.to_json()), "mpr": json.loads(mpr.to_json()),
                "surgical": json.loads(surgical.to_json()),
                "learning": json.loads(learning.to_json()),
                "metrics": metrics + ("\n\n" + warnings if warnings else ""),
                "planning": planning, "learning_summary": learning_summary, "token": token,
                "downloads": [{"name": name} for name in exports[token]],
            })
        except Exception as exc:  # user-visible validation / inference errors
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.get("/api/download/{token}/{name}")
    def download(token: str, name: str) -> FileResponse:
        path = exports.get(token, {}).get(name)
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail="Export is no longer available.")
        return FileResponse(path, filename=name)

    return api
