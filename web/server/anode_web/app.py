"""AnoDe web API and static frontend.

    ANODE_BUNDLE=/path/to/bundle ANODE_REPOS=/path/to/repos uvicorn anode_web.app:app --port 7860

Environment:
    ANODE_BUNDLE / ANODE_BUNDLE_REPO   model bundle (see bundle.py)
    ANODE_REPOS                        folder with the official CSAD/ clone
    ANODE_DATA                         where custom products are stored (default /tmp/anode)
    ANODE_FRONTEND                     built frontend (default ../../frontend/dist)
"""

from __future__ import annotations

import io
import json
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps
from pydantic import BaseModel

from anode import qc

from . import demo
from .bundle import Bundle
from .heatmap import data_url, overlay, segmentation

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "data" / "benchmark.json").read_text())
DEMO_SCORES = json.loads((HERE / "data" / "demo_scores.json").read_text())
REPOS = Path(os.environ.get("ANODE_REPOS", "/opt/repos"))
FRONTEND = Path(os.environ.get("ANODE_FRONTEND", HERE.parents[1] / "frontend" / "dist"))
MAX_FILES, MAX_BYTES, MAX_SIDE = 64, 12 * 1024 * 1024, 1024

app = FastAPI(title="AnoDe", version="1.0")
bundle = Bundle.locate()
_runtimes: dict = {}
_rt_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=1)  # one training job at a time on a small CPU


def runtime(name: str):
    """Load model runtimes lazily, so the server starts fast."""
    with _rt_lock:
        if name not in _runtimes:
            from . import runtimes as R
            if name == "csad":
                _runtimes[name] = R.CsadRuntime(REPOS / "CSAD")
            elif name == "patchcore":
                _runtimes[name] = R.PatchCoreRuntime(Path(os.environ.get("ANODE_DATA", "/tmp/anode")))
        return _runtimes[name]


# ---------------------------------------------------------------- products

def _benchmark_for(p: dict) -> dict:
    ds = DATA["datasets"].get(p["dataset"], {})
    metric = "auroc_loco_mean" if p["dataset"] == "mvtec_loco" else "image_auroc"
    return {m: rows[p["category"]].get(metric) for m, rows in ds.get("per_category", {}).items()
            if p["category"] in rows}


def _public(p: dict) -> dict:
    return {
        "id": p["id"], "name": p["name"], "dataset": p["dataset"], "category": p["category"], "kind": p["kind"],
        "model": p["model"], "threshold": p["threshold"], "custom": False, "has_map": p["model"] != "csad",
        "samples": [{**s, "url": f"/bundle/{s['file']}", "mask_url": f"/bundle/{s['mask']}" if s.get("mask") else None}
                    for s in p["samples"]],
        "benchmark": _benchmark_for(p),
        "val_scores": p.get("val_scores", []),
    }


def _public_custom(cp) -> dict:
    return {"id": cp.id, "name": cp.name, "dataset": "custom", "category": cp.id, "kind": "custom",
            "model": "patchcore", "threshold": cp.threshold, "custom": True, "has_map": True, "samples": [],
            "benchmark": {}, "val_scores": cp.val_scores, "n_train": cp.n_train, "n_val": cp.n_val,
            "created": cp.created}


def _find(pid: str) -> dict:
    if pid in bundle.products:
        return _public(bundle.products[pid])
    if pid.startswith("custom-"):
        rt = runtime("patchcore")
        if pid in rt.products:
            return _public_custom(rt.products[pid])
    raise HTTPException(404, f"unknown product {pid}")


@app.get("/api/health")
def health():
    return {"ok": True, "products": len(bundle.products), "loaded": sorted(_runtimes)}


@app.get("/api/products")
def products(custom: str = ""):
    """Bundled products, plus the custom products whose ids the browser
    remembers (custom products are private to the browser that made them)."""
    out = [_public(p) for p in bundle.products.values()]
    ids = [i for i in custom.split(",") if i.startswith("custom-")]
    if ids:
        rt = runtime("patchcore")
        out += [_public_custom(rt.products[i]) for i in ids if i in rt.products]
    return out


@app.get("/api/products/{pid}")
def product(pid: str):
    return _find(pid)


@app.get("/api/benchmark")
def benchmark():
    return DATA


# ---------------------------------------------------------------- inspection

def _load(data: bytes) -> Image.Image:
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "image larger than 12 MB")
    try:
        im = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    except Exception:
        raise HTTPException(400, "not a readable image")
    im.thumbnail((MAX_SIDE, MAX_SIDE))
    return im


def _predict(p: dict, images: list[Image.Image]):
    if p["custom"]:
        return runtime("patchcore").predict(p["id"], images)
    raw = bundle.products[p["id"]]
    if p["model"] == "patchcore":
        return runtime("patchcore").predict(p["id"], images, bank_file=bundle.path(raw["model_file"]))
    return runtime("csad").predict(raw, bundle.path(raw["model_file"]), images)


def _results(p: dict, names: list[str], images: list[Image.Image]) -> list[dict]:
    out = []
    for name, pred in zip(names, _predict(p, images)):
        r = {"name": name, "score": pred.score, "normalized": pred.score / p["threshold"],
             "verdict": "NOK" if pred.score >= p["threshold"] else "OK",
             "latency_ms": round(pred.latency_ms, 1), "view": data_url(pred.view)}
        if pred.amap is not None:
            r["heatmap"] = data_url(overlay(pred.view, pred.amap, p["threshold"]))
            r["segmentation"] = data_url(segmentation(pred.view, pred.amap, p["threshold"]))
        out.append(r)
    return out


@app.post("/api/inspect/{pid}")
async def inspect(pid: str, files: list[UploadFile] = File(...)):
    p = _find(pid)
    if len(files) > MAX_FILES:
        raise HTTPException(413, f"at most {MAX_FILES} images per request")
    images = [_load(await f.read()) for f in files]
    return _results(p, [f.filename or f"image-{i}" for i, f in enumerate(files)], images)


@app.post("/api/inspect/{pid}/samples")
def inspect_samples(pid: str, indices: list[int] | None = None):
    """Run the model on bundled sample images (no upload needed)."""
    p = _find(pid)
    if p["custom"]:
        raise HTTPException(400, "custom products have no bundled samples")
    samples = p["samples"] if indices is None else [p["samples"][i] for i in indices if 0 <= i < len(p["samples"])]
    images = [Image.open(bundle.path(s["file"])).convert("RGB") for s in samples]
    res = _results(p, [Path(s["file"]).name for s in samples], images)
    for r, s in zip(res, samples):
        r["truth"] = "NOK" if s["label"] else "OK"
        r["defect_type"] = s["defect_type"]
    return res


# ---------------------------------------------------------------- custom products

@app.post("/api/products/custom")
async def create_custom(name: str = Form(...), files: list[UploadFile] = File(...)):
    if not 10 <= len(files) <= 100:
        raise HTTPException(400, "upload between 10 and 100 normal images")
    images = [_load(await f.read()) for f in files]
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "queued", "progress": "waiting for the CPU", "started": time.time()}

    def work():
        job = _jobs[job_id]
        try:
            job.update(status="running", progress="loading the backbone")
            cp = runtime("patchcore").train(name, images, progress=lambda m: job.update(progress=m))
            job.update(status="done", progress="ready", product_id=cp.id, seconds=round(time.time() - job["started"]))
        except Exception as e:  # report the error to the client
            job.update(status="error", progress=str(e))

    _executor.submit(work)
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    if job_id not in _jobs:
        raise HTTPException(404, "unknown job")
    return _jobs[job_id]


# ---------------------------------------------------------------- quality control

class Batch(BaseModel):
    n: int
    defective: int
    scores: list[float] = []


class QCRequest(BaseModel):
    batches: list[Batch]


@app.post("/api/qc")
def qc_report(req: QCRequest):
    """P-chart over the batches and process capability of the normalised
    scores (USL = 1, i.e. the decision threshold), overall and per batch."""
    if not req.batches:
        raise HTTPException(400, "no batches")
    pc = qc.p_chart([b.defective for b in req.batches], [b.n for b in req.batches])
    all_scores = [s for b in req.batches for s in b.scores]
    per_batch = [qc.capability(b.scores, usl=1.0) if len(b.scores) >= 2 else None for b in req.batches]
    return {"p_chart": pc, "capability": qc.capability(all_scores, usl=1.0) if len(all_scores) >= 2 else None,
            "capability_per_batch": per_batch}


@app.get("/api/demo-history/{pid}")
def demo_history(pid: str, days: int = 30):
    p = _find(pid)
    if p["custom"] or p["category"] not in DEMO_SCORES:
        return []
    raw = bundle.products[pid]
    return demo.history(pid, DEMO_SCORES[p["category"]], raw["threshold"], days=min(max(days, 7), 90))


# ---------------------------------------------------------------- static files

@app.get("/bundle/samples/{cat}/{name}")
def sample_file(cat: str, name: str):
    try:
        return FileResponse(bundle.path(f"samples/{cat}/{name}"), headers={"Cache-Control": "public, max-age=86400"})
    except ValueError:
        raise HTTPException(404)


if FRONTEND.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND / "assets"), name="assets")

    @app.get("/{path:path}")
    def spa(path: str):
        if path.startswith(("api/", "bundle/")):  # unknown API or bundle path: a real 404, not the app shell
            raise HTTPException(404)
        f = FRONTEND / path
        if path and f.is_file() and FRONTEND in f.resolve().parents:
            return FileResponse(f)
        return FileResponse(FRONTEND / "index.html")
