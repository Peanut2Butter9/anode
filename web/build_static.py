"""Precompute everything the static website needs.

    python web/build_static.py --bundle bundle --out web/static-data

Free Hugging Face Spaces only serve static files, so the public demo cannot run
models. This script runs the deployed models once on every bundled sample
image, with the same server code as the full app. It writes:

    products.json            the product list, as /api/products returns it
    inspect/<id>.json        per-sample results, as /api/inspect/<id>/samples returns them
    inspect/<id>/*.jpg       the model's view, heatmap and segmentation images
    benchmark.json           benchmark tables
    demo_scores.json         test-set score pools for the simulated history

Original sample images are not copied: the site loads them from the model
bundle on the Hugging Face Hub (``--samples-url``).
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bundle", type=Path, default=ROOT / "bundle")
    p.add_argument("--repos", type=Path, default=ROOT / "third_party")
    p.add_argument("--out", type=Path, default=ROOT / "web" / "static-data")
    p.add_argument("--samples-url", default="https://huggingface.co/Arzz/anode-models/resolve/main")
    a = p.parse_args()

    os.environ["ANODE_BUNDLE"] = str(a.bundle.resolve())
    os.environ["ANODE_REPOS"] = str(a.repos.resolve())
    os.environ["ANODE_DATA"] = "/tmp/anode-static"
    sys.path.insert(0, str(ROOT / "web" / "server"))
    import anode_web.app as server

    out = a.out.resolve()
    shutil.rmtree(out, ignore_errors=True)
    (out / "inspect").mkdir(parents=True)
    products = []
    for p in server.products():
        pid = p["id"]
        # Samples load from the Hub; inference results become static files
        for s in p["samples"]:
            s["url"] = f"{a.samples_url}/{s['file']}"
            s["mask_url"] = f"{a.samples_url}/{s['mask']}" if s.get("mask") else None
        products.append(p)
        results = server.inspect_samples(pid)
        d = out / "inspect" / pid
        d.mkdir()
        for r, s in zip(results, p["samples"]):
            stem = Path(s["file"]).stem
            for key in ("view", "heatmap", "segmentation"):
                if r.get(key):
                    _, b64 = r[key].split(",", 1)
                    (d / f"{stem}_{key}.jpg").write_bytes(base64.b64decode(b64))
                    r[key] = f"data/inspect/{pid}/{stem}_{key}.jpg"
        (out / "inspect" / f"{pid}.json").write_text(json.dumps(results))
        ok = sum(r["truth"] == r["verdict"] for r in results)
        print(f"{pid:20s} {p['model']:9s} {ok}/{len(results)} correct", flush=True)
    (out / "products.json").write_text(json.dumps(products))
    shutil.copy(ROOT / "web" / "server" / "anode_web" / "data" / "benchmark.json", out / "benchmark.json")
    shutil.copy(ROOT / "web" / "server" / "anode_web" / "data" / "demo_scores.json", out / "demo_scores.json")
    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"static data: {len(products)} products, {total / 1e6:.1f} MB in {out}")


if __name__ == "__main__":
    main()
