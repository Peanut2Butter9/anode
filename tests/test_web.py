"""Web API endpoints that do not need real model checkpoints."""

import importlib
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

pytest.importorskip("fastapi")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "web" / "server"))


@pytest.fixture
def client(tmp_path, monkeypatch):
    b = tmp_path / "bundle"
    (b / "models" / "glass").mkdir(parents=True)
    (b / "models" / "glass" / "bottle.pth").write_bytes(b"")
    (b / "samples" / "bottle").mkdir(parents=True)
    Image.new("RGB", (8, 8)).save(b / "samples" / "bottle" / "00_good.jpg")
    manifest = {"products": [
        {"id": "bottle", "dataset": "mvtec_ad", "category": "bottle", "model": "glass",
         "model_file": "models/glass/bottle.pth", "threshold": 0.5, "val_scores": [0.1, 0.2],
         "samples": [{"file": "samples/bottle/00_good.jpg", "label": 0, "defect_type": "good"}]},
        # Its checkpoint is missing, so a partial bundle simply leaves it out
        {"id": "cable", "dataset": "mvtec_ad", "category": "cable", "model": "glass",
         "model_file": "models/glass/cable.pth", "threshold": 0.5, "samples": []},
    ]}
    (b / "manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setenv("ANODE_BUNDLE", str(b))
    monkeypatch.setenv("ANODE_DATA", str(tmp_path / "data"))
    from fastapi.testclient import TestClient
    import anode_web.app as app_module
    importlib.reload(app_module)
    return TestClient(app_module.app)


def test_health_and_products(client):
    assert client.get("/api/health").json()["products"] == 1
    products = client.get("/api/products").json()
    assert [p["id"] for p in products] == ["bottle"]
    p = products[0]
    assert p["name"] == "Bottle" and p["kind"] == "object" and p["has_map"]
    assert p["samples"][0]["url"] == "/bundle/samples/bottle/00_good.jpg"
    assert client.get(p["samples"][0]["url"]).status_code == 200


def test_unknown_product_is_404(client):
    assert client.get("/api/products/nope").status_code == 404


def test_sample_path_traversal_is_blocked(client):
    for url in ("/bundle/samples/..%2F..%2F/manifest.json", "/bundle/samples/bottle/..%2F..%2Fmanifest.json",
                "/bundle/manifest.json", "/api/nope"):
        r = client.get(url)
        assert r.status_code == 404 and b'"products"' not in r.content, url


def test_qc_endpoint(client):
    r = client.post("/api/qc", json={"batches": [
        {"n": 100, "defective": 3, "scores": [0.2, 0.4, 1.3]},
        {"n": 120, "defective": 30, "scores": [0.3, 0.5]},
    ]}).json()
    assert len(r["p_chart"]["points"]) == 2 and r["p_chart"]["points"][1]["out_of_control"]
    assert r["capability"]["usl"] == 1.0


def test_demo_history_is_deterministic(client, monkeypatch):
    import anode_web.app as app_module
    monkeypatch.setitem(app_module.DEMO_SCORES, "bottle", {"good": [0.1, 0.2, 0.3], "defective": [0.9, 1.2]})
    a = client.get("/api/demo-history/bottle?days=14").json()
    b = client.get("/api/demo-history/bottle?days=14").json()
    assert a == b and len(a) >= 10
    assert all(x["demo"] and x["n"] == len(x["scores"]) for x in a)
    assert all(x["defective"] == sum(s >= 1 for s in x["scores"]) for x in a)
