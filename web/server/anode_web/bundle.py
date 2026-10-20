"""The model bundle: manifest, checkpoints and sample images.

The bundle is produced by kaggle/bundle/run.py. The server finds it in this
order:

1. ``ANODE_BUNDLE``: a local folder (development).
2. ``ANODE_BUNDLE_REPO``: a Hugging Face Hub repository (deployment), downloaded
   once into the Hub cache.
"""

from __future__ import annotations

import json
import os
from functools import cached_property
from pathlib import Path

DISPLAY_NAMES = {
    "metal_nut": "Metal nut", "breakfast_box": "Breakfast box", "juice_bottle": "Juice bottle",
    "screw_bag": "Screw bag", "splicing_connectors": "Splicing connectors",
}
TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}


def display_name(category: str) -> str:
    return DISPLAY_NAMES.get(category, category.replace("_", " ").capitalize())


def kind(dataset: str, category: str) -> str:
    if dataset == "mvtec_loco":
        return "logical"
    return "texture" if category in TEXTURES else "object"


class Bundle:
    def __init__(self, root: Path):
        self.root = Path(root)
        if not (self.root / "manifest.json").exists():
            raise FileNotFoundError(f"no manifest.json in {self.root}")

    @classmethod
    def locate(cls) -> "Bundle":
        if os.environ.get("ANODE_BUNDLE"):
            return cls(Path(os.environ["ANODE_BUNDLE"]))
        repo = os.environ.get("ANODE_BUNDLE_REPO")
        if repo:
            from huggingface_hub import snapshot_download
            return cls(Path(snapshot_download(repo_id=repo, repo_type="model")))
        raise RuntimeError("set ANODE_BUNDLE (local folder) or ANODE_BUNDLE_REPO (Hugging Face repo id)")

    @cached_property
    def manifest(self) -> dict:
        return json.loads((self.root / "manifest.json").read_text())

    @cached_property
    def products(self) -> dict[str, dict]:
        out = {}
        for p in self.manifest["products"]:
            # A partial bundle (development) may hold only some checkpoints
            if (self.root / p["model_file"]).exists():
                out[p["id"]] = {**p, "name": display_name(p["category"]), "kind": kind(p["dataset"], p["category"])}
        return out

    def path(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if self.root.resolve() not in p.parents:
            raise ValueError("path outside the bundle")
        return p
