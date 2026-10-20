"""Publish AnoDe to the Hugging Face Hub (needs `hf auth login` first).

    python web/deploy.py bundle --repo USER/anode-models --path bundle
    python web/deploy.py static --repo USER/anode
    python web/deploy.py space  --repo USER/anode --bundle-repo USER/anode-models

``bundle`` uploads the model bundle (checkpoints, samples, manifest) to a model
repository. ``static`` publishes the free static demo: the frontend built with
VITE_STATIC=1 plus the results precomputed by web/build_static.py. ``space``
deploys the full app (live inference) as a Docker Space. That needs a paid
plan on Hugging Face since 2026.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
SPACE_FILES = ["Dockerfile", ".dockerignore", "pyproject.toml", "src", "web/server", "web/frontend"]
SKIP = {"node_modules", "dist", "__pycache__", ".pytest_cache"}

SPACE_README = """---
title: AnoDe
emoji: 🔍
colorFrom: green
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: cc-by-nc-sa-4.0
short_description: Visual anomaly detection and quality control
---

# AnoDe

Visual anomaly detection and statistical quality control for manufacturing,
benchmarked on MVTec AD and MVTec LOCO AD. Code, benchmark and paper:
{github}

Sample images: MVTec AD and MVTec LOCO AD, (c) MVTec Software GmbH,
CC BY-NC-SA 4.0. Non-commercial demonstration.
"""


STATIC_README = """---
title: AnoDe
emoji: 🔍
colorFrom: green
colorTo: indigo
sdk: static
app_file: index.html
pinned: false
license: cc-by-nc-sa-4.0
short_description: Visual anomaly detection and quality control
---

# AnoDe (static demo)

Visual anomaly detection and statistical quality control for manufacturing,
benchmarked on MVTec AD and MVTec LOCO AD. This free static build shows
model results that were precomputed for the sample images. Live inference runs
with the full app (see the repository): {github}

Sample images: MVTec AD and MVTec LOCO AD, (c) MVTec Software GmbH,
CC BY-NC-SA 4.0. Non-commercial demonstration.
"""


def upload_static(api: HfApi, repo: str, github: str) -> None:
    import subprocess
    fe = ROOT / "web" / "frontend"
    subprocess.run(["npm", "run", "build:static"], cwd=fe, check=True)
    dist = fe / "dist-static"
    data = ROOT / "web" / "static-data"
    if not (data / "products.json").exists():
        raise SystemExit("run web/build_static.py first")
    shutil.rmtree(dist / "data", ignore_errors=True)
    shutil.copytree(data, dist / "data")
    (dist / "README.md").write_text(STATIC_README.format(github=github))
    api.create_repo(repo, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="space", folder_path=str(dist), commit_message="Deploy AnoDe (static)")
    print(f"https://huggingface.co/spaces/{repo}")


def upload_bundle(api: HfApi, repo: str, path: Path) -> None:
    api.create_repo(repo, repo_type="model", exist_ok=True)
    api.upload_folder(repo_id=repo, repo_type="model", folder_path=str(path),
                      commit_message="AnoDe model bundle")
    print(f"https://huggingface.co/{repo}")


def upload_space(api: HfApi, repo: str, bundle_repo: str, github: str) -> None:
    api.create_repo(repo, repo_type="space", space_sdk="docker", exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp)
        for rel in SPACE_FILES:
            src = ROOT / rel
            if src.is_dir():
                shutil.copytree(src, stage / rel, ignore=lambda d, names: [n for n in names if n in SKIP])
            else:
                (stage / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, stage / rel)
        (stage / "README.md").write_text(SPACE_README.format(github=github))
        api.upload_folder(repo_id=repo, repo_type="space", folder_path=str(stage), commit_message="Deploy AnoDe")
    api.add_space_variable(repo, "ANODE_BUNDLE_REPO", bundle_repo)
    print(f"https://huggingface.co/spaces/{repo}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--repo", required=True)
    b.add_argument("--path", type=Path, default=ROOT / "bundle")
    st = sub.add_parser("static")
    st.add_argument("--repo", required=True)
    st.add_argument("--github", default="https://github.com/Peanut2Butter9/anode")
    s = sub.add_parser("space")
    s.add_argument("--repo", required=True)
    s.add_argument("--bundle-repo", required=True)
    s.add_argument("--github", default="https://github.com/Peanut2Butter9/anode")
    a = p.parse_args()
    api = HfApi()
    if a.cmd == "bundle":
        upload_bundle(api, a.repo, a.path)
    elif a.cmd == "static":
        upload_static(api, a.repo, a.github)
    else:
        upload_space(api, a.repo, a.bundle_repo, a.github)


if __name__ == "__main__":
    main()
