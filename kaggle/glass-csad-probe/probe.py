"""Kaggle notebook (CPU): set up the official GLASS and CSAD code and checkpoints.

1. Clone both official repositories at pinned commits.
2. Download the released checkpoints from the authors' Google Drive into
   /kaggle/working, so the GPU notebooks can mount this output instead of
   downloading again.
3. Check that both codebases import on Kaggle's Python / PyTorch.
4. Smoke-test one small category per model on CPU and compare the AUROC with
   the published value.

Every step is reported in probe_report.json, and one failing step does not
stop the others.
"""

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

WORK = Path("/kaggle/working")
REPOS = Path("/tmp/repos")  # code stays out of the saved output
GLASS_COMMIT = "f788d567db552820d744ca2d93e989265aad8c45"
CSAD_COMMIT = "d46cf856b99bf056088b5f510fd984e04813583a"
GLASS_CKPT_FOLDER = "https://drive.google.com/drive/folders/1Hjlr-CcXwnhWfrWUCJJCooBI_pMP4N1C"
CSAD_MODELS_ID = "1V_X0sMFgpplAZhH4_iCjiqNs1AIXAxT3"  # pytorch_models.zip
CSAD_MASKS_ID = "1CHg697k-jyYRK2v-RtIRjikvuPGOzlHQ"  # masks.zip (semantic pseudo-labels, needed for training)

report = {}


def step(name):
    def wrap(fn):
        t0 = time.time()
        print(f"\n===== {name} =====", flush=True)
        try:
            out = fn()
            report[name] = {"ok": True, "seconds": round(time.time() - t0, 1), "result": out}
        except Exception as e:
            traceback.print_exc()
            report[name] = {"ok": False, "seconds": round(time.time() - t0, 1), "error": f"{type(e).__name__}: {e}"}
        (WORK / "probe_report.json").write_text(json.dumps(report, indent=2, default=str))
        return fn
    return wrap


def sh(cmd: str, cwd=None) -> str:
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
    print(r.stdout[-3000:], r.stderr[-3000:], flush=True)
    if r.returncode:
        raise RuntimeError(f"exit {r.returncode}: {r.stderr[-500:]}")
    return r.stdout


def tree(root: Path) -> dict:
    return {str(p.relative_to(root)): p.stat().st_size for p in sorted(root.rglob("*")) if p.is_file()}


@step("environment")
def _():
    import numpy
    import torch
    import torchvision
    return {"python": sys.version.split()[0], "torch": torch.__version__, "torchvision": torchvision.__version__,
            "numpy": numpy.__version__}


@step("pip install")
def _():
    wheel = next(Path("/kaggle/input").rglob("anode-*.whl"))
    sh(f"{sys.executable} -m pip install -q --no-deps {wheel}")
    sh(f"{sys.executable} -m pip install -q gdown imgaug openpyxl einops empatches")
    return sh(f"{sys.executable} -m pip list 2>/dev/null | grep -iE 'gdown|imgaug|openpyxl|einops|empatches|timm|tensorboard|opencv|scikit-image|pandas|click'")


@step("clone repos")
def _():
    REPOS.mkdir(parents=True, exist_ok=True)
    for name, url, commit in (("GLASS", "https://github.com/cqylunlun/GLASS", GLASS_COMMIT),
                              ("CSAD", "https://github.com/Tokichan/CSAD", CSAD_COMMIT)):
        sh(f"git clone -q {url} {REPOS / name}")
        sh(f"git -C {REPOS / name} checkout -q {commit}")
    return {n: sh(f"git -C {REPOS / n} rev-parse HEAD").strip() for n in ("GLASS", "CSAD")}


@step("download GLASS checkpoints")
def _():
    out = WORK / "glass_ckpt"
    sh(f"gdown --folder {GLASS_CKPT_FOLDER} -O {out} --quiet")
    return tree(out)


@step("download CSAD checkpoints")
def _():
    out = WORK / "csad_ckpt"
    out.mkdir(exist_ok=True)
    sh(f"gdown {CSAD_MODELS_ID} -O {out / 'pytorch_models.zip'} --quiet")
    sh(f"cd {out} && unzip -q -o pytorch_models.zip && rm pytorch_models.zip")
    return tree(out)


@step("download CSAD pseudo-label masks")
def _():
    out = WORK / "csad_masks"
    out.mkdir(exist_ok=True)
    sh(f"gdown {CSAD_MASKS_ID} -O {out / 'masks.zip'} --quiet")
    sh(f"cd {out} && unzip -q -o masks.zip && rm masks.zip")
    files = tree(out)
    return {"n_files": len(files), "total_mb": round(sum(files.values()) / 1e6, 1),
            "top_dirs": sorted({k.split("/")[0] + "/" + k.split("/")[1] for k in files if k.count("/") >= 1})[:20]}


def data_roots():
    from anode.kaggle import dataset_roots
    return dataset_roots()


@step("GLASS import + toothbrush smoke test")
def _():
    import numpy as np
    from sklearn.metrics import roc_auc_score
    os.chdir(REPOS / "GLASS")  # the code reads ./datasets/excel/... relative paths
    sys.path.insert(0, str(REPOS / "GLASS"))
    import torch
    import backbones
    import glass
    from datasets.mvtec import DatasetSplit, MVTecDataset

    roots = data_roots()
    cat = "toothbrush"
    ckpt = next((WORK / "glass_ckpt").rglob(f"mvtec_{cat}/ckpt_best_*.pth"))
    device = torch.device("cpu")
    model = glass.GLASS(device)
    model.load(backbone=backbones.load("wideresnet50"), layers_to_extract_from=["layer2", "layer3"], device=device,
               input_shape=(3, 288, 288), pretrain_embed_dimension=1536, target_embed_dimension=1536, patchsize=3,
               meta_epochs=640, eval_epochs=1, dsc_layers=2, dsc_hidden=1024, dsc_margin=0.5, train_backbone=False,
               pre_proj=1, mining=1, noise=0.015, radius=0.75, p=0.5, lr=0.0001, svd=0, step=20, limit=392)
    sd = torch.load(ckpt, map_location=device)
    model.discriminator.load_state_dict(sd["discriminator"])
    model.pre_projection.load_state_dict(sd["pre_projection"])
    ds = MVTecDataset(str(roots["mvtec_ad"]), "/kaggle/input", classname=cat, resize=288, imagesize=288,
                      split=DatasetSplit.TEST)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, num_workers=2)
    scores, labels = [], []
    for b in loader:
        s, _ = model._predict(b["image"])
        scores += [float(x) for x in s]
        labels += b["is_anomaly"].tolist()
    return {"ckpt": ckpt.name, "n_test": len(scores), "image_auroc": roc_auc_score(labels, scores),
            "published": 1.0}


@step("CSAD import + juice_bottle smoke test")
def _():
    import numpy as np
    import torch
    from sklearn.metrics import roc_auc_score
    os.chdir(REPOS / "CSAD")
    sys.path.insert(0, str(REPOS / "CSAD"))
    from models.onnx_model import CSAD_ONNX
    from data_loader import MVTecLOCODataset

    roots = data_roots()
    cat = "juice_bottle"
    path = next((WORK / "csad_ckpt").rglob(f"{cat}.pth"))
    sd = torch.load(path, map_location="cpu")
    model = CSAD_ONNX(dim=512, num_classes=sd["segmentor.fc2.conv3.weight"].shape[0])
    model.load_state_dict(sd)
    model.eval()
    ds = MVTecLOCODataset(root=str(roots["mvtec_loco"]), image_size=256, phase="test", category=cat, to_gpu=False)
    res = {"logical": ([], []), "structural": ([], [])}
    with torch.no_grad():
        for i in range(len(ds)):
            s = ds[i]
            score = float(model(s["image"].unsqueeze(0)).item())
            kind = Path(s["path"]).parent.name
            for k in ("logical", "structural"):
                if kind == "good" or kind.startswith(k):
                    res[k][0].append(int(kind != "good"))
                    res[k][1].append(score)
    auc = {k: roc_auc_score(*v) for k, v in res.items()}
    return {"ckpt": str(path.relative_to(WORK)), "n_test": len(ds), **auc, "mean": (auc["logical"] + auc["structural"]) / 2}


print(json.dumps(report, indent=2, default=str))
print("ALL OK" if all(v["ok"] for v in report.values()) else "SOME STEPS FAILED")
