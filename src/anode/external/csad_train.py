"""Train CSAD from scratch with the official code, keeping the last checkpoint.

The official scripts (github.com/Tokichan/CSAD, commit d46cf856) evaluate on
the test set during training and keep the best checkpoint: the segmentor every
2 epochs, the LGST every 1,000 iterations. Here both stages run their full
schedule and keep the last state. Everything else follows the official
scripts, for one category per call:

* ``segmentor``: ``train_segmentor.py`` (120 epochs, the first 50 supervised
  only), then the patch-histogram scores and parameters that
  ``export_model.py`` needs (fitted on train and validation images).
* ``lgst``: ``CSAD.py`` (60,000 iterations). With ``eval_freq`` equal to the
  number of iterations, the official loop evaluates once, after the last
  iteration, and so saves the last state.
* ``export``: ``export_model.py --format torch`` packs both stages into
  ``ckpt/pytorch_models/<category>.pth``, the format ``csad-official`` scores.

Three fixes so the code runs on Linux with current packages:

* The official dataset classes pair images with pseudo-label maps by
  ``glob`` order, which is only sorted on Windows, so ``glob.glob`` is sorted.
* The ImageNet penalty set ``Maysee/tiny-imagenet`` moved to
  ``zh-plus/tiny-imagenet`` on the Hugging Face Hub.
* The checkpoints pickle whole modules, which ``torch.load`` refuses since
  PyTorch 2.6 unless ``weights_only=False``.

``--epochs`` and ``--iterations`` shorten a stage, and every stage writes its
timings to ``--out`` (``timing_<stage>_<category>.json``), so the same code
also serves as a timing probe.

With ``--eval-every N`` and ``--ts-dir DIR``, a stage also evaluates on the
test set every N epochs or iterations, as the official loops do, and keeps
the state with the best test score in DIR (same file layout as the repo).
Training still runs to the end and the repo keeps the last state. DIR exists
only to measure how optimistic the official test-set selection is
(``csad_testselected``); ``export --from-dir DIR`` packs that state.

    python -m anode.external.csad_train prepare   --repo CSAD --root MVTEC_LOCO --masks MASKS --category pushpins
    python -m anode.external.csad_train segmentor --repo CSAD --category pushpins --out OUT [--eval-every 10 --ts-dir TS]
    python -m anode.external.csad_train lgst      --repo CSAD --category pushpins --seed 0 --out OUT [--eval-every 2000 --ts-dir TS]
    python -m anode.external.csad_train export    --repo CSAD --category pushpins [--from-dir TS]
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

PENALTY_SET = "zh-plus/tiny-imagenet"


def _patch() -> None:
    if getattr(glob, "_anode_sorted", False):
        return
    unsorted = glob.glob
    glob.glob = lambda *a, **k: sorted(unsorted(*a, **k))
    glob._anode_sorted = True
    torch.load = functools.partial(torch.load, weights_only=False)


def _enter(repo: Path) -> Path:
    """Import the official modules from ``repo`` (they use paths relative to it)."""
    repo = Path(repo).resolve()
    _patch()
    for name in [m for m in sys.modules if m == "datasets" or m.startswith("datasets.")]:
        del sys.modules[name]  # GLASS ships a ``datasets`` package that shadows HuggingFace's
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.chdir(repo)
    for d in ("ckpt", "anomaly_score"):
        (repo / d).mkdir(exist_ok=True)
    return repo


@contextlib.contextmanager
def _cwd(path: Path):
    """The official code writes its outputs relative to the working directory."""
    old = os.getcwd()
    for d in ("ckpt", "anomaly_score"):
        (Path(path) / d).mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _copy(src: Path, dst: Path, names: list[str]) -> None:
    for name in names:
        (dst / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / name, dst / name)


def _sync() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def _write(out: Path | None, name: str, info: dict) -> None:
    info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    info["peak_gpu_mb"] = round(torch.cuda.max_memory_allocated() / 2**20) if torch.cuda.is_available() else 0
    print(json.dumps(info, indent=1), flush=True)
    if out:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{name}.json").write_text(json.dumps(info, indent=1))


def prepare(repo: Path, root: Path, masks: Path, category: str) -> None:
    """Lay out ``datasets/`` as the official scripts expect. The masks are
    copied because training writes segmentation outputs next to them."""
    ds = Path(repo) / "datasets"
    ds.mkdir(exist_ok=True)
    link = ds / "mvtec_loco_anomaly_detection"
    if not link.exists():
        link.symlink_to(Path(root).resolve(), target_is_directory=True)
    dst = ds / "masks" / category
    if not dst.exists():
        shutil.copytree(Path(masks) / category, dst)


def _seg_files(category: str) -> list[str]:
    return [f"ckpt/segmentor_{category}_256.pth",
            *(f"anomaly_score/{category}_patchhist_{s}" for s in ("params.npz", "logi_score.npy", "struc_score.npy",
                                                                   "val_score.npy"))]


def _lgst_files(category: str) -> list[str]:
    return [f"ckpt/best_{category}.pth", f"anomaly_score/{category}_LGST_val_score.npy"]


def segmentor(repo: Path, category: str, epochs: int | None = None, seed: int = 42,
              eval_every: int = 0, ts_dir: Path | None = None, out: Path | None = None) -> dict:
    """``train_segmentor.py`` for one category, keeping the last epoch.
    ``eval_every`` > 0 also logs the test AUROCs every that many epochs (the
    official loop selects on them every 2 epochs); with ``ts_dir`` the
    test-best epoch is kept there."""
    repo = _enter(repo)
    scratch = repo / "_eval_scratch"
    import timm
    from torch.utils.data import DataLoader
    from train_segmentor import read_config, seed_everything
    from models.segmentation.dataloader import SegmentDataset, SegmentDatasetTest
    from models.segmentation.loss import (ClassBalancedDiceLoss, EntropyLoss, FocalLoss, HistLoss,
                                          MulticlassCrossEntropyLoss)
    from models.segmentation.patch_histogram import test_patch_histogram
    from models.segmentation.segmentor import InfiniteDataloader, SegmentorTrainer

    t_start = _sync()
    seed_everything(seed)
    config = read_config(repo / f"configs/segmentor/{category}.yaml")
    config["image_size"] = 256
    for key in ("train_image_path", "test_image_path", "val_image_path", "mask_root", "model_path"):
        config[key] = config[key].replace("category", category).replace("PROJECT_ROOT", str(repo))
    config["model_path"] = config["model_path"].replace(".pth", f"_{config['image_size']}.pth")
    config["category"] = category
    total_epochs = config["epoch"] if epochs is None else epochs

    sup = SegmentDataset(image_path=config["train_image_path"], sup=True, mask_root=config["mask_root"],
                         image_size=config["image_size"], config=config)
    config["pad2resize"] = sup.pad2resize
    unsup = SegmentDataset(image_path=config["train_image_path"], sup=False, mask_root=config["mask_root"],
                           image_size=config["image_size"], config=config)
    loader = functools.partial(DataLoader, batch_size=4, shuffle=True, drop_last=True, pin_memory=False,
                               num_workers=0)
    sup_loader, unsup_loader = loader(sup), loader(unsup)
    single = functools.partial(DataLoader, batch_size=1, shuffle=False)
    test_loader = single(SegmentDatasetTest(image_path=config["test_image_path"], image_size=256))
    train_loader = single(SegmentDatasetTest(image_path=config["train_image_path"], image_size=256))
    val_loader = single(SegmentDatasetTest(image_path=config["val_image_path"], image_size=256))

    encoder = timm.create_model("hf_hub:timm/wide_resnet50_2.tv2_in1k", pretrained=True, features_only=True,
                                out_indices=[1, 2, 3])
    for p in encoder.parameters():
        p.requires_grad = False
    trainer = SegmentorTrainer(encoder, config)
    # The start of SegmentorTrainer.fit, unchanged.
    trainer.optimizer = torch.optim.Adam(trainer.segmentor.parameters(), lr=config["lr"], weight_decay=1e-5)
    trainer.loss_dict = {"ce": MulticlassCrossEntropyLoss(ignore_index=None), "focal": FocalLoss(ignore_index=None),
                         "dice": ClassBalancedDiceLoss(ignore_index=None), "hist": HistLoss(ignore_index=0),
                         "entropy": EntropyLoss()}
    trainer.loss_weight = config["loss_weight"]
    iters_per_epoch = len(sup_loader)
    sup_inf, unsup_inf = InfiniteDataloader(sup_loader), InfiniteDataloader(unsup_loader)
    t_setup = _sync()

    def patch_hist(save: bool):
        trainer.segmentor.eval()
        with torch.no_grad():
            return test_patch_histogram(train_loader=train_loader, val_loader=val_loader, test_loader=test_loader,
                                        encoder=trainer.encoder, segmentor=trainer.segmentor, category=category,
                                        patch_size=[256, 128], overlap_ratio=[0, 0], save_score=save)

    import tqdm
    bar = tqdm.tqdm(range(total_epochs), desc=f"segmentor {category}")
    epoch_s, losses, test_log, eval_s = [], [], [], []
    best = {"mean": float("-inf")}
    for epoch in bar:
        t0 = _sync()
        trainer.segmentor.train()
        losses.append(trainer.train_one_epoch(bar, sup_inf, unsup_inf, iters_per_epoch,
                                              epoch < config["sup_only_epoch"]) / iters_per_epoch)
        epoch_s.append(_sync() - t0)
        if eval_every and (epoch + 1) % eval_every == 0 and epoch + 1 < total_epochs:
            t0 = _sync()
            with _cwd(scratch):  # scores are written here and kept only if this epoch is the test-best
                logi, struc = patch_hist(save=True)
            entry = {"epoch": epoch + 1, "logical": logi, "structural": struc, "mean": (logi + struc) / 2}
            test_log.append(entry)
            if ts_dir and entry["mean"] > best["mean"]:
                best = entry
                trainer.save(scratch / _seg_files(category)[0])
                _copy(scratch, Path(ts_dir), _seg_files(category))
            eval_s.append(_sync() - t0)
            print(f"epoch {epoch + 1}: test logical {logi:.2f} structural {struc:.2f} (logged only)", flush=True)

    t0 = _sync()
    trainer.segmentor.eval()
    trainer.save(config["model_path"])
    logi, struc = patch_hist(save=True)
    t_final = _sync() - t0
    last = {"epoch": total_epochs, "logical": logi, "structural": struc, "mean": (logi + struc) / 2}
    if ts_dir and last["mean"] > best["mean"]:
        best = last
        _copy(repo, Path(ts_dir), _seg_files(category))
    sup_only = [s for i, s in enumerate(epoch_s) if i < config["sup_only_epoch"]]
    joint = [s for i, s in enumerate(epoch_s) if i >= config["sup_only_epoch"]]
    info = {
        "stage": "segmentor", "category": category, "seed": seed, "epochs": total_epochs,
        "official_epochs": config["epoch"], "sup_only_epochs": config["sup_only_epoch"],
        "iters_per_epoch": iters_per_epoch, "n_sup": len(sup), "n_unsup": len(unsup),
        "n_train": len(train_loader), "n_val": len(val_loader), "n_test": len(test_loader),
        "setup_s": t_setup - t_start, "epoch_s": epoch_s,
        "sup_only_epoch_s": sum(sup_only) / len(sup_only) if sup_only else None,
        "joint_epoch_s": sum(joint) / len(joint) if joint else None,
        "eval_s": eval_s, "final_s": t_final, "loss": losses, "test_log": test_log,
        "last": last, "test_best": best if ts_dir else None, "total_s": _sync() - t_start,
    }
    _write(out, f"timing_segmentor_{category}", info)
    return info


def lgst(repo: Path, category: str, iterations: int | None = None, seed: int = 0, eval_every: int = 0,
         ts_dir: Path | None = None, out: Path | None = None) -> dict:
    """``CSAD.py`` for one category, evaluating (and so saving) only after
    the last iteration. ``eval_every`` > 0 also logs the official test
    evaluation every that many iterations (the official loop selects on it
    every 1,000); with ``ts_dir`` the test-best state is kept there."""
    repo = _enter(repo)
    import data_loader
    load = data_loader.hf_load_dataset
    data_loader.hf_load_dataset = lambda name, *a, **k: load(PENALTY_SET if name == "Maysee/tiny-imagenet" else name,
                                                              *a, **k)
    import CSAD as official

    t_start = _sync()
    config = official.read_config("./configs/mvtecloco_train.yaml")
    if iterations is not None:
        config["Model"]["iterations"] = iterations
    config["eval_freq"] = config["Model"]["iterations"]
    config.update(category=category, seed=seed, run_name="LGST")
    official.config = config  # CSAD.train reads the module-level config of the official __main__
    model = official.CSAD(config)
    t_setup = _sync()

    total = config["Model"]["iterations"]
    test_log, eval_s = [], []
    best = {"mean": float("-inf")}

    def test_eval(n: int) -> None:
        """The official intermediate evaluation, without its checkpoint selection."""
        t0 = _sync()
        for m in (model.teacher, model.student, model.autoencoder):
            m.eval()
        with torch.no_grad():
            model.q_st_start, model.q_st_end, model.q_ae_start, model.q_ae_end = model.map_normalization(
                validation_loader=model.validation_loader)
            logi, struc = model.test(test_set=model.test_set, iteration=n - 1)
            entry = {"iteration": n, "logical": logi, "structural": struc, "mean": (logi + struc) / 2}
            test_log.append(entry)
            if ts_dir and entry["mean"] > best["mean"]:
                best.clear()
                best.update(entry)
                torch.save({"teacher": model.teacher, "student": model.student, "autoencoder": model.autoencoder,
                            "teacher_mean": model.teacher_mean, "teacher_std": model.teacher_std,
                            "q_ae_end": model.q_ae_end, "q_ae_start": model.q_ae_start,
                            "q_st_end": model.q_st_end, "q_st_start": model.q_st_start},
                           Path(ts_dir) / _lgst_files(category)[0])
                with _cwd(Path(ts_dir)):
                    model.val(test_set=model.validation_loader, iteration=n - 1, save_img=True)
        model.student.train()
        model.autoencoder.train()
        eval_s.append(_sync() - t0)
        print(f"iteration {n}: test logical {logi:.2f} structural {struc:.2f} (logged only)", flush=True)

    # Time the training steps alone: the official loop logs to TensorBoard
    # every 10 iterations and runs the one evaluation inside the last step.
    marks: list[tuple[int, float]] = []
    rates: list[float] = []
    step = model.scheduler.step
    if ts_dir:
        (Path(ts_dir) / "ckpt").mkdir(parents=True, exist_ok=True)

    def timed_step(*a, **k):
        step(*a, **k)
        n = model.scheduler.last_epoch
        if n % 100 == 0:
            now = _sync()
            if marks:
                rates.append((n - marks[-1][0]) / (now - marks[-1][1]))
            marks.append((n, now))
        if eval_every and n % eval_every == 0 and n < total:
            test_eval(n)
            if marks and marks[-1][0] == n:
                marks[-1] = (n, _sync())  # leave the evaluation out of the speed

    model.scheduler.step = timed_step
    best_auc, _, logi, struc = model.train()
    t_end = _sync()
    last = {"iteration": total, "logical": logi, "structural": struc, "mean": best_auc}
    if ts_dir and last["mean"] > best["mean"]:
        best = last
        _copy(repo, Path(ts_dir), _lgst_files(category))
    steady = rates[1:] or rates
    info = {
        "stage": "lgst", "category": category, "seed": seed, "iterations": total,
        "official_iterations": 60000, "batch_size": config["Model"]["batch_size"],
        "n_train": len(model.train_set), "n_val": len(model.validation_set), "n_test": len(model.test_set),
        "n_penalty": len(model.penalty_set) if model.pretrain_penalty else 0,
        "setup_s": t_setup - t_start, "it_per_s": sum(steady) / len(steady) if steady else None,
        "train_s": marks[-1][1] - t_setup if marks else None, "eval_s": eval_s,
        "final_eval_s": t_end - marks[-1][1] if marks else None, "test_log": test_log,
        "last": last, "test_best": best if ts_dir else None, "total_s": t_end - t_start,
    }
    _write(out, f"timing_lgst_{category}_seed{seed}", info)
    return info


def export(repo: Path, category: str, from_dir: Path | None = None, out: Path | None = None) -> Path:
    """Pack both stages from ``from_dir`` (default: the repo, i.e. the last
    states) into ``<from_dir>/ckpt/pytorch_models/<category>.pth``."""
    repo = _enter(repo)
    from export_model import torch_export_model

    base = Path(from_dir).resolve() if from_dir else repo
    t0 = _sync()
    with _cwd(base):
        torch_export_model(category)
    path = base / "ckpt" / "pytorch_models" / f"{category}.pth"
    _write(out, f"timing_export_{category}", {"stage": "export", "category": category, "total_s": _sync() - t0})
    return path


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "segmentor", "lgst", "export"):
        s = sub.add_parser(name)
        s.add_argument("--repo", type=Path, required=True)
        s.add_argument("--category", required=True)
        s.add_argument("--out", type=Path)
        if name == "prepare":
            s.add_argument("--root", type=Path, required=True)
            s.add_argument("--masks", type=Path, required=True)
        if name in ("segmentor", "lgst"):
            s.add_argument("--eval-every", type=int, default=0)
            s.add_argument("--ts-dir", type=Path, help="keep the test-best state here (analysis only)")
        if name == "segmentor":
            s.add_argument("--epochs", type=int)
            s.add_argument("--seed", type=int, default=42)
        if name == "lgst":
            s.add_argument("--iterations", type=int)
            s.add_argument("--seed", type=int, default=0)
        if name == "export":
            s.add_argument("--from-dir", type=Path, help="export the states kept here instead of the last ones")
    a = p.parse_args(argv)
    if a.cmd == "prepare":
        prepare(a.repo, a.root, a.masks, a.category)
    elif a.cmd == "segmentor":
        segmentor(a.repo, a.category, a.epochs, a.seed, a.eval_every, a.ts_dir, a.out)
    elif a.cmd == "lgst":
        lgst(a.repo, a.category, a.iterations, a.seed, a.eval_every, a.ts_dir, a.out)
    else:
        print(export(a.repo, a.category, a.from_dir, a.out))


if __name__ == "__main__":
    main()
