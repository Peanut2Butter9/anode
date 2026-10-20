"""Detection and localisation metrics."""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.metrics import precision_recall_curve, roc_auc_score


def image_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores))


def pixel_auroc(masks: np.ndarray, maps: np.ndarray) -> float:
    """``masks`` (N, H, W) bool and ``maps`` (N, H, W) float, flattened together."""
    return float(roc_auc_score(masks.ravel(), maps.ravel()))


def f1_max(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    """Best F1 over all thresholds, and the threshold that reaches it.

    It picks the threshold using the test labels, so it is an optimistic upper
    bound. It is not a deployable operating point.
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    best = int(np.argmax(f1[:-1]))  # the last point has no threshold
    return float(f1[best]), float(thresholds[best])


def validation_threshold(val_scores: np.ndarray, k: float = 3.0) -> float:
    """Decision threshold from normal validation images only: mean + k * std.

    This uses the same 3-sigma rule as a Shewhart control chart, and it never
    looks at test labels.
    """
    return float(np.mean(val_scores) + k * np.std(val_scores, ddof=1))


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = scores >= threshold
    tp = int(np.sum(pred & (labels == 1)))
    tn = int(np.sum(~pred & (labels == 0)))
    fp = int(np.sum(pred & (labels == 0)))
    fn = int(np.sum(~pred & (labels == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "accuracy": (tp + tn) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def aupro(masks: np.ndarray, maps: np.ndarray, fpr_limit: float = 0.3) -> float:
    """Area under the per-region-overlap curve up to ``fpr_limit``, normalised to [0, 1].

    It follows the reference implementation released with MVTec AD (Bergmann et
    al., 2021). Every pixel is a threshold. Each connected ground-truth region
    counts equally, whatever its size, and the FPR is computed over all normal
    pixels.
    """
    masks = masks.astype(bool)
    n_regions = 0
    fp_change = np.zeros(masks.shape, dtype=np.float32)
    pro_change = np.zeros(masks.shape, dtype=np.float32)
    structure = np.ones((3, 3), dtype=int)
    for i, m in enumerate(masks):
        labeled, n = ndimage.label(m, structure=structure)
        n_regions += n
        fp_change[i][labeled == 0] = 1.0
        if n:
            sizes = np.bincount(labeled.ravel())
            region = labeled > 0
            pro_change[i][region] = 1.0 / sizes[labeled[region]]
    n_ok = fp_change.sum()
    if n_regions == 0 or n_ok == 0:
        return float("nan")

    order = np.argsort(maps.ravel(), kind="stable")[::-1]
    scores_sorted = maps.ravel()[order]
    fpr = np.cumsum(fp_change.ravel()[order], dtype=np.float64) / n_ok
    pro = np.cumsum(pro_change.ravel()[order], dtype=np.float64) / n_regions
    del order

    # Keep one point per distinct threshold (the last pixel with that score)
    keep = np.append(np.diff(scores_sorted) != 0, True)
    fpr = np.concatenate([[0.0], fpr[keep]])
    pro = np.concatenate([[0.0], pro[keep]])

    # Integrate up to fpr_limit, interpolating the last segment
    end = np.searchsorted(fpr, fpr_limit, side="right")
    x, y = fpr[:end], pro[:end]
    if end < len(fpr) and x[-1] < fpr_limit:
        x0, x1, y0, y1 = fpr[end - 1], fpr[end], pro[end - 1], pro[end]
        y_lim = y0 + (y1 - y0) * (fpr_limit - x0) / (x1 - x0)
        x, y = np.append(x, fpr_limit), np.append(y, y_lim)
    return float(np.trapezoid(y, x) / fpr_limit) if hasattr(np, "trapezoid") else float(np.trapz(y, x) / fpr_limit)
