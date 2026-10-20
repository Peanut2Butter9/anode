"""Statistical process control for inspection results.

- ``p_chart``: Shewhart P-chart of the defective fraction per batch, with
  3-sigma limits that depend on each batch's size.
- ``capability``: process capability of the continuous anomaly score against
  an upper specification limit (USL). A larger score means "more anomalous",
  so only the upper side matters and Cpk = Cpu = (USL - mean) / (3 sigma).
  With the score normalised by the model's decision threshold, USL = 1.

  That formula assumes normally distributed scores. A batch that mixes good
  and defective parts gives a bimodal distribution with an inflated sigma, so
  we also report the yield-based equivalent Phi^-1(1 - p) / 3, where p is the
  observed fraction above the USL (the "Z-bench" reading of Cpk).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.stats import norm


def p_chart(defectives: list[int], sizes: list[int], k: float = 3.0) -> dict:
    """Center line p-bar = total defectives / total inspected. Per batch i:
    UCL_i = p-bar + k * sqrt(p-bar (1 - p-bar) / n_i), and LCL_i is the same
    with a minus sign, clipped at 0."""
    d = np.asarray(defectives, dtype=float)
    n = np.asarray(sizes, dtype=float)
    if len(d) != len(n) or not len(n) or (n <= 0).any():
        raise ValueError("need one positive batch size per batch")
    p_bar = float(d.sum() / n.sum())
    sigma = np.sqrt(p_bar * (1 - p_bar) / n)
    p = d / n
    ucl = np.minimum(1.0, p_bar + k * sigma)
    lcl = np.maximum(0.0, p_bar - k * sigma)
    return {
        "p_bar": p_bar,
        "points": [
            {"p": float(pi), "ucl": float(u), "lcl": float(lo), "out_of_control": bool(pi > u or pi < lo)}
            for pi, u, lo in zip(p, ucl, lcl)
        ],
    }


def capability(scores, usl: float, lsl: float | None = None) -> dict:
    """Process capability of ``scores`` against the spec limits.

    Returns mean, std (sample), Cpu, and Cpk (= Cpu for a one-sided upper
    spec). Cp needs both limits, so it is only given when ``lsl`` is set.
    """
    x = np.asarray(scores, dtype=float)
    if len(x) < 2:
        raise ValueError("need at least two scores")
    mu, sigma = float(x.mean()), float(x.std(ddof=1))
    if sigma == 0:
        sigma = math.ulp(1.0)
    cpu = (usl - mu) / (3 * sigma)
    above = float((x > usl).mean())
    out = {"mean": mu, "std": sigma, "usl": usl, "cpu": cpu, "cpk": cpu, "cp": None,
           "fraction_above_usl": above, "cpk_from_yield": cpk_from_yield(above)}
    if lsl is not None:
        cpl = (mu - lsl) / (3 * sigma)
        out.update(lsl=lsl, cpl=cpl, cpk=min(cpu, cpl), cp=(usl - lsl) / (6 * sigma))
    out["rating"] = rate(out["cpk"])
    return out


def cpk_from_yield(p: float) -> float | None:
    """Cpk equivalent of a defective fraction ``p``: Phi^-1(1 - p) / 3.
    None when p is 0 (the index is unbounded) or 1."""
    if not 0 < p < 1:
        return None
    return float(norm.ppf(1 - p) / 3)


def rate(cpk: float) -> str:
    """Common reading of Cpk: >= 1.33 capable, 1.00-1.33 marginal, < 1 not capable."""
    if cpk >= 1.33:
        return "capable"
    if cpk >= 1.0:
        return "marginal"
    return "not capable"
