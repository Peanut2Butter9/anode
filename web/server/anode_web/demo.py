"""Simulated production history so the report pages have data on first visit.

Each simulated unit draws a real anomaly score from the product's test set
(good or defective, at the batch's true defect rate). The decision threshold
then classifies it exactly as in a live inspection. One batch has a raised
defect rate (an "assignable cause"), so the P-chart shows what an
out-of-control point looks like.
"""

from __future__ import annotations

import datetime as dt
import hashlib

import numpy as np


def history(product_id: str, scores: dict, threshold: float, days: int = 30, today: dt.date | None = None) -> list[dict]:
    seed = int(hashlib.sha256(product_id.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    good = np.asarray(scores["good"], dtype=float)
    bad = np.asarray(scores["defective"], dtype=float)
    today = today or dt.date.today()
    dates = [today - dt.timedelta(days=d) for d in range(days, 0, -1)]
    dates = [d for d in dates if d.weekday() < 6]  # Monday to Saturday
    spike = int(rng.integers(len(dates) // 3, len(dates)))
    base_rate = float(rng.uniform(0.02, 0.05))
    batches = []
    for i, day in enumerate(dates):
        n = int(rng.integers(80, 161))
        rate = base_rate * (4 if i == spike else 1)
        is_bad = rng.random(n) < rate
        s = np.where(is_bad, rng.choice(bad, n), rng.choice(good, n)) / threshold
        flagged = s >= 1.0
        batches.append({
            "id": f"demo-{product_id}-{day.isoformat()}",
            "date": day.isoformat(),
            "batch_no": f"B{day.strftime('%y%m%d')}-{i + 1:02d}",
            "n": n,
            "defective": int(flagged.sum()),
            "scores": [round(float(x), 4) for x in s],
            "demo": True,
        })
    return batches
