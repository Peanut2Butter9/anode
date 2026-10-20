import math

import numpy as np
import pytest

from anode import qc


def test_p_chart_limits_follow_batch_size():
    r = qc.p_chart([5, 10, 2], [100, 200, 50])
    p_bar = 17 / 350
    assert math.isclose(r["p_bar"], p_bar)
    for pt, n in zip(r["points"], [100, 200, 50]):
        assert math.isclose(pt["ucl"], p_bar + 3 * math.sqrt(p_bar * (1 - p_bar) / n))
        assert pt["lcl"] == max(0.0, p_bar - 3 * math.sqrt(p_bar * (1 - p_bar) / n))
    # Larger batches get tighter limits
    assert r["points"][1]["ucl"] < r["points"][0]["ucl"] < r["points"][2]["ucl"]


def test_p_chart_flags_out_of_control_batch():
    r = qc.p_chart([1, 1, 1, 1, 30], [100] * 5)
    assert [pt["out_of_control"] for pt in r["points"]] == [False, False, False, False, True]


def test_p_chart_rejects_bad_input():
    with pytest.raises(ValueError):
        qc.p_chart([1], [0])


def test_capability_one_sided():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.1, 10_000)
    c = qc.capability(x, usl=1.0)
    assert abs(c["cpk"] - (1.0 - 0.5) / (3 * 0.1)) < 0.03  # about 1.67
    assert c["cp"] is None and c["rating"] == "capable"


def test_capability_two_sided_takes_the_worse_side():
    c = qc.capability([0.2, 0.3, 0.4], usl=1.0, lsl=0.0)
    assert math.isclose(c["cpk"], min(c["cpu"], c["cpl"])) and c["cp"] is not None


def test_cpk_from_yield():
    # 0.135% above a one-sided limit is the classic 3-sigma process, Cpk = 1
    assert abs(qc.cpk_from_yield(0.00135) - 1.0) < 1e-3
    assert qc.cpk_from_yield(0.0) is None
    c = qc.capability([0.1] * 97 + [5.0] * 3, usl=1.0)
    assert abs(c["cpk_from_yield"] - qc.cpk_from_yield(0.03)) < 1e-12
    assert c["cpk_from_yield"] > c["cpk"]  # the bimodal mixture deflates the normal-theory index


def test_rating_bands():
    assert [qc.rate(v) for v in (1.5, 1.1, 0.5)] == ["capable", "marginal", "not capable"]
