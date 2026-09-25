import numpy as np
import pandas as pd

from casmi.harness.compare import compare, gate, paired_bootstrap


def _run(rr):
    rr = np.asarray(rr, float)
    return pd.DataFrame({"molecule_id": [f"m{i:03d}" for i in range(len(rr))], "rr": rr,
                         "in_candidates": rr > 0, "rank": np.where(rr > 0, 1.0, np.nan), "leak": False})


def test_identical_runs_have_zero_width_ci():
    a = np.random.default_rng(1).random(200)
    r = paired_bootstrap(a, a)
    assert r["delta"] == r["lo"] == r["hi"] == 0.0


def test_constant_shift_is_recovered_exactly():
    a = np.random.default_rng(2).random(200)
    r = paired_bootstrap(a, a + 0.1)
    assert np.isclose(r["delta"], 0.1) and np.isclose(r["lo"], 0.1) and np.isclose(r["hi"], 0.1)


def test_pairing_narrows_ci_vs_noise():
    rng = np.random.default_rng(3)
    a = rng.random(250)
    b = a + rng.normal(0.02, 0.01, 250)          # small consistent gain on top of big per-molecule variance
    r = paired_bootstrap(a, b)
    assert r["lo"] > 0 and r["hi"] < 0.04


def test_compare_and_gate():
    rng = np.random.default_rng(4)
    base = {r: _run(rng.random(100)) for r in ("C1", "C2", "C3")}
    cand = {r: df.assign(rr=df["rr"] + (0.05 if r == "C2" else 0.0)) for r, df in base.items()}
    res = compare(base, cand, {"C1": 0.5, "C2": 0.35, "C3": 0.15})
    assert res["shared_resample"]
    assert np.isclose(res["est_lb"]["delta"], 0.35 * 0.05)
    g = gate(res, "C2")
    assert g["works"] and g["c1_lo_gt_minus_0_01"]
