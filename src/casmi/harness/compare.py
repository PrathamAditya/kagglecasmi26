"""Paired bootstrap comparison of two harness runs (CLAUDE.md eval protocol). Shared by every run.

uv run python -m casmi.harness.compare --anchor experiments/exp_016_v27_anchor/results \
    --candidate experiments/exp_004_entropy_search/results --config configs/exp_016.yaml \
    [--target C2] [--out experiments/exp_016_v27_anchor/compare_exp004.json]

Per regime: MRR, Recall@cand, Recall@25, n and the paired bootstrap 95% CI of the MRR delta
(candidate − anchor), resampling MOLECULES. est LB uses `lb_weights` from the config (printed).
When every regime covers the same molecules (the 250-set), one set of molecule indices is drawn
per resample and shared across regimes, so the est-LB CI keeps their correlation; otherwise
(val_v2: disjoint regime slices) each regime is resampled independently.
Gate: target-regime delta CI lo > 0 AND C1 delta CI lo > −0.01.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import load_config

REGIMES = ("C1", "C2", "C3")


def load_run(results_dir: str | Path) -> dict[str, pd.DataFrame]:
    out = {}
    for r in REGIMES:
        f = Path(results_dir) / r / "per_molecule.csv"
        if f.is_file():
            df = pd.read_csv(f).sort_values("molecule_id").reset_index(drop=True)
            out[r] = df[["molecule_id", "rr", "in_candidates", "rank", "leak"]]
    return out


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 1000, seed: int = 0,
                     idx: np.ndarray | None = None) -> dict:
    """Mean delta (b − a) and percentile 95% CI over resampled units. `idx` (n_boot × n) lets callers
    share resample indices across metrics."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape and a.ndim == 1
    if idx is None:
        idx = np.random.default_rng(seed).integers(0, len(a), size=(n_boot, len(a)))
    d = b - a
    boots = d[idx].mean(axis=1)
    return dict(delta=float(d.mean()), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)),
                boots=boots)


def _metrics(df: pd.DataFrame) -> dict:
    return dict(n=len(df), mrr=float(df["rr"].mean()), recall_at_cand=float(df["in_candidates"].mean()),
                recall_at_25=float(df["rank"].notna().mean()), leak_hits=int(df["leak"].sum()))


def compare(anchor: dict, cand: dict, weights: dict, n_boot: int = 1000, seed: int = 0) -> dict:
    regimes = [r for r in REGIMES if r in anchor and r in cand]
    for r in regimes:
        if anchor[r]["molecule_id"].tolist() != cand[r]["molecule_id"].tolist():
            raise ValueError(f"{r}: anchor and candidate were not run on the same molecules")
    same_mols = len({tuple(anchor[r]["molecule_id"]) for r in regimes}) == 1
    rng = np.random.default_rng(seed)
    shared = rng.integers(0, len(anchor[regimes[0]]), size=(n_boot, len(anchor[regimes[0]]))) if same_mols else None

    out, boots_by_regime = {"regimes": {}, "shared_resample": same_mols, "n_boot": n_boot, "seed": seed}, {}
    for r in regimes:
        idx = shared if same_mols else rng.integers(0, len(anchor[r]), size=(n_boot, len(anchor[r])))
        bs = paired_bootstrap(anchor[r]["rr"].to_numpy(), cand[r]["rr"].to_numpy(), idx=idx)
        boots_by_regime[r] = bs.pop("boots")
        out["regimes"][r] = dict(anchor=_metrics(anchor[r]), candidate=_metrics(cand[r]), mrr_delta=bs)

    wsum = sum(weights[r] for r in regimes)
    est = lambda run: sum(weights[r] * float(run[r]["rr"].mean()) for r in regimes) / wsum
    lb_boots = sum(weights[r] * boots_by_regime[r] for r in regimes) / wsum
    out["est_lb"] = dict(weights={r: weights[r] for r in regimes}, anchor=est(anchor), candidate=est(cand),
                         delta=est(cand) - est(anchor), lo=float(np.percentile(lb_boots, 2.5)),
                         hi=float(np.percentile(lb_boots, 97.5)))
    return out


def gate(res: dict, target: str | None) -> dict:
    reg = res["regimes"]
    c1_ok = "C1" not in reg or reg["C1"]["mrr_delta"]["lo"] > -0.01
    tgt_ok = None if target is None else reg[target]["mrr_delta"]["lo"] > 0
    return dict(target=target, target_lo_gt_0=tgt_ok, c1_lo_gt_minus_0_01=c1_ok,
                works=bool(c1_ok and tgt_ok) if target else None)


def format_table(res: dict, anchor_name: str, cand_name: str) -> str:
    lines = [f"candidate = {cand_name}   anchor = {anchor_name}   (Δ = candidate − anchor, paired bootstrap "
             f"{res['n_boot']}×, 95% CI, shared resample={res['shared_resample']})",
             "regime    n  anchor MRR  cand MRR   Δ MRR [lo, hi]            cand R@cand  cand R@25  leak"]
    for r, v in res["regimes"].items():
        a, c, d = v["anchor"], v["candidate"], v["mrr_delta"]
        lines.append(f"{r:<6} {c['n']:>4}  {a['mrr']:>10.4f}  {c['mrr']:>8.4f}   {d['delta']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]"
                     f"   {c['recall_at_cand']:>10.3f}  {c['recall_at_25']:>9.3f}  {c['leak_hits']}/{a['leak_hits']}")
    e = res["est_lb"]
    w = ", ".join(f"{k}={v}" for k, v in e["weights"].items())
    lines.append(f"est LB (weights {w}): anchor {e['anchor']:.4f}  cand {e['candidate']:.4f}  "
                 f"Δ {e['delta']:+.4f} [{e['lo']:+.4f}, {e['hi']:+.4f}]")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--config", required=True, help="source of lb_weights")
    ap.add_argument("--target", choices=REGIMES)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out")
    args = ap.parse_args()
    weights = load_config(args.config)["lb_weights"]
    res = compare(load_run(args.anchor), load_run(args.candidate), weights, args.n_boot, args.seed)
    res["gate"] = gate(res, args.target)
    res.update(anchor=args.anchor, candidate=args.candidate)
    print(format_table(res, args.anchor, args.candidate))
    print("gate:", json.dumps(res["gate"]))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
