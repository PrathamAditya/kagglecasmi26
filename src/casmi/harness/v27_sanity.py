"""exp_016 A1 tooling: calibrate V27's ppm offset once, build notebook-reference inputs, compare.

uv run python -m casmi.harness.v27_sanity calibrate --config configs/exp_016.yaml
uv run python -m casmi.harness.v27_sanity ref       --config configs/exp_016.yaml --regime C1
uv run python -m casmi.harness.evaluate --config configs/exp_016.yaml --limit 20 --regimes C1 C2
uv run python -m casmi.harness.v27_sanity compare   --config configs/exp_016.yaml --regime C1

`ref` writes the SAME regime-filtered library the harness gives the port (same exclusion code,
same --limit molecules, same row order) to a parquet, lays it out like /kaggle/input, and runs
the notebook's own cells on it (casmi.harness.v27_notebook_ref).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..chem import IK14Cache
from ..data import load_config, load_pool, scan_train
from ..predictors import build_predictor
from ..predictors.exp016_v27 import LIBRARY_COLUMNS
from .evaluate import exclusion, tautomer_twins


def _val(cfg: dict, n: int):
    val_dir = Path(cfg["paths"]["val_dir"])
    queries = pd.read_parquet(val_dir / "queries.parquet")
    labels = pd.read_parquet(val_dir / "labels.parquet").sort_values("molecule_id").reset_index(drop=True)
    if n:
        labels = labels.head(n)
        queries = queries.loc[queries["molecule_id"].isin(labels["molecule_id"])].reset_index(drop=True)
    return queries, labels


def _regime_train(cfg: dict, regime: str, labels: pd.DataFrame):
    paths = cfg["paths"]
    train_lf = scan_train(paths["train"])
    cache = IK14Cache(Path(paths["cache_dir"]) / "metric_ik14.parquet")
    excl_iks = labels["inchikey14"].tolist() + tautomer_twins(train_lf, labels, cache)
    return train_lf.filter(~exclusion(regime, cfg["val"]["lib"], excl_iks))


def cmd_calibrate(cfg: dict, args) -> None:
    """Label-free: V27's own calibrate_ppm_offset on the full val query file with the C1 library."""
    queries, labels = _val(cfg, 0)
    params = dict(cfg["predictor"]["params"], ppm_offset="calibrate")
    pred = build_predictor(cfg["predictor"]["name"], params)
    pred.prepare(_regime_train(cfg, "C1", labels), load_pool(cfg["paths"]["pool"]), queries)
    out = Path(cfg["paths"]["out_dir"]); out.mkdir(parents=True, exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(pred.calibration, indent=2))
    print(json.dumps(pred.calibration, indent=2))


def cmd_ref(cfg: dict, args) -> None:
    s = cfg["sanity"]
    offset = cfg["predictor"]["params"]["ppm_offset"]
    if offset == "calibrate":
        raise SystemExit("freeze ppm_offset in the config first (run `calibrate`)")
    queries, labels = _val(cfg, s["n_molecules"])
    base = Path(s["ref_dir"]) / args.regime
    comp, art = base / "input" / "comp", base / "input" / "artifacts"
    comp.mkdir(parents=True, exist_ok=True); art.mkdir(parents=True, exist_ok=True)

    train_pq = comp / "train.parquet"
    if not train_pq.exists():
        _regime_train(cfg, args.regime, labels).select(LIBRARY_COLUMNS).sink_parquet(train_pq)
    queries.to_parquet(comp / "test.parquet")
    pd.DataFrame({"molecule_id": sorted(queries["molecule_id"].unique())}).to_csv(
        comp / "sample_submission.csv", index=False)
    p = cfg["predictor"]["params"]
    links = [Path(p["pool_npz"]), Path(p["rankers"]), *sorted(Path(p["fp_models"]).glob("fp_*.pt"))]
    for src in links:
        dst = art / src.name
        if not dst.exists():
            dst.symlink_to(src.resolve())

    cmd = [sys.executable, "-m", "casmi.harness.v27_notebook_ref", "--notebook", str(Path(s["notebook"]).resolve()),
           "--input-root", str((base / "input").resolve()), "--workdir", str((base / "run").resolve()),
           "--ppm-offset", str(offset)]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=dict(os.environ))


def _strip_pad(tokens: list[str], pad: str = "CCO") -> list[str]:
    """Undo the notebook's pad_candidates(): windows with < 25 candidates are padded with 'CCO'."""
    while tokens and tokens[-1] == pad:
        tokens = tokens[:-1]
    return tokens


def cmd_compare(cfg: dict, args) -> None:
    s = cfg["sanity"]
    ref = pd.read_csv(Path(s["ref_dir"]) / args.regime / "run" / "submission.csv")
    ref = {m: _strip_pad(v.split(";")) for m, v in zip(ref["molecule_id"], ref["smiles"])}
    port = json.loads((Path(cfg["paths"]["out_dir"]) / args.regime / "top25.json").read_text())
    rows = []
    for m in sorted(ref):
        a, b = ref[m], port.get(m, [])
        first_diff = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
        rows.append(dict(molecule_id=m, n_ref=len(a), n_port=len(b), top1_equal=bool(b) and a[0] == b[0],
                         overlap25=len(set(a) & set(b)),
                         identical_order=a == b, first_diff_rank=None if first_diff is None else first_diff + 1))
    df = pd.DataFrame(rows)
    out = Path(cfg["paths"]["out_dir"]) / args.regime / "sanity_vs_notebook.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    df["overlap_ok"] = df["overlap25"] >= np.minimum(df[["n_ref", "n_port"]].max(axis=1), 25) - 1   # >= 24/25
    ok = bool(df["top1_equal"].all() and df["overlap_ok"].all())
    summary = dict(regime=args.regime, n=len(df), top1_equal=int(df["top1_equal"].sum()),
                   identical_order=int(df["identical_order"].sum()), min_overlap25=int(df["overlap25"].min()),
                   gate_pass=ok)
    print(json.dumps(summary))
    (out.with_suffix(".json")).write_text(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["calibrate", "ref", "compare"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--regime", default="C1")
    args = ap.parse_args()
    cfg = load_config(args.config)
    {"calibrate": cmd_calibrate, "ref": cmd_ref, "compare": cmd_compare}[args.cmd](cfg, args)


if __name__ == "__main__":
    main()
