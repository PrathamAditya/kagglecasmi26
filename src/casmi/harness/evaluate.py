"""Score a predictor on the validation set under C1/C2/C3 (ADR-002), with loss decomposition.

uv run python -m casmi.harness.evaluate --config configs/exp_001.yaml [--limit 20] [--regimes C1]
"""
from __future__ import annotations

import argparse
import json
import time
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from ..chem import IK14Cache, mass_tol, neutral_mass
from ..data import load_config, load_pool, scan_train
from ..predictors import build_predictor

TOP_K = 25
BUCKETS = ["hit@1", "hit@2-25", "cand_not_top25", "reachable_missed", "unreachable"]

def git_state() -> dict:
    run = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "--short", "HEAD"), "dirty": bool(run("status", "--porcelain", "--", "src", "configs"))}

def exclusion(regime: str, lib: str, val_iks: list[str]) -> pl.Expr:
    expr = pl.col("ingest_lib") == lib                          # C1: remove the query spectra themselves
    if regime in ("C2", "C3"):
        expr = expr | pl.col("inchikey14").is_in(val_iks)       # C2/C3: remove every spectrum of the structure
    return expr

def regime_pool(pool: pd.DataFrame, regime: str, labels: pd.DataFrame, ref_counts: dict) -> pd.DataFrame:
    """Pool as it would look for this regime. `in_train` must not leak structures that have no reference spectra."""
    pool = pool.copy()
    no_ref = set()
    for ik, mk in zip(labels["inchikey14"], labels["metric_ik14"]):
        if ref_counts.get(ik, 0) == 0:
            no_ref |= {ik, mk}
    pool.loc[pool["metric_ik14"].isin(no_ref), "in_train"] = False
    if regime == "C3":
        pool = pool.loc[~pool["metric_ik14"].isin(set(labels["metric_ik14"]) | set(labels["inchikey14"]))]
    return pool.reset_index(drop=True)


def diagnostics(queries, labels, ref: pd.DataFrame, pool: pd.DataFrame, ppm: float, min_da: float) -> pd.DataFrame:
    """Per molecule: is the truth reachable at all, and how crowded is the precursor window?"""
    ref_mz = np.sort(ref["precursor_mz"].to_numpy(np.float64))
    truth_ref = ref.loc[ref["inchikey14"].isin(set(labels["inchikey14"]))]
    truth_ref_by_ik = {ik: g for ik, g in truth_ref.groupby("inchikey14")}
    pool_mass = pool.groupby("metric_ik14")["exact_mass"].first().to_dict()
    rows = []
    for lab in labels.itertuples(index=False):
        q = queries.loc[queries["molecule_id"] == lab.molecule_id]
        g = truth_ref_by_ik.get(lab.inchikey14)
        window_max, same_adduct_hit, pool_hit = 0, False, False
        tm = pool_mass.get(lab.metric_ik14)
        for s in q.itertuples(index=False):
            mz = float(s.precursor_mz); tol = mass_tol(mz, ppm, min_da)
            window_max = max(window_max, int(np.searchsorted(ref_mz, mz + tol, "right") - np.searchsorted(ref_mz, mz - tol, "left")))
            if g is not None:
                same_adduct_hit |= bool(((g["adduct"] == s.adduct) & ((g["precursor_mz"] - mz).abs() <= tol)).any())
            m = neutral_mass(mz, s.adduct)
            if tm is not None and m is not None:
                pool_hit |= abs(tm - m) <= mass_tol(m, ppm, min_da)
        rows.append(dict(molecule_id=lab.molecule_id, n_ref_truth=0 if g is None else len(g),
                         ref_truth_same_adduct_in_tol=same_adduct_hit, window_max=window_max,
                         truth_in_pool=tm is not None, pool_truth_in_tol=pool_hit))
    return pd.DataFrame(rows)


def score(preds: dict, labels: pd.DataFrame, cache: IK14Cache, regime: str) -> pd.DataFrame:
    tops = {m: list(dict.fromkeys(c.smiles for c in cands))[:TOP_K] for m, cands in preds.items()}
    keymap = cache.get_many([s for t in tops.values() for s in t]); cache.save()
    rows = []
    for lab in labels.itertuples(index=False):
        cands, top = preds[lab.molecule_id], tops[lab.molecule_id]
        rank = next((i + 1 for i, s in enumerate(top) if keymap[s] == lab.metric_ik14), None)
        truth = {lab.inchikey14, lab.metric_ik14}
        in_cand = rank is not None or any(c.key in truth for c in cands)
        src = {c.smiles: c.source for c in cands}
        leak = regime != "C1" and any(src.get(s) == "library" and keymap[s] == lab.metric_ik14 for s in top)
        rows.append(dict(molecule_id=lab.molecule_id, rank=rank, rr=0.0 if rank is None else 1.0 / rank,
                         in_candidates=in_cand, n_candidates=len(cands), leak=leak,
                         top3=" | ".join(top[:3])))
    return pd.DataFrame(rows)


def bucket(r) -> str:
    if r["rank"] == 1: return "hit@1"
    if pd.notna(r["rank"]): return "hit@2-25"
    if r["in_candidates"]: return "cand_not_top25"
    if r["ref_truth_same_adduct_in_tol"] or r["pool_truth_in_tol"]: return "reachable_missed"
    return "unreachable"


def run_regime(regime, cfg, queries, labels, train_lf, pool_all, cache, out_dir) -> dict:
    paths, vcfg, dcfg = cfg["paths"], cfg["val"], cfg.get("diag", {"ppm": 20.0, "min_da": 0.01})
    excl = exclusion(regime, vcfg["lib"], labels["inchikey14"].tolist() + tautomer_twins(train_lf, labels, cache))
    ref_lf = train_lf.filter(~excl)
    ref = ref_lf.select(["precursor_mz", "adduct", "inchikey14"]).collect().to_pandas()
    ref_counts = ref["inchikey14"].value_counts().to_dict()
    pool = regime_pool(pool_all, regime, labels, ref_counts)

    predictor = build_predictor(cfg["predictor"]["name"], cfg["predictor"].get("params"))
    t0 = time.time()
    predictor.prepare(ref_lf, pool, queries)
    t_prep = time.time() - t0
    preds = {m: predictor.predict(g) for m, g in queries.groupby("molecule_id", sort=True)}
    t_total = time.time() - t0

    per = score(preds, labels, cache, regime).merge(
        diagnostics(queries, labels, ref, pool, dcfg["ppm"], dcfg["min_da"]), on="molecule_id")
    per = labels[["molecule_id", "smiles", "adducts", "n_spectra"]].merge(per, on="molecule_id")
    per["bucket"] = per.apply(bucket, axis=1)

    rdir = out_dir / regime; rdir.mkdir(parents=True, exist_ok=True)
    per.to_csv(rdir / "per_molecule.csv", index=False)
    worst = per.sort_values(["rr", "in_candidates", "window_max"], ascending=[True, True, False]).head(20)
    worst.to_csv(rdir / "worst20.csv", index=False)

    counts = per["bucket"].value_counts()
    summary = dict(
        regime=regime, molecules=len(per), mrr=round(per["rr"].mean(), 4),
        recall_at_cand=round(per["in_candidates"].mean(), 4), recall_at_25=round(per["rank"].notna().mean(), 4),
        buckets={b: int(counts.get(b, 0)) for b in BUCKETS},
        # Ceiling analysis: MRR if everything in the candidate set were ranked #1.
        oracle_mrr_given_candidates=round(per["in_candidates"].mean(), 4),
        crowded_windows_gt256=int((per["window_max"] > 256).sum()),
        leak_hits=int(per["leak"].sum()), runtime_prepare_s=round(t_prep, 1), runtime_total_s=round(t_total, 1),
    )
    (rdir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary

def tautomer_twins(train_lf: pl.LazyFrame, labels: pd.DataFrame, cache: IK14Cache) -> list[str]:
    """Train ik14s whose METRIC key equals a val molecule's (tautomers). Only same-formula structures checked."""
    val_iks = labels["inchikey14"].tolist()
    forms = train_lf.filter(pl.col("inchikey14").is_in(val_iks)).select("molecular_formula").unique().collect()
    cand = (train_lf.filter(pl.col("molecular_formula").is_in(forms["molecular_formula"].to_list())
                            & ~pl.col("inchikey14").is_in(val_iks))
            .select("inchikey14", "normalized_smiles").unique().collect())
    keys = cache.get_many(cand["normalized_smiles"].to_list()); cache.save()
    targets = set(labels["metric_ik14"])
    return sorted({ik for ik, smi in zip(cand["inchikey14"], cand["normalized_smiles"]) if keys[smi] in targets})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--regimes", nargs="*")
    ap.add_argument("--limit", type=int, default=0, help="first N molecules only (smoke test)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths = cfg["paths"]
    val_dir, out_dir = Path(paths["val_dir"]), Path(paths["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = pd.read_parquet(val_dir / "queries.parquet")
    labels = pd.read_parquet(val_dir / "labels.parquet").sort_values("molecule_id").reset_index(drop=True)
    if args.limit:
        labels = labels.head(args.limit)
        queries = queries.loc[queries["molecule_id"].isin(labels["molecule_id"])].reset_index(drop=True)

    cache = IK14Cache(Path(paths["cache_dir"]) / "metric_ik14.parquet")
    train_lf, pool_all = scan_train(paths["train"]), load_pool(paths["pool"])
    regimes = args.regimes or cfg["regimes"]
    summaries = {r: run_regime(r, cfg, queries, labels, train_lf, pool_all, cache, out_dir) for r in regimes}

    w = cfg["lb_weights"]
    est = sum(w[r] * summaries[r]["mrr"] for r in summaries) / sum(w[r] for r in summaries)
    final = dict(exp_id=cfg["exp_id"], predictor=cfg["predictor"], limit=args.limit, est_lb=round(est, 4),
                 regimes=summaries)
    (out_dir / "results.json").write_text(json.dumps(final, indent=2))

    print(json.dumps(final, indent=2))
    c = {r: f"{summaries[r]['mrr']:.3f}" if r in summaries else "" for r in ("C1", "C2", "C3")}
    print("\nexp-log row:")
    print(f"| {cfg['exp_id'][-3:]} | done | | | | {c['C1']} | {c['C2']} | {c['C3']} | {est:.3f} | | | | |")


if __name__ == "__main__":
    main()
