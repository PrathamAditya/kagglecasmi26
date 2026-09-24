"""Build the frozen validation set (ADR-002) from enveda-np-examples.

uv run python -m casmi.harness.build_val --config configs/exp_001.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from ..chem import TEST_ADDUCTS, IK14Cache
from ..data import load_config, scan_train, to_test_schema


def _trim(mzs, ints, precursor):
    """Mimic test cleaning: drop peaks above precursor + 2 Da, renormalize to base peak = 1."""
    mzs, ints = np.asarray(mzs, float), np.asarray(ints, float)
    keep = np.isfinite(mzs) & np.isfinite(ints) & (mzs <= precursor + 2.0)
    mzs, ints = mzs[keep], ints[keep]
    if ints.size and ints.max() > 0:
        ints = ints / ints.max()
    return mzs, ints


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    cfg = load_config(args.config)
    paths, vcfg = cfg["paths"], cfg["val"]
    out = Path(paths["val_dir"]); out.mkdir(parents=True, exist_ok=True)

    raw = scan_train(paths["train"]).filter(pl.col("ingest_lib") == vcfg["lib"]).collect().to_pandas()
    report: dict = {"lib": vcfg["lib"], "raw_spectra": len(raw), "raw_molecules": int(raw["inchikey14"].nunique())}
    report["adduct_counts_raw"] = raw["adduct"].value_counts(dropna=False).to_dict()

    # Precursor ppm error by adduct — the number exp_002 needs for tolerance.
    ppm = raw["precursor_error_ppm"].abs()
    report["precursor_abs_ppm_quantiles"] = {
        str(q): float(ppm.quantile(q)) for q in (0.5, 0.9, 0.95, 0.99, 1.0) if ppm.notna().any()
    }

    keep = raw["adduct"].isin(TEST_ADDUCTS)
    keep &= ppm.isna() | (ppm <= vcfg["max_precursor_ppm"])
    bpi = raw["base_peak_intensity"]
    keep &= bpi.isna() | (bpi >= vcfg["min_base_peak"])
    val = raw.loc[keep].copy()

    trimmed = [_trim(m, i, p) for m, i, p in zip(val["ms2_mzs"], val["ms2_normalized_intensities"], val["precursor_mz"])]
    val["ms2_mzs"] = [t[0] for t in trimmed]
    val["ms2_normalized_intensities"] = [t[1] for t in trimmed]
    val = val.loc[val["ms2_mzs"].map(len) > 0]

    iks = sorted(val["inchikey14"].unique())
    mid = {ik: f"v_{i:04d}" for i, ik in enumerate(iks)}
    val["molecule_id"] = val["inchikey14"].map(mid)
    if "spectrum_id" not in val.columns or val["spectrum_id"].isna().any():
        val["spectrum_id"] = "s_" + val["train_row"].astype(str)

    labels = (val.groupby("molecule_id")
              .agg(inchikey14=("inchikey14", "first"),
                   smiles=("normalized_smiles", lambda s: s.mode().iat[0]),
                   n_spectra=("spectrum_id", "size"),
                   adducts=("adduct", lambda s: ",".join(sorted(set(s)))))
              .reset_index())
    cache = IK14Cache(Path(paths["cache_dir"]) / "metric_ik14.parquet")
    keys = cache.get_many(labels["smiles"].tolist()); cache.save()
    labels["metric_ik14"] = labels["smiles"].map(keys)

    to_test_schema(val).to_parquet(out / "queries.parquet")
    labels.to_parquet(out / "labels.parquet")
    (out / "val_ik14.txt").write_text("\n".join(iks) + "\n")

    report.update(kept_spectra=len(val), kept_molecules=len(labels),
                  adduct_counts_kept=val["adduct"].value_counts().to_dict(),
                  spectra_per_molecule=labels["n_spectra"].describe().round(2).to_dict(),
                  ik14_metric_differs_from_raw=int((labels["metric_ik14"] != labels["inchikey14"]).sum()))
    (out / "build_report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
