"""Step 2b: build the training targets (one 6930-bit fingerprint per training structure).

Only structures that have at least one spectrum with one of the ten test adducts are kept,
so training data and test conditions match.

Run:  uv run src/scripts/build_fp_targets.py <path/to/train.parquet> [out.npz]
Output (default artifacts/fpnet/fp_targets.npz): ik14 (strings), smiles (strings), fp (n, 867 uint8).
"""
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from casmi.fpnet import fp_recipe

TEN = ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
       "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"]


def main():
    train_path = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "artifacts/fpnet/fp_targets.npz"
    df = pd.read_parquet(train_path, columns=["inchikey14", "normalized_smiles", "adduct"])
    print("train spectra:", f"{len(df):,}", "| unique structures:", f"{df.inchikey14.nunique():,}")

    per = df.groupby("adduct").agg(spectra=("inchikey14", "size"), molecules=("inchikey14", "nunique"))
    print("\nper-adduct counts for the ten test adducts:")
    print(per.reindex(TEN).fillna(0).astype(int).to_string())
    print("\nten most common adducts overall (to see what we are dropping):")
    print(per.sort_values("spectra", ascending=False).head(10).to_string())

    df = df[df.adduct.isin(TEN)].dropna(subset=["inchikey14", "normalized_smiles"])
    print("\nkept spectra:", f"{len(df):,}", "| kept structures:", f"{df.inchikey14.nunique():,}")
    mols = df.drop_duplicates("inchikey14")[["inchikey14", "normalized_smiles"]].reset_index(drop=True)

    t0 = time.time()
    with Pool(max(1, os.cpu_count() - 1), initializer=fp_recipe.init) as p:
        res = p.map(fp_recipe.smiles_to_packed, mols.normalized_smiles.tolist(), chunksize=1000)
    ok = np.array([r is not None for r in res])
    print(f"fingerprints done in {time.time() - t0:.0f}s | parse failures: {(~ok).sum():,}")

    fp = np.stack([r for r in res if r is not None])
    mols = mols[ok].reset_index(drop=True)
    dens = fp_recipe.unpack(fp[:20000]).mean()
    print(f"fp array {fp.shape} | mean bit density (first 20k) {dens:.4f}")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    np.savez_compressed(out, ik14=mols.inchikey14.to_numpy().astype(str),
                        smiles=mols.normalized_smiles.to_numpy().astype(str), fp=fp)
    print("saved", out)


if __name__ == "__main__":
    main()