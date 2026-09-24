"""Tiny fake train/pool with the real schemas — smoke-tests the harness end to end (not a benchmark)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt

from casmi.chem import NEUTRAL_OFFSET

SMILES = ["OCC1OC(O)C(O)C(O)C1O", "CN1C=NC2=C1C(=O)N(C)C(=O)N2C", "O=C(O)c1ccccc1O", "NCCc1ccc(O)cc1",
          "CC(=O)Nc1ccc(O)cc1", "O=c1cc(-c2ccccc2)oc2cc(O)cc(O)c12", "OC(=O)C=Cc1ccc(O)c(O)c1",
          "COc1cc(C=CC(=O)O)ccc1O", "O=C1C=CC(=O)C=C1", "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
          "OC1C(O)C(O)C(OC2=CC=CC=C2)OC1CO", "O=c1c(O)c(-c2ccc(O)c(O)c2)oc2cc(O)cc(O)c12",
          "c1ccc2[nH]ccc2c1", "OC(=O)CCC(=O)O", "CC(O)C(=O)O", "Oc1ccc(C=Cc2cc(O)cc(O)c2)cc1"]
rng = np.random.default_rng(0)


def spectrum(seed_mass, n=12):
    mz = np.sort(rng.uniform(40, seed_mass, n)); it = rng.uniform(0.01, 1, n); it /= it.max()
    return mz, it


def main(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, smi in enumerate(SMILES):
        mol = Chem.MolFromSmiles(smi); M = ExactMolWt(mol); ik = Chem.MolToInchiKey(mol)
        base_mz, base_it = spectrum(M)
        libs = ["gnps", "riken"] + (["enveda-np-examples"] if i < 10 else [])
        for lib in libs:
            for adduct in ["[M+H]+", "[M-H]-", "[M+Na]+"][: 1 + i % 3]:
                mz = M - NEUTRAL_OFFSET[adduct]
                rows.append(dict(spectrum_id=f"{lib}_{i}_{adduct}", ms2_mzs=base_mz + rng.normal(0, 0.002, base_mz.size),
                                 ms2_normalized_intensities=base_it, base_peak_intensity=5000.0, adduct=adduct,
                                 ionization_mode="positive" if adduct.endswith("+") else "negative",
                                 instrument_type="timsTOF", precursor_mz=mz, collision_energy_ev=[20.0],
                                 collision_energy_orig="20", collision_energy_orig_units="eV",
                                 normalized_smiles=smi, inchikey=ik, inchikey14=ik[:14], molecular_formula="",
                                 ingest_lib=lib, adduct_orig=adduct, precursor_error_ppm=1.0, num_peaks=base_mz.size))
    pd.DataFrame(rows).to_parquet(out / "train.parquet")
    pool = pd.DataFrame([dict(metric_ik14=Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14], smiles=s,
                              exact_mass=ExactMolWt(Chem.MolFromSmiles(s)), in_coconut=True, in_train=True)
                         for s in SMILES])
    pool.to_parquet(out / "pool.parquet")
    print(f"train {len(rows)} spectra, pool {len(pool)}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
