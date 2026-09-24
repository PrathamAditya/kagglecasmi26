"""Mass arithmetic and the metric's structure key. Single source of truth — never redefine offsets elsewhere."""
from __future__ import annotations

import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

# Monoisotopic masses
H, C, N, O = 1.00782503207, 12.0, 14.0030740048, 15.99491461956
NA, K, CL, E = 22.9897692809, 38.96370668, 34.96885268, 0.000548579909
H2O, CH2O2 = 2 * H + O, C + 2 * H + 2 * O
PROTON = H - E

# Neutral monoisotopic mass M = precursor_mz + offset (all test adducts are singly charged).
NEUTRAL_OFFSET: dict[str, float] = {
    "[M+H]+": -PROTON,
    "[M+NH4]+": -(N + 4 * H - E),
    "[M-H2O+H]+": H2O - PROTON,
    "[M-2H2O+H]+": 2 * H2O - PROTON,
    "[M+Na]+": -(NA - E),
    "[M+K]+": -(K - E),
    "[M-H]-": PROTON,
    "[M-H2O-H]-": H2O + PROTON,
    "[M+CH2O2-H]-": -(CH2O2 - H + E),
    "[M+Cl]-": -(CL + E),
}
TEST_ADDUCTS: tuple[str, ...] = tuple(NEUTRAL_OFFSET)


def neutral_mass(precursor_mz: float, adduct: str | None) -> float | None:
    offset = NEUTRAL_OFFSET.get(adduct or "")
    if offset is None or precursor_mz is None or not math.isfinite(precursor_mz) or precursor_mz <= 0:
        return None
    return float(precursor_mz) + offset


def mass_tol(mass: float, ppm: float = 20.0, min_da: float = 0.01) -> float:
    return max(min_da, abs(mass) * ppm * 1e-6)


# ── Metric key ──────────────────────────────────────────────────────────────
# Competition: RDKit tautomer canonicalization → InChIKey first block.
# TODO(exp_001): diff this against the official metric notebook and copy its exact function if it differs.
_ENUMERATOR = None


def _init_rdkit() -> None:
    global _ENUMERATOR
    from rdkit import RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    _ENUMERATOR = rdMolStandardize.TautomerEnumerator()


def metric_ik14(smiles: str) -> str | None:
    from rdkit import Chem

    if _ENUMERATOR is None:
        _init_rdkit()
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return None
    try:
        mol = _ENUMERATOR.Canonicalize(mol)
    except Exception:
        pass
    key = Chem.MolToInchiKey(mol)
    return key[:14] if key else None


class IK14Cache:
    """Disk-backed smiles → metric InChIKey14 map (tautomer canonicalization is slow)."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._map: dict[str, str | None] = {}
        self._dirty = False
        if self.path.is_file():
            df = pd.read_parquet(self.path)
            self._map = {s: (k or None) for s, k in zip(df["smiles"], df["ik14"])}

    def get_many(self, smiles: list[str], workers: int | None = None) -> dict[str, str | None]:
        misses = sorted({s for s in smiles if s not in self._map})
        if misses:
            if len(misses) < 500 or workers == 1:
                values = [metric_ik14(s) for s in misses]
            else:
                with ProcessPoolExecutor(workers, initializer=_init_rdkit) as ex:
                    values = list(ex.map(metric_ik14, misses, chunksize=64))
            self._map.update(zip(misses, values))
            self._dirty = True
        return {s: self._map.get(s) for s in smiles}

    def save(self) -> None:
        if not self._dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"smiles": list(self._map), "ik14": [v or "" for v in self._map.values()]}).to_parquet(self.path)
        self._dirty = False
