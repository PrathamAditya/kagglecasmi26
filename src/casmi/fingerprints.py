"""Morgan fingerprints + Tanimoto, cached by SMILES.

Needed for analog PROPAGATION: exp_003 could only score a pool candidate if that exact
structure appeared in the train-derived analog set, so every genuinely novel class-2
structure scored 0.0. V27 instead compares candidate fingerprints against the fingerprints
of its analog NEIGHBOURS, so structural similarity transfers to structures never measured.
"""
from __future__ import annotations

import numpy as np


class FingerprintCache:
    def __init__(self, n_bits: int = 2048, radius: int = 2):
        self.n_bits, self.radius = n_bits, radius
        self._cache: dict[str, np.ndarray | None] = {}
        self._gen = None

    def _generator(self):
        if self._gen is None:
            from rdkit import RDLogger
            from rdkit.Chem import rdFingerprintGenerator

            RDLogger.DisableLog("rdApp.*")
            self._gen = rdFingerprintGenerator.GetMorganGenerator(
                radius=self.radius, fpSize=self.n_bits)
        return self._gen

    def get(self, smiles: str) -> np.ndarray | None:
        if smiles in self._cache:
            return self._cache[smiles]
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles) if smiles else None
        fp = None
        if mol is not None:
            bv = self._generator().GetFingerprintAsNumPy(mol)
            fp = np.asarray(bv, dtype=np.float32)
        self._cache[smiles] = fp
        return fp

    def matrix(self, smiles_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Returns (matrix of valid fingerprints, index array of which inputs were valid)."""
        rows, keep = [], []
        for i, s in enumerate(smiles_list):
            fp = self.get(s)
            if fp is not None:
                rows.append(fp)
                keep.append(i)
        if not rows:
            return np.zeros((0, self.n_bits), np.float32), np.zeros(0, np.int64)
        return np.vstack(rows), np.asarray(keep, np.int64)


def tanimoto_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Pairwise Tanimoto between rows of a and rows of b (binary fingerprints)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((a.shape[0], b.shape[0]), np.float32)
    inter = a @ b.T
    asum, bsum = a.sum(1), b.sum(1)
    return inter / (asum[:, None] + bsum[None, :] - inter + 1e-9)