"""exp_002 — candidate-generation correctness.

Three changes vs exp_000, all about *recall*, none about the similarity function:
1. Library is indexed by the reference's NEUTRAL mass, computed from its labelled
   molecular_formula. Matching is therefore adduct-agnostic: a [M+Na]+ query finds
   [M+H]+ references. exp_000 matched raw precursor m/z, so only same-adduct hits existed.
2. All 10 test adducts get a correct neutral mass (chem.NEUTRAL_OFFSET); tolerance is
   configurable and defaults to 10 ppm (val p99 = 6.6).
3. The candidate cap counts STRUCTURES, not spectra. exp_000's 256 nearest spectra were
   often a handful of structures measured many times; the same cosine budget now covers
   ~max_structures distinct structures, each represented by <= spectra_per_structure spectra.

Dedup and the final list use the metric InChIKey14 (tautomer-canonical), not raw keys.
Ranking of pool candidates is deliberately unchanged — that is exp_003's job.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from ..chem import NEUTRAL_OFFSET, IK14Cache, formula_mass, mass_tol, neutral_mass
from .base import Candidate, Predictor

BUCKET = 0.02


def greedy_cosine(q_mz, q_int, q_order, q_norm, c_mz, c_int, tol) -> float:
    """Unchanged from exp_000 so score comparisons stay meaningful."""
    c_norm = float(np.linalg.norm(c_int))
    if q_norm == 0.0 or c_norm == 0.0:
        return 0.0
    used = np.zeros(c_mz.size, bool)
    dot = 0.0
    for i in q_order:
        d = np.abs(c_mz - q_mz[i])
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            dot += q_int[i] * c_int[j]
    return float(dot / (q_norm * c_norm))


class Exp002Predictor(Predictor):
    name = "exp002"

    def __init__(self, ppm=10.0, min_da=0.005, fragment_tol=0.01, max_structures=400,
                 spectra_per_structure=3, cross_adduct_penalty=0.7, strong_score=0.20,
                 max_library_guesses=12, cache_dir="cache", **kw):
        super().__init__(**kw)
        self.ppm, self.min_da, self.fragment_tol = ppm, min_da, fragment_tol
        self.max_structures, self.spectra_per_structure = max_structures, spectra_per_structure
        self.cross_adduct_penalty, self.strong_score = cross_adduct_penalty, strong_score
        self.max_library_guesses = max_library_guesses
        self.cache = IK14Cache(f"{cache_dir}/metric_ik14.parquet")

    # ── index ──────────────────────────────────────────────────────────────
    def _query_masses(self, queries: pd.DataFrame) -> np.ndarray:
        out = []
        for mz, ad in zip(queries["precursor_mz"], queries["adduct"]):
            m = neutral_mass(mz, ad)
            if m is not None:
                out.append(m)
            else:                                  # unknown adduct: try every test adduct
                out.extend(float(mz) + o for o in NEUTRAL_OFFSET.values()
                           if mz and np.isfinite(mz) and float(mz) + o > 0)
        return np.asarray(out, float)

    def prepare(self, train: pl.LazyFrame, pool: pd.DataFrame, queries: pd.DataFrame) -> None:
        masses = self._query_masses(queries)
        tol = np.maximum(self.min_da, masses * self.ppm * 1e-6)
        centers, radii = np.floor(masses / BUCKET).astype(np.int64), np.ceil(tol / BUCKET).astype(np.int64)
        buckets = np.unique(np.concatenate([np.arange(c - r, c + r + 1) for c, r in zip(centers, radii)])) \
            if masses.size else np.empty(0, np.int64)
        target = pl.DataFrame({"_b": buckets.astype(np.int64)}).lazy()

        # Formula → neutral mass, on the unique formulas only (cheap), then bucket-join.
        forms = train.select("molecular_formula").unique().collect()
        fmass = {f: formula_mass(f) for f in forms["molecular_formula"].to_list()}
        fmap = pl.DataFrame({"molecular_formula": list(fmass), "neutral_mass": list(fmass.values())}).lazy()

        lf = (train.join(fmap, on="molecular_formula", how="inner")
              .with_columns((pl.col("neutral_mass") / BUCKET).floor().cast(pl.Int64).alias("_b"),
                            pl.col("inchikey14").cast(pl.String).str.strip_chars().str.to_uppercase().alias("key"))
              .filter(pl.col("neutral_mass").is_finite() & (pl.col("neutral_mass") > 0)
                      & pl.col("key").str.len_chars().gt(0)
                      & pl.col("normalized_smiles").cast(pl.String).str.len_chars().gt(0))
              .join(target, on="_b", how="semi"))

        # Keep at most `spectra_per_structure` spectra per (structure, adduct): the most
        # informative ones (peak count), which spreads collision energies in practice.
        ref = (lf.with_columns(pl.col("ms2_mzs").list.len().alias("_np"))
               .sort(["key", "adduct", "_np"], descending=[False, False, True])
               .group_by(["key", "adduct"], maintain_order=True).head(self.spectra_per_structure)
               .select(["key", "adduct", "neutral_mass", "normalized_smiles",
                        "ms2_mzs", "ms2_normalized_intensities", "_np"])
               .sort("neutral_mass").collect().to_pandas())

        self.ref_mass = ref["neutral_mass"].to_numpy(np.float64)
        self.ref_key = ref["key"].to_numpy(dtype=str)
        self.ref_adduct = ref["adduct"].to_numpy(dtype=str)
        self.ref_smiles = ref["normalized_smiles"].to_numpy(dtype=str)
        self.ref_mzs, self.ref_ints = ref["ms2_mzs"].to_numpy(), ref["ms2_normalized_intensities"].to_numpy()

        p = pool.copy()
        p["exact_mass"] = pd.to_numeric(p["exact_mass"], errors="coerce")
        p = p.dropna(subset=["metric_ik14", "smiles", "exact_mass"])
        p = p.loc[(p["exact_mass"] > 0) & p["smiles"].astype(str).str.len().gt(0)]
        p = p.drop_duplicates("metric_ik14").sort_values("exact_mass").reset_index(drop=True)
        self.pool_mass = p["exact_mass"].to_numpy(np.float64)
        self.pool_key = p["metric_ik14"].astype(str).to_numpy(dtype=str)
        self.pool_smiles = p["smiles"].astype(str).to_numpy(dtype=str)
        self.pool_train = p["in_train"].to_numpy(np.int8)
        self.pool_coco = p["in_coconut"].to_numpy(np.int8)

    # ── search ─────────────────────────────────────────────────────────────
    def _window(self, arr: np.ndarray, mass: float) -> tuple[int, int]:
        t = mass_tol(mass, self.ppm, self.min_da)
        return (int(np.searchsorted(arr, mass - t, "left")), int(np.searchsorted(arr, mass + t, "right")))

    def _library(self, rows: pd.DataFrame) -> dict[str, tuple[float, str]]:
        best: dict[str, tuple[float, str]] = {}
        for r in rows.itertuples(index=False):
            m = neutral_mass(r.precursor_mz, r.adduct)
            if m is None:
                continue
            q_mz = np.asarray([] if r.ms2_mzs is None else r.ms2_mzs, float)
            q_in = np.asarray([] if r.ms2_normalized_intensities is None else r.ms2_normalized_intensities, float)
            n = min(q_mz.size, q_in.size)
            q_mz, q_in = q_mz[:n], q_in[:n]
            ok = np.isfinite(q_mz) & np.isfinite(q_in)
            q_mz, q_in = q_mz[ok], q_in[ok]
            if q_mz.size == 0:
                continue
            q_order, q_norm = np.lexsort((q_mz, -q_in)), float(np.linalg.norm(q_in))

            lo, hi = self._window(self.ref_mass, m)
            idx = np.arange(lo, hi)
            if idx.size == 0:
                continue
            same = (self.ref_adduct[idx] == r.adduct)
            # Cap on STRUCTURES: same-adduct first, then nearest in mass.
            order = np.lexsort((np.abs(self.ref_mass[idx] - m), ~same))
            idx = idx[order]
            seen: dict[str, int] = {}
            keep = []
            for pos, k in enumerate(self.ref_key[idx]):
                c = seen.get(k, 0)
                if (c == 0 and len(seen) >= self.max_structures) or c >= self.spectra_per_structure:
                    continue
                seen[k] = c + 1
                keep.append(pos)
            for pos in keep:
                i = idx[pos]
                s = greedy_cosine(q_mz, q_in, q_order, q_norm,
                                  np.asarray(self.ref_mzs[i], float), np.asarray(self.ref_ints[i], float),
                                  self.fragment_tol)
                if self.ref_adduct[i] != r.adduct:
                    s *= self.cross_adduct_penalty
                k = str(self.ref_key[i])
                if s > best.get(k, (-1.0, ""))[0]:
                    best[k] = (s, str(self.ref_smiles[i]))
        return best

    def _pool(self, rows: pd.DataFrame) -> list[tuple[str, str, tuple]]:
        best: dict[str, tuple[str, str, tuple]] = {}
        for r in rows.itertuples(index=False):
            m = neutral_mass(r.precursor_mz, r.adduct)
            if m is None:
                continue
            lo, hi = self._window(self.pool_mass, m)
            for i in range(lo, hi):
                k = str(self.pool_key[i])
                rank = (-int(self.pool_train[i]), -int(self.pool_coco[i]), abs(self.pool_mass[i] - m))
                if k not in best or rank < best[k][2]:
                    best[k] = (k, str(self.pool_smiles[i]), rank)
        return sorted(best.values(), key=lambda x: x[2])

    def predict(self, rows: pd.DataFrame) -> list[Candidate]:
        lib = sorted(self._library(rows).items(), key=lambda kv: -kv[1][0])
        pool = self._pool(rows)

        ordered: list[Candidate] = []
        strong = [Candidate(s, k, "library", sc) for k, (sc, s) in lib if sc >= self.strong_score]
        ordered += strong[: self.max_library_guesses]
        lib_keys = {c.key for c in strong}
        ordered += [Candidate(s, k, "pool") for k, s, _ in pool if k not in lib_keys]
        ordered += [Candidate(s, k, "library", sc) for k, (sc, s) in lib if sc < self.strong_score]

        # Dedup on the METRIC key (tautomer-canonical) so duplicates don't burn slots.
        head = ordered[:60]
        keys = self.cache.get_many([c.smiles for c in head])
        self.cache.save()
        out, seen = [], set()
        for c in head:
            k = keys.get(c.smiles) or c.smiles
            if k not in seen:
                seen.add(k)
                out.append(c)
        return out + ordered[60:]