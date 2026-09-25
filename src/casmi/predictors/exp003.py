"""exp_003 — zero-cap entropy search + analog (mass-shift) channel.

Replaces exp_002's per-structure spectrum cap entirely, instead of tuning it further:
the cap only existed because pure-Python cosine was too slow to score every spectrum in
the mass window. search_kernels.search_window (numba, parallel) makes that unnecessary —
every reference spectrum in the ppm window is scored directly (external reference: a
community notebook using this approach, with no candidate truncation, scored 0.320 public
LB vs our 0.127; see docs/research/literature.md).

Two evidence channels:
  A. Library match  — direct entropy similarity, neutral-mass windowed (adduct-agnostic,
     from exp_002). This is C1's channel.
  B. Analog match    — mass-shifted entropy similarity against one representative spectrum
     per unique structure, over a wide window. This gives POOL candidates a real ranking
     signal (structural similarity to what was actually measured) instead of exp_000/002's
     mass-error sort, which is what was leaving 137/250 val molecules below rank 25 in C2.

Both channels reuse chem.py's neutral_mass/formula_mass (single source of truth for adducts).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from ..chem import IK14Cache, formula_mass, mass_tol, neutral_mass
from ..search_kernels import search_window, search_window_shift
from .base import Candidate, Predictor

BUCKET = 0.02


def _spectrum(mz, it):
    mz = np.asarray([] if mz is None else mz, np.float32)
    it = np.asarray([] if it is None else it, np.float32)
    n = min(mz.size, it.size)
    mz, it = mz[:n], it[:n]
    ok = np.isfinite(mz) & np.isfinite(it)
    return mz[ok], it[ok]


class Exp003Predictor(Predictor):
    name = "exp003"

    def __init__(self, ppm=10.0, min_da=0.005, ppm_offset=0.0, fragment_tol=0.01,
                 int_floor=0.002, max_peaks=256, ent_weight=True,
                 analog_win_da=200.0, n_analog=100, sim_power=4.0,
                 strong_score=0.20, max_library_guesses=12, max_pool_ranked=25,
                 cache_dir="cache", **kw):
        super().__init__(**kw)
        self.ppm, self.min_da, self.ppm_offset = ppm, min_da, ppm_offset
        self.fragment_tol, self.int_floor, self.max_peaks, self.ent_weight = (
            fragment_tol, int_floor, max_peaks, ent_weight)
        self.analog_win_da, self.n_analog, self.sim_power = analog_win_da, n_analog, sim_power
        self.strong_score, self.max_library_guesses, self.max_pool_ranked = (
            strong_score, max_library_guesses, max_pool_ranked)
        self.cache = IK14Cache(f"{cache_dir}/metric_ik14.parquet")

    # ── index ──────────────────────────────────────────────────────────────
    def _query_mass(self, mz, adduct) -> float | None:
        m = neutral_mass(mz, adduct)
        return None if m is None else m * (1.0 - self.ppm_offset * 1e-6)

    def prepare(self, train: pl.LazyFrame, pool: pd.DataFrame, queries: pd.DataFrame) -> None:
        masses = [m for mz, ad in zip(queries["precursor_mz"], queries["adduct"])
                  if (m := self._query_mass(mz, ad)) is not None]
        masses = np.asarray(masses, np.float64)
        tol = np.maximum(self.min_da, masses * self.ppm * 1e-6)
        centers, radii = np.floor(masses / BUCKET).astype(np.int64), np.ceil(tol / BUCKET).astype(np.int64)
        buckets = np.unique(np.concatenate([np.arange(c - r, c + r + 1) for c, r in zip(centers, radii)])) \
            if masses.size else np.empty(0, np.int64)
        target = pl.DataFrame({"_b": buckets.astype(np.int64)}).lazy()

        forms = train.select("molecular_formula").unique().collect()
        fmass = {f: formula_mass(f) for f in forms["molecular_formula"].to_list()}
        fmap = pl.DataFrame({"molecular_formula": list(fmass), "neutral_mass": list(fmass.values())}).lazy()

        lf = (train.join(fmap, on="molecular_formula", how="inner")
              .with_columns((pl.col("neutral_mass") / BUCKET).floor().cast(pl.Int64).alias("_b"),
                            pl.col("inchikey14").cast(pl.String).str.strip_chars().str.to_uppercase().alias("key"))
              .filter(pl.col("neutral_mass").is_finite() & (pl.col("neutral_mass") > 0)
                      & pl.col("key").str.len_chars().gt(0)
                      & pl.col("normalized_smiles").cast(pl.String).str.len_chars().gt(0)))

        # Channel A reference: every spectrum in the query windows (no per-structure cap).
        ref = (lf.join(target, on="_b", how="semi")
               .select(["key", "adduct", "neutral_mass", "normalized_smiles",
                        "ms2_mzs", "ms2_normalized_intensities"])
               .sort("neutral_mass").collect().to_pandas())
        self._ref_mass = ref["neutral_mass"].to_numpy(np.float64)
        self._ref_key = ref["key"].to_numpy(dtype=str)
        self._ref_adduct = ref["adduct"].to_numpy(dtype=str)
        self._ref_smiles = ref["normalized_smiles"].to_numpy(dtype=str)
        self._pack_spectra(ref["ms2_mzs"], ref["ms2_normalized_intensities"], "_ref")

        # Channel B reference: one representative (richest) spectrum per unique structure,
        # over a much wider window, used for mass-shifted analog search.
        rep = (lf.with_columns(pl.col("ms2_mzs").list.len().alias("_np"))
               .sort(["key", "_np"], descending=[False, True])
               .group_by("key", maintain_order=True).head(1)
               .select(["key", "neutral_mass", "normalized_smiles", "ms2_mzs", "ms2_normalized_intensities"])
               .sort("neutral_mass").collect().to_pandas())
        self._rep_mass = rep["neutral_mass"].to_numpy(np.float64)
        self._rep_key = rep["key"].to_numpy(dtype=str)
        self._rep_smiles = rep["normalized_smiles"].to_numpy(dtype=str)
        self._pack_spectra(rep["ms2_mzs"], rep["ms2_normalized_intensities"], "_rep")

        p = pool.copy()
        p["exact_mass"] = pd.to_numeric(p["exact_mass"], errors="coerce")
        p = p.dropna(subset=["metric_ik14", "smiles", "exact_mass"])
        p = p.loc[(p["exact_mass"] > 0) & p["smiles"].astype(str).str.len().gt(0)]
        p = p.drop_duplicates("metric_ik14").sort_values("exact_mass").reset_index(drop=True)
        self._pool_mass = p["exact_mass"].to_numpy(np.float64)
        self._pool_key = p["metric_ik14"].astype(str).to_numpy(dtype=str)
        self._pool_smiles = p["smiles"].astype(str).to_numpy(dtype=str)

    def _pack_spectra(self, mzs, ints, prefix: str) -> None:
        """CSR-style flat arrays: offsets[i]:offsets[i+1] slices out spectrum i. Numba wants flat float32 arrays."""
        offsets = np.zeros(len(mzs) + 1, np.int64)
        for i, m in enumerate(mzs):
            offsets[i + 1] = offsets[i] + (0 if m is None else len(m))
        allmz, allin = np.empty(offsets[-1], np.float32), np.empty(offsets[-1], np.float32)
        for i, (m, v) in enumerate(zip(mzs, ints)):
            if m is None:
                continue
            a, b = offsets[i], offsets[i + 1]
            allmz[a:b] = np.asarray(m, np.float32)
            allin[a:b] = np.asarray(v, np.float32)
        setattr(self, f"{prefix}_off", offsets)
        setattr(self, f"{prefix}_mz", allmz)
        setattr(self, f"{prefix}_in", allin)

    # ── search ─────────────────────────────────────────────────────────────
    def _window(self, arr: np.ndarray, mass: float, ppm: float | None = None) -> tuple[int, int]:
        t = mass_tol(mass, self.ppm if ppm is None else ppm, self.min_da)
        return int(np.searchsorted(arr, mass - t, "left")), int(np.searchsorted(arr, mass + t, "right"))

    def _library(self, rows: pd.DataFrame) -> dict[str, tuple[float, str]]:
        best: dict[str, tuple[float, str]] = {}
        for r in rows.itertuples(index=False):
            m = self._query_mass(r.precursor_mz, r.adduct)
            if m is None:
                continue
            qmz, qit = _spectrum(r.ms2_mzs, r.ms2_normalized_intensities)
            if qmz.size == 0:
                continue
            lo, hi = self._window(self._ref_mass, m)
            if hi <= lo:
                continue
            idx = np.arange(lo, hi, dtype=np.int64)
            scores = search_window(qmz, qit, idx, self._ref_off, self._ref_mz, self._ref_in,
                                    self.fragment_tol, self.int_floor, self.max_peaks, self.ent_weight)
            for i, s in zip(idx, scores):
                k = str(self._ref_key[i])
                if s > best.get(k, (-1.0, ""))[0]:
                    best[k] = (float(s), str(self._ref_smiles[i]))
        return best

    def _analog(self, rows: pd.DataFrame) -> dict[str, float]:
        """Best mass-shifted similarity to ANY unique structure's representative spectrum,
        over a wide window — the signal used to rank pool candidates."""
        best: dict[str, float] = {}
        for r in rows.itertuples(index=False):
            m = self._query_mass(r.precursor_mz, r.adduct)
            if m is None:
                continue
            qmz, qit = _spectrum(r.ms2_mzs, r.ms2_normalized_intensities)
            if qmz.size == 0:
                continue
            lo, hi = (int(np.searchsorted(self._rep_mass, m - self.analog_win_da, "left")),
                      int(np.searchsorted(self._rep_mass, m + self.analog_win_da, "right")))
            if hi <= lo:
                continue
            idx = np.arange(lo, hi, dtype=np.int64)
            shift = (m - self._rep_mass[idx]).astype(np.float32)
            scores = search_window_shift(qmz, qit, idx, self._rep_off, self._rep_mz, self._rep_in,
                                         self.fragment_tol, self.int_floor, self.max_peaks,
                                         self.ent_weight, shift)
            for i, s in zip(idx, scores):
                k = str(self._rep_key[i])
                if s > best.get(k, -1.0):
                    best[k] = float(s)
        return best

    def _pool(self, rows: pd.DataFrame, analog: dict[str, float]) -> list[tuple[str, str, float]]:
        cand: dict[str, tuple[str, float]] = {}
        for r in rows.itertuples(index=False):
            m = self._query_mass(r.precursor_mz, r.adduct)
            if m is None:
                continue
            lo, hi = self._window(self._pool_mass, m)
            for i in range(lo, hi):
                k = str(self._pool_key[i])
                if k not in cand:
                    cand[k] = (str(self._pool_smiles[i]), analog.get(k, 0.0))
        return sorted(((k, s, sc) for k, (s, sc) in cand.items()), key=lambda x: -x[2])

    def predict(self, rows: pd.DataFrame) -> list[Candidate]:
        lib = sorted(self._library(rows).items(), key=lambda kv: -kv[1][0])
        analog = self._analog(rows)
        pool = self._pool(rows, analog)

        ordered: list[Candidate] = []
        strong = [Candidate(s, k, "library", sc) for k, (sc, s) in lib if sc >= self.strong_score]
        ordered += strong[: self.max_library_guesses]
        lib_keys = {c.key for c in strong}
        ordered += [Candidate(s, k, "pool", sc) for k, s, sc in pool if k not in lib_keys]
        ordered += [Candidate(s, k, "library", sc) for k, (sc, s) in lib if sc < self.strong_score]

        head = ordered[:60]
        keys = self.cache.get_many([c.smiles for c in head])
        self.cache.save()
        out, seen = [], set()
        for c in head:
            key = keys.get(c.smiles) or c.smiles
            if key not in seen:
                seen.add(key)
                out.append(c)
        return out + ordered[60:]