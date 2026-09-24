"""exp_000 ported to the Predictor interface. Reproduces the original ranking exactly (bugs included)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import polars as pl

from ..legacy import exp000_original as L
from .base import Candidate, Predictor


def assemble(train_hits, external_hits) -> list[str]:
    """Copy of the guess-assembly part of L._predict (verified by check_legacy)."""
    library_keys = {hit[0] for hit in train_hits if hit[2] > 0.0}
    guesses: list[str] = []
    used: set[str] = set()
    for _key, smi, score, _, _ in train_hits:
        if score < L.STRONG_SCORE or len(guesses) >= L.MAX_SPECTRAL_GUESSES:
            continue
        if smi not in used:
            guesses.append(smi); used.add(smi)
    for key, smi in external_hits:
        if len(guesses) >= L.MAX_GUESSES:
            break
        if key not in library_keys and smi not in used:
            guesses.append(smi); used.add(smi)
    for _key, smi, _, _, _ in train_hits:
        if len(guesses) >= L.MAX_GUESSES:
            break
        if smi not in used:
            guesses.append(smi); used.add(smi)
    return guesses or ["CCO"]


class Exp000Predictor(Predictor):
    name = "exp000"

    def prepare(self, train: pl.LazyFrame, pool: pd.DataFrame, queries: pd.DataFrame) -> None:
        self._checks_left = int(self.params.get("check_legacy", 0))
        prec = pd.to_numeric(queries["precursor_mz"], errors="coerce").to_numpy(np.float64)
        prec = prec[np.isfinite(prec) & (prec > 0)]
        centers = np.floor(prec / L.BUCKET_WIDTH).astype(np.int64)
        radii = np.ceil(np.maximum(0.01, prec * 20e-6) / L.BUCKET_WIDTH).astype(np.int64)
        buckets = np.unique(np.concatenate([np.arange(c - r, c + r + 1) for c, r in zip(centers, radii)])) \
            if centers.size else np.empty(0, np.int64)
        target = pl.DataFrame({"_bucket": buckets.astype(np.int64)}).lazy()

        frame = (
            train.with_columns(
                pl.col("precursor_mz").cast(pl.Float64, strict=False),
                (pl.col("precursor_mz").cast(pl.Float64, strict=False) / L.BUCKET_WIDTH).floor().cast(pl.Int64).alias("_bucket"),
                pl.col("inchikey14").cast(pl.String).str.strip_chars().str.to_uppercase().alias("_key"),
            )
            .filter(pl.col("precursor_mz").is_finite() & (pl.col("precursor_mz") > 0.0)
                    & pl.col("_key").str.len_chars().gt(0)
                    & pl.col("normalized_smiles").cast(pl.String).str.strip_chars().str.len_chars().gt(0))
            .join(target, on="_bucket", how="semi")
            .select(L.TRAIN_COLUMNS + ["_key", "train_row"])
            .sort(["precursor_mz", "_key", "normalized_smiles", "train_row"])
            .collect()
            .to_pandas()
        )

        p = pool.copy()
        p["exact_mass"] = pd.to_numeric(p["exact_mass"], errors="coerce")
        p = p.dropna(subset=["metric_ik14", "smiles", "exact_mass"])
        p["_key"] = p["metric_ik14"].astype(str).str.strip().str.upper()
        p["_in_train"] = p["in_train"].fillna(False).astype(np.int8)
        p["_in_coconut"] = p["in_coconut"].fillna(False).astype(np.int8)
        p = p.loc[np.isfinite(p["exact_mass"].to_numpy()) & (p["exact_mass"].to_numpy() > 0.0)
                  & p["_key"].str.len().gt(0) & p["smiles"].astype(str).str.strip().str.len().gt(0)]
        p["_row"] = np.arange(len(p), dtype=np.int64)
        p = p.sort_values(["exact_mass", "_key", "smiles", "_in_train", "_in_coconut", "_row"],
                          ascending=[True, True, True, False, False, True], kind="mergesort")
        p = p.drop_duplicates("_key").sort_values("exact_mass", kind="mergesort").reset_index(drop=True)

        self.index = L.RetrievalIndex(
            frame["precursor_mz"].to_numpy(np.float64), frame["_key"].to_numpy(dtype=str),
            frame["normalized_smiles"].to_numpy(dtype=str), frame["train_row"].to_numpy(np.int64),
            frame["ms2_mzs"].to_numpy(), frame["ms2_normalized_intensities"].to_numpy(),
            p["exact_mass"].to_numpy(np.float64), p["_key"].to_numpy(dtype=str), p["smiles"].to_numpy(dtype=str),
            p["_row"].to_numpy(np.int64), p["_in_train"].to_numpy(np.int8), p["_in_coconut"].to_numpy(np.int8),
        )

    def predict(self, rows: pd.DataFrame) -> list[Candidate]:
        train_hits = L._rank_train(rows, self.index)
        external_hits = L._rank_external(rows, self.index)
        guesses = assemble(train_hits, external_hits)
        if self._checks_left > 0:
            self._checks_left -= 1
            legacy = L._predict(rows, self.index)[0]
            assert legacy == guesses, "exp000 port diverged from the original script"

        origin: dict[str, tuple[str, str, float]] = {}
        for key, smi, score, _, _ in train_hits:
            origin.setdefault(smi, (key, "library", score))
        for key, smi in external_hits:
            origin.setdefault(smi, (key, "pool", float("nan")))
        out = [Candidate(s, *origin.get(s, ("", "fallback", float("nan")))) for s in guesses]
        seen = set(guesses)
        for smi, (key, src, score) in origin.items():   # tail: everything generated, for Recall@cand
            if smi not in seen:
                out.append(Candidate(smi, key, src, score)); seen.add(smi)
        return out
