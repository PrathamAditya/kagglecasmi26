#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "numpy",
#     "pandas",
#     "polars",
# ]
# ///
# â”€â”€â”€ How to run â”€â”€â”€
# 1. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
# 2. Run locally: uv run retrieval_spectral_fast.py
# 3. In Kaggle, run this script after mounting the competition and pool files.
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
import pandas as pd
import polars as pl

ROOT: Final = Path("/kaggle/input"); WORK: Final = Path("/kaggle/working")
MAX_GUESSES: Final = 25; MAX_CANDIDATES: Final = 256
MAX_SPECTRAL_GUESSES: Final = 12; STRONG_SCORE: Final = 0.20
FRAGMENT_TOLERANCE: Final = 0.01; BUCKET_WIDTH: Final = 0.02
PROTON: Final = 1.007276466621
ADDUCT_OFFSETS: Final = {
    "[m+h]+": -PROTON, "[m-h]-": PROTON, "[m+na]+": -22.989218,
    "[m+nh4]+": -18.033823, "[m+k]+": -38.963158, "[m+cl]-": -34.968853,
    "[m+hcoo]-": -44.998201, "[m+ch3coo]-": -59.013305, "[m+no3]-": -62.004804,
    "[m+2h]2+": -2.0 * PROTON, "[m-2h]2-": 2.0 * PROTON,
}
DOUBLE_ADDUCTS: Final = {"[m+2h]2+": 2.0, "[m-2h]2-": 2.0}
TRAIN_COLUMNS: Final = ["precursor_mz", "ms2_mzs", "ms2_normalized_intensities", "normalized_smiles", "inchikey14"]
POOL_COLUMNS: Final = ["metric_ik14", "smiles", "exact_mass", "in_coconut", "in_train"]
Spectrum = tuple[np.ndarray, np.ndarray]; Query = tuple[np.ndarray, np.ndarray, np.ndarray, float]
TrainHit = tuple[str, str, float, float, int]


class RetrievalIndex(NamedTuple):
    train_masses: np.ndarray; train_keys: np.ndarray; train_smiles: np.ndarray; train_rows: np.ndarray; train_mzs: np.ndarray
    train_intensities: np.ndarray; pool_masses: np.ndarray; pool_keys: np.ndarray; pool_smiles: np.ndarray
    pool_rows: np.ndarray; pool_in_train: np.ndarray; pool_in_coconut: np.ndarray


def _tolerance(mass: float) -> float:
    return max(0.01, abs(mass) * 20.0 / 1_000_000.0)


def _spectrum(mzs, intensities) -> Spectrum:
    peaks = np.asarray([] if mzs is None else mzs, dtype=np.float64)
    values = np.asarray([] if intensities is None else intensities, dtype=np.float64)
    if peaks.ndim != 1 or values.ndim != 1:
        return peaks.reshape(0), values.reshape(0)
    size = min(peaks.size, values.size)
    valid = np.isfinite(peaks[:size]) & np.isfinite(values[:size])
    return peaks[:size][valid], values[:size][valid]


def fragment_cosine(query: Query, candidate: Spectrum) -> float:
    query_mzs, query_values, query_order, query_norm = query
    candidate_mzs, candidate_values = candidate
    candidate_norm = float(np.linalg.norm(candidate_values))
    if query_norm == 0.0 or candidate_norm == 0.0:
        return 0.0
    used = np.zeros(candidate_mzs.size, dtype=bool)
    dot = 0.0
    for query_index in query_order:
        distances = np.abs(candidate_mzs - query_mzs[query_index])
        distances[used] = np.inf
        candidate_index = int(np.argmin(distances))
        if distances[candidate_index] <= FRAGMENT_TOLERANCE:
            used[candidate_index] = True
            dot += query_values[query_index] * candidate_values[candidate_index]
    return float(dot / (query_norm * candidate_norm))


def _neutral_masses(row) -> tuple[tuple[float, int], ...]:
    precursor = float(row.precursor_mz)
    if not np.isfinite(precursor) or precursor <= 0.0:
        return ()
    adduct = "" if pd.isna(row.adduct) else str(row.adduct).strip().casefold()
    if adduct in DOUBLE_ADDUCTS:
        return ((precursor * DOUBLE_ADDUCTS[adduct] + ADDUCT_OFFSETS[adduct], 0),)
    if adduct in ADDUCT_OFFSETS:
        return ((precursor + ADDUCT_OFFSETS[adduct], 0),)
    primary_rank = 0 if not adduct else 1
    return ((precursor, primary_rank), (precursor - PROTON, primary_rank + 1))


def _rank_train(rows, index: RetrievalIndex, limit: int = MAX_CANDIDATES) -> list[TrainHit]:
    masses, keys, smiles = index.train_masses, index.train_keys, index.train_smiles
    source_rows = index.train_rows
    train_mzs, train_intensities = index.train_mzs, index.train_intensities
    best: dict[str, TrainHit] = {}
    for row in rows.itertuples(index=False):
        precursor = float(row.precursor_mz)
        if not np.isfinite(precursor) or precursor <= 0.0:
            continue
        query_mzs, query_values = _spectrum(row.ms2_mzs, row.ms2_normalized_intensities)
        query = (query_mzs, query_values, np.lexsort((query_mzs, -query_values)),
                 float(np.linalg.norm(query_values)))
        if query_mzs.size == 0:
            continue
        tolerance = _tolerance(precursor)
        left = int(np.searchsorted(masses, precursor - tolerance, side="left"))
        right = int(np.searchsorted(masses, precursor + tolerance, side="right"))
        indices = np.arange(left, right)
        # ponytail: 256 rows/query is the CPU ceiling; widen only if coverage misses them.
        if indices.size > limit:
            distances = np.abs(masses[indices] - precursor)
            order = np.lexsort((source_rows[indices], keys[indices], distances))
            indices = indices[order[:limit]]
        for index in indices:
            candidate = _spectrum(train_mzs[index], train_intensities[index])
            score = fragment_cosine(query, candidate)
            key = str(keys[index])
            error = float(abs(masses[index] - precursor))
            candidate_smiles = str(smiles[index])
            source = int(source_rows[index])
            rank = (-score, error, candidate_smiles, source)
            previous = best.get(key)
            if previous is None or rank < (-previous[2], previous[3], previous[1], previous[4]):
                best[key] = (key, candidate_smiles, score, error, source)
    ordered = sorted(best.values(), key=lambda hit: (-hit[2], hit[3], hit[0], hit[1], hit[4]))
    return ordered[:limit]


def _rank_external(rows, index: RetrievalIndex, limit: int = MAX_CANDIDATES) -> list[tuple[str, str]]:
    masses, keys, smiles = index.pool_masses, index.pool_keys, index.pool_smiles
    in_train, in_coconut = index.pool_in_train, index.pool_in_coconut
    source_rows = index.pool_rows
    best: dict[str, tuple[str, str, tuple]] = {}
    for row in rows.itertuples(index=False):
        for neutral, shift_rank in _neutral_masses(row):
            tolerance = _tolerance(neutral)
            left = int(np.searchsorted(masses, neutral - tolerance, side="left"))
            right = int(np.searchsorted(masses, neutral + tolerance, side="right"))
            indices = np.arange(left, right)
            if indices.size > limit:
                errors = np.abs(masses[indices] - neutral)
                shift_ranks = np.full(indices.size, shift_rank, dtype=np.int8)
                order = np.lexsort((smiles[indices], keys[indices], errors, -in_coconut[indices], -in_train[indices], shift_ranks))
                indices = indices[order[:limit]]
            for index in indices:
                key = str(keys[index])
                candidate_smiles = str(smiles[index])
                rank = (-int(in_train[index]), -int(in_coconut[index]), shift_rank,
                        float(abs(masses[index] - neutral)), candidate_smiles, int(source_rows[index]))
                previous = best.get(key)
                if previous is None or rank < previous[2]:
                    best[key] = (key, candidate_smiles, rank)
    ordered = sorted(best.values(), key=lambda hit: hit[2])
    return [(key, candidate_smiles) for key, candidate_smiles, _ in ordered[:limit]]


def _predict(rows, index: RetrievalIndex) -> tuple[list[str], int, int, int, bool]:
    train_hits = _rank_train(rows, index)
    external_hits = _rank_external(rows, index)
    library_keys = {hit[0] for hit in train_hits if hit[2] > 0.0}
    guesses: list[str] = []
    used_smiles: set[str] = set()
    strong_count = 0
    for key, candidate_smiles, score, _, _ in train_hits:
        if score < STRONG_SCORE or len(guesses) >= MAX_SPECTRAL_GUESSES:
            continue
        if candidate_smiles not in used_smiles:
            guesses.append(candidate_smiles)
            used_smiles.add(candidate_smiles)
            strong_count += 1
    external_count = 0
    for key, candidate_smiles in external_hits:
        if len(guesses) >= MAX_GUESSES:
            break
        if key not in library_keys and candidate_smiles not in used_smiles:
            guesses.append(candidate_smiles)
            used_smiles.add(candidate_smiles)
            external_count += 1
    for key, candidate_smiles, _, _, _ in train_hits:
        if len(guesses) >= MAX_GUESSES:
            break
        if candidate_smiles not in used_smiles:
            guesses.append(candidate_smiles)
            used_smiles.add(candidate_smiles)
    fallback = not guesses
    return (["CCO"] if fallback else guesses, len(train_hits), strong_count, external_count, fallback)


def main() -> None:
    train_paths = sorted(ROOT.rglob("train.parquet")); pool_paths = sorted(ROOT.rglob("pool.parquet"))
    assert train_paths and pool_paths, "train.parquet and pool.parquet must be mounted"
    train_path = train_paths[0]; pool_path = pool_paths[0]
    test_path = train_path.with_name("test.parquet"); sample_path = train_path.with_name("sample_submission.csv")
    assert test_path.is_file() and sample_path.is_file(), "test data are incomplete"
    sample = pd.read_csv(sample_path, dtype={"molecule_id": str}); sample_ids = sample["molecule_id"].astype(str).tolist()

    test_targets = pd.read_parquet(test_path, columns=["molecule_id", "precursor_mz"])
    precursor_values = pd.to_numeric(
        test_targets.loc[test_targets["molecule_id"].astype(str).isin(sample_ids), "precursor_mz"], errors="coerce"
    ).to_numpy(dtype=np.float64)
    valid_precursors = precursor_values[np.isfinite(precursor_values) & (precursor_values > 0.0)]
    centers = np.floor(valid_precursors / BUCKET_WIDTH).astype(np.int64)
    # radius=ceil(tolerance/0.02) includes every train row within max(0.01 Da, 20 ppm).
    radii = np.ceil(np.maximum(0.01, np.abs(valid_precursors) * 20.0 / 1_000_000.0) / BUCKET_WIDTH).astype(np.int64)
    target_buckets = np.unique(np.concatenate([np.arange(c-r, c+r+1, dtype=np.int64) for c, r in zip(centers, radii, strict=True)])) if centers.size else np.empty(0, dtype=np.int64)
    target_frame = pl.DataFrame({"_bucket": target_buckets}, schema={"_bucket": pl.Int64}).lazy()

    train_frame = (
        pl.scan_parquet(train_path, row_index_name="_source_row")
        .with_columns(
            pl.col("precursor_mz").cast(pl.Float64, strict=False).alias("precursor_mz"),
            (pl.col("precursor_mz").cast(pl.Float64, strict=False) / BUCKET_WIDTH).floor().cast(pl.Int64).alias("_bucket"),
            pl.col("inchikey14").cast(pl.String).str.strip_chars().str.to_uppercase().alias("_key"),
        )
        .filter(pl.col("precursor_mz").is_finite() & (pl.col("precursor_mz") > 0.0)
                & pl.col("_key").str.len_chars().gt(0)
                & pl.col("normalized_smiles").cast(pl.String).str.strip_chars().str.len_chars().gt(0))
        .join(target_frame, on="_bucket", how="semi")
        .select(TRAIN_COLUMNS + ["_key", "_source_row"])
        .sort(["precursor_mz", "_key", "normalized_smiles", "_source_row"])
        .collect()
    )
    selected_train_rows = int(train_frame.height)
    train = train_frame.rename({"_source_row": "_row"}).to_pandas()

    pool = pd.read_parquet(pool_path, columns=POOL_COLUMNS)
    pool["exact_mass"] = pd.to_numeric(pool["exact_mass"], errors="coerce")
    pool = pool.dropna(subset=["metric_ik14", "smiles", "exact_mass"])
    pool["_key"] = pool["metric_ik14"].astype(str).str.strip().str.upper()
    pool["_in_train"] = pool["in_train"].fillna(False).astype(np.int8)
    pool["_in_coconut"] = pool["in_coconut"].fillna(False).astype(np.int8)
    pool = pool.loc[np.isfinite(pool["exact_mass"].to_numpy()) & (pool["exact_mass"].to_numpy() > 0.0)
                    & pool["_key"].str.len().gt(0) & pool["smiles"].astype(str).str.strip().str.len().gt(0)]
    pool["_row"] = np.arange(len(pool), dtype=np.int64)
    pool = pool.sort_values(["exact_mass", "_key", "smiles", "_in_train", "_in_coconut", "_row"],
                            ascending=[True, True, True, False, False, True], kind="mergesort")
    pool = pool.drop_duplicates("_key").sort_values("exact_mass", kind="mergesort").reset_index(drop=True)
    index = RetrievalIndex(
        train["precursor_mz"].to_numpy(dtype=np.float64), train["_key"].to_numpy(dtype=str),
        train["normalized_smiles"].to_numpy(dtype=str), train["_row"].to_numpy(dtype=np.int64),
        train["ms2_mzs"].to_numpy(), train["ms2_normalized_intensities"].to_numpy(),
        pool["exact_mass"].to_numpy(dtype=np.float64), pool["_key"].to_numpy(dtype=str),
        pool["smiles"].to_numpy(dtype=str), pool["_row"].to_numpy(dtype=np.int64),
        pool["_in_train"].to_numpy(dtype=np.int8), pool["_in_coconut"].to_numpy(dtype=np.int8),
    )

    test = pd.read_parquet(test_path)
    if "adduct" not in test.columns:
        test["adduct"] = None
    test["_molecule_id"] = test["molecule_id"].astype(str)
    groups = {str(key): frame for key, frame in test.groupby("_molecule_id", sort=False)}
    empty = test.iloc[0:0]; guess_lists: list[list[str]] = []
    spectral_molecules = strong_molecules = spectral_candidates = external_guesses = fallback_count = 0
    for molecule_id in sample_ids:
        guesses, spectral_count, strong_count, external_count, fallback = _predict(groups.get(molecule_id, empty), index)
        guess_lists.append(guesses); spectral_molecules += int(spectral_count > 0)
        strong_molecules += int(strong_count > 0); spectral_candidates += spectral_count
        external_guesses += external_count; fallback_count += int(fallback)

    submission = pd.DataFrame({"molecule_id": sample_ids, "smiles": [";".join(x) for x in guess_lists]})
    parts = submission["smiles"].str.split(";")
    assert list(submission.columns) == ["molecule_id", "smiles"]
    assert len(submission) == len(sample) and submission["molecule_id"].tolist() == sample_ids
    assert submission["smiles"].notna().all() and parts.map(len).between(1, MAX_GUESSES).all()
    assert parts.map(lambda values: all(values) and len(values) == len(set(values))).all()
    assert not submission["smiles"].str.contains(";;").any()
    output_path = WORK / "submission.csv"; WORK.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False)
    precursors = pd.to_numeric(test["precursor_mz"], errors="coerce").to_numpy(dtype=np.float64)
    diagnostics = {
        "selected_train_rows": selected_train_rows, "train_rows": int(len(train)), "pool_rows": int(len(pool)),
        "test_rows": int(len(test)), "query_count": int(len(sample_ids)),
        "invalid_test_precursors": int((~np.isfinite(precursors) | (precursors <= 0.0)).sum()),
        "molecules_with_library_candidates": int(spectral_molecules),
        "molecules_with_strong_spectral": int(strong_molecules),
        "library_spectral_candidates": int(spectral_candidates), "external_guesses": int(external_guesses),
        "fallback_guesses": int(fallback_count),
        "max_guesses": int(max((len(values) for values in guess_lists), default=0)),
        "library_candidate_cap": MAX_CANDIDATES, "submission_path": str(output_path),
    }
    print(json.dumps(diagnostics, sort_keys=True))


if __name__ == "__main__":
    main()
