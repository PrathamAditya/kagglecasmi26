"""Paths, config and loaders shared by the harness and the Kaggle runner."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import polars as pl
import yaml

# Exactly the columns the hidden test set has. Predictors only ever see these.
TEST_COLUMNS = [
    "molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities", "base_peak_intensity", "adduct",
    "ionization_mode", "instrument_type", "precursor_mz", "collision_energy_ev", "collision_energy_orig",
    "collision_energy_orig_units",
]
POOL_COLUMNS = ["metric_ik14", "smiles", "exact_mass", "in_coconut", "in_train"]


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def discover(filename: str, root: str | Path = "/kaggle/input") -> Path:
    hits = sorted(Path(root).rglob(filename))
    if not hits:
        raise FileNotFoundError(f"{filename} not found under {root}")
    return hits[0]


def resolve(path_value: str, filename: str) -> Path:
    return discover(filename) if path_value == "auto" else Path(path_value)


def scan_train(path: str | Path) -> pl.LazyFrame:
    """Lazy train frame with a stable `train_row` id (row position in the file)."""
    return pl.scan_parquet(path).with_row_index("train_row")


def load_pool(path: str | Path) -> pd.DataFrame:
    pool = pd.read_parquet(path, columns=POOL_COLUMNS)
    pool["in_train"] = pool["in_train"].fillna(False).astype(bool)
    pool["in_coconut"] = pool["in_coconut"].fillna(False).astype(bool)
    return pool


def to_test_schema(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in TEST_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[TEST_COLUMNS].reset_index(drop=True)
