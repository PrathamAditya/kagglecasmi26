"""The one interface every approach implements. Harness and Kaggle runner only talk to this."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd
import polars as pl


@dataclass(frozen=True)
class Candidate:
    smiles: str
    key: str = ""            # InChIKey14 as known to the source (train raw ik14 or pool metric_ik14)
    source: str = ""         # "library" | "pool" | "generated" | "fallback"
    score: float = float("nan")


class Predictor(ABC):
    name = "base"
    # Structure keys (raw InChIKey14 + metric key) the harness removed from the pool for this regime (C3).
    # Predictors that carry a PRIVATE candidate store (not the `pool` frame) must drop these themselves.
    pool_exclude_keys: frozenset[str] = frozenset()

    def __init__(self, **params):
        self.params = params

    @abstractmethod
    def prepare(self, train: pl.LazyFrame, pool: pd.DataFrame, queries: pd.DataFrame) -> None:
        """Build indexes. `train` is already filtered for the regime; `queries` has TEST_COLUMNS only."""

    @abstractmethod
    def predict(self, rows: pd.DataFrame) -> list[Candidate]:
        """All spectra of ONE molecule → FULL ranked candidate list (may exceed 25; used for Recall@cand)."""
