"""Predictor registry. Add one line per new predictor; configs refer to it by name."""
from __future__ import annotations

import importlib

from .base import Candidate, Predictor

REGISTRY = {
    "exp000": "casmi.predictors.exp000:Exp000Predictor",
    "exp002": "casmi.predictors.exp002:Exp002Predictor",
    "exp003": "casmi.predictors.exp003:Exp003Predictor",
    "exp004": "casmi.predictors.exp004:Exp004Predictor",
}

def build_predictor(name: str, params: dict | None = None) -> Predictor:
    module, cls = REGISTRY[name].split(":")
    return getattr(importlib.import_module(module), cls)(**(params or {}))


__all__ = ["Candidate", "Predictor", "build_predictor", "REGISTRY"]
