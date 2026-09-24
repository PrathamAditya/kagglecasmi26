"""Kaggle runner: same Predictor as the harness → submission.csv. Local and LB code paths never diverge.

Kaggle cell:
    import sys; sys.path.insert(0, "/kaggle/input/<repo-dataset>/src")
    !python /kaggle/input/<repo-dataset>/kaggle/submit.py --config /kaggle/input/<repo-dataset>/configs/exp_000.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from casmi.data import load_config, load_pool, resolve, scan_train, to_test_schema  # noqa: E402
from casmi.predictors import build_predictor  # noqa: E402

MAX_GUESSES = 25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="/kaggle/working/submission.csv")
    args = ap.parse_args()
    cfg = load_config(args.config)
    k = cfg.get("kaggle", {})
    train_path = resolve(k.get("train", "auto"), "train.parquet")
    pool_path = resolve(k.get("pool", "auto"), "pool.parquet")
    test_path = train_path.with_name("test.parquet")
    sample = pd.read_csv(train_path.with_name("sample_submission.csv"), dtype={"molecule_id": str})
    ids = sample["molecule_id"].tolist()

    t0 = time.time()
    test = to_test_schema(pd.read_parquet(test_path))
    test["molecule_id"] = test["molecule_id"].astype(str)
    predictor = build_predictor(cfg["predictor"]["name"], cfg["predictor"].get("params"))
    predictor.prepare(scan_train(train_path), load_pool(pool_path), test)
    groups = dict(tuple(test.groupby("molecule_id", sort=False)))

    rows, fallbacks = [], 0
    for mid in ids:
        g = groups.get(mid)
        cands = predictor.predict(g) if g is not None else []
        top = [s for s in dict.fromkeys(c.smiles for c in cands) if s and ";" not in s][:MAX_GUESSES]
        if not top:
            top, fallbacks = ["CCO"], fallbacks + 1
        rows.append((mid, ";".join(top)))

    sub = pd.DataFrame(rows, columns=["molecule_id", "smiles"])
    parts = sub["smiles"].str.split(";")
    assert sub["molecule_id"].tolist() == ids and sub["smiles"].notna().all()
    assert parts.map(len).between(1, MAX_GUESSES).all()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(args.out, index=False)
    print(json.dumps({"molecules": len(ids), "fallbacks": fallbacks, "runtime_s": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
