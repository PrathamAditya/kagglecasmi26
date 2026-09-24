# exp_001 — Validation harness + exp_000 loss decomposition
- Owner: Pratham · Date: 2026-09-24 · Branch: `exp/001-val-harness` · Parent: exp_000 · Config: `configs/exp_001.yaml`

## Hypothesis
exp_000 loses most of its MRR to candidate generation (truth never scored), not to ranking.
Falsified if `cand_not_top25` + `hit@2-25` > `reachable_missed` in C1.

## Change vs parent
No model change. Adds: frozen val set, C1/C2/C3 regimes, Predictor interface, exp_000 port (verified identical to the original on 5 molecules), Kaggle runner sharing the same code.

## Attacks
Measurement. All classes.

## Success criterion (set before running)
1. Runs end-to-end on real data in < 15 min on the local machine.
2. Loss buckets + 20 worst molecules per regime written.
3. `leak_hits` = 0 in C2/C3 (else exclusion is leaking via tautomer ik14 mismatch → fix before trusting C2).

## How to run
```
uv run python -m casmi.harness.build_val --config configs/exp_001.yaml
uv run python -m casmi.harness.evaluate  --config configs/exp_001.yaml --limit 20   # smoke
uv run python -m casmi.harness.evaluate  --config configs/exp_001.yaml             # full
```

## Results
| metric | C1 | C2 | C3 |
|---|---|---|---|
| MRR@25 | | | |
| Recall@cand | | | |
| Recall@25 | | | |
| hit@1 / hit@2-25 / cand_not_top25 / reachable_missed / unreachable | | | |
| crowded windows (>256) | | | |
| leak_hits | | | |
| runtime | | | |

est. LB (0.5/0.35/0.15): ___ vs public 0.127 → first calibration point.

From `build_report.json`: precursor |ppm| p50/p95/p99 = ___ → sets exp_002 tolerance.

## Failure analysis (fill from worst20.csv)
- reachable_missed with window_max > 256 → audit #1 (cap) confirmed?
- reachable_missed with ref_truth_same_adduct_in_tol = False but pool_truth_in_tol = True, water-loss/formate adducts → audit #3 confirmed?
- cand_not_top25 in C2 → audit #4 (random isomer ranking).

## Verdict
## Next
exp_002: fix the bucket that dominates.
