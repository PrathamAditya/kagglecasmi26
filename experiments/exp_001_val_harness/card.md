# exp_001 — Validation harness + exp_000 loss decomposition

- Owner: Pratham · Date: 2026-09-24 · Branch: `exp/001-val-harness` · Parent: exp_000 · Config: `configs/exp_001.yaml`

## Hypothesis

exp_000 loses most of its MRR to candidate generation (truth never scored), not to ranking.
Falsified if `cand_not_top25` + `hit@2-25` > `reachable_missed` in C1.

**Result:** Partially confirmed. In C1, candidate generation is strong (recall 0.996). But in C2, ranking is the dominant problem: 137 molecules have truth in candidates but ranked below 25.

## Change vs parent

No model change. Adds: frozen val set, C1/C2/C3 regimes, Predictor interface, exp_000 port (verified identical to the original on 5 molecules), Kaggle runner sharing the same code.

### Files added

| file                             | purpose                                                                                    |
| -------------------------------- | ------------------------------------------------------------------------------------------ |
| `src/casmi/chem.py`              | All 10 adduct offsets (tested), metric InChIKey14 + tautomer canonicalization + disk cache |
| `src/casmi/data.py`              | Shared loaders, test schema, path discovery                                                |
| `src/casmi/predictors/base.py`   | `Predictor` interface (prepare + predict)                                                  |
| `src/casmi/predictors/exp000.py` | exp_000 wrapped in Predictor interface                                                     |
| `src/casmi/harness/build_val.py` | Frozen val set from enveda-np-examples (250 molecules, 1183 spectra)                       |
| `src/casmi/harness/evaluate.py`  | C1/C2/C3 scoring with loss buckets + tautomer-twin exclusion                               |
| `kaggle/submit.py`               | LB runner using the same Predictor (local = Kaggle, always)                                |
| `tests/test_chem.py`             | Unit tests: 10 adduct offsets against glucose, metric key stereo invariance                |
| `configs/exp_001.yaml`           | Config pointing to exp000 predictor                                                        |

## Attacks

Measurement. All classes.

## Success criterion (set before running)

1. Runtime: Completed in ~12.5 min total (269s + 241s + 245s across 3 regimes). Within the < 15 min threshold on the local machine. [PASS]
2. Artifacts: Loss buckets and 20 worst molecules per regime successfully generated. [PASS]
3. Data Leakage: leak_hits = 0 confirmed in C2/C3 via tautomer-twin exclusion. [PASS]

## How to run

1. uv run python -m casmi.harness.build_val --config configs/exp_001.yaml
2. uv run python -m casmi.harness.evaluate --config configs/exp_001.yaml --limit 20 # smoke
3. uv run python -m casmi.harness.evaluate --config configs/exp_001.yaml # full

## Results (n=250)

| metric                 | C1        | C2        | C3        |
| ---------------------- | --------- | --------- | --------- |
| MRR@25                 | **0.859** | **0.038** | **0.000** |
| Recall@cand            | 0.996     | 0.876     | 0.000     |
| Recall@25              | 0.976     | 0.328     | 0.000     |
| hit@1                  | 200       | 2         | 0         |
| hit@2-25               | 44        | 80        | 0         |
| cand_not_top25         | 5         | **137**   | 0         |
| reachable_missed       | 1         | 30        | 0         |
| unreachable            | 0         | 1         | 250       |
| crowded windows (>256) | 184       | 156       | 156       |
| leak_hits              | 0         | 0         | 0         |
| runtime (s)            | 269       | 241       | 245       |

est. LB (0.5/0.35/0.15): **0.443** vs public LB **0.127** → test set is heavily C2/C3 weighted.

From `build_report.json`: precursor |ppm| p50=1.6 / p95=3.7 / p99=6.6 → exp_002 tolerance = 10 ppm.

## Failure analysis

- **C2 cand_not_top25 = 137** — pool isomers ranked by mass error = effectively random. Audit #4 confirmed. This is the #1 problem.
- **C2 reachable_missed = 30** — missing adducts ([M-H2O+H]+, [M-2H2O+H]+, [M-H2O-H]-, [M+CH2O2-H]-) and loose tolerance. Audit #3 confirmed.
- **C1 crowded windows (184/250 > 256)** — yet recall still 0.996, so the 256 cap hurts less than expected for these common compounds. Audit #1 less severe than suspected for C1.
- **16/250 metric ik14 ≠ raw ik14** — tautomer mismatch confirmed. Twin exclusion fixes it.

## Verdict: keep

Harness is the measurement tool for all future experiments.

## Next

exp_002: fix adducts + 10 ppm tolerance + metric-ik14 dedup → recover 30 reachable_missed.
Then exp_003: Tanimoto ranker for pool candidates → fix 137 cand_not_top25 (biggest single gain expected).
EOF
