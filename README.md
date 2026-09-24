# CASMI26 — Molecule ID from MS/MS

Kaggle: Enveda CASMI 2026 · Metric: MRR@25 on InChIKey14 (tautomer-canonical) · Deadline: 2026-12-14
Team: Pratham + <teammate>

## Status (update after every experiment)

| exp | date | owner | one-line change | C1 | C2 | C3 | est. LB | public LB | verdict |
|-----|------|-------|-----------------|----|----|----|---------|-----------|---------|
| 000 | 2026-09-24 | Pratham | CPU spectral retrieval + pool fallback | – | – | – | – | **0.127** | baseline, 8 bugs found |
| 001 | 2026-09-24 | Pratham | validation harness (code done, awaiting real run) | | | | | | |

Leaderboard reference (2026-09-24): #1 0.417, #5 0.387. Track our own delta, not theirs.

## Quickstart
```
uv sync --extra dev
# put train.parquet + pool.parquet in data/raw/
uv run pytest -q                                                      # adduct math + metric key
uv run python -m casmi.harness.build_val --config configs/exp_001.yaml
uv run python -m casmi.harness.evaluate  --config configs/exp_001.yaml
```
New approach = new file in `src/casmi/predictors/` implementing `Predictor` + one line in `REGISTRY` + a config.
The harness (local) and `kaggle/submit.py` (LB) run the same predictor, so a local gain is the code that gets submitted.

## Code map
| Path | Role |
|---|---|
| `src/casmi/chem.py` | Adduct offsets (all 10, tested), metric InChIKey14 + disk cache. Only place mass math lives. |
| `src/casmi/data.py` | Test schema, loaders, Kaggle path discovery |
| `src/casmi/predictors/` | `base.Predictor` interface, registry, one module per approach |
| `src/casmi/harness/` | `build_val` (frozen val set), `evaluate` (C1/C2/C3 + loss buckets) |
| `src/casmi/legacy/` | Original exp_000 script, untouched (regression reference) |
| `kaggle/submit.py` | LB runner |

## How to read this repo

| Path | What it is | When to touch it |
|------|-----------|------------------|
| `docs/plan.md` | Current roadmap, diagnosis, decision gates | Re-version when a gate changes the plan |
| `docs/decisions/ADR-*.md` | Why we chose X over Y | Any change of direction |
| `experiments/exp-log.md` | One row per experiment (the index) | Before starting (claim) and after finishing |
| `experiments/exp_NNN_slug/card.md` | Full record of one experiment | During and after the run |
| `configs/exp_NNN.yaml` | Exact config that produced the numbers | Committed with the result |
| `docs/research/literature.md` | Papers/tools + why they matter to us | When you read something useful |

## Rules (keep us from trying random stuff)

1. **One hypothesis, one primary change per experiment.** If two things change, you learn nothing.
2. **Write the success criterion before running.** Decided afterwards = not a criterion.
3. **Local harness is the judge** (ADR-002). Public LB only at milestones; each LB score is a calibration point.
4. **Claim before you start**: add your row to `exp-log.md` with owner + status `running`. No duplicate work.
5. **Negative results get logged** with the same care. A discarded idea with a reason is a win.
6. **Change of direction → ADR.** Parameter tweaks don't need one.
7. Branch `exp/NNN-slug`; merge to `main` only if verdict = keep. `main` always produces a valid submission.
