# exp_000 — CPU spectral retrieval + pool fallback
- Owner: Pratham · Date: 2026-09-24 · Code: `casmi_spectral_retrieval_cpu.py` · Config: `configs/exp_000.yaml`

## Method
1. Per test spectrum: train spectra within max(0.01 Da, 20 ppm) of the raw precursor m/z; keep the 256 nearest by mass.
2. Greedy cosine, 0.01 Da tolerance, raw normalized intensities. Best score per ik14 across the molecule's spectra.
3. Ranking: spectral hits with score ≥ 0.20 (max 12) → pool candidates by (in_train, in_coconut, adduct-shift rank, mass error) → remaining spectral hits. Max 25.

## Results
| metric | value |
|---|---|
| public LB | **0.127** |
| local C1/C2/C3 | not measured (no harness yet) → exp_001 |

## Audit (to verify in exp_001, fix in 002–003)
1. 256-candidate cap by mass distance drops the true compound (biggest suspect).
2. Train matched on raw precursor m/z → same-adduct only.
3. Missing adducts: [M-H2O+H]+, [M-2H2O+H]+, [M-H2O-H]-, [M+CH2O2-H]- → wrong neutral mass for pool lookup.
4. Pool isomers ranked by mass error → effectively random within a formula.
5. No cleaning (precursor peak kept, no floor, no intensity transform).
6. 20 ppm pool tolerance, not measured for timsTOF.
7. Dedup by SMILES rather than metric ik14.
8. Cl⁻ offset off by one electron mass.

## Verdict
Baseline. Superseded once exp_002 lands. Kept runnable for regression checks.
