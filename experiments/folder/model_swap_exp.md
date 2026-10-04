# CASMI26 — Model Swap Log (own baseline → V27 → own FPNet → DreaMS)

Metric: public LB MRR@25 (~130 molecules visible; same code reruns identically — 0.334 reproduced exactly, so gaps are signal). Anchor to beat: **0.344**.

## TL;DR

- Every swap of a _learned component_ into the V27 pipeline lost to the frozen original. Best own-encoder result: **DreaMS-20k + fixed pool = 0.299** vs frozen FPNet + fixed pool = **0.315**, vs anchor **0.344** (frozen FPNet + original pool).
- Local metrics misled 3×; **LB is the only trusted arbiter** until the dev metric is fixed.
- Leading hypothesis (untested): downstream ranker was trained on the _frozen_ FPNet's output distribution → any swapped encoder is mis-calibrated for it. Next test: own ranker on DreaMS-20k outputs.

## Pipeline slots (V27 = 4 channels: library match, mass-shifted analog, MetFrag-lite, FPNet+GBDT ranker)

| Slot                | Frozen (V27)                                                             | Swapped variants                                                                                                                                              |
| ------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Candidate pool      | `pool_712k_precomputed.npz` (COCONUT+ChEBI+LIPID MAPS+train; no PubChem) | `pool_712k_fixedfp.npz` (435,743/712,199 rows = COCONUT fps recomputed to the train recipe; Morgan r2/4096 + r3/4096 + RDKit/2048 + MACCS/167, `fp_bits.npy`) |
| Spectrum→FP encoder | `fp_merged_m1.pt`, `fp_single_s2.pt`                                     | own plain FPNet (20k/30k), DreaMS fine-tune (5k/20k/40k) — all 6930-bit target                                                                                |
| Ranker              | V17 LightGBM/ranker ensemble (`rank_train.npz`, provenance lost)         | exp_020 / `ranker_v1_bal` (LambdaRank) — failed; own ranker on DreaMS-20k — in progress                                                                       |

## Swap chronology (public LB)

| #   | Change vs previous                                                                           | Encoder | Pool  | LB        | Δ vs ref |
| --- | -------------------------------------------------------------------------------------------- | ------- | ----- | --------- | -------- |
| 0   | exp_000 own CPU retrieval + pool fallback                                                    | none    | —     | 0.127     | —        |
| 1   | V27 community reference                                                                      | frozen  | orig  | 0.320     | ref      |
| 2   | 3rd-party V27 reproduction                                                                   | frozen  | orig  | 0.290     | −0.030   |
| 3   | V27 + patches (ppm self-calib, mass floor, pooled frag workers, tautomer cache, tail_refill) | frozen  | orig  | 0.334     | +0.014   |
| 4   | Fork `c01c30` of #3 → **anchor**                                                             | frozen  | orig  | **0.344** | +0.010   |
| 5   | #3 + fixed pool only (exp_021b)                                                              | frozen  | fixed | 0.315     | −0.019   |
| 6   | #5 + `has_ref` penalty α=0.3 (exp_021c)                                                      | frozen  | fixed | 0.313     | −0.002   |
| 7   | Own LambdaRank on simulated C2 groups, gated lvmax≥0.70 (exp_020; `ranker_v1_bal`) ¹         | frozen  | orig  | 0.304     | −0.030   |
| 8   | Own FPNet 20k (plain1) into #4                                                               | own     | orig  | 0.272     | −0.072   |
| 9   | Own FPNet 30k (plain1_ext30k) into #4                                                        | own     | orig  | 0.271     | −0.073   |
| 10  | Own FPNet 20k + fixed pool                                                                   | own     | fixed | 0.266     | −0.078   |
| 11  | DreaMS-5k + fixed pool                                                                       | DreaMS  | fixed | 0.279     | −0.065   |
| 12  | DreaMS-20k + fixed pool                                                                      | DreaMS  | fixed | **0.299** | −0.045   |
| 13  | DreaMS-40k (continued from 20k state) + fixed pool                                           | DreaMS  | fixed | 0.292     | −0.052   |

¹ Memory logs exp_020 and `ranker_v1_bal` both at 0.304 — confirm whether same run.
DreaMS-30k trained (`dreams_30k_final.pt`), not submitted. DreaMS-5k + original pool: planned, no result recorded.

## Pool-only effect (encoder held fixed)

- Frozen: 0.334 → 0.315 (**−0.019**). Own-20k: 0.272 → 0.266 (**−0.006**).
- Local harness predicted the _opposite sign_ (+0.158 C2, est_lb 0.592→0.654): local truths are train rows (fps unchanged by fix); hidden C2/C3 truths are COCONUT rows (fps changed). Local pool comparisons are valid only within one pool.
- Pool bug origin: COCONUT fps loaded pre-made from `coco_fp.npy` (different upstream recipe); train fps computed in-notebook with the correct recipe.

## Encoder-only effect (fixed pool, ranker frozen)

Frozen **0.315** > DreaMS-20k 0.299 > DreaMS-40k 0.292 > DreaMS-5k 0.279 > own-20k 0.266.

- DreaMS over own FPNet at 20k: **+0.033 LB**; frozen over DreaMS-20k: **+0.016**.
- More DreaMS steps past 20k did not help (40k −0.007).

## Training details

- **Own FPNet (plain1):** Modal; 20k steps, extended to 30k. Beat frozen FPNet locally; lost on LB (−0.07).
- **DreaMS fine-tune:** pretrained DreaMS encoder, own plain-PyTorch copy, same 6930-bit FP target, Modal. 5k run, fresh 20k run, then 30k and 40k as separate continuations from the 20k state (reuses compute).
- **Local at matched steps:** 5k — DreaMS per-molecule MRR 0.291 vs own 0.167 (pretraining head start). 20k — 0.564 vs 0.573 (level) at ~2.7× cost/step.

## Local-metric failures (why LB decides)

1. Simulated-C2 ranker validation → exp_020 passed locally, 0.304 on LB.
2. Pool fix → +0.158 local C2, −0.019 LB.
3. Own FPNet → beat frozen locally, −0.07 on LB.
4. Isomer-panel dev metric (same-formula isomers, tautomer twins marked): 5/6 pairs concordant with LB, but own-20k (0.683) vs DreaMS-5k (0.528) flipped; frozen scores 0.830 on all panels yet 0.497 on NP proxy (below own-20k 0.549, DreaMS-20k 0.574). **Not used to pick between models.**

## Known open bugs

- `_logits_raw` in `exp016_v27.py` (~L631–632): FPNet row-alignment bug (drops empty peak lists without dropping metadata rows); fix exists in refill notebook, never backported.
- exp_021c submission had heavy duplicate candidates (2–6 unique SMILES of 25 in worst cases); root cause unidentified, parked.
- Which config produced the diffed "0.315" CSV unconfirmed (conflated with exp_021c 0.313).

## Artifacts

| Kaggle dataset                        | Contents                                                                                           |
| ------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `prathamaditya/casmi26-fpnet-own-20k` | `fp_own_20k_single.pt`, `fp_own_30k_single.pt` (name must match `fp_*.pt` for the notebook loader) |
| `prathamaditya/casmi26-dreams-5k-20k` | DreaMS 5k + 20k checkpoints + self-contained loader                                                |
| `prathamaditya/casmi26-ranker-mols`   | validation molecule list for ranker groups                                                         |
| `prathamaditya/casmi26-ranker-v1`     | `ranker_v1_bal.txt` (failed, 0.304)                                                                |
| `casmi26-pool-fixedfp`                | `pool_712k_fixedfp.npz`                                                                            |

Notebooks: `baseline-334-dreams-5k-model-and-fixed-pool.ipynb`, `fork-of-casmi-four-channel-v27-c01c30` (anchor), `ranker-groups-dreams20k`.

## Next

1. Train own ranker on DreaMS-20k outputs (fixed pool kept); groups built for all 4,229 validation molecules (`ranker-groups-dreams20k`).
2. Fair test = DreaMS-20k + own ranker vs frozen FPNet + frozen ranker (same fixed pool).
3. Then: PubChem pool (exp_023, other account), C3 generation.
