# Experiment log: CASMI26 timeline

Rule: **log entry first, then submit.** Every Kaggle submission gets a row in Table A. Every idea you test gets an `exp_NNN` in Table B with a card.
Status: planned · running · done · failed · abandoned. Verdict: keep · discard · inconclusive.
C1/C2/C3 = local MRR@25 on the 250 enveda-np-examples val set (ADR-002). n=60 is a smoke test only (CLAUDE.md), never a basis for a decision.
Anchor = the best LB submission so far: **S05 / exp_005, 0.334**.

---

## A. Kaggle submissions (chronological, source of truth for LB)

| sub | date       | owner   | notebook @ version               | exp | change (short)                                                                | LB        | Δ vs anchor | verdict           |
| --- | ---------- | ------- | -------------------------------- | --- | ----------------------------------------------------------------------------- | --------- | ----------- | ----------------- |
| S01 | 2026-09-24 | Sameer  | CASMI External Retrieval v16     | –   | COCONUT pool mass retrieval + adduct correction                               | 0.122     | –           | superseded        |
| S02 | 2026-09-24 | Sameer  | CASMI Spectral Retrieval CPU v25 | 000 | Mass-filtered train cosine retrieval + COCONUT fill                           | 0.127     | –           | baseline          |
| S03 | 2026-09-24 | Sameer  | casmi-four-channel-v27 v1        | –   | Public V27: tautomer dedup + adduct-grouped FPNet                             | 0.320     | –           | superseded        |
| S04 | 2026-09-24 | Sameer  | casmi-four-channel-0339 v1       | –   | Public four-channel reproduction                                              | 0.290     | –           | discard           |
| S05 | 2026-09-24 | Pratham | casmi-four-channel-v27 v2        | 005 | V27 + ppm self-calib, abs mass floor, pooled frag workers, tautomer-key cache | **0.334** | anchor      | **keep (anchor)** |
| S06 | 2026-09-25 | Sameer  | casmi-four-channel-v27 v4 (V28)  | 017 | Per-mode ppm calibration + 20 ppm tail refill                                 | 0.324     | −0.010      | discard           |
| S07 | 2026-09-26 | Pratham | notebook389d54872b v2            | 020 | Gated LambdaRank on own groups (broken pool fps)                              | 0.304     | −0.030      | discard           |
| S08 | 2026-09-26 | Sameer  | casmi-v29-gated-ranker v4 (V29)  | 019 | Gated LambdaMART, fold-clean harness, no FPNet                                | 0.243     | −0.091      | discard           |
| S09 | 2026-09-26 | Pratham | casmi-four-channel-v27 (exp_021) | 021 | S05 + fixed-fp pool                                                           | **ERROR** | –           | fix & resubmit    |

S03 → S05 is +0.014, which is within LB noise. The S05 patches aren't proven to help; S05 is simply the best run so far.

---

## B. Experiments (idea-level)

| exp     | status | owner   | parent     | hypothesis (short)                                                    | n        | C1    | C2    | C3  | est LB\* | LB (sub)    | verdict                                      | card                                   |
| ------- | ------ | ------- | ---------- | --------------------------------------------------------------------- | -------- | ----- | ----- | --- | -------- | ----------- | -------------------------------------------- | -------------------------------------- |
| 000     | done   | Pratham | –          | CPU cosine retrieval + pool fallback                                  | –        | –     | –     | –   | –        | 0.127 (S02) | baseline                                     | exp_000_baseline_retrieval/card.md     |
| 001     | done   | Pratham | 000        | Val harness; exp_000 re-scored with loss buckets                      | 250      | 0.859 | 0.038 | 0   | 0.443    | –           | keep (infra)                                 | exp_001_val_harness/card.md            |
| 002     | done   | Pratham | 001        | Neutral-mass match, all 10 adducts, 10 ppm, structure cap, ik14 dedup | 250      | 0.892 | 0.069 | 0   | –        | –           | keep: recall fixed (C2 R@cand 0.876→0.996)   | exp_002_correctness/card.md (TODO)     |
| 003     | done   | Pratham | 002        | Zero-cap entropy search + analog channel                              | 60/60/20 | 0.714 | 0.074 | 0   | –        | –           | inconclusive (smoke only)                    | exp_003_entropy_search/card.md (TODO)  |
| 004     | done   | Pratham | 003        | 003 + V27 params (8.5 ppm, +1.4 offset, sim_power 4, coconut bonus)   | 60       | 0.717 | 0.074 | –   | –        | –           | inconclusive (smoke only); superseded by 016 | exp_004_entropy_search/card.md (TODO)  |
| 005     | done   | Pratham | public V27 | Patched V27 notebook as reference submission                          | –        | –     | –     | –   | –        | 0.334 (S05) | keep (anchor)                                | reference/casmi-four-channel-v27.ipynb |
| 006–015 | –      | –       | –          | **unused**, don't backfill ids                                        |          |       |       |     |          |             |                                              |                                        |
| 016     | done   | Pratham | 005        | Port S05 into harness; notebook-reference sanity                      | 250      | 0.875 | 0.442 | 0   | 0.592    | = S05       | keep (local anchor)                          | exp_016_v27_anchor/card.md (TODO)      |
| 017     | done   | Sameer  | 005        | Per-mode ppm calibration + tail refill                                | –        | –     | –     | –   | –        | 0.324 (S06) | discard                                      | exp_017_v28_calib/card.md              |
| 018     | done   | Pratham | 016        | Build training groups: 2,500 NP (b00–b03 C2, b04 C1) + val b99        | –        | –     | –     | –   | –        | –           | keep (infra), rebuild on fixed pool          | exp_018_training_groups/card.md        |
| 019     | done   | Sameer  | 005        | Gated LambdaMART, fold-clean harness, no FPNet                        | –        | –     | –     | –   | –        | 0.243 (S08) | discard                                      | exp_019_v29_ranker/card.md             |
| 020     | done   | Pratham | 018        | Gated LambdaRank on 018 groups                                        | b99      | –     | 0.69  | –   | –        | 0.304 (S07) | discard: fp-source shortcut                  | exp_020_gated_lambdarank/card.md       |
| 021     | failed | Pratham | 005        | Fix COCONUT pool fps to V17 recipe                                    | –        | TODO  | TODO  | 0   | –        | ERROR (S09) | fix & resubmit                               | exp_021_fixed_pool/card.md             |

\*est LB uses the placeholder weights 0.5/0.35/0.15. These are wrong (see F6); don't use this column for decisions.

---

## C. Findings (carry forward)

- **F1 (001):** exp_000 loses C2 to ranking, not recall. 137/250 C2 molecules had the truth among candidates but ranked below 25.
- **F2 (002):** the adduct and tolerance fixes closed the recall gap (C2 R@cand 0.996, reachable_missed 30→0). C2 MRR stayed near zero (0.069) → ranking is the bottleneck.
- **F3 (016):** the port matches the notebook exactly (20/20 identical top-25 on C1 and C2). Full-250 local anchor: C1 0.875, C2 0.442. FPNet + GBDT ranking is what lifted C2 from 0.07 to 0.44.
- **F4 (017, 019):** ppm recalibration (−0.010) and dropping FPNet (−0.091) both hurt the LB.
- **F5 (020, 021 diag):** COCONUT pool rows (436k) weren't fingerprinted with the V17 recipe (0/300 match, ~316 bits off). A ranker learns "right recipe = truth" → local C2 0.69 but LB 0.304.
- **F6 (fit from S02 and S05, assuming C3 = 0):** solving `w1·C1 + w2·C2 = LB` for exp_001/S02 and exp_016/S05 gives **w1 ≈ 0.13, w2 ≈ 0.51, w3 ≈ 0.37**. The test is mostly C2 and C3, not C1. This rests on 2 points and assumes local C1/C2 transfer to the test (ADR-002 says local C1 is optimistic), so treat it as direction, not truth. Replace `lb_weights` in configs with a TODO note until a third clean submission confirms it.

---

## D. Run blocks (append one per run, newest last)

### exp_001: validation harness + exp_000 loss decomposition

- git: 306a0aa config: configs/exp_001.yaml anchor: exp_000
- n: 250/250/250 · leak_hits 0/0/0 · runtime ~4 min/regime
- C1 0.859 (R@cand 0.996, R@25 0.976) | C2 0.038 (R@cand 0.876, R@25 0.328) | C3 0
- verdict: keep (infra)

### exp_002: candidate-generation correctness

- git: a151622 config: configs/exp_002.yaml anchor: exp_001
- n: 250/250/250 · leak_hits 0
- C1 0.892 (R@cand 0.996, R@25 0.996) | C2 0.069 (R@cand 0.996, R@25 0.404; cand_not_top25 148) | C3 0
- verdict: keep. Recall solved; ranking is the problem.

### exp_003 / exp_004: entropy search (smoke, n=60)

- git: TODO (likely 6b7e4c7) configs: exp_003.yaml, exp_004.yaml
- 003: C1 0.714 | C2 0.074 · 004: C1 0.717 | C2 0.074
- note: `exp_004/results/results.json` has `exp_id: exp_003`, a mislabel to fix in the harness.
- verdict: inconclusive; direction abandoned in favour of porting V27 (016).

### exp_016: V27 anchor port (full 250)

- git: d51e63e config: configs/exp_016.yaml anchor: exp_005 (S05)
- sanity n=20: C1/C2 20/20 identical top-25 vs notebook → PASS
- ppm offset +1.284 (label-free, n=37 pairs, MAD 0.78)
- C1 0.875 (R@cand 1.0, R@25 0.992) | C2 0.442 (R@cand 1.0, R@25 0.864; cand_not_top25 34) | C3 0
- runtime 2.3–3.3 s/mol CPU · peak RSS 14.7 GB · leak_hits 0
- verdict: keep (local anchor)

### exp_020: gated LambdaRank

- git: fa4b7ad config: configs/exp_020.yaml (TODO)
- local C2 0.69 (b99) → LB 0.304 (S07)
- verdict: discard. Groups built on the inconsistent pool (F5); retrain after 021.

### exp_021: fixed-fp pool

- git: 6695afe config: configs/exp_021.yaml (TODO)
- change: pool_712k_precomputed → pool_712k_fixedfp.npz; everything else = S05
- LB: notebook threw exception (S09). Error: TODO paste log
- verdict: pending

### exp_021c (a03): has_ref penalty on top of fixed pool

- git: 8a0bd5b (predictor patch) + alpha set to 0.3 for submission
- local: C1 0.883 (exp_021: 0.889) | C2 0.607 (exp_021: 0.599, Δ +0.0075 [+0.0014,+0.0158], paired bootstrap)
- LB: PENDING
- verdict: pending

### exp_021c (a03): has_ref penalty on fixed pool — CLOSED

- LB: 0.313 (vs anchor 0.334, vs fixed-pool-alone 0.315)
- local: C1 0.883, C2 0.607 (Δ+0.0075 vs exp_021, CI [+0.0014,+0.0158])
- verdict: discard. Local C2 gain did not transfer to LB — confirms patching frozen
  ranker inputs is not the lever. Local C2 on our val set is not a reliable proxy for
  hidden-set C2 when the ranker itself is untouched. exp_022 (retrain) is now the only
  path forward.

### Duplication bug investigation (2026-09-29) — session with Claude, UNRESOLVED

- Diffing two submission CSVs (labelled 0.334 and 0.315 by user) found severe
  candidate-list duplication in the 0.315 file for many molecules
  (e.g. m_8d75c7: 2/25 unique SMILES, m_e5b5fa: 4/25; n_new=1, n_dropped up to
  24 vs the 0.334 file — confirms duplication, not genuine reordering).
- Ruled out: `_metric_unique_candidates`'s fallback (guards `if smi not in out`,
  cannot itself produce literal duplicates) — confirmed against actual repo
  code (src/casmi/predictors/exp016_v27.py), not just the notebook.
- Ruled out: no per-molecule try/except with nearest-mass fallback exists
  anywhere in exp016_v27.py. Earlier notes describing such a fallback in
  "313_fixedfp_norefill" do not match any code found in the current repo —
  either that code was never merged, or belongs to a notebook not yet located.
- Ruled out: duplicate SMILES strings in the pool itself. Checked
  pool_712k_fixedfp.npz directly — 712,199 distinct SMILES / 712,199 rows,
  zero duplicates.
- UNRESOLVED — config mismatch discovered late in session: the "0.315" CSV
  we diffed was never confirmed against a specific exp_id/config. Investigation
  conflated it with exp_021c (0.313, has_ref penalty) — but per exp-log.md,
  0.315 should be the fixed-pool-alone run (exp_021, no penalty), a different
  config. Has_ref-penalty code (ref_penalty_alpha/sim in exp016_v27.py) may be
  entirely irrelevant to this bug. NEEDS CONFIRMATION: which exp_id/config
  produced submission_0315.csv before resuming this thread.
- Found (separate, real, unrelated bug): `_logits_raw` in exp016_v27.py
  (~line 631-632) still has the pre-334_refill FPNet row-alignment bug —
  drops empty peak lists without dropping matching metadata rows. Never
  backported from the notebook fix that resolved it. Affects every experiment
  run through exp016_v27, present-tense, including exp_021c.
- Next step when resumed: confirm the config for submission_0315.csv, then
  instrument the real predictor at the \_metric_unique_candidates call site to
  print `len(cand)` vs `len(set(cand.tolist()))` — checks whether duplication
  starts upstream (four-channel merge produces repeated pool indices) or is
  introduced during dedup itself.
- Decision: parking this to start on own-FPNet build (see plan.md /
  docs/decisions.md). Not blocking — FPNet validation uses the class-1/class-2
  harness split, which does not depend on this bug being resolved.

  # Own FPNet — build log and run `plain1` (2026-09-30)

> Paste at the end of `experiments/exp-log.md`. Assign the next free `exp_id` (proposed run name: `plain1`).
> Numbers below are copied from terminal output you pasted; anything not seen is marked **[fill in]**.

## 1. Why this experiment

- V27 anchor 0.334 LB. Class 1 is solved by library match; the gap is class 2/3.
- Frozen FPNet is a black box (no training code in repo, unknown training data, may have seen validation molecules).
- Decision: **build our own FPNet first (plain baseline, trained on Modal); DreaMS fine-tuning comes later as a separate comparison.** Component swap only; V27 stays the fallback until our version wins on validated evidence.
- FPNet is the one component the local harness can validate, because it never touches pool fingerprints.

## 2. Status at a glance

| Item                                            | State                                                                       |
| ----------------------------------------------- | --------------------------------------------------------------------------- |
| Architecture recovered + copied (`model.py`)    | done, trains and evaluates                                                  |
| Fingerprint recipe verified against pool        | done, 0 mismatches / 3000                                                   |
| Targets built                                   | done (269,846 molecules)                                                    |
| Full prep                                       | done (1,927,596 spectra)                                                    |
| Local smoke + overfit tests                     | done (pipeline works; overfit test only partial pass)                       |
| Modal smoke                                     | done (setup works, ~1.3-2.9k spec/s)                                        |
| **Run `plain1` (20,000 steps)**                 | **in progress; last seen step 15,000/20,000** — final numbers **[fill in]** |
| Own vs frozen FPNet comparison (`eval_ckpt.py`) | **not run yet**                                                             |
| Swap into V27 / harness comparison              | not started                                                                 |

## 3. What we did, in order

1. **Recovered the FPNet architecture** from the two frozen checkpoints (`artifacts/fp_models/fp_merged_m1.pt`, `fp_single_s2.pt`): d=512, 6 layers, 6930 output bits, 36.0M parameters, trained 24,000 / 20,000 steps. The class itself exists only in the V27 notebook ("CELL 6 — FPNet verbatim"); `src/` has no FPNet class. Attention heads = 8 is **assumed** (default in the class; not stored in checkpoints).
2. **Copied the class** to `src/casmi/fpnet/model.py` (same parameter names, so frozen weights load strictly). `check_fpnet_copy.py` (frozen-vs-copy equality) failed on an import at first and was **not re-run** — we chose to train first. Status: unverified equality.
3. **Verified the fingerprint recipe** (`fp_recipe.py`, from `fix_pool_fps.py`: Morgan r2/4096 + Morgan r3/4096 + RDKit FP/2048 + MACCS/167, bit-selected by `fp_bits.npy`, packed to 867 bytes). `check_fp_recipe.py` on `pool_712k_fixedfp.npz` (712,199 rows): sampled 3000, parse failures 0, mismatches 0.
4. **Built training targets** (`build_fp_targets.py`): 2,539,608 spectra / 275,810 structures in `train.parquet`; kept only the ten test adducts → 2,176,877 spectra / 269,846 structures; parse failures 0; 122 s; mean bit density 0.1289. Output `artifacts/fpnet/fp_targets.npz`.

   | adduct       | spectra   | molecules |
   | ------------ | --------- | --------- |
   | [M+H]+       | 1,324,585 | 243,970   |
   | [M-H]-       | 465,688   | 98,485    |
   | [M+Na]+      | 178,092   | 25,112    |
   | [M+CH2O2-H]- | 83,188    | 20,723    |
   | [M+NH4]+     | 53,452    | 8,890     |
   | [M+Cl]-      | 27,511    | 7,148     |
   | [M-H2O+H]+   | 21,682    | 5,594     |
   | [M+K]+       | 18,955    | 4,335     |
   | [M-2H2O+H]+  | 3,439     | 1,002     |
   | [M-H2O-H]-   | 285       | 149       |

   Thin adducts: `[M-H2O-H]-` and `[M-2H2O+H]+`. Dropped: dimers etc. (~340k spectra) — not in test.

5. **Wrote data prep** (`data.py`, `prep.py`): peak cleaning identical to the notebook (`prep_peaks`: drop > precursor+1.5 Da, 0.1% floor, ≤128 peaks with 8 per 50 Da window, sqrt intensities), plus label-quality filters (|precursor ppm| ≤ 10, raw base peak ≥ 1000 — both mirror how the test set was cleaned). Validation = all molecules of `enveda-np-examples` + 1.5% hash-chosen molecules, held out **across all libraries** (by `inchikey14`).
6. **Local GPU:** the project's torch was a CPU-only build (`2.14.0+cpu`). Installed a CUDA build into the venv only (`uv pip install --reinstall-package torch --torch-backend cu130 torch`); always run with `uv run --no-sync` so the lock file doesn't revert it. GTX 1650 ≈ 70 spectra/s (fp32, batch 32).
7. **Small prep** (`--limit 200000`): reached only `enveda-180` + `drug_plus` (file is sorted by library): 143,291 spectra kept (val 2,311), 20 s. Smoke test 200 steps: loss plateau 0.243 = bit-frequency level, per-molecule MRR 0.0130 vs chance 0.0128 (nothing learned yet, expected).
8. **Added diagnostics** to `train.py`: train-probe (same size as validation but from training molecules), chance level, per-molecule MRR (average logits over a molecule's spectra), natural-product proxy (holdout-library molecules), and a prior-only baseline printed once at start.
9. **Overfit test** (`--train-subset 2000`, 600 steps, batch 32): train-probe MRR 0.0136 → 0.0210 → 0.0279 (steps 200/400/600) while validation MRR stayed 0.0146 → 0.0178 → 0.0188; probe loss 0.2416 → 0.2310, validation loss flat ≈ 0.241. Verdict: partial pass — spectra reach the model and the probe detects memorization, but memorizing was slow (target was probe loss < ~0.15).
10. **Full prep** (315 s) → `artifacts/fpnet_data`:
    - Candidates (test adduct + known structure) 2,176,877. Dropped: precursor ppm > 10 → 172,924 (7.9%); base peak < 1000 → 70,537 (3.2%); no peaks left → 5,820.
    - **Kept 1,927,596 spectra** (train 1,856,291 / val 71,305), 266,850 molecules, mean 46.3 peaks/spectrum; 153,822 kept spectra have < 5 peaks.
    - Missing collision energy (set to 25 eV; `--drop-null-ce` NOT used because it would remove all of `gnps`): gnps 124,427; spectraverse 42,994; riken 19,793; msdial 14,976; mona 12,575; drug_plus 2,390; massbank 973; masaryk 650 (total 218,778 = 11%).
    - Spectra per library: enveda-180 821,960; pluskal_ms2 472,353; riken 245,646; gnps 124,427; massbank 89,828; mona 85,228; spectraverse 46,839; msdial 37,093; drug_plus 2,390; enveda-np-examples 1,182; masaryk 650.
11. **Modal smoke** (app `ap-pKGFIug5xD48imVjz4qJIY`, 300 steps, A10G): image build, volume, imports, data, eval, checkpointing all OK; 1,301-2,879 spec/s; per-molecule MRR 0.0030 vs chance 0.0026 (learning rate never left warmup — expected).
12. **Run `plain1`** launched on Modal (details below).

## 4. Config of `plain1`

|               |                                                                                                                                                                                        |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Model         | FPNet, d=512, 6 layers, 8 heads (assumed), dropout 0.1, 36.0M params, 6930 outputs                                                                                                     |
| Loss          | BCE with logits, mean over bits and batch                                                                                                                                              |
| Optimizer     | AdamW, betas (0.9, 0.999), weight decay 0.01 (matrices only), grad clip 1.0                                                                                                            |
| LR            | peak 3e-4, linear warmup 1,000 steps, cosine to 10% (3e-5) at step 20,000                                                                                                              |
| Batch / steps | 256 / 20,000 (≈5.1M samples ≈ 2.8 passes over 1.86M train spectra)                                                                                                                     |
| Precision     | bf16 autocast (loss in fp32)                                                                                                                                                           |
| Sampling      | uniform over spectra; `lib_weights` none, `mol_balance` 0, `instr_dropout` 0                                                                                                           |
| Prep filters  | adducts = ten test adducts; ppm ≤ 10; base peak ≥ 1000; min peaks 1; null CE → 25 eV                                                                                                   |
| Validation    | 71,305 spectra of 3,4xx molecules (1.5% hash + enveda-np-examples); evaluation samples 20,000 spectra = 3,280 molecules, every 1,000 steps; best checkpoint chosen by per-molecule MRR |
| Hardware      | Modal A10G, 8 CPU, 32 GB, 6 loader workers; ≈ 1,630 spec/s steady                                                                                                                      |
| Seed / resume | seed 0; resumable (`last_state.pt` every 500 steps + at each eval)                                                                                                                     |

Launch: `modal run --detach src/scripts/modal_fpnet.py --stage train --run-name plain1`
Outputs on volume `casmi-fpnet-vol`: `/runs/plain1/fpnet_own_best.pt`, `fpnet_own_final.pt`, `last_state.pt`.

## 5. Results so far (evaluations I have seen)

Metric: rank of the true molecule among **3,280 validation molecules** (NOT the 712k pool — do not compare with the leaderboard). Chance MRR ≈ 0.0026.

| step                     | train-probe loss              | train-probe MRR | val loss | per-spectrum MRR | per-molecule MRR | top1  | top10 | NP proxy (248 mol) |
| ------------------------ | ----------------------------- | --------------- | -------- | ---------------- | ---------------- | ----- | ----- | ------------------ |
| 14,000                   | 0.15378                       | 0.6182          | 0.16491  | 0.4813           | 0.5135           | 0.437 | 0.641 | 0.5672             |
| 15,000                   | 0.15196                       | 0.6474          | 0.16718  | 0.4913           | 0.5382           | 0.463 | 0.669 | 0.5715             |
| 20,000 (final)           | 0.14515                       | 0.6970          | 0.16847  | 0.5106           | 0.5733           | 0.501 | 0.695 | 0.5925             |

## 6. Files used

All under `src/`. Sizes are approximate (each 1 byte less after download).

| File                                | Role                                                 | Status                                                                                             |
| ----------------------------------- | ---------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| `casmi/fpnet/model.py` (3.9 KB)     | FPNet architecture (copy of notebook CELL 6)         | used by all training/eval                                                                          |
| `casmi/fpnet/fp_recipe.py` (1.7 KB) | SMILES → 6930-bit packed fingerprint                 | verified vs pool                                                                                   |
| `casmi/fpnet/data.py` (4.8 KB)      | peak cleaning, batching, `SpecData`                  | ran on real data                                                                                   |
| `casmi/fpnet/prep.py` (8.0 KB)      | train.parquet → compact arrays + filters + val split | ran on real data                                                                                   |
| `casmi/fpnet/train.py` (~15 KB)     | training loop, eval, probes, prior baseline          | ran locally + Modal; **confirm version used for plain1 (14,965 B = includes prior-only baseline)** |
| `scripts/build_fp_targets.py`       | fingerprints for all training structures             | ran                                                                                                |
| `scripts/check_fp_recipe.py`        | recipe check vs pool                                 | ran, passed                                                                                        |
| `scripts/check_fpnet_copy.py`       | frozen-vs-copy equality                              | **not run successfully**                                                                           |
| `scripts/modal_fpnet.py`            | Modal prep/smoke/train                               | ran (smoke + plain1)                                                                               |
| `scripts/eval_ckpt.py`              | score any checkpoint on our exact validation split   | **written, not yet run**                                                                           |

Data/artifacts: `train.parquet` (competition), `artifacts/fpnet/fp_targets.npz`, `artifacts/fpnet_data/` (full prep), `artifacts/fpnet_data_small/` (test only), `artifacts/coconut_candidates/fp_bits.npy`, `artifacts/v17_builder/pool_712k_fixedfp.npz`, frozen weights in `artifacts/fp_models/`, Modal volume `casmi-fpnet-vol`.

## 7. Issues, gotchas, open questions

- A wrong file was once saved as `fp_recipe.py` (it was `build_fp_targets.py`); fixed. Check `wc -c` after copying files.
- `resume` ignores changed settings: use a fresh `--out-dir` / run name per experiment.
- Local torch was CPU-only; use `uv run --no-sync` with the CUDA build.
- Row-alignment bug in the FPNet inference path (notebook `_logits_from` uses `rows[:B]` after dropping empty spectra; project notes place it near `_logits_raw` in `exp016_v27.py`, ~line 631). Not fixed. Fix before any harness comparison or swap.
- `exp016_v27.py` in the repo contains no FPNet class (grep for `qkv` found nothing) → where the harness gets FPNet from is unconfirmed.
- Data page says "one of the seven above" but lists ten adducts; we use ten.
- Frozen FPNet may have trained on our validation molecules (its number = upper bound).
- Only 248 natural-product proxy molecules → wide confidence intervals; local MRR (3,280 candidates) cannot predict the leaderboard. Local wins misled twice before (pool swap, `has_ref` penalty).
- COCONUT is not used in training (structures only, no spectra); it enters at inference via the candidate pool.

## 8. Next steps

1. Let `plain1` finish; record final table above; download `fpnet_own_best.pt`.
2. Run `eval_ckpt.py` on ours + `fp_single_s2.pt` + `fp_merged_m1.pt` (also serves as the bit-order check: frozen near chance = bit order/input mismatch).
3. Ablations, one change each, same steps, two seeds for a noise estimate: cosine/soft-Tanimoto loss, natural-product `lib_weights`, `instr_dropout 0.3`, `mol_balance 0.5`, m/z-jitter + peak-dropout augmentation, looser ppm filter (20-30), "unknown CE" marker instead of 25 eV, longer training, peak LR 1e-4 / 6e-4. Judge on per-molecule MRR and the NP proxy with bootstrap CIs.
4. Final config: retrain including the 250 `enveda-np-examples` molecules (upweighted); train the merged-spectrum variant (keep "merged" in the filename); optional small seed ensemble.
5. Fix the alignment bug, then swap into V27 one change per submission; check the frozen ranker's feature distributions first (frozen ranker was trained on the old FPNet's outputs).
6. Ranker retraining only after FPNet is solid, with class-stratified validation.
7. DreaMS (later): confirm checkpoint loads in its own Python 3.11 env, add `--model dreams`, compare at matched steps (600 / 2,000 / 5,000)
