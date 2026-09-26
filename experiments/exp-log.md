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
