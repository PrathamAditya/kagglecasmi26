# plan.md — Phase A/B: anchor on V27, then build real ranker training data

## Why this plan (read once)
exp_004–015 all failed for three shared reasons, not because the ideas were bad:
1. **Wrong anchor.** Everything was compared to our own pipeline (est. LB 0.40,
   unverified). The only pipeline with a known LB score is the V27 notebook (0.334),
   and it was never run through the harness with its own pretrained artifacts.
2. **Tiny training sets.** exp_007/010 trained rankers on 25–91 groups. V27's GBDT was
   trained on far more. A ranker on 91 groups can't beat it.
3. **Training distribution had no C1.** Leave-structure-out training groups remove the
   truth from the library, so the ranker never saw "high library sim = answer". That is
   exactly why C1 collapsed in exp_007 and exp_010. Fix the data, not the routing
   (exp_013–015 showed routing on max lib sim can't separate C1 from C2).

Also: n=60 decisions are below noise (SE of MRR ≈ 0.05). All decisions move to full n.

Target architecture (context for later phases, do **not** build yet):
`candidates = pool window ∪ analog-edit generator`
→ `features = V27's 31 features + V27 GBDT prob (stacked) + new features`
→ `our LambdaRank, trained on thousands of mixed-regime groups`.
The V27 prob as a feature is what protects C1; the edit generator is what unlocks C3.

---

## Phase A — V27 anchor + harness hardening (exp_016)  ← do today

### A1. Run V27 as-is through the harness
- Wrap the V27 notebook logic as `src/pipelines/v27.py` (import, don't rewrite the
  scoring). Uses the frozen pool, FPNet checkpoints, GBDT rankers.
- The harness must remove the held-out structure (by InChIKey14 **and** metric
  tautomer key) from the library for C2, and from library + pool for C3.
- Output the standard per-regime table.

### A2. Harness fixes
- Paired bootstrap CI of deltas (1,000 resamples) — shared function, used by every run.
- Enlarge validation beyond enveda-np-examples' 250: add a held-out NP slice
  (riken + gnps + massbank, NP-like, prefer QTOF/timsTOF, ≤ 1 structure per scaffold),
  target ~600 molecules total, split by regime. Freeze the list in
  `data/val_ids_v2.parquet`. Keep the 250-set results reported separately for continuity.
- Explain the C1 drop: exp_001 C1 = 0.859, exp_004+ C1 ≈ 0.745. Find the commit that
  caused it (bisect on the 250-set, C1 only). Report the cause; do not fix unless trivial.
- Record in `docs/decisions.md` the class weights used for "est LB".

### Gate A (all must hold)
- V27-in-harness runs end to end, leak_hits = 0.
- Table printed for both the 250-set and val_v2.
- V27's est LB vs our exp_004 est LB, with CI. **V27 becomes the anchor for all future
  runs** unless exp_004 beats it with CI lower bound > 0.
- C1 regression cause identified (or documented as unresolved with bisect evidence).

Estimated effort: ~half a day. Runtime: V27 is ~30–60 s/molecule on CPU with frag
channel; if val_v2 takes >4 h, run it on Kaggle CPU / Modal and report.

---

## Phase B — Training-group factory (exp_017)  ← start only after Gate A passes

Goal: thousands of candidate groups with labels, in the **same mixed regime
distribution** as test.

- `src/data/group_factory.py`: sample N structures from NP-like training libraries,
  disjoint by InChIKey14 **and** Murcko scaffold from both val sets.
- For each structure, assign a regime and build the group accordingly:
  - C1 (≈35%): truth stays in library and pool.
  - C2 (≈45%): truth removed from library, kept in pool.
  - C3 (≈20%): truth removed from library and pool (group labelled "no positive" for
    now; used later for the edit generator).
- Candidates = V27 candidate list (pool window). Features = V27's 31 features
  **plus the V27 GBDT probability** plus `n_candidates`, `rank_in_v27`.
- Save as sharded parquet: `group_id, regime, cand_key, label, f_0..f_k`.
- Start with N = 3,000; measure throughput; scale to 10k+ if time allows (Kaggle CPU
  or Modal — parallelise by shard).
- Report: groups built, positives per regime, % groups where truth is in the window,
  seconds per group.

### Gate B
- ≥ 3,000 groups, ≥ 1,000 with a positive in C2 regime.
- Zero overlap with val sets (key + scaffold), asserted in code.
- Spot check 5 groups by hand: truth label correct under the metric key.

---

## Out of scope for this plan
- Training the new ranker (Phase C), edit generator for C3 (Phase D).
- Any change to the V27 artifacts.
- Any LB submission. Validate via Kaggle "commit" only if a runtime check is needed.

## Report back
Overwrite `handoff.md` after Gate A and again after Gate B. Include the tables and
anything surprising (especially the C1-drop bisect and V27-vs-exp_004 comparison).
