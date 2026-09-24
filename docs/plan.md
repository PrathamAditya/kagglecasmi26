# Plan v1 — 10 iterations (2026-09-24)

## 1. Where we are
exp_000 (CPU spectral retrieval + pool fallback) → public LB **0.127**. Leader 0.417.
The gap is mostly pipeline correctness and ranking, not model capacity. Fix those before training anything big.

## 2. Diagnosis of exp_000 (ranked by suspected impact — confirm with exp_001)

1. **Candidate truncation.** Each spectrum is scored against only the 256 train spectra nearest in precursor m/z. A ±20 ppm window over 2.5M spectra (1.15M from enveda-180) holds far more than 256; the cut is by mass distance, which says nothing about correctness. The true compound is often never scored. Root cause: pure-Python O(n·m) cosine.
2. **Library search keyed on raw precursor m/z** → same-adduct only. Test [M+Na]+ vs train that only has [M+H]+ = invisible, while the window fills with unrelated compounds sharing the ion m/z.
3. **4 of 10 test adducts missing** from `ADDUCT_OFFSETS`: [M-H2O+H]+, [M-2H2O+H]+, [M-H2O-H]-, [M+CH2O2-H]- (dict has `[m+hcoo]-`, which never matches). Fallback uses precursor or precursor−H as neutral mass → wrong pool candidates for all of these. Water-loss adducts are common for glycosides/polyols, i.e. natural products.
4. **Pool (class 2) ranking is effectively random.** Isomers share a formula and therefore exact mass; sorting by (in_train, in_coconut, mass error) cannot separate structures.
5. **No spectrum cleaning.** Raw intensities, precursor peak kept, no noise floor → cosine dominated by 1–2 peaks, often the precursor, which every isobar "shares".
6. **20 ppm pool tolerance** is loose for timsTOF → more formulas → more isomers to rank. Measure real error on enveda-np-examples.
7. **Dedup on SMILES, not metric InChIKey14** → stereo/tautomer duplicates can burn slots (train `inchikey14` is not tautomer-canonical like the metric).
8. Cl⁻ offset uses the atom mass (off by one electron, 0.55 mDa). Cosmetic.

Correct neutral-mass offsets (M = precursor_mz + offset):

| adduct | offset |
|---|---|
| [M+H]+ | −1.007276 |
| [M+NH4]+ | −18.033823 |
| [M-H2O+H]+ | +17.003289 |
| [M-2H2O+H]+ | +35.013853 |
| [M+Na]+ | −22.989218 |
| [M+K]+ | −38.963158 |
| [M-H]- | +1.007276 |
| [M-H2O-H]- | +19.017841 |
| [M+CH2O2-H]- | −44.998203 |
| [M+Cl]- | −34.969402 |

## 3. Principle
**MRR = P(true structure in candidate set) × ranking quality within it.**
Every experiment states which factor it attacks and for which novelty class. The harness reports both.

## 4. Scoreboard (every experiment reports all of these — see ADR-002)
- **C1 MRR** — queries: enveda-np-examples; reference: train minus enveda-np-examples rows.
- **C2 MRR** — also remove every train spectrum of the query's ik14; structure stays in the pool.
- **C3 MRR** — also remove it from the pool (≈0 until analog/de-novo work).
- **Recall@cand** and **Recall@25** per class.
- **Est. LB** = w1·C1 + w2·C2 + w3·C3. Placeholder weights 0.5/0.35/0.15; refit from our LB submissions.
- Kaggle CPU runtime (budget 9 h, offline).

## 5. Roadmap

| exp | hypothesis / change | attacks | cost | success criterion (set now) |
|-----|--------------------|---------|------|-----------------------------|
| 001 | Validation harness + exp_000 re-scored with recall/rank decomposition | measurement | 0.5 d | Harness runs < 15 min; per-class numbers + 20 worst molecules dumped |
| 002 | Correctness: all 10 adducts, neutral-mass candidate generation over unique train structures + pool, measured ppm tolerance, metric-ik14 dedup | recall C1/C2 | 0.5 d | Recall@cand C2 ≥ 0.9 |
| 003 | Fast full search (FlashEntropy via `ms_entropy`, or sparse binned cosine), no cap; cleaning (drop > precursor−1.6, 1% floor, top-128, entropy/sqrt weighting); aggregate over a molecule's spectra (max vs sum-top-2) | ranking C1 | 1 d | C1 MRR +0.10 over 002 (expected largest single jump) |
| 004 | Reference side: consensus spectra per (ik14, adduct) merged over CEs to match merged multi-CE queries; cross-adduct matches with penalty; prefer QTOF/timsTOF refs | ranking C1 | 1 d | C1 MRR +0.02 |
| 005 | Class-2 ranker v1, no training: analog search → top-k neighbor structures → score pool candidates by similarity-weighted max Tanimoto + in_coconut prior + formula plausibility (fragment sub-formula explainability) | ranking C2 | 1 d | C2 MRR ≥ 2× exp_003's C2 |
| 006 | Learned spectrum→fingerprint encoder (repurpose our transformer: 4096-bit Morgan head and/or contrastive); score = Tanimoto(pred, cand) | ranking C2 | 2 d, Kaggle GPU | Beats 005 on C2 by ≥ 0.02, else keep only as a feature |
| 007 | LightGBM LambdaRank over the union of candidates (features: best cosine/entropy, #spectra matched, same-adduct, ref instrument, ppm error, in_coconut, in_train, FP score, neighbor-Tanimoto, formula score, #isomers) | ranking all | 1 d | Est. LB +0.02 over best heuristic |
| 008 | Slot allocation: confidence-based interleaving of library vs pool candidates | ranking all | 0.5 d | Est. LB +0.005 |
| 009 | Class-3 lite: biotransformation edits of top neighbors (±O, ±CH2, ±H2O, +hexose, +pentose, +malonyl, sulfate) via RDKit reaction SMARTS, keep formula-consistent ones, rank with 006 | recall C3 | 1.5 d | C3 MRR > 0.02 |
| 010 | Ensemble / tune / harden final notebook (offline wheels, runtime, fallbacks) | all | 1 d | Final submission valid under 9 h |

## 6. Decision gates
- After **003**: C1 MRR < 0.6 → stay on retrieval (cleaning, reference side) before touching class 2.
- After **005**: C2 Recall@cand < 0.8 → fix formula/pool coverage before learned ranking.
- **006** proceeds as a primary scorer only if it clears its criterion; otherwise it becomes a 007 feature.
- **No SMILES generation before 009** (ADR-001).
- Refit class weights after each LB submission. If fitted C3 share > 0.3, revisit ADR-001.

## 7. Two-person split
- **Track A — retrieval** (002, 003, 004, 008)
- **Track B — ranking/learning** (005, 006, 007, 009)
- 001 first, one owner, both review. Track B can start 006 training in parallel from day 2 (independent of retrieval code).

## 8. LB submission policy
Submit at 002, 003, 005, 007 and final. Nothing else. Every LB score gets logged next to its local est. LB — if they stop moving together, the harness is wrong and fixing it comes first.
