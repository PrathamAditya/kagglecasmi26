## exp_023 — PubChem class-2 candidate pool (2026-10-03)

**Goal:** make PubChem-only class-2 test molecules reachable. Today they score 0 because they are absent from the 712k pool (train + COCONUT).

**Status:**

- Pools built and uploaded to Kaggle.
- LB submissions S1/S2 pending.
- Local harness run blocked (see Issues).
- Ranker retrain not started.

---

### Approach

- **Annotation-based selection, not an NP-likeness sweep.**
  - Tier 1 = CIDs with organism links (PubChem `taxonomy2cid`: LOTUS, NPASS, KNApSAcK, NPAtlas, FooDB, HMDB…).
  - Tier 2 = PubChemLite for Exposomics (drugs, agro, food, metabolites; excluded by COCONUT).
- CIDs mapped to preferred CID, then standardised with a recipe calibrated against train, then filtered after standardisation.
- Filters: mass 150–1200 Da; elements CHNOPS + F, Cl, Br, I, Se; no net charge, radicals or isotopes.
- Dedup vs pool on the **scorer key** (tautomer-canonical IK14) plus plain IK14, then collapse within PubChem by scorer key.
- Fingerprints: same V17 recipe and `fp_bits.npy` as the pool, computed from the stored SMILES.
- Code: `pubchem_c2/` (steps 00–10). Outputs: `casmi26/artifacts/pubchem_c2/`.
- Environment: `casmi` venv (RDKit 2026.03.3) via `~/casmi_env.sh`, run from `casmi26-project/` with `CASMI_ROOT=casmi26`.

### Inputs (downloaded 2026-10-03)

| File                                  | Size     | Note                                                     |
| ------------------------------------- | -------- | -------------------------------------------------------- |
| `Target/taxonomy2cid.tsv.gz`          | 56 MB    | Renamed by PubChem from `taxonomy2cid.gz` (old URL 404s) |
| `Compound/Extras/CID-SMILES.gz`       | 1,487 MB |                                                          |
| `Compound/Extras/CID-Preferred.gz`    | 13 MB    |                                                          |
| `Compound/Extras/CID-PMID.gz`         | 326 MB   | `n_pmid` feature                                         |
| `PubChemLite_exposomics_20260925.csv` | 200 MB   | Zenodo record 22953038, CC BY 4.0, md5 ok                |
| `Target/pathway2cid.tsv.gz`           | —        | Available, **not used** (enumerated lipids → dilution)   |

### Checks

- **00 parity (3,000 pool rows):**
  - Fingerprints 3000/3000 bit-identical.
  - Mass = CalcExactMolWt (max diff 6e-5, float32).
  - Pool keys = plain IK14; scorer key agrees on only **93.3%**, so dedup must use the scorer key.
  - Charged rows 0.07%.
- **04 standardiser calibration (5,000 train ∩ PubChemLite):**

  | Recipe                       | fp identical |
  | ---------------------------- | ------------ |
  | raw                          | 0.9714       |
  | lf_unch                      | 0.9714       |
  | **cleanup_lf_unch (chosen)** | **0.9744**   |
  | cleanup_lf_unch_taut         | 0.9336       |

  Conclusion: train `normalized_smiles` are **not** tautomer-canonical. Fingerprint the non-tautomerised form; use the tautomer key only for dedup.

- **05 smoke (20k):** 644 mol/s. Outcomes: ok 73.3%, mass 19.6%, elements 3.7%, charged 3.4%, parse ~0. Full-run reject table not recorded.
- **07 fingerprints:** 312,208 rows. Guard 1 (pool re-fingerprint) and guard 2 (new rows from stored SMILES) both 1000/1000 bit-identical.

### Result: new structures (06)

|                            | Rows                                                                                            |
| -------------------------- | ----------------------------------------------------------------------------------------------- |
| Removed as already in pool | 308,679 (scorer key) + 652 (plain key)                                                          |
| **New structures**         | **312,208 (+43.8% pool)**                                                                       |
| Tier 1                     | 12,539 (HMDB 11,420 · KNApSAcK 380 · NPASS 137 · LOTUS 131 · NPAtlas 16 · FooDB 13 · other 566) |
| Tier 2 only                | 299,669                                                                                         |

Known NPs are already covered by COCONUT, so Tier 1 adds few rows. The addition is about 96% exposome/synthetic.

### Result: class-2 reachability proxy (09)

Proxy: NP-library train structures, assuming they had no spectra. Is each one in COCONUT, or only reachable via PubChem? `--ext` = `COCONUT4MetFrag_april.csv` (423,706 rows, 394,632 unique IK14).

| Group              | n       | COCONUT | +Tier 1 only | +Tier 2 only | Total reachable | **Newly reachable** |
| ------------------ | ------- | ------- | ------------ | ------------ | --------------- | ------------------- |
| NP libs            | 64,674  | 37.3%   | +4.1         | +8.6         | 50.0%           | **12.7%**           |
| GNPS               | 45,750  | 37.7%   | +2.8         | +9.7         | 50.2%           | 12.5%               |
| RIKEN              | 15,892  | 77.5%   | +2.8         | +5.4         | 85.7%           | 8.2%                |
| enveda-np-examples | 250     | 98.8%   | +1.2         | 0.0          | 100%            | 1.2%                |
| Synthetic libs     | 228,412 | 4.0%    | +0.2         | +2.0         | 6.2%            | 2.2%                |

Per added row, Tier 1 is about 10× more efficient than Tier 2 (+4.1 points from 12.5k rows vs +8.6 points from 300k rows).

_First 09 run was invalid:_ `ext` was taken from the pool, whose COCONUT rows were already deduped against train, so `ext%` was 0 and the gain was overstated. Fixed by loading raw COCONUT via `--ext`.

### Result: Tier-2 filter scan (10)

| Filter                | Rows added         | Share of Tier-2 gain kept | Gain per 10k rows |
| --------------------- | ------------------ | ------------------------- | ----------------- |
| ALL Tier 2            | 299,669            | 100%                      | 198               |
| **AnnoTypeCount ≥ 2** | **41,618 (13.9%)** | **60.3%**                 | **860**           |
| AnnoTypeCount ≥ 3     | 16,318             | 39.7%                     | 1,444             |
| drugmedicinfo > 0     | 4,844              | 33.3%                     | 4,081             |
| NP-likeness > 0       | 63,821             | 19.4%                     | 181               |

- **NP-likeness is useless as a filter:** gain tracks rows at every threshold.
- **Annotation filters are strongly enriched.** This is flattered by the proxy, since train structures are well-studied and real class-2 molecules are less so.

### Pool variants (08)

| Variant                        | Base pool | File                                       | Rows      | Median candidates / window | Windows > 256 |
| ------------------------------ | --------- | ------------------------------------------ | --------- | -------------------------- | ------------- |
| Baseline                       | fixed     | `pool_712k_fixedfp.npz`                    | 712,199   | 62                         | 9.5%          |
| A (Tier 1)                     | fixed     | `pool_724k_pubchem_t1.npz`                 | 724,738   | 62                         | 9.5%          |
| **B (Tier 1 + Tier 2 anno≥2)** | fixed     | `pool_766k_pubchem_t12_anno2.npz`          | 766,356   | 64                         | 9.7%          |
| C (all)                        | fixed     | `pool_1024k_pubchem_t12.npz`               | 1,024,407 | 80                         | 11.7%         |
| A                              | original  | `pool_724k_pubchem_t1_origpool.npz`        | 724,738   | —                          | —             |
| B                              | original  | `pool_766k_pubchem_t12_anno2_origpool.npz` | 766,356   | —                          | —             |

- **B is the front-runner:** about 73% of the reachability gain (≈9.3 of 12.7 points) for +3% crowding.
- Rows are re-sorted by mass. `base_idx` maps to original pool rows, and `src` is 0/1/2 (pool / Tier 1 / Tier 2).

### Kaggle

- Dataset: `prathamaditya/casmi26-pubchem-pool` (private).
  - Pools: A/B/C (fixed) plus A/B `_origpool`.
  - `pubchem_new_meta.parquet` (features, join on `cid`).
  - `pool_tkeys.npy` (scorer keys of the original 712k rows).
  - Provenance JSON/CSV.
- Licence: PubChem = public domain; PubChemLite = CC BY 4.0 (Schymanski et al., J Cheminform 2021).

### LB tests (frozen rankers; only change = pool)

| ID  | Notebook                      | Pool                                       | Baseline | LB    |
| --- | ----------------------------- | ------------------------------------------ | -------- | ----- |
| S1  | lb-315 (fixed pool)           | `pool_766k_pubchem_t12_anno2.npz`          | 0.315    | _TBD_ |
| S2  | lb-334-refill (original pool) | `pool_766k_pubchem_t12_anno2_origpool.npz` | 0.334    | _TBD_ |

How to read the results:

- Within ±0.02 of baseline → dilution safe.
- Up in both → PubChem helps even with the frozen ranker.
- Down in both → inspect the new decoys.
- Public LB resolution is about ±0.03.

### Issues

- Taxonomy URL renamed (404). Fixed in `config.py` / `patch_pw.py`.
- Path/env confusion (`CASMI_ROOT`, `casmi_env.sh`, stray `data/` and `artifacts/` in the parent folder). Resolved: run PubChem scripts from `casmi26-project/` with the `casmi` venv, and repo tools from `casmi26/` with `uv run`.
- RDKit tautomer-limit warnings in the pool-key job. Harmless (the scorer uses the same defaults); silenced in `common.py`.
- YAML error in the new config (unquoted `:` in the description). Fixed.
- **Open:** local harness `uv run python -m casmi.harness.evaluate --config configs/exp_023b_pubchem_poolB.yaml` fails with `NameError: load_neural_models` in `src/casmi/predictors/exp016_v27.py:707`.
  - Torch 2.14.0+cpu imports fine, and `exp016_v27.py` is git-clean.
  - The function is defined nowhere the predictor can see. Unrelated to the pool; exp_021 would fail the same way.

### Caveats

- `--ext` covers COCONUT April only (395k IK14). The pool's non-train part is 436k, so another source (ChEBI / LIPID MAPS?) is missing and the gain is somewhat overstated.
- The proxy uses train structures (better studied than true class 2). This inflates PubChem coverage and especially the annotation filters.
- The frozen ranker cannot use annotation features, so the LB tests measure dilution and incidental gain only.

### Decisions

- Do not filter by NP-likeness (keep it as a feature only).
- B is the main candidate; A is the safe fallback; C is a reference only.
- Ranker retrain: attach annotation features **by structure key to every pool row** (train and COCONUT included). Never use `src` as a feature. Use out-of-fold spectrum-model scores.

### Next

1. Record S1/S2 LB.
2. Fix the harness `load_neural_models` NameError (`git log -S load_neural_models`), then run the exp_023b local dilution check (paired bootstrap via `casmi.harness.compare`).
3. Write the key-level feature table (all pool rows).
4. Retrain the ranker on pool B; compare A/B/C on structure-held-out validation.

### Open questions

- Own fine-tuned DreaMS fingerprint model (`dreams_fp_20k.pt`, drop-in for FPNet): what is `NBITS` (V17 6,930-bit mask?), and was it trained on all train structures or with held-out folds? This decides the step-07 extension and leakage handling.
- `ranker_cv.py`:
  - 2 positives per group in all 819 groups.
  - `M constant within group: False`.
  - The file was modified while being run (git `M`).
  - Verify before trusting LambdaRank (M0 0.757 / M1 0.628) vs GBDT (M0 0.741 / M1 0.645). These are statistically tied anyway.
