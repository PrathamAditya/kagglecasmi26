# Experiment log (index)

Status: planned · running · done · abandoned. Verdict: keep · discard · inconclusive.
Full records in `exp_NNN_slug/card.md`. Numbers: C1/C2/C3 = local MRR@25 per regime (ADR-002).

| exp | status | owner | parent | hypothesis (short) | C1 | C2 | C3 | est. LB | LB | Δ vs parent | verdict | card |
|-----|--------|-------|--------|--------------------|----|----|----|---------|----|-------------|---------|------|
| 000 | done | Pratham | – | CPU cosine retrieval + mass-matched pool fallback | – | – | – | – | 0.127 | – | baseline | exp_000_baseline_retrieval/card.md |
| 001 | running | Pratham | 000 | Harness: measure recall vs ranking loss per class | | | | | | | | exp_001_val_harness/card.md |
| 002 | planned | | 001 | Adduct/neutral-mass/tolerance/dedup fixes raise recall | | | | | | | | |
| 003 | planned | | 002 | Full-index entropy search + cleaning raises C1 ranking | | | | | | | | |

---
Entries below follow the CLAUDE.md format (one block per run).

### exp_016 (A1 sanity, n=20) — V27 port vs notebook run on identical inputs
- git: (this commit)  config: configs/exp_016.yaml  anchor: n/a (port check)
- change: V27 patched notebook ported to src/casmi/predictors/exp016_v27.py; notebook cells exec'd verbatim on the same regime-filtered library (src/casmi/harness/v27_notebook_ref.py)
- n: C1=20 C2=20 C3=20 (smoke only — no decisions)
- port vs notebook: C1 top-1 20/20, identical top-25 order 20/20; C2 top-1 20/20, identical order 20/20 (C3 not comparable: notebook has no pool-removal path)
- C1 MRR 0.867  R@cand 1.00  R@25 1.00  |  C2 MRR 0.468  R@cand 1.00  R@25 0.95  |  C3 MRR 0.000 (pool removal: 22 rows)
- ppm offset: +1.284 ppm (V27 calibrate_ppm_offset on 250 val queries, C1 library; n=37 pairs, label-free)
- runtime: ~3.5 s/molecule CPU (12 threads), prepare ~45 s/regime, peak RSS 14.7 GB   leak_hits: 0/0/0
- verdict: port PASS — proceed to full eval. NOTE: C2=0.47 is implausibly high vs LB 0.334; see handoff (FPNet/GBDT training-set leak suspected).
