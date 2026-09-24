# Experiment log (index)

Status: planned · running · done · abandoned. Verdict: keep · discard · inconclusive.
Full records in `exp_NNN_slug/card.md`. Numbers: C1/C2/C3 = local MRR@25 per regime (ADR-002).

| exp | status | owner | parent | hypothesis (short) | C1 | C2 | C3 | est. LB | LB | Δ vs parent | verdict | card |
|-----|--------|-------|--------|--------------------|----|----|----|---------|----|-------------|---------|------|
| 000 | done | Pratham | – | CPU cosine retrieval + mass-matched pool fallback | – | – | – | – | 0.127 | – | baseline | exp_000_baseline_retrieval/card.md |
| 001 | running | Pratham | 000 | Harness: measure recall vs ranking loss per class | | | | | | | | exp_001_val_harness/card.md |
| 002 | planned | | 001 | Adduct/neutral-mass/tolerance/dedup fixes raise recall | | | | | | | | |
| 003 | planned | | 002 | Full-index entropy search + cleaning raises C1 ranking | | | | | | | | |
