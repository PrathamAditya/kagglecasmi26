# ADR-002 — Validation protocol
- Status: proposed
- Date: 2026-09-24

## Context
- The visible `test.parquet` is copied from train → any score on it is leaked (identical spectra). Never tune on it.
- Public LB = partial test, limited submissions → too noisy/slow to iterate on.
- The test is timsTOF natural products; `enveda-np-examples` (250 molecules, 1,151 spectra) is the only same-instrument, same-pipeline NP set.

## Decision
- Query set: enveda-np-examples, grouped by molecule (ik14). Split frozen in `data/val_ik14.txt`, committed.
- Three regimes per run:
  - **C1**: reference = train minus enveda-np-examples rows.
  - **C2**: reference = train minus all rows with the query ik14; pool keeps it.
  - **C3**: C2 + drop the query ik14 from the pool.
- Report MRR, Recall@cand, Recall@25 per regime + est. LB + runtime.
- Anything trained (encoder, reranker) must exclude val ik14 from its training data. Reranker training queries: leave-structure-out from NP-rich libs (riken, gnps, msdial, mona, massbank), QTOF preferred.
- Every run dumps its 20 worst molecules for failure analysis.

## Known bias
enveda-np compounds were chosen to be common → C1 here is optimistic vs the real test. Track deltas, trust the LB calibration for absolute level.

## Revisit if
Local est. LB and public LB move in opposite directions on two consecutive submissions.
