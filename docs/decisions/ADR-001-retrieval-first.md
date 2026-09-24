# ADR-001 — Retrieval + learned ranking first; de novo generation last
- Status: proposed (confirm with teammate)
- Date: 2026-09-24

## Context
- The metric only rewards an exact 2D match (InChIKey14) within 25 guesses.
- Class 1 (public spectra exist) and class 2 (in PubChem/COCONUT) are reachable by retrieval over train + pool.
- On public benchmarks (MassSpecGym) exact-match top-1 for de novo generation is near zero.
- exp_000 at 0.127 vs leader 0.417 — the gap is plausibly explained by retrieval bugs and random class-2 ranking (plan §2).

## Options
| option | pros | cons |
|---|---|---|
| A. Transformer spectrum→SMILES | covers class 3 in principle | near-zero exact match; expensive; no help for class 1/2 |
| B. Retrieval + heuristic ranking | cheap, fixes biggest losses | ceiling on class 2 |
| C. B + learned spectrum encoder as a scorer + LTR reranker | reuses our transformer; attacks class 2 | needs clean leave-structure-out training data |

## Decision
C. The transformer is repurposed as a spectrum encoder (fingerprint head / contrastive), not a SMILES decoder.

## Consequences
We stop investing in the SMILES-decoding baseline. Class 3 is handled later by analog edits (exp_009).

## Revisit if
Fitted class-3 share from LB calibration > 0.3, or C1+C2 plateau with headroom only in C3.
