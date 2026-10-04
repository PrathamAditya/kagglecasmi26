## 2026-10-04 · Asset: casmi26-assets (RDKit wheels + local scorer)

**Purpose:** Offline, version-pinned RDKit and a local scorer that replicates the official CASMI26 matching. Submission notebooks run with internet off, and the scorer pins RDKit 2026.03.3.

**Source notebook:** `prathamaditya/rdkit-2026-03-3` (built with Internet On, Kaggle image Python 3.13)
**Dataset:** `casmi26-assets` (Private)

### Contents

| Path                                      | Notes                                   |
| ----------------------------------------- | --------------------------------------- |
| `rdkit_wheels/rdkit-2026.3.3-cp313-*.whl` | 37.19 MB                                |
| `rdkit_wheels/numpy-2.5.3-cp313-*.whl`    | 16.71 MB                                |
| `rdkit_wheels/pillow-12.3.0-cp313-*.whl`  | 6.94 MB                                 |
| `casmi_scorer.py`                         | `key14(smi)` and `mrr25(preds, truths)` |

### Matching logic (mirrors official scorer)

1. Parse SMILES with RDKit.
2. Canonicalize the tautomer (`TautomerEnumerator.Canonicalize`).
3. Compute the InChIKey and keep block 1 (**key14**, the first 14 chars = 2D connectivity).
4. A guess is correct if its key14 equals the answer's key14. Stereochemistry is ignored.
5. MRR@25: score = 1/rank of the first correct guess, 0 if none in the top 25.

`key14` is cached with `lru_cache`, because tautomer canonicalization is slow.

### Usage (internet off)

    !pip install --no-index --find-links=/kaggle/input/casmi26-assets/rdkit_wheels rdkit==2026.3.3 -q
    import sys; sys.path.insert(0, "/kaggle/input/casmi26-assets")
    from casmi_scorer import key14, mrr25

Confirm the mount path with `!ls /kaggle/input` the first time.

### Verification

- [x] Wheels downloaded (rdkit, numpy, pillow, all cp313)
- [ ] Dataset created from notebook output
- [ ] Fresh notebook, internet off: `rdkit.__version__ == 2026.03.3`
- [ ] Both glucose SMILES from the Overview return `WQZGKKKJIJFFOK`

### Notes

- Wheels are tied to Python 3.13. If Kaggle changes its image Python version, rebuild the dataset.
- Use this same RDKit version for candidate canonicalization and dedup (by key14) in the pipeline, so local scores don't drift from the leaderboard.
- Planned: add fingerprints, retrieval indexes and checkpoints to this dataset as new versions.
