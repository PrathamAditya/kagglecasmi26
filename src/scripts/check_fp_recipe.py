"""Step 2a check: does our fp_recipe reproduce the pool's stored fingerprints exactly?

Run:  uv run src/scripts/check_fp_recipe.py artifacts/v17_builder/pool_712k_fixedfp.npz
(any pool file made by fix_pool_fps.py works; it has 'smiles' and 'fp'.)
Expected: 'mismatching rows: 0' for the random sample.
"""
import sys

import numpy as np

from casmi.fpnet import fp_recipe

z = np.load(sys.argv[1], allow_pickle=True)
smiles, fps = z["smiles"], z["fp"]
print("pool rows", len(smiles), "| stored fp shape", fps.shape, fps.dtype)

rng = np.random.default_rng(0)
idx = rng.choice(len(smiles), size=3000, replace=False)
fp_recipe.init()
bad = fail = 0
for i in idx:
    f = fp_recipe.smiles_to_packed(str(smiles[i]))
    if f is None:
        fail += 1
    elif not np.array_equal(f, fps[i]):
        bad += 1
print(f"sampled {len(idx)} | parse failures: {fail} | mismatching rows: {bad}")