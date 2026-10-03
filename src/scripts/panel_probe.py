import glob, os
import numpy as np

def describe(path):
    print("==", path, os.path.getsize(path), "bytes")
    z = np.load(path, allow_pickle=True)
    for k in z.files:
        a = z[k]
        print(" ", k, a.shape, a.dtype)
        if a.dtype.kind in "OUS" and a.shape and a.shape[0] > 0:
            print("     sample:", a[:3])

describe("artifacts/v17_builder/pool_712k_fixedfp.npz")
describe("artifacts/fpnet/fp_targets.npz")

print("== artifacts/fpnet_data")
for p in sorted(glob.glob("artifacts/fpnet_data/*")):
    print(" ", p, os.path.getsize(p) if os.path.isfile(p) else "dir")

for p in sorted(glob.glob("artifacts/fpnet_data/*meta*")):
    if p.endswith(".parquet"):
        import pandas as pd
        m = pd.read_parquet(p)
        print("== meta", p, len(m))
        print(m.dtypes)
        print(m.head(3).T)
        for c in ("is_val", "is_holdout"):
            if c in m:
                print(m[c].value_counts())
    elif p.endswith(".npz"):
        describe(p)
    else:
        print("== meta (unhandled type)", p)

import pyarrow.parquet as pq
found = False
for p in ("/home/amoeba/Desktop/enveda-CASMI26-molecule-id-mass-spectra/train.parquet", "train.parquet", "../train.parquet", "../data/train.parquet"):
    if os.path.exists(p):
        print("== train.parquet schema at", p)
        print(pq.read_schema(p).names)
        found = True
        break
if not found:
    print("train.parquet not found at the guessed paths")
