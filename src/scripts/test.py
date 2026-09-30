# import pandas as pd

# a = pd.read_csv('data/test/local_rank_refill_0334.csv').set_index('inchikey14')
# b = pd.read_csv('data/test/local_rank_refill_0320.csv').set_index('inchikey14')
# c = pd.read_csv('data/test/local_rank_refill_0315.csv').set_index('inchikey14')

# common = a.index.intersection(b.index).intersection(c.index)
# for k in common:
#     ranks = [a.loc[k, 'rank'], b.loc[k, 'rank'], c.loc[k, 'rank']]
#     if 1 in ranks and not all(r == 1 for r in ranks):
#         print(k, '0334:', a.loc[k, 'rank'], a.loc[k, 'top1_smiles'],
#                   '| 0320:', b.loc[k, 'rank'], b.loc[k, 'top1_smiles'],
#                   '| 0315:', c.loc[k, 'rank'], c.loc[k, 'top1_smiles'])

# import numpy as np
# z = np.load("artifacts/v17_builder/pool_712k_fixedfp.npz", allow_pickle=True)
# print({k: (z[k].shape, z[k].dtype) for k in z.files})

# import numpy as np, pandas as pd
# from collections import defaultdict

# o = np.load("artifacts/v17_builder/pool_712k_precomputed.npz", allow_pickle=True)
# f = np.load("artifacts/v17_builder/pool_712k_fixedfp.npz", allow_pickle=True)
# keys = o["keys"]
# assert (keys == f["keys"]).all(), "pools are in different order"

# changed = (o["fp"] != f["fp"]).any(axis=1)
# print("example keys:", keys[:3])
# print("rows:", len(keys), "| unique keys:", len(set(keys)))
# print("rows with changed fp:", changed.sum())

# idx = defaultdict(list)
# for i, k in enumerate(keys):
#     idx[k].append(i)

# truth = pd.read_parquet(
#     "/home/amoeba/Desktop/enveda-CASMI26-molecule-id-mass-spectra/train.parquet",
#     columns=["inchikey14", "ingest_lib"],
#     filters=[("ingest_lib", "==", "enveda-np-examples")],
# )["inchikey14"].unique()

# in_pool = [k for k in truth if k in idx]
# n_changed = sum(changed[idx[k]].any() for k in in_pool)
# n_multi = sum(len(idx[k]) > 1 for k in in_pool)
# print("local truths:", len(truth), "| found in pool:", len(in_pool))
# print("truths whose pool row has a CHANGED fp:", n_changed)
# print("truths with more than one pool row:", n_multi)


# import numpy as np, hashlib
# a = np.load('artifacts/coconut_candidates/fp_bits.npy')
# print(a.shape, a.dtype, hashlib.md5(a.tobytes()).hexdigest())

# import pandas as pd
# import numpy as np

# a = pd.read_csv("data/submission_files/submission_0334.csv").set_index("molecule_id")["smiles"].str.split(";")
# b = pd.read_csv("data/submission_files/submission_0315.csv").set_index("molecule_id")["smiles"].str.split(";")

# assert set(a.index) == set(b.index)
# mols = a.index

# rows = []
# for mol in mols:
#     la, lb = a[mol], b[mol]
#     pos_b = {s: i for i, s in enumerate(lb)}          # rank of each 334-candidate in 315's list
#     ranks_in_b = [pos_b.get(s, np.nan) for s in la]    # NaN = dropped out of top-25 entirely

#     first_diff = next((i for i, (x, y) in enumerate(zip(la, lb)) if x != y), len(la))
#     top1_same  = la[0] == lb[0]
#     top5_set   = set(la[:5]) == set(lb[:5])            # same 5, ignoring order
#     dropped    = sum(np.isnan(ranks_in_b))              # 334-candidates gone from 315's top-25
#     new_in_b   = len(set(lb) - set(la))                 # candidates in 315 not in 334 at all

#     # mean |rank shift| for candidates that survived in both lists
#     shifts = [abs(i - r) for i, r in enumerate(ranks_in_b) if not np.isnan(r)]
#     mean_shift = np.mean(shifts) if shifts else 0.0

#     rows.append(dict(molecule_id=mol, first_diff_rank=first_diff, top1_same=top1_same,
#                       top5_set_same=top5_set, n_dropped=dropped, n_new=new_in_b,
#                       mean_rank_shift=mean_shift, list_len_a=len(la), list_len_b=len(lb)))

# diff = pd.DataFrame(rows).set_index("molecule_id")

# print("=== Headline stats over", len(diff), "molecules ===")
# print(f"Identical lists (no change at all):     {(diff.first_diff_rank == diff.list_len_a).sum()}")
# print(f"Top-1 unchanged:                         {diff.top1_same.sum()} ({diff.top1_same.mean():.1%})")
# print(f"Top-5 set unchanged (order-independent): {diff.top5_set_same.sum()} ({diff.top5_set_same.mean():.1%})")
# print()
# print("First differing rank position (0-indexed), distribution:")
# print(diff.first_diff_rank.describe())
# print()
# print("Candidates dropped out of top-25 (334→315), per molecule:")
# print(diff.n_dropped.describe())
# print(f"Molecules losing >=1 candidate from top-25: {(diff.n_dropped > 0).sum()}")
# print()
# print("Mean |rank shift| among surviving candidates, per molecule:")
# print(diff.mean_rank_shift.describe())
# print()
# print("Top 10 most-disrupted molecules (by n_dropped, then mean_rank_shift):")
# print(diff.sort_values(["n_dropped", "mean_rank_shift"], ascending=False).head(10))


# for mol in ["m_8d75c7", "m_e5b5fa", "m_0cc9ea", "m_bcd97d", "m_c50497"]:
#     la, lb = a[mol], b[mol]
#     print(mol,
#           "| unique in A:", len(set(la)), "/", len(la),
#           "| unique in B:", len(set(lb)), "/", len(lb))

# import numpy as np
# f = np.load("artifacts/v17_builder/pool_712k_fixedfp.npz", allow_pickle=True)
# smis = f["smiles"]
# u, counts = np.unique(smis, return_counts=True)
# dupe_smis = u[counts > 1]
# print("distinct SMILES:", len(u), "/ total rows:", len(smis))
# print("SMILES strings shared by >1 pool row:", len(dupe_smis))
# print("worst offenders:", sorted(counts, reverse=True)[:10])

# import torch, numpy as np
# for p in ["artifacts/fp_models/fp_merged_m1.pt", "artifacts/fp_models/fp_single_s2.pt"]:
#     sd = torch.load(p, map_location="cpu")
#     sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
#     print(p)
#     for k, v in sd.items():
#         print(" ", k, tuple(v.shape) if hasattr(v, "shape") else type(v))
# print(np.load("artifacts/coconut_candidates/fp_bits.npy").shape)

import torch
for p in ["artifacts/fp_models/fp_merged_m1.pt", "artifacts/fp_models/fp_single_s2.pt"]:
    ck = torch.load(p, map_location="cpu")
    print(p, "step", ck["step"], "nbits", ck["nbits"], "d", ck["d"], "layers", ck["layers"])
    for k, v in ck["model"].items():
        print("  ", k, tuple(v.shape))