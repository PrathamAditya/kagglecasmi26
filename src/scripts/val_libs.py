import numpy as np
m = np.load("artifacts/fpnet_data/meta.npz", allow_pickle=True)
lib, mol, v, h = m["lib"], m["mol"], m["is_val"], m["is_holdout"]
names = [str(x) for x in m["lib_names"]]
print("lib | val spectra | val molecules | train spectra | train molecules")
for i, n in enumerate(names):
    a = (lib == i) & v
    b = (lib == i) & ~v
    print(n, int(a.sum()), len(np.unique(mol[a])), int(b.sum()), len(np.unique(mol[b])))
nh = v & ~h
print("non-holdout val: spectra", int(nh.sum()), "molecules", len(np.unique(mol[nh])))
c = np.unique(mol[nh], return_counts=True)[1]
print("spectra per non-holdout val molecule: median", float(np.median(c)), "| p10", float(np.percentile(c, 10)), "| p90", float(np.percentile(c, 90)), "| molecules with >=3 spectra", int((c >= 3).sum()))
