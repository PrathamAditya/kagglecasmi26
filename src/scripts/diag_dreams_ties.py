import sys
import numpy as np
from casmi.fpnet.data import SpecData

D = SpecData(sys.argv[1])
lens = np.diff(D.offsets)
for i in (9552, 11760, 17072, 32502, 46955):
    s, e = D.offsets[i], D.offsets[i + 1]
    it = np.sort(D.it[s:e].astype(np.float64) ** 2)[::-1]
    print(f"id {i}: {len(it)} peaks | rank-100 intensity {it[99]:.6g} | rank-101 {it[100]:.6g} | tie at cutoff: {it[99] == it[100]}")
rng = np.random.default_rng(0)
big = np.where(lens > 100)[0]
pick = rng.choice(big, size=min(2000, len(big)), replace=False)
tie = 0
for i in pick:
    it = np.sort(D.it[D.offsets[i]:D.offsets[i + 1]])[::-1]
    tie += int(it[99] == it[100])
print(f"random spectra with >100 peaks: {len(pick)}, tie at the 100/101 cutoff: {tie / len(pick):.3f}")
