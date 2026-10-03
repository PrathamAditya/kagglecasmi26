import sys, os, time, math
sys.path.insert(0, "src")
from pathlib import Path
import numpy as np

def main():
    from casmi.chem import IK14Cache
    P = np.load("artifacts/panel/panel_v1.npz", allow_pickle=True)
    pool = np.load("artifacts/v17_builder/pool_712k_fixedfp.npz", allow_pickle=True)
    psmi = pool["smiles"]
    rows, offsets, truth_row = P["rows"], P["offsets"], P["truth_row"]
    is_np, val_ik = P["is_np"], P["val_ik14"].astype(str)

    uniq = np.unique(rows)
    smiles = [str(psmi[i]) for i in uniq]
    cache = IK14Cache(Path("artifacts/panel/metric_ik14_panel.parquet"))
    t0 = time.time()
    CH = 20000
    for s in range(0, len(smiles), CH):
        cache.get_many(smiles[s:s + CH])
        cache.save()
        print("canonicalised", min(s + CH, len(smiles)), "of", len(smiles), "| elapsed s", round(time.time() - t0), flush=True)
    keys = cache.get_many(smiles)
    keymap = dict(zip(uniq.tolist(), [keys[s] for s in smiles]))
    print("rows with failed metric key:", sum(v is None for v in keymap.values()))

    n = len(truth_row)
    correct = np.zeros(len(rows), bool)
    n_correct = np.zeros(n, np.int64)
    sizes = np.diff(offsets)
    truth_fail = 0
    truth_differs = np.zeros(n, bool)
    for v in range(n):
        a, b = offsets[v], offsets[v + 1]
        r = rows[a:b]
        tk = keymap[int(truth_row[v])]
        if tk is None:
            truth_fail += 1
            m = r == truth_row[v]
        else:
            truth_differs[v] = tk != val_ik[v]
            ks = np.array([keymap[int(x)] or "" for x in r])
            m = (ks == tk) | (r == truth_row[v])
        correct[a:b] = m
        n_correct[v] = m.sum()
    print("truths with failed metric key:", truth_fail)
    print("truth metric key != raw ik14: ALL", int(truth_differs.sum()), "of", n, "| NP", int(truth_differs[is_np].sum()), "of", int(is_np.sum()))

    twins = n_correct - 1
    for name, sel in (("ALL", np.ones(n, bool)), ("NP", is_np)):
        t = twins[sel]
        print(name, "panels with >=1 twin:", int((t > 0).sum()), "of", len(t), "| max twins", int(t.max()), "| total twins", int(t.sum()))

    d = sizes - n_correct
    for name, sel in (("ALL", np.ones(n, bool)), ("NP", is_np)):
        x = d[sel]
        print(name, "decoys per panel:", {lab: int(((x >= lo) & (x <= hi)).sum()) for lab, lo, hi in
              [("0", 0, 0), ("1", 1, 1), ("2-4", 2, 4), ("5-19", 5, 19), ("20-99", 20, 99), ("100+", 100, 10**9)]})

    chance = np.zeros(n)
    for v in range(n):
        N, c = int(sizes[v]), int(n_correct[v])
        chance[v] = sum((1.0 / k) * math.comb(N - k, c - 1) / math.comb(N, c) for k in range(1, N - c + 2))
    for name, sel in (("ALL", np.ones(n, bool)), ("NP", is_np)):
        s2 = sel & (d > 0)
        print(name, "random-ranking MRR: all panels", round(float(chance[sel].mean()), 4),
              "| panels with >=1 decoy", round(float(chance[s2].mean()), 4), "(n =", int(s2.sum()), ")")

    out = "artifacts/panel/panel_v2.npz"
    np.savez_compressed(out, **{k: P[k] for k in P.files}, correct_mask=correct, n_correct=n_correct)
    print("saved", out, os.path.getsize(out), "bytes")

if __name__ == "__main__":
    main()
