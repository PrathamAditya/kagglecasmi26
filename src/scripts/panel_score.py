import sys, math
import numpy as np

P = np.load("artifacts/panel/panel_v2.npz", allow_pickle=True)
pool = np.load("artifacts/v17_builder/pool_712k_fixedfp.npz", allow_pickle=True)
fp = pool["fp"]
NB = int(pool["nbits"])
rows, offsets, correct = P["rows"], P["offsets"], P["correct_mask"]
n_correct, is_np = P["n_correct"], P["is_np"]
sizes = np.diff(offsets)

names, lbs, L = [], {}, {}
for arg in sys.argv[1:]:
    nm, _, lb = arg.partition(":")
    names.append(nm)
    lbs[nm] = float(lb) if lb else None
    z = np.load("artifacts/panel/logits_" + nm + ".npz")
    L[nm] = (z["ml"], z["cnt"])
n = min(len(L[nm][0]) for nm in names)
print("models", names, "| panels scored", n)

rr = {nm: np.full(n, np.nan) for nm in names}
t1 = {nm: np.full(n, np.nan) for nm in names}
tie = {nm: np.zeros(n, bool) for nm in names}
for v in range(n):
    a, b = offsets[v], offsets[v + 1]
    r, cm = rows[a:b], correct[a:b]
    B = np.unpackbits(fp[r], axis=1)[:, :NB].astype(np.float32)
    for nm in names:
        ml, cnt = L[nm]
        if cnt[v] == 0:
            continue
        s = B @ ml[v]
        cs = s[cm].max()
        wrong = s[~cm]
        rank = 1 + int((wrong >= cs).sum())
        tie[nm][v] = bool((wrong == cs).any())
        rr[nm][v] = 1.0 / rank
        t1[nm][v] = float(rank == 1)

d = (sizes - n_correct)[:n]
chance = np.zeros(n)
for v in range(n):
    N, c = int(sizes[v]), int(n_correct[v])
    chance[v] = sum((1.0 / k) * math.comb(N - k, c - 1) / math.comb(N, c) for k in range(1, N - c + 2))

valid = np.ones(n, bool)
for nm in names:
    valid &= ~np.isnan(rr[nm])
print("panels valid for all models:", int(valid.sum()))
npm = is_np[:n]
groups = {"ALL": valid, "ALL_with_decoys": valid & (d > 0), "NP": valid & npm, "NP_with_decoys": valid & npm & (d > 0)}

for g, sel in groups.items():
    print("\n==", g, "| panels", int(sel.sum()), "| random-ranking MRR", round(float(chance[sel].mean()), 4))
    print("  model            LB      MRR     top1    tie_panels")
    for nm in names:
        print("  %-16s %-7s %.4f  %.4f  %d" % (nm, lbs[nm], rr[nm][sel].mean(), t1[nm][sel].mean(), int(tie[nm][sel].sum())))

sel = groups["ALL_with_decoys"]
known = sorted([nm for nm in names if lbs[nm] is not None], key=lambda x: lbs[x])
rng = np.random.default_rng(0)
ids = np.flatnonzero(sel)
bi = rng.integers(0, len(ids), (2000, len(ids)))
print("\n== paired bootstrap on ALL_with_decoys (MRR delta, higher-LB model minus lower-LB model)")
for i in range(len(known) - 1):
    lo, hi = known[i], known[i + 1]
    x = rr[hi][ids] - rr[lo][ids]
    bs = x[bi].mean(1)
    print("  %s (%.3f) vs %s (%.3f): delta %+.4f  95%% CI [%+.4f, %+.4f]" % (hi, lbs[hi], lo, lbs[lo], x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
conc = tot = 0
for i in range(len(known)):
    for j in range(i + 1, len(known)):
        tot += 1
        conc += rr[known[j]][ids].mean() > rr[known[i]][ids].mean()
print("\nconcordant pairs with leaderboard order:", conc, "of", tot)
