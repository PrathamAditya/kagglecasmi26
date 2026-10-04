import sys, glob, time
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

W1 = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
NPW = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0
K, TOPK, ROOT = 5, 25, "artifacts/ranker_groups"
GBM = dict(max_depth=10, max_iter=500, learning_rate=0.03, min_samples_leaf=80, l2_regularization=1.0)

files = sorted(glob.glob(ROOT + "/**/part_*.npz", recursive=True))
print("parts:", len(files))
Xl, yl, recs, n0 = [], [], [], 0
for f in files:
    z = np.load(f, allow_pickle=True)
    X, y, g = z["X"], z["y"], z["g"]
    mt = pd.DataFrame(z["meta"]).set_index("gid")
    b = np.flatnonzero(np.diff(g)) + 1
    for a, e in zip(np.r_[0, b], np.r_[b, len(g)]):
        m = mt.loc[g[a]]
        recs.append(dict(ik=str(m["ik"]), is_np=int(m["is_np"]), reg=int(m["reg"]), n=int(e - a), s=n0 + a, e=n0 + e))
    Xl.append(X); yl.append(y); n0 += len(g)
X = np.concatenate(Xl); y = np.concatenate(yl)
G = pd.DataFrame(recs)
print("groups:", len(G), "| rows:", len(X), "| positives:", int(y.sum()), "| NP molecules:", G[G.is_np == 1].ik.nunique())

fs = pd.read_csv(glob.glob(ROOT + "/**/frozen_summary.csv", recursive=True)[0])
assert len(fs) == len(G), (len(fs), len(G))
assert (fs.reg.values == G.reg.values).all() and (fs.is_np.values == G.is_np.values).all() and (fs.n_cand.values == G.n.values).all(), "frozen_summary rows are not aligned with the parts"
rank_fro = np.rint(1.0 / fs.rr.values).astype(int)

iks = np.array(sorted(G.ik.unique()))
perm = np.random.default_rng(0).permutation(len(iks))
G["fold"] = G.ik.map(dict(zip(iks[perm], np.arange(len(iks)) % K)))

gw = np.where(G.reg.values == 1, W1, 1.0 - W1) * np.where(G.is_np.values == 1, NPW, 1.0)
gw = gw / G.groupby(["ik", "reg"])["n"].transform("size").values
sw = np.repeat(gw, G.n.values)
fold_row = np.repeat(G.fold.values, G.n.values)
S, E, F = G["s"].to_numpy(), G["e"].to_numpy(), G["fold"].to_numpy()

rank_new = np.zeros(len(G), int)
for k in range(K):
    t0 = time.time()
    tr = fold_row != k
    m = HistGradientBoostingClassifier(random_state=0, **GBM).fit(X[tr], y[tr], sample_weight=sw[tr])
    for gi in np.flatnonzero(F == k):
        p = m.predict_proba(X[S[gi]:E[gi]])[:, 1]
        t = int(np.flatnonzero(y[S[gi]:E[gi]])[0])
        rank_new[gi] = int((p >= p[t]).sum())
    print("fold", k + 1, "of", K, "| iterations", m.n_iter_, "| secs", round(time.time() - t0), flush=True)

def rr(r):
    return np.where(r <= TOPK, 1.0 / r, 0.0)

d = pd.DataFrame({"ik": G.ik, "reg": G.reg, "is_np": G.is_np, "frozen": rr(rank_fro), "new": rr(rank_new),
                  "top1_frozen": rank_fro == 1, "top1_new": rank_new == 1})
print("\nW1 =", W1, "| NPW =", NPW)
print(d.groupby(["reg", "is_np"]).agg(groups=("new", "size"), mrr_frozen=("frozen", "mean"), mrr_new=("new", "mean"),
                                        top1_frozen=("top1_frozen", "mean"), top1_new=("top1_new", "mean")).round(4))
d["diff"] = d["new"] - d["frozen"]
rng = np.random.default_rng(0)
for name, sel in [("NP removed", (d.is_np == 1) & (d.reg == 0)), ("NP kept", (d.is_np == 1) & (d.reg == 1)),
                  ("non-NP removed", (d.is_np == 0) & (d.reg == 0)), ("non-NP kept", (d.is_np == 0) & (d.reg == 1))]:
    per = d[sel].groupby("ik")["diff"].mean().to_numpy()
    bs = per[rng.integers(0, len(per), (2000, len(per)))].mean(1)
    print("%s: molecules %d | new - frozen = %+.4f  95%% CI [%+.4f, %+.4f]" % (name, len(per), per.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)))
d.to_csv(ROOT + "/cv_W%s_NPW%s.csv" % (W1, NPW), index=False)
