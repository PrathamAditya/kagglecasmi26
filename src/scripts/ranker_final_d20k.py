import glob, os, time
import numpy as np, pandas as pd, joblib, sklearn
from sklearn.ensemble import HistGradientBoostingClassifier

NPW, PRIORS, SEEDS = 4.0, (0.30, 0.60), (0, 1, 2, 3)
GBM = dict(max_depth=10, max_iter=500, learning_rate=0.03, min_samples_leaf=80, l2_regularization=1.0)
ROOT, OUT = "artifacts/ranker_groups", "artifacts/ranker_d20k/rankers_d20k_v1.joblib"
os.makedirs("artifacts/ranker_d20k", exist_ok=True)

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
print("groups:", len(G), "| rows:", len(X), "| positives:", int(y.sum()))
cnt = G.groupby(["ik", "reg"])["n"].transform("size").values

models = []
for w1 in PRIORS:
    gw = np.where(G.reg.values == 1, w1, 1.0 - w1) * np.where(G.is_np.values == 1, NPW, 1.0) / cnt
    sw = np.repeat(gw, G.n.values)
    for seed in SEEDS:
        t0 = time.time()
        m = HistGradientBoostingClassifier(random_state=seed, **GBM).fit(X, y, sample_weight=sw)
        models.append(m)
        print("prior", w1, "seed", seed, "| iterations", m.n_iter_, "| secs", round(time.time() - t0), flush=True)

joblib.dump(models, OUT)
chk = joblib.load(OUT)
print("saved", OUT, os.path.getsize(OUT), "bytes | models", len(chk), "| features", {m.n_features_in_ for m in chk}, "| sklearn", sklearn.__version__)
rr = []
for gi in np.flatnonzero((G.is_np.values == 1) & (G.reg.values == 0))[:200]:
    s, e = int(G.s.iloc[gi]), int(G.e.iloc[gi])
    p = np.mean([m.predict_proba(X[s:e])[:, 1] for m in chk], axis=0)
    rr.append(1.0 / (p >= p[int(np.flatnonzero(y[s:e])[0])]).sum())
print("in-sample NP-removed MRR, first 200 groups (training data, only a sanity check):", round(float(np.mean(rr)), 4))
