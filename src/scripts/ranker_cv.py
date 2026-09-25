# scripts/ranker_cv.py  —  grouped 5-fold CV on rank_train.npz, MRR per regime
import numpy as np, lightgbm as lgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

z = np.load('artifacts/ranker_training_features/rank_train.npz')
X, Y, M, G = z['X'], z['Y'].astype(int), z['M'].astype(int), z['G']
SG = G * 2 + M    
# sanity: group structure
ug, gi = np.unique(G, return_inverse=True)
pos_per_g = np.bincount(gi, weights=Y)
print(f"groups={len(ug)}  size median={np.median(np.bincount(gi)):.0f}  "
      f"pos/group: {np.unique(pos_per_g, return_counts=True)}")
print("M constant within group:", all(len(np.unique(M[G == g])) == 1 for g in ug[:500]))

def mrr(scores, idx):
    out = {0: [], 1: []}
    for g in np.unique(SG[idx]):
        m = idx[SG[idx] == g]
        if Y[m].sum() == 0: continue
        order = m[np.argsort(-scores[np.searchsorted(idx, m)])]
        r = int(np.where(Y[order] == 1)[0][0]) + 1
        out[int(M[m[0]])].append(1 / r if r <= 25 else 0.0)
    return {k: (np.mean(v), len(v)) for k, v in out.items()}

gkf = GroupKFold(5)
res = {'v27_gbdt': [], 'lambdarank': []}
for f, (tr, te) in enumerate(gkf.split(X, Y, G)):
    tr, te = np.sort(tr), np.sort(te)
    # 1) V27 recipe (1 prior x 1 seed for speed)
    W = np.where(M[tr] == 0, 0.3, 0.7)
    h = HistGradientBoostingClassifier(max_depth=6, max_iter=500, learning_rate=0.03,
                                       min_samples_leaf=80, l2_regularization=1.0, random_state=0)
    h.fit(X[tr], Y[tr], sample_weight=W)
    res['v27_gbdt'].append(mrr(h.predict_proba(X[te])[:, 1], te))
    # 2) LambdaRank
    o = np.argsort(SG[tr], kind='stable'); trs = tr[o]
    _, cnt = np.unique(SG[trs], return_counts=True)
    r = lgb.LGBMRanker(objective='lambdarank', n_estimators=600, learning_rate=0.03,
                       num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                       colsample_bytree=0.8, lambdarank_truncation_level=25, verbose=-1)
    r.fit(X[trs], Y[trs], group=cnt)
    res['lambdarank'].append(mrr(r.predict(X[te]), te))
    print(f"fold {f}: v27={res['v27_gbdt'][-1]}  lr={res['lambdarank'][-1]}")

for k, v in res.items():
    for reg in (0, 1):
        vals = [d[reg][0] for d in v]
        print(f"{k:<12} M={reg}  MRR={np.mean(vals):.4f} ± {np.std(vals):.4f}")