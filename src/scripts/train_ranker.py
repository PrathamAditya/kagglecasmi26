import glob, numpy as np, lightgbm as lgb

def load(path):
    z = np.load(path)
    X, y, g, p = z['X'], z['y'].astype(int), z['g'], z['p_v27']
    feats, rows = [], []
    for gid in np.unique(g):
        m = np.where(g == gid)[0]
        pr = p[m]; order = np.argsort(-pr, kind='stable')
        rk = np.empty(len(m)); rk[order] = np.arange(1, len(m) + 1)
        extra = np.c_[pr, rk / len(m), pr - pr.max(), np.full(len(m), np.log1p(len(m)))]
        feats.append(np.c_[X[m], extra]); rows.append(m)
    idx = np.concatenate(rows)
    gkey = np.char.add(f"{z['batch']}_{z['regime']}_", g[idx].astype(str))
    return np.vstack(feats).astype(np.float32), y[idx], gkey, p[idx], str(z['regime'])

def mrr_by_group(score, y, gk):
    out = []
    for k in np.unique(gk):
        m = gk == k
        if y[m].sum() == 0: out.append(0.0); continue
        r = int(np.where(y[m][np.argsort(-score[m], kind='stable')] == 1)[0][0]) + 1
        out.append(1 / r if r <= 25 else 0.0)
    return np.array(out)

def boot_delta(a, b, n=1000, seed=0):
    rng = np.random.default_rng(seed); d = a - b
    s = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return d.mean(), np.percentile(s, [2.5, 97.5])

if __name__ == '__main__':
    train = [load(f) for f in sorted(glob.glob('artifacts/groups/groups_b0*.npz'))]
    Xtr = np.vstack([t[0] for t in train]); ytr = np.concatenate([t[1] for t in train])
    gtr = np.concatenate([t[2] for t in train])
    keep = np.isin(gtr, np.unique(gtr[ytr == 1]))
    Xtr, ytr, gtr = Xtr[keep], ytr[keep], gtr[keep]
    o = np.argsort(gtr, kind='stable'); Xtr, ytr, gtr = Xtr[o], ytr[o], gtr[o]
    groups_all = np.unique(gtr)
    print(f"train: {len(groups_all)} groups, {len(ytr):,} candidates")

    val = {r: load(f'artifacts/groups/groups_b99_{r}.npz') for r in ('C1', 'C2')}

    def fit(group_subset):
        m = np.isin(gtr, group_subset)
        _, cnt = np.unique(gtr[m], return_counts=True)
        r = lgb.LGBMRanker(objective='lambdarank', n_estimators=800, learning_rate=0.03,
                           num_leaves=31, min_child_samples=40, subsample=0.8, subsample_freq=1,
                           colsample_bytree=0.8, lambdarank_truncation_level=25, verbose=-1)
        r.fit(Xtr[m], ytr[m], group=cnt)
        return r

    model = fit(groups_all)
    for reg, (X, y, gk, p, _) in val.items():
        base = mrr_by_group(p, y, gk); new = mrr_by_group(model.predict(X), y, gk)
        d, (lo, hi) = boot_delta(new, base)
        print(f"{reg}: V27 {base.mean():.4f}  new {new.mean():.4f}  Δ {d:+.4f} [{lo:+.3f}, {hi:+.3f}]  n={len(base)}")

    rng = np.random.default_rng(0)
    X2, y2, g2, p2, _ = val['C2']
    for n in (500, 1000, 1500, 2000, len(groups_all)):
        if n > len(groups_all): continue
        sub = rng.choice(groups_all, n, replace=False)
        print(f"  train groups {n:>5}: C2 MRR {mrr_by_group(fit(sub).predict(X2), y2, g2).mean():.4f}")

    model.booster_.save_model('artifacts/groups/ranker_v1.txt')