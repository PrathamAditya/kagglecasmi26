# src/scripts/diag_shortcut.py
import numpy as np, pyarrow.parquet as pq, lightgbm as lgb, glob, sys
sys.path.insert(0, 'src/scripts')
from scripts.train_ranker import load, mrr_by_group

TRAIN = '/home/amoeba/Desktop/enveda-CASMI26-molecule-id-mass-spectra/train.parquet'
train_ik = set(pq.read_table(TRAIN, columns=['inchikey14']).column('inchikey14').to_pylist())

NAMES = (['lv','lv_rank','lvmax','lv-lvmax','lv>0','ap','ap_rank','apmax','ap-apmax','a1',
          'best_tan','top_tan','mean_tan','top_sim','log_nc']
         + [f'fpnet{i}' for i in range(6)] + [f'frag{i}' for i in range(4)]
         + [f'xfp{i}' for i in range(6)] + ['p_v27','p_rank','p_gap','log_n'])

# 1) does the shortcut exist? in C2 val: among candidates with no library hit, how many are train structures?
z = np.load('artifacts/groups/groups_b99_C2.npz')
ck, g, y, lv = z['cand_key'], z['g'], z['y'], z['X'][:, 0]
uniq = []
for gid in np.unique(g):
    m = g == gid
    if y[m].sum() == 0: continue
    zero_train = [(k in train_ik) and l == 0 for k, l in zip(ck[m], lv[m])]
    uniq.append(sum(zero_train) == 1 and zero_train[int(np.where(y[m] == 1)[0][0])])
print(f"C2 val: truth is the ONLY train-structure with lv=0 in {np.mean(uniq):.1%} of groups")

# 2) retrain with ablations
tr = [load(f) for f in sorted(glob.glob('artifacts/groups/groups_b0*.npz'))]
X = np.vstack([t[0] for t in tr]); Y = np.concatenate([t[1] for t in tr]); G = np.concatenate([t[2] for t in tr])
keep = np.isin(G, np.unique(G[Y == 1])); X, Y, G = X[keep], Y[keep], G[keep]
o = np.argsort(G, kind='stable'); X, Y, G = X[o], Y[o], G[o]
_, cnt = np.unique(G, return_counts=True)
val = {r: load(f'artifacts/groups/groups_b99_{r}.npz') for r in ('C1', 'C2')}

def run(cols, label):
    r = lgb.LGBMRanker(objective='lambdarank', n_estimators=800, learning_rate=0.03, num_leaves=31,
                       min_child_samples=40, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                       lambdarank_truncation_level=25, verbose=-1)
    r.fit(X[:, cols], Y, group=cnt)
    s = {k: mrr_by_group(r.predict(v[0][:, cols]), v[1], v[2]).mean() for k, v in val.items()}
    print(f"{label:<28} C1 {s['C1']:.4f}   C2 {s['C2']:.4f}")
    return r

allc = list(range(35))
full = run(allc, 'all 35 features')
imp = full.booster_.feature_importance('gain'); top = np.argsort(-imp)[:10]
print('top gain:', [(NAMES[i], round(imp[i] / imp.sum(), 3)) for i in top])
fp_cols = list(range(15, 21)) + list(range(25, 31))
run([c for c in allc if c not in fp_cols], 'no FPNet features')
run([c for c in allc if c not in range(0, 5)], 'no library features')
run([c for c in allc if c not in fp_cols and c not in range(0, 5)], 'no FPNet + no library')
# oversample C1 groups to 50/50 with C2
c1 = np.char.find(G, '_C1_') >= 0
g_c1, g_c2 = np.unique(G[c1]), np.unique(G[~c1])
rep = max(1, len(g_c2) // len(g_c1))
Xb = [X[~c1]]; Yb = [Y[~c1]]; Gb = [G[~c1]]
for r in range(rep):
    Xb.append(X[c1]); Yb.append(Y[c1]); Gb.append(np.char.add(G[c1], f'_r{r}'))
X, Y, G = np.vstack(Xb), np.concatenate(Yb), np.concatenate(Gb)
o = np.argsort(G, kind='stable'); X, Y, G = X[o], Y[o], G[o]
_, cnt = np.unique(G, return_counts=True)
run(allc, 'BALANCED all features')
an_cols = list(range(5, 14))
run([c for c in allc if c not in an_cols], 'BALANCED no analog features')