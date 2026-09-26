# src/scripts/diag_fp_origin.py
import numpy as np, pickle, pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator as G, MACCSkeys

TRAIN = '/home/amoeba/Desktop/enveda-CASMI26-molecule-id-mass-spectra/train.parquet'
pool = np.load('artifacts/v17_builder/pool_712k_precomputed.npz', allow_pickle=True)
sel = np.load('artifacts/coconut_candidates/fp_bits.npy')
coco = set(pickle.load(open('artifacts/coconut_candidates/coco_meta.pkl', 'rb'))['keys'])
train_ik = set(pq.read_table(TRAIN, columns=['inchikey14']).column('inchikey14').to_pylist())
nb = int(pool['nbits'])

gens = [G.GetMorganGenerator(radius=2, fpSize=4096), G.GetMorganGenerator(radius=3, fpSize=4096),
        G.GetRDKitFPGenerator(fpSize=2048)]
def recompute(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    a = [g.GetFingerprintAsNumPy(m).astype(np.uint8) for g in gens]
    mk = np.zeros(167, np.uint8)
    for i in MACCSkeys.GenMACCSKeys(m).GetOnBits(): mk[i] = 1
    return np.concatenate(a + [mk])[sel]

keys = pool['keys']
origin = np.array(['train' if k in train_ik else ('coco' if k in coco else 'other') for k in keys])
print({o: int((origin == o).sum()) for o in np.unique(origin)})

rng = np.random.default_rng(0)
for o in ('train', 'coco'):
    idx = rng.choice(np.where(origin == o)[0], 300, replace=False)
    stored = np.unpackbits(pool['fp'][idx], axis=1)[:, :nb]
    ok, diffs, pc = 0, [], []
    for j, i in enumerate(idx):
        f = recompute(str(pool['smiles'][i]))
        if f is None: continue
        d = int((f != stored[j]).sum()); diffs.append(d); ok += d == 0; pc.append(stored[j].sum())
    print(f"{o:>5}: exact match {ok}/{len(diffs)}  mean bit diff {np.mean(diffs):.2f}  "
          f"mean popcount {np.mean(pc):.0f}")

v = [k for k in np.load('artifacts/groups/groups_b99_C2.npz')['group_key']]
print(f"val compounds in COCONUT: {np.mean([k in coco for k in v]):.1%}")