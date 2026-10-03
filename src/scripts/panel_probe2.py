import time
import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
RDLogger.DisableLog("rdApp.*")

TRAIN = "/home/amoeba/Desktop/enveda-CASMI26-molecule-id-mass-spectra/train.parquet"

meta = np.load("artifacts/fpnet_data/meta.npz", allow_pickle=True)
mol, is_val, is_ho = meta["mol"], meta["is_val"], meta["is_holdout"]
lib, lib_names = meta["lib"], list(meta["lib_names"])
ft_ik = np.load("artifacts/fpnet/fp_targets.npz", allow_pickle=True)["ik14"]
pool = np.load("artifacts/v17_builder/pool_712k_fixedfp.npz", allow_pickle=True)
pkeys = pool["keys"].astype(str)
psmi = pool["smiles"]

print("mol min/max", mol.min(), mol.max(), "| fp_targets rows", len(ft_ik), "| unique mol in meta", len(np.unique(mol)))
val_mols = np.unique(mol[is_val])
ho_mols = np.unique(mol[is_ho])
nonval_mols = np.unique(mol[~is_val])
print("val molecules", len(val_mols), "| holdout molecules", len(ho_mols), "| val&holdout molecules", len(np.intersect1d(val_mols, ho_mols)))
print("molecules with both val and non-val spectra", len(np.intersect1d(val_mols, nonval_mols)))
print("spectra: val", int(is_val.sum()), "holdout", int(is_ho.sum()), "val&holdout", int((is_val & is_ho).sum()))

i = lib_names.index("enveda-np-examples")
np_mols = np.unique(mol[lib == i])
print("enveda-np-examples: molecules", len(np_mols), "| of which is_val", int(np.isin(np_mols, val_mols).sum()))
t = pq.read_table(TRAIN, columns=["inchikey14"], filters=[("ingest_lib", "==", "enveda-np-examples")])
tp_np = set(t.column("inchikey14").to_pylist())
print("np-examples ik14 in train.parquet", len(tp_np), "| mapped via mol->fp_targets matches", len(tp_np & set(ft_ik[np_mols])))

val_ik = ft_ik[val_mols]
print("val molecules in pool by ik14:", int(np.isin(val_ik, pkeys).sum()), "of", len(val_ik))
print("all fp_targets molecules in pool:", int(np.isin(ft_ik, pkeys).sum()), "of", len(ft_ik))
print("pool unique keys", len(np.unique(pkeys)), "of", len(pkeys), "| pool rows with key in fp_targets", int(np.isin(pkeys, ft_ik).sum()))

tp = pq.read_table(TRAIN, columns=["inchikey14", "molecular_formula"]).to_pandas().drop_duplicates("inchikey14")
print("train structures", len(tp), "| null formulas", int(tp["molecular_formula"].isna().sum()))
fmap = dict(zip(tp["inchikey14"], tp["molecular_formula"]))
vf = [fmap.get(k) for k in val_ik]
print("val molecules with formula:", sum(f is not None for f in vf), "of", len(vf), "| examples", vf[:5])

rng = np.random.default_rng(0)
idx = np.flatnonzero(np.isin(pkeys, ft_ik))
idx = rng.choice(idx, 3000, replace=False)
t0 = time.time(); same = bad = 0; ex = []
for j in idx:
    m = Chem.MolFromSmiles(psmi[j])
    if m is None:
        bad += 1; continue
    f = CalcMolFormula(m)
    if f == fmap.get(pkeys[j]):
        same += 1
    elif len(ex) < 5:
        ex.append((psmi[j], f, fmap.get(pkeys[j])))
print("train-row sample: RDKit formula == train formula", same, "of 3000 | parse fails", bad, "| secs", round(time.time() - t0, 1))
print("mismatch examples (smiles, rdkit, train):", ex)

idx2 = np.flatnonzero(~np.isin(pkeys, ft_ik))
idx2 = rng.choice(idx2, 3000, replace=False)
t0 = time.time(); bad2 = 0
for j in idx2:
    m = Chem.MolFromSmiles(psmi[j])
    if m is None:
        bad2 += 1
    else:
        CalcMolFormula(m)
print("non-train pool sample: parse fails", bad2, "of 3000 | secs", round(time.time() - t0, 1))
