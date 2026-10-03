import os, time
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
RDLogger.DisableLog("rdApp.*")

TRAIN = "/home/amoeba/Desktop/enveda-CASMI26-molecule-id-mass-spectra/train.parquet"
POOL = "artifacts/v17_builder/pool_712k_fixedfp.npz"
FORM = "artifacts/panel/pool_formulas.npy"
OUT = "artifacts/panel/panel_v1.npz"
os.makedirs("artifacts/panel", exist_ok=True)

pool = np.load(POOL, allow_pickle=True)
pkeys = pool["keys"].astype(str)
psmi = pool["smiles"]

if os.path.exists(FORM):
    forms = np.load(FORM)
else:
    t0 = time.time()
    out = []
    for s in psmi:
        m = Chem.MolFromSmiles(s)
        out.append(CalcMolFormula(m) if m is not None else "")
    forms = np.array(out, dtype="U40")
    np.save(FORM, forms)
    print("pool formulas computed in", round(time.time() - t0), "s")
print("pool rows", len(forms), "| empty formula", int((forms == "").sum()))

meta = np.load("artifacts/fpnet_data/meta.npz", allow_pickle=True)
mol, is_val, lib = meta["mol"], meta["is_val"], meta["lib"]
lib_names = list(meta["lib_names"])
ft_ik = np.load("artifacts/fpnet/fp_targets.npz", allow_pickle=True)["ik14"]

val_mols = np.unique(mol[is_val])
val_ik = ft_ik[val_mols]
np_mols = np.unique(mol[lib == lib_names.index("enveda-np-examples")])
is_np = np.isin(val_mols, np_mols)

row_of = {k: i for i, k in enumerate(pkeys)}
truth_rows = np.array([row_of[k] for k in val_ik], dtype=np.int64)
val_form = forms[truth_rows]
assert (val_form != "").all(), "some truth formulas are empty"
print("val molecules", len(val_mols), "| NP", int(is_np.sum()))

tp = pq.read_table(TRAIN, columns=["inchikey14", "molecular_formula"]).to_pandas().drop_duplicates("inchikey14")
fmap = dict(zip(tp["inchikey14"], tp["molecular_formula"]))
print("val formula RDKit vs train.parquet mismatches:", sum(fmap.get(k) != f for k, f in zip(val_ik, val_form)))

by_f = defaultdict(list)
for i, f in enumerate(forms):
    by_f[f].append(i)

n = len(val_mols)
rows_list = []
truth_pos = np.zeros(n, np.int32)
sizes = np.zeros(n, np.int64)
for v in range(n):
    r = np.array(by_f[val_form[v]], dtype=np.int32)
    pos = np.flatnonzero(r == truth_rows[v])
    assert len(pos) == 1
    truth_pos[v] = pos[0]
    sizes[v] = len(r)
    rows_list.append(r)
offsets = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
rows = np.concatenate(rows_list)

np.savez_compressed(OUT, val_mol=val_mols, val_ik14=val_ik, val_formula=val_form, is_np=is_np,
                    truth_row=truth_rows, truth_pos=truth_pos, offsets=offsets, rows=rows)

def dist(s, name):
    print(name, "n", len(s), "| median", float(np.median(s)), "| mean", round(float(s.mean()), 1), "| max", int(s.max()))
    for lo, hi in [(1, 1), (2, 2), (3, 5), (6, 20), (21, 100), (101, 1000), (1001, 10**9)]:
        print("   size", lo, "to", hi, ":", int(((s >= lo) & (s <= hi)).sum()))

dist(sizes, "ALL")
dist(sizes[is_np], "NP")
print("total panel rows", len(rows), "| unique pool rows", len(np.unique(rows)))
print("saved", OUT, os.path.getsize(OUT), "bytes")
