# src/scripts/fix_pool_fps.py  —  recompute all pool FPs with the V17 recipe
import numpy as np, os
from multiprocessing import Pool
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator as G, MACCSkeys

SEL = np.load('artifacts/coconut_candidates/fp_bits.npy')
_g = None
def _init():
    global _g; RDLogger.DisableLog('rdApp.*')
    _g = [G.GetMorganGenerator(radius=2, fpSize=4096), G.GetMorganGenerator(radius=3, fpSize=4096),
          G.GetRDKitFPGenerator(fpSize=2048)]
def fp(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    a = [g.GetFingerprintAsNumPy(m).astype(np.uint8) for g in _g]
    mk = np.zeros(167, np.uint8)
    for i in MACCSkeys.GenMACCSKeys(m).GetOnBits(): mk[i] = 1
    return np.packbits(np.concatenate(a + [mk])[SEL])

if __name__ == '__main__':
    z = np.load('artifacts/v17_builder/pool_712k_precomputed.npz', allow_pickle=True)
    fps = z['fp'].copy(); smiles = [str(s) for s in z['smiles']]
    with Pool(os.cpu_count() - 1, initializer=_init) as p:
        new = p.map(fp, smiles, chunksize=2000)
    bad = changed = 0
    for i, f in enumerate(new):
        if f is None: bad += 1; continue          # keep old fp if SMILES fails to parse
        if not np.array_equal(f, fps[i]): changed += 1
        fps[i] = f
    print(f"rows {len(new):,}  changed {changed:,}  parse-fail kept old {bad:,}")
    np.savez_compressed('artifacts/v17_builder/pool_712k_fixedfp.npz',
                        fp=fps, mass=z['mass'], keys=z['keys'], smiles=z['smiles'], nbits=z['nbits'])