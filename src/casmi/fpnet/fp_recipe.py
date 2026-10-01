"""SMILES -> 6930-bit fingerprint (packed to 867 bytes), identical to the V17 recipe used
by src/scripts/fix_pool_fps.py: Morgan r2/4096 + Morgan r3/4096 + RDKit FP/2048 + MACCS/167,
then bit-selected with artifacts/coconut_candidates/fp_bits.npy.

Keep this identical to fix_pool_fps.py. If the bits differ from the pool's, FPNet would learn
bits that don't line up with the candidates. check_fp_recipe.py verifies this against the pool.
"""
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys
from rdkit.Chem import rdFingerprintGenerator as G

SEL_PATH = "artifacts/coconut_candidates/fp_bits.npy"
NBITS = 6930
NBYTES = (NBITS + 7) // 8  # 867

_SEL = None
_GENS = None


def init(sel_path=SEL_PATH):
    """Call once per process (also as the multiprocessing initializer)."""
    global _SEL, _GENS
    RDLogger.DisableLog("rdApp.*")
    _SEL = np.load(sel_path)
    _GENS = [G.GetMorganGenerator(radius=2, fpSize=4096),
             G.GetMorganGenerator(radius=3, fpSize=4096),
             G.GetRDKitFPGenerator(fpSize=2048)]


def smiles_to_packed(smi):
    """Returns uint8 array of length 867 (packed bits), or None if the SMILES doesn't parse."""
    if _GENS is None:
        init()
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    a = [g.GetFingerprintAsNumPy(m).astype(np.uint8) for g in _GENS]
    mk = np.zeros(167, np.uint8)
    for i in MACCSkeys.GenMACCSKeys(m).GetOnBits():
        mk[i] = 1
    return np.packbits(np.concatenate(a + [mk])[_SEL])


def unpack(packed, nbits=NBITS):
    """packed uint8 (..., 867) -> uint8 bits (..., 6930)."""
    return np.unpackbits(packed, axis=-1)[..., :nbits]