"""Spectrum cleaning + batch building for FPNet training (numpy only, no torch).

prep_peaks / instr_family / ce_of follow the V27 notebook ("CELL 6") so our model sees the same
kind of input as the frozen FPNet. One deliberate difference: prep_peaks precomputes the 'already
selected' set once instead of rebuilding it inside the loop (same result, avoids a slow O(n^2) path).
"""
import os

import numpy as np

from casmi.fpnet.model import ADDUCT_IX

TEN = ["[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
       "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-"]
MAX_PEAKS_NN = 128
NBITS = 6930
NBYTES = 867
DEFAULT_CE = 25.0  # same fallback the frozen pipeline uses when collision energy is missing


def prep_peaks(mz, inten, prec_mz, max_peaks=MAX_PEAKS_NN, floor=1e-3, win=50.0, per_win=8):
    empty = (np.zeros(0, np.float32), np.zeros(0, np.float32))
    mz = np.asarray(mz, np.float64)
    it = np.asarray(inten, np.float64)
    if len(mz) == 0:
        return empty
    keep = mz <= prec_mz + 1.5
    mz, it = mz[keep], it[keep]
    if len(mz) == 0:
        return empty
    mx = it.max()
    if mx <= 0:
        return empty
    keep = it >= floor * mx
    mz, it = mz[keep], it[keep]
    if len(mz) > max_peaks:
        order = np.argsort(-it)
        bucket = (mz // win).astype(np.int64)
        cnt = {}
        sel = []
        for i in order:
            b = bucket[i]
            c = cnt.get(b, 0)
            if c < per_win:
                cnt[b] = c + 1
                sel.append(i)
        sel = np.array(sel)
        if len(sel) > max_peaks:
            sel = sel[np.argsort(-it[sel])[:max_peaks]]
        elif len(sel) < max_peaks:
            chosen = set(sel.tolist())
            rest = np.array([i for i in order if i not in chosen])
            need = max_peaks - len(sel)
            if len(rest):
                sel = np.concatenate([sel, rest[:need]])
        mz, it = mz[sel], it[sel]
    o = np.argsort(mz)
    mz, it = mz[o], it[o]
    v = np.sqrt(it / it.max())
    return mz.astype(np.float32), v.astype(np.float32)


def instr_family(s):
    if s is None:
        return 4
    t = str(s).lower()
    if 'timstof' in t:
        return 0
    if 'orbitrap' in t or 'qft' in t or 'ftms' in t or 'hybrid ft' in t or 'itft' in t or 'exactive' in t:
        return 1
    if 'tof' in t:
        return 2
    if 'trap' in t or 'qq' in t:
        return 3
    return 4


def ce_raw(v):
    """Mean collision energy in eV, or NaN if missing/unusable (multi-value entries are averaged)."""
    try:
        if v is None:
            return float("nan")
        a = np.atleast_1d(np.asarray(v, dtype=np.float64))
        if len(a) == 0:
            return float("nan")
        m = float(np.mean(a))
        return m if np.isfinite(m) else float("nan")
    except Exception:
        return float("nan")


def ce_of(v):
    """Mean collision energy in eV; DEFAULT_CE if missing/unusable."""
    m = ce_raw(v)
    return DEFAULT_CE if np.isnan(m) else m


class SpecData:
    """Everything prep.py writes, loaded into memory. Peaks are stored flat with offsets."""

    def __init__(self, d):
        self.mz = np.load(os.path.join(d, "peaks_mz.npy"))
        self.it = np.load(os.path.join(d, "peaks_it.npy"))
        self.offsets = np.load(os.path.join(d, "offsets.npy"))
        m = np.load(os.path.join(d, "meta.npz"), allow_pickle=True)
        self.prec, self.ad, self.ins = m["prec"], m["ad"], m["ins"]
        self.ce, self.mode, self.mol = m["ce"], m["mode"], m["mol"]
        self.lib, self.is_val = m["lib"], m["is_val"]
        self.lib_names = [str(x) for x in m["lib_names"]]
        self.n = len(self.prec)
        # spectra of molecules from the holdout library (natural-product proxy); absent in older prep output
        self.is_holdout = m["is_holdout"] if "is_holdout" in m.files else np.zeros(self.n, bool)


def make_batch(D, idx):
    """Pad the spectra in idx into arrays the model expects. Returns a dict of numpy arrays."""
    idx = np.asarray(idx)
    starts, ends = D.offsets[idx], D.offsets[idx + 1]
    lens = ends - starts
    B, N = len(idx), int(lens.max())
    mz = np.zeros((B, N), np.float32)
    it = np.zeros((B, N), np.float32)
    pad = np.ones((B, N), bool)
    for i, (a, b) in enumerate(zip(starts, ends)):
        k = b - a
        mz[i, :k] = D.mz[a:b]
        it[i, :k] = D.it[a:b]
        pad[i, :k] = False
    return dict(mz=mz, it=it, pad=pad, prec=D.prec[idx], ad=D.ad[idx].astype(np.int64),
                ins=D.ins[idx].astype(np.int64), ce=D.ce[idx], mode=D.mode[idx])


def unpack_targets(fp_packed, mol_idx):
    """packed (n_mol, 867) uint8 + molecule indices -> float32 (B, 6930) 0/1 targets."""
    return np.unpackbits(fp_packed[mol_idx], axis=1)[:, :NBITS].astype(np.float32)