"""exp_016 — V27 (patched, public LB 0.334) ported as-is behind the Predictor interface.

Source: reference/casmi-four-channel-v27.ipynb (cells 0, 2–9). The scoring logic below is a
transcription, not a re-implementation: kernels, feature order, patch set (A1 ppm offset,
A2 absolute mass floor, B2 fragment soft cap, C1 pooled frag workers, C2 memoised tautomer
keys) are copied verbatim. The only changes are I/O:

  * `CFG.X` globals  -> `cfg.x` fields of V27Config (values come from configs/exp_016.yaml).
  * load_library(TRAIN) -> load_library(table) on the harness's regime-filtered train frame
    (same columns, same file row order — build_rep tie-breaks on row order).
  * find_file(...)      -> explicit artifact paths from the config.
  * calibrate_ppm_offset runs only when ppm_offset == "calibrate"; otherwise the frozen,
    logged value from the config is used (the notebook calibrates on the file it predicts).
  * C3 support: `pool_exclude_keys` (set by the harness) drops the held-out structures from
    V27's private 712k pool by raw InChIKey14 AND metric tautomer key.
  * predict() returns V27's 25 metric-unique picks first, then the rest of the ranked window
    (for Recall@cand). The first 25 are exactly what the notebook writes to submission.csv.

Frozen artifacts are loaded read-only and never modified. `rep_analog_scaffolds.npz` is NOT
used: the notebook never loads it (it rebuilds the analog set from the library), and it
contains every train structure, so using it would leak the truth in C2/C3.
"""
from __future__ import annotations

import gc
import glob
import math
import os
import time
from dataclasses import dataclass, fields
from functools import partial
from multiprocessing import Pool as MPool

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
from numba import njit, prange

from .base import Candidate, Predictor


# =====================================================================================
#  CELL 0 + CELL 8 CONFIG ADDITIONS
# =====================================================================================
@dataclass
class V27Config:
    ppm_win: float = 8.5
    ppm_fallback: float = 30.0
    int_floor: float = 0.002
    max_peaks: int = 256
    mz_tol: float = 0.01
    int_power: float = 1.0
    ent_weight: bool = True
    analog_win: float = 200.0
    n_analog: int = 100
    sim_power: float = 4.0
    topn: int = 25
    ppm_offset: float = 0.0
    min_da: float = 0.0018
    frag_soft_cap: int = 34
    frag_hard_cap: int = 80
    max_abs_offset: float = 5.0
    use_analog_fallback: bool = False
    frag_workers: int = 4
    nfeat: int = 31
    # calibrate_ppm_offset() arguments (cell 8 defaults)
    calib_n_sample: int = 150
    calib_sim_thresh: float = 0.90
    calib_search_ppm: float = 25.0
    calib_seed: int = 0
    calib_min_pairs: int = 20
    # exp_021c: demote candidates that HAVE reference spectra but match the query poorly.
    # Hidden C2/C3 truths never have public spectra. alpha=1.0 disables (== exp_021).
    ref_penalty_alpha: float = 1.0
    ref_penalty_sim: float = 0.5


# =====================================================================================
#  CELL 2 — numba kernels (verbatim)
# =====================================================================================
@njit(cache=True, fastmath=True)
def _clean(mz, it, floor, topk, power, ent_weight):
    n = len(mz)
    if n == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    mx = 0.0
    for i in range(n):
        if it[i] > mx: mx = it[i]
    if mx <= 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    thr = floor * mx
    c = 0
    for i in range(n):
        if it[i] >= thr: c += 1
    idx = np.empty(c, np.int64)
    j = 0
    for i in range(n):
        if it[i] >= thr:
            idx[j] = i; j += 1
    if c > topk:
        v = np.empty(c, np.float32)
        for i in range(c): v[i] = it[idx[i]]
        o = np.argsort(v)[c - topk:]
        k2 = np.empty(topk, np.int64)
        for i in range(topk): k2[i] = idx[o[i]]
        k2.sort()
        idx = k2
        c = topk
    om = np.empty(c, np.float32)
    oi = np.empty(c, np.float32)
    s = 0.0
    for i in range(c):
        om[i] = mz[idx[i]]
        v = it[idx[i]] ** power
        oi[i] = v
        s += v
    if s > 0:
        for i in range(c): oi[i] /= s
    if ent_weight:
        S = 0.0
        for i in range(c):
            if oi[i] > 0: S -= oi[i] * np.log(oi[i])
        if S < 3.0:
            w = 0.25 + 0.25 * S
            s2 = 0.0
            for i in range(c):
                oi[i] = oi[i] ** w
                s2 += oi[i]
            if s2 > 0:
                for i in range(c): oi[i] /= s2
    return om, oi


@njit(cache=True, fastmath=True)
def entropy_sim(qmz, qp, cmz, cp, tol):
    i = 0; j = 0; n = len(qmz); m = len(cmz)
    SA = 0.0
    for x in range(n):
        if qp[x] > 0: SA -= qp[x] * np.log(qp[x])
    SB = 0.0
    for x in range(m):
        if cp[x] > 0: SB -= cp[x] * np.log(cp[x])
    SAB = 0.0; tot = 0.0
    buf = np.empty(n + m, np.float64); b = 0
    while i < n and j < m:
        d = qmz[i] - cmz[j]
        if d < -tol:
            buf[b] = qp[i]; i += 1; b += 1
        elif d > tol:
            buf[b] = cp[j]; j += 1; b += 1
        else:
            buf[b] = qp[i] + cp[j]; i += 1; j += 1; b += 1
    while i < n: buf[b] = qp[i]; i += 1; b += 1
    while j < m: buf[b] = cp[j]; j += 1; b += 1
    for x in range(b): tot += buf[x]
    if tot <= 0: return 0.0
    for x in range(b):
        v = buf[x] / tot
        if v > 0: SAB -= v * np.log(v)
    return 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)


@njit(cache=True, fastmath=True)
def entropy_sim_shift(qmz, qp, cmz, cp, tol, shift):
    """Evaluates max of direct and mass-shifted spectral entropy similarity."""
    a = entropy_sim(qmz, qp, cmz, cp, tol)
    if shift > -0.001 and shift < 0.001:
        return a
    sm = np.empty(len(cmz), np.float32)
    for i in range(len(cmz)):
        sm[i] = cmz[i] + shift
    b = entropy_sim(qmz, qp, sm, cp, tol)
    return a if a > b else b


# =====================================================================================
#  CELL 3 — adduct physics & reference library (verbatim, I/O adapted)
# =====================================================================================
MASS = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, P=30.97376163,
            S=31.97207100, F=18.99840322, Cl=34.96885268, Br=78.9183371, I=126.904473,
            Na=22.9897692809, K=38.96370668, Si=27.9769265325, B=11.0093054, Se=79.9165213)
E = 0.00054857990
PROTON = MASS['H'] - E
H2O = 2 * MASS['H'] + MASS['O']
NH4 = MASS['N'] + 4 * MASS['H']
FORMATE = MASS['C'] + 2 * MASS['H'] + 2 * MASS['O']
ACETATE = 2 * MASS['C'] + 4 * MASS['H'] + 2 * MASS['O']

ADDUCTS = {
    "[M+H]+": (1, 1, PROTON), "[M+NH4]+": (1, 1, NH4 - E), "[M+Na]+": (1, 1, MASS['Na'] - E),
    "[M+K]+": (1, 1, MASS['K'] - E), "[M-H2O+H]+": (1, 1, PROTON - H2O), "[M-2H2O+H]+": (1, 1, PROTON - 2*H2O),
    "[M+2H]2+": (1, 2, 2 * PROTON), "[M]+": (1, 1, -E), "[M-H2O]+": (1, 1, -E - H2O),
    "[M+CH3OH+H]+": (1, 1, PROTON + MASS['C'] + 4 * MASS['H'] + MASS['O']),
    "[M+CH3CN+H]+": (1, 1, PROTON + 2 * MASS['C'] + 3 * MASS['H'] + MASS['N']),
    "[M-H]-": (1, 1, -PROTON), "[M-H2O-H]-": (1, 1, -PROTON - H2O), "[M+CH2O2-H]-": (1, 1, FORMATE - PROTON),
    "[M+C2H4O2-H]-": (1, 1, ACETATE - PROTON), "[M+Cl]-": (1, 1, MASS['Cl'] + E), "[M]-": (1, 1, E),
    "[M-2H]-": (1, 2, -2 * PROTON), "[M+Na-2H]-": (1, 1, MASS['Na'] - 2 * PROTON),
    "[2M+H]+": (2, 1, PROTON), "[2M+Na]+": (2, 1, MASS['Na'] - E), "[2M+NH4]+": (2, 1, NH4 - E),
    "[2M+K]+": (2, 1, MASS['K'] - E), "[2M-H]-": (2, 1, -PROTON), "[2M+CH2O2-H]-": (2, 1, FORMATE - PROTON),
    "[2M+C2H4O2-H]-": (2, 1, ACETATE - PROTON), "[2M+Na-2H]-": (2, 1, MASS['Na'] - 2 * PROTON),
    "[3M+H]+": (3, 1, PROTON), "[3M-H]-": (3, 1, -PROTON),
}
LIBRARY_COLUMNS = ['inchikey14', 'normalized_smiles', 'adduct', 'precursor_mz',
                   'ms2_mzs', 'ms2_normalized_intensities']


def neutral_mass(mz, adduct):
    out = np.full(len(mz), np.nan)
    ad = np.asarray(adduct, dtype=object)
    for a, (n, z, d) in ADDUCTS.items():
        m = (ad == a)
        if m.any():
            out[m] = (mz[m] * z - d) / n
    return out


def _list_flat(col: pa.ChunkedArray) -> tuple[np.ndarray, np.ndarray]:
    """offsets + flat float32 values of a list column (cell 3 used .offsets/.values directly;
    rebased here in case the combined array does not start at offset 0)."""
    arr = col.combine_chunks()
    off = arr.offsets.to_numpy().astype(np.int64)
    vals = arr.values.to_numpy(zero_copy_only=False).astype(np.float32)
    return off - off[0], vals[off[0]:off[-1]]


def load_library(t: pa.Table):
    t0 = time.time()
    off, allmz = _list_flat(t.column('ms2_mzs'))
    off_i, allin = _list_flat(t.column('ms2_normalized_intensities'))
    assert np.array_equal(off, off_i), "ms2_mzs / ms2_normalized_intensities length mismatch"
    prec = t.column('precursor_mz').to_numpy(zero_copy_only=False).astype(np.float64)
    add = np.asarray(t.column('adduct').cast(pa.string()).to_pylist(), dtype=object)
    ik  = np.asarray(t.column('inchikey14').cast(pa.string()).to_pylist(), dtype=object)
    smi = np.asarray(t.column('normalized_smiles').cast(pa.string()).to_pylist(), dtype=object)
    nm  = neutral_mass(prec, add)
    ok  = np.isfinite(nm)
    order = np.argsort(np.where(ok, nm, 1e18), kind='mergesort')
    best = {}
    for k, s in zip(ik, smi):
        if k and s and k not in best:
            best[k] = s
    print(f"[V27] Reference Library loaded: {len(off)-1:,} spectra / {len(best):,} unique structures ({time.time()-t0:.1f}s)", flush=True)
    return dict(off=off, mz=allmz, it=allin, nm=nm, ik=ik, best=best,
                order=order, snm=nm[order], n_ok=int(ok.sum()))


def lib_window(L, target, tol):
    lo = np.searchsorted(L['snm'][:L['n_ok']], target - tol, 'left')
    hi = np.searchsorted(L['snm'][:L['n_ok']], target + tol, 'right')
    return L['order'][lo:hi]


def build_rep(L):
    """Extracts the single richest representative spectrum per unique structure for analog retrieval."""
    npk = np.diff(L['off'])
    best = {}; ik = L['ik']
    for i in range(len(ik)):
        k = ik[i]
        if k and (k not in best or npk[i] > npk[best[k]]):
            best[k] = i
    rep = np.array(sorted(best.values()))
    nm = L['nm'][rep]
    ok = np.isfinite(nm)
    rep = rep[ok]; nm = nm[ok]; key = ik[rep]
    o = np.argsort(nm)
    print(f"[V27] Built representative analog set: {len(rep):,} unique scaffolds.", flush=True)
    return rep[o], key[o], nm[o]


# =====================================================================================
#  CELL 4 + TIER A2 — candidate pool (verbatim)
# =====================================================================================
class CandidatePool:
    def __init__(self, npz):
        self.fp = npz['fp']
        self.mass = npz['mass']
        self.keys = npz['keys']
        self.smiles = npz['smiles']
        self.nbits = int(npz['nbits'])
        self.k2i = {k: i for i, k in enumerate(self.keys)}

    def window(self, target, ppm, min_da=0.0):
        tol = max(target * ppm / 1e6, min_da)
        lo = int(np.searchsorted(self.mass, target - tol, side='left'))
        hi = int(np.searchsorted(self.mass, target + tol, side='right'))
        return np.arange(lo, hi)

    def fps(self, indices):
        return np.unpackbits(self.fp[indices], axis=1)[:, :self.nbits]

    def drop(self, remove: np.ndarray) -> None:
        """Harness C3 support: remove rows (mass order is preserved)."""
        keep = np.ones(len(self.mass), bool); keep[remove] = False
        self.fp, self.mass = self.fp[keep], self.mass[keep]
        self.keys, self.smiles = self.keys[keep], self.smiles[keep]
        self.k2i = {k: i for i, k in enumerate(self.keys)}


# =====================================================================================
#  CELL 5 — evidence channels (verbatim; lib_sim/analog_sim are the final cell-8 versions)
# =====================================================================================
AMU = {'C':12.0, 'H':1.00782503207, 'N':14.0030740048, 'O':15.9949146196, 'P':30.97376163,
       'S':31.97207100, 'F':18.99840322, 'Cl':34.96885268, 'Br':78.9183371, 'I':126.904473,
       'Na':22.9897692809, 'K':38.96370668, 'Si':27.9769265325, 'B':11.0093054, 'Se':79.9165213}
H_ATOM = AMU['H']
PROTON_MASS = H_ATOM - 0.00054857990


def clean_spectrum(cfg, mz, it):
    return _clean(np.asarray(mz, np.float32), np.asarray(it, np.float32),
                  cfg.int_floor, cfg.max_peaks, cfg.int_power, cfg.ent_weight)


def mol_graph(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    n = m.GetNumAtoms()
    w = np.zeros(n)
    for a in m.GetAtoms():
        w[a.GetIdx()] = AMU.get(a.GetSymbol(), 0.0) + a.GetTotalNumHs() * H_ATOM
    if (w == 0).any(): return None
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
    return w, bonds, n


def _components(n, bonds, drop):
    adj = [[] for _ in range(n)]
    for i, (a, b) in enumerate(bonds):
        if i in drop: continue
        adj[a].append(b); adj[b].append(a)
    seen = np.zeros(n, bool); comps = []
    for s in range(n):
        if seen[s]: continue
        stack = [s]; seen[s] = True; cur = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]: seen[v] = True; stack.append(v); cur.append(v)
        comps.append(cur)
    return comps


def fragment_masses(smi, max_breaks=2, max_bonds=34, hard_cap=80):
    """TIER B2 version (the one active in the LB run): soft cap -> 1 break, hard cap -> give up."""
    g = mol_graph(smi)
    if g is None:
        return np.zeros(0)
    w, bonds, n = g
    nb = len(bonds)
    if nb == 0 or nb > hard_cap:
        return np.array([w.sum()])
    if nb > max_bonds:
        max_breaks = 1                      # O(nb) instead of O(nb^2)
    out = {w.sum()}
    for i in range(nb):
        for c in _components(n, bonds, {i}):
            out.add(float(w[c].sum()))
    if max_breaks >= 2:
        for i in range(nb):
            for j in range(i + 1, nb):
                for c in _components(n, bonds, {i, j}):
                    out.add(float(w[c].sum()))
    return np.array(sorted(out))


def explain_score(frag_mass, peak_mz, peak_int, mode=1.0, tol=0.01, h_shifts=(-2, -1, 0, 1, 2)):
    if len(frag_mass) == 0 or len(peak_mz) == 0: return 0.0
    ion = []
    for dh in h_shifts:
        ion.append(frag_mass + dh * H_ATOM + (PROTON_MASS if mode > 0 else -PROTON_MASS))
    ion = np.sort(np.concatenate(ion))
    w = np.sqrt(np.asarray(peak_int, float)); tot = w.sum()
    if tot <= 0: return 0.0
    idx = np.searchsorted(ion, peak_mz)
    ok = np.zeros(len(peak_mz), bool)
    for off in (-1, 0):
        k = np.clip(idx + off, 0, len(ion) - 1)
        ok |= np.abs(ion[k] - peak_mz) <= tol
    return float(w[ok].sum() / tot)


def _frag_masses_wrapper(smi, max_bonds=34, hard_cap=80):
    try: return fragment_masses(smi, max_bonds=max_bonds, hard_cap=hard_cap)
    except Exception: return np.zeros(0)


# TIER C1 — process pool hoisted out of the per-molecule loop (module singleton, as in the notebook).
_FRAG_POOL = None


def _get_frag_pool(workers=4):
    global _FRAG_POOL
    if _FRAG_POOL is None:
        _FRAG_POOL = MPool(workers)
    return _FRAG_POOL


def frag_scores(cfg, cand_smiles, specs, mode):
    fn = partial(_frag_masses_wrapper, max_bonds=cfg.frag_soft_cap, hard_cap=cfg.frag_hard_cap)
    frags = _get_frag_pool(cfg.frag_workers).map(fn, cand_smiles, chunksize=8)
    peaks = []
    for mz, it in specs:
        m2, i2 = _clean(np.asarray(mz, np.float32), np.asarray(it, np.float32),
                        cfg.int_floor, cfg.max_peaks, 1.0, False)
        peaks.append((np.asarray(m2, float), np.asarray(i2, float)))
    out = np.zeros(len(cand_smiles), np.float32)
    for j, f in enumerate(frags):
        out[j] = max((explain_score(f, a, b, mode=mode, tol=cfg.mz_tol) for a, b in peaks),
                     default=0.0)
    return out


# =====================================================================================
#  CELL 7 — 31-feature extraction (verbatim)
# =====================================================================================
def _rank_norm(x):
    o = np.argsort(-x)
    r = np.empty(len(x))
    r[o] = np.arange(len(x))
    return r / max(1, len(x) - 1)


def _z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)


def rank_features(cand_fp, cand_lib, analog_fp, analog_sim_vals, model_logits=None, frag=None, P_SIM=4.0):
    nc = cand_fp.shape[0]
    cf = cand_fp.astype(np.float32)
    cs = cf.sum(1)
    lv = np.asarray(cand_lib, np.float32)
    lvmax = float(lv.max()) if nc else 0.0
    if analog_fp is not None and len(analog_sim_vals):
        af = analog_fp.astype(np.float32)
        asum = af.sum(1)
        inter = cf @ af.T
        tan = inter / (cs[:, None] + asum[None, :] - inter + 1e-9)
        w = np.clip(np.asarray(analog_sim_vals, np.float32), 0, None)
        ap = (tan * (w ** P_SIM)[None, :]).max(1)
        a1 = (tan * w[None, :]).max(1)
        best_tan = tan.max(1)
        top_tan = tan[:, 0]
        top_sim = float(w[0])
        mean_tan = (tan * (w ** P_SIM)[None, :]).sum(1) / ((w ** P_SIM).sum() + 1e-9)
    else:
        ap = a1 = best_tan = top_tan = mean_tan = np.zeros(nc, np.float32)
        top_sim = 0.0
    apmax = float(ap.max()) if nc else 0.0
    if model_logits is not None:
        raw = cf @ np.asarray(model_logits, np.float32)
        nrm = raw / np.sqrt(np.maximum(cs, 1.0))
        mfeat = [_z(raw), _rank_norm(raw), raw - raw.max(), _z(nrm), _rank_norm(nrm),
                 (raw == raw.max()).astype(np.float32)]
    else:
        mfeat = [np.zeros(nc, np.float32)] * 6
    if model_logits is not None and nc:
        mr = _rank_norm(cf @ np.asarray(model_logits, np.float32))
        lbest = int(np.argmax(lv)) if lvmax > 0 else -1
        agree = float(1.0 - mr[lbest]) if lbest >= 0 else 0.0
        abest = int(np.argmax(ap)) if apmax > 0 else -1
        agree_a = float(1.0 - mr[abest]) if abest >= 0 else 0.0
        xfeat = [lv * (1.0 - mr), ap * (1.0 - mr), np.full(nc, agree), np.full(nc, agree_a),
                 np.full(nc, agree * lvmax), np.full(nc, float(np.corrcoef(lv, -mr)[0, 1]) if lv.std() > 1e-9 else 0.0)]
    else:
        xfeat = [np.zeros(nc, np.float32)] * 6
    if frag is not None:
        fr = np.asarray(frag, np.float32)
        ffeat = [fr, _rank_norm(fr), fr - fr.max() if nc else fr, _z(fr)]
    else:
        ffeat = [np.zeros(nc, np.float32)] * 4
    return np.column_stack([
        lv, _rank_norm(lv), np.full(nc, lvmax), lv - lvmax, (lv > 0).astype(float),
        ap, _rank_norm(ap), np.full(nc, apmax), ap - apmax,
        a1, best_tan, top_tan, mean_tan, np.full(nc, top_sim),
        np.full(nc, np.log(max(nc, 1))),
        *mfeat, *ffeat, *xfeat,
    ]).astype(np.float32)


# =====================================================================================
#  CELL 8 — patch set (verbatim; final definitions only)
# =====================================================================================
def calibrate_ppm_offset(cfg, L, pool, te):
    """TIER A1 — self-calibrate the precursor offset on the file being predicted (label-free)."""
    rng = np.random.default_rng(cfg.calib_seed)
    mids = te.molecule_id.unique()
    pick = rng.choice(mids, size=min(cfg.calib_n_sample, len(mids)), replace=False)

    saved, cfg.ppm_win = cfg.ppm_win, cfg.calib_search_ppm   # widen so edge hits are not lost
    errs = []
    try:
        for mid in pick:
            sub = te[te.molecule_id == mid]
            nms = sub.nm.values[np.isfinite(sub.nm.values)]
            if not len(nms):
                continue
            target = float(np.median(nms))
            specs = [(r.ms2_mzs, r.ms2_normalized_intensities) for r in sub.itertuples()]
            hits = lib_sim(cfg, L, specs, target)
            if not hits:
                continue
            k, s = max(hits.items(), key=lambda kv: kv[1])
            if s < cfg.calib_sim_thresh:
                continue
            i = pool.k2i.get(k, -1)
            if i < 0:
                continue
            errs.append((target - pool.mass[i]) / pool.mass[i] * 1e6)
    finally:
        cfg.ppm_win = saved

    errs = np.asarray(errs, float)
    if errs.size < cfg.calib_min_pairs:
        print(f"[WARN] only {errs.size} calibration pairs — keeping offset at 0.0")
        return 0.0, errs
    med = float(np.median(errs))
    mad = float(np.median(np.abs(errs - med)))
    print(f"[CALIB] n={errs.size}  median={med:+.3f} ppm  MAD={mad:.3f}  "
          f"p5={np.percentile(errs, 5):+.2f}  p95={np.percentile(errs, 95):+.2f}")
    return med, errs


_TAUTOMERS = None
_TAUT_CACHE: dict = {}


def _metric_key(smi):
    """TIER C2 — memoised tautomer key (RDKit defaults, exactly as the notebook)."""
    global _TAUTOMERS
    if not smi:
        return None
    s = str(smi)
    if s in _TAUT_CACHE:
        return _TAUT_CACHE[s]
    from rdkit import Chem
    if _TAUTOMERS is None:
        from rdkit import RDLogger
        from rdkit.Chem.MolStandardize import rdMolStandardize
        RDLogger.DisableLog('rdApp.*')
        _TAUTOMERS = rdMolStandardize.TautomerEnumerator()
    try:
        mol = Chem.MolFromSmiles(s)
        k = None if mol is None else Chem.MolToInchiKey(_TAUTOMERS.Canonicalize(mol)).split('-')[0]
    except Exception:
        k = None
    _TAUT_CACHE[s] = k
    return k


def _metric_unique_candidates(ordered_smis, topn=25):
    """Fills exactly 25 slots with distinct tautomer-canonical InChIKeys."""
    out, seen = [], set()
    for smi in ordered_smis:
        key = _metric_key(smi)
        if key and key not in seen:
            out.append(smi); seen.add(key)
        elif not key and smi not in seen:
            out.append(smi); seen.add(smi)
        if len(out) == topn:
            break
    for smi in ordered_smis:                      # fallback to avoid empty slots
        if len(out) == topn: break
        if smi not in out: out.append(smi)
    return out


@njit(cache=True, fastmath=True, parallel=True)
def _preclean_fill(off, allmz, allin, uoff, umz, uin, lens, floor, topk, power, ent_weight):
    n = len(off) - 1
    for i in prange(n):
        a = off[i]; b = off[i + 1]
        if b <= a:
            lens[i] = 0
            continue
        cm, cp = _clean(allmz[a:b], allin[a:b], floor, topk, power, ent_weight)
        m = len(cm)
        lens[i] = m
        ua = uoff[i]
        for j in range(m):
            umz[ua + j] = cm[j]
            uin[ua + j] = cp[j]


@njit(cache=True, parallel=True)
def _compact(uoff, umz, uin, lens, coff, cmz, cin):
    for i in prange(len(lens)):
        ua = uoff[i]; ca = coff[i]
        for j in range(lens[i]):
            cmz[ca + j] = umz[ua + j]
            cin[ca + j] = uin[ua + j]


def preclean_library(cfg, L):
    """Runs _clean() once per reference spectrum and stores the result as CSR arrays
    L['coff'], L['cmz'], L['cin']. Frees the raw peak arrays afterwards."""
    t0 = time.time()
    off = L['off']; n = len(off) - 1
    ulen = np.minimum(np.diff(off), cfg.max_peaks)          # upper bound per spectrum
    uoff = np.zeros(n + 1, np.int64); np.cumsum(ulen, out=uoff[1:])
    umz = np.empty(uoff[-1], np.float32); uin = np.empty(uoff[-1], np.float32)
    lens = np.zeros(n, np.int64)
    _preclean_fill(off, L['mz'], L['it'], uoff, umz, uin, lens,
                   cfg.int_floor, cfg.max_peaks, cfg.int_power, cfg.ent_weight)
    coff = np.zeros(n + 1, np.int64); np.cumsum(lens, out=coff[1:])
    cmz = np.empty(coff[-1], np.float32); cin = np.empty(coff[-1], np.float32)
    _compact(uoff, umz, uin, lens, coff, cmz, cin)
    L['coff'], L['cmz'], L['cin'] = coff, cmz, cin
    raw = L['mz'].nbytes + L['it'].nbytes
    del umz, uin, L['mz'], L['it']; gc.collect()
    print(f"[V27 PRECLEAN] {n:,} spectra  raw {raw/1e9:.2f} GB -> cleaned {(cmz.nbytes+cin.nbytes)/1e9:.2f} GB  "
          f"({time.time()-t0:.1f}s)", flush=True)


@njit(cache=True, fastmath=True, parallel=True)
def search_pre(qmz, qp, cand, coff, cmz, cin, tol):
    out = np.zeros(len(cand), np.float32)
    for k in prange(len(cand)):
        c = cand[k]; a = coff[c]; b = coff[c + 1]
        if b <= a: continue
        out[k] = entropy_sim(qmz, qp, cmz[a:b], cin[a:b], tol)
    return out


@njit(cache=True, fastmath=True, parallel=True)
def search_shift_pre(qmz, qp, cand, coff, cmz, cin, tol, shift):
    out = np.zeros(len(cand), np.float32)
    for k in prange(len(cand)):
        c = cand[k]; a = coff[c]; b = coff[c + 1]
        if b <= a: continue
        out[k] = entropy_sim_shift(qmz, qp, cmz[a:b], cin[a:b], tol, shift[k])
    return out


def lib_sim(cfg, L, specs, target):
    """CLASS 1: direct match on pre-cleaned references, with the absolute tolerance floor."""
    tol = max(target * cfg.ppm_win / 1e6, cfg.min_da)
    cand = lib_window(L, target, tol)
    if len(cand) == 0:
        return {}
    agg = {}
    for mz, it in specs:
        qm, qp = clean_spectrum(cfg, mz, it)
        if len(qm) == 0:
            continue
        sc = search_pre(qm, qp, cand, L['coff'], L['cmz'], L['cin'], cfg.mz_tol)
        for c, s in zip(cand, sc):
            k = L['ik'][c]
            if s > agg.get(k, -1.0):
                agg[k] = float(s)
    return agg


def analog_sim(cfg, L, specs, target, rep, rep_key, rep_nm):
    """CLASS 2: mass-shifted entropy match across the wide window, on pre-cleaned references."""
    lo = np.searchsorted(rep_nm, target - cfg.analog_win, 'left')
    hi = np.searchsorted(rep_nm, target + cfg.analog_win, 'right')
    cand = rep[lo:hi]
    if len(cand) == 0:
        return []
    shift = (target - rep_nm[lo:hi]).astype(np.float32)
    ckey = rep_key[lo:hi]
    agg = {}
    for mz, it in specs:
        qm, qp = clean_spectrum(cfg, mz, it)
        if len(qm) == 0:
            continue
        sc = search_shift_pre(qm, qp, cand, L['coff'], L['cmz'], L['cin'], cfg.mz_tol, shift)
        for c, k, s in zip(cand, ckey, sc):
            if s > agg.get(k, -1.0):
                agg[k] = float(s)
    return sorted(agg.items(), key=lambda x: -x[1])[:cfg.n_analog]


# =====================================================================================
#  Predictor wrapper (cell 9 main loop, one molecule per predict() call)
# =====================================================================================
class Exp016V27Predictor(Predictor):
    name = "exp016_v27"

    def __init__(self, pool_npz: str, rankers: str, fp_models: str, device: str = "cpu",
                 torch_threads: int = 0, ppm_offset="calibrate", **v27):
        super().__init__(pool_npz=pool_npz, rankers=rankers, fp_models=fp_models, device=device,
                         torch_threads=torch_threads, ppm_offset=ppm_offset, **v27)
        known = {f.name for f in fields(V27Config)}
        unknown = set(v27) - known
        if unknown:
            raise ValueError(f"unknown V27 params: {sorted(unknown)}")
        self.cfg = V27Config(**v27)
        self.ppm_offset_mode = ppm_offset
        self.paths = dict(pool_npz=pool_npz, rankers=rankers, fp_models=fp_models)
        self.device, self.torch_threads = device, torch_threads
        self.calibration: dict = {}
        self.n_floor_bound = 0

    # ── index ──────────────────────────────────────────────────────────────
    def prepare(self, train: pl.LazyFrame, pool: pd.DataFrame, queries: pd.DataFrame) -> None:
        import joblib
        import torch
        cfg = self.cfg
        if self.torch_threads:
            torch.set_num_threads(self.torch_threads)
        _get_frag_pool(cfg.frag_workers)       # fork frag workers before the model + library load (cell 9)

        self._model = load_neural_models(glob.glob(os.path.join(self.paths["fp_models"], "fp_*.pt")), self.device)
        self._rankers = joblib.load(self.paths["rankers"])
        print(f"[V27] Loaded {len(self._rankers)} GBDT rankers (NFEAT={cfg.nfeat})")

        t0 = time.time()
        npz = np.load(self.paths["pool_npz"], allow_pickle=True)
        self.pool = CandidatePool(npz)
        print(f"[V27] Candidate pool: {len(self.pool.mass):,} structures ({time.time()-t0:.1f}s)")
        self._drop_excluded(queries)

        table = train.select(LIBRARY_COLUMNS).collect().to_arrow()
        self.L = load_library(table)
        del table; gc.collect()
        preclean_library(cfg, self.L)
        self.rep, self.rep_key, self.rep_nm = build_rep(self.L)
        self.ref_keys = {k for k in self.L['ik'] if k}   # structures with spectra in THIS (regime-filtered) library
        print(f"[V27] ref_keys: {len(self.ref_keys):,} structures with reference spectra")

        if self.ppm_offset_mode == "calibrate":
            te = queries.copy()
            te['nm'] = neutral_mass(te.precursor_mz.values.astype(np.float64), te.adduct.values)
            off, errs = calibrate_ppm_offset(cfg, self.L, self.pool, te)
            self.calibration = dict(n_pairs=int(errs.size), offset_ppm=off,
                                    mad=float(np.median(np.abs(errs - np.median(errs)))) if errs.size else None)
        else:
            off = float(self.ppm_offset_mode)
        if abs(off) > cfg.max_abs_offset:
            print(f"[WARN] calibrated offset {off:+.2f} ppm exceeds sanity bound — "
                  f"reverting to 0.0 and widening the window instead")
            off = 0.0
            cfg.ppm_win = max(cfg.ppm_win, 10.0)
        cfg.ppm_offset = off
        print(f"[V27 CFG] PPM_WIN={cfg.ppm_win}  PPM_OFFSET={cfg.ppm_offset:+.3f}  MIN_DA={cfg.min_da}  "
              f"ANALOG_FALLBACK={cfg.use_analog_fallback}", flush=True)
        if cfg.use_analog_fallback:
            raise NotImplementedError("Tier B1 analog fallback was off in the LB run; not ported")

    def _drop_excluded(self, queries: pd.DataFrame) -> None:
        """C3: drop held-out structures from the private pool by raw key AND metric tautomer key.
        Metric keys are only computed inside the widest window any query can reach."""
        ex = set(self.pool_exclude_keys)
        if not ex:
            return
        cfg = self.cfg
        q = queries[["molecule_id"]].copy()
        q["nm"] = neutral_mass(queries.precursor_mz.values.astype(np.float64), queries.adduct.values)
        targets = q.loc[np.isfinite(q["nm"])].groupby("molecule_id")["nm"].median().to_numpy()
        worst_ppm = cfg.ppm_fallback + cfg.max_abs_offset          # widest window predict() can open
        reach = set()
        for m in targets:
            reach.update(self.pool.window(m, worst_ppm, cfg.min_da).tolist())
        by_key = {i for i in range(len(self.pool.keys)) if self.pool.keys[i] in ex}
        by_metric = {i for i in reach - by_key if _metric_key(self.pool.smiles[i]) in ex}
        drop = sorted(by_key | by_metric)
        self.pool.drop(np.asarray(drop, np.int64))
        print(f"[V27] pool_exclude: removed {len(drop)} pool rows ({len(by_metric)} by metric key only)")

    # ── one molecule (cell 9 loop body) ────────────────────────────────────
    def predict(self, rows: pd.DataFrame) -> list[Candidate]:
        cfg, L, pool = self.cfg, self.L, self.pool
        sub = rows.copy()
        sub['nm'] = neutral_mass(sub.precursor_mz.values.astype(np.float64), sub.adduct.values)
        nms = sub.nm.values[np.isfinite(sub.nm.values)]
        smis: list[str] = []
        ranked: list[tuple[str, str, float]] = []
        if len(nms):
            target = float(np.median(nms))
            specs = [(r.ms2_mzs, r.ms2_normalized_intensities) for r in sub.itertuples()]

            lib_hits = lib_sim(cfg, L, specs, target)
            analogs = analog_sim(cfg, L, specs, target, self.rep, self.rep_key, self.rep_nm)

            pool_target = target * (1.0 - cfg.ppm_offset * 1e-6)
            if pool_target * cfg.ppm_win / 1e6 < cfg.min_da:
                self.n_floor_bound += 1
            cand = pool.window(pool_target, cfg.ppm_win, cfg.min_da)
            if len(cand) == 0:
                cand = pool.window(pool_target, cfg.ppm_fallback, cfg.min_da)

            if len(cand):
                cfp = pool.fps(cand)
                lv = np.array([lib_hits.get(pool.keys[c], 0.0) for c in cand], np.float32)
                ids, sims_l = [], []
                for k, s in analogs:
                    i = pool.k2i.get(k, -1)
                    if i >= 0:
                        ids.append(i); sims_l.append(s)
                afp = pool.fps(np.array(ids)) if ids else None
                sims = np.array(sims_l, np.float32)

                ion_mode = float(np.mean([1.0 if m == 'positive' else -1.0 for m in sub.ionization_mode]))
                fsc = frag_scores(cfg, [pool.smiles[c] for c in cand], specs, ion_mode)
                zlog = model_logits(self._model, sub)

                X = rank_features(cfp, lv, afp, sims, zlog, fsc, P_SIM=cfg.sim_power)[:, :cfg.nfeat]
                p = np.mean([m.predict_proba(X)[:, 1] for m in self._rankers], axis=0)
                if cfg.ref_penalty_alpha != 1.0:
                    has_ref = np.fromiter((pool.keys[c] in self.ref_keys for c in cand), bool, len(cand))
                    pen = has_ref & (lv < cfg.ref_penalty_sim)
                    p = np.where(pen, p * cfg.ref_penalty_alpha, p)
                order = np.argsort(-p)

                ordered_raw_smis = [pool.smiles[cand[i]] for i in order]
                smis = _metric_unique_candidates(ordered_raw_smis, topn=cfg.topn)
                ranked = [(str(pool.smiles[cand[i]]), str(pool.keys[cand[i]]), float(p[i])) for i in order]

        if not smis:
            return [Candidate("CCO", "", "fallback")]
        info = {}
        for s, k, sc in ranked:
            info.setdefault(s, (k, sc))
        out = [Candidate(s, info[s][0], "pool", info[s][1]) for s in smis]
        head = set(smis)
        out += [Candidate(s, k, "pool", sc) for s, k, sc in ranked if s not in head]
        return out
