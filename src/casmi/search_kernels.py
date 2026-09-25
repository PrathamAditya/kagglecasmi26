"""JIT spectral-entropy search. Replaces exp_000/002's pure-Python O(n*m) cosine loop and,
with it, the need for any per-structure candidate cap: every spectrum in the mass window
gets scored directly. Entropy similarity (Li et al. 2021, Nat. Methods) also tends to
outperform plain cosine for library search. Adapted from the community v27 notebook
(scored 0.320 public LB) — same core kernel, wired into our Predictor interface.
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=True, fastmath=True)
def clean(mz, it, floor, topk, ent_weight):
    """Intensity floor -> top-k -> normalize -> optional low-entropy sharpening."""
    n = len(mz)
    if n == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    mx = 0.0
    for i in range(n):
        if it[i] > mx:
            mx = it[i]
    if mx <= 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    thr = floor * mx
    idx = np.empty(n, np.int64)
    c = 0
    for i in range(n):
        if it[i] >= thr:
            idx[c] = i
            c += 1
    if c > topk:
        v = np.empty(c, np.float32)
        for i in range(c):
            v[i] = it[idx[i]]
        o = np.argsort(v)[c - topk:]
        k2 = np.empty(topk, np.int64)
        for i in range(topk):
            k2[i] = idx[o[i]]
        k2.sort()
        idx, c = k2, topk
    om, oi, s = np.empty(c, np.float32), np.empty(c, np.float32), 0.0
    for i in range(c):
        om[i] = mz[idx[i]]
        oi[i] = it[idx[i]]
        s += oi[i]
    if s > 0:
        for i in range(c):
            oi[i] /= s
    if ent_weight:
        S = 0.0
        for i in range(c):
            if oi[i] > 0:
                S -= oi[i] * np.log(oi[i])
        if S < 3.0:                     # low-complexity spectrum: sharpen so a few peaks don't dominate
            w = 0.25 + 0.25 * S
            s2 = 0.0
            for i in range(c):
                oi[i] = oi[i] ** w
                s2 += oi[i]
            if s2 > 0:
                for i in range(c):
                    oi[i] /= s2
    return om, oi


@njit(cache=True, fastmath=True)
def entropy_sim(qmz, qp, cmz, cp, tol) -> float:
    """1 - normalized entropy of the merged spectrum. Peaks within tol are summed, not matched 1:1."""
    i = j = b = 0
    n, m = len(qmz), len(cmz)
    SA = SB = 0.0
    for x in range(n):
        if qp[x] > 0:
            SA -= qp[x] * np.log(qp[x])
    for x in range(m):
        if cp[x] > 0:
            SB -= cp[x] * np.log(cp[x])
    buf = np.empty(n + m, np.float64)
    while i < n and j < m:
        d = qmz[i] - cmz[j]
        if d < -tol:
            buf[b] = qp[i]; i += 1; b += 1
        elif d > tol:
            buf[b] = cp[j]; j += 1; b += 1
        else:
            buf[b] = qp[i] + cp[j]; i += 1; j += 1; b += 1
    while i < n:
        buf[b] = qp[i]; i += 1; b += 1
    while j < m:
        buf[b] = cp[j]; j += 1; b += 1
    tot = 0.0
    for x in range(b):
        tot += buf[x]
    if tot <= 0:
        return 0.0
    SAB = 0.0
    for x in range(b):
        v = buf[x] / tot
        if v > 0:
            SAB -= v * np.log(v)
    return 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)


@njit(cache=True, fastmath=True, parallel=True)
def search_window(qmz, qp, cand_idx, off, allmz, allin, tol, floor, topk, ent_weight):
    """Score the query against every reference spectrum listed in cand_idx. No cap — parallel over candidates."""
    out = np.zeros(len(cand_idx), np.float32)
    for k in prange(len(cand_idx)):
        c = cand_idx[k]
        a, b = off[c], off[c + 1]
        if b <= a:
            continue
        cm, cp = clean(allmz[a:b], allin[a:b], floor, topk, ent_weight)
        if len(cm) == 0:
            continue
        out[k] = entropy_sim(qmz, qp, cm, cp, tol)
    return out


@njit(cache=True, fastmath=True, parallel=True)
def search_window_shift(qmz, qp, cand_idx, off, allmz, allin, tol, floor, topk, ent_weight, shift):
    """Like search_window, but also tries each reference shifted by `shift[k]` (analog / mass-offset match)
    and keeps whichever alignment scores higher. shift = query_neutral_mass - candidate_neutral_mass."""
    out = np.zeros(len(cand_idx), np.float32)
    for k in prange(len(cand_idx)):
        c = cand_idx[k]
        a, b = off[c], off[c + 1]
        if b <= a:
            continue
        cm, cp = clean(allmz[a:b], allin[a:b], floor, topk, ent_weight)
        if len(cm) == 0:
            continue
        direct = entropy_sim(qmz, qp, cm, cp, tol)
        s = shift[k]
        if -1e-3 < s < 1e-3:
            out[k] = direct
            continue
        shifted_mz = cm + s
        out[k] = max(direct, entropy_sim(qmz, qp, shifted_mz, cp, tol))
    return out