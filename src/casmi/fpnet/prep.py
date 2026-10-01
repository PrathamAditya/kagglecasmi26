"""Step 3: turn train.parquet into compact arrays FPNet can train on.

Keeps only spectra that (a) have one of the ten test adducts, (b) belong to a structure in
fp_targets.npz, (c) still have peaks after cleaning. Molecules of the holdout library
(default enveda-np-examples = our local harness set) plus a small hash-chosen fraction of all
other molecules are marked validation, ACROSS ALL LIBRARIES, so validation molecules never
leak into training through another library.

Run:  uv run python -m casmi.fpnet.prep <train.parquet> <fp_targets.npz> <out_dir> [--limit N]
"""
import argparse
import os
import time
import zlib
from multiprocessing import Pool

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from casmi.fpnet.data import ADDUCT_IX, DEFAULT_CE, TEN, ce_raw, instr_family, prep_peaks

COLS = ["inchikey14", "ingest_lib", "adduct", "ionization_mode", "instrument_type",
        "precursor_mz", "collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities",
        "precursor_error_ppm", "base_peak_intensity"]


def _work(args):
    mz, it, prec = args
    return prep_peaks(mz, it, prec)


def _is_val_hash(ik14, frac):
    return (zlib.crc32(ik14.encode()) % 100000) < frac * 100000


def run(train, targets, out_dir, limit=0, val_frac=0.015, holdout_lib="enveda-np-examples",
        workers=None, batch_rows=20000, max_ppm=10.0, min_base_peak=1000.0, min_peaks=1, drop_null_ce=False):
    t0 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    tg = np.load(targets, allow_pickle=True)
    ik14 = [str(x) for x in tg["ik14"]]
    mol_of = {k: i for i, k in enumerate(ik14)}
    print(f"targets: {len(ik14):,} molecules")

    pf = pq.ParquetFile(train)
    total = pf.metadata.num_rows
    print(f"train.parquet: {total:,} rows" + (f" (limit {limit:,})" if limit else ""))

    # pass 1: which molecules are held out (holdout library, any spectrum of that molecule)
    holdout = set()
    seen = 0
    for b in pf.iter_batches(batch_size=200000, columns=["inchikey14", "ingest_lib"]):
        df = b.to_pandas()
        holdout.update(df.inchikey14[df.ingest_lib == holdout_lib].tolist())
        seen += len(df)
        if limit and seen >= limit:
            break
    is_val_mol = np.array([(k in holdout) or _is_val_hash(k, val_frac) for k in ik14])
    hold_mol = np.array([k in holdout for k in ik14])
    print(f"holdout-library molecules: {len(holdout):,} | validation molecules in targets: "
          f"{int(is_val_mol.sum()):,} of {len(ik14):,}")

    # pass 2: clean peaks and collect metadata
    libs = {}
    st = dict(candidates=0, bad_ppm=0, low_base_peak=0, few_peaks=0, null_ce_dropped=0)
    null_ce = {}
    mzs, its, lens = [], [], []
    M = {k: [] for k in ("prec", "ad", "ins", "ce", "mode", "mol", "lib")}
    seen = 0
    with Pool(workers or max(1, os.cpu_count() - 1)) as pool:
        for b in pf.iter_batches(batch_size=batch_rows, columns=COLS):
            df = b.to_pandas()
            seen += len(df)
            df = df[df.adduct.isin(TEN) & df.inchikey14.isin(mol_of)]
            st["candidates"] += len(df)
            # label-quality / test-mirroring filters (NaN compares False, so unknown values are kept)
            ppm = pd.to_numeric(df.precursor_error_ppm, errors="coerce").abs()
            bp = pd.to_numeric(df.base_peak_intensity, errors="coerce")
            bad_ppm = (ppm > max_ppm).to_numpy()
            low_bp = ((bp < min_base_peak).to_numpy()) & ~bad_ppm
            st["bad_ppm"] += int(bad_ppm.sum()); st["low_base_peak"] += int(low_bp.sum())
            df = df[~bad_ppm & ~low_bp]
            if len(df):
                jobs = [(r.ms2_mzs, r.ms2_normalized_intensities, float(r.precursor_mz))
                        for r in df.itertuples()]
                cleaned = pool.map(_work, jobs, chunksize=500)
                for r, (mz, it) in zip(df.itertuples(), cleaned):
                    if len(mz) < max(1, min_peaks):
                        st["few_peaks"] += 1
                        continue
                    ce = ce_raw(r.collision_energy_ev)
                    if np.isnan(ce):
                        null_ce[r.ingest_lib] = null_ce.get(r.ingest_lib, 0) + 1
                        if drop_null_ce:
                            st["null_ce_dropped"] += 1
                            continue
                        ce = DEFAULT_CE
                    mzs.append(mz); its.append(it); lens.append(len(mz))
                    M["prec"].append(float(r.precursor_mz))
                    M["ad"].append(ADDUCT_IX[r.adduct])
                    M["ins"].append(instr_family(r.instrument_type))
                    M["ce"].append(ce)
                    M["mode"].append(1.0 if r.ionization_mode == "positive" else -1.0)
                    M["mol"].append(mol_of[r.inchikey14])
                    M["lib"].append(libs.setdefault(r.ingest_lib, len(libs)))
            print(f"  rows read {seen:,} | spectra kept {len(lens):,} | {time.time() - t0:.0f}s", flush=True)
            if limit and seen >= limit:
                break

    offsets = np.zeros(len(lens) + 1, np.int64)
    offsets[1:] = np.cumsum(lens)
    mol = np.array(M["mol"], np.int32)
    np.save(os.path.join(out_dir, "peaks_mz.npy"), np.concatenate(mzs))
    np.save(os.path.join(out_dir, "peaks_it.npy"), np.concatenate(its))
    np.save(os.path.join(out_dir, "offsets.npy"), offsets)
    lib_names = np.array(sorted(libs, key=libs.get))
    np.savez(os.path.join(out_dir, "meta.npz"),
             prec=np.array(M["prec"], np.float32), ad=np.array(M["ad"], np.int16),
             ins=np.array(M["ins"], np.int8), ce=np.array(M["ce"], np.float32),
             mode=np.array(M["mode"], np.float32), mol=mol,
             lib=np.array(M["lib"], np.int8), is_val=is_val_mol[mol], is_holdout=hold_mol[mol], lib_names=lib_names)

    is_val = is_val_mol[mol]
    print("\nfilter report (spectra with a test adduct and a known structure):")
    print(f"  candidates {st['candidates']:,} | dropped: |precursor ppm| > {max_ppm}: {st['bad_ppm']:,} | "
          f"base peak < {min_base_peak:g}: {st['low_base_peak']:,} | < {min_peaks} peaks: {st['few_peaks']:,} | "
          f"null collision energy (drop_null_ce={drop_null_ce}): {st['null_ce_dropped']:,}")
    print("  spectra with missing collision energy, by library (set to "
          f"{DEFAULT_CE} eV unless --drop-null-ce): {null_ce}")
    print(f"  spectra kept with < 5 peaks: {int((np.array(lens) < 5).sum()):,}")
    print(f"\nDONE in {time.time() - t0:.0f}s -> {out_dir}")
    print(f"spectra: {len(lens):,} (train {int((~is_val).sum()):,} | val {int(is_val.sum()):,})")
    print(f"molecules with spectra: {len(set(mol.tolist())):,} | mean peaks/spectrum {np.mean(lens):.1f}")
    lib_arr = np.array(M["lib"])
    for name, i in sorted(libs.items(), key=lambda x: x[1]):
        print(f"  {name:22s} spectra {int((lib_arr == i).sum()):>9,}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("train"); ap.add_argument("targets"); ap.add_argument("out_dir")
    ap.add_argument("--limit", type=int, default=0, help="stop after ~N rows (smoke test)")
    ap.add_argument("--val-frac", type=float, default=0.015)
    ap.add_argument("--holdout-lib", default="enveda-np-examples")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--max-ppm", type=float, default=10.0, help="drop spectra with |precursor_error_ppm| above this")
    ap.add_argument("--min-base-peak", type=float, default=1000.0, help="drop spectra with raw base peak below this (test set did the same)")
    ap.add_argument("--min-peaks", type=int, default=1)
    ap.add_argument("--drop-null-ce", action="store_true", help="drop spectra with no collision energy instead of using 25 eV")
    a = ap.parse_args()
    run(a.train, a.targets, a.out_dir, a.limit, a.val_frac, a.holdout_lib, a.workers,
        max_ppm=a.max_ppm, min_base_peak=a.min_base_peak, min_peaks=a.min_peaks, drop_null_ce=a.drop_null_ce)