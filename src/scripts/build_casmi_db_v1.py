#!/usr/bin/env python3
"""
build_casmi_db_v1.py - build CASMI26 candidate DB v1 in the exact schema of the V17/V27 pool.

Stages (each resumable; `all` runs them in order):
  inspect   Reverse-engineer the pool's fingerprint spec. Aborts unless 100% bit match.
  coconut   Standardise COCONUT 2.0 (every parsable entry kept; filters only reported).
  pubchem   Stream CID-SMILES.gz in chunks -> filters -> NP-likeness -> parquet shards.
  report    Dedupe PubChem by InChIKey14; print DB size at several NP thresholds.
  build     Needs --np-threshold. FPs + metric (tautomer) keys, union, collapse, write npz.
  verify    Schema check, 100% pool-retention check, FP/key/mass spot-check on new rows.

Typical run:
  python build_casmi_db_v1.py inspect
  python build_casmi_db_v1.py coconut --coconut-file downloads/coconut_csv-09-2026.zip
  python build_casmi_db_v1.py pubchem --download --limit-chunks 4     # smoke test + rate
  python build_casmi_db_v1.py pubchem --download                      # full (resumes)
  python build_casmi_db_v1.py report
  python build_casmi_db_v1.py build --np-threshold 0.5
  python build_casmi_db_v1.py verify

PubChem is never held in memory: it is read line-by-line from the .gz, processed in
fixed-size chunks with bounded in-flight work, and each chunk is committed to disk
(parquet shard + json stats, json written last = commit marker).
"""
import argparse
import contextlib
import datetime as dt
import gzip
import hashlib
import io
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, as_completed, wait
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------
RDKIT_PIN = "2026.03.3"
ALLOWED_Z = {1, 6, 7, 8, 9, 15, 16, 17, 35, 53}  # H C N O F P S Cl Br I
ALLOWED_BRACKET_SYMS = {"H", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I",
                        "c", "n", "o", "p", "s"}
MASS_MIN, MASS_MAX = 150.0, 1200.0
PUBCHEM_URL = "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"
NP_EDGES = np.round(np.arange(-6.0, 6.0001, 0.1), 2)
REPORT_THRESHOLDS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
POOL_SCHEMA = {"fp", "mass", "keys", "smiles", "nbits"}
REJECT_ORDER = ["malformed", "parse", "standardize", "elements", "charged",
                "isotope", "mass", "inchi"]

BRACKET_RE = re.compile(r"\[(\d*)([A-Z][a-z]?|[a-z][a-z]?)")
BRACKET_STRIP_RE = re.compile(r"\[[^\]]*\]")
BARE_BORON_RE = re.compile(r"B(?!r)|b")

_G = {}  # per-process globals (RDKit objects, fp function, NP model)


# --------------------------------------------------------------------------------------
# Small utilities
# --------------------------------------------------------------------------------------
def log(msg):
    print(f"[{dt.datetime.now():%H:%M:%S}] {msg}", flush=True)


def die(msg):
    print(f"\nERROR: {msg}\n", file=sys.stderr, flush=True)
    sys.exit(1)


def sha256_file(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(block), b""):
            h.update(b)
    return h.hexdigest()


def atomic_write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=_json_default))
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def load_json(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def git_info():
    here = Path(__file__).resolve().parent
    for cwd in (here, Path.cwd()):
        try:
            h = subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True,
                               text=True, timeout=10)
            if h.returncode == 0:
                d = subprocess.run(["git", "status", "--porcelain"], cwd=cwd,
                                   capture_output=True, text=True, timeout=10)
                return {"commit": h.stdout.strip(), "dirty": bool(d.stdout.strip())}
        except Exception:
            pass
    return {"commit": None, "dirty": None}


def to_str_list(arr):
    if arr.dtype.kind == "S":
        return [x.decode("ascii") for x in arr]
    return [str(x) for x in arr]


def keys_to_s14(keys):
    return np.array([k.encode("ascii") if isinstance(k, str) else k for k in keys], dtype="S14")


class Timer:
    def __init__(self, work, stage):
        self.path = Path(work) / "timings.json"
        self.stage = stage

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            t = load_json(self.path, {})
            t[self.stage] = round(time.time() - self.t0, 1)
            atomic_write_json(self.path, t)
            log(f"stage '{self.stage}' finished in {t[self.stage] / 60:.1f} min")


def run_bounded(executor, fn, tasks, max_inflight, on_done):
    """Submit tasks lazily with backpressure (Pool.imap would read the whole iterable)."""
    inflight = set()
    for t in tasks:
        inflight.add(executor.submit(fn, t))
        if len(inflight) >= max_inflight:
            done, inflight = wait(inflight, return_when=FIRST_COMPLETED)
            for d in done:
                on_done(d.result())
    for d in as_completed(inflight):
        on_done(d.result())


# --------------------------------------------------------------------------------------
# RDKit primitives (all run inside workers; objects created once per process)
# --------------------------------------------------------------------------------------
def check_rdkit(allow_mismatch):
    import rdkit
    if rdkit.__version__ != RDKIT_PIN:
        msg = (f"RDKit {rdkit.__version__} found, pin is {RDKIT_PIN} "
               f"(pip install rdkit=={RDKIT_PIN}).")
        if not allow_mismatch:
            die(msg + " Use --allow-version-mismatch only for smoke tests.")
        log("WARNING: " + msg)
    return rdkit.__version__


def load_np_model():
    from rdkit import RDConfig
    p = os.path.join(RDConfig.RDContribDir, "NP_Score")
    if not os.path.isfile(os.path.join(p, "npscorer.py")):
        die(f"RDKit Contrib NP_Score not found at {p}")
    if p not in sys.path:
        sys.path.insert(0, p)
    import npscorer
    with contextlib.redirect_stderr(io.StringIO()):
        model = npscorer.readNPModel()
    return npscorer, model


def make_fp_fn(spec):
    """Returns f(mol) -> packed uint8 bytes, exactly as the pool stores them."""
    radius, nbits = spec["radius"], spec["nbits"]
    bitorder = spec.get("bitorder", "big")
    if spec["api"] == "generator":
        from rdkit.Chem import rdFingerprintGenerator as G
        kw = dict(radius=radius, fpSize=nbits, includeChirality=spec["chirality"],
                  countSimulation=spec["count_sim"])
        if spec["features"]:
            kw["atomInvariantsGenerator"] = G.GetMorganFeatureAtomInvGen()
        gen = G.GetMorganGenerator(**kw)

        def bits(mol):
            return gen.GetFingerprintAsNumPy(mol).astype(bool)
    else:
        from rdkit import DataStructs
        from rdkit.Chem import rdMolDescriptors as D

        def bits(mol):
            bv = D.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits,
                                                 useChirality=spec["chirality"],
                                                 useFeatures=spec["features"])
            a = np.zeros((nbits,), dtype=np.uint8)
            DataStructs.ConvertToNumpyArray(bv, a)
            return a.astype(bool)

    def packed(mol):
        return np.packbits(bits(mol), bitorder=bitorder)

    packed.bits = bits
    return packed


def init_worker(fp_spec=None, need_np=False):
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    _G.clear()
    _G["fp"] = make_fp_fn(fp_spec) if fp_spec else None
    _G["nbytes"] = fp_spec["nbytes"] if fp_spec else None
    if need_np:
        _G["npscorer"], _G["npmodel"] = load_np_model()


def prefilter(smi):
    """Cheap string rejects for single-fragment SMILES (salts must go through LFC first)."""
    if "." in smi:
        return None
    for iso, sym in BRACKET_RE.findall(smi):
        if sym not in ALLOWED_BRACKET_SYMS:
            return "elements"
    if BARE_BORON_RE.search(BRACKET_STRIP_RE.sub("", smi)):
        return "elements"
    for iso, sym in BRACKET_RE.findall(smi):
        if iso:
            return "isotope"
    return None


def standardize(smi):
    """parse -> Cleanup -> largest fragment -> uncharge. Returns (mol, reject_reason, n_frags)."""
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize as S
    if "lfc" not in _G:
        _G["lfc"] = S.LargestFragmentChooser(preferOrganic=True)
        _G["unch"] = S.Uncharger()
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, "parse", 0
    try:
        mol = S.Cleanup(mol)
        nfrag = len(Chem.GetMolFrags(mol))
        if nfrag > 1:
            mol = _G["lfc"].choose(mol)
        mol = _G["unch"].uncharge(mol)
        Chem.SanitizeMol(mol)
    except Exception:
        return None, "standardize", 0
    return mol, None, nfrag


def filter_reason(mol):
    """First failing PubChem filter (or None) and the neutral monoisotopic mass."""
    from rdkit.Chem import rdMolDescriptors
    atoms = list(mol.GetAtoms())
    if any(a.GetAtomicNum() not in ALLOWED_Z for a in atoms):
        return "elements", None
    if sum(a.GetFormalCharge() for a in atoms) != 0 or \
            any(a.GetNumRadicalElectrons() for a in atoms):
        return "charged", None
    if any(a.GetIsotope() for a in atoms):
        return "isotope", None
    m = rdMolDescriptors.CalcExactMolWt(mol)
    if not (MASS_MIN <= m <= MASS_MAX):
        return "mass", m
    return None, m


def metric_key(mol):
    """Competition match key: tautomer-canonical InChIKey14.
    Mirrors the Overview description; if the metric notebook uses different
    TautomerEnumerator settings or a pre-standardisation step, mirror it here."""
    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize as S
    if "taut" not in _G:
        _G["taut"] = S.TautomerEnumerator()
    try:
        t = _G["taut"].Canonicalize(mol)
        k = Chem.MolToInchiKey(t)
        return k[:14] if k else ""
    except Exception:
        return ""


# --------------------------------------------------------------------------------------
# Pool loading / fp-spec detection
# --------------------------------------------------------------------------------------
def load_pool(path):
    path = Path(path)
    if not path.exists():
        die(f"pool not found: {path}")
    z = np.load(path, allow_pickle=True)
    files = set(z.files)
    if files != POOL_SCHEMA:
        die(f"pool keys {sorted(files)} != expected {sorted(POOL_SCHEMA)}; "
            "script only knows how to reproduce that schema.")
    pool = {k: z[k] for k in z.files}
    fp = pool["fp"]
    if fp.ndim != 2:
        die(f"pool fp has shape {fp.shape}; expected 2-D (N, packed)")
    pool["fp_u8"] = np.ascontiguousarray(fp).view(np.uint8).reshape(fp.shape[0], -1)
    pool["nbits_int"] = int(np.asarray(pool["nbits"]).ravel()[0])
    with zipfile.ZipFile(path) as zf:
        pool["compressed"] = any(i.compress_type != zipfile.ZIP_STORED for i in zf.infolist())
    return pool


def candidate_specs(nbits):
    for api in ("generator", "legacy"):
        for radius in (2, 3, 1):
            for chirality in (False, True):
                for features in (False, True):
                    for count_sim in ((False, True) if api == "generator" else (False,)):
                        yield dict(api=api, radius=radius, nbits=nbits, chirality=chirality,
                                   features=features, count_sim=count_sim)


def require_spec(work):
    spec = load_json(Path(work) / "pool_spec.json")
    if not spec or spec.get("match_rate") != 1.0:
        die("no verified fingerprint spec (pool_spec.json with match_rate 1.0). Run `inspect`.")
    return spec


def cmd_inspect(a):
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    rdv = check_rdkit(a.allow_version_mismatch)
    init_worker()
    pool = load_pool(a.pool)
    n = len(pool["keys"])
    nbits = pool["nbits_int"]
    log(f"pool: {n:,} rows, nbits={nbits}, fp {pool['fp'].dtype}{pool['fp'].shape}, "
        f"mass {pool['mass'].dtype}, keys {pool['keys'].dtype}, smiles {pool['smiles'].dtype}, "
        f"compressed={pool['compressed']}")
    for k in ("fp", "mass", "keys", "smiles"):
        if len(pool[k]) != n:
            die(f"length mismatch: {k} has {len(pool[k])}, keys has {n}")
    if pool["fp_u8"].shape[1] * 8 < nbits:
        die("packed fp width smaller than nbits")
    mass_sorted = bool(np.all(np.diff(pool["mass"].astype(np.float64)) >= 0))
    n_unique_keys = len(set(to_str_list(pool["keys"])))

    rng = np.random.default_rng(0)
    idx = rng.choice(n, size=min(a.n_check, n), replace=False)
    smiles = to_str_list(pool["smiles"][idx])
    keys = to_str_list(pool["keys"][idx])
    mols, ok_idx = [], []
    for j, s in enumerate(smiles):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            mols.append(m)
            ok_idx.append(j)
    if len(mols) < len(smiles):
        log(f"WARNING: {len(smiles) - len(mols)} sampled pool SMILES failed to parse")
    ok_idx = np.array(ok_idx)
    targets = {bo: np.unpackbits(pool["fp_u8"][idx][ok_idx], axis=1, bitorder=bo)[:, :nbits]
               .astype(bool) for bo in ("big", "little")}

    results = []
    for spec in candidate_specs(nbits):
        try:
            fn = make_fp_fn(spec)
            got = np.stack([fn.bits(m) for m in mols])
        except Exception as e:
            results.append((0.0, spec, f"error: {e}"))
            continue
        for bo, tgt in targets.items():
            rate = float(np.mean(np.all(got == tgt, axis=1)))
            results.append((rate, {**spec, "bitorder": bo}, ""))
    results.sort(key=lambda r: -r[0])
    log("top fingerprint-spec candidates (exact row match rate):")
    for rate, spec, err in results[:6]:
        print(f"   {rate:7.2%}  {spec} {err}")

    # mass definition + key definition checks on the same sample
    mass_calc = np.array([rdMolDescriptors.CalcExactMolWt(m) for m in mols])
    mass_diff = float(np.max(np.abs(mass_calc - pool["mass"][idx][ok_idx].astype(np.float64))))
    k_plain = [Chem.MolToInchiKey(m)[:14] for m in mols]
    k_taut = [metric_key(m) for m in mols]
    keys_ok = [keys[j] for j in ok_idx]
    plain_rate = float(np.mean([x == y for x, y in zip(k_plain, keys_ok)]))
    taut_rate = float(np.mean([x == y for x, y in zip(k_taut, keys_ok)]))

    best_rate, best_spec, _ = results[0]
    spec_out = {**best_spec, "nbytes": int(pool["fp_u8"].shape[1]),
                "storage_dtype": str(pool["fp"].dtype), "match_rate": best_rate,
                "n_checked": len(mols), "rdkit": rdv,
                "pool_mass_def": "CalcExactMolWt(neutral mol from pool SMILES)",
                "pool_mass_max_abs_diff": mass_diff,
                "pool_keys_match_plain_inchikey14": plain_rate,
                "pool_keys_match_tautomer_inchikey14": taut_rate,
                "pool_mass_sorted": mass_sorted, "pool_rows": n,
                "pool_unique_keys": n_unique_keys,
                "pool_dtypes": {k: str(pool[k].dtype) for k in POOL_SCHEMA},
                "pool_compressed": pool["compressed"]}
    atomic_write_json(Path(a.work) / "pool_spec.json", spec_out)
    log(f"mass: max |ExactMolWt - pool mass| = {mass_diff:.2e}; sorted={mass_sorted}")
    log(f"keys: plain InChIKey14 match {plain_rate:.1%}, tautomer-canonical {taut_rate:.1%}; "
        f"unique keys {n_unique_keys:,}/{n:,}")
    if best_rate < 1.0:
        die(f"best fingerprint match only {best_rate:.2%} - NOT proceeding. Check how the "
            "pool FPs were built (pre-standardisation? other FP type?) and extend candidate_specs.")
    if mass_diff > 1e-3:
        die("pool mass is not neutral monoisotopic mass of the stored SMILES; "
            "new entries would be inconsistent. Fix mass definition before proceeding.")
    if max(plain_rate, taut_rate) < 0.99:
        die("pool keys match neither plain nor tautomer InChIKey14 on >=99% of the sample.")
    if taut_rate > plain_rate:
        log("NOTE: pool keys look tautomer-canonical; new rows use plain InChIKey14 in `keys` "
            "(dedupe on metric key is still applied). Adjust if V27 depends on this.")
    log(f"fingerprint spec VERIFIED (100% of {len(mols)}): {best_spec}")


# --------------------------------------------------------------------------------------
# COCONUT
# --------------------------------------------------------------------------------------
def iter_coconut(path, chunk):
    """Yields (ids, smiles) lists from COCONUT CSV / CSV.zip / SDF(.gz)."""
    import pandas as pd
    path = Path(path)
    name = path.name.lower()
    if name.endswith((".sdf", ".sdf.gz")):
        from rdkit import Chem
        fh = gzip.open(path, "rb") if name.endswith(".gz") else open(path, "rb")
        ids, smis = [], []
        for i, m in enumerate(Chem.ForwardSDMolSupplier(fh)):
            if m is None:
                ids.append(f"sdf_{i}")
                smis.append("")
            else:
                ids.append(m.GetProp("identifier") if m.HasProp("identifier") else f"sdf_{i}")
                smis.append(Chem.MolToSmiles(m))
            if len(smis) == chunk:
                yield ids, smis
                ids, smis = [], []
        if smis:
            yield ids, smis
        return
    if name.endswith(".zip"):
        zf = zipfile.ZipFile(path)
        csvs = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not csvs:
            die("no .csv inside COCONUT zip")
        opener = lambda: zf.open(csvs[0])  # noqa: E731
    else:
        opener = lambda: open(path, "rb")  # noqa: E731
    with opener() as f:
        cols = list(pd.read_csv(f, nrows=0).columns)
    smi_col = next((c for c in ("canonical_smiles", "smiles", "SMILES", "absolute_smiles")
                    if c in cols), None)
    id_col = next((c for c in ("identifier", "coconut_id", "id") if c in cols), None)
    if smi_col is None:
        die(f"no SMILES column in COCONUT file; columns: {cols}")
    log(f"COCONUT columns used: smiles={smi_col}, id={id_col}")
    usecols = [c for c in (id_col, smi_col) if c]
    with opener() as f:
        for df in pd.read_csv(f, usecols=usecols, chunksize=chunk, dtype=str,
                              keep_default_na=False):
            ids = df[id_col].tolist() if id_col else [""] * len(df)
            yield ids, df[smi_col].tolist()


def process_coconut_chunk(task):
    from rdkit import Chem
    chunk_id, ids, smis, out_dir = task
    st = Counter()
    rows = {"id": [], "smiles": [], "key14": [], "mass": [], "np": [], "would_fail": []}
    for cid, smi in zip(ids, smis):
        st["input"] += 1
        mol, why, nfrag = standardize(smi) if smi else (None, "parse", 0)
        if mol is None:
            st["rej_" + why] += 1
            continue
        if nfrag > 1:
            st["multi_fragment_largest_kept"] += 1
        k = Chem.MolToInchiKey(mol)
        if not k:
            st["rej_inchi"] += 1
            continue
        reason, m = filter_reason(mol)
        if m is None:
            from rdkit.Chem import rdMolDescriptors
            m = rdMolDescriptors.CalcExactMolWt(mol)
        if reason:
            st["info_would_fail_" + reason] += 1
        st["kept"] += 1
        rows["id"].append(cid)
        rows["smiles"].append(Chem.MolToSmiles(mol))
        rows["key14"].append(k[:14])
        rows["mass"].append(m)
        rows["np"].append(_G["npscorer"].scoreMol(mol, _G["npmodel"]))
        rows["would_fail"].append(reason or "")
    write_shard(Path(out_dir), f"coconut_{chunk_id:05d}", rows,
                {"id": pa.string(), "smiles": pa.string(), "key14": pa.string(),
                 "mass": pa.float64(), "np": pa.float32(), "would_fail": pa.string()}, st, None)
    return chunk_id, dict(st)


def write_shard(out_dir, stem, rows, schema, stats, hist):
    tbl = pa.table({k: pa.array(v, type=schema[k]) for k, v in rows.items()})
    tmp = out_dir / f"{stem}.parquet.tmp"
    pq.write_table(tbl, tmp, compression="zstd")
    os.replace(tmp, out_dir / f"{stem}.parquet")
    atomic_write_json(out_dir / f"{stem}.json",
                      {"stats": stats, "hist": None if hist is None else hist.tolist()})


def cmd_coconut(a):
    check_rdkit(a.allow_version_mismatch)
    if not a.coconut_file:
        die("--coconut-file is required (download the latest COCONUT 2.0 CSV/SDF from "
            "https://coconut.naturalproducts.net/download)")
    out = Path(a.work) / "coconut"
    out.mkdir(parents=True, exist_ok=True)
    src = Path(a.coconut_file)
    with Timer(a.work, "coconut"):
        meta = {"path": str(src.resolve()), "sha256": sha256_file(src),
                "size_bytes": src.stat().st_size,
                "file_mtime": dt.datetime.fromtimestamp(src.stat().st_mtime).isoformat(),
                "download_date": a.coconut_date or "unknown (pass --coconut-date)"}
        atomic_write_json(out / "source.json", meta)
        done = {p.stem for p in out.glob("coconut_*.json")}
        totals = Counter()

        def tasks():
            for i, (ids, smis) in enumerate(iter_coconut(src, a.coconut_chunk)):
                if f"coconut_{i:05d}" in done:
                    continue
                yield (i, ids, smis, str(out))

        def on_done(res):
            cid, st = res
            totals.update(st)
            log(f"coconut chunk {cid} done ({st.get('input', 0):,} rows)")

        with ProcessPoolExecutor(a.workers, initializer=init_worker,
                                 initargs=(None, True)) as ex:
            run_bounded(ex, process_coconut_chunk, tasks(), 2 * a.workers, on_done)
    st = sum_stats(out, "coconut_")
    log(f"COCONUT totals: {dict(st)}")


def sum_stats(d, prefix):
    tot, hist = Counter(), None
    for p in sorted(Path(d).glob(f"{prefix}*.json")):
        j = json.loads(p.read_text())
        tot.update(j["stats"])
        if j.get("hist") is not None:
            h = np.array(j["hist"])
            hist = h if hist is None else hist + h
    return (tot, hist) if prefix.startswith("pub") else tot


# --------------------------------------------------------------------------------------
# PubChem
# --------------------------------------------------------------------------------------
def download(url, dest):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    meta_path = dest.with_name(dest.name + ".source.json")
    if dest.exists() and meta_path.exists():
        log(f"using existing download {dest}")
        return load_json(meta_path)
    part = dest.with_name(dest.name + ".part")
    have = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
    log(f"downloading {url} (resume from {have / 1e9:.2f} GB)")
    with urllib.request.urlopen(req, timeout=120) as r:
        if have and r.status != 206:
            have = 0  # server ignored Range -> restart
        total = int(r.headers.get("Content-Length", 0)) + have
        lastmod = r.headers.get("Last-Modified")
        with open(part, "ab" if have else "wb") as f:
            got, t0 = have, time.time()
            while True:
                b = r.read(1 << 23)
                if not b:
                    break
                f.write(b)
                got += len(b)
                if time.time() - t0 > 30:
                    log(f"  {got / 1e9:.2f}/{total / 1e9:.2f} GB")
                    t0 = time.time()
    os.replace(part, dest)
    meta = {"url": url, "last_modified": lastmod, "downloaded_at": dt.datetime.now(dt.timezone.utc).isoformat(), "size_bytes": dest.stat().st_size}
    atomic_write_json(meta_path, meta)
    return meta


def iter_line_chunks(path, chunk_size):
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        buf, cid = [], 0
        for line in f:
            buf.append(line)
            if len(buf) == chunk_size:
                yield cid, buf
                buf, cid = [], cid + 1
        if buf:
            yield cid, buf


def process_pubchem_chunk(task):
    from rdkit import Chem
    chunk_id, lines, out_dir, np_floor = task
    st = Counter()
    scores_all = []
    rows = {"cid": [], "smiles": [], "key14": [], "mass": [], "np": []}
    for line in lines:
        st["input"] += 1
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 2 or not parts[1]:
            st["rej_malformed"] += 1
            continue
        cid_s, smi = parts[0], parts[1]
        why = prefilter(smi)
        if why:
            st["rej_" + why] += 1
            continue
        mol, why, nfrag = standardize(smi)
        if mol is None:
            st["rej_" + why] += 1
            continue
        if nfrag > 1:
            st["multi_fragment_largest_kept"] += 1
        why, m = filter_reason(mol)
        if why:
            st["rej_" + why] += 1
            continue
        k = Chem.MolToInchiKey(mol)
        if not k:
            st["rej_inchi"] += 1
            continue
        s = float(_G["npscorer"].scoreMol(mol, _G["npmodel"]))
        scores_all.append(s)
        st["pass_filters"] += 1
        if s < np_floor:
            st["below_np_floor"] += 1
            continue
        rows["cid"].append(int(cid_s) if cid_s.isdigit() else -1)
        rows["smiles"].append(Chem.MolToSmiles(mol))
        rows["key14"].append(k[:14])
        rows["mass"].append(m)
        rows["np"].append(s)
    hist, _ = np.histogram(np.clip(scores_all, NP_EDGES[0], NP_EDGES[-1] - 1e-9), bins=NP_EDGES)
    write_shard(Path(out_dir), f"pub_{chunk_id:05d}", rows,
                {"cid": pa.int64(), "smiles": pa.string(), "key14": pa.string(),
                 "mass": pa.float64(), "np": pa.float32()}, st, hist)
    return chunk_id, dict(st)


def cmd_pubchem(a):
    check_rdkit(a.allow_version_mismatch)
    work = Path(a.work)
    out = work / "pubchem"
    out.mkdir(parents=True, exist_ok=True)
    gz = Path(a.pubchem_file) if a.pubchem_file else work.parent / "downloads" / "CID-SMILES.gz"
    if a.download and not a.pubchem_file:
        src_meta = download(PUBCHEM_URL, gz)
    elif gz.exists():
        src_meta = load_json(gz.with_name(gz.name + ".source.json")) or {
            "url": "local file", "path": str(gz.resolve()), "size_bytes": gz.stat().st_size,
            "file_mtime": dt.datetime.fromtimestamp(gz.stat().st_mtime).isoformat()}
    else:
        die(f"{gz} not found; pass --download or --pubchem-file")
    atomic_write_json(out / "source.json", {**src_meta, "np_floor": a.np_floor,
                                             "chunk_size": a.chunk_size})
    done = {p.stem for p in out.glob("pub_*.json")}
    log(f"{len(done)} chunks already done; resuming")
    stats = {"n": 0, "t0": time.time(), "chunks": 0}

    with Timer(a.work, "pubchem" if not a.limit_chunks else "pubchem_smoke"):
        def tasks():
            for cid, lines in iter_line_chunks(gz, a.chunk_size):
                if a.limit_chunks and cid >= a.limit_chunks:
                    return
                if f"pub_{cid:05d}" in done:
                    continue
                yield (cid, lines, str(out), a.np_floor)

        def on_done(res):
            cid, st = res
            stats["n"] += st["input"]
            stats["chunks"] += 1
            rate = stats["n"] / max(time.time() - stats["t0"], 1e-9)
            eta_h = (a.expected_rows - len(done) * a.chunk_size - stats["n"]) / rate / 3600
            log(f"chunk {cid:5d}: {st['input']:,} in, {st.get('pass_filters', 0):,} pass | "
                f"{rate:,.0f} mol/s | ETA ~{max(eta_h, 0):.1f} h")

        with ProcessPoolExecutor(a.workers, initializer=init_worker,
                                 initargs=(None, True)) as ex:
            run_bounded(ex, process_pubchem_chunk, tasks(), 2 * a.workers, on_done)
    tot, _ = sum_stats(out, "pub_")
    log(f"PubChem totals so far: {dict(tot)}")


def funnel(tot):
    """Sequential counts remaining after each filter (each row counted at first failure)."""
    remaining = tot.get("input", 0)
    out = [("input", remaining)]
    for r in REJECT_ORDER:
        remaining -= tot.get("rej_" + r, 0)
        out.append((f"after_{r}", remaining))
    return out


# --------------------------------------------------------------------------------------
# Report: dedupe + NP threshold table
# --------------------------------------------------------------------------------------
def coconut_table(work):
    parts = sorted((Path(work) / "coconut").glob("coconut_*.parquet"))
    if not parts:
        die("no COCONUT shards; run `coconut` first")
    return pa.concat_tables([pq.read_table(p) for p in parts])


def cmd_report(a):
    work = Path(a.work)
    pool = load_pool(a.pool)
    pool_keys = keys_to_s14(to_str_list(pool["keys"]))
    coco = coconut_table(work)
    coco_keys = keys_to_s14(coco.column("key14").to_pylist())
    coco_first = np.unique(coco_keys)
    coco_new = np.setdiff1d(coco_first, pool_keys)
    existing = np.union1d(pool_keys, coco_new)
    n_pool, n_coco_new = len(pool["keys"]), len(coco_new)

    shards = sorted((work / "pubchem").glob("pub_*.parquet"))
    if not shards:
        die("no PubChem shards; run `pubchem` first")
    k_list, np_list, sh_list, row_list = [], [], [], []
    for p in shards:
        sid = int(p.stem.split("_")[1])
        t = pq.read_table(p, columns=["key14", "np"])
        k_list.append(keys_to_s14(t.column("key14").to_pylist()))
        np_list.append(t.column("np").to_numpy())
        sh_list.append(np.full(t.num_rows, sid, dtype=np.int32))
        row_list.append(np.arange(t.num_rows, dtype=np.int32))
    keys = np.concatenate(k_list)
    del k_list
    nps, sh, row = np.concatenate(np_list), np.concatenate(sh_list), np.concatenate(row_list)
    order = np.argsort(keys, kind="stable")  # stable -> lowest CID wins within a key
    ks = keys[order]
    first = np.ones(len(ks), dtype=bool)
    first[1:] = ks[1:] != ks[:-1]
    uidx = order[first]
    n_stored, n_unique = len(keys), len(uidx)
    is_existing = np.isin(keys[uidx], existing)
    new_np = nps[uidx][~is_existing]

    tot, hist = sum_stats(work / "pubchem", "pub_")
    rows = []
    for t in REPORT_THRESHOLDS:
        n_new = int((new_np >= t).sum())
        rows.append({"np_threshold": t, "pubchem_new": n_new,
                     "total_before_taut_collapse": n_pool + n_coco_new + n_new})
    srt = np.sort(new_np)[::-1]
    targets = {}
    for tgt in (5_000_000, 7_500_000, 10_000_000):
        need = tgt - n_pool - n_coco_new
        if need <= 0:
            targets[tgt] = "pool+COCONUT already exceed target"
        elif need > len(srt):
            targets[tgt] = f"unreachable (only {len(srt):,} new PubChem above np_floor)"
        else:
            targets[tgt] = float(srt[need - 1])

    rep = {"funnel_pubchem": funnel(tot), "pubchem_stats": dict(tot),
           "pubchem_rows_stored_above_floor": n_stored,
           "pubchem_unique_key14_above_floor": n_unique,
           "pubchem_key14_duplicates_collapsed": n_stored - n_unique,
           "pubchem_already_in_pool_or_coconut": int(is_existing.sum()),
           "pool_rows": n_pool, "coconut_kept": coco.num_rows,
           "coconut_unique_key14": len(coco_first), "coconut_new_vs_pool": n_coco_new,
           "coconut_stats": dict(sum_stats(work / "coconut", "coconut_")),
           "threshold_table": rows, "threshold_for_total": targets,
           "np_hist_all_passing_filters": {"edges": NP_EDGES.tolist(),
                                           "counts": None if hist is None else hist.tolist()},
           "np_floor": load_json(work / "pubchem" / "source.json", {}).get("np_floor")}
    atomic_write_json(work / "report.json", rep)
    np.savez(work / "pubchem_unique.npz", shard=sh[uidx], row=row[uidx], np=nps[uidx],
             existing=is_existing)

    print("\nPubChem filter funnel (rows remaining):")
    for name, v in rep["funnel_pubchem"]:
        print(f"   {name:<20} {v:>14,}")
    print(f"   {'NP >= floor':<20} {n_stored:>14,}   (floor {rep['np_floor']})")
    print(f"   {'unique InChIKey14':<20} {n_unique:>14,}")
    print(f"   {'new vs pool+COCONUT':<20} {len(new_np):>14,}")
    print(f"\npool {n_pool:,} + COCONUT new {n_coco_new:,} + PubChem new at threshold:")
    print(f"   {'NP thr':>7} {'PubChem new':>13} {'total DB':>13}")
    for r in rows:
        print(f"   {r['np_threshold']:>7.2f} {r['pubchem_new']:>13,} "
              f"{r['total_before_taut_collapse']:>13,}")
    print("\nNP threshold giving total size:")
    for k, v in targets.items():
        print(f"   {k:>11,}: {v if isinstance(v, str) else f'{v:.3f}'}")
    print("\nTotals are before tautomer (metric-key) collapse, which `build` reports.")


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------
def fp_taut_task(task):
    """Recompute FP (optional) + metric key from SMILES. Writes an npz checkpoint."""
    from rdkit import Chem
    kind, out_path, payload, do_fp = task
    if kind == "pubchem":
        shard_path, rows = payload
        t = pq.read_table(shard_path, columns=["cid", "smiles", "key14", "mass"])
        t = t.take(pa.array(rows))
        smiles, key14 = t.column("smiles").to_pylist(), t.column("key14").to_pylist()
        mass, ids = t.column("mass").to_numpy(), t.column("cid").to_pylist()
    else:
        smiles, key14, mass, ids = payload
    n = len(smiles)
    nb = _G["nbytes"]
    fps = np.zeros((n if do_fp else 0, nb), dtype=np.uint8)
    taut, ok = [], np.ones(n, dtype=bool)
    for i, s in enumerate(smiles):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            ok[i] = False
            taut.append("")
            continue
        if do_fp:
            p = _G["fp"](mol)
            fps[i, :len(p)] = p
        taut.append(metric_key(mol))
    res = {"fp": fps, "taut": np.array(taut, dtype="S14"), "ok": ok}
    if kind != "pool":
        res.update(smiles=np.array(smiles, dtype=object), key14=np.array(key14, dtype="S14"),
                   mass=np.asarray(mass, dtype=np.float64), ids=np.array(ids, dtype=object))
    tmp = out_path + ".tmp"
    with open(tmp, "wb") as f:
        np.savez(f, **res)
    os.replace(tmp, out_path)
    return out_path, n, int((~ok).sum())


def run_fp_tasks(a, spec, tasks, label):
    tasks = [t for t in tasks if not Path(t[1]).exists()]
    if not tasks:
        log(f"{label}: all checkpoints present")
        return
    log(f"{label}: {len(tasks)} tasks to run")
    prog = {"n": 0, "t0": time.time()}

    def on_done(res):
        prog["n"] += res[1]
        rate = prog["n"] / max(time.time() - prog["t0"], 1e-9)
        log(f"  {label}: +{res[1]:,} ({prog['n']:,} total, {rate:,.0f} mol/s)")

    with ProcessPoolExecutor(a.workers, initializer=init_worker, initargs=(spec, False)) as ex:
        run_bounded(ex, fp_taut_task, iter(tasks), 2 * a.workers, on_done)


def load_parts(paths, allow_pickle=True):
    parts = [dict(np.load(p, allow_pickle=allow_pickle)) for p in paths]
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else None


def like_dtype(template, values):
    k = template.dtype.kind
    if k == "U":
        w = max(template.dtype.itemsize // 4, max((len(v) for v in values), default=1))
        return np.array(values, dtype=f"<U{w}")
    if k == "S":
        w = max(template.dtype.itemsize, max((len(v) for v in values), default=1))
        return np.array([v.encode("ascii") for v in values], dtype=f"S{w}")
    return np.array(values, dtype=object)


def cmd_build(a):
    rdv = check_rdkit(a.allow_version_mismatch)
    if a.np_threshold is None:
        die("--np-threshold is required (run `report` and pick one)")
    work = Path(a.work)
    spec = require_spec(work)
    rep = load_json(work / "report.json") or die("run `report` first")
    pool = load_pool(a.pool)
    n_pool = len(pool["keys"])
    pool_keys_str = to_str_list(pool["keys"])
    pool_smiles_str = to_str_list(pool["smiles"])
    common = work / "build_common"
    thr_dir = work / f"build_thr{a.np_threshold:+.3f}"
    common.mkdir(parents=True, exist_ok=True)
    thr_dir.mkdir(parents=True, exist_ok=True)
    B = a.fp_chunk

    with Timer(a.work, f"build_thr{a.np_threshold:+.3f}"):
        # 1) pool metric keys (no FP recompute: pool FPs are copied byte-for-byte)
        tasks = [("pool", str(common / f"pool_{i // B:05d}.npz"),
                  (pool_smiles_str[i:i + B], None, None, None), False)
                 for i in range(0, n_pool, B)]
        run_fp_tasks(a, spec, tasks, "pool metric keys")
        pool_taut = load_parts([t[1] for t in tasks])["taut"]

        # 2) COCONUT: dedupe key14, drop anything already in pool
        coco = coconut_table(work)
        ck = coco.column("key14").to_pylist()
        pool_set = set(pool_keys_str)
        seen, sel = set(), []
        for i, k in enumerate(ck):
            if k in pool_set or k in seen:
                continue
            seen.add(k)
            sel.append(i)
        cs = coco.take(pa.array(sel, type=pa.int64()))
        c_smi, c_key = cs.column("smiles").to_pylist(), cs.column("key14").to_pylist()
        c_mass, c_id = cs.column("mass").to_numpy(), cs.column("id").to_pylist()
        tasks = [("coconut", str(common / f"coconut_{i // B:05d}.npz"),
                  (c_smi[i:i + B], c_key[i:i + B], c_mass[i:i + B], c_id[i:i + B]), True)
                 for i in range(0, len(c_smi), B)]
        run_fp_tasks(a, spec, tasks, "COCONUT fp+keys")
        C = load_parts([t[1] for t in tasks]) if tasks else None

        # 3) PubChem selection at threshold
        u = np.load(work / "pubchem_unique.npz")
        m = (u["np"] >= a.np_threshold) & ~u["existing"]
        sh, rw = u["shard"][m], u["row"][m]
        tasks = []
        for sid in np.unique(sh):
            rows = np.sort(rw[sh == sid])
            for j in range(0, len(rows), B):
                tasks.append(("pubchem", str(thr_dir / f"pub_{sid:05d}_{j // B:03d}.npz"),
                              (str(work / "pubchem" / f"pub_{sid:05d}.parquet"),
                               rows[j:j + B].tolist()), True))
        log(f"PubChem selected at NP >= {a.np_threshold}: {int(m.sum()):,}")
        run_fp_tasks(a, spec, tasks, "PubChem fp+keys")
        P = load_parts([t[1] for t in tasks]) if tasks else None

        # 4) metric-key collapse (pool never dropped; precedence pool > COCONUT > PubChem)
        pool_taut_valid = [t for t in pool_taut if t]
        seen_t = set(pool_taut_valid)
        counts = {"pool_rows": n_pool,
                  "pool_metric_key_fail": int(len(pool_taut) - len(pool_taut_valid)),
                  "pool_internal_metric_collisions_kept": len(pool_taut_valid) - len(seen_t)}

        def collapse(D, name):
            if D is None:
                counts.update({f"{name}_candidates": 0, f"{name}_reparse_fail": 0,
                               f"{name}_metric_collapsed": 0, f"{name}_added": 0})
                return np.zeros(0, dtype=np.int64)
            keep = []
            n_fail = int((~D["ok"]).sum())
            n_col = 0
            for i in range(len(D["ok"])):
                if not D["ok"][i]:
                    continue
                t = D["taut"][i]
                if t and t in seen_t:
                    n_col += 1
                    continue
                if t:
                    seen_t.add(t)
                keep.append(i)
            counts.update({f"{name}_candidates": len(D["ok"]), f"{name}_reparse_fail": n_fail,
                           f"{name}_metric_collapsed": n_col, f"{name}_added": len(keep),
                           f"{name}_metric_key_fail": int(sum(1 for t in D["taut"] if not t))})
            return np.array(keep, dtype=np.int64)

        kc = collapse(C, "coconut")
        kp = collapse(P, "pubchem")
        log(f"metric-key collapse: {counts}")

        # 5) assemble in pool schema, stable sort by mass
        new_smiles = [] + ([str(s) for s in C["smiles"][kc]] if C is not None else []) + \
                     ([str(s) for s in P["smiles"][kp]] if P is not None else [])
        new_keys = ([k.decode() for k in C["key14"][kc]] if C is not None else []) + \
                   ([k.decode() for k in P["key14"][kp]] if P is not None else [])
        fp_parts = [pool["fp_u8"]] + ([C["fp"][kc]] if C is not None else []) + \
                   ([P["fp"][kp]] if P is not None else [])
        mass_parts = [pool["mass"]] + \
                     ([C["mass"][kc].astype(pool["mass"].dtype)] if C is not None else []) + \
                     ([P["mass"][kp].astype(pool["mass"].dtype)] if P is not None else [])
        mass_all = np.concatenate(mass_parts)
        order = np.argsort(mass_all, kind="stable")
        del C, P
        fp_all = np.concatenate(fp_parts)[order]
        fp_all = np.ascontiguousarray(fp_all).view(pool["fp"].dtype) \
            .reshape(len(order), pool["fp"].shape[1])
        all_smiles = pool_smiles_str + new_smiles
        all_keys = pool_keys_str + new_keys
        del new_smiles, new_keys
        smiles_arr = like_dtype(pool["smiles"], [all_smiles[i] for i in order])
        keys_arr = like_dtype(pool["keys"], [all_keys[i] for i in order])
        del all_smiles, all_keys
        if smiles_arr.dtype.kind == "U":
            log(f"smiles stored as {smiles_arr.dtype} ({smiles_arr.nbytes / 1e9:.1f} GB in RAM) "
                "to match pool schema")
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        saver = np.savez_compressed if pool["compressed"] else np.savez
        with open(tmp, "wb") as f:
            saver(f, fp=fp_all, mass=mass_all[order], keys=keys_arr, smiles=smiles_arr,
                  nbits=pool["nbits"])
        os.replace(tmp, out)
        n_total = len(order)
        log(f"wrote {out} ({out.stat().st_size / 1e9:.2f} GB, {n_total:,} rows)")

    meta = {
        "db": "casmi_db_v1", "built_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "rows": n_total, "file": str(out), "file_size_bytes": out.stat().st_size,
        "schema": {k: str(v.dtype) for k, v in
                   {"fp": fp_all, "mass": mass_all, "keys": keys_arr, "smiles": smiles_arr}
                   .items()}, "sorted_by": "mass (stable)",
        "fp_spec": spec, "rdkit": rdv, "rdkit_pin": RDKIT_PIN, "python": sys.version,
        "git": git_info(), "script_sha256": sha256_file(__file__),
        "sources": {
            "pool": {"path": str(Path(a.pool).resolve()), "sha256": sha256_file(a.pool),
                     "rows": n_pool},
            "coconut": load_json(work / "coconut" / "source.json"),
            "pubchem": load_json(work / "pubchem" / "source.json")},
        "filters": {
            "pubchem": {"standardisation": "MolFromSmiles -> rdMolStandardize.Cleanup -> "
                                           "LargestFragmentChooser(preferOrganic) -> Uncharger",
                        "elements": "subset of C,H,N,O,P,S,F,Cl,Br,I",
                        "neutral": "net formal charge 0 after uncharging, no radicals",
                        "isotopes": "rejected if any atom has an isotope label",
                        "mass_da": [MASS_MIN, MASS_MAX], "mass_def": "CalcExactMolWt (neutral)",
                        "np_score": "RDKit Contrib NP_Score (Ertl 2008)",
                        "np_floor_stored": rep.get("np_floor"),
                        "np_threshold": a.np_threshold},
            "coconut": "all parsable entries kept (same standardisation, no filters)",
            "pool": "all rows kept byte-for-byte (fp, mass, keys, smiles)",
            "dedupe": "InChIKey14 (pool > COCONUT > PubChem, lowest CID first), then "
                      "tautomer-canonical InChIKey14 metric key (pool rows never dropped)"},
        "counts": {**counts, "pubchem_funnel": rep["funnel_pubchem"],
                   "pubchem_unique_key14_above_floor": rep["pubchem_unique_key14_above_floor"],
                   "pubchem_key14_duplicates_collapsed":
                       rep["pubchem_key14_duplicates_collapsed"],
                   "pubchem_already_in_pool_or_coconut":
                       rep["pubchem_already_in_pool_or_coconut"],
                   "coconut_stats": rep["coconut_stats"],
                   "coconut_new_vs_pool_key14": rep["coconut_new_vs_pool"],
                   "threshold_table": rep["threshold_table"], "final_rows": n_total},
        "timings_s": load_json(work / "timings.json", {}),
    }
    atomic_write_json(out.with_name("meta.json"), meta)
    log(f"wrote {out.with_name('meta.json')}")


# --------------------------------------------------------------------------------------
# Verify
# --------------------------------------------------------------------------------------
def cmd_verify(a):
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors
    check_rdkit(a.allow_version_mismatch)
    work = Path(a.work)
    spec = require_spec(work)
    init_worker(spec)
    with Timer(a.work, "verify"):
        pool = load_pool(a.pool)
        db = load_pool(a.out)  # same schema check + fp_u8 view
        res = {}
        res["schema_keys_equal"] = True  # load_pool enforces
        res["dtype_kinds_equal"] = all(pool[k].dtype.kind == db[k].dtype.kind
                                       for k in ("fp", "mass", "keys", "smiles"))
        res["fp_dtype_shape_equal"] = (pool["fp"].dtype == db["fp"].dtype and
                                       pool["fp"].shape[1] == db["fp"].shape[1])
        res["nbits_equal"] = pool["nbits_int"] == db["nbits_int"]
        res["mass_sorted"] = bool(np.all(np.diff(db["mass"].astype(np.float64)) >= 0))
        dk = to_str_list(db["keys"])
        res["rows"] = len(dk)
        res["duplicate_key14_rows"] = len(dk) - len(set(dk))

        # retention: every pool row must exist with identical (key, smiles, mass, fp)
        pk, ps = to_str_list(pool["keys"]), to_str_list(pool["smiles"])
        pset = set(pk)
        ds = db["smiles"]
        index = {}
        for i, k in enumerate(dk):
            if k in pset:
                index.setdefault((k, str(ds[i])), []).append(i)
        retained = 0
        for i in range(len(pk)):
            cand = index.get((pk[i], ps[i]), [])
            if any(db["mass"][j] == pool["mass"][i] and
                   np.array_equal(db["fp_u8"][j], pool["fp_u8"][i]) for j in cand):
                retained += 1
        res["pool_retained"] = retained
        res["pool_retained_fraction"] = retained / len(pk)

        # spot-check new rows
        new_idx = np.array([i for i, k in enumerate(dk) if k not in pset])
        res["new_rows"] = len(new_idx)
        rng = np.random.default_rng(1)
        samp = rng.choice(new_idx, size=min(a.n_check * 5, len(new_idx)), replace=False) \
            if len(new_idx) else []
        fp_ok = key_ok = mass_ok = 0
        for j in samp:
            mol = Chem.MolFromSmiles(str(ds[j]))
            p = _G["fp"](mol)
            fp_ok += np.array_equal(db["fp_u8"][j][:len(p)], p)
            key_ok += Chem.MolToInchiKey(mol)[:14] == dk[j]
            mass_ok += abs(rdMolDescriptors.CalcExactMolWt(mol) - float(db["mass"][j])) < 1e-3
        n = max(len(samp), 1)
        res.update(new_spotcheck_n=len(samp), new_fp_match=fp_ok / n, new_key_match=key_ok / n,
                   new_mass_match=mass_ok / n)
        res["file_size_bytes"] = Path(a.out).stat().st_size
        res["passed"] = bool(res["dtype_kinds_equal"] and res["fp_dtype_shape_equal"] and
                             res["nbits_equal"] and res["mass_sorted"] and
                             res["pool_retained_fraction"] == 1.0 and
                             (not len(samp) or (fp_ok == key_ok == mass_ok == len(samp))))
    meta_path = Path(a.out).with_name("meta.json")
    meta = load_json(meta_path, {})
    meta["verify"] = res
    meta["timings_s"] = load_json(work / "timings.json", {})
    atomic_write_json(meta_path, meta)
    for k, v in res.items():
        print(f"   {k:<28} {v}")
    if not res["passed"]:
        die("verification FAILED - do not use this DB")
    log("verification passed")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main():
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--pool", default="artifacts/v17_builder/pool_712k_precomputed.npz")
    base.add_argument("--work", default="artifacts/casmi_db_v1/work")
    base.add_argument("--out", default="artifacts/casmi_db_v1/casmi_db_v1.npz")
    base.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    base.add_argument("--allow-version-mismatch", action="store_true")
    base.add_argument("--n-check", type=int, default=200)
    base.add_argument("--coconut-file")
    base.add_argument("--coconut-date", help="date you downloaded COCONUT (for meta.json)")
    base.add_argument("--coconut-chunk", type=int, default=50_000)
    base.add_argument("--pubchem-file", help="local CID-SMILES.gz (skips download)")
    base.add_argument("--download", action="store_true", help="download CID-SMILES.gz")
    base.add_argument("--chunk-size", type=int, default=250_000)
    base.add_argument("--limit-chunks", type=int, default=0, help="smoke test: first N chunks")
    base.add_argument("--expected-rows", type=int, default=122_000_000, help="for ETA only")
    base.add_argument("--np-floor", type=float, default=-2.0,
                      help="rows below this NP score are not stored (histogram still kept)")
    base.add_argument("--np-threshold", type=float)
    base.add_argument("--fp-chunk", type=int, default=20_000)
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("inspect", "coconut", "pubchem", "report", "build", "verify", "all"):
        sub.add_parser(name, parents=[base])
    a = p.parse_args()
    Path(a.work).mkdir(parents=True, exist_ok=True)

    if a.cmd == "inspect":
        cmd_inspect(a)
    elif a.cmd == "coconut":
        require_spec(a.work)
        cmd_coconut(a)
    elif a.cmd == "pubchem":
        require_spec(a.work)
        cmd_pubchem(a)
    elif a.cmd == "report":
        cmd_report(a)
    elif a.cmd == "build":
        cmd_build(a)
    elif a.cmd == "verify":
        cmd_verify(a)
    elif a.cmd == "all":
        cmd_inspect(a)
        cmd_coconut(a)
        cmd_pubchem(a)
        cmd_report(a)
        if a.np_threshold is None:
            log("stopping after report: choose a threshold, then run "
                "`build --np-threshold X` and `verify`")
            return
        cmd_build(a)
        cmd_verify(a)


if __name__ == "__main__":
    main()
