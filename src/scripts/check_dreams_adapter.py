import argparse, ast
import numpy as np
import torch
from casmi.fpnet.data import SpecData, make_batch
from casmi.fpnet.dreams_model import build_dreams, to_dreams_spec

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--dreams-ckpt", required=True)
ap.add_argument("--spectra-py", required=True)
ap.add_argument("--n", type=int, default=300)
a = ap.parse_args()

want = {"pad_peak_list", "trim_peak_list", "to_rel_intensity", "prepend_precursor_peak"}
ns = {"np": np}
for node in ast.parse(open(a.spectra_py).read()).body:
    if isinstance(node, ast.FunctionDef) and node.name in want:
        exec(compile(ast.Module(body=[node], type_ignores=[]), a.spectra_py, "exec"), ns)
print("DreaMS reference functions found:", sorted(w for w in want if w in ns))

D = SpecData(a.data_dir)
lens = np.diff(D.offsets)


def ref_spec(i, n=100):
    s, e = D.offsets[i], D.offsets[i + 1]
    sp = np.stack([D.mz[s:e].astype(np.float64), D.it[s:e].astype(np.float64) ** 2], axis=1)
    sp = ns["trim_peak_list"](sp.T, n).T
    sp = ns["pad_peak_list"](sp.T, target_len=n).T
    sp = ns["to_rel_intensity"](sp.T).T
    return ns["prepend_precursor_peak"](sp, float(D.prec[i]), 1.1, high=True)


def mine(ids):
    b = make_batch(D, np.asarray(ids))
    t = {k: torch.from_numpy(v) for k, v in b.items()}
    return to_dreams_spec(t["mz"], t["it"], t["pad"], t["prec"], 100).numpy()


rng = np.random.default_rng(0)
idx = np.sort(rng.choice(np.where(lens > 0)[0], size=min(a.n, int((lens > 0).sum())), replace=False))
m_all = mine(idx)
d_batch = [float(np.abs(m_all[k] - ref_spec(i)).max()) for k, i in enumerate(idx)]
d_single = [float(np.abs(mine([i])[0] - ref_spec(i)).max()) for i in idx[:60]]
for name, d, ids in (("batched", d_batch, idx), ("single-spectrum", d_single, idx[:60])):
    bad = [(int(i), int(lens[i]), round(x, 4)) for i, x in zip(ids, d) if x > 1e-4]
    print(f"adapter vs DreaMS preprocessing, {name}: {len(ids)} spectra, max abs diff {max(d):.2e}, "
          f"{len(bad)} above 1e-4 (id, n_peaks, diff): {bad[:5]}")

ne = np.where(lens > 0)[0]
mx = np.zeros(D.n, np.float32)
mx[ne] = np.maximum.reduceat(D.mz, D.offsets[:-1][ne])
for name, sel in (("train", ~D.is_val), ("val", D.is_val)):
    print(f"{name}: {int(sel.sum()):,} spectra | precursor > 1000: {float((D.prec[sel] > 1000).mean()):.4f} "
          f"| any peak > 1000: {float((mx[sel] > 1000).mean()):.4f} | empty: {int((lens[sel] == 0).sum())}")

model = build_dreams({"dreams_ckpt": a.dreams_ckpt, "grad_ckpt": False}).eval()
print("wrapper params: %.1fM" % (sum(p.numel() for p in model.parameters()) / 1e6))
sel = idx[:8]
t = {k: torch.from_numpy(v) for k, v in make_batch(D, sel).items()}
with torch.no_grad():
    logits = model(t["mz"], t["it"], t["pad"], t["prec"], t["ad"], t["ins"], t["ce"], t["mode"])
    e1 = model.encoder(torch.from_numpy(mine(sel)))[:, 0]
    e2 = model.encoder(torch.from_numpy(np.stack([ref_spec(i) for i in sel]).astype(np.float32)))[:, 0]
print("logits:", tuple(logits.shape), "finite:", bool(torch.isfinite(logits).all()),
      "| encoder token-0 diff (adapter vs DreaMS-preprocessed input): %.2e (abs max %.2f)" % (
          (e1 - e2).abs().max().item(), e1.abs().max().item()))
