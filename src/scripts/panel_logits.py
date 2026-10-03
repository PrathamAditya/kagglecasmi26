import sys, os, time, argparse
sys.path.insert(0, "src")
import numpy as np
import torch
import torch.nn.functional as F
from casmi.fpnet.data import SpecData, make_batch, unpack_targets, NBITS
try:
    from casmi.fpnet.train import forward, to_dev
    print("using forward/to_dev from casmi.fpnet.train")
except Exception as e:
    print("WARNING: could not import forward/to_dev from train.py:", repr(e)[:200], "- using fallback")
    def to_dev(b, dev):
        return {k: v.to(dev) for k, v in b.items()}
    def forward(model, b):
        return model(b["mz"], b["it"], b["pad"], b["prec"], b["ad"], b["ins"], b["ce"], b["mode"])

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--limit", type=int, default=0, help="first N panel molecules only (smoke test)")
ap.add_argument("--bs", type=int, default=64)
ap.add_argument("--panel", default="artifacts/panel/panel_v2.npz")
ap.add_argument("--data-dir", default="artifacts/fpnet_data")
ap.add_argument("--targets", default="artifacts/fpnet/fp_targets.npz")
ap.add_argument("--encoder", default="artifacts/dreams/dreams_encoder.pt")
ap.add_argument("--out-dir", default="artifacts/panel")
a = ap.parse_args()

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", dev)
os.makedirs(a.out_dir, exist_ok=True)
P = np.load(a.panel, allow_pickle=True)
val_mol = P["val_mol"]
n_mol = len(val_mol)
D = SpecData(a.data_dir)
fp = np.load(a.targets)["fp"]

idx = np.flatnonzero(D.is_val)
mp = np.searchsorted(val_mol, D.mol[idx])
assert (val_mol[np.minimum(mp, n_mol - 1)] == D.mol[idx]).all()
if a.limit:
    keep = mp < a.limit
    idx, mp = idx[keep], mp[keep]
    n_mol = a.limit
lens = D.offsets[idx + 1] - D.offsets[idx]
nz = lens > 0
print("val spectra", len(idx), "| empty peak lists skipped", int((~nz).sum()))
idx, mp, lens = idx[nz], mp[nz], lens[nz]
order = np.argsort(lens, kind="stable")
idx, mp = idx[order], mp[order]

ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
if ck.get("arch") == "DreaMSFP":
    from casmi.fpnet.dreams_model import build_dreams
    model = build_dreams({"dreams_ckpt": a.encoder, "grad_ckpt": False})
else:
    from casmi.fpnet.model import FPNet
    model = FPNet(ck["nbits"], d=ck["d"], layers=ck["layers"])
print(model.load_state_dict(ck["model"]))
model.to(dev).eval()
print("arch", ck.get("arch", "FPNet"), "| step", ck.get("step"))

spec_path = os.path.join(a.out_dir, "spec_" + a.name + ".npy")
spec = np.lib.format.open_memmap(spec_path, mode="w+", dtype=np.float16, shape=(len(idx), NBITS))
msum = torch.zeros(n_mol, NBITS, device=dev)
mcnt = torch.zeros(n_mol, device=dev)
bce = 0.0
t0 = time.time()
nb = (len(idx) + a.bs - 1) // a.bs
with torch.no_grad():
    for bi, s in enumerate(range(0, len(idx), a.bs)):
        j = idx[s:s + a.bs]
        m = torch.from_numpy(mp[s:s + a.bs]).to(dev)
        b = make_batch(D, j)
        b = to_dev({k: torch.from_numpy(v) for k, v in b.items()}, dev)
        lg = forward(model, b).float()
        y = torch.from_numpy(unpack_targets(fp, D.mol[j])).to(dev)
        bce += F.binary_cross_entropy_with_logits(lg, y, reduction="sum").item()
        msum.index_add_(0, m, lg)
        mcnt.index_add_(0, m, torch.ones(len(j), device=dev))
        spec[s:s + len(j)] = lg.half().cpu().numpy()
        if bi % 100 == 0 or bi == nb - 1:
            el = time.time() - t0
            print("batch", bi + 1, "of", nb, "| elapsed s", round(el), "| eta s", round(el / (bi + 1) * (nb - bi - 1)), flush=True)
spec.flush()
del spec

ml = (msum / mcnt.clamp(min=1)[:, None]).cpu().numpy()
cnt = mcnt.cpu().numpy()
out = os.path.join(a.out_dir, "logits_" + a.name + ".npz")
np.savez(out, ml=ml, cnt=cnt, spec_mp=mp, spec_idx=idx)
print("saved", out, os.path.getsize(out), "bytes |", spec_path, os.path.getsize(spec_path), "bytes")
print("molecules with 0 spectra scored:", int((cnt == 0).sum()), "of", n_mol)
print("val BCE per bit:", bce / (len(idx) * NBITS))
