"""Step 1 check: does our copied FPNet (casmi.fpnet.model) behave identically to the
original class inside exp016_v27.py, when both hold the frozen weights?

Run:  uv run src/scripts/check_fpnet_copy.py
Expected: 'strict load OK' twice, then max abs diff = 0.0 (or ~1e-6) for both checkpoints.
"""
import numpy as np
import torch

from casmi.fpnet.model import FPNet as OurFPNet
from casmi.predictors.exp016_v27 import _build_fpnet_classes

CKPTS = ["artifacts/fp_models/fp_merged_m1.pt", "artifacts/fp_models/fp_single_s2.pt"]

OrigFPNet = _build_fpnet_classes()
rng = np.random.default_rng(0)

# one small fake batch: 4 spectra, up to 30 peaks each
B, N = 4, 30
n_peaks = np.array([30, 12, 5, 1])
mz = np.zeros((B, N), np.float32)
it = np.zeros((B, N), np.float32)
pad = np.ones((B, N), bool)
for i, n in enumerate(n_peaks):
    mz[i, :n] = np.sort(rng.uniform(50, 400, n))
    it[i, :n] = rng.uniform(0.05, 1.0, n)
    pad[i, :n] = False
prec = np.full(B, 420.0, np.float32)
args = (torch.tensor(mz), torch.tensor(it), torch.tensor(pad), torch.tensor(prec),
        torch.tensor([0, 1, 7, 25]), torch.tensor([0, 1, 2, 4]),
        torch.tensor([20.0, 35.0, 50.0, 25.0]), torch.tensor([1.0, 1.0, -1.0, 1.0]))

for path in CKPTS:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    orig = OrigFPNet(ck["nbits"], d=ck["d"], layers=ck["layers"]).eval()
    ours = OurFPNet(ck["nbits"], d=ck["d"], layers=ck["layers"]).eval()
    orig.load_state_dict(ck["model"])   # strict by default
    ours.load_state_dict(ck["model"])   # strict by default
    print(path, "-> strict load OK")
    with torch.no_grad():
        a, b = orig(*args), ours(*args)
    print("   output shape", tuple(b.shape), "| max abs diff =", float((a - b).abs().max()))