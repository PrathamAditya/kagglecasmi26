import importlib.util, sys
import numpy as np
import torch
from casmi.fpnet.data import SpecData, make_batch
from casmi.fpnet.dreams_encoder import DreaMSEncoder
from casmi.fpnet.dreams_model import DreaMSFP

pkg, slim, final, data = sys.argv[1:5]
sp = importlib.util.spec_from_file_location("dreams_fp_infer", pkg)
mod = importlib.util.module_from_spec(sp)
sp.loader.exec_module(mod)
m1 = mod.load_dreams_fp(slim)
m0 = DreaMSFP(DreaMSEncoder())
m0.load_state_dict(torch.load(final, map_location="cpu", weights_only=False)["model"], strict=True)
m0.eval()
D = SpecData(data)
idx = np.sort(np.random.default_rng(0).choice(D.n, 16, replace=False))
b = {k: torch.from_numpy(v) for k, v in make_batch(D, idx).items()}
args = (b["mz"], b["it"], b["pad"], b["prec"], b["ad"], b["ins"], b["ce"], b["mode"])
with torch.no_grad():
    l0, l1 = m0(*args), m1(*args)
print("max abs logit diff, package vs training code: %.2e" % (l0 - l1).abs().max().item())
print("logits: mean %.3f std %.3f min %.2f max %.2f | mean sigmoid %.4f" % (
    l1.mean().item(), l1.std().item(), l1.min().item(), l1.max().item(), torch.sigmoid(l1).mean().item()))
