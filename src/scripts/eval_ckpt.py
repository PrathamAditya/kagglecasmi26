"""Score one or more FPNet checkpoints (ours and/or the frozen ones) on OUR validation set,
with exactly the same split, spectra sample and metrics that train.py prints.

Run:
  uv run --no-sync python src/scripts/eval_ckpt.py --data-dir artifacts/fpnet_data \
      --targets artifacts/fpnet/fp_targets.npz \
      --ckpt artifacts/fpnet_runs/plain1/fpnet_own_best.pt artifacts/fp_models/fp_single_s2.pt artifacts/fp_models/fp_merged_m1.pt

Reading the frozen models' numbers:
  * near chance  -> its output bit order does NOT match our fingerprint (or the input features differ)
  * far above chance -> bit order matches. But it may have TRAINED on our validation molecules, so treat it
    as an upper bound, not a fair head-to-head. fp_merged_m1 was built for merged multi-energy spectra;
    scoring it per spectrum is not its intended use.
"""
import argparse

import numpy as np
import torch

from casmi.fpnet.data import NBITS, SpecData, make_batch, unpack_targets
from casmi.fpnet.model import FPNet
from casmi.fpnet.train import evaluate, forward, prior_baseline, to_dev


@torch.no_grad()
def scorer_comparison(model, D, fp, val_idx, cand, dev, bs=256):
    """Per-molecule MRR under three different ways of scoring a candidate against the predicted fingerprint.
    A model trained with a different loss can look bad under one scorer and fine under another, so the
    comparison between models is only trustworthy if it holds under all three.
      soft-Tanimoto(prob): what train.py reports
      cosine(prob):        cosine between predicted probabilities and the candidate's bits
      log-likelihood:      sum of the model's logits over the candidate's on-bits (= ranking by -BCE)"""
    model.eval()
    cand_fp = torch.from_numpy(unpack_targets(fp, cand)).to(dev)
    msum = torch.zeros(len(cand), NBITS, device=dev)
    mcnt = torch.zeros(len(cand), device=dev)
    pos = np.searchsorted(cand, D.mol[val_idx])
    for s in range(0, len(val_idx), bs):
        idx = val_idx[s:s + bs]
        b = to_dev({k: torch.from_numpy(v) for k, v in make_batch(D, idx).items()}, dev)
        z = forward(model, b).float()
        p = torch.from_numpy(pos[s:s + bs]).to(dev)
        msum.index_add_(0, p, z)
        mcnt.index_add_(0, p, torch.ones(len(idx), device=dev))
    z = msum / mcnt[:, None].clamp(min=1)
    prob = torch.sigmoid(z)
    inter = prob @ cand_fp.T
    scores = {
        "soft-Tanimoto(prob)": inter / (prob.sum(1, keepdim=True) + cand_fp.sum(1)[None] - inter + 1e-6),
        "cosine(prob)": inter / (prob.norm(dim=1, keepdim=True) * cand_fp.norm(dim=1)[None] + 1e-6),
        "log-likelihood": z @ cand_fp.T,
    }
    hold = np.zeros(len(cand), bool)
    hold[pos] = D.is_holdout[val_idx]
    hold_t = torch.from_numpy(hold).to(dev)
    out = {}
    for name, S in scores.items():
        rank = (S > S.diagonal()[:, None]).sum(1) + 1     # candidate i is molecule i
        inv = 1.0 / rank.float()
        out[name] = (inv.mean().item(), (rank <= 1).float().mean().item(), (rank <= 10).float().mean().item(),
                     inv[hold_t].mean().item() if hold.any() else float("nan"))
    return out

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", required=True)
ap.add_argument("--targets", required=True)
ap.add_argument("--ckpt", nargs="+", required=True)
ap.add_argument("--max-val-spectra", type=int, default=20000)
ap.add_argument("--device", default="auto")
a = ap.parse_args()

dev = torch.device(a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
D = SpecData(a.data_dir)
fp = np.load(a.targets, allow_pickle=True)["fp"]
train_idx = np.where(~D.is_val)[0]
val_all = np.where(D.is_val)[0]
rng = np.random.default_rng(0)   # same recipe as train.run(), so the same 20,000 validation spectra
val_idx = np.sort(rng.choice(val_all, size=min(a.max_val_spectra, len(val_all)), replace=False))
cand = np.unique(D.mol[val_idx])
print(f"device {dev} | val spectra {len(val_idx):,} | val molecules {len(cand):,}")

pm, ph = prior_baseline(D, fp, val_idx, cand, train_idx, dev)
print(f"PRIOR-ONLY BASELINE: per-molecule MRR {pm:.4f}" + (f" | natural-product proxy {ph:.4f}" if ph is not None else ""))

for path in a.ckpt:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = FPNet(ck["nbits"], d=ck["d"], layers=ck["layers"]).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    r = evaluate(model, D, fp, val_idx, cand, dev, None)
    print(f"\n{path} (step {ck.get('step')})")
    print(f"  val_loss {r['val_loss']:.5f} | per-spectrum MRR {r['mrr']:.4f} top1 {r['top1']:.3f} top10 {r['top10']:.3f}")
    print(f"  per-MOLECULE MRR {r['mol_mrr']:.4f} top1 {r['mol_top1']:.3f} top10 {r['mol_top10']:.3f} "
          f"(chance {r['chance_mrr']:.4f})")
    if r["hold_mrr"] is not None:
        print(f"  natural-product proxy MRR {r['hold_mrr']:.4f} on {r['n_hold']} molecules")
    print("  scorer comparison (per-molecule MRR | top1 | top10 | natural-product proxy MRR):")
    for name, (m, t1, t10, npm) in scorer_comparison(model, D, fp, val_idx, cand, dev).items():
        print(f"    {name:22s} {m:.4f} | {t1:.3f} | {t10:.3f} | {npm:.4f}")
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()