"""DreaMS backbone + FPNet-style conditioning head -> 6930 fingerprint logits.
Same forward signature as FPNet, so train.py's forward()/evaluate() work unchanged."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from casmi.fpnet.data import NBITS
from casmi.fpnet.dreams_encoder import DreaMSEncoder, load_slim
from casmi.fpnet.model import ADDUCT_LIST, INSTR_LIST

DREAMS_PEAKS = 100
PREC_INTENS = 1.1


def to_dreams_spec(mz, it, pad, prec, n_peaks=DREAMS_PEAKS):
    """(B,N) padded arrays (sqrt intensities, pad=True where empty) -> (B, n_peaks+1, 2) DreaMS input."""
    B, N = mz.shape
    lin = (it.float() ** 2).masked_fill(pad, 0.0)
    mzf = mz.float().masked_fill(pad, 0.0)
    if N < n_peaks:
        extra = n_peaks - N
        mzf = F.pad(mzf, (0, extra))
        lin = F.pad(lin, (0, extra))
        pad = F.pad(pad, (0, extra), value=True)
    elif N > n_peaks:
        top = lin.topk(n_peaks, dim=1).indices
        sel_pad = pad.gather(1, top)
        sel_mz = mzf.gather(1, top)
        sel_it = lin.gather(1, top)
        order = sel_mz.masked_fill(sel_pad, float("inf")).argsort(dim=1)
        mzf = sel_mz.gather(1, order)
        lin = sel_it.gather(1, order)
        pad = sel_pad.gather(1, order)
    lin = lin / lin.max(dim=1, keepdim=True).values.clamp(min=1e-12)
    real = (~pad).float()
    peaks = torch.stack([mzf * real, lin * real], dim=-1)
    pt = torch.stack([prec.float(), torch.full_like(prec.float(), PREC_INTENS)], dim=-1).unsqueeze(1)
    return torch.cat([pt, peaks], dim=1)


class DreaMSFP(nn.Module):
    def __init__(self, encoder, nbits=NBITS, drop=0.1, n_peaks=DREAMS_PEAKS):
        super().__init__()
        self.encoder = encoder
        self.n_peaks = n_peaks
        d = encoder.d_model
        self.ad = nn.Embedding(len(ADDUCT_LIST), d)
        self.ins = nn.Embedding(len(INSTR_LIST), d)
        self.gl = nn.Linear(3, d)
        nn.init.normal_(self.ad.weight, std=0.02)
        nn.init.normal_(self.ins.weight, std=0.02)
        nn.init.zeros_(self.gl.weight)
        nn.init.zeros_(self.gl.bias)
        self.head = nn.Sequential(nn.Linear(d, 2048), nn.GELU(), nn.Dropout(drop), nn.Linear(2048, nbits))

    def forward(self, mz, it, pad, prec, ad, ins, ce, mode):
        spec = to_dreams_spec(mz, it, pad, prec, self.n_peaks)
        h = self.encoder(spec)[:, 0]
        feats = torch.stack([ce.float() / 100.0, mode.float(), torch.log1p(prec.float()) / 10.0], dim=-1)
        g = self.gl(feats) + self.ad(ad) + self.ins(ins)
        return self.head(h + g)


def build_dreams(cfg):
    enc = load_slim(cfg["dreams_ckpt"])
    model = DreaMSFP(enc)
    if cfg.get("grad_ckpt"):
        enc.transformer_encoder.grad_ckpt = True
    return model


def fp_loss(logits, y, kind="bce"):
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(logits, y)
    if kind == "cosine":   # my guess at a cosine objective, not DreaMS's own head code (unread)
        return (1.0 - F.cosine_similarity(torch.sigmoid(logits), y, dim=-1, eps=1e-8)).mean()
    raise ValueError(kind)


def make_dreams_opt(model, cfg):
    """Backbone gets lr_mult = backbone_lr / lr; head and conditioning get 1.0. Same warmup/cosine shape."""
    mult = cfg["backbone_lr"] / cfg["lr"]
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        enc = name.startswith("encoder.")
        dec = p.ndim >= 2
        g = groups.setdefault((enc, dec), {"params": [], "weight_decay": cfg["wd"] if dec else 0.0,
                                           "lr_mult": mult if enc else 1.0})
        g["params"].append(p)
    return torch.optim.AdamW(list(groups.values()), lr=cfg["lr"], betas=(0.9, 0.999))
