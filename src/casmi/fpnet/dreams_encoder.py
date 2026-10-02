import math, pickle, types
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class _U(pickle.Unpickler):
    def find_class(self, mod, name):
        try:
            return super().find_class(mod, name)
        except (ModuleNotFoundError, AttributeError):
            def setstate(self, s):
                self.__dict__["_state"] = s
            return type(name, (), {"__setstate__": setstate})


_shim = types.ModuleType("shim")
_shim.Unpickler = _U
_shim.Pickler = pickle.Pickler
_shim.UnpicklingError = pickle.UnpicklingError
_shim.load = pickle.load
_shim.dump = pickle.dump


class Fourier(nn.Module):
    def __init__(self, x_min=1e-4, x_max=1000.0):
        super().__init__()
        b = [1 / (x_min * i) for i in range(2, math.ceil(1 / x_min), 2)]
        b += [1 / (1 * i) for i in range(2, math.ceil(x_max), 1)]
        self.register_buffer("b", torch.tensor(b).unsqueeze(0))

    def forward(self, x):
        x = 2 * math.pi * x @ self.b
        return torch.cat((torch.cos(x), torch.sin(x)), dim=-1)


class FF(nn.Module):
    def __init__(self, d_in, d_hid, d_out, depth, dropout):
        super().__init__()
        layers = []
        for l in range(depth):
            a = d_in if l == 0 else d_hid
            b = d_out if l == depth - 1 else d_hid
            layers.append(nn.Linear(a, b))
            if l != depth - 1:
                layers.append(nn.Dropout(dropout))
            layers.append(nn.ReLU())
        self.ff = nn.Sequential(*layers)

    def forward(self, x):
        return self.ff(x)


class MHA(nn.Module):
    def __init__(self, d, h, p):
        super().__init__()
        self.d, self.h, self.p = d, h, p
        self.hd = d // h
        self.scale = self.hd ** -0.5
        self.weights = nn.Parameter(torch.empty(4 * d, d))

    def forward(self, x, mask, bias):
        B, n, d = x.shape
        q, k, v = F.linear(x, self.weights[: 3 * d]).chunk(3, dim=-1)
        q = q.reshape(B, n, self.h, self.hd).transpose(1, 2)
        k = k.reshape(B, n, self.h, self.hd).transpose(1, 2)
        v = v.reshape(B, n, self.h, self.hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)).float() * self.scale + bias.float()
        att = att.masked_fill(mask[:, None, :, None], -1e9)
        att = F.softmax(att, dim=-1).to(v.dtype)
        att = F.dropout(att, p=self.p, training=self.training)
        out = (att @ v).transpose(1, 2).reshape(B, n, d)
        return F.linear(out, self.weights[3 * d:])


class FFN(nn.Module):
    def __init__(self, d, p):
        super().__init__()
        self.p = p
        self.in_proj = nn.Linear(d, 4 * d, bias=False)
        self.out_proj = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        y = F.relu(self.in_proj(x))
        y = F.dropout(y, p=self.p, training=self.training)
        return self.out_proj(y)


class TEnc(nn.Module):
    def __init__(self, d, L, h, att_p, ff_p, res_p):
        super().__init__()
        self.L, self.res_p = L, res_p
        self.grad_ckpt = False
        self.atts = nn.ModuleList([MHA(d, h, att_p) for _ in range(L)])
        self.ffs = nn.ModuleList([FFN(d, ff_p) for _ in range(L)])
        self.scales = nn.ModuleList([nn.LayerNorm(d) for _ in range(2 * L + 1)])

    def layer(self, i, x, mask, bias):
        r = x
        x = self.atts[i](self.scales[2 * i](x), mask, bias)
        x = r + F.dropout(x, p=self.res_p, training=self.training)
        r = x
        x = self.ffs[i](self.scales[2 * i + 1](x))
        return r + F.dropout(x, p=self.res_p, training=self.training)

    def forward(self, x, mask, bias):
        x = F.dropout(x, p=self.res_p, training=self.training)
        for i in range(self.L):
            if self.grad_ckpt and self.training:
                x = checkpoint(self.layer, i, x, mask, bias, use_reentrant=False)
            else:
                x = self.layer(i, x, mask, bias)
        return self.scales[-1](x)


class DreaMSEncoder(nn.Module):
    def __init__(self, n_layers=7, n_heads=8, d_fourier=980, d_peak=44,
                 ff_fourier_d=512, max_mz=1000.0, dropout=0.1):
        super().__init__()
        d = d_fourier + d_peak
        self.d_model, self.max_mz = d, max_mz
        self.fourier_enc = Fourier()
        self.ff_fourier = FF(2 * self.fourier_enc.b.shape[1], ff_fourier_d, d_fourier, 5, dropout)
        self.ff_peak = FF(2, d_peak, d_peak, 1, dropout)
        self.transformer_encoder = TEnc(d, n_layers, n_heads, dropout, dropout, dropout)

    def forward(self, spec):
        # spec: (B, 101, 2) = (m/z raw, linear relative intensity); m/z == 0 is padding
        mask = spec[:, :, 0] == 0
        with torch.autocast(device_type=spec.device.type, enabled=False):
            fe = self.fourier_enc(spec[..., :1].float())
        four = self.ff_fourier(fe)
        scale = torch.tensor([self.max_mz, 1.0], device=spec.device, dtype=spec.dtype)
        peak = self.ff_peak(spec / scale)
        x = torch.cat([peak, four.to(peak.dtype)], dim=-1)
        s = four.float().sum(-1)
        bias = (s[:, :, None] - s[:, None, :])[:, None]
        return self.transformer_encoder(x, mask, bias)


def load_dreams_encoder(path):
    ck = torch.load(path, map_location="cpu", weights_only=False, pickle_module=_shim)
    hp = vars(ck["hyper_parameters"]["args"])
    assert hp["n_layers"] == 7 and hp["n_heads"] == 8 and hp["d_model"] == 1024
    assert hp["d_fourier"] == 980 and hp["d_peak"] == 44 and hp["no_transformer_bias"]
    model = DreaMSEncoder()
    sd = {k: v for k, v in ck["state_dict"].items() if k != "fourier_enc.Fourier frequencies"}
    keys = set(model.state_dict().keys())
    model.load_state_dict({k: v for k, v in sd.items() if k in keys}, strict=True)
    return model, [k for k in sd if k not in keys]


def load_slim(path):
    ck = torch.load(path, map_location="cpu")
    model = DreaMSEncoder()
    model.load_state_dict(ck["state_dict"], strict=True)
    return model
