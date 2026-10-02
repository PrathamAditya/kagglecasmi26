p = "src/casmi/fpnet/train.py"
s = open(p).read()
reps = [
    ("from casmi.fpnet.model import FPNet\n",
     "from casmi.fpnet.model import FPNet\nfrom casmi.fpnet.dreams_model import build_dreams, fp_loss, make_dreams_opt\n"),
    ('    model = FPNet(NBITS, d=cfg["d"], layers=cfg["layers"]).to(dev)\n',
     '    if cfg["model"] == "dreams":\n        model = build_dreams(cfg).to(dev)\n    else:\n        model = FPNet(NBITS, d=cfg["d"], layers=cfg["layers"]).to(dev)\n'),
    ('    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))\n',
     '    if cfg["model"] == "dreams":\n        opt = make_dreams_opt(model, cfg)\n    scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))\n'),
    ('            g["lr"] = lr_at(step)\n',
     '            g["lr"] = lr_at(step) * g.get("lr_mult", 1.0)\n'),
    ('        loss = F.binary_cross_entropy_with_logits(logits.float(), b["y"])\n',
     '        loss = fp_loss(logits.float(), b["y"], cfg["loss"])\n'),
    ('mol_balance=0.0, train_subset=0)',
     'mol_balance=0.0, train_subset=0,\n                model="fpnet", loss="bce", backbone_lr=5e-5, dreams_ckpt="", grad_ckpt=False)'),
    ('torch.save({"model": model.state_dict(), "step": step, "nbits": NBITS, "d": d, "layers": layers}, path)',
     'torch.save({"model": model.state_dict(), "step": step, "nbits": NBITS, "d": d, "layers": layers,\n                "arch": type(model).__name__}, path)'),
]
for old, new in reps:
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"ABORT: anchor found {n} times (need 1), nothing written:\n{old}")
open(p + ".bak", "w").write(s)
for old, new in reps:
    s = s.replace(old, new)
open(p, "w").write(s)
print("patched", p, "- backup at", p + ".bak")
