p = "src/casmi/fpnet/train.py"
s = open(p).read()
reps = [
    ("evaluate(model, D, fp, val_idx, cand_mols, dev, amp_dtype)",
     'evaluate(model, D, fp, val_idx, cand_mols, dev, amp_dtype, bs=cfg["eval_bs"])'),
    ("evaluate(model, D, fp, probe_idx, probe_mols, dev, amp_dtype)",
     'evaluate(model, D, fp, probe_idx, probe_mols, dev, amp_dtype, bs=cfg["eval_bs"])'),
    ('dreams_ckpt="", grad_ckpt=False)',
     'dreams_ckpt="", grad_ckpt=False, eval_bs=256)'),
]
for old, new in reps:
    n = s.count(old)
    if n != 1:
        raise SystemExit(f"ABORT: anchor found {n} times (need 1), nothing written:\n{old}")
open(p + ".bak2", "w").write(s)
for old, new in reps:
    s = s.replace(old, new)
open(p, "w").write(s)
print("patched", p, "- backup at", p + ".bak2")
