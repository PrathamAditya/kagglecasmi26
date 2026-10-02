p = "src/scripts/modal_dreams.py"
s = open(p).read()
copy_block = '''    vol.reload()
    if init_run:
        import os, shutil
        src = f"{VOL_PATH}/runs/{init_run}/last_state.pt"
        dst_dir = f"{VOL_PATH}/runs/{run_name}"
        dst = f"{dst_dir}/last_state.pt"
        os.makedirs(dst_dir, exist_ok=True)
        if not os.path.exists(dst):
            if not os.path.exists(src):
                raise SystemExit(f"missing {src}")
            shutil.copy2(src, dst)
            vol.commit()
            print(f"copied {src} -> {dst} ({os.path.getsize(dst)} bytes)", flush=True)
        else:
            print(f"{dst} already exists: resuming it, not copying", flush=True)
    cfg = T.default_cfg()
'''
reps = [
    ("loss: str, eval_every: int, eval_bs: int, grad_ckpt: str, max_val: int):",
     'loss: str, eval_every: int, eval_bs: int, grad_ckpt: str, max_val: int, init_run: str = ""):'),
    ("    vol.reload()\n    cfg = T.default_cfg()\n", copy_block),
    ('loss: str = "bce", eval_every: int = 1000, eval_bs: int = 64, grad_ckpt: str = "false"):',
     'loss: str = "bce", eval_every: int = 1000, eval_bs: int = 64, grad_ckpt: str = "false",\n         init_run: str = ""):'),
    ("eval_every, eval_bs, grad_ckpt, 20000)", "eval_every, eval_bs, grad_ckpt, 20000, init_run)"),
]
for old, new in reps:
    n = s.count(old)
    if n != 1:
        raise SystemExit("ABORT: anchor found %d times (need 1), nothing written:\n%s" % (n, old))
open(p + ".bak", "w").write(s)
for old, new in reps:
    s = s.replace(old, new)
open(p, "w").write(s)
print("patched", p, "- backup at", p + ".bak")
