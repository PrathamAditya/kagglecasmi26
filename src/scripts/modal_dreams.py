"""Matched runs of FPNet vs DreaMS-backbone on Modal (same data, schedule shape, evaluation).

  modal run src/scripts/modal_dreams.py --stage smoke --model dreams
  modal run src/scripts/modal_dreams.py --stage smoke --model dreams --grad-ckpt true   (if it runs out of memory)
  modal run src/scripts/modal_dreams.py --stage smoke --model fpnet                      (speed reference)
"""
import modal

GPU = "A10G"
VOL_PATH = "/vol"

app = modal.App("casmi-dreams")
vol = modal.Volume.from_name("casmi-fpnet-vol", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "numpy", "pandas", "pyarrow", "rdkit")
         .add_local_python_source("casmi"))


@app.function(image=image, volumes={VOL_PATH: vol}, gpu=GPU, cpu=8, memory=32768, timeout=6 * 3600)
def train(run_name: str, model: str, steps: int, batch_size: int, warmup: int, lr: float, backbone_lr: float,
          loss: str, eval_every: int, eval_bs: int, grad_ckpt: str, max_val: int, init_run: str = ""):
    import torch
    from casmi.fpnet import train as T
    vol.reload()
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
    cfg.update(data_dir=f"{VOL_PATH}/fpnet_data", targets=f"{VOL_PATH}/fp_targets.npz",
               out_dir=f"{VOL_PATH}/runs/{run_name}", steps=steps, batch_size=batch_size, warmup=warmup, lr=lr,
               eval_every=eval_every, workers=6, amp="bf16", model=model, loss=loss, backbone_lr=backbone_lr,
               dreams_ckpt=f"{VOL_PATH}/dreams_encoder.pt", grad_ckpt=(grad_ckpt.lower() == "true"),
               eval_bs=eval_bs, max_val_spectra=max_val)
    T.run(cfg, on_save=vol.commit)
    print("peak GPU memory (GB):", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True)
    vol.commit()


@app.local_entrypoint()
def main(stage: str = "smoke", model: str = "dreams", run_name: str = "", steps: int = 5000,
         batch_size: int = 256, warmup: int = 500, lr: float = 3e-4, backbone_lr: float = 5e-5,
         loss: str = "bce", eval_every: int = 1000, eval_bs: int = 64, grad_ckpt: str = "false",
         init_run: str = ""):
    if stage == "smoke":
        train.remote(run_name or f"smoke_{model}", model, 300, batch_size, 50, lr, backbone_lr, loss, 150,
                     eval_bs, grad_ckpt, 2000)
    elif stage == "train":
        train.remote(run_name or f"{model}_{steps}", model, steps, batch_size, warmup, lr, backbone_lr, loss,
                     eval_every, eval_bs, grad_ckpt, 20000, init_run)
    else:
        raise SystemExit("stage must be smoke or train")
