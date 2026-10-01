"""Run FPNet data prep and training on Modal.

One-time setup (from the repo root):
  modal volume create casmi-fpnet-vol
  modal volume put casmi-fpnet-vol <path/to/train.parquet> /train.parquet
  modal volume put casmi-fpnet-vol artifacts/fpnet/fp_targets.npz /fp_targets.npz

Then:
  modal run src/scripts/modal_fpnet.py --stage prep                    # CPU job, writes /fpnet_data on the volume
  modal run src/scripts/modal_fpnet.py --stage smoke                   # 300 GPU steps: proves it runs + gives spec/s
  modal run src/scripts/modal_fpnet.py --stage train --run-name plain1 # the real run (resumes if interrupted)

Download the result:
  modal volume get casmi-fpnet-vol /runs/plain1/fpnet_own_best.pt .
"""
import modal

GPU = "A10G"          # change to "A100-40GB" etc. if you want faster; check Modal's current prices first
VOL_PATH = "/vol"

app = modal.App("casmi-fpnet")
vol = modal.Volume.from_name("casmi-fpnet-vol", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "numpy", "pandas", "pyarrow", "rdkit")
         .add_local_python_source("casmi"))


@app.function(image=image, volumes={VOL_PATH: vol}, cpu=8, memory=32768, timeout=3 * 3600)
def prep(limit: int = 0):
    from casmi.fpnet import prep as P
    vol.reload()
    out = f"{VOL_PATH}/fpnet_data" + ("_small" if limit else "")
    P.run(f"{VOL_PATH}/train.parquet", f"{VOL_PATH}/fp_targets.npz", out, limit=limit)
    vol.commit()


@app.function(image=image, volumes={VOL_PATH: vol}, gpu=GPU, cpu=8, memory=32768, timeout=6 * 3600)
def train(run_name: str, steps: int, batch_size: int, lr: float, eval_every: int, lib_weights: str, data_dir: str,
          instr_dropout: float = 0.0, mol_balance: float = 0.0):
    from casmi.fpnet import train as T
    vol.reload()
    cfg = T.default_cfg()
    cfg.update(data_dir=f"{VOL_PATH}/{data_dir}", targets=f"{VOL_PATH}/fp_targets.npz",
               out_dir=f"{VOL_PATH}/runs/{run_name}", steps=steps, batch_size=batch_size, lr=lr,
               eval_every=eval_every, lib_weights=lib_weights, workers=6, amp="bf16",
               instr_dropout=instr_dropout, mol_balance=mol_balance)
    T.run(cfg, on_save=vol.commit)   # commit on every checkpoint so an interruption loses little
    vol.commit()


@app.local_entrypoint()
def main(stage: str = "smoke", run_name: str = "plain1", steps: int = 20000, batch_size: int = 256,
         lr: float = 3e-4, eval_every: int = 1000, lib_weights: str = "", limit: int = 0,
         instr_dropout: float = 0.0, mol_balance: float = 0.0):
    if stage == "prep":
        prep.remote(limit)
    elif stage == "smoke":
        train.remote("smoke", 300, batch_size, lr, 150, lib_weights, "fpnet_data")
    elif stage == "train":
        train.remote(run_name, steps, batch_size, lr, eval_every, lib_weights, "fpnet_data", instr_dropout, mol_balance)
    else:
        raise SystemExit("stage must be prep, smoke or train")