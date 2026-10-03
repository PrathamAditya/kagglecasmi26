"""Panel logits on Modal, one A10G container per model.

  modal run src/scripts/modal_panel.py --models frozen,d20k --limit 60     (smoke)
  modal run src/scripts/modal_panel.py                                     (all six models, full)
Outputs land on the volume under /panel_out/ (logits_<name>.npz and spec_<name>.npy).
"""
import modal

GPU = "A10G"
V = "/vol"
MODELS = {
    "frozen": V + "/frozen/fp_single_s2.pt",
    "own20k": V + "/runs/plain1/fpnet_own_final.pt",
    "d5k": V + "/runs/dreams_5k/fpnet_own_final.pt",
    "d20k": V + "/runs/dreams_20k/fpnet_own_final.pt",
    "d30k": V + "/runs/dreams_30k/fpnet_own_final.pt",
    "d40k": V + "/runs/dreams_40k/fpnet_own_final.pt",
}

app = modal.App("casmi-panel")
vol = modal.Volume.from_name("casmi-fpnet-vol")
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("torch", "numpy", "pandas", "pyarrow", "rdkit")
         .add_local_python_source("casmi")
         .add_local_file("src/scripts/panel_logits.py", "/root/panel_logits.py"))


@app.function(image=image, volumes={V: vol}, gpu=GPU, cpu=4, memory=32768, timeout=2 * 3600)
def run(out_name: str, ckpt: str, limit: int, bs: int):
    import os, subprocess
    vol.reload()
    need = [ckpt, V + "/panel/panel_v2.npz", V + "/fpnet_data/meta.npz", V + "/fp_targets.npz", V + "/dreams_encoder.pt"]
    missing = [p for p in need if not os.path.exists(p)]
    if missing:
        raise SystemExit("missing on volume: " + ", ".join(missing))
    cmd = ["python", "/root/panel_logits.py", "--name", out_name, "--ckpt", ckpt, "--limit", str(limit), "--bs", str(bs),
           "--panel", V + "/panel/panel_v2.npz", "--data-dir", V + "/fpnet_data", "--targets", V + "/fp_targets.npz",
           "--encoder", V + "/dreams_encoder.pt", "--out-dir", V + "/panel_out"]
    subprocess.run(cmd, check=True, cwd="/root")
    vol.commit()
    return out_name


@app.local_entrypoint()
def main(models: str = "frozen,own20k,d5k,d20k,d30k,d40k", limit: int = 0, bs: int = 128):
    names = [m.strip() for m in models.split(",") if m.strip()]
    for n in names:
        if n not in MODELS:
            raise SystemExit("unknown model " + n + "; choose from " + ", ".join(MODELS))
    prefix = "modal_smoke_" if limit else "modal_"
    args = [(prefix + n, MODELS[n], limit, bs) for n in names]
    for done in run.starmap(args):
        print("finished", done)
