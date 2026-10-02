import os, sys, torch
from casmi.fpnet.model import ADDUCT_LIST, INSTR_LIST

src_ckpt, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
enc = open("src/casmi/fpnet/dreams_encoder.py").read()
mod = open("src/casmi/fpnet/dreams_model.py").read()
for text, marker in ((enc, "class Fourier("), (enc, "def load_dreams_encoder("),
                     (mod, "DREAMS_PEAKS = 100"), (mod, "def build_dreams(")):
    if text.count(marker) != 1:
        raise SystemExit("ABORT: marker not found exactly once: " + marker)
body_enc = enc[enc.index("class Fourier("):enc.index("def load_dreams_encoder(")]
body_mod = mod[mod.index("DREAMS_PEAKS = 100"):mod.index("def build_dreams(")]
header = (
    '"""Self-contained DreaMS-backbone fingerprint model (plain PyTorch, no casmi imports)."""\n'
    "import math\nimport torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"
    "from torch.utils.checkpoint import checkpoint\n\n"
    "NBITS = 6930\nADDUCT_LIST = %r\nINSTR_LIST = %r\n\n" % (ADDUCT_LIST, INSTR_LIST)
)
footer = (
    "\n\ndef load_dreams_fp(path, device='cpu'):\n"
    "    ck = torch.load(path, map_location='cpu')\n"
    "    m = DreaMSFP(DreaMSEncoder())\n"
    "    m.load_state_dict(ck['state_dict'], strict=True)\n"
    "    return m.to(device).eval()\n"
)
open(os.path.join(out_dir, "dreams_fp_infer.py"), "w").write(header + body_enc + "\n" + body_mod + footer)
ck = torch.load(src_ckpt, map_location="cpu", weights_only=False)
torch.save({"state_dict": ck["model"], "step": ck.get("step"), "arch": ck.get("arch")},
           os.path.join(out_dir, "dreams_fp_5k.pt"))
print("wrote", out_dir, "| trained steps:", ck.get("step"), "| arch:", ck.get("arch"))
