"""Execute the V27 notebook's own code cells, unmodified, on local inputs (exp_016 sanity reference).

uv run python -m casmi.harness.v27_notebook_ref --notebook reference/casmi-four-channel-v27.ipynb \
    --input-root cache/v27_ref/C1/input --workdir cache/v27_ref/C1/run --ppm-offset 1.9

The only interventions, all outside the scoring logic:
  * the literal '/kaggle/input' in cell sources is replaced by --input-root (a directory laid
    out like Kaggle's: comp/{train,test}.parquet + sample_submission.csv, artifacts/*);
  * matplotlib is stubbed (cell 1 imports it, only the dashboard cell 10 uses it; cell 10 is skipped);
  * calibrate_ppm_offset is replaced by a function returning --ppm-offset, so the reference and
    the port use the same frozen offset (the notebook would otherwise calibrate on 20 molecules).
Cells execute in this module's globals so multiprocessing can pickle the frag worker function.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types

ap = argparse.ArgumentParser()
ap.add_argument("--notebook", required=True)
ap.add_argument("--input-root", required=True)
ap.add_argument("--workdir", required=True)
ap.add_argument("--ppm-offset", type=float, required=True)
ap.add_argument("--last-cell", type=int, default=9)
_args = ap.parse_args()

_mpl = types.ModuleType("matplotlib"); _plt = types.ModuleType("matplotlib.pyplot")
_mpl.pyplot = _plt
sys.modules.setdefault("matplotlib", _mpl); sys.modules.setdefault("matplotlib.pyplot", _plt)

_cells = [c for c in json.load(open(_args.notebook))["cells"] if c["cell_type"] == "code"]
_root = os.path.abspath(_args.input_root)
_nb_path = os.path.abspath(_args.notebook)
os.makedirs(_args.workdir, exist_ok=True)
os.chdir(_args.workdir)

_FROZEN_OFFSET = _args.ppm_offset
for _i, _c in enumerate(_cells[: _args.last_cell + 1]):
    _src = "".join(_c["source"]).replace("/kaggle/input", _root)
    if _i == 9:
        def calibrate_ppm_offset(*_a, **_k):  # noqa: F811 — frozen offset, see module docstring
            print(f"[REF] calibrate_ppm_offset overridden -> {_FROZEN_OFFSET:+.4f} ppm")
            return _FROZEN_OFFSET
    print(f"[REF] ---- executing notebook cell {_i} ----", flush=True)
    _file = os.path.abspath(f"nb_cell{_i:02d}.py")      # numba cache=True needs a real source file
    with open(_file, "w") as _f:
        _f.write(_src)
    exec(compile(_src, _file, "exec"), globals())
