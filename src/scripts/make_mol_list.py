import numpy as np, pandas as pd
P = np.load("artifacts/panel/panel_v2.npz", allow_pickle=True)
ik, is_np, nc = P["val_ik14"].astype(str), P["is_np"], P["n_correct"]
keep = nc == 1
df = pd.DataFrame({"inchikey14": ik[keep], "is_np": is_np[keep].astype(int)})
df.to_csv("kaggle_upload/ranker_mols/val_molecules.csv", index=False)
print("kept", len(df), "| NP kept", int(df.is_np.sum()), "| dropped (twins)", int((~keep).sum()))
