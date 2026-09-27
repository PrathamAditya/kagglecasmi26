import pandas as pd

train = pd.read_parquet("/home/amoeba/Desktop/enveda-CASMI26-molecule-id-mass-spectra/train.parquet")
np_examples = train[train["ingest_lib"] == "enveda-np-examples"]

print(np_examples["molecular_formula"].nunique())
print(np_examples[["molecule_id", "normalized_smiles", "inchikey14", "adduct", "collision_energy_ev"]].head())