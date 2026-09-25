import numpy as np
z = np.load('/run/media/amoeba/HDD/Kaggle/casmi26-project/casmi26/artifacts/rank_train.npz')
for k in z.files: print(k, z[k].shape, z[k].dtype)
print('Y mean', z['Y'].mean(), '  M values', np.unique(z['M'], return_counts=True))