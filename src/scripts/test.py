import numpy as np
z = np.load('artifacts/ranker_training_features/rank_train.npz')
print("keys:", z.files)
for k in z.files:
    print(k, z[k].dtype, z[k].shape)
print("G unique:", np.unique(z['G'], return_counts=True))
print("M unique:", np.unique(z['M'], return_counts=True))