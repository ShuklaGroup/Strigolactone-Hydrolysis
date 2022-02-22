import numpy as np

pmf = np.load("pmf_kcal_mol.npy")
ts_image = np.argmax(pmf)
barrier = pmf[ts_image]

print(ts_image)
print(barrier)
