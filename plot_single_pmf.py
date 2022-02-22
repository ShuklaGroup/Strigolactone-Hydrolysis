import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rc('savefig', dpi=500)
matplotlib.rc('font',family='Helvetica-Normal',size=24)

pmf = np.load("pmf_kcal_mol.npy")
pmf_D14 = np.load("../../../AtD14_hydr_path1/string_step1/iteration_26/pmf_kcal_mol.npy")

fig, ax = plt.subplots()
plt.plot(pmf, color='blue')
plt.plot(pmf_D14, color='red')

plt.xlim(0, 20)
#plt.ylim(0, 30)
plt.ylim(-10, 100)

plt.xlabel("Image")
plt.ylabel("PMF (kcal/mol)")

fig.tight_layout()

plt.savefig("HTL7_step1_pmf.png", transparent=True)
