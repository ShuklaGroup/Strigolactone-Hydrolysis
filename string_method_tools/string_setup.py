import numpy as np
import mdtraj as md

from string_sim_class import StringSim

name = "4ih4_smd_step3"
top = "4ih4_withlig_setup.psf"
n_images = 20

#cv1 = (np.array([[4087, 1440],[1440, 4089]]), np.array([1,-1]))
cv1 = (np.array([4080, 3750]), np.array([1]))
#cv2 = (np.array([1439, 4084]), np.array([1]))
#cv2 = (np.array([[1440, 3750],[1440, 1439]]), np.array([1,-1]))
#cv3 = (np.array([[1439, 1440],[1440, 4089]]), np.array([1,-1]))
colvars = [cv1]

sim = StringSim(name, top, n_images, colvars)
#centers = sim.get_initial("4ih4_smd_madd_step1.dcd")
#print(centers)

#sim.write_colvars(centers, [1,1])

sim.sim_setup(0, smd_traj="4ih4_smd_step3a.dcd")
#sim.sim_setup(0, smd_traj="4ih4_smd_step1.dcd")
