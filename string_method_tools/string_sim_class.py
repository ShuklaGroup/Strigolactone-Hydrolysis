import numpy as np
import scipy
import scipy.interpolate
import mdtraj as md

import os

class StringSim(object):

    def __init__(self, name, top, n_images, colvars, n_reps=5, temperature=300):
        self.name = name
        self.topfile = top
        self.n_images = n_images
        self.colvars = colvars #Currently only supports linear combinations of distances!
        self.n_reps = n_reps
        self.temperature = 300

    def sim_setup(self, iteration, smd_traj="smd.dcd", fix_ends=False):

        cwd = os.getcwd()

        #Make directories
        if not os.path.isdir("iteration_%d"%iteration):
            os.mkdir("iteration_%d"%iteration)

        os.chdir("iteration_%d"%iteration)

        print(os.getcwd())

        for i in range(self.n_images):
            if not os.path.isdir("image_%d"%i):
                os.mkdir("image_%d"%i)

            os.chdir("image_%d"%i)

            for j in range(self.n_reps):
                if not os.path.isdir("rep_%d"%j):
                    os.mkdir("rep_%d"%j)
                if not os.path.isdir("rep_%d/qmmm_exec"%j):
                    os.mkdir("rep_%d/qmmm_exec"%j)

            os.chdir("..")

        if iteration == 0:
            print("Computing initial centers from SMD trajectory...")
            centers = self.get_initial("../%s"%smd_traj)

        else:
            print("Computing updated centers from previous iteration...")
            centers = self.update_centers(iteration, fix_ends=fix_ends)

            print(os.getcwd())

            self.write_coor(iteration)

        #print("Computing force constants...")
        #force_constants = self.compute_force_constants(centers)

        force_constants = 100*np.ones(np.shape(centers))

        print("Writing colvars files...")
        self.write_colvars(centers, force_constants)

        print("Writing simulation config files...")
        if iteration == 0:
            self.write_sim_config(iteration)
        else:
            self.write_sim_config_restart(iteration)

        #print("Writing simulation config files...")
        #self.write_sim_config_drift(iteration)

        #print("Writing Blue Waters job scripts...")
        #self.write_job_script_bw(iteration)

    def get_initial(self, smd_traj, initial_reparam=False):
        """Extract initial images from SMD trajectory and compute initial centers"""

        traj = md.load(smd_traj, top="../sys/%s"%self.topfile)
        indices = np.round(np.linspace(0, len(traj)-1, self.n_images)).astype(int)
        frames = traj[indices]

        for i in range(len(indices)):
            frames[i].save_pdb("./image_%d/%s_image%d.pdb"%(i, self.name, i))

        #Compute initial restraint centers
        centers = np.zeros((len(self.colvars), len(indices)))

        for i in range(len(self.colvars)):
            pairs = self.colvars[i][0]
            coeff = self.colvars[i][1]

            if len(np.shape(pairs)) == 1:
                pairs = np.reshape(pairs, (1,2))

            dist = 10*md.compute_distances(frames, pairs-1)

            try:
                centers[i,:] = np.matmul(dist, coeff)
            except:
                centers[i,:] = dist*coeff

        np.savetxt("centers_iter0.txt", centers)

        return centers

    def update_centers(self, iteration, fix_ends=False):

        drifted_centers = self._compute_drift(iteration, fix_ends=fix_ends)

        new_centers = np.zeros(np.shape(drifted_centers))

        if np.shape(drifted_centers)[1] < 2:
            print(drifted_centers)
            new_centers = drifted_centers

        elif np.shape(drifted_centers)[1] == 2:
            curve_func = scipy.interpolate.interp1d(drifted_centers[:,0], drifted_centers[:,1], kind='linear') #Cubic spline interp
            x_grid = np.linspace(drifted_centers[0,0], drifted_centers[-1,0], 2000) #Fine grid along curve
            grid_spacing = x_grid[1] - x_grid[0]
            curve_points = curve_func(x_grid)

            #Calculate cumulative arc lengths
            c_arc_length = np.zeros(np.shape(x_grid)[0])
            for i in range(np.shape(x_grid)[0] - 1):
                c_arc_length[i+1] = c_arc_length[i] + np.sqrt(grid_spacing**2 + (curve_points[i+1] - curve_points[i])**2)

            equal_arc_lengths = np.linspace(c_arc_length[0], c_arc_length[-1], np.shape(drifted_centers)[0])

            for i in range(np.shape(drifted_centers)[0]):
                ind = np.argmin(np.abs(c_arc_length - equal_arc_lengths[i]))
                new_centers[i,0] = x_grid[ind]
                new_centers[i,1] = curve_func(x_grid[ind])

        else:
            #curve_func = scipy.interpolate.interp2d(drifted_centers[:,0], drifted_centers[:,1], drifted_centers[:,2], kind='cubic')
            x_grid = np.linspace(drifted_centers[0,0], drifted_centers[-1,0], 2000)
            y_grid = np.linspace(drifted_centers[0,1], drifted_centers[-1,1], 2000)
            x_grid_spacing = x_grid[1] - x_grid[0]
            y_grid_spacing = y_grid[1] - y_grid[0]

            curve_points = scipy.interpolate.griddata(drifted_centers[:,0:2], drifted_centers[:,2], np.vstack((x_grid, y_grid)).T, method='linear')
            print(np.shape(x_grid))
            print(np.shape(curve_points))
            #curve_func = scipy.interpolate.interp2d(drifted_centers[:,0], drifted_centers[:,1], drifted_centers[:,2], kind='cubic')

            #curve_points = curve_func(x_grid, y_grid)
            #curve_points = np.zeros(np.shape(x_grid)[0])
            #for i in range(np.size(curve_points)):
            #    curve_points[i] = curve_func(x_grid[i], y_grid[i])

            #Calculate cumulative arc lengths
            c_arc_length = np.zeros(np.shape(x_grid)[0])

            for i in range(np.shape(x_grid)[0] - 1):

                c_arc_length[i+1] = c_arc_length[i] + np.sqrt(x_grid_spacing**2 + y_grid_spacing**2 + (curve_points[i+1] - curve_points[i])**2)

            equal_arc_lengths = np.linspace(c_arc_length[0], c_arc_length[-1], np.shape(drifted_centers)[0])

            for i in range(np.shape(drifted_centers)[0]):
                ind = np.argmin(np.abs(c_arc_length - equal_arc_lengths[i]))
                new_centers[i,0] = x_grid[ind]
                new_centers[i,1] = y_grid[ind]
                #new_centers[i,2] = curve_func(x_grid[ind], y_grid[ind])
                new_centers[i,2] = curve_points[ind]

            #TEMP
            #new_centers = drifted_centers

        np.savetxt("centers_iter%d.txt"%iteration, new_centers)

        print("NEW CENTERS")
        print(new_centers)

        return new_centers

    def smooth_new_centers(self, old_centers, new_centers):
        
        center_shift = new_centers - old_centers
        if np.max(center_shift) > 0.5:
            ratio = np.max(center_shift)/0.5
            new_centers = old_centers + ratio*center_shift
        else:
            pass

        return new_centers
    
    def _compute_drift(self, iteration, fix_ends = False):

        old_centers = np.loadtxt("../iteration_%d/centers_iter%d.txt"%(iteration-1, iteration-1))

        if len(np.shape(old_centers)) == 1:
            colvars_avg = old_centers.reshape((self.n_images,1))
            return colvars_avg

        elif np.shape(old_centers)[0] < np.shape(old_centers)[1]:
            old_centers = old_centers.T

        print("OLD CENTERS")
        print(old_centers)
       
        #Load centers from colvars
        colvars_avg = np.zeros((self.n_images, len(self.colvars))) #n_images x n_colvars
        sample_count = np.zeros(np.shape(colvars_avg)) #n_images x n_colvars

        for i in range(self.n_images):
            for j in range(self.n_reps):
                try:
                    if iteration == 1:
                        cv = np.loadtxt("../iteration_%d/image_%d/rep_%d/%s_im%d_rep%d.colvars.traj"%(iteration-1, i, j, self.name, i, j), usecols=list(range(1,len(self.colvars)+1))) #n_samples x n_colvars
                    else:
                        cv = np.loadtxt("../iteration_%d/image_%d/rep_%d/%s_im%d_rep%d-%d.colvars.traj"%(iteration-1, i, j, self.name, i, j, iteration-1), usecols=list(range(1,len(self.colvars)+1))) #n_samples x n_colvars


                    colvars_avg[i,:] += np.sum(cv, axis=0)
                    sample_count[i,:] += np.shape(cv)[0]
                except:
                    pass

            colvars_avg[i,:] /= sample_count[i,:]

        print("DRIFTED CENTERS")
        print(colvars_avg)

        if fix_ends:
            colvars_avg[0,:] = old_centers[0,:]
            colvars_avg[-1,:] = old_centers[-1,:]

            print("FIXED ENDS")
            print(colvars_avg)

        return colvars_avg

    def compute_force_constants(self, centers, scaled=True):

        if scaled:
            colvars_range = np.ones(len(self.colvars))
        else:
            colvars_range = np.zeros(len(self.colvars)) #Max - min for each dimension
            for i in range(len(self.colvars)):
                colvars_range[i] = np.max(centers[i,:]) - np.min(centers[i,:])

        force_constants = np.zeros(len(self.colvars))
        for i in range(len(self.colvars)):
            spacing = colvars_range[i]/(self.n_images-1)
            force_constants[i] = 0.00198*self.temperature/(0.5*spacing)**2 

        return force_constants

    def write_coor(self, iteration):

        for image in range(self.n_images):
            for rep in range(self.n_reps):
                if iteration == 1:
                    try:
                        if not os.path.isfile("../iteration_0/image_%d/rep_%d/%s_im%d_rep%d.coor"%(image, rep, self.name, image, rep)):
                            tr = md.load("../iteration_0/image_%d/rep_%d/%s_im%d_rep%d.dcd"%(image, rep, self.name, image, rep), top="../sys/%s"%self.topfile)
                            tr[-1].save_pdb("../iteration_0/image_%d/rep_%d/%s_im%d_rep%d.coor"%(image, rep, self.name, image, rep))
                    except:
                        print("WARNING: No coor file for image %d rep %d"%(image, rep))
 
                else:
                    try:
                        if not os.path.isfile("../iteration_%d/image_%d/rep_%d/%s_im%d_rep%d-%d.coor"%(iteration-1, image, rep, self.name, image, rep, iteration-1)):
                            tr = md.load("../iteration_%d/image_%d/rep_%d/%s_im%d_rep%d-%d.dcd"%(iteration-1, image, rep, self.name, image, rep, iteration-1), top="../sys/%s"%self.topfile)
                            tr[-1].save_pdb("../iteration_%d/image_%d/rep_%d/%s_im%d_rep%d-%d.coor"%(iteration-1, image, rep, self.name, image, rep, iteration-1))
                    except:
                        print("WARNING: No coor file for image %d rep %d"%(image, rep))
                
    def write_colvars(self, centers, force_constants):

        print(np.shape(centers))
        print(self.n_images)

        if np.shape(centers)[0] == self.n_images:
            centers = centers.T
        
        for image in range(self.n_images):

            with open("./image_%d/%s_im%d_colvars.conf"%(image, self.name, image), 'w') as colvars_file:

                #Loop through and write all the colvars
                for i in range(len(self.colvars)):
                    colvars_file.write("colvar { \n")
                    colvars_file.write("    name cv%d\n"%i)

                    #If more than one distance in a colvar, write custom function to compute linear combination
                    if len(self.colvars[i][1]) > 1:
                        cfstring = ""

                        for j in range(len(self.colvars[i][1])):
                            if self.colvars[i][1][j] != 1:
                                cfstring += "%d*"%self.colvars[i][1][j]
                            cfstring += "r%d"%j
                            if j+1 < len(self.colvars[i][1]):
                                if self.colvars[i][1][j+1] > 0:
                                    cfstring += "+"

                        colvars_file.write("    customFunction %s \n"%cfstring)

                    #Loop through atom pairs and write distances

                    if len(self.colvars[i][1]) == 1:
                        colvars_file.write("    distance { \n")
                        colvars_file.write("        name r0 \n")
                        colvars_file.write("        group1 { atomNumbers %d }\n"%self.colvars[i][0][0])
                        colvars_file.write("        group2 { atomNumbers %d }\n"%self.colvars[i][0][1])
                        colvars_file.write("    }\n")

                    else:
                        for j in range(len(self.colvars[i])):
                            colvars_file.write("    distance { \n")
                            colvars_file.write("        name r%d \n"%j)
                            colvars_file.write("        group1 { atomNumbers %d }\n"%self.colvars[i][0][j,0])
                            colvars_file.write("        group2 { atomNumbers %d }\n"%self.colvars[i][0][j,1])
                            colvars_file.write("    }\n")
                        
                    colvars_file.write("}\n\n")
                    
                #Iterate through colvars again and write harmonic blocks
                for i in range(len(self.colvars)):
                    colvars_file.write("harmonic {\n")
                    colvars_file.write("    colvars cv%d\n"%i)
                    #colvars_file.write("    forceConstant %f\n"%force_constants[i])
                    colvars_file.write("    forceConstant 100\n")
                    colvars_file.write("    centers %f\n"%centers[i, image])
                    colvars_file.write("}\n\n")

                colvars_file.write("colvarsTrajfrequency 1")

                colvars_file.close()

    def write_sim_config_restart(self, iteration):
        """This is very hardcoded at the moment. Will generalize later."""

        for image in range(self.n_images):
            for rep in range(self.n_reps):
                with open("./image_%d/rep_%d/%s_im%d_rep%d.conf"%(image, rep, self.name, image, rep), 'w') as sim_config:
                    sim_config.write("#Simulation configuration file \n")
                    sim_config.write("cutoff 16 \n")
                    sim_config.write("pairlistdist 18.0 \n")
                    sim_config.write("switching on \n")
                    sim_config.write("switchdist 15 \n")
                    if iteration == 1:
                        sim_config.write("coordinates ../../../iteration_%d/image_%d/rep_%d/%s_im%d_rep%d.coor \n"%(iteration-1, image, rep, self.name, image, rep))
                    else:
                        sim_config.write("coordinates ../../../iteration_%d/image_%d/rep_%d/%s_im%d_rep%d-%d.coor \n"%(iteration-1, image, rep, self.name, image, rep, iteration-1))
                    sim_config.write("structure ../../../sys/%s \n\n"%self.topfile)
                    sim_config.write("temperature %d \n\n"%self.temperature)
                    sim_config.write("constraints off \n\n")
                    if iteration == 1:
                        sim_config.write("extendedsystem ../../../iteration_%d/image_%d/rep_%d/%s_im%d_rep%d.restart.xsc \n"%(iteration-1, image, rep, self.name, image, rep))
                    else:
                        sim_config.write("extendedsystem ../../../iteration_%d/image_%d/rep_%d/%s_im%d_rep%d-%d.restart.xsc \n"%(iteration-1, image, rep, self.name, image, rep, iteration-1))
                    ##WRITE OUTPUT PARAMETERS
                    sim_config.write("binaryoutput no \n")
                    sim_config.write("outputname %s_im%d_rep%d-%d \n"%(self.name, image, rep, iteration))
                    sim_config.write("outputenergies 1 \n")
                    sim_config.write("outputtiming 1 \n")
                    sim_config.write("outputpressure 1 \n")
                    sim_config.write("binaryrestart yes \n")
                    sim_config.write("dcdfreq 1 \n")
                    sim_config.write("XSTFreq 1 \n")
                    sim_config.write("restartfreq 10 \n\n")

                    sim_config.write("langevin on \n")
                    sim_config.write("langevintemp %d \n"%self.temperature)
                    sim_config.write("langevinHydrogen on \n")
                    sim_config.write("langevindamping 50 \n\n")
                    sim_config.write("usegrouppressure yes \n")
                    sim_config.write("useflexiblecell no \n")
                    sim_config.write("useConstantArea no \n")
                    sim_config.write("langevinpiston on \n")
                    sim_config.write("langevinpistontarget 1 \n")
                    sim_config.write("langevinpistonperiod 200 \n")
                    sim_config.write("langevinpistondecay 100 \n")
                    sim_config.write("langevinpistontemp 60 \n\n")

                    sim_config.write("timestep 0.5 \n")
                    sim_config.write("fullElectfrequency 1 \n")
                    sim_config.write("stepspercycle 1 \n\n")

                    sim_config.write("paratypecharmm on \n")
                    sim_config.write("parameters ../../../sys/toppar_water_ions_namd.str \n")
                    sim_config.write("parameters ../../../sys/toppar_all36_carb_glycopeptide.str \n")
                    sim_config.write("parameters ../../../sys/par_all36_lipid.prm \n")
                    sim_config.write("parameters ../../../sys/par_all36_na.prm \n")
                    sim_config.write("parameters ../../../sys/par_all36_prot.prm \n")
                    sim_config.write("parameters ../../../sys/par_all36_carb.prm \n")
                    sim_config.write("parameters ../../../sys/par_all36_cgenff.prm \n")
                    sim_config.write("parameters ../../../sys/GR24_withH_qwikmd.str \n")
                    sim_config.write("exclude scaled1-4 \n")
                    sim_config.write("1-4scaling 1.0 \n")
                    sim_config.write("rigidbonds none \n\n")

                    sim_config.write("gbis off \n\n")

                    sim_config.write("qmForces on \n")
                    if "4ih4" in self.name:
                        sim_config.write("qmParamPDB ../../../sys/4ih4_withlig_qmInput_withWater.pdb \n")
                    elif "5z7y" in self.name:
                        sim_config.write("qmParamPDB ../../../sys/5z7y_withlig_qmInput.pdb \n")
                    else:
                        sim_config.write("qmParamPDB ../../../sys/qmInput.pdb \n")
                    sim_config.write("qmColumn beta \n")
                    sim_config.write("QMSimsPerNode 1 \n")
                    sim_config.write("qmBondColumn occ \n")
                    sim_config.write("qmBondDist off \n")
                    sim_config.write("QMBondValueType len \n")
                    sim_config.write("QMSwitching on \n")
                    sim_config.write("QMSwitchingType Switch \n")
                    sim_config.write("qmBaseDir [pwd]/qmmm_exec/ \n")
                    sim_config.write("qmReplaceAll OFF \n")
                    sim_config.write("QMVdWParams off \n")
                    sim_config.write("qmElecEmbed on \n")
                    sim_config.write("QMPCStride 1 \n")
                    sim_config.write("QMCustomPCSelection off \n")
                    sim_config.write("QMLiveSolventSel off \n")
                    #sim_config.write("qmConfigLine \"! B3LYP 6-31G PAL8 EnGrad TightSCF\" \n")
                    #sim_config.write("qmConfigLine \"%%output PrintLevel Mini Print\[ P_Mulliken \] 1 Print\[P_AtCharges_M\] 1 end\" \n")
                    sim_config.write("qmMult 1 1 \n")
                    sim_config.write("qmCharge 1 -1.00 \n") #QM region net charge
                    sim_config.write("qmSoftware custom \n")
                    #sim_config.write("qmSoftware orca \n")
                    #sim_config.write("qmExecPath /u/sciteam/jimingc2/orca_4_0_1_2_linux_x86-64_openmpi202/orca \n") #QM exec path
                    sim_config.write("qmExecPath [pwd]/../../../run_terachem.py \n") #QM exec path
                    sim_config.write("QMChargeMode mulliken \n")
                    sim_config.write("QMOutStride 1 \n")
                    sim_config.write("QMPositionOutStride 1 \n\n")

                    sim_config.write("colvars on \n")
                    sim_config.write("colvarsConfig ../%s_im%d_colvars.conf\n"%(self.name, image))
                    sim_config.write("run 200\n")
                    sim_config.close()

    def write_sim_config(self, iteration, restraints=True):
        """This is very hardcoded at the moment. Will generalize later."""

        for image in range(self.n_images):
            for rep in range(self.n_reps):
                with open("./image_%d/rep_%d/%s_im%d_rep%d.conf"%(image, rep, self.name, image, rep), 'w') as sim_config:
                    sim_config.write("#Simulation configuration file \n")
                    sim_config.write("cutoff 16 \n")
                    sim_config.write("pairlistdist 18.0 \n")
                    sim_config.write("switching on \n")
                    sim_config.write("switchdist 15 \n")
                    if restraints == True:
                        sim_config.write("coordinates ../%s_image%d.pdb \n"%(self.name, image))
                    else:
                        sim_config.write("coordinates %s_im%d_rep%d_equil.restart.coor \n"%(self.name, image, rep))
                    sim_config.write("structure ../../../sys/%s \n\n"%self.topfile)
                    sim_config.write("temperature %d \n\n"%self.temperature)
                    sim_config.write("constraints off \n\n")
                    ##WRITE OUTPUT PARAMETERS
                    sim_config.write("binaryoutput no \n")
                    sim_config.write("outputname %s_im%d_rep%d \n"%(self.name, image, rep))
                    sim_config.write("outputenergies 1 \n")
                    sim_config.write("outputtiming 1 \n")
                    sim_config.write("outputpressure 1 \n")
                    sim_config.write("binaryrestart yes \n")
                    sim_config.write("dcdfreq 1 \n")
                    sim_config.write("XSTFreq 1 \n")
                    sim_config.write("restartfreq 10 \n\n")

                    sim_config.write("langevin on \n")
                    sim_config.write("langevintemp %d \n"%self.temperature)
                    sim_config.write("langevinHydrogen on \n")
                    sim_config.write("langevindamping 50 \n\n")
                    sim_config.write("usegrouppressure yes \n")
                    sim_config.write("useflexiblecell no \n")
                    sim_config.write("useConstantArea no \n")
                    sim_config.write("langevinpiston on \n")
                    sim_config.write("langevinpistontarget 1 \n")
                    sim_config.write("langevinpistonperiod 200 \n")
                    sim_config.write("langevinpistondecay 100 \n")
                    sim_config.write("langevinpistontemp 60 \n\n")

                    sim_config.write("timestep 0.5 \n")
                    sim_config.write("fullElectfrequency 1 \n")
                    sim_config.write("stepspercycle 1 \n\n")

                    sim_config.write("paratypecharmm on \n")
                    sim_config.write("parameters ../../../sys/toppar_water_ions_namd.str \n")
                    sim_config.write("parameters ../../../sys/toppar_all36_carb_glycopeptide.str \n")
                    sim_config.write("parameters ../../../sys/par_all36_lipid.prm \n")
                    sim_config.write("parameters ../../../sys/par_all36_na.prm \n")
                    sim_config.write("parameters ../../../sys/par_all36_prot.prm \n")
                    sim_config.write("parameters ../../../sys/par_all36_carb.prm \n")
                    sim_config.write("parameters ../../../sys/par_all36_cgenff.prm \n")
                    sim_config.write("parameters ../../../sys/GR24_withH_qwikmd.str \n")
                    sim_config.write("exclude scaled1-4 \n")
                    sim_config.write("1-4scaling 1.0 \n")
                    sim_config.write("rigidbonds none \n\n")

                    sim_config.write("gbis off \n\n")

                    sim_config.write("qmForces on \n")
                    #sim_config.write("qmParamPDB ../../../sys/5z7y_withlig_qmInput.pdb \n")
                    #sim_config.write("qmParamPDB ../../../sys/5z7y_withlig_qmInput.pdb \n")

                    if "4ih4" in self.name:
                        sim_config.write("qmParamPDB ../../../sys/4ih4_withlig_qmInput_withWater.pdb \n")
                    elif "5z7y" in self.name:
                        sim_config.write("qmParamPDB ../../../sys/5z7y_withlig_qmInput.pdb \n")
                    else:
                        sim_config.write("qmParamPDB ../../../sys/qmInput.pdb \n")

                    sim_config.write("qmColumn beta \n")
                    sim_config.write("QMSimsPerNode 1 \n")
                    sim_config.write("qmBondColumn occ \n")
                    sim_config.write("qmBondDist off \n")
                    sim_config.write("QMBondValueType len \n")
                    sim_config.write("QMSwitching on \n")
                    sim_config.write("QMSwitchingType Switch \n")
                    sim_config.write("qmBaseDir [pwd]/qmmm_exec/ \n")
                    sim_config.write("qmReplaceAll OFF \n")
                    sim_config.write("QMVdWParams off \n")
                    sim_config.write("qmElecEmbed on \n")
                    sim_config.write("QMPCStride 1 \n")
                    sim_config.write("QMCustomPCSelection off \n")
                    sim_config.write("QMLiveSolventSel off \n")
                    #sim_config.write("qmConfigLine \"! B3LYP 6-31G PAL8 EnGrad TightSCF\" \n")
                    #sim_config.write("qmConfigLine \"%%output PrintLevel Mini Print\[ P_Mulliken \] 1 Print\[P_AtCharges_M\] 1 end\" \n")
                    sim_config.write("qmMult 1 1 \n")
                    sim_config.write("qmCharge 1 -1.00 \n") #QM region net charge
                    sim_config.write("qmSoftware custom \n")
                    #sim_config.write("qmSoftware orca \n")
                    #sim_config.write("qmExecPath /u/sciteam/jimingc2/orca_4_0_1_2_linux_x86-64_openmpi202/orca \n") #QM exec path
                    sim_config.write("qmExecPath [pwd]/../../../run_terachem.py \n") #QM exec path
                    sim_config.write("QMChargeMode mulliken \n")
                    sim_config.write("QMOutStride 1 \n")
                    sim_config.write("QMPositionOutStride 1 \n\n")

                    sim_config.write("colvars on \n")
                    sim_config.write("colvarsConfig ../%s_im%d_colvars.conf\n"%(self.name, image))
                    sim_config.write("run 2000\n")
                    sim_config.close()


    def write_sim_config_drift(self, iteration, restraints=False):
        """This is very hardcoded at the moment. Will generalize later."""

        for image in range(self.n_images):
            for rep in range(self.n_reps):
                with open("./image_%d/rep_%d/%s_im%d_rep%d_drift.conf"%(image, rep, self.name, image, rep), 'w') as sim_config:
                    sim_config.write("#Simulation configuration file \n")
                    sim_config.write("cutoff 16 \n")
                    sim_config.write("pairlistdist 18.0 \n")
                    sim_config.write("switching on \n")
                    sim_config.write("switchdist 15 \n")
                    if restraints == True:
                        sim_config.write("coordinates ../%s_image%d.pdb \n"%(self.name, image))
                    else:
                        sim_config.write("coordinates %s_im%d_rep%d.coor \n"%(self.name, image, rep))
                    sim_config.write("structure ../../../sys/%s \n\n"%self.topfile)
                    sim_config.write("temperature %d \n\n"%self.temperature)
                    sim_config.write("constraints off \n\n")
                    ##WRITE OUTPUT PARAMETERS
                    sim_config.write("binaryoutput no \n")
                    sim_config.write("outputname %s_im%d_rep%d_drift \n"%(self.name, image, rep))
                    sim_config.write("outputenergies 1 \n")
                    sim_config.write("outputtiming 1 \n")
                    sim_config.write("outputpressure 1 \n")
                    sim_config.write("binaryrestart yes \n")
                    sim_config.write("dcdfreq 1 \n")
                    sim_config.write("XSTFreq 1 \n")
                    sim_config.write("restartfreq 10 \n\n")

                    sim_config.write("langevin on \n")
                    sim_config.write("langevintemp %d \n"%self.temperature)
                    sim_config.write("langevinHydrogen on \n")
                    sim_config.write("langevindamping 50 \n\n")
                    sim_config.write("usegrouppressure yes \n")
                    sim_config.write("useflexiblecell no \n")
                    sim_config.write("useConstantArea no \n")
                    sim_config.write("langevinpiston off \n")
                    sim_config.write("langevinpistontarget 1 \n")
                    sim_config.write("langevinpistonperiod 200 \n")
                    sim_config.write("langevinpistondecay 100 \n")
                    sim_config.write("langevinpistontemp 60 \n\n")

                    sim_config.write("timestep 0.5 \n")
                    sim_config.write("fullElectfrequency 1 \n")
                    sim_config.write("stepspercycle 1 \n\n")

                    sim_config.write("paratypecharmm on \n")
                    sim_config.write("parameters ../../../sys/toppar_water_ions_namd.str \n")
                    sim_config.write("parameters ../../../sys/toppar_all36_carb_glycopeptide.str \n")
                    sim_config.write("parameters ../../../sys/par_all36_lipid.prm \n")
                    sim_config.write("parameters ../../../sys/par_all36_na.prm \n")
                    sim_config.write("parameters ../../../sys/par_all36_prot.prm \n")
                    sim_config.write("parameters ../../../sys/par_all36_carb.prm \n")
                    sim_config.write("parameters ../../../sys/par_all36_cgenff.prm \n")
                    sim_config.write("parameters ../../../sys/GR24_withH_qwikmd.str \n")
                    sim_config.write("exclude scaled1-4 \n")
                    sim_config.write("1-4scaling 1.0 \n")
                    sim_config.write("rigidbonds none \n\n")

                    sim_config.write("gbis off \n\n")

                    sim_config.write("qmForces on \n")

                    if "4ih4" in self.name:
                        sim_config.write("qmParamPDB ../../../sys/4ih4_withlig_qmInput_withWater.pdb \n")
                    elif "5z7y" in self.name:
                        sim_config.write("qmParamPDB ../../../sys/5z7y_withlig_qmInput.pdb \n")
                    else:
                        sim_config.write("qmParamPDB ../../../sys/qmInput.pdb \n")

                    sim_config.write("qmColumn beta \n")
                    sim_config.write("QMSimsPerNode 1 \n")
                    sim_config.write("qmBondColumn occ \n")
                    sim_config.write("qmBondDist off \n")
                    sim_config.write("QMBondValueType len \n")
                    sim_config.write("QMSwitching on \n")
                    sim_config.write("QMSwitchingType Switch \n")
                    sim_config.write("qmBaseDir [pwd]/qmmm_exec/ \n")
                    sim_config.write("qmReplaceAll OFF \n")
                    sim_config.write("QMVdWParams off \n")
                    sim_config.write("qmElecEmbed on \n")
                    sim_config.write("QMPCStride 1 \n")
                    sim_config.write("QMCustomPCSelection off \n")
                    sim_config.write("QMLiveSolventSel off \n")
                    #sim_config.write("qmConfigLine \"! B3LYP 6-31G PAL8 EnGrad TightSCF\" \n")
                    #sim_config.write("qmConfigLine \"%%output PrintLevel Mini Print\[ P_Mulliken \] 1 Print\[P_AtCharges_M\] 1 end\" \n")
                    sim_config.write("qmMult 1 1 \n")
                    sim_config.write("qmCharge 1 -1.00 \n") #QM region net charge
                    sim_config.write("qmSoftware custom \n")
                    #sim_config.write("qmSoftware orca \n")
                    #sim_config.write("qmExecPath /u/sciteam/jimingc2/orca_4_0_1_2_linux_x86-64_openmpi202/orca \n") #QM exec path
                    sim_config.write("qmExecPath [pwd]/../../../run_terachem.py \n") #QM exec path
                    sim_config.write("QMChargeMode mulliken \n")
                    sim_config.write("QMOutStride 1 \n")
                    sim_config.write("QMPositionOutStride 1 \n\n")

                    sim_config.write("colvars off \n")
                    #sim_config.write("colvarsConfig ../%s_im%d_colvars.conf\n"%(self.name, image))
                    sim_config.write("run 2000\n")
                    sim_config.close()


    def write_job_script_bw(self, iteration):

        for image in range(self.n_images):
            for rep in range(self.n_reps):
                with open("./image_%d/rep_%d/%s.pbs"%(image, rep, self.name), 'w') as job_script:
                    job_script.write("#!/bin/tcsh \n")
                    job_script.write("#PBS -V \n")
                    job_script.write("#PBS -j oe \n")
                    job_script.write("#PBS -N %s_im%d_rep%d \n"%(self.name, image, rep))
                    job_script.write("#PBS -q low \n")
                    job_script.write("#PBS -l walltime=47:00:00,nodes=1:ppn=10:xk \n\n")

                    job_script.write("cd $PBS_O_WORKDIR \n\n")

                    job_script.write("source /opt/modules/3.2.10.4/init/tcsh \n")
                    job_script.write("module swap PrgEnv-cray PrgEnv-gnu \n")
                    job_script.write("module load rca \n")
                    job_script.write("module load craype-hugepages8M \n")
                    job_script.write("module list \n")
                    job_script.write("module load stat \n\n")

                    job_script.write("setenv HUGETLB_DEFAULT_PAGE_SIZE 8M \n")
                    job_script.write("setenv HUGETLB_MORECORE no \n")
                    job_script.write("setenv ATP_ENABLED 1 \n")
                    job_script.write("ulimit -c unlimited \n\n")

                    job_script.write("aprun -n 10 -N 10 -d 1 /u/sciteam/jphillip/NAMD_2.13/CRAY-XE-MPI-BlueWaters/namd2 %s.conf +p1 > %s_im%d_rep%d.log"%(self.name, self.name, image, rep))
