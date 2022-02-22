import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
plt.rc('savefig', dpi=500)
matplotlib.rc('font',family='Helvetica-Normal',size=24)

import argparse
import os

from calc_string_pmf import calc_pmf

global GAS_CONSTANT
GAS_CONSTANT = 0.00198588

class StringConvergenceChecker(object):

    def __init__(self, name, n_iterations, n_images, n_reps, n_colvars=2, iters_use='all'):
        self.name = name
        self.n_iterations = n_iterations
        self.n_images = n_images
        self.n_reps = n_reps

        if iters_use == 'all':
            self.iters_use = list(range(n_iterations))
        else:
            self.iters_use = iters_use

        self.n_colvars = n_colvars

        self.centers = self._load_centers()


    def _load_centers(self):

        centers_all = []

        for i in range(len(self.iters_use)):
            centers = np.loadtxt("iteration_%d/centers_iter%d.txt"%(self.iters_use[i],self.iters_use[i]))
            if np.shape(centers)[0] < np.shape(centers)[1]:
                centers = centers.T

            centers_all.append(centers)

        return centers_all

    def _load_iter_colvars(self, iteration):
        """Load colvars for given iteration"""

        colvars_iter = []

        for i in range(self.n_images):

            cv_im = []

            for j in range(self.n_reps):
                if iteration == 0:
                    try:
                        cv = np.loadtxt("iteration_%d/image_%d/rep_%d/%s_im%d_rep%d.colvars.traj"%(iteration,i,j,self.name,i,j), usecols=list(range(1, self.n_colvars+1)))
                        cv_im.append(cv)
                    except:
                        pass
                else:
                    try:
                        cv = np.loadtxt("iteration_%d/image_%d/rep_%d/%s_im%d_rep%d-%d.colvars.traj"%(iteration,i,j,self.name,i,j,iteration), usecols=list(range(1, self.n_colvars+1)))
                        cv_im.append(cv)
                    except:
                        pass

            cv_im = np.vstack(cv_im)
            colvars_iter.append(cv_im)

        return colvars_iter

    def calc_center_deviation(self, n_avg=5):

        dev_all = np.zeros(self.n_iterations-n_avg)

        for i in range(self.n_iterations-n_avg):
            avg_center = np.zeros((self.n_images, self.n_colvars))
            for j in range(n_avg):
                avg_center += self.centers[i+j]
            avg_center /= n_avg
 
            dev = np.sqrt(((1/self.n_images)/np.sum(self.centers[i+n_avg-1] - avg_center)**2))

            dev_all[i] = dev

        return dev_all

    def plot_strings(self, iters_use='all', labels=["CV1","CV2","CV3"]):

        if iters_use == 'all':
            iters_use = list(range(self.n_iterations))

        print(iters_use)

        centers_plot = []
        for i in range(len(iters_use)):
            centers = np.loadtxt("iteration_%d/centers_iter%d.txt"%(iters_use[i],iters_use[i]))
            if np.shape(centers)[0] < np.shape(centers)[1]:
                centers = centers.T

            centers_plot.append(centers)

            #plt.plot(centers[:,0], centers[:,1], label="Iter=%d"%iters_use[i])

        if np.shape(centers_plot[0])[1] == 2:

            line_segs = LineCollection(centers_plot) #Make line collection
            line_segs.set_array(np.array(iters_use)) #Color assignments
            line_segs.set_cmap('jet')
            line_segs.set_clim(vmin=0, vmax=25)

            fig, ax = plt.subplots()
            #ax.set_xlim(np.round(np.min(centers_plot[0][:,0])-1), np.round(np.max(centers_plot[0][:,0])+1))
            ax.set_xlim(np.round(centers_plot[0][0,0]+np.sign(centers_plot[0][0,0])), np.round(centers_plot[0][-1,0]+np.sign(centers_plot[0][-1,0])))
            #ax.set_xlim(lims[0], lims[1])
            #ax.set_ylim(np.round(np.min(centers_plot[0][:,1])-1), np.round(np.max(centers_plot[0][:,1])+1))
            ax.set_ylim(np.round(centers_plot[0][0,1]-np.sign(centers_plot[0][0,1])), np.round(centers_plot[0][-1,1]+np.sign(centers_plot[0][-1,1])))
            #ax.set_ylim(lims[2], lims[3])
            ax.add_collection(line_segs)
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
            cbar = fig.colorbar(line_segs)
            cbar.set_label("Iteration")
            cbar.ax.set_yticklabels(np.linspace(0,25,6, dtype=int))
            cbar.ax.tick_params(labelsize=20)

            ax.xaxis.set_major_locator(plt.MultipleLocator(1.0))
            ax.yaxis.set_major_locator(plt.MultipleLocator(1.0))

            fig.tight_layout()
            plt.savefig("string_%s.png"%self.name, transparent=True)

        else: #For that one step...

            for i in range(3):
                for j in range(i+1,3):

                    centers_plot_use = [np.vstack((x[:,i], x[:,j])).T for x in centers_plot]

                    line_segs = LineCollection(centers_plot_use) #Make line collection
                    line_segs.set_array(np.array(iters_use)) #Color assignments
                    line_segs.set_cmap('jet')
                    line_segs.set_clim(vmin=0, vmax=25)

                    fig, ax = plt.subplots()
                    ax.set_xlim(np.floor(np.min(centers_plot_use[0][:,0])), np.ceil(np.max(centers_plot_use[0][:,0])))
                    ax.set_ylim(np.floor(np.min(centers_plot_use[0][:,1])), np.ceil(np.max(centers_plot_use[0][:,1])))
                    ax.add_collection(line_segs)
                    plt.xlabel(labels[i])
                    plt.ylabel(labels[j])
                    cbar = fig.colorbar(line_segs)
                    cbar.set_label("Iteration")
                    cbar.ax.set_yticklabels(np.linspace(0,25,6, dtype=int))
                    cbar.ax.tick_params(labelsize=20)

                    fig.tight_layout()
                    plt.savefig("string_%s_cv%d_%d.png"%(self.name, i, j), transparent=True)

    def plot_iter_pmfs(self, n_dim=2, n_reps=5, k=100, N_max=1005, temp=300, lims=(-1, 20, 0, 100)):

        pmfs_all = []

        for i in self.iters_use:
            os.chdir("iteration_%d"%i)
            if not os.path.isfile("pmf_kcal_mol.npy"):
                f_k = calc_pmf(self.name, i, self.n_images, n_dim, n_reps, k, N_max)

            pmf_iter = np.load("pmf_kcal_mol.npy")
            image_ind = np.array(range(self.n_images))

            pmfs_all.append(np.vstack((image_ind, pmf_iter)).T)
            print(pmfs_all[0])
            #plt.plot(np.array(range(self.n_images)), pmf_iter, label="Iter=%d"%i)
            os.chdir("..")

        line_segs = LineCollection(pmfs_all) #Make line collection
        print(self.iters_use)
        line_segs.set_array(np.array(self.iters_use)) #Color assignments
        line_segs.set_cmap('jet')
        line_segs.set_clim(vmin=0, vmax=25)

        fig, ax = plt.subplots()
        ax.set_xlim(lims[0], lims[1])
        ax.set_ylim(lims[2], lims[3])
        ax.add_collection(line_segs)
        plt.xlabel("Image")
        plt.ylabel("PMF (kcal/mol)")
        cbar = fig.colorbar(line_segs)
        cbar.set_label("Iteration")
        cbar.ax.set_yticklabels(np.linspace(0,25,6, dtype=int))
        cbar.ax.tick_params(labelsize=20)

        fig.tight_layout()
        plt.savefig("pmfs_%s.png"%self.name, transparent=True)

    def force_calc(self, iteration, force_constant):

        #Load colvars
        colvars_iter = self._load_iter_colvars(iteration)
        centers_iter = self.centers[0]

        forces_all = []
        forces_avg = np.zeros(np.shape(centers_iter))

        for i in range(self.n_images):
            forces_im = force_constant*(colvars_iter[i] - centers_iter[i,:])
            forces_all.append(forces_im)
            forces_avg[i,:] = np.mean(forces_im, axis=0)

        return forces_all, forces_avg

    def calc_normal_force_2d(self, centers=None, iteration=None, force_constant=None, forces=None):

        if centers == None:
            centers = self.centers[-1]

        #Calculate tangents at each image
        diff = centers[1:,:] - centers[:-1,:]
        dy_dx = diff[:,1]/diff[:,0]
        deriv_all = np.zeros(self.n_images)

        deriv_all[0] = dy_dx[0]
        deriv_all[-1] = dy_dx[-1]

        deriv_all[1:-1] = (dy_dx[1:] + dy_dx[:-1])/2
    
        #Compute normal vectors
        normal_vecs = np.ones(np.shape(centers))
        normal_vecs[:,1] = -1/deriv_all

        norms = np.linalg.norm(normal_vecs, axis=1)

        for i in range(self.n_images):
            normal_vecs[i,:] /= norms[i]

        #Get forces
        if forces is None:
            forces_all, forces_avg = self.force_calc(iteration, force_constant)
        else:
            forces_all = forces[0]
            forces_avg = forces[1]
        
        normal_forces = np.zeros(self.n_images)
        for i in range(self.n_images):
            normal_forces[i] = np.dot(normal_vecs[i,:], forces_avg[i,:])

        return normal_forces

    def calc_plot_normal_force_3d(self, iteration=None, force_constant=None, forces=None):

        #Just for that one step, again

        forces_all, forces_avg = self.force_calc(iteration, force_constant)
        normal_forces_all = []

        for i in range(3):
            for j in range(i+1,3):
                forces_use = [np.vstack((x[:,i], x[:,j])).T for x in forces_avg]
                normal_forces_all.append(self.calc_normal_force_2d(centers=self.centers, iteration=None, force_constant=100, forces=forces_use))

        return normal_forces_all

    def calc_normal_force_nd(self, iteration=None, force_constant=None, forces=None):
        """More than two colvars. Just because of that one freaking step."""

        centers = self.centers[0]

        #Calculate tangents at each image
        diff = centers[1:,:] - centers[:-1,:]
        dy_dx = np.zeros((self.n_images-1, self.n_colvars-1))

        for i in range(self.n_colvars - 1):
            dy_dx[:,i] = diff[:]

        deriv_all = np.zeros((self.n_images, self.n_colvars-1))

        deriv_all[0] = dy_dx[0]
        deriv_all[-1] = dy_dx[-1]

        deriv_all[1:-1] = (dy_dx[1:] + dy_dx[:-1])/2
    
        #Compute normal vectors
        normal_vecs = np.ones(np.shape(centers))
        normal_vecs[:,1] = -1/deriv_all

        norms = np.linalg.norm(normal_vecs, axis=1)

        for i in range(self.n_images):
            normal_vecs[i,:] /= norms[i]

        #Get forces
        if forces is None:
            forces_all, forces_avg = self.force_calc(iteration, force_constant)
        else:
            forces_all = forces[0]
            forces_avg = forces[1]
        
        normal_forces = np.zeros(self.n_images)
        for i in range(self.n_images):
            normal_forces[i] = np.dot(normal_vecs[i,:], forces_avg[i,:])

        return normal_forces
             
    def plot_time_trace(self, vals_all, savename="timetrace.png"):

        for i in range(len(vals_all)):
            plt.plot(vals_all[i])

        plt.savefig("%s"%savename)

    def plot_iter_trace(self, vals_all, savename="itertrace.png"):

        if len(np.shape(vals_all[0])) > 1:
            for i in range(self.n_colvars):

                plt.figure()
                vals_alliters = np.zeros((self.n_images, self.n_iterations))

                for j in range(self.n_iterations):
                    vals_alliters[:,j] = vals_all[j][:,i]

                plt.plot(vals_alliters.T) 
                plt.xlabel("Iteration")
                plt.ylabel("Force (kcal/mol*Angstrom)")
                plt.savefig("%d_%s"%(i, savename)) 

        else:
            plt.figure()
            vals_alliters = np.zeros((self.n_images, self.n_iterations))

            for j in range(self.n_iterations):
                vals_alliters[:,j] = vals_all[j]

            plt.plot(vals_alliters.T)
            plt.xlabel("Iteration")
            plt.ylabel("Force (kcal/mol*Angstrom)")
            plt.savefig("%s"%savename)
            
            print("WTF IS THIS")
            print(vals_alliters)

def get_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--n_iterations", type=int)
    parser.add_argument("--n_images", type=int, default=20)
    parser.add_argument("--n_reps", type=int, default=5)
    parser.add_argument("--n_colvars", type=int, default=2)
    parser.add_argument("--iters_use", type=int, nargs='+')
    args = parser.parse_args()

    return args

if __name__=="__main__":

    args = get_args()

    conv_checker = StringConvergenceChecker(args.name, args.n_iterations, args.n_images, args.n_reps, iters_use=args.iters_use, n_colvars=args.n_colvars)

    #dev = conv_checker.calc_center_deviation()
    #print(dev)

    conv_checker.plot_strings()

    #conv_checker.plot_iter_pmfs(n_dim=3)

    #forces_avg_all = []
    #for i in range(args.n_iterations):
    #    forces_all, forces_avg = conv_checker.force_calc(i, 100)
    #    print(forces_avg)
    #    print(np.shape(forces_avg))
    #    forces_avg_all.append(forces_avg)

    #conv_checker.plot_iter_trace(forces_avg_all)

    #f_normal_avg_all = []
    #for i in range(args.n_iterations):
    #    f_normal = conv_checker.calc_normal_force_2d(iteration=i, force_constant=100)
    #    print(f_normal)
    #    f_normal_avg_all.append(f_normal)

    #conv_checker.plot_iter_trace(f_normal_avg_all, savename="f_normal.png")

