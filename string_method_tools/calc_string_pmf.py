import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rc('savefig',dpi=500)
import pymbar
from pymbar import timeseries
import argparse

GAS_CONSTANT = 0.00198588

def calc_pmf(name, iteration, K, n_dim, n_reps, spring_constant, N_max, temperature=300, bin0=0, data_use_frac=1.0):

    N_k = np.zeros(K, np.int32) #Number of snapshots per window
    K_k = np.zeros(K) #Spring constants
    #r0_k = np.zeros((K, n_dim)) #Centers
    r_kn = np.zeros((K, n_dim, N_max)) #Colvars for each snapshot
    u_kn = np.zeros((K, N_max)) #Reduced potential energy (to be calculated)
    g_k = np.zeros(K) #Timeseries for subsampling
    nbins = 100 #Number of bins for pmf

    K_k[:] = spring_constant
    #r0_k[:] = np.linspace(min_center, max_center, K)
    if iteration == 0:
        r0_k = np.loadtxt("centers_iter%d.txt"%iteration).T
    else:
        r0_k = np.loadtxt("centers_iter%d.txt"%iteration)

    print(r0_k)

    ##Load colvar values from simulations
    for i in range(bin0,K+bin0):
        for j in range(n_reps):
            try:
                print("image_%d/rep_%d/%s_im%d_rep%d-%d.colvars.traj"%(i,j,name,i,j,iteration))
                if iteration == 0:
                    colvars_data = np.loadtxt("image_%d/rep_%d/%s_im%d_rep%d.colvars.traj"%(i,j,name,i,j))
                else:
                    colvars_data = np.loadtxt("image_%d/rep_%d/%s_im%d_rep%d-%d.colvars.traj"%(i,j,name,i,j,iteration))

                print("np.loadtxt passed")

                #Remove first column
                colvars_data = colvars_data[:,1:]

                #colvars_data = np.hstack(colvars_data)

                print("hstack passed")
                n_k = np.shape(colvars_data)[0]

                #print(n_k)
                #print(N_k[i-bin0])
                #print(N_k[i-bin0] + n_k)
                #print(np.shape(r_kn))
                #print(np.shape(r_kn[i, :, N_k[i]:(N_k[i] + n_k)]))
                #print(np.shape(r_kn[i-bin0, :, N_k[i-bin0]:(N_k[i-bin0] + n_k)]))

                r_kn[i-bin0, :, N_k[i-bin0]:(N_k[i-bin0] + n_k)] = colvars_data.T

                print("r_kn updated")

                N_k[i-bin0] += n_k
            except:
                print("WARNING: One or more data files failed to load")
                pass

        #SUBSAMPLING
        #g_k[i] = timeseries.statisticalInefficiency(r_kn[i,:])
        #print(g_k[i])
        #indices = timeseries.subsampleCorrelatedData(r_kn[i,:], g=g_k[i])

        indices = list(range(N_k[i-bin0]))[::1]
        N_k[i-bin0] = len(indices)
        r_kn[i-bin0,:,0:N_k[i-bin0]] = r_kn[i-bin0,:,indices].T


        if data_use_frac < 1.0:
            indices = np.random.choice(range(N_k[i-bin0]), size=(int(data_use_frac*N_k[i-bin0]),), replace=False) 

            N_k[i-bin0] = len(indices)
            r_kn[i-bin0,:,0:N_k[i-bin0]] = r_kn[i-bin0,:,sorted(indices)].T

    ##Calculate u_kln from centers and actual sep distances
    #for i in range(K):
    #    for j in range(N_max):
    #        if r_kn[i,j] != 0:
    #            u_kn[i,j] = 1.0/(GAS_CONSTANT*temperature) * (K_k[i]/2.0) * (r_kn[i,j] - r0_k[i])**2

    u_kln = np.zeros((K,K,N_max))
    for k in range(K):
        for n in range(N_k[k]):
            ##CHANGE THIS TO N-D EQUIVALENT
            
            #print(np.shape(r_kn[k,:,n]))
            #print(np.shape(r0_k))
            #dr = r_kn[k,:,n] - r0_k[:,k]
            dr = r_kn[k,:,n] - r0_k


            #print(np.shape(dr))
            
            #print(np.shape(u_kln[k,:,n]))
            #print(np.shape(dr**2))
            #print(np.shape(K_k))
            #print(np.shape((1.0/(GAS_CONSTANT*temperature)) * (K_k/2.0) * dr**2))
            #if len(np.shape(dr)) == 1:
            #    u_kln[k,:,n] = (1.0/(GAS_CONSTANT*temperature)) * (K_k/2.0) * dr**2
            #else:
            u_kln[k,:,n] = (1.0/(GAS_CONSTANT*temperature)) * (K_k/2.0) * np.sum(dr**2, axis=1)
            #u_kln[k,:,n] = (1.0/(GAS_CONSTANT*temperature)) * (K_k/2.0) * np.sum(dr**2)

    print(r_kn)
    print(u_kln)

    ##Calculate bins
    #r_min = np.min(r_kn[np.nonzero(r_kn)])
    #r_max = np.max(r_kn[np.nonzero(r_kn)])
    #delta = (r_max - r_min)/nbins
    #bin_center_i = np.zeros(nbins)
    #for i in range(nbins):
    #    bin_center_i[i] = r_min + delta/2 + delta*i

    #Bin data
    #bin_kn = np.zeros((K, N_max))
    #for k in range(K):
    #    for n in range(N_k[k]):
    #        bin_kn[k,n] = int((r_kn[k,n] - r_min) / delta)

    ##Run MBAR and calculate pmf
    mbar = pymbar.MBAR(u_kln, N_k, verbose = True, maximum_iterations=10000)

    f_k = mbar.getFreeEnergyDifferences()

    #(f_i, df_i) = mbar.computePMF(u_kn, bin_kn, nbins)

    return f_k

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--n_windows", default=20, type=int)
    parser.add_argument("--n_dim", type=int)
    parser.add_argument("--n_reps", default=5, type=int)
    parser.add_argument("--k", default=100, type=float)
    #parser.add_argument("--colvar_col", type=int)
    parser.add_argument("--N_max", type=int, default=3000)
    parser.add_argument("--temp", type=float, default=300)
    parser.add_argument("--error_bar", type=bool)

    args = parser.parse_args()

    return args

if __name__=="__main__":

    args = get_args()

    if not args.error_bar:

        f_k = calc_pmf(args.name, args.iteration, args.n_windows, args.n_dim, args.n_reps, args.k, args.N_max)

        print(f_k[0][0])
        print(f_k[0][0]*GAS_CONSTANT*args.temp)

        np.save("pmf_kcal_mol.npy", f_k[0][0]*GAS_CONSTANT*args.temp)

    if args.error_bar:
        pmfs_all = []
        for i in range(5):
            f_k = calc_pmf(args.name, args.iteration, args.n_windows, args.n_dim, args.n_reps, args.k, args.N_max, data_use_frac=0.8)
            pmf = f_k[0][0]*GAS_CONSTANT*args.temp

            np.save("pmf_kcal_mol_%d.npy"%i, pmf)
            pmfs_all.append(pmf)

        pmfs_all = np.vstack(pmfs_all)

        avg = np.mean(pmfs_all, axis=0)
        np.save("pmf_avg_over_5.npy", avg)

        print(avg)

        error_bars = np.std(pmfs_all, axis=0)
        np.save("pmf_error_bars.npy", error_bars)

        print(error_bars)
