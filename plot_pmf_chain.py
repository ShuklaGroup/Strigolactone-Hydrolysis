import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rc('savefig', dpi=500)
matplotlib.rc('font',family='Helvetica-Normal',size=24)

def load_pmfs(rxn_step_dir, iter_dir):

    pmfs_all = []

    for i in range(len(rxn_step_dir)):
        pmf = np.load("%s/iteration_%d/pmf_kcal_mol.npy"%(rxn_step_dir[i], iter_dir[i]))
        if i > 0:
            pmf[:] += pmfs_all[i-1][-1]

        print(pmf)

        pmfs_all.append(pmf)

    pmfs_all = np.hstack(pmfs_all)
    print(pmfs_all)

    return pmfs_all

def plot_pmfs(pmfs_all, pmfs_other=None, savename="pmf_chain"):

    fig, ax = plt.subplots()

    plt.plot(pmfs_all, color='blue')

    if pmfs_other is not None:
        x_vals = np.hstack((np.array(range(40)), np.array(range(41,80,2))))
        plt.plot(x_vals, pmfs_other, color='green')

    #plt.xlim(0, 40)
    plt.xlim(0, 80)
    #plt.ylim(0, 200)
    plt.ylim(0, 250)

    plt.grid(linestyle=':')

    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    ax.yaxis.set_major_locator(plt.MultipleLocator(50))

    plt.gca().set_aspect(aspect=40/250, adjustable='box')
    #plt.gca().set_aspect(aspect=20/200, adjustable='box')

    plt.gca().axes.get_xaxis().set_ticks([])

    plt.xlabel("Reaction Coordinate")
    plt.ylabel("PMF (kcal/mol)")

    fig.tight_layout()

    plt.savefig("%s.png"%savename, transparent=True)

if __name__=="__main__":

    #rxn_step_dir = ["string_step1_fixed_ends","string_step2"]
    #iter_dir = [22, 22]

    rxn_step_dir = ["string_step1","string_step2","string_step3","string_step4"]
    iter_dir = [26, 26, 1, 26]

    alt_rxn_step_dir = ["string_step1","string_step2","string_step3_alt"]
    alt_iter_dir = [26, 26, 25]

    pmfs_all = load_pmfs(rxn_step_dir, iter_dir)
    pmfs_other = load_pmfs(alt_rxn_step_dir, alt_iter_dir)

    plot_pmfs(pmfs_all, pmfs_other=pmfs_other, savename="pmf_chain_acyl")
