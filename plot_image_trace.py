import numpy as np
import mdtraj as md
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rc('savefig', dpi=500)
matplotlib.rc('font',family='Helvetica-Normal',size=24)

import glob
import os

def calc_distances(iteration, pairs, topfile):

    trajs = glob.glob("iteration_%d/image_*/rep*/*%d.dcd"%(iteration, iteration))

    for traj in trajs:
        for pair in pairs:

            t = md.load(traj, top="%s"%topfile)
            d = md.compute_distances(t, [pair])

            np.save("./analysis/%s_%d-%d.npy"%(traj.split('/')[-1], pair[0], pair[1]), d)

def calc_image_trace(iteration, n_images, pair):
    
    avg_per_image = np.zeros(n_images)

    for i in range(n_images):
        distances = glob.glob("./analysis/*im%d_*%d-%d.npy"%(i, pair[0], pair[1]))

        n_counts = 0
        running_sum = 0

        for dist in distances:
            data = np.load(dist)
            n_counts += np.size(data)
            running_sum += np.sum(data)

        avg_per_image[i] = running_sum/n_counts

    return avg_per_image

if __name__=="__main__":

    #Acyl pathway

    #pairs = np.array([[1438, 4082], [1438, 1439], [3749, 1439], [4088, 1439], [4088, 4082]])
    #labels = ["S97:OG-GR24:C16", "S97:OG-S97:HG", "H247:NE-S97:HG", "GR24:O5-S97:HG", "GR24:O5-GR24:C16"]
    #steered = [1, 1, 1, 0, 0]

    #pairs = np.array([[4086, 4079], [4088, 1439], [4086, 1439]])
    #labels = ["GR24:O3-GR24:C13", "GR24:O5-S97:HG", "GR24:O3-S97:HG"]
    #steered = [1, 1, 1]

    #pairs = np.array([[3749, 4079], [4088, 1439], [4086, 1439]])
    #labels = ["H247:NE-GR24:C13", "GR24:O5-S97:HG", "GR24:O3-S97:HG"]
    #steered = [1, 0, 0] 

    pairs = np.array([[3749, 4079], [4088, 4082], [4082, 1438], [4081, 1438]])
    labels = ["H247:NE-GR24:C13", "GR24:O5-GR24:C16", "GR24:C16-S97:OG", "GR24:C15-S97:OG"]
    steered = [1, 1, 1, 0] 

    #Michael addition pathway
    #pairs = np.array([[1438, 4078], [1438, 1439], [3749, 1439]])
    #labels = ["S97:OG-GR24:C12", "S97:OG-S97:HG", "H247:NE-S97:HG"]
    #steered = [1, 1, 1]

    #pairs = np.array([[4079, 4088], [4078, 4086]])
    #labels = ["GR24:C12-GR24:O3", "GR24:C13-GR24:O5"]
    #steered = [1, 1]

    #calc_distances(22, pairs, "../sys/4ih4_withlig_setup.psf")
    y_max = 6
    ts_ind = 18

    plt.figure()
    fig, ax = plt.subplots()

    for i in range(np.shape(pairs)[0]):
        avg = calc_image_trace(25, 20, pairs[i])
        print(10*avg)

        if steered[i]:
            plt.plot(10*avg, label=labels[i])
        else:
            plt.plot(10*avg, label=labels[i], linestyle='--')

    plt.legend(fancybox=True, frameon=True, edgecolor='k', fontsize=12, loc=1, framealpha=1)

    plt.xlim(0,20)
    plt.ylim(0,y_max)

    ts_line = np.linspace(y_max/16, y_max-(y_max/16), 9)
    #plt.scatter(ts_ind*np.ones(np.size(ts_line)), ts_line, color='k', marker='*')
    plt.scatter(ts_ind*np.ones(np.size(ts_line)), ts_line, color='k', marker='$\ddag$', s=175)

    plt.grid(linestyle=":")

    plt.xlabel("Image")
    #plt.ylabel("Bond Distance ($\AA^3$)")
    plt.ylabel("Bond Distance")

    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)

    plt.gca().set_aspect(aspect=0.6*20/y_max, adjustable='box')

    fig.tight_layout()

    plt.savefig("acyl_step3_alt_image_trace.png", transparent=True)
