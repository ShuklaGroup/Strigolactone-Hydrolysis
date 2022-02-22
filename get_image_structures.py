import numpy as np
import mdtraj as md
import glob
import argparse

def draw_structures(name, iteration, n_images=20, n_reps=5):

    frames = []

    for i in range(n_images):
        #Get possible dcd files
        trajs = glob.glob("iteration_%d/image_%d/rep*/*%d.coor"%(iteration, i, iteration))
        print(len(trajs))
        print(trajs)
        traj_use = trajs[np.random.randint(len(trajs))]

        frames.append(md.load_pdb(traj_use))

    traj = md.join(frames)
    traj.save_xtc("%s_iteration%d_image_samples.xtc"%(name, iteration))


def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--name", type=str)
    parser.add_argument("--iteration", type=int)
    parser.add_argument("--n_images", type=int, default=20)
    parser.add_argument("--n_reps", type=int, default=5)

    args = parser.parse_args()

    return args

if __name__=="__main__":

    args = get_args()
    
    draw_structures(args.name, args.iteration, args.n_images, args.n_reps)
