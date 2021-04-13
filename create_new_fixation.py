import numpy as np
import argparse
import os
from collect_ppo_demonstrations import calculate_z
import tqdm


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--fixation', type=float, required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    pbar = tqdm.tqdm(total=len(os.listdir(args.dataset_path)))
    for folder in os.listdir(args.dataset_path):
        np_filename = 'z_observations_'+str(args.fixation).replace('.', '')+'.npy'
        if os.path.exists(os.path.join(args.dataset_path, folder, np_filename)):
            pbar.update(1)
            continue
        states = np.load(os.path.join(args.dataset_path, folder, "states.npy"))
        actions = np.load(os.path.join(args.dataset_path, folder, "actions.npy"))
        new_z_observations = np.zeros((states.shape[0], 1), dtype=np.float32)
        for i, s in enumerate(states):
            new_z_observations[i] = calculate_z(s[0], s[2], args.fixation)
        np.save(os.path.join(args.dataset_path, folder, np_filename), new_z_observations)
        pbar.update(1)
    pbar.close()

