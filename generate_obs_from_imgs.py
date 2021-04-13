import numpy as np
import argparse
import os
from collect_ppo_demonstrations import calculate_z
import tqdm
from train_perception_model import FeatureNetDepth
import torch
from cartpole_env import CartPoleEnv
import misc_utils as mu

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--start_index', type=int)
    parser.add_argument('--end_index', type=int)

    parser.add_argument('--rendering', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # load model
    feature_net = FeatureNetDepth(height=120, width=100)
    feature_net.load_state_dict(torch.load(os.path.join(args.model_path, 'feature_net.pt')))
    feature_net.to(device)
    feature_net.eval()

    # fixation_suffix = str(args.fixation).replace('.', '')
    folders = sorted(os.listdir(args.dataset_path))
    num_eps = len(folders)
    start_index = 0 if args.start_index is None else args.start_index
    end_index = num_eps if args.end_index is None else args.end_index

    pbar = tqdm.tqdm(total=end_index - start_index)
    accumulated_err = 0
    for i in range(start_index, end_index):
        folder = folders[i]
        np_filename = 'z_observations_p_10.npy'
        if os.path.exists(os.path.join(args.dataset_path, folder, np_filename)):
            pbar.update(1)
            continue

        depth_imgs = np.load(os.path.join(args.dataset_path, folder, 'depth_imgs_10.npy'))
        gt_z_observations = np.load(os.path.join(args.dataset_path, folder, "z_observations_10.npy"))
        # new_z_observations = np.zeros((depth_imgs.shape[0], 1), dtype=np.float32)

        input_x = torch.tensor(depth_imgs[:, None, :, :])
        input_x = input_x.to(device)
        z = feature_net(input_x)
        new_z_observations = z.detach().cpu().numpy()

        err = np.absolute(gt_z_observations - new_z_observations)
        accumulated_err += np.sum(err)
        np.save(os.path.join(args.dataset_path, folder, np_filename), new_z_observations)
        pbar.update(1)
        pbar.set_description('err dist: {:.7f}'.format(accumulated_err / ((i-start_index+1) * 500)))
    pbar.close()
