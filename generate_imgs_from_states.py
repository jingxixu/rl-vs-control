import numpy as np
import argparse
import os
from collect_ppo_demonstrations import calculate_z
import tqdm
from train_perception_model import FeatureNetDepth
# import torch
from cartpole_env import CartPoleEnv
import misc_utils as mu

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--fixation', type=float, required=True)
    parser.add_argument('--start_index', type=int)
    parser.add_argument('--end_index', type=int)

    parser.add_argument('--rendering', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    # load model
    # feature_net = FeatureNetDepth()
    # feature_net.load_state_dict(torch.load('/home/jxu/noise_perception_ws/noisy-perception/perception_checkpoint/2020-04-20_12-24-07/epoch_1066_loss_0.0000004_dist_0.0000018/feature_net.pt'))
    # feature_net.to(device)
    # feature_net.eval()

    # create env
    mu.configure_pybullet(rendering=args.rendering, debug=args.debug, yaw=27.5, pitch=-33.7, dist=1.8)
    fixation_suffix = str(args.fixation).replace('.', '')
    cartpole_env = mu.make_cart_pole_env(fixation=args.fixation, ob_type='full', action_dims=2, history_size=200)
    cartpole_env.cartpole.make_invisible()
    cartpole_env.reset()

    folders = sorted(os.listdir(args.dataset_path))
    num_eps = len(folders)
    start_index = 0 if args.start_index is None else args.start_index
    end_index = num_eps if args.end_index is None else args.end_index

    pbar = tqdm.tqdm(total=end_index - start_index)
    for i in range(start_index, end_index):
        folder = folders[i]
        np_filename = 'z_observations_p_' + str(args.fixation).replace('.', '') + '.npy'
        if os.path.exists(os.path.join(args.dataset_path, folder, 'depth_imgs_'+fixation_suffix+'.npy')):
            pbar.update(1)
            continue
        states = np.load(os.path.join(args.dataset_path, folder, "states.npy"))
        # rgb_imgs = np.zeros((states.shape[0], cartpole_env.camera.height, cartpole_env.camera.width, 3), dtype=np.uint8)
        depth_imgs = np.zeros((states.shape[0], cartpole_env.camera.height, 100), dtype=np.float32)
        gt_z_observations = np.load(os.path.join(args.dataset_path, folder, "z_observations_10.npy"))
        new_z_observations = np.zeros((states.shape[0], 1), dtype=np.float32)
        for i, s in enumerate(states):
            cartpole_env.cartpole.reset_state(s)
            rgb_im, depth_im, seg_im, depth_pixels = cartpole_env.camera.get_data()
            # rgb_im = mu.cut_image(rgb_im, w_pixs=30, h_pixs=0)
            depth_im = mu.cut_image(depth_im, h_pixs=0, w_pixs=30)
            # rgb_imgs[i] = rgb_im
            depth_imgs[i] = depth_im

            # input_x = torch.tensor(depth_im[None, None, ...].astype(np.float32))
            # input_x = input_x.to(device)
            # z = feature_net(input_x)
            # new_z_observations[i] = z.item()
        err = gt_z_observations - new_z_observations
        # np.save(os.path.join(args.dataset_path, folder, np_filename), new_z_observations)
        # np.save(os.path.join(args.dataset_path, folder, 'rgb_imgs_'+fixation_suffix), rgb_imgs)
        np.save(os.path.join(args.dataset_path, folder, 'depth_imgs_'+fixation_suffix), depth_imgs)
        pbar.update(1)
    pbar.close()
