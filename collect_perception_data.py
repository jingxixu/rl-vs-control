from cartpole_env import CartPole, CartPoleEnv
import argparse
import os
import time
import misc_utils as mu
import numpy as np
import pybullet_utils as pu
from math import radians, sin
import tqdm
import pybullet as p


def get_args():
    parser = argparse.ArgumentParser(description='Collect training data')

    parser.add_argument('--fixation', default=1.0, type=float)
    parser.add_argument('--num_samples', default=50000, type=int)
    parser.add_argument('--save_dir', required=True, type=str)
    parser.add_argument('--height', type=int, default=120)
    parser.add_argument('--width', type=int, default=160)
    parser.add_argument('--seed', type=int, default=10)

    parser.add_argument('--rendering', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    # seed
    np.random.seed(args.seed)
    mu.configure_pybullet(rendering=args.rendering, debug=args.debug, yaw=27.5, pitch=-33.7, dist=1.8)

    cartpole_env = mu.make_cart_pole_env(fixation=args.fixation, ob_type='z_sequence', img_size=(args.height, args.width))
    cartpole_env.reset()

    # plot camera pose
    camera_pose = pu.pose_from_matrix(cartpole_env.camera.pose_matrix)
    pu.create_frame_marker(camera_pose)

    pbar = tqdm.tqdm(total=args.num_samples, desc="sampled x: | sampled theta: ")
    observations = []
    for i in range(args.num_samples):
        image_folder_path = os.path.join(args.save_dir, 'fixation_'+mu.convert_float_to_fixstr(args.fixation), '{:05}'.format(i))
        if not os.path.exists(image_folder_path):
            os.makedirs(image_folder_path)

        # sample a random state for the cartpole
        x = np.random.uniform(cartpole_env.distance_threshold_low, cartpole_env.distance_threshold_high)
        theta = np.random.uniform(cartpole_env.angle_threshold_low, cartpole_env.angle_threshold_high)
        # print("sampled x: {}, sampled theta: {}".format(x, theta))
        cartpole_env.cartpole.reset_slider_cart_joint(x, 0)
        cartpole_env.cartpole.reset_cart_pole_joint(radians(theta), 0)

        cartpole_env.cartpole.make_invisible()
        rgb_im, depth_im, seg_im, depth_pixels = cartpole_env.camera.get_data()
        cartpole_env.cartpole.make_visible()
        z = mu.calculate_z(x, radians(theta), args.fixation)
        observations.append(z)
        # mu.show_rgb(rgb_im)
        # mu.show_depth(depth_im)
        depth_im = mu.cut_image_by_percentage(depth_im, 0, 0.375)
        rgb_im = mu.cut_image_by_percentage(rgb_im, 0, 0.375)

        # save stuff
        np.save(os.path.join(image_folder_path, 'rgb.npy'), rgb_im)
        np.save(os.path.join(image_folder_path, 'depth.npy'), depth_im)
        np.save(os.path.join(image_folder_path, 'state.npy'), np.array(cartpole_env.cartpole.get_state()))
        pbar.update(1)
        pbar.set_description("sampled x: {:+.6f} | sampled theta: {:+.6f} | z: {:+.6f}".format(x, theta, z))
    np.save(os.path.join(args.save_dir, 'observations.npy'), np.array(observations, dtype=np.float32))
