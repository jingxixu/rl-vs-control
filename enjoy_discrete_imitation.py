from train_imitation import PolicyNet
import torch
import gym
from PPO_discrete import PPO, Memory
from PIL import Image
import torch
import misc_utils as mu
from cartpole_env import CartPoleEnv
import os
import argparse
import time
import numpy as np
import random


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable_gui', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--realtime', action='store_true', default=False)

    # hyper parameters
    parser.add_argument('--ob_type', type=str, default='z_sequence')
    parser.add_argument('--history_size', type=int, default=200)
    parser.add_argument('--fixation', type=float, default=1.0)

    args = parser.parse_args()

    args.rendering = not args.disable_gui
    args.realtime = False if args.disable_gui else args.realtime

    return args


if __name__ == "__main__":
    model_path = 'assets/expert_0144600.pth'
    policy_net = PolicyNet()
    policy_net.load_state_dict(torch.load(model_path))

    args = get_args()
    mu.configure_pybullet(rendering=args.rendering, debug=args.debug, yaw=27.5, pitch=-33.7, dist=1.8)
    env = CartPoleEnv(timestep=0.02,
                      initial_position=[0, 0, 0],
                      initial_orientation=[0, 0, 0, 1],
                      fixation=1,
                      # using a template for fixation
                      urdf_path=os.path.join('assets', 'cartpole_template.urdf'),
                      angle_threshold_low=-15,
                      angle_threshold_high=15,
                      distance_threshold_low=-2,
                      distance_threshold_high=2,
                      force_mag=10,
                      camera_position=(-1.7, 0, 0.4),
                      camera_lookat=(0.5, 0, 0.4),
                      camera_up_direction=(0, 0, 1),
                      ob_type=args.ob_type,
                      history_size=args.history_size,
                      std_dev=None)  # creating environment

    n_episodes = 100
    max_timesteps = 500


    ep_rewards = []
    successes = []
    policy_net.eval()
    for ep in range(1, n_episodes + 1):
        ep_reward = 0
        success = True
        state = env.reset()
        for t in range(max_timesteps-1):
            state = torch.tensor(state.astype(np.float32))
            state = state[None, :]
            action = policy_net(state)
            action = torch.argmax(action).item()
            # action = random.choice([0, 1])
            state, reward, done, _ = env.step(action)
            if args.realtime:
                time.sleep(0.02)
            ep_reward += reward
            if done:
                success = False
                break

        print('Episode: {}\tReward: {}\tSuccess: {}'.format(ep, int(ep_reward), success))
        successes.append(success)
        ep_rewards.append(ep_reward)
        env.close()

    print("Success rate: {} \t Average episode reward: {}".format(np.average(successes), np.average(ep_rewards)))