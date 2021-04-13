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


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--disable_gui', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--realtime', action='store_true', default=False)
    parser.add_argument('--pnoise', action='store_true', default=False)
    parser.add_argument('--perception_model_path', type=str)

    # hyper parameters
    parser.add_argument('--ob_type', type=str, default='full')
    parser.add_argument('--action_dims', type=int, required=True)
    parser.add_argument('--fixation', type=float, default=1.0)

    args = parser.parse_args()

    args.rendering = not args.disable_gui
    args.realtime = False if args.disable_gui else args.realtime

    return args


def test():
    args = get_args()
    mu.configure_pybullet(rendering=args.rendering, debug=args.debug, yaw=27.5, pitch=-33.7, dist=1.8)
    env = mu.make_cart_pole_env(args.fixation, args.ob_type, args.action_dims, pnoise=args.pnoise, model_path=args.perception_model_path)
    state_dim = env.observation_space.shape[0]
    n_latent_var = 64  # number of variables in hidden layer
    lr = 0.0007
    betas = (0.9, 0.999)
    gamma = 0.99  # discount factor
    K_epochs = 4  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    #############################################

    n_episodes = 100
    max_timesteps = 500

    ppo = PPO(state_dim, args.action_dims, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)

    ppo.policy_old.load_state_dict(torch.load(args.model_path))

    ep_rewards = []
    successes = []
    displacements = []
    for ep in range(1, n_episodes + 1):
        ep_reward = 0
        success = True
        state = env.reset()
        for t in range(max_timesteps):
            action = ppo.policy_old.act_deterministic(state)
            state, reward, done, _ = env.step(action)
            displacements.append(state[0])
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
    print("Average displacement form center: {:5f}".format(np.absolute(np.array(displacements)).mean()))


if __name__ == '__main__':
    test()
