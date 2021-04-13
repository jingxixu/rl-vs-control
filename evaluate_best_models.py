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
import pprint
import json
from sac import SAC


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--save_dir', type=str, required=True)
    parser.add_argument('--limit_epoch', type=int, help='default: None, estimate epochs before this')
    parser.add_argument('--num_epoch', type=int, default=20, help='default: 20, num of top models to evaluate')
    parser.add_argument('--perception_model_path', type=str)

    parser.add_argument('--rendering', action='store_true', default=False)
    parser.add_argument('--realtime', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    args.realtime = False if not args.rendering else args.realtime
    return args


if __name__ == "__main__":
    args = get_args()

    # pick a checkpoint folder to get metadata
    dirs = os.listdir(args.save_dir)
    print()
    with open(os.path.join(args.save_dir, dirs[0], 'metadata.json')) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        metadata = json.load(file)
    pprint.pprint(metadata, indent=4)
    print()

    epoch_folders = [f for f in os.listdir(args.save_dir) if os.path.isdir(os.path.join(args.save_dir, f))]
    # retrieve running reward
    epoch_rrewards = []
    epoch_nums = []
    for e in epoch_folders:
        with open(os.path.join(args.save_dir, e, 'metadata.json')) as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            epoch_metadata = json.load(file)
        epoch_rrewards.append(epoch_metadata['running reward'])
        epoch_nums.append(epoch_metadata['episode'])

    # sort these models using the episode number and limit by max episode
    epoch_folders = [x for e, x in sorted(zip(epoch_nums, epoch_folders)) if e <= args.limit_epoch]
    epoch_rrewards = [x for e, x in sorted(zip(epoch_nums, epoch_rrewards)) if e <= args.limit_epoch]

    # sort these models using the accuracy
    epoch_folders = [x for _, x in sorted(zip(epoch_rrewards, epoch_folders))]
    epoch_folders.reverse()

    # pick top models from recent models
    epoch_folders = epoch_folders[:args.num_epoch]

    # evaluate top folders
    # right now everything is high resolution
    mu.configure_pybullet(rendering=args.rendering, debug=args.debug, yaw=27.5, pitch=-33.7, dist=1.8)
    env = mu.make_cart_pole_env(fixation=metadata['args']['fixation'],
                                ob_type=metadata['args']['ob_type'],
                                pnoise=metadata['args']['pnoise'],
                                rgb=metadata['args']['rgb'],
                                model_path=metadata['args']['perception_model_path'])

    success_rates = []
    avg_rewards = []
    epoch_rrewards = []
    for i, e in enumerate(epoch_folders):
        epoch = int(e.split('_')[1])
        with open(os.path.join(args.save_dir, e, 'metadata.json')) as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            epoch_metadata = json.load(file)
        rreward = epoch_metadata['running reward']
        epoch_rrewards.append(rreward)

        # Agent
        agent = SAC(env.observation_space.shape[0], env.action_space, args)
        agent.load_model(os.path.join(args.save_dir, e))

        n_episodes = 100
        max_timesteps = 500

        ep_rewards = []
        successes = []
        for ep in range(1, n_episodes + 1):
            states = []
            actions = []
            z_observations = []
            ep_reward = 0
            success = True
            state = env.reset()
            for t in range(max_timesteps):
                states.append(state)
                action = np.float64(agent.select_action(state, evaluate=True)[0])
                state, reward, done, _ = env.step(action)
                z_observations.append(env.cartpole.get_fixation_position())
                if args.realtime:
                    time.sleep(0.02)
                ep_reward += reward
                actions.append(action)
                if done:
                    success = False
                    break

            print('Episode: {}\tReward: {}\tSuccess: {}'.format(ep, int(ep_reward), success))
            successes.append(success)
            ep_rewards.append(ep_reward)
            env.close()


        print("num_trials: {}, epoch: {},  rreward: {}, success rate: {}, avg_reward: {}".
              format(n_episodes, epoch, rreward, np.average(successes), np.average(ep_rewards)))
        success_rates.append(np.average(successes))
        avg_rewards.append(np.average(ep_rewards))
    print("avg running reward: {}, avg success rate: {}, avg reward: {}".
          format(np.average(epoch_rrewards), np.average(success_rates), np.average(avg_rewards)))
