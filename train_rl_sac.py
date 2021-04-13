import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
from sac import SAC, ReplayMemory
from torch.utils.tensorboard import SummaryWriter
import os
import time
import misc_utils as mu
from collections import deque
import random
import pprint


def get_args():
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    parser.add_argument('--policy_type', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', action='store_true',
                        help='Evaluates a policy a every eval_interval episode (default: False)')
    parser.add_argument('--no_save', action='store_true',
                        help='Saves a policy every save_interval episode (default: False)')
    parser.add_argument('--eval_interval', type=int, default=10,
                        help='Evaluates a policy every eval_interval episode (default: 10)')
    parser.add_argument('--save_interval', type=int, default=10,
                        help='Saves a policy every save_interval episode (default: 10)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=10, metavar='N',
                        help='random seed (default: 10)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=10000000, metavar='N',
                        help='maximum number of steps (default: 10M)')
    parser.add_argument('--max_ep_steps', type=int, default=500,
                        help='maximum number of steps in an episode (default: 500)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--no_cuda', action="store_true",
                        help='run on CUDA (default: False)')
    # -----------------------------
    # Normally no need to touch above parameters

    parser.add_argument('--ob_type', type=str, default='full',
                        help='obersation type, one of full, z_sequence or pixel (default: full)')
    parser.add_argument('--fixation', type=float, default=1.0)
    parser.add_argument('--pnoise', action='store_true', default=False)
    parser.add_argument('--perception_model_path', type=str)
    parser.add_argument('--resolution', type=str, default='high')
    parser.add_argument('--probabilistic', action='store_true', default=False)
    parser.add_argument('--rgb', action='store_true', default=False)
    parser.add_argument('--append_actions_ob', action='store_true', default=False)
    parser.add_argument('--delay', type=int, default=0)

    parser.add_argument('--rendering', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--log_dir', type=str, default='logs/rl_sac',
                        help='Directory path to save logs (default: logs/rl_sac)')
    parser.add_argument('--save_dir', type=str, default='models/rl_sac',
                        help='Directory path to save checkpoints (default: models/rl_sac)')
    parser.add_argument('--suffix', type=str)
    parser.add_argument('--gravity', type=float, default=9.8)
    parser.add_argument('--timestep', type=int, default=1,
                        help='multiples of 1/240, default pybullet time step')
    parser.add_argument('--duration', type=float, default=10,
                        help='duration in seconds of this episode')
    args = parser.parse_args()

    args.timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    prefix = "fix_{:.2f}_noise_".format(args.fixation) if args.pnoise else "fix_{:.2f}_".format(args.fixation)
    if args.ob_type == 'full':
        if args.pnoise:
            prefix = 'full_' + prefix
        else:
            prefix = 'full_'
    args.save_dir = os.path.join(args.save_dir, prefix+args.timestr) if args.suffix is None \
        else os.path.join(args.save_dir, prefix+args.timestr+'_'+args.suffix)
    args.log_dir = os.path.join(args.log_dir, prefix+args.timestr) if args.suffix is None \
        else os.path.join(args.log_dir, prefix+args.timestr+'_'+args.suffix)
    if args.perception_model_path is None and args.pnoise:
        assert args.fixation is not None
        args.perception_model_path = mu.get_perception_model_path(args.ob_type, args.fixation, args.rgb, args.resolution)
    # modify alpha for rgb images
    if args.rgb:
        args.alpha = 0.01
    args.max_ep_steps = round(args.duration / (args.timestep * 1/240))
    return args


if __name__ == "__main__":
    start_time = time.time()
    args = get_args()

    print()
    pprint.pprint(vars(args), indent=4)
    print()

    # Environment
    mu.configure_pybullet(rendering=args.rendering, debug=args.debug, gravity=args.gravity)
    height, width, height_after, width_after = mu.map_resolution(args.resolution)
    env = mu.make_cart_pole_env(fixation=args.fixation,
                                ob_type=args.ob_type,
                                action_dims=1,
                                pnoise=args.pnoise,
                                model_path=args.perception_model_path,
                                probabilistic=args.probabilistic,
                                rgb=args.rgb,
                                img_size=(height, width),
                                delay=args.delay,
                                append_actions_ob=args.append_actions_ob,
                                timestep=args.timestep)

    # this seed does not seem to make each run (training curve) identical
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    random.seed(args.seed)

    # Agent
    agent = SAC(env.observation_space.shape[0],
                env.action_space,
                gamma=args.gamma,
                tau=args.tau,
                alpha=args.alpha,
                policy_type=args.policy_type,
                target_update_interval=args.target_update_interval,
                automatic_entropy_tuning=args.automatic_entropy_tuning,
                no_cuda=args.no_cuda,
                hidden_size=args.hidden_size,
                lr=args.lr)

    # Tesnorboard
    writer = SummaryWriter(log_dir=args.log_dir)

    # Memory
    memory = ReplayMemory(args.replay_size)

    # Training Loop
    total_numsteps = 0
    updates = 0
    episode_rewards_list = deque(maxlen=100)
    running_reward = 0.0  # the average reward over the past 100 episodes

    for i_episode in itertools.count(1):
        episode_reward = 0
        episode_steps = 0
        done = False
        state = env.reset()

        while not done and episode_steps < args.max_ep_steps:
            if args.start_steps > total_numsteps:
                action = env.action_space.sample()  # Sample random action
            else:
                action = agent.select_action(state)  # Sample action from policy

            if len(memory) > args.batch_size:
                # Number of updates per step in environment
                for i in range(args.updates_per_step):
                    # Update parameters of all the networks
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                         args.batch_size,
                                                                                                         updates)

                    # writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    # writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    # writer.add_scalar('loss/policy', policy_loss, updates)
                    # writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    # writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    writer.add_scalar('reward/train/running_reward_step', running_reward, total_numsteps)
                    updates += 1

            next_state, reward, done, _ = env.step(action)  # Step
            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward

            # Ignore the "done" signal if it comes from hitting the time horizon.
            # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
            # For my code, done is not true just because of hitting time horizon
            mask = float(not done)

            memory.push(state, action, reward, next_state, mask)  # Append transition to memory

            state = next_state

        # episode ends
        if total_numsteps > args.num_steps:
            break

        episode_rewards_list.append(episode_reward)
        running_reward = np.array(episode_rewards_list).mean()
        training_time = mu.convert_second(time.time() - start_time)

        writer.add_scalar('reward/train/episode_reward', episode_reward, i_episode)
        writer.add_scalar('reward/train/running_reward', running_reward, i_episode)
        print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}, rreward: {}, time: {}".
              format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2), round(running_reward, 2), training_time))

        if i_episode % args.eval_interval == 0 and args.eval:
            avg_reward = 0.
            episodes = 10
            for _ in range(episodes):
                state = env.reset()
                episode_reward = 0
                done = False
                while not done:
                    action = agent.select_action(state, evaluate=True)

                    next_state, reward, done, _ = env.step(action)
                    episode_reward += reward

                    state = next_state
                avg_reward += episode_reward
            avg_reward /= episodes

            writer.add_scalar('reward/test/avg_reward', avg_reward, i_episode)

            print("----------------------------------------")
            print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
            print("----------------------------------------")

        if i_episode % args.save_interval == 0 and not args.no_save:
            folder_name = 'episode_{}_rreward_{}'.format(i_episode, int(round(running_reward)))
            folder_path = os.path.join(args.save_dir, folder_name)
            actor_path, critic_path = agent.save_model(folder_path)

            checkpoint_metadata = {
                'episode': i_episode,
                'step': total_numsteps,
                'update': updates,
                'running reward': running_reward,
                'episode reward': episode_reward,
                'training time': training_time,
                'args': vars(args)}

            mu.save_json(checkpoint_metadata, os.path.join(folder_path, 'metadata.json'))
            print("----------------------------------------")
            print('Saving models to {} and {}'.format(actor_path, critic_path))
            print("----------------------------------------")

    env.close()
