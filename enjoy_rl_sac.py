import gym
from PPO_discrete import PPO, Memory
from PIL import Image
import torch
import misc_utils as mu
import os
import argparse
import time
import numpy as np
from sac import SAC


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rendering', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--realtime', action='store_true', default=False)

    parser.add_argument('--pnoise', action='store_true', default=False)
    parser.add_argument('--perception_model_path', type=str)
    parser.add_argument('--ob_type', type=str, default='full')
    parser.add_argument('--fixation', type=float, default=1.0)
    parser.add_argument('--delay', type=int, default=0)

    # sac parameters
    parser.add_argument('--policy', default="Gaussian",
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
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_steps', type=int, default=10000001, metavar='N',
                        help='maximum number of steps (default: 10000000)')
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

    args = parser.parse_args()
    args.realtime = False if not args.rendering else args.realtime
    return args


def test():
    args = get_args()
    mu.configure_pybullet(rendering=args.rendering, debug=args.debug, yaw=27.5, pitch=-33.7, dist=1.8)
    env = mu.make_cart_pole_env(args.fixation,
                                args.ob_type,
                                pnoise=args.pnoise,
                                model_path=args.perception_model_path,
                                delay=args.delay)

    n_episodes = 100
    max_timesteps = 500

    # Agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    agent.load_model(args.model_path)
    # agent.policy.load_state_dict(torch.load('sac_save/2020-06-22_10-56-51/policy_0000330.pth'))
    # agent.critic.load_state_dict(torch.load('sac_save/2020-06-22_10-56-51/critic_0000330.pth'))
    # agent.critic_target.load_state_dict(torch.load('sac_save/2020-06-22_10-56-51/critic_target_0000330.pth'))

    ep_rewards = []
    successes = []
    displacements = []
    list_actions = []
    list_z_observations = []
    list_states = []
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
            displacements.append(state[0])
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
        list_actions.append(np.array(actions))
        list_z_observations.append(np.array(z_observations))
        list_states.append(states)

    # import scipy.io
    # scipy.io.savemat('sysid_data_100.mat', dict(actions=list_actions, observations=list_z_observations, states=list_states))

    print("Success rate: {} \t Average episode reward: {}".format(np.average(successes), np.average(ep_rewards)))
    print("Average displacement form center: {:5f}".format(np.absolute(np.array(displacements)).mean()))

    # plotting the expert trajectories
    # import matplotlib.pyplot as plt
    # t = range(500)
    # fig, axs = plt.subplots(3, 4)
    #
    # # fixation = 1.0
    # axs[0, 0].plot(t, list_actions[0])
    # # axs[0, 0].set_title('example 1')
    #
    # axs[0, 1].plot(t, list_actions[1])
    # # axs[0, 1].set_title('example 1')
    #
    # axs[0, 2].plot(t, list_actions[2])
    # # axs[0, 2].set_title('example 2')
    #
    # axs[0, 3].plot(t, list_actions[3])
    # # axs[0, 3].set_title('example 2')
    #
    # # fixation = 0.8
    # axs[1, 0].plot(t, list_actions[3])
    # # axs[1, 0].set_title('example 3')
    #
    # axs[1, 1].plot(t, list_actions[4])
    # # axs[1, 1].set_title('example 3')
    #
    # axs[1, 2].plot(t, list_actions[5])
    # # axs[1, 2].set_title('example 4')
    #
    # axs[1, 3].plot(t, list_actions[6])
    # # axs[1, 3].set_title('example 4')
    #
    # # fixation = 0.5
    # axs[2, 0].plot(t, list_actions[7])
    # # axs[2, 0].set_title('example 5')
    #
    # axs[2, 1].plot(t, list_actions[8])
    # # axs[2, 1].set_title('example 5')
    #
    # axs[2, 2].plot(t, list_actions[8])
    # # axs[2, 2].set_title('example 6')
    #
    # axs[2, 3].plot(t, list_actions[9])
    # # axs[2, 3].set_title('example 6')
    #
    # # for ax in axs.flat:
    # #     ax.set(xlabel='time step', ylabel='z')
    #
    # # Hide x labels and tick labels for top plots and y ticks for right plots.
    # # for ax in axs.flat:
    # #     ax.label_outer()

    # plt.show()


if __name__ == '__main__':
    test()
