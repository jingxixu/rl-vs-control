import gym
#from PPO_discrete import PPO, Memory
from PIL import Image
import torch
import os
import sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)
import misc_utils as mu
import argparse
import time
import numpy as np
from sac import SAC
from enjoy_hinf_controller import DiscreteController
from scipy.io import loadmat
import pprint



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rendering', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--realtime', action='store_true', default=False)

    parser.add_argument('--full_state', type=bool, default=False)
    parser.add_argument('--ob_type', type = str, default = 'z_sequence')
    parser.add_argument('--pnoise', type = bool, default=False)
    parser.add_argument('--rgb', type=bool, default=False)
    parser.add_argument('--fixation', type=float, default=1.0)
    parser.add_argument('--policy', type=str, default='random')
    parser.add_argument('--resolution', type=str, default='high')
    parser.add_argument('--policy_model_path', type=str)
    parser.add_argument('--true_dynamics', action='store_true', default=False)

    parser.add_argument('--zero_initial', action='store_true', default=False)
    parser.add_argument('--action_noise', type=float, default=0)
    parser.add_argument('--save_fnm', type=str)
    parser.add_argument('--save_folder', type=str)
    parser.add_argument('--num_traj', type=int, default=10000)
    parser.add_argument('--trial_number', type=int, default=0)
    #parser.add_argument('--seed', type=int, default = None)

    args = parser.parse_args()
    args.realtime = False if not args.rendering else args.realtime

    if args.full_state:
        args.ob_type = 'full'
    else:
        args.ob_type = 'z_sequence'
    if args.save_fnm is None:
        fixstr = mu.convert_float_to_fixstr(args.fixation)
        args.save_fnm = 'fix_'+fixstr+'_'+args.policy
        args.save_fnm += '_obtype_'+args.ob_type
        if args.pnoise:
            args.save_fnm += '_noise'
            if args.rgb:
                args.save_fnm += '_rgb'
        args.save_fnm += '.mat'

    if args.save_folder is None:
        args.save_folder = os.path.join("control_experiment_data", "trial"+str(args.trial_number), "datasets", "sysid")

    args.perception_model_path = mu.get_perception_model_path(args.ob_type, args.fixation, rgb=args.rgb, resolution=args.resolution) \
        if args.pnoise else None

    return args


def test(args):
    #np.random.seed(10)

    print()
    pprint.pprint(vars(args), indent=4)
    print()

    mu.configure_pybullet(rendering=args.rendering, debug=args.debug, yaw=27.5, pitch=-33.7, dist=1.8)
    if args.true_dynamics:
        env = mu.make_cart_pole_env_true_dyanmics(fixation=args.fixation,
                                                  ob_type='z_sequence',
                                                  pnoise=args.pnoise,
                                                  model_path=args.perception_model_path)
    else:
        env = mu.make_cart_pole_env(fixation=args.fixation,
                                    ob_type=args.ob_type,
                                    pnoise=args.pnoise,
                                    model_path=args.perception_model_path,
                                    rgb=args.rgb
                                ) 

    n_episodes = args.num_traj
    max_timesteps = 500

    # Agent
    if args.policy == 'rl_sac':
        agent = SAC(env.observation_space.shape[0], env.action_space)
        agent.load_model(args.policy_model_path)
    elif args.policy == 'hinf':
        K = loadmat('robust_control/params/hinf.mat')
        A, B, C, D, Nx = K['A'], K['B'], K['C'], K['D'], K['Nx'].squeeze()
        controller = DiscreteController(A, B, C, D, Nx)

    ep_rewards = []
    successes = []
    displacements = []
    list_actions = []
    list_observations = []
    list_states = []
    for ep in range(1, n_episodes + 1):
        states = []
        observations = []
        z_observations = []
        actions = []
        ep_reward = 0
        success = True
        ob = env.reset(randstate=np.zeros(4)) if args.zero_initial else env.reset()

        if args.policy == 'hinf' and args.zero_initial:
            # some initial random actions for hinf
            for i in range(1):
                states.append(np.array(env.cartpole.get_state()))
                z_observations.append(ob[-1])
                action = np.random.normal()
                ob, reward, done, _ = env.step(action)
                actions.append(action)

        for t in range(max_timesteps):
            states.append(np.array(env.cartpole.get_state()))

            observations.append(ob)
            z_observations.append(ob[-1])
            if args.policy == 'random':
                action = np.random.uniform(-10, 10)
            elif args.policy == 'hinf':
                action = controller.control(env.cartpole.get_fixation_position())
            elif args.policy == 'rl_sac':
                action = np.float64(agent.select_action(ob, evaluate=True)[0])
            else:
                raise TypeError
            

            action += np.random.normal(0, scale=args.action_noise)
            actions.append(action)
            ob, reward, done, _ = env.step(action)
            displacements.append(env.cartpole.get_state()[0])
            if args.realtime:
                time.sleep(0.02)
            
            ep_reward += reward
            if done:
                success = False
                break

        if len(z_observations) < len(actions):
            observations.append(ob)
            z_observations.append(ob[-1])

        print('Episode: {}\tReward: {}\tSuccess: {}'.format(ep, int(ep_reward), success))
        successes.append(success)
        ep_rewards.append(ep_reward)
        list_actions.append(actions)
        if args.ob_type == 'full':
            list_observations.append(observations)
        else:
            list_observations.append(z_observations)
        list_states.append(states)
        #print(list_observations)

    import scipy.io
    if not os.path.isdir(args.save_folder):
        os.makedirs(args.save_folder)
    scipy.io.savemat(os.path.join(args.save_folder, args.save_fnm), dict(actions=list_actions, observations=list_observations, states=list_states))
    print("Success rate: {} \t Average episode reward: {}".format(np.average(successes), np.average(ep_rewards)))
    print("Average displacement form center: {:5f}".format(np.absolute(np.array(displacements)).mean()))

class Argument(object):
    __isfrozen = False
    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError( "%r is a frozen class" % self )
        object.__setattr__(self, key, value)

def collect(fixation=1.0, ob_type='z_sequence', pnoise=False, rgb=False, num_traj = 10000, trial_number = 0):
    args = Argument()
    args.fixation = fixation
    args.pnoise = pnoise
    args.ob_type = ob_type
    args.rgb = rgb
    args.rendering = False
    args.debug = False
    args.realtime = False
    args.policy = 'random'
    args.resolution = 'high'
    args.true_dynamics = False
    args.zero_initial = False
    args.acion_noise = 0
    args.num_traj = num_traj
    args.action_noise = 0 
    args.save_folder = os.path.join("control_experiment_data", "trial"+str(trial_number), "datasets", "sysid")
    fixstr = mu.convert_float_to_fixstr(args.fixation)
    args.save_fnm = 'fix_'+fixstr+'_'+args.policy
    args.save_fnm += '_obtype_'+args.ob_type
    if args.pnoise:
        args.save_fnm += '_noise'
        if args.rgb:
            args.save_fnm += '_rgb'
    args.save_fnm += '.mat'
    args.perception_model_path = mu.get_perception_model_path(args.ob_type, args.fixation, rgb=args.rgb, resolution=args.resolution) \
        if args.pnoise else None
    test(args)

if __name__ == '__main__':
    args = get_args()
    test(args)
