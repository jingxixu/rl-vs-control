import numpy as np
import matplotlib.pyplot as plt
import misc_utils as mu
import time
from scipy.io import loadmat
import argparse
import torch
from sac import SAC
from gym import spaces

class ContinuousController:
    def __init__(self, A, B, C, D, Nx, x0=None):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Nx = Nx
        self.x0 = x0
        self.x = np.zeros((Nx, 1)) if self.x0 is None else self.x0

    def control(self, z):
        x_dot = self.A.dot(self.x) + self.B.dot(z)
        u = self.C.dot(self.x) + self.D.dot(z)
        self.x = self.x + x_dot * 0.02
        return u.item()

    def reset(self):
        self.x = np.zeros((Nx, 1)) if self.x0 is None else self.x0

class DiscreteController:
    def __init__(self, A, B, C, D, Nx, x0=None):
        # time step is 0.02s
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Nx = Nx
        self.x0 = x0
        self.x = np.zeros((Nx, 1)) if self.x0 is None else self.x0

    def control(self, z):
        #z = np.vstack([z[:1], z[2:3]])      
        x_next = self.A.dot(self.x) + self.B.dot(z)
        u = self.C.dot(self.x) + self.D.dot(z)
        self.x = x_next
        return u.item()

    def reset(self):
        self.x = np.zeros((self.Nx, 1)) if self.x0 is None else self.x0


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixation', type=float, default=1.0)
    parser.add_argument('--pnoise', type=bool, default=False)
    parser.add_argument('--rgb', type=bool, default=False)
    parser.add_argument('--ob_type', type=str, default='z_sequence')
    parser.add_argument('--trial_number', type=int, default=0)
    parser.add_argument('--amount_of_data', type=int, default=1e6)
    parser.add_argument('--full_state', type=bool, default=False)
    parser.add_argument('--model_fnm', type=str, default=None)
    parser.add_argument('--resolution', type=str, default='high')
    parser.add_argument('--rendering', type=bool, default=False)
    parser.add_argument('--realtime', type=bool, default=False)
    args = parser.parse_args()

    args.perception_model_path = mu.get_perception_model_path(args.ob_type, args.fixation, rgb=args.rgb, resolution=args.resolution) \
        if args.pnoise else None
    args.fixstr = mu.convert_float_to_fixstr(args.fixation)
    args.model_fnm = 'control_experiment_data/trial'+str(args.trial_number)+'/params/hinf/hinf_'+args.fixstr+'_obtype_'+args.ob_type
    if args.pnoise:
        args.model_fnm += '_pnoise'
        if args.rgb:
            args.model_fnm += '_rgb'
    if args.amount_of_data < 1e6:
        args.model_fnm += str(amt)
    args.model_fnm += '.mat'

    return args

class Argument(object):
    __isfrozen = False
    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError( "%r is a frozen class" % self )
        object.__setattr__(self, key, value)

def enjoy(args):
    fixation = args.fixation
    pnoise = args.pnoise
    rgb = args.rgb
    ob_type = args.ob_type
    trial_number = args.trial_number
    amt = args.amount_of_data
    
    
    n_trials = 100
    K = loadmat(args.model_fnm)
    print('\n\nloaded hinf: {}\n\n'.format(args.model_fnm))
    A, B, C, D, Nx = K['A'], K['B'], K['C'], K['D'], K['Nx'].squeeze()

    mu.configure_pybullet(rendering=args.rendering)
    controller = DiscreteController(A, B, C, D, Nx)
    env = mu.make_cart_pole_env(fixation=args.fixation,
                                    ob_type=args.ob_type,
                                    pnoise=args.pnoise,
                                    model_path=args.perception_model_path,
                                    rgb=args.rgb) 

    action_space = spaces.Box(low=-10, high=10, shape=(1,), dtype=np.float32)
    observation_space = spaces.Box(-np.inf, np.inf, shape=(4,))

    success_list = []
    ep_reward_list = []
    for ep in range(n_trials):
        state = env.reset()
        controller.reset()

        success = True
        ep_reward = 0
        actions = []
        observations = []
        for i in range(500):
            if args.ob_type == 'full':
                observations.append(state)
                action = controller.control(np.expand_dims(state,1))
            else:
                observations.append(state[-1])
                action = controller.control(observations[-1])
            if i < 1:
                action = 0
            actions.append(action)
            state, reward, done, _ = env.step(action)
            ep_reward += reward

            if args.realtime:
                time.sleep(0.02)
            if done:
                success = False
                break
        print('Episode: {}\tReward: {}\tSuccess: {}'.format(ep, int(ep_reward), success))
        success_list.append(success)
        ep_reward_list.append(ep_reward)
        env.close()

    print("Success rate: {} \t Average episode reward: {}".format(np.average(success_list), np.average(ep_reward_list)))

if __name__ == '__main__':
    args = get_args()
    enjoy(args)
