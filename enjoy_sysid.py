import numpy as np
import os
import misc_utils as mu
from scipy.io import loadmat
import matplotlib.pyplot as plt
import argparse


""" 
Use the last part (test ratio) of the sysid dataset to evaluate the performance of the learned dynamic model 
and then plot the prediction. The model can have 4 dimensional output (full state) or 1 dimensional output (z).
Also print the fitness. 
"""


class ContinuousDynamicModel:
    def __init__(self, A, B, C, D, Nx, x0=None):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Nx = Nx
        self.x = np.zeros((Nx, 1)) if x0 is None else x0

    def predict_ahead(self, step_forces, x0=None):
        x = self.x if x0 is None else x0
        xs = [x]
        ys = []
        for f in step_forces:
            d_x = self.A.dot(x) + self.B.dot(f)
            y = self.C.dot(x) + self.D.dot(f)
            x = x + d_x * 0.02
            xs.append(x)
            ys.append(y.item())
        return ys, xs


class DiscreteDynamicModel:
    def __init__(self, A, B, C, D, x0=None):
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.Nx = np.shape(A)[0]
        self.Ny = np.shape(C)[0]
        self.x = np.zeros((self.Nx, 1)) if x0 is None else x0

    def predict_ahead(self, step_forces, x0=None):
        x = self.x if x0 is None else x0
        xs = [x]
        ys = []
        for f in step_forces:
            x = self.A.dot(x) + self.B.dot(f)
            y = self.C.dot(x) + self.D.dot(f)
            xs.append(x)
            ys.append(np.squeeze(y))
        return ys, xs


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_fnm', type=str, default='dynamics_discrete_ob.mat')
    parser.add_argument('--data_path', type=str, default='datasets/sysid/fix_100_random.mat')
    parser.add_argument('--test_ratio', type=float, default=0.1)
    args = parser.parse_args()
    return args


def calculate_fitness(y, y_hat):
    y = np.array(y)
    y_hat = np.array(y_hat)
    # NRMSE fitness (normalized by standard deviation)
    fitness = 100 * (1 - np.linalg.norm(y - y_hat, axis=0) / np.linalg.norm(y - np.mean(y, axis=0), axis=0))
    return fitness


if __name__ == "__main__":
    args = get_args()
    model_patameters = loadmat(os.path.join('robust_control', 'params', 'dynamic_model', args.model_fnm))
    data = loadmat(args.data_path)

    dynamic_model = DiscreteDynamicModel(model_patameters['A'],
                                         model_patameters['B'],
                                         model_patameters['C'],
                                         model_patameters['D'])

    # data[actions] is (1, n_trajs) object array, each object array is (1, n_steps) array
    action_list = np.squeeze(data['actions']).tolist()
    action_list = [np.squeeze(actions) for actions in action_list]
    observation_list = np.squeeze(data['observations']).tolist()
    observation_list = [np.squeeze(observations) for observations in observation_list]
    state_list = np.squeeze(data['states']).tolist()
    state_list = [np.squeeze(states) for states in state_list]
    if dynamic_model.Ny == 4:
        # data is collected by (s, a, o)
        # s2 corresponds to o1
        # when the dynamic output is of dim 4, use cartpole full state as observation
        action_list = [actions[:-1] for actions in action_list]
        observation_list = [states[1:, :] for states in state_list]

    num_test = int(len(action_list) * args.test_ratio)
    action_list = action_list[-num_test:]
    observation_list = observation_list[-num_test:]
    state_list = state_list[-num_test:]

    predicted_observation_list = []
    predicted_internal_state_list = []
    for actions, observations, states in zip(action_list, observation_list, state_list):
        predicted_observations, predicted_states = dynamic_model.predict_ahead(actions)
        predicted_observation_list.append(predicted_observations)
        predicted_internal_state_list.append(predicted_states)

    fitnesses = np.zeros((num_test, dynamic_model.Ny))
    for i in range(num_test):
        fitness = calculate_fitness(observation_list[i], predicted_observation_list[i])
        fitnesses[i, :] = fitness
    avg_fitness = np.mean(fitnesses, axis=0)
    print("Average NRMSE fitness along all dimensions:\n{}".format(avg_fitness))

    # visualize
    for i in range(num_test):
        t = range(len(predicted_observation_list[i]))

        if dynamic_model.Ny != 1:
            fig, axs = plt.subplots(dynamic_model.Ny, 1, figsize=(12, 12))
            for y in range(dynamic_model.Ny):
                axs[y].plot(t, np.array(predicted_observation_list[i])[:, y], label='predicted')
                axs[y].plot(t, np.array(observation_list[i])[:, y], label='gt')
                axs[y].set(xlabel='time step', ylabel='output[{}]'.format(y))
            # Hide x labels
            for ax in axs.flat:
                ax.label_outer()
            plt.legend()
            plt.show()
        else:
            fig, axs = plt.subplots(dynamic_model.Ny, 1, figsize=(12, 12))
            axs.plot(t, np.array(predicted_observation_list[i]), label='predicted')
            axs.plot(t, np.array(observation_list[i]), label='gt')
            axs.set(xlabel='time step', ylabel='output')
            plt.legend()
            plt.show()

    print('finieshed')
