import misc_utils as mu
import numpy as np
import time
import tqdm
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--true_dynamics', action='store_true')
    parser.add_argument('--num_trials', type=int, default=100)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    mu.configure_pybullet(rendering=False, debug=False)
    if args.true_dynamics:
        env = mu.make_cart_pole_env_true_dyanmics(
            fixation=1.0,
            ob_type="z_sequence"
        )
    else:
        env = mu.make_cart_pole_env(
            fixation=1.0,
            ob_type="z_sequence"
        )

    # recorded data
    deltas = []
    states = []
    z_observations = []

    pbar = tqdm.tqdm(total=args.num_trials)
    for e in range(args.num_trials):
        env.reset()
        initial_state = env.cartpole.get_state()
        states.append(initial_state)
        initial_z = env.cartpole.get_fixation_position()
        z_observations.append(initial_z)
        old_z = initial_z
        # print("initial state: {}, initial z: {}".format(initial_state, initial_z))
        for i in range(500):
            action = np.random.uniform(-10, 10)
            observation, reward, done, info = env.step(action)
            time.sleep(0.02)
            state = env.cartpole.get_state()
            states.append(state)
            z = env.cartpole.get_fixation_position()
            z_observations.append(z)
            deltas.append(z - old_z)
            # print(action, state, z - old_z)
            old_z = z
            if done:
                break
        pbar.update(1)

    pbar.close()

    # get stats
    print('average delta: {}'.format(np.average(np.abs(np.array(deltas)))))
    states_npy = np.array(states)
    print('state mean: {}'.format(np.mean(np.array(states), axis=0)))
    print('state std: {}'.format(np.std(np.array(states), axis=0)))