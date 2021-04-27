import numpy as np
import scipy.linalg as la
import argparse
import scipy.io
import misc_utils as mu
import os 

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ob_type', type = str, default = 'z_sequence')
    parser.add_argument('--pnoise', type = bool, default=False)
    parser.add_argument('--rgb', type=bool, default=False)
    parser.add_argument('--fixation', type=float, default=1.0)
    parser.add_argument('--id_horizon', type=int, default=10)
    parser.add_argument('--trial_number', type=int, default=0)
    parser.add_argument('--amount_of_data', type=int, default=np.inf)

    args = parser.parse_args()
    return args

def idFromState(observations, actions):
    n = observations[0].shape[1]
    m = actions[0].T.shape[1]

    T = len(observations)
   
    regressors = np.zeros((0,n+m))
    targets = np.zeros((0,n))
    for i in range(T):
        regressors = np.vstack([regressors, np.hstack([observations[i][:-1], actions[i].T[:-1]])])    
        targets = np.vstack([targets, observations[i][1:]])

    AB = la.solve(regressors.T@regressors+0e-5*np.eye(n+m), regressors.T@targets)

    A = AB[:n].T
    B = AB[n:n+m].T
    C = np.eye(n)
    D = np.zeros((n,m))
    return (A,B,C,D) 

def idFromObservations(observations, actions, idAlg = 'ARXHK', p = 10, Nx = 4, amt=np.inf):
    n = observations[0].T.shape[1]
    m = actions[0].T.shape[1]
    T = len(observations)
    t = 0

    if idAlg == 'ARXHK':
        Z = np.zeros((0, 2*p))
        Y = np.zeros(0)
        
        for i in range(T):
            obs = observations[i].squeeze()
            act = actions[i].squeeze()
            t = t + len(obs)-p
            Y = np.hstack([Y, obs[p:]])
            regressors = np.zeros((len(obs)-p, 0))
            for j in range(p):
                regressors = np.hstack([regressors, np.vstack([act[j:j-p], obs[j:j-p]]).T])
            Z = np.vstack([Z, regressors])
            if t > amt:
                break
        
        Z = Z
        Y = Y

        H = la.solve(Z.T@Z, Z.T@Y)
        Hankel = np.zeros((0, 2*p))
        for j in range(p):
            if j == 0:
                Hankel = np.vstack([Hankel, H])
            else:
                Hankel = np.vstack([Hankel, np.hstack([np.zeros(2*j), H[:-2*j]])])

        U, Sigma, VT = la.svd(Hankel)
        V = VT.T
        U = U[:, :Nx]
        V = V[:, :Nx]

        S = np.sqrt(Sigma[:Nx])
        S = np.diag(S)
        Observability = U@S

        Controllability = S@V.T
        Chat = Observability[:n]
        BKhat = Controllability[:, -(m+n):]

        Bhat = BKhat[:,:m]
        Khat = BKhat[:,m:]

        AKhat = la.solve(Observability[:-n].T@Observability[:-n], Observability[:-n].T@Observability[n:])

        Ahat = AKhat + Khat@Chat

    return (Ahat, Bhat, Chat, np.zeros((n, m)))

class Argument(object):
    __isfrozen = False
    def __setattr__(self, key, value):
        if self.__isfrozen and not hasattr(self, key):
            raise TypeError( "%r is a frozen class" % self )
        object.__setattr__(self, key, value)
  
def performID(args):
    fixation = args.fixation
    pnoise = args.pnoise 
    rgb = args.rgb
    ob_type = args.ob_type
    trial_number=args.trial_number
    p = args.id_horizon
    amt=args.amount_of_data
    
    dataset_folder = 'control_experiment_data/trial'+str(trial_number) + '/datasets/sysid/'
    dataset_fnm = 'fix_'+ mu.convert_float_to_fixstr(args.fixation)+'_random'+'_obtype_'+\
            args.ob_type
   
    if pnoise:
        dataset_fnm += '_noise'
        if rgb:
            dataset_fnm += '_rgb'

    dataset = scipy.io.loadmat(os.path.join(dataset_folder, dataset_fnm))
    actions = dataset['actions'].squeeze()#[:num_traj]
    observations = dataset['observations'].squeeze()#[:num_traj]
    states = dataset['states'].squeeze()


    if args.ob_type == 'full':
        A, B, C, D = idFromState(observations,actions)
    else:
        A, B, C, D = idFromObservations(observations, actions, amt = amt,p = p)

    save_dir = 'control_experiment_data/trial'+str(trial_number) + '/params/dynamic_model'
    save_fnm = 'fix_'+mu.convert_float_to_fixstr(args.fixation)
    save_fnm = save_fnm + '_obtype_' + args.ob_type
    if args.pnoise:
        save_fnm = save_fnm + '_pnoise'
        if args.rgb:
            save_fnm = save_fnm + '_rgb'
    if amt < 1e6:
        save_fnm = save_fnm + str(amt)
    save_fnm = save_fnm + '.mat'

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    scipy.io.savemat(os.path.join(save_dir, save_fnm), {"A": A, "B": B, "C": C, "D": D})
    
    return A,B,C,D

if __name__ == '__main__':
    args = get_args()
    performID(args)
