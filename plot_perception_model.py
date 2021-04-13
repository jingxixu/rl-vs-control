from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import misc_utils as mu
import torch.nn.functional as F
import argparse
import torch
import tqdm
import time
import json
import random
from perception_model import PerceptionDataset, FeatureNetDepth, ProbFeatureNetDepth, CalibrationNet, ProbFeatureNetRGB, FeatureNetRGB
from torch.distributions import MultivariateNormal, Normal, Independent
import matplotlib.pyplot as plt

""" Use the learned perception model on the test set again and plot the predicted mean with variance """


def get_args():
    parser = argparse.ArgumentParser(description='Collect training data')

    parser.add_argument('--fixation', default=1.0, type=float)
    parser.add_argument('--perception_model_path', type=str)
    parser.add_argument('--calibration_model_path', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--probabilistic', action="store_true", default=False)
    parser.add_argument('--calibration', action="store_true", default=False)
    args = parser.parse_args()

    # fixed args
    # keep the same as train_perception_model.py to make sure they use the same test set
    args.no_cuda = False
    args.test_batch_size = 256
    args.seed = 10
    args.split = 0.2

    if args.dataset_path is None:
        assert args.fixation is not None
        fixstr = mu.convert_float_to_fixstr(args.fixation)
        args.dataset_path = os.path.join('datasets', 'perception', 'fixation_' + fixstr)

    return args


if __name__ == "__main__":
    args = get_args()

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = PerceptionDataset(args.dataset_path, rgb=True)
    dataset_size = len(dataset)
    test_size = int(np.floor(args.split * dataset_size))
    train_size = dataset_size - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True)

    model = ProbFeatureNetRGB(height=dataset.img_height, width=dataset.img_width).to(device) if args.probabilistic \
        else FeatureNetRGB(height=dataset.img_height, width=dataset.img_width).to(device)
    model.load_state_dict(torch.load(os.path.join(args.perception_model_path, 'feature_net.pt')))
    model.eval()

    if args.calibration:
        model = CalibrationNet(model).to(device)
        model.load_state_dict(torch.load(os.path.join(args.calibration_model_path, 'feature_net.pt')))
        model.eval()

    gt_z_observations = []
    predicted_mean = []
    predicted_std = []
    num_samples = 0
    loss = 0
    error_distance = 0
    for batch_idx, (data, target_z) in enumerate(test_loader):
        data, target_z = data.to(device), target_z.to(device)
        output = model(data)
        output = output.squeeze()

        # update epoch stats
        output_np = output.detach().cpu().numpy()
        target_z_np = target_z.detach().cpu().numpy()
        gt_z_observations += target_z_np.tolist()
        predicted_mean += output_np[:, 0].tolist()
        predicted_std += output_np[:, 1].tolist()

        if args.probabilistic:
            m = Normal(output[:, 0], output[:, 1])
            loss += (-m.log_prob(target_z).sum()).item()
            error_distance += np.sum(np.absolute(output_np[:, 0] - target_z_np))
        else:
            loss += F.mse_loss(output, target_z, reduction='sum').item()
            error_distance += np.sum(np.absolute(output_np - target_z_np))
        num_samples += len(target_z)

    loss /= num_samples
    error_distance /= num_samples

    # combine_and_filtered = filter(lambda x: -0.25 < x[0] < 0.25, zip(gt_z_observations, predicted_mean, predicted_std))
    # gt_z_observations, predicted_mean, predicted_std = zip(*combine_and_filtered)
    plt.scatter(np.abs(np.array(predicted_mean) - np.array(gt_z_observations)), np.abs(predicted_std), marker='.', s=1)
    plt.xlabel('error')
    plt.ylabel('std')

    # combine = sorted(zip(gt_z_observations, predicted_mean, predicted_std), key=lambda pair: pair[0])
    # gt_z_observations, predicted_mean, predicted_std = zip(*combine)
    # plt.scatter(np.array(predicted_mean) - np.array(gt_z_observations), gt_z_observations, marker='.', s=1)
    # axis = plt.gca()
    # axis.fill_betweenx(gt_z_observations, -np.array(predicted_std), np.array(predicted_std), alpha=0.2)

    # plt.scatter(range(len(gt_z_observations)), np.array(predicted_mean) - np.array(gt_z_observations), marker='.', s=1)
    # axis = plt.gca()
    # axis.fill_between(range(len(gt_z_observations)), -np.array(predicted_std), np.array(predicted_std), alpha=0.2)
    plt.show()
    print("correlation coefficient: {}".format(
        np.corrcoef(np.abs(np.array(predicted_mean) - np.array(gt_z_observations)), np.abs(predicted_std))))
    print('finished')
