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

""" Use the learned perception model on the same test set again and calculate new evaluation metrics """


def get_args():
    parser = argparse.ArgumentParser(description='Collect training data')

    parser.add_argument('--fixation', default=1.0, type=float)
    parser.add_argument('--perception_model_path', type=str)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--probabilistic', action="store_true", default=False)
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
        args.dataset_path = os.path.join('datasets', 'perception', 'high', 'fixation_' + fixstr)

    return args


if __name__ == "__main__":
    args = get_args()

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # load saved args
    model_args = json.load(open(os.path.join(args.perception_model_path, 'checkpoint_metadata.json')))['args']

    dataset = PerceptionDataset(args.dataset_path, rgb=model_args['rgb'])
    dataset_size = len(dataset)
    test_size = int(np.floor(args.split * dataset_size))
    train_size = dataset_size - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True)

    model = FeatureNetDepth(height=dataset.img_height, width=dataset.img_width).to(device) if not model_args['rgb'] else \
        FeatureNetRGB(height=dataset.img_height, width=dataset.img_width).to(device)
    model.load_state_dict(torch.load(os.path.join(args.perception_model_path, 'feature_net.pt')))
    model.eval()

    gt_z_observations = []
    predicted_mean = []
    predicted_std = []
    num_samples = 0
    loss = 0
    error_distance = 0
    outputs = np.zeros(0)
    targets = np.zeros(0)
    for batch_idx, (data, target_z) in enumerate(test_loader):
        data, target_z = data.to(device), target_z.to(device)
        output = model(data)
        output = output.squeeze()

        # update epoch stats
        output_np = output.detach().cpu().numpy()
        target_z_np = target_z.detach().cpu().numpy()
        outputs = np.concatenate((outputs, output_np))
        targets = np.concatenate((targets, target_z_np))

        loss += F.mse_loss(output, target_z, reduction='sum').item()
        error_distance += np.sum(np.absolute(output_np - target_z_np))
        num_samples += len(target_z)

    loss /= num_samples
    error_distance /= num_samples
    RMS = np.sqrt(np.mean((outputs - targets)**2))
    relativeRMS = np.sqrt(np.mean(((outputs - targets) / targets)**2))
    normalizedRMS = RMS / (max(targets) - min(targets))
    print(f'nomalizedRMS: {normalizedRMS}')
    print(f'relativeRMS: {relativeRMS}')
    print(f'RMS: {RMS}')

