"""
Using a single depth image to predict the z observation.
"""


import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import numpy as np
import misc_utils as mu
import torch.nn.functional as F
import torch.optim as optim
import argparse
import torch
from torch.utils.tensorboard import SummaryWriter
import tqdm
import time
import json
import random
from perception_model import PerceptionDataset, FeatureNetDepth, ProbFeatureNetDepth, FeatureNetRGB, ProbFeatureNetRGB
from torch.distributions import MultivariateNormal, Normal, Independent

np.set_printoptions(suppress=True)


def get_args():
    parser = argparse.ArgumentParser()
    # training setup
    parser.add_argument('--dataset_path', type=str,
                        help='if not specified, use fixation to point to the correct dataset')
    parser.add_argument('--suffix', type=str,
                        help='save folder name suffix')
    parser.add_argument('--fixation', type=float)
    parser.add_argument('--seed', type=int, default=10,
                        help='random seed (default: 10)')
    parser.add_argument('--no_cuda', action='store_true',
                        help='disables CUDA training')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test_batch_size', type=int, default=256,
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of epochs to train (default: 1000)')
    parser.add_argument('--log_dir', type=str, default='logs/perception',
                        help='the path to the directory to save logs')
    parser.add_argument('--save_dir', type=str, default='models/perception',
                        help='the path to the directory to save models')
    parser.add_argument('--split', type=float, default=0.2,
                        help='proportion of the test set (default: 0.2)')
    parser.add_argument('--sample_size', type=int,
                        help="using a subset of the original dataset")
    parser.add_argument('--lr', type=float, default=0.001)

    # hyper-parameters
    parser.add_argument('--rgb', action='store_true', default=False,
                        help="use rgb data instead of depth images")
    parser.add_argument('--probabilistic', action='store_true', default=False,
                        help="predict mean and variance of the z observation")
    parser.add_argument('--resolution', type=str, default="high",
                        help="resolution of the image to use")

    args = parser.parse_args()

    # lower learning rate for perception model with rgb
    if args.rgb:
        args.lr = 0.0001

    if args.dataset_path is None:
        assert args.fixation is not None
        fixstr = mu.convert_float_to_fixstr(args.fixation)
        args.dataset_path = os.path.join('datasets', 'perception', args.resolution, 'fixation_'+fixstr)

    timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, timestr+'_'+args.suffix) if args.suffix is not None \
        else os.path.join(args.save_dir, timestr)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.log_dir = os.path.join(args.log_dir, timestr+'_'+args.suffix) if args.suffix is not None \
        else os.path.join(args.log_dir, timestr)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    return args


def train(model, device, train_loader, optimizer, epoch, writer):
    """ train one epoch """
    probabilistic = (type(model).__name__ == 'ProbFeatureNetDepth' or type(model).__name__ ==  'ProbFeatureNetRGB')
    model.train()
    pbar = tqdm.tqdm(total=len(train_loader.dataset), desc='Train | Epoch: | Loss: | Error Distance: ')
    epoch_loss = 0.0
    epoch_error_distance = 0.0
    num_samples = 0
    for batch_idx, (data, target_z) in enumerate(train_loader):
        data, target_z = data.to(device), target_z.to(device)
        optimizer.zero_grad()
        output = model(data)
        output = output.squeeze()
        if probabilistic:
            m = Normal(output[:, 0], output[:, 1])
            loss = -m.log_prob(target_z).mean()
        else:
            loss = F.mse_loss(output, target_z)
        loss.backward()
        optimizer.step()

        # update epoch stats
        output = output.detach()
        target_z = target_z.detach()
        if probabilistic:
            epoch_loss += (-m.log_prob(target_z).sum()).item()
        else:
            epoch_loss += F.mse_loss(output, target_z, reduction='sum').item()
        output = output.cpu().numpy()
        target_z = target_z.cpu().numpy()
        if probabilistic:
            epoch_error_distance += np.sum(np.absolute(output[:, 0] - target_z))
        else:
            epoch_error_distance += np.sum(np.absolute(output - target_z))

        num_samples += len(target_z)
        pbar.update(len(target_z))
        pbar.set_description(
            'Train | Epoch: {} | Loss: {:.6f} | Error Distance: {:.6f}'.format(epoch, epoch_loss / num_samples,
                                                                               epoch_error_distance / num_samples))

    epoch_loss /= num_samples
    epoch_error_distance /= num_samples
    writer.add_scalar('Train/Loss', epoch_loss, epoch)
    writer.add_scalar('Train/Error_distance', epoch_error_distance, epoch)
    pbar.close()
    return epoch_loss, epoch_error_distance


def test(model, device, test_loader, epoch, writer):
    probabilistic = (type(model).__name__ == 'ProbFeatureNetDepth' or type(model).__name__ ==  'ProbFeatureNetRGB')
    model.eval()
    epoch_loss = 0.0
    epoch_error_distance = 0.0
    num_samples = 0
    pbar = tqdm.tqdm(total=len(test_loader.dataset), desc='Test | Epoch: | Loss: | Error Distance: ')
    with torch.no_grad():
        for batch_idx, (data, target_z) in enumerate(test_loader):
            data, target_z = data.to(device), target_z.to(device)
            output = model(data)
            output = output.squeeze()
            if probabilistic:
                m = Normal(output[:, 0], output[:, 1])
                epoch_loss += (-m.log_prob(target_z).sum()).item()  # sum up batch loss
            else:
                epoch_loss += F.mse_loss(output, target_z, reduction='sum').item()  # sum up batch loss

            # calculate euclidean distance error
            output = output.detach().cpu().numpy()
            target_z = target_z.detach().cpu().numpy()
            if probabilistic:
                epoch_error_distance += np.sum(np.absolute(output[:, 0] - target_z))
            else:
                epoch_error_distance += np.sum(np.absolute(output - target_z))

            num_samples += len(target_z)
            pbar.update(len(target_z))
            pbar.set_description(
                'Test | Epoch: {} | Loss: {:.6f} | Error Distance: {:.6f}'.format(epoch, epoch_loss / num_samples,
                                                                                  epoch_error_distance / num_samples))

        epoch_loss /= num_samples
        epoch_error_distance /= num_samples
        writer.add_scalar('Test/Loss', epoch_loss, epoch)
        writer.add_scalar('Test/Error_distance', epoch_error_distance, epoch)
        pbar.close()
        return epoch_loss, epoch_error_distance


if __name__ == "__main__":
    args = get_args()

    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    dataset = PerceptionDataset(args.dataset_path, rgb=args.rgb, sample_size=args.sample_size)
    dataset_size = len(dataset)
    test_size = int(np.floor(args.split * dataset_size))
    train_size = dataset_size - test_size
    train_set, test_set = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=True)

    model = mu.construct_model(dataset.img_width, dataset.img_height, args.rgb, args.probabilistic, fs=False)
    writer = SummaryWriter(log_dir=args.log_dir)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # scheduler = ReduceLROnPlateau(optimizer, 'max', min_lr=1e-5)

    for epoch in range(args.epochs):
        train_epoch_loss, train_epoch_error_distance = train(model, device, train_loader, optimizer, epoch, writer)
        test_epoch_loss, test_epoch_error_distance = test(model, device, test_loader, epoch, writer)
        # scheduler.step(test_accuracy)

        # save checkpoint
        epoch_dir_name = "epoch_{:04}_loss_{:.7f}_dist_{:.7f}".format(epoch, test_epoch_loss, test_epoch_error_distance)
        epoch_dir = os.path.join(args.save_dir, epoch_dir_name)
        os.makedirs(epoch_dir)
        checkpoint_metadata = {
            'epoch': epoch,
            'train_epoch_loss': train_epoch_loss,
            'train_epoch_error_distance': train_epoch_error_distance,
            'test_epoch_loss': test_epoch_loss,
            'test_epoch_error_distance': test_epoch_error_distance,
            'args': vars(args)}
        json.dump(checkpoint_metadata, open(os.path.join(epoch_dir, 'checkpoint_metadata.json'), 'w'), indent=4)
        torch.save(model.state_dict(), os.path.join(epoch_dir, "feature_net.pt"))
