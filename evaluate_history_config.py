from __future__ import division
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split, IterableDataset
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
import h5py

""" Try using different history sizes and skips for evaluating the performance of predicting full state """

np.set_printoptions(suppress=True)


class PixelMLDataset(IterableDataset):
    def __init__(self, dataset_path, fixation, start, end, history_size, image_skip):
        self.image_skip = image_skip
        self.history_size = history_size
        self.episode_len = 500
        self.dataset_path = dataset_path
        self.fixation = fixation
        self.start = start
        self.end = end

        self.fixation_suffix = str(self.fixation).replace('.', '')

        # load all the data, this might not work if the dataset is huge and memory is small
        self.folder_names = os.listdir(self.dataset_path)
        self.folder_names.sort()
        self.n_traj = len(self.folder_names)

        # (n, 500)
        self.actions = np.zeros((len(self.folder_names), self.episode_len), dtype=np.int64)
        self.states = np.zeros((len(self.folder_names), self.episode_len, 4), dtype=np.float32)
        for i, folder_name in enumerate(self.folder_names):
            self.actions[i] = np.squeeze(np.load(os.path.join(self.dataset_path, folder_name, 'actions.npy')))
            self.states[i] = np.squeeze(np.load(os.path.join(self.dataset_path, folder_name, 'states.npy')))

        self.depths_file = h5py.File(name='merged_depths.hdf5', mode='r')
        self.depths = self.depths_file['merged_dataset']

        print(np.mean(self.states.reshape(-1, 4), axis=0))
        print(np.std(self.states.reshape(-1, 4), axis=0))

        # load an example depth
        sample_depth = self.depths[0, 0]
        self.img_height, self.img_width = sample_depth.shape

    def __len__(self):
        return len(self.folder_names[self.start:self.end]) * self.episode_len

    def __iter__(self):
        for ep_idx, folder_name in enumerate(self.folder_names[self.start:self.end]):
            depth_imgs = np.load(os.path.join(self.dataset_path, folder_name, 'depth_imgs_10.npy'))
            for wp_idx in range(self.episode_len):
                # y = self.actions[ep_idx, wp_idx]
                y = self.states[ep_idx, wp_idx]
                frame_idx_list = []
                frame_idx = wp_idx
                while frame_idx >= 0 and len(frame_idx_list) < self.history_size:
                    frame_idx_list.append(frame_idx)
                    frame_idx -= (self.image_skip + 1)
                frame_idx_list.reverse()
                num_paddings = self.history_size - len(frame_idx_list)

                padding = np.zeros((num_paddings, ) + depth_imgs[0].shape, dtype=np.float32)
                x = np.vstack((padding, depth_imgs[frame_idx_list]))
                # x = x[None, ...]
                # print(x.shape, y.shape)
                yield x, y


class PixelMLDatasetMap(Dataset):
    def __init__(self, dataset_path, fixation):
        self.image_skip = 1
        self.history_size = 100
        self.episode_len = 500
        self.dataset_path = dataset_path
        self.fixation = fixation

        self.fixation_suffix = str(self.fixation).replace('.', '')

        # load all the data, this might not work if the dataset is huge and memory is small
        self.folder_names = os.listdir(self.dataset_path)
        self.folder_names.sort()
        self.n_traj = len(self.folder_names)

        # (n, 500)
        self.actions = np.zeros((len(self.folder_names), self.episode_len), dtype=np.int64)
        self.states = np.zeros((len(self.folder_names), self.episode_len, 4), dtype=np.float32)
        for i, folder_name in enumerate(self.folder_names):
            self.actions[i] = np.squeeze(np.load(os.path.join(self.dataset_path, folder_name, 'actions.npy')))
            self.states[i] = np.squeeze(np.load(os.path.join(self.dataset_path, folder_name, 'states.npy')))

        self.depths_file = h5py.File(name='merged_depths.hdf5', mode='r')
        self.depths = self.depths_file['merged_dataset']

        # load an example depth
        sample_depth = self.depths[0, 0]
        self.img_height, self.img_width = sample_depth.shape

    def __len__(self):
        return len(self.folder_names) * self.episode_len

    def __getitem__(self, idx):
        # calculate episode idx
        ep_idx = idx // self.episode_len
        wp_idx = idx % self.episode_len

        # construct a sample with history
        if wp_idx + 1 < self.history_size:
            n_repeats = self.history_size - wp_idx - 1
            x = self.depths[ep_idx, 0:wp_idx + 1, ...]
            x = np.append([self.depths[ep_idx, 0, ...]] * n_repeats, x, axis=0)
        else:
            x = self.depths[ep_idx, wp_idx + 1 - self.history_size:wp_idx + 1, ...]
        x = x[None, ...]
        # y = self.actions[ep_idx, wp_idx]
        y = self.states[ep_idx, wp_idx]
        return x, y


# -------------------------------------------------------------------------------------------
# Networks


class PixelMLNetDepth(nn.Module):
    def __init__(self, width, height, history_size):
        super(PixelMLNetDepth, self).__init__()
        self.width = width
        self.height = height

        self.net_conv2d = nn.Sequential(
            nn.Conv2d(history_size, 32, 3, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

        self.net_conv3d = nn.Sequential(
            nn.Conv3d(1, 32, 3, 2),
            nn.ReLU(),
            nn.Conv3d(32, 64, 3, 2),
            nn.ReLU(),
            nn.Conv3d(64, 128, 3, 2),
            nn.ReLU(),
            nn.Conv3d(128, 256, 3, 2),
            nn.ReLU(),
            nn.Conv3d(256, 256, 3, 2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 64),
            nn.Linear(64, 2)
        )

    def forward(self, depth):
        y = self.net_conv2d(depth)
        return y


def get_args():
    parser = argparse.ArgumentParser()
    # training setup
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--history_size', type=int, required=True)
    parser.add_argument('--image_skip', type=int, required=True)
    parser.add_argument('--fixation', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=10,
                        help='random seed (default: 10)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='input batch size for training (default: 256)')
    parser.add_argument('--test_batch_size', type=int, default=128,
                        help='input batch size for testing (default: 256)')
    parser.add_argument('--epochs', type=int, default=10000,
                        help='number of epochs to train (default: 10000)')
    parser.add_argument('--log_dir', type=str, default='pixel_ml_logs',
                        help='the path to the directory to save logs')
    parser.add_argument('--save_dir', type=str, default='pixel_ml_checkpoint',
                        help='the path to the directory to save models')
    parser.add_argument('--split', type=float, default=0.2,
                        help='proportion of the test set (default: 0.2)')
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    timestr = time.strftime("%Y-%m-%d_%H-%M-%S")
    args.save_dir = os.path.join(args.save_dir, timestr)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args.log_dir = os.path.join(args.log_dir, timestr)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    return args


def train(model, device, train_loader, optimizer, epoch, writer):
    """ train one epoch """
    model.train()
    pbar = tqdm.tqdm(total=len(train_loader.dataset), desc='Train | Epoch: | Loss: | Acc: ')
    epoch_loss = 0.0
    num_samples = 0
    correct = 0
    for batch_idx, (data, target_z) in enumerate(train_loader):
        data, target_z = data.to(device), target_z.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target_z)
        loss.backward()
        optimizer.step()

        # update epoch stats
        output = output.detach()
        target_z = target_z.detach()
        epoch_loss += F.mse_loss(output, target_z, reduction='sum').item()

        num_samples += len(target_z)
        pbar.update(len(target_z))
        pbar.set_description(
            'Train | Epoch: {} | Loss: {:.6f} | Acc: {:.6f}'.format(epoch, epoch_loss / num_samples,
                                                                    correct / num_samples))

    epoch_loss /= (num_samples * 4)
    writer.add_scalar('Train/Loss', epoch_loss, epoch)
    pbar.close()
    return epoch_loss, 0


def test(model, device, test_loader, epoch, writer):
    model.eval()
    epoch_loss = 0.0
    num_samples = 0
    correct = 0
    pbar = tqdm.tqdm(total=len(test_loader.dataset), desc='Test | Epoch: | Loss: | Acc: ')
    accumulate_error = np.zeros(4, dtype=np.float32)
    with torch.no_grad():
        for batch_idx, (data, target_z) in enumerate(test_loader):
            data, target_z = data.to(device), target_z.to(device)
            output = model(data)
            epoch_loss += F.mse_loss(output, target_z, reduction='sum').item()  # sum up batch loss

            # calculate accuracy
            output = output.detach().cpu().numpy()
            target_z = target_z.detach().cpu().numpy()
            accumulate_error += np.sum((output - target_z)**2, axis=0)

            num_samples += len(target_z)
            pbar.update(len(target_z))
            pbar.set_description(
                'Test | Epoch: {} | Loss: {:.6f} | Acc: {:.6f}'.format(epoch, epoch_loss / num_samples,
                                                                                  correct / num_samples))

        epoch_loss /= (num_samples * 4)
        accumulate_error /= num_samples
        print(accumulate_error)
        writer.add_scalar('Test/Loss', epoch_loss, epoch)
        pbar.close()
        return epoch_loss, 0


if __name__ == "__main__":
    args = get_args()
    model_metadata = vars(args)
    json.dump(model_metadata, open(os.path.join(args.save_dir, 'model_metadata.json'), 'w'), indent=4)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # dataset = PixelMLDataset(args.dataset_path, fixation=args.fixation)
    # dataset_size = len(dataset)
    # test_size = int(np.floor(args.split * dataset_size))
    # train_size = dataset_size - test_size
    # train_set, test_set = random_split(dataset, [train_size, test_size])
    train_set = PixelMLDataset(args.dataset_path, fixation=args.fixation, start=0, end=800, history_size=args.history_size, image_skip=args.image_skip)
    test_set = PixelMLDataset(args.dataset_path, fixation=args.fixation, start=800, end=1000, history_size=args.history_size, image_skip=args.image_skip)

    train_loader = DataLoader(train_set, batch_size=args.batch_size)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size)

    model = PixelMLNetDepth(height=train_set.img_height, width=train_set.img_width, history_size=args.history_size).to(device)
    writer = SummaryWriter(log_dir=args.log_dir)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    # scheduler = ReduceLROnPlateau(optimizer, 'max', min_lr=1e-5)

    for epoch in range(1, args.epochs + 1):
        train_epoch_loss, train_epoch_acc = train(model, device, train_loader, optimizer, epoch, writer)
        test_epoch_loss, test_epoch_acc = test(model, device, test_loader, epoch, writer)
        # scheduler.step(test_accuracy)

        # save checkpoint
        epoch_dir_name = "epoch_{:04}_loss_{:.7f}_acc_{:.7f}".format(epoch, test_epoch_loss, test_epoch_acc)
        epoch_dir = os.path.join(args.save_dir, epoch_dir_name)
        os.makedirs(epoch_dir)
        checkpoint_metadata = {
            'train_epoch_loss': train_epoch_loss,
            'train_epoch_acc': train_epoch_acc,
            'test_epoch_loss': test_epoch_loss,
            'test_epoch_acc': test_epoch_acc}
        json.dump(checkpoint_metadata, open(os.path.join(epoch_dir, 'checkpoint_metadata.json'), 'w'), indent=4)
        torch.save(model.state_dict(), os.path.join(epoch_dir, "feature_net.pt"))
