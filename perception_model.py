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


class PerceptionDataset(Dataset):
    def __init__(self, dataset_path, rgb, sample_size=None):
        self.history_size = 200
        self.episode_len = 500
        self.dataset_path = dataset_path
        self.rgb = rgb

        # load all the observations, this might not work if the dataset is huge and memory is small
        self.folder_names = [f for f in os.listdir(self.dataset_path) if
                             os.path.isdir(os.path.join(self.dataset_path, f))]
        self.folder_names.sort()
        self.n_traj = len(self.folder_names)
        self.z_observations = np.load(os.path.join(self.dataset_path, 'observations.npy'))

        if sample_size is not None:
            indices = random.sample(range(self.n_traj), sample_size)
            self.folder_names = [self.folder_names[i] for i in indices]
            self.z_observations = self.z_observations[indices]
            self.n_traj = sample_size

        # load a sample
        if rgb:
            sample_rgb = np.load(os.path.join(self.dataset_path, self.folder_names[0], 'rgb.npy'))
            self.img_height, self.img_width, _ = sample_rgb.shape
        else:
            sample_depth = np.load(os.path.join(self.dataset_path, self.folder_names[0], 'depth.npy'))
            self.img_height, self.img_width = sample_depth.shape

    def __len__(self):
        return len(self.folder_names)

    def __getitem__(self, idx):
        folder_name = self.folder_names[idx]
        if self.rgb:
            rgb_im = np.load(os.path.join(self.dataset_path, folder_name, 'rgb.npy'))
            rgb_im = np.transpose(rgb_im, (2, 0, 1))
            return rgb_im, self.z_observations[idx]
        else:
            depth = np.load(os.path.join(self.dataset_path, folder_name, 'depth.npy'))
            depth = depth[None, ...]
            return depth, self.z_observations[idx]


class PerceptionFSDataset(Dataset):
    """ dataset for predicting full state with a sequence of history images"""
    def __init__(self, dataset_path, rgb=False):
        self.rgb = rgb
        self.X_dataset_path = os.path.join(dataset_path, 'rgb', 'X') if self.rgb \
            else os.path.join(dataset_path, 'depth', 'X')
        self.Y_dataset_path = os.path.join(dataset_path, 'rgb', 'Y') if self.rgb \
            else os.path.join(dataset_path, 'depth', 'Y')

        self.xs = os.listdir(self.X_dataset_path)
        self.dataset_path = dataset_path

        # take a sample
        sample_x = np.load(os.path.join(self.X_dataset_path, self.xs[0]))
        self.img_height, self.img_width = sample_x.shape[2], sample_x.shape[3]

    def __len__(self):
        return len(self.xs)

    def __getitem__(self, idx):
        x = np.load(os.path.join(self.X_dataset_path, str(idx) + '.npy'))
        x = np.squeeze(x)
        y = np.load(os.path.join(self.Y_dataset_path, str(idx) + '.npy'))
        return x, y.astype(np.float32)


# -------------------------------------------------------------------------------------------
# Networks


class FeatureNetDepth(nn.Module):
    probabilistic = False
    rgb = False

    def __init__(self, height, width):
        super(FeatureNetDepth, self).__init__()
        self.height = height
        self.width = width

        if self.height == 120 and self.width == 100:
            # n, 1, 120, 100
            self.model = nn.Sequential(nn.Conv2d(1, 32, 3, 2),
                                       # n, 32, 59, 49
                                       nn.ReLU(),
                                       nn.Conv2d(32, 64, 3, 2),
                                       # n, 64, 29, 24
                                       nn.ReLU(),
                                       nn.Conv2d(64, 128, 3, 2),
                                       # n, 128, 14, 11
                                       nn.ReLU(),
                                       nn.Conv2d(128, 256, 3, 2),
                                       # n, 256, 6, 5
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, 3, 2),
                                       # n, 256, 2, 2
                                       nn.ReLU(),
                                       nn.Flatten(),
                                       # n, 1024
                                       nn.Linear(1024, 64),
                                       # n, 64
                                       nn.ReLU(),
                                       nn.Linear(64, 1)
                                       # n, 1
                                       )
        elif self.height == 80 and self.width == 67:
            # n, 1, 80, 67
            self.model = nn.Sequential(nn.Conv2d(1, 32, 3, 2),
                                       # n, 32, 39, 33
                                       nn.ReLU(),
                                       nn.Conv2d(32, 64, 3, 2),
                                       # n, 64, 19, 16
                                       nn.ReLU(),
                                       nn.Conv2d(64, 128, 3, 2),
                                       # n, 128, 9, 7
                                       nn.ReLU(),
                                       nn.Conv2d(128, 256, 3, 2),
                                       # n, 256, 4, 3
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, 3, 2),
                                       # n, 256, 1, 1
                                       nn.ReLU(),
                                       nn.Flatten(),
                                       # n, 256
                                       nn.Linear(256, 64),
                                       # n, 64
                                       nn.ReLU(),
                                       nn.Linear(64, 1)
                                       # n, 1
                                       )
        elif self.height == 60 and self.width == 50:
            # low resolution uses one less layer
            # n, 1, 60, 50
            self.model = nn.Sequential(nn.Conv2d(1, 32, 3, 2),
                                       # n, 32, 29, 24
                                       nn.ReLU(),
                                       nn.Conv2d(32, 64, 3, 2),
                                       # n, 64, 14, 11
                                       nn.ReLU(),
                                       nn.Conv2d(64, 128, 3, 2),
                                       # n, 128, 6, 5
                                       nn.ReLU(),
                                       nn.Conv2d(128, 256, 3, 2),
                                       # n, 256, 2, 2
                                       nn.ReLU(),
                                       nn.Flatten(),
                                       # n, 1024
                                       nn.Linear(1024, 64),
                                       # n, 64
                                       nn.ReLU(),
                                       nn.Linear(64, 1)
                                       # n, 1
                                       )
        else:
            raise TypeError('Unsupported image size!')

    def forward(self, x):
        output = self.model(x)
        return output


class FullStateNetDepth(nn.Module):
    """ Predicting the full state with a sequence of past depth images """
    probabilistic = False
    rgb = False

    def __init__(self, height, width, history_size):
        super(FullStateNetDepth, self).__init__()
        self.height = height
        self.width = width
        self.history_size = history_size

        if self.history_size == 5:
            if self.height == 120 and self.width == 100:
                # n, 5, 120, 100
                self.model = nn.Sequential(nn.Conv2d(5, 32, 3, 2),
                                           # n, 32, 59, 49
                                           nn.ReLU(),
                                           nn.Conv2d(32, 64, 3, 2),
                                           # n, 64, 29, 24
                                           nn.ReLU(),
                                           nn.Conv2d(64, 128, 3, 2),
                                           # n, 128, 14, 11
                                           nn.ReLU(),
                                           nn.Conv2d(128, 256, 3, 2),
                                           # n, 256, 6, 5
                                           nn.ReLU(),
                                           nn.Conv2d(256, 256, 3, 2),
                                           # n, 256, 2, 2
                                           nn.ReLU(),
                                           nn.Flatten(),
                                           # n, 1024
                                           nn.Linear(1024, 64),
                                           # n, 64
                                           nn.ReLU(),
                                           nn.Linear(64, 4)
                                           # n, 4
                                           )
            else:
                raise TypeError('Unsupported image size!')

    def forward(self, x):
        output = self.model(x)
        return output


class Squeeze(torch.nn.Module):
    def forward(self, x):
        return torch.squeeze(x, dim=2)  # only squeeze dim 2 andn ignore dim 0 (n)


class FullStateNetRGB(nn.Module):
    """ Predicting the full state with a sequence of past rgb images """
    probabilistic = False
    rgb = False

    def __init__(self, height, width, history_size):
        super(FullStateNetRGB, self).__init__()
        self.height = height
        self.width = width
        self.history_size = history_size

        if self.history_size == 5:
            if self.height == 120 and self.width == 100:
                # n, 3, 5, 120, 100
                self.model = nn.Sequential(nn.Conv3d(3, 32, 3, 2),
                                           # n, 32, 2, 59, 49
                                           nn.ReLU(),
                                           nn.Conv3d(32, 64, kernel_size=(2, 3, 3), stride=2),
                                           # n, 64, 1, 29, 24
                                           Squeeze(),
                                           # n, 64, 29, 24
                                           nn.ReLU(),
                                           nn.Conv2d(64, 128, 3, 2),
                                           # n, 128, 14, 11
                                           nn.ReLU(),
                                           nn.Conv2d(128, 256, 3, 2),
                                           # n, 256, 6, 5
                                           nn.ReLU(),
                                           nn.Conv2d(256, 256, 3, 2),
                                           # n, 256, 2, 2
                                           nn.ReLU(),
                                           nn.Flatten(),
                                           # n, 1024
                                           nn.Linear(1024, 64),
                                           # n, 64
                                           nn.ReLU(),
                                           nn.Linear(64, 4)
                                           # n, 4
                                           )
            else:
                raise TypeError('Unsupported image size!')

    def forward(self, x):
        x = torch.transpose(x, 1, 2).float()
        output = self.model(x)
        return output


class ProbFeatureNetDepth(nn.Module):
    probabilistic = True
    rgb = False

    def __init__(self, width, height):
        super(ProbFeatureNetDepth, self).__init__()
        self.width = width
        self.height = height

        self.conv1 = nn.Conv2d(1, 32, 3, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, 2)
        self.conv4 = nn.Conv2d(128, 256, 3, 2)
        self.conv5 = nn.Conv2d(256, 256, 3, 2)

        # TODO fix this
        if self.height == 240 and self.width == 320:
            self.fc1 = nn.Linear(2048, 64)
        elif self.height == 120 and self.width == 100:
            self.conv6 = nn.Conv2d(256, 256, 3, 2)
            self.fc1 = nn.Linear(1024, 64)
        else:
            raise TypeError('Unsupported image size!')

        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        if self.height == 240 and self.width == 320:
            x = self.conv6(x)
            x = F.relu(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        mean = x[:, 0]
        std = F.sigmoid(x[:, 1])
        z = torch.stack([mean, std], 1)
        return z


class FeatureNetRGB(nn.Module):
    probabilistic = False
    rgb = True

    def __init__(self, height, width):
        super(FeatureNetRGB, self).__init__()
        self.height = height
        self.width = width

        if self.height == 120 and self.width == 100:
            # n, 3, 120, 100
            self.model = nn.Sequential(nn.Conv2d(3, 32, 3, 2),
                                       # n, 32, 59, 49
                                       nn.ReLU(),
                                       nn.Conv2d(32, 64, 3, 2),
                                       # n, 64, 29, 24
                                       nn.ReLU(),
                                       nn.Conv2d(64, 128, 3, 2),
                                       # n, 128, 14, 11
                                       nn.ReLU(),
                                       nn.Conv2d(128, 256, 3, 2),
                                       # n, 256, 6, 5
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, 3, 2),
                                       # n, 256, 2, 2
                                       nn.ReLU(),
                                       nn.Flatten(),
                                       # n, 1024
                                       nn.Linear(1024, 64),
                                       # n, 64
                                       nn.ReLU(),
                                       nn.Linear(64, 1)
                                       # n, 1
                                       )
        elif self.height == 80 and self.width == 67:
            # n, 3, 80, 67
            self.model = nn.Sequential(nn.Conv2d(3, 32, 3, 2),
                                       # n, 32, 39, 33
                                       nn.ReLU(),
                                       nn.Conv2d(32, 64, 3, 2),
                                       # n, 64, 19, 16
                                       nn.ReLU(),
                                       nn.Conv2d(64, 128, 3, 2),
                                       # n, 128, 9, 7
                                       nn.ReLU(),
                                       nn.Conv2d(128, 256, 3, 2),
                                       # n, 256, 4, 3
                                       nn.ReLU(),
                                       nn.Conv2d(256, 256, 3, 2),
                                       # n, 256, 1, 1
                                       nn.ReLU(),
                                       nn.Flatten(),
                                       # n, 256
                                       nn.Linear(256, 64),
                                       # n, 64
                                       nn.ReLU(),
                                       nn.Linear(64, 1)
                                       # n, 1
                                       )
        elif self.height == 60 and self.width == 50:
            # low resolution uses one less layer
            # n, 3, 60, 50
            self.model = nn.Sequential(nn.Conv2d(3, 32, 3, 2),
                                       # n, 32, 29, 24
                                       nn.ReLU(),
                                       nn.Conv2d(32, 64, 3, 2),
                                       # n, 64, 14, 11
                                       nn.ReLU(),
                                       nn.Conv2d(64, 128, 3, 2),
                                       # n, 128, 6, 5
                                       nn.ReLU(),
                                       nn.Conv2d(128, 256, 3, 2),
                                       # n, 256, 2, 2
                                       nn.ReLU(),
                                       nn.Flatten(),
                                       # n, 1024
                                       nn.Linear(1024, 64),
                                       # n, 64
                                       nn.ReLU(),
                                       nn.Linear(64, 1)
                                       # n, 1
                                       )
        else:
            raise TypeError('Unsupported image size!')

    def forward(self, x):
        # convert rgb data to float tensor
        output = self.model(x.float())
        return output


class ProbFeatureNetRGB(nn.Module):
    Probabilistic = True
    rgb = True

    def __init__(self, height, width):
        super(ProbFeatureNetRGB, self).__init__()
        self.height = height
        self.width = width

        self.conv1 = nn.Conv2d(3, 32, 3, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, 2)
        self.conv3 = nn.Conv2d(64, 128, 3, 2)
        self.conv4 = nn.Conv2d(128, 256, 3, 2)
        self.conv5 = nn.Conv2d(256, 256, 3, 2)

        # TODO fix this
        if self.height == 240 and self.width == 320:
            self.fc1 = nn.Linear(2048, 64)
        elif self.height == 120 and self.width == 100:
            self.conv6 = nn.Conv2d(256, 256, 3, 2)
            self.fc1 = nn.Linear(1024, 64)
        else:
            raise TypeError('Unsupported image size!')

        self.fc2 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.conv1(x.float())
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv5(x)
        x = F.relu(x)
        if self.height == 240 and self.width == 320:
            x = self.conv6(x)
            x = F.relu(x)
        x = torch.flatten(x, start_dim=1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        mean = x[:, 0]
        std = F.sigmoid(x[:, 1])
        z = torch.stack([mean, std], 1)
        return z


class CalibrationNet(nn.Module):
    def __init__(self, perception_model):
        super(CalibrationNet, self).__init__()
        self.fc1 = nn.Linear(2, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 2)
        self.perception_model = perception_model

    def forward(self, x):
        with torch.no_grad():
            x = self.perception_model(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        mean = x[:, 0]
        std = F.sigmoid(x[:, 1])
        x = torch.stack([mean, std], 1)
        return x


class EnsembleModel:
    """ member models in this should not be probabilistic """

    def __init__(self, ensemble_model_path, width, height, rgb):
        perception_folder_names = os.listdir(ensemble_model_path)
        self.models = []
        for pfn in perception_folder_names:
            model = mu.construct_model(width, height, rgb, probabilistic=False)
            model.load_state_dict(torch.load(os.path.join(ensemble_model_path, pfn, 'feature_net.pt')))
            model.eval()
            self.models.append(model)

    def predict_feature(self, image):
        """ image can be (n, h, w) or (h, w), returns (n, 1) or a scalar respectively """
        results = [mu.predict_feature(m, image) for m in self.models]
        if image.ndim == 3:
            results = np.hstack(results)
            mean = np.mean(results, axis=-1)
            std = np.std(results, axis=-1)
        elif image.ndim == 2:
            mean = np.array(results).mean()
            std = np.array(results).std()
        else:
            raise TypeError
        return mean, std
