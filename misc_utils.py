import pprint
from collections import OrderedDict
import os
import csv
import pybullet_data
import pybullet as p
import pybullet_utils as pu
import pandas as pd
from math import radians, cos, sin
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
import pybullet_utils as pu
import yaml
import json

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def configure_pybullet(rendering=False,
                       debug=False,
                       yaw=50.0,
                       pitch=-35.0,
                       dist=1.2,
                       target=(0.0, 0.0, 0.0),
                       gravity=9.8):
    if not rendering:
        p.connect(p.DIRECT)
    else:
        p.connect(p.GUI_SERVER)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    if not debug:
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    pu.reset_camera(yaw=yaw, pitch=pitch, dist=dist, target=target)
    p.setPhysicsEngineParameter(enableFileCaching=0)
    p.resetSimulation()
    p.setGravity(0, 0, -gravity)


def write_csv_line(result_file_path, result):
    """ write a line in a csv file; create the file and write the first line if the file does not already exist """
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(result)
    result = OrderedDict(result)
    file_exists = os.path.exists(result_file_path)
    with open(result_file_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, result.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)


def sample_position(lower_limits=(0, 0, 0), upper_limits=(1, 1, 1)):
    """
    :param lower_limits: lower limits of (x, y, z)
    :param upper_limits: upper limits of (x, y, z)
    :return: a position between lower_limits and upper_limits
    """
    return np.random.uniform(low=lower_limits, high=upper_limits)


def show_rgb(img, title='rgb', ticks_off=False):
    """
    :param img: np.array, (height, width, 3), uint8
    """
    # RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # cv2.imshow('rgb image', RGB_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # # plt.imshow(img)
    # return RGB_img
    plt.title(title)
    plt.imshow(img)
    plt.tight_layout()
    if ticks_off:
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        plt.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelleft=False)  # labels along the bottom edge are off
    plt.pause(0.0001)


def show_depth(img, title='depth', ticks_off=False):
    """
    :param img: np.array, (height, width), float32, processed distance in the range of [0, 1]
    """
    # plt.figure()  # if you want a separate figure window
    plt.title(title)
    plt.imshow(img, cmap='gray')
    plt.tight_layout()
    if ticks_off:
        plt.tick_params(
            axis='x',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            bottom=False,  # ticks along the bottom edge are off
            top=False,  # ticks along the top edge are off
            labelbottom=False)  # labels along the bottom edge are off
        plt.tick_params(
            axis='y',  # changes apply to the x-axis
            which='both',  # both major and minor ticks are affected
            left=False,  # ticks along the bottom edge are off
            right=False,  # ticks along the top edge are off
            labelleft=False)  # labels along the bottom edge are off
    plt.pause(0.0001)


def show_rgbs(imgs, title='rgb'):
    """
    :param imgs: np.array, (n, 3, height, width), uint8
    """
    for rgb in imgs:
        rgb = np.transpose(rgb, (1, 2, 0))
        show_rgb(rgb)


def show_depths(imgs, title='depth'):
    """
    :param imgs: np.array, (n, 1, height, width), float32, processed distance in the range of [0, 1]
    """
    # plt.figure()  # if you want a separate figure window
    for depth in imgs:
        depth = np.squeeze(depth)
        show_depth(depth)


def show_segmentation(img):
    """
    :param img: np.array, (height, width), int32, each pixel corresponds to an object id
    """
    # plt.figure()  # if you want a separate figure window
    plt.imshow(img)
    plt.tight_layout()
    plt.pause(0.0001)


def draw_workspace(lower_limits, upper_limits, rgb_color=(0, 1, 0)):
    markers = []
    lines = [((lower_limits[0], lower_limits[1], lower_limits[2]), (lower_limits[0], upper_limits[1], lower_limits[2])),
             ((lower_limits[0], lower_limits[1], lower_limits[2]), (upper_limits[0], lower_limits[1], lower_limits[2])),
             ((lower_limits[0], upper_limits[1], lower_limits[2]), (upper_limits[0], upper_limits[1], lower_limits[2])),
             ((upper_limits[0], lower_limits[1], lower_limits[2]), (upper_limits[0], upper_limits[1], lower_limits[2])),

             ((lower_limits[0], lower_limits[1], upper_limits[2]), (lower_limits[0], upper_limits[1], upper_limits[2])),
             ((lower_limits[0], lower_limits[1], upper_limits[2]), (upper_limits[0], lower_limits[1], upper_limits[2])),
             ((lower_limits[0], upper_limits[1], upper_limits[2]), (upper_limits[0], upper_limits[1], upper_limits[2])),
             ((upper_limits[0], lower_limits[1], upper_limits[2]), (upper_limits[0], upper_limits[1], upper_limits[2])),

             ((lower_limits[0], lower_limits[1], lower_limits[2]), (lower_limits[0], lower_limits[1], upper_limits[2])),
             ((lower_limits[0], upper_limits[1], lower_limits[2]), (lower_limits[0], upper_limits[1], upper_limits[2])),
             ((upper_limits[0], lower_limits[1], lower_limits[2]), (upper_limits[0], lower_limits[1], upper_limits[2])),
             ((upper_limits[0], upper_limits[1], lower_limits[2]), (upper_limits[0], upper_limits[1], upper_limits[2]))]

    for start_pos, end_pos in lines:
        markers.append(pu.draw_line(start_pos, end_pos, rgb_color))
    return markers


def save_yaml(data, path):
    with open(path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)


def save_json(data, path):
    # save checkpoint
    json.dump(data, open(path, 'w'), indent=4)


def scale(data, low, high):
    """ scale numpy array to [low, high]"""
    low = float(low)
    high = float(high)
    data = data.astype(np.float32)

    max_value = np.max(data)
    min_value = np.min(data)
    range_value = max_value - min_value
    data = low + (high - low) * (data - min_value) / range_value
    return data


def create_fixation_urdf(fixation,
                         length=1.0,
                         mass=0.1,
                         urdf_path='assets/cartpole_template.urdf',
                         new_urdf_path='assets/cartpole_fixation.urdf'):
    """
    create new urdf based on the template urdf
    :param fixation: the ratio (0 - 1) of the height of the fixation point from bottom to top
    """
    # set_up urdf
    low_length = fixation * length
    high_length = (1 - fixation) * length
    low_z = low_length / 2.0
    high_z = low_length + high_length / 2.0
    low_mass = mass * fixation
    high_mass = mass * (1 - fixation)

    os.system('cp {} {}'.format(urdf_path, new_urdf_path))
    sed_cmd = "sed -i 's|{}|{:.3f}|g' {}".format('low_length', low_length, new_urdf_path)
    os.system(sed_cmd)
    sed_cmd = "sed -i 's|{}|{:.3f}|g' {}".format('low_z', low_z, new_urdf_path)
    os.system(sed_cmd)
    sed_cmd = "sed -i 's|{}|{:.3f}|g' {}".format('low_mass', low_mass, new_urdf_path)
    os.system(sed_cmd)
    sed_cmd = "sed -i 's|{}|{:.3f}|g' {}".format('high_length', high_length, new_urdf_path)
    os.system(sed_cmd)
    sed_cmd = "sed -i 's|{}|{:.3f}|g' {}".format('high_z', high_z, new_urdf_path)
    os.system(sed_cmd)
    sed_cmd = "sed -i 's|{}|{:.3f}|g' {}".format('high_mass', high_mass, new_urdf_path)
    os.system(sed_cmd)
    return new_urdf_path


def make_cart_pole_env(fixation,
                       ob_type,
                       action_dims=1,  # default continuous
                       history_size=None,
                       std_dev=None,
                       pnoise=False,
                       model_path=None,
                       probabilistic=False,
                       img_size=(120, 160),
                       rgb=False,
                       delay=0,
                       append_actions_ob=False,
                       timestep=5):
    """ Wrapper function for making the env with some default values for all scripts """
    from cartpole_env import CartPoleEnv
    if history_size is None:
        if ob_type == "z_sequence":
            history_size = 200
        elif ob_type == "full" and pnoise:
            history_size = 5
    env = CartPoleEnv(timestep=timestep,
                      initial_position=[0, 0, 0],
                      initial_orientation=[0, 0, 0, 1],
                      fixation=fixation,
                      # using a template for fixation
                      urdf_path=os.path.join('assets', 'cartpole_template.urdf'),
                      angle_threshold_low=-15,
                      angle_threshold_high=15,
                      distance_threshold_low=-0.6,
                      distance_threshold_high=0.6,
                      force_mag=10,
                      camera_position=(-2.0, 0, 0.4),
                      camera_lookat=(0.5, 0, 0.7),
                      camera_up_direction=(0, 0, 1),
                      img_size=img_size,
                      ob_type=ob_type,
                      action_dims=action_dims,
                      history_size=history_size,
                      pnoise=pnoise,
                      model_path=model_path,
                      std_dev=std_dev,
                      probabilistic=probabilistic,
                      rgb=rgb,
                      delay=delay,
                      append_actions_ob=append_actions_ob)
    return env


def make_cart_pole_env_true_dyanmics(fixation,
                                     ob_type,
                                     pnoise=False,
                                     model_path=None,
                                     probabilistic=False,
                                     image_size=(120, 160),
                                     history_size=200,
                                     delay=0):
    """ Wrapper function for making the env with some default values for all scripts """

    from cartpole_env_true_dynamics import CartPoleEnv
    from math import radians
    env = CartPoleEnv(
        urdf_path="assets/cartpole_template.urdf",
        initial_cart_pos=0,
        initial_pole_pos=0,
        fixation=fixation,
        angle_threshold_low=radians(-15),
        angle_threshold_high=radians(15),
        distance_threshold_low=-0.6,
        distance_threshold_high=0.6,
        force_mag=10,
        camera_position=(-2.0, 0, 0.4),
        camera_lookat=(0.5, 0, 0.7),
        camera_up_direction=(0, 0, 1),
        image_size=image_size,
        ob_type=ob_type,
        history_size=history_size,
        pnoise=pnoise,
        model_path=model_path,
        std_dev=0,
        probabilistic=probabilistic,
        delay=delay
    )
    return env


def convert_action_to_force(action, action_dims):
    if action_dims == 1:
        # continuous
        return action
    elif action_dims == 2:
        action_map = {
            0: -10,
            1: 10
        }
    elif action_dims == 3:
        action_map = {
            0: -10,
            1: 0,
            2: 10
        }
    elif action_dims == 7:
        action_map = {
            0: -10,
            1: -5,
            2: -1,
            3: 0,
            4: 1,
            5: 5,
            6: 10
        }
    elif action_dims == 6:
        action_map = {
            0: -10,
            1: -5,
            2: -1,
            3: 1,
            4: 5,
            5: 10
        }
    else:
        raise TypeError('action dim {} is not supported!'.format(action_dims))
    return action_map[action]


def compute_image_size_after_cut_by_percentage(height, width, h_p, w_p):
    height_after = height - 2 * int(height * h_p / 2)
    width_after = width - 2 * int(width * w_p / 2)
    return height_after, width_after


def cut_image_by_percentage(img_arr, h_p, w_p):
    h = img_arr.shape[0]
    w = img_arr.shape[1]
    h_pixs = int(h * h_p / 2)
    w_pixs = int(w * w_p / 2)
    return cut_image(img_arr, h_pixs, w_pixs)


def cut_image(img_arr, h_pixs, w_pixs):
    """ cut the image from left and right by num_pixs pixels """
    h = img_arr.shape[0]
    w = img_arr.shape[1]
    img_arr = img_arr[h_pixs:h - h_pixs, w_pixs:w - w_pixs, ...]
    return img_arr


def predict_feature_fs(feature_net, images, rgb=False):
    """
    for depth:
    images are of shape (history_size, h, w) or (n, history_size, h, w)
    for rgb:
    images are of shape (history_size, h, w, 3) or (n, history_size, h, w, 3)
    """
    with torch.no_grad():
        if rgb:
            if images.ndim == 5:
                # not implemented
                pass
            elif images.ndim == 4:
                images = np.transpose(images, [0, 3, 1, 2])
                input_x = torch.tensor(images[None, ...])
                input_x = input_x.to(device)
                output = feature_net(input_x)
                output = output.cpu().numpy().squeeze()
            else:
                raise TypeError('unsupported depth image shape')
        else:
            if images.ndim == 4:
                raise NotImplementedError
            elif images.ndim == 3:  # (history_size, h, w)
                input_x = torch.tensor(images[None, ...])
                input_x = input_x.to(device)
                output = feature_net(input_x)
                output = output.cpu().numpy().squeeze()
            else:
                raise TypeError('unsupported image shape')

    return output


def predict_feature(feature_net, image, rgb=False):
    """ depth_im can be np array of size (n, h, w) or (h, w), returns (n, 1) or a scalar respectively """

    if rgb:
        image = np.transpose(image, [2, 0, 1])
        if image.ndim == 4:
            # not implemented
            pass
        elif image.ndim == 3:
            input_x = torch.tensor(image[None, :, :, :])
            input_x = input_x.to(device)
            z = feature_net(input_x)
            z = z.item()
        else:
            raise TypeError('unsupported depth image shape')
    else:
        if image.ndim == 3:
            input_x = torch.tensor(image[:, None, :, :])
            input_x = input_x.to(device)
            z = feature_net(input_x)
            z = z.detach().cpu().numpy()

        elif image.ndim == 2:
            input_x = torch.tensor(image[None, None, :, :])
            input_x = input_x.to(device)
            z = feature_net(input_x)
            z = z.item()
        else:
            raise TypeError('unsupported depth image shape')

    return z


def predict_feature_probabilistic(feature_net, depth_im):
    """ depth_im can be (n, h, w) or (h, w), returns a (n, 2) numpy array or z, std respectively """

    if depth_im.ndim == 3:
        input_x = torch.tensor(depth_im[:, None, :, :])
        input_x = input_x.to(device)
        output = feature_net(input_x)
        return output.detach().cpu().numpy()

    elif depth_im.ndim == 2:
        input_x = torch.tensor(depth_im[None, None, :, :])
        input_x = input_x.to(device)
        output = feature_net(input_x)
        z, std = output.detach().cpu().numpy().squeeze()
        return z, std

    else:
        raise TypeError('unsupported depth image shape')


def convert_second(seconds):
    day = seconds // (24 * 3600)
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return "%02dd%02dh%02dm%02ds" % (day, hour, minutes, seconds)


def convert_float_to_fixstr(f):
    return "{:.2f}".format(f).replace('.', '')


def get_perception_model_path(ob_type, f, rgb, resolution):
    model_path = os.path.join('assets', 'models', 'perception')
    if ob_type == "full":
        model_path = os.path.join(model_path, 'fs', 'rgb') if rgb else os.path.join(model_path, 'fs', 'depth')
    else:
        model_path = os.path.join(model_path, 'z', 'rgb') if rgb else os.path.join(model_path, 'z', 'depth')
        model_path = os.path.join(model_path, resolution)

    fixstr = convert_float_to_fixstr(f)
    model_path = os.path.join(model_path, 'fixation_' + fixstr)
    folder_nm = [n for n in os.listdir(model_path)
                 if os.path.isdir(os.path.join(model_path, n))][0]
    return os.path.join(model_path, folder_nm)


def construct_model(width, height, rgb, probabilistic, fs):
    from perception_model import PerceptionDataset, FeatureNetDepth, ProbFeatureNetDepth, FeatureNetRGB, \
        ProbFeatureNetRGB, FullStateNetDepth, FullStateNetRGB
    if fs:
        if rgb:
            model = FullStateNetRGB(width=width, height=height, history_size=5).to(device)
        else:
            model = FullStateNetDepth(width=width, height=height, history_size=5).to(device)
        return model
    else:
        if rgb and probabilistic:
            model = ProbFeatureNetRGB(height, width).to(device)
        elif rgb and not probabilistic:
            model = FeatureNetRGB(height, width).to(device)
        elif not rgb and probabilistic:
            model = ProbFeatureNetDepth(width, height).to(device)
        else:
            model = FeatureNetDepth(height, width).to(device)
        return model


def map_resolution(resolution):
    if resolution == 'high':
        return 120, 160, 120, 100
    if resolution == 'medium':
        return 80, 107, 80, 67
    if resolution == 'low':
        return 60, 80, 60, 50


def calculate_z(cart_p, joint_p, fixation):
    z = cart_p + sin(joint_p) * fixation  # default length is 1
    return z
