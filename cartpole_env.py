import os, inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import logging
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import subprocess
import pybullet as p
from camera import RealSenseD415
import pybullet_utils as pu
import misc_utils as mu
from math import radians, sin, cos, pi
from perception_model import FeatureNetDepth, ProbFeatureNetDepth, FeatureNetRGB, FullStateNetDepth, FullStateNetRGB
import torch

logger = logging.getLogger(__name__)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CartPole:
    """ A controller of the cartpole """
    CART_LINK = 0
    LOW_POLE_LINK = 1
    HIGH_POLE_LINK = 2
    SLIDER_LINK = -1
    CART_POLE_JOINT = 1
    SLIDER_CART_JOINT = 0

    def __init__(self, urdf_path, initial_position, initial_orientation, fixation):
        self.urdf_path = urdf_path
        self.initial_position = initial_position
        self.initial_orientation = initial_orientation
        self.fixation = fixation
        self.fixation_urdf_path = mu.create_fixation_urdf(fixation=fixation, urdf_path=urdf_path)
        self.id = p.loadURDF(self.fixation_urdf_path, initial_position, initial_orientation, useFixedBase=True)
        p.changeDynamics(self.id, -1, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.id, 0, linearDamping=0, angularDamping=0)
        p.changeDynamics(self.id, 1, linearDamping=0, angularDamping=0)

        # TODO why is these two lines important
        p.setJointMotorControl2(self.id, 1, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.id, 0, p.VELOCITY_CONTROL, force=0)

    def reset_cart_pole_joint(self, position_value, velocity_value):
        p.resetJointState(self.id, self.CART_POLE_JOINT, position_value, velocity_value)

    def reset_slider_cart_joint(self, position_value, velocity_value):
        p.resetJointState(self.id, self.SLIDER_CART_JOINT, position_value, velocity_value)

    def apply_force_on_slider_cart_joint(self, force):
        p.setJointMotorControl2(self.id, self.SLIDER_CART_JOINT, p.TORQUE_CONTROL, force=force)

    def get_fixation_position(self):
        return self.get_cart_position() + sin(self.get_pole_position()) * self.fixation  # default length is 1

    def get_cart_position(self):
        # the link pose of cart does not match prismatic joint position 100% after some simulation
        return p.getJointState(self.id, self.SLIDER_CART_JOINT)[0]

    def get_cart_velocity(self):
        return p.getJointState(self.id, self.SLIDER_CART_JOINT)[1]

    def get_pole_position(self):
        # angular joint position
        return p.getJointState(self.id, self.CART_POLE_JOINT)[0]

    def get_pole_velocity(self):
        # angular joint velocity
        return p.getJointState(self.id, self.CART_POLE_JOINT)[1]

    def reset_state(self, state):
        """ state is a list of four values: cart position + cart velocity + joint position + joint velocity """
        self.reset_slider_cart_joint(state[0], state[1])
        self.reset_cart_pole_joint(state[2], state[3])

    def get_state(self):
        """ state is a list of four values: cart position + cart velocity + joint position + joint velocity """
        return [self.get_cart_position(), self.get_cart_velocity(), self.get_pole_position(), self.get_pole_velocity()]

    def make_invisible(self):
        p.changeVisualShape(self.id, self.HIGH_POLE_LINK, rgbaColor=(0, 0, 0, 0))

    def make_visible(self):
        p.changeVisualShape(self.id, self.HIGH_POLE_LINK, rgbaColor=(0, 0, 0, 1))


class CartPoleEnv(gym.Env):
    # TODO I think this is useless
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def __init__(self,
                 urdf_path,
                 initial_position,
                 initial_orientation,
                 fixation,
                 timestep,
                 angle_threshold_low,
                 angle_threshold_high,
                 distance_threshold_low,
                 distance_threshold_high,
                 force_mag,
                 camera_position,
                 camera_lookat,
                 camera_up_direction,
                 img_size,
                 ob_type,
                 action_dims,
                 append_actions_ob,
                 history_size=None,
                 std_dev=None,
                 pnoise=False,
                 model_path=None,
                 probabilistic=False,
                 rgb=False, 
                 delay=0):
        self.urdf_path = urdf_path
        self.initial_position = initial_position
        self.initial_orientation = initial_orientation
        self.fixation = fixation
        self.timestep = timestep
        self.angle_threshold_low = angle_threshold_low
        self.angle_threshold_high = angle_threshold_high
        self.distance_threshold_low = distance_threshold_low
        self.distance_threshold_high = distance_threshold_high
        self.force_mag = force_mag
        self.height, self.width = img_size
        self.height_after, self.width_after =  \
            mu.compute_image_size_after_cut_by_percentage(self.height, self.width, 0, 0.375)
        self.rgb = rgb
        self.camera_position = camera_position
        self.camera_lookat = camera_lookat
        self.camera_up_direction = camera_up_direction
        self.camera = RealSenseD415(self.camera_position, self.camera_lookat, self.camera_up_direction, (self.height, self.width))
        self.ob_type = ob_type
        self.action_dims = action_dims
        self.history_size = history_size
        self.append_actions_ob = append_actions_ob
        self.std_dev = std_dev
        self.pnoise = pnoise
        self.model_path = model_path
        self.steps_beyond_done = None
        self.probabilistic = probabilistic
        self.delay = delay
        self.action_history = None

        # initialize world by loading objects
        self.cartpole = CartPole(self.urdf_path, self.initial_position, self.initial_orientation, self.fixation)

        if self.pnoise:
            assert self.model_path is not None
            # load model
            if self.ob_type == 'full':
                if self.rgb:
                    self.feature_net = FullStateNetRGB(height=self.height_after, width=self.width_after, history_size=self.history_size).to(device)
                else:
                    self.feature_net = FullStateNetDepth(height=self.height_after, width=self.width_after, history_size=self.history_size).to(device)
            elif self.probabilistic:
                self.feature_net = ProbFeatureNetDepth(height=self.height_after, width=self.width_after).to(device)
            else:
                if self.rgb:
                    self.feature_net = FeatureNetRGB(height=self.height_after, width=self.width_after).to(device)
                else:
                    self.feature_net = FeatureNetDepth(height=self.height_after, width=self.width_after).to(device)
            self.feature_net.load_state_dict(torch.load(os.path.join(self.model_path, 'feature_net.pt'), map_location = device))
            self.feature_net.to(device)
            self.feature_net.eval()
            self.cartpole.make_invisible()

        if action_dims == 1:
            self.action_space = spaces.Box(low=-self.force_mag, high=self.force_mag, shape=(1,), dtype=np.float32)
        else:
            self.action_space = spaces.Discrete(action_dims)
        self.observation = None
        if self.ob_type == 'full':
            if self.pnoise:
                self.image_space = np.zeros((self.history_size, self.height_after, self.width_after, 3), dtype=np.float32) \
                    if self.rgb else np.zeros((self.history_size, self.height_after, self.width_after), dtype=np.float32)
            # cart position + cart velocity + joint position + joint velocity
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(4,))
        elif self.ob_type == 'z_sequence':
            assert self.history_size is not None
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.history_size * 2,)) if \
                self.probabilistic or self.append_actions_ob else spaces.Box(-np.inf, np.inf, shape=(self.history_size,))
        elif self.ob_type == 'pixel':
            assert self.history_size is not None
            self.observation_space = spaces.Box(-np.inf, np.inf,
                                                shape=(self.history_size, self.camera.height, self.camera.width, 3))
        else:
            raise TypeError
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        """ return observation, reward, done, info """
        # rgb_im, depth_im, seg_im, depth_pixels = self.camera.get_data()
        # force = mu.convert_action_to_force(action, self.action_dims)
        force = action  # action now is a continuous force
        self.cartpole.apply_force_on_slider_cart_joint(force)
        pu.step_multiple(self.timestep)
        # p.stepSimulation()
        if self.ob_type == 'full':
            if self.pnoise:
                rgb_im, depth_im, seg_im, depth_pixels = self.camera.get_data()
                image = rgb_im if self.rgb else depth_im
                image = mu.cut_image_by_percentage(image, 0, 0.375)
                self.image_space = np.append(self.image_space, image[None, ...], axis=0)
                self.image_space = self.image_space[1:, ...]
                self.observation = mu.predict_feature_fs(self.feature_net, self.image_space, self.rgb)
            else:
                self.observation = np.array(self.cartpole.get_state())
            # ground truth used to check done
            x, x_dot, theta, theta_dot = self.cartpole.get_state()
        elif self.ob_type == 'z_sequence':
            x, x_dot, theta, theta_dot = self.cartpole.get_state()
            if self.pnoise:
                rgb_im, depth_im, seg_im, depth_pixels = self.camera.get_data()
                image = rgb_im if self.rgb else depth_im
                image = mu.cut_image_by_percentage(image, 0, 0.375)
                if self.probabilistic:
                    z, std = mu.predict_feature_probabilistic(self.feature_net, image)
                else:
                    z = mu.predict_feature(self.feature_net, image, self.rgb)
                if self.std_dev is not None:
                    z = z + np.random.normal(loc=0, scale=self.std_dev)
                if self.probabilistic:
                    self.observation = np.append(self.observation, [z, std])
                    self.observation = self.observation[2:]
                else:
                    self.observation = np.append(self.observation, z)
                    self.observation = self.observation[1:]
            else:
                z = self.cartpole.get_fixation_position()
                if self.std_dev is not None:
                    z = z + np.random.normal(loc=0, scale=self.std_dev)
                self.observation = np.append(self.observation, z)
                self.observation = self.observation[1:]
        elif self.ob_type == 'pixel':
            x, x_dot, theta, theta_dot = self.cartpole.get_state()
            rgb_im, depth_im, seg_im, depth_pixels = self.camera.get_data()
            self.observation = np.vstack((self.observation, rgb_im[None, ...]))
            self.observation = self.observation[1:]
        else:
            raise TypeError("type must be one of full, z_sequence or pixel")

        # done indicates a failure
        done = x < self.distance_threshold_low \
               or x > self.distance_threshold_high \
               or theta < radians(self.angle_threshold_low) \
               or theta > radians(self.angle_threshold_high)
        done = bool(done)

        # handling step when the cartpole already fails
        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            # Pole just fell!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warning(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        self.buffer = self.buffer[1:]+[self.observation]
        if self.append_actions_ob:
            return np.concatenate((self.buffer[0], self.action_history)), reward, done, {}
        else:
            return self.buffer[0], reward, done, {}

    def reset(self, randstate=None):
        """ return an initial observation """
        self.steps_beyond_done = None
        randstate = self.np_random.uniform(low=-0.05, high=0.05, size=(4,)) if randstate is None else randstate
        self.cartpole.reset_state(randstate)
        self.action_history = np.zeros(self.history_size)
        if self.ob_type == 'full':
            if self.pnoise:
                self.image_space = np.zeros((self.history_size, self.height_after, self.width_after, 3), dtype=np.float32) \
                    if self.rgb else np.zeros((self.history_size, self.height_after, self.width_after), dtype=np.float32)
                rgb_im, depth_im, seg_im, depth_pixels = self.camera.get_data()
                image = rgb_im if self.rgb else depth_im
                image = mu.cut_image_by_percentage(image, 0, 0.375)
                self.image_space = np.append(self.image_space, image[None, ...], axis=0)
                self.image_space = self.image_space[1:, ...]
                x, x_dot, theta, theta_dot = mu.predict_feature_fs(self.feature_net, self.image_space, self.rgb)
                self.observation = np.array([x, x_dot, theta, theta_dot])
            else:
                self.observation = np.array(self.cartpole.get_state())
        elif self.ob_type == 'z_sequence':
            if self.pnoise:
                rgb_im, depth_im, seg_im, depth_pixels = self.camera.get_data()
                image = rgb_im if self.rgb else depth_im
                image = mu.cut_image_by_percentage(image, 0, 0.375)
                if self.probabilistic:
                    initial_z, initial_std = mu.predict_feature_probabilistic(self.feature_net, image)
                else:
                    initial_z = mu.predict_feature(self.feature_net, image, self.rgb)
                if self.std_dev is not None:
                    initial_z = initial_z + np.random.normal(loc=0, scale=self.std_dev)
                if self.probabilistic:
                    self.observation = np.array([initial_z, initial_std] * self.history_size)
                else:
                    self.observation = np.array([initial_z] * self.history_size)
            else:
                initial_z = self.cartpole.get_fixation_position()
                if self.std_dev is not None:
                    initial_z = initial_z + np.random.normal(loc=0, scale=self.std_dev)
                self.observation = np.array([initial_z] * self.history_size)
        elif self.ob_type == 'pixel':
            rgb_im, depth_im, seg_im, depth_pixels = self.camera.get_data()
            self.observation = np.tile(rgb_im, (self.history_size, 1, 1, 1))
        else:
            raise TypeError
        # print("-----------reset simulation---------------")
        # print(randstate)
        self.buffer = [self.observation]*(self.delay+1)

        if self.append_actions_ob:
            return np.concatenate((self.buffer[0], self.action_history))
        else:
            return self.buffer[0]
   
    def configure(self, args):
        pass


if __name__ == "__main__":
    mu.configure_pybullet(rendering=True, debug=True)
    env = mu.make_cart_pole_env(
        fixation=0.7,
        ob_type="z_sequence",
        pnoise=True,
        model_path='assets/models/perception/depth/high/fixation_100/epoch_0959_loss_0.0000003_dist_0.0004324/')

    for e in range(1):
        env.reset()
        print("initial: {}".format(env.cartpole.get_state()))
        for i in range(500):
            action = np.random.uniform(-10, 10)
            observation, reward, done, info = env.step(action)
            time.sleep(0.02)
            print(action, env.cartpole.get_state(), env.cartpole.get_fixation_position())
            if done:
                break

    print('end')
