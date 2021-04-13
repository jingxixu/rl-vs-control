#!/usr/bin/env python

import numpy as np
import threading
import time
import pybullet as p


class RealSenseD415:
    # Mimic RealSense D415 parameters
    z_near = 0.01
    z_far = 10.0
    fov_w = 69.40

    def __init__(self, position, lookat, up_direction, size=(120, 160)):
        # original image size = (240, 320)
        self.height = size[0]
        self.width = size[1]

        self.focal_length = (float(self.width) / 2) / np.tan((np.pi * self.fov_w / 180) / 2)
        self.fov_h = (np.arctan((float(self.height) / 2) / self.focal_length) * 2 / np.pi) * 180
        self.projection_matrix = p.computeProjectionMatrixFOV(fov=self.fov_h,
                                                              aspect=float(self.width) / float(self.height),
                                                              nearVal=self.z_near,
                                                              farVal=self.z_far)  # notes: 1) FOV is vertical FOV 2) aspect must be float

        self.intrinsics = np.array([[self.focal_length, 0, float(self.width) / 2],
                                    [0, self.focal_length, float(self.height) / 2],
                                    [0, 0, 1]])

        self.max_fps = 10  # higher numbers will reduce PyBullet framerate
        self.position = position
        self.lookat = lookat
        self.up_direction = up_direction

        self.view_matrix = p.computeViewMatrix(self.position, self.lookat, self.up_direction)
        pose_matrix = np.linalg.inv(np.array(self.view_matrix).reshape(4, 4).T)
        pose_matrix[:, 1:3] = -pose_matrix[:, 1:3]  # TODO: fix flipped up and forward vectors (quick hack)
        self.pose_matrix = pose_matrix

        # Start thread to stream RGB-D images
        # self.color_im = None
        # self.depth_im = None
        # self.segm_mask = None
        # self.get_data()
        # stream_thread = threading.Thread(target=self.stream)
        # stream_thread.daemon = True
        # stream_thread.start()

        # def stream(self):
        #     while True:
        #         self.get_data()
        #         time.sleep(1. / self.max_fps)

        # Get latest RGB-D image from camera

    def get_data(self, noise=False):
        """

        :return rgb_im: np.array, (height, width, 3), uint8
        :return depth_im: np.array, (height, width), float32, real distance in the range of [z_near, z_far]
        :return seg_im: np.array, (height, width), int32, object id
        :return depth_pixels: np.array, (height, width), float32, processed distance in the range of [0, 1]
        """
        width, height, rgba_pixels, depth_pixels, seg_im = p.getCameraImage(self.width,
                                                                            self.height,
                                                                            self.view_matrix,
                                                                            self.projection_matrix,
                                                                            shadow=1,
                                                                            renderer=p.ER_TINY_RENDERER)

        rgb_im = rgba_pixels[:, :, :3]  # remove alpha channel
        z_buffer = depth_pixels
        depth_im = (2.0 * self.z_near * self.z_far) / (
                self.z_far + self.z_near - (2.0 * z_buffer - 1.0) * (self.z_far - self.z_near))
        if noise:
            depth_im += np.random.normal(loc=0, scale=0.005, size=(self.image_size[0], self.image_size[1]))
        depth_im = depth_im
        return rgb_im, depth_im, seg_im, depth_pixels

    @staticmethod
    def mask_objects_in_depth(object_ids, depth_im, seg_im):
        depth_mask = np.full_like(depth_im, False)
        for i in object_ids:
            depth_mask = np.logical_or(depth_mask, seg_im == i)
        new_depth_im = np.where(depth_mask, depth_im, np.zeros_like(depth_im, dtype=np.float32))
        return new_depth_im

    @staticmethod
    def mask_objects_in_rgb(object_ids, rgb_im, seg_im):
        rgb_mask = np.full_like(rgb_im, False)
        for i in object_ids:
            rgb_mask = np.logical_or(rgb_mask, np.tile(seg_im[..., None], (1, 1, 3)) == i)
        new_rgb_im = np.where(rgb_mask, rgb_im, np.zeros_like(rgb_im, dtype=np.uint8))
        return new_rgb_im

    def get_rgbd(self, masked=False, object_ids=None):
        rgb_im, depth_im, seg_im, depth_pixels = self.get_data()
        if masked:
            # mask out everything except target
            assert object_ids is not None, "please provide a list of object ids"
            depth_im = RealSenseD415.mask_objects_in_depth(object_ids, depth_im, seg_im)
            rgb_im = RealSenseD415.mask_objects_in_rgb(object_ids, rgb_im, seg_im)
        rgbd_im = np.concatenate((rgb_im, depth_im[..., None]), axis=2)
        return rgbd_im
