
from simple_waymo_open_dataset_reader import utils as waymo_utils
from simple_waymo_open_dataset_reader import dataset_pb2
import numpy as np
from easydict import EasyDict as edict


class Lidar():
    def __init__(self):
        self.frame = None
        self.names = edict()
        self.names.top = dataset_pb2.LaserName.TOP

    def init_frame(self, frame, lidar_name):
        self.frame = frame
        self.lidar_name = lidar_name
        # extract lidar data and range image
        self.lidar = waymo_utils.get(self.frame.lasers, self.lidar_name)

        self.range_image, self.camera_projection, self.range_image_pose = \
            waymo_utils.parse_range_image_and_camera_projection(self.lidar)

        self.lidar_calib = waymo_utils.get(self.frame.context.laser_calibrations,
                                           self.names.top)
