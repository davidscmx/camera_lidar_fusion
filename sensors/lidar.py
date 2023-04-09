
from simple_waymo_open_dataset_reader import utils as waymo_utils
from simple_waymo_open_dataset_reader import dataset_pb2

from enum import Enum
import numpy as np
from easydict import EasyDict as edict
import math


class Lidar:
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

        self.calibration = waymo_utils.get(self.frame.context.laser_calibrations,
                                           self.names.top)
        self.range_image_height = self.range_image[0]
        self.range_image_width = self.range_image[1]

    def get_pcl_range_image(self) -> np.ndarray:
        """
            Returns the point cloud from the range image in cartesian coordinates,
            x, y, z, and with the intensity as the last column. Formerly called
            pcl_from_range_image
        """
        # transforms the Waymo Open Data range image into a 3D point-cloud
        pcl, pcl_attr = self.project_to_pointcloud()
        # stack point cloud and lidar intensity
        points_all = np.column_stack((pcl, pcl_attr[:, 1]))
        return points_all

    def get_pixel_pose(self):
        if self.range_image_pose is None:
            pixel_pose = None
        else:
            pixel_pose = waymo_utils.get_rotation_matrix(self.range_image_pose[:, :, 0],
                                                         self.range_image_pose[:, :, 1],
                                                         self.range_image_pose[:, :, 2])
            translation = self.range_image_pose[:, :, 3:]

            pixel_pose = np.block([
                    [pixel_pose, translation[:, :, :, np.newaxis]],
                    [np.zeros_like(translation)[:, :, np.newaxis],
                     np.ones_like(translation[:, :, 0])[:, :, np.newaxis, np.newaxis]]])

        return pixel_pose

    def project_to_pointcloud(self):
        """ Create a pointcloud in vehicle space from LIDAR range image. """
        # inclinations/pitches have to be reversed in order so that the
        # first angle corresponds to the top-most measurement.
        beam_inclinations = waymo_utils.compute_beam_inclinations(self.calibration,
                                                                  self.range_image_height)
        beam_inclinations = np.flip(beam_inclinations)

        # Compute the α angle betweetn the x and y axis from the extrinsic calibration matrix
        #         [cosαcosβ ... ]
        # [R,t] = [sinαcosβ ... ]
        #         [-sinβ ...    ]
        extrinsic = np.array(self.calibration.extrinsic.transform).reshape(4, 4)
        frame_pose = np.array(self.frame.pose.transform).reshape(4, 4)

        ri_polar = waymo_utils.compute_range_image_polar(self.range_image[:, :, 0],
                                                         extrinsic, beam_inclinations)

        pixel_pose = self.get_pixel_pose()

        ri_cartesian = waymo_utils.compute_range_image_cartesian(ri_polar, extrinsic,
                                                                 pixel_pose, frame_pose)
        ri_cartesian = ri_cartesian.transpose(1, 2, 0)

        mask = self.range_image[:, :, 0] > 0

        return ri_cartesian[mask, :3], self.range_image[mask]
