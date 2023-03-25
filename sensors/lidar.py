
from simple_waymo_open_dataset_reader import utils as waymo_utils
from simple_waymo_open_dataset_reader import dataset_pb2
import numpy as np
from easydict import EasyDict as edict


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

        lidar = [obj for obj in frame.lasers if obj.name == self.lidar_name][0]


    def compute_beam_inclinations(self):
        """ Compute the inclination angle for each beam in a range image. """

        if len(self.calibration.beam_inclinations) > 0:
            return np.array(self.calibration.beam_inclinations)
        else:
            inclination_min = self.calibration.beam_inclination_min
            inclination_max = self.calibration.beam_inclination_max

        return np.linspace(inclination_min, inclination_max, self.range_image_height)

    #def range_image_to_3d_pcl(self):
    #    # Formerly pcl_from_range_image
    #    # transforms the Waymo Open Data range image into a 3D point-cloud
    #    pcl, pcl_attr = project_to_pointcloud(self.frame, self.range_image, 
    #                                          self.camera_projection, self.range_image_pose, 
    #                                          self.lidar_calib)
#
    #    # stack point cloud and lidar intensity
    #    points_all = np.column_stack((pcl, pcl_attr[:, 1]))
#
    #    return points_all
#
    #def project_to_pointcloud(self, frame, ri, camera_projection, range_image_pose, calibration):
    #    """ Create a pointcloud in vehicle space from LIDAR range image. """
    #    beam_inclinations = self.compute_beam_inclinations()
    #    
    #    # inclinations/pitches have to be reversed in order so that the
    #    # first angle corresponds to the top-most measurement.
    #    beam_inclinations = np.flip(beam_inclinations)
#
    #    # Compute the α angle betweetn the x and y axis from the extrinsic calibration matrix
    #    #         [cosαcosβ ... ]
    #    # [R,t] = [sinαcosβ ... ]
    #    #         [-sinβ ...    ]
    #    extrinsic = np.array(self.calibration.extrinsic.transform).reshape(4,4)
    #    frame_pose = np.array(self.frame.pose.transform).reshape(4,4)
#
    #    ri_polar = self.compute_range_image_polar(ri[:,:,0], extrinsic, beam_inclinations)
#
    #    if range_image_pose is None:
    #        pixel_pose = None
    #    else:
    #        pixel_pose = get_rotation_matrix(range_image_pose[:,:,0], range_image_pose[:,:,1], range_image_pose[:,:,2])
    #        translation = range_image_pose[:,:,3:]
    #        pixel_pose = np.block([
    #            [pixel_pose, translation[:,:,:,np.newaxis]],
    #            [np.zeros_like(translation)[:,:,np.newaxis],np.ones_like(translation[:,:,0])[:,:,np.newaxis,np.newaxis]]])
#
#
    #    ri_cartesian = compute_range_image_cartesian(ri_polar, extrinsic, pixel_pose, frame_pose)
    #    ri_cartesian = ri_cartesian.transpose(1,2,0)
#
    #    mask = ri[:,:,0] > 0
#
    #    return ri_cartesian[mask,:3], ri[mask]
#
#
    #def compute_range_image_polar(self, extrinsic, inclination):
    #    """ Convert a range image to polar coordinates. """
#
    #    height = self.range_image_height
    #    width = self.range_image_width
#
    #    az_correction = math.atan2(extrinsic[1,0], extrinsic[0,0])
#
    #    azimuth = np.linspace(np.pi, -np.pi, width) - az_correction
#
    #    # expand inclination and azimuth such that every range image cell has 
    #    # its own appropiate value pair
    #    azimuth_tiled = np.broadcast_to(azimuth[np.newaxis, :], (height, width))
    #    inclination_tiled = np.broadcast_to(inclination[:, np.newaxis], (height,width))
#
    #    return np.stack((azimuth_tiled, inclination_tiled, range_image))
#