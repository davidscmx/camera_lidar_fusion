

from enum import Enum
import numpy as np
import open3d
import time
import cv2

import matplotlib
import matplotlib.cm
matplotlib.use("cairo")

from simple_waymo_open_dataset_reader import dataset_pb2
from simple_waymo_open_dataset_reader import utils as waymo_utils

class RangeImgChannel(Enum):
    Range = 0
    Intensity = 1
    Elongation = 2
    Is_in_no_label_zone = 3


class LidarVisualizer:
    def __init__(self, lidar):
        self.lidar = lidar
        self.lidar_frame_counter = 0

    def crop_channel_azimuth(self, img_channel, division_factor):
        opening_angle = int(img_channel.shape[1] / division_factor)
        img_channel_center = int(img_channel.shape[1] / 2)

        img_center_minus = img_channel_center - opening_angle
        img_center_plus = img_channel_center + opening_angle

        img_channel = img_channel[:, img_center_minus:img_center_plus]

        return img_channel

    def map_to_8bit(self, range_image, channel):
        img_channel = range_image[:, :, channel]

        if channel == RangeImgChannel.Range.value:
            diff = np.amax(img_channel) - np.amin(img_channel)
            img_channel = (img_channel / diff) * 255
            img_channel = img_channel.astype(np.uint8)
        elif channel == RangeImgChannel.Intensity.value:
            img_channel = self.contrast_adjustment(img_channel)
        return img_channel

    def get_img_selected_channel(self, channel, crop_azimuth=True):
        self.lidar.range_image[self.lidar.range_image < 0] = 0.0

        img_selected = self.map_to_8bit(self.lidar.range_image, channel=channel.value)
        if crop_azimuth:
            img_selected = self.crop_channel_azimuth(img_selected, 8)
        return img_selected

    def contrast_adjustment(self, img):
        """
            Heuristic approach to this lidar-specific problem: multiply the entire
            intensity image with half or the whole the value of the max.
            intensity value.
            #TODO understand how the normalization is done when converting to uint8
        """
        constant = np.amax(img) / 2
        img = (constant * img * 255) / (np.amax(img) - np.amin(img))
        return img

    def show_pcl(self, pcl, vis):
        # TODO improve visualization
        if not self.lidar_frame_counter:
            vis.create_window()

        pcd = open3d.geometry.PointCloud()

        # Remove intensity channel
        pcl = pcl[:, :-1]
        pcd.points = open3d.utility.Vector3dVector(pcl)
        open3d.visualization.draw_geometries([pcd])

        if not self.lidar_frame_counter:
            vis.add_geometry(pcd)
        else:
            vis.clear_geometries()
            vis.update_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()

        time.sleep(2)
        # TODO: Fix saving as dark image
        vis.capture_screen_image(f"./media/depth_image_{self.lidar_frame_counter}.png")

        vis.destroy_window()
        self.lidar_frame_counter += 1

    def display_laser_on_image(self, img, camera_calibration):

        pcl, pcl_attr = self.lidar.project_to_pointcloud()
        # get transformation matrix from vehicle frame to image
        vehicle_to_image = waymo_utils.get_image_transform(camera_calibration)

        # Convert the pointcloud to homogeneous coordinates.
        pcl1 = np.concatenate((pcl, np.ones_like(pcl[:, 0:1])), axis=1)

        # Transform the point cloud to image space.
        proj_pcl = np.einsum('ij,bj->bi', vehicle_to_image, pcl1)

        # Filter LIDAR points which are behind the camera.
        mask = proj_pcl[:, 2] > 0
        proj_pcl = proj_pcl[mask]
        proj_pcl_attr = pcl_attr[mask]

        # Project the point cloud onto the image.
        proj_pcl = proj_pcl[:, :2]/proj_pcl[:, 2:3]

        # Filter points which are outside the image.
        mask = np.logical_and(
            np.logical_and(proj_pcl[:, 0] > 0, proj_pcl[:, 0] < img.shape[1]),
            np.logical_and(proj_pcl[:, 1] > 0, proj_pcl[:, 1] < img.shape[1]))

        proj_pcl = proj_pcl[mask]
        proj_pcl_attr = proj_pcl_attr[mask]

        # Colour code the points based on distance.
        cmap = matplotlib.cm.get_cmap("viridis")
        coloured_intensity = 255 * cmap(proj_pcl_attr[:, 0]/30)

        # Draw a circle for each point.
        for i in range(proj_pcl.shape[0]):
            cv2.circle(img, (int(proj_pcl[i, 0]), int(proj_pcl[i, 1])), 1, coloured_intensity[i])

