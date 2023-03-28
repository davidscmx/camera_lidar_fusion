
from simple_waymo_open_dataset_reader import dataset_pb2

from enum import Enum
import numpy as np
import cv2
import matplotlib.pyplot as plt


class RangeImgChannel(Enum):
    Range = 0
    Intensity = 1
    Elongation = 2
    Is_in_no_label_zone = 3


class LidarVisualizer:
    def __init__(self, lidar):
        self.lidar = lidar

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
