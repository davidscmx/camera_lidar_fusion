from shapely.geometry import Polygon
from utilities.config import lidar_config
import numpy as np


class Object3d:
    def __init__(self, label):
        self.type = label.type
        self.center_x = label.box.center_x
        self.center_y = label.box.center_y
        self.center_z = label.box.center_z
        self.height = label.box.height
        self.width = label.box.width
        self.length = label.box.length
        self.heading = label.box.heading


class LabelsLoader():
    def __init__(self, labels, labels_config=None):
        self.labels_config = labels_config
        # convert_labels_into_objects
        self.labels = labels
        self.objects_3d: list = [Object3d(label) for label in labels]
        self.objects_3d_valid: list = [Object3d(label) for label in labels
                                       if self.is_label_inside_detection_area(Object3d(label))]

    def is_label_inside_detection_area(self, object_3d, min_overlap=0.5):
        # convert current label object into Polygon object

        center_x = object_3d.center_x
        center_y = object_3d.center_y
        width = object_3d.width
        length = object_3d.length
        yaw = object_3d.heading

        label_obj_corners = self.compute_box_corners(center_x, center_y, width, length, yaw)
        label_obj_poly = Polygon(label_obj_corners)

        # convert detection are into polygon
        da_w = lidar_config.diff_x  # width
        da_l = lidar_config.diff_y  # length
        da_x = lidar_config.lim_x[0] + da_w/2  # center in x
        da_y = lidar_config.lim_y[0] + da_l/2  # center in y

        da_corners = self.compute_box_corners(da_x, da_y, da_w, da_l, 0)
        da_poly = Polygon(da_corners)

        # check if detection area contains label object
        intersection = da_poly.intersection(label_obj_poly)
        overlap = intersection.area / label_obj_poly.area

        return False if (overlap <= min_overlap) else True

    # compute location of each corner of a box and returns [front_left, rear_left, rear_right, front_right]
    def compute_box_corners(self, x, y, w, l, yaw):
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)

        fl = (x - w / 2 * cos_yaw - l / 2 * sin_yaw,  # front left
              y - w / 2 * sin_yaw + l / 2 * cos_yaw)

        rl = (x - w / 2 * cos_yaw + l / 2 * sin_yaw,  # rear left
              y - w / 2 * sin_yaw - l / 2 * cos_yaw)

        rr = (x + w / 2 * cos_yaw + l / 2 * sin_yaw,  # rear right
              y + w / 2 * sin_yaw - l / 2 * cos_yaw)

        fr = (x + w / 2 * cos_yaw - l / 2 * sin_yaw,  # front right
              y + w / 2 * sin_yaw + l / 2 * cos_yaw)

        return [fl, rl, rr, fr]
