

import numpy as np
import cv2

from simple_waymo_open_dataset_reader import label_pb2
from utilities.config import bev_config, lidar_config, obj_config


class ObjectPixel:
    def __init__(self):
        id: int
        x: float
        y: float
        z: float
        w: float
        l: float
        yaw: float


class Bev:
    def __init__(self):
        pass

    def convert_from_metric_into_px_coords(self, object_3d):
        ob_3d = object_3d
        obj_px = ObjectPixel()
        obj_px.id = ob_3d.type

        obj_px.x = (ob_3d.center_y - lidar_config.lim_y[0]) / \
                   (lidar_config.diff_y) * bev_config.width

        obj_px.y = (ob_3d.center_x - lidar_config.lim_x[0]) / \
                   (lidar_config.diff_x) * bev_config.height

        obj_px.z = ob_3d.center_z - lidar_config.lim_z[0]
        obj_px.w = ob_3d.width / (lidar_config.diff_y) * bev_config.width
        obj_px.l = ob_3d.length / (lidar_config.diff_x) * bev_config.height
        obj_px.yaw = -ob_3d.heading

        return obj_px

    def get_object_corners_within_bev_image(self, obj_px):
        bev_corners = np.zeros((4, 2), dtype=np.float32)

        cos_yaw = np.cos(obj_px.yaw)
        sin_yaw = np.sin(obj_px.yaw)

        bev_corners[0, 0] = obj_px.x - obj_px.w / 2 * cos_yaw - obj_px.l / 2 * sin_yaw  # front left
        bev_corners[0, 1] = obj_px.y - obj_px.w / 2 * sin_yaw + obj_px.l / 2 * cos_yaw
        bev_corners[1, 0] = obj_px.x - obj_px.w / 2 * cos_yaw + obj_px.l / 2 * sin_yaw  # rear left
        bev_corners[1, 1] = obj_px.y - obj_px.w / 2 * sin_yaw - obj_px.l / 2 * cos_yaw
        bev_corners[2, 0] = obj_px.x + obj_px.w / 2 * cos_yaw + obj_px.l / 2 * sin_yaw  # rear right
        bev_corners[2, 1] = obj_px.y + obj_px.w / 2 * sin_yaw - obj_px.l / 2 * cos_yaw
        bev_corners[3, 0] = obj_px.x + obj_px.w / 2 * cos_yaw - obj_px.l / 2 * sin_yaw  # front right
        bev_corners[3, 1] = obj_px.y + obj_px.w / 2 * sin_yaw + obj_px.l / 2 * cos_yaw

        return bev_corners

    def project_detections_into_bev(self, bev_map, objects_3d, color=[]):

        for object_3d in objects_3d:
            # extract detection
            if object_3d.type != label_pb2.Label.TYPE_VEHICLE:
                continue
            obj_px = self.convert_from_metric_into_px_coords(object_3d)

            # draw object bounding box into birds-eye view
            if not color:
                color = obj_config.colors[int(obj_px.id)]

            bev_corners = self.get_object_corners_within_bev_image(obj_px)

            # draw object as box
            corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
            cv2.polylines(bev_map, [corners_int], True, color, 2)

            # draw colored line to identify object front
            corners_int = corners_int.reshape(-1, 2)

            cv2.line(bev_map, (corners_int[0, 0], corners_int[0, 1]),
                     (corners_int[3, 0], corners_int[3, 1]),
                     (255, 255, 0), 2)
