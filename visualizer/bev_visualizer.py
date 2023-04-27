
from simple_waymo_open_dataset_reader import label_pb2
from simple_waymo_open_dataset_reader import utils as waymo_utils
from data_loaders.labels_loader import LabelsLoader
from sensors.bev import Bev
from utilities.config import bev_config
import numpy as np
import cv2


class BevVisualizer:
    def __init__(self, bev_maps, labels):
        self.bev = Bev()
        self.bev_map = self.process_bev(bev_maps)
        # Google protobuf object
        self.labels_loader = LabelsLoader(labels)

    def process_bev(self, bev_maps):
        bev_map = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        bev_map = cv2.resize(bev_map, (bev_config.width, bev_config.height))
        return bev_map

    def show_objects_in_bev(self):
        """
            Takes detections or labels
        """
        label_objects_3d = self.labels_loader.objects_3d_valid
        self.bev.project_detections_into_bev(self.bev_map, label_objects_3d)

        self.bev_map = cv2.rotate(self.bev_map, cv2.ROTATE_180)
        cv2.imshow('labels (green) vs. detected objects (red)', self.bev_map)
        cv2.waitKey(0)
