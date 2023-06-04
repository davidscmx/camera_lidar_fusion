from data_loaders.labels_loader import LabelsLoader
from sensors.bev import Bev
from utilities.config import bev_config
import numpy as np
import cv2


class BevVisualizer:
    def __init__(self, bev_maps, labels):

        # Google protobuf object
        self.labels_loader = LabelsLoader(labels)

        self.bev = Bev()
        self.bev_map = self._process_bev(bev_maps)
        self.bev_map_with_labels = self._get_labels_in_bev_image()

    def _process_bev(self, bev_maps):
        bev_map = (bev_maps.squeeze().permute(
            1, 2, 0).numpy() * 255).astype(np.uint8)
        bev_map = cv2.resize(bev_map, (bev_config.width, bev_config.height))
        return bev_map

    def _get_labels_in_bev_image(self):
        """
            Takes detections or labels
        """
        label_objects_3d = self.labels_loader.objects_3d_valid
        self.bev.project_detections_into_bev(self.bev_map, label_objects_3d)

        return cv2.rotate(self.bev_map, cv2.ROTATE_180)
