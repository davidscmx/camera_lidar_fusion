
import numpy as np

from simple_waymo_open_dataset_reader import label_pb2
from simple_waymo_open_dataset_reader import utils as waymo_utils
from data_loaders.labels_loader import LabelsLoader

class CameraVisualizer():
    def __init__(self, image, labels, camera_calibration):
        labels_loader = LabelsLoader(labels)
        self.image = image
        self.labels = labels_loader.labels
        self.camera_calibration = camera_calibration
        # get transformation matrix from vehicle frame to image
        self.vehicle_to_image = waymo_utils.get_image_transform(self.camera_calibration)

    def project_laser_labels_into_image(self):
        image_with_labels = np.copy(self.image)        
        for label in self.labels:
            waymo_utils.draw_3d_box(image_with_labels, self.vehicle_to_image, label, colour=(255, 0, 0))                          
        return image_with_labels

    