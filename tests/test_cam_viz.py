
import cv2
import git
import numpy as np
import unittest
from pathlib import Path

from data_loaders.camera_loader import CameraLoader
from visualizer.cam_visualizer import CameraVisualizer

from simple_waymo_open_dataset_reader import dataset_pb2
from simple_waymo_open_dataset_reader import WaymoDataFileReader


class TestCameraVisualization(unittest.TestCase):
    def setUp(self):
        tf_rec = ("training_segment-10963653239323173269_1924_000_1944_000"
                  "_with_camera_labels_part0.tfrecord")

        git_repo = git.Repo("./", search_parent_directories=True)
        test_tf_record = Path(git_repo.working_dir) / "waymo_data" / tf_rec
        
        self.datafile = WaymoDataFileReader(test_tf_record)

        frame = next(iter(self.datafile))

        cam_loader = CameraLoader()
        cam_loader.set_frame(frame)
        cam_loader.set_selected_camera(cam_loader.camera_names.front)
        
        img = cam_loader.decode_single_image()
        
        self.cam_viz = CameraVisualizer(img, frame.laser_labels, 
                                        cam_loader.get_camera_calibration())

    def test_project_laser_labels_into_image(self):
        img_with_labels = self.cam_viz.project_laser_labels_into_image()        

        ref_img = "training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels_part0_laser_labels.png"
        ref_img_dir = Path("tests/reference_data")
        ref_img_path = ref_img_dir / ref_img
        assert ref_img_path.exists(), "Image not found"
        ref_img = cv2.imread(str(ref_img_path))
        
        self.assertIsNone(np.testing.assert_array_equal(img_with_labels, ref_img, verbose=True))

    def tearDown(self):
        self.datafile.file.close()


if __name__ == "__main__":
    unittest.main()
