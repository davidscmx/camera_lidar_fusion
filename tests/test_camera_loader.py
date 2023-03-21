
import cv2
import git
import numpy as np
import unittest
from pathlib import Path

from data_loaders.camera_loader import CameraLoader
from simple_waymo_open_dataset_reader import dataset_pb2
from simple_waymo_open_dataset_reader import WaymoDataFileReader


class TestCameraLoader(unittest.TestCase):
    def setUp(self):
        tf_rec = ("training_segment-10963653239323173269_1924_000_1944_000"
                  "_with_camera_labels_part0.tfrecord")

        git_repo = git.Repo("./", search_parent_directories=True)
        test_tf_record = Path(git_repo.working_dir) / "waymo_data" / tf_rec

        self.datafile = WaymoDataFileReader(test_tf_record)

        self.frame = next(iter(self.datafile))

        self.cam_loader = CameraLoader()
        self.cam_loader.set_frame(self.frame)
        self.cam_loader.set_selected_camera(self.cam_loader.camera_names.front)

        self.camera_image = self.cam_loader.load_camera_data_structure()

    def test_camera_names(self):
        self.assertEqual(self.cam_loader.camera_names.front, dataset_pb2.CameraName.FRONT)

        self.assertEqual(self.cam_loader.camera_names.front_right,
                         dataset_pb2.CameraName.FRONT_RIGHT)

        self.assertEqual(self.cam_loader.camera_names.front_left,
                         dataset_pb2.CameraName.FRONT_LEFT)

    def test_load_camera_data_structure(self):
        self.assertIsInstance(self.camera_image, dataset_pb2.CameraImage)

    def test_convert_image_to_rgb(self):
        self.img = self.cam_loader.convert_image_to_rgb(self.camera_image)
        self.assertEqual(self.img.shape[2], 3)

    def test_resize_img(self):
        img = self.cam_loader.convert_image_to_rgb(self.camera_image)
        original_shape = img.shape
        resized_img = self.cam_loader.resize_img(img, factor=0.5)
        self.assertEqual(original_shape[0]/2, resized_img.shape[0])
        self.assertEqual(original_shape[1]/2, resized_img.shape[1])

        resized_img_to_dims = self.cam_loader.resize_img_to_dims(img, (500, 500))
        self.assertEqual(500, resized_img_to_dims.shape[0])
        self.assertEqual(500, resized_img_to_dims.shape[1])

    def test_decode_single_image(self):
        img = self.cam_loader.decode_single_image()
        ref_img = "training_segment-10963653239323173269_1924_000_1944_000_with_camera_labels_front_camera.png"
        ref_img_dir = Path("tests/reference_data")
        ref_img_path = ref_img_dir / ref_img
        assert ref_img_path.exists(), "Image not found"
        ref_img = cv2.imread(str(ref_img_path))
        self.assertIsNone(np.testing.assert_array_equal(img, ref_img, verbose=True))

    def tearDown(self):
        self.datafile.file.close()


if __name__ == "__main__":
    unittest.main()
