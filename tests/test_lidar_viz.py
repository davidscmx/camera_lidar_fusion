
import cv2
import git
import numpy as np
import unittest
from pathlib import Path

from sensors.lidar import Lidar
from visualizer.lidar_visualizer import LidarVisualizer

from simple_waymo_open_dataset_reader import dataset_pb2
from simple_waymo_open_dataset_reader import WaymoDataFileReader


class TestLidarVisualization(unittest.TestCase):
    def setUp(self):
        tf_rec = ("training_segment-10963653239323173269_1924_000_1944_000"
                  "_with_camera_labels_part0.tfrecord")

        git_repo = git.Repo("./", search_parent_directories=True)
        test_tf_record = Path(git_repo.working_dir) / "waymo_data" / tf_rec

        self.datafile = WaymoDataFileReader(test_tf_record)

        frame = next(iter(self.datafile))

        lidar = Lidar()
        lidar_viz = LidarVisualizer(lidar)
        lidar.init_frame(frame, lidar.names.top)

    def tearDown(self):
        self.datafile.file.close()


if __name__ == "__main__":
    unittest.main()
