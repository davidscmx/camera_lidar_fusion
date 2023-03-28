import git
import unittest
from pathlib import Path

from sensors.lidar import Lidar
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
        self.lidar = Lidar()
        self.lidar.init_frame(self.frame, self.lidar.names.top)

    def test_lidar_names(self):
        self.assertEqual(self.lidar.names.top, dataset_pb2.LaserName.TOP)

    def test_init_frame(self):
        self.assertTupleEqual(self.lidar.range_image.shape, (64, 2650, 4))
        self.assertEqual(self.lidar.calibration.extrinsic.transform[-5], 2.184)

    def test_compute_beam_inclinations(self):
        beam_inclinations = self.lidar.compute_beam_inclinations()
        self.assertEqual(len(beam_inclinations.tolist()), 64)
        self.assertAlmostEqual(beam_inclinations[0], -0.30677331)
        self.assertAlmostEqual(beam_inclinations[-1], 0.04198772)
        
    def tearDown(self):
        self.datafile.file.close()
