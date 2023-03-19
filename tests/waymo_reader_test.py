
import git
import sys
from pathlib import Path
from unittest import TestCase
from simple_waymo_open_dataset_reader import WaymoDataFileReader


class ImportSimpleWaymoDatasetReader(TestCase):

    def setUp(self):

        tf_rec = ("training_segment-10963653239323173269_1924_000_1944_000"
                  "_with_camera_labels_part0.tfrecord")

        git_repo = git.Repo("./", search_parent_directories=True)
        self.test_tf_record = Path(git_repo.working_dir) / "waymo_data" / tf_rec

    def test_count_frames(self):
        datafile = WaymoDataFileReader(self.test_tf_record)
        table = datafile.get_record_table()
        num_frames = len(table)
        self.assertEqual(num_frames, 2)
        datafile.file.close()
