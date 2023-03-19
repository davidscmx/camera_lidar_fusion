
import sys
from unittest import TestCase
from simple_waymo_open_dataset_reader import WaymoDataFileReader


class ImportSimpleWaymoDatasetReader(TestCase):
    def test_imported_correctly(self):
        print()
        assert "simple_waymo_open_dataset_reader" in sys.modules.keys(), \
            "simple_waymo_open_dataset_reader not found"
