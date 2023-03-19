
import sys
from unittest import TestCase


class ImportSimpleWaymoDatasetReader(TestCase):
    def test_imported_correctly(self):
        assert "simple_waymo_open_dataset_reader" in sys.modules.keys(), \
            "simple_waymo_open_dataset_reader"
