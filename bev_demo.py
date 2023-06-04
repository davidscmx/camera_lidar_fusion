import cv2
from pathlib import Path
from collections import namedtuple

# Waymo open dataset reader
from simple_waymo_open_dataset_reader import WaymoDataFileReader

from sensors.lidar import Lidar
from sensors.pcl import Pcl
from visualizer.bev_visualizer import BevVisualizer

frame_range = namedtuple("frame_range", ["start", "end"])
f_range = frame_range(0, 100)

tf_rec = ("training_segment-10963653239323173269_1924_000_1944_000"
          "_with_camera_labels.tfrecord")

test_tf_record = Path("waymo_data") / tf_rec
datafile = WaymoDataFileReader(test_tf_record)
lidar = Lidar()

for cnt_frame in range(f_range.start, f_range.end):
    frame = next(iter(datafile))

    lidar.init_frame(frame, lidar.names.top)

    pcl = Pcl(lidar.get_pcl_range_image())

    # Instantiate BEV Visualizer
    bev_viz = BevVisualizer(pcl.assembled_bev_from_maps, frame.laser_labels)

    cv2.imshow('labels (green) vs. detected objects (red)', bev_viz.bev_map_with_labels)
    cv2.waitKey(0)
