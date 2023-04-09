import cv2
from pathlib import Path
from collections import namedtuple
import open3d

# Waymo open dataset reader
from simple_waymo_open_dataset_reader import WaymoDataFileReader

from data_loaders.camera_loader import CameraLoader

from sensors.lidar import Lidar
from visualizer.cam_visualizer import CameraVisualizer
from visualizer.lidar_visualizer import LidarVisualizer, RangeImgChannel

frame_range = namedtuple("frame_range", ["start", "end"])
f_range = frame_range(0, 100)

tf_rec = ("training_segment-10963653239323173269_1924_000_1944_000"
          "_with_camera_labels_part0.tfrecord")

test_tf_record = Path("waymo_data") / tf_rec
datafile = WaymoDataFileReader(test_tf_record)


cam_loader = CameraLoader()
lidar = Lidar()
lidar_viz = LidarVisualizer(lidar)
vis = open3d.visualization.VisualizerWithKeyCallback()

for cnt_frame in range(f_range.start, f_range.end):
    frame = next(iter(datafile))
    cam_loader.set_frame(frame)
    cam_loader.set_selected_camera(cam_loader.camera_names.front)
    img = cam_loader.decode_single_image()

    cam_viz = CameraVisualizer(img, frame.laser_labels,
                               cam_loader.get_camera_calibration())

    #img = cam_viz.project_laser_labels_into_image()
    lidar.init_frame(frame, lidar.names.top)
    #ri_range = lidar_viz.get_img_selected_channel(RangeImgChannel.Range)
    #ri_intensity = lidar_viz.get_img_selected_channel(RangeImgChannel.Intensity)

    lidar_viz.display_laser_on_image(img, cam_loader.get_camera_calibration())

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.imshow("img", img)
    cv2.waitKey(0)

