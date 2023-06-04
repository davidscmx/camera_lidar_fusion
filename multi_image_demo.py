from pathlib import Path
from collections import namedtuple

# Waymo open dataset reader
from simple_waymo_open_dataset_reader import WaymoDataFileReader
from data_loaders.camera_loader import CameraLoader

from sensors.lidar import Lidar
from sensors.pcl import Pcl

from visualizer.cam_visualizer import CameraVisualizer
from visualizer.multi_image_drawer import MultiImageDrawer
from visualizer.bev_visualizer import BevVisualizer

frame_range = namedtuple("frame_range", ["start", "end"])
f_range = frame_range(0, 100)

tf_rec = ("training_segment-10963653239323173269_1924_000_1944_000"
          "_with_camera_labels.tfrecord")

test_tf_record = Path("waymo_data") / tf_rec
datafile = WaymoDataFileReader(test_tf_record)

cam_loader = CameraLoader()
lidar = Lidar()

for cnt_frame in range(f_range.start, f_range.end):
    frame = next(iter(datafile))
    cam_loader.set_frame(frame)
    cam_loader.set_selected_camera(cam_loader.camera_names.front)
    img = cam_loader.decode_single_image()

    cam_viz = CameraVisualizer(img, frame.laser_labels,
                               cam_loader.get_camera_calibration())

    lidar.init_frame(frame, lidar.names.top)
    pcl = Pcl(lidar.get_pcl_range_image())

    # Instantiate BEV Visualizer
    bev_viz = BevVisualizer(pcl.assembled_bev_from_maps, frame.laser_labels)

    img_with_gt = cam_viz.project_laser_labels_into_image()

    images_dic = {}
    images_dic["img_with_gt"] = img_with_gt # 3d image
    images_dic["pcl.intensity_map_1d_map"] = pcl.intensity_map_1d_map
    images_dic["pcl.density_map_1d_map"] = pcl.density_map_1d_map
    images_dic["pcl.height_map_1d_map"] = pcl.height_map_1d_map
    images_dic["bev_labels"] = bev_viz.bev_map_with_labels

    multi = MultiImageDrawer(images_dic)
