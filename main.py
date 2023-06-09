import cv2
from pathlib import Path
from collections import namedtuple
import open3d

# Waymo open dataset reader
from simple_waymo_open_dataset_reader import WaymoDataFileReader

from data_loaders.camera_loader import CameraLoader

from sensors.lidar import Lidar
from sensors.pcl import Pcl

from visualizer.cam_visualizer import CameraVisualizer
from visualizer.lidar_visualizer import LidarVisualizer, RangeImgChannel
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
lidar_viz = LidarVisualizer(lidar)
vis = open3d.visualization.VisualizerWithKeyCallback()


for cnt_frame in range(f_range.start, f_range.end):

    frame = next(iter(datafile))
    cam_loader.set_frame(frame)
    cam_loader.set_selected_camera(cam_loader.camera_names.front)
    img = cam_loader.decode_single_image()

    cam_viz = CameraVisualizer(img, frame.laser_labels,
                               cam_loader.get_camera_calibration())

    lidar.init_frame(frame, lidar.names.top)
    pcl = Pcl(lidar.get_pcl_range_image())

    #lidar_viz.draw_1D_map(pcl.intensity_map, "intensity")
    #lidar_viz.draw_1D_map(pcl.height_map, "height")
    #lidar_viz.draw_1D_map(pcl.density_map, "density")
#
    #lidar_viz.show_bev(pcl.assembled_bev_from_maps, pcl.bev_height, pcl.bev_width)
    #img_with_gt = cam_viz.project_laser_labels_into_image()
    #print(img_with_gt.shape)
    #ri_range = lidar_viz.get_img_selected_channel(RangeImgChannel.Range)
    #ri_intensity = lidar_viz.get_img_selected_channel(RangeImgChannel.Intensity)

    #images_dic = {}
    #images_dic["pcl.intensity_map"] = pcl.intensity_map
    #images_dic["pcl.height_map"] = pcl.height_map
    #images_dic["pcl.density_map"] = pcl.density_map
    #images_dic["pcl.assembled_bev"] = pcl.assembled_bev_from_maps
    #images_dic["img_with_gt"] = img_with_gt
#
    #multi = MultiImageDrawer(images_dic)
