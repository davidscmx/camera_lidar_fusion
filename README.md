# Camera/Lidar Fusion with The Waymo Dataset



[![Python package](https://github.com/davidscmx/camera_lidar_fusion/actions/workflows/python.yml/badge.svg)](https://github.com/davidscmx/camera_lidar_fusion/actions/workflows/python.yml)



## Front camera image with top laser 3D labels

<img src="media/range_image_explanation.png"/>

## What are "Range Images"?
They are an alternative way to represent lidar scans.
This data structure holds 3d points as a 360 degree "photo" of the scanning environment with the row
dimension denoting the elevation angle of the laser beam and the column dimension denoting the azimuth angle.

<img src="media/range_image_explanation.png"/>

## Lidar Resolution

- 64 lines and 2650 columns

Shape of the lidar range frame:
(64, 2650, 4)

64 in elevation/pitch
2650 in azimuth/yaw
Resolution in the azimuth:
360 / 2650 0.1358 degrees * (60 angular minutes/ degree) ~ 8 angular minutes

<img src="media/point_cloud_waymo.gif"/>
