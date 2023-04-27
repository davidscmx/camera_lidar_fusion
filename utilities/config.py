from easydict import EasyDict as edict

lidar_config = edict()
lidar_config.lim_x = (0, 50)
lidar_config.diff_x = lidar_config.lim_x[1] - lidar_config.lim_x[0]
lidar_config.lim_y = (-25, 25)
lidar_config.diff_y = lidar_config.lim_y[1] - lidar_config.lim_y[0]
lidar_config.lim_z = (-1, 3)
lidar_config.bev_width = 608
lidar_config.bev_height = 608

bev_config = edict()
bev_config.width = 608
bev_config.height = 608

det_config = edict()
det_config.conf_thresh = 0.5
det_config.model = 'darknet'

obj_config = edict()
obj_config.colors = [[0, 255, 255],  # 'Pedestrian': 0
                     [0, 0, 255],    # 'Car': 1
                     [255, 0, 0]]   # 'Cyclist': 2