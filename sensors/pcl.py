
import numpy as np
import torch

from utilities.config import lidar_config



INTENSITY_DIM = 3
HEIGHT_DIM = 2

class Pcl:
    def __init__(self, pcl):
        self.pcl = pcl
        self.lim_x = lidar_config.lim_x
        self.lim_y = lidar_config.lim_y
        self.lim_z = lidar_config.lim_z
        self.bev_height = lidar_config.bev_height
        self.bev_width = lidar_config.bev_width

        self.discretize_for_bev()

    def crop_point_cloud(self):
        mask = np.where((self.pcl[:, 0] >= self.lim_x[0]) & (self.pcl[:, 0] <= self.lim_x[1]) &
                        (self.pcl[:, 1] >= self.lim_y[0]) & (self.pcl[:, 1] <= self.lim_y[1]) &
                        (self.pcl[:, 2] >= self.lim_z[0]) & (self.pcl[:, 2] <= self.lim_z[1]))

        self.pcl = self.pcl[mask]



    def discretize_for_bev(self):
        # remove lidar points outside detection area and with too low reflectivity
        self.crop_point_cloud()

        bev_discret = (self.lim_x[1] - self.lim_x[0]) / self.bev_height

        self.pcl[:, 0] = np.int_(np.floor(self.pcl[:, 0] / bev_discret))
        # Make sure that no negative bev-coordinates occur by adding half of the width
        self.pcl[:, 1] = np.int_(np.floor(self.pcl[:, 1] / bev_discret) +
                                 (self.bev_width + 1) / 2)

        # shift level of ground plane to avoid flipping from 0 to
        # 255 for neighboring pixels
        self.pcl[:, 2] = self.pcl[:, 2] - self.lim_z[0]

    def get_sorted_lidar_pcl_according_to_dim(self, dim2sort=None):

        proc_pcl = np.copy(self.pcl)

        if dim2sort == INTENSITY_DIM:
            proc_pcl[proc_pcl[:, 3] > 1.0, 3] = 1.0

        # sort points such that in case of identical BEV grid coordinates,
        # the points in each grid cell are arranged based on their intensity
        indices = np.lexsort((-proc_pcl[:, dim2sort],
                              proc_pcl[:, 1],
                              proc_pcl[:, 0]))

        proc_pcl = proc_pcl[indices]
        # only keep one point per grid cell
        _, indices = np.unique(proc_pcl[:, 0:2], axis=0, return_index=True)
        topsorted_pcl = proc_pcl[indices]

        return topsorted_pcl

    @property
    def intensity_map(self):
        intensity_map = np.zeros((self.bev_height + 1, self.bev_width + 1))
        topsorted_pcl = self.get_sorted_lidar_pcl_according_to_dim(dim2sort=3)

        top_sorted_x, top_sorted_y, top_sorted_z = self._get_xyz_components(topsorted_pcl)

        intensity_map[np.int_(top_sorted_x), np.int_(top_sorted_y)] = \
            top_sorted_z / (np.amax(top_sorted_z) - np.amin(top_sorted_z))

        return intensity_map

    @property
    def height_map(self):
        """
            Assign the height value of each unique entry in lidar_top_pcl
            to the height map and make sure that each entry is normalized on
            the difference between the upper and lower height defined in the
            config file
        """
        pcl_height = self.get_sorted_lidar_pcl_according_to_dim(dim2sort=2)

        height_map = np.zeros((self.bev_height + 1, self.bev_width + 1))

        pcl_height_x, pcl_height_y, pcl_height_z = self._get_xyz_components(pcl_height)

        height_map[np.int_(pcl_height_x), np.int_(pcl_height_y)] = \
            pcl_height_z / float(np.abs(self.lim_z[1] - self.lim_z[0]))

        return height_map

    @property
    def density_map(self):
        # Compute density layer of the BEV map
        density_map = np.zeros((self.bev_height + 1, self.bev_width + 1))
        _, _, counts = np.unique(self.pcl[:, 0:2], axis=0, return_index=True, return_counts=True)
        normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64))

        topsorted_pcl = self.get_sorted_lidar_pcl_according_to_dim(dim2sort=2)
        top_sorted_x, top_sorted_y, _ = self._get_xyz_components(topsorted_pcl)
        density_map[np.int_(top_sorted_x), np.int_(top_sorted_y)] = normalizedCounts

        return density_map

    def _get_xyz_components(self, pcl):
        pcl_x = pcl[:, 0]
        pcl_y = pcl[:, 1]
        pcl_z = pcl[:, 2]

        return (pcl_x, pcl_y, pcl_z)

    def assemble_bev_from_maps(density_map, intensity_map, height_map, configs):
        # assemble 3-channel bev-map from individual maps
        bev_map = np.zeros((3, self.bev_height, self.bev_width))

        bev_map[2, :, :] = density_map[:self.bev_height, :self.bev_width]  # r_map
        bev_map[1, :, :] = height_map[:self.bev_height, :self.bev_width]  # g_map
        bev_map[0, :, :] = intensity_map[:self.bev_height, :self.bev_width]  # b_map

        # expand dimension of bev_map before converting into a tensor
        s1, s2, s3 = bev_map.shape
        bev_maps = np.zeros((1, s1, s2, s3))
        bev_maps[0] = bev_map

        bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
        input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()

        return input_bev_maps