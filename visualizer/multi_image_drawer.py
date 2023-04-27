import cv2
import numpy as np

THREE_DIMS = 3
class MultiImageDrawer:
    def __init__(self, images: dict, width=480, height=320):
        # Resize the images to a fixed size (e.g., 400x400)
        width2 = 320
        for key in images:
            if "pcl" in key:

                if "bev" in key:
                    images[key] = self.get_bev(images[key], 320, 320)
                else:
                    images[key] = self.get_1D_map(images[key])
                    images[key] = cv2.resize(images[key], (320, 320))
            if "img" in key:
                images[key] = cv2.resize(images[key], (width, height))

        canvas = np.zeros((2*height, width+(2*width2), THREE_DIMS), dtype=np.uint8)

        # Copy the images onto the canvas
        canvas[0:height, 0:width] = images["img_with_gt"]
        canvas[0:height, width:width+width2] = self.tile_gray(images["pcl.height_map"])
        canvas[0:height, width+width2:width+(2*width2)] = self.tile_gray(images["pcl.intensity_map"])

        canvas[height:2*height, 0:width] = images["img_with_gt"]
        canvas[height:2*height, width:width+width2] = images["pcl.assembled_bev"]
        canvas[height:2*height, width+width2:width+(2*width2)] = self.tile_gray(images["pcl.height_map"])

        # Display the canvas
        cv2.namedWindow("Six Images", cv2.WINDOW_NORMAL)
        cv2.imshow("Six Images", canvas)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def tile_gray(self, img):
        img = np.tile(img[..., np.newaxis], (1, 1, 3))
        return img

    def get_1D_map(self, custom_map):
        custom_map = custom_map * 255
        custom_map = custom_map.astype(np.uint8)
        custom_map = cv2.rotate(custom_map, cv2.ROTATE_180)

        return custom_map

    def get_bev(self, bev_maps, height, width):
        bev_map = (bev_maps.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        bev_map = cv2.resize(bev_map, (width, height))
        bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
        return bev_map
