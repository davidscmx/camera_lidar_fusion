import cv2
import numpy as np

THREE_DIMS = 3


def draw_images_side_by_side(images, max_images_per_row, resize):
    # Calculate the dimensions of the canvas
    num_images = len(images)
    num_rows = (num_images + max_images_per_row - 1) // max_images_per_row
    canvas_height = max(image.shape[0] for image in images)
    canvas_width = max_images_per_row * max(image.shape[1] for image in images)

    resized_images = [cv2.resize(image, resize) for image in images]
    canvas_height = max(resize[1] for _ in resized_images)
    canvas_width = max_images_per_row * resize[0]

    # Create a blank canvas
    canvas = np.zeros((canvas_height * num_rows, canvas_width, 3), dtype=np.uint8)

    # Paste the images onto the canvas
    x_offset = 0
    y_offset = 0
    count = 0

    for image in resized_images:
        print("image shape", image.shape)
        canvas[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image
        x_offset += image.shape[1]
        count += 1

        # Move to the next row if maximum images per row is reached
        if count % max_images_per_row == 0:
            y_offset += canvas_height
            x_offset = 0

    # Display the canvas using OpenCV
    cv2.imshow("Images Side by Side", canvas)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class MultiImageDrawer:
    def __init__(self, images: dict, width=480, height=320):
        draw_images_side_by_side(images.values(), max_images_per_row=3,
                                 resize=(width, height))
