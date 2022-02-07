"""
Canny Edge Detection

Question 4

Author: Derek Wang
"""
import numpy as np
from skimage import io, data, color, feature
import matplotlib.pyplot as plt
import math
from template_matching import get_gradient


def point_formatter(col, row, col_size, row_size, img):
    if not 0 <= col < col_size or not 0 <= row < row_size:
        return 0.
    return img[col][row]


def question4(image):
    """
    Performs canny edge detection
    :param image:
    :return:
    """
    x_derivative = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    y_derivative = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    result = []

    # get gradient of image
    mag, dir = get_gradient(image, x_derivative, y_derivative)
    for y in range(len(image)):
        new = []
        for x in range(len(image[0])):
            degree = dir[y][x] * (180 / math.pi)

            # degree are [0, 180] or [0, -180]
            rd_degree = round(degree / 45) * 45
            opp_degree = -rd_degree
            dc = {
                0: (0, 1),
                45: (-1, 1),
                90: (-1, 0),
                135: (-1, -1),
                180: (0, -1),
                -180: (0, -1),
                -135: (1, -1),
                -90: (1, 0),
                -45: (1, 1),
            }
            pixel = mag[y][x]
            side1 = point_formatter(y + dc[rd_degree][0], x + dc[rd_degree][1], len(mag), len(mag[0]), mag)
            side2 = point_formatter(y + dc[opp_degree][0], x + dc[opp_degree][1], len(mag), len(mag[0]), mag)

            if pixel == max(pixel, side1, side2):
                new.append(pixel)
            else:
                new.append(0)

        result.append(new)
    return result


# if __name__ == "__main__":
#     print("Run Assignment1 Question 4 Results")
#     original = np.array(io.imread('waldo.png', as_gray=True))
#     plt.imshow(original, cmap='gray')
#     plt.title("Original")
#     plt.show()
#
#
#     canny = question4(original)
#     plt.imshow(canny, cmap='gray')
#     plt.title("Canny")
#     plt.show()
