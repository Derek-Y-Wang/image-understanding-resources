"""
Code Solutions for Assignment1

Question 3

Author: Derek Wang
"""
import numpy as np
from skimage import io
from scipy.ndimage import convolve
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def point_formatter(col, row, col_size, row_size, img):
    """
    Padding for points outsize of image
    :param col:
    :param row:
    :param col_size:
    :param row_size:
    :param img:
    :return:
    """
    if not 0 <= col < col_size or not 0 <= row < row_size:
        return 0.
    return img[col][row]


def get_gradient(image, x_deriv, y_deriv):
    """
    Takes in the derivative filter in x direction and y direction
    to get the image matrix based on magnitude and a matrix based on direction
    :param image:
    :param x_deriv:
    :param y_deriv:
    :return:
    """
    result = []
    dir_result = []
    x_deriv = np.fliplr(np.flipud(x_deriv))
    y_deriv = np.fliplr(np.flipud(y_deriv))
    for y in range(len(image)):
        new = []
        dir = []
        for x in range(len(image[0])):
            v0 = point_formatter(y - 1, x - 1, len(image), len(image[0]), image)
            v1 = point_formatter(y - 1, x, len(image), len(image[0]), image)
            v2 = point_formatter(y - 1, x + 1, len(image), len(image[0]), image)

            v3 = point_formatter(y, x - 1, len(image), len(image[0]), image)
            v4 = point_formatter(y, x, len(image), len(image[0]), image)
            v5 = point_formatter(y, x + 1, len(image), len(image[0]), image)

            v6 = point_formatter(y + 1, x - 1, len(image), len(image[0]), image)
            v7 = point_formatter(y + 1, x, len(image), len(image[0]), image)
            v8 = point_formatter(y + 1, x + 1, len(image), len(image[0]), image)
            kernal_space_matrix = np.array([
                [v0, v1, v2],
                [v3, v4, v5],
                [v6, v7, v8]
            ])
            new_x = np.sum(convolve(kernal_space_matrix, x_deriv, mode='constant'))
            new_y = np.sum(convolve(kernal_space_matrix, y_deriv, mode='constant'))

            new_xy = np.sqrt(np.power(new_x, 2) + np.power(new_y, 2))
            new_dir = np.arctan2(new_y, new_x)
            new.append(new_xy)
            dir.append(new_dir)

        result.append(new)
        dir_result.append(dir)
    # print(np.array(dir_result))
    return np.array(result), np.array(dir_result)


def match_template(image, template):
    temp_h, temp_w = template.shape
    row, col = float('inf'), float('inf')
    min_sum = float('inf')
    print(template.shape)
    for y in range(len(image)):
        for x in range(len(image[0])):
            given = image[y:y+temp_h, x:x+temp_w]
            if given.shape != template.shape:
                continue
            diff = np.sum(np.absolute(given - template))
            if diff <= min_sum:
                row, col = y, x
                min_sum = diff

    return row, col


# if __name__ == "__main__":
#     print("Run Assignment1 Question 3 Results")
#     original_waldo = io.imread('waldo.png', as_gray=True)
#     original_template = io.imread('template.png', as_gray=True)
#
#     plt.imshow(original_waldo, cmap='gray')
#     plt.title('original_waldo')
#     plt.show()
#
#     plt.imshow(original_template, cmap='gray')
#     plt.title('original_template')
#     plt.show()
#
#     x_derivative = np.array([
#         [-1, 0, 1],
#         [-2, 0, 2],
#         [-1, 0, 1]
#     ])
#     y_derivative = np.array([
#         [1, 2, 1],
#         [0, 0, 0],
#         [-1, -2, -1]
#     ])
#
#     # question 3a
#     grad_magnitudes_waldo = get_gradient(original_waldo, x_derivative, y_derivative)[0]
#     grad_magnitudes_template = get_gradient(original_template, x_derivative, y_derivative)[0]
#
#     plt.imshow(grad_magnitudes_waldo, cmap='gray')
#     plt.title('gradient mag waldo')
#     plt.show()
#
#     plt.imshow(grad_magnitudes_template, cmap='gray')
#     plt.title('gradient mag template')
#     plt.show()
#
#     # question 3b
#     y, x = match_template(grad_magnitudes_waldo, grad_magnitudes_template)
#     temp_h, temp_w = original_template.shape
#     print(y, x)
#     if y and x:
#         # original_waldo[y:y+temp_h, x:x+temp_w] = 255
#         plt.imshow(original_waldo, cmap='gray')
#         circ = patches.Rectangle((x, y), temp_w, temp_h, color='red', fill=False)
#         plt.gca().add_patch(circ)
#         plt.show()

