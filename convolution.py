"""

"""

import numpy as np
from skimage import io, data, color
import matplotlib.pyplot as plt
from timeit import default_timer as timer


def point_formatter(col, row, col_size, row_size, img):
    """
    This function is for padding when a point of the kernal is out of the
    image range

    for 0 padding
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


def convolution(img, kernal):
    """
    Convolves image by a 3x3 kernal
    :param img:
    :param kernal:
    :return:
    """
    image = img
    formatted_image = []
    # kernal = np.fliplr(np.flipud(kernal))
    for y in range(len(image)):
        new_row = []
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
            kernal_space_matrix = np.fliplr(np.flipud(kernal_space_matrix))

            processed_pixel = np.sum(kernal * kernal_space_matrix)
            new_row.append(processed_pixel)
        formatted_image.append(new_row)

    print('finish 1a.')
    return np.array(formatted_image)


def check_for_seperable(kernal):
    """
    Determines if kernal is seperable
    :param kernal:
    :return:
    """
    svd = np.linalg.svd(kernal)
    # print(f'non zeros {np.count_nonzero(np.round(rank[1], decimals=5))}')
    return np.count_nonzero(np.round(svd[1], decimals=5)) == 1


def convolution_based_on_seperable(img, kernal):
    """
    If kernal is seperable splits it into 1D arrays which are then used
    to convolve image. Reduces run time
    :param img:
    :param kernal:
    :return:
    """
    if not check_for_seperable(kernal):
        return False

    U, S, V = np.linalg.svd(kernal)
    s = np.argmax(np.round(S[1], decimals=5))
    row = np.sqrt(S[s]) * U[s]
    col = np.sqrt(S[s]) * V[s].T

    inter_img = []
    output_img = []
    for y in range(len(img)):
        n_row = []
        for x in range(len(img[0])):
            v1 = point_formatter(y, x - 1, len(img), len(img[0]), img)
            v2 = point_formatter(y, x, len(img), len(img[0]), img)
            v3 = point_formatter(y, x + 1, len(img), len(img[0]), img)
            conv = row * np.array([v1, v2, v3])
            n_row.append(np.sum(conv))
        inter_img.append(n_row)

    inter_img = np.array(inter_img)

    for y in range(len(inter_img)):
        n_col = []
        for x in range(len(inter_img[0])):
            v1 = point_formatter(y, x - 1, len(inter_img[0]), len(inter_img), inter_img)
            v2 = point_formatter(y, x, len(inter_img[0]), len(inter_img), inter_img)
            v3 = point_formatter(y, x + 1, len(inter_img[0]), len(inter_img), inter_img)
            conv = col * np.array([v1, v2, v3])
            n_col.append(np.sum(conv))
        output_img.append(n_col)
    print('Finished 1c.')
    return np.array(output_img)




# if __name__ == "__main__":
#     print("Run Assignment1 Question 1 Results")
#     original = io.imread('waldo.png', as_gray=True)
#     plt.imshow(original, cmap='gray')
#     plt.title('original')
#     plt.show()
#
#     filter_2D = np.array([
#         [0, 0.125, 0],
#         [0.5, 0.5, 0.125],
#         [0, 0.5, 0]
#     ])
#
#     # Question 1a
#     q1a_start = timer()
#     q1 = question1_a(original, filter_2D)
#     q1a_end = timer()
#     print(f'Q1a Process Time: {q1a_end - q1a_start}')
#     plt.imshow(q1, cmap='gray')
#     plt.title('1a')
#     plt.show()
#
#     # Question 1b and 1c
#     separable_filter = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#     # question1_b(separable_filter)
#
#     q1c_start = timer()
#     q1c = question1_c(original, separable_filter)
#     q1c_end = timer()
#     print(f'Q1c Process Time: {q1c_end - q1c_start}')
#     plt.imshow(q1c, cmap='gray')
#     plt.title('1bc')
#     plt.show()

