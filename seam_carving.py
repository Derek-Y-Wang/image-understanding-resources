import numpy as np
from skimage import io
# from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from scipy.ndimage.filters import convolve


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

# question 1a
def get_gradient(image):
    """
    Takes in the derivative filter in x direction and y direction
    to get the image matrix based on magnitude and a matrix based on direction
    :param image:
    :param x_deriv:
    :param y_deriv:
    :return:
    """
    x_deriv = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])
    y_deriv = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])
    result = []
    x_deriv = np.fliplr(np.flipud(x_deriv))
    y_deriv = np.fliplr(np.flipud(y_deriv))
    for y in range(len(image)):
        new = []
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
            new.append(new_xy)

        result.append(new)
    return np.array(result)


def find_min_energy(energy_map):
    r, c = energy_map.shape
    for y in range(r):
        for x in range(c):
            if y == 0:
                continue
            if x == 0:
                loc = np.argmin(energy_map[y - 1, 0:2])
                min_energy = energy_map[y - 1, loc]
            else:
                loc = np.argmin(energy_map[y - 1, x - 1:x + 2])
                min_energy = energy_map[y - 1, loc + x - 1]
            energy_map[y, x] += min_energy
    return energy_map


def get_path(energy_map):
    r, c = energy_map.shape
    path = []
    path_index = energy_map[0].argmin()
    path.append(path_index)
    for i in range(r - 1):
        col_start = max(path_index - 1, 0)
        col_end = min(path_index + 2, c)
        path_index = (path_index - 1) + energy_map[i + 1, col_start:col_end].argmin()
        path.append(path_index)
    return path


def show_seam(cost, img):
    for y in range(len(img)):
        img[y][cost[y]] = 1

    return img


def remove_seam(path, img):
    result = []
    for y in range(len(img)):
        result.append(np.delete(img[y], path[y], 0))

    result = np.array(result)
    return result


def remove_multiple_seams(original, n_removed):
    print(original.shape)
    prev = original
    for i in range(n_removed):
        gradient_mag = get_gradient(prev)
        energy = find_min_energy(gradient_mag)
        # plt.imshow(energy, cmap='gray')
        # plt.title("energy")
        # plt.show()

        path = get_path(energy)
        new_img = remove_seam(path, prev)
        prev = new_img
    print(prev.shape)
    return prev


if __name__ == "__main__":
    original_img = io.imread('castle.png', as_gray=True)

    plt.imshow(original_img, cmap='gray')
    plt.title('original')
    plt.show()

    # gradient_mag = get_gradient(original_img)
    # plt.imshow(gradient_mag, cmap='gray')
    # plt.title("gradient mag castle")
    # plt.show()
    #
    # seam = find_seams(gradient_mag)
    # plt.imshow(seam, cmap='gray')
    # plt.title("seam")
    # plt.show()


    # new_img = remove_col(original_img, gradient_mag)
    # plt.imshow(new_img, cmap='gray')
    # plt.title("removed one col")
    # plt.show()

    # path = get_path(seam)
    #
    # new_img = show_seam(path, original_img)
    # # new_img = remove_seam(path, gradient_mag)
    # plt.imshow(new_img, cmap='gray')
    # plt.title("image with seam")
    # plt.show()

    final = remove_multiple_seams(original_img, 50)
    plt.imshow(final, cmap='gray')
    plt.title("reduced")
    plt.show()

