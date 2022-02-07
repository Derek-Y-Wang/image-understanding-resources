"""
Gaussian Blur

Author: Derek Wang
"""
from scipy.ndimage import convolve
import numpy as np
from skimage import io, color, filters
import math
import matplotlib.pyplot as plt


def gaussian_formula(u, v, sigma=1):
    """
    Implentation of the Gaussian formula in 2D
    :param u:
    :param v:
    :param sigma:
    :return:
    """
    e = -(math.pow(u, 2) + math.pow(v, 2)) / (2 * (math.pow(sigma, 2)))

    result = np.exp(e) / (2 * math.pi * math.pow(sigma, 2))
    return result


def create_gaussian(size, sigma=1):
    """
    Creates a Gaussian Matrix of size n and float value sigma.

    SIZE OF MATRiX SHOULD BE ODD NUMBERED
    :param size:
    :param sigma:
    :return:
    """
    result = []

    for i in range(size):
        r = []
        for j in range(size):
            y, x = i - (size // 2), j - (size // 2)
            r.append(gaussian_formula(y, x, sigma))
        result.append(r)

    # print(np.array(result))
    return np.array(result)


def blur_image(image, size, sigma=1):
    """
    Creates a Gaussian matrix and convolves it against the image
    :param image:
    :param size:
    :param sigma:
    :return:
    """
    filter = create_gaussian(size, sigma)
    return convolve(image, filter, cval=0.0, mode='constant')


# if __name__ == "__main__":
#     print("Run Assignment1 Question 2 Results")
#     original = io.imread('waldo.png', as_gray=True)
#     plt.imshow(original, cmap='gray')
#     plt.title("Original")
#     plt.show()
#
#     # Question 2
#     sigma = 1
#     size = 3
#     # size should be odd
#     new = question2(original, size, sigma)
#     print(new.shape)
#     plt.imshow(new, cmap='gray')
#     plt.title("Gaussian")
#     plt.show()


