import numpy as np
from skimage import io
# from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import signal as sig
import matplotlib.patches as patches


def get_gradient(image):
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

    return sig.convolve2d(image, x_deriv, mode='same'), sig.convolve2d(image,
                                                                       y_deriv,
                                                                       mode='same')


def harris_edge_detection(bw_img, alpha, threshold):
    Ix, Iy = get_gradient(bw_img)
    # print(Ix.shape, Iy.shape)
    Ix_2, Iy_2, Ixy = Ix ** 2, Iy ** 2, Ix * Iy
    det = gaussian_filter(Ix_2, sigma=3) * gaussian_filter(Iy_2,
                                                           sigma=3) - gaussian_filter(
        Ixy, sigma=3)
    trace = gaussian_filter(Ix_2, sigma=3) + gaussian_filter(Iy_2, sigma=3)

    R = det - alpha * (trace ** 2)
    corners = np.argwhere(R > threshold)

    new_corners = []
    # non maxima supression
    for i in corners:
        y, x = i
        checkable = []
        for d in [(0, 0), (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1),
                  (1, 0), (1, 1)]:
            if 0 <= y + d[0] < len(R) and 0 <= x + d[1] < len(R[0]) and \
                    R[y + d[0]][x + d[1]] > threshold:
                checkable.append(R[y + d[0]][x + d[1]])

        if max(checkable) == R[y][x]:
            new_corners.append((y, x))
    return new_corners


if __name__ == "__main__":
    original_img = io.imread('building.jpg')
    bw_img = io.imread('building.jpg', as_gray=True)

    plt.imshow(bw_img, cmap='gray')
    plt.title('original')
    plt.show()

    corners = harris_edge_detection(bw_img, 0.05, 0.05)

    plt.imshow(original_img)
    plt.title('corner')
    for i in corners:
        y, x = i
        circ = patches.Circle((x, y), 1, color='red', fill=True)
        plt.gca().add_patch(circ)
    plt.show()
