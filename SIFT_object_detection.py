import numpy as np
from skimage import io
# from scipy.ndimage import convolve
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import signal as sig
import matplotlib.patches as patches
import random
import cv2


def feature_extraction(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    return sift.detectAndCompute(gray, None)


def plot_features(img):
    kp, des = feature_extraction(img)
    points = random.sample(range(len(kp)), 100)
    kp = [kp[i] for i in points]

    plt.imshow(img)
    plt.title('features img')
    for point in kp:
        x, y, size = point.pt[0], point.pt[1], point.size
        circ = patches.Circle((x, y), size, color='red', fill=False)
        plt.gca().add_patch(circ)

        circ = patches.Circle((x, y), 1, color='yellow', fill=True)
        plt.gca().add_patch(circ)

    plt.show()


def matcher(ref, test, threshold):
    ref_kp, ref_des = feature_extraction(ref)
    test_kp, test_des = feature_extraction(test)

    original_points = []
    threshold_points = []
    for ref in range(len(ref_des)):
        ref_point, ref_d = ref_kp[ref], ref_des[ref]
        temp = []
        for test in range(len(test_kp)):
            test_point, test_d = test_kp[test], test_des[test]
            dist = np.sqrt(np.sum((ref_d - test_d) ** 2))
            temp.append((dist, ref_point, ref_d, test_point, test_d))

        temp.sort(key=lambda x: x[0])
        # print(temp[0][0], temp[1][0], temp[0][0]/temp[1][0])
        ratio = temp[0][0]/temp[1][0]
        if ratio < threshold:
            threshold_points.append((ratio, temp[0][3]))
            original_points.append((ratio, temp[0][1]))
    # print(threshold_points)
    return threshold_points, original_points


def plot_matches(img, points):
    plt.imshow(img)
    plt.title('match img')

    colors = {0: ('red', 'yellow'),
              1: ('blue', 'cyan'),
              2: ('orange', 'white')}
    count = 0
    for point in points:
        x, y, size = point[1].pt[0], point[1].pt[1], point[1].size
        circ = patches.Circle((x, y), size, color=colors[count][0], fill=False)
        plt.gca().add_patch(circ)

        circ = patches.Circle((x, y), 1, color=colors[count][1], fill=True)
        plt.gca().add_patch(circ)
        count += 1

    plt.show()


def affine_transformation(ref_points, test_points):
    P = []
    P_prime = []
    for p in range(len(ref_points)):
        x, y = ref_points[p][1].pt[0], ref_points[p][1].pt[1]

        P.append([x, y, 0, 0, 1, 0])
        P.append([0, 0, x, y, 0, 1])

        t_x, t_y = test_points[p][1].pt[0], test_points[p][1].pt[1]
        P_prime.append(t_x)
        P_prime.append(t_y)

    a = np.linalg.lstsq(np.array(P), np.array(P_prime), rcond=None)
    return np.array([a[0]])


def plot_affine(img, affine, ref_img):
    ref_img_matrix = np.array([[0, len(ref_img[0]), len(ref_img[0]), 0],
                               [0, 0, len(ref_img), len(ref_img)],
                               [1, 1, 1, 1]])
    new_affine = np.array([[affine[0][0], affine[0][1], affine[0][4]],
                           [affine[0][2], affine[0][3], affine[0][5]]])
    points = np.matmul(new_affine, ref_img_matrix).astype(np.int32).T
    plt.imshow(img)
    plt.title('match img')
    coords = []
    for p in points:
        coords.append(p)
    coords.append(coords[0])
    xs, ys = zip(*coords)
    plt.plot(xs, ys, 'red')
    plt.show()


if __name__ == "__main__":
    ref_img = io.imread('reference.png')
    ref_img_bw = io.imread('reference.png', as_gray=True)

    test_img = io.imread('test.png')
    test_img_bw = io.imread('test.png', as_gray=True)
    # plt.imshow(ref_img_bw, cmap='gray')
    # plt.title('ref img')
    # plt.show()

    test_img2 = io.imread('test2.png')
    test_img_bw2 = io.imread('test2.png', as_gray=True)

    # extracted = feature_extraction(ref_img)
    # plot_features(ref_img)
    # plot_features(test_img)
    # plot_features(test_img2)

    new_match, original_match = matcher(ref_img, test_img, 0.8)
    new_match.sort(key=lambda x: x[0])
    original_match.sort(key=lambda x: x[0])

    new_match2, original_match2 = matcher(ref_img, test_img2, 0.8)
    new_match2.sort(key=lambda x: x[0])
    original_match2.sort(key=lambda x: x[0])

    # plot_matches(ref_img, original_match[:3])
    # plot_matches(test_img, new_match[:3])
    # plot_matches(ref_img, original_match2[:3])
    # plot_matches(test_img2, new_match2[:3])
    affine_points1 = affine_transformation(original_match[:3], new_match[:3])
    plot_affine(test_img, affine_points1, ref_img)

    affine_points2 = affine_transformation(original_match2[:3], new_match2[:3])
    plot_affine(test_img2, affine_points2, ref_img)



