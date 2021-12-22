import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

IMG = 'flower.jpg'

img = cv.imread(IMG)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

mask = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
# mask = np.array([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]])

def get_new_intensity(m1, m2):
    total = 0
    for i in range(len(m1)):
        total += np.dot(m1[i], m2[i]).sum()
    return total


result = []
for i in range(1, len(img) - 1):
    new = []
    for j in range(1, len(img[0]) - 1):
        print(f'{i}/{len(img)})')
        top = [img[i - 1][j - 1], img[i - 1][j], img[i - 1][j + 1]]
        mid = [img[i][j - 1], img[i][j], img[i][j + 1]]
        bottom = [img[i + 1][j - 1], img[i + 1][j], img[i + 1][j + 1]]
        square = np.array([top, mid, bottom])

        # mapped_ops = np.dot(np.subtract(square, mask), mask)
        # print(mapped_ops.sum())
        new.append(get_new_intensity(square, mask))

    result.append(new)

print(np.uint8(result))
cv.imshow('test', np.array(np.uint8(result)))
cv.imshow('original', img)
cv.waitKey(0)
cv.destroyAllWindows()
