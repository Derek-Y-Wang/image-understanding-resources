import cv2 as cv
import numpy as np
import math as m
#import matplotlib.pyplot as plt

IMG = 'lion.jpeg'

img = cv.imread(IMG)
img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
print(img)


filters = {
    'sharpness' : np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]),
    'blur' : np.array([[0.0625, 0.125, 0.0625], [0.125, 0.25, 0.125], [0.0625, 0.125, 0.0625]]),
    'prewitt' : np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]),
    'sobel' : np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
}

def gaussianFormula(i, j):
    SIGMA = 8
    PI = m.pi
    E = m.e
    return ((1 / (2 * PI * (SIGMA ** 2))) * (E ** ((-1/2) * ((i ** 2 + j ** 2) / SIGMA **2))))
    
def get_new_intensity(m1, m2):
    total = 0
    for i in range(len(m1)):
        total += np.dot(m1[i], m2[i]).sum()
    return total

def fuzzyFilter(square):
    pixelValue = 0
    
    for i in range(len(square)):
        for j in range(len(square)):
            pixelValue += np.multiply(square, gaussianFormula(i, j))
    return pixelValue

result = []
def applyFilter(filter):
    for i in range(1, len(img) - 1):
        new = []
        for j in range(1, len(img[0]) - 1):
            print(f'{i}/{len(img)}')
            top = [img[i - 1][j - 1], img[i - 1][j], img[i - 1][j + 1]]
            mid = [img[i][j - 1], img[i][j], img[i][j + 1]]
            bottom = [img[i + 1][j - 1], img[i + 1][j], img[i + 1][j + 1]]
            square = np.array([top, mid, bottom])

            # mapped_ops = np.dot(np.subtract(square, mask), mask)
            # print(mapped_ops.sum())
            new.append(get_new_intensity(square, filter))

        result.append(new)
  
def applyFuzzyFilter():
    for i in range(1, len(img) - 1):
        new = []
        for j in range(1, len(img[0]) - 1):
            print(f'{i}/{len(img)}')
            top = [img[i - 1][j - 1], img[i - 1][j], img[i - 1][j + 1]]
            mid = [img[i][j - 1], img[i][j], img[i][j + 1]]
            bottom = [img[i + 1][j - 1], img[i + 1][j], img[i + 1][j + 1]]
            square = np.array([top, mid, bottom])
            print(square)
            break

            # mapped_ops = np.dot(np.subtract(square, mask), mask)
            # print(mapped_ops.sum())
            new.append(fuzzyFilter(square))

        result.append(new)

def main():
    applyFilter(filters['sobel'])
    print(np.uint8(result))
    cv.imshow('test', np.array(np.uint8(result)))
    cv.imshow('original', img)
    cv.waitKey(0)
    cv.destroyAllWindows()
if __name__ == '__main__':
    main()