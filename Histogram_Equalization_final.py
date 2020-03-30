import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

if __name__ == "__main__":

    image = cv2.imread("ekmek.jpg", 0)
    rows, cols = image.shape
    print("Image Dimensions: ",rows,cols)

    cdf = np.zeros(256,np.uint32) #cumulative distribution function
    equalized_image = np.zeros((rows,cols),np.uint8)

    for i in range(rows):
        for j in range(cols):
            cdf[image[i][j]] += 1 #calculate cdf
    #Calculate Histogram through Formula
    histogram = cdf / (rows * cols)
    histogram = histogram * 255
    for i in range(1,256):
        histogram[i] = histogram[i] + histogram[i-1]
    histogram = np.round(histogram)
    histogram = histogram.astype("uint8")
    #Calculate Equalized Histogram
    for i in range(256):
        a = np.where(image == i)
        a = np.asarray(a)
        for j in range(a.shape[1]):
            equalized_image[a[0][j]][a[1][j]] = histogram[i]
    plt.subplot(3, 1, 1)
    plt.title('Original Histogram')
    plt.axis([0, 256, 0, 8000])
    plt.hist(cdf, bins=256)
    plt.subplot(3, 1, 3)
    plt.title('New Equalized Histogram')
    plt.hist(histogram, bins=256)
    plt.show()
    cv2.imwrite("pout_new.png", equalized_image)
    cv2.imshow("original",image)
    cv2.imshow("new",equalized_image)
    cv2.waitKey()

