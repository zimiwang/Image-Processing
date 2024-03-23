import csv
import os
import random
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.io import imread, imshow
import matplotlib.image as mpimg
import scipy
from matplotlib import pyplot


# Implement the Gaussian filter on the fourier domain
def Gaussian_low_filter(img, D0):
    M, N = img.shape
    H = np.zeros((M, N), dtype=np.float32)
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            H[u, v] = np.exp(-D ** 2 / (2 * D0 * D0))
    return H

# Set tge grayscale weights
weights = [0.3, 0.6, 0.1]
# Read images
image_0 = './images/1.jpg'
image_1 = './images/2.jpg'
image_2 = './images/3.jpg'
image_3 = './images/4.jpg'
image_4 = './images/5.jpg'
image_5 = './images/6.jpg'

# Convert images to grayscale
img_0 = cv2.imread(image_0)
img_1 = cv2.imread(image_1)
img_2 = cv2.imread(image_2)
img_3 = cv2.imread(image_3)
img_4 = cv2.imread(image_4)
img_5 = cv2.imread(image_5)


def convert_grayscale(readImage):
    if len(readImage.shape) == 3:
        grayscale_image = np.dot(readImage[..., :3], [0.3, 0.6, 0.1])
    else:
        grayscale_image = readImage

    grayscale_image = grayscale_image.astype(np.uint8)
    return grayscale_image


gray_0 = convert_grayscale(img_0)
gray_1 = convert_grayscale(img_1)
gray_2 = convert_grayscale(img_2)
gray_3 = convert_grayscale(img_3)
gray_4 = convert_grayscale(img_4)
gray_5 = convert_grayscale(img_5)

# Plot original images
plt.imshow(gray_3, cmap='gray')
plt.axis('off')
plt.show()

# Convert image in Fourier domain
F = np.fft.fft2(gray_3)
Fshift = np.fft.fftshift(F)

# Plot the image after Fourier transformation
plt.imshow(np.log1p(np.abs(Fshift)), cmap='gray')
plt.axis('off')
plt.show()

# Apply Gaussian filter
H = Gaussian_low_filter(gray_3, 50)

# Plot filtered image
plt.imshow(H, cmap='gray')
plt.axis('off')
plt.show()

# Applied Ideal Low Pass Filter
Gshift = Fshift * H

# Plot images
plt.imshow(np.log1p(np.abs(Gshift)), cmap='gray')
plt.axis('off')
plt.show()

# Inverse Fourier Transform to general image
G = np.fft.ifftshift(Gshift)

# Plot images
plt.imshow(np.log1p(np.abs(G)), cmap='gray')
plt.axis('off')
plt.show()

g = np.abs(np.fft.ifft2(G))

plt.imshow(g, cmap='gray')
plt.axis('off')
plt.show()
