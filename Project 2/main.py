# Ziming Wang

import numpy as np
import numpy.matlib
from skimage.io import imread, imshow
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import random
from skimage.util import random_noise
from sklearn.metrics import mean_squared_error
import scipy

weights = [0.3, 0.6, 0.1]

image_0 = './images/panda.jpg'
image_1 = './images/houndog1.png'
image_2 = './images/pika.jpg'
image_3 = './images/sun.jpg'
image_4 = './images/person.png'


# Part 1 - Four types noise added to images
def convert_grayscale(readImage):
    if len(readImage.shape) == 3:
        grayscale_image = np.dot(readImage[..., :3], [0.3, 0.6, 0.1])
    else:
        grayscale_image = readImage

    grayscale_image = grayscale_image.astype(np.uint8)
    return grayscale_image


def gaussian_noise(image, m, v):
    img = cv2.imread(image)
    # image = convert_grayscale(image)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    noise = random_noise(img, mode="gaussian", mean=m, var=v)
    noise = np.array(255 * noise, dtype='uint8')

    return noise


def salt_pepper_noise(image, a):
    img = cv2.imread(image)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # This function returns a floating-point image on the range [0, 1]
    noise = random_noise(img, mode='s&p', amount=a)

    # Changed it to 'uint8' and from [0,255]
    noise = np.array(255 * noise, dtype='uint8')

    return noise


def speckle_noise(image, v):
    img = cv2.imread(image)

    noise = random_noise(img, mode='speckle', var=v, clip=True)
    noise = np.array(255 * noise, dtype='uint8')
    return noise


def poisson_noise(image, s):
    img = cv2.imread(image)

    noise = random_noise(img, mode='poisson', seed=s, clip=True)
    noise = np.array(255 * noise, dtype='uint8')

    return noise


# Par2 - Linear Filter
def box_filter(readImage, r):

    noise_img = convert_grayscale(readImage)
    copy_img = noise_img.copy()
    image = np.zeros(noise_img.shape)
    row = noise_img.shape[0]
    col = noise_img.shape[1]

    # Process the row
    # Get the sum of the row
    noise_img = np.cumsum(noise_img, 0)
    image[:r + 1, :] = noise_img[r:2 * r + 1, :]
    image[r + 1:row - r, :] = noise_img[2 * r + 1:, :] - noise_img[:row - 2 * r - 1, :]
    image[row - r:, :] = np.matlib.repmat(noise_img[row - 1, :], r, 1) - noise_img[row - 2 * r - 1:row - r - 1, :]

    # Process the column
    # Get the sum of the column
    noise_img = np.cumsum(image, 1)
    image[:, :r + 1] = noise_img[:, r:2 * r + 1]
    image[:, r + 1:col - r] = noise_img[:, 2 * r + 1:] - noise_img[:, :col - 2 * r - 1]
    image[:, col - r:] = np.matlib.repmat(noise_img[:, col - 1].reshape(-1, 1), 1, r) - noise_img[:,
                                                                                        col - 2 * r - 1:col - r - 1]
    image = np.uint8(image*255/np.max(image))

    return copy_img,image


def gaussian_filter(noise_image, i, j):
    denoise_img = cv2.GaussianBlur(noise_image, (i, j), cv2.BORDER_REFLECT_101)

    return denoise_img


def laplacian_filter(noise_image, gray, ddepth, kernel_size):
    denoise_img = cv2.Laplacian(noise_image, ddepth, ksize=kernel_size)

    return denoise_img


# Part 3 - Nonlinear Filter
def median_filter(noise_image, val):
    median = cv2.medianBlur(noise_image, val)

    return median


def bilateral_filter(noise_image, d, sigmaColor, sigmaSpace):
    bilateral = cv2.bilateralFilter(noise_image, d, sigmaColor, sigmaSpace)

    return bilateral


def nonlocal_mean_filter(noise_image):
    if len(noise_image) == 1:
        out = cv2.fastNlMeansDenoising(noise_image, None, 10, 10, 7, 21)
    else:
        out = cv2.fastNlMeansDenoisingColored(noise_image, None, 10, 10, 7, 21)

    return out


def mse(img1, img2):
    MSE = np.mean((img1 - img2) ** 2)
    return MSE

def MSE(img1, img2):
      squared_diff = (img1 -img2) ** 2
      summed = np.sum(squared_diff)

      #img1 and 2 should have same shape
      num_pix = img1.shape[0] * img1.shape[1]
      err = summed / num_pix
      return err

# Part 1
noise_img = gaussian_noise(image_0, 0, 0.05)
# noise_img = salt_pepper_noise(image_4, 0.02)
# noise_img = speckle_noise(image_4, 0.09)
# noise_img = poisson_noise(image_4, 42)

# Display the noise image
cv2.imshow('Noise Image', noise_img)
cv2.waitKey(0)

# Part 2
# noise_img, filtered_img = box_filter(noise_img, 3)           # Box Filter
filtered_img = gaussian_filter(noise_img, 5, 5)   # Gaussian Filter
# filtered_img = laplacian_filter(noise_img, None, -1, 5) # Laplacian Filter

cv2.imshow('Linear Filtering', filtered_img)
cv2.waitKey(0)

print(MSE(noise_img, filtered_img))

# Part 3
# filtered_img = median_filter(noise_img, 5)  # Median filter
# filtered_img = bilateral_filter(noise_img, 15, 75, 75)      # Bilateral filter
filtered_img = nonlocal_mean_filter(noise_img)  # Non-Local Mean Filter

cv2.imshow('Non-linear Filtering', filtered_img)
cv2.waitKey(0)
print(mse(noise_img, filtered_img))


# noise_choice = input("Please input noise function number: (1-gaussian, 2-salt pepper, 3-speckle, 4-poisson)")
#
# if noise_choice == 1:
#
#     noise_img = gaussian_noise(image_2, 0, 0.05)
#
#     cv2.imshow('Gaussian Noise Image', noise_img)
#     cv2.waitKey(0)
#
#     denoise_choice = input("Please input noise function number: (1-gaussian, 2-salt pepper, 3-speckle, 4-poisson)")
#
# elif noise_choice == 2:
#     noise_img = salt_pepper_noise(image_2, 0.02)
#
#     cv2.imshow('Salt Pepper Noise Image', noise_img)
#     cv2.waitKey(0)
# elif noise_choice == 3:
#     noise_img = speckle_noise(image_2, 0.09)
#
#     cv2.imshow('Speckle Noise Image', noise_img)
#     cv2.waitKey(0)
# elif noise_choice == 4:
#     noise_img = poisson_noise(image_2, 42)
#
#     cv2.imshow('Poisson Noise Image', noise_img)
#     cv2.waitKey(0)
# else:
#     print("Invalid input, please select correct number")
