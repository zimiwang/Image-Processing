# Import important libraries
import numpy as np
import cv2 as cv
import os
import math
from skimage.io import imread, imshow
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from skimage import io
import skimage.color
import skimage.filters
import skimage.measure
from skimage.segmentation import flood, flood_fill
from skimage import data, exposure, img_as_float

weights = [0.3, 0.6, 0.1]

image_0 = './images/xray.png'
image_1 = './images/airplane.jpg'
image_2 = './images/church.tif'
image_3 = './images/shapes_noise.tif'
image_4 = './images/portal.tif'
image_5 = './images/chang.tif'
image_6 = './images/crowd.tif'
image_7 = './images/turkeys.tif'


# Part 1 - Create a Histogram
def build_histogram(readImage, bins):
    # Read the image and save it as numpy array
    image = cv.imread(readImage, 0)
    # Convert the image to grayscale
    if len(image.shape) == 3:
        grayscale_image = np.dot(image[..., :3], [0.3, 0.6, 0.1])
    else:
        grayscale_image = image

    # Get image dimensions
    row, col = grayscale_image.shape

    # Create empty numpy array
    hist = np.zeros([256])
    # Go through every pixel
    for x in range(row):
        for y in range(col):
            # When passing a pixel, the hist[3] increases by 1 if the intensity is 3 (example)
            hist[grayscale_image[x, y]] = hist[grayscale_image[x, y]] + 1

    # Get the intensity number and the number of pixels
    intensity_num = np.arange(0, 256)
    pixels_num = np.array(np.zeros(256), np.int32)
    for i in range(len(intensity_num)):
        pixels_num[i] = hist[i]

    # Display the grayscale image
    fig, ax = plt.subplots(1, 1)
    ax.imshow(grayscale_image, cmap='gray')

    # Display the histogram of the grayscale
    plt.figure()
    plt.title("Grayscale Image Histogram")
    plt.xlabel("bin values")
    plt.ylabel("bin count")
    plt.xlim([0, 256])
    plt.bar(intensity_num, pixels_num, width=256 / bins, align='edge')
    plt.show()

    # Return the 2D-array, first column is the center (mean) of each bin in range and second is the number of pixels with the intensity in a range
    avg_intensity = []
    pixels_num_range = []
    global int_sum, pix_sum
    global count_intensity, count_pix_num
    count_intensity = 0
    count_pix_num = 0

    for x in range(int(256 / bins)):
        int_sum = 0
        for y in range(int(bins)):
            int_sum = int_sum + hist[count_intensity]
            count_intensity = count_intensity + 1

        avg_intensity.append(int_sum / bins)

    for z in range(int(256 / bins)):
        pix_sum = 0
        for k in range(bins):
            pix_sum = pix_sum + pixels_num[count_pix_num]
            count_pix_num = count_pix_num + 1

        pixels_num_range.append(pix_sum)

    hist_array = np.array([avg_intensity, pixels_num_range])
    return hist_array


# For part 2
def plot_hist(readImage, bins):
    # Read the image and save it as numpy array
    image = cv.imread(readImage, 0)
    # Convert the image to grayscale
    if len(image.shape) == 3:
        grayscale_image = np.dot(image[..., :3], [0.3, 0.6, 0.1])
    else:
        grayscale_image = image

    # Get image dimensions
    row, col = grayscale_image.shape

    # Create empty numpy array
    hist = np.zeros([256])
    # Go through every pixel
    for x in range(row):
        for y in range(col):
            # When passing a pixel, the hist[3] increases by 1 if the intensity is 3 (example)
            hist[grayscale_image[x, y]] = hist[grayscale_image[x, y]] + 1

    # Get the intensity number and the number of pixels
    intensity_num = np.arange(0, 256)
    pixels_num = np.array(np.zeros(256), np.int32)
    for i in range(len(intensity_num)):
        pixels_num[i] = hist[i]

    # Display the histogram of the grayscale
    plt.figure()
    plt.title("Grayscale Image Histogram")
    plt.xlabel("bin values")
    plt.ylabel("bin count")
    plt.xlim([0, 256])
    plt.bar(intensity_num, pixels_num, width=256 / bins, align='edge')
    plt.show()


# For part 2
def display_hist_threshold(new_grayscale):
    b, bins, patches = plt.hist(new_grayscale, 255)
    plt.xlim([0, 255])
    plt.title('After double-sides threshold')
    plt.show()


# Part 2 - double-sides threshold
def regions_components(readImage, T_low, T_high):
    # Read the image and save it as numpy array
    image = cv.imread(readImage, 0)
    # image = image[..., ::-1]
    fig, ax = plt.subplots(1, 2)

    # Convert the image to grayscale
    if len(image.shape) == 3:
        grayscale_image = np.dot(image[..., :3], [0.3, 0.6, 0.1])
    else:
        grayscale_image = image

    # # Copy the original image
    # old_grayscale = grayscale_image

    # Display the grayscale
    ax[0].imshow(grayscale_image, cmap="gray")

    # Perform double-sides threshold
    row, col = grayscale_image.shape
    for i in range(row):
        for j in range(col):
            if grayscale_image[i, j] > T_high:
                grayscale_image[i, j] = 0
            elif grayscale_image[i, j] < T_low:
                grayscale_image[i, j] = 0
            else:
                grayscale_image[i, j] = 255

    # Display the grayscale after thresholding
    ax[1].imshow(grayscale_image, cmap="gray")

    display_hist_threshold(grayscale_image)

    return grayscale_image


# Part 2 - Connected Components
def connected_components(image, T_low, T_high):
    new_grayscale = regions_components(image, T_low, T_high)

    # Perform measure module from skimage
    labeled_image, count = skimage.measure.label(new_grayscale, connectivity=2, return_num=True)

    # Convert the labeled image to colored image
    colored_label_image = skimage.color.label2rgb(labeled_image, bg_label=0)

    # Display the colored image
    fig, ax = plt.subplots(1, 1)
    plt.imshow(colored_label_image)

    return labeled_image, count


# Part 2 - flood fill
def flood_fill_algo(image_name, x, y, replace_color, tolerance):
    # Read the image
    image = cv.imread(image_name, 0)

    # Run the flood fill algorithm from skimage
    # Second parameter is the start pixel
    # Third parameter is the color for replacing
    # Forth parameter is adjacent values
    filled_image = flood_fill(image, (x, y), replace_color, tolerance=tolerance)

    fig, ax = plt.subplots(ncols=2, figsize=(10, 5))
    # Display the original image
    ax[0].imshow(image, cmap=plt.cm.gray)
    ax[0].set_title('Original')

    # Display the second image
    ax[1].imshow(filled_image, cmap=plt.cm.gray)
    ax[1].plot(76, 76, 'ro')  # seed point
    ax[1].set_title('flood fill')

    plt.show()


# Plot the image with its histogram for part 3
def plot_img_and_hist(image, axes, bins):
    # Convert the image to float datatype
    image = img_as_float(image)
    ax_img, ax_hist = axes
    # Create a twin axes
    ax_cdf = ax_hist.twinx()

    # Display image
    ax_img.imshow(image, cmap=plt.cm.gray)
    # Display histogram
    ax_hist.hist(image.ravel(), bins, histtype='step', color='black')
    ax_hist.set_xlim(0, 1)

    # Display cumulative distribution
    img_cdf, bins = exposure.cumulative_distribution(image, bins)
    ax_cdf.plot(bins, img_cdf, 'r')

    ax = [ax_img, ax_hist, ax_cdf]
    return ax


# Part 3 - Histogram equalization
def histogram_equalization(image):
    image = cv.imread(image, 0)

    # Adaptive Equalization
    img_adap_equal = exposure.equalize_adapthist(image, clip_limit=0.03)

    # Display results
    fig = plt.figure(figsize=(8, 5))
    axes = np.zeros((2, 2), dtype=object)
    axes[0, 0] = fig.add_subplot(2, 2, 1)
    for i in range(1, 2):
        axes[0, i] = fig.add_subplot(2, 2, 1 + i, sharex=axes[0, 0], sharey=axes[0, 0])
    for i in range(0, 2):
        axes[1, i] = fig.add_subplot(2, 2, 3 + i)

    # Display the original image with its histogram
    ax = plot_img_and_hist(image, axes[:, 0], 256)
    ax[0].set_title('Original image')

    ax[1].set_ylabel('Number of pixels')

    # Display the adaptive equalization image with its histogram
    ax = plot_img_and_hist(img_adap_equal, axes[:, 1], 256)
    ax[0].set_title('Adaptive equalization')

    plt.show()


# Run the build_histogram function - part 1
hist_array = build_histogram(image_0, bins=64)

# Run the double sides threshold function - part 2
new_grayscale = regions_components(image_3,25,90)

# Display the histogram and its new image after thresholding - part 2
plot_hist(image_3, bins=64)
# display_hist_threshold(new_grayscale)

# Run Part connected components - part 2
# Attention: the connected_components() cannot run independently. You must run the regions_components() first with same parameters
connected_components(image_3, 25, 90)

# Run the flood fill algorithm - part 2
flood_fill_algo(image_5, 76, 76, 255, 15)

# Run the adaptive equalization algorithm  - part 3
histogram_equalization(image_0)
