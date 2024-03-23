# Ziming Wang
# CS 6640 Image Processing
# Prof.Tolga Tasdizen
# 9/2/2022

# Import some important Libraries
import numpy as np
from skimage.io import imread, imshow
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# Read the image from the image folder
image = mpimg.imread('./images/airplane.jpg')

# Save the weight for image processing
weights = [0.3, 0.6, 0.1]


# The first approach to convert the image to grayscale
def grayImage_1():
    # Getting red, green, blue channels from the imported image
    x, y, z = image[:, :, 0], image[:, :, 1], image[:, :, 2]

    # Convert the color images to grayscale
    grayscale_image = x * weights[0] + y * weights[1] + z * weights[2]

    # Display the images in a grid
    fig, ax = plt.subplots(1, 1)
    ax.imshow(grayscale_image, cmap='gray')

    # Show the image
    fig.show()
    plt.show()

    # Save the image grid as a new file
    mpimg.imsave("new_grayImage.jpg", grayscale_image, cmap='gray')

# The second approach to convert the image to grayscale
def grayImage_2():
    # Convert the color images to grayscale by using the dot product function from 1.c
    grayscale_image = np.dot(image, weights)

    # Display the images in a grid
    fig, ax = plt.subplots(1, 1)
    ax.imshow(grayscale_image, cmap='gray')

    # Show the image
    fig.show()
    plt.show()
    # Save the image grid as a new file
    mpimg.imsave("new_grayImage.jpg", grayscale_image, cmap='gray')

# Main function
if __name__ == "__main__":
    grayImage_1()
    #grayImage_2()
