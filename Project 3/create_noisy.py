import csv
import os
import random
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy.matlib
from skimage.io import imread, imshow
import matplotlib.image as mpimg
from skimage.util import random_noise
from sklearn.metrics import mean_squared_error
import scipy


def write_csv(csv_file_name, training_dict_list):
    csv_columns = ["RefImageName", "NoiseType", "NoisyImage"]
    try:
        with open(csv_file_name, "w") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
            for data in training_dict_list:
                writer.writerow(data)
    except IOError:
        print("I/O error")


# Part 1 - Four types noise added to images
def convert_grayscale(readImage):
    if len(readImage.shape) == 3:
        grayscale_image = np.dot(readImage[..., :3], [0.3, 0.6, 0.1])
    else:
        grayscale_image = readImage

    grayscale_image = grayscale_image.astype(np.uint8)
    return grayscale_image


def gaussian_noise(img, m, v):
    #img = cv2.imread(image)
    # image = convert_grayscale(image)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    noise = random_noise(img, mode="gaussian", mean=m, var=v)
    noise = np.array(255 * noise, dtype='uint8')

    return noise


def salt_pepper_noise(img, a):
    #img = cv2.imread(image)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # This function returns a floating-point image on the range [0, 1]
    noise = random_noise(img, mode='s&p', amount=a)

    # Changed it to 'uint8' and from [0,255]
    noise = np.array(255 * noise, dtype='uint8')

    return noise


def speckle_noise(img, v):
    #img = cv2.imread(image)

    noise = random_noise(img, mode='speckle', var=v, clip=True)
    noise = np.array(255 * noise, dtype='uint8')
    return noise


def poisson_noise(img, s):
    #img = cv2.imread(image)

    noise = random_noise(img, mode='poisson', seed=s, clip=True)
    noise = np.array(255 * noise, dtype='uint8')

    return noise


def add_noise(image, noise_indicator):
    """
    image is of size m x n with values in the range (0,1).
    noise_indicator is type of noise that needs to be added to the image
    noise_indicator == 0 indicates an addition of Gaussian noise with mean 0 and var 0.08
    noise_indicator == 1 indicates an addition of salt and pepper noise with intensity variation of 0.08
    noise_indicator == 2 indicates an addition of Poisson noise
    noise_indicator == 3 indicates an addition of speckle noise of mean 0 and var 0.05

    This function should return a noisy version of the input image
    """
    noisy = None

    if noise_indicator == 0:
        print("Adding gaussian noise")
        noisy = gaussian_noise(image, 0, 0.08)
    elif noise_indicator == 1:
        print("Adding salt&pepper noise")
        noisy = salt_pepper_noise(image, 0.08)
    elif noise_indicator == 2:
        print("Adding poisson noise")
        noisy = poisson_noise(image, 42)
    elif noise_indicator == 3:
        print("Adding speckle noise")
        noisy = speckle_noise(image, 0.05)
    else:
        print("Wrong input, default add gaussian noise to the image")
        noisy = gaussian_noise(image, 0, 0.08)

    # TODO: implement add_noise
    # raise NotImplementedError

    return noisy


def main(directory, train, num_of_samples, noise_indicator_low, noise_indicator_high):
    """
    Main driver function for noise generator
    """
    if train == 1:
        name_csv = pd.read_csv(directory + "file_name_train.csv")
        csv_file_name = directory + "../training.csv"
        directory_name = directory + "../training/"
        training_dict_list = [dict() for x in range(num_of_samples)]
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)
    else:
        name_csv = pd.read_csv(directory + "file_name_test.csv")
        csv_file_name = directory + "../testing.csv"
        directory_name = directory + "../testing/"
        training_dict_list = [dict() for x in range(num_of_samples)]
        if not os.path.exists(directory_name):
            os.makedirs(directory_name)

    for i in range(num_of_samples):
        dataset_idx = random.randint(
            0, len(name_csv) - 1
        )  # Choose an image randomly from the dataset
        # Read the image from a path
        img_name = os.path.join(directory, name_csv.iloc[dataset_idx, 0])
        image = io.imread(img_name)

        # Normalize the image to range (0,1)
        max_pixel_value = np.max(image)
        image = image / max_pixel_value

        # Choosing the noise randomly
        noise_type = random.randint(noise_indicator_low, noise_indicator_high)
        noisy_image = add_noise(image, noise_type)
        training_dict_list[i] = {
            "RefImageName": img_name,
            "NoiseType": noise_type,
            "NoisyImage": str(i) + ".png",
        }
        io.imsave(directory_name + str(i) + ".png", noisy_image)
    write_csv(csv_file_name, training_dict_list)


if __name__ == "__main__":
    main("./nn_data/cats/raw/", 1, 800, 0, 3)  ## creating 800 Samples of Training Data
    main("./nn_data/cats/raw/", 0, 400, 0, 3)  ## creating 400 Samples of Testing Data
    main(
        "./nn_data/pokemon/raw/", 1, 800, 0, 3
    )  ## creating 800 Samples of Training Data
    main(
        "./nn_data/pokemon/raw/", 0, 400, 0, 3
    )  ## creating 400 Samples of Testing Data
