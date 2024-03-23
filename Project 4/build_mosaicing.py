import numpy as np
import cv2
from matplotlib import pyplot


# Convert the image to grayscale
def convert_grayscale(readImage):
    if len(readImage.shape) == 3:
        grayscale_image = np.dot(readImage[..., :3], [0.3, 0.6, 0.1])
    else:
        grayscale_image = readImage

    grayscale_image = grayscale_image.astype(np.uint8)
    return grayscale_image


# Implement the Gaussian filter on the Fourier Domain
def Gaussian_low_filter(img, D0):
    M, N = img.shape
    H = np.zeros((M, N), dtype=np.float32)
    for u in range(M):
        for v in range(N):
            D = np.sqrt((u - M / 2) ** 2 + (v - N / 2) ** 2)
            H[u, v] = np.exp(-D ** 2 / (2 * D0 * D0))
    return H


# Implement the phase correlation and use filter on it
def phase_correlation_filter(a, b):
    H_1 = Gaussian_low_filter(a, 50)
    H_2 = Gaussian_low_filter(b, 50)
    G_a = np.fft.fft2(a)
    G_b = np.fft.fft2(b)
    G_a = G_a * H_1
    G_b = G_b * H_2
    conj_b = np.ma.conjugate(G_b)
    R = G_a * conj_b
    R /= np.absolute(R)

    r = np.fft.ifft2(R).real
    return r


# With filter
def image_mosaicing(img1, img2):
    # compute phase correlation with filter
    corrimg = phase_correlation_filter(img1, img2)
    r, c = np.unravel_index(corrimg.argmax(), corrimg.shape)
    threshold = np.max(corrimg)

    # print(threshold)

    # Plot the images: original image with phase correlation point
    pyplot.imshow(img1, cmap='gray')
    pyplot.plot([c], [r], 'ro')
    pyplot.show()
    # The second image
    pyplot.imshow(img2, cmap='gray')
    pyplot.show()
    # The image of Fourier domain
    pyplot.figure(figsize=[8, 8])
    pyplot.imshow(corrimg, cmap='gray')
    pyplot.show()

    return corrimg, r, c, threshold


# Combine two images into one by using the Fourier Transformation
def conbime(gray_1, gray_2, r, c, size):
    # Get the height and width of each image
    img1_y, img1_x = gray_1.shape
    img2_y, img2_x = gray_2.shape

    # Find the peak and location
    y = size - r
    x = c

    # Find the center point of the canvas
    center_r = int(r / 2)
    center_c = int(c / 2)

    # If the point is located at the bottom of left
    if c > center_c and r > center_r:
        canvas_x = img1_x + img2_x - (img2_x - y)
        canvas_y = img1_y + img2_y - (img2_y - x)

        canvas = np.zeros((canvas_x, canvas_y))

        canvas[y:y + img1_y, :img1_x] = gray_1
        canvas[:img2_y, x:x + img2_x] = gray_2

        pyplot.imshow(canvas, cmap='gray')
        pyplot.show()

    # If the point is located at the top of left
    if c > center_c and r <= center_r:
        y = r

        # Compute the size of the empty canvas
        canvas_x = img1_x + img2_x - (img2_x - y)
        canvas_y = img1_y + img2_y - (img2_y - x)

        # Build the empty canvas
        canvas = np.zeros((canvas_x, canvas_y))

        # Paste the image on the canvas
        canvas[y:y + img2_y, x:x + img2_x] = gray_2
        canvas[:img1_y, :img1_x] = gray_1

        # Plot the image
        pyplot.imshow(canvas, cmap='gray')
        pyplot.show()


def get_peaks(imgs, general_size):
    peaks = []

    for index, img in enumerate(imgs):
        if index < 3:
            corrimg, r, c, t= image_mosaicing(imgs[index + 1], imgs[index + 2])
            # y = general_size - r
            # x = c
            a = [index + 2, r, c]
            peaks.append(a)

    return peaks


def build_mosaic(imgs, general_size):
    peaks = get_peaks(img_gray, general_size)
    final_image = imgs[0]

    for i in range(len(peaks)):

        image = imgs[peaks[i][0]]
        r = peaks[i][1]
        c = peaks[i][2]

        img1_y, img1_x = final_image.shape
        img2_y, img2_x = image.shape

        y = general_size - r
        x = c

        center_r = int(r / 2)
        center_c = int(c / 2)

        if c > center_c and r > center_r:
            canvas_x = img1_x + img2_x - (img2_x - y)
            canvas_y = img1_y + img2_y - (img2_y - x)

            canvas = np.zeros((canvas_x, canvas_y))

            canvas[y:y + img1_y, :img1_x] = final_image
            canvas[:img2_y, x:x + img2_x] = imgs[peaks[i][0]]

            final_image = canvas

            pyplot.imshow(canvas, cmap='gray')
            pyplot.show()

        if c > center_c and r <= center_r:
            y = r

            canvas_x = img1_x + img2_x - (img2_x - y)
            canvas_y = img1_y + img2_y - (img2_y - x)

            canvas = np.zeros((canvas_x, canvas_y))

            canvas[y:y + img2_y, x:x + img2_x] = imgs[peaks[i][0]]
            canvas[:img1_y, :img1_x] = final_image

            final_image = canvas

            pyplot.imshow(canvas, cmap='gray')
            pyplot.show()

    return canvas


# Set tge grayscale weights
weights = [0.3, 0.6, 0.1]

# Read images
image_0 = './images/1.jpg'
image_1 = './images/2.jpg'
image_2 = './images/3.jpg'
image_3 = './images/4.jpg'
image_4 = './images/5.jpg'
image_5 = './images/6.jpg'
img_0 = cv2.imread(image_0)
img_1 = cv2.imread(image_1)
img_2 = cv2.imread(image_2)
img_3 = cv2.imread(image_3)
img_4 = cv2.imread(image_4)
img_5 = cv2.imread(image_5)

# Convert images to grayscale
gray_0 = convert_grayscale(img_0)
gray_1 = convert_grayscale(img_1)
gray_2 = convert_grayscale(img_2)
gray_3 = convert_grayscale(img_3)
gray_4 = convert_grayscale(img_4)
gray_5 = convert_grayscale(img_5)

# Compute the phase correlation of two images
corrimg, r, c, threshold = image_mosaicing(gray_0, gray_5)

# Set threshold for the non-overlapping images
if threshold < 0.04:
    conbime(gray_0, gray_5, r, c, 183)
else:
    print("The two images don't have overlapping area.")

# Combine the images to one
img_gray = [gray_0, gray_1, gray_2, gray_3, gray_4, gray_5]
final_image = build_mosaic(img_gray, len(img_gray[0]))
