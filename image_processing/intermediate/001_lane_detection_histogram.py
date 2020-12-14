# Inspired by Self-Driving-Car Nano Degree from Udacity

# Note that following is done for an image which has already gone through Calibration, Threholding and
# Perspective Tranformation Steps

import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# Read in an image and grayscale it
image_path = os.path.join(os.getcwd(), "../../samples/roads/warped-road.jpg")

if not os.path.exists(image_path):
    print("Image does not exist!")
    exit()

image = mpimg.imread(image_path)


def hist(img):
    # Grab only the bottom half of the image
    # Lane lines are likely to be mostly vertical nearest to the car
    bottom_half = img[img.shape[0] // 2:, :]

    # Sum across image pixels vertically - make sure to set an `axis`
    # i.e. the highest areas of vertical lines should be larger values
    histogram = np.sum(bottom_half, axis=0)

    return histogram


# Create histogram of image binary activations
histogram = hist(image)

# Visualize the resulting histogram
plt.plot(histogram)
plt.show()
