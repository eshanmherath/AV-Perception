# Inspired by Self-Driving-Car Nano Degree from Udacity

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

image_path = os.path.join(os.getcwd(), "../../samples/roads/road_1.jpg")

if not os.path.exists(image_path):
    print("Image does not exist!")
    exit()

image = mpimg.imread(image_path)

ysize = image.shape[0]
xsize = image.shape[1]

# Note: always make a copy rather than simply using "="
color_select = np.copy(image)

red_threshold = 200
green_threshold = 200
blue_threshold = 200
rgb_threshold = [red_threshold, green_threshold, blue_threshold]

thresholds = (image[:, :, 0] < rgb_threshold[0]) \
             | (image[:, :, 1] < rgb_threshold[1]) \
             | (image[:, :, 2] < rgb_threshold[2])

# color_select, pixels that were above the threshold have been retained,
# and pixels below the threshold have been blacked out.
color_select[thresholds] = [0, 0, 0]

plt.imshow(color_select)
plt.show()
