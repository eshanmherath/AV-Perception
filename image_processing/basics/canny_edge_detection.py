# Inspired by Self-Driving-Car Nano Degree from Udacity

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

image_path = os.path.join(os.getcwd(), "../../samples/roads/road_1.jpg")

if not os.path.exists(image_path):
    print("Image does not exist!")
    exit()

image = mpimg.imread(image_path)

plt.imshow(image)
plt.show()

gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # grayscale conversion
plt.imshow(gray, cmap='gray')
plt.show()

"""
We'll also include Gaussian smoothing, before running Canny, 
which is essentially a way of suppressing noise and spurious gradients by averaging
kernel_size for Gaussian smoothing to be any odd number. A larger kernel_size implies averaging, 
or smoothing, over a larger area
"""
kernel_size = 3
blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size), 0)
plt.imshow(blur_gray, cmap='gray')
plt.show()

"""
The algorithm will first detect strong edge (strong gradient) pixels above the high_threshold, 
and reject pixels below the low_threshold. Next, pixels with values between the low_threshold 
and high_threshold will be included as long as they are connected to strong edges. 
The output edges is a binary image with white pixels tracing out the detected edges and black everywhere else. 

As far as a ratio of low_threshold to high_threshold, John Canny himself recommended a low 
to high ratio of 1:2 or 1:3.

cv2.Canny() applies a 5x5 Gaussian internally

"""

low_threshold = 50
high_threshold = 150
edges = cv2.Canny(gray, low_threshold, high_threshold)

# Display the image
plt.imshow(edges, cmap='Greys_r')
plt.show()
