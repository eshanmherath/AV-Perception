# Inspired by Self-Driving-Car Nano Degree from Udacity

import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# Read in an image and grayscale it
image_path = os.path.join(os.getcwd(), "../../samples/roads/road_2.jpg")

if not os.path.exists(image_path):
    print("Image does not exist!")
    exit()

image = mpimg.imread(image_path)


def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    # Apply the following steps to img

    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)

    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output


# Run the function
gradient_binary_x = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)
gradient_binary_y = abs_sobel_thresh(image, orient='y', thresh_min=20, thresh_max=100)

# Plot the result
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
f.tight_layout()

ax1.imshow(image)
ax1.set_title('Original Image', fontsize=20)

ax2.imshow(gradient_binary_x, cmap='gray')
ax2.set_title('Sobel X (Vertical Edges)', fontsize=20)

ax3.imshow(gradient_binary_y, cmap='gray')
ax3.set_title('Sobel Y (Horizontal Edges)', fontsize=20)

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
