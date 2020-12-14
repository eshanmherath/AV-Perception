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


def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    s_channel = hls[:, :, 2]
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the derivative in x
    abs_sobelx = np.absolute(sobelx)  # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    # Stack each channel
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    return color_binary


result = pipeline(image)

# Plot the result
f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
f.tight_layout()

ax1.imshow(image)
ax1.set_title('Original Image', fontsize=20)

ax2.imshow(result)
ax2.set_title('Pipeline Result', fontsize=20)

result_gray_temp = cv2.cvtColor(result, cv2.COLOR_HLS2RGB)
result_gray = cv2.cvtColor(result_gray_temp, cv2.COLOR_RGB2GRAY)
ax3.imshow(result_gray, cmap='gray')
ax3.set_title('Pipeline Result (Gray)', fontsize=20)

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
