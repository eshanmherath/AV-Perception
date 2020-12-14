# Inspired by Self-Driving-Car Nano Degree from Udacity

# Notice that S channel is still doing a fairly robust job of
# picking up the lines under very different color and contrast conditions

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

thresh = (180, 255)
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
binary = np.zeros_like(gray)
binary[(gray > thresh[0]) & (gray <= thresh[1])] = 1

# Plot the result
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(image)
ax1.set_title('Original Image', fontsize=20)

ax2.imshow(binary, cmap='gray')
ax2.set_title('Gray Binary', fontsize=20)

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
H = hls[:, :, 0]
L = hls[:, :, 1]
S = hls[:, :, 2]

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
f.tight_layout()

ax1.imshow(H, cmap='gray')
ax1.set_title('Hue Channel', fontsize=20)

ax2.imshow(L, cmap='gray')
ax2.set_title('Lightness Channel', fontsize=20)

ax3.imshow(S, cmap='gray')
ax3.set_title('Saturation Channel', fontsize=20)

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

# Using S Channel

thresh = (90, 255)
binary = np.zeros_like(S)
binary[(S > thresh[0]) & (S <= thresh[1])] = 1

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(image)
ax1.set_title('Original Image', fontsize=20)

ax2.imshow(binary, cmap='gray')
ax2.set_title('Saturation Binary', fontsize=20)

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

# Using H Channel

thresh = (15, 100)
binary = np.zeros_like(H)
binary[(H > thresh[0]) & (H <= thresh[1])] = 1

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()

ax1.imshow(image)
ax1.set_title('Original Image', fontsize=20)

ax2.imshow(binary, cmap='gray')
ax2.set_title('Hue Binary', fontsize=20)

plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()

"""
It's worth noting, however, that the R channel still does rather well on the white lines, 
perhaps even better than the S channel. 
As with gradients, it's worth considering how you might combine various color thresholds 
to make the most robust identification of the lines.
"""
