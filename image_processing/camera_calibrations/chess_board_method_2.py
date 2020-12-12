# Inspired by Self-Driving-Car Nano Degree from Udacity

"""
There are two main steps to this process: use chessboard images to obtain image points and object points, and then
use the OpenCV functions cv2.calibrateCamera() and cv2.undistort() to compute the calibration and un-distortion.
"""

import glob
import os
import pickle

import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = os.path.join(os.getcwd(), "../../samples/chess_boards/distorted_images/GO*.jpg")

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
real_object_points = np.zeros((6 * 8, 3), np.float32)
real_object_points[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
object_points = []  # 3d points in real world space
image_points = []  # 2d points in image plane.

# Make a list of calibration images
print("Reading training images..")
images = glob.glob(image_path)

# Step through the list and search for chessboard corners
for idx, image_name in enumerate(images):
    test_image = cv2.imread(image_name)
    gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (8, 6), None)

    # If found, add object points, image points
    if ret:
        object_points.append(real_object_points)
        image_points.append(corners)

        # Draw and display the corners

        # cv2.drawChessboardCorners(test_image, (8, 6), corners, ret)
        # cv2.imshow('img', test_image)
        # cv2.waitKey(500)

cv2.destroyAllWindows()

# Calibration

print("Reading test image..")
test_image_path = os.path.join(os.getcwd(), "../../samples/chess_boards/distorted_images/test_image.jpg")
test_image = cv2.imread(test_image_path)
image_size = (test_image.shape[1], test_image.shape[0])

print("Camera calibration..")
# Do camera calibration given object points and image points
ret, camera_matrix, distortion_coefficients, rotation_vector, translation_vector \
    = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

print("Un-distorting the test image..")
undistorted_image = cv2.undistort(test_image, camera_matrix, distortion_coefficients, None, camera_matrix)

print("Saving Camera calibration values..")
dist_pickle = {"mtx": camera_matrix, "dist": distortion_coefficients}  # You can save other values also
pickle.dump(dist_pickle, open("camera_calibrations.pickle", "wb"))

# Visualize un-distortion

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(test_image)
ax1.set_title('Original Image', fontsize=30)
ax2.imshow(undistorted_image)
ax2.set_title('Undistorted Image', fontsize=30)
plt.show()
print("Calibration Complete.")
