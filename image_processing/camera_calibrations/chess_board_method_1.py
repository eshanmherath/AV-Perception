# Inspired by Self-Driving-Car Nano Degree from Udacity

import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

image_path = os.path.join(os.getcwd(), "../../samples/chess_boards/chess_board_0.png")
# image_path = os.path.join(os.getcwd(), "../../samples/chess_boards/training_data/image1.tif")

if not os.path.exists(image_path):
    print("Image does not exist!")
    exit()

img = cv2.imread(image_path)
plt.imshow(img)
plt.show()

object_points = []  # 3D points in real world space
image_points = []  # 2D points in image plane

# Known objects coordinates of the chess board corners for a 8*6 board
# These 3D coordinates start from left top corner (corner with 4 squares, not the image corner) being (0,0,0)
# Right bottom being (7, 5, 0)
# z is 0 for all points since the board is on a flat image plane

real_object_points = np.zeros((8 * 6, 3), np.float32)

# Filling x, y
real_object_points[:, :2] = np.mgrid[0:8, 0:6].T.reshape(-1, 2)

print(real_object_points)
nx = 8
ny = 6

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

if ret:
    image_points.append(corners)
    object_points.append(real_object_points)

    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    plt.imshow(img)
    plt.show()