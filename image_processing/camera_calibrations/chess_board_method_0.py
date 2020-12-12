# Inspired by Self-Driving-Car Nano Degree from Udacity

import os
import cv2
import matplotlib.pyplot as plt

image_path = os.path.join(os.getcwd(), "../../samples/chess_boards/chess_board_0.png")

if not os.path.exists(image_path):
    print("Image does not exist!")
    exit()

nx = 8  # number of inside corners in x direction
ny = 6  # number of inside corners in y direction

img = cv2.imread(image_path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

if ret == True:
    cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
    plt.imshow(img)
    plt.show()

