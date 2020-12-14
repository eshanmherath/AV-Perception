# Inspired by Self-Driving-Car Nano Degree from Udacity

# Note that following is done for an image which has already gone through Calibration, Threholding and
# Perspective Tranformation Steps

import os

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

# Read in an image and grayscale it
image_path = os.path.join(os.getcwd(), "../../samples/roads/warped-road.jpg")

if not os.path.exists(image_path):
    print("Image does not exist!")
    exit()

binary_warped_image = mpimg.imread(image_path)


def find_lane_pixels(binary_warped):
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)

    # Create an output image to draw on and visualize the result
    out_image = np.dstack((binary_warped, binary_warped, binary_warped))

    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0] // 2)
    left_x_base = np.argmax(histogram[:midpoint])
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint

    # HYPER-PARAMETERS
    # Choose the number of sliding windows
    n_windows = 9
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    min_pixels = 50

    # Set height of windows - based on n_windows above and image shape
    window_height = np.int(binary_warped.shape[0] // n_windows)

    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzero_y = np.array(nonzero[0])
    nonzero_x = np.array(nonzero[1])

    # Current positions to be updated later for each window in n_windows
    left_x_current = left_x_base
    right_x_current = right_x_base

    # Create empty lists to receive left and right lane pixel indices
    left_lane_indexes = []
    right_lane_indexes = []

    # Step through the windows one by one
    for window in range(n_windows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height

        win_x_left_low = left_x_current - margin
        win_x_left_high = left_x_current + margin

        win_x_right_low = right_x_current - margin
        win_x_right_high = right_x_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_image, (win_x_left_low, win_y_low),
                      (win_x_left_high, win_y_high), (0, 255, 0), 2)

        cv2.rectangle(out_image, (win_x_right_low, win_y_low),
                      (win_x_right_high, win_y_high), (0, 255, 0), 2)

        # Identify the nonzero pixels in x and y within the window #
        good_left_indexes = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                             (nonzero_x >= win_x_left_low) & (nonzero_x < win_x_left_high)).nonzero()[0]

        good_right_indexes = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) &
                              (nonzero_x >= win_x_right_low) & (nonzero_x < win_x_right_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_indexes.append(good_left_indexes)
        right_lane_indexes.append(good_right_indexes)

        # If you found > min_pixels pixels, recenter next window on their mean position
        if len(good_left_indexes) > min_pixels:
            left_x_current = np.int(np.mean(nonzero_x[good_left_indexes]))
        if len(good_right_indexes) > min_pixels:
            right_x_current = np.int(np.mean(nonzero_x[good_right_indexes]))

    # Concatenate the arrays of indices (previously was a list of lists of pixels)
    try:
        left_lane_indexes = np.concatenate(left_lane_indexes)
        right_lane_indexes = np.concatenate(right_lane_indexes)
    except ValueError:
        # Avoids an error if the above is not implemented fully
        pass

    # Extract left and right line pixel positions
    left_x = nonzero_x[left_lane_indexes]
    left_y = nonzero_y[left_lane_indexes]
    right_x = nonzero_x[right_lane_indexes]
    right_y = nonzero_y[right_lane_indexes]

    return left_x, left_y, right_x, right_y, out_image


def fit_polynomial(binary_warped):
    # Find our lane pixels first
    left_x, lefty, right_x, right_y, out_image = find_lane_pixels(binary_warped)

    # Fit a second order polynomial to each using `np.polyfit`
    left_fit = np.polyfit(lefty, left_x, 2)
    right_fit = np.polyfit(right_y, right_x, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    
    try:
        left_fit_x = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
        right_fit_x = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    except TypeError:
        # Avoids an error if `left` and `right_fit` are still none or incorrect
        print('The function failed to fit a line!')
        left_fit_x = 1 * ploty ** 2 + 1 * ploty
        right_fit_x = 1 * ploty ** 2 + 1 * ploty

    """
    Take note of how we fit the lines above - while normally you calculate a y-value for a given x, 
    here we do the opposite. Why? Because we expect our lane lines to be (mostly) vertically-oriented.
    """

    # Visualization
    # Colors in the left and right lane regions
    out_image[lefty, left_x] = [255, 0, 0]
    out_image[right_y, right_x] = [0, 0, 255]

    # Plots the left and right polynomials on the lane lines
    plt.plot(left_fit_x, ploty, color='yellow')
    plt.plot(right_fit_x, ploty, color='yellow')

    return out_image


out_img = fit_polynomial(binary_warped_image)

plt.imshow(out_img)
plt.show()
