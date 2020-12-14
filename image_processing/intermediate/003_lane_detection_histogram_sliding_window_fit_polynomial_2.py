# Inspired by Self-Driving-Car Nano Degree from Udacity

# Note that following is done for an image which has already gone through Calibration, Threholding and
# Perspective Tranformation Steps

# using the full algorithm from before and starting fresh on every frame may seem inefficient,
# as the lane lines don't necessarily move a lot from frame to frame.
# once you know where the lines are in one frame of video,
# you can do a highly targeted search for them in the next frame.

# This is equivalent to using a customized region of interest for each frame of video,
# and should help you track the lanes through sharp curves and tricky conditions.
# If you lose track of the lines, go back to your sliding windows search or other method to rediscover them.

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

print("Incomplete")
exit()

binary_warped_image = mpimg.imread(image_path)

# Polynomial fit values from the previous frame
# Make sure to grab the actual values from the previous step in your project!
left_fit = np.array([2.13935315e-04, -3.77507980e-01, 4.76902175e+02])
right_fit = np.array([4.17622148e-04, -4.93848953e-01, 1.11806170e+03])


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


def search_around_poly(binary_warped):
    # HYPER-PARAMETER
    # Choose the width of the margin around the previous polynomial to search
    # The quiz grader expects 100 here, but feel free to tune on your own!
    margin = 100

    # Grab activated pixels
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    ### TO-DO: Set the area of search based on activated x-values ###
    ### within the +/- margin of our polynomial function ###
    ### Hint: consider the window areas for the similarly named variables ###
    ### in the previous quiz, but change the windows to our new search area ###
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy +
                                   left_fit[2] - margin)) & (nonzerox < (left_fit[0] * (nonzeroy ** 2) +
                                                                         left_fit[1] * nonzeroy + left_fit[
                                                                             2] + margin)))
    right_lane_inds = ((nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy +
                                    right_fit[2] - margin)) & (nonzerox < (right_fit[0] * (nonzeroy ** 2) +
                                                                           right_fit[1] * nonzeroy + right_fit[
                                                                               2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit new polynomials
    left_fitx, right_fitx, ploty = fit_poly(binary_warped.shape, leftx, lefty, rightx, righty)

    ## Visualization ##
    # Create an image to draw on and an image to show the selection window
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    window_img = np.zeros_like(out_img)
    # Color in left and right line pixels
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    # Generate a polygon to illustrate the search window area
    # And recast the x and y points into usable format for cv2.fillPoly()
    left_line_window1 = np.array([np.transpose(np.vstack([left_fitx - margin, ploty]))])
    left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx + margin,
                                                                    ploty])))])
    left_line_pts = np.hstack((left_line_window1, left_line_window2))
    right_line_window1 = np.array([np.transpose(np.vstack([right_fitx - margin, ploty]))])
    right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx + margin,
                                                                     ploty])))])
    right_line_pts = np.hstack((right_line_window1, right_line_window2))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
    cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
    result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

    # Plot the polynomial lines onto the image
    plt.plot(left_fitx, ploty, color='yellow')
    plt.plot(right_fitx, ploty, color='yellow')
    ## End visualization steps ##

    return result


def fit_poly(img_shape, leftx, lefty, rightx, righty):
    ### TO-DO: Fit a second order polynomial to each with np.polyfit() ###
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, img_shape[0] - 1, img_shape[0])
    ### TO-DO: Calc both polynomials using ploty, left_fit and right_fit ###
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    return left_fitx, right_fitx, ploty


out_image = search_around_poly(binary_warped_image)

plt.imshow(out_image)
plt.show()


"""
Fitting on Large Curves
In current implementation of sliding window search what happens when we arrive at the left or right edge of an image, 
such as when there is a large curve on the road ahead. 
If min_pixel is not achieved (i.e. the curve ran off the image), 
the starting position of our next window doesn't change, 
so it is just positioned directly above the previous window. 
This will repeat for however many windows are left in n_windows, 
stacking the sliding windows vertically against the side of the image, 
and likely leading to an imperfect polynomial fit.
"""