import cv2
import numpy as np
import os
import glob

# Define the chessboard size (number of inner corners)
CHECKERBOARD = (5, 8)
# Define the size of each square on the chessboard in millimeters
square_size_mm = 29.8
# Define termination criteria for corner detection refinement
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# Create lists to store 3D world points and 2D image points
threedpoints = []
twodpoints = []

# Create an array to represent 3D coordinates of chessboard corners
objectp3d = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objectp3d[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objectp3d *= square_size_mm

# Initialize variables to store image dimensions
prev_img_shape = None
# Get a list of image file paths in the current directory
images = glob.glob('jpg_chessboard/*.jpg')


# Loop through each image in the list
for filename in images:
    image = cv2.imread(filename)
    grayColor = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners in the grayscale image
    ret, corners = cv2.findChessboardCorners(
        grayColor, CHECKERBOARD,
        cv2.CALIB_CB_ADAPTIVE_THRESH +
        cv2.CALIB_CB_FAST_CHECK +
        cv2.CALIB_CB_NORMALIZE_IMAGE)

    # If corners are found, append 3D and 2D points to their respective lists
    if ret == True:
        threedpoints.append(objectp3d)
        corners2 = cv2.cornerSubPix(
            grayColor, corners, (5, 5), (-1, -1), criteria)

        twodpoints.append(corners2)
        # Draw detected corners on the image
        image = cv2.drawChessboardCorners(image,
                                          CHECKERBOARD,
                                          corners2, ret)
    else:
        print("Min corners not found")
        exit(0)

    screen_width, screen_height = 1920, 1080
    resized_image = cv2.resize(image, (screen_width, screen_height))


    cv2.imshow('Resized Image', resized_image)
    cv2.waitKey(0)

cv2.destroyAllWindows()

h, w = image.shape[:2]

# Perform camera calibration by
# passing the value of the above found out 3D points (threedpoints)
# and its corresponding pixel coordinates of the
# detected corners (twodpoints)
ret, matrix, distortion, r_vecs, t_vecs = cv2.calibrateCamera(
    threedpoints, twodpoints, grayColor.shape[::-1], None, None)

# Displaying required output
print("Camera matrix:")
print(matrix)

print("\nDistortion coefficient:")
print(distortion)

print("\nRotation Vectors:")
print(r_vecs)

print("\nTranslation Vectors:")
print(t_vecs)

