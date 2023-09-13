import numpy as np
import cv2
import glob
import cv2.aruco as aruco

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

def calibrate():
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,6,0)
    # In camera calibration, it's essential to prepare a set of 3D object points to represent known physical
    # locations in the real world. These object points serve as reference points with known coordinates in a
    # three-dimensional space. When combined with their corresponding 2D image points (detected in images taken
    # with the camera), they allow for the estimation of camera parameters such as the camera matrix and
    # distortion coefficients. This step is crucial for accurately calibrating the camera and correcting lens
    # distortions.

    objp = np.zeros((5*8,3), np.float32)
    objp[:, :2] = np.mgrid[0:8, 0:5].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob('*.jpg')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (8, 5), None)
        print(ret, corners)
        # If found, add object points, image points (after refining them)
        if ret:
            objpoints.append(objp)
            # The cv2.cornerSubPix function takes the initial corner coordinates (corners) and the grayscale image (gray)
            # as input, along with other parameters, and performs sub-pixel corner refinement. It refines the initial corner
            # positions to achieve sub-pixel accuracy, which is valuable in various computer vision tasks.

            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            img = cv2.drawChessboardCorners(img, (8, 5), corners2, ret)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print(ret, mtx, dist, rvecs, tvecs)
    return [ret, mtx, dist, rvecs, tvecs]
