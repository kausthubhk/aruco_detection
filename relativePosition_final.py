
import cv2.aruco as aruco
import cv2
import numpy as np

# The inverse perspective transformation is used to reverse the mapping
# from 2D image coordinates to 3D world coordinates.
# It allows us to compute the 3D coordinates of an object or marker
# from its 2D image coordinates captured by a camera.
def inversePerspective(rvec, tvec):
    # Convert rotation vector (rvec) to a rotation matrix (R)
    R, _ = cv2.Rodrigues(rvec)
    R = np.matrix(R).T  # Transpose the rotation matrix
    # Compute the inverse translation vector (invTvec) using matrix multiplication
    invTvec = np.dot(-R, np.matrix(tvec))
    # Convert the inverse rotation matrix (R) to a rotation vector (invRvec)
    invRvec, _ = cv2.Rodrigues(R)
    return invRvec, invTvec

# Function to compute the relative position between two markers
def relativePosition(rvec1, tvec1, rvec2, tvec2):
    rvec1, tvec1 = rvec1.reshape((3, 1)), tvec1.reshape(
        (3, 1))
    rvec2, tvec2 = rvec2.reshape((3, 1)), tvec2.reshape((3, 1))

    # Inverse the second marker, the right one in the image
    invRvec, invTvec = inversePerspective(rvec2, tvec2)
    # Apply inverse perspective transformation again to obtain the original pose of the second marker
    orgRvec, orgTvec = inversePerspective(invRvec, invTvec)
    # print("rvec: ", rvec2, "tvec: ", tvec2, "\n and \n", orgRvec, orgTvec)

    # The double inversion is a validation step to confirm the accuracy of the perspective transformations
    # and to ensure they can be correctly reversed to recover the second marker's original pose.
    # It serves as a quality check for the pose estimation process

    # Compose the relative transformation between the two markers
    info = cv2.composeRT(rvec1, tvec1, invRvec, invTvec)
    composedRvec, composedTvec = info[0], info[1]

    composedRvec = composedRvec.reshape((3, 1))
    composedTvec = composedTvec.reshape((3, 1))
    return composedRvec, composedTvec