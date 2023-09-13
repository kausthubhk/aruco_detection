import cv2
import numpy as np
#THIS PYTHON FILE WAS USED TO GET THE ID OF EACH ARUCO MARKER SO THAT IT COULD BE PASSED AS ARGUMENTS . WE CAN ONLY DETECT THE MARKER
#IF WE KNOW THE ID OF THE MARKER AND THIS PYTHON SNIPPET WAS USED TO GET THE ID OF THE MARKER

# Load the dictionary
aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)

# Load image and convert to grayscale
image = cv2.imread('aruco/aruco_image2.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect ArUco markers
corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict)

if ids is not None:
    # Print the IDs of detected ArUco markers
    for i in range(len(ids)):
        print(f"Detected ArUco marker with ID: {ids[i]}")
else:
    print("No ArUco markers detected.")
