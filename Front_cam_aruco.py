import numpy as np
import cv2
import cv2.aruco as aruco
import keyboard
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px
import pandas as pd

import argparse
# Import functions from other modules
from calibrate_final import calibrate
from saveCoefficients_final import saveCoefficients
from relativePosition_final import relativePosition
from loadCoefficients_final import loadCoefficients

cap = cv2.VideoCapture(0)

# Function to draw lines connecting marker points
def draw(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    # img = cv2.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color

    # Loop through pairs of corner points and draw lines connecting them on the input image 'img'.
    for i,j in zip(range(4),range(4)):
        img = cv2.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    return img



# Define a function to track ArUco markers and visualize their relative positions.
def track(matrix_coefficients, distortion_coefficients, markerIDs):
    pointCircle = (0, 0)
    marker_data = {} # Initialize a dictionary to store marker data
    # Create a 3D subplot grid
    fig = plt.figure()
    num_subplots = len(markerIDs) * (len(markerIDs) - 1) // 2
    subplot_rows = int(np.sqrt(num_subplots))
    subplot_cols = int(np.ceil(num_subplots / subplot_rows))
    subplot_index = 1
    while True:
        ret, frame = cap.read()
        # operations on the frame come here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Change grayscale
        aruco_dict = aruco.Dictionary_get(aruco.DICT_5X5_250)  # Use 5x5 dictionary to find markers
        parameters = aruco.DetectorParameters_create()  # Marker detection parameters

        # lists of ids and the corners beloning to each id
        corners, ids, rejected_img_points = aruco.detectMarkers(gray, aruco_dict,
                                                                parameters=parameters,
                                                                cameraMatrix=matrix_coefficients,
                                                                distCoeff=distortion_coefficients)

        if np.all(ids is not None):  # If there are markers found by detector
            zipped = zip(ids, corners)
            sorted_zipped = sorted(zipped, key=lambda x: x[0])  # Sort based on IDs
            ids, corners = zip(*sorted_zipped)
            # Iterate through all the detected ArUco markers identified by their unique IDs.
            for i in range(len(ids)):
                # Estimate the pose (position and orientation) of the current marker (i) in 3D space.
                rvec, tvec, markerPoints = aruco.estimatePoseSingleMarkers(corners[i], 0.02, matrix_coefficients,
                                                                           distortion_coefficients)
                # Create keys for storing marker-specific data based on their IDs.
                marker_key = f"MarkerID{ids[i]}"
                rvec_key = f"{marker_key}Rvec"
                tvec_key = f"{marker_key}Tvec"
                # print("here are the marker keys")     - THIS WAS FOR DEBUGGING
                # print(marker_key)
                # print("here are the rvec keys")
                # print(rvec_key)
                # print("here are the tvec keys")
                # print(tvec_key)

                # Check if the rotation and translation keys exist in marker_data dictionary.
                # If not, initialize them as empty lists to store data for this marker.
                if rvec_key not in marker_data:
                    marker_data[rvec_key] = []
                if tvec_key not in marker_data:
                    marker_data[tvec_key] = []

                marker_data[rvec_key].append(rvec)
                marker_data[tvec_key].append(tvec)
                #print(marker_data)
                aruco.drawDetectedMarkers(frame, corners)  # Draw A square around the markers

        if keyboard.is_pressed('c') and len(marker_data) >= 2:
                print("C HAS BEEN PRESSED")
                # Extract rvec and tvec values for each marker
                rvec_values = [np.array(marker_data[f"MarkerID[{i}]Rvec"]).reshape((-1, 3, 1)) for i in markerIDs]
                tvec_values = [np.array(marker_data[f"MarkerID[{i}]Tvec"]).reshape((-1, 3, 1)) for i in markerIDs]

                for i in range(len(markerIDs)):
                    for j in range(i + 1, len(markerIDs)):
                        rvec1 = rvec_values[i][0]  # Extract the first rvec from the list
                        tvec1 = tvec_values[i][0]  # Extract the first tvec from the list
                        rvec2 = rvec_values[j][0]  # Extract the first rvec from the list
                        tvec2 = tvec_values[j][0]  # Extract the first tvec from the list

                        # Call the relativePosition function
                        composed_rvec, composed_tvec = relativePosition(rvec1, tvec1, rvec2, tvec2)

                        # Print or do something with the composed rvec and tvec
                        print(f"Relative position between Marker {i} and Marker {j}:")
                        print("Composed rvec:", composed_rvec)
                        print("Composed tvec:", composed_tvec)

                        ax = fig.add_subplot(subplot_rows, subplot_cols, subplot_index, projection='3d')
                        subplot_index += 1
                        # Plot the relative positions
                        ax.quiver(tvec1[0], tvec1[1], tvec1[2], composed_tvec[0], composed_tvec[1], composed_tvec[2],
                                  color='b', label='Translation')
                        ax.quiver(tvec1[0], tvec1[1], tvec1[2], composed_rvec[0], composed_rvec[1], composed_rvec[2],
                                  color='r', label='Rotation')
                        # Set labels for axes
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')

                # Add legend
                ax.legend()

                # Show the plot
                plt.show()
        # Display the resulting frame
        cv2.imshow('frame', frame)
        # Wait 3 milliseconds for an interaction. Check the key and do the corresponding job.
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print(key)
            print("q has been pressed")# Quit
            break

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Aruco Marker Tracking')
    parser.add_argument('--coefficients', metavar='bool', required=True,
                        help='File name for matrix coefficients and distortion coefficients')
    parser.add_argument('--markerIDs', nargs='+', type=int, required=True, help='List of marker IDs')
    # Parse the arguments and take action for that.
    args = parser.parse_args()
    # Get the marker IDs from the arguments
    markerIDs = args.markerIDs
    num_marker_ids = len(markerIDs)

    # Now you have the marker IDs in the list markerIDs and the number of marker IDs in num_marker_ids
    # You can use them in the rest of your code
    print("Marker IDs:", markerIDs)
    print("Number of marker IDs:", num_marker_ids)
    if args.coefficients == '1':
        # Load camera matrix (mtx) and distortion coefficients (dist) from a file
        mtx, dist = loadCoefficients()
        print(mtx,dist)
        ret = True
    else:
        # If 'coefficients' argument is not '1', perform camera calibration
        ret, mtx, dist, rvecs, tvecs = calibrate()
        print(ret, mtx, dist, rvecs, tvecs)
        # Save the newly calibrated coefficients (mtx, dist) to a file
        saveCoefficients(mtx, dist)
    print("Calibration is completed. Starting tracking sequence.")
    if ret:
        track(mtx, dist, markerIDs)