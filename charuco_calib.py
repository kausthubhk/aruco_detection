import numpy as np
import cv2
import glob


CHARUCO_BOARD_SIZE = (5, 7)
SQUARE_SIZE = 0.02  # 2 cm

def calibrate_camera(image_folder):

    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_1000)
    charuco_board = cv2.aruco.CharucoBoard_create(CHARUCO_BOARD_SIZE[0], CHARUCO_BOARD_SIZE[1], SQUARE_SIZE, 0.8, aruco_dict)

    obj_points = []
    img_points = []


    images = glob.glob(image_folder + "/*.png")

    for image_path in images:
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect charuco board corners and IDs
        corners, ids, rejected_points = cv2.aruco.detectMarkers(gray_image, aruco_dict)
        _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(corners, ids, gray_image, charuco_board)

        # If charuco board was found, add points for calibration
        if charuco_corners is not None and charuco_ids is not None:
            img_points.append(charuco_corners)
            obj_points.append(charuco_board.chessboardCorners)

    # Perform camera calibration
    _, camera_matrix, dist_coeffs, _, _ = cv2.aruco.calibrateCameraCharuco(obj_points, img_points, charuco_board, gray_image.shape[::-1], None, None)

    # Print calibration results
    print("Camera matrix:")
    print(camera_matrix)
    print("\nDistortion coefficients:")
    print(dist_coeffs)

    # Save calibration parameters to a file
    np.savez("camera_calibration.npz", camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)

    return camera_matrix, dist_coeffs

if __name__ == "__main__":
    image_folder = r"C:\Users\Kausthubh\PycharmProjects\chessboard_calib"
    camera_matrix, dist_coeffs = calibrate_camera(image_folder)
