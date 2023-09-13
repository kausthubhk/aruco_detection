import cv2
import cv2.aruco as aruco
def saveCoefficients(mtx, dist):
    cv_file = cv2.FileStorage("calibrationCoefficients.yaml", cv2.FILE_STORAGE_WRITE)
    cv_file.write("camera_matrix", mtx)
    cv_file.write("dist_coeff", dist)
    # note you *release* you don't close() a FileStorage object
    cv_file.release()