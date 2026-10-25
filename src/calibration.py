import glob

import cv2 as cv
import numpy as np

import params


def calibrate(show=True):
    print("Calibrating camera...")
    criteria = (cv.TERM_CRITERIA_MAX_ITER + cv.TERM_CRITERIA_EPS, 30, 0.001)
    pattern = (params.CHESS_COLS, params.CHESS_ROWS)

    object_points = np.zeros((params.CHESS_COLS * params.CHESS_ROWS, 3), np.float32)
    object_points[:, :2] = np.mgrid[0:params.CHESS_COLS, 0:params.CHESS_ROWS].T.reshape(-1, 2)

    object_points_array = []
    img_points_array = []

    for path in sorted(glob.glob(params.CALIB_IMAGES)):
        img = cv.imread(path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, pattern, None)
        if ret:
            corners = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            object_points_array.append(object_points)
            img_points_array.append(corners)
            cv.drawChessboardCorners(img, pattern, corners, ret)
        if show:
            cv.imshow("Chessboard", img)
            cv.waitKey(params.CALIB_SHOW_DELAY)

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(object_points_array, img_points_array, gray.shape[::-1], None, None)
    np.savez(params.CALIB_FILE, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs, size=gray.shape[::-1])
    print("Calibration saved to", params.CALIB_FILE)
    return mtx, dist


def load_calibration():
    try:
        calibration = np.load(params.CALIB_FILE)
    except FileNotFoundError:
        calibrate(show=False)
        calibration = np.load(params.CALIB_FILE)
    return calibration["mtx"], calibration["dist"], tuple(calibration["size"])


def undistort(img, mtx, dist):
    return cv.undistort(img, mtx, dist, None, mtx)


def undistort_if_calibrated(img, calib):
    mtx, dist, size = calib
    height, width = img.shape[:2]
    if (width, height) != size:
        return img
    return undistort(img, mtx, dist)


if __name__ == "__main__":
    mtx, dist = calibrate()

    chess = cv.imread("course_files_export/Zadatak/camera_cal/calibration1.jpg")
    road = cv.imread("course_files_export/Zadatak/test_images/test1.jpg")

    cv.imshow("Chessboard", np.hstack((chess, undistort(chess, mtx, dist))))
    cv.imshow("Road", np.hstack((road, undistort(road, mtx, dist))))
    cv.waitKey(0)
    cv.destroyAllWindows()
