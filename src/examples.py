import os

import cv2 as cv
import numpy as np

import params
import lane
import calibration
import main


def save(name, img):
    path = params.EXAMPLES_DIR + "/" + name
    cv.imwrite(path, img)
    print("Sacuvano:", path)


def generate_examples(calib):
    os.makedirs(params.EXAMPLES_DIR, exist_ok=True)
    mtx, dist, size = calib

    chess = cv.imread("course_files_export/Zadatak/camera_cal/calibration1.jpg")
    save("undistort_chessboard.jpg", np.hstack((chess, calibration.undistort(chess, mtx, dist))))

    road = cv.imread("course_files_export/Zadatak/test_images/test1.jpg")
    undistorted = calibration.undistort(road, mtx, dist)
    save("undistort_test1.jpg", np.hstack((road, undistorted)))

    roi = lane.crop_roi(undistorted)
    gray = lane.to_gray(roi)
    sobel = lane.sobel_x_binary(gray)
    color = lane.color_binary(roi)
    blank = np.zeros(sobel.shape, dtype='uint8')
    save("binary_components_test1.jpg", cv.merge([color, sobel, blank]))
    save("binary_test1.jpg", cv.add(sobel, color))

    binary = cv.add(sobel, color)
    height, width = undistorted.shape[:2]
    M, Minv = lane.perspective_matrices(undistorted)
    warped = lane.warper(roi, M, (width, height))
    warped_binary = lane.warper(binary, M, (width, height))
    save("warped_test1.jpg", np.hstack((warped, cv.merge([warped_binary, warped_binary, warped_binary]))))

    straight = calibration.undistort(cv.imread("course_files_export/Zadatak/test_images/straight_lines1.jpg"), mtx, dist)
    M, Minv = lane.perspective_matrices(straight)
    straight_warped = lane.warper(lane.crop_roi(straight), M, (width, height))
    src, dst = lane.warp_points(straight)
    lane.draw_trapezoid(straight)
    lane.draw_polygon(straight_warped, dst, params.TRAPEZ_COLOR)
    save("warped_straight_lines.jpg", np.hstack((straight, straight_warped)))

    for name in ["test2", "test1"]:
        road = calibration.undistort(cv.imread("course_files_export/Zadatak/test_images/" + name + ".jpg"), mtx, dist)
        roi = lane.crop_roi(road)
        M, Minv = lane.perspective_matrices(road)
        sobel_warped = lane.warper(lane.sobel_x_binary(lane.to_gray(roi)), M, (width, height))
        color_warped = lane.warper(lane.color_binary(roi), M, (width, height))
        warped_binary = cv.add(sobel_warped, color_warped)
        hist = lane.histogram(sobel_warped, color_warped)
        left, right, windows = lane.find_lane_pixels(warped_binary, hist)
        left_fit = lane.fit_polynomial(left)
        right_fit = lane.fit_polynomial(right)
        fit_img = lane.draw_lane_fit(warped_binary, left, right, windows, left_fit, right_fit)
        hist_img = cv.resize(lane.draw_histogram(hist), (width, params.HISTOGRAM_HEIGHT))
        save("color_fit_lines_" + name + ".jpg", np.vstack((fit_img, hist_img)))
        print(name, "levi fit:", left_fit, "desni fit:", right_fit)

    for name in ["test2", "test3", "straight_lines1"]:
        result, info, debug = main.process_frame(cv.imread("course_files_export/Zadatak/test_images/" + name + ".jpg"), calib)
        save("example_output_" + name + ".jpg", result)
        main.report(info)

    video = cv.VideoCapture("course_files_export/Zadatak/test_videos/project_video03.mp4")
    for i in range(780):
        ret, frame = video.read()
    video.release()
    result, info, debug = main.process_frame(frame, calib)
    save("example_output_warning.jpg", result)
    main.report(info)


if __name__ == "__main__":
    generate_examples(calibration.load_calibration())
