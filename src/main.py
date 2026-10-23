import sys

import cv2 as cv

import params
import lane
import calibration


def process_frame(frame, calib):
    undistorted = calibration.undistort_if_calibrated(frame, calib)
    roi = lane.crop_roi(undistorted)
    gray = lane.to_gray(roi)
    binary = cv.add(lane.sobel_x_binary(gray), lane.color_binary(roi))

    height, width = undistorted.shape[:2]
    M, Minv = lane.perspective_matrices(undistorted)
    warped_binary = lane.warper(binary, M, (width, height))

    left, right, windows = lane.find_lane_pixels(warped_binary)
    left_fit = lane.fit_polynomial(left)
    right_fit = lane.fit_polynomial(right)
    if left_fit is None or right_fit is None:
        return undistorted, None

    left_fit_m = lane.fit_in_meters(left)
    right_fit_m = lane.fit_in_meters(right)
    y_m = (height - 1) * params.YM_PER_PIX
    left_radius = lane.curvature_radius(left_fit_m, y_m)
    right_radius = lane.curvature_radius(right_fit_m, y_m)
    offset = lane.vehicle_offset(left_fit, right_fit, M, roi.shape, height)

    lane_area = lane.draw_lane_area(warped_binary.shape, left_fit, right_fit)
    result = lane.overlay_lane(undistorted, lane_area, Minv)
    if lane.is_departing(offset):
        lane.draw_warning(result)
    return result, (left_radius, right_radius, offset)


def report(info):
    if info is None:
        print("Linije trake nisu pronadjene")
        return
    left_radius, right_radius, offset = info
    side = "desno" if offset > 0 else "levo"
    print(f"Zakrivljenost: leva {left_radius:.0f} m, desna {right_radius:.0f} m | vozilo {abs(offset):.2f} m {side} od centra")


def warning_message(info):
    offset = info[2]
    side = "DESNO" if offset > 0 else "LEVO"
    return f"UPOZORENJE: napustanje trake {side} ({abs(offset):.2f} m od centra)"


def show_phase(result):
    cv.imshow("Result", result)


def run_image(path, calib):
    img = cv.imread(path)
    if img is None:
        print("Ne mogu da ucitam sliku:", path)
        return
    result, info = process_frame(img, calib)
    report(info)
    if info is not None and lane.is_departing(info[2]):
        print(warning_message(info))
    show_phase(result)
    cv.waitKey(0)


def run_video(path, calib):
    cap = cv.VideoCapture(path)
    frame_index = 0
    warned = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cap = cv.VideoCapture(path)
            frame_index = 0
            continue
        result, info = process_frame(frame, calib)
        if frame_index % params.PRINT_EVERY == 0:
            report(info)
        departing = info is not None and lane.is_departing(info[2])
        if departing and not warned:
            print(warning_message(info))
        warned = departing
        show_phase(result)
        frame_index += 1
        if cv.waitKey(params.FRAME_DELAY) & 0xFF == 27: break
    cap.release()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else params.DEFAULT_INPUT
    calib = calibration.load_calibration()
    if path.endswith(".mp4"):
        run_video(path, calib)
    else:
        run_image(path, calib)
    cv.destroyAllWindows()
