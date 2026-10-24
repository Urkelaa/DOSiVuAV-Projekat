import sys

import cv2 as cv
import numpy as np

import params
import lane
import calibration
import tracker as tracking


def process_frame(frame, calib, tracker=None):
    undistorted = calibration.undistort_if_calibrated(frame, calib)
    roi = lane.crop_roi(undistorted)
    gray = lane.to_gray(roi)
    sobel = lane.sobel_x_binary(gray)
    color = lane.color_binary(roi)

    height, width = undistorted.shape[:2]
    M, Minv = lane.perspective_matrices(undistorted)
    sobel_warped = lane.warper(sobel, M, (width, height))
    color_warped = lane.warper(color, M, (width, height))
    warped_binary = cv.add(sobel_warped, color_warped)
    hist = lane.histogram(sobel_warped, color_warped)

    if tracking.has_lines(tracker):
        previous_fits = (tracker["left_fit"], tracker["right_fit"])
        left, right = lane.search_around_poly(warped_binary, previous_fits[0], previous_fits[1])
        windows = []
        source = "pretraga"
    else:
        previous_fits = None
        left, right, windows = lane.find_lane_pixels(warped_binary, hist)
        source = "prozori"

    raw_left_fit = lane.fit_polynomial(left)
    raw_right_fit = lane.fit_polynomial(right)
    if tracker is not None:
        left_fit, right_fit, accepted = tracking.update(tracker, raw_left_fit, raw_right_fit, warped_binary.shape)
    else:
        left_fit, right_fit = raw_left_fit, raw_right_fit
        accepted = lane.is_valid(left_fit, right_fit, warped_binary.shape)

    debug = {
        "undistorted": undistorted,
        "roi": roi,
        "M": M,
        "sobel": sobel,
        "color": color,
        "warped_binary": warped_binary,
        "hist": hist,
        "left": left,
        "right": right,
        "windows": windows,
        "previous_fits": previous_fits,
        "raw_left_fit": raw_left_fit,
        "raw_right_fit": raw_right_fit,
        "left_fit": left_fit,
        "right_fit": right_fit,
        "source": source,
        "accepted": accepted,
    }

    if left_fit is None or right_fit is None:
        result = undistorted.copy()
        lane.draw_info(result, None)
        return result, None, debug

    y_m = (height - 1) * lane.ym_per_pix(height)
    left_radius = lane.curvature_radius(lane.fit_to_meters(left_fit, width, height), y_m)
    right_radius = lane.curvature_radius(lane.fit_to_meters(right_fit, width, height), y_m)
    offset = lane.vehicle_offset(left_fit, right_fit, M, roi.shape, warped_binary.shape)

    lane_area = lane.draw_lane_area(warped_binary.shape, left_fit, right_fit)
    result = lane.overlay_lane(undistorted, lane_area, Minv)
    info = (left_radius, right_radius, offset)
    if lane.is_departing(offset):
        lane.draw_warning(result)
    lane.draw_info(result, info)
    return result, info, debug


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


def debug_report(frame_index, info, debug):
    left_x, left_y = debug["left"]
    right_x, right_y = debug["right"]
    text = f"[{frame_index}] {debug['source']}"
    if len(debug["windows"]) > 0:
        left_start = debug["windows"][0][0] + params.WINDOW_MARGIN
        right_start = debug["windows"][1][0] + params.WINDOW_MARGIN
        text += f" start L={left_start} D={right_start}"
    text += f" | piksela L={len(left_x)} D={len(right_x)}"
    raw_left_fit = debug["raw_left_fit"]
    raw_right_fit = debug["raw_right_fit"]
    if raw_left_fit is not None and raw_right_fit is not None:
        height = debug["warped_binary"].shape[0]
        widths = lane.lane_widths(raw_left_fit, raw_right_fit, height)
        text += f" | sirina dno={widths[0]:.0f} sred={widths[1]:.0f} vrh={widths[2]:.0f} px"
    text += " | prihvacen" if debug["accepted"] else " | ODBACEN"
    if info is not None:
        left_radius, right_radius, offset = info
        text += f" | R L={left_radius:.0f} D={right_radius:.0f} m | pomeraj {offset:+.2f} m"
    print(text)


def show_debug(debug):
    size = (params.DEBUG_WIDTH, params.DEBUG_HEIGHT)

    preview = debug["undistorted"].copy()
    x1, y1, x2, y2 = lane.roi_bounds(preview)
    cv.rectangle(preview, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
    lane.draw_trapezoid(preview)

    blank = np.zeros(debug["sobel"].shape, dtype='uint8')
    components = cv.merge([debug["color"], debug["sobel"], blank])
    components_height = params.DEBUG_WIDTH * components.shape[0] // components.shape[1]

    warped_binary = debug["warped_binary"]
    height, width = warped_binary.shape
    warped = lane.warper(debug["roi"], debug["M"], (width, height))
    fit_img = lane.draw_lane_fit(warped_binary, debug["left"], debug["right"], debug["windows"], debug["raw_left_fit"], debug["raw_right_fit"])
    if debug["previous_fits"] is not None:
        lane.draw_search_band(fit_img, debug["previous_fits"][0], debug["previous_fits"][1])
    hist_img = lane.draw_histogram(debug["hist"])

    cv.imshow("1 Undistorted + ROI + trapez", cv.resize(preview, size))
    cv.imshow("2 Binary ROI (zeleno Sobel, plavo boja)", cv.resize(components, (params.DEBUG_WIDTH, components_height)))
    cv.imshow("3a Warped", cv.resize(warped, size))
    cv.imshow("3b Warped binary", cv.resize(warped_binary, size))
    cv.imshow("4 Histogram", cv.resize(hist_img, (params.DEBUG_WIDTH, params.HISTOGRAM_HEIGHT // 2)))
    cv.imshow("5 Lane fit", cv.resize(fit_img, size))


def show_phase(result):
    cv.imshow("Result", result)


def run_image(path, calib, debug_mode):
    img = cv.imread(path)
    if img is None:
        print("Ne mogu da ucitam sliku:", path)
        return
    result, info, debug = process_frame(img, calib)
    report(info)
    if info is not None and lane.is_departing(info[2]):
        print(warning_message(info))
    if debug_mode:
        debug_report(0, info, debug)
        show_debug(debug)
    show_phase(result)
    cv.waitKey(0)


def run_video(path, calib, debug_mode):
    cap = cv.VideoCapture(path)
    tracker = tracking.new_tracker()
    frame_index = 0
    warned = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            cap.release()
            cap = cv.VideoCapture(path)
            tracker = tracking.new_tracker()
            frame_index = 0
            continue
        result, info, debug = process_frame(frame, calib, tracker)
        if debug_mode:
            debug_report(frame_index, info, debug)
            show_debug(debug)
        elif frame_index % params.PRINT_EVERY == 0:
            report(info)
        departing = info is not None and lane.is_departing(info[2])
        if departing and not warned:
            print(warning_message(info))
        warned = departing
        show_phase(result)
        frame_index += 1
        key = cv.waitKey(params.FRAME_DELAY) & 0xFF
        if key == 27: break
        if key == 32:
            cv.waitKey(0)
    cap.release()


if __name__ == "__main__":
    debug_mode = "--debug" in sys.argv
    args = [arg for arg in sys.argv[1:] if arg != "--debug"]
    path = args[0] if len(args) > 0 else params.DEFAULT_INPUT
    calib = calibration.load_calibration()
    if path.endswith(".mp4"):
        run_video(path, calib, debug_mode)
    else:
        run_image(path, calib, debug_mode)
    cv.destroyAllWindows()
