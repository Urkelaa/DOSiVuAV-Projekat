from math import cos, sin

import cv2 as cv
import numpy as np

import params


def to_gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)


def roi_bounds(img):
    height, width = img.shape[:2]
    x1 = int(width * params.ROI_LEFT)
    x2 = int(width * params.ROI_RIGHT)
    y1 = int(height * params.ROI_TOP)
    y2 = int(height * params.ROI_BOTTOM)
    return x1, y1, x2, y2


def crop_roi(img):
    x1, y1, x2, y2 = roi_bounds(img)
    return img[y1:y2, x1:x2]


def blur(img):
    return cv.GaussianBlur(img, params.GAUSS_KERNEL, 0)


def edges(img):
    return cv.Canny(img, params.CANNY_LOW, params.CANNY_HIGH, apertureSize=3)


def hough_lines(edges):
    return cv.HoughLines(edges, params.HOUGH_RHO, params.HOUGH_THETA, params.HOUGH_THRESHOLD)


def x_at(rho, theta, y):
    return (rho - y * sin(theta)) / cos(theta)


def is_horizontal(theta):
    return abs(theta - np.pi / 2) < params.HORIZONTAL_MARGIN


def select_lanes(lines, width, height):
    left = None
    right = None
    if lines is None:
        return left, right
    for line in lines:
        rho, theta = line[0]
        if is_horizontal(theta):
            continue
        x_bottom = x_at(rho, theta, height)
        if left is None and theta < np.pi / 2 and x_bottom < width / 2:
            left = (rho, theta)
        if right is None and theta > np.pi / 2 and x_bottom >= width / 2:
            right = (rho, theta)
        if left is not None and right is not None:
            break
    return left, right


def line_points(line, height, offset):
    rho, theta = line
    ox, oy = offset
    p1 = (int(x_at(rho, theta, 0)) + ox, oy)
    p2 = (int(x_at(rho, theta, height)) + ox, height + oy)
    return p1, p2


def draw_lane(img, line, height, offset, color):
    if line is None:
        return
    p1, p2 = line_points(line, height, offset)
    cv.line(img, p1, p2, color, params.LANE_THICKNESS)


def sobel_x_binary(gray):
    sobel_x = cv.Sobel(gray, ddepth=cv.CV_32F, dx=1, dy=0, ksize=params.SOBEL_KSIZE)
    abs_x = cv.convertScaleAbs(sobel_x)
    return cv.inRange(abs_x, params.SOBEL_LOW, params.SOBEL_HIGH)


def yellow_binary(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    return cv.inRange(hsv, params.YELLOW_LOW, params.YELLOW_HIGH)


def white_binary(img):
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    return cv.inRange(hsv, params.WHITE_LOW, params.WHITE_HIGH)


def color_binary(img):
    return cv.add(yellow_binary(img), white_binary(img))


def combined_binary(img):
    gray = to_gray(img)
    return cv.add(sobel_x_binary(gray), color_binary(img))


def warp_points(img):
    height, width = img.shape[:2]
    x1, y1, x2, y2 = roi_bounds(img)
    src = params.WARP_SRC * np.float32([width, height]) - np.float32([x1, y1])
    dst = params.WARP_DST * np.float32([width, height])
    return src, dst


def perspective_matrices(img):
    src, dst = warp_points(img)
    M = cv.getPerspectiveTransform(src, dst)
    Minv = cv.getPerspectiveTransform(dst, src)
    return M, Minv


def warper(img, M, size):
    return cv.warpPerspective(img, M, size, flags=cv.INTER_NEAREST)


def draw_polygon(img, points, color):
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        cv.line(img, (int(x1), int(y1)), (int(x2), int(y2)), color, params.TRAPEZ_THICKNESS)


def draw_trapezoid(img):
    height, width = img.shape[:2]
    draw_polygon(img, params.WARP_SRC * np.float32([width, height]), params.TRAPEZ_COLOR)


def histogram(binary_warped):
    height = binary_warped.shape[0]
    return np.sum(binary_warped[height // 2:, :] // 255, axis=0)


def find_lane_pixels(binary_warped):
    height, width = binary_warped.shape
    hist = histogram(binary_warped)
    midpoint = width // 2
    leftx_current = int(np.argmax(hist[:midpoint]))
    rightx_current = int(np.argmax(hist[midpoint:])) + midpoint

    nonzeroy, nonzerox = binary_warped.nonzero()
    left_mask = np.zeros(len(nonzerox), dtype=bool)
    right_mask = np.zeros(len(nonzerox), dtype=bool)
    windows = []

    window_height = height // params.N_WINDOWS
    for window in range(params.N_WINDOWS):
        y_low = height - (window + 1) * window_height
        y_high = height - window * window_height
        in_rows = (nonzeroy >= y_low) & (nonzeroy < y_high)

        left_low = leftx_current - params.WINDOW_MARGIN
        left_high = leftx_current + params.WINDOW_MARGIN
        right_low = rightx_current - params.WINDOW_MARGIN
        right_high = rightx_current + params.WINDOW_MARGIN
        windows.append((left_low, y_low, left_high, y_high))
        windows.append((right_low, y_low, right_high, y_high))

        good_left = in_rows & (nonzerox >= left_low) & (nonzerox < left_high)
        good_right = in_rows & (nonzerox >= right_low) & (nonzerox < right_high)
        left_mask = left_mask | good_left
        right_mask = right_mask | good_right

        if good_left.sum() > params.WINDOW_MINPIX:
            leftx_current = int(np.mean(nonzerox[good_left]))
        if good_right.sum() > params.WINDOW_MINPIX:
            rightx_current = int(np.mean(nonzerox[good_right]))

    left = (nonzerox[left_mask], nonzeroy[left_mask])
    right = (nonzerox[right_mask], nonzeroy[right_mask])
    return left, right, windows


def fit_polynomial(pixels):
    x, y = pixels
    if len(x) < params.MIN_FIT_PIXELS:
        return None
    return np.polyfit(y, x, 2)


def fit_x(fit, y):
    a, b, c = fit
    return a * y * y + b * y + c


def draw_fit(img, fit, color):
    if fit is None:
        return
    height = img.shape[0]
    previous = None
    for y in range(0, height + params.CURVE_STEP, params.CURVE_STEP):
        point = (int(fit_x(fit, y)), y)
        if previous is not None:
            cv.line(img, previous, point, color, params.FIT_THICKNESS)
        previous = point


def draw_lane_fit(binary_warped, left, right, windows, left_fit, right_fit):
    out = cv.merge([binary_warped, binary_warped, binary_warped])
    out[left[1], left[0]] = params.LEFT_PIXELS_COLOR
    out[right[1], right[0]] = params.RIGHT_PIXELS_COLOR
    for x1, y1, x2, y2 in windows:
        cv.rectangle(out, (x1, y1), (x2, y2), params.WINDOW_COLOR, 2)
    draw_fit(out, left_fit, params.FIT_COLOR)
    draw_fit(out, right_fit, params.FIT_COLOR)
    return out


def draw_histogram(hist):
    height = params.HISTOGRAM_HEIGHT
    img = np.zeros((height, len(hist), 3), dtype='uint8')
    scale = (height - 1) / max(hist.max(), 1)
    for x in range(len(hist)):
        cv.line(img, (x, height - 1), (x, height - 1 - int(hist[x] * scale)), (255, 255, 255), 1)
    return img


def fit_in_meters(pixels):
    x, y = pixels
    if len(x) < params.MIN_FIT_PIXELS:
        return None
    return np.polyfit(y * params.YM_PER_PIX, x * params.XM_PER_PIX, 2)


def curvature_radius(fit_m, y_m):
    a, b, c = fit_m
    if a == 0:
        return float("inf")
    return (1 + (2 * a * y_m + b) ** 2) ** 1.5 / abs(2 * a)


def warp_point(M, x, y):
    px, py, pw = M @ np.float32([x, y, 1])
    return px / pw, py / pw


def vehicle_offset(left_fit, right_fit, M, roi_shape, warped_height):
    roi_height, roi_width = roi_shape[:2]
    car_x, car_y = warp_point(M, roi_width / 2, roi_height - 1)
    y = warped_height - 1
    lane_center = (fit_x(left_fit, y) + fit_x(right_fit, y)) / 2
    return (car_x - lane_center) * params.XM_PER_PIX


def draw_lane_area(shape, left_fit, right_fit):
    height, width = shape[:2]
    area = np.zeros((height, width, 3), dtype='uint8')
    for y in range(height):
        x_left = int(fit_x(left_fit, y))
        x_right = int(fit_x(right_fit, y))
        cv.line(area, (x_left, y), (x_right, y), params.LANE_AREA_COLOR, 1)
    return area


def overlay_lane(undistorted, lane_area, Minv):
    result = undistorted.copy()
    x1, y1, x2, y2 = roi_bounds(undistorted)
    roi = result[y1:y2, x1:x2]
    lane_roi = warper(lane_area, Minv, (x2 - x1, y2 - y1))
    result[y1:y2, x1:x2] = cv.addWeighted(roi, 1, lane_roi, params.LANE_AREA_WEIGHT, 0)
    return result


def is_departing(offset):
    return abs(offset) > params.DEPARTURE_THRESHOLD


def draw_warning(img):
    height, width = img.shape[:2]
    cv.rectangle(img, (0, 0), (width - 1, height - 1), params.WARNING_COLOR, params.WARNING_THICKNESS)
