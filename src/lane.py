import cv2 as cv

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


def crop_roi(gray):
    # Korak 1b: isecanje ROI-a (donji deo kadra gde je put)
    x1, y1, x2, y2 = roi_bounds(gray)
    return gray[y1:y2, x1:x2]
