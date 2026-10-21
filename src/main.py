# Pokretanje: python3 src/main.py [putanja do slike ili videa]
import sys

import cv2 as cv

import params
import lane


def process_frame(frame):
    gray = lane.to_gray(frame)
    roi = lane.crop_roi(gray)
    show_phase(frame, gray, roi)


def show_phase(frame, gray, roi):
    preview = frame.copy()
    x1, y1, x2, y2 = lane.roi_bounds(frame)
    cv.rectangle(preview, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 2)
    cv.imshow("Original + ROI", preview)
    cv.imshow("Gray", gray)
    cv.imshow("ROI", roi)


def run_image(path):
    img = cv.imread(path)
    if img is None:
        print("Ne mogu da ucitam sliku:", path)
        return
    process_frame(img)
    cv.waitKey(0)


def run_video(path):
    cap = cv.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        process_frame(frame)
        if cv.waitKey(params.FRAME_DELAY) & 0xFF == 27: break  # ESC prekida
    cap.release()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else params.DEFAULT_INPUT
    if path.endswith(".mp4"):
        run_video(path)
    else:
        run_image(path)
    cv.destroyAllWindows()
