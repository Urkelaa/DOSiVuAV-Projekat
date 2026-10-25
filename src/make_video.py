import os
import sys

import cv2 as cv

import params
import calibration
import tracker as tracking
import main


def make_video(input_path, output_path, calib):
    os.makedirs(params.OUTPUT_VIDEOS_DIR, exist_ok=True)
    cap = cv.VideoCapture(input_path)
    fps = cap.get(cv.CAP_PROP_FPS)
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    writer = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    tracker = tracking.new_tracker()
    frame_index = 0
    warnings = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        result, info, debug = main.process_frame(frame, calib, tracker)
        if info is not None and main.lane.is_departing(info[2]):
            warnings += 1
        writer.write(result)
        frame_index += 1
    cap.release()
    writer.release()
    print("Sacuvano:", output_path, "| kadrova:", frame_index, "| kadrova sa upozorenjem:", warnings)


if __name__ == "__main__":
    calib = calibration.load_calibration()
    for input_path in sys.argv[1:]:
        name = input_path.split("/")[-1]
        make_video(input_path, params.OUTPUT_VIDEOS_DIR + "/" + name, calib)
