import cv2
import numpy as np

DEFAULT_INPUT = "course_files_export/Zadatak/test_images/test1.jpg"


ROI_TOP = 0.59      
ROI_BOTTOM = 0.95   
ROI_LEFT = 0.00     
ROI_RIGHT = 1.00    

FRAME_DELAY = 30

GAUSS_KERNEL = (5, 5)

CANNY_LOW = 120
CANNY_HIGH = 170

HOUGH_RHO = 1
HOUGH_THETA = np.pi / 180
HOUGH_THRESHOLD = 50

HORIZONTAL_MARGIN = np.pi / 180 * 20

LEFT_LANE_COLOR = (255, 0, 0)
RIGHT_LANE_COLOR = (0, 255, 0)
LANE_THICKNESS = 3

CALIB_IMAGES = "course_files_export/Zadatak/camera_cal/calibration*.jpg"
CALIB_FILE = "calib.npz"
CHESS_COLS = 9
CHESS_ROWS = 6
CALIB_SHOW_DELAY = 200

SOBEL_KSIZE = 3
SOBEL_LOW = 50
SOBEL_HIGH = 255

YELLOW_LOW = np.array([15, 100, 120])
YELLOW_HIGH = np.array([35, 255, 255])

WHITE_LOW = np.array([0, 0, 200])
WHITE_HIGH = np.array([180, 30, 255])

EXAMPLES_DIR = "output_images"

WARP_SRC = np.float32([[0.457, 0.639], [0.548, 0.639], [0.820, 0.950], [0.201, 0.950]])
WARP_DST = np.float32([[0.25, 0.0], [0.75, 0.0], [0.75, 1.0], [0.25, 1.0]])
TRAPEZ_COLOR = (0, 0, 255)
TRAPEZ_THICKNESS = 2

COLOR_HIST_WEIGHT = 3

N_WINDOWS = 9
WINDOW_MARGIN = 100
WINDOW_MINPIX = 50
MIN_FIT_PIXELS = 100

WINDOW_COLOR = (0, 255, 0)
LEFT_PIXELS_COLOR = (0, 0, 255)
RIGHT_PIXELS_COLOR = (255, 0, 0)
FIT_COLOR = (0, 255, 255)
FIT_THICKNESS = 3
CURVE_STEP = 10
HISTOGRAM_HEIGHT = 200

LANE_WIDTH_M = 3.7
VISIBLE_LENGTH_M = 24

LANE_AREA_COLOR = (0, 255, 0)
LANE_AREA_WEIGHT = 0.3

DEPARTURE_THRESHOLD = 0.5
WARNING_COLOR = (0, 0, 255)
WARNING_THICKNESS = 20
PRINT_EVERY = 25

DEBUG_WIDTH = 640
DEBUG_HEIGHT = 360

SEARCH_MARGIN = 80
HISTORY_LENGTH = 5
MAX_BAD_FRAMES = 5
MIN_WIDTH_RATIO = 0.75
MAX_WIDTH_RATIO = 1.25
MAX_WIDTH_DIFF_RATIO = 0.25
MAX_SHIFT_RATIO = 0.15

OUTPUT_VIDEOS_DIR = "output_videos"
TEST_VIDEOS_DIR = "course_files_export/Zadatak/test_videos"

WRITEUP_IMAGES = [
    "undistort_chessboard.jpg",
    "undistort_test1.jpg",
    "binary_test1.jpg",
    "binary_components_test1.jpg",
    "warped_straight_lines.jpg",
    "warped_test1.jpg",
    "color_fit_lines_test2.jpg",
    "example_output_test2.jpg",
    "example_output_warning.jpg",
]
WRITEUP_VIDEOS = ["project_video03.mp4"]

TEXT_FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 1.0
TEXT_THICKNESS = 2
TEXT_COLOR = (255, 255, 255)
TEXT_OUTLINE_COLOR = (0, 0, 0)
TEXT_ORIGIN = (30, 50)
TEXT_LINE_HEIGHT = 45
STRAIGHT_RADIUS = 3000
