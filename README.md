# DOSiVuAV-Projekat
Projekat iz DOSiVuAV - Uros Radojevic

---

**Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: output_images/undistort_chessboard.jpg "Undistorted"
[image2]: output_images/undistort_test1.jpg "Road Transformed"
[image3]: output_images/binary_test1.jpg "Binary Example"
[image3b]: output_images/binary_components_test1.jpg "Binary Components"
[image4]: output_images/warped_straight_lines.jpg "Warp Example"
[image4b]: output_images/warped_test1.jpg "Warped Binary"
[image5]: output_images/color_fit_lines_test2.jpg "Fit Visual"
[image6]: output_images/example_output_test2.jpg "Output"
[image6c]: output_images/example_output_warning.jpg "Output Warning"
[video1]: output_videos/project_video03.mp4 "Video"

---

## Important info

- Python3 version used for running code is 3.13.15.
- After pulling code from remote it is important to run the code before taking a look at this documentation because source pictures are generated after initial run.

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.

You're reading it!

All the code is in the `src/` folder. The main pipeline is in `main.py`, the individual steps are in `lane.py`, camera calibration is in `calibration.py` and all the parameters are collected in `params.py`. The program is started with:

```
python3 src/main.py <image or video>
```

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The calibration code is in `src/calibration.py`.

Calibration is done with the chessboard images from the `camera_cal/` folder. The idea is simple: the exact shape of a chessboard is known (a regular grid of squares), so by comparing that ideal grid with where the corners actually appear in the image, it is possible to determine how the camera lens bends the picture.

On every image, the inner corners of the board are detected with `cv.findChessboardCorners` and their positions are refined with `cv.cornerSubPix`. The found corners, together with the ideal grid positions, are passed to `cv.calibrateCamera`, which returns the camera matrix and the distortion coefficients. On a few images the board is partly cut off, so those are skipped. The result is saved to a file, so calibration only has to be done once (if the file is missing, it runs automatically when the program starts).

With these values any image from the same camera can be corrected using `cv.undistort`. On the example below, the edges of the board are visibly curved in the original image and straight after the correction:

![alt text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Distortion correction is the first step applied to every frame. Here is an example on one of the test images (left original, right corrected). The change is subtle in the middle of the image and most visible near the edges, for example on the car on the right and the hood at the bottom:

![alt text][image2]

The calibration only applies to the camera it was made for, so the smaller videos recorded with a different camera are processed without this correction.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code is in `src/lane.py` (`sobel_x_binary()`, `color_binary()`).

Before thresholding, the lower part of the frame where the road is (region of interest) is cropped. This removes the sky, the trees and the hood of the car, which would otherwise only add noise.

The binary image is a combination of two masks:

- **Gradient mask.** The image is converted to grayscale and the Sobel operator is applied in the x direction. Lane lines are mostly vertical in the image, so the change in brightness from left to right is strong at their edges. Pixels with a strong enough gradient are kept.
- **Color mask.** The image is converted to the HSV color space and only pixels that are either yellow or white are kept. HSV is convenient here because the color (hue) stays almost the same in sunlight and in shadow, only the brightness changes.

The two masks are combined, so a pixel is part of the binary image if either of them detects it. They complement each other well: on light concrete the yellow line has almost the same brightness as the road, so the gradient barely sees it while the color mask finds it easily. On the other hand, lines in shadow are often too dark for the color mask but still have a clear edge.

Here is the contribution of each mask (green is the gradient, blue is the color mask):

![alt text][image3b]

And the final binary image:

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code is in `src/lane.py` (`perspective_matrices()`, `warper()`), and the points are defined in `src/params.py`.

To get a view of the road from above, four points are chosen on an image with a straight road: a trapezoid that follows the two lane lines from close to the car up to just below the horizon. These points are mapped to a rectangle, so that the lane lines become vertical and parallel. The points used for the 1280×720 images are:

| Source        | Destination   |
|:-------------:|:-------------:|
| 585, 460      | 320, 0        |
| 701, 460      | 960, 0        |
| 1050, 684     | 960, 720      |
| 257, 684      | 320, 720      |

The transformation matrix is computed with `cv.getPerspectiveTransform` and applied with `cv.warpPerspective`. The inverse matrix is also computed (by swapping source and destination points), which is used later to draw the result back onto the original image.

The transform was checked on the straight road images: if the points are chosen well, the lines in the warped image should be straight and parallel, which is the case here:

![alt text][image4]

An example on a test image, the warped frame and the warped binary image. Most of the noise from the sides of the road is gone and mostly just the two lane lines remain:

![alt text][image4b]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The code is in `src/lane.py` (`histogram()`, `find_lane_pixels()`, `fit_polynomial()`).

**Finding the starting points.** In the bird's-eye view the lane lines are almost vertical near the car. Counting the white pixels in each column of the lower half of the image gives a histogram with two clear peaks, one for each line. The highest peak on the left side is where the left line starts, and the highest peak on the right side is where the right line starts. Since the color mask is more reliable than the gradient (the gradient also reacts to the texture of the road), color pixels count more in this histogram.

**Sliding windows.** Starting from these two positions, a small window is moved up the image, one step at a time. All white pixels inside the window are assigned to that line. If enough pixels are found, the next window is centered on their average position, so the windows follow the line even when it curves.

**Fitting the polynomial.** A second order polynomial is fitted through the pixels of each line:

x = A·y² + B·y + C

The polynomial is written as x in terms of y (and not the other way around) because the lines are close to vertical, so for each height y there is exactly one x position of the line. The coefficients have a simple meaning: C is the position of the line, B is its direction (slope) and A describes how much it bends. For a straight line A is zero.

Here is an example on a curved road. Green rectangles are the sliding windows, red and blue are the pixels of the left and right line, the yellow curves are the fitted polynomials and the histogram is shown below:

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The code is in `src/lane.py` (`curvature_radius()`, `vehicle_offset()`).

**From pixels to meters.** The polynomial is in pixels, but the radius should be in meters. In the bird's-eye view, the lane is 640 pixels wide, and a standard highway lane is 3.7 m wide, which gives the horizontal scale. The vertical scale comes from the dashed lane lines: one dash and the gap after it are 12 m long in total (3 m + 9 m), which tells how many meters of road one pixel represents vertically. With these two scales, the polynomial coefficients are converted to meters.

**Radius of curvature.** The radius of curvature at some point of a curve is the radius of the circle that best fits the curve at that point. A sharp curve has a small radius, a gentle curve has a large one, and a straight road has an infinite radius. For the curve x = A·y² + B·y + C it is computed from the first and second derivative:

x' = 2·A·y + B, x'' = 2·A

R = (1 + x'²)^(3/2) / |x''| = (1 + (2·A·y + B)²)^(3/2) / |2·A|

It is evaluated at the bottom of the image, which is the closest point to the car, and the average of the left and right line is shown.

**Vehicle position.** The camera is mounted in the middle of the car, so the middle of the image represents the position of the car. The center of the lane is halfway between the left and right line at the bottom of the image. The difference between these two positions, converted to meters, is how far the car is from the center of the lane, and its sign tells whether the car is to the left or to the right.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The code is in `src/lane.py` (`draw_lane_area()`, `overlay_lane()`, `draw_info()`).

In an empty bird's-eye image, the area between the two fitted lines is filled with green. This image is then warped back to the original perspective with the inverse matrix and blended over the frame. The radius of curvature and the position of the car are written in the top left corner:

![alt text][image6]

There is also a lane departure warning: if the car moves more than 0.5 m away from the center of the lane, a red frame is drawn around the image together with a warning message. 0.5 m was chosen so that the warning appears before the wheels actually reach the line:

![alt text][image6c]

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to the video result](output_videos/project_video03.mp4)

Results for the other videos are in the `output_videos/` folder.

In videos, frames are not processed completely independently, because consecutive frames are very similar and this can be used (the code is in `src/tracker.py`):

- **Searching around the previous lines.** Once the lines are found, in the next frame pixels are searched for only close to where the lines were before, instead of starting from scratch with the histogram and windows. This is faster and noise far from the lines cannot pull them away.
- **Sanity check.** A new result is accepted only if it makes sense: the lane has a realistic width, the two lines are roughly parallel, and they did not jump too far compared to the previous frame.
- **Falling back to previous lines.** If a frame does not pass the check, the lines from the previous frames are used instead. If this happens several frames in a row, the search starts again from the histogram.
- **Smoothing.** The displayed lines are the average of the last few accepted frames, so the drawn lane and the numbers don't jitter from frame to frame.

There is also a debug mode (`python3 src/main.py --debug <video>`) which shows all the intermediate steps, which was very useful for finding out where the pipeline goes wrong.

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The pipeline works well on the project video, but there are several situations where it struggles:

- **Different cameras and resolutions.** The calibration and the perspective points were made for the camera that recorded the 1280×720 images. The 960×540 videos come from a different camera, so their distortion is not corrected at all, and the bird's-eye view uses points that do not quite match their road geometry. They still get a reasonable result, but the curvature is less accurate. A separate distortion correction matrix and a separate perspective transform for the 960×540 resolution (chosen on one of its straight road images) would fix this. The same applies to any differently mounted camera.
- **Things that look like lane lines.** Road texture, concrete barriers and seams between two kinds of asphalt create strong vertical edges, which the gradient mask picks up. On one of the test images this pulled the right line to the wrong place, which was solved by giving the color mask more weight when looking for the starting points of the lines. Seams in the asphalt can still confuse the pipeline. Using gradient pixels only where the color mask also finds something nearby would help.
- **Lighting.** In strong sunlight the asphalt becomes almost white and passes the white color threshold, while in deep shadows the lines become too dark for it. Thresholds that adapt to the brightness of the frame would make this more robust.
- **Sharp curves.** The perspective transform and the second order polynomial work well for gentle highway curves. On a winding road (like the hardest challenge video) the lines leave the area covered by the transform, and a single parabola cannot describe an S-shaped curve. A shorter region of interest would help with this.
- **Assumptions.** The position of the car assumes the camera is exactly in the middle of the car, and the conversion to meters assumes standard lane width and dash length. If these are not true, the numbers are off by a bit, and the lane departure warning can trigger a little too early or too late. The warning could also be improved by taking into account the direction the car is heading relative to the lane, not only its position.


