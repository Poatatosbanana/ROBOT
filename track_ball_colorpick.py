#!/usr/bin/env python3
#
# 9-tennis
#
# Single-camera ball tracking with a click-to-choose color picker.
# Left-click the ball in the camera window to set the HSV color filter from
# whatever you clicked on. No motor control, no stereo, no speed math -- just
# detect-and-mark plus live color selection.
#
# Controls:
#   * Left-click the ball  -> set the color filter from the clicked point
#   * 'q' or ESC           -> quit
#
# Why the color space matters (do not "fix" this):
#   BallDetector.detect() builds its mask with cv2.cvtColor(frame,
#   COLOR_RGB2HSV) even though the camera delivers BGR. The picker below uses
#   the *same* COLOR_RGB2HSV call, so the color you click lands in the exact
#   same HSV space the mask thresholds. Matching the detector is what makes the
#   clicked color equal the filtered color -- consistency beats "correctness".
#

import cv2
import numpy as np

from example_ball_detector import BallDetector
from gretchen.camera import Camera

# Camera device index. ROBOT's stereo scripts use 0 (camera A) and 1 (camera B);
# the first-steps course examples used 2. Change this to match your setup.
CAMERA_INDEX = 0

FRAME_WIN = "Ball tracking (left-click the ball to set its color)"


def pick_color_from_click(detector, frame, x, y,
                          patch=9, h_margin=12, s_margin=70, v_margin=70,
                          s_floor=40, v_floor=80):
    """Sample the color under a click and update `detector`'s HSV filter.

    Samples a `patch` x `patch` window centered on (x, y), takes the median
    H/S/V (robust to a hot pixel or a specular highlight), and centers the
    detector's colorLower/colorUpper on it with fixed margins. A high V (and
    S) floor keeps a dark/black background from leaking back into the mask.

    IMPORTANT: converts with COLOR_RGB2HSV to match BallDetector.detect(), so
    the sampled color lands in the same HSV space the mask is thresholded in.

    Returns (h, s, v, lower, upper) for logging / on-screen feedback.
    """
    h_img, w_img = frame.shape[:2]

    # Clamp a patch-sized window to the frame so clicks near the border are safe.
    half = patch // 2
    x0 = max(x - half, 0)
    y0 = max(y - half, 0)
    x1 = min(x + half + 1, w_img)
    y1 = min(y + half + 1, h_img)

    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)   # match BallDetector.detect()
    region = hsv[y0:y1, x0:x1].reshape(-1, 3)

    # Median per channel: the robust "average" of the patch.
    h, s, v = [int(c) for c in np.median(region, axis=0)]

    lower = (int(np.clip(h - h_margin, 0, 179)),
             max(s - s_margin, s_floor),
             max(v - v_margin, v_floor))
    upper = (int(np.clip(h + h_margin, 0, 179)), 255, 255)

    detector.colorLower = lower
    detector.colorUpper = upper
    return h, s, v, lower, upper


# Small holder so the mouse callback can reach the detector and the newest frame.
class _State:
    detector = None
    frame = None


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and _State.frame is not None:
        h, s, v, lower, upper = pick_color_from_click(_State.detector, _State.frame, x, y)
        print("Clicked ({}, {}) -> HSV median ({}, {}, {}) | filter {} .. {}"
              .format(x, y, h, s, v, lower, upper))


def main():
    camera = Camera(CAMERA_INDEX)
    camera.start()

    detector = BallDetector()
    _State.detector = detector

    cv2.namedWindow(FRAME_WIN)
    cv2.setMouseCallback(FRAME_WIN, on_mouse)

    while True:
        ret, img, timestamp = camera.getImage()
        if not ret or img is None:
            continue

        # Keep the newest frame available to the click callback (copy so the
        # callback samples a stable image, not one being drawn on).
        _State.frame = img.copy()

        # detect() marks all detected circles on `img` and returns the largest.
        img, center = detector.detect(img)

        # HUD
        cv2.putText(img, "left-click the ball to set its color; q/ESC to quit",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, "filter {} .. {}".format(tuple(detector.colorLower),
                                                  tuple(detector.colorUpper)),
                    (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        if center is not None:
            cv2.putText(img, "ball @ {}".format(center),
                        (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow(FRAME_WIN, img)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):   # q or ESC
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
