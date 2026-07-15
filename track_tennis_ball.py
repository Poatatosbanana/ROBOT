#!/usr/bin/env python3
#
# First Steps in Programming a Humanoid AI Robot
#
# Detect and mark a bright yellow-green tennis ball on a black background.
# (Detection + marking only -- this program does NOT move the robot.)
#
# Controls:
#   * Left-click the ball in the "Frame" window to sample its color and
#     re-center the HSV filter around it (the "click-to-pick" tuner).
#   * Use the sliders in the "Controls" window to refine the HSV bounds.
#   * Press 'm' to toggle the mask preview, 'b' to toggle the optional blur.
#   * Press 'q' or ESC to quit.
#
# Tuning tips for a black background:
#   * Keep the V (brightness) LOW slider fairly high (~80) so the black
#     backdrop is rejected. Raise S LOW to drop any gray clutter.
#   * Widen H only as far as needed to keep the ball stable between
#     yellow and green as the lighting changes.
#

import cv2
import time
import numpy as np

from tennis_ball_detector import TennisBallDetector
from gretchen.camera import Camera

#
# Device index / path for the camera.
#   Ubuntu/Linux: '/dev/grt_cam'   Mac: 0 or a /dev/cu.* path   Windows: 0
# example_2_detect_ball.py used Camera(2); change this to match your setup.
#
CAMERA_DEVICE = 2

CONTROLS_WIN = "Controls"
FRAME_WIN = "Frame"
MASK_WIN = "Mask"


def _noop(_):
    pass


def build_controls(detector):
    """Create the trackbar window, seeded from the detector's current bounds."""
    cv2.namedWindow(CONTROLS_WIN)
    lo, hi = detector.colorLower, detector.colorUpper
    cv2.createTrackbar("H low",  CONTROLS_WIN, int(lo[0]), 179, _noop)
    cv2.createTrackbar("H high", CONTROLS_WIN, int(hi[0]), 179, _noop)
    cv2.createTrackbar("S low",  CONTROLS_WIN, int(lo[1]), 255, _noop)
    cv2.createTrackbar("S high", CONTROLS_WIN, int(hi[1]), 255, _noop)
    cv2.createTrackbar("V low",  CONTROLS_WIN, int(lo[2]), 255, _noop)
    cv2.createTrackbar("V high", CONTROLS_WIN, int(hi[2]), 255, _noop)


def read_controls():
    """Read all six sliders back into (lower, upper) tuples."""
    lower = (cv2.getTrackbarPos("H low",  CONTROLS_WIN),
             cv2.getTrackbarPos("S low",  CONTROLS_WIN),
             cv2.getTrackbarPos("V low",  CONTROLS_WIN))
    upper = (cv2.getTrackbarPos("H high", CONTROLS_WIN),
             cv2.getTrackbarPos("S high", CONTROLS_WIN),
             cv2.getTrackbarPos("V high", CONTROLS_WIN))
    return lower, upper


def write_controls(lower, upper):
    """Push (lower, upper) back onto the sliders (after a click-to-pick)."""
    cv2.setTrackbarPos("H low",  CONTROLS_WIN, int(lower[0]))
    cv2.setTrackbarPos("S low",  CONTROLS_WIN, int(lower[1]))
    cv2.setTrackbarPos("V low",  CONTROLS_WIN, int(lower[2]))
    cv2.setTrackbarPos("H high", CONTROLS_WIN, int(upper[0]))
    cv2.setTrackbarPos("S high", CONTROLS_WIN, int(upper[1]))
    cv2.setTrackbarPos("V high", CONTROLS_WIN, int(upper[2]))


# Shared handle so the mouse callback can reach the detector + latest frame.
class _State:
    detector = None
    frame = None


def on_mouse(event, x, y, flags, param):
    """Left-click samples the pixel under the cursor and re-centers HSV."""
    if event == cv2.EVENT_LBUTTONDOWN and _State.frame is not None:
        h, s, v = _State.detector.sample_pixel(_State.frame, x, y)
        write_controls(_State.detector.colorLower, _State.detector.colorUpper)
        print("Sampled HSV at ({}, {}): H={} S={} V={} -> bounds {} .. {}"
              .format(x, y, h, s, v,
                      tuple(int(c) for c in _State.detector.colorLower),
                      tuple(int(c) for c in _State.detector.colorUpper)))


def main():
    camera = Camera(CAMERA_DEVICE)
    camera.start()

    detector = TennisBallDetector(
        color_lower=(25, 60, 80),    # optic-yellow -> green, high V floor
        color_upper=(70, 255, 255),
        process_scale=0.5,           # detect at half-res for speed
        use_blur=False,              # black bg is clean; blur usually not needed
        use_morph=True,              # single MORPH_OPEN pass
    )

    _State.detector = detector

    cv2.namedWindow(FRAME_WIN)
    cv2.setMouseCallback(FRAME_WIN, on_mouse)
    build_controls(detector)

    show_mask = True
    prev_t = time.time()
    fps = 0.0

    while True:
        ret, img, timestamp = camera.getImage()
        if not ret or img is None:
            continue

        # Keep the newest full-res frame available to the click callback.
        _State.frame = img.copy()

        # Sliders override the bounds every frame (click-to-pick writes to them).
        lower, upper = read_controls()
        detector.set_bounds(lower, upper)

        frame, center, radius, mask = detector.detect(img)

        # --- FPS estimate (smoothed) ---
        now = time.time()
        dt = now - prev_t
        prev_t = now
        if dt > 0:
            fps = 0.9 * fps + 0.1 * (1.0 / dt)

        # HUD
        cv2.putText(frame, "FPS: {:4.1f}".format(fps), (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if center is not None:
            cv2.putText(frame, "ball: {} r={}".format(center, radius),
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow(FRAME_WIN, frame)
        if show_mask:
            cv2.imshow(MASK_WIN, mask)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):        # q or ESC
            break
        elif key == ord('m'):
            show_mask = not show_mask
            if not show_mask:
                cv2.destroyWindow(MASK_WIN)
        elif key == ord('b'):
            detector.use_blur = not detector.use_blur
            print("blur:", detector.use_blur)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
