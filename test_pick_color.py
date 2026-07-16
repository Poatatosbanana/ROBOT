#!/usr/bin/env python3
#
# 9-tennis
#
# Camera-free test for pick_color_from_click (track_ball_colorpick.py).
# Builds synthetic frames -- no camera, no motor -- and checks that:
#   1. clicking a bright-yellow ball on black sets a filter that then detects
#      that ball at the expected location, and
#   2. clicking the black background does NOT produce a filter that selects the
#      whole frame (the V/S floors must reject the dark background).
#
# The detector and the picker both convert with COLOR_RGB2HSV, so this also
# guards that the picker stays consistent with BallDetector's mask.
#
# Run:  python test_pick_color.py
#

import cv2
import numpy as np

# BallDetector.detect() calls cv2.imshow("Filter", ...); stub it so the test
# needs no display (and pops no windows).
cv2.imshow = lambda *a, **k: None

from example_ball_detector import BallDetector
from track_ball_colorpick import pick_color_from_click


def make_yellow_ball_frame(center, radius=50, size=(480, 640)):
    """Black BGR frame with a filled bright-yellow (BGR 0,255,255) ball."""
    frame = np.zeros((size[0], size[1], 3), dtype=np.uint8)
    cv2.circle(frame, center, radius, (0, 255, 255), -1)
    return frame


def test_click_ball_detects_it():
    cx, cy = 400, 240
    frame = make_yellow_ball_frame((cx, cy))
    det = BallDetector()

    pick_color_from_click(det, frame, cx, cy)

    _, center = det.detect(frame.copy())
    assert center is not None, "ball not detected after clicking it"
    dx, dy = abs(center[0] - cx), abs(center[1] - cy)
    assert dx <= 5 and dy <= 5, "detected center {} too far from {}".format(center, (cx, cy))
    print("PASS: click on ball -> detected at {} (target {})".format(center, (cx, cy)))


def test_click_background_selects_nothing():
    cx, cy = 400, 240
    frame = make_yellow_ball_frame((cx, cy))
    det = BallDetector()

    # Click a black background pixel far from the ball.
    pick_color_from_click(det, frame, 20, 20)

    # Reproduce the detector's masking to measure coverage.
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, det.colorLower, det.colorUpper)
    coverage = float(np.count_nonzero(mask)) / mask.size
    assert coverage < 0.01, "background click selected {:.1%} of the frame".format(coverage)

    _, center = det.detect(frame.copy())
    assert center is None, "background click should not detect a ball, got {}".format(center)
    print("PASS: click on background -> mask coverage {:.3%}, no ball".format(coverage))


if __name__ == '__main__':
    test_click_ball_detects_it()
    test_click_background_selects_nothing()
    print("All tests passed.")
