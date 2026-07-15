#
# First Steps in Programming a Humanoid AI Robot
#
# Tennis ball detector class (yellow-green ball on a black background)
# Used by track_tennis_ball.py
#
# This is an optimized variant of example_ball_detector.BallDetector.
# Key differences and why they matter:
#
#   * COLOR_BGR2HSV (not RGB2HSV): cv2.VideoCapture returns BGR frames.
#     The original example used COLOR_RGB2HSV, which swaps R/B and shifts
#     hue. That "works" for green (center of the range) but is wrong for
#     yellow-green. Tune your HSV values against this BGR2HSV conversion.
#
#   * Black background => gate hard on Value (brightness) and Saturation,
#     and keep Hue forgiving. Black is low-V/low-S, so a high V floor
#     removes the entire backdrop in one step and the ball is almost the
#     only bright, saturated blob left.
#
#   * No bilateral filter: bilateralFilter(d=15) was the most expensive op
#     in the original pipeline. On a clean black background it is
#     unnecessary; a small Gaussian blur (optional) is ~10-50x cheaper.
#
#   * Process at reduced resolution (process_scale): 4x fewer pixels at
#     0.5 scale. Coordinates are scaled back to full resolution on return.
#
#   * Single MORPH_OPEN instead of erode(iter=3)+dilate(iter=2): removes
#     speckle in one pass with no net change in blob size, and is easier
#     to reason about than several asymmetric 3x3 passes.
#
#   * Relaxed circularity when the ball is moving: motion blur turns a
#     rolling ball into an ellipse that fails a strict circularity test,
#     so we lower the threshold once we detect motion.
#
#   * Light constant-velocity predictor: bridges 1-2 dropped frames and
#     gives a "lead" estimate, which helps at ~10 fps where the ball can
#     jump ~20 cm between frames.
#

import numpy as np
import cv2
import imutils


class TennisBallDetector:
    PI = 3.141592

    def __init__(self,
                 color_lower=(25, 60, 80),
                 color_upper=(70, 255, 255),
                 process_scale=0.5,
                 min_radius=10,
                 circularity_strict=0.70,
                 circularity_moving=0.45,
                 use_blur=False,
                 use_morph=True):
        #
        # HSV bounds (H: 0..179, S: 0..255, V: 0..255) for a BGR2HSV image.
        # Defaults target optic-yellow -> green with a high V floor so the
        # black background is rejected outright. Tune with the trackbars.
        #
        self.colorLower = np.array(color_lower, dtype=np.uint8)
        self.colorUpper = np.array(color_upper, dtype=np.uint8)

        # Detection is run on a downscaled copy for speed.
        self.process_scale = float(process_scale)

        # Ignore anything smaller than this (radius, in full-res pixels).
        self.min_radius = min_radius

        # Circularity = contourArea / enclosingCircleArea.
        # Strict when the ball is still; relaxed when it is moving/blurred.
        self.circularity_strict = circularity_strict
        self.circularity_moving = circularity_moving
        self.moving_speed_px = 6.0   # px/frame (full-res) that counts as "moving"

        self.use_blur = use_blur     # optional cheap denoise before HSV
        self.use_morph = use_morph   # single MORPH_OPEN to kill speckle

        # 5x5 elliptical structuring element, built once and reused.
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        # --- Predictor state (constant-velocity) ---
        self.prev_center = None      # last measured center (full-res)
        self.velocity = (0.0, 0.0)   # px/frame, smoothed
        self.misses = 0              # consecutive frames without a detection

    # ------------------------------------------------------------------ #
    #  Runtime tuning helpers (used by the trackbar / click callbacks)
    # ------------------------------------------------------------------ #
    def set_bounds(self, lower, upper):
        self.colorLower = np.array(lower, dtype=np.uint8)
        self.colorUpper = np.array(upper, dtype=np.uint8)

    def sample_pixel(self, bgr_frame, x, y, h_margin=12, s_margin=70, v_margin=70):
        """Sample the pixel at (x, y) on a full-res BGR frame and center the
        HSV bounds around it. Great for click-to-pick tuning. The V floor is
        kept high so the black background stays rejected."""
        hsv = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2HSV)
        h, s, v = [int(c) for c in hsv[y, x]]
        lower = (max(h - h_margin, 0),
                 max(s - s_margin, 40),
                 max(v - v_margin, 60))
        upper = (min(h + h_margin, 179), 255, 255)
        self.set_bounds(lower, upper)
        return (h, s, v)

    # ------------------------------------------------------------------ #
    #  Core detection
    # ------------------------------------------------------------------ #
    def _build_mask(self, frame):
        """Return (mask, inv_scale) where mask is the binary ball mask on the
        downscaled image and inv_scale maps its coords back to full-res."""
        scale = self.process_scale
        if scale != 1.0:
            small = cv2.resize(frame, None, fx=scale, fy=scale,
                               interpolation=cv2.INTER_AREA)
        else:
            small = frame

        if self.use_blur:
            small = cv2.GaussianBlur(small, (5, 5), 0)

        # Camera frames are BGR -> convert with BGR2HSV (see class docstring).
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.colorLower, self.colorUpper)

        if self.use_morph:
            # One opening pass: erosion then dilation with the same kernel,
            # so speckle is removed without shrinking the ball.
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)

        return mask, (1.0 / scale)

    def detect(self, frame):
        """Detect the tennis ball and annotate `frame` in place.

        Returns:
            frame     : the annotated BGR frame
            center    : (x, y) of the best ball in full-res px, or None
            radius    : radius in full-res px, or 0
            mask      : the binary mask (downscaled) for preview/debugging
        """
        mask, inv_scale = self._build_mask(frame)

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)

        # Relax circularity if the ball was moving fast last frame.
        speed = (self.velocity[0] ** 2 + self.velocity[1] ** 2) ** 0.5
        circ_thresh = (self.circularity_moving
                       if speed >= self.moving_speed_px
                       else self.circularity_strict)

        circles = []
        max_radius = 0
        max_center = None
        for cnt in cnts:
            contour_area = cv2.contourArea(cnt)
            ((x, y), radius) = cv2.minEnclosingCircle(cnt)

            # Scale back to full-res coordinates.
            x *= inv_scale
            y *= inv_scale
            radius *= inv_scale

            if radius < self.min_radius:
                continue

            circle_area = self.PI * radius * radius
            # contour_area was measured on the small image; compare on equal
            # footing by scaling the circle area down, or the contour up.
            contour_area_full = contour_area * (inv_scale ** 2)
            if circle_area <= 0:
                continue

            if contour_area_full / circle_area > circ_thresh:
                center = (int(x), int(y))
                circles.append((center, int(radius)))
                if radius > max_radius:
                    max_radius = radius
                    max_center = center

        # --- Update the constant-velocity predictor ---
        predicted = self._update_predictor(max_center)

        # Draw all accepted circles (measured detections) in yellow.
        for (center, radius) in circles:
            cv2.circle(frame, center, radius, (0, 255, 255), 2)
            cv2.circle(frame, center, 3, (0, 255, 255), -1)

        # If we missed this frame but have a recent track, draw the predicted
        # position in orange so you can see the "ghost" the predictor supplies.
        if max_center is None and predicted is not None:
            cv2.circle(frame, predicted, 6, (0, 140, 255), 2)
            cv2.putText(frame, "predicted", (predicted[0] + 8, predicted[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 140, 255), 1)

        report_center = max_center if max_center is not None else predicted
        return frame, report_center, max_radius, mask

    def _update_predictor(self, measured_center):
        """Smoothly track velocity; return a predicted center when a frame is
        missed. Returns the predicted position (or None if no track yet)."""
        if measured_center is not None:
            if self.prev_center is not None:
                vx = measured_center[0] - self.prev_center[0]
                vy = measured_center[1] - self.prev_center[1]
                # Exponential smoothing to damp jitter.
                a = 0.5
                self.velocity = (a * vx + (1 - a) * self.velocity[0],
                                 a * vy + (1 - a) * self.velocity[1])
            self.prev_center = measured_center
            self.misses = 0
            return measured_center

        # Missed detection: coast along the last known velocity, but only for
        # a couple of frames before giving up (the ball likely left the frame).
        self.misses += 1
        if self.prev_center is not None and self.misses <= 2:
            px = int(self.prev_center[0] + self.velocity[0])
            py = int(self.prev_center[1] + self.velocity[1])
            self.prev_center = (px, py)
            return (px, py)

        # Track lost.
        self.prev_center = None
        self.velocity = (0.0, 0.0)
        return None
