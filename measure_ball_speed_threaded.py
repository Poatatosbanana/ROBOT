#!/usr/bin/env python3
#
# 9-tennis
#
# Threaded version of measure_ball_speed_stationary.py: one background
# thread captures both cameras back-to-back (as close together in time as
# possible) and detects on both, so a triangulated pair is always from two
# genuinely near-simultaneous frames - not two independently-timed ones.
# The main thread runs the Kalman filter's predict() at a fast, independent
# rate and only calls update() when a fresh synchronized pair is ready.
# Also reports how fast each of the two loops (main + the capture/detect
# thread) is actually running.
#
# Same camera/intrinsic/extrinsic placeholders as measure_ball_speed_stationary.py.
#
# Press a key (with a camera window focused) to exit.
#

import time
import threading
import cv2
import numpy as np
from filterpy.kalman import KalmanFilter
from example_ball_detector import BallDetector
from gretchen.camera import Camera

CAMERA_A_INDEX = 0
CAMERA_B_INDEX = 1

CAMERA_A_FX, CAMERA_A_FY, CAMERA_A_CX, CAMERA_A_CY = 1140.68, 1140.68, 319.5, 239.5
CAMERA_B_FX, CAMERA_B_FY, CAMERA_B_CX, CAMERA_B_CY = 1140.68, 1140.68, 319.5, 239.5

CAMERA_B_R = np.eye(3)
CAMERA_B_T = np.array([0.5, 0.0, 0.0])

ACCEL_NOISE = 500.0
MEASUREMENT_NOISE_STD = 0.02  # metres
PRINT_INTERVAL_S = 0.2


def make_projection_matrix(fx, fy, cx, cy, R, t):
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0, 0, 1]])
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt


def triangulate(P_a, P_b, point_a, point_b):
    pts_a = np.array([[point_a[0]], [point_a[1]]], dtype=np.float64)
    pts_b = np.array([[point_b[0]], [point_b[1]]], dtype=np.float64)
    point_4d = cv2.triangulatePoints(P_a, P_b, pts_a, pts_b)
    return (point_4d[:3, 0] / point_4d[3, 0])


class RateCounter:
    def __init__(self):
        self.count = 0
        self.last_check = time.time()

    def tick(self):
        self.count += 1

    def rate_and_reset(self):
        now = time.time()
        elapsed = now - self.last_check
        rate = self.count / elapsed if elapsed > 0 else 0.0
        self.count = 0
        self.last_check = now
        return rate


class CaptureDetectWorker:
    # Captures both cameras back-to-back and detects on both, every cycle,
    # so a stored pair always comes from two near-simultaneous frames.
    def __init__(self, camera_a_index, camera_b_index):
        self.camera_a = Camera(camera_a_index)
        self.camera_b = Camera(camera_b_index)
        self.detector_a = BallDetector()
        self.detector_b = BallDetector()

        self.lock = threading.Lock()
        self.latest_image_a = None
        self.latest_image_b = None
        self.latest_center_a = None
        self.latest_center_b = None
        self.latest_pair_id = 0  # incremented each successful capture cycle

        self.rate_counter = RateCounter()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.camera_a.start()
        self.camera_b.start()
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=2.0)

    def _run(self):
        while not self.stop_event.is_set():
            ret_a, img_a, ts_a = self.camera_a.getImage()
            ret_b, img_b, ts_b = self.camera_b.getImage()

            if not ret_a or img_a is None or not ret_b or img_b is None:
                continue

            (img_a, center_a) = self.detector_a.detect(img_a)
            (img_b, center_b) = self.detector_b.detect(img_b)

            with self.lock:
                self.latest_image_a = img_a
                self.latest_image_b = img_b
                self.latest_center_a = center_a
                self.latest_center_b = center_b
                self.latest_pair_id += 1
            self.rate_counter.tick()

    def get_latest(self):
        with self.lock:
            return (self.latest_image_a, self.latest_image_b,
                    self.latest_center_a, self.latest_center_b,
                    self.latest_pair_id)

    def get_rate(self):
        return self.rate_counter.rate_and_reset()


def make_kalman_filter():
    kf = KalmanFilter(dim_x=6, dim_z=3)
    kf.x = np.zeros(6)
    kf.P *= 500.0
    kf.H = np.zeros((3, 6))
    kf.H[0, 0] = kf.H[1, 1] = kf.H[2, 2] = 1.0
    kf.R = np.eye(3) * (MEASUREMENT_NOISE_STD ** 2)
    return kf


def set_process_model(kf, dt):
    F = np.eye(6)
    F[0, 3] = F[1, 4] = F[2, 5] = dt

    q = ACCEL_NOISE
    Q = np.zeros((6, 6))
    for i in range(3):
        Q[i, i] = q * dt**4 / 4
        Q[i, i + 3] = Q[i + 3, i] = q * dt**3 / 2
        Q[i + 3, i + 3] = q * dt**2

    kf.F = F
    kf.Q = Q


def main():
    print("Camera intrinsics/extrinsics are still PLACEHOLDERS - replace them "
          "with calibrate_stereo.py's printed output before trusting any "
          "speed number this prints.")

    worker = CaptureDetectWorker(CAMERA_A_INDEX, CAMERA_B_INDEX)
    worker.start()

    cv2.namedWindow("Camera A")
    cv2.namedWindow("Camera B")

    P_a = make_projection_matrix(CAMERA_A_FX, CAMERA_A_FY, CAMERA_A_CX, CAMERA_A_CY,
                                  np.eye(3), np.zeros(3))
    P_b = make_projection_matrix(CAMERA_B_FX, CAMERA_B_FY, CAMERA_B_CX, CAMERA_B_CY,
                                  CAMERA_B_R, CAMERA_B_T)

    kf = make_kalman_filter()
    initialized = False
    last_time = time.time()
    last_print = 0.0
    main_rate = RateCounter()
    last_used_pair_id = 0

    try:
        while True:
            now = time.time()
            dt = now - last_time
            last_time = now
            if dt > 0:
                set_process_model(kf, dt)
                kf.predict()
            main_rate.tick()

            img_a, img_b, center_a, center_b, pair_id = worker.get_latest()

            if img_a is not None:
                cv2.imshow("Camera A", img_a)
            if img_b is not None:
                cv2.imshow("Camera B", img_b)

            if (pair_id != last_used_pair_id
                    and center_a is not None and center_b is not None):
                point = triangulate(P_a, P_b, center_a, center_b)
                if not initialized:
                    kf.x[0:3] = point
                    initialized = True
                else:
                    kf.update(point)
                last_used_pair_id = pair_id

            if now - last_print > PRINT_INTERVAL_S:
                if initialized:
                    speed = float(np.linalg.norm(kf.x[3:6]))
                    speed_str = f"speed: {speed:.2f} m/s ({speed * 3.6:.1f} km/h)"
                else:
                    speed_str = "speed: (not initialized yet)"

                print(f"{speed_str}  |  loop rates (Hz) - "
                      f"main: {main_rate.rate_and_reset():.1f}  "
                      f"capture/detect: {worker.get_rate():.1f}")
                last_print = now

            key = cv2.waitKey(1)
            if key > 0:
                break
    finally:
        worker.stop()


if __name__ == '__main__':
    main()
