#!/usr/bin/env python3
#
# 9-tennis
#
# Helps tune ACCEL_NOISE for measure_ball_speed_stationary.py's Kalman
# filter. Unlike MEASUREMENT_NOISE_STD, this isn't a single clean
# measurement - instead this records one real triangulated trajectory
# (e.g. the ball swinging on a string), then replays that *same* recording
# through the filter with several candidate ACCEL_NOISE values so you can
# compare them fairly and pick by eye: too high looks jittery/overreacts
# to noise, too low looks sluggish/lags behind real motion.
#
# Set MEASUREMENT_NOISE_STD below to whatever measure_triangulation_noise.py
# found - this script only varies ACCEL_NOISE, not measurement noise.
#
# Controls: RECORD_DURATION_S of live triangulation happens automatically
# when you run this - start moving the ball right away.
#

import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import KalmanFilter
from example_ball_detector import BallDetector
from gretchen.camera import Camera

CAMERA_A_INDEX = 0
CAMERA_B_INDEX = 1

CAMERA_A_FX, CAMERA_A_FY, CAMERA_A_CX, CAMERA_A_CY = 1140.68, 1140.68, 319.5, 239.5
CAMERA_B_FX, CAMERA_B_FY, CAMERA_B_CX, CAMERA_B_CY = 1140.68, 1140.68, 319.5, 239.5

CAMERA_B_R = np.eye(3)
CAMERA_B_T = np.array([0.5, 0.0, 0.0])

MEASUREMENT_NOISE_STD = 0.02  # metres - set from measure_triangulation_noise.py

RECORD_DURATION_S = 8.0
ACCEL_NOISE_TRIALS = [50.0, 500.0, 5000.0]


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


def record_trajectory():
    camera_a = Camera(CAMERA_A_INDEX)
    camera_b = Camera(CAMERA_B_INDEX)
    camera_a.start()
    camera_b.start()

    detector_a = BallDetector()
    detector_b = BallDetector()

    P_a = make_projection_matrix(CAMERA_A_FX, CAMERA_A_FY, CAMERA_A_CX, CAMERA_A_CY,
                                  np.eye(3), np.zeros(3))
    P_b = make_projection_matrix(CAMERA_B_FX, CAMERA_B_FY, CAMERA_B_CX, CAMERA_B_CY,
                                  CAMERA_B_R, CAMERA_B_T)

    print(f"Recording {RECORD_DURATION_S:.0f}s of real ball motion now - "
          f"start moving it (e.g. swing it on a string).")

    times = []
    points = []
    start = time.time()

    while time.time() - start < RECORD_DURATION_S:
        ret_a, img_a, _ = camera_a.getImage()
        ret_b, img_b, _ = camera_b.getImage()

        if not ret_a or not ret_b or img_a is None or img_b is None:
            continue

        (_, center_a) = detector_a.detect(img_a)
        (_, center_b) = detector_b.detect(img_b)

        if center_a is not None and center_b is not None:
            point = triangulate(P_a, P_b, center_a, center_b)
            times.append(time.time() - start)
            points.append(point)

    print(f"Recorded {len(points)} triangulated samples.")
    return times, points


def make_kalman_filter(accel_noise):
    kf = KalmanFilter(dim_x=6, dim_z=3)
    kf.x = np.zeros(6)
    kf.P *= 500.0
    kf.H = np.zeros((3, 6))
    kf.H[0, 0] = kf.H[1, 1] = kf.H[2, 2] = 1.0
    kf.R = np.eye(3) * (MEASUREMENT_NOISE_STD ** 2)
    return kf


def set_process_model(kf, dt, accel_noise):
    F = np.eye(6)
    F[0, 3] = F[1, 4] = F[2, 5] = dt

    Q = np.zeros((6, 6))
    for i in range(3):
        Q[i, i] = accel_noise * dt**4 / 4
        Q[i, i + 3] = Q[i + 3, i] = accel_noise * dt**3 / 2
        Q[i + 3, i + 3] = accel_noise * dt**2

    kf.F = F
    kf.Q = Q


def replay(times, points, accel_noise):
    kf = make_kalman_filter(accel_noise)
    speeds = []
    last_t = times[0]

    for i, (t, point) in enumerate(zip(times, points)):
        dt = t - last_t
        last_t = t
        if i == 0:
            kf.x[0:3] = point
        else:
            if dt > 0:
                set_process_model(kf, dt, accel_noise)
                kf.predict()
            kf.update(point)
        speeds.append(np.linalg.norm(kf.x[3:6]))

    return speeds


def main():
    times, points = record_trajectory()
    if len(points) < 2:
        print("Not enough successful triangulations recorded.")
        return

    points = np.array(points)

    fig, (ax_pos, ax_speed) = plt.subplots(2, 1, sharex=True)

    ax_pos.plot(times, points[:, 2], marker='.', color='black', label='raw z')
    ax_pos.set_ylabel("z (m)")
    ax_pos.legend()

    for accel_noise in ACCEL_NOISE_TRIALS:
        speeds = replay(times, points, accel_noise)
        ax_speed.plot(times, speeds, marker='.', label=f"ACCEL_NOISE={accel_noise:g}")

    ax_speed.set_ylabel("filtered speed (m/s)")
    ax_speed.set_xlabel("time (s)")
    ax_speed.legend()
    fig.suptitle("ACCEL_NOISE comparison on the same recorded trajectory")
    plt.show()


if __name__ == '__main__':
    main()
