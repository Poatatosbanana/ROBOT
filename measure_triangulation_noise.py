#!/usr/bin/env python3
#
# 9-tennis
#
# Measures MEASUREMENT_NOISE_STD for measure_ball_speed_stationary.py's
# Kalman filter. Put the ball at a fixed, static position in view of both
# cameras and don't touch it while this runs - it triangulates repeatedly
# with no filtering, then reports the spread (std dev) of the raw
# triangulated position per axis. That spread is your real measurement
# noise number.
#
# Uses the same camera/intrinsic/extrinsic placeholders as
# measure_ball_speed_stationary.py - fill those in first.
#

import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from example_ball_detector import BallDetector
from gretchen.camera import Camera

CAMERA_A_INDEX = 0
CAMERA_B_INDEX = 1

CAMERA_A_FX, CAMERA_A_FY, CAMERA_A_CX, CAMERA_A_CY = 1140.68, 1140.68, 319.5, 239.5
CAMERA_B_FX, CAMERA_B_FY, CAMERA_B_CX, CAMERA_B_CY = 1140.68, 1140.68, 319.5, 239.5

CAMERA_B_R = np.eye(3)
CAMERA_B_T = np.array([0.5, 0.0, 0.0])

DURATION_S = 10.0


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


def main():
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

    print(f"Logging static triangulated position for {DURATION_S:.0f}s - "
          f"do not move the ball.")

    times = []
    points = []
    start = time.time()

    while time.time() - start < DURATION_S:
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

    if len(points) < 2:
        print("Not enough successful triangulations to measure noise from.")
        return

    points = np.array(points)
    mean = points.mean(axis=0)
    std = points.std(axis=0)

    print()
    print(f"Samples: {len(points)}")
    print(f"Mean position (m): x={mean[0]:.4f} y={mean[1]:.4f} z={mean[2]:.4f}")
    print(f"Std dev (m):        x={std[0]:.4f} y={std[1]:.4f} z={std[2]:.4f}")
    print(f"Overall std dev (m): {np.linalg.norm(std):.4f}")
    print()
    print("Use the overall std dev (or the largest per-axis value) as "
          "MEASUREMENT_NOISE_STD in measure_ball_speed_stationary.py.")

    fig, axes = plt.subplots(3, 1, sharex=True)
    labels = ['x', 'y', 'z']
    for i in range(3):
        axes[i].plot(times, points[:, i], marker='.')
        axes[i].axhline(mean[i], color='gray', linestyle='--')
        axes[i].set_ylabel(f"{labels[i]} (m)")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Triangulated position noise while stationary")
    plt.show()


if __name__ == '__main__':
    main()
