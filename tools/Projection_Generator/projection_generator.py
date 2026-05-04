#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2 
import numpy as np
import config

from mouse_events import MouseEvents

PIXEL_W = 1920
PIXEL_H = 1080

CAMERA_W = 1920
CAMERA_H = 1080

POINTS_CHECK = False
# 좌표점은 좌상->좌하-> 우상 -> 우하 
# 변경할 좌표를 4x2 행렬로 전환 
# 이동할 좌표를 설정
mouse_events = MouseEvents()
mouse_events2 = MouseEvents()

# path = 'test.jpg'
# frame = cv2.imread(path)
# success = True

use_undistort = False
if os.path.exists("distortion_coefficients.npy") and os.path.exists("calibration_matrix.npy"):
    use_undistort = True
    dist = np.load("distortion_coefficients.npy")
    mtx = np.load("calibration_matrix.npy")


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)

success, frame = cap.read()
if use_undistort:
    frame = cv2.undistort(frame, mtx, dist, None, None)
if config.use_rotate_camera:
    frame = cv2.rotate(frame, cv2.ROTATE_180)
if config.use_flip_camera:
    frame = cv2.flip(frame, 1)

height, width  = frame.shape[:2]
cv2.imshow('origin', frame)
if success == True:
    copy_frame = frame.copy()
    cv2.setMouseCallback('origin',mouse_events.onMouse, copy_frame)
    cv2.waitKey(0)
    # 원본 이미지에서 원하는 지점의 좌표
    source_coord = np.float32( mouse_events.points )
    
    # dest coord는 Pixel Coordinate의 크기로 정의
    destination_coord = np.float32( [ [0,0], [0,PIXEL_H], [PIXEL_W,0], [PIXEL_W,PIXEL_H] ] )
    # 원본 -> 변환 미지수 8개를 구한다
    M = cv2.getPerspectiveTransform(source_coord, destination_coord)
    # 변환 -> 원본 미지수 8개를 구한다
    M2 = cv2.getPerspectiveTransform(destination_coord, source_coord)
	# 원본 -> 변환의 역행렬을 구한다 
    INV_M = np.linalg.pinv(M)
    
print("===== Projection Parameters =====")
print(f"M_cp = np.array([[ {M[0][0]: .8E}, {M[0][1]: .8E}, {M[0][2]: .8E} ],")
print(f"                 [ {M[1][0]: .8E}, {M[1][1]: .8E}, {M[1][2]: .8E} ],")
print(f"                 [ {M[2][0]: .8E}, {M[2][1]: .8E}, {M[2][2]: .8E} ]])")

print()
print(f"DISPLAY_POINTS = {mouse_events.points[0:4]}")


transformed = cv2.warpPerspective(frame, M, (PIXEL_W, PIXEL_H))
cv2.imshow('transformed', transformed)
cv2.waitKey(0)