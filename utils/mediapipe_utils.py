#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import math
import copy

import utils.coord_utils as coord

FINGER_LANDMARKS = [
    ( 8,  7,  6), # Index Finger
    (12, 11, 10), # Middle Finger
    (16, 15, 14), # Ring Finger
    (20, 19, 18)  # Pinky
]

def calculate_angle(point1, point2):
    '''Calculate the angle between two points'''
    return math.atan2((point2[1] - point1[1]), (point2[0] - point1[0]))

def check_finger_angles(hand_landmarks, threshold=0.1):
    '''Check if the finger angles are similar'''
    angles = []
    depths = []
    for tip, mip, pip in FINGER_LANDMARKS:
        angle = calculate_angle(hand_landmarks[tip], hand_landmarks[pip])
        angles.append(angle)
        depths.append(hand_landmarks[tip][2])
        depths.append(hand_landmarks[mip][2])
        depths.append(hand_landmarks[pip][2])

    # Get average
    angle_avg = sum(angles) / len(angles)
    depth_avg = sum(depths) / len(depths)

    # Get error between average
    diff_angles = [abs(angles[i] - angle_avg) for i in range(len(angles))]
    diff_depths = [abs(depths[i] - depth_avg) for i in range(len(depths))]
    
    # Check threshold
    # TODO: Remove hard-coded thresholds
    return all(diff <= threshold for diff in diff_angles) and all(abs(diff) <= 10 for diff in diff_depths)

def convert_coordinate(hand_raw):
    '''Convert the coordinate from camera to pixel'''
    hand = copy.deepcopy(hand_raw)

    xList = []
    yList = []
    for i, lm in enumerate(hand["lmList"]):
        lm[0:2] = coord.fromCameraToPixelPoint(lm, rounded=True)
        hand["lmList"][i] = lm
        xList.append(lm[0])
        yList.append(lm[1])

    xmin, xmax = min(xList), max(xList)
    ymin, ymax = min(yList), max(yList)
    boxW, boxH = xmax - xmin, ymax - ymin
    hand["bbox"] = xmin, ymin, boxW, boxH
    
    hand["center"] = coord.fromCameraToPixelPoint(hand['center'], rounded=True)

    return hand