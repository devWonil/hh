#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import cv2
import os
import mediapipe as mp
import numpy as np
import json
import ffmpegcv

frameRate = round(1000/30)

mpHands = mp.solutions.hands
mpDraw = mp.solutions.drawing_utils

print(os.getcwd())

filename = 'records/20250513-211445_fd9225d20b5169e2b295534bb571506d'

def fromCameraToPixelPoint(point, M_cp, rounded=False):
    '''Calculate pixel coordinate from camera coordinate. (point: [X, Y])'''
    point_homo = np.ones((3, 1))
    point_homo[0] = point[0]
    point_homo[1] = point[1]

    calc_point = M_cp @ point_homo
    calc_point /= calc_point[2] # Normalize

    if rounded:
        return [round(calc_point[0,0]), round(calc_point[1,0])]
    else:
        return [calc_point[0,0], calc_point[1,0]]

# Open mediapipe records
with open(filename + '.json', 'r') as f:
    json_load = json.load(f)
    metadata = json_load['metadata']
    overlays = json_load['detects']
    time_file = overlays[0]['time']

    M_cp = np.array(metadata['M_cp'])

# Open videos
cap = cv2.VideoCapture(filename + '.mp4')
writer = ffmpegcv.VideoWriterNV("overlay_output.mp4", 'h264', 30)

frame_index = 0
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)

while True:
    retval, img = cap.read()
    if not(retval):	# 프레임정보를 정상적으로 읽지 못하면
        break  # while문을 빠져나가기
    print(frame_index, overlays[frame_index]['time'], ":", round(overlays[frame_index]['time'] - time_file, 5))
    time_file = overlays[frame_index]['time']
    mp_overlays = [overlays[frame_index]['mediapipe']]#, overlays[i]['mediapipe_unmasked']]
    lm_colors = [(0, 0, 255), (255, 0, 0)] # Green for masked, Blue for unmasked

    for mp_overlay, lm_color in zip(mp_overlays, lm_colors):
        for hand in mp_overlay:
            bbox = hand['bbox_cam']
            hand_score = hand['score']
            if 'landmarks_cam' in hand:
                handLms = hand['landmarks_cam']
            else:
                handLms = hand['landmarks']
            handtype = hand['type']

            # Index 0
            cv2.circle(img, handLms[0][0:2], 5, lm_color, -1)

            # Index 1 to 20
            for i in range(0, 5):
                cv2.circle(img, handLms[i * 4 + 1][0:2], 4, lm_color, 2) # 동그라미
                # Draw triangle (세모)
                triangle_pts = np.array([handLms[i * 4 + 2][0:2], 
                                       [handLms[i * 4 + 2][0] - 4, handLms[i * 4 + 2][1] + 6],
                                       [handLms[i * 4 + 2][0] + 4, handLms[i * 4 + 2][1] + 6]], np.int32)
                cv2.polylines(img, [triangle_pts], True, lm_color, 2)
                
                # Draw square (네모)
                x, y = handLms[i * 4 + 3][0:2]
                cv2.rectangle(img, (int(x-3), int(y-3)), (int(x+3), int(y+3)), lm_color, 2)
                
                # Draw filled star (별)
                x, y = handLms[i * 4 + 4][0:2]
                size = 6
                # Create star points
                star_points = np.array([
                    [int(x), int(y - size)],                        # top point
                    [int(x + size * 0.3), int(y - size * 0.3)],     # upper right
                    [int(x + size), int(y - size * 0.1)],           # right point
                    [int(x + size * 0.4), int(y + size * 0.3)],     # lower right
                    [int(x + size * 0.6), int(y + size)],           # bottom right
                    [int(x), int(y + size * 0.6)],                  # bottom point
                    [int(x - size * 0.6), int(y + size)],           # bottom left
                    [int(x - size * 0.4), int(y + size * 0.3)],     # lower left
                    [int(x - size), int(y - size * 0.1)],           # left point
                    [int(x - size * 0.3), int(y - size * 0.3)],     # upper left
                ], np.int32)
                cv2.fillPoly(img, [star_points], lm_color)
            
            cv2.putText(img, f"{handtype} {hand_score:.4f}", (bbox[0] - 30, bbox[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                        1.4, (255, 0, 255), 2)
    
    # Write the frame to the output video
    writer.write(img)
    cv2.imshow('frame', img)	# 프레임 보여주기
    key = cv2.waitKey(frameRate)  # frameRate msec동안 한 프레임을 보여준다
    
    if key == 27: # ESC
        break

    frame_index += 1
        
if cap.isOpened():	# 영상 파일(카메라)이 정상적으로 열렸는지(초기화되었는지) 여부
    cap.release()	# 영상 파일(카메라) 사용을 종료
    
cv2.destroyAllWindows()