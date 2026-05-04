#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import cv2
import numpy as np

class MouseEvents:
    def __init__(self) -> None:
        self.points = []

    def left_button_down(self, x, y, frame):
        self.points.append([x, y])  
        cv2.circle(frame, (x,y),5,(0,0,255),-1)
        cv2.imshow('origin',frame)
        print(x,y)     

    def right_button_down(self,x,y, frame):
        self.points.pop()
        print(x,y)      

    def onMouse(self, event, x, y, flags, img):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.left_button_down(x,y, img)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.right_button_down(x,y,img)