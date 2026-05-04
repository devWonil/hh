#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import math
import time

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # Hide welcome message
import pygame

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from client import SULIVAN_Client

import config

import utils.coord_utils as coord
import utils.state as STATE
from utils.shape import OBB, check_collision_obb_obb
import logging

logger = logging.getLogger(__name__)

class ObjectButton:
    def __init__(self, client: "SULIVAN_Client", place_point, cls, place_time, complete_callback, callback_arg=None):
        self._client = client
        self.scenario = self._client.scenario
        x, y, w, h = place_point
        self.place_obb = OBB(x + w / 2, y + h / 2, w, h, 0)

        try:
            # Convert class name to index
            if type(cls) == int:
                self.cls = cls
            else:
                self.cls = self._client.yolo_thread.indexes[cls]
        except KeyError:
            logger.error(f"Class '{cls}' not found in YOLO indexes.")
            self.cls = -1 # Actual indexes starts from 0

        self.place_time = place_time
        self.complete_callback = complete_callback
        self.callback_arg = callback_arg
        
        # === Hover state ===
        # NOT_STARTED: Not hovering
        # WORKING: Hovering now
        # COMPLETED: Completed
        self.hover_state = STATE.NOT_STARTED
        self.hover_start_time = None

        # Images
        self.default_image = None
        self.default_image_rect = None
        self.incorrect_image = None
        self.incorrect_image_rect = None
        self.correct_image = None
        self.correct_image_rect = None

    def loadImage(self, image_path, image_bbox):
        '''Load and resize button image'''
        image_path = self.scenario.path_util.getPath(image_path)
        image         = pygame.image.load(image_path).convert_alpha()
        image_resized = pygame.transform.smoothscale(image, image_bbox[2:4])

        rect   = image_resized.get_rect()
        rect.x = image_bbox[0]
        rect.y = image_bbox[1]

        return image_resized, rect

    def setDefaultImage(self, path, bbox):
        '''Set detection's default image'''
        if path is not None:
            self.default_image, self.default_image_rect = self.loadImage(path, bbox)
        else:
            self.default_image, self.default_image_rect = None, None
            
    def setIncorrectImage(self, path, bbox):
        '''Set detection's default image'''
        if path is not None:
            self.incorrect_image, self.incorrect_image_rect = self.loadImage(path, bbox)
        else:
            self.incorrect_image, self.incorrect_image_rect = None, None
            
    def setCorrectImage(self, path, bbox):
        '''Set detection's default image'''
        if path is not None:
            self.correct_image, self.correct_image_rect = self.loadImage(path, bbox)
        else:
            self.correct_image, self.correct_image_rect = None, None
        
        
    def checkPosition(self, detections):
        '''Loop: check YOLO detection data and handle assigned event'''
        time_to_clear = self.place_time

        # Build OBB for each detection
        detection_datas = []
        for detection in detections:
            det = detection.copy()
            cx, cy, w, h = det['bbox']
            det['det_obb'] = OBB(cx, cy, w, h, math.degrees(det['angle']))
            detection_datas.append(det)

        # Handle placing
        keep_timer = False
        incorrect_class_in_place_point = False
        correct_class_in_place_point = False
        draw_corners = None
        for detection in detection_datas:

            # Check collide (OBB-OBB, AABB is treated as angle=0 OBB)
            if check_collision_obb_obb(self.place_obb, detection['det_obb']):
                # State 0->1
                if self.hover_state == STATE.NOT_STARTED:
                    self.hover_state = STATE.WORKING
                    self.yolo_track_id = detection['track_id']
                    self.hover_start_time = time.time()
                    keep_timer = True

                # State 1
                elif self.hover_state == STATE.WORKING:
                    # Keep timer when track id is same
                    if detection['track_id'] == self.yolo_track_id:
                        keep_timer = True

                    # State 1->2 (Clear!)
                    if self.cls == detection['class']:
                        correct_class_in_place_point = True
                        if time.time() - self.hover_start_time > time_to_clear / 1000:
                            self.complete_callback(self.callback_arg)
                        draw_corners = detection['corners']

                    # Incorrect class: Showing incorrect image
                    else:
                        incorrect_class_in_place_point = True

        # Reset hover start time when correct object not presents
        if not correct_class_in_place_point:
            self.hover_start_time = time.time()

        # Reset when object is disappeared
        if not keep_timer:
            self.hover_state      = STATE.NOT_STARTED
            self.hover_start_time = None
            self.yolo_track_id    = None

        # Display default image
        if self.default_image is not None:
            self._client.SCREEN.blit(self.default_image, self.default_image_rect)

        # Display correct image
        if correct_class_in_place_point and self.correct_image is not None:
            self._client.SCREEN.blit(self.correct_image, self.correct_image_rect)
        # Display incorrect image
        elif incorrect_class_in_place_point and self.incorrect_image is not None:
            self._client.SCREEN.blit(self.incorrect_image, self.incorrect_image_rect)

        # Display YOLO bbox
        if config.show_yolo_bbox_to_screen and draw_corners is not None:
            pygame.draw.polygon(self._client.SCREEN, (0, 255, 0), draw_corners, 6)