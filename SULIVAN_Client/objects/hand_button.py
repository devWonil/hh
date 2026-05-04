#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import time
import threading

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # Hide welcome message
import pygame

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from client import SULIVAN_Client

import utils.state as STATE
from utils.shape import OBB, Circle, check_collision_obb_circle

class HandButton():
    # Variables 
    buttons = []
    _active_lock = threading.Lock()
    active_button = None
    active_button_mp = None
    active_button_state = STATE.NOT_STARTED

    @classmethod
    def getActiveButtonSnapshot(cls):
        """Return a consistent snapshot of (active_button, active_button_mp, active_button_state)."""
        with cls._active_lock:
            return cls.active_button, cls.active_button_mp, cls.active_button_state

    def __init__(self, client: "SULIVAN_Client", getHands, name, bbox, angle_deg, hover_time, complete_callback, callback_arg=None):
        # Define methods
        self._client = client
        self.scenario = client.scenario
        self.getHands = getHands

        # Attributes
        self.name = name
        HandButton.buttons.append(self.name)

        x, y, w, h = bbox
        center_x = x + w / 2
        center_y = y + h / 2
        self.obb = OBB(center_x, center_y, w, h, angle_deg)
        self.hover_time = hover_time

        # New variables for stability check
        self.hover_start_timer = (self.hover_time * 0.2) / 1000
        self.hover_end_timer = self.hover_time / 1000
        self.initial_hand_pos = None
        self.movement_threshold = 10 # pixels

        self.complete_callback = complete_callback
        self.callback_arg = callback_arg
        
        self.use_up_image = False
        self.use_down_image = False

        self.handle_grasp = False
        self.hover_hand = None
        self.desired_work_hand = None

        # === Hover state ===
        #  NOT_STARTED: Not hovering
        #  WORKING: Hovering now
        #  COMPLETED: Completed
        self.hover_state = STATE.NOT_STARTED
        self.hover_timer = None
        
        # Saves button state
        self.is_button_down = False

        # Press up/down effect sounds
        self.press_down_sound = None
        self.press_down_end_sound = None
    
    def __del__(self):
        '''Delete from instance list when the button is deleted'''
        HandButton.buttons.remove(self.name)
        with HandButton._active_lock:
            if HandButton.active_button == self.name:
                HandButton.active_button = None
                HandButton.active_button_mp = None
                HandButton.active_button_state = STATE.NOT_STARTED

    def setHandleGrasp(self, change):
        '''Set handle grasp option.
           if set true, button action works only when hand is grasped
        '''
        self.handle_grasp = change

    def setDesiredWorkHand(self, hand):
        '''Set desired work hand option.
           if set, button action works only with desired work hand
        '''
        self.desired_work_hand = hand

    def loadImage(self, image_path):
        '''Load and resize button image'''
        image_path = self.scenario.path_util.getPath(image_path)
        return pygame.image.load(image_path).convert_alpha()

    def setPressUpImage(self, image_path):
        '''Set button's press up(Not pressed) image'''
        if image_path is not None:
            image_surface = self.loadImage(image_path)
            self.obb.add_image('up', image_surface)
            self.obb.set_active_image('up') # 기본값
            self.use_up_image = True
        
    def setPressDownImage(self, image_path):
        '''Set button's press down(Pressed) image'''
        if image_path is not None:
            image_surface = self.loadImage(image_path)
            self.obb.add_image('down', image_surface)
            self.use_down_image = True

    def setPressDownSound(self, sound_path):
        '''Set button's sound feedback when press downs'''
        sound_path = self.scenario.path_util.getPath(sound_path)
        self.press_down_sound = pygame.mixer.Sound(sound_path)
        self.press_down_sound.set_volume(0.7)
        
    def setPressDownEndSound(self, sound_path):
        '''Set button's sound feedback when press down and holds'''
        sound_path = self.scenario.path_util.getPath(sound_path)
        self.press_down_end_sound = pygame.mixer.Sound(sound_path)
        self.press_down_end_sound.set_volume(0.7)

    def checkHover(self):
        '''Check hands collide with button and run a callback function when condition is satisfied.'''
        # Handle display delay
        if hasattr(self, 'display_delay') and time.time() - self.scenario.step_start_time < self.display_delay / 1000:
            return

        hands = self.getHands()
        is_collided = False

        for hand in hands.copy():
            hand_type    = hand['type']
            hand_circle_info = hand.get('hand_circle')
            if hand_circle_info is None:
                continue
            
            # Check hand type(L/R)
            if (self.desired_work_hand is not None) and (hand_type != self.desired_work_hand):
                continue # Prevent abnormal action

            # ===== Check collision =====
            center_x, center_y = hand_circle_info['center']
            radius = hand_circle_info['radius']

            # Create circle for collision detection
            hand_circle = Circle(center_x, center_y, radius)

            # Check collision between circle and hover_point
            if check_collision_obb_circle(self.obb, hand_circle):
                colide_hand = True
            else:
                colide_hand = False

            if colide_hand:
                is_collided = True
                # === Scenario 0: Start hover ===
                if self.hover_state == STATE.NOT_STARTED:
                    self.hover_state = STATE.WORKING
                    self.hover_hand = hand_type
                    self.hover_timer = time.time()
                    self.initial_hand_pos = (center_x, center_y)
                    with HandButton._active_lock:
                        if HandButton.active_button != self.name:
                            HandButton.active_button = self.name
                            HandButton.active_button_mp = hand
                            HandButton.active_button_state = STATE.WORKING

                # === Scenario 1: hovering ===
                elif self.hover_state == STATE.WORKING:
                    with HandButton._active_lock:
                        HandButton.active_button = self.name
                        HandButton.active_button_mp = hand

                    # Check stability
                    if self.initial_hand_pos is not None:
                        dist = ((center_x - self.initial_hand_pos[0])**2 + (center_y - self.initial_hand_pos[1])**2)**0.5
                        if self.handle_grasp and dist > self.movement_threshold:
                            # Moved too much, reset timer and position
                            self.hover_timer = time.time()
                            self.initial_hand_pos = (center_x, center_y)
                            self.is_button_down = False
                            continue

                    elapsed_time = time.time() - self.hover_timer

                    # Play press down sound
                    if elapsed_time > self.hover_start_timer:
                        if not self.is_button_down:
                            self.is_button_down = True
                            with HandButton._active_lock:
                                if HandButton.active_button == self.name:
                                    if self.press_down_sound is not None:
                                        self.press_down_sound.play()
                    
                    # Check hover time
                    if elapsed_time > self.hover_end_timer:
                        self.hover_state = STATE.COMPLETED
                        with HandButton._active_lock:
                            if HandButton.active_button == self.name:
                                if self.press_down_end_sound is not None:
                                    self.press_down_end_sound.play()
                        
                        # Run pre-defined callback(e.g. goToNextStep())
                        self.scenario.setRecentHand(hand_type)
                        self.complete_callback(self.callback_arg)
                        with HandButton._active_lock:
                            if HandButton.active_button == self.name:
                                HandButton.active_button_state = STATE.COMPLETED
                # === Scenario 2: Clear! ===
                elif self.hover_state == STATE.COMPLETED:
                    pass

        # Reset hover state if not collided with any hands
        if not is_collided:
            self.hover_state = STATE.NOT_STARTED
            self.hover_hand = None
            self.hover_timer = None
            self.is_button_down = False
            self.initial_hand_pos = None
            if HandButton.active_button == self.name:
                with HandButton._active_lock:
                    HandButton.active_button = None
                    HandButton.active_button_mp = None
                    HandButton.active_button_state = STATE.NOT_STARTED

        # Display button
        if self.is_button_down and self.use_down_image:
            self.obb.set_active_image('down')
            self.obb.draw(self._client.SCREEN)
        elif self.use_up_image:
            self.obb.set_active_image('up')
            self.obb.draw(self._client.SCREEN)

    def getHoverHand(self):
        '''Returns type of hovering hand. (e.g. "Left", "Right")'''
        return self.hover_hand