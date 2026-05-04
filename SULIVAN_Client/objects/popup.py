#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import time

import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # Hide welcome message
import pygame

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from client import SULIVAN_Client

import utils.state as STATE

class Popup():
    def __init__(self, client: "SULIVAN_Client", name, image_path, bbox, display_time, complete_callback, callback_arg=None):
        self._client = client
        self.scenario = client.scenario
        self.name = name
        self.bbox = bbox # (X, Y, W, H)
        self.display_time = display_time
        self.complete_callback = complete_callback
        self.callback_arg = callback_arg

        self.triggered = False

        # === Display state ===
        #  NOT_STARTED: Not displaying
        #  WORKING: Displaying now
        #  COMPLETED: Completed
        self.display_state = STATE.NOT_STARTED
        self.display_timer = None

        # Load and resize button image
        image_path = self.scenario.path_util.getPath(image_path)
        image         = pygame.image.load(image_path).convert_alpha()
        image_resized = pygame.transform.smoothscale(image, self.bbox[2:4])

        rect   = image_resized.get_rect()
        rect.x = self.bbox[0]
        rect.y = self.bbox[1]

        self.image      = image_resized
        self.image_rect = rect
        
        self.popup_sound = None
        
    def setPopupSound(self, sound_path):
        '''Set button's sound feedback when press downs'''
        sound_path = self.scenario.path_util.getPath(sound_path)
        self.popup_sound = pygame.mixer.Sound(sound_path)
        self.popup_sound.set_volume(0.7)
    
    def trigger(self):
        '''Trigger the popup'''
        if not self.triggered:
            self.triggered = True

    def getDisplayState(self):
        '''Get current display state'''
        return self.display_state
    
    def checkDisplayTime(self):
        '''Check display time and run a callback function when condition is satisfied.'''
        if self.triggered:
            # === Scenario 0: Start hover ===
            if self.display_state == STATE.NOT_STARTED:
                self.display_state = STATE.WORKING
                self.display_timer = time.time()
                # Play popup sound
                if self.popup_sound is not None:
                    self.popup_sound.play()

            # === Scenario 1: hovering ===
            elif self.display_state == STATE.WORKING:
                # Check display time
                if time.time() - self.display_timer > self.display_time / 1000:
                    self.display_state = STATE.COMPLETED

            # === Scenario 2: Clear! ===
            elif self.display_state == STATE.COMPLETED:
                # Run pre-defined callback(e.g. goToNextStep())
                self.complete_callback(self.callback_arg)

                # Reset state
                self.display_state = STATE.NOT_STARTED
                self.triggered = False
            
            # Display popup
            self._client.SCREEN.blit(self.image, self.image_rect)
        
        return self.display_state