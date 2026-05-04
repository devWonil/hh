#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import os
import cv2
import numpy as np
import time
import threading
import json
from collections import deque

os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide" # Hide welcome message
import pygame

import logging

import config
import utils.coord_utils as coord

logger = logging.getLogger(__name__)
import utils.state as STATE

print("==============================================================================")
print("  SULIVAN : Synergistic Understanding and Learning with Interactive robotics, ")
print("            computer Vision, Augmented reality, and Neural networks           ")
print("  Copyright 2026 PNU IRLab All rights reserved.                               ")
print("==============================================================================")

# ================== Camera, Medidpipe, YOLO ==================
from .threads.camera_thread import CameraThread
from .threads.mediapipe_thread import MediaPipeThread
from .threads.yolo_thread import YOLOThread
from .threads.video_recorder_thread import VideoRecorderThread
from .threads.tts_thread import TTSThread
from .threads.vlm_thread import VLMThread

from .scenario import Scenario

from .objects.lpips import LPIPSModelLoader

# ====================== Main client ======================
class SULIVAN_Client():
    def __init__(self):
        ''' Load pygame and function threads, then start threads '''
        # Var to handle exit
        self.is_exit = False

        # ================== Init pygame ==================
        pygame.init()

        # Set position of window, create window without frame
        if config.borderless_pygame_window:
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"-{config.PIXELS_WIDTH},0"
            self.SCREEN = pygame.display.set_mode((config.PIXELS_WIDTH, config.PIXELS_HEIGHT), pygame.NOFRAME)
        else:
            self.SCREEN = pygame.display.set_mode((config.PIXELS_WIDTH, config.PIXELS_HEIGHT))

        pygame.display.set_caption("SULIVAN")
        self.display_images = deque([np.zeros((config.PIXELS_HEIGHT, config.PIXELS_WIDTH, 3), np.uint8)], maxlen=6)
        self._display_lock = threading.Lock()

        # Black masks for LPIPS
        self.black_masks = []
        self._black_masks_lock = threading.Lock()

        self.clock = pygame.time.Clock()

        # Load default background
        self.default_bg = pygame.image.load("assets/backgrounds/default.png").convert()
        self.default_bg_rect = self.default_bg.get_rect()
        if self.default_bg_rect.width != config.PIXELS_WIDTH or self.default_bg_rect.height != config.PIXELS_HEIGHT:
            self.default_bg = pygame.transform.smoothscale(self.default_bg, (config.PIXELS_WIDTH, config.PIXELS_HEIGHT))
            self.default_bg_rect = self.default_bg.get_rect()
        self.show_default_bg = True

        # Init threads
        if config.undistort_camera:
            self.camera_thread = CameraThread(self, config.camera_device, is_rotate=config.use_rotate_camera, use_flip_camera=config.use_flip_camera,
                                               use_camera_projection=config.use_camera_projection,
                                               mtx=config.filename_calibration_matrix, dist=config.filename_distortion_coefficients)
        else:
            self.camera_thread = CameraThread(self, config.camera_device, is_rotate=config.use_rotate_camera, use_flip_camera=config.use_flip_camera,
                                               use_camera_projection=config.use_camera_projection)
        
        self.mp_thread     = MediaPipeThread(self)
        self.yolo_thread   = YOLOThread(self)
        
        self.record_thread = None # Will be created when scenario is loaded

        self.client_thread = threading.Thread(target=self.loop)
        self._client_stop_event = threading.Event()

        self.tts_thread = TTSThread()
            
        # Init scenario handler, Scenario will loaded separately
        self.scenario = None

        # Run threads
        self.camera_thread.start()

        # Wait for camera thread is ready
        self.camera_thread._ready_event.wait()
        logger.info("Camera thread loaded")
        
        self.mp_thread.start()

        self.mp_thread._ready_event.wait()
        logger.info("Mediapipe thread loaded")
        
        self.yolo_thread.start()

        # Wait for mediapipe and YOLO thread is ready
        self.yolo_thread._ready_event.wait()
        logger.info("YOLO thread loaded")

        self.tts_thread.start()

        # Load LPIPS model
        LPIPSModelLoader.get_model()

        # VLM (Vision-Language Model) worker
        self._vlm_thread = None
        if getattr(config, 'VLM_ENABLED', False):
            self._vlm_thread = VLMThread()
            self._vlm_thread.start()
            while not self._vlm_thread.ready:
                time.sleep(0.1)
            if self._vlm_thread.is_available():
                logger.info("VLM thread loaded")
            else:
                logger.warning("VLM thread started but provider not available")
            
        logger.info("All threads are ready!")

    def stop(self):
        ''' Stop threads and exit '''
        # Prevent duplicate calls
        if self.is_exit:
            return
        self.is_exit = True

        # Exit client thread
        self._client_stop_event.set()

        if self.record_thread is not None:
            self.record_thread.stop()

        self.camera_thread.stop()
        self.mp_thread.stop()
        self.yolo_thread.stop()
        self.stop_tts()
        if self.tts_thread is not None:
            self.tts_thread.stop()
            self.tts_thread.join(timeout=1)
        if self._vlm_thread is not None:
            self._vlm_thread.stop()
            self._vlm_thread.join(timeout=2)
        
        logger.info("Client stopped")

    def isScenarioLoaded(self):
        return self.scenario is not None
    
    def loadScenario(self, scenario_name):
        ''' Load steps from scenario file '''
        if not self.isScenarioLoaded():
            # Run video_record thread if use
            start_time = time.time()
            if config.record_video:
                self.record_thread = VideoRecorderThread(self, scenario_name, start_time)
                self.record_thread.start()
            logger.info(f"Load scenario: {scenario_name}")
            self.yolo_thread.cleanTrackHistory()
            self.scenario = Scenario(self, scenario_name, start_time)
            self.scenario.start()

    def unloadScenario(self):
        ''' Unload scenario '''
        if self.isScenarioLoaded():
            logger.info("Unload scenario")

            # Stop a guide sound
            pygame.mixer.stop()
            self.stop_tts()

            # Stop Recording
            if config.record_video:
                filename_recorded = self.record_thread.stop()
                self.record_thread.join()
                self.record_thread = None
            else:
                filename_recorded = None
            
            # Save train records
            if config.record_video:
                self.scenario.train_recorder.recordAction("Scenario ended.")
                self.scenario.train_recorder.saveStepRecord() ## Save last record
                
                # Finalize train record
                train_record = self.scenario.train_recorder.finalizeRecord(filename_recorded)

                # Save train record to local file
                try:
                    filename_train_record = os.path.splitext(filename_recorded)[0] + '_train'
                    with open(f"{config.record_dir}/{filename_train_record}.json", 'w', encoding='utf-8') as f:
                        json.dump(train_record, f, indent=4, ensure_ascii=False)
                except Exception as e:
                    logger.error(f"Failed to save train record locally: {e}")

            self.yolo_thread.setMode(STATE.PAUSED)
            self.scenario = None
    
    def restartScenario(self):
        ''' Restart the current scenario '''
        logger.info("Restarting scenario...")
        if self.isScenarioLoaded():
            scenario_name = self.scenario.name
            logger.info(f"Unloading {scenario_name}...")
            self.unloadScenario()
            logger.info(f"Loading {scenario_name}...")
            self.loadScenario(scenario_name)
        else:
            logger.warning("No scenario to restart")
    
    def getDisplayImage(self):
        ''' Get the image to display on pygame screen '''
        with self._display_lock:
            return self.display_images[-1].copy()

    def addBlackMask(self, bbox):
        ''' Add a black mask to the screen. bbox: [x, y, w, h] '''
        x, y, w, h = bbox
        rect = pygame.Rect(x, y, w, h)
        with self._black_masks_lock:
            self.black_masks.append(rect)

    def removeBlackMask(self, bbox):
        ''' Remove a black mask. bbox: [x, y, w, h] '''
        x, y, w, h = bbox
        rect = pygame.Rect(x, y, w, h)
        with self._black_masks_lock:
            if rect in self.black_masks:
                self.black_masks.remove(rect)

    # =================================== TTS API ===================================
    def play_tts(
        self,
        text: str,
        *,
        rate: str | None = None,
        pitch: str | None = None,
        voice: str | None = None,
        volume: float | None = None,
        wait: bool = False,
    ):
        """Queue a text-to-speech request via the EdgeTTS worker."""

        if self.tts_thread is None:
            logger.warning("TTS thread is not initialized")
            return None

        return self.tts_thread.speak(
            text=text,
            rate=rate,
            pitch=pitch,
            voice=voice,
            volume=volume,
            wait=wait,
        )

    def stop_tts(self) -> None:
        """Stop the currently playing TTS audio."""

        if self.tts_thread is not None:
            self.tts_thread.stop_playback()

    def get_tts_cache_path(self, text: str, *, rate: str | None = None, pitch: str | None = None):
        """Return cache file path for the provided TTS request."""

        if self.tts_thread is None:
            return None
        return self.tts_thread.get_cache_path(text, rate=rate, pitch=pitch)

    # =================================== VLM API ===================================
    def request_vlm_feedback(
        self,
        custom_prompt: str | None = None,
        *,
        wait: bool = False,
    ):
        """Request VLM feedback for current work state.
        
        This method captures the current camera frame and current step info,
        then sends them to the VLM for analysis and feedback.
        
        Args:
            custom_prompt: Optional custom prompt to override step's vlm_prompt
            wait: If True, block until response is ready
            
        Returns:
            threading.Event if wait=False, None otherwise
        """
        if self._vlm_thread is None or not self._vlm_thread.is_available():
            logger.warning("VLM thread is not available")
            return None

        if not self.isScenarioLoaded():
            logger.warning("No scenario loaded, cannot request VLM feedback")
            return None

        # Get current camera frame
        frame = self.camera_thread.getImage()
        if frame is None:
            logger.warning("No camera frame available")
            return None

        # Get current step info from scenario
        step_info = self.scenario.getCurrentStepInfo()
        
        # Collect LPIPS images for context
        lpips_images = self._collect_lpips_images()

        return self._vlm_thread.request_feedback(
            frame=frame,
            step_info=step_info,
            custom_prompt=custom_prompt,
            lpips_images=lpips_images,
            wait=wait,
        )

    def _collect_lpips_images(self):
        """Collect LPIPS reference and comparison images for VLM context.
        
        Returns:
            List of LPIPSImageInfo objects or None if no LPIPS objects
        """
        if not self.isScenarioLoaded():
            return None
        
        lpips_objects = self.scenario.lpips_objects
        if not lpips_objects:
            return None
        
        from .threads.vlm_thread import LPIPSImageInfo
        
        lpips_images = []
        for name, lpips_obj in lpips_objects.items():
            context = lpips_obj.get_vlm_context()
            lpips_images.append(LPIPSImageInfo(
                name=context["name"],
                reference_image=context["reference_image"],
                comparison_image=context["comparison_image"],
                last_score=context["last_score"],
                threshold=context["threshold"],
            ))
        
        return lpips_images if lpips_images else None

    def get_vlm_result(self):
        """Get the latest VLM response.
        
        Returns:
            VLMResponse object or None if no response available
        """
        if self._vlm_thread is None:
            return None
        return self._vlm_thread.get_latest_response()

    def is_vlm_available(self) -> bool:
        """Check if VLM is available for requests."""
        return self._vlm_thread is not None and self._vlm_thread.is_available()

    def is_vlm_processing(self) -> bool:
        """Check if VLM is currently processing a request."""
        if self._vlm_thread is None:
            return False
        return self._vlm_thread.is_processing()

    def is_vlm_rate_limited(self) -> bool:
        """Check if VLM is currently in rate limit cooldown."""
        if self._vlm_thread is None:
            return False
        return self._vlm_thread.is_rate_limited()

    def get_vlm_rate_limit_remaining(self) -> float:
        """Get remaining rate limit cooldown time in seconds."""
        if self._vlm_thread is None:
            return 0
        return self._vlm_thread.get_rate_limit_remaining()

    # =================================== Loop ===================================
    def run(self):
        self.client_thread.start()
    
    def loop(self):
        ''' Main loop function: Handle scenario '''
        while not self._client_stop_event.is_set():
            try:
                # Handle exit
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self._client_stop_event.set()
                
                # ================== Update pygame screen ==================
                # Wait for loading scenario
                if not self.isScenarioLoaded():
                    if self.show_default_bg:
                        self.SCREEN.fill((0,0,0))
                        self.SCREEN.blit(self.default_bg, self.default_bg_rect)
                        self.updateScreen()
                    self.clock.tick(30)
                    continue

                # Fill screen with black (No lights)
                self.SCREEN.fill((0,0,0))
                
                # Handle pause
                if self.scenario.getRunningState() != STATE.RUNNING:
                    self.clock.tick(30)
                    continue

                # Get background object from ScenarioHandler
                # Images need to be resized by ScenarioHandler
                bg = self.scenario.getBackground()
                bg_rect = bg.get_rect()
                bg_rect.x = 0
                bg_rect.y = 0
                self.SCREEN.blit(bg, bg_rect) # Apply background image to screen
                
                # Handle guide video
                guide_video, guide_video_bbox = self.scenario.getGuideVideo()
                # Show video if guide video object exists
                if guide_video is not None:
                    # Replay a video without vanishing
                    try:
                        if guide_video.frame >= (guide_video.frame_count - 6) or not guide_video.active:
                            guide_video.restart()
                        guide_video.draw(self.SCREEN, guide_video_bbox[0:2])
                    except:
                        pass
                
                # Run Scenario Loop once (Check conditions, handle objects)
                if self.isScenarioLoaded():
                    try:
                        # Run Scenario main loop
                        self.scenario.loop()
                        # Draw information area
                        if getattr(self.scenario, 'is_display_runtime', False) is True:
                            self.scenario.drawInformationArea()
                    except Exception as e:
                        logger.error(f"An error occurred in scenario loop: {e}")
                        if getattr(config, 'stop_scenario_on_error', True) is True:
                            self.unloadScenario()
                
                # Draw bboxes of hands
                if config.show_hands_bbox_to_screen:
                    hands = self.mp_thread.getHands()

                    for hand in hands.copy():
                        if hand['type'] == 'Left':
                            bbox_color = [255, 0, 0] # Blue
                        else:
                            bbox_color = [0, 0, 255] # Red

                        hand_bbox = hand['bbox']
                        hand_rect = pygame.Rect(hand_bbox[0], hand_bbox[1], hand_bbox[2], hand_bbox[3])
                        pygame.draw.rect(self.SCREEN, bbox_color, hand_rect, 6)
                
                # Draw black masks
                with self._black_masks_lock:
                    black_masks_snapshot = self.black_masks.copy()
                for rect in black_masks_snapshot:
                    pygame.draw.rect(self.SCREEN, (0, 0, 0), rect)

                self.updateScreen()
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.clock.tick(30)

        # Exit
        if not self.is_exit:
            self.stop()

    
    def setShowDefaultBackground(self, show: bool):
        self.show_default_bg = show

    def displayBackground(self, image_path=None, image_cv=None):
        # Fill screen with black (No lights)
        self.SCREEN.fill((0,0,0))
        if image_path is not None:
            bg = pygame.image.load(image_path).convert()
            bg_rect = bg.get_rect()
            self.SCREEN.blit(bg, bg_rect)
        elif image_cv is not None:
            bg = pygame.image.frombuffer(image_cv.tostring(), image_cv.shape[1::-1], "BGR")
            bg_rect = bg.get_rect()
            self.SCREEN.blit(bg, bg_rect)
        pygame.display.flip()
    
    def updateScreen(self):
        pygame.display.flip()
        screen = pygame.display.get_surface()
        capture = pygame.surfarray.pixels3d(screen)
        frame = cv2.cvtColor(capture.transpose([1, 0, 2]), cv2.COLOR_RGB2BGR)
        del capture
        with self._display_lock:
            self.display_images.append(frame)

        self.clock.tick(30)

if __name__ == '__main__':

    print("Please run main.py!")