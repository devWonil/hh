#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import cv2
import numpy as np
import time
import threading
import queue

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from client import SULIVAN_Client

import config
import logging

logger = logging.getLogger(__name__)
from utils.mask_projector import MaskProjector

FRAME_RATE = 30

class CameraThread(threading.Thread):
    def __init__(self, client: "SULIVAN_Client", device,
                 show_window=False, is_rotate=False, use_flip_camera=False,
                 use_camera_projection=False, dist=None, mtx=None):
        self._client = client
        self.ready = False
        self._ready_event = threading.Event()
        self.show_window = show_window
        self.is_rotate = is_rotate
        self.use_flip_camera = use_flip_camera
        self.use_camera_projection = use_camera_projection
        self.mask_projector = MaskProjector()

        # Check if doubled resolution capture is enabled
        self.capture_doubled_resolution = getattr(config, 'capture_doubled_resolution', False)
        self.base_width = config.CAMERA_WIDTH
        self.base_height = config.CAMERA_HEIGHT

        # Load camera distort parameters
        self.use_undistort = False
        if dist is not None and mtx is not None:
            try:
                self.dist = np.load(dist)
                self.mtx = np.load(mtx)
                if self.capture_doubled_resolution:
                    self.mtx_doubled = self.mtx.copy()
                    self.mtx_doubled *= 2.0
                    self.mtx_doubled[2][2] = 1.0
                self.use_undistort = True
                
                # Precompute undistortion maps for faster processing
                self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                    self.mtx, self.dist, None, self.mtx, 
                    (self.base_width, self.base_height), cv2.CV_32FC1
                )

                self.mapx_doubled, self.mapy_doubled = None, None
                if self.capture_doubled_resolution:
                    self.mapx_doubled, self.mapy_doubled = cv2.initUndistortRectifyMap(
                        self.mtx_doubled, self.dist, None, self.mtx_doubled, 
                        (self.base_width * 2, self.base_height * 2), cv2.CV_32FC1
                    )
            except:
                pass

        # Calculate actual capture resolution
        if self.capture_doubled_resolution:
            self.capture_width = self.base_width * 2
            self.capture_height = self.base_height * 2
            logger.info(f"Doubled resolution capture enabled: {self.capture_width}x{self.capture_height}")
        else:
            self.capture_width = self.base_width
            self.capture_height = self.base_height

        # ====== Open a device ======
        self.cap = cv2.VideoCapture(device)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.capture_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
        self.cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
        self.cap.set(cv2.CAP_PROP_FOCUS, 0)

        logger.info(f"FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")

        # Define camera brightness
        try:
            camera_brightness = config.CAMERA_BRIGHTNESS
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, camera_brightness)
        except AttributeError:
            logger.warning("Camera brightness is not defined in config.py")
        
        self._stop_event = threading.Event()
        super().__init__()
        self.name = "Camera"
        self.daemon = True

        self.img = None
        self.img_raw = None
        
        # Image queue and capture thread for separate image capture
        self.image_queue = queue.Queue(maxsize=1)  # Small buffer to keep latest frames
        # Queue for doubled resolution images (for LPIPS comparison)
        self.image_queue_doubled = queue.Queue(maxsize=1) if self.capture_doubled_resolution else None
        self.capture_thread = None
        self.capture_lock = threading.Lock()

    def stop(self):
        self._stop_event.set()
        # Wait for capture thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1)

    def getImage(self):
        '''Get the image from camera'''
        with self.capture_lock:
            return self.img.copy() if self.img is not None else None
    
    def getRawImage(self):
        '''Get the raw image from camera'''
        with self.capture_lock:
            return self.img_raw.copy() if self.img_raw is not None else None

    def _capture_worker(self):
        '''Background thread worker for continuous image capture'''
        while not self._stop_event.is_set():
            try:
                # Capture a image from camera
                success, img_raw = self.cap.read()
                if not success:
                    continue
                
                # ======================= Pre-Processing =======================
                # Store doubled resolution image before resizing
                if self.capture_doubled_resolution:
                    # Resize to base resolution for normal processing
                    img = cv2.resize(img_raw, (self.base_width, self.base_height), 
                                     interpolation=cv2.INTER_AREA)
                else:
                    img = img_raw
                
                # Handle camera undistortion
                if self.use_undistort:
                    img = cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)
                
                # Handle rotating image
                if self.is_rotate:
                    img = cv2.rotate(img, cv2.ROTATE_180)
                
                # Handle flipping image
                if self.use_flip_camera:
                    img = cv2.flip(img, 1) # Flip left/right
                
                # Put the processed image in queue (non-blocking)
                try:
                    self.image_queue.put_nowait(img)
                except queue.Full:
                    # Remove old frame and add new one
                    try:
                        self.image_queue.get_nowait()
                        self.image_queue.put_nowait(img)
                    except queue.Empty:
                        pass
                
                # Put the doubled resolution image in queue (non-blocking)
                if self.capture_doubled_resolution and img_raw is not None:
                    try:
                        self.image_queue_doubled.put_nowait(img_raw)
                    except queue.Full:
                        # Remove old frame and add new one
                        try:
                            self.image_queue_doubled.get_nowait()
                            self.image_queue_doubled.put_nowait(img_raw)
                        except queue.Empty:
                            pass
                        
            except Exception as e:
                logger.error(f"Error in capture worker: {e}")
                break
    
    def captureImage(self):
        '''Get the latest captured image from the queue'''
        try:
            # Get the latest image from queue
            img = self.image_queue.get_nowait()
            return img
        except queue.Empty:
            # If no image available, return None
            return None
    
    def captureDoubledImage(self):
        '''Get the latest doubled resolution image from the queue.
        This is useful for LPIPS image comparison with higher quality images.
        Returns None if capture_doubled_resolution is disabled or no image available.'''
        if not self.capture_doubled_resolution or self.image_queue_doubled is None:
            return None
        try:
            # Get the latest doubled resolution image from queue
            img = self.image_queue_doubled.get_nowait()
            
            # Handle camera undistortion
            if self.use_undistort and self.mapx_doubled is not None and self.mapy_doubled is not None:
                img = cv2.remap(img, self.mapx_doubled, self.mapy_doubled, cv2.INTER_LINEAR)

            # Handle rotating image
            if self.is_rotate:
                img = cv2.rotate(img, cv2.ROTATE_180)
            
            # Handle flipping image
            if self.use_flip_camera:
                img = cv2.flip(img, 1) # Flip left/right
            
            return img
        except queue.Empty:
            # If no image available, return None
            return None
    
    def getDoubledImage(self):
        '''Get the latest doubled resolution image without removing it from queue.
        Returns a copy of the most recently captured doubled resolution image.
        Returns None if capture_doubled_resolution is disabled.'''
        if not self.capture_doubled_resolution or self.image_queue_doubled is None:
            return None
        
        # Use lock to make get+put atomic and avoid race conditions
        with self.capture_lock:
            try:
                img = self.image_queue_doubled.get_nowait()
                # Put it back immediately under the same lock
                try:
                    self.image_queue_doubled.put_nowait(img)
                except queue.Full:
                    pass
            except queue.Empty:
                return None
        
        # Process outside lock (heavy operations)
        if self.use_undistort and self.mapx_doubled is not None and self.mapy_doubled is not None:
            img = cv2.remap(img, self.mapx_doubled, self.mapy_doubled, cv2.INTER_LINEAR)
        if self.is_rotate:
            img = cv2.rotate(img, cv2.ROTATE_180)
        if self.use_flip_camera:
            img = cv2.flip(img, 1)
        
        return img
    
    def isDoubledResolutionEnabled(self):
        '''Check if doubled resolution capture is enabled.'''
        return self.capture_doubled_resolution

    def _start_capture_thread(self):
        '''Start the background capture thread'''
        if self.capture_thread is None or not self.capture_thread.is_alive():
            self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
            self.capture_thread.start()

    def start_calibration(self):
        if self.mask_projector:
            def display_func(img):
                self._client.displayBackground(image_cv=img)
                
            def capture_func():
                return self.getRawImage()

            def set_show_default_bg_func(show):
                return self._client.setShowDefaultBackground(show)
            
            # Run calibration in a separate thread
            threading.Thread(target=self.mask_projector.calibrate, 
                             args=(display_func, capture_func, set_show_default_bg_func),
                             daemon=True).start()

    def run(self):
        '''Main camera thread loop'''
        # Start the background capture thread
        self._start_capture_thread()
        
        while True:
            # Check scenario is completed
            if self._stop_event.is_set():
                break
            
            # Get latest captured image
            img = self.captureImage()
            if img is None:
                time.sleep(0.01)  # Small delay if no image available
                continue

            # Prepare masked image outside lock (heavy GPU operation)
            if self.mask_projector is not None:
                display_img = self._client.getDisplayImage()
                masked_img = self.mask_projector.getMaskedImage(img, display_img)
            else:
                masked_img = None

            # Quick assignment under lock
            with self.capture_lock:
                self.img_raw = img
                self.img = masked_img if masked_img is not None else img

            # Mark as ready
            if not self.ready:
                self.ready = True
                self._ready_event.set()
        
            # Show camera image
            if config.display_raw_camera_window or self.show_window:
                cv2.imshow("Camera", img)
                if cv2.waitKey(1) == ord("q"): # q 누를 시 웹캠 종료
                    break
                    
        # Mark as exit when loop is completed
        self.cap.release()
        cv2.destroyAllWindows()