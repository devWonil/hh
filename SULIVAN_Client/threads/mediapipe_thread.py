#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import os
import time
import threading
import urllib.request

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from client import SULIVAN_Client

import config
from utils.mediapipe_utils import FINGER_LANDMARKS, check_finger_angles, convert_coordinate

# Hand landmark connections for drawing (matching mp.solutions.hands.HAND_CONNECTIONS)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),           # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),           # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),       # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),     # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20),    # Pinky
    (0, 17),                                    # Palm base
]

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "hand_landmarker.task")


def _ensure_model():
    """Download the hand_landmarker model if not present."""
    if not os.path.exists(MODEL_PATH):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        print(f"[MediaPipe] Downloading hand_landmarker.task model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print(f"[MediaPipe] Model downloaded successfully.")


def _draw_hand_landmarks(image, landmarks_px):
    """Draw hand landmarks and connections on the image."""
    # Draw connections
    for start_idx, end_idx in HAND_CONNECTIONS:
        x1, y1 = landmarks_px[start_idx][0], landmarks_px[start_idx][1]
        x2, y2 = landmarks_px[end_idx][0], landmarks_px[end_idx][1]
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Draw landmark points
    for lm in landmarks_px:
        cv2.circle(image, (lm[0], lm[1]), 5, (255, 0, 255), cv2.FILLED)


class MediaPipeThread(threading.Thread):
    def __init__(self, client: "SULIVAN_Client", get_image_func=None):
        self._client = client
        self.get_image_func = get_image_func
        self.ready = False
        self._ready_event = threading.Event()

        # Download model if needed
        _ensure_model()

        # Create HandLandmarker with mp.tasks API
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=4,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.7,
        )
        self.detector = vision.HandLandmarker.create_from_options(options)

        self._stop_event = threading.Event()
        super().__init__()
        self.name = "MediaPipe"
        self.daemon = True

        # Mediapipe returns shrunken bbox of all direction by specified pixels
        self.mp_inner_offset = 0
        
        # define hands list
        # element: {'type': hand_type, 'score': hand_score, 'bbox': bbox_pixel, 'bbox_cam': bbox_cam,
        #                              'landmarks': lm_pixel, 'landmarks_cam': lm_cam,
        #                              'hand_circle': hand_circle, 'hand_circle_cam': hand_circle_cam} )
        self.hands = []

        self.raw_image = None
        self.mp_image = None
        self._data_lock = threading.Lock()
    
    def stop(self):
        self._stop_event.set()

    def getRawImage(self):
        ''' For time sync in video recorder '''
        with self._data_lock:
            return self.raw_image.copy() if self.mp_image is not None else None
    
    def getImage(self):
        '''Get the image with mediapipe detection result overlays'''
        with self._data_lock:
            return self.mp_image.copy() if self.mp_image is not None else None
    
    def getHands(self):
        '''Get the detection results'''
        with self._data_lock:
            return list(self.hands)
    
    def run(self):
        last_timestamp_ms = 0

        while True:
            if self._stop_event.is_set():
                break
            # Run mediapipe
            if self.get_image_func is not None:
                _raw = self.get_image_func()
            else:
                _raw = self._client.camera_thread.getImage()
            if _raw is None:
                self._stop_event.wait(0.005)
                continue

            # _raw is already a private copy from getImage(), so we can
            # store a copy for raw_image (read by others) and use _raw
            # directly for drawing landmarks (this thread only).
            with self._data_lock:
                self.raw_image = _raw.copy()

            img = _raw
            h, w, _ = img.shape

            # Convert BGR to RGB for MediaPipe
            rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

            # Ensure strictly increasing timestamp (required by VIDEO mode)
            current_ms = int(time.monotonic() * 1000)
            frame_timestamp_ms = max(current_ms, last_timestamp_ms + 1)
            last_timestamp_ms = frame_timestamp_ms

            # Detect hands using mp.tasks HandLandmarker
            result = self.detector.detect_for_video(mp_image, frame_timestamp_ms)

            flip_type = not config.use_flip_camera

            # Build hands list
            hands = []
            for i in range(len(result.hand_landmarks)):
                hand_landmarks = result.hand_landmarks[i]
                handedness = result.handedness[i][0]

                # Convert normalized landmarks to pixel coordinates (matching cvzone format)
                lm_list = []
                for lm in hand_landmarks:
                    px = int(lm.x * w)
                    py = int(lm.y * h)
                    pz = int(lm.z * w)  # z scaled by width (same as cvzone)
                    lm_list.append([px, py, pz])

                # Compute bbox (x, y, w, h)
                x_list = [lm[0] for lm in lm_list]
                y_list = [lm[1] for lm in lm_list]
                xmin, xmax = min(x_list), max(x_list)
                ymin, ymax = min(y_list), max(y_list)
                bbox = (xmin, ymin, xmax - xmin, ymax - ymin)

                # Compute center
                cx = (xmin + xmax) // 2
                cy = (ymin + ymax) // 2

                # Hand type with flipType handling (same as cvzone)
                hand_type = handedness.category_name
                if flip_type:
                    hand_type = "Right" if hand_type == "Left" else "Left"

                hand_dict = {
                    "lmList": lm_list,
                    "bbox": bbox,
                    "center": (cx, cy),
                    "type": hand_type,
                }
                hands.append(hand_dict)

                # Draw landmarks & connections on image
                _draw_hand_landmarks(img, lm_list)

                # Draw bbox
                cv2.rectangle(img,
                              (bbox[0] - 20, bbox[1] - 20),
                              (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                              (255, 0, 255), 2)
                cv2.putText(img, hand_type,
                            (bbox[0] - 30, bbox[1] - 30),
                            cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

            # Init hand rects
            hands_return = []

            # Check if any hands are detected
            for i, hand in enumerate(hands):
                hand_pixel = convert_coordinate(hand)
                lm_pixel   = hand_pixel["lmList"]
                bbox       = hand_pixel["bbox"]
                hand_type  = hand_pixel["type"]

                bbox_cam = hand["bbox"]
                lm_cam   = hand["lmList"]
                
                # Apply offsets
                bbox_pixel = []
                bbox_pixel.append(bbox[0] - self.mp_inner_offset)
                bbox_pixel.append(bbox[1] - self.mp_inner_offset)
                bbox_pixel.append(bbox[2] + 2 * self.mp_inner_offset)
                bbox_pixel.append(bbox[3] + 2 * self.mp_inner_offset)
                
                # Create hand circle info (Used by other objects)
                index_tip = lm_pixel[FINGER_LANDMARKS[0][0]]
                pinky_tip = lm_pixel[FINGER_LANDMARKS[3][1]]
                center_x = (index_tip[0] + pinky_tip[0]) // 2
                center_y = (index_tip[1] + pinky_tip[1]) // 2
                radius = int(((index_tip[0] - pinky_tip[0]) ** 2 + (index_tip[1] - pinky_tip[1]) ** 2) ** 0.5 / 2)
                hand_circle = {'center': (center_x, center_y), 'radius': radius}

                index_tip_cam = lm_cam[FINGER_LANDMARKS[0][0]]
                pinky_tip_cam = lm_cam[FINGER_LANDMARKS[3][1]]
                center_x_cam = (index_tip_cam[0] + pinky_tip_cam[0]) // 2
                center_y_cam = (index_tip_cam[1] + pinky_tip_cam[1]) // 2
                radius_cam = int(((index_tip_cam[0] - pinky_tip_cam[0]) ** 2 + (index_tip_cam[1] - pinky_tip_cam[1]) ** 2) ** 0.5 / 2)
                hand_circle_cam = {'center': (center_x_cam, center_y_cam), 'radius': radius_cam}

                # Draw hand circle
                cv2.circle(img, hand_circle_cam['center'], hand_circle_cam['radius'], color=(255, 0, 0), thickness=2)

                
                hand_score = result.handedness[i][0].score
                cv2.putText(img, f"{hand_score:.4f}", (bbox_cam[0] + 60, bbox_cam[1] - 30), cv2.FONT_HERSHEY_PLAIN,
                                1.4, (255, 0, 255), 2)

                # Append hand to hands list
                hands_return.append( {'type': hand_type, 'score': hand_score, 'bbox': bbox_pixel, 'bbox_cam': bbox_cam,
                                      'landmarks': lm_pixel, 'landmarks_cam': lm_cam,
                                      'hand_circle': hand_circle, 'hand_circle_cam': hand_circle_cam} )

            # Update mp_image with overlays and hand detection results
            with self._data_lock:
                self.mp_image = img
                self.hands = hands_return if hands_return else []

            # Mark as ready
            if not self.ready:
                self.ready = True
                self._ready_event.set()

        # Clean up detector resources
        self.detector.close()