#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import numpy as np

DEBUG_MODE = False

# =============== Text-to-Speech ===============
EDGE_TTS_VOICE = "ko-KR-SunHiNeural"
EDGE_TTS_RATE = "-10%"
EDGE_TTS_PITCH = "+0Hz"
EDGE_TTS_VOLUME = 0.8
EDGE_TTS_CACHE_DIR = ".cache/tts"

# =============== Camera ===============
# Camera devices number
# Define camera to use
camera_device = 0

# Define camera resolution
CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
capture_doubled_resolution = False

# Use Projection
# If you tests on desk, You must be turned off
use_camera_projection = True

# Flip left/right
use_flip_camera = False
# Rotate image to 180 deg
use_rotate_camera = True

display_raw_camera_window = False

# Use Camera calibration matrix
undistort_camera = False
filename_calibration_matrix = "calibration_matrix.npy"
filename_distortion_coefficients = "distortion_coefficients.npy"

# Record video of camera
record_video = True
record_dir = 'records'

# Define directory of logs
logs_dir = "logs"

# ======== Projection Parameters =======
# Define projection area size in pixel/physical unit(cm)
PIXELS_WIDTH  = 1920
PIXELS_HEIGHT = 1080
LENGTH_WIDTH  = 100.0
LENGTH_HEIGHT = 56.25

# Screen + Extra spaces on human-side
M_cp = np.array([[  1.05931446E+00, -3.85205258E-01,  2.11862892E+01 ],
                 [  3.12882625E-03,  1.64471967E+00, -5.24965941E+02 ],
                 [ -6.12070364E-06, -3.76853910E-04,  1.00000000E+00 ]])

DISPLAY_POINTS = [[96, 319], [264, 781], [1673, 316], [1528, 775]]

# Define camera brightness
# CAMERA_BRIGHTNESS = 160

# Show bounding boxes of hands on projector screen
show_hands_bbox_to_screen = False

# Show bounding boxes of YOLO detections on projector screen
show_yolo_bbox_to_screen = False

# borderless pygame window
borderless_pygame_window = True

# Use mask projector
use_project_masking = True
use_rgb_masking = False
use_compensate_hls = False

# Define YOLO model filename
filename_yolo = "models/yolov8m.pt"

# Use Oriented Bounding Box (OBB) detection
# The filename must be ended with "-obb.pt"
use_obb = False
filename_yolo_obb = "models/yolo26m-obb.pt"

# =============== VLM (Vision-Language Model) ===============
# Provider: "gemini" (Google Gemini) or "local" (local model, not yet implemented)
VLM_PROVIDER = "gemini"

# API Key for Gemini (set your own key here or use environment variable)
VLM_API_KEY = ""  # TODO: Set your Gemini API key

# Model name (for Gemini: "gemini-2.0-flash", "gemini-1.5-pro", etc.)
VLM_MODEL = "gemini-2.5-flash"

# Enable/disable VLM feature
VLM_ENABLED = True

# Auto check interval in milliseconds (0 to disable auto check)
# Recommended: 5000~10000ms to balance cost and responsiveness
VLM_AUTO_CHECK_INTERVAL = 0  # Disabled by default, set > 0 to enable

# Maximum tokens in VLM response (shorter = faster, less cost)
VLM_MAX_TOKENS = 1024

# Temperature for response generation (0.0 = deterministic, 1.0 = creative)
VLM_TEMPERATURE = 0.7

# Default prompt template
# Available placeholders: {step_name}, {step_description}, {completion_conditions}, {custom_prompt}
VLM_DEFAULT_PROMPT_TEMPLATE = \
"""당신은 작업 가이드 시스템의 AI 어시스턴트입니다.
작업자가 현재 수행 중인 작업 단계: {step_name}

이번 단계에서 해야 할 것:
{step_description}

완료 조건:
{completion_conditions}

{custom_prompt}

위 정보와 제공된 카메라 이미지를 보고, 작업자가 현재 작업을 올바르게 수행하고 있는지 평가하고,
도움이 될 수 있는 피드백을 한국어로 한 문장으로 간결하게 제공하세요.
문제가 없다면 "잘 하고 있습니다"라고만 답하세요."""

# Output method for VLM feedback: "tts" (text-to-speech)
VLM_OUTPUT_METHOD = "tts"

# Local model path (for future use with local VLM models)
VLM_LOCAL_MODEL_PATH = ""