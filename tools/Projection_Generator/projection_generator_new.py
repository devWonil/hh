#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
import config

PIXEL_W = 1920
PIXEL_H = 1080

CAMERA_W = 1920
CAMERA_H = 1080

os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
os.environ['SDL_VIDEO_WINDOW_POS'] = f"-{PIXEL_W},0"

import cv2 
import cv2.aruco as aruco
import numpy as np
import pygame


use_undistort = False
if os.path.exists("distortion_coefficients.npy") and os.path.exists("calibration_matrix.npy"):
    use_undistort = True
    dist = np.load("distortion_coefficients.npy")
    mtx = np.load("calibration_matrix.npy")
    
cap = cv2.VideoCapture(config.camera_device)

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_H)

# Display a checkerboard pattern for calibration
pygame.init()

# Charuco pattern settings
width = PIXEL_W
height = PIXEL_H
squares_x = 20  # Number of squares in X
squares_y = 11   # Number of squares in Y
square_length = 100 # Arbitrary units, will be scaled by generateImage
marker_length = 70  # 70% of square

dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
board = aruco.CharucoBoard((squares_x, squares_y), square_length, marker_length, dictionary)

# Generate image
margin = 50
board_image = board.generateImage((width, height), marginSize=margin)

# Convert to BGR to draw colored border
board_image_display = cv2.cvtColor(board_image, cv2.COLOR_GRAY2BGR)
# Draw 5px red border
cv2.rectangle(board_image_display, (0, 0), (width-1, height-1), (0, 0, 255), 5)

# Create Charuco detector
charuco_detector = aruco.CharucoDetector(board)

# Detect corners on the generated image (Ground Truth)
# We detect on the generated image to get the exact pixel coordinates of the corners
# as they appear on the projector.
corners_origin, ids_origin, _, _ = charuco_detector.detectBoard(board_image)

if ids_origin is None or len(ids_origin) == 0:
    print("Error: Could not detect corners on the generated board image.")
    exit()

# Prepare for display
# Convert to RGB for Pygame (OpenCV is BGR)
board_image_rgb = cv2.cvtColor(board_image_display, cv2.COLOR_BGR2RGB)
# Transpose for Pygame [y, x, c] -> [x, y, c]
chekcerboard_for_display = np.transpose(board_image_rgb, (1, 0, 2))

# Pygame screen setup
screen = pygame.display.set_mode((PIXEL_W, PIXEL_H), pygame.NOFRAME)
pygame.display.set_caption("Checkerboard Display")

# Convert final_checkerboard to Pygame surface
checkerboard_surface = pygame.surfarray.make_surface(chekcerboard_for_display)

# Display chessboard on Pygame screen
screen.blit(pygame.transform.scale(checkerboard_surface, (PIXEL_W, PIXEL_H)), (0, 0))

# Update the display
pygame.display.flip()

# # Wait for a short time to display the checkerboard
pygame.time.wait(1000) # Wait a bit longer for the projector to stabilize

success, frame = cap.read()
if use_undistort:
    frame = cv2.undistort(frame, mtx, dist, None, None)
if config.use_rotate_camera:
    frame = cv2.rotate(frame, cv2.ROTATE_180)
if config.use_flip_camera:
    frame = cv2.flip(frame, 1)

# Convert the captured frame to grayscale for corner detection
gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Find the Charuco corners in the captured frame
corners_frame, ids_frame, _, _ = charuco_detector.detectBoard(gray_frame)

if ids_frame is not None and len(ids_frame) > 4:
    # Match points based on IDs
    # We need to find the intersection of IDs between origin and frame
    
    # Create a dictionary for source points: id -> point
    # ids_origin is (N, 1), corners_origin is (N, 1, 2)
    src_dict = {}
    for i in range(len(ids_origin)):
        src_dict[ids_origin[i][0]] = corners_origin[i][0]
        
    obj_points = []
    img_points = []
    
    for i in range(len(ids_frame)):
        id_val = ids_frame[i][0]
        if id_val in src_dict:
            obj_points.append(src_dict[id_val]) # Projector pixel (Source)
            img_points.append(corners_frame[i][0]) # Camera pixel (Destination)
            
    obj_points = np.array(obj_points, dtype=np.float32)
    img_points = np.array(img_points, dtype=np.float32)

    if len(obj_points) < 4:
        print("Not enough common points found.")
    else:
        # Compute the homography matrix between the two sets of corners
        # H maps Camera (img_points) -> Projector (obj_points)
        H, status = cv2.findHomography(img_points, obj_points, cv2.RANSAC, 5.0)
        INV_H = np.linalg.pinv(H)

        # Calculate DISPLAY_POINTS
        DISPLAY_POINTS = []
        PIXEL_POINTS = [[0, 0], [0, PIXEL_H], [PIXEL_W, 0], [PIXEL_W, PIXEL_H]]
        for point in PIXEL_POINTS:
            point_homo = np.ones((3, 1))
            point_homo[0] = point[0]
            point_homo[1] = point[1]

            point_camera = INV_H @ point_homo
            point_camera /= point_camera[2]

            DISPLAY_POINTS.append([round(point_camera[0,0]), round(point_camera[1,0])])

        print("# ===== Projection Parameters =====")
        print(f"M_cp = np.array([[ {H[0][0]: .8E}, {H[0][1]: .8E}, {H[0][2]: .8E} ],")
        print(f"                 [ {H[1][0]: .8E}, {H[1][1]: .8E}, {H[1][2]: .8E} ],")
        print(f"                 [ {H[2][0]: .8E}, {H[2][1]: .8E}, {H[2][2]: .8E} ]])")

        print()
        print(f"DISPLAY_POINTS = {DISPLAY_POINTS}")

        # Draw corners_origin on the checkerboard image
        checkerboard_with_corners = board_image_display.copy()
        if ids_origin is not None:
            cv2.aruco.drawDetectedCornersCharuco(checkerboard_with_corners, corners_origin, ids_origin)
        checkerboard_with_corners = cv2.resize(checkerboard_with_corners, (PIXEL_W//2, PIXEL_H//2))

        # Draw corners_frame on the captured frame
        frame_with_corners = frame.copy()
        if ids_frame is not None:
            cv2.aruco.drawDetectedCornersCharuco(frame_with_corners, corners_frame, ids_frame)
        frame_with_corners = cv2.resize(frame_with_corners, (PIXEL_W//2, PIXEL_H//2))

        # Display the images with corners
        cv2.imshow("Checkerboard with Corners", checkerboard_with_corners)
        cv2.imshow("Captured Frame with Corners", frame_with_corners)

        # Wait for a key press to close the windows
        cv2.waitKey(0)
        cv2.destroyAllWindows()
else:
    print("Checkerboard corners not found in frame.")

# Close PyGame
pygame.quit()