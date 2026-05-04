#  SULIVAN : Synergistic Understanding and Learning with Interactive robotics,
#            computer Vision, Augmented reality, and Neural networks
#  Copyright 2026 PNU IRLab All rights reserved.
#
#  Made by Jibaek Oh (jibaek8809@pusan.ac.kr), Jihoon Yoon (face5921@pusan.ac.kr),
#          HyeonUk Kang (hwkang0318@pusan.ac.kr)

import cupy as cp
import numpy as np

import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import config

from utils.cv2_cupy import warpPerspective, cvtColor

import cv2

'''
cv2_cupy test script for perspective transformation using CuPy
'''

M_cp = np.array([[  1.15023549E+00, -7.56023325E-02, -1.36176001E+02 ],
                 [  1.77520917E-02,  1.17660864E+00, -1.56390248E+02 ],
                 [  7.38722464E-07, -7.22291193E-05,  1.00000000E+00 ]])

# Camera capture and display example

def main():
    """
    Main function to capture from camera and apply perspective transformation
    """
    # Initialize camera (camera index 0)
    cap = cv2.VideoCapture(0)
    
    # Set camera properties (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 60)
    
    # Define output size for transformed image
    output_size = (1920, 1080)  # (width, height)
    
    print("Press 'q' to quit")
    
    try:
        while True:
            # Capture frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Apply perspective transformation
            transformed_frame = warpPerspective(frame, config.M_cp, output_size)
            # Convert color space if needed (e.g., BGR to RGB)
            transformed_frame = cvtColor(transformed_frame, cv2.COLOR_BGR2HLS)
            
            # Convert back to numpy for display (if needed)
            if isinstance(transformed_frame, cp.ndarray):
                transformed_frame_np = cp.asnumpy(transformed_frame)
            else:
                transformed_frame_np = transformed_frame
            
            # Display original and transformed images
            cv2.imshow('Original', frame)
            cv2.imshow('Transformed', transformed_frame_np.astype(np.uint8))
            
            # Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import cv2  # For display
    main()